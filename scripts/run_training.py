#!/usr/bin/env python3
"""
run_training.py
===============
Main training orchestrator for the Personalized Query Rewriter.

Executes the full training pipeline:
    0. Load and preprocess data
    1. Pre-train user encoder (contrastive)
    2. Stage 1: Continual Pre-Training (CPT) on domain text
    3. Stage 2: Supervised Fine-Tuning (SFT) with user context
    4. Stage 3: GRPO alignment with retrieval rewards
    5. Train personalization gate

Usage:
    python -m scripts.run_training --config config/config.yaml
    python -m scripts.run_training --config config/config.yaml --stage sft  # Single stage
    python -m scripts.run_training --config config/config.yaml --stage grpo --resume outputs/sft/best
"""

import argparse
import logging
import os
import sys

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personalized_query_rewriter.data.data_loader import (
    load_raw_data,
    split_data,
    build_session_reformulation_pairs,
    build_click_through_pairs,
    identify_ambiguous_queries,
)
from personalized_query_rewriter.data.feature_engineering import UserFeatureEncoder
from personalized_query_rewriter.models.personalized_rewriter import PersonalizedQueryRewriter
from personalized_query_rewriter.training.train_user_encoder import UserEncoderTrainer
from personalized_query_rewriter.training.stage1_cpt import CPTTrainer
from personalized_query_rewriter.training.stage2_sft import SFTTrainer
from personalized_query_rewriter.training.stage3_grpo import GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Personalized Query Rewriter")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "user_encoder", "cpt", "sft", "grpo", "gate"],
                        help="Training stage to run (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--data", type=str, default=None,
                        help="Override data path from config")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = args.data or config["data"]["input_csv"]

    # --- Device setup ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # =================================================================
    # Step 0: Load and preprocess data
    # =================================================================
    logger.info("=" * 60)
    logger.info("STEP 0: Loading and preprocessing data")
    logger.info("=" * 60)

    df = load_raw_data(data_path, min_query_length=config["data"]["min_query_length"])
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
        seed=config["data"]["random_seed"],
    )

    # Feature encoder
    feature_encoder = UserFeatureEncoder(include_derived=True)
    logger.info("Feature encoder: %d features", feature_encoder.num_features)

    # Build training data for each stage
    logger.info("Building session reformulation pairs...")
    reformulation_pairs = build_session_reformulation_pairs(train_df)

    logger.info("Building click-through pairs...")
    click_pairs = build_click_through_pairs(train_df)

    logger.info("Identifying ambiguous queries...")
    train_df = identify_ambiguous_queries(train_df)

    # Extract text lists for CPT
    train_texts = train_df["query_clean"].dropna().str.lower().tolist()
    val_texts = val_df["query_clean"].dropna().str.lower().tolist()

    # =================================================================
    # Initialize model
    # =================================================================
    logger.info("=" * 60)
    logger.info("Initializing PersonalizedQueryRewriter")
    logger.info("=" * 60)

    # Update config with actual feature count
    config["ft_transformer"]["num_features"] = feature_encoder.num_features

    load_llm = args.stage in ("all", "cpt", "sft", "grpo")
    model = PersonalizedQueryRewriter(config, load_llm=load_llm)

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model.load_components(args.resume)

    model.to(device)

    # =================================================================
    # Stage 0: Pre-train User Encoder (Contrastive)
    # =================================================================
    if args.stage in ("all", "user_encoder"):
        logger.info("=" * 60)
        logger.info("STAGE 0: User Encoder Contrastive Pre-training")
        logger.info("=" * 60)

        trainer = UserEncoderTrainer(
            user_encoder=model.user_encoder,
            config=config["training_user_encoder"],
            train_df=train_df,
            val_df=val_df,
            feature_encoder=feature_encoder,
        )
        history = trainer.train()
        logger.info("User encoder training complete. Final val loss: %.4f",
                     history["val_losses"][-1])

    # =================================================================
    # Stage 1: Continual Pre-Training (CPT)
    # =================================================================
    if args.stage in ("all", "cpt"):
        logger.info("=" * 60)
        logger.info("STAGE 1: Continual Pre-Training (CPT)")
        logger.info("=" * 60)

        trainer = CPTTrainer(
            model=model,
            config=config["training_cpt"],
            train_texts=train_texts,
            val_texts=val_texts,
        )
        history = trainer.train()
        logger.info("CPT complete. Final val loss: %.4f",
                     history["val_losses"][-1])

    # =================================================================
    # Stage 2: Supervised Fine-Tuning (SFT)
    # =================================================================
    if args.stage in ("all", "sft"):
        logger.info("=" * 60)
        logger.info("STAGE 2: Supervised Fine-Tuning (SFT)")
        logger.info("=" * 60)

        # Combine reformulation pairs and click pairs for SFT
        import pandas as pd
        sft_train = pd.concat([reformulation_pairs, click_pairs], ignore_index=True)
        sft_val = build_session_reformulation_pairs(val_df)

        if len(sft_train) == 0:
            logger.warning("No SFT training pairs found. Using click pairs from train set.")
            sft_train = click_pairs

        trainer = SFTTrainer(
            model=model,
            config=config["training_sft"],
            train_df=sft_train,
            val_df=sft_val if len(sft_val) > 0 else sft_train.sample(frac=0.1),
            feature_encoder=feature_encoder,
        )
        history = trainer.train()
        logger.info("SFT complete. Final val loss: %.4f",
                     history["val_losses"][-1])

    # =================================================================
    # Stage 3: GRPO Alignment
    # =================================================================
    if args.stage in ("all", "grpo"):
        logger.info("=" * 60)
        logger.info("STAGE 3: GRPO Alignment")
        logger.info("=" * 60)

        grpo_train = click_pairs  # Use click pairs with gold navlinks

        trainer = GRPOTrainer(
            model=model,
            config=config["training_grpo"],
            train_df=grpo_train,
            feature_encoder=feature_encoder,
            search_index_fn=None,  # Plug in actual search index in production
        )
        history = trainer.train()
        logger.info("GRPO complete. Final avg reward: %.4f",
                     history["avg_rewards"][-1] if history["avg_rewards"] else 0)

    # =================================================================
    # Save final model
    # =================================================================
    final_dir = "outputs/final"
    model.save_components(final_dir)
    logger.info("=" * 60)
    logger.info("Training complete! Final model saved to: %s", final_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
