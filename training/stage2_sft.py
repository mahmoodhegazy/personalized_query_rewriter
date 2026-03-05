"""
stage2_sft.py
=============
Stage 2: Supervised Fine-Tuning (SFT) with user context injection.

Purpose:
    Train the full personalization pipeline (FT-Transformer + E2P + LoRA)
    on supervised query rewriting pairs with user profile features.

Method:
    - Input: [E2P_prefix_tokens] + "rewrite: {original_query}" → "{target_rewrite}"
    - User features encoded by FT-Transformer, projected by E2P into prefix tokens.
    - LoRA adapters in the LLM are fine-tuned alongside user_encoder and E2P.
    - Training pairs come from:
        (a) Session reformulation pairs (WeWrite posterior mining)
        (b) Click-through pairs (query → clicked navlink/intent)

Reference:
    QLoRA (4-bit + LoRA) enables fine-tuning on a single A100 with ~0.16%
    of parameters trained. SFT provides the foundation but tends to overfit
    to surface patterns without subsequent GRPO alignment.
"""

import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..data.dataset import SFTDataset, cpt_collate_fn
from ..data.feature_engineering import UserFeatureEncoder
from ..models.personalized_rewriter import PersonalizedQueryRewriter

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer with user context injection.

    Trains: FT-Transformer + E2P + LLM LoRA (gate is frozen).

    Args:
        model: PersonalizedQueryRewriter instance (post-CPT).
        config: Training configuration (config.yaml['training_sft']).
        train_df: Training DataFrame with reformulation/click pairs.
        val_df: Validation DataFrame.
        feature_encoder: UserFeatureEncoder instance.
    """

    def __init__(
        self,
        model: PersonalizedQueryRewriter,
        config: Dict[str, Any],
        train_df,
        val_df,
        feature_encoder: UserFeatureEncoder,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Set training mode: user_encoder + e2p + LoRA
        self.model.set_training_mode("sft")

        # Create datasets
        self.train_dataset = SFTDataset(
            tokenizer=model.tokenizer,
            df=train_df,
            feature_encoder=feature_encoder,
            max_length=config["max_length"],
        )
        self.val_dataset = SFTDataset(
            tokenizer=model.tokenizer,
            df=val_df,
            feature_encoder=feature_encoder,
            max_length=config["max_length"],
        )

        pad_id = model.tokenizer.pad_token_id
        self.collate_fn = partial(cpt_collate_fn, pad_token_id=pad_id)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        # Optimizer with different LRs for different components
        param_groups = [
            {
                "params": [p for p in model.user_encoder.parameters() if p.requires_grad],
                "lr": config["learning_rate"] * 2,  # Higher LR for user encoder
                "name": "user_encoder",
            },
            {
                "params": [p for p in model.e2p.parameters() if p.requires_grad],
                "lr": config["learning_rate"] * 2,
                "name": "e2p",
            },
            {
                "params": [
                    p for n, p in model.llm.named_parameters()
                    if p.requires_grad and "lora_" in n
                ],
                "lr": config["learning_rate"],
                "name": "llm_lora",
            },
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config["weight_decay"],
        )

        total_steps = (
            len(self.train_loader)
            // config["gradient_accumulation_steps"]
            * config["epochs"]
        )
        warmup_steps = int(total_steps * config["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.grad_accum_steps = config["gradient_accumulation_steps"]
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = 3
        self.patience_counter = 0

    def train(self) -> Dict[str, List[float]]:
        """
        Run SFT training loop with early stopping.

        Returns:
            Dict with train_losses, val_losses, gate_scores per epoch.
        """
        history = {
            "train_losses": [],
            "val_losses": [],
            "avg_gate_scores": [],
        }

        for epoch in range(self.config["epochs"]):
            train_loss, avg_gate = self._train_epoch(epoch)
            history["train_losses"].append(train_loss)
            history["avg_gate_scores"].append(avg_gate)

            val_loss = self._validate(epoch)
            history["val_losses"].append(val_loss)

            logger.info(
                "SFT Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, avg_gate=%.3f",
                epoch + 1, self.config["epochs"], train_loss, val_loss, avg_gate,
            )

            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.model.save_components(os.path.join(self.output_dir, "best"))
                logger.info("  New best model (val_loss=%.4f)", val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info("  Early stopping triggered at epoch %d", epoch + 1)
                    break

            self.model.save_components(
                os.path.join(self.output_dir, f"epoch_{epoch+1}")
            )

        return history

    def _train_epoch(self, epoch: int) -> tuple:
        """Run one training epoch. Returns (avg_loss, avg_gate_score)."""
        self.model.train()
        total_loss = 0.0
        total_gate = 0.0
        num_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            user_features = batch["user_features"].to(self.device)

            # Forward with user features
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                user_features=user_features,
            )

            loss = outputs["loss"] / self.grad_accum_steps
            loss.backward()

            total_loss += outputs["loss"].item()
            if "gate_score" in outputs:
                total_gate += outputs["gate_score"].mean().item()
            num_steps += 1

            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (step + 1) % self.config["logging_steps"] == 0:
                avg_loss = total_loss / num_steps
                avg_gate = total_gate / num_steps
                logger.info(
                    "  SFT [Epoch %d, Step %d/%d] loss=%.4f, gate=%.3f",
                    epoch + 1, step + 1, len(self.train_loader), avg_loss, avg_gate,
                )

        avg_loss = total_loss / max(num_steps, 1)
        avg_gate = total_gate / max(num_steps, 1)
        return avg_loss, avg_gate

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_steps = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            user_features = batch["user_features"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                user_features=user_features,
            )

            total_loss += outputs["loss"].item()
            num_steps += 1

        return total_loss / max(num_steps, 1)
