"""
dataset.py
==========
PyTorch Dataset classes for each training stage:

    1. CPTDataset         - Continual pre-training on Chase domain text
    2. SFTDataset         - Supervised fine-tuning with user-conditioned rewrites
    3. GRPODataset        - GRPO alignment with reward signals
    4. UserEncoderContrastiveDataset - Contrastive learning for FT-Transformer

Each dataset handles tokenization and user feature encoding internally.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_engineering import UserFeatureEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: Continual Pre-Training Dataset
# =============================================================================

class CPTDataset(Dataset):
    """
    Dataset for continual pre-training (CPT) on Chase domain text.

    Reuses the existing "complete: {partial_query}" prefix task from the
    Gemma3 notebook, but also introduces a new task prefix:
        "rewrite: {query}" → "{expanded_query}"

    This teaches the model Chase-specific vocabulary and query patterns
    before any personalization is added.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: List of query strings (from CDA utterances).
        max_length: Maximum sequence length.
        min_prefix_chars: Minimum prefix length for completion task.
        task_mix: Probability of "complete" vs "rewrite" task.
    """

    def __init__(
        self,
        tokenizer,
        texts: List[str],
        max_length: int = 128,
        min_prefix_chars: int = 3,
        task_mix: float = 0.7,  # 70% complete, 30% domain MLM
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.min_prefix_chars = min_prefix_chars
        self.task_mix = task_mix

        # Filter texts that are too short
        self.valid_indices = [
            i for i, t in enumerate(texts) if len(t) > min_prefix_chars
        ]
        logger.info("CPTDataset: %d valid samples from %d texts",
                     len(self.valid_indices), len(texts))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[self.valid_indices[idx]]

        if random.random() < self.task_mix:
            # Task 1: Prefix completion (existing task)
            prefix_len = random.randint(self.min_prefix_chars, len(text) - 1)
            partial = text[:prefix_len]
            remaining = text[prefix_len:]
            prompt = f"complete: {partial}"
            full_text = f"{prompt}{remaining}{self.tokenizer.eos_token}"
        else:
            # Task 2: Domain language modeling (full query reconstruction)
            # Masked span reconstruction: mask a random span and predict it
            tokens = text.split()
            if len(tokens) >= 3:
                mask_start = random.randint(0, len(tokens) - 2)
                mask_end = random.randint(mask_start + 1, min(mask_start + 3, len(tokens)))
                masked_tokens = tokens[:mask_start] + ["<mask>"] + tokens[mask_end:]
                prompt = f"rewrite: {' '.join(masked_tokens)}"
                full_text = f"{prompt} → {text}{self.tokenizer.eos_token}"
            else:
                prompt = f"rewrite: {text}"
                full_text = f"{prompt} → {text}{self.tokenizer.eos_token}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Labels: mask the prompt portion (only predict the target)
        prompt_enc = self.tokenizer(prompt, add_special_tokens=True)
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Align lengths
        labels = _align_length(labels, len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# Stage 2: Supervised Fine-Tuning Dataset
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning with user context injection.

    Input format:
        [user_features_tensor] + "rewrite: {original_query}" → "{personalized_rewrite}"

    The user features are encoded as a separate tensor and will be injected
    via the E2P prefix mechanism during forward pass.

    Args:
        tokenizer: HuggingFace tokenizer.
        df: DataFrame with columns: original_query, rewritten_query, user_product_list.
        feature_encoder: UserFeatureEncoder instance.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        tokenizer,
        df: pd.DataFrame,
        feature_encoder: UserFeatureEncoder,
        max_length: int = 128,
        include_click_pairs: bool = True,
    ):
        self.tokenizer = tokenizer
        self.feature_encoder = feature_encoder
        self.max_length = max_length

        # Build training examples
        self.examples = []
        for _, row in df.iterrows():
            original = row.get("original_query", row.get("query_clean", ""))
            target = row.get("rewritten_query", row.get("query_clean", ""))
            products = row.get("user_product_list", [])

            if isinstance(products, str):
                products = [p.strip() for p in products.split(",") if p.strip()]

            if original and target:
                self.examples.append({
                    "original_query": original,
                    "target_query": target,
                    "user_products": products,
                    "navlink": row.get("navlink", ""),
                    "intent": row.get("intent", ""),
                })

        logger.info("SFTDataset: %d training examples", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        # Encode user features
        user_features = self.feature_encoder.encode_to_tensor(ex["user_products"])

        # Build prompt and target
        prompt = f"rewrite: {ex['original_query']}"
        target = ex["target_query"]
        full_text = f"{prompt} → {target}{self.tokenizer.eos_token}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Mask prompt tokens in labels
        prompt_enc = self.tokenizer(f"{prompt} → ", add_special_tokens=True)
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = _align_length(labels, len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "user_features": user_features,
        }


# =============================================================================
# Stage 3: GRPO Alignment Dataset
# =============================================================================

class GRPODataset(Dataset):
    """
    Dataset for Group Relative Policy Optimization (GRPO).

    Each sample contains:
        - The original query and user features (for generation)
        - The gold navlink/intent (for reward computation)
        - The reference rewrite from SFT model (for KL regularization)

    During training, the model generates `num_generations` candidate rewrites
    per query. Rewards are computed for each, and policy gradients are
    estimated from the group-relative advantages.

    Args:
        tokenizer: HuggingFace tokenizer.
        df: DataFrame with: query_clean, user_product_list, navlink, intent.
        feature_encoder: UserFeatureEncoder instance.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        tokenizer,
        df: pd.DataFrame,
        feature_encoder: UserFeatureEncoder,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.feature_encoder = feature_encoder
        self.max_length = max_length

        self.examples = []
        for _, row in df.iterrows():
            query = row.get("query_clean", "")
            products = row.get("user_product_list", [])
            if isinstance(products, str):
                products = [p.strip() for p in products.split(",") if p.strip()]

            self.examples.append({
                "query": query,
                "user_products": products,
                "navlink": row.get("navlink", ""),
                "intent": row.get("intent", ""),
            })

        logger.info("GRPODataset: %d examples", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        user_features = self.feature_encoder.encode_to_tensor(ex["user_products"])

        prompt = f"rewrite: {ex['query']}"
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "input_ids": prompt_enc["input_ids"].squeeze(0),
            "attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "user_features": user_features,
            "gold_navlink": ex["navlink"] or "",
            "gold_intent": ex["intent"] or "",
            "original_query": ex["query"],
        }


# =============================================================================
# User Encoder Contrastive Dataset
# =============================================================================

class UserEncoderContrastiveDataset(Dataset):
    """
    Contrastive learning dataset for training the FT-Transformer user encoder.

    Following CoPPS (KDD 2023): learn user embeddings such that
    (user_profile, clicked_feature) pairs are close while
    (user_profile, unclicked_feature) pairs are distant.

    Positive pairs:  (user_features, navlink_embedding) for clicked interactions.
    Negative pairs:  (user_features, navlink_embedding) for other users' clicks
                     (in-batch negatives via InfoNCE).

    Args:
        df: DataFrame with: user_product_list, navlink, clicked_bool.
        feature_encoder: UserFeatureEncoder instance.
        navlink_vocab: Dict mapping navlink strings to integer IDs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_encoder: UserFeatureEncoder,
        navlink_vocab: Optional[Dict[str, int]] = None,
    ):
        self.feature_encoder = feature_encoder

        # Only keep clicked interactions with valid navlinks
        clicked = df[df["clicked_bool"] & df["Selected_Navlink"].notna()].copy()

        # Build navlink vocabulary if not provided
        if navlink_vocab is None:
            unique_navlinks = clicked["Selected_Navlink"].unique().tolist()
            self.navlink_vocab = {nl: i for i, nl in enumerate(unique_navlinks)}
        else:
            self.navlink_vocab = navlink_vocab

        self.num_navlinks = len(self.navlink_vocab)

        self.examples = []
        for _, row in clicked.iterrows():
            products = row.get("user_product_list", [])
            navlink = row["Selected_Navlink"]
            if navlink in self.navlink_vocab:
                self.examples.append({
                    "user_products": products,
                    "navlink_id": self.navlink_vocab[navlink],
                })

        logger.info("UserEncoderContrastiveDataset: %d positive pairs, %d navlinks",
                     len(self.examples), self.num_navlinks)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        user_features = self.feature_encoder.encode_to_tensor(ex["user_products"])
        navlink_id = torch.tensor(ex["navlink_id"], dtype=torch.long)

        return {
            "user_features": user_features,
            "navlink_id": navlink_id,
        }


# =============================================================================
# Collate Functions
# =============================================================================

def cpt_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate for CPT/SFT: pad input_ids, attention_mask, labels."""
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_masks, labels_list = [], [], []
    user_features_list = []

    for x in batch:
        ids = list(x["input_ids"])
        mask = list(x["attention_mask"])
        labs = list(x["labels"])

        labs = _align_length(labs, len(ids))
        pad_len = max_len - len(ids)

        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append(mask + [0] * pad_len)
        labels_list.append(labs + [-100] * pad_len)

        if "user_features" in x:
            user_features_list.append(x["user_features"])

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }

    if user_features_list:
        result["user_features"] = torch.stack(user_features_list)

    return result


def grpo_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, Any]:
    """Collate for GRPO: pad prompts + keep metadata."""
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids, attention_masks, user_features = [], [], []
    gold_navlinks, gold_intents, original_queries = [], [], []

    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad_len = max_len - ids.size(0)

        input_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        )
        user_features.append(x["user_features"])
        gold_navlinks.append(x["gold_navlink"])
        gold_intents.append(x["gold_intent"])
        original_queries.append(x["original_query"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "user_features": torch.stack(user_features),
        "gold_navlinks": gold_navlinks,
        "gold_intents": gold_intents,
        "original_queries": original_queries,
    }


# =============================================================================
# Helpers
# =============================================================================

def _align_length(labels: List[int], target_len: int) -> List[int]:
    """Ensure labels list matches target length."""
    if len(labels) < target_len:
        labels = labels + [-100] * (target_len - len(labels))
    elif len(labels) > target_len:
        labels = labels[:target_len]
    return labels
