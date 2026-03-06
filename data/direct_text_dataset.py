"""
direct_text_dataset.py
======================
PyTorch Dataset for the direct-text rewriter route (no embeddings/E2P).

Serializes user products directly into the prompt text, so the LLM
receives them as regular tokens instead of soft prefix embeddings.

Training format:
    "User products: Sapphire Reserve, Mobile Banking. rewrite: rewards → chase sapphire reserve points redemption"
"""

import logging
import random
from typing import Any, Dict, List

import pandas as pd
from torch.utils.data import Dataset

from ..models.direct_text_rewriter import DirectTextRewriter

logger = logging.getLogger(__name__)


class DirectTextSFTDataset(Dataset):
    """
    SFT dataset for the direct-text rewriter.

    Each example encodes the user products as literal text in the prompt,
    rather than as a separate feature tensor processed by FT-Transformer + E2P.

    Args:
        tokenizer: HuggingFace tokenizer.
        df: DataFrame with columns: original_query (or query_clean),
            rewritten_query, user_product_list.
        max_length: Maximum token length.
    """

    def __init__(
        self,
        tokenizer,
        df: pd.DataFrame,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

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
                })

        logger.info("DirectTextSFTDataset: %d training examples", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]

        prompt = DirectTextRewriter.build_prompt(
            ex["original_query"], ex["user_products"]
        )
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

        # Mask prompt tokens in labels (only predict the target)
        prompt_enc = self.tokenizer(f"{prompt} → ", add_special_tokens=True)
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Align
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
