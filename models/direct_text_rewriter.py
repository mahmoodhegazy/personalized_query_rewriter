"""
direct_text_rewriter.py
=======================
Alternate personalized query rewriter that injects user products directly
as natural-language text tokens (no FT-Transformer / E2P embedding pipeline).

Instead of:
    user_features → FT-Transformer → E2P → soft prefix tokens → LLM

This route does:
    "User products: Sapphire Reserve, Mobile Banking. rewrite: {query}" → LLM

This serves as a baseline to compare against the embedding+projection approach
on both quality and latency.  It follows the TabLLM "natural language
serialization" paradigm documented in the survey.

Trade-offs vs. the E2P route:
    + Simpler pipeline — no extra learned modules
    + No separate user-encoder training stage required
    - Consumes input tokens proportional to product count
    - Cannot encode derived/aggregate features (diversity score, etc.)
    - Latency grows with product list length
    - Harder to cache (prompt varies per user)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


@dataclass
class DirectRewriteResult:
    """Result from the direct-text rewriter."""
    original_query: str
    rewritten_query: str
    latency_ms: float
    prompt_tokens: int
    alternatives: List[str]


class DirectTextRewriter(nn.Module):
    """
    Query rewriter that serializes user products as text in the prompt.

    Architecture:
        prompt = "User products: {csv_products}. rewrite: {query}"
        → Gemma3-1B + LoRA → rewritten query

    Args:
        config: Configuration dict (same schema as config.yaml).
        load_llm: Whether to load the LLM weights.
    """

    def __init__(self, config: Dict[str, Any], load_llm: bool = True):
        super().__init__()
        self.config = config

        self.tokenizer = None
        self.llm = None
        if load_llm:
            self._load_llm(config["llm"])

    def _load_llm(self, llm_cfg: Dict):
        model_name = llm_cfg["model_name"]
        logger.info("DirectTextRewriter loading LLM: %s", model_name)

        dtype_str = llm_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model.config.pad_token_id = self.tokenizer.eos_token_id

        lora_cfg = llm_cfg["lora"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        )
        self.llm = get_peft_model(base_model, lora_config)
        self.llm.enable_input_require_grads()

        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.llm.parameters())
        logger.info("DirectTextRewriter LLM: %s/%s trainable params (%.2f%%)",
                     f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))

    @staticmethod
    def build_prompt(query: str, user_products: List[str]) -> str:
        """
        Serialize user products and query into a single text prompt.

        Format:
            User products: Sapphire Reserve, Mobile Banking. rewrite: rewards
        """
        products_str = ", ".join(user_products) if user_products else "none"
        return f"User products: {products_str}. rewrite: {query}"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard causal-LM forward for training.

        The prompt already contains user products as text tokens,
        so no separate user_features tensor is needed.
        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        query: str,
        user_products: List[str],
        max_new_tokens: int = 50,
        num_beams: int = 5,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate personalized rewrites with products serialized in the prompt."""
        self.eval()
        device = next(self.parameters()).device

        prompt = self.build_prompt(query, user_products)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Strip the prompt portion
            if "rewrite:" in text:
                text = text.split("rewrite:", 1)[-1]
            if "→" in text:
                text = text.split("→", 1)[-1]
            results.append(text.strip())

        return list(dict.fromkeys(results))

    def rewrite(
        self,
        query: str,
        user_products: List[str],
        max_new_tokens: int = 50,
        num_beams: int = 5,
    ) -> DirectRewriteResult:
        """Rewrite with latency measurement."""
        start = time.perf_counter()

        prompt = self.build_prompt(query, user_products)
        prompt_tokens = len(self.tokenizer.encode(prompt))

        rewrites = self.generate(
            query, user_products,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return DirectRewriteResult(
            original_query=query,
            rewritten_query=rewrites[0] if rewrites else query,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            alternatives=rewrites[1:] if len(rewrites) > 1 else [],
        )

    def save_pretrained(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)
        if self.llm is not None:
            self.llm.save_pretrained(os.path.join(output_dir, "llm_lora"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    def get_param_count(self) -> Dict[str, int]:
        if self.llm is None:
            return {"trainable": 0, "total": 0}
        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.llm.parameters())
        return {"trainable": trainable, "total": total}
