"""
pipeline.py
===========
Production-ready inference pipeline for personalized query rewriting.

Latency budget allocation (~50ms total):
    - User embedding lookup:  <10ms (pre-computed, cached)
    - E2P projection:         <2ms
    - Personalization gate:   <1ms
    - LLM inference:          <25ms (INT8 quantized)
    - Network overhead:       <12ms

Optimizations:
    1. Pre-computed user embeddings cached in memory (or Redis in production).
    2. Semantic cache for query → rewrite (avoid redundant LLM calls).
    3. Gate bypasses LLM entirely for unambiguous queries.
    4. KV cache reuse for prefix-sharing queries (type-ahead).

Usage:
    pipeline = PersonalizedRewritePipeline.from_pretrained("outputs/grpo/final")
    result = pipeline.rewrite("rewards", user_products=["Sapphire Reserve", "Mobile Banking"])
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import yaml

from ..data.feature_engineering import UserFeatureEncoder
from ..models.personalized_rewriter import PersonalizedQueryRewriter
from .cache import SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """Result of a personalized query rewrite."""
    original_query: str
    rewritten_query: str
    was_personalized: bool
    gate_score: float
    latency_ms: float
    cache_hit: bool
    alternatives: List[str]


class PersonalizedRewritePipeline:
    """
    Production inference pipeline for personalized query rewriting.

    Handles the full flow:
        1. Check semantic cache.
        2. Encode user features (from cache or compute).
        3. Check personalization gate.
        4. Generate rewrite (if needed).
        5. Update caches.

    Args:
        model: Trained PersonalizedQueryRewriter.
        feature_encoder: UserFeatureEncoder instance.
        config: Inference configuration dict.
    """

    def __init__(
        self,
        model: PersonalizedQueryRewriter,
        feature_encoder: UserFeatureEncoder,
        config: Dict[str, Any] = None,
    ):
        self.model = model
        self.model.eval()
        self.feature_encoder = feature_encoder
        self.config = config or {}
        self.device = next(model.parameters()).device

        # User embedding cache: {user_id: (embedding_tensor, timestamp)}
        self.user_embed_cache: Dict[str, tuple] = {}
        self.user_embed_ttl = self.config.get("user_embedding_cache", {}).get(
            "ttl_seconds", 86400
        )

        # Semantic cache
        cache_cfg = self.config.get("semantic_cache", {})
        self.semantic_cache = None
        if cache_cfg.get("enabled", False):
            self.semantic_cache = SemanticCache(
                similarity_threshold=cache_cfg.get("similarity_threshold", 0.92),
                max_size=cache_cfg.get("max_cache_size", 100000),
                ttl_seconds=cache_cfg.get("ttl_seconds", 3600),
            )

        # Generation parameters
        self.max_new_tokens = self.config.get("max_new_tokens", 50)
        self.num_beams = self.config.get("num_beams", 5)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config_path: str = "config/config.yaml",
    ) -> "PersonalizedRewritePipeline":
        """Load pipeline from a saved model directory."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        feature_encoder = UserFeatureEncoder(include_derived=True)

        model = PersonalizedQueryRewriter(config, load_llm=True)
        model.load_components(model_dir)

        return cls(
            model=model,
            feature_encoder=feature_encoder,
            config=config.get("inference", {}),
        )

    def rewrite(
        self,
        query: str,
        user_products: List[str],
        user_id: Optional[str] = None,
        force_personalize: bool = False,
        num_alternatives: int = 1,
    ) -> RewriteResult:
        """
        Rewrite a query with personalization.

        Args:
            query: Original user query.
            user_products: List of user's Chase products.
            user_id: Optional user ID for caching.
            force_personalize: Skip gate, always personalize.
            num_alternatives: Number of alternative rewrites.

        Returns:
            RewriteResult with rewrite, metadata, and timing.
        """
        start_time = time.perf_counter()
        cache_hit = False

        # --- Step 1: Check semantic cache ---
        if self.semantic_cache is not None and user_id:
            cached = self.semantic_cache.get(query, user_id)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return RewriteResult(
                    original_query=query,
                    rewritten_query=cached,
                    was_personalized=True,
                    gate_score=1.0,
                    latency_ms=elapsed_ms,
                    cache_hit=True,
                    alternatives=[],
                )

        # --- Step 2: Encode user features ---
        user_features = self.feature_encoder.encode_to_tensor(user_products)

        # Check user embedding cache
        user_embed = None
        if user_id and user_id in self.user_embed_cache:
            cached_embed, cached_time = self.user_embed_cache[user_id]
            if (time.time() - cached_time) < self.user_embed_ttl:
                user_embed = cached_embed

        if user_embed is None:
            user_embed = self.model.user_encoder(
                user_features.unsqueeze(0).to(self.device)
            )
            if user_id:
                self.user_embed_cache[user_id] = (user_embed, time.time())

        # --- Step 3: Check personalization gate ---
        gate_score = 1.0
        should_personalize = force_personalize

        if not force_personalize:
            inputs = self.model.tokenizer(
                f"rewrite: {query}", return_tensors="pt"
            ).to(self.device)
            embed_layer = self.model.llm.get_input_embeddings()
            input_embeds = embed_layer(inputs["input_ids"])
            query_embed = input_embeds.mean(dim=1)

            gate_score = self.model.gate(
                user_embed, query_embed
            ).item()
            should_personalize = gate_score > self.model.gate.threshold

        # --- Step 4: Generate rewrite ---
        if should_personalize:
            rewrites = self.model.generate(
                query=query,
                user_features=user_features,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                num_return_sequences=max(num_alternatives, 1),
                use_gate=False,  # Already checked above
            )
            rewritten = rewrites[0] if rewrites else query
            alternatives = rewrites[1:] if len(rewrites) > 1 else []
        else:
            rewritten = query
            alternatives = []

        # --- Step 5: Update cache ---
        if self.semantic_cache is not None and user_id and should_personalize:
            self.semantic_cache.put(query, user_id, rewritten)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            was_personalized=should_personalize,
            gate_score=gate_score,
            latency_ms=elapsed_ms,
            cache_hit=cache_hit,
            alternatives=alternatives,
        )

    def rewrite_batch(
        self,
        queries: List[str],
        user_products_list: List[List[str]],
        user_ids: Optional[List[str]] = None,
    ) -> List[RewriteResult]:
        """Rewrite a batch of queries."""
        if user_ids is None:
            user_ids = [None] * len(queries)

        return [
            self.rewrite(q, up, uid)
            for q, up, uid in zip(queries, user_products_list, user_ids)
        ]

    def benchmark_latency(
        self,
        queries: List[str],
        user_products: List[str],
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Returns p50, p90, p99, and mean latency in milliseconds.
        """
        latencies = []

        for _ in range(n_runs):
            result = self.rewrite(queries[0], user_products)
            latencies.append(result.latency_ms)

        latencies = sorted(latencies)
        n = len(latencies)

        return {
            "p50_ms": latencies[n // 2],
            "p90_ms": latencies[int(n * 0.9)],
            "p99_ms": latencies[int(n * 0.99)],
            "mean_ms": sum(latencies) / n,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
        }
