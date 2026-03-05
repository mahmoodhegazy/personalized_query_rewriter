"""
test_training.py
================
Tests for training pipeline components (without requiring GPU or LLM).
Tests data flow, loss computation, and collation logic.
"""

import pytest
import torch

from personalized_query_rewriter.data.dataset import (
    _align_length,
    cpt_collate_fn,
)
from personalized_query_rewriter.evaluation.metrics import (
    compute_ndcg,
    compute_mrr,
    compute_recall_at_k,
    compute_bleu,
    compute_rouge_l,
    compute_reformulation_rate,
    compute_click_through_rate,
    compute_stratified_metrics,
)
from personalized_query_rewriter.evaluation.bias_correction import (
    InversePropensityWeighting,
    RelevanceSaturationCorrector,
)
from personalized_query_rewriter.inference.cache import SemanticCache


# =============================================================================
# Dataset Utility Tests
# =============================================================================

class TestDatasetUtils:
    def test_align_length_pad(self):
        labels = [1, 2, 3]
        result = _align_length(labels, 5)
        assert result == [1, 2, 3, -100, -100]

    def test_align_length_truncate(self):
        labels = [1, 2, 3, 4, 5]
        result = _align_length(labels, 3)
        assert result == [1, 2, 3]

    def test_align_length_exact(self):
        labels = [1, 2, 3]
        result = _align_length(labels, 3)
        assert result == [1, 2, 3]

    def test_collate_fn(self):
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        ]
        result = cpt_collate_fn(batch, pad_token_id=0)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)
        # Second sample should be padded
        assert result["input_ids"][1, 2].item() == 0
        assert result["attention_mask"][1, 2].item() == 0
        assert result["labels"][1, 2].item() == -100


# =============================================================================
# Evaluation Metrics Tests
# =============================================================================

class TestMetrics:
    def test_ndcg_perfect(self):
        gold = ["/a", "/b"]
        pred = [["/a", "/c"], ["/b", "/d"]]
        ndcg = compute_ndcg(gold, pred, k=5)
        assert ndcg == 1.0  # Gold at position 0 for both

    def test_ndcg_partial(self):
        gold = ["/a", "/b"]
        pred = [["/c", "/a"], ["/b", "/d"]]
        ndcg = compute_ndcg(gold, pred, k=5)
        assert 0.5 < ndcg < 1.0  # First has gold at pos 1, second at pos 0

    def test_ndcg_zero(self):
        gold = ["/a"]
        pred = [["/b", "/c"]]
        ndcg = compute_ndcg(gold, pred, k=2)
        assert ndcg == 0.0

    def test_mrr_perfect(self):
        gold = ["/a", "/b"]
        pred = [["/a", "/c"], ["/b", "/d"]]
        mrr = compute_mrr(gold, pred, k=5)
        assert mrr == 1.0

    def test_mrr_second_position(self):
        gold = ["/a"]
        pred = [["/b", "/a"]]
        mrr = compute_mrr(gold, pred, k=5)
        assert abs(mrr - 0.5) < 0.01

    def test_recall_at_k(self):
        gold = ["/a", "/b", "/c"]
        pred = [["/a", "/d"], ["/d", "/e"], ["/c", "/b"]]
        recall = compute_recall_at_k(gold, pred, k=2)
        assert abs(recall - 2/3) < 0.01

    def test_bleu_identical(self):
        refs = ["check my balance"]
        hyps = ["check my balance"]
        bleu = compute_bleu(refs, hyps)
        assert bleu == 1.0

    def test_bleu_different(self):
        refs = ["check my balance"]
        hyps = ["what is the weather today"]
        bleu = compute_bleu(refs, hyps)
        assert bleu < 0.1

    def test_rouge_l(self):
        refs = ["check my account balance"]
        hyps = ["check account balance"]
        rouge = compute_rouge_l(refs, hyps)
        assert 0.5 < rouge < 1.0

    def test_reformulation_rate(self):
        original = ["rewards", "balance", "fees"]
        follow_up = ["sapphire rewards", None, None]
        rate = compute_reformulation_rate(original, follow_up)
        assert abs(rate - 1/3) < 0.01

    def test_click_through_rate(self):
        clicked = [True, True, False, False, True]
        result = compute_click_through_rate(clicked)
        assert abs(result["ctr"] - 0.6) < 0.01

    def test_stratified_metrics(self):
        gold = ["/a", "/b", "/c", "/d"]
        pred = [["/a"], ["/b"], ["/x"], ["/d"]]
        segments = ["premium", "premium", "basic", "basic"]
        results = compute_stratified_metrics(gold, pred, segments, k=5)
        assert "premium" in results
        assert "basic" in results
        assert results["premium"]["count"] == 2
        assert results["basic"]["count"] == 2


# =============================================================================
# Bias Correction Tests
# =============================================================================

class TestBiasCorrection:
    def test_ipw_power_law(self):
        ipw = InversePropensityWeighting(method="power_law", eta=1.0)
        # Position 1 should have lowest weight (highest propensity)
        w1 = ipw.get_weight(1)
        w5 = ipw.get_weight(5)
        assert w1 < w5  # Lower position = higher propensity = lower weight

    def test_ipw_correction(self):
        ipw = InversePropensityWeighting()
        weights = ipw.correct_click_labels([1, 3, 5])
        assert len(weights) == 3
        assert all(w > 0 for w in weights)
        # Weights should increase with position
        assert weights[0] <= weights[1] <= weights[2]

    def test_relevance_saturation(self):
        corrector = RelevanceSaturationCorrector()
        # Item right after last click should be unreliable negative
        w_near = corrector.get_negative_weight(position=3, last_click_position=2, total_clicks=1)
        w_far = corrector.get_negative_weight(position=10, last_click_position=2, total_clicks=1)
        assert w_near < w_far  # Near items are less reliable negatives

    def test_saturation_many_clicks(self):
        corrector = RelevanceSaturationCorrector()
        # With 3+ clicks, all negatives are reliable
        w = corrector.get_negative_weight(position=3, last_click_position=2, total_clicks=3)
        assert w == 1.0


# =============================================================================
# Semantic Cache Tests
# =============================================================================

class TestSemanticCache:
    def test_put_get(self):
        cache = SemanticCache()
        cache.put("rewards", "premium", "sapphire rewards balance")
        result = cache.get("rewards", "premium")
        assert result == "sapphire rewards balance"

    def test_miss(self):
        cache = SemanticCache()
        result = cache.get("rewards", "premium")
        assert result is None

    def test_different_segments(self):
        cache = SemanticCache()
        cache.put("rewards", "premium", "sapphire rewards")
        cache.put("rewards", "basic", "freedom rewards")
        assert cache.get("rewards", "premium") == "sapphire rewards"
        assert cache.get("rewards", "basic") == "freedom rewards"

    def test_eviction(self):
        cache = SemanticCache(max_size=3)
        cache.put("q1", "s1", "r1")
        cache.put("q2", "s1", "r2")
        cache.put("q3", "s1", "r3")
        cache.put("q4", "s1", "r4")  # Should evict q1
        assert cache.get("q1", "s1") is None
        assert cache.get("q4", "s1") == "r4"

    def test_stats(self):
        cache = SemanticCache()
        cache.put("q1", "s1", "r1")
        cache.get("q1", "s1")  # Hit
        cache.get("q2", "s1")  # Miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
