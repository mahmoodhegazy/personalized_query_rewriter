"""
metrics.py
==========
Evaluation metrics for personalized query rewriting.

Four metric families (following the survey recommendations):

1. Retrieval Quality: NDCG@K, MRR@K, Recall@K
   - Measured on held-out click data with temporal splits.

2. Rewrite Quality: BLEU, ROUGE-L
   - Compare generated rewrites to gold reformulations from session logs.

3. Personalization-Specific: Reformulation rate, CTR, dwell time.
   - WeWrite-style metrics measuring actual search improvement.

4. Stratified Analysis: Metrics broken down by user segment.
   - Detect "domain seesaw" where gains for one segment hurt another.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Retrieval Quality Metrics
# =============================================================================

def compute_ndcg(
    gold_navlinks: List[str],
    predicted_rankings: List[List[str]],
    k: int = 10,
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    For binary relevance (clicked/not clicked), computes the gain from
    having the gold navlink at each position in the predicted ranking.

    Args:
        gold_navlinks: List of gold navlink strings (one per query).
        predicted_rankings: List of ranked navlink lists (one per query).
        k: Cutoff rank.

    Returns:
        Mean NDCG@K across all queries.
    """
    ndcg_scores = []
    for gold, predicted in zip(gold_navlinks, predicted_rankings):
        if not gold:
            continue

        # DCG: binary relevance
        dcg = 0.0
        for i, nav in enumerate(predicted[:k]):
            if nav == gold:
                dcg += 1.0 / math.log2(i + 2)  # Position 0 → log2(2)

        # Ideal DCG: gold is at position 0
        idcg = 1.0 / math.log2(2)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_mrr(
    gold_navlinks: List[str],
    predicted_rankings: List[List[str]],
    k: int = 10,
) -> float:
    """
    Compute MRR@K (Mean Reciprocal Rank).

    Args:
        gold_navlinks: Gold navlink per query.
        predicted_rankings: Ranked predictions per query.
        k: Cutoff rank.

    Returns:
        Mean Reciprocal Rank.
    """
    rr_scores = []
    for gold, predicted in zip(gold_navlinks, predicted_rankings):
        if not gold:
            continue

        rr = 0.0
        for i, nav in enumerate(predicted[:k]):
            if nav == gold:
                rr = 1.0 / (i + 1)
                break

        rr_scores.append(rr)

    return np.mean(rr_scores) if rr_scores else 0.0


def compute_recall_at_k(
    gold_navlinks: List[str],
    predicted_rankings: List[List[str]],
    k: int = 10,
) -> float:
    """
    Compute Recall@K (fraction of queries where gold is in top-K).

    Args:
        gold_navlinks: Gold navlink per query.
        predicted_rankings: Ranked predictions per query.
        k: Cutoff rank.

    Returns:
        Recall@K score.
    """
    hits = 0
    total = 0
    for gold, predicted in zip(gold_navlinks, predicted_rankings):
        if not gold:
            continue
        total += 1
        if gold in predicted[:k]:
            hits += 1

    return hits / max(total, 1)


# =============================================================================
# 2. Rewrite Quality Metrics
# =============================================================================

def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
) -> float:
    """
    Compute corpus-level BLEU score.

    Simplified implementation for query-length texts (typically 2-10 tokens).
    Uses modified precision with brevity penalty.

    Args:
        references: Gold rewrite strings.
        hypotheses: Predicted rewrite strings.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score [0, 1].
    """
    def _get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    total_match = [0] * max_n
    total_count = [0] * max_n
    total_ref_len = 0
    total_hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()

        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = _get_ngrams(ref_tokens, n)
            hyp_ngrams = _get_ngrams(hyp_tokens, n)

            if not hyp_ngrams:
                continue

            ref_counts = defaultdict(int)
            for ng in ref_ngrams:
                ref_counts[ng] += 1

            matches = 0
            for ng in hyp_ngrams:
                if ref_counts[ng] > 0:
                    matches += 1
                    ref_counts[ng] -= 1

            total_match[n-1] += matches
            total_count[n-1] += len(hyp_ngrams)

    # Compute modified precision per n-gram
    precisions = []
    for n in range(max_n):
        if total_count[n] > 0:
            precisions.append(total_match[n] / total_count[n])
        else:
            precisions.append(0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    if total_hyp_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - total_ref_len / total_hyp_len))

    return bp * math.exp(log_avg)


def compute_rouge_l(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence based F1).

    Args:
        references: Gold rewrite strings.
        hypotheses: Predicted rewrite strings.

    Returns:
        Mean ROUGE-L F1 score.
    """
    def _lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()

        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        lcs = _lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        scores.append(f1)

    return np.mean(scores) if scores else 0.0


# =============================================================================
# 3. Personalization-Specific Metrics
# =============================================================================

def compute_reformulation_rate(
    original_queries: List[str],
    follow_up_queries: List[Optional[str]],
) -> float:
    """
    Compute query reformulation rate.

    Lower is better: if the personalized rewrite works, users shouldn't
    need to reformulate their query.

    Args:
        original_queries: Initial queries.
        follow_up_queries: Subsequent queries (None if no reformulation).

    Returns:
        Fraction of queries that were reformulated.
    """
    reformulated = sum(1 for q in follow_up_queries if q is not None)
    return reformulated / max(len(original_queries), 1)


def compute_click_through_rate(
    clicked: List[bool],
    dwell_times: Optional[List[float]] = None,
    min_dwell_time: float = 10.0,
) -> Dict[str, float]:
    """
    Compute click-through metrics.

    Following WeWrite: count click-throughs with >10 seconds dwell time
    as quality clicks (not bounces).

    Args:
        clicked: Whether each query resulted in a click.
        dwell_times: Time spent on clicked page (seconds). None if unavailable.
        min_dwell_time: Minimum dwell time for quality click.

    Returns:
        Dict with 'ctr', 'quality_ctr'.
    """
    total = len(clicked)
    if total == 0:
        return {"ctr": 0.0, "quality_ctr": 0.0}

    ctr = sum(clicked) / total

    quality_ctr = ctr  # Default if no dwell time data
    if dwell_times is not None:
        quality_clicks = sum(
            1 for c, d in zip(clicked, dwell_times)
            if c and d is not None and d >= min_dwell_time
        )
        quality_ctr = quality_clicks / total

    return {"ctr": ctr, "quality_ctr": quality_ctr}


# =============================================================================
# 4. Stratified Analysis
# =============================================================================

def compute_stratified_metrics(
    gold_navlinks: List[str],
    predicted_rankings: List[List[str]],
    segment_labels: List[str],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute retrieval metrics stratified by user segment.

    Detects the "domain seesaw" problem: gains for one segment at another's
    expense.

    Args:
        gold_navlinks: Gold navlinks per query.
        predicted_rankings: Predicted rankings per query.
        segment_labels: Segment name per query (e.g., "premium", "basic").
        k: Cutoff rank.

    Returns:
        Nested dict: {segment: {metric: value}}.
    """
    # Group by segment
    segments = defaultdict(lambda: {"gold": [], "predicted": []})
    for gold, pred, seg in zip(gold_navlinks, predicted_rankings, segment_labels):
        segments[seg]["gold"].append(gold)
        segments[seg]["predicted"].append(pred)

    results = {}
    for seg_name, data in segments.items():
        results[seg_name] = {
            f"ndcg@{k}": compute_ndcg(data["gold"], data["predicted"], k),
            f"mrr@{k}": compute_mrr(data["gold"], data["predicted"], k),
            f"recall@{k}": compute_recall_at_k(data["gold"], data["predicted"], k),
            "count": len(data["gold"]),
        }

    return results
