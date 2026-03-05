"""
evaluator.py
============
End-to-end evaluation pipeline for the personalized query rewriter.

Runs the complete evaluation suite:
    1. Generate rewrites for test queries (with and without personalization).
    2. Retrieve results using the search index (or simulated).
    3. Compute all metrics (retrieval, rewrite quality, personalization).
    4. Stratified analysis by user segment.
    5. Compare personalized vs non-personalized baselines.
    6. Repeat N times and report variance (SIGIR-AP 2025 recommendation).
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..data.feature_engineering import UserFeatureEncoder
from ..models.personalized_rewriter import PersonalizedQueryRewriter
from . import metrics

logger = logging.getLogger(__name__)


class RewriterEvaluator:
    """
    Comprehensive evaluator for the personalized query rewriter.

    Args:
        model: Trained PersonalizedQueryRewriter.
        feature_encoder: UserFeatureEncoder instance.
        search_fn: Callable that simulates search retrieval.
            Signature: fn(query: str) -> List[str] (returns ranked navlinks).
        config: Evaluation configuration dict.
    """

    def __init__(
        self,
        model: PersonalizedQueryRewriter,
        feature_encoder: UserFeatureEncoder,
        search_fn=None,
        config: Dict[str, Any] = None,
    ):
        self.model = model
        self.feature_encoder = feature_encoder
        self.search_fn = search_fn
        self.config = config or {}
        self.device = next(model.parameters()).device

    def evaluate(
        self,
        test_df: pd.DataFrame,
        n_repeats: int = 1,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on test set.

        Following SIGIR-AP 2025: repeat N times and report mean ± std
        to account for LLM generation stochasticity.

        Args:
            test_df: Test DataFrame with query, user products, gold navlink.
            n_repeats: Number of evaluation repeats.

        Returns:
            Dict with metric results, comparisons, and stratified analysis.
        """
        all_results = []

        for run in range(n_repeats):
            logger.info("Evaluation run %d/%d", run + 1, n_repeats)
            result = self._single_evaluation(test_df, seed=42 + run)
            all_results.append(result)

        # Aggregate across runs
        aggregated = self._aggregate_results(all_results)

        # Add comparison: personalized vs non-personalized
        comparison = self._compare_baselines(test_df)
        aggregated["baseline_comparison"] = comparison

        return aggregated

    def _single_evaluation(
        self,
        test_df: pd.DataFrame,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Run a single evaluation pass."""
        torch.manual_seed(seed)
        self.model.eval()

        # Collect predictions
        original_queries = []
        rewritten_queries = []
        gold_navlinks = []
        predicted_rankings = []
        clicked_list = []
        segment_labels = []

        for _, row in test_df.iterrows():
            query = row.get("query_clean", "")
            products = row.get("user_product_list", [])
            if isinstance(products, str):
                products = [p.strip() for p in products.split(",") if p.strip()]
            gold_nav = row.get("navlink", row.get("Selected_Navlink", ""))
            was_clicked = row.get("clicked_bool", False)

            # Generate personalized rewrite
            user_features = self.feature_encoder.encode_to_tensor(products)
            rewrites = self.model.generate(
                query=query,
                user_features=user_features,
                use_gate=True,
            )
            rewritten = rewrites[0] if rewrites else query

            # Retrieve results for rewritten query
            if self.search_fn is not None:
                ranking = self.search_fn(rewritten)
            else:
                ranking = []

            original_queries.append(query)
            rewritten_queries.append(rewritten)
            gold_navlinks.append(gold_nav)
            predicted_rankings.append(ranking)
            clicked_list.append(was_clicked)

            # Segment label
            segment = self._get_segment(products)
            segment_labels.append(segment)

        # Compute metrics
        result = {}

        # Retrieval metrics
        for k in [5, 10]:
            result[f"ndcg@{k}"] = metrics.compute_ndcg(
                gold_navlinks, predicted_rankings, k
            )
            result[f"mrr@{k}"] = metrics.compute_mrr(
                gold_navlinks, predicted_rankings, k
            )
            result[f"recall@{k}"] = metrics.compute_recall_at_k(
                gold_navlinks, predicted_rankings, k
            )

        # Rewrite quality (compare to gold = original query for passthrough,
        # or to reformulated query if available)
        result["bleu"] = metrics.compute_bleu(original_queries, rewritten_queries)
        result["rouge_l"] = metrics.compute_rouge_l(original_queries, rewritten_queries)

        # Personalization metrics
        ctr_metrics = metrics.compute_click_through_rate(clicked_list)
        result.update(ctr_metrics)

        # Rewrite diversity: fraction of queries that were actually rewritten
        n_rewritten = sum(
            1 for o, r in zip(original_queries, rewritten_queries) if o != r
        )
        result["rewrite_rate"] = n_rewritten / max(len(original_queries), 1)

        # Stratified metrics
        result["stratified"] = metrics.compute_stratified_metrics(
            gold_navlinks, predicted_rankings, segment_labels, k=10
        )

        return result

    def _compare_baselines(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compare personalized vs non-personalized rewriting.

        Runs evaluation with:
            1. Personalized (gate ON): Full pipeline.
            2. Personalized (gate OFF): Always personalize.
            3. Non-personalized: No user features.
        """
        comparisons = {}

        # Personalized with gate
        comparisons["personalized_gated"] = self._single_evaluation(test_df)

        # Non-personalized baseline
        logger.info("Running non-personalized baseline...")
        self.model.eval()

        gold_navlinks, predicted_rankings = [], []

        for _, row in test_df.iterrows():
            query = row.get("query_clean", "")
            gold_nav = row.get("navlink", row.get("Selected_Navlink", ""))

            # Generate WITHOUT user features
            rewrites = self.model.generate(query=query, user_features=None)
            rewritten = rewrites[0] if rewrites else query

            if self.search_fn is not None:
                ranking = self.search_fn(rewritten)
            else:
                ranking = []

            gold_navlinks.append(gold_nav)
            predicted_rankings.append(ranking)

        comparisons["non_personalized"] = {
            "ndcg@10": metrics.compute_ndcg(gold_navlinks, predicted_rankings, 10),
            "mrr@10": metrics.compute_mrr(gold_navlinks, predicted_rankings, 10),
            "recall@10": metrics.compute_recall_at_k(gold_navlinks, predicted_rankings, 10),
        }

        # Compute deltas
        for metric in ["ndcg@10", "mrr@10", "recall@10"]:
            p_val = comparisons["personalized_gated"].get(metric, 0)
            np_val = comparisons["non_personalized"].get(metric, 0)
            delta = p_val - np_val
            comparisons[f"delta_{metric}"] = delta
            logger.info("  %s: personalized=%.4f, baseline=%.4f, delta=%+.4f",
                         metric, p_val, np_val, delta)

        return comparisons

    def _aggregate_results(
        self, results: List[Dict],
    ) -> Dict[str, Any]:
        """Aggregate results across multiple runs, reporting mean ± std."""
        if len(results) == 1:
            return results[0]

        aggregated = {}
        numeric_keys = [
            k for k in results[0]
            if isinstance(results[0][k], (int, float))
        ]

        for key in numeric_keys:
            values = [r[key] for r in results]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return aggregated

    def _get_segment(self, products: List[str]) -> str:
        """Assign user to a segment based on their products."""
        product_set = set(products)

        has_premium = bool(product_set & {
            "Sapphire Reserve", "Sapphire Preferred", "Chase Private Client"
        })
        has_business = bool(product_set & {
            "Ink Business Card", "Business Checking", "Business Credit Card"
        })
        has_mortgage = "Mortgage" in product_set
        has_investment = "Investment Account" in product_set

        if has_premium:
            return "premium"
        elif has_business:
            return "business"
        elif has_mortgage or has_investment:
            return "wealth"
        elif len(products) <= 2:
            return "basic"
        else:
            return "standard"

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        lines = ["=" * 60, "PERSONALIZED QUERY REWRITER — EVALUATION REPORT", "=" * 60, ""]

        # Overall metrics
        lines.append("OVERALL METRICS")
        lines.append("-" * 40)
        for key, val in results.items():
            if isinstance(val, dict) and "mean" in val:
                lines.append(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")
            elif isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")

        # Baseline comparison
        if "baseline_comparison" in results:
            comp = results["baseline_comparison"]
            lines.extend(["", "BASELINE COMPARISON", "-" * 40])
            for key in ["delta_ndcg@10", "delta_mrr@10", "delta_recall@10"]:
                if key in comp:
                    lines.append(f"  {key}: {comp[key]:+.4f}")

        # Stratified
        for key, val in results.items():
            if key == "stratified" and isinstance(val, dict):
                lines.extend(["", "STRATIFIED ANALYSIS", "-" * 40])
                for seg, seg_metrics in val.items():
                    lines.append(f"  [{seg}] ({seg_metrics.get('count', '?')} queries)")
                    for mk, mv in seg_metrics.items():
                        if mk != "count":
                            lines.append(f"    {mk}: {mv:.4f}")

        lines.append("")
        return "\n".join(lines)
