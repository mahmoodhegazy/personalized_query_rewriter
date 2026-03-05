#!/usr/bin/env python3
"""
run_eval.py
===========
Run comprehensive evaluation on the trained personalized query rewriter.

Usage:
    python -m scripts.run_eval --model outputs/final --data data/raw/query_user_profile.csv
    python -m scripts.run_eval --model outputs/final --repeats 5  # Report variance
"""

import argparse
import json
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personalized_query_rewriter.data.data_loader import load_raw_data, split_data
from personalized_query_rewriter.data.feature_engineering import UserFeatureEncoder
from personalized_query_rewriter.models.personalized_rewriter import PersonalizedQueryRewriter
from personalized_query_rewriter.evaluation.evaluator import RewriterEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Personalized Query Rewriter")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint dir")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Override test data path")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Evaluation repeats (for variance estimation)")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json",
                        help="Output results file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = args.data or config["data"]["input_csv"]
    df = load_raw_data(data_path)
    _, _, test_df = split_data(df, seed=config["data"]["random_seed"])

    # Load model
    feature_encoder = UserFeatureEncoder(include_derived=True)
    config["ft_transformer"]["num_features"] = feature_encoder.num_features

    model = PersonalizedQueryRewriter(config, load_llm=True)
    model.load_components(args.model)
    model.to(device)
    model.eval()

    # Run evaluation
    evaluator = RewriterEvaluator(
        model=model,
        feature_encoder=feature_encoder,
        config=config.get("evaluation", {}),
    )

    logger.info("Running evaluation with %d repeats on %d test samples...",
                args.repeats, len(test_df))
    results = evaluator.evaluate(test_df, n_repeats=args.repeats)

    # Generate report
    report = evaluator.generate_report(results)
    print(report)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Convert non-serializable values
    def _serialize(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.item()
        if hasattr(obj, "__float__"):
            return float(obj)
        return str(obj)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)

    logger.info("Results saved to %s", args.output)

    # Save report
    report_path = args.output.replace(".json", ".txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
