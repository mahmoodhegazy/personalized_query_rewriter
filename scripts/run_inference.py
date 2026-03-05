#!/usr/bin/env python3
"""
run_inference.py
================
Run personalized query rewriting inference with interactive or batch mode.

Usage:
    # Interactive mode
    python -m scripts.run_inference --model outputs/final --interactive

    # Batch mode from CSV
    python -m scripts.run_inference --model outputs/final --input queries.csv --output rewrites.csv

    # Single query
    python -m scripts.run_inference --model outputs/final \\
        --query "rewards" \\
        --products "Sapphire Reserve,Mobile Banking"

    # Latency benchmark
    python -m scripts.run_inference --model outputs/final --benchmark
"""

import argparse
import csv
import json
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personalized_query_rewriter.data.feature_engineering import UserFeatureEncoder
from personalized_query_rewriter.models.personalized_rewriter import PersonalizedQueryRewriter
from personalized_query_rewriter.inference.pipeline import PersonalizedRewritePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Personalized Query Rewriting")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint dir")
    parser.add_argument("--config", type=str, default="config/config.yaml")

    # Mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interactive", action="store_true", help="Interactive mode")
    group.add_argument("--query", type=str, help="Single query to rewrite")
    group.add_argument("--input", type=str, help="Input CSV for batch mode")
    group.add_argument("--benchmark", action="store_true", help="Latency benchmark")

    parser.add_argument("--products", type=str, default="",
                        help="Comma-separated user products (for --query mode)")
    parser.add_argument("--output", type=str, default="outputs/rewrites.csv",
                        help="Output CSV for batch mode")
    parser.add_argument("--num-alternatives", type=int, default=3,
                        help="Number of alternative rewrites")

    return parser.parse_args()


def load_pipeline(args) -> PersonalizedRewritePipeline:
    """Load the inference pipeline."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    feature_encoder = UserFeatureEncoder(include_derived=True)
    config["ft_transformer"]["num_features"] = feature_encoder.num_features

    model = PersonalizedQueryRewriter(config, load_llm=True)
    model.load_components(args.model)
    model.eval()

    pipeline = PersonalizedRewritePipeline(
        model=model,
        feature_encoder=feature_encoder,
        config=config.get("inference", {}),
    )

    return pipeline


def run_single(pipeline, query, products, num_alternatives=3):
    """Run a single query and display results."""
    product_list = [p.strip() for p in products.split(",") if p.strip()]

    result = pipeline.rewrite(
        query=query,
        user_products=product_list,
        num_alternatives=num_alternatives,
    )

    print(f"\n{'='*60}")
    print(f"  Original:      {result.original_query}")
    print(f"  Rewritten:     {result.rewritten_query}")
    print(f"  Personalized:  {result.was_personalized}")
    print(f"  Gate Score:    {result.gate_score:.3f}")
    print(f"  Latency:       {result.latency_ms:.1f}ms")
    print(f"  Cache Hit:     {result.cache_hit}")
    if result.alternatives:
        print(f"  Alternatives:")
        for i, alt in enumerate(result.alternatives, 1):
            print(f"    {i}. {alt}")
    print(f"{'='*60}\n")

    return result


def run_interactive(pipeline):
    """Interactive REPL mode."""
    print("\n" + "=" * 60)
    print("  PERSONALIZED QUERY REWRITER — Interactive Mode")
    print("  Type 'quit' to exit, 'products <list>' to set products")
    print("=" * 60 + "\n")

    current_products = "Personal Checking, Debit Card, Mobile Banking"
    print(f"  Current products: {current_products}")
    print()

    while True:
        try:
            user_input = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower().startswith("products "):
            current_products = user_input[9:].strip()
            print(f"  Updated products: {current_products}\n")
            continue

        run_single(pipeline, user_input, current_products)


def run_batch(pipeline, input_path, output_path, num_alternatives=3):
    """Batch mode: read queries from CSV, write rewrites."""
    import pandas as pd

    df = pd.read_csv(input_path)
    results = []

    for idx, row in df.iterrows():
        query = row.get("Query", row.get("query", ""))
        products = row.get("User Products", row.get("products", ""))

        if isinstance(products, str):
            product_list = [p.strip() for p in products.split(",") if p.strip()]
        else:
            product_list = []

        result = pipeline.rewrite(
            query=query,
            user_products=product_list,
            num_alternatives=num_alternatives,
        )

        results.append({
            "original_query": result.original_query,
            "rewritten_query": result.rewritten_query,
            "was_personalized": result.was_personalized,
            "gate_score": round(result.gate_score, 3),
            "latency_ms": round(result.latency_ms, 1),
            "alternatives": "; ".join(result.alternatives),
        })

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d/%d queries", idx + 1, len(df))

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Batch results saved to %s (%d queries)", output_path, len(results_df))


def run_benchmark(pipeline):
    """Latency benchmark."""
    print("\n" + "=" * 60)
    print("  LATENCY BENCHMARK")
    print("=" * 60)

    test_cases = [
        ("rewards", ["Sapphire Reserve", "Mobile Banking"]),
        ("check balance", ["Personal Checking", "Debit Card"]),
        ("travel", ["Sapphire Reserve", "United Card", "Ultimate Rewards"]),
        ("fees", ["Chase Total Checking", "Online Banking"]),
        ("mortgage payment", ["Mortgage", "Personal Checking"]),
    ]

    for query, products in test_cases:
        latency_stats = pipeline.benchmark_latency(
            queries=[query],
            user_products=products,
            n_runs=50,
        )
        print(f"\n  Query: '{query}'")
        print(f"    Products: {', '.join(products)}")
        print(f"    p50={latency_stats['p50_ms']:.1f}ms, "
              f"p90={latency_stats['p90_ms']:.1f}ms, "
              f"p99={latency_stats['p99_ms']:.1f}ms, "
              f"mean={latency_stats['mean_ms']:.1f}ms")

    print(f"\n{'='*60}\n")


def main():
    args = parse_args()
    pipeline = load_pipeline(args)

    if args.interactive:
        run_interactive(pipeline)
    elif args.query:
        run_single(pipeline, args.query, args.products, args.num_alternatives)
    elif args.input:
        run_batch(pipeline, args.input, args.output, args.num_alternatives)
    elif args.benchmark:
        run_benchmark(pipeline)
    else:
        # Default: run demo examples
        print("\nRunning demo examples...\n")
        demos = [
            ("rewards", "Sapphire Reserve, Chase Freedom Unlimited, Ultimate Rewards"),
            ("rewards", "Personal Checking, Debit Card"),
            ("fees", "Sapphire Reserve, Mobile Banking"),
            ("fees", "Student Checking, High School Checking"),
            ("transfer", "Personal Checking, Personal Savings, Mobile Banking"),
            ("increase limit", "Chase Freedom, Personal Credit Card"),
            ("travel", "Sapphire Reserve, United Card, Ultimate Rewards"),
            ("travel", "Personal Checking, Auto Loan"),
        ]
        for query, products in demos:
            run_single(pipeline, query, products)


if __name__ == "__main__":
    main()
