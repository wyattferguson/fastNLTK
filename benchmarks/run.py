#!/usr/bin/env python3
"""
Benchmark runner — CLI entry point.

Usage:
    python -m benchmarks.run                          # Run + print table
    python -m benchmarks.run --save                   # Run + save to auto path
    python -m benchmarks.run --save results/latest.json
    python -m benchmarks.run --regression results/baseline.json
    python -m benchmarks.run --regression --threshold 0.20
    python -m benchmarks.run --ci                      # Run + save + check regressions
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.harness import (
    save_results,
    load_results,
    print_table,
    print_regression_table,
    check_regression,
    RESULTS_DIR,
)
from benchmarks.bench_suite import run_all


def main():
    parser = argparse.ArgumentParser(
        description="fastNLTK benchmark harness",
    )
    parser.add_argument(
        "--save", nargs="?", const="", default=None,
        help="Save results to JSON file (default: auto path)",
    )
    parser.add_argument(
        "--regression", nargs="?", const="", default=None,
        help="Compare results against a baseline JSON file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Regression threshold (fractional, default 0.15 = 15%%)",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: run, save, check regressions against last baseline",
    )
    args = parser.parse_args()

    # Resolve paths
    save_path = args.save
    if args.save == "":
        save_path = ""
    if args.regression == "":
        args.regression = ""  # will be resolved later

    # Run
    results = run_all()

    # Print table
    print_table(results, "fastNLTK Benchmarks")

    # Save
    if args.save is not None:
        path = save_results(results, save_path)
        print(f"\n  Results saved to: {path}")

    # Regression check
    baseline_path = args.regression
    if args.ci:
        # CI mode: find most recent baseline
        candidates = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
            reverse=True,
        )
        if candidates:
            baseline_path = os.path.join(RESULTS_DIR, candidates[0])
        else:
            # Save current as baseline and exit clean
            save_results(results, os.path.join(RESULTS_DIR, "baseline.json"))
            print("\n  No baseline found. Saved current as baseline.")
            return 0

    if baseline_path:
        if not os.path.exists(baseline_path):
            print(f"\n  Baseline not found: {baseline_path}")
            return 1
        baseline = load_results(baseline_path)
        regressions = check_regression(results, baseline.results, args.threshold)
        if regressions:
            print_regression_table(regressions)
            return 1
        else:
            print(f"\n  ✅ All within {args.threshold*100:.0f}% of baseline.")

    # Save in CI mode too (after comparison)
    if args.ci and baseline_path:
        save_results(results, os.path.join(RESULTS_DIR, "baseline.json"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
