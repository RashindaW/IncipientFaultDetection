#!/usr/bin/env python3
"""Compare baseline methods against DySTGAT.

This script loads trained models or their saved metrics and generates
comparison tables in Markdown and LaTeX formats.

Usage:
    # Compare all trained baselines
    python compare_baselines.py --results-dir results/ims-raw \
        --output comparison_table.md

    # Compare specific methods
    python compare_baselines.py --methods lstm_vae,usad,gdn,dystgat \
        --results-dir results/ims-raw

    # Generate LaTeX table
    python compare_baselines.py --results-dir results/ims-raw \
        --format latex --output comparison_table.tex
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from baselines import list_baselines, get_baseline_description
from baselines.utils.metrics import format_metrics_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline methods against DySTGAT"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing trained model results/metrics",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to compare (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for comparison table",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "latex", "csv", "all"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="faults_all",
        help="Test set to compare (default: faults_all)",
    )
    parser.add_argument(
        "--include-dystgat",
        action="store_true",
        default=True,
        help="Include DySTGAT results if available",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="auc_roc",
        choices=["auc_roc", "f1", "best_f1", "tea_auc", "tea_best_f1", "name"],
        help="Metric to sort results by",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)",
    )

    return parser.parse_args()


def load_metrics_csv(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load metrics from CSV file.

    Args:
        filepath: Path to metrics CSV

    Returns:
        Dict mapping test_set names to metrics dicts
    """
    metrics = {}

    if not os.path.exists(filepath):
        return metrics

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_set = row.get("test_set", "unknown")
            metrics[test_set] = {
                k: float(v) if v and k not in ["method", "test_set"] else v
                for k, v in row.items()
                if k not in ["test_set"]
            }

    return metrics


def find_metrics_files(
    results_dir: str, methods: Optional[List[str]] = None
) -> Dict[str, str]:
    """Find metrics files for each method.

    Args:
        results_dir: Directory to search
        methods: Optional list of methods to look for

    Returns:
        Dict mapping method names to metrics file paths
    """
    found = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return found

    # Look for standard naming patterns
    patterns = [
        "{method}_metrics.csv",
        "{method}/metrics.csv",
        "{method}/detailed_test_metrics.csv",
        "metrics_{method}.csv",
    ]

    search_methods = methods or list_baselines() + ["dystgat", "dyedgegat"]

    for method in search_methods:
        for pattern in patterns:
            filepath = results_path / pattern.format(method=method)
            if filepath.exists():
                found[method] = str(filepath)
                break

    return found


def load_all_metrics(
    results_dir: str,
    methods: Optional[List[str]] = None,
    test_set: str = "faults_all",
) -> Dict[str, Dict[str, float]]:
    """Load metrics for all methods.

    Args:
        results_dir: Directory containing results
        methods: Optional list of methods to load
        test_set: Which test set to extract metrics for

    Returns:
        Dict mapping method names to their metrics
    """
    all_metrics = {}

    # Find metrics files
    metrics_files = find_metrics_files(results_dir, methods)

    for method, filepath in metrics_files.items():
        method_metrics = load_metrics_csv(filepath)
        if test_set in method_metrics:
            all_metrics[method] = method_metrics[test_set]
        elif method_metrics:
            # Take first available test set
            first_key = next(iter(method_metrics))
            all_metrics[method] = method_metrics[first_key]

    return all_metrics


def format_latex_table(
    metrics_dict: Dict[str, Dict[str, float]],
    columns: List[str] = None,
    column_headers: List[str] = None,
    highlight_best: bool = True,
) -> str:
    """Format metrics as a LaTeX table with best results highlighted.

    Args:
        metrics_dict: Dict mapping method names to metrics
        columns: List of metric column keys
        column_headers: Display names for columns
        highlight_best: Bold the best value in each column

    Returns:
        LaTeX table string
    """
    if columns is None:
        columns = ["auc_roc", "f1", "best_f1", "tea_auc", "tea_best_f1"]
    if column_headers is None:
        column_headers = ["AUC-ROC", "F1", "Best F1", "TEA AUC", "TEA F1"]

    # Find best values for each column
    best_values = {}
    if highlight_best:
        for col in columns:
            values = [m.get(col, 0) for m in metrics_dict.values()]
            if values:
                best_values[col] = max(values)

    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(columns) + "}")
    lines.append(r"\toprule")
    lines.append("Method & " + " & ".join(column_headers) + r" \\")
    lines.append(r"\midrule")

    for method, metrics in metrics_dict.items():
        # Format method name
        display_name = method.replace("_", "-").upper()
        if method == "dystgat":
            display_name = r"\textbf{DySTGAT (Ours)}"

        values = []
        for col in columns:
            val = metrics.get(col, 0.0)
            formatted = f"{val:.4f}"

            # Highlight best
            if highlight_best and col in best_values:
                if abs(val - best_values[col]) < 1e-6:
                    formatted = r"\textbf{" + formatted + "}"

            values.append(formatted)

        row = f"{display_name} & " + " & ".join(values) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def format_markdown_table(
    metrics_dict: Dict[str, Dict[str, float]],
    columns: List[str] = None,
    column_headers: List[str] = None,
    highlight_best: bool = True,
) -> str:
    """Format metrics as a Markdown table with best results highlighted.

    Args:
        metrics_dict: Dict mapping method names to metrics
        columns: List of metric column keys
        column_headers: Display names for columns
        highlight_best: Bold the best value in each column

    Returns:
        Markdown table string
    """
    if columns is None:
        columns = ["auc_roc", "f1", "best_f1", "tea_auc", "tea_best_f1"]
    if column_headers is None:
        column_headers = ["AUC-ROC", "F1", "Best F1", "TEA AUC", "TEA F1"]

    # Find best values for each column
    best_values = {}
    if highlight_best:
        for col in columns:
            values = [m.get(col, 0) for m in metrics_dict.values()]
            if values:
                best_values[col] = max(values)

    lines = []

    # Header
    header = "| Method | " + " | ".join(column_headers) + " |"
    separator = "|" + "|".join(["---" for _ in range(len(columns) + 1)]) + "|"
    lines.append(header)
    lines.append(separator)

    for method, metrics in metrics_dict.items():
        # Format method name
        display_name = method.replace("_", "-").upper()
        if method == "dystgat":
            display_name = "**DySTGAT (Ours)**"

        values = []
        for col in columns:
            val = metrics.get(col, 0.0)
            formatted = f"{val:.4f}"

            # Highlight best
            if highlight_best and col in best_values:
                if abs(val - best_values[col]) < 1e-6:
                    formatted = f"**{formatted}**"

            values.append(formatted)

        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    return "\n".join(lines)


def format_csv_table(
    metrics_dict: Dict[str, Dict[str, float]],
    columns: List[str] = None,
) -> str:
    """Format metrics as CSV.

    Args:
        metrics_dict: Dict mapping method names to metrics
        columns: List of metric column keys

    Returns:
        CSV string
    """
    if columns is None:
        columns = ["auc_roc", "f1", "best_f1", "tea_auc", "tea_best_f1"]

    lines = []
    lines.append("method," + ",".join(columns))

    for method, metrics in metrics_dict.items():
        values = [str(metrics.get(col, 0.0)) for col in columns]
        lines.append(f"{method}," + ",".join(values))

    return "\n".join(lines)


def main():
    """Main comparison function."""
    args = parse_args()

    # Parse methods list
    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]

    # Load all metrics
    all_metrics = load_all_metrics(
        args.results_dir,
        methods=methods,
        test_set=args.test_set,
    )

    if not all_metrics:
        print(f"No metrics found in {args.results_dir}")
        print("Make sure you have trained baselines and saved their metrics.")
        sys.exit(1)

    print(f"Found metrics for {len(all_metrics)} methods:")
    for method in all_metrics:
        print(f"  - {method}")

    # Sort results
    if args.sort_by != "name":
        sorted_items = sorted(
            all_metrics.items(),
            key=lambda x: x[1].get(args.sort_by, 0),
            reverse=not args.ascending,
        )
    else:
        sorted_items = sorted(all_metrics.items(), reverse=not args.ascending)

    all_metrics = dict(sorted_items)

    # Format tables
    columns = ["auc_roc", "f1", "best_f1", "tea_auc", "tea_best_f1"]
    column_headers = ["AUC-ROC", "F1", "Best F1", "TEA AUC", "TEA F1"]

    if args.format == "markdown" or args.format == "all":
        md_table = format_markdown_table(all_metrics, columns, column_headers)
        print("\n## Comparison Table (Markdown)\n")
        print(md_table)

        if args.output and args.format == "markdown":
            output_path = args.output
        elif args.format == "all":
            output_path = os.path.join(args.results_dir, "comparison.md")
        else:
            output_path = None

        if output_path:
            with open(output_path, "w") as f:
                f.write("# Baseline Comparison Results\n\n")
                f.write(f"Test set: {args.test_set}\n\n")
                f.write(md_table)
                f.write("\n")
            print(f"\nSaved to: {output_path}")

    if args.format == "latex" or args.format == "all":
        latex_table = format_latex_table(all_metrics, columns, column_headers)
        print("\n## Comparison Table (LaTeX)\n")
        print(latex_table)

        if args.output and args.format == "latex":
            output_path = args.output
        elif args.format == "all":
            output_path = os.path.join(args.results_dir, "comparison.tex")
        else:
            output_path = None

        if output_path:
            with open(output_path, "w") as f:
                f.write("% Baseline Comparison Results\n")
                f.write(f"% Test set: {args.test_set}\n\n")
                f.write(latex_table)
                f.write("\n")
            print(f"\nSaved to: {output_path}")

    if args.format == "csv" or args.format == "all":
        csv_table = format_csv_table(all_metrics, columns)

        if args.output and args.format == "csv":
            output_path = args.output
        elif args.format == "all":
            output_path = os.path.join(args.results_dir, "comparison.csv")
        else:
            output_path = None

        if output_path:
            with open(output_path, "w") as f:
                f.write(csv_table)
                f.write("\n")
            print(f"\nSaved to: {output_path}")

    # Print method descriptions
    print("\n## Method Descriptions\n")
    for method in all_metrics:
        if method in list_baselines():
            desc = get_baseline_description(method)
            print(f"- **{method}**: {desc}")
        elif method == "dystgat":
            print(f"- **DySTGAT**: Our method with dual temporal-spectral view")

    print("\nDone.")


if __name__ == "__main__":
    main()
