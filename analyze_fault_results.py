#!/usr/bin/env python3
"""
Analyze fault detection results and generate summary statistics.
Run this after executing the fault tests to get quantitative insights.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = "outputs/anomaly_scores"

FAULT_NAMES = [
    "Fault1_DisplayCaseDoorOpen",
    "Fault2_IceAccumulation",
    "Fault3_EvapValveFailure",
    "Fault4_MTEvapFanFailure",
    "Fault5_CondAPBlock",
    "Fault6_MTEvapAPBlock",
]


def analyze_single_fault(fault_name: str, results_dir: str) -> dict:
    """Analyze a single fault dataset."""
    csv_path = os.path.join(results_dir, f"{fault_name}_anomaly_scores.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping {fault_name}")
        return None
    
    df = pd.read_csv(csv_path)
    
    stats = {
        "fault_name": fault_name,
        "num_samples": df["sample_id"].nunique(),
        "total_timesteps": len(df),
        "mean_score": df["anomaly_score"].mean(),
        "median_score": df["anomaly_score"].median(),
        "std_score": df["anomaly_score"].std(),
        "min_score": df["anomaly_score"].min(),
        "max_score": df["anomaly_score"].max(),
        "p95_score": df["anomaly_score"].quantile(0.95),
        "p99_score": df["anomaly_score"].quantile(0.99),
    }
    
    return stats


def find_detection_time(fault_name: str, results_dir: str, threshold: float = 2.0) -> dict:
    """Find when fault is first detected above threshold."""
    csv_path = os.path.join(results_dir, f"{fault_name}_anomaly_scores.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Find first detection
    detections = df[df["anomaly_score"] > threshold]
    if len(detections) == 0:
        return {
            "fault_name": fault_name,
            "first_detection": "Not detected",
            "detection_sample": None,
        }
    
    first = detections.iloc[0]
    return {
        "fault_name": fault_name,
        "first_detection": first["timestamp"],
        "detection_sample": int(first["sample_id"]),
        "detection_score": float(first["anomaly_score"]),
    }


def plot_comparative_scores(all_stats: list, output_path: str):
    """Create comparative bar plot of anomaly scores across faults."""
    if not all_stats:
        print("No statistics available for plotting")
        return
    
    df = pd.DataFrame(all_stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean scores
    ax = axes[0, 0]
    df.plot(x="fault_name", y="mean_score", kind="bar", ax=ax, legend=False, color="steelblue")
    ax.set_title("Mean Anomaly Score by Fault Type")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Score")
    ax.tick_params(axis='x', rotation=45)
    
    # Max scores
    ax = axes[0, 1]
    df.plot(x="fault_name", y="max_score", kind="bar", ax=ax, legend=False, color="coral")
    ax.set_title("Maximum Anomaly Score by Fault Type")
    ax.set_xlabel("")
    ax.set_ylabel("Max Score")
    ax.tick_params(axis='x', rotation=45)
    
    # 95th percentile
    ax = axes[1, 0]
    df.plot(x="fault_name", y="p95_score", kind="bar", ax=ax, legend=False, color="mediumseagreen")
    ax.set_title("95th Percentile Anomaly Score")
    ax.set_xlabel("")
    ax.set_ylabel("P95 Score")
    ax.tick_params(axis='x', rotation=45)
    
    # Standard deviation
    ax = axes[1, 1]
    df.plot(x="fault_name", y="std_score", kind="bar", ax=ax, legend=False, color="mediumpurple")
    ax.set_title("Score Variability (Std Dev)")
    ax.set_xlabel("")
    ax.set_ylabel("Std Dev")
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparative plot to {output_path}")


def main():
    print("=" * 60)
    print("Fault Detection Results Analysis")
    print("=" * 60)
    print()
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Results directory '{OUTPUT_DIR}' not found.")
        print("Please run the fault tests first using:")
        print("  python run_all_fault_tests.py")
        return
    
    # Collect statistics for all faults
    all_stats = []
    all_detections = []
    
    for fault in FAULT_NAMES:
        stats = analyze_single_fault(fault, OUTPUT_DIR)
        if stats:
            all_stats.append(stats)
        
        detection = find_detection_time(fault, OUTPUT_DIR, threshold=2.0)
        if detection:
            all_detections.append(detection)
    
    # Print summary statistics
    if all_stats:
        print("\n" + "=" * 60)
        print("Summary Statistics for Each Fault")
        print("=" * 60)
        
        df_stats = pd.DataFrame(all_stats)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df_stats.to_string(index=False))
        
        # Save to CSV
        stats_csv = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
        df_stats.to_csv(stats_csv, index=False)
        print(f"\n✓ Saved summary statistics to {stats_csv}")
    
    # Print detection times
    if all_detections:
        print("\n" + "=" * 60)
        print("Fault Detection Times (Threshold = 2.0)")
        print("=" * 60)
        
        df_detect = pd.DataFrame(all_detections)
        print(df_detect.to_string(index=False))
        
        # Save to CSV
        detect_csv = os.path.join(OUTPUT_DIR, "detection_times.csv")
        df_detect.to_csv(detect_csv, index=False)
        print(f"\n✓ Saved detection times to {detect_csv}")
    
    # Generate comparative plots
    if all_stats:
        plot_path = os.path.join(OUTPUT_DIR, "comparative_analysis.png")
        plot_comparative_scores(all_stats, plot_path)
    
    # Rank faults by detectability
    if all_stats:
        print("\n" + "=" * 60)
        print("Fault Detectability Ranking (by mean score)")
        print("=" * 60)
        df_ranked = pd.DataFrame(all_stats).sort_values("mean_score", ascending=False)
        for i, row in enumerate(df_ranked.itertuples(), 1):
            print(f"{i}. {row.fault_name}: {row.mean_score:.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

