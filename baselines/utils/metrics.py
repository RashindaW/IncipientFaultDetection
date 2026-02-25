"""Shared evaluation metrics for baseline anomaly detection models.

These metrics are consistent with those used in train_dualstage.py for fair comparison.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


def threshold_evt_pot(val_scores, false_alarm_rate=0.05, initial_percentile=90):
    """EVT Peaks-Over-Threshold with GPD tail fit.

    Uses only validation baseline scores — no test data leakage.
    Ref: Siffer et al. (2017) "Anomaly Detection in Streams with EVT" KDD.
    """
    u = np.percentile(val_scores, initial_percentile)
    exceedances = val_scores[val_scores > u] - u
    n_total, n_exceed = len(val_scores), len(exceedances)
    if n_exceed < 10:
        return np.percentile(val_scores, 100 * (1 - false_alarm_rate))
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    if abs(shape) < 1e-8:
        return u + scale * np.log(n_exceed / (n_total * false_alarm_rate))
    return u + (scale / shape) * ((n_exceed / (n_total * false_alarm_rate)) ** shape - 1)


def compute_auc_roc(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
) -> float:
    """Compute AUC-ROC score.

    Args:
        baseline_scores: Anomaly scores for healthy/baseline samples
        fault_scores: Anomaly scores for faulty samples

    Returns:
        AUC-ROC score
    """
    # Create binary labels: 0 for baseline (healthy), 1 for fault
    y_true = np.concatenate([
        np.zeros(len(baseline_scores)),
        np.ones(len(fault_scores))
    ])
    y_scores = np.concatenate([baseline_scores, fault_scores])

    return roc_auc_score(y_true, y_scores)


def compute_f1_at_threshold(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
    percentile: float = 95.0,
) -> Tuple[float, float, float, float]:
    """Compute F1 score at a threshold defined by baseline percentile.

    Args:
        baseline_scores: Anomaly scores for healthy samples
        fault_scores: Anomaly scores for faulty samples
        percentile: Percentile of baseline scores to use as threshold

    Returns:
        Tuple of (f1, precision, recall, threshold)
    """
    threshold = np.percentile(baseline_scores, percentile)

    # Predictions: 1 if score > threshold, 0 otherwise
    baseline_preds = (baseline_scores > threshold).astype(int)
    fault_preds = (fault_scores > threshold).astype(int)

    y_true = np.concatenate([
        np.zeros(len(baseline_scores)),
        np.ones(len(fault_scores))
    ])
    y_pred = np.concatenate([baseline_preds, fault_preds])

    # Handle edge cases
    if np.sum(y_pred) == 0:
        return 0.0, 0.0, 0.0, threshold

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return f1, precision, recall, threshold


def compute_best_f1(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute best F1 score across all thresholds using precision-recall curve.

    Args:
        baseline_scores: Anomaly scores for healthy samples
        fault_scores: Anomaly scores for faulty samples

    Returns:
        Tuple of (best_f1, best_threshold)
    """
    y_true = np.concatenate([
        np.zeros(len(baseline_scores)),
        np.ones(len(fault_scores))
    ])
    y_scores = np.concatenate([baseline_scores, fault_scores])

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 for each threshold
    # Note: precision and recall have one more element than thresholds
    f1_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    if len(f1_scores) == 0:
        return 0.0, 0.0

    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx], thresholds[best_idx]


def compute_detection_delay(
    fault_scores: np.ndarray,
    threshold: float,
    min_consecutive: int = 1,
) -> int:
    """Compute detection delay (samples until first detection).

    Args:
        fault_scores: Anomaly scores for faulty samples (in temporal order)
        threshold: Detection threshold
        min_consecutive: Minimum consecutive samples above threshold

    Returns:
        Number of samples until first detection (-1 if never detected)
    """
    above_threshold = fault_scores > threshold

    if min_consecutive == 1:
        first_detection = np.argmax(above_threshold)
        if above_threshold[first_detection]:
            return first_detection
        return -1

    # Find first occurrence of min_consecutive True values
    for i in range(len(above_threshold) - min_consecutive + 1):
        if all(above_threshold[i:i + min_consecutive]):
            return i

    return -1


def compute_tea_metrics(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
    window_sizes: Optional[list] = None,
) -> Dict[str, float]:
    """Compute TEA (Temporal Evidence Accumulation) metrics.

    This applies exponential smoothing to accumulate evidence over time,
    which is useful for bearing degradation detection.

    Args:
        baseline_scores: Anomaly scores for healthy samples
        fault_scores: Anomaly scores for faulty samples
        window_sizes: Window sizes to try for smoothing

    Returns:
        Dict with TEA metrics (tea_auc, tea_best_f1, tea_best_window)
    """
    if window_sizes is None:
        window_sizes = [30, 60, 180]

    best_tea_auc = 0.0
    best_tea_f1 = 0.0
    best_window = window_sizes[0]

    for window in window_sizes:
        # Apply exponential moving average
        alpha = 2.0 / (window + 1)

        tea_baseline = _apply_ema(baseline_scores, alpha)
        tea_fault = _apply_ema(fault_scores, alpha)

        # Compute metrics on TEA scores
        tea_auc = compute_auc_roc(tea_baseline, tea_fault)
        tea_f1, _ = compute_best_f1(tea_baseline, tea_fault)

        if tea_auc > best_tea_auc:
            best_tea_auc = tea_auc
            best_tea_f1 = tea_f1
            best_window = window

    return {
        'tea_auc': best_tea_auc,
        'tea_best_f1': best_tea_f1,
        'tea_best_window': best_window,
    }


def _apply_ema(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average to scores.

    Args:
        scores: Input scores
        alpha: Smoothing factor (0 < alpha <= 1)

    Returns:
        Smoothed scores
    """
    ema = np.zeros_like(scores)
    ema[0] = scores[0]

    for i in range(1, len(scores)):
        ema[i] = alpha * scores[i] + (1 - alpha) * ema[i - 1]

    return ema


def compute_all_metrics(
    baseline_scores: np.ndarray,
    fault_scores: np.ndarray,
    include_tea: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        baseline_scores: Anomaly scores for healthy samples
        fault_scores: Anomaly scores for faulty samples
        include_tea: Whether to include TEA metrics

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # AUC-ROC
    metrics['auc_roc'] = compute_auc_roc(baseline_scores, fault_scores)

    # F1 at 95th percentile threshold
    f1, precision, recall, threshold = compute_f1_at_threshold(
        baseline_scores, fault_scores, percentile=95.0
    )
    metrics['f1'] = f1
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['threshold'] = threshold

    # Best F1
    best_f1, best_threshold = compute_best_f1(baseline_scores, fault_scores)
    metrics['best_f1'] = best_f1
    metrics['best_threshold'] = best_threshold

    # Detection delay at best threshold
    delay = compute_detection_delay(fault_scores, best_threshold)
    metrics['detection_delay'] = delay

    # TEA metrics
    if include_tea:
        tea_metrics = compute_tea_metrics(baseline_scores, fault_scores)
        metrics.update(tea_metrics)

    return metrics


def format_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    format_type: str = "markdown",
) -> str:
    """Format metrics as a comparison table.

    Args:
        metrics_dict: Dict mapping method names to their metrics
        format_type: "markdown" or "latex"

    Returns:
        Formatted table string
    """
    # Define columns
    columns = ['auc_roc', 'f1', 'best_f1', 'tea_auc', 'tea_best_f1']
    column_headers = ['AUC-ROC', 'F1', 'Best F1', 'TEA AUC', 'TEA F1']

    if format_type == "markdown":
        # Header
        lines = []
        header = "| Method | " + " | ".join(column_headers) + " |"
        separator = "|" + "|".join(["---" for _ in range(len(columns) + 1)]) + "|"
        lines.append(header)
        lines.append(separator)

        # Rows
        for method, metrics in metrics_dict.items():
            values = [f"{metrics.get(col, 0.0):.4f}" for col in columns]
            row = f"| {method} | " + " | ".join(values) + " |"
            lines.append(row)

        return "\n".join(lines)

    elif format_type == "latex":
        lines = []
        lines.append(r"\begin{tabular}{l" + "c" * len(columns) + "}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(column_headers) + r" \\")
        lines.append(r"\midrule")

        for method, metrics in metrics_dict.items():
            values = [f"{metrics.get(col, 0.0):.4f}" for col in columns]
            row = f"{method} & " + " & ".join(values) + r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format type: {format_type}")
