"""Metrics for comparing residue scores with Start2Fold protection classes."""
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def compute_roc_auc(residue_scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC; returns nan when all labels are same class."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, residue_scores))


def compute_mcc(predicted: set, ground_truth: set, n_residues: int) -> float:
    labels = np.zeros(n_residues, dtype=int)
    preds = np.zeros(n_residues, dtype=int)
    for r in ground_truth:
        if 1 <= r <= n_residues:
            labels[r - 1] = 1
    for r in predicted:
        if 1 <= r <= n_residues:
            preds[r - 1] = 1
    return float(matthews_corrcoef(labels, preds))


def compute_precision_at_k(residue_scores: np.ndarray, ground_truth: set, k: int) -> float:
    """Precision among top-k residues versus a reference protection set."""
    if k <= 0:
        return float("nan")
    top_k = set(np.argsort(residue_scores)[::-1][:k] + 1)  # convert to 1-indexed
    return len(top_k & ground_truth) / k


def bootstrap_metric_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05):
    """
    Bootstrap 95% CI for the median of a per-protein metric list.
    Returns (ci_lo, ci_hi, median).
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_medians = [np.median(arr[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return lo, hi, float(np.median(arr))


def aggregate_dataset_metrics(metrics_list: list[dict]) -> dict:
    """
    Compute dataset-level summary (median, IQR, bootstrap CI 95%) for each
    scalar metric in a list of per-protein dicts. Skips NaN values.
    """
    if not metrics_list:
        return {}
    keys = [k for k, v in metrics_list[0].items() if isinstance(v, (int, float))]
    result = {}
    for key in keys:
        vals = [m[key] for m in metrics_list
                if m.get(key) is not None and not np.isnan(float(m.get(key, float("nan"))))]
        if not vals:
            result[key] = {"median": float("nan"), "iqr_lo": float("nan"),
                           "iqr_hi": float("nan"), "ci_lo": float("nan"),
                           "ci_hi": float("nan"), "n": 0}
            continue
        arr = np.array(vals)
        lo, hi, med = bootstrap_metric_ci(vals)
        result[key] = {
            "median": float(np.median(arr)),
            "iqr_lo": float(np.percentile(arr, 25)),
            "iqr_hi": float(np.percentile(arr, 75)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(vals),
        }
    return result
