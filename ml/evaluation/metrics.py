"""
Evaluation metrics for the Research Intelligence Model.

Computed metrics
----------------
  category_accuracy  : top-1 accuracy on 20-class area classification
  category_f1        : macro-averaged F1 across all categories
  category_top3_acc  : top-3 accuracy (useful for long-tail categories)
  impact_mse         : mean-squared error on normalised impact score
  impact_mae         : mean absolute error on normalised impact score
  impact_r2          : R² coefficient of determination
  quality_accuracy   : accuracy on binary abstract-quality label
  quality_f1         : weighted F1 for binary quality classification
  quality_auc        : ROC-AUC (binary quality)
"""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _safe(fn, *args, **kwargs) -> float:
    """Call metric function; return NaN on failure."""
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:
        logger.debug("Metric computation failed: %s", exc)
        return float("nan")


def top_k_accuracy(preds_logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy from raw logit arrays."""
    top_k = np.argsort(preds_logits, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def compute_metrics(
    category_preds: np.ndarray,    # (N,)   argmax predictions
    category_labels: np.ndarray,   # (N,)   ground-truth indices
    impact_preds: np.ndarray,      # (N,)   continuous predictions [0,1]
    impact_targets: np.ndarray,    # (N,)   ground-truth [0,1]
    quality_preds: np.ndarray,     # (N,)   argmax binary predictions
    quality_labels: np.ndarray,    # (N,)   ground-truth {0,1}
    category_logits: np.ndarray = None,  # (N, C)  raw logits for top-k
) -> Dict[str, float]:
    """Compute and return all evaluation metrics as a flat dict."""

    metrics: Dict[str, float] = {}

    # ── Classification ─────────────────────────────────────────────────────
    metrics["category_accuracy"] = _safe(accuracy_score, category_labels, category_preds)
    metrics["category_f1"] = _safe(
        f1_score, category_labels, category_preds,
        average="macro", zero_division=0
    )
    if category_logits is not None:
        metrics["category_top3_acc"] = top_k_accuracy(category_logits, category_labels, k=3)

    # ── Impact regression ─────────────────────────────────────────────────
    metrics["impact_mse"] = _safe(mean_squared_error, impact_targets, impact_preds)
    metrics["impact_mae"] = _safe(mean_absolute_error, impact_targets, impact_preds)
    metrics["impact_r2"]  = _safe(r2_score, impact_targets, impact_preds)

    # ── Quality classification ────────────────────────────────────────────
    metrics["quality_accuracy"] = _safe(accuracy_score, quality_labels, quality_preds)
    metrics["quality_f1"] = _safe(
        f1_score, quality_labels, quality_preds,
        average="weighted", zero_division=0
    )
    # AUC needs probability scores; use raw pred as proxy (0 or 1 degrades gracefully)
    if len(np.unique(quality_labels)) > 1:
        metrics["quality_auc"] = _safe(roc_auc_score, quality_labels, quality_preds)

    return metrics


def per_category_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    category_names: list,
) -> Dict[str, Dict[str, float]]:
    """Return per-category precision, recall, F1, and support."""
    from sklearn.metrics import classification_report
    report = classification_report(
        labels, preds, target_names=category_names, output_dict=True, zero_division=0
    )
    return {k: v for k, v in report.items() if k in category_names}


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a human-readable string summary of metrics."""
    lines = []
    groups = [
        ("Category Classification",
         ["category_accuracy", "category_f1", "category_top3_acc"]),
        ("Impact Regression",
         ["impact_mse", "impact_mae", "impact_r2"]),
        ("Abstract Quality",
         ["quality_accuracy", "quality_f1", "quality_auc"]),
    ]
    for title, keys in groups:
        lines.append(f"\n  {title}")
        for k in keys:
            if k in metrics:
                lines.append(f"    {k:<28}: {metrics[k]:.4f}")
    return "\n".join(lines)
