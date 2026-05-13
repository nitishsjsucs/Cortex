"""
Visualisation utilities: training curves, confusion matrix, regression plots,
per-category F1 bars, and ablation comparison charts.

All functions save to disk and return the figure path.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Helpers ────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    val_f1s:      List[float],
    output_path:  str = "ml/outputs/training_curves.png",
) -> str:
    """Plot loss + F1 across epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_losses,   marker="s", label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[1].plot(epochs, val_f1s, marker="^", color="forestgreen")
    axes[1].set_title("Validation Category F1 (macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Research Intelligence Model — Training Progress", fontsize=14)
    return _save(fig, output_path)


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:       np.ndarray,
    preds:        np.ndarray,
    class_names:  List[str],
    output_path:  str = "ml/outputs/confusion_matrix.png",
    normalise:    bool = True,
) -> str:
    cm = confusion_matrix(labels, preds, normalize="true" if normalise else None)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[c.split(".")[-1] for c in class_names],  # short labels
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45,
              values_format=".2f" if normalise else "d")
    ax.set_title(
        f"Category Classification — {'Normalised ' if normalise else ''}Confusion Matrix",
        pad=12
    )
    return _save(fig, output_path)


# ── Regression scatter ──────────────────────────────────────────────────────

def plot_regression_scatter(
    targets:     np.ndarray,
    preds:       np.ndarray,
    output_path: str = "ml/outputs/impact_scatter.png",
) -> str:
    from scipy.stats import pearsonr
    r, _ = pearsonr(targets, preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(targets, preds, alpha=0.35, s=18, color="steelblue", label="Samples")
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Ground-truth Impact Score")
    ax.set_ylabel("Predicted Impact Score")
    ax.set_title(f"Impact Regression — Pearson r = {r:.3f}")
    ax.legend()
    return _save(fig, output_path)


# ── Per-category F1 bar chart ────────────────────────────────────────────────

def plot_per_category_f1(
    per_cat_metrics: Dict[str, Dict[str, float]],
    category_names:  List[str],
    output_path:     str = "ml/outputs/per_category_f1.png",
) -> str:
    names = [c.split(".")[-1] for c in category_names]
    f1s   = [per_cat_metrics.get(c, {}).get("f1-score", 0.0) for c in category_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if v >= 0.6 else "#e67e22" if v >= 0.4 else "#e74c3c" for v in f1s]
    ax.bar(names, f1s, color=colors)
    ax.axhline(np.mean(f1s), color="navy", linestyle="--", lw=1.5,
               label=f"Macro avg = {np.mean(f1s):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Research Area")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Category F1 (test set)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    return _save(fig, output_path)


# ── Ablation comparison ──────────────────────────────────────────────────────

def plot_ablation_results(
    study_name:  str,
    conditions:  List[str],
    f1_scores:   List[float],
    output_path: str = "ml/outputs/ablation.png",
    baseline_f1: Optional[float] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3498db"] * len(conditions)
    if baseline_f1 is not None:
        ax.axhline(baseline_f1, color="red", linestyle="--", lw=1.5, label="Baseline")

    ax.bar(conditions, f1_scores, color=colors, edgecolor="white")
    ax.set_ylim(0, max(f1_scores) * 1.15)
    ax.set_ylabel("Validation Category F1")
    ax.set_title(f"Ablation Study — {study_name}")
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    return _save(fig, output_path)


# ── Sweep results ────────────────────────────────────────────────────────────

def plot_sweep_parallel(
    param_names:  List[str],
    param_values: List[List[float]],
    f1_scores:    List[float],
    output_path:  str = "ml/outputs/sweep_parallel.png",
) -> str:
    """
    Parallel coordinates plot for hyperparameter sweep results.
    Each line = one trial, colour-coded by F1 score.
    """
    try:
        from pandas.plotting import parallel_coordinates
        import pandas as pd
    except ImportError:
        return ""

    data = {name: vals for name, vals in zip(param_names, param_values)}
    data["f1_score"] = f1_scores
    df = pd.DataFrame(data)
    df["f1_bin"] = pd.qcut(df["f1_score"], q=4, labels=["low", "med-low", "med-high", "high"])

    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(df.drop("f1_score", axis=1), "f1_bin", colormap="RdYlGn", ax=ax)
    ax.set_title("Hyperparameter Sweep — Parallel Coordinates")
    return _save(fig, output_path)
