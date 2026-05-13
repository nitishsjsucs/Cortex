"""
Standalone evaluation script for a trained Research Intelligence Model.

Usage:
    python -m ml.evaluation.evaluate \
        --checkpoint ml/outputs/models/best_model.pt \
        --data_path  ml/data/arxiv.csv \
        --output_dir ml/outputs/eval
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from ml.models.research_classifier import ResearchIntelligenceModel
from ml.data.arxiv_loader import ArxivDataset, load_dataframe, TARGET_CATEGORIES
from ml.evaluation.metrics import compute_metrics, per_category_metrics, format_metrics
from ml.visualizations.plot_metrics import (
    plot_confusion_matrix,
    plot_regression_scatter,
    plot_per_category_f1,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "ml/outputs/eval",
    batch_size: int = 64,
    max_length: int = 256,
    mlflow_run_id: str = None,
) -> dict:
    """
    Load a checkpoint, run full evaluation, log to MLflow, and
    save plots + JSON report to `output_dir`.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResearchIntelligenceModel()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)

    # ── Load data ───────────────────────────────────────────────────────────
    df = load_dataframe(data_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = ArxivDataset(df, tokenizer, max_length=max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Collect predictions ─────────────────────────────────────────────────
    cat_preds, cat_labels = [], []
    cat_logits_all = []
    imp_preds, imp_targets = [], []
    qual_preds, qual_labels = [], []

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(ids, mask)

        cat_logits_all.extend(out["category_logits"].cpu().numpy())
        cat_preds.extend(out["category_logits"].argmax(-1).cpu().numpy())
        cat_labels.extend(batch["category_labels"].numpy())
        imp_preds.extend(out["impact_score"].cpu().numpy())
        imp_targets.extend(batch["impact_scores"].numpy())
        qual_preds.extend(out["quality_logits"].argmax(-1).cpu().numpy())
        qual_labels.extend(batch["quality_labels"].numpy())

    cat_preds    = np.array(cat_preds)
    cat_labels   = np.array(cat_labels)
    cat_logits   = np.array(cat_logits_all)
    imp_preds    = np.array(imp_preds)
    imp_targets  = np.array(imp_targets)
    qual_preds   = np.array(qual_preds)
    qual_labels  = np.array(qual_labels)

    # ── Compute metrics ──────────────────────────────────────────────────────
    metrics = compute_metrics(
        cat_preds, cat_labels,
        imp_preds, imp_targets,
        qual_preds, qual_labels,
        category_logits=cat_logits,
    )
    per_cat = per_category_metrics(cat_preds, cat_labels, TARGET_CATEGORIES)

    logger.info("Evaluation results:%s", format_metrics(metrics))

    # ── Plots ────────────────────────────────────────────────────────────────
    cm_path = Path(output_dir) / "confusion_matrix.png"
    plot_confusion_matrix(cat_labels, cat_preds, TARGET_CATEGORIES, str(cm_path))

    scatter_path = Path(output_dir) / "impact_scatter.png"
    plot_regression_scatter(imp_targets, imp_preds, str(scatter_path))

    per_cat_path = Path(output_dir) / "per_category_f1.png"
    plot_per_category_f1(per_cat, TARGET_CATEGORIES, str(per_cat_path))

    # ── Save JSON report ─────────────────────────────────────────────────────
    report = {"summary": metrics, "per_category": per_cat}
    report_path = Path(output_dir) / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)

    # ── Log to MLflow ────────────────────────────────────────────────────────
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics({f"test/{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(scatter_path))
            mlflow.log_artifact(str(per_cat_path))
            mlflow.log_artifact(str(report_path))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--output_dir", default="ml/outputs/eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mlflow_run_id", default=None)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        mlflow_run_id=args.mlflow_run_id,
    )


if __name__ == "__main__":
    main()
