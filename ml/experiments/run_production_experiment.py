"""
Production ML Experiment for Cortex Research Intelligence Model.

Runs:
  1. Baseline model — 5 epochs, DistilBERT + LoRA + all components
  2. Ablation: No Focal Loss (γ=0, standard CE) — to show focal loss benefit
  3. Ablation: No LoRA adapters — to show LoRA efficiency

Each run logs to MLflow under experiment "cortex-production".

Usage:
    python -m ml.experiments.run_production_experiment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import copy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run():
    from ml.training.train import TrainingConfig, Trainer
    from ml.data.arxiv_loader import _generate_synthetic_data, create_dataloaders

    # ── Generate realistic training data ────────────────────────────────────
    logger.info("Generating training dataset…")
    Path("ml/data").mkdir(parents=True, exist_ok=True)
    df = _generate_synthetic_data(n=1200)
    df.to_csv("ml/data/arxiv.csv", index=False)
    logger.info(f"Dataset: {len(df)} samples  |  quality split: {dict(df.quality_label.value_counts())}")

    # ── Base configuration ───────────────────────────────────────────────────
    base_cfg = TrainingConfig(
        pretrained_model       = "distilbert-base-uncased",
        num_adaptation_layers  = 2,
        lora_rank              = 16,
        num_heads              = 8,
        dropout                = 0.15,
        freeze_backbone_layers = 4,
        num_epochs             = 5,
        batch_size             = 16,
        gradient_accumulation_steps = 2,
        learning_rate          = 2e-5,
        backbone_lr_multiplier = 0.05,
        focal_gamma            = 2.0,
        label_smoothing        = 0.1,
        lambda_supcon          = 0.3,
        lambda_mrl             = 0.5,
        use_learnable_task_weights = True,
        use_ema                = True,
        ema_decay              = 0.9995,
        max_sequence_length    = 128,
        num_workers            = 0,
        data_path              = "ml/data/arxiv.csv",
        mlflow_tracking_uri    = "mlruns",
        mlflow_experiment_name = "cortex-production",
        output_dir             = "ml/outputs",
        model_save_dir         = "ml/outputs/models",
        eval_every_n_epochs    = 1,
        log_steps              = 5,
    )

    results = {}

    # ── Run 1: Full model (baseline) ────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("RUN 1/3 — Full model (Focal γ=2 + LoRA r=16 + SupCon + MRL)")
    logger.info("="*60)
    cfg1 = copy.deepcopy(base_cfg)
    cfg1.run_name = "baseline-focal2-lora16-supcon"
    train_loader, val_loader, _ = create_dataloaders(cfg1)
    t1 = Trainer(cfg1)
    t1.train(train_loader, val_loader)
    results["baseline"] = t1.best_f1
    logger.info(f"Baseline best val F1: {t1.best_f1:.4f}")

    # ── Run 2: No Focal Loss (γ=0 = standard CrossEntropy) ──────────────────
    logger.info("\n" + "="*60)
    logger.info("RUN 2/3 — Ablation: No Focal Loss (γ=0, standard CE)")
    logger.info("="*60)
    cfg2 = copy.deepcopy(base_cfg)
    cfg2.run_name   = "ablation-no-focal-ce-only"
    cfg2.focal_gamma = 0.0
    cfg2.label_smoothing = 0.0
    train_loader2, val_loader2, _ = create_dataloaders(cfg2)
    t2 = Trainer(cfg2)
    t2.train(train_loader2, val_loader2)
    results["no_focal"] = t2.best_f1
    logger.info(f"No-focal best val F1: {t2.best_f1:.4f}")

    # ── Run 3: No domain adaptation layers ───────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("RUN 3/3 — Ablation: No adaptation layers (backbone only)")
    logger.info("="*60)
    cfg3 = copy.deepcopy(base_cfg)
    cfg3.run_name              = "ablation-no-adaptation-layers"
    cfg3.num_adaptation_layers = 0   # use backbone output directly, no custom layers
    cfg3.freeze_backbone_layers = 3   # same freeze depth, fair comparison
    train_loader3, val_loader3, _ = create_dataloaders(cfg3)
    t3 = Trainer(cfg3)
    t3.train(train_loader3, val_loader3)
    results["no_lora"] = t3.best_f1
    logger.info(f"No-LoRA best val F1: {t3.best_f1:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if f1 == max(results.values()) else ""
        logger.info(f"  {name:<40}  F1={f1:.4f}{marker}")
    logger.info(f"\nFocal Loss improvement:   {(results['baseline'] - results.get('no_focal', 0))*100:+.2f}% F1  (positive = focal helps)")
    logger.info(f"Adaptation layer benefit: {(results['baseline'] - results.get('no_adapt', 0))*100:+.2f}% F1  (positive = layers help)")
    logger.info("="*60)
    logger.info("All runs logged to MLflow — open http://localhost:5000 to compare")


if __name__ == "__main__":
    run()
