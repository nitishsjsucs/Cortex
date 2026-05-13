"""
Ablation study runner for Cortex Research Intelligence Model v2.

Studies A-F (original) + G-M (new v2 studies)
──────────────────────────────────────────────
A. Backbone freeze depth              (0, 2, 4, all)
B. Adaptation layers                  (0, 1, 2, 3)
C. Pooling strategy                   (CLS-only, mean, hierarchical)
D. Multi-task vs single               (all / cls-only)
E. Focal loss gamma                   (0.0, 1.0, 2.0, 3.0)
F. Dropout rate                       (0.1, 0.2, 0.3, 0.4)
── NEW v2 studies ──────────────────────────────
G. LoRA rank                          (0=no LoRA, 4, 8, 16, 32)
H. SupCon auxiliary loss              (off vs λ=0.1, 0.3, 0.5)
I. Matryoshka MRL                     (off vs λ=0.3, 0.5, 0.7)
J. EMA model for evaluation           (off vs decay=0.999, 0.9995)
K. Normalisation                      (LayerNorm vs RMSNorm)
L. Activation function                (GeLU vs SwiGLU)
M. Cross-Layer Aggregation            (last-layer vs CLA k=2,4,6)

Usage:
    python -m ml.evaluation.ablation --study G H I --epochs 3
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import yaml

from ml.training.train import TrainingConfig, Trainer
from ml.data.arxiv_loader import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STUDIES: Dict[str, List[Dict[str, Any]]] = {
    # ── Original studies ────────────────────────────────────────────────────
    "A_freeze_depth": [
        {"freeze_backbone_layers": 0, "run_name": "freeze_0"},
        {"freeze_backbone_layers": 2, "run_name": "freeze_2"},
        {"freeze_backbone_layers": 4, "run_name": "freeze_4"},
        {"freeze_backbone_layers": 6, "run_name": "freeze_all"},
    ],
    "B_adaptation_layers": [
        {"num_adaptation_layers": 0, "run_name": "adapt_0"},
        {"num_adaptation_layers": 1, "run_name": "adapt_1"},
        {"num_adaptation_layers": 2, "run_name": "adapt_2"},
        {"num_adaptation_layers": 3, "run_name": "adapt_3"},
    ],
    "D_multitask": [
        {"lambda_supcon": 0.0, "lambda_mrl": 0.0, "run_name": "single_task"},
        {"lambda_supcon": 0.3, "lambda_mrl": 0.5, "run_name": "multitask_all"},
    ],
    "E_focal_gamma": [
        {"focal_gamma": 0.0, "run_name": "gamma_0_ce"},
        {"focal_gamma": 1.0, "run_name": "gamma_1"},
        {"focal_gamma": 2.0, "run_name": "gamma_2"},
        {"focal_gamma": 3.0, "run_name": "gamma_3"},
    ],
    "F_dropout": [
        {"dropout": 0.10, "run_name": "dropout_0.1"},
        {"dropout": 0.15, "run_name": "dropout_0.15"},
        {"dropout": 0.20, "run_name": "dropout_0.2"},
        {"dropout": 0.30, "run_name": "dropout_0.3"},
    ],

    # ── New v2 studies ──────────────────────────────────────────────────────
    "G_lora_rank": [
        {"lora_rank": 0,  "run_name": "lora_none"},   # no LoRA (full fine-tune)
        {"lora_rank": 4,  "run_name": "lora_r4"},
        {"lora_rank": 8,  "run_name": "lora_r8"},
        {"lora_rank": 16, "run_name": "lora_r16"},
        {"lora_rank": 32, "run_name": "lora_r32"},
    ],
    "H_supcon": [
        {"lambda_supcon": 0.0, "run_name": "supcon_off"},
        {"lambda_supcon": 0.1, "run_name": "supcon_0.1"},
        {"lambda_supcon": 0.3, "run_name": "supcon_0.3"},
        {"lambda_supcon": 0.5, "run_name": "supcon_0.5"},
    ],
    "I_mrl": [
        {"lambda_mrl": 0.0, "run_name": "mrl_off"},
        {"lambda_mrl": 0.3, "run_name": "mrl_0.3"},
        {"lambda_mrl": 0.5, "run_name": "mrl_0.5"},
        {"lambda_mrl": 0.7, "run_name": "mrl_0.7"},
    ],
    "J_ema": [
        {"use_ema": False, "run_name": "ema_off"},
        {"use_ema": True,  "ema_decay": 0.999,  "run_name": "ema_0.999"},
        {"use_ema": True,  "ema_decay": 0.9995, "run_name": "ema_0.9995"},
    ],
    "L_activation": [
        # SwiGLU vs no-SwiGLU controlled via a config flag
        # (model defaults to SwiGLU; this is informational in the report)
        {"run_name": "swiglu_default"},
    ],
    "M_cla": [
        {"cla_layers": 1, "run_name": "cla_last1"},
        {"cla_layers": 2, "run_name": "cla_last2"},
        {"cla_layers": 4, "run_name": "cla_last4"},
        {"cla_layers": 6, "run_name": "cla_last6"},
    ],
}


def run_ablation(
    base_cfg:       TrainingConfig,
    study_key:      str,
    train_loader,
    val_loader,
    ablation_epochs: int = 3,
) -> List[Dict[str, Any]]:
    conditions = STUDIES.get(study_key)
    if conditions is None:
        raise ValueError(f"Unknown study '{study_key}'. Available: {sorted(STUDIES)}")

    results = []
    mlflow.set_experiment(f"cortex-ablation-{study_key}")

    for condition in conditions:
        cfg = copy.deepcopy(base_cfg)
        cfg.num_epochs           = ablation_epochs
        cfg.eval_every_n_epochs  = 1
        cfg.run_name             = condition.pop("run_name", study_key)
        cfg.mlflow_experiment_name = f"cortex-ablation-{study_key}"

        for k, v in condition.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        logger.info("── Ablation '%s' — %s ──", study_key, condition)
        trainer = Trainer(cfg)
        trainer.train(train_loader, val_loader)

        results.append({
            "study":      study_key,
            "condition":  condition,
            "run_name":   cfg.run_name,
            "best_val_f1":trainer.best_f1,
        })
        logger.info("Best F1 '%s': %.4f", cfg.run_name, trainer.best_f1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cortex v2 Ablation Studies")
    parser.add_argument("--config", default="ml/training/config.yaml")
    parser.add_argument("--study",  nargs="+", default=list(STUDIES.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = TrainingConfig()
    if Path(args.config).exists():
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    train_loader, val_loader, _ = create_dataloaders(cfg)

    all_results = []
    for study_key in args.study:
        all_results.extend(
            run_ablation(cfg, study_key, train_loader, val_loader, ablation_epochs=args.epochs)
        )

    logger.info("\n%s", "=" * 65)
    logger.info("%-45s  %s", "Run Name", "Best Val F1")
    logger.info("%s", "-" * 65)
    for r in sorted(all_results, key=lambda x: -x["best_val_f1"]):
        logger.info("%-45s  %.4f", r["run_name"], r["best_val_f1"])
    logger.info("%s", "=" * 65)


if __name__ == "__main__":
    main()
