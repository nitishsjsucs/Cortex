"""
Hyperparameter sweep using Optuna.

Searches over:
  - learning_rate        [1e-5, 5e-5]
  - dropout              [0.1, 0.4]
  - num_adaptation_layers [1, 3]
  - freeze_backbone_layers [0, 4]
  - batch_size           {16, 32, 64}
  - focal_gamma          [0.0, 3.0]

Each trial trains for `sweep_epochs` quick epochs and reports
validation category-F1 as the optimisation objective.

All trials are tracked in MLflow under "research-intelligence-sweep".

Usage:
    python -m ml.experiments.run_sweep \
        --n_trials 20 \
        --sweep_epochs 3 \
        --config ml/training/config.yaml
"""

import argparse
import copy
import logging
from pathlib import Path

import mlflow
import optuna
import yaml

from ml.training.train import TrainingConfig, Trainer
from ml.data.arxiv_loader import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_objective(base_cfg: TrainingConfig, sweep_epochs: int):
    """Return an Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        cfg.num_epochs           = sweep_epochs
        cfg.eval_every_n_epochs  = 1

        # ── Suggest hyperparameters ────────────────────────────────────────
        cfg.learning_rate         = trial.suggest_float("lr",          1e-5, 5e-5, log=True)
        cfg.dropout               = trial.suggest_float("dropout",     0.1,  0.4)
        cfg.num_adaptation_layers = trial.suggest_int("adapt_layers",  1,    3)
        cfg.freeze_backbone_layers= trial.suggest_int("freeze_layers", 0,    4)
        cfg.batch_size            = trial.suggest_categorical("batch_size", [16, 32, 64])
        cfg.focal_gamma           = trial.suggest_float("focal_gamma", 0.0,  3.0)
        cfg.run_name              = f"sweep_trial_{trial.number}"

        mlflow.set_experiment("research-intelligence-sweep")

        # Re-build dataloaders for the (possibly different) batch size
        train_loader, val_loader, _ = create_dataloaders(cfg)

        trainer = Trainer(cfg)
        trainer.train(train_loader, val_loader)

        # Report best validation F1 as the objective
        return trainer.best_f1

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="ml/training/config.yaml")
    parser.add_argument("--n_trials",      type=int, default=20)
    parser.add_argument("--sweep_epochs",  type=int, default=3)
    parser.add_argument("--study_name",    default="research-intelligence-sweep")
    parser.add_argument("--storage",       default=None,
                        help="Optuna storage URL (e.g. sqlite:///sweep.db)")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if Path(args.config).exists():
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    logger.info("Starting sweep: %d trials, %d epochs each", args.n_trials, args.sweep_epochs)
    study.optimize(
        make_objective(cfg, args.sweep_epochs),
        n_trials=args.n_trials,
        timeout=None,
    )

    best = study.best_trial
    logger.info("\n%s", "=" * 60)
    logger.info("Best trial:  #%d  F1=%.4f", best.number, best.value)
    logger.info("  Params: %s", best.params)
    logger.info("%s", "=" * 60)

    # Save best params to YAML
    out_path = Path("ml/outputs/best_sweep_params.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best.params, f)
    logger.info("Best params saved to %s", out_path)


if __name__ == "__main__":
    main()
