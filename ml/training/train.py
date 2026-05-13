"""
Training pipeline for Cortex Research Intelligence Model v2.

New in v2
─────────
- EMA (Exponential Moving Average) model for evaluation  [Polyak 1992]
- Multi-stage training: frozen → gradual unfreeze         [ULMFiT 2018]
- Gradient checkpointing for memory-efficient fine-tuning
- bf16 + fp16 mixed precision
- SupCon warmup annealing (0 → λ over first 500 steps)
- Per-step global_step passed to loss for SupCon warmup
- Separate LoRA parameter group with 2× head LR

Run:
    python -m ml.training.train --config ml/training/config.yaml
"""

import argparse
import dataclasses
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ml.models.research_classifier import CortexResearchModel
from ml.models.components import ModelEMA
from ml.data.arxiv_loader import create_dataloaders
from ml.training.losses import MultiTaskLoss
from ml.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Training configuration ─────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    pretrained_model:       str   = "distilbert-base-uncased"
    num_adaptation_layers:  int   = 3
    lora_rank:              int   = 16
    lora_alpha:             float = 32.0
    num_heads:              int   = 12
    dropout:                float = 0.15
    drop_path_rate:         float = 0.1
    freeze_backbone_layers: int   = 2
    cla_layers:             int   = 4
    proj_dim:               int   = 128
    use_alternating_attention: bool = True
    window_size:            int   = 64

    # Training
    num_epochs:                 int   = 10
    batch_size:                 int   = 32
    gradient_accumulation_steps:int   = 2
    learning_rate:              float = 2e-5
    backbone_lr_multiplier:     float = 0.05   # backbone gets 5× smaller LR
    lora_lr_multiplier:         float = 2.0    # LoRA params get 2× higher LR
    weight_decay:               float = 0.01
    warmup_ratio:               float = 0.1
    max_grad_norm:              float = 1.0
    fp16:                       bool  = False
    bf16:                       bool  = False
    gradient_checkpointing:     bool  = False

    # Multi-stage unfreeze (ULMFiT recipe)
    unfreeze_schedule: List[int] = field(default_factory=lambda: [3, 6])  # unfreeze at these epochs

    # EMA
    ema_decay:   float = 0.9995
    ema_warmup:  int   = 2000
    use_ema:     bool  = True

    # Data
    max_sequence_length: int   = 256
    train_size:          float = 0.8
    val_size:            float = 0.1
    num_workers:         int   = 0
    data_path:           str   = "ml/data/arxiv.csv"

    # Loss
    focal_gamma:            float = 2.0
    label_smoothing:        float = 0.1
    lambda_supcon:          float = 0.3
    lambda_mrl:             float = 0.5
    lambda_rank:            float = 0.3
    supcon_warmup_steps:    int   = 500
    use_learnable_task_weights: bool = True

    # Paths
    output_dir:      str = "ml/outputs"
    model_save_dir:  str = "ml/outputs/models"

    # MLflow
    mlflow_tracking_uri:    str   = "mlruns"
    mlflow_experiment_name: str   = "research-intelligence-v2"
    run_name:               Optional[str] = None

    # Misc
    seed:                int = 42
    log_steps:           int = 10
    eval_every_n_epochs: int = 1


# ── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.model_save_dir).mkdir(parents=True, exist_ok=True)

        # Mixed precision
        use_amp   = (cfg.fp16 or cfg.bf16) and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        self.scaler     = GradScaler(enabled=cfg.fp16 and torch.cuda.is_available())
        self.autocast_ctx = lambda: autocast(enabled=use_amp, dtype=amp_dtype)

        self.best_f1     = float("-inf")
        self.global_step = 0
        self.ema: Optional[ModelEMA] = None

    # ── Model + optimizer setup ────────────────────────────────────────────

    def _build_model(self) -> CortexResearchModel:
        model = CortexResearchModel(
            backbone_name          = self.cfg.pretrained_model,
            num_adaptation_layers  = self.cfg.num_adaptation_layers,
            lora_rank              = self.cfg.lora_rank,
            lora_alpha             = self.cfg.lora_alpha,
            num_heads              = self.cfg.num_heads,
            dropout                = self.cfg.dropout,
            drop_path_rate         = self.cfg.drop_path_rate,
            freeze_backbone_layers = self.cfg.freeze_backbone_layers,
            cla_layers             = self.cfg.cla_layers,
            proj_dim               = self.cfg.proj_dim,
            use_alternating_attention = self.cfg.use_alternating_attention,
            window_size            = self.cfg.window_size,
        ).to(self.device)

        if self.cfg.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        logger.info(
            "Model v2 — total: %dM  trainable: %dM  LoRA: %dk",
            model.num_parameters // 1_000_000,
            model.num_trainable_parameters // 1_000_000,
            model.lora_parameters // 1_000,
        )
        return model

    def _build_optimizer_and_scheduler(self, model: CortexResearchModel, total_steps: int):
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        lora_ids     = {id(p) for n, p in model.named_parameters() if "lora" in n.lower()}

        param_groups = [
            # Backbone parameters (slowest LR — protect pre-trained knowledge)
            {
                "params": [p for p in model.parameters()
                           if id(p) in backbone_ids and p.requires_grad],
                "lr":   self.cfg.learning_rate * self.cfg.backbone_lr_multiplier,
                "name": "backbone",
            },
            # LoRA adapters (slightly higher LR for fast adaptation)
            {
                "params": [p for p in model.parameters()
                           if id(p) in lora_ids and p.requires_grad],
                "lr":   self.cfg.learning_rate * self.cfg.lora_lr_multiplier,
                "name": "lora",
            },
            # All other task-specific parameters
            {
                "params": [p for p in model.parameters()
                           if id(p) not in backbone_ids and id(p) not in lora_ids and p.requires_grad],
                "lr":   self.cfg.learning_rate,
                "name": "head",
            },
        ]
        # Remove empty groups
        param_groups = [g for g in param_groups if g["params"]]

        optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay, eps=1e-8, betas=(0.9, 0.98))

        warmup_steps = max(1, int(total_steps * self.cfg.warmup_ratio))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7),
            ],
            milestones=[warmup_steps],
        )
        return optimizer, scheduler

    def _unfreeze_backbone(self, model: CortexResearchModel, epoch: int):
        """Progressive backbone unfreezing (ULMFiT-style)."""
        if epoch in self.cfg.unfreeze_schedule:
            unfrozen = 0
            for attr in ["transformer.layer", "encoder.layer", "layers"]:
                try:
                    layers = eval(f"model.backbone.{attr}")
                    for layer in layers:
                        for p in layer.parameters():
                            if not p.requires_grad:
                                p.requires_grad_(True)
                                unfrozen += 1
                    if unfrozen > 0:
                        logger.info("Epoch %d: unfroze %d backbone params", epoch, unfrozen)
                        break
                except Exception:
                    continue

    # ── Training step ──────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model: CortexResearchModel,
        loader: DataLoader,
        optimizer,
        scheduler,
        criterion: MultiTaskLoss,
    ) -> Dict[str, float]:
        model.train()
        totals: Dict[str, float] = {k: 0.0 for k in
            ["total","classification","supcon","mrl","regression","quality"]}
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            ids    = batch["input_ids"].to(self.device)
            mask   = batch["attention_mask"].to(self.device)
            cat    = batch["category_labels"].to(self.device)
            imp    = batch["impact_scores"].to(self.device)
            qual   = batch["quality_labels"].to(self.device)

            with self.autocast_ctx():
                out = model(ids, mask)
                tw  = model.task_weights() if self.cfg.use_learnable_task_weights else None
                losses = criterion(
                    category_logits = out["category_logits"],
                    impact_pred     = out["impact_score"],
                    quality_logits  = out["quality_logits"],
                    proj_emb        = out["proj_emb"],
                    mrl_logits      = out["mrl_logits"],
                    category_labels = cat,
                    impact_targets  = imp,
                    quality_labels  = qual,
                    task_weights    = tw,
                    global_step     = self.global_step,
                )

            scaled = losses["total"] / self.cfg.gradient_accumulation_steps
            self.scaler.scale(scaled).backward()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

                # Update EMA
                if self.ema is not None:
                    self.ema.update(model)

                if self.global_step % self.cfg.log_steps == 0:
                    mlflow.log_metrics({
                        "train/total_loss":          losses["total"].item(),
                        "train/classification_loss": losses["classification"].item(),
                        "train/supcon_loss":         losses["supcon"].item(),
                        "train/mrl_loss":            losses["mrl"].item(),
                        "train/regression_loss":     losses["regression"].item(),
                        "train/quality_loss":        losses["quality"].item(),
                        "train/lr_head":             optimizer.param_groups[-1]["lr"],
                        "train/lr_backbone":         optimizer.param_groups[0]["lr"],
                        "train/task_w_cls":          tw[0].item() if tw is not None else 0.33,
                        "train/task_w_reg":          tw[1].item() if tw is not None else 0.33,
                        "train/task_w_qual":         tw[2].item() if tw is not None else 0.33,
                    }, step=self.global_step)

        return {k: v / n_batches for k, v in totals.items()}

    # ── Evaluation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,   # can be EMA shadow model
        loader: DataLoader,
        criterion: MultiTaskLoss,
    ) -> Dict[str, float]:
        model.eval()
        cat_preds, cat_labels = [], []
        imp_preds, imp_targets = [], []
        qual_preds, qual_labels = [], []
        val_loss = 0.0
        n_batches = 0

        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            out = model(ids, mask) if isinstance(model, CortexResearchModel) else \
                  model(ids, mask)

            losses = criterion(
                category_logits = out["category_logits"],
                impact_pred     = out["impact_score"],
                quality_logits  = out["quality_logits"],
                proj_emb        = out.get("proj_emb", torch.zeros(ids.shape[0], 128, device=self.device)),
                mrl_logits      = out.get("mrl_logits", [out["category_logits"]] * 4),
                category_labels = batch["category_labels"].to(self.device),
                impact_targets  = batch["impact_scores"].to(self.device),
                quality_labels  = batch["quality_labels"].to(self.device),
                global_step     = self.global_step,
            )
            val_loss  += losses["total"].item()
            n_batches += 1

            cat_preds.extend(out["category_logits"].argmax(-1).cpu().numpy())
            cat_labels.extend(batch["category_labels"].numpy())
            imp_preds.extend(out["impact_score"].cpu().numpy())
            imp_targets.extend(batch["impact_scores"].numpy())
            qual_preds.extend(out["quality_logits"].argmax(-1).cpu().numpy())
            qual_labels.extend(batch["quality_labels"].numpy())

        metrics = compute_metrics(
            np.array(cat_preds),  np.array(cat_labels),
            np.array(imp_preds),  np.array(imp_targets),
            np.array(qual_preds), np.array(qual_labels),
        )
        metrics["val_loss"] = val_loss / n_batches
        return metrics

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        model:    CortexResearchModel,
        optimizer,
        metrics:  Dict[str, float],
        epoch:    int,
    ) -> Path:
        ckpt = {
            "epoch":            epoch,
            "global_step":      self.global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "metrics":          metrics,
            "config":           asdict(self.cfg),
            "version":          "2.0",
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()

        path = Path(self.cfg.model_save_dir) / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(ckpt, path)

        if metrics.get("category_f1", 0.0) > self.best_f1:
            self.best_f1 = metrics["category_f1"]
            best = Path(self.cfg.model_save_dir) / "best_model.pt"
            torch.save(ckpt, best)
            mlflow.pytorch.log_model(model, "best_model")
            logger.info("✓ New best  —  val F1 = %.4f (epoch %d)", self.best_f1, epoch)
        return path

    # ── Main train loop ────────────────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> CortexResearchModel:
        effective_steps = (
            len(train_loader) * self.cfg.num_epochs
            // self.cfg.gradient_accumulation_steps
        )

        model     = self._build_model()
        optimizer, scheduler = self._build_optimizer_and_scheduler(model, effective_steps)
        criterion = MultiTaskLoss(
            focal_gamma        = self.cfg.focal_gamma,
            label_smoothing    = self.cfg.label_smoothing,
            lambda_supcon      = self.cfg.lambda_supcon,
            lambda_mrl         = self.cfg.lambda_mrl,
            lambda_rank        = self.cfg.lambda_rank,
            supcon_warmup_steps= self.cfg.supcon_warmup_steps,
        )

        if self.cfg.use_ema:
            self.ema = ModelEMA(model, decay=self.cfg.ema_decay, warm_up=self.cfg.ema_warmup)
            logger.info("EMA enabled (decay=%.4f, warmup=%d)", self.cfg.ema_decay, self.cfg.ema_warmup)

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow_experiment_name)

        with mlflow.start_run(run_name=self.cfg.run_name) as run:
            mlflow.log_params({**asdict(self.cfg),
                               "num_parameters":          model.num_parameters,
                               "num_trainable_parameters":model.num_trainable_parameters,
                               "lora_parameters":         model.lora_parameters,
                               "device":                  str(self.device)})
            logger.info("MLflow run: %s", run.info.run_id)

            # Model summary at training start
            mlflow.log_text(
                f"CortexResearchModel v2\n"
                f"Backbone:    {model._backbone_name}\n"
                f"Total:       {model.num_parameters/1e6:.1f}M\n"
                f"Trainable:   {model.num_trainable_parameters/1e6:.1f}M\n"
                f"LoRA params: {model.lora_parameters/1e3:.0f}k\n",
                "model_summary.txt"
            )

            for epoch in range(1, self.cfg.num_epochs + 1):
                t0 = time.time()

                # Progressive unfreeze (ULMFiT)
                self._unfreeze_backbone(model, epoch)

                train_losses = self._train_epoch(model, train_loader, optimizer, scheduler, criterion)

                val_metrics: Dict[str, float] = {}
                if epoch % self.cfg.eval_every_n_epochs == 0:
                    # Evaluate EMA model if available (usually better)
                    eval_model = self.ema.shadow if self.ema else model
                    val_metrics = self._evaluate(eval_model, val_loader, criterion)

                    mlflow.log_metrics({
                        "epoch/train_loss":         train_losses["total"],
                        "epoch/train_supcon":       train_losses["supcon"],
                        "epoch/train_mrl":          train_losses["mrl"],
                        "epoch/val_loss":           val_metrics["val_loss"],
                        "epoch/category_accuracy":  val_metrics["category_accuracy"],
                        "epoch/category_f1":        val_metrics["category_f1"],
                        "epoch/impact_mse":         val_metrics["impact_mse"],
                        "epoch/impact_r2":          val_metrics["impact_r2"],
                        "epoch/quality_accuracy":   val_metrics["quality_accuracy"],
                        "epoch/quality_f1":         val_metrics["quality_f1"],
                    }, step=epoch)

                    self._save_checkpoint(model, optimizer, val_metrics, epoch)

                elapsed = time.time() - t0
                logger.info(
                    "Epoch %3d/%d  [%.1fs]  "
                    "train=%.4f  (cls=%.3f sup=%.3f mrl=%.3f)  "
                    "val_F1=%.4f  val_loss=%.4f",
                    epoch, self.cfg.num_epochs, elapsed,
                    train_losses["total"],
                    train_losses["classification"],
                    train_losses["supcon"],
                    train_losses["mrl"],
                    val_metrics.get("category_f1", float("nan")),
                    val_metrics.get("val_loss", float("nan")),
                )

            mlflow.log_metric("best_val_f1", self.best_f1)

        logger.info("Training complete.  Best val F1: %.4f", self.best_f1)
        return model


# ── CLI entry-point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Cortex Research Intelligence Model v2")
    parser.add_argument("--config",   type=str, default="ml/training/config.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    if Path(args.config).exists():
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    if args.run_name:
        cfg.run_name = args.run_name

    train_loader, val_loader, _ = create_dataloaders(cfg)
    Trainer(cfg).train(train_loader, val_loader)


if __name__ == "__main__":
    main()
