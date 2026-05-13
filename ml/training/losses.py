"""
Loss functions for Cortex Research Intelligence Model v2.

All losses reference the original papers that introduced them.

Losses
──────
FocalLoss               — Lin et al., ICCV 2017 (arXiv:1708.02002)
AsymmetricLoss          — Ben-Baruch et al., ICCV 2021 (arXiv:2009.14119)
LabelSmoothingCE        — Szegedy et al., CVPR 2016 (InceptionV4 recipe)
SupervisedContrastive   — Khosla et al., NeurIPS 2020 (arXiv:2004.11362)
MatryoshkaLoss          — Kusupati et al., NeurIPS 2022 (arXiv:2205.13147)
DiffSpearmanLoss        — Differentiable Spearman ρ (Blondel et al., 2020)
HuberLoss               — Huber, 1964 (AMS) — robust regression
MultiTaskLoss           — Kendall et al., CVPR 2018 (arXiv:1705.07115)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss — Lin et al., ICCV 2017
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = −α(1 − p_t)^γ log(p_t)

    Reduces contribution of easy examples (high p_t) so the optimiser
    focuses on hard, misclassified samples.  Essential for the long-tailed
    arXiv category distribution where cs.LG >> econ.EM.

    With γ=2 the loss for a well-classified example (p_t=0.9) is
    100× smaller than for a hard example (p_t=0.1).
    """

    def __init__(
        self,
        num_classes: int,
        gamma:       float = 2.0,
        alpha:       Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()

        # Label smoothing: soft targets
        if self.label_smoothing > 0:
            smooth  = self.label_smoothing / self.num_classes
            one_hot = torch.zeros_like(prob).scatter_(1, targets.unsqueeze(1), 1.0)
            targets_soft = one_hot * (1 - self.label_smoothing) + smooth
            nll = -(log_prob * targets_soft).sum(-1)
        else:
            nll = F.nll_loss(log_prob, targets, weight=self.alpha, reduction="none")

        p_t   = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t) ** self.gamma * nll
        return focal.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Asymmetric Loss — Ben-Baruch et al., ICCV 2021
# ══════════════════════════════════════════════════════════════════════════════

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for imbalanced binary / multi-label classification.

    Applies *different* focusing parameters for positive and negative samples:
    - Positive:  (1 − p)^γ_pos · CE(p)
    - Negative:  p_shifted^γ_neg · CE(p_shifted)
                 where p_shifted = max(0, p − m_neg)

    The shift margin m_neg discards very easy negatives (p ≪ m_neg → loss=0),
    concentrating gradient on hard negatives.  This is especially important
    for quality classification where "poor" abstracts are rare.

    Default γ_pos=0, γ_neg=4, m_neg=0.05 as in the original paper.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        m_neg:     float = 0.05,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_pos    = gamma_pos
        self.gamma_neg    = gamma_neg
        self.m_neg        = m_neg
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # For binary classification
        prob     = torch.sigmoid(logits[:, 1] - logits[:, 0])
        targets  = targets.float()

        # Positive / negative probabilities
        p_pos    = prob
        p_neg    = (prob - self.m_neg).clamp(min=0)

        # Binary CE
        ce_pos   = -targets * torch.log(p_pos.clamp(min=1e-8))
        ce_neg   = -(1 - targets) * torch.log((1 - p_neg).clamp(min=1e-8))

        focal_pos = (1 - p_pos) ** self.gamma_pos * ce_pos
        focal_neg = p_neg       ** self.gamma_neg * ce_neg

        loss = focal_pos + focal_neg
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Supervised Contrastive Loss — Khosla et al., NeurIPS 2020
# ══════════════════════════════════════════════════════════════════════════════

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.

    Uses ground-truth category labels to define positives:
      Positives  = all other samples in the mini-batch with the same label
      Negatives  = all samples with a different label

    L_sup = (1/|I|) Σᵢ (−1/|P(i)|) Σ_{p∈P(i)}
                log( exp(zᵢ·z_p/τ) / Σ_{a≠i} exp(zᵢ·z_a/τ) )

    where z are L2-normalised projection embeddings and τ is temperature.

    Key advantage over instance discrimination (SimCLR):
    Multiple positives per anchor make gradients richer and more consistent.
    Category F1 improvements of 3-8% reported on text classification benchmarks.

    Args:
        temperature: τ in the softmax.  0.07 follows SimCLR / SupCon papers.
        min_positives: minimum positives per anchor to include in loss.
    """

    def __init__(self, temperature: float = 0.07, min_positives: int = 1):
        super().__init__()
        self.temp = temperature
        self.min_pos = min_positives

    def forward(
        self,
        proj_emb: torch.Tensor,  # (B, D) — L2-normalised projections
        labels:   torch.Tensor,  # (B,)   — integer class labels
    ) -> torch.Tensor:
        B = proj_emb.shape[0]
        device = proj_emb.device

        # Pairwise cosine similarity (proj_emb already L2-normalised)
        sim = torch.mm(proj_emb, proj_emb.t()) / self.temp  # (B, B)

        diag     = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag  # (B,B)

        valid = pos_mask.sum(1) >= self.min_pos
        if valid.sum() == 0:
            return proj_emb.sum() * 0.0  # differentiable zero

        # Use -1e4 (not -inf) on diagonal to avoid inf - inf = NaN in log_prob
        sim_no_self = sim.masked_fill(diag, -1e4)
        log_denom   = torch.logsumexp(sim_no_self, dim=1, keepdim=True)  # (B,1)

        log_prob = sim - log_denom  # (B, B)

        # Mean log-prob over positive pairs for each anchor
        mean_log_prob_pos = (
            (log_prob * pos_mask.float()).sum(1)
            / pos_mask.float().sum(1).clamp(min=1)
        )

        return -mean_log_prob_pos[valid].mean()


# ══════════════════════════════════════════════════════════════════════════════
# Matryoshka Loss — Kusupati et al., NeurIPS 2022
# ══════════════════════════════════════════════════════════════════════════════

class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning auxiliary loss.

    Computes a classification loss at each embedding granularity
    [d, d//2, d//4, d//8] and aggregates with 1/log(dᵢ) weights.

    The weighting emphasises smaller granularities (harder objectives),
    ensuring the model produces useful compact representations.

    At inference, any prefix [0:d'] of the embedding can be used directly.
    """

    def __init__(
        self,
        granularities: List[int] = (1, 2, 4, 8),
        base_loss:     nn.Module = None,
    ):
        super().__init__()
        self.granularities = granularities
        self.base_loss = base_loss or nn.CrossEntropyLoss(label_smoothing=0.1)

        # 1/log(granularity) weighting — emphasise smaller dims
        w = torch.tensor([1.0 / math.log(g + 1) for g in granularities])
        self.register_buffer("weights", w / w.sum())  # normalised

    def forward(
        self,
        mrl_logits: List[torch.Tensor],  # one per granularity
        labels:     torch.Tensor,
    ) -> torch.Tensor:
        assert len(mrl_logits) == len(self.granularities)
        loss = sum(
            self.weights[i].item() * self.base_loss(mrl_logits[i], labels)
            for i in range(len(self.granularities))
        )
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable Rank Correlation Loss — Blondel et al., 2020
# ══════════════════════════════════════════════════════════════════════════════

class SoftSpearmanLoss(nn.Module):
    """
    Differentiable approximation of Spearman rank correlation loss.

    For impact-score regression the *ordering* of predictions matters
    more than absolute values.  MSE treats all errors equally; rank-based
    losses focus on getting relative ordering right.

    Approach:
        1. Soft-rank both predictions and targets using a temperature τ
           (Blondel et al., ICML 2020 — "Fast Differentiable Sorting")
        2. Compute Spearman ρ on the soft ranks
        3. Return (1 − ρ) / 2 ∈ [0, 1]  (0 = perfect rank correlation)

    Combined with Huber loss:
        L_impact = λ_huber · L_huber + λ_rank · L_rank
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temp = temperature

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Temperature-smoothed ranking via pairwise comparisons."""
        diff = (x.unsqueeze(-1) - x.unsqueeze(-2)) / self.temp
        return torch.sigmoid(diff).sum(-1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_rank   = self._soft_rank(preds)
        target_rank = self._soft_rank(targets)

        # Centre ranks
        pr = pred_rank   - pred_rank.mean()
        tr = target_rank - target_rank.mean()

        # Spearman ρ via normalised dot product of centred ranks
        rho = (pr * tr).sum() / (pr.norm() * tr.norm()).clamp(min=1e-8)
        return (1.0 - rho) / 2.0  # convert to loss ∈ [0, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Task Loss — Kendall et al., CVPR 2018
# ══════════════════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Combines all task losses using:
      - Focal Loss + SupCon + MRL for category classification
      - Huber + Soft-Spearman for impact regression
      - AsymmetricLoss for binary quality classification
      - Learnable task uncertainty weights from model.task_log_weights

    The uncertainty-weighted combination avoids manual weight tuning:
      L = Σᵢ [ (1/2σᵢ²) Lᵢ + log σᵢ ]
    which is approximately equivalent to softmax(log_weights) × L.

    Additional regularisation:
      - Label smoothing (ε=0.1) in classification to prevent overconfidence
      - SupCon weight annealed: 0 → λ_supcon over first warmup steps
      - MRL weight constant at λ_mrl
    """

    def __init__(
        self,
        num_classes:          int   = 20,
        focal_gamma:          float = 2.0,
        label_smoothing:      float = 0.1,
        asym_gamma_neg:       float = 4.0,
        asym_margin:          float = 0.05,
        huber_delta:          float = 1.0,
        lambda_supcon:        float = 0.3,
        lambda_mrl:           float = 0.5,
        lambda_rank:          float = 0.3,
        supcon_temp:          float = 0.07,
        supcon_warmup_steps:  int   = 500,
    ):
        super().__init__()
        self.lambda_supcon       = lambda_supcon
        self.lambda_mrl          = lambda_mrl
        self.lambda_rank         = lambda_rank
        self.supcon_warmup_steps = supcon_warmup_steps

        self.focal      = FocalLoss(num_classes, gamma=focal_gamma, label_smoothing=label_smoothing)
        self.supcon     = SupervisedContrastiveLoss(temperature=supcon_temp)
        self.mrl        = MatryoshkaLoss()
        self.huber      = nn.HuberLoss(delta=huber_delta, reduction="mean")
        self.rank_loss  = SoftSpearmanLoss()
        self.asym_loss  = AsymmetricLoss(gamma_neg=asym_gamma_neg, m_neg=asym_margin)

    def forward(
        self,
        # Predictions
        category_logits: torch.Tensor,  # (B, 20)
        impact_pred:     torch.Tensor,  # (B,)
        quality_logits:  torch.Tensor,  # (B, 2)
        proj_emb:        torch.Tensor,  # (B, D) L2-normalised
        mrl_logits:      List[torch.Tensor],

        # Targets
        category_labels: torch.Tensor,  # (B,)
        impact_targets:  torch.Tensor,  # (B,) ∈ [0,1]
        quality_labels:  torch.Tensor,  # (B,) ∈ {0,1}

        # Multi-task uncertainty weights from model
        task_weights:    Optional[torch.Tensor] = None,
        global_step:     int = 0,
    ) -> Dict[str, torch.Tensor]:

        # ── Classification ─────────────────────────────────────────────────
        l_cls    = self.focal(category_logits, category_labels)

        # SupCon with linear warmup — skip computation entirely at step 0 to avoid
        # any edge-case NaN * 0 = NaN from IEEE 754 (e.g. batch with all unique labels)
        supcon_w = min(1.0, global_step / max(self.supcon_warmup_steps, 1)) * self.lambda_supcon
        if supcon_w == 0.0:
            l_supcon = proj_emb.sum() * 0.0   # differentiable zero
        else:
            l_supcon = self.supcon(proj_emb, category_labels) * supcon_w

        # MRL auxiliary
        l_mrl    = self.mrl(mrl_logits, category_labels) * self.lambda_mrl

        l_cat_total = l_cls + l_supcon + l_mrl

        # ── Impact regression ───────────────────────────────────────────────
        l_huber = self.huber(impact_pred, impact_targets.float())
        l_rank  = self.rank_loss(impact_pred, impact_targets.float()) * self.lambda_rank
        l_reg   = l_huber + l_rank

        # ── Quality classification ──────────────────────────────────────────
        l_qual  = self.asym_loss(quality_logits, quality_labels)

        # ── Uncertainty-weighted combination ───────────────────────────────
        if task_weights is not None:
            w = task_weights
            l_total = w[0] * l_cat_total + w[1] * l_reg + w[2] * l_qual
        else:
            l_total = l_cat_total + l_reg + l_qual

        return {
            "total":           l_total,
            "classification":  l_cls,
            "supcon":          l_supcon,
            "mrl":             l_mrl,
            "regression":      l_reg,
            "quality":         l_qual,
        }
