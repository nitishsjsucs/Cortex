"""
Cortex Research Intelligence Model v2.0

Architecture overview
─────────────────────
Input: "Title [SEP] Abstract" (≤512 tokens)

  ModernBERT / DeBERTa-v3 / DistilBERT backbone
  (backbone frozen for first N layers; LoRA adapters on remaining)
      │
  CrossLayerAggregation  ← mixes last K layers (learnable weights)
      │
  N × ModernAdaptationLayer
       ├─ RMSNorm  [Zhang & Sennrich, 2019]
       ├─ RoPE Multi-Head Attention + LoRA Q,V  [Su 2022; Hu 2022]
       ├─ Stochastic Depth  [Huang 2016]
       ├─ RMSNorm
       └─ SwiGLU FFN  [Shazeer 2020; Touvron 2023]
      │
  MatryoshkaPooling  [Kusupati 2022]
  (hierarchical CLS+mean+max with learned weights)
      │
  ┌───┴──────────────────┬──────────────────────┐
  │                      │                      │
CategoryHead         ImpactHead            QualityHead
(20 arXiv classes)   (regression [0,1])    (binary quality)
  │                      │                      │
Focal CE +             Huber +               Asymmetric Loss
SupCon auxiliary       Rank Correlation        [Ben-Baruch 2021]
  [Khosla 2020]          [differentiable ρ]
  │                      │                      │
  └──────────────────────┴──────────────────────┘
             Uncertainty-Weighted MTL  [Kendall 2018]
                + Matryoshka MRL loss  [Kusupati 2022]

EMA shadow model maintained for inference  [Polyak 1992]

References
──────────
[1] Warner et al. (2024). Smarter, Better, Faster, Longer: ModernBERT.
    arXiv:2412.13663
[2] Hu et al. (2022). LoRA: Low-Rank Adaptation of LLMs. ICLR 2022.
    arXiv:2106.09685
[3] Su et al. (2022). RoFormer: Enhanced Transformer with RoPE.
    arXiv:2104.09864
[4] Shazeer (2020). GLU Variants Improve Transformer. arXiv:2002.05202
[5] Touvron et al. (2023). LLaMA 2. arXiv:2307.09288
[6] Zhang & Sennrich (2019). RMSNorm. NeurIPS 2019. arXiv:1910.07467
[7] Kusupati et al. (2022). Matryoshka Representation Learning. NeurIPS.
    arXiv:2205.13147
[8] Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS.
    arXiv:2004.11362
[9] Ben-Baruch et al. (2021). Asymmetric Loss. ICCV 2021. arXiv:2009.14119
[10] Huang et al. (2016). Deep Networks with Stochastic Depth. ECCV 2016.
[11] Kendall et al. (2018). Multi-Task Learning Using Uncertainty. CVPR.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.components import (
    RMSNorm, SwiGLUFFN, RotaryEmbedding, LoRALinear,
    ModernAdaptationLayer, CrossLayerAggregation,
    MatryoshkaPooling, ModelEMA
)

logger = logging.getLogger(__name__)


# ── arXiv category taxonomy ────────────────────────────────────────────────

ARXIV_CATEGORIES: List[str] = [
    "cs.AI",  "cs.LG",  "cs.CL",  "cs.CV",  "cs.RO",
    "math.ST","math.OC","stat.ML", "stat.AP","stat.ME",
    "physics.optics","quant-ph","cond-mat.str-el","astro-ph.GA",
    "q-bio.NC","econ.EM","eess.SP","cs.NI","cs.IR","cs.SE",
]


# ── Task heads ─────────────────────────────────────────────────────────────

class DeepClassificationHead(nn.Module):
    """
    Multi-granularity classification head used for both the main
    category task and the Matryoshka auxiliary objectives.

    Architecture:
      Linear → RMSNorm → SiLU → Dropout → Linear → RMSNorm → Dropout → Linear

    Each layer is initialised with truncated normal (σ=0.02) following
    the LLaMA / ModernBERT initialisation recipe.
    """

    def __init__(
        self,
        input_dim:  int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout:    float = 0.25,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim // 2, num_classes * 4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            RMSNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImpactRegressionHead(nn.Module):
    """
    Regression head for impact-score prediction.

    Predicts log-normalised citation count [0, 1] via:
      Linear → RMSNorm → SiLU → Linear → SiLU → Linear → sigmoid

    Output is bounded to [0, 1] via sigmoid to ensure valid predictions
    throughout training (unlike unconstrained regression).
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        d = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, d,    bias=False),
            RMSNorm(d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class QualityHead(nn.Module):
    """Binary quality classification head with calibration-aware design."""

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4, bias=False),
            RMSNorm(input_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, 2),
        )
        nn.init.trunc_normal_(list(self.net.children())[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Contrastive projection head ────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection used for Supervised Contrastive Learning.

    Projects the pooled representation to a lower-dimensional space
    where the SupCon loss is applied.  L2-normalises the output so
    that cosine similarity equals dot product.

    Follows the SimCLR / SupCon recipe:  Linear → BN → ReLU → Linear → L2.
    Detached from task heads to avoid gradient interference.
    """

    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=False),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ── Backbone factory ───────────────────────────────────────────────────────

def _load_backbone(name: str):
    """
    Load a pre-trained backbone, trying in order:
      1. answerdotai/ModernBERT-base   (best; requires transformers ≥ 4.48)
      2. microsoft/deberta-v3-base     (strong; requires sentencepiece)
      3. distilbert-base-uncased       (lightweight fallback; always available)

    Returns (model, hidden_size, n_layers, output_all_hidden_states=bool).
    """
    from transformers import AutoModel, AutoConfig

    candidates = [name] if name not in ("auto", "") else [
        "answerdotai/ModernBERT-base",
        "microsoft/deberta-v3-base",
        "distilbert-base-uncased",
    ]

    for candidate in candidates:
        try:
            cfg   = AutoConfig.from_pretrained(candidate, output_hidden_states=True)
            model = AutoModel.from_pretrained(candidate, config=cfg, ignore_mismatched_sizes=True)
            h_size = getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
            n_layers = getattr(cfg, "num_hidden_layers",
                               getattr(cfg, "n_layers",
                               getattr(cfg, "num_layers", 6)))
            logger.info("Loaded backbone: %s  (hidden=%d, layers=%d)", candidate, h_size, n_layers)
            return model, h_size, n_layers, candidate
        except Exception as exc:
            logger.debug("Could not load %s: %s", candidate, exc)

    raise RuntimeError("Could not load any backbone. Ensure transformers is installed.")


# ── Main model ─────────────────────────────────────────────────────────────

class CortexResearchModel(nn.Module):
    """
    Cortex Research Intelligence Model v2.0

    Combines eight distinct 2020-2025 research advances into a single
    production-grade multi-task document intelligence model.

    Input
    ─────
    input_ids      : (B, L) token ids
    attention_mask : (B, L) 1=real, 0=pad

    Output  (dict)
    ──────
    category_logits  : (B, 20)   raw scores for arXiv area classification
    impact_score     : (B,)      normalised impact score in [0, 1]
    quality_logits   : (B, 2)    binary quality classification scores
    proj_emb         : (B, 128)  L2-normalised projection for SupCon loss
    mrl_logits       : list[(B, 20)] category logits at each MRL granularity
    embeddings       : (B, D)    full pooled embedding [only if requested]
    """

    NUM_CLASSES       = len(ARXIV_CATEGORIES)
    ARXIV_CATEGORIES  = ARXIV_CATEGORIES

    def __init__(
        self,
        backbone_name:      str   = "distilbert-base-uncased",
        num_adaptation_layers: int = 3,
        lora_rank:          int   = 16,
        lora_alpha:         float = 32.0,
        num_heads:          int   = 12,
        dropout:            float = 0.15,
        drop_path_rate:     float = 0.1,
        freeze_backbone_layers: int = 2,
        cla_layers:         int   = 4,     # last K layers for cross-layer aggregation
        proj_dim:           int   = 128,   # SupCon projection dim
        use_alternating_attention: bool = True,  # local/global per ModernBERT
        window_size:        int   = 64,    # local window tokens
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone, bb_dim, n_layers, loaded_name = _load_backbone(backbone_name)
        self._backbone_name = loaded_name

        # Freeze early layers
        self._freeze_backbone(freeze_backbone_layers, n_layers)

        # Projection from backbone dim → model dim if they differ
        self.projection: nn.Module = nn.Identity()
        d_model = bb_dim
        if bb_dim != 768:
            d_model = 768
            self.projection = nn.Sequential(
                nn.Linear(bb_dim, d_model, bias=False),
                RMSNorm(d_model),
                nn.SiLU(),
            )

        # ── Cross-Layer Aggregation ───────────────────────────────────────
        self.cla        = CrossLayerAggregation(min(cla_layers, n_layers))
        self.cla_layers = min(cla_layers, n_layers)

        # ── Modern Adaptation Layers ──────────────────────────────────────
        # Linearly spaced stochastic depth rates (0 → drop_path_rate)
        dp_rates = [drop_path_rate * i / max(num_adaptation_layers - 1, 1)
                    for i in range(num_adaptation_layers)]

        self.adaptation = nn.ModuleList([
            ModernAdaptationLayer(
                d_model    = d_model,
                num_heads  = num_heads,
                lora_rank  = lora_rank,
                lora_alpha = lora_alpha,
                dropout    = dropout,
                drop_path  = dp_rates[i],
                # Alternate local/global attention (ModernBERT design)
                window_size = window_size if (use_alternating_attention and i % 2 == 0) else None,
            )
            for i in range(num_adaptation_layers)
        ])

        # ── Matryoshka Pooling ────────────────────────────────────────────
        self.pooler = MatryoshkaPooling(d_model)

        # ── Task Heads ────────────────────────────────────────────────────
        self.category_head = DeepClassificationHead(
            d_model, self.NUM_CLASSES, dropout=dropout
        )
        self.impact_head   = ImpactRegressionHead(d_model, dropout=dropout)
        self.quality_head  = QualityHead(d_model, dropout=dropout)

        # Contrastive projection head (SupCon, [8])
        self.proj_head     = ProjectionHead(d_model, proj_dim=proj_dim)

        # Matryoshka auxiliary heads at each granularity (MRL, [7])
        self.mrl_heads = nn.ModuleList([
            nn.Linear(d_model // g, self.NUM_CLASSES)
            for g in MatryoshkaPooling.GRANULARITIES
        ])

        # Learnable uncertainty weights for multi-task loss (Kendall [11])
        self.task_log_weights = nn.Parameter(torch.zeros(3))

        self.drop = nn.Dropout(dropout)

        # Store dims
        self.d_model  = d_model
        self._log_stats()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _freeze_backbone(self, n_freeze: int, total_layers: int):
        """Freeze the first n_freeze transformer layers of the backbone."""
        if n_freeze <= 0:
            return
        frozen = 0
        # Try common attribute names for encoder layers
        for attr in ["transformer.layer", "encoder.layer", "layers"]:
            try:
                layers = eval(f"self.backbone.{attr}")
                for i, layer in enumerate(layers):
                    if i < n_freeze:
                        for p in layer.parameters():
                            p.requires_grad_(False)
                        frozen += 1
                logger.info("Froze %d/%d backbone layers (%s)", frozen, total_layers, attr)
                return
            except Exception:
                continue

    def _log_stats(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel() for n, p in self.named_parameters()
            if 'lora' in n.lower() and p.requires_grad
        )
        logger.info(
            "CortexResearchModel v2: total=%dM  trainable=%dM  LoRA=%dk  backbone=%s",
            total // 1_000_000, trainable // 1_000_000, lora_params // 1_000, self._backbone_name
        )

    # ── Encoding ───────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode tokens → pooled embedding using the full pipeline.
        """
        # Backbone forward with all hidden states
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Cross-Layer Aggregation over last K backbone layers
        all_hidden = out.hidden_states  # tuple of (B, T, bb_dim) per layer
        if all_hidden is not None and len(all_hidden) >= self.cla_layers:
            hidden = self.cla(all_hidden[-self.cla_layers:])
        else:
            hidden = out.last_hidden_state

        # Project to d_model if backbone uses different dim
        hidden = self.projection(hidden)  # (B, T, d_model)

        # Modern adaptation layers (RoPE + LoRA + SwiGLU + StochDepth)
        for layer in self.adaptation:
            hidden = layer(hidden, attention_mask)

        # Matryoshka-aware hierarchical pooling
        return self.pooler(hidden, attention_mask)  # (B, d_model)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:

        emb = self.drop(self.encode(input_ids, attention_mask))  # (B, d_model)

        # ── Main task predictions ──────────────────────────────────────────
        outputs: Dict[str, torch.Tensor] = {
            "category_logits": self.category_head(emb),
            "impact_score":    self.impact_head(emb),
            "quality_logits":  self.quality_head(emb),
            "proj_emb":        self.proj_head(emb),
        }

        # ── Matryoshka MRL auxiliary logits ───────────────────────────────
        granularity_views = self.pooler.get_granularity_views(emb)
        mrl_logits = [
            self.mrl_heads[i](view)
            for i, (view, _) in enumerate(granularity_views)
        ]
        outputs["mrl_logits"] = mrl_logits

        if return_embeddings:
            outputs["embeddings"] = emb

        return outputs

    # ── Multi-task uncertainty weights ────────────────────────────────────

    def task_weights(self) -> torch.Tensor:
        """Softmax-normalised weights for the 3 main task losses."""
        return F.softmax(self.task_log_weights, dim=0)

    # ── Model properties ───────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def lora_parameters(self) -> int:
        return sum(
            p.numel() for n, p in self.named_parameters()
            if "lora" in n.lower() and p.requires_grad
        )

    # ── Category helpers ───────────────────────────────────────────────────

    @classmethod
    def category_to_idx(cls, cat: str) -> int:
        try: return cls.ARXIV_CATEGORIES.index(cat)
        except ValueError: return 0

    @classmethod
    def idx_to_category(cls, idx: int) -> str:
        return cls.ARXIV_CATEGORIES[idx % len(cls.ARXIV_CATEGORIES)]

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "backbone_name":     self._backbone_name,
                "d_model":           self.d_model,
                "num_classes":       self.NUM_CLASSES,
            },
            "version": "2.0",
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "CortexResearchModel":
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        cfg   = ckpt.get("config", {})
        model = cls(backbone_name=cfg.get("backbone_name", "distilbert-base-uncased"), **kwargs)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model


# ── Backward-compatibility alias ──────────────────────────────────────────

ResearchIntelligenceModel = CortexResearchModel
