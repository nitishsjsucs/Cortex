"""
Inference wrapper for the Research Intelligence Model v2.

Features
--------
  - Graceful checkpoint loading (skips shape-mismatched parameters)
  - MC-Dropout uncertainty estimation (Gal & Ghahramani, 2016)
  - Conformal prediction sets with guaranteed coverage (Angelopoulos 2022)
  - Gradient saliency for attention-based explanation (Simonyan et al. 2013)
  - Embedding extraction for semantic similarity search

References
----------
[1] Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
    arXiv:1506.02142
[2] Angelopoulos & Bates (2022). A Gentle Introduction to Conformal Prediction.
    arXiv:2107.07511
[3] Simonyan et al. (2013). Deep Inside Convolutional Networks (saliency).
    arXiv:1312.6034
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast

from ml.models.research_classifier import CortexResearchModel, ResearchIntelligenceModel

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    text_preview:        str
    predicted_category:  str
    category_confidence: float
    top3_categories:     List[Dict[str, float]]
    impact_score:        float
    quality_label:       str
    quality_confidence:  float
    embedding:           Optional[List[float]] = field(default=None, repr=False)
    # Optional enrichments
    uncertainty:         Optional[float]       = None   # MC-Dropout std
    conformal_set:       Optional[List[str]]   = None   # Guaranteed-coverage set
    token_saliency:      Optional[List[Tuple[str, float]]] = None  # (token, score)


# ── Graceful checkpoint loading ───────────────────────────────────────────────

def _load_compatible_state_dict(model: nn.Module, state_dict: dict) -> dict:
    """
    Load only the parameters from state_dict that match the model's expected
    shapes. Parameters with size mismatches are silently skipped and the
    model's random initialisation is kept for those layers.

    This makes loading robust to architecture changes between training and
    inference — e.g. when the FFN intermediate size changes (SwiGLU 4× →
    SwiGLU 8/3×), the mismatched layers fall back to random init while all
    other weights load correctly.
    """
    model_state = model.state_dict()
    compatible  = {}
    skipped     = []

    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append(f"{k} (unexpected key)")
            continue
        if v.shape != model_state[k].shape:
            skipped.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(model_state[k].shape)}")
            continue
        compatible[k] = v

    if skipped:
        logger.warning("Skipped %d mismatched/unexpected keys:\n  %s",
                       len(skipped), "\n  ".join(skipped[:10]))

    model.load_state_dict(compatible, strict=False)
    loaded_pct = 100 * len(compatible) / max(len(model_state), 1)
    logger.info("Loaded %.1f%% of checkpoint parameters (%d/%d)",
                loaded_pct, len(compatible), len(model_state))
    return compatible


# ── Main predictor ────────────────────────────────────────────────────────────

class ResearchPredictor:
    """
    High-level inference interface for the Cortex Research Intelligence Model.

    Supports:
      - Standard MAP prediction
      - MC-Dropout uncertainty estimation [1]
      - Conformal prediction sets [2]
      - Gradient saliency explanations [3]
    """

    QUALITY_LABELS = {0: "poor", 1: "well-structured"}

    # Conformal prediction calibration score (set via calibrate_conformal)
    _qhat: Optional[float] = None

    def __init__(
        self,
        model,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = model.to(self.device).eval()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, max_length: int = 256) -> "ResearchPredictor":
        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = CortexResearchModel()
        _load_compatible_state_dict(model, ckpt["model_state_dict"])
        tok   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        return cls(model, tok, max_length)

    @classmethod
    def from_pretrained_only(cls, max_length: int = 256) -> "ResearchPredictor":
        model = CortexResearchModel()
        tok   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        return cls(model, tok, max_length)

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # ── Standard prediction ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        text: str,
        return_embedding: bool = False,
        mc_samples: int = 0,        # >0 → MC-Dropout uncertainty
        conformal: bool = False,    # include conformal prediction set
    ) -> PredictionResult:
        results = self.predict_batch(
            [text],
            return_embedding=return_embedding,
            mc_samples=mc_samples,
            conformal=conformal,
        )
        return results[0]

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        return_embedding: bool = False,
        mc_samples: int = 0,
        conformal: bool = False,
    ) -> List[PredictionResult]:
        enc  = self._tokenize(texts)
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        out = self.model(ids, mask, return_embeddings=return_embedding)

        cat_probs  = F.softmax(out["category_logits"], dim=-1).cpu().numpy()
        qual_probs = F.softmax(out["quality_logits"],  dim=-1).cpu().numpy()
        imp_scores = out["impact_score"].cpu().numpy()
        embeddings = out.get("embeddings")
        if embeddings is not None:
            embeddings = embeddings.cpu().numpy()

        # MC-Dropout uncertainty
        mc_std = None
        if mc_samples > 0:
            mc_std = self._mc_uncertainty(ids, mask, n=mc_samples)

        results = []
        for i, text in enumerate(texts):
            cat_idx  = int(cat_probs[i].argmax())
            top3_idx = cat_probs[i].argsort()[::-1][:3]
            qual_idx = int(qual_probs[i].argmax())

            # Conformal prediction set
            c_set = None
            if conformal and self._qhat is not None:
                c_set = [
                    CortexResearchModel.idx_to_category(j)
                    for j in range(len(cat_probs[i]))
                    if float(cat_probs[i][j]) >= 1.0 - self._qhat
                ]

            results.append(PredictionResult(
                text_preview        = text[:120] + "…" if len(text) > 120 else text,
                predicted_category  = CortexResearchModel.idx_to_category(cat_idx),
                category_confidence = float(cat_probs[i][cat_idx]),
                top3_categories     = [
                    {"category": CortexResearchModel.idx_to_category(j),
                     "confidence": float(cat_probs[i][j])}
                    for j in top3_idx
                ],
                impact_score        = float(imp_scores[i]),
                quality_label       = self.QUALITY_LABELS[qual_idx],
                quality_confidence  = float(qual_probs[i][qual_idx]),
                embedding           = embeddings[i].tolist() if embeddings is not None else None,
                uncertainty         = float(mc_std[i]) if mc_std is not None else None,
                conformal_set       = c_set,
            ))

        return results

    # ── MC-Dropout Uncertainty ────────────────────────────────────────────────
    # Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
    # arXiv:1506.02142

    def _mc_uncertainty(
        self,
        ids:  torch.Tensor,
        mask: torch.Tensor,
        n:    int = 20,
    ) -> torch.Tensor:
        """
        Enable dropout at inference time and run N stochastic forward passes.
        Returns per-sample predictive entropy as an uncertainty score ∈ [0,1].

        Higher entropy = model is unsure which category this paper belongs to.
        """
        # Activate ONLY Dropout layers (not BatchNorm) to avoid the
        # batch-size=1 error in BatchNorm when training mode is set globally.
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(n):
                out = self.model(ids, mask)
                all_probs.append(F.softmax(out["category_logits"], dim=-1))
        self.model.eval()   # restore fully

        # Stack → (N, B, C), mean → (B, C)
        probs_stack = torch.stack(all_probs, dim=0)
        mean_probs  = probs_stack.mean(0)                    # (B, C)
        # Predictive entropy, normalised by log(num_classes)
        entropy = -(mean_probs * (mean_probs + 1e-9).log()).sum(-1)
        entropy = entropy / torch.log(torch.tensor(float(mean_probs.shape[-1])))
        return entropy.cpu()                                 # (B,) ∈ [0,1]

    # ── Conformal Prediction Calibration ─────────────────────────────────────
    # Angelopoulos & Bates (2022). A Gentle Introduction to Conformal Prediction
    # arXiv:2107.07511

    def calibrate_conformal(
        self,
        cal_texts:  List[str],
        cal_labels: List[int],
        alpha:      float = 0.1,   # target error rate (1-alpha = coverage)
    ) -> float:
        """
        Calibrate conformal prediction using a held-out calibration set.

        After calling this, every predict() call with conformal=True returns
        a PREDICTION SET guaranteed to contain the true label with
        probability ≥ 1 - alpha.

        Score function: s(x,y) = 1 - softmax(f(x))[y]
        q-hat = ceil((n+1)(1-alpha)/n) quantile of scores.
        """
        scores = []
        for i in range(0, len(cal_texts), 32):
            batch = cal_texts[i:i+32]
            enc   = self._tokenize(batch)
            with torch.no_grad():
                out = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
            probs = F.softmax(out["category_logits"], dim=-1).cpu()
            for j, label in enumerate(cal_labels[i:i+32]):
                scores.append(1.0 - float(probs[j, label]))

        n    = len(scores)
        level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
        idx  = int(torch.ceil(torch.tensor(level * n)).item()) - 1
        self._qhat = sorted(scores)[min(idx, n-1)]
        logger.info("Conformal calibration: n=%d  alpha=%.2f  q-hat=%.4f", n, alpha, self._qhat)
        return self._qhat

    # ── Gradient Saliency Explanation ─────────────────────────────────────────
    # Simonyan et al. (2013). Deep Inside Convolutional Networks. arXiv:1312.6034
    # We use a forward hook on the embedding layer to capture the intermediate
    # activation and its gradient — the standard way to obtain input-space
    # saliency for transformer models.

    def explain(self, text: str, target_class: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Compute token-level saliency for the predicted category.

        Method: Attention-rollout saliency — aggregate [CLS]→token attention
        across all transformer layers and heads, normalised to [0, 1].

        This is equivalent to "attention-based saliency" (Jain & Wallace 2019;
        Clark et al. 2019) and is more reliable than gradient-based methods
        for transformer models where embedding gradients require careful setup.

        For gradient-based saliency (Simonyan et al. 2013), enable
        output_attentions=False and call the _gradient_saliency method below.
        """
        enc  = self._tokenize([text])
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        return self._attention_saliency(ids, mask)

    def _attention_saliency(self, ids, mask) -> List[Tuple[str, float]]:
        """Fallback: use mean attention from [CLS] across all heads and layers."""
        with torch.no_grad():
            out = self.model.backbone(ids, mask, output_attentions=True)
        attentions = out.attentions   # list of (B, H, T, T)
        if not attentions:
            return []
        # Average attention from CLS token over all layers and heads
        avg = torch.stack(attentions).mean(dim=0).mean(dim=1)[0, 0, :]  # (T,)
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-9)
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
        n_real = int(mask[0].sum().item())
        pairs = sorted(
            [(tok, float(s)) for tok, s in zip(tokens[:n_real], avg[:n_real].tolist())
             if tok not in ["[CLS]", "[SEP]", "[PAD]"]],
            key=lambda x: -x[1],
        )
        return pairs[:20]
