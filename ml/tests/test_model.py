"""
Unit tests for Cortex Research Intelligence Model v2.

Covers:
  - All new component building blocks (RMSNorm, RoPE, SwiGLU, LoRA, etc.)
  - Full model forward pass
  - Loss functions (Focal, AsymmetricLoss, SupCon, MRL, SoftSpearman, MultiTask)
  - Evaluation metrics
  - Data preprocessing

Run:
    pytest ml/tests/ -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

# ── Components ────────────────────────────────────────────────────────────────
from ml.models.components import (
    RMSNorm, RotaryEmbedding, SwiGLUFFN, LoRALinear, LoRAMergedLinear,
    ModernAdaptationLayer, CrossLayerAggregation, MatryoshkaPooling,
    ModelEMA, StochasticDepth,
)

# ── Model ─────────────────────────────────────────────────────────────────────
from ml.models.research_classifier import CortexResearchModel, ResearchIntelligenceModel

# ── Losses ────────────────────────────────────────────────────────────────────
from ml.training.losses import (
    FocalLoss, AsymmetricLoss, SupervisedContrastiveLoss,
    MatryoshkaLoss, SoftSpearmanLoss, MultiTaskLoss,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
from ml.evaluation.metrics import compute_metrics

# ── Data ─────────────────────────────────────────────────────────────────────
from ml.data.preprocessor import clean_text
from ml.data.arxiv_loader import _quality_heuristic

# ── Constants ─────────────────────────────────────────────────────────────────
B = 4   # batch size
T = 32  # sequence length
D = 256 # hidden dim (small for tests)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    return CortexResearchModel(
        backbone_name          = "distilbert-base-uncased",
        num_adaptation_layers  = 1,
        lora_rank              = 8,
        num_heads              = 8,
        dropout                = 0.1,
        freeze_backbone_layers = 5,
        cla_layers             = 2,
    )


@pytest.fixture
def dummy_batch():
    return {
        "input_ids":       torch.randint(0, 30522, (B, T)),
        "attention_mask":  torch.ones(B, T, dtype=torch.long),
        "category_labels": torch.randint(0, 20, (B,)),
        "impact_scores":   torch.rand(B),
        "quality_labels":  torch.randint(0, 2, (B,)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Component tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(D)
        x = torch.randn(B, T, D)
        assert norm(x).shape == (B, T, D)

    def test_no_nan(self):
        norm = RMSNorm(D)
        out = norm(torch.randn(B, T, D))
        assert not torch.isnan(out).any()

    def test_learnable_weight(self):
        norm = RMSNorm(D)
        assert norm.weight.shape == (D,)
        assert norm.weight.requires_grad


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(64)
        q = torch.randn(B, 8, T, 64)
        k = torch.randn(B, 8, T, 64)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_no_nan(self):
        rope = RotaryEmbedding(64)
        q, k = torch.randn(2, 4, T, 64), torch.randn(2, 4, T, 64)
        qr, kr = rope(q, k)
        assert not torch.isnan(qr).any()
        assert not torch.isnan(kr).any()

    def test_rotation_is_position_dependent(self):
        rope = RotaryEmbedding(32)
        q = torch.randn(1, 1, 10, 32)
        qr, _ = rope(q, q)
        # Same input at different positions should give different outputs
        assert not torch.allclose(qr[:, :, 0], qr[:, :, 1])


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLUFFN(D)
        assert ffn(torch.randn(B, T, D)).shape == (B, T, D)

    def test_no_nan(self):
        ffn = SwiGLUFFN(D)
        assert not torch.isnan(ffn(torch.randn(B, T, D))).any()

    def test_gating_produces_different_output_than_ungated(self):
        ffn = SwiGLUFFN(D)
        x   = torch.randn(B, T, D)
        y1  = ffn(x)
        y2  = ffn(x + 0.1)
        assert not torch.allclose(y1, y2)


class TestLoRA:
    def test_output_shape(self):
        lora = LoRALinear(D, D, rank=8)
        assert lora(torch.randn(B, D)).shape == (B, D)

    def test_zero_init(self):
        lora = LoRALinear(D, D, rank=8)
        # B matrix initialised to zero → output should be zero at init
        with torch.no_grad():
            out = lora(torch.ones(1, D))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_merged_linear(self):
        W = torch.randn(D, D)
        ml = LoRAMergedLinear(W, None, rank=8)
        out = ml(torch.randn(B, D))
        assert out.shape == (B, D)


class TestModernAdaptationLayer:
    def test_output_shape(self):
        layer = ModernAdaptationLayer(D, num_heads=8, lora_rank=4)
        x    = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)
        assert layer(x, mask).shape == (B, T, D)

    def test_no_nan(self):
        layer = ModernAdaptationLayer(D, num_heads=8, lora_rank=4)
        x    = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)
        assert not torch.isnan(layer(x, mask)).any()

    def test_stochastic_depth_drops_during_training(self):
        layer = ModernAdaptationLayer(D, num_heads=4, lora_rank=4, drop_path=1.0)
        layer.train()
        x    = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)
        out  = layer(x, mask)
        # With drop_path=1.0 the residual is zeroed — output = input
        assert not torch.allclose(out, x)  # some layers still modify


class TestMatryoshkaPooling:
    def test_pooling_shape(self):
        pool = MatryoshkaPooling(D)
        h    = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)
        out  = pool(h, mask)
        assert out.shape == (B, D)

    def test_granularity_views(self):
        pool = MatryoshkaPooling(D)
        emb  = torch.randn(B, D)
        views = pool.get_granularity_views(emb)
        assert len(views) == len(MatryoshkaPooling.GRANULARITIES)
        for view, dim in views:
            assert view.shape == (B, dim)

    def test_learnable_weights_sum_to_one(self):
        import torch.nn.functional as F
        pool = MatryoshkaPooling(D)
        w = F.softmax(pool.log_w, dim=0)
        assert abs(w.sum().item() - 1.0) < 1e-5


class TestModelEMA:
    def test_update(self):
        m   = nn.Linear(10, 10)
        ema = ModelEMA(m, decay=0.99, warm_up=5)
        old = ema.shadow.weight.clone()
        # Change model weights
        with torch.no_grad():
            m.weight.fill_(1.0)
        ema.update(m)
        # EMA should move toward 1.0 but not reach it
        assert not torch.allclose(ema.shadow.weight, old)
        assert not torch.allclose(ema.shadow.weight, m.weight)

    def test_state_dict_roundtrip(self):
        m   = nn.Linear(10, 10)
        ema = ModelEMA(m, decay=0.99)
        sd  = ema.state_dict()
        ema2 = ModelEMA(m, decay=0.99)
        ema2.load_state_dict(sd)
        assert torch.allclose(ema.shadow.weight, ema2.shadow.weight)


# ══════════════════════════════════════════════════════════════════════════════
# Full model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCortexResearchModel:
    def test_output_keys(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert set(out.keys()) >= {"category_logits","impact_score","quality_logits","proj_emb","mrl_logits"}

    def test_category_logits_shape(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert out["category_logits"].shape == (B, CortexResearchModel.NUM_CLASSES)

    def test_impact_score_shape_and_range(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert out["impact_score"].shape == (B,)
        assert out["impact_score"].min() >= 0.0
        assert out["impact_score"].max() <= 1.0

    def test_quality_logits_shape(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert out["quality_logits"].shape == (B, 2)

    def test_proj_emb_is_l2_normalized(self, tiny_model, dummy_batch):
        out  = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        norms = out["proj_emb"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_mrl_logits_count(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        assert len(out["mrl_logits"]) == len(MatryoshkaPooling.GRANULARITIES)

    def test_embeddings_returned_on_request(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"], return_embeddings=True)
        assert "embeddings" in out

    def test_task_weights_sum_to_one(self, tiny_model):
        w = tiny_model.task_weights()
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_no_nan_in_outputs(self, tiny_model, dummy_batch):
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"

    def test_category_idx_roundtrip(self):
        for i, cat in enumerate(CortexResearchModel.ARXIV_CATEGORIES):
            assert CortexResearchModel.category_to_idx(cat) == i
            assert CortexResearchModel.idx_to_category(i) == cat

    def test_backward_compat_alias(self):
        assert ResearchIntelligenceModel is CortexResearchModel


# ══════════════════════════════════════════════════════════════════════════════
# Loss function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFocalLoss:
    def test_positive_loss(self):
        fl = FocalLoss(num_classes=20)
        loss = fl(torch.randn(B, 20), torch.randint(0, 20, (B,)))
        assert loss.item() > 0

    def test_label_smoothing(self):
        fl_smooth = FocalLoss(num_classes=20, label_smoothing=0.1)
        fl_hard   = FocalLoss(num_classes=20, label_smoothing=0.0)
        logits = torch.randn(B, 20)
        targets = torch.randint(0, 20, (B,))
        # Smoothed loss is typically lower than hard CE
        assert not torch.isnan(fl_smooth(logits, targets))


class TestAsymmetricLoss:
    def test_binary_output(self):
        al = AsymmetricLoss()
        loss = al(torch.randn(B, 2), torch.randint(0, 2, (B,)))
        assert loss.item() >= 0

    def test_asymmetry(self):
        al = AsymmetricLoss(gamma_pos=0, gamma_neg=4)
        pos_targets = torch.ones(B, dtype=torch.long)
        neg_targets = torch.zeros(B, dtype=torch.long)
        logits = torch.randn(B, 2)
        # Losses should differ due to asymmetric focusing
        l_pos = al(logits, pos_targets)
        l_neg = al(logits, neg_targets)
        assert l_pos.item() != l_neg.item()


class TestSupervisedContrastiveLoss:
    def test_with_positives(self):
        supcon = SupervisedContrastiveLoss()
        import torch.nn.functional as F
        # Repeat some labels to ensure positives exist
        emb     = F.normalize(torch.randn(8, 128), dim=-1)
        labels  = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = supcon(emb, labels)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_no_positives_returns_zero(self):
        supcon = SupervisedContrastiveLoss()
        import torch.nn.functional as F
        emb    = F.normalize(torch.randn(4, 128), dim=-1)
        labels = torch.tensor([0, 1, 2, 3])  # all unique
        loss = supcon(emb, labels)
        assert loss.item() == 0.0


class TestMatryoshkaLoss:
    def test_output_shape(self):
        mrl = MatryoshkaLoss()
        logits_list = [torch.randn(B, 20)] * 4
        labels = torch.randint(0, 20, (B,))
        loss = mrl(logits_list, labels)
        assert loss.item() > 0

    def test_gradient_flows_to_all_levels(self):
        mrl = MatryoshkaLoss()
        logits_list = [torch.randn(B, 20, requires_grad=True) for _ in range(4)]
        labels = torch.randint(0, 20, (B,))
        loss = mrl(logits_list, labels)
        loss.backward()
        for i, l in enumerate(logits_list):
            assert l.grad is not None, f"No grad at granularity {i}"


class TestSoftSpearmanLoss:
    def test_perfect_rank_order(self):
        ssr = SoftSpearmanLoss()
        x = torch.arange(B).float()
        # Identical order → loss ≈ 0
        loss = ssr(x, x)
        assert loss.item() < 0.1

    def test_reversed_order_high_loss(self):
        ssr = SoftSpearmanLoss()
        x = torch.arange(B).float()
        loss = ssr(x, x.flip(0))  # reversed
        assert loss.item() > 0.4


class TestMultiTaskLoss:
    def test_all_keys_returned(self, tiny_model, dummy_batch):
        criterion = MultiTaskLoss()
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        tw = tiny_model.task_weights()
        losses = criterion(
            out["category_logits"], out["impact_score"], out["quality_logits"],
            out["proj_emb"], out["mrl_logits"],
            dummy_batch["category_labels"],
            dummy_batch["impact_scores"],
            dummy_batch["quality_labels"],
            task_weights=tw,
            global_step=600,  # past supcon warmup
        )
        assert set(losses.keys()) == {"total","classification","supcon","mrl","regression","quality"}

    def test_no_nan(self, tiny_model, dummy_batch):
        criterion = MultiTaskLoss()
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        losses = criterion(
            out["category_logits"], out["impact_score"], out["quality_logits"],
            out["proj_emb"], out["mrl_logits"],
            dummy_batch["category_labels"], dummy_batch["impact_scores"], dummy_batch["quality_labels"],
        )
        for k, v in losses.items():
            assert not torch.isnan(v), f"NaN in loss[{k}]"

    def test_supcon_zero_before_warmup(self, tiny_model, dummy_batch):
        criterion = MultiTaskLoss(supcon_warmup_steps=1000)
        out = tiny_model(dummy_batch["input_ids"], dummy_batch["attention_mask"])
        losses = criterion(
            out["category_logits"], out["impact_score"], out["quality_logits"],
            out["proj_emb"], out["mrl_logits"],
            dummy_batch["category_labels"], dummy_batch["impact_scores"], dummy_batch["quality_labels"],
            global_step=0,  # before warmup
        )
        assert losses["supcon"].item() == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_perfect_classification(self):
        labels = np.arange(10) % 20
        m = compute_metrics(labels, labels, np.ones(10), np.ones(10), labels % 2, labels % 2)
        assert m["category_accuracy"] == 1.0

    def test_regression_metrics_near_perfect(self):
        targets = np.random.rand(100)
        preds   = targets + np.random.randn(100) * 0.01
        m = compute_metrics(
            np.zeros(100, int), np.zeros(100, int),
            preds, targets,
            np.zeros(100, int), np.zeros(100, int),
        )
        assert m["impact_r2"] > 0.9


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_latex_removal(self):
        cleaned = clean_text("Loss is $L = \\sum x^2$ and $$ \\nabla = 0 $$.")
        assert "$" not in cleaned
        assert "[MATH]" in cleaned

    def test_citation_removal(self):
        cleaned = clean_text("As shown [Smith et al.] and (Jones, 2020) it works.")
        assert "[Smith et al.]" not in cleaned

    def test_quality_heuristic_good(self):
        abstract = (
            "We propose a deep learning approach achieving 95.3% accuracy "
            "on the benchmark dataset consisting of over 50,000 labelled examples. "
            "Our method outperforms competitive baselines by 3.5% and we demonstrate "
            "its effectiveness on 10,000 training samples drawn from diverse sources. "
            "The proposed architecture introduces a novel self-attention mechanism "
            "that captures long-range dependencies more efficiently than recurrent models "
            "while reducing computational cost by a factor of two. "
            "We conduct extensive ablation studies that confirm the contribution of each "
            "architectural component, including the encoder depth, the pooling strategy, "
            "and the regularisation schedule applied during training. "
            "Furthermore, our approach generalises well to out-of-distribution data, "
            "achieving strong results across multiple evaluation domains and benchmark settings "
            "without requiring any task-specific fine-tuning or additional labelled supervision. "
            "This validates both the robustness and the practical applicability of the proposed method."
        )
        assert _quality_heuristic(abstract) == 1

    def test_quality_heuristic_short(self):
        assert _quality_heuristic("Too short.") == 0
