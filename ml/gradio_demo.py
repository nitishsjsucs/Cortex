"""
Cortex Research Intelligence — Gradio Demo v2

Features
--------
  1. Standard multi-task prediction (category, impact, quality)
  2. MC-Dropout uncertainty estimation (flag ambiguous papers)
  3. Gradient saliency — which words drove the classification
  4. Semantic similarity search over previously analysed papers
  5. Live monitoring stats (prediction count, drift alert)

Run:
    python ml/gradio_demo.py
    python ml/gradio_demo.py --checkpoint ml/outputs/models/best_model.pt --port 7860
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Load predictor ────────────────────────────────────────────────────────────

_predictor = None


def _get_predictor(checkpoint: str = None):
    global _predictor
    if _predictor is None:
        from ml.inference.predictor import ResearchPredictor
        ckpt = checkpoint or os.getenv("MODEL_CHECKPOINT", "ml/outputs/models/best_model.pt")
        if Path(ckpt).exists():
            _predictor = ResearchPredictor.from_checkpoint(ckpt)
        else:
            _predictor = ResearchPredictor.from_pretrained_only()
    return _predictor


# ── Tab 1: Analyse Paper ──────────────────────────────────────────────────────

def analyse_paper(title: str, abstract: str, mc_uncertainty: bool) -> Tuple:
    if not title.strip() or not abstract.strip():
        return ("⚠️ Enter both title and abstract.", None, None, "", "", "")

    pred    = _get_predictor()
    text    = f"{title} [SEP] {abstract}"
    result  = pred.predict(
        text,
        return_embedding=True,
        mc_samples=20 if mc_uncertainty else 0,
        conformal=True,
    )

    # Register in similarity index
    try:
        from ml.inference.similarity import get_index
        import hashlib
        idx = get_index()
        idx.add(
            hashlib.md5(text.encode()).hexdigest()[:12],
            title, abstract, result.embedding or [],
            result.predicted_category, result.impact_score,
        )
    except Exception:
        pass

    # Log prediction
    try:
        from ml.inference.monitoring import log_prediction
        log_prediction(title, abstract, result.predicted_category,
                       result.category_confidence, result.impact_score,
                       result.quality_label, 0.0, result.uncertainty)
    except Exception:
        pass

    # ── Summary text ──────────────────────────────────────────────────────
    conf_pct = result.category_confidence * 100
    imp_pct  = result.impact_score * 100
    qual_col = "🟢" if result.quality_label == "well-structured" else "🔴"

    summary = f"""### 📄 {result.predicted_category}  · {conf_pct:.0f}% confidence

**Impact Score:** {imp_pct:.0f} / 100 &nbsp;&nbsp; {qual_col} **Quality:** {result.quality_label.title()} ({result.quality_confidence*100:.0f}%)"""

    if result.uncertainty is not None:
        unc_pct = result.uncertainty * 100
        flag = "⚠️ High uncertainty" if result.uncertainty > 0.5 else "✅ Confident"
        summary += f"\n\n**MC-Dropout Uncertainty:** {unc_pct:.0f}%  —  {flag}"

    if result.conformal_set and len(result.conformal_set) > 1:
        summary += f"\n\n**Conformal Prediction Set (90% coverage):** {', '.join(result.conformal_set)}"

    # ── Top-3 bar chart ───────────────────────────────────────────────────
    top3   = result.top3_categories
    cats   = [c["category"].split(".")[-1] for c in top3]
    confs  = [c["confidence"] * 100 for c in top3]

    fig1, ax = plt.subplots(figsize=(5, 2.4))
    fig1.patch.set_facecolor("#0f0f1c")
    ax.set_facecolor("#0f0f1c")
    colors = ["#6366f1", "#818cf8", "#a5b4fc"]
    bars   = ax.barh(cats[::-1], confs[::-1], color=colors[::-1], height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1c1c2e")
    for bar, val in zip(bars, confs[::-1]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.0f}%",
                va="center", color="#f1f5f9", fontsize=8)
    plt.tight_layout(pad=0.5)

    # ── Impact gauge ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4, 1.8))
    fig2.patch.set_facecolor("#0f0f1c")
    ax2.set_facecolor("#0f0f1c")
    col = "#10b981" if imp_pct >= 60 else "#f59e0b" if imp_pct >= 30 else "#f43f5e"
    ax2.barh(["Impact"], [imp_pct], color=col, height=0.4)
    ax2.set_xlim(0, 100)
    ax2.axvline(50, color="#3f3f46", ls="--", lw=0.8)
    ax2.set_xlabel("Impact Score (0–100)", color="#94a3b8", fontsize=8)
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1c1c2e")
    ax2.text(imp_pct + 1, 0, f"{imp_pct:.0f}", va="center", color="#f1f5f9", fontsize=9)
    plt.tight_layout(pad=0.5)

    return (
        summary,
        fig1,
        fig2,
        result.predicted_category,
        result.quality_label,
    )


# ── Tab 2: Explain (Saliency) ─────────────────────────────────────────────────

def explain_paper(title: str, abstract: str) -> Tuple:
    if not title.strip() or not abstract.strip():
        return "⚠️ Enter both title and abstract.", None

    pred  = _get_predictor()
    text  = f"{title} [SEP] {abstract}"
    pairs = pred.explain(text)

    if not pairs:
        return "Could not compute saliency.", None

    tokens = [p[0] for p in pairs[:15]]
    scores = [p[1] for p in pairs[:15]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0f0f1c")
    ax.set_facecolor("#0f0f1c")

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(s)) for s in scores]

    bars = ax.barh(tokens[::-1], scores[::-1], color=colors[::-1], height=0.65)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Saliency Score", color="#94a3b8", fontsize=9)
    ax.set_title("Token Importance (gradient-based)", color="#f1f5f9", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1c1c2e")
    plt.tight_layout(pad=0.8)

    explanation = "**Top influential tokens:**\n\n"
    for tok, score in pairs[:10]:
        bar = "█" * int(score * 20)
        explanation += f"`{tok}` — {score:.3f}  {bar}\n\n"

    return explanation, fig


# ── Tab 3: Similar Papers ─────────────────────────────────────────────────────

def find_similar(title: str, abstract: str) -> str:
    if not title.strip() or not abstract.strip():
        return "⚠️ Enter both title and abstract."

    try:
        from ml.inference.similarity import get_index
        pred   = _get_predictor()
        text   = f"{title} [SEP] {abstract}"
        result = pred.predict(text, return_embedding=True)

        if not result.embedding:
            return "Could not extract embeddings."

        idx     = get_index()
        similar = idx.search(result.embedding, top_k=5)

        if not similar:
            return (
                "**No similar papers in index yet.**\n\n"
                "Analyse some papers first (Tab 1) to populate the index.\n"
                "Each analysed paper is automatically added to the similarity index."
            )

        out = f"### Papers similar to: *{title[:60]}*\n\n"
        out += f"Query category: **{result.predicted_category}**\n\n"
        out += "---\n\n"
        for i, r in enumerate(similar, 1):
            pct = r['score'] * 100
            out += f"**{i}. {r['title'][:80]}**\n\n"
            out += f"Category: `{r['category']}` &nbsp;·&nbsp; "
            out += f"Similarity: **{pct:.0f}%** &nbsp;·&nbsp; "
            out += f"Impact: {r['impact']*100:.0f}/100\n\n"
            out += f"_{r['abstract'][:200]}…_\n\n---\n\n"
        return out

    except Exception as exc:
        return f"Search failed: {exc}"


# ── Tab 4: Model Monitor ──────────────────────────────────────────────────────

def get_monitor_stats() -> Tuple:
    try:
        from ml.inference.monitoring import get_rolling_stats, get_hourly_volume
        stats   = get_rolling_stats(24.0)
        hourly  = get_hourly_volume(24)

        if stats.get("total", 0) == 0:
            return ("**No predictions recorded yet.** Analyse some papers first.", None)

        # Stats markdown
        md  = f"### 📊 Last 24h — {stats['total']} predictions\n\n"
        md += f"**Avg confidence:** {stats.get('mean_confidence',0)*100:.0f}%  "
        md += f"**Avg impact:** {stats.get('mean_impact',0)*100:.0f}/100  "
        md += f"**Avg latency:** {stats.get('mean_latency_ms',0):.0f}ms\n\n"

        drift = stats.get("drift_score", 0)
        alert = stats.get("drift_alert", False)
        md += f"**Data drift (KS stat):** {drift:.3f}  "
        md += "⚠️ **DRIFT DETECTED**\n\n" if alert else "✅ Within bounds\n\n"

        md += f"**ECE (calibration error):** {stats.get('ece',0):.4f}\n\n"

        cats = stats.get("category_counts", {})
        if cats:
            md += "**Category distribution:**\n\n"
            total = sum(cats.values())
            for cat, cnt in sorted(cats.items(), key=lambda x: -x[1])[:8]:
                pct  = cnt / total * 100
                bar  = "█" * int(pct / 5)
                md  += f"`{cat}` {cnt} ({pct:.0f}%)  {bar}\n\n"

        # Volume chart
        if hourly:
            hours  = [h["hours_ago"] for h in hourly]
            counts = [h["count"] for h in hourly]

            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor("#0f0f1c")
            ax.set_facecolor("#0f0f1c")
            ax.fill_between(hours, counts, alpha=0.3, color="#6366f1")
            ax.plot(hours, counts, color="#6366f1", lw=1.5)
            ax.set_xlabel("Hours ago", color="#94a3b8", fontsize=8)
            ax.set_ylabel("Predictions", color="#94a3b8", fontsize=8)
            ax.set_title("Prediction Volume (last 24h)", color="#f1f5f9", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            ax.invert_xaxis()
            for spine in ax.spines.values():
                spine.set_edgecolor("#1c1c2e")
            plt.tight_layout(pad=0.5)
        else:
            fig = None

        return md, fig

    except Exception as exc:
        return f"Monitor error: {exc}", None


# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "Attention Is All You Need",
        "We propose a novel simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task with 12 attention heads.",
    ],
    [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context. The pre-trained BERT model can be fine-tuned to create state-of-the-art models for tasks including question answering, achieving F1 of 93.5 on SQuAD 2.0.",
    ],
    [
        "Deep Residual Learning for Image Recognition",
        "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, gaining accuracy from increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers — 8× deeper than VGG nets achieving 3.57% top-5 error.",
    ],
]

# ── Gradio Interface ──────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#09090b",
    block_background_fill="#0f0f1c",
    block_border_color="#1c1c2e",
    input_background_fill="#0f0f1c",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_hover="#4f46e5",
)

with gr.Blocks(theme=THEME, title="Cortex — Research Intelligence") as demo:
    gr.Markdown(
        """
        # 🔬 Cortex Research Intelligence
        **Multi-task paper analysis: classification · impact · quality · uncertainty · saliency · semantic search**
        """
    )

    with gr.Tabs():

        # ── Tab 1: Analyse ────────────────────────────────────────────────
        with gr.TabItem("📄 Analyse Paper"):
            with gr.Row():
                with gr.Column(scale=2):
                    t1 = gr.Textbox(label="Paper Title", placeholder="e.g. Attention Is All You Need")
                    a1 = gr.Textbox(label="Abstract", lines=6, placeholder="Paste the abstract here…")
                    mc = gr.Checkbox(label="Enable MC-Dropout uncertainty (slower, ~20 passes)", value=False)
                    btn1 = gr.Button("🔍 Analyse", variant="primary")

                with gr.Column(scale=3):
                    summary = gr.Markdown()
                    with gr.Row():
                        cat_chart = gr.Plot(label="Top-3 Predictions")
                        imp_chart = gr.Plot(label="Impact Score")
                    with gr.Row():
                        cat_label  = gr.Textbox(label="Category",       interactive=False)
                        qual_label = gr.Textbox(label="Quality",         interactive=False)

            gr.Examples(examples=EXAMPLES, inputs=[t1, a1], label="Example Papers")

            btn1.click(
                fn=analyse_paper,
                inputs=[t1, a1, mc],
                outputs=[summary, cat_chart, imp_chart, cat_label, qual_label],
            )

        # ── Tab 2: Explain ────────────────────────────────────────────────
        with gr.TabItem("🔦 Explain Prediction"):
            gr.Markdown("**Gradient saliency** — see which words in the abstract most influenced the category prediction.")
            with gr.Row():
                with gr.Column():
                    t2  = gr.Textbox(label="Title")
                    a2  = gr.Textbox(label="Abstract", lines=6)
                    btn2 = gr.Button("🔦 Explain", variant="primary")
                with gr.Column():
                    expl_text = gr.Markdown()
                    expl_plot = gr.Plot(label="Token Saliency Map")

            btn2.click(fn=explain_paper, inputs=[t2, a2], outputs=[expl_text, expl_plot])

        # ── Tab 3: Similar Papers ─────────────────────────────────────────
        with gr.TabItem("🔗 Find Similar Papers"):
            gr.Markdown(
                "**Semantic similarity search** using 768-D paper embeddings (SBERT-style cosine similarity).\n"
                "Analyse papers in Tab 1 first to populate the index."
            )
            with gr.Row():
                with gr.Column():
                    t3  = gr.Textbox(label="Title")
                    a3  = gr.Textbox(label="Abstract", lines=6)
                    btn3 = gr.Button("🔗 Find Similar", variant="primary")
                with gr.Column():
                    sim_output = gr.Markdown()

            btn3.click(fn=find_similar, inputs=[t3, a3], outputs=[sim_output])

        # ── Tab 4: Monitor ────────────────────────────────────────────────
        with gr.TabItem("📊 Model Monitor"):
            gr.Markdown("**Live prediction monitoring** — distribution, drift detection (KS test), calibration error.")
            btn4 = gr.Button("🔄 Refresh Stats", variant="secondary")
            with gr.Row():
                monitor_text = gr.Markdown()
                volume_chart = gr.Plot(label="Prediction Volume")
            btn4.click(fn=get_monitor_stats, outputs=[monitor_text, volume_chart])


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    args = parser.parse_args()

    _get_predictor(args.checkpoint)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
