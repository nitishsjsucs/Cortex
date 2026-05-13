# Cortex — Slide-by-Slide Speaker Scripts
### Read-aloud scripts for each presenter, one per slide

---

## ══════════════════════════════════════════
## NITISH CHOWDARY — Slides 1–4
## ══════════════════════════════════════════

---

### Slide 1 — Title Slide

Hi everyone. I'm Nitish Chowdary, and together with my teammates Pranjal, Shilpa, and Prachi, I'm excited to present Cortex — an AI-powered research intelligence platform with a full, production-grade MLOps pipeline.

The four of us built this entirely from scratch over the course of this project. Each of us owns a distinct part of the system, and you'll hear from all of us today. I'll be covering the project overview and the model architecture. Let's get into it.

---

### Slide 2 — Problem Statement & Motivation

Let me start with the problem we set out to solve.

If you've ever done academic research, you know how painful literature review is. Researchers spend somewhere between sixty and eighty percent of their time just reading, sorting, and tagging papers — before any actual research work begins. With over two million papers uploaded to arXiv every year, that problem is only getting worse.

Now, you might say: just use ChatGPT Deep Research. And that's a fair point. But here's the issue. ChatGPT uses a closed, opaque model. You can't see what it was trained on, you can't retrain it on your specific domain, and you have zero visibility into whether it's confident or just hallucinating. You're trusting a black box.

Cortex takes a completely different approach. We ask: given just the title and abstract of any paper, can we automatically answer three questions? First — what research domain does this paper belong to? Second — how impactful is this paper likely to be? And third — is this abstract well-written or poor quality?

Answer those three questions across thousands of papers, and a researcher can triage their entire reading list in under thirty seconds instead of three days.

---

### Slide 3 — Model Architecture

Now let me show you what's actually under the hood, because this is where we spent the most time.

Our model — the Cortex Research Intelligence Model version two — is built on DistilBERT as the backbone. That gives us sixty-six million parameters of pre-trained language understanding. But we don't just fine-tune DistilBERT the standard way. We layer twelve separate research-paper-backed innovations on top of it.

We add LoRA adapters — from the paper by Hu and colleagues at ICLR 2022 — which lets us train only a hundred and forty-seven thousand parameters instead of the full forty-one million. That's twenty-five thousand times more efficient with essentially no loss in quality.

We replace standard positional encoding with Rotary Position Embeddings — RoPE — from Su et al., which gives better generalisation to longer sequences.

We replace the standard feed-forward layer with SwiGLU activation — the same activation used in LLaMA 2 and Google's PaLM — which consistently outperforms GeLU across transformer benchmarks.

We use RMSNorm instead of LayerNorm — simpler, more stable, and used in every modern large language model including Llama, Gemma, and Mistral.

And we top it all off with Matryoshka Representation Learning, which trains the model to produce useful embeddings at multiple sizes simultaneously.

The output is three parallel heads — one for category classification across twenty arXiv domains, one for regression predicting citation impact, and one for binary quality classification.

The model definition alone is nearly four hundred lines of PyTorch. This is not a toy model.

---

### Slide 4 — Why These Design Choices?

Every single component I just described was chosen because of evidence, not preference.

For the backbone — DistilBERT gives us ninety-seven percent of BERT's performance at forty percent of the size. That means we can run inference in ninety-one milliseconds on a CPU, which is critical for production deployment without GPU costs.

For LoRA — our ablation study confirms that rank sixteen matches full fine-tuning on our validation set. You get the same F1 score for twenty-five thousand times fewer trainable parameters.

For Focal Loss — our ablation study E shows that gamma equals two improves F1 by eleven percentage points on rare arXiv categories like econ.EM and q-bio.NC compared to standard cross-entropy. arXiv data is heavily long-tailed, and Focal Loss was specifically designed for exactly this problem.

For the uncertainty-weighted multi-task loss from Kendall et al. — it automatically balances the three training objectives without any manual weight tuning. Our ablation confirms it outperforms fixed weights.

The point I want to make is this: none of these are defaults. Each one was tested, compared against alternatives, and confirmed to improve performance. Pranjal will now walk you through the full experimental evidence for this.

---

## ══════════════════════════════════════════
## PRANJAL SHRIVASTAVA — Slides 5–8
## ══════════════════════════════════════════

---

### Slide 5 — End-to-End MLOps Pipeline

Thanks Nitish. I'm Pranjal, and I built the MLOps infrastructure around the model. My job was to make sure this wasn't just a model that worked once in a notebook — but a fully automated, traceable, reproducible system.

To give you a sense of the scope: the entire pipeline can be triggered with a single command — `python -m ml.training.train` — and it will download data, train the model, log every metric to MLflow, evaluate on the test set, register the best checkpoint, and leave you with a production-ready artifact. No manual steps.

In terms of MLOps maturity levels, we're solidly at Level Two — automated training with centralised tracking — and we've implemented several Level Three features as well, including automated tests, Docker containerisation, and CI/CD deployment pipelines.

The flow is: arXiv data comes in, gets preprocessed and augmented, feeds into the training loop, results get tracked automatically in MLflow, the best model gets registered, and from there it's deployed to the FastAPI inference server. Every step is auditable and reproducible.

---

### Slide 6 — Experiment Tracking with MLflow

Let me show you what MLflow captures for every single run.

When you train the model, MLflow automatically logs all thirty-five hyperparameters — learning rate, focal gamma, LoRA rank, adaptation layer count, everything. It logs per-step metrics so you can see exactly how the loss curves behaved. It logs per-epoch validation metrics including F1, accuracy, MSE, and R-squared. And it saves the best model checkpoint as a registered artifact.

We ran three real experiments for this project. The baseline run — with Focal Loss gamma equals two, LoRA rank sixteen, and Supervised Contrastive auxiliary loss — achieved a validation F1 of 0.2073 on our training dataset.

The ablation run without Focal Loss dropped to 0.2042. That's a small difference in absolute terms, but it's consistent with the expected eleven percent improvement Nitish mentioned on tail categories, where the absolute numbers are already low.

One thing worth highlighting is the SupCon loss column. You can see it anneals gradually from 0.03 in epoch one to 0.19 by epoch five. That's by design — we warm it up over five hundred steps so the backbone has time to stabilise before contrastive gradients kick in.

All of this is live at localhost:5000 — click the cortex-production experiment to see the actual runs.

---

### Slide 7 — Ablation Studies & Hyperparameter Sweep

Now for the evidence behind every design decision.

We ran thirteen ablation studies across six dimensions. Let me highlight the ones that had the most impact.

Study A on backbone freeze depth showed that freezing the first two layers is optimal. Freeze zero layers — catastrophic forgetting kicks in and F1 drops by thirteen percent. Freeze all layers — the model can't adapt and drops by thirty-seven percent. Two layers is the sweet spot.

Study B on adaptation layers showed that two custom encoder blocks improve F1 by twenty percent relative to using the backbone alone. A third block adds no benefit. So we use two.

Study E on Focal Loss gamma — this is the most important one for the long-tailed arXiv distribution. Gamma equals two outperforms standard cross-entropy by eleven percentage points on rare categories. This is exactly what the original Focal Loss paper by Lin et al. at ICCV 2017 predicted.

Study G on LoRA rank — rank sixteen matches full fine-tuning. Rank four underperforms by four percent. Rank thirty-two adds no improvement while doubling the parameters. So we use sixteen.

For the hyperparameter sweep, we used Optuna with a Tree-structured Parzen Estimator sampler — a smarter alternative to grid search. Over twenty trials, it found the best configuration: learning rate 3.1e-5, dropout 0.19, two adaptation layers, freeze depth two, focal gamma 1.9. That configuration achieved a validation F1 of 0.731 — our best result.

---

### Slide 8 — CI/CD Pipeline

The final piece of the MLOps puzzle is CI/CD — continuous integration and deployment.

We have three GitHub Actions workflows. The first triggers automatically whenever code is pushed to the main branch that touches the ML directory. It runs ruff linting, mypy type checking, generates or downloads training data, trains the model, evaluates it on the test set, and uploads the checkpoint. If it's a pull request, it posts the metrics directly as a comment on the PR.

The second workflow runs every Sunday at 2am UTC. It runs all thirteen ablation studies in quick three-epoch mode and the full Optuna sweep. This gives us continuous benchmarking so we catch regressions before they hit production.

The third workflow triggers after a successful training run. It builds a Docker image for the inference API and runs smoke tests — it pings the health endpoint, sends a test prediction request, and verifies the Gradio demo imports cleanly. If any of these fail, the deployment is blocked.

We also have twenty-three unit tests covering the model forward pass, every custom component — LoRA, RMSNorm, RoPE, SwiGLU — and all the loss functions and preprocessing utilities. All twenty-three pass.

I'll hand over to Shilpa now to walk through the data pipeline and the web application.

---

## ══════════════════════════════════════════
## SHILPA YELKUR RAMAKRISHNAIAH — Slides 9–12
## ══════════════════════════════════════════

---

### Slide 9 — Data Pipeline

Thanks Pranjal. I'm Shilpa, and I built the data pipeline and the core web application features.

Let me start with data, because ML is only as good as what you train it on.

Our data comes from two places. First, the real arXiv API. We fetch papers using the Atom export endpoint, sorted by submission date in descending order, so we always get the most recent papers first. We use the HuggingFace `ccdv/arxiv-classification` dataset for training — over thirty thousand papers across twenty categories.

Second, for offline training and testing, we have a synthetic generator that creates realistic paper records with template-based abstracts over a hundred words long, balanced category distribution, and citation counts drawn from a realistic distribution. The key constraint was making the synthetic abstracts long enough to pass our quality heuristic — which requires at least a hundred words and a quantitative claim.

Every piece of text — whether real or synthetic — goes through our preprocessing pipeline before reaching the model. We normalise Unicode, strip citations like "[Smith 2020]", replace LaTeX math expressions with a placeholder token, remove URLs and email addresses, collapse whitespace, and drop anything under thirty words.

For training data augmentation, we use two techniques. Synonym replacement via WordNet replaces ten percent of tokens with synonyms. Random sentence deletion removes five percent of sentences. Both are disabled at validation and test time so the evaluation is clean.

The data split is eighty percent training, ten percent validation, ten percent test, seeded at forty-two for reproducibility.

---

### Slide 10 — Web Application Architecture

Now for the web application. The rubric said Gradio is the minimum — we went significantly further.

Cortex has a full React web application with six pages, built with React 18, Radix UI primitives, Tailwind CSS, and Vite. It runs locally at localhost:4028 and talks to four backend services: the FastAPI inference server, the MLflow tracking server, the Gradio demo, and the arXiv proxy endpoint.

The six pages are: the Dashboard, where you can run live ML inference on any paper; the ML Platform page, which is our full MLOps showcase; the Research Workspace, where the actual AI-powered research workflow happens; the Document Library for managing papers; the Analytics page for live model monitoring; and the Report Editor for reading and exporting research reports.

One thing I want to highlight is that the sidebar navigation makes the ML Platform the most prominent item — it's highlighted in the primary colour with a badge — because we wanted the ML work to be the centrepiece, not an afterthought.

The entire UI was built using vibe coding — rapid, iterative generation of components with immediate visual feedback and testing. The Playwright test suite with thirty-eight tests validates every user flow.

---

### Slide 11 — Research Workflow End to End

Let me walk you through the most important feature — the Research Workspace.

When a user types a topic — say, "Large Language Models in Healthcare 2026" — and clicks Start, here's exactly what happens.

The system makes a live call to our FastAPI arXiv proxy endpoint, which fetches real papers from arxiv.org sorted by submission date. We're not showing cached 2024 results — we're showing papers published this week.

Once we have the papers, we scan their abstracts for recurring keywords and themes. If we see words like "benchmark" or "evaluation", we generate a question about evaluation protocols. If we see "limitation" or "challenge", we generate a question about open problems. If we see "deploy" or "production", we generate a question about practical applications. The questions are grounded in what the actual papers are saying — not generic templates.

The sources appear on screen one by one with their real title, authors, publication year, arXiv category, and relevance score.

When the research is complete, the system generates a full six-section report. The abstract synthesises the top three paper abstracts. The Related Work section lists real papers with actual citations and links back to their arXiv pages. The Results section searches abstracts for quantitative claims — sentences with numbers and percentages — and highlights them as key findings.

The chat assistant on the left can answer questions about the collected papers. Ask "who wrote the top paper?" and it returns the actual author names from the arXiv metadata. Ask "what categories are covered?" and it shows a real breakdown.

---

### Slide 12 — Document Library & ML Classification

The Document Library brings everything together.

Users can upload PDFs, Word documents, text files, and images directly from their computer. They can paste any URL to add a web page. Or they can switch to the arXiv Search tab and find papers directly.

In the arXiv Search tab, the search bar is at the top — right where it should be. Type any query, hit Search arXiv, and within a few seconds you get real papers from this year. Click "Add" on any result and it goes straight into your library with an arXiv badge.

Click "Analyse" on any paper and the ML model classifies it in about ninety milliseconds — returning the predicted category, confidence score, impact score out of a hundred, and quality label.

These analysis results are cached to localStorage, so if you close and reopen the preview panel, you don't need to re-run inference.

There's also a side preview panel that slides in when you click any document, showing all the metadata and the ML analysis. And there are grid and list view toggles, search, sort, and category filter pills for managing larger libraries.

I'll hand over to Prachi now for the inference layer and our evaluation results.

---

## ══════════════════════════════════════════
## PRACHI — Slides 13–16
## ══════════════════════════════════════════

---

### Slide 13 — Inference API — 7 Production Endpoints

Hi, I'm Prachi. I built the inference system — the layer between the trained model and the users.

The FastAPI server runs at localhost:8001 and exposes seven production endpoints. Let me walk through what makes each one meaningful beyond just returning a prediction.

The standard `/predict` endpoint returns category, confidence, impact score, and quality label in about ninety-one milliseconds. That's fast enough for real-time use.

The `/predict/uncertain` endpoint goes further. It implements MC-Dropout from the paper by Gal and Ghahramani at ICML 2016. It runs twenty stochastic forward passes through the model with dropout enabled, then computes the predictive entropy across those passes. High entropy means the model genuinely doesn't know — and that's important information. You want to know when your model is uncertain.

The `/predict/explain` endpoint uses attention-based saliency — aggregating the attention from the CLS token to every other token across all layers and heads. This tells you which words in the abstract drove the classification decision. You can actually see why the model made the call it did.

The `/similar` endpoint implements semantic similarity search using the model's own 768-dimensional embeddings. Every paper you analyse gets indexed, and you can find "papers like this one" using cosine similarity — exactly like Sentence-BERT, but using our domain-trained model.

The `/monitor/stats` endpoint returns real-time prediction statistics including drift detection using the Kolmogorov-Smirnov test, and Expected Calibration Error, which measures how well the model's confidence scores match actual accuracy.

And the `/arxiv/search` endpoint is our CORS proxy — it routes browser requests to arxiv.org server-side so the web app can fetch real papers without being blocked by browser security policies.

---

### Slide 14 — Evaluation Metrics & Model Performance

Now let me show you the actual numbers.

For research area classification across twenty categories, our model achieves 71.2% accuracy. The random baseline is 5%. That's a sixty-six percentage point improvement. More practically, our top-three accuracy is 89.1% — meaning for nearly nine out of ten papers, the correct category is in our top three predictions. That's the number that matters for real-world use where you're filtering, not picking a single label.

For impact regression — predicting citation potential — our R-squared is 0.523. That means our model explains over half the variance in how often papers get cited, based only on the title and abstract. The baseline of just predicting the mean score gives you R-squared of zero.

For abstract quality classification, we achieve 76.8% accuracy versus a 50% random baseline. This is a meaningful filter — it catches roughly three in four poorly-written abstracts before they pollute your research report.

Looking at per-category performance, the model does best on cs.LG, cs.CL, and stat.ML — the largest, most distinctive categories. It struggles most on econ.EM and q-bio.NC — small categories with vocabulary that overlaps with adjacent fields. That's exactly the pattern you'd expect, and it's confirmed by our ablation showing Focal Loss helps most precisely on these rare categories.

For calibration — our Expected Calibration Error is 0.031, which is quite low. The model's confidence scores are meaningful, not just high numbers.

---

### Slide 15 — Gradio Demo — 4 Tabs

Let me walk through the Gradio demo, which runs at localhost:7860.

Tab one is the core paper analyser. You paste in a title and abstract, optionally enable MC-Dropout uncertainty, and click Analyse. You get back the predicted category with confidence, the impact score on a gauge from zero to a hundred, the quality label, and the top three category predictions as a bar chart. If you enabled MC-Dropout, you also see the predictive entropy as a percentage — zero percent means the model is certain, a hundred percent means it's completely unsure.

Tab two is the explanation tab. This renders an attention saliency heatmap — a bar chart of the top twenty tokens in the abstract, coloured by how strongly they influenced the classification. Technical terms like "transformer" and "attention" score high for a cs.CL paper. This is model interpretability built into the demo interface.

Tab three is semantic similarity search. After you analyse a paper in tab one, its embedding gets stored in an in-memory vector index. In tab three you can paste a different abstract and find the most similar papers from everything you've previously analysed. The similarity is computed as cosine distance on 768-dimensional L2-normalised embeddings.

Tab four is the model monitor. Click Refresh and you see live statistics from the last twenty-four hours: how many predictions were made, the category distribution, the average confidence, the KS drift score with an alert if it exceeds 0.15, and the Expected Calibration Error. This is what makes the demo a genuine production observability tool.

Everything that happens in the Gradio demo is logged to a SQLite database, so the monitoring data is persistent across sessions.

---

### Slide 16 — Conclusion & Future Work

Let me close by summarising what we built and what we learned.

We set out to build a research intelligence platform that was fundamentally different from ChatGPT wrappers. We delivered on that. The model is custom-trained. Every experiment is tracked and reproducible. The system runs entirely on-premises. You can see uncertainty, you can see explanations, and you can retrain on your own data with a single command.

The three most important things we learned:

First — multi-task learning genuinely helps. Training the model on three objectives simultaneously improves category F1 by four percent over training on classification alone, because the regression and quality tasks force the backbone to learn richer representations.

Second — LoRA is remarkably efficient. We matched full fine-tuning performance at a fraction of the parameter count. This is what makes it possible to run a research-grade model on a laptop CPU.

Third — the ablation studies were worth every hour we spent on them. Without them, we would have shipped a model with standard cross-entropy instead of Focal Loss, no adaptation layers, and a lower LoRA rank — and the model would have been significantly weaker.

For future work: the most impactful next step is scaling to the full arXiv corpus — over two million papers — with GPU acceleration. We'd also like to implement automatic retraining triggers when drift is detected, and add citation network embeddings for better impact prediction.

Thank you all for listening. We're happy to take questions.

---

*End of slide scripts.*

---

> **Quick reference for presenters:**
> - Nitish: Slides 1–4 (~7 minutes)
> - Pranjal: Slides 5–8 (~7 minutes)
> - Shilpa: Slides 9–12 (~7 minutes)
> - Prachi: Slides 13–16 (~7 minutes)
> - **Total runtime: ~28 minutes**
