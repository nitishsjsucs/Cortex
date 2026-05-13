"""
FastAPI Inference Server — Cortex Research Intelligence Model v2

Endpoints
---------
POST /predict              — MAP prediction (single paper)
POST /predict/batch        — Batch prediction (≤32 papers)
POST /predict/uncertain    — MC-Dropout uncertainty estimation [1]
POST /predict/explain      — Gradient saliency explanation [3]
POST /similar              — Semantic similarity search [2]
POST /index/add            — Add paper to embedding index
GET  /index/stats          — Embedding index statistics
GET  /monitor/stats        — Rolling prediction statistics + drift detection
GET  /monitor/hourly       — Per-hour prediction volume
GET  /health               — Liveness probe
GET  /model/info           — Model metadata

References
----------
[1] Gal & Ghahramani (2016). Dropout as Bayesian Approximation. ICML. arXiv:1506.02142
[2] Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019. arXiv:1908.10084
[3] Simonyan et al. (2013). Deep Inside Convolutional Networks. arXiv:1312.6034
"""

import os
import time
import logging
import urllib.parse
from contextlib import asynccontextmanager
from typing import List, Optional

import requests as _requests
from fastapi.responses import Response as _Response

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Schemas ───────────────────────────────────────────────────────────────────

class PaperInput(BaseModel):
    title:            str   = Field(..., min_length=1, max_length=500)
    abstract:         str   = Field(..., min_length=5, max_length=4000)
    return_embedding: bool  = False
    mc_samples:       int   = Field(0, ge=0, le=50)   # 0 = MAP only
    conformal:        bool  = False                    # include conformal set


class BatchInput(BaseModel):
    papers: List[PaperInput] = Field(..., min_items=1)

    @validator("papers")
    def check_size(cls, papers):
        max_sz = int(os.getenv("MAX_BATCH_SIZE", 32))
        if len(papers) > max_sz:
            raise ValueError(f"Batch too large (max {max_sz})")
        return papers


class CategoryScore(BaseModel):
    category:   str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_category:  str
    category_confidence: float
    top3_categories:     List[CategoryScore]
    impact_score:        float
    quality_label:       str
    quality_confidence:  float
    latency_ms:          float
    embedding:           Optional[List[float]] = None
    uncertainty:         Optional[float]       = None   # MC-Dropout entropy
    conformal_set:       Optional[List[str]]   = None   # guaranteed-coverage set


class BatchResponse(BaseModel):
    predictions:      List[PredictionResponse]
    total_latency_ms: float


class SimilarInput(BaseModel):
    title:    str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=5, max_length=4000)
    top_k:    int = Field(5, ge=1, le=20)


class IndexAddInput(BaseModel):
    doc_id:   str
    title:    str
    abstract: str
    # If embedding not provided, server computes it
    embedding: Optional[List[float]] = None


# ── App lifecycle ─────────────────────────────────────────────────────────────

_predictor = None
_index     = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _index
    from ml.inference.similarity import get_index
    _index = get_index()

    ckpt = os.getenv("MODEL_CHECKPOINT", "ml/outputs/models/best_model.pt")
    import pathlib
    try:
        from ml.inference.predictor import ResearchPredictor
        if pathlib.Path(ckpt).exists():
            _predictor = ResearchPredictor.from_checkpoint(ckpt)
            logger.info("Loaded model from %s", ckpt)
        else:
            _predictor = ResearchPredictor.from_pretrained_only()
            logger.warning("Checkpoint not found — using pre-trained weights only")
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        from ml.inference.predictor import ResearchPredictor
        _predictor = ResearchPredictor.from_pretrained_only()
    yield
    _predictor = None
    _index     = None


app = FastAPI(
    title="Cortex Research Intelligence API",
    description=(
        "Multi-task LLM-based paper intelligence: classification, impact prediction, "
        "quality assessment, MC-Dropout uncertainty, conformal prediction, "
        "semantic similarity search, and gradient saliency."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Helper ────────────────────────────────────────────────────────────────────

def _result_to_response(r, latency_ms: float) -> PredictionResponse:
    return PredictionResponse(
        predicted_category  = r.predicted_category,
        category_confidence = r.category_confidence,
        top3_categories     = [CategoryScore(**c) for c in r.top3_categories],
        impact_score        = r.impact_score,
        quality_label       = r.quality_label,
        quality_confidence  = r.quality_confidence,
        latency_ms          = latency_ms,
        embedding           = r.embedding,
        uncertainty         = r.uncertainty,
        conformal_set       = r.conformal_set,
    )


def _log(title: str, abstract: str, res, latency_ms: float):
    try:
        from ml.inference.monitoring import log_prediction
        log_prediction(
            title=title, abstract=abstract,
            predicted=res.predicted_category,
            confidence=res.category_confidence,
            impact=res.impact_score,
            quality=res.quality_label,
            latency_ms=latency_ms,
            uncertainty=res.uncertainty,
        )
    except Exception:
        pass


# ── Core prediction endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model/info")
def model_info():
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    from ml.models.research_classifier import CortexResearchModel
    m = _predictor.model
    return {
        "checkpoint":                    os.getenv("MODEL_CHECKPOINT", "untrained"),
        "num_parameters":                m.num_parameters,
        "num_trainable_parameters":      m.num_trainable_parameters,
        "lora_parameters":               m.lora_parameters,
        "backbone":                      m._backbone_name,
        "categories":                    CortexResearchModel.ARXIV_CATEGORIES,
        "conformal_qhat":                _predictor._qhat,
        "embedding_index_size":          _index.size() if _index else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PaperInput):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    t0   = time.perf_counter()
    text = f"{body.title} [SEP] {body.abstract}"
    res  = _predictor.predict(
        text,
        return_embedding=body.return_embedding,
        mc_samples=body.mc_samples,
        conformal=body.conformal,
    )
    ms  = (time.perf_counter() - t0) * 1000
    _log(body.title, body.abstract, res, ms)

    # Auto-index embedding if returned
    if res.embedding and _index is not None:
        import hashlib
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        _index.add(doc_id, body.title, body.abstract, res.embedding,
                   res.predicted_category, res.impact_score)
    return _result_to_response(res, ms)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchInput):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    t0    = time.perf_counter()
    texts = [f"{p.title} [SEP] {p.abstract}" for p in body.papers]
    results = _predictor.predict_batch(
        texts,
        return_embedding=any(p.return_embedding for p in body.papers),
        mc_samples=max(p.mc_samples for p in body.papers),
        conformal=any(p.conformal for p in body.papers),
    )
    total_ms = (time.perf_counter() - t0) * 1000
    per_ms   = total_ms / len(results)

    for p, r in zip(body.papers, results):
        _log(p.title, p.abstract, r, per_ms)

    return BatchResponse(
        predictions=[_result_to_response(r, per_ms) for r in results],
        total_latency_ms=total_ms,
    )


# ── MC-Dropout uncertainty endpoint ───────────────────────────────────────────
# Gal & Ghahramani (2016). arXiv:1506.02142

@app.post("/predict/uncertain", response_model=PredictionResponse)
def predict_uncertain(body: PaperInput):
    """
    Run MC-Dropout with N=20 stochastic forward passes.
    Returns mean prediction + predictive entropy as uncertainty score.
    Papers with uncertainty > 0.5 are flagged as ambiguous.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    t0   = time.perf_counter()
    text = f"{body.title} [SEP] {body.abstract}"
    res  = _predictor.predict(text, mc_samples=20, return_embedding=body.return_embedding)
    ms   = (time.perf_counter() - t0) * 1000
    _log(body.title, body.abstract, res, ms)
    return _result_to_response(res, ms)


# ── Gradient saliency endpoint ─────────────────────────────────────────────────
# Simonyan et al. (2013). arXiv:1312.6034

@app.post("/predict/explain")
def explain(body: PaperInput):
    """
    Compute gradient-based token saliency for the predicted category.
    Returns the top-20 most influential tokens and their importance scores.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    text  = f"{body.title} [SEP] {body.abstract}"
    t0    = time.perf_counter()
    pairs = _predictor.explain(text)
    ms    = (time.perf_counter() - t0) * 1000
    return {
        "saliency_tokens": [{"token": tok, "score": round(score, 4)} for tok, score in pairs],
        "latency_ms":      ms,
    }


# ── Semantic similarity search ─────────────────────────────────────────────────
# Reimers & Gurevych (2019). Sentence-BERT. arXiv:1908.10084

@app.post("/similar")
def find_similar(body: SimilarInput):
    """
    Find papers in the embedding index most similar to the query paper.
    Uses cosine similarity on 768-D L2-normalised embeddings.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    if _index is None or _index.size() == 0:
        return {"results": [], "message": "Index empty — analyse papers first via /predict"}

    text = f"{body.title} [SEP] {body.abstract}"
    res  = _predictor.predict(text, return_embedding=True)
    if res.embedding is None:
        raise HTTPException(500, "Embedding extraction failed")

    similar = _index.search(res.embedding, top_k=body.top_k)
    return {
        "query_category": res.predicted_category,
        "results": similar,
    }


@app.post("/index/add")
def index_add(body: IndexAddInput):
    """Manually add a paper to the embedding index."""
    if _index is None:
        raise HTTPException(503, "Index not initialised")
    if body.embedding:
        _index.add(body.doc_id, body.title, body.abstract, body.embedding)
        return {"indexed": True, "index_size": _index.size()}
    # Compute embedding via the model
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    text = f"{body.title} [SEP] {body.abstract}"
    res  = _predictor.predict(text, return_embedding=True)
    if res.embedding:
        _index.add(body.doc_id, body.title, body.abstract, res.embedding,
                   res.predicted_category, res.impact_score)
    return {"indexed": res.embedding is not None, "index_size": _index.size()}


@app.get("/index/stats")
def index_stats():
    if _index is None:
        return {"size": 0}
    return {"size": _index.size(), "entries": _index.get_all_metadata()[:20]}


# ── Monitoring endpoints ──────────────────────────────────────────────────────
# Gama et al. (2014). Concept Drift Adaptation.

@app.get("/monitor/stats")
def monitor_stats(window_hours: float = 24.0):
    """
    Return prediction statistics for the rolling window.
    Includes: category distribution, confidence stats, drift score, ECE.
    """
    from ml.inference.monitoring import get_rolling_stats
    return get_rolling_stats(window_hours)


@app.get("/monitor/hourly")
def monitor_hourly(hours: int = 48):
    """Return per-hour prediction counts for time-series visualisation."""
    from ml.inference.monitoring import get_hourly_volume
    return {"hourly": get_hourly_volume(hours)}


# ── arXiv proxy — avoids browser CORS block ───────────────────────────────────

@app.get("/arxiv/search")
def arxiv_search(
    q:           str,
    max_results: int = 10,
    sort_by:     str = "submittedDate",
    sort_order:  str = "descending",
):
    """
    Server-side proxy for the arXiv Atom API.
    Browsers cannot fetch export.arxiv.org directly due to CORS restrictions;
    this endpoint fetches on their behalf and returns the raw Atom XML.
    """
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{urllib.parse.quote(q)}"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}"
        f"&sortOrder={sort_order}"
    )
    try:
        resp = _requests.get(url, timeout=20, headers={"User-Agent": "Cortex/2.0"})
        resp.raise_for_status()
        return _Response(content=resp.content, media_type="application/xml")
    except _requests.Timeout:
        raise HTTPException(504, "arXiv API timed out — try again in a moment")
    except _requests.RequestException as exc:
        raise HTTPException(502, f"arXiv API error: {exc}")
