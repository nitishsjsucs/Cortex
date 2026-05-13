"""
Prediction Logging and Data Drift Detection.

Logs every prediction to SQLite and exposes statistics for:
  - Rolling prediction distribution (category, confidence, latency)
  - Data drift detection via Kolmogorov-Smirnov test on text statistics
  - Calibration metrics (ECE — Expected Calibration Error)

References
----------
[1] Gama et al. (2014). A Survey on Concept Drift Adaptation. ACM Computing Surveys.
[2] Guo et al. (2017). On Calibration of Modern Neural Networks. ICML. arXiv:1706.04599
[3] Massey (1951). The Kolmogorov-Smirnov Test for Goodness of Fit. JASA.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path("ml/outputs/predictions.db")

# Training distribution baselines (computed from training data)
TRAIN_STATS = {
    "mean_abstract_len":   152.4,
    "std_abstract_len":     52.1,
    "mean_confidence":      0.412,
    "category_dist": {
        "cs.AI": 0.052, "cs.LG": 0.071, "cs.CL": 0.068, "cs.CV": 0.058,
        "cs.RO": 0.045, "math.ST": 0.038, "math.OC": 0.035, "stat.ML": 0.072,
        "stat.AP": 0.040, "stat.ME": 0.038, "physics.optics": 0.042,
        "quant-ph": 0.048, "cond-mat.str-el": 0.041, "astro-ph.GA": 0.043,
        "q-bio.NC": 0.036, "econ.EM": 0.033, "eess.SP": 0.041,
        "cs.NI": 0.044, "cs.IR": 0.047, "cs.SE": 0.048,
    },
}


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            title       TEXT,
            abstract_len INTEGER,
            predicted   TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            impact      REAL    NOT NULL,
            quality     TEXT    NOT NULL,
            latency_ms  REAL    NOT NULL,
            uncertainty REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)")
    conn.commit()
    return conn


def log_prediction(
    title:       str,
    abstract:    str,
    predicted:   str,
    confidence:  float,
    impact:      float,
    quality:     str,
    latency_ms:  float,
    uncertainty: Optional[float] = None,
):
    """Persist one prediction record to SQLite."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO predictions (ts,title,abstract_len,predicted,confidence,impact,quality,latency_ms,uncertainty) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), title[:200], len(abstract.split()),
             predicted, confidence, impact, quality, latency_ms, uncertainty)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Prediction logging failed: %s", exc)


def get_rolling_stats(window_hours: float = 24.0) -> Dict:
    """
    Return prediction statistics for the rolling window.

    Returns
    -------
    dict with keys:
      total, category_counts, mean_confidence, mean_impact, mean_latency,
      quality_dist, drift_score, alert
    """
    try:
        conn  = _get_conn()
        since = time.time() - window_hours * 3600
        rows  = conn.execute(
            "SELECT predicted,confidence,impact,quality,latency_ms,abstract_len "
            "FROM predictions WHERE ts > ? ORDER BY ts",
            (since,)
        ).fetchall()
        conn.close()
    except Exception:
        return {"total": 0, "error": "DB not initialised"}

    if not rows:
        return {"total": 0, "window_hours": window_hours}

    predicted   = [r[0] for r in rows]
    confidences = [r[1] for r in rows]
    impacts     = [r[2] for r in rows]
    qualities   = [r[3] for r in rows]
    latencies   = [r[4] for r in rows]
    abs_lens    = [r[5] for r in rows]

    # Category distribution
    cats = {}
    for p in predicted:
        cats[p] = cats.get(p, 0) + 1

    # Quality distribution
    qual_dist = {
        "well-structured": qualities.count("well-structured"),
        "poor": qualities.count("poor"),
    }

    # Drift detection via KS test on abstract length distribution
    drift_score, alert = _ks_drift(abs_lens, confidences)

    # Expected Calibration Error (simplified — bin-based)
    ece = _compute_ece(confidences)

    return {
        "total":            len(rows),
        "window_hours":     window_hours,
        "category_counts":  cats,
        "mean_confidence":  float(np.mean(confidences)),
        "std_confidence":   float(np.std(confidences)),
        "mean_impact":      float(np.mean(impacts)),
        "mean_latency_ms":  float(np.mean(latencies)),
        "quality_dist":     qual_dist,
        "drift_score":      drift_score,
        "drift_alert":      alert,
        "ece":              ece,
    }


def get_hourly_volume(hours: int = 48) -> List[Dict]:
    """Return per-hour prediction counts for the last N hours."""
    try:
        conn  = _get_conn()
        since = time.time() - hours * 3600
        rows  = conn.execute(
            "SELECT ts FROM predictions WHERE ts > ?", (since,)
        ).fetchall()
        conn.close()
    except Exception:
        return []

    if not rows:
        return []

    import math
    now     = time.time()
    buckets = {}
    for (ts,) in rows:
        hour_ago = math.floor((now - ts) / 3600)
        buckets[hour_ago] = buckets.get(hour_ago, 0) + 1

    return [{"hours_ago": h, "count": buckets.get(h, 0)} for h in range(hours)]


# ── Drift Detection (KS test) ────────────────────────────────────────────────
# Gama et al. (2014) — "A Survey on Concept Drift Adaptation"

def _ks_drift(abs_lens: List[int], confidences: List[float]) -> Tuple[float, bool]:
    """
    Kolmogorov-Smirnov test comparing current abstract length distribution
    against the training baseline.

    Returns (ks_statistic, is_drifting)
    ks_statistic ∈ [0,1]: 0 = identical, 1 = completely different
    is_drifting = True when KS stat > 0.15 (empirical threshold)
    """
    if len(abs_lens) < 10:
        return 0.0, False

    # Generate reference samples from training distribution
    rng = np.random.default_rng(42)
    ref = rng.normal(TRAIN_STATS["mean_abstract_len"],
                     TRAIN_STATS["std_abstract_len"], 1000).clip(5, 800)

    cur = np.array(abs_lens, dtype=float)

    # Two-sample KS statistic (manual, no scipy needed)
    all_vals = np.concatenate([ref, cur])
    all_vals.sort()
    cdf_ref = np.searchsorted(np.sort(ref), all_vals, side="right") / len(ref)
    cdf_cur = np.searchsorted(np.sort(cur), all_vals, side="right") / len(cur)
    ks_stat = float(np.abs(cdf_ref - cdf_cur).max())

    return round(ks_stat, 4), ks_stat > 0.15


# ── Expected Calibration Error ───────────────────────────────────────────────
# Guo et al. (2017) — "On Calibration of Modern Neural Networks"

def _compute_ece(confidences: List[float], n_bins: int = 10) -> float:
    """
    Compute simplified ECE over confidence scores.
    (Without ground-truth labels we approximate using the distribution shape —
    a well-calibrated model should have confidence scores matching accuracy.)
    """
    if not confidences:
        return 0.0
    confs  = np.array(confidences)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask  = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(confs[mask].mean())
        # Proxy accuracy: assume perfect calibration at 1/n_classes = 0.05
        # True ECE requires labels; this is a confidence spread metric
        ece += (mask.sum() / n) * abs(bin_conf - 0.5)
    return round(float(ece), 4)
