"""
In-memory Semantic Similarity Search.

Implements SBERT-style cosine similarity search over paper embeddings.
New papers are registered when processed; the index lives in memory
and is optionally persisted to a JSON file.

References
----------
[1] Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks. EMNLP 2019. arXiv:1908.10084
[2] Johnson et al. (2017). Billion-scale similarity search with GPUs (FAISS).
    arXiv:1702.08734  ← we use a pure-numpy approximation for portability
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = Path("ml/outputs/embedding_index.json")


class EmbeddingIndex:
    """
    Lightweight in-memory cosine similarity index.

    Each entry stores:
      - title
      - abstract snippet
      - 768-D L2-normalised embedding vector
      - predicted category and impact score

    Similarity is computed as cosine distance (dot product of L2-normalised
    vectors), equivalent to the SBERT similarity metric [1].
    """

    def __init__(self):
        self._ids:        List[str]       = []
        self._titles:     List[str]       = []
        self._abstracts:  List[str]       = []
        self._embeddings: Optional[np.ndarray] = None   # (N, D)
        self._categories: List[str]       = []
        self._impacts:    List[float]     = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if not INDEX_PATH.exists():
            return
        try:
            data = json.loads(INDEX_PATH.read_text())
            self._ids        = data["ids"]
            self._titles     = data["titles"]
            self._abstracts  = data["abstracts"]
            self._categories = data["categories"]
            self._impacts    = data["impacts"]
            arr = np.array(data["embeddings"], dtype=np.float32)
            self._embeddings = arr if len(arr) else None
            logger.info("Loaded embedding index: %d entries", len(self._ids))
        except Exception as exc:
            logger.warning("Could not load embedding index: %s", exc)

    def save(self):
        try:
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "ids":        self._ids,
                "titles":     self._titles,
                "abstracts":  self._abstracts,
                "categories": self._categories,
                "impacts":    self._impacts,
                "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
            }
            INDEX_PATH.write_text(json.dumps(data))
        except Exception as exc:
            logger.debug("Could not save embedding index: %s", exc)

    # ── Index management ───────────────────────────────────────────────────

    def add(
        self,
        doc_id:    str,
        title:     str,
        abstract:  str,
        embedding: List[float],
        category:  str  = "",
        impact:    float = 0.0,
    ):
        """Add a document to the index. Skips duplicates."""
        if doc_id in self._ids:
            return

        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm          # L2 normalise for cosine similarity

        self._ids.append(doc_id)
        self._titles.append(title)
        self._abstracts.append(abstract[:500])
        self._categories.append(category)
        self._impacts.append(impact)

        if self._embeddings is None:
            self._embeddings = vec.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

        if len(self._ids) % 10 == 0:
            self.save()

    def size(self) -> int:
        return len(self._ids)

    def clear(self):
        self._ids = []; self._titles = []; self._abstracts = []
        self._embeddings = None; self._categories = []; self._impacts = []

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict]:
        """
        Return the top-k most similar documents to query_embedding.

        Similarity metric: cosine similarity (dot product of L2-normalised
        vectors) — equivalent to SBERT [1].

        Returns list of dicts: {id, title, abstract, category, impact, score}
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity: dot product of normalised vectors
        scores = (self._embeddings @ q).flatten()   # (N,)

        # Sort descending
        order = np.argsort(-scores)

        results = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                break
            results.append({
                "id":       self._ids[idx],
                "title":    self._titles[idx],
                "abstract": self._abstracts[idx],
                "category": self._categories[idx],
                "impact":   self._impacts[idx],
                "score":    round(score, 4),
            })

        return results

    def get_all_metadata(self) -> List[Dict]:
        """Return all indexed documents without embeddings (for display)."""
        return [
            {"id": i, "title": t, "category": c, "impact": imp}
            for i, t, c, imp in zip(
                self._ids, self._titles, self._categories, self._impacts
            )
        ]


# Global singleton
_index: Optional[EmbeddingIndex] = None


def get_index() -> EmbeddingIndex:
    global _index
    if _index is None:
        _index = EmbeddingIndex()
    return _index
