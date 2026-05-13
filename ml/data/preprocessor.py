"""
Text preprocessing and data augmentation utilities for arXiv papers.

Preprocessing steps applied before tokenisation:
  1. Unicode normalisation and whitespace collapse
  2. LaTeX math removal / placeholder substitution
  3. Reference / citation strip
  4. Length gate (drop very short abstracts < 30 words)

Augmentation strategies (applied only during training):
  A. Synonym replacement via WordNet
  B. Random sentence deletion (probability p)
  C. Back-translation stub (disabled by default — requires MT model)
"""

import re
import unicodedata
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Cleaning ────────────────────────────────────────────────────────────────

_LATEX_INLINE   = re.compile(r'\$[^$]{1,120}\$')
_LATEX_DISPLAY  = re.compile(r'\$\$[\s\S]{1,500}?\$\$')
_CITATION_REF   = re.compile(r'\[[^\]]{1,40}\]|\(([A-Z][a-z]+,?\s?)+\d{4}\w?\)')
_MULTI_SPACE    = re.compile(r'[ \t]{2,}')
_URL            = re.compile(r'https?://\S+|www\.\S+')
_EMAIL          = re.compile(r'\S+@\S+\.\S+')


def clean_text(text: str) -> str:
    """Apply standard cleaning pipeline to a single string."""
    if not isinstance(text, str):
        return ""

    # Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # Remove citations and URLs FIRST (before inserting math placeholders,
    # so the [MATH] tag is not accidentally caught by the citation bracket regex)
    text = _CITATION_REF.sub("", text)
    text = _URL.sub("", text)
    text = _EMAIL.sub("", text)

    # Replace display math, then inline math, with a placeholder
    text = _LATEX_DISPLAY.sub(" [MATH] ", text)
    text = _LATEX_INLINE.sub(" [MATH] ", text)

    # Collapse whitespace and normalise newlines
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def is_valid_abstract(abstract: str, min_words: int = 30) -> bool:
    """Return True if abstract meets minimum quality requirements."""
    return len(abstract.split()) >= min_words


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate an entire dataframe.

    Columns expected: title, abstract (others passed through).
    Rows with too-short abstracts are dropped.
    """
    df = df.copy()
    df["title"]    = df.get("title", pd.Series([""] * len(df))).apply(clean_text)
    df["abstract"] = df.get("abstract", pd.Series([""] * len(df))).apply(clean_text)

    before = len(df)
    df = df[df["abstract"].apply(is_valid_abstract)]
    logger.info("Dropped %d rows with short abstracts (kept %d)", before - len(df), len(df))

    return df.reset_index(drop=True)


# ── Augmentation ────────────────────────────────────────────────────────────

def synonym_replace(text: str, p: float = 0.1) -> str:
    """
    Replace random tokens with a WordNet synonym.
    Falls back silently if NLTK / WordNet not available.
    """
    try:
        import random
        from nltk.corpus import wordnet  # type: ignore

        words  = text.split()
        result = []
        for word in words:
            if random.random() < p:
                synsets = wordnet.synsets(word)
                if synsets:
                    lemmas = synsets[0].lemmas()
                    replacements = [l.name().replace("_", " ") for l in lemmas if l.name() != word]
                    if replacements:
                        result.append(random.choice(replacements))
                        continue
            result.append(word)
        return " ".join(result)
    except Exception:
        return text


def random_sentence_delete(text: str, p: float = 0.1) -> str:
    """Delete each sentence independently with probability `p`."""
    import random
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = [s for s in sentences if random.random() > p or len(sentences) <= 2]
    return " ".join(kept) if kept else text


class TextAugmenter:
    """
    Applies a configurable augmentation pipeline to text strings.

    Usage::

        aug = TextAugmenter(synonym_p=0.1, delete_p=0.05)
        augmented = aug(original_text)
    """

    def __init__(
        self,
        synonym_p: float = 0.1,
        delete_p: float = 0.05,
        enabled: bool = True,
    ):
        self.synonym_p = synonym_p
        self.delete_p  = delete_p
        self.enabled   = enabled

    def __call__(self, text: str) -> str:
        if not self.enabled:
            return text
        text = synonym_replace(text, p=self.synonym_p)
        text = random_sentence_delete(text, p=self.delete_p)
        return text
