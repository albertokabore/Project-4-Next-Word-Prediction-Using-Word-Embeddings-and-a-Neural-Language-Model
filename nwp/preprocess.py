"""Steps 2-5 - Clean the text, lowercase it, remove punctuation and tokenize.

The same ``normalize_sentence`` function is used for training data and for
user prompts at prediction time, so both are processed identically.
"""

from __future__ import annotations

import re
import unicodedata

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG", re.I)

# Typographic characters mapped to plain ASCII before anything else.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": " ", "—": " ", "…": "...", " ": " ",
})

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
# Anything that is not a letter, digit, apostrophe or space is punctuation.
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
# Apostrophes are kept only inside words ("don't", "holmes's").
_STRAY_APOSTROPHE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")
_WHITESPACE = re.compile(r"\s+")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the book itself, dropping Project Gutenberg's licence header/footer."""
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def clean_text(text: str) -> str:
    """Remove boilerplate, normalise unicode and drop non-ASCII symbols."""
    text = strip_gutenberg_boilerplate(text)
    text = text.translate(_CHAR_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.replace("\r\n", "\n")


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    text = _NON_WORD.sub(" ", text)
    text = _STRAY_APOSTROPHE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return text.split()


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation and blank lines (before punctuation is removed)."""
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s and s.strip()]


def normalize_sentence(sentence: str) -> list[str]:
    """Normalize characters, lowercase, remove punctuation and tokenize."""
    return tokenize(remove_punctuation(to_lowercase(clean_text(sentence))))


def preprocess_corpus(raw_text: str) -> list[list[str]]:
    """Full preprocessing: raw book text -> list of tokenized sentences."""
    cleaned = clean_text(raw_text)
    sentences = (normalize_sentence(s) for s in split_sentences(cleaned))
    return [tokens for tokens in sentences if tokens]
