"""Step 8 - Convert words to vectors with Word2Vec."""

from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec

from .vocab import Vocabulary


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    negative: int,
    epochs: int,
    seed: int,
) -> Word2Vec:
    """Learn a vector for each word from the words that appear around it.

    A fixed seed and workers=1 reduce training variation. Reproducing runs
    across Python processes also requires a fixed PYTHONHASHSEED; package
    versions and hardware can still affect results.
    """
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        seed=seed,
        workers=1,
    )


def build_embedding_matrix(w2v: Word2Vec, vocab: Vocabulary, seed: int = 0) -> tuple[np.ndarray, int]:
    """Row i of the matrix is the Word2Vec vector of vocabulary word i.

    <pad> is all zeros. <unk> is initialized with the mean learned vector;
    this is a practical choice, not a guarantee of a neutral meaning.
    Words Word2Vec did not learn get small random vectors. Returns the matrix
    and how many vocabulary words were covered by Word2Vec.
    """
    dim = w2v.vector_size
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[vocab.pad_id] = 0.0
    matrix[vocab.unk_id] = w2v.wv.vectors.mean(axis=0)

    covered = 0
    for word, idx in vocab.word2idx.items():
        if idx > vocab.unk_id and word in w2v.wv.key_to_index:
            matrix[idx] = w2v.wv[word]
            covered += 1
    return matrix, covered
