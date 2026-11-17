import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nwp.config import Config
from nwp.embeddings import build_embedding_matrix, train_word2vec
from nwp.model import NextWordLSTM
from nwp.predict import Predictor
from nwp.preprocess import (
    clean_text,
    normalize_sentence,
    preprocess_corpus,
    remove_punctuation,
    strip_gutenberg_boilerplate,
    tokenize,
    to_lowercase,
)
from nwp.sequences import NextWordDataset, make_sequences, split_sentences
from nwp.trainer import evaluate, train
from nwp.vocab import PAD, UNK, Vocabulary

RAW = (
    "Header licence text\n"
    "*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
    "“My dear Watson,” said Holmes. “It’s quite simple!”\n\n"
    "The door opened—slowly.\n"
    "*** END OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
    "Footer licence text"
)


# --- preprocessing -------------------------------------------------------

def test_boilerplate_is_removed():
    body = strip_gutenberg_boilerplate(RAW)
    assert "Header" not in body and "Footer" not in body
    assert "Holmes" in body


def test_clean_text_makes_ascii():
    assert clean_text(RAW).isascii()


def test_lowercase():
    assert to_lowercase("Sherlock HOLMES") == "sherlock holmes"


def test_remove_punctuation_keeps_inner_apostrophes():
    assert remove_punctuation("it's 'quite' simple, watson!") == "it's quite simple watson"


def test_tokenize():
    assert tokenize("my dear watson") == ["my", "dear", "watson"]


def test_preprocess_corpus_splits_sentences():
    sentences = preprocess_corpus(RAW)
    assert sentences == [
        ["my", "dear", "watson", "said", "holmes"],
        ["it's", "quite", "simple"],
        ["the", "door", "opened", "slowly"],
    ]


def test_prompt_and_corpus_normalised_identically():
    assert normalize_sentence("My dear, WATSON!") == ["my", "dear", "watson"]


def test_unicode_prompt_matches_corpus():
    text = "“It’s café—quite simple,” said Holmes."
    assert normalize_sentence(text) == preprocess_corpus(text)[0]
    assert normalize_sentence(text) == ["it's", "cafe", "quite", "simple", "said", "holmes"]


# --- vocabulary ----------------------------------------------------------

def test_vocab_specials_and_min_freq():
    vocab = Vocabulary.build([["a", "b", "a"], ["a", "c", "b"]], min_freq=2)
    assert vocab.idx2word == [PAD, UNK, "a", "b"]
    assert vocab.encode(["a", "c"]) == [2, vocab.unk_id]
    assert vocab.decode([2, 3]) == ["a", "b"]


def test_vocab_roundtrip(tmp_path):
    vocab = Vocabulary.build([["x", "y", "x"]])
    vocab.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").idx2word == vocab.idx2word


# --- sequences -----------------------------------------------------------

def test_make_sequences_sliding_window():
    vocab = Vocabulary.build([["a", "b", "c", "d"]])
    x, y = make_sequences([["a", "b"], ["c", "d"]], vocab, context_size=2)
    a, b, c, d = vocab.encode(["a", "b", "c", "d"])
    assert x.tolist() == [[0, 0], [0, a], [a, b], [b, c]]
    assert y.tolist() == [a, b, c, d]


def test_make_sequences_skips_unknown_targets():
    vocab = Vocabulary.build([["a", "a"]], min_freq=2)
    _, y = make_sequences([["a", "zzz", "a"]], vocab, context_size=1)
    assert vocab.unk_id not in y.tolist()


def test_split_is_ordered_and_complete():
    data = [[str(i)] for i in range(10)]
    tr, va, te = split_sentences(data, 0.8, 0.1)
    assert tr + va + te == data and len(tr) == 8


def test_config_validation():
    with pytest.raises(ValueError):
        Config(train_frac=0.95, val_frac=0.1)


# --- embeddings, model and training ---------------------------------------

@pytest.fixture(scope="module")
def tiny_setup():
    torch.manual_seed(0)
    sentences = [["the", "cat", "sat", "on", "the", "mat"]] * 40 + [["the", "dog", "ran", "to", "the", "park"]] * 40
    vocab = Vocabulary.build(sentences)
    w2v = train_word2vec(sentences, vector_size=16, window=2, min_count=1, sg=1, negative=5, epochs=5, seed=0)
    matrix, covered = build_embedding_matrix(w2v, vocab)
    x, y = make_sequences(sentences, vocab, context_size=3)
    return vocab, w2v, matrix, covered, x, y


def test_embedding_matrix(tiny_setup):
    vocab, w2v, matrix, covered, *_ = tiny_setup
    assert matrix.shape == (len(vocab), 16)
    assert covered == len(vocab) - 2
    assert np.all(matrix[vocab.pad_id] == 0)
    assert np.allclose(matrix[vocab.word2idx["cat"]], w2v.wv["cat"])


def test_model_output_shape(tiny_setup):
    vocab, _, matrix, *_ = tiny_setup
    model = NextWordLSTM(matrix, hidden_size=8, num_layers=1, dropout=0.0)
    assert model(torch.zeros((4, 3), dtype=torch.long)).shape == (4, len(vocab))


def test_predictor_excludes_special_tokens(tiny_setup):
    vocab, _, matrix, *_ = tiny_setup
    model = NextWordLSTM(matrix, hidden_size=8, num_layers=1, dropout=0.0)
    predictor = Predictor(model, vocab, context_size=3)
    # Very high temperatures exposed epsilon smoothing that reintroduced
    # banned tokens into the sampling distribution.
    assert torch.equal(predictor._probabilities(["the"], temperature=1e12)[:2], torch.zeros(2))
    assert not {PAD, UNK}.intersection(predictor.generate("the", 30, temperature=1e12, seed=0).split())
    with pytest.raises(ValueError, match="k must"):
        predictor.predict_next("the", k=len(vocab))


def test_training_learns_the_pattern(tiny_setup, tmp_path):
    vocab, _, matrix, _, x, y = tiny_setup
    loader = DataLoader(NextWordDataset(x, y), batch_size=32, shuffle=True)
    model = NextWordLSTM(matrix, hidden_size=32, num_layers=1, dropout=0.0)
    device = torch.device("cpu")
    before = evaluate(model, loader, device).loss
    history = train(model, loader, loader, device, epochs=15, learning_rate=1e-2, weight_decay=0.0,
                    grad_clip=1.0, patience=5, freeze_embeddings_epochs=1)
    after = evaluate(model, loader, device)
    assert after.loss < before
    assert history.best_epoch >= 1

    predictor = Predictor(model, vocab, context_size=3)
    assert predictor.predict_next("the cat sat", k=1)[0][0] == "on"
    assert predictor.predict_next("the dog ran", k=1)[0][0] == "to"
    assert len(predictor.generate("the cat", n_words=4).split()) == 6
