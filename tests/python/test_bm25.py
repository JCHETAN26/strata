"""BM25 and text analysis bindings.

- Scores agree with bm25s (method="lucene": Lucene idf, exact lengths) on SciFact.
- Porter stems agree with Martin Porter's official output and with NLTK (MARTIN_EXTENSIONS).
- The SciFact run reproduces Anserini's published BM25 flat results (bench/validate_bm25_beir.py).
Dataset-dependent tests skip when data/beir/scifact or data/porter is missing.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

strata = pytest.importorskip("strata", reason="bindings not built: run `pip install -e .`")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCIFACT = REPO_ROOT / "data" / "beir" / "scifact"
PORTER = REPO_ROOT / "data" / "porter"
needs_scifact = pytest.mark.skipif(
    not (SCIFACT / "corpus.jsonl").exists(), reason="run scripts/prepare_beir.py scifact"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


# --- Analyzer ----------


def test_analyzer_presets() -> None:
    anserini = strata.Analyzer.anserini_english()
    assert "stemmer='porter'" in repr(anserini) and "lucene_english" in repr(anserini)
    text = "The cats' toys aren't running; John's generalizations"
    assert anserini.analyze(text) == ["cat", "toi", "aren't", "run", "john", "gener"]
    assert strata.Analyzer.plain().analyze("Don't U.S. 3.14") == ["don", "t", "u", "s", "3", "14"]
    custom = strata.Analyzer(tokenizer="unicode", stopwords="lucene_english")
    assert custom.analyze("The U.S. is big") == ["u.s", "big"]
    with pytest.raises(ValueError):
        strata.Analyzer(tokenizer="whitespace")
    with pytest.raises(ValueError):
        strata.Analyzer(stemmer="snowball")


def test_porter_stem_examples() -> None:
    assert strata.porter_stem("generalizations") == "gener"
    assert strata.porter_stem("ponies") == "poni"
    assert strata.porter_stem("is") == "is"


@pytest.mark.skipif(not (PORTER / "voc.txt").exists(), reason="run scripts/fetch_porter_vocab.py")
def test_porter_matches_official_output() -> None:
    words = (PORTER / "voc.txt").read_text().split()
    stems = (PORTER / "output.txt").read_text().split()
    assert len(words) == len(stems) > 20_000
    mismatches = [(w, s, strata.porter_stem(w)) for w, s in zip(words, stems, strict=True)
                  if strata.porter_stem(w) != s]  # fmt: skip
    assert not mismatches, mismatches[:20]


@needs_scifact
def test_porter_matches_nltk_on_scifact_vocabulary() -> None:
    nltk_porter = pytest.importorskip("nltk.stem.porter")
    stemmer = nltk_porter.PorterStemmer(mode=nltk_porter.PorterStemmer.MARTIN_EXTENSIONS)
    unstemmed = strata.Analyzer(english_possessive=True, stopwords="lucene_english")
    vocabulary: set[str] = set()
    for doc in load_jsonl(SCIFACT / "corpus.jsonl")[:2000]:
        vocabulary.update(unstemmed.analyze(f"{doc['title']}\n{doc['text']}"))
    ascii_words = sorted(w for w in vocabulary if w.isascii())
    assert len(ascii_words) > 5000
    mismatches = [(w, stemmer.stem(w, to_lowercase=False), strata.porter_stem(w))
                  for w in ascii_words
                  if stemmer.stem(w, to_lowercase=False) != strata.porter_stem(w)]  # fmt: skip
    assert not mismatches, mismatches[:20]


# --- BM25 ----------


def test_bm25_basics() -> None:
    index = strata.Bm25Index()
    assert (index.k1, index.b, index.length_encoding) == (
        pytest.approx(0.9),
        pytest.approx(0.4),
        "lucene",
    )
    np.testing.assert_array_equal(index.add(["the runner was running", "unrelated text"]), [0, 1])
    assert index.add("a third runner") == 2
    ids, scores = index.search("runners", 5)  # Porter: runners -> runner (not run)
    assert ids.shape == scores.shape == (5,)
    assert list(ids[:2]) == [0, 2] and ids[2] == -1
    # Both docs have length 2 and one "runner": equal scores, ties by ascending id (Lucene order).
    assert scores[0] == scores[1] > 0 and np.isneginf(scores[2:]).all()
    batch_ids, batch_scores = index.search(["runners", "text"], 3)
    assert batch_ids.shape == (2, 3)
    np.testing.assert_array_equal(batch_ids[0], ids[:3])
    np.testing.assert_array_equal(batch_scores[0], scores[:3])
    # "the runner was running" -> runner, run ("the", "was" are stopwords)
    assert index.doc_length(0) == 2 and index.doc_freq("run") == 1 and index.doc_freq("runner") == 2
    assert (len(index), index.doc_count, index.vocabulary_size) == (3, 3, 5)


def test_bm25_tokens_api_and_errors() -> None:
    index = strata.Bm25Index(length_encoding="exact", analyzer=strata.Analyzer.plain())
    index.add_tokens([["a", "b"], ["b", "c", "c"]])
    ids, _ = index.search_tokens(["c"], 2)
    assert list(ids) == [1, -1]
    assert index.search_tokens([["b"], ["zzz"]], 2)[0].tolist() == [[0, 1], [-1, -1]]
    index.remove(1)
    assert index.is_deleted(1) and index.live_size == 1
    assert index.search_tokens(["c"], 2)[0].tolist() == [-1, -1]
    with pytest.raises(KeyError):
        index.remove(1)
    with pytest.raises(ValueError):
        strata.Bm25Index(length_encoding="approximate")
    with pytest.raises(ValueError):
        strata.Bm25Index(b=2.0)


def test_bm25_and_vector_index_share_ids() -> None:
    texts = ["alpha beta", "beta gamma", "gamma delta", "delta alpha"]
    vectors = np.eye(4, dtype=np.float32)
    bm25 = strata.Bm25Index()
    dense = strata.BruteForceIndex(4)
    np.testing.assert_array_equal(bm25.add(texts), dense.add(vectors))
    bm25.remove(2)
    dense.remove(2)
    assert bm25.live_size == dense.live_size
    assert bm25.add("epsilon") == int(dense.add(np.ones(4, np.float32))[0]) == 4
    # A text query and a vector query about the same document return the same id.
    assert bm25.search("gamma", 1)[0][0] == dense.search(vectors[1], 1)[0][0] == 1


def test_bm25_concurrent_adds_and_searches() -> None:
    index = strata.Bm25Index()
    index.add([f"document number {i} about topic {i % 7}" for i in range(200)])

    def writer(batch: int) -> np.ndarray:
        return index.add([f"extra {batch} {j} topic" for j in range(20)])

    def reader(q: int) -> None:
        ids, _ = index.search([f"topic {q % 7}"] * 10, 5, threads=1)
        assert (ids[:, 0] >= 0).all()

    with ThreadPoolExecutor(8) as pool:
        added = list(pool.map(writer, range(10)))
        list(pool.map(reader, range(40)))
    assert len(index) == 400
    np.testing.assert_array_equal(np.sort(np.concatenate(added)), np.arange(200, 400))


@needs_scifact
def test_bm25_matches_bm25s_lucene_variant() -> None:
    bm25s = pytest.importorskip("bm25s")
    corpus = load_jsonl(SCIFACT / "corpus.jsonl")[:2000]
    queries = [q["text"] for q in load_jsonl(SCIFACT / "queries.jsonl")[:100]]
    analyzer = strata.Analyzer.anserini_english()
    doc_tokens = [analyzer.analyze(f"{d['title']}\n{d['text']}") for d in corpus]

    ours = strata.Bm25Index(k1=0.9, b=0.4, length_encoding="exact")
    ours.add_tokens(doc_tokens)
    theirs = bm25s.BM25(k1=0.9, b=0.4, method="lucene", idf_method="lucene")
    theirs.index(doc_tokens, show_progress=False)
    vocabulary = set(theirs.vocab_dict)

    compared = 0
    for query in queries:
        tokens = [t for t in analyzer.analyze(query) if t in vocabulary]
        if not tokens:
            continue
        expected = np.asarray(theirs.get_scores(tokens), dtype=np.float64)
        ids, scores = ours.search_tokens(tokens, len(corpus))
        got = np.zeros(len(corpus))
        got[ids[ids >= 0]] = scores[ids >= 0]
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6, err_msg=query)
        compared += 1
    assert compared > 90


@needs_scifact
def test_scifact_reproduces_anserini() -> None:
    import validate_bm25_beir

    assert validate_bm25_beir.main(["--dataset", "scifact"]) == 0
