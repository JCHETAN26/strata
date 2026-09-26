"""Fusion bindings, HybridIndex, embedding input formatting, and the SciFact hybrid evaluation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

strata = pytest.importorskip("strata", reason="bindings not built: run `pip install -e .`")

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = REPO_ROOT / "data" / "embeddings" / "scifact" / "bge-small-en-v1.5"


# --- Fusion vs a pure-Python reference ----------


def rrf_reference(lists: list[list[int]], k: int, rrf_k: float) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def weighted_reference(
    lists: list[list[tuple[int, float]]], weights: list[float], k: int
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked, w in zip(lists, weights, strict=True):
        if not ranked:
            continue
        values = [s for _, s in ranked]
        lo, hi = min(values), max(values)
        for doc, s in ranked:
            norm = (s - lo) / (hi - lo) if hi > lo else 1.0
            scores[doc] = scores.get(doc, 0.0) + w * norm
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def random_lists(rng: np.random.Generator, rows: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((rows, width), -1, dtype=np.int64)
    scores = np.full((rows, width), -np.inf, dtype=np.float32)
    for r in range(rows):
        n = int(rng.integers(0, width + 1))
        ids[r, :n] = rng.choice(50, size=n, replace=False)
        scores[r, :n] = np.sort(rng.normal(size=n).astype(np.float32))[::-1]
    return ids, scores


@pytest.mark.parametrize("seed", range(3))
def test_fuse_rrf_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, _ = random_lists(rng, 20, 15)
    b, _ = random_lists(rng, 20, 15)
    ids, scores = strata.fuse_rrf([a, b], 10, rrf_k=60.0)
    for r in range(20):
        expected = rrf_reference(
            [[i for i in a[r] if i >= 0], [i for i in b[r] if i >= 0]], 10, 60.0
        )
        n = len(expected)
        assert ids[r, :n].tolist() == [d for d, _ in expected]
        np.testing.assert_allclose(scores[r, :n], [s for _, s in expected], rtol=1e-6)
        assert (ids[r, n:] == -1).all() and np.isneginf(scores[r, n:]).all()


@pytest.mark.parametrize("seed", range(3))
def test_fuse_weighted_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a_ids, a_scores = random_lists(rng, 20, 15)
    b_ids, b_scores = random_lists(rng, 20, 15)
    ids, scores = strata.fuse_weighted([a_ids, b_ids], [a_scores, b_scores], [0.3, 0.7], 10)
    for r in range(20):
        lists = [
            [(int(i), float(s)) for i, s in zip(a_ids[r], a_scores[r], strict=True) if i >= 0],
            [(int(i), float(s)) for i, s in zip(b_ids[r], b_scores[r], strict=True) if i >= 0],
        ]
        expected = weighted_reference(lists, [0.3, 0.7], 10)
        n = len(expected)
        assert ids[r, :n].tolist() == [d for d, _ in expected]
        np.testing.assert_allclose(scores[r, :n], [s for _, s in expected], rtol=1e-5, atol=1e-6)


def test_fusion_argument_errors() -> None:
    ids = np.zeros((2, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        strata.fuse_rrf([], 5)
    with pytest.raises(ValueError):
        strata.fuse_rrf([ids, np.zeros((3, 3), dtype=np.int64)], 5)
    with pytest.raises(ValueError):
        strata.fuse_rrf([ids], 5, rrf_k=-1.0)
    with pytest.raises(ValueError):
        strata.fuse_weighted([ids], [np.zeros((2, 3), np.float32)], [0.5, 0.5], 5)
    with pytest.raises(ValueError):
        strata.fuse_weighted([ids], [np.zeros((2, 2), np.float32)], [1.0], 5)


# --- HybridIndex ----------


def small_hybrid() -> tuple[strata.HybridIndex, np.ndarray]:
    texts = ["alpha beta", "beta gamma", "gamma delta", "delta alpha"]
    vectors = np.eye(4, dtype=np.float32)
    index = strata.HybridIndex(4, metric="ip")
    np.testing.assert_array_equal(index.add(["a", "b", "c", "d"], texts, vectors), [0, 1, 2, 3])
    return index, vectors


def test_hybrid_methods_match_underlying_indexes() -> None:
    index, vectors = small_hybrid()
    dense_ids, _ = index.search(["gamma"], vectors[1:2], 2, method="dense")
    assert dense_ids.tolist() == index.vector_index.search(vectors[1:2], 2)[0].tolist()
    bm25_ids, bm25_scores = index.search(["gamma"], vectors[1:2], 2, method="bm25")
    ref_ids, ref_scores = index.bm25.search(["gamma"], 2)
    np.testing.assert_array_equal(bm25_ids, ref_ids)
    np.testing.assert_array_equal(bm25_scores, ref_scores)
    # b is first for both retrievers, so it wins every fusion.
    for method in ("rrf", "weighted"):
        docs, _ = index.search_doc_ids(["gamma"], vectors[1:2], 3, method=method)
        assert docs[0][0] == "b"


def test_hybrid_validation_leaves_indexes_untouched() -> None:
    index, vectors = small_hybrid()
    with pytest.raises(ValueError, match="already present"):
        index.add(["e", "a"], ["x", "y"], vectors[:2])
    with pytest.raises(ValueError, match="same length"):
        index.add(["e"], ["x", "y"], vectors[:2])
    with pytest.raises(ValueError, match="duplicate"):
        index.add(["e", "e"], ["x", "y"], vectors[:2])
    with pytest.raises(ValueError, match="shape"):
        index.add(["e"], ["x"], np.ones((1, 5), np.float32))
    assert len(index.vector_index) == len(index.bm25) == len(index) == 4
    with pytest.raises(ValueError):
        index.search(["q"], vectors[:1], 2, method="magic")
    with pytest.raises(ValueError):
        index.search(["q"], vectors[:1], 2, method="weighted", weight=1.5)
    with pytest.raises(ValueError):
        strata.HybridIndex(4, vector_index=index.vector_index)  # not empty


def test_hybrid_remove_applies_to_both() -> None:
    index, vectors = small_hybrid()
    index.remove("b")
    assert index.vector_index.is_deleted(1) and index.bm25.is_deleted(1) and len(index) == 3
    for method in ("dense", "bm25", "rrf", "weighted"):
        docs, _ = index.search_doc_ids(["beta gamma"], vectors[1:2], 4, method=method)
        assert "b" not in docs[0], method
    with pytest.raises(KeyError):
        index.remove("b")
    assert index.add(["e"], ["epsilon"], np.ones((1, 4), np.float32)).tolist() == [4]


def test_hybrid_concurrent_adds_keep_ids_aligned() -> None:
    index = strata.HybridIndex(8, metric="ip")
    rng = np.random.default_rng(0)

    def writer(batch: int) -> None:
        index.add([f"{batch}-{j}" for j in range(25)], [f"doc {batch} {j}" for j in range(25)],
                  rng.standard_normal((25, 8)).astype(np.float32))  # fmt: skip

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(writer, range(16)))
    assert len(index.vector_index) == len(index.bm25) == 400
    # Every external id maps to the internal id whose text it was added with.
    for internal in range(0, 400, 37):
        batch, j = index.doc_id(internal).split("-")
        assert index.bm25.search(f"doc {batch} {j}", 1)[0][0] >= 0


# --- Embedding input formatting ----------


def test_embedding_input_formatting() -> None:
    from embed_beir import MODELS, format_docs, format_queries

    bge, e5 = MODELS["bge-small-en-v1.5"], MODELS["e5-small-v2"]
    docs = [{"title": "Aspirin", "text": "lowers stroke risk"}, {"title": "", "text": "no title"}]
    # bge: instruction on queries only; passages unchanged.
    assert format_queries(bge, ["q"]) == [
        "Represent this sentence for searching relevant passages: q"
    ]
    assert format_docs(bge, docs) == ["Aspirin lowers stroke risk", "no title"]
    # e5: query: / passage: prefixes.
    assert format_queries(e5, ["q"]) == ["query: q"]
    assert format_docs(e5, docs) == ["passage: Aspirin lowers stroke risk", "passage: no title"]
    for spec in MODELS.values():
        assert spec.normalize and len(spec.revision) == 40  # pinned to a full commit hash


# --- SciFact end to end ----------


def test_weight_rule_prefers_dev_then_train_then_fixed(tmp_path: Path) -> None:
    from eval_hybrid_beir import choose_tuning_split

    qrels = tmp_path / "qrels"
    qrels.mkdir()
    (qrels / "test.tsv").write_text("query-id\tcorpus-id\tscore\n")
    assert choose_tuning_split(qrels) is None  # -> fixed weight
    (qrels / "train.tsv").write_text("")
    assert choose_tuning_split(qrels) == "train"
    (qrels / "dev.tsv").write_text("")
    assert choose_tuning_split(qrels) == "dev"


@pytest.mark.skipif(not (EMBEDDINGS / "meta.json").exists(), reason="run scripts/embed_beir.py")
def test_scifact_hybrid_baselines_match_published(tmp_path: Path) -> None:
    import eval_hybrid_beir

    assert eval_hybrid_beir.main(["--dataset", "scifact", "--out-dir", str(tmp_path)]) == 0
    (result,) = tmp_path.glob("*.json")
    record = json.loads(result.read_text())
    # SciFact has train but no dev split: the rule picks train.
    assert record["protocol"]["weight_source"] == "train"
    assert record["embedding"]["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert record["baselines_match_published"]
    # Six pairs per metric, each with a CI around its mean difference, from saved per-query data.
    pairs = record["significance"]["nDCG@10"]
    assert len(pairs) == 6
    for c in pairs:
        assert c["ci_low"] <= c["mean_diff"] <= c["ci_high"]
        assert 0 < c["p_value"] <= c["p_holm"] <= 1
        assert c["queries"] == 300 == len(record["per_query"][c["a"]]["nDCG@10"])
