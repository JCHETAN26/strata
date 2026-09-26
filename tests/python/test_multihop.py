"""Two-hop retrieval: query expansion, candidate ordering, combination, and the evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from multihop import EXPANSIONS, combine, expansion_text, hop2_candidates, hop2_query
from rerank import Reranker

REPO_ROOT = Path(__file__).resolve().parents[2]
BEIR = REPO_ROOT / "data" / "beir"
EMBEDDINGS = REPO_ROOT / "data" / "embeddings"
TEST = "hotpotqa-subset-n100-seed0-bg20000"
DEV = "hotpotqa-dev-subset-n100-seed0-bg20000"
RERANK_RESULT = next(iter(sorted((REPO_ROOT / "results" / "rerank").glob(f"{TEST}-*.json"))), None)


def test_expansions() -> None:
    sentences = ["", "Tom was born in Oslo.", " He moved to Paris."]
    assert expansion_text("Tom", sentences, "title") == "Tom"
    assert expansion_text("Tom", sentences, "title_first_sentence") == "Tom Tom was born in Oslo."
    assert (
        expansion_text("Tom", sentences, "full") == "Tom Tom was born in Oslo. He moved to Paris."
    )
    assert hop2_query("Where?", "Tom", sentences, "title") == "Where? Tom"
    assert set(EXPANSIONS) == {"title", "title_first_sentence", "full"}
    with pytest.raises(ValueError):
        expansion_text("Tom", sentences, "abstract")


def test_hop2_candidates_round_robin_without_scores() -> None:
    per_p = [(["a", "b", "c"], ["b", "d", "e"]), (["f", "a"], ["g"])]
    # Rank 1 of all four lists, then rank 2, ...; duplicates dropped.
    assert hop2_candidates(per_p, 2) == ["a", "b", "f", "g", "d"]
    assert hop2_candidates(per_p, 1) == ["a", "b", "f", "g"]
    assert hop2_candidates([], 3) == []


def test_hop2_candidates_best_score_across_expanded_passages() -> None:
    per_p = [(["a", "b"], ["c"]), (["b", "d"], [])]
    scores = [{"a": 1.0, "b": 0.0, "c": 2.0}, {"b": 5.0, "d": 1.0}]
    # b gets its best score (5, from p2); a and d tie at 1.0 and keep round-robin order.
    assert hop2_candidates(per_p, 2, scores) == ["b", "c", "a", "d"]
    # At depth 1, d is not in p2's pool, and b only enters via p2's pool (score 5).
    assert hop2_candidates(per_p, 1, scores) == ["b", "c", "a"]


def test_combine_keeps_hop1_prefix_then_new_hop2() -> None:
    hop1 = ["h1", "h2", "h3", "h4", "h5"]
    assert combine(hop1, ["h2", "x", "y"], keep=2, k=5) == ["h1", "h2", "x", "y", "h3"]
    assert combine(hop1, ["x"], keep=4, k=5) == ["h1", "h2", "h3", "h4", "x"]
    assert combine(hop1, [], keep=1, k=5) == hop1  # nothing new: hop 1 unchanged
    assert combine(["h1"], ["h1"], keep=1, k=5) == ["h1"]


class HashEncoder:
    """Deterministic 384-d unit vectors from text hashes (no model download)."""

    def __call__(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha1(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


class OverlapCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


@pytest.mark.skipif(
    not all((EMBEDDINGS / n / "meta.json").exists() for n in (TEST, DEV)) or RERANK_RESULT is None,
    reason="needs both subsets' embeddings and a rerank result",
)
def test_eval_multihop_end_to_end(tmp_path: Path) -> None:
    import eval_multihop

    reranker = Reranker("bge-reranker-base", model=OverlapCrossEncoder())
    args = ["--rerank-result", str(RERANK_RESULT), "--out-dir", str(tmp_path)]
    assert eval_multihop.main(args, reranker=reranker, encode=HashEncoder()) == 0
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    baseline = json.loads(RERANK_RESULT.read_text())

    assert record["tune_dataset"] == DEV
    assert len(record["tuning"]["stage1"]) == 3 * 3 * 3 * 4
    assert len(record["tuning"]["stage2"]) == 2 * 4
    no_model, bge = record["configs"]["multihop_no_model"], record["configs"]["multihop_bge"]
    assert not no_model["rerank"] and bge["rerank"]
    assert (bge["m"], bge["expansion"]) == (no_model["m"], no_model["expansion"])
    # The chosen configurations are the best on dev.
    best1 = max(r["R@5"] for r in record["tuning"]["stage1"])
    assert any(
        r["R@5"] == best1 and not r["rerank"] and r["m"] == no_model["m"]
        for r in record["tuning"]["stage1"]
    )

    s = record["summary"]
    for m in ("fused_top5", "union_top5_no_model", "rerank_bge-reranker-base"):
        assert s[m]["R@5"] == pytest.approx(baseline["summary"][m]["R@5"])
    for m in ("multihop_no_model", "multihop_bge"):
        assert s[m]["mean_passages"] == 5.0
        n = record["questions_by_type"]
        weighted = (
            s[m]["R@5_bridge"] * n["bridge"] + s[m]["R@5_comparison"] * n["comparison"]
        ) / 100
        assert weighted == pytest.approx(s[m]["R@5"])
        assert record["ceiling_all_relevant_in_pool"][m] >= s[m]["all_relevant@5"]
        assert record["latency"][m]["total"]["ms_mean"] > 0
        # Every multi-hop top 5 starts with `keep` fused passages.
        keep = record["configs"][m]["keep"]
        for q, top in record["top5"][m].items():
            assert top[:keep] == baseline["top5"]["fused_top5"][q][:keep]
    for pairs in record["significance"].values():
        assert [(c["a"], c["b"]) for c in pairs] == [tuple(p) for p in eval_multihop.PLANNED]
    assert len(record["comparison_check"]) == 4
    assert all(c["questions"] == 19 for c in record["comparison_check"])


def test_joint_rank_best_score_over_pools() -> None:
    from multihop import joint_rank

    single = (["a", "b", "c"], {"a": 3.0, "b": 1.0, "c": 0.0})
    hop2 = (["c", "d"], {"c": 5.0, "d": 1.0})
    # c's best score is 5 (from hop 2); b and d tie at 1.0, b first (single-hop pool first).
    assert joint_rank([single, hop2], 5) == ["c", "a", "b", "d"]
    assert joint_rank([single, hop2], 2) == ["c", "a"]
    assert joint_rank([single], 5) == ["a", "b", "c"]
    assert joint_rank([], 5) == []


MULTIHOP_RESULT = next(
    iter(sorted((REPO_ROOT / "results" / "multihop").glob(f"{TEST}-2*.json"))), None
)


@pytest.mark.skipif(
    not all((EMBEDDINGS / n / "meta.json").exists() for n in (TEST, DEV))
    or RERANK_RESULT is None
    or MULTIHOP_RESULT is None,
    reason="needs both subsets' embeddings, a rerank result and a multi-hop result",
)
def test_eval_multihop_joint_end_to_end(tmp_path: Path) -> None:
    import eval_multihop_joint

    reranker = Reranker("bge-reranker-base", model=OverlapCrossEncoder())
    args = [
        "--rerank-result",
        str(RERANK_RESULT),
        "--multihop-result",
        str(MULTIHOP_RESULT),
        "--out-dir",
        str(tmp_path),
    ]
    assert eval_multihop_joint.main(args, reranker=reranker, encode=HashEncoder()) == 0
    record = json.loads(next(tmp_path.glob("*-joint-*.json")).read_text())
    assert len(record["tuning"]) == 8
    best = max(r["R@5"] for r in record["tuning"])
    assert any(
        r["R@5"] == best and all(r[k] == v for k, v in record["config"].items())
        for r in record["tuning"]
    )
    s = record["summary"]
    assert set(s) == {
        "fused_top5",
        "union_top5_no_model",
        "rerank_bge-reranker-base",
        "multihop_no_model",
        "multihop_bge",
        "joint_bge",
    }
    assert s["joint_bge"]["mean_passages"] == 5.0
    assert record["ceiling_all_relevant_in_pool"] >= s["joint_bge"]["all_relevant@5"]
    for pairs in record["significance"].values():
        assert [(c["a"], c["b"]) for c in pairs] == [tuple(p) for p in eval_multihop_joint.PLANNED]
    assert [c["baseline"] for c in record["comparison_check"]] == [
        "fused_top5",
        "rerank_bge-reranker-base",
    ]
