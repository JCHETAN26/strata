"""BEIR-setting HotpotQA: subset building, cost estimation, and the evaluation script."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from answer import (
    MAX_TOKENS,
    PER_REQUEST_TOKENS,
    TOKENS_PER_CHAR,
    Passage,
    build_request,
    cost_usd,
    estimate_input_tokens,
    parse_response,
    request_chars,
)
from prepare_hotpotqa_beir import keep_background, normalized, split_sentences

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET = REPO_ROOT / "data" / "beir" / "hotpotqa-subset-n100-seed0-bg20000"
EMBEDDINGS = REPO_ROOT / "data" / "embeddings" / SUBSET.name / "meta.json"


def test_split_sentences_is_lossless() -> None:
    text = 'Paris is big. It has 2.1M people! "Quoted" start? (Aside) yes. e.g. this stays.'
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert pieces[0] == "Paris is big." and pieces[1] == " It has 2.1M people!"
    assert split_sentences("") == [] and split_sentences("One.") == ["One."]


def test_background_sampling_is_deterministic_and_uniform() -> None:
    ids = [str(i) for i in range(200_000)]
    kept = [i for i in ids if keep_background(i, 0, 0.01)]
    assert kept == [i for i in ids if keep_background(i, 0, 0.01)]
    assert 1700 < len(kept) < 2300  # ~1%
    assert kept != [i for i in ids if keep_background(i, 1, 0.01)]  # seed matters


def test_normalization_ignores_whitespace_and_nfkc() -> None:
    assert normalized("Scanian (\xa0\xa0 ) is") == normalized("Scanian (   ) is")


# --- Cost estimation ----------


def test_offline_estimate_and_pricing() -> None:
    request = build_request("q?", [Passage("d", "T", ["One sentence.", " Two."])])
    tokens, method = estimate_input_tokens(request)
    assert method == "estimate"
    assert tokens == round(PER_REQUEST_TOKENS + TOKENS_PER_CHAR * request_chars(request))
    assert cost_usd(1_000_000, 0) == pytest.approx(1.0)
    assert cost_usd(0, 1_000_000) == pytest.approx(5.0)


def test_exact_count_uses_count_tokens_without_sampling_params() -> None:
    calls: list[dict] = []

    class Counter:
        def __init__(self) -> None:
            self.messages = self

        def count_tokens(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(input_tokens=321)

    request = build_request("q?", [Passage("d", "T", ["One."])])
    assert estimate_input_tokens(request, Counter()) == (321, "count_tokens")
    assert set(calls[0]) == {"model", "system", "messages"}  # count_tokens rejects max_tokens etc.


# --- End to end with a scripted generator ----------


class OracleGenerator:
    """Answers correctly and cites the gold supporting facts present in its passages."""

    cache_dir = None
    client = None

    def __init__(
        self, by_question: dict[str, dict], stop_reasons: dict[str, str] | None = None
    ) -> None:
        self.by_question = by_question
        self.stop_reasons = stop_reasons or {}
        self.calls = 0

    def is_cached(self, question: str, passages: list[Passage]) -> bool:
        return False

    def answer(self, question: str, passages: list[Passage]) -> Any:
        self.calls += 1
        gold = self.by_question[question]
        facts = {tuple(f) for f in gold["supporting_facts"]}
        blocks = []
        for pi, p in enumerate(passages):
            kept = [i for i, s in enumerate(p.sentences) if s.strip()]
            for b, sid in enumerate(kept):
                if (p.title, sid) in facts:
                    cite = {
                        "type": "content_block_location",
                        "cited_text": p.sentences[sid],
                        "document_index": pi,
                        "document_title": p.title,
                        "start_block_index": b,
                        "end_block_index": b + 1,
                    }
                    blocks.append({"type": "text", "text": "Because", "citations": [cite]})
        blocks.append({"type": "text", "text": f"\nAnswer: {gold['answer']}"})
        response = {
            "model": "fake",
            "content": blocks,
            "stop_reason": self.stop_reasons.get(question, "end_turn"),
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        return parse_response(response, question, passages, digest="x", from_cache=False)


@pytest.mark.skipif(not EMBEDDINGS.exists(), reason="build the subset and run the estimate once")
def test_eval_estimates_by_default_and_scores_an_oracle(tmp_path: Path) -> None:
    import eval_hotpotqa_beir

    queries = [json.loads(line) for line in (SUBSET / "queries.jsonl").open()]
    answers = {
        json.loads(line)["_id"]: json.loads(line) for line in (SUBSET / "answers.jsonl").open()
    }
    oracle = OracleGenerator({q["text"]: answers[q["_id"]] for q in queries})
    common = ["--subset", SUBSET.name, "--out-dir", str(tmp_path)]

    assert eval_hotpotqa_beir.main(common, oracle) == 0  # estimate only
    assert oracle.calls == 0 and not list(tmp_path.glob("*.json"))
    assert eval_hotpotqa_beir.main([*common, "--run", "--max-cost-usd", "0.01"], oracle) == 3
    assert oracle.calls == 0  # refused: worst case above the cap

    assert eval_hotpotqa_beir.main([*common, "--run", "--max-cost-usd", "100"], oracle) == 0
    (result,) = tmp_path.glob("*.json")
    record = json.loads(result.read_text())
    gold = record["results"]["gold"]["summary"]
    for metric in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1"):
        assert gold[metric] == pytest.approx(1.0), metric
    # Retrieved: every citation the oracle makes is a gold fact, but questions whose gold
    # passages were not retrieved have nothing to cite, and the official metric scores an empty
    # prediction as precision 0. So check citations directly, and that recall is capped.
    retrieved = record["results"]["retrieved"]
    for r in retrieved["records"]:
        facts = {tuple(f) for f in answers[r["id"]]["supporting_facts"]}
        assert {tuple(c) for c in r["cited_sentences"]} <= facts
    assert retrieved["summary"]["sp_recall"] < 1.0  # both gold in top 5 for < 100% of queries
    assert record["cost_estimate"]["gold"]["worst_case_cost_usd"] > 0
    assert record["dataset"]["comparable_to_full_beir"] is False
    assert MAX_TOKENS == 512
    assert record["results"]["gold"]["stop_reasons"] == {"end_turn": 100}
    assert record["results"]["gold"]["flagged"] == []
    per_request = record["cost_estimate"]["gold"]["per_request"]
    assert len(per_request) == 100 and all(r["chars"] > 0 for r in per_request.values())


@pytest.mark.skipif(not EMBEDDINGS.exists(), reason="build the subset and run the estimate once")
def test_truncated_answers_are_flagged_not_scored(tmp_path: Path) -> None:
    import eval_hotpotqa_beir

    queries = [json.loads(line) for line in (SUBSET / "queries.jsonl").open()]
    answers = {
        json.loads(line)["_id"]: json.loads(line) for line in (SUBSET / "answers.jsonl").open()
    }
    truncated = queries[0]
    oracle = OracleGenerator(
        {q["text"]: answers[q["_id"]] for q in queries}, {truncated["text"]: "max_tokens"}
    )
    args = ["--subset", SUBSET.name, "--out-dir", str(tmp_path), "--run", "--max-cost-usd", "100"]
    assert eval_hotpotqa_beir.main(args, oracle) == 0
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    gold = record["results"]["gold"]
    assert gold["flagged"] == [
        {"id": truncated["_id"], "stop_reason": "max_tokens", "output_tokens": 10}
    ]
    assert gold["stop_reasons"] == {"end_turn": 99, "max_tokens": 1}
    assert gold["summary"]["questions"] == 99 and gold["summary"]["em"] == pytest.approx(1.0)
    assert gold["summary_flagged_as_zero"]["em"] == pytest.approx(0.99)
    assert all(c["queries"] == 99 for c in record["significance"]["f1"])  # paired on complete ones
