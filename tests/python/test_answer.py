"""Cited answer generation: request shape, response parsing, caching (no network).

The live test at the bottom runs only when ANTHROPIC_API_KEY is available (env or .env).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from answer import (
    MODEL,
    SYSTEM_PROMPT,
    AnswerGenerator,
    Passage,
    build_request,
    load_env,
    parse_response,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

PASSAGES = [
    Passage("d1", "Eiffel Tower", ["The Eiffel Tower is in Paris.", "", " It opened in 1889."]),
    Passage("d2", "Paris", ["Paris is the capital of France.", "It is on the Seine."]),
]


def fake_response(blocks: list[dict[str, Any]], stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": blocks,
        "stop_reason": stop,
        "usage": {"input_tokens": 1200, "output_tokens": 80},
    }


def cite(doc: int, start: int, end: int, text: str = "") -> dict[str, Any]:
    return {
        "type": "content_block_location",
        "cited_text": text,
        "document_index": doc,
        "document_title": PASSAGES[doc].title,
        "start_block_index": start,
        "end_block_index": end,
    }


def test_request_shape() -> None:
    request = build_request("When did the tower open?", PASSAGES)
    assert request["model"] == "claude-haiku-4-5"
    assert request["temperature"] == 0.0
    assert request["system"] == SYSTEM_PROMPT
    assert "output_config" not in request  # incompatible with citations
    content = request["messages"][0]["content"]
    doc = content[0]
    assert doc["type"] == "document" and doc["citations"] == {"enabled": True}
    assert doc["source"]["type"] == "content" and doc["title"] == "Eiffel Tower"
    # Empty sentences are not sent as blocks (they would shift nothing: see parsing test).
    assert [b["text"] for b in doc["source"]["content"]] == [
        "The Eiffel Tower is in Paris.",
        " It opened in 1889.",
    ]
    assert content[-1] == {"type": "text", "text": "Question: When did the tower open?"}


def test_parse_maps_blocks_back_to_sentence_ids() -> None:
    response = fake_response(
        [
            {"type": "text", "text": "The tower opened in 1889", "citations": [cite(0, 1, 2)]},
            {"type": "text", "text": ", in Paris", "citations": [cite(1, 0, 1), cite(0, 0, 1)]},
            {"type": "text", "text": ".\nAnswer: 1889"},
        ]
    )
    answer = parse_response(response, "q", PASSAGES, digest="h", from_cache=False)
    assert answer.ok and answer.short_answer == "1889" and not answer.abstained
    # Block 1 of passage 0 is sentence 2 (sentence 1 was empty and skipped).
    assert answer.cited_sentences == {("Eiffel Tower", 2), ("Paris", 0), ("Eiffel Tower", 0)}
    assert all("Answer:" not in s.text for s in answer.statements)
    assert answer.input_tokens == 1200 and answer.cost_usd == pytest.approx(1600 / 1e6)


def test_parse_abstention_and_failures() -> None:
    abstain = parse_response(
        fake_response([{"type": "text", "text": "The documents do not say.\nAnswer: unknown"}]),
        "q",
        PASSAGES,
        digest="h",
        from_cache=False,
    )
    assert abstain.abstained and abstain.ok and not abstain.cited_sentences
    truncated = parse_response(
        fake_response([{"type": "text", "text": "The tower"}], stop="max_tokens"),
        "q",
        PASSAGES,
        digest="h",
        from_cache=False,
    )
    assert not truncated.ok and truncated.short_answer == ""
    # Citations pointing outside the passage list are ignored rather than crashing.
    bad = parse_response(
        fake_response(
            [
                {"type": "text", "text": "x", "citations": [cite(0, 0, 1) | {"document_index": 7}]},
                {"type": "text", "text": "\nAnswer: x"},
            ]
        ),
        "q",
        PASSAGES,
        digest="h",
        from_cache=False,
    )
    assert bad.ok and not bad.cited_sentences


class FakeMessage:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return self._data


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append(kwargs)
        return FakeMessage(
            fake_response(
                [
                    {"type": "text", "text": "Opened 1889", "citations": [cite(0, 1, 2)]},
                    {"type": "text", "text": "\nAnswer: 1889"},
                ]
            )
        )


def test_generator_caches_by_request(tmp_path: Path) -> None:
    client = FakeClient()
    generator = AnswerGenerator(client, cache_dir=tmp_path)
    first = generator.answer("When did it open?", PASSAGES)
    second = generator.answer("When did it open?", PASSAGES)
    assert len(client.calls) == 1  # second answer came from the cache
    assert not first.from_cache and second.from_cache
    assert first.short_answer == second.short_answer == "1889"
    assert first.cited_sentences == second.cited_sentences
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["request"]["model"] == MODEL and cached["response"]["id"] == "msg_test"
    generator.answer("A different question?", PASSAGES)
    assert len(client.calls) == 2


def has_api_key() -> bool:
    load_env(REPO_ROOT)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(not has_api_key(), reason="needs ANTHROPIC_API_KEY (env or .env)")
def test_live_cited_answer(tmp_path: Path) -> None:
    answer = AnswerGenerator(cache_dir=tmp_path).answer("When did the Eiffel Tower open?", PASSAGES)
    assert answer.ok, answer.text
    assert "1889" in answer.short_answer
    assert ("Eiffel Tower", 2) in answer.cited_sentences


# --- HotpotQA evaluation end to end with a scripted generator ----------

HOTPOT = REPO_ROOT / "data" / "hotpotqa" / "subset-n100-seed0.json"


class OracleGenerator:
    """Answers with the gold answer and cites exactly the gold supporting facts it can see."""

    def __init__(self, gold: dict[str, dict]) -> None:
        self.gold = gold

    def answer(self, question: str, passages: list[Passage]) -> Any:
        q = self.gold[question]
        facts = {tuple(f) for f in q["supporting_facts"]}
        blocks = []
        for pi, p in enumerate(passages):
            kept = [i for i, s in enumerate(p.sentences) if s.strip()]
            for b, sid in enumerate(kept):
                if (p.title, sid) in facts:
                    c = cite(0, b, b + 1) | {
                        "document_index": pi,
                        "document_title": p.title,
                        "cited_text": p.sentences[sid],
                    }
                    blocks.append({"type": "text", "text": "Because", "citations": [c]})
        blocks.append({"type": "text", "text": f"\nAnswer: {q['answer']}"})
        return parse_response(
            fake_response(blocks), question, passages, digest="x", from_cache=False
        )


@pytest.mark.skipif(not HOTPOT.exists(), reason="run scripts/prepare_hotpotqa.py")
def test_hotpot_eval_scores_an_oracle_perfectly(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    import eval_hotpotqa

    questions = json.loads(HOTPOT.read_text())[:8]
    data = tmp_path / "data"
    data.mkdir()
    (data / "tiny.json").write_text(json.dumps(questions))
    (data / "tiny.meta.json").write_text(json.dumps({"n": len(questions)}))
    gen = OracleGenerator({q["question"]: q for q in questions})
    status = eval_hotpotqa.main(
        ["--subset", "tiny", "--data-dir", str(data), "--out-dir", str(tmp_path / "out")], gen
    )
    assert status == 0
    (result,) = (tmp_path / "out").glob("hotpotqa-tiny-*[0-9].json")
    record = json.loads(result.read_text())
    distractor = record["results"]["distractor"]["summary"]
    # All gold paragraphs are present in the distractor setting: every official metric is 1.
    for metric in ("em", "f1", "sp_em", "sp_f1", "joint_em", "joint_f1"):
        assert distractor[metric] == pytest.approx(1.0), metric
    assert distractor["citation_coverage"] == 1.0 and distractor["abstention_rate"] == 0.0
    # Retrieved setting: supporting-fact recall is capped by what retrieval found.
    retrieved = record["results"]["retrieved"]["summary"]
    assert retrieved["em"] == 1.0 and retrieved["sp_prec"] == pytest.approx(1.0)
    assert retrieved["sp_recall"] <= 1.0
    assert set(record["significance"]) == {"f1", "sp_f1", "joint_f1"}
    preds = json.loads(next((tmp_path / "out").glob("*.distractor.pred.json")).read_text())
    assert set(preds) == {"answer", "sp"} and len(preds["answer"]) == 8
