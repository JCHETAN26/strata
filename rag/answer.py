"""Cited answer generation with Claude (claude-haiku-4-5), grounded in retrieved passages.

Each passage is sent as a *custom content* document whose content blocks are its sentences, with
citations enabled. Claude's citations then come back as content_block_location ranges, i.e. exact
sentence indices in a known passage, so groundedness can be measured against sentence-level
evidence (HotpotQA supporting facts) instead of inferred from text.

Output: a brief explanation whose statements carry citations, and a final line
"Answer: <short answer>" (or "Answer: unknown") used for exact-match / F1 scoring. Citations and
structured outputs are incompatible in the API, so the short answer is a marked line, not JSON.

Reproducibility and cost: temperature 0, a fixed prompt version, and an on-disk cache keyed by
the SHA-256 of the full request, so re-running an evaluation costs nothing and returns the same
answers. The cache stores the raw API response alongside the parsed result.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODEL = "claude-haiku-4-5"
PROMPT_VERSION = "cited-answer-v1"
MAX_TOKENS = 1024
TEMPERATURE = 0.0
# Claude Haiku 4.5 list prices (USD per million tokens), for cost reporting.
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}

SYSTEM_PROMPT = """You answer questions using only the documents provided.

Write a brief explanation (one to three sentences) of how the documents answer the question, \
citing the sentences that support each claim. Use only information in the documents; if they \
do not contain the answer, say so.

End with a final line in exactly this form:
Answer: <the shortest answer: a name, number, date, short phrase, or yes/no>
If the documents do not contain the answer, the final line must be:
Answer: unknown"""

ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    sentences: list[str]


@dataclass(frozen=True)
class Citation:
    passage_index: int
    doc_id: str
    title: str
    sentence_ids: list[int]  # indices into Passage.sentences
    cited_text: str


@dataclass
class Statement:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class CitedAnswer:
    question: str
    short_answer: str
    abstained: bool
    statements: list[Statement]
    text: str
    stop_reason: str
    ok: bool  # stop_reason == "end_turn" and an "Answer:" line was found
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    request_hash: str
    from_cache: bool

    @property
    def cited_sentences(self) -> set[tuple[str, int]]:
        """(title, sentence index) pairs cited anywhere, HotpotQA supporting-fact format."""
        return {
            (c.title, s) for st in self.statements for c in st.citations for s in c.sentence_ids
        }

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
        ) / 1e6

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cited_sentences"] = sorted(self.cited_sentences)
        d["cost_usd"] = self.cost_usd
        return d


def _blocks(passage: Passage) -> tuple[list[dict[str, str]], list[int]]:
    """Non-empty sentences as content blocks, and each block's original sentence index."""
    blocks, index = [], []
    for i, sentence in enumerate(passage.sentences):
        if sentence.strip():
            blocks.append({"type": "text", "text": sentence})
            index.append(i)
    return blocks, index


def build_request(
    question: str,
    passages: list[Passage],
    *,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict[str, Any]:
    """Keyword arguments for client.messages.create()."""
    content: list[dict[str, Any]] = []
    for p in passages:
        blocks, _ = _blocks(p)
        if not blocks:
            blocks = [{"type": "text", "text": "(empty)"}]
        content.append(
            {
                "type": "document",
                "source": {"type": "content", "content": blocks},
                "title": p.title,
                "citations": {"enabled": True},
            }
        )
    content.append({"type": "text", "text": f"Question: {question}"})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
    }


def request_hash(request: dict[str, Any]) -> str:
    payload = {"prompt_version": PROMPT_VERSION, "request": request}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_response(
    response: dict[str, Any],
    question: str,
    passages: list[Passage],
    *,
    digest: str,
    from_cache: bool,
) -> CitedAnswer:
    """Turn a Messages API response (as a dict) into a CitedAnswer."""
    index_maps = [_blocks(p)[1] for p in passages]
    statements: list[Statement] = []
    for block in response.get("content", []):
        if block.get("type") != "text":
            continue
        citations = []
        for c in block.get("citations") or []:
            if c.get("type") != "content_block_location":
                continue
            pi = c["document_index"]
            if not 0 <= pi < len(passages):
                continue
            mapping = index_maps[pi]
            blocks = range(c["start_block_index"], min(c["end_block_index"], len(mapping)))
            citations.append(
                Citation(
                    passage_index=pi,
                    doc_id=passages[pi].doc_id,
                    title=passages[pi].title,
                    sentence_ids=[mapping[b] for b in blocks],
                    cited_text=c.get("cited_text", ""),
                )
            )
        statements.append(Statement(text=block.get("text", ""), citations=citations))

    text = "".join(s.text for s in statements)
    matches = ANSWER_LINE.findall(text)
    short = matches[-1].strip() if matches else ""
    # The "Answer:" line is bookkeeping, not a claim: drop it from the statements.
    if matches:
        for st in statements:
            st.text = ANSWER_LINE.sub("", st.text)
        statements = [st for st in statements if st.text.strip() or st.citations]
    usage = response.get("usage", {})
    stop = response.get("stop_reason", "")
    return CitedAnswer(
        question=question,
        short_answer=short,
        abstained=short.lower().rstrip(".") == "unknown",
        statements=statements,
        text=text,
        stop_reason=stop,
        ok=stop == "end_turn" and bool(matches),
        model=response.get("model", ""),
        prompt_version=PROMPT_VERSION,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        request_hash=digest,
        from_cache=from_cache,
    )


class AnswerGenerator:
    """Generates cited answers, caching raw responses on disk by request hash.

    client: an anthropic.Anthropic (or anything with a compatible messages.create); created on
    first use if None, reading credentials the SDK's usual way (ANTHROPIC_API_KEY, which
    load_env() can pull from the repo's .env). Thread safety: answer() may be called from
    several threads; cache files are written atomically.
    """

    def __init__(
        self, client: Any | None = None, *, cache_dir: Path | None = None, model: str = MODEL
    ) -> None:
        self._client = client
        self.model = model
        self.cache_dir = cache_dir

    def is_cached(self, question: str, passages: list[Passage]) -> bool:
        if self.cache_dir is None:
            return False
        digest = request_hash(build_request(question, passages, model=self.model))
        return (self.cache_dir / f"{digest}.json").exists()

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    def answer(self, question: str, passages: list[Passage]) -> CitedAnswer:
        request = build_request(question, passages, model=self.model)
        digest = request_hash(request)
        cached = self.cache_dir / f"{digest}.json" if self.cache_dir else None
        if cached is not None and cached.exists():
            response = json.loads(cached.read_text())["response"]
            return parse_response(response, question, passages, digest=digest, from_cache=True)
        message = self.client.messages.create(**request)
        response = message.to_dict() if hasattr(message, "to_dict") else dict(message)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": request, "response": response}))
            tmp.replace(cached)
        return parse_response(response, question, passages, digest=digest, from_cache=False)


# --- Cost estimation ----------

# Offline token estimate: English prose runs about 4 characters per token; 3.5 is used to lean
# high, and citation-enabled documents add chunk markup and a system-prompt addition (the API
# docs say "a slight increase in input tokens"), budgeted at 25%.
CHARS_PER_TOKEN = 3.5
CITATION_OVERHEAD = 1.25
EXPECTED_OUTPUT_TOKENS = 150  # a 1-3 sentence cited explanation plus the Answer line


def request_chars(request: dict[str, Any]) -> int:
    chars = len(request["system"])
    for block in request["messages"][0]["content"]:
        if block["type"] == "document":
            chars += len(block.get("title", ""))
            chars += sum(len(b["text"]) for b in block["source"]["content"])
        else:
            chars += len(block["text"])
    return chars


def estimate_input_tokens(request: dict[str, Any], client: Any | None = None) -> tuple[int, str]:
    """Input tokens for one request: exact via the free count_tokens endpoint when a client is
    given, else the offline estimate. Returns (tokens, "count_tokens" | "estimate")."""
    if client is not None:
        counted = client.messages.count_tokens(
            model=request["model"], system=request["system"], messages=request["messages"]
        )
        return counted.input_tokens, "count_tokens"
    return round(request_chars(request) / CHARS_PER_TOKEN * CITATION_OVERHEAD), "estimate"


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_PER_MTOK["input"] + output_tokens * PRICE_PER_MTOK["output"]) / 1e6


def load_env(repo_root: Path) -> None:
    """Load ANTHROPIC_API_KEY (and friends) from <repo>/.env if present. Never logs values."""
    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env", override=False)
