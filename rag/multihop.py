"""Two-hop retrieval for bridge questions.

HotpotQA's bridge questions need a passage the question never names (the "second hop"). The first
hop is usually already retrieved: in the BEIR subset, 34 of 36 missing gold passages had their
first-hop partner in the top 5. So:

1. Hop 1: the normal hybrid retrieval ranking H1.
2. For each of the top m hop-1 passages p, a hop-2 query = question + expansion(p), where the
   expansion is p's title, title + first sentence, or full text. BM25 and dense retrieval each
   return their top `depth` passages for every hop-2 query.
3. Hop-2 candidates are ordered either without a model (round-robin over those lists) or by a
   cross-encoder that reads (hop-2 query, candidate): the expanded query carries the bridge the
   question lacks. A candidate reached from several p keeps its best score.
4. Final list: H1[:keep], then hop-2 candidates not already chosen, then the rest of H1.

Joint variant (joint_rank): instead of a keep rule, the single-hop pool (scored against the
question) and the hop-2 pools (scored against their hop-2 queries) are merged, and every
candidate competes on its best cross-encoder score.

All parameters are tuned on a dev split (bench/eval_multihop.py, bench/eval_multihop_joint.py);
this module only computes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rerank import union_pool

EXPANSIONS = ("title", "title_first_sentence", "full")


@dataclass(frozen=True)
class MultiHopConfig:
    m: int  # hop-1 passages to expand
    expansion: str  # one of EXPANSIONS
    depth: int  # per-retriever hop-2 depth
    rerank: bool  # order hop-2 candidates with the cross-encoder
    keep: int  # hop-1 passages kept at the top of the final list

    def to_dict(self) -> dict:
        return asdict(self)


def expansion_text(title: str, sentences: list[str], kind: str) -> str:
    if kind == "title":
        return title
    if kind == "title_first_sentence":
        first = next((s for s in sentences if s.strip()), "")
        return f"{title} {first.strip()}".strip()
    if kind == "full":
        return f"{title} {''.join(sentences).strip()}".strip()
    raise ValueError(f"unknown expansion {kind!r}")


def hop2_query(question: str, title: str, sentences: list[str], kind: str) -> str:
    return f"{question} {expansion_text(title, sentences, kind)}"


def hop2_candidates(
    per_p: list[tuple[list[str], list[str]]],
    depth: int,
    per_p_scores: list[dict[str, float]] | None = None,
) -> list[str]:
    """Hop-2 candidates from each expanded passage's (BM25 list, dense list), top `depth` of each.

    Without scores: the round-robin union over all 2m lists. With per_p_scores (cross-encoder
    scores for each p's candidates): sorted by the best score a candidate gets from any p whose
    pool it is in; ties keep round-robin order."""
    pools = [union_pool([bm25, dense], depth) for bm25, dense in per_p]
    merged = union_pool([lst for pair in per_p for lst in pair], depth)
    if per_p_scores is None:
        return merged
    best: dict[str, float] = {}
    for pool, scores in zip(pools, per_p_scores, strict=True):
        for doc in pool:
            best[doc] = max(best.get(doc, float("-inf")), scores[doc])
    return sorted(merged, key=lambda d: -best[d])


def combine(hop1: list[str], hop2: list[str], keep: int, k: int) -> list[str]:
    """H1[:keep], then unseen hop-2 candidates, then the rest of H1; the first k."""
    out: list[str] = []
    seen: set[str] = set()
    for doc in [*hop1[:keep], *hop2, *hop1[keep:]]:
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
        if len(out) == k:
            break
    return out


def joint_rank(pools: list[tuple[list[str], dict[str, float]]], k: int) -> list[str]:
    """Rank the union of several scored pools (each: candidate ids, cross-encoder scores against
    that pool's query) by each candidate's best score over the pools that contain it. Ties keep
    first-appearance order, with pools in the given order (single-hop first). Top k."""
    best: dict[str, float] = {}
    order: list[str] = []
    for pool, scores in pools:
        for doc in pool:
            if doc not in best:
                order.append(doc)
                best[doc] = scores[doc]
            else:
                best[doc] = max(best[doc], scores[doc])
    return sorted(order, key=lambda d: -best[d])[:k]
