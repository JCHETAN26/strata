"""Embed a BEIR dataset with a pinned sentence-embedding model.

    uv sync --group embed
    uv run python scripts/embed_beir.py --dataset scifact --model bge-small-en-v1.5

Writes data/embeddings/<dataset>/<model>/:
    corpus.fbin, queries.fbin        L2-normalized float32 embeddings (.fbin format)
    corpus_ids.json, query_ids.json  row i -> BEIR document / query id
    meta.json                        model, pinned revision, input formatting, normalization,
                                     device, library versions, dataset checksum

Every model's Hugging Face revision is pinned in MODELS below, so runs on different machines load
identical weights (embeddings can still differ slightly across devices: CPU vs CUDA vs MPS
arithmetic; the device is recorded). Each model gets the input formatting its authors require:
bge adds a retrieval instruction to queries only, e5 prefixes "query: " / "passage: ".
Documents are formatted as title + " " + text (BEIR's convention for dense models).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_datasets import write_bin  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    hf_name: str
    revision: str  # Hugging Face commit hash: pins the exact weights and config
    query_prefix: str
    doc_prefix: str
    normalize: bool
    source: str  # where the formatting rules come from


MODELS: dict[str, ModelSpec] = {
    "bge-small-en-v1.5": ModelSpec(
        hf_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        # Model card: add this instruction to retrieval queries; never to passages.
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        normalize=True,
        source="https://huggingface.co/BAAI/bge-small-en-v1.5 (model card, query instruction)",
    ),
    "e5-small-v2": ModelSpec(
        hf_name="intfloat/e5-small-v2",
        revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        # Model card: every input starts with "query: " or "passage: ".
        query_prefix="query: ",
        doc_prefix="passage: ",
        normalize=True,
        source="https://huggingface.co/intfloat/e5-small-v2 (model card usage example)",
    ),
}


def format_queries(spec: ModelSpec, queries: list[str]) -> list[str]:
    return [spec.query_prefix + q for q in queries]


def format_docs(spec: ModelSpec, docs: list[dict]) -> list[str]:
    return [spec.doc_prefix + f"{d.get('title', '')} {d['text']}".strip() for d in docs]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="bge-small-en-v1.5", choices=sorted(MODELS))
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    spec = MODELS[args.model]
    data = REPO_ROOT / "data" / "beir" / args.dataset
    corpus = load_jsonl(data / "corpus.jsonl")
    queries = load_jsonl(data / "queries.jsonl")
    out = REPO_ROOT / "data" / "embeddings" / args.dataset / args.model
    out.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(spec.hf_name, revision=spec.revision, device=args.device)
    encode = {
        "batch_size": args.batch_size,
        "normalize_embeddings": spec.normalize,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    start = time.perf_counter()
    doc_emb = model.encode(format_docs(spec, corpus), **encode).astype(np.float32)
    query_emb = model.encode(format_queries(spec, [q["text"] for q in queries]), **encode)
    query_emb = query_emb.astype(np.float32)
    seconds = time.perf_counter() - start

    for name, emb in (("corpus", doc_emb), ("queries", query_emb)):
        norms = np.linalg.norm(emb, axis=1)
        if spec.normalize and not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(f"{name} embeddings are not L2-normalized (norms {norms.min()}..)")
        write_bin(out / f"{name}.fbin", np.ascontiguousarray(emb))
    (out / "corpus_ids.json").write_text(json.dumps([d["_id"] for d in corpus]))
    (out / "query_ids.json").write_text(json.dumps([q["_id"] for q in queries]))

    beir_meta = json.loads((data / "meta.json").read_text())
    meta = {
        "dataset": args.dataset,
        "dataset_sha256": beir_meta.get("source_sha256"),
        "model_key": args.model,
        **asdict(spec),
        "doc_format": "doc_prefix + title + ' ' + text",
        "query_format": "query_prefix + text",
        "l2_normalized": spec.normalize,
        "dim": int(doc_emb.shape[1]),
        "max_seq_length": model.max_seq_length,
        "pooling": str(model[1]) if len(model) > 1 else "unknown",
        "device": args.device,
        "batch_size": args.batch_size,
        "num_docs": len(corpus),
        "num_queries": len(queries),
        "seconds": seconds,
        "versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "machine": platform.machine(),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({len(corpus)} docs, {len(queries)} queries, dim {meta['dim']}, "
          f"{seconds:.0f} s on {args.device})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
