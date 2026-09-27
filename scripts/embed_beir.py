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
import os
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
    parser.add_argument(
        "--shard-size",
        type=int,
        default=200_000,
        help="corpus rows encoded before flushing to disk. Each shard is appended to corpus.fbin "
        "with a progress sidecar, so a run killed at a session limit resumes from the last "
        "completed shard instead of starting over. Smaller shards bound peak encode RAM "
        "(shard_size x dim x 4 bytes) and make resume finer-grained. <=0 means one shard.",
    )
    args = parser.parse_args(argv)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    spec = MODELS[args.model]
    data = REPO_ROOT / "data" / "beir" / args.dataset
    corpus_path = data / "corpus.jsonl"
    queries = load_jsonl(data / "queries.jsonl")
    out = REPO_ROOT / "data" / "embeddings" / args.dataset / args.model
    out.mkdir(parents=True, exist_ok=True)

    # One cheap streaming pass for the row ids and the count (needed before writing the header and
    # for corpus_ids.json); it does not hold the passage text in memory.
    doc_ids = [json.loads(line)["_id"] for line in corpus_path.open()]
    total = len(doc_ids)

    model = SentenceTransformer(spec.hf_name, revision=spec.revision, device=args.device)
    dim = int(model.get_sentence_embedding_dimension())
    itemsize, header_bytes = 4, 8  # float32 body; two little-endian uint32 for (rows, dim)
    shard_size = args.shard_size if args.shard_size > 0 else total
    corpus_fbin = out / "corpus.fbin"
    partial = out / "corpus.fbin.partial"
    progress_path = out / "corpus.progress.json"
    signature = {
        "model": spec.hf_name,
        "revision": spec.revision,
        "dim": dim,
        "total": total,
        "normalize": spec.normalize,
        "doc_prefix": spec.doc_prefix,
    }

    print(
        f"corpus.fbin ~{(header_bytes + total * dim * itemsize) / 2**30:.2f} GiB; "
        f"peak encode RAM ~{shard_size * dim * itemsize / 2**30:.2f} GiB "
        f"(shard {shard_size} x dim {dim}). Exact dense retrieval later loads all {total:,} "
        f"vectors (~{total * dim * itemsize / 2**30:.2f} GiB) into RAM."
    )

    # Resume: a completed corpus.fbin is trusted as-is; otherwise a matching .partial resumes from
    # the rows already on disk. The file length is the source of truth for how many rows are done.
    rows_done = 0
    if corpus_fbin.exists():
        rows_done = total
        print(f"corpus.fbin already complete ({total:,} rows); skipping corpus embedding")
    elif partial.exists() and progress_path.exists():
        prog = json.loads(progress_path.read_text())
        if prog.get("signature") == signature and partial.stat().st_size >= header_bytes:
            rows_done = min(prog.get("rows_done", 0), (partial.stat().st_size - header_bytes)
                            // (dim * itemsize), total)  # fmt: skip
            print(f"resuming corpus embedding from row {rows_done:,}/{total:,}")
        else:
            print("progress signature changed; restarting corpus embedding", file=sys.stderr)
            partial.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)

    seconds = 0.0
    if rows_done < total:
        with partial.open("r+b" if partial.exists() else "wb") as f, corpus_path.open() as cf:
            if partial.stat().st_size < header_bytes:
                np.array([total, dim], dtype="<u4").tofile(f)
            f.seek(header_bytes + rows_done * dim * itemsize)
            f.truncate()  # drop any half-written tail past the last acknowledged shard
            for _ in range(rows_done):  # skip passages already embedded
                cf.readline()
            batch: list[dict] = []
            idx = rows_done

            def flush(batch: list[dict], idx: int) -> int:
                nonlocal seconds
                if not batch:
                    return idx
                t0 = time.perf_counter()
                emb = model.encode(
                    format_docs(spec, batch),
                    batch_size=args.batch_size,
                    normalize_embeddings=spec.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                ).astype(np.float32)
                seconds += time.perf_counter() - t0
                norms = np.linalg.norm(emb, axis=1)
                if spec.normalize and not np.allclose(norms, 1.0, atol=1e-4):
                    raise RuntimeError(f"corpus embeddings not L2-normalized (min {norms.min()})")
                np.ascontiguousarray(emb, dtype="<f4").tofile(f)
                f.flush()
                os.fsync(f.fileno())
                idx += len(emb)
                progress_path.write_text(json.dumps({"signature": signature, "rows_done": idx}))
                print(f"  corpus rows {idx:,}/{total:,} embedded")
                return idx

            for line in cf:
                batch.append(json.loads(line))
                if len(batch) >= shard_size:
                    idx = flush(batch, idx)
                    batch = []
            idx = flush(batch, idx)
        if idx != total:
            raise RuntimeError(f"embedded {idx} corpus rows, expected {total}")

    # Finalize: rename the completed .partial into place. This also covers the rare case where a
    # prior run embedded every row but was killed before the rename (rows_done == total above, so
    # the write block is skipped), so corpus.fbin is always created once the partial is complete.
    if not corpus_fbin.exists() and partial.exists():
        partial.rename(corpus_fbin)
        progress_path.unlink(missing_ok=True)

    # Queries are small: (re)compute only if missing.
    if not (out / "queries.fbin").exists():
        query_emb = model.encode(
            format_queries(spec, [q["text"] for q in queries]),
            batch_size=args.batch_size,
            normalize_embeddings=spec.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        norms = np.linalg.norm(query_emb, axis=1)
        if spec.normalize and not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(f"query embeddings not L2-normalized (min norm {norms.min()})")
        write_bin(out / "queries.fbin", np.ascontiguousarray(query_emb))
    (out / "corpus_ids.json").write_text(json.dumps(doc_ids))
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
        "dim": dim,
        "max_seq_length": model.max_seq_length,
        "pooling": str(model[1]) if len(model) > 1 else "unknown",
        "device": args.device,
        "batch_size": args.batch_size,
        "shard_size": shard_size,
        "num_shards": (total + shard_size - 1) // shard_size,
        "num_docs": total,
        "num_queries": len(queries),
        "seconds": seconds,
        "seconds_note": "corpus encode time for rows embedded in this run (0 if resumed complete)",
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
    print(f"wrote {out} ({total} docs, {len(queries)} queries, dim {dim}, "
          f"{seconds:.0f} s this run on {args.device})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
