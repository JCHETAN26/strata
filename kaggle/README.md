# Running Strata's GPU stages on Kaggle

The development laptop (MacBook Air M2) can't do this work, and the IdeaPad's disk is full
(2.6 GB free — not enough for CUDA PyTorch, let alone the ~7.5 GB HotpotQA embeddings). So the
GPU stages run in a **Kaggle notebook** instead. This directory is the runner:

| File | What it is |
|------|------------|
| `setup.sh` | Builds the C++ bindings (vcpkg + `uv sync`) and installs the GPU deps. Idempotent. |
| `run_stages.py` | Runs each stage, records the GPU/env in every result, resumes on restart. |
| `notebook.ipynb` | A ready-made Kaggle notebook with the cells below. |

**What runs, and what it costs**

| Stage | What | Anthropic API? |
|-------|------|----------------|
| `env` | Print + save the exact GPU and environment | no |
| `stage2` | Full-corpus BEIR HotpotQA (5.2M passages): BM25 / dense / hybrid retrieval vs published BEIR references | no |
| `stage3` | SciFact cross-encoder reranking on the GPU, tuned on the train split | no |
| `stage4` | GPU reranking latency on the HotpotQA subset, printed next to the committed CPU numbers | no |
| `stage2b` | *Optional:* HotpotQA answer generation on the BEIR **subset** (the split with gold answers; not the full corpus) | **yes — off by default** |

Stages 2–4 never call the Anthropic API. Only `stage2b` does, and it estimates the cost and
refuses to spend unless you pass `--enable-api` and the worst case is under `--max-cost-usd`.

---

## 1. One-time account setup

1. A Kaggle account with **phone verification** (required to enable GPU + internet in notebooks).
2. A **GitHub personal access token** with read access to `JCHETAN26/strata` (a fine-grained
   token with *Contents: read* on that repo is enough; a classic `repo`-scoped token also works).

## 2. Create the notebook

1. On kaggle.com: **Create → New Notebook**.
2. **Settings (right sidebar):**
   - **Accelerator:** pick a **GPU**. **GPU T4 x2** or **GPU P100** both work; the runner uses one
     GPU. Prefer whichever the session offers with the **most RAM** — Stage 2's dense retrieval
     over 5.2M passages needs about **15 GiB of RAM** (see *Stage 2 memory* below).
   - **Internet:** **On** (needed to clone the repo, install PyTorch, and download the datasets).
   - **Persistence:** set **Variables and Files** to persist, so `/kaggle/working` survives a
     restart and Stage 2 resumes instead of re-embedding.
3. **Add the token as a secret:** **Add-ons → Secrets → Add a new secret**, label
   **`GITHUB_TOKEN`**, value = your token. Attach it to the notebook.

## 3. Notebook cells

Either open `notebook.ipynb` from this directory, or paste these cells in order.

**Cell 1 — clone (the token is read from the secret, never printed, and scrubbed afterward):**
```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
os.chdir("/kaggle/working")
if not os.path.isdir("strata"):
    # `git clone` does not echo the URL, so the token never reaches the notebook output.
    subprocess.run(
        ["git", "clone", "--depth", "1",
         f"https://x-access-token:{token}@github.com/JCHETAN26/strata.git"],
        check=True, stdout=subprocess.DEVNULL,
    )
    # Remove the token from the stored remote so it never persists on disk.
    subprocess.run(["git", "-C", "strata", "remote", "set-url", "origin",
                    "https://github.com/JCHETAN26/strata.git"], check=True)
del token
print("cloned:", os.path.isdir("strata"))
```

**Cell 2 — build + install (≈ 5–12 min the first time; instant on a warm restart):**
```python
%cd /kaggle/working/strata
!bash kaggle/setup.sh
```

**Cell 3 — environment (records the exact GPU; also embedded in every result file):**
```python
!source /kaggle/working/strata-env.sh && uv run python kaggle/run_stages.py env
```

**Cell 4 — Stage 2 (full-corpus HotpotQA retrieval):**
```python
!source /kaggle/working/strata-env.sh && uv run python kaggle/run_stages.py stage2
```

**Cell 5 — Stage 3 (SciFact reranking on GPU):**
```python
!source /kaggle/working/strata-env.sh && uv run python kaggle/run_stages.py stage3
```

**Cell 6 — Stage 4 (GPU reranking latency vs CPU):**
```python
!source /kaggle/working/strata-env.sh && uv run python kaggle/run_stages.py stage4
```

**Cell 7 — Stage 2b (optional, spends money — leave commented until you decide):**
```python
# Estimate only (free): prints token counts and worst-case cost, calls nothing.
!source /kaggle/working/strata-env.sh && uv run python kaggle/run_stages.py stage2b
# To actually generate answers, uncomment and set your own cap. Requires ANTHROPIC_API_KEY
# (add it as a Kaggle secret and export it, or put it in strata/.env):
# !source /kaggle/working/strata-env.sh && ANTHROPIC_API_KEY=$YOUR_KEY \
#   uv run python kaggle/run_stages.py stage2b --enable-api --max-cost-usd 1.00
```

## 4. Stage 2 memory

Before embedding, `stage2` prints an estimate:

- corpus embeddings on disk (`corpus.fbin`): **~7.5 GiB** (5,233,329 × 384 × float32);
- peak RAM for exact dense retrieval: **~15 GiB** (the corpus vectors plus the index's own copy).

If that does **not** fit safely in the session's RAM, the runner **stops before embedding** and
tells you the options (attach a higher-RAM accelerator; or `--allow-large-memory` to try anyway;
or `--embed-only` to produce the embeddings now and run retrieval elsewhere). Embedding itself is
cheap on RAM (bounded by one shard) and is always safe to run.

## 5. Resuming after a session ends

Kaggle GPU sessions have a wall-clock limit. Embedding is **sharded and resumable**: each shard is
appended to `corpus.fbin` with a progress sidecar, so if a session ends mid-embed, just **rerun
Cell 2 then Cell 4** — `setup.sh` skips work already done, and `stage2` resumes from the last
completed shard (you'll see `resuming corpus embedding from row N`). A finished `corpus.fbin` is
detected and skipped entirely. This only works if **Persistence** is on (Step 2).

## 6. What to download

The only outputs meant to leave Kaggle are the **result JSONs** — embeddings and indexes stay on
Kaggle. Each stage copies its results into **`/kaggle/working/strata-results/`**. Download that
folder from the notebook's **Output** tab (or the Data panel). It contains:

- `environment.json` — the GPU/session details;
- `hotpotqa-bge-small-en-v1.5-*.json` — Stage 2 retrieval (with `baseline_checks` vs published);
- `scifact-*.json` — Stage 3 reranking;
- `hotpotqa-subset-*-*.json` + `rerank_latency_cpu_vs_gpu.json` — Stage 4 latency;
- and any `stage2b` answer-generation result.

Commit those into `results/` in the repo when you're back on a normal machine.

## 7. GPU not detected

If Cell 2/3 report `CUDA not available`:

- Confirm a **GPU** accelerator is attached (Settings → Accelerator) and the session was restarted
  after attaching it.
- The pinned PyTorch (`torch==2.14.0`, CUDA 13 build in `uv.lock`) needs a driver new enough for
  CUDA 13. If Kaggle's image ships an older driver, `torch.cuda.is_available()` will be `False`.
  Check the driver in Cell 3's output (`accelerator.gpus[].driver_version`). If it's too old,
  either use a newer Kaggle image (Settings → Environment → *Always use latest*) or install a
  PyTorch build matching Kaggle's CUDA before running the stages. The stages will otherwise fall
  back to CPU, which is correct but slow — the runner warns loudly rather than doing this silently.
