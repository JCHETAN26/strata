# AWS benchmark session: plan

Status: **prepared, not run.** Nothing has been created in AWS. Prices and quotas below were read
from the account on 2026-10-01 (AWS Pricing API, 7-day spot price history, Service Quotas); the
Terraform was validated and planned (no apply). The runbook is [`aws/README.md`](../aws/README.md).

The session produces the final x86 numbers for Phase 9. Every result comes from a script in
`bench/`, records the instance type, CPU, kernel, commit, and library versions, and is written
next to the Mac development results under names that do not overwrite them.

## What runs

Two stages, run one after the other (the account's vCPU quota is 32, see below).

### Main stage: one c7i.8xlarge (parts A–E)

| part | what | how | outputs |
|---|---|---|---|
| **A** | Recall@10 vs QPS, Strata vs hnswlib vs FAISS (HNSW) plus brute force, **SIFT1M** and **GloVe-100** | `run_hnsw_curves.py`: same M=16 and ef_construction=200 for all, ef_search 10–320, **5 runs per point**, single-threaded build and search pinned to one physical core | `results/hnsw/hnsw_vs_reference_x86.md`, plots `results/plots/hnsw_vs_reference_x86_*.png`, raw records `results/search/<dataset>/` |
| **B** | The same at **10M vectors** (BIGANN-10M) | Builds use 16 threads for every library (single-threaded 10M builds take hours each); search single-threaded; 3 runs per point | `results/hnsw/hnsw_vs_reference_10m.md` + plot |
| **C** | **Thread scaling 1–16** | Search: `run_search_scaling.py`, 1, 2, 4, 8, 12, 16 threads, each run pinned to that many *distinct physical cores*, plus 32 (all hardware threads) as the SMT point; one index loaded from a snapshot. Build: `run_hnsw_build_scaling.py` at 1, 2, 4, 8, 16 threads. 3 interleaved rounds each | `results/search_scaling/scaling_sift1m.md`, `results/hnsw_build/build_scaling_sift1m.md` |
| **D** | **Filtered search crossover at 1M and 10M**, including the **1–3% sweep** and auto's **fallback rate** | `run_hnsw_filter_bench.py` at 0.1, 0.5, 1, 1.3, 1.5, 2, 2.5, 3, 5, 10, 50% selectivity, random and cluster-correlated filters; derives the crossover, then measures auto at it (how often it pre-filters, how often the graph falls back) | `results/hnsw_filter/filter_sift1m.md`, `filter_bigann10m.md`, plots |
| **E** | **Linux `perf` profile of search** (search_layer) | `linux-profile` build (release + frame pointers); index from a snapshot, so only search is profiled; `perf stat` (cycles, IPC, cache and TLB misses), `perf record --call-graph fp`, `perf annotate` of the hottest symbols; SIFT1M and 10M | `results/profiles/aws/` |

Setup before part A: build and test (`linux-release`, including the AVX2-vs-scalar kernel tests),
build the server (`linux-server-release`, also the first x86 build of the Linux server presets)
and run its tests, build `linux-profile`, and prepare SIFT1M, GloVe-100, and BIGANN-10M. The 10M
base is the first 10M vectors of the BIGANN 1B file, fetched with an HTTP range request; the
published 10M ground truth is spot-checked against an exact search (`scripts/prepare_datasets.py`).

### Cluster stage: 4 shards + coordinator + client (part F)

| part | what | how | outputs |
|---|---|---|---|
| **F** | **Multi-machine sharding scaling** and **tail latency on separate machines** | SIFT1M over 1, 2, then 4 shard machines behind a coordinator, TLS + token on every hop; the load client on its own machine. Closed loop at 1, 8, 32, 64 in flight for capacity; open loop at 25, 50, 75% of capacity for p50/p90/p99/p99.9 (latency from each query's due time, so no coordinated omission; the client's own lateness is reported per row). 3 runs per point | `results/server/sharding_aws.md`, raw `results/server/sharding/aws/` |

The cluster stage uses the server binaries built on the main stage (copied back to the laptop,
then to each machine), so it needs no compiler and starts in minutes.

## Account constraints found

- **vCPU quota: 32** (on-demand and spot, standard families, in both us-east-1 and us-east-2).
  The main instance alone uses 32, so the cluster cannot run at the same time. `up.sh` checks
  the quota and refuses a stage that would exceed it. *Optional:* requesting 64 (free; approval
  can take a day) would allow the cluster to use c7i.2xlarge shards, which have more cores per shard.
- No existing instances, volumes, or non-default VPCs in either region, so the leftover check
  (`check_clean.sh`) has a clean baseline. One unrelated key pair exists (another project's).
- The GitHub repository is public, so the instances clone it at the pinned commit without
  credentials. (`buildplan.md` still lists "make the repo public" as open; it appears to be done.)

## Instances, region, market

| role | type | vCPU / physical cores | RAM | on-demand $/h | spot $/h (us-east-2, last 7 days) |
|---|---|---|---|---:|---:|
| main (A–E) | **c7i.8xlarge** (Sapphire Rapids, AVX2 + AVX-512) | 32 / 16 | 64 GiB | 1.428 | 0.339–0.460 |
| shard ×4 (F) | c7i.xlarge | 4 / 2 | 8 GiB | 0.1785 | not queried; ~0.05 by ratio |
| coordinator (F) | c7i.xlarge | 4 / 2 | 8 GiB | 0.1785 | as above |
| client (F) | c7i.2xlarge | 8 / 4 | 16 GiB | 0.357 | 0.091–0.120 |

- **Why c7i.8xlarge:** 16 physical cores, which is exactly the 1–16 scaling curve, with each point
  on distinct cores (SMT siblings idle) and the 32-thread SMT point as an extra. 64 GiB holds the
  10M run comfortably: about 7 GB per HNSW index plus about 10 GB peak while FAISS copies the base,
  one library at a time. Its AVX2 is the target ISA, and Intel matches most published hnswlib and
  FAISS numbers. Alternatives considered:
  - **c7a.8xlarge** (AMD Genoa, 32 physical cores, no SMT, $1.642/h): a cleaner scaling story,
    but 15% dearer.
  - **c7i.4xlarge** ($0.714/h, 8 cores, 32 GiB): too few cores for 1–16.
  - **m7i.8xlarge** (128 GiB, $1.613/h): memory not needed.
  - **c7i.metal-24xl** ($4.284/h): full PMU access, but only worth it if perf counters turn out
    to be missing (see risks).
- **Region us-east-2 (Ohio):** the same on-demand price as us-east-1, cheaper and steadier spot,
  and both types are offered in every AZ. One AZ (us-east-2a) for everything; the cluster is in a
  *cluster placement group*, so latency measures Strata, not cross-rack distance.
- **On-demand, not spot (recommended).** Spot would save about $14 of a ~$20 session. An
  interruption during the 10M builds or the filter sweep would lose up to an hour of work, plus
  setup again. It would also split one part's runs across two machines, which spoils the variance
  comparisons. `market = "spot"` is a one-variable switch if you prefer it.

## Cost estimate

On-demand prices above. Durations are estimates for this hardware, from the Mac runs scaled to
x86 and the measured sizes; the contingency covers slower-than-expected steps and debugging.

| item | hours | $ |
|---|---:|---:|
| Setup: toolchain, vcpkg (gRPC from source is the slow part), builds + tests, datasets | 1.0 | 1.43 |
| A: SIFT1M + GloVe-100, 3 libraries, single-threaded builds, 5 runs per point | 2.0 | 2.86 |
| B: BIGANN-10M, 3 libraries, 16-thread builds, 3 runs per point | 1.5 | 2.14 |
| C: search scaling (7 thread counts × 3 rounds) + build scaling (5 × 3 rounds) | 1.0 | 1.43 |
| D: filter crossover, 11 selectivities × 2 filter kinds, at 1M and 10M | 2.75 | 3.93 |
| E: perf at 1M and 10M | 0.5 | 0.71 |
| Main contingency (+30%) | 2.6 | 3.71 |
| **Main stage (c7i.8xlarge)** | **11.4** | **16.21** |
| F: cluster setup, three bulk loads (1M HNSW inserts per shard count), measurements, with contingency | 2.5 | 3.12 |
| EBS gp3 (150 GB main ~12 h; 6 × 30 GB cluster ~2.5 h) | | 0.24 |
| Public IPv4 addresses ($0.005/h each) | | 0.14 |
| Data out (results to the laptop, < 1 GB); in-AZ traffic between instances is free | | < 0.10 |
| **Total, on-demand** | | **≈ 20** |
| Total if both stages used spot (same hours) | | ≈ 6 |

Safety nets, all in the Terraform:
- **Auto-termination:** each instance schedules its own shutdown 14 h after boot, and shutdown
  means terminate. A forgotten teardown costs at most about $20 more, not days.
- **Tags:** everything is tagged `Project=strata-bench`, so `check_clean.sh` can find leftovers.
- **Teardown check:** `teardown.sh` ends with that check and fails if anything remains.

Suggested, but not created by these scripts: an AWS Budgets alert at $40 (free) before applying.

## Methodology notes

- **Variance:**
  - Parts A, B, and D run each point 3–5 times and report mean ± stdev.
  - Parts C and F interleave thread counts or shard counts across rounds, so drift affects all of
    them equally.
  - The coordinator benchmark (already done on the Mac) alternates modes within each round.
- **Pinning:**
  - Single-threaded runs are pinned to one physical core (core 1; core 0 is left for the OS).
  - Multi-threaded runs use N distinct physical cores.
  - The SMT point is labeled as such.
  - `hardware.topology` in every record states the core and thread counts.
- **Fair comparison:** the same M, ef_construction, and ef_search grid for all three libraries.
  hnswlib and FAISS QPS come from one batched Python call, so Python overhead is amortized; their
  per-query latencies include Python overhead and are not compared. On x86, hnswlib uses its
  AVX/SSE paths. This removes the Mac caveat, where hnswlib had no NEON path.
- **Recorded with every result:**
  - instance type, AZ, and AMI (IMDSv2);
  - CPU model, physical cores, SMT, L3 size, AVX-512 flag, kernel, and CPU governor;
  - the commit (each instance checks out the pushed `HEAD`; uncommitted changes are refused);
  - FAISS version and compile options, and the hnswlib version.

## Risks and decision points

1. **perf hardware counters.** Below full-socket sizes, EC2 exposes only a subset of PMU events.
   Part E checks for them and falls back to `cpu-clock` sampling, which still gives the call
   graph and annotated hot loops, but not cache-miss counts. If cache misses are wanted and
   missing, 30 minutes on c7i.metal-24xl would cost about $2.15 (a variable change and a rerun of
   part E). That is your call after seeing part E.
2. **hnswlib build flags.** If PyPI has no wheel for the platform, hnswlib compiles from source on
   the instance (build-essential is installed for that). Its setup chooses SIMD flags at build
   time. The record keeps its version; the setup log shows the compile line.
3. **The cluster stage's single-threaded ingest.** A shard inserts through `Collection::insert`
   one vector at a time (`--sync none` for the bulk load). One shard holding all 1M takes an
   estimated 10–15 minutes. That is acceptable for SIFT1M; a 10M sharded run would need parallel
   ingest first, so it is not in this plan.
4. **Durations are estimates.** The scripts resume where they stopped: finished parts and runs
   are skipped. A long part can be stopped and rerun without losing what finished.

## After the session

- Commit `results/` from AWS. Replace the development numbers in `README.md` and `docs/design.md`
  with the x86 tables, keeping the Mac numbers as development history.
- Fill the metrics tracker in `buildplan.md`.
- The Oracle ARM machine supplies the final NEON numbers and the first ARM build of the server
  presets.
