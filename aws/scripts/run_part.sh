#!/usr/bin/env bash
# Runs ON the main instance (started by run_main.sh), one benchmark part per call:
#
#   run_part.sh ann       A: recall@10 vs QPS, Strata vs hnswlib vs FAISS, SIFT1M + GloVe-100
#   run_part.sh 10m       B: the same on BIGANN-10M, 16-thread builds for every library
#   run_part.sh threads   C: search QPS at 1-16 threads (+32 SMT), build time at 1-16 threads
#   run_part.sh filter    D: filtered-search crossover at 1M and 10M, 0.1-50% incl. 1-3% sweep
#   run_part.sh perf      E: perf stat + perf record of search, search_layer annotated
#
# The bench scripts default to build/release; setup links it to build/linux-release.
# Every result goes where the Mac results go (results/...), under names that do not overwrite
# them (an "_x86" / "_10m" suffix or a dataset the Mac never ran). Logs: ~/logs/<part>.log.
set -euo pipefail
part=${1:?usage: run_part.sh ann|10m|threads|filter|perf}
cd ~/strata
export VCPKG_ROOT=~/vcpkg
py=.venv/bin/python

# One logical CPU per physical core, lowest sibling first: pinning to these keeps two benchmark
# threads off the same core's SMT pair.
physical_cpus() {
  lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | sort -t, -k3,3n -k2,2n -k1,1n \
    | awk -F, '!seen[$3","$2]++ {print $1}' | head -n "$1" | paste -sd, -
}
one_core=$(physical_cpus 2 | cut -d, -f2)  # core 1, leaving core 0 for the OS and sshd
sixteen=$(physical_cpus 16)

case $part in
  ann)
    # Single-threaded build and search (ann-benchmarks style), pinned to one physical core. Five
    # runs per point. The 10 s pause replaces the Mac's 60 s cool-down (no thermal limit here).
    # Primary, like for like: everything at AVX2 (.venv-avx2 has hnswlib built without AVX-512;
    # FAISS is held to AVX2). Second: hnswlib and FAISS at AVX-512, reusing the Strata runs.
    taskset -c "$one_core" .venv-avx2/bin/python bench/run_hnsw_curves.py \
      --datasets sift1m glove100 --runs 5 --cooldown 10 --simd avx2 --name hnsw_vs_reference_x86
    taskset -c "$one_core" $py bench/run_hnsw_curves.py --datasets sift1m glove100 \
      --runs 5 --cooldown 10 --simd native --skip-strata --name hnsw_vs_reference_x86_avx512
    ;;

  10m)
    # A single-threaded build of 10M vectors takes hours per library; all three build with 16
    # threads (recorded in each record's build_params). Search is still one thread. Same two
    # comparisons as part A.
    taskset -c "$sixteen" .venv-avx2/bin/python bench/run_hnsw_curves.py --datasets bigann10m \
      --runs 3 --cooldown 10 --build-threads 16 --simd avx2 --name hnsw_vs_reference_10m
    taskset -c "$sixteen" $py bench/run_hnsw_curves.py --datasets bigann10m \
      --runs 3 --cooldown 10 --build-threads 16 --simd native --skip-strata \
      --name hnsw_vs_reference_10m_avx512
    ;;

  threads)
    # The script pins each run itself (N distinct physical cores; all hardware threads for 32).
    $py bench/run_search_scaling.py --dataset sift1m --threads 1,2,4,8,12,16,32 --rounds 3
    taskset -c "$sixteen" $py bench/run_hnsw_build_scaling.py --dataset sift1m \
      --threads 1,2,4,8,16 --rounds 3 --cooldown 10
    ;;

  filter)
    # Same selectivities at 1M and 10M, dense around the auto threshold (1.3%) to measure how
    # often auto falls back. Single-threaded search, pinned; the one-time index build uses 16.
    sel=0.001,0.005,0.01,0.013,0.015,0.02,0.025,0.03,0.05,0.1,0.5
    # 200 queries per point at 10M: very selective filters run at a few QPS there, and 500 would
    # stretch the part by hours for no gain in precision.
    taskset -c "$sixteen" $py bench/run_hnsw_filter_bench.py --dataset sift1m \
      --selectivities "$sel" --runs 3 --cooldown 10 --build-threads 16 --max-queries 500
    taskset -c "$sixteen" $py bench/run_hnsw_filter_bench.py --dataset bigann10m \
      --selectivities "$sel" --runs 3 --cooldown 10 --build-threads 16 --max-queries 200
    ;;

  perf)
    out=results/profiles/aws
    mkdir -p "$out"
    harness=build/linux-profile/bench/strata_search
    for ds in sift1m bigann10m; do
      snap=data/$ds/perf_index.snap
      args=(--data "data/$ds" --metric l2 --index hnsw --ef-search 80 --runs 10 --build-threads 16
            --snapshot "$snap")
      # Build (and save) the index outside the profile, then profile search only.
      [[ -f "$snap" ]] || "$harness" "${args[@]}" --runs 1 >/dev/null
      # Which hardware events this instance exposes (EC2 exposes a subset below full-socket
      # sizes); the record keeps whatever was available.
      perf stat -e cycles,instructions,branch-misses,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses \
        -o "$out/$ds-perf-stat.txt" -- taskset -c "$one_core" "$harness" "${args[@]}" >/dev/null
      event=cycles
      perf stat -e cycles -x, -- true 2>&1 | grep -q '<not supported>' && event=cpu-clock
      perf record -e "$event" -F 4999 --call-graph fp -o "/tmp/$ds.perf.data" -- \
        taskset -c "$one_core" "$harness" "${args[@]}" >/dev/null
      perf report -i "/tmp/$ds.perf.data" --stdio --no-children --percent-limit 0.5 \
        > "$out/$ds-perf-report.txt"
      perf report -i "/tmp/$ds.perf.data" --stdio --children --percent-limit 2 -g none \
        > "$out/$ds-perf-report-children.txt"
      # Annotate the five hottest symbols. search_layer is a template and may be inlined into
      # its caller, so its loop shows up under whichever symbol it was inlined into.
      perf report -i "/tmp/$ds.perf.data" --stdio --no-children -F sym --percent-limit 1 \
        | grep -v '^#' | sed -n 's/^ *\[\.\] //p' | head -n 5 \
        | while read -r sym; do
            safe=$(echo "$sym" | tr -c 'A-Za-z0-9_' _ | cut -c1-60)
            perf annotate -i "/tmp/$ds.perf.data" --stdio -s "$sym" > "$out/$ds-annotate-$safe.txt" 2>/dev/null || true
          done
      echo "{\"dataset\": \"$ds\", \"event\": \"$event\", \"commit\": \"$(git rev-parse HEAD)\", \"instance\": \"$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)" http://169.254.169.254/latest/meta-data/instance-type)\", \"cpu\": \"$(lscpu | sed -n 's/Model name: *//p')\", \"kernel\": \"$(uname -r)\", \"perf\": \"$(perf --version)\"}" \
        > "$out/$ds-meta.json"
    done
    ;;

  *)
    echo "unknown part $part"; exit 2 ;;
esac
echo "part $part done"
