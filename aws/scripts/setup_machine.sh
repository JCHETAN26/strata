#!/usr/bin/env bash
# Runs ON an instance (Ubuntu 24.04, x86_64), started by run_main.sh / run_sharding.sh:
#
#   setup_machine.sh main   COMMIT   # toolchain, vcpkg, builds + tests, Python env, datasets
#   setup_machine.sh client COMMIT   # repo + Python env + SIFT1M, for the load client
#   setup_machine.sh node            # shards / coordinator: just a directory for the binaries
#
# The repo is cloned from GitHub at COMMIT (detached), so every result records a commit anyone can
# check out. Idempotent: rerunning skips finished steps.
set -euo pipefail

role=${1:?usage: setup_machine.sh main|client|node [COMMIT]}
commit=${2:-}
repo_url=https://github.com/JCHETAN26/strata.git
cd ~

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

apt_install() {
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq "$@"
}

clone_repo() {
  if [[ ! -d strata/.git ]]; then
    git clone -q "$repo_url" strata
  fi
  git -C strata fetch -q origin
  git -C strata checkout -q --detach "$commit"
  log "repo at $(git -C strata rev-parse HEAD)"
}

# Python env from the lockfile's bench group, without building the project's own extension
# (the benchmarks drive the C++ binaries; only hnswlib, FAISS, numpy, h5py, matplotlib are needed).
python_env() {
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  cd ~/strata
  uv venv -q --python 3.12 .venv
  uv export -q --frozen --only-group bench --no-emit-project --no-hashes -o /tmp/bench-req.txt
  uv pip install -q --python .venv/bin/python -r /tmp/bench-req.txt
  .venv/bin/python -c "import faiss, hnswlib; print('faiss', faiss.__version__, faiss.get_compile_options()); print('hnswlib', getattr(hnswlib, '__version__', '?'))"
  cd ~
}

case $role in
  node)
    mkdir -p ~/bin ~/certs
    sudo sysctl -q -w net.core.somaxconn=4096
    log "node ready"
    ;;

  client)
    [[ -n "$commit" ]] || { echo "client needs COMMIT"; exit 2; }
    apt_install build-essential git curl rsync tmux  # a compiler: hnswlib may build from source
    mkdir -p ~/bin ~/certs
    clone_repo
    python_env
    (cd ~/strata && .venv/bin/python scripts/prepare_datasets.py sift1m)
    log "client ready"
    ;;

  main)
    [[ -n "$commit" ]] || { echo "main needs COMMIT"; exit 2; }
    apt_install build-essential gcc-13 g++-13 cmake ninja-build git curl zip unzip tar pkg-config \
      autoconf automake libtool perl bison flex linux-libc-dev rsync tmux jq \
      linux-tools-common "linux-tools-$(uname -r)" linux-tools-aws util-linux
    # perf: allow unprivileged profiling and kernel symbols for this session.
    sudo sysctl -q -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
    clone_repo

    if [[ ! -x ~/vcpkg/vcpkg ]]; then
      git clone -q https://github.com/microsoft/vcpkg.git ~/vcpkg
      ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
    fi
    export VCPKG_ROOT=~/vcpkg
    grep -q VCPKG_ROOT ~/.profile || echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.profile

    cd ~/strata
    log "building linux-release (tests included)"
    cmake --preset linux-release >/dev/null
    cmake --build --preset linux-release
    ln -sfn linux-release build/release  # the bench scripts' default --build-dir
    log "testing on x86 (AVX2 kernels vs scalar included)"
    ctest --preset linux-release --output-on-failure
    log "building linux-server-release (gRPC from vcpkg: the slow step)"
    cmake --preset linux-server-release >/dev/null
    cmake --build --preset linux-server-release
    ctest --preset linux-server-release --output-on-failure
    log "building linux-profile (for perf)"
    cmake --preset linux-profile >/dev/null
    cmake --build --preset linux-profile --target strata_search
    # Binaries for the cluster stage (statically linked against vcpkg's gRPC; same Ubuntu image).
    tar -czf ~/server-binaries.tar.gz -C build/linux-server-release/server \
      strata_shard strata_coordinator strata_load
    cd ~

    python_env
    log "datasets"
    (cd ~/strata && .venv/bin/python scripts/prepare_datasets.py sift1m glove100 bigann10m)
    log "main ready"
    ;;

  *)
    echo "unknown role $role"; exit 2 ;;
esac
