#!/usr/bin/env bash
# Build Strata's Python bindings and install the GPU embedding/reranking dependencies on Kaggle.
# Run this from inside the cloned repo, AFTER kaggle/README.md's clone cell:
#
#     bash kaggle/setup.sh
#
# It is safe to re-run: vcpkg, the build, and the dependency sync are all incremental, so a
# restarted session skips work that is already done. Nothing here prints the GitHub token.
#
# Layout choices for Kaggle's split disk:
#   - The uv environment and cache go on /kaggle/temp (ephemeral, roomy) so the ~7 GB CUDA-PyTorch
#     install does not eat the ~20 GB persistent /kaggle/working quota.
#   - Embeddings, indexes, datasets, and results live under the repo in /kaggle/working (persistent),
#     so a killed session resumes and the results survive to download.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pinned vcpkg baseline (must match vcpkg.json "builtin-baseline").
VCPKG_BASELINE="617ef1c0c422737d117eda00a471ea1be5bad088"
TEMP_ROOT="${STRATA_TEMP_ROOT:-/kaggle/temp}"
[ -d "$TEMP_ROOT" ] || TEMP_ROOT="$(mktemp -d)"   # off-Kaggle fallback

export VCPKG_ROOT="${VCPKG_ROOT:-$TEMP_ROOT/vcpkg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TEMP_ROOT/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TEMP_ROOT/strata-venv}"
export STRATA_OUTPUT_DIR="${STRATA_OUTPUT_DIR:-/kaggle/working/strata-results}"

echo "== toolchain =="
command -v ninja >/dev/null 2>&1 || pip install --quiet ninja
command -v cmake >/dev/null 2>&1 || pip install --quiet cmake
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "cmake $(cmake --version | head -1); ninja $(ninja --version); uv $(uv --version)"

echo "== vcpkg (pinned $VCPKG_BASELINE) =="
if [ ! -x "$VCPKG_ROOT/vcpkg" ]; then
  if [ ! -d "$VCPKG_ROOT/.git" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
  fi
  git -C "$VCPKG_ROOT" fetch --depth 1 origin "$VCPKG_BASELINE"
  git -C "$VCPKG_ROOT" checkout --quiet "$VCPKG_BASELINE"
  "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
fi

echo "== dependencies + bindings (uv sync builds the strata extension) =="
# Only the groups the stages need: embed (CUDA PyTorch, sentence-transformers) and rag (anthropic,
# for the optional answer-generation stage). VCPKG_ROOT is exported so the C++ build finds it.
uv sync --no-default-groups --group embed --group rag

echo "== verify =="
uv run python -c "import strata; print('strata build_info:', strata.build_info())"
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available. Check that a GPU accelerator is attached and that the "
          "Kaggle driver supports the pinned CUDA build (see kaggle/README.md 'GPU not detected').")
PY

# Persist the environment so later notebook cells reuse the same venv/toolchain.
ENV_FILE="/kaggle/working/strata-env.sh"
[ -d /kaggle/working ] || ENV_FILE="$REPO_ROOT/.kaggle-env.sh"
cat > "$ENV_FILE" <<EOF
export VCPKG_ROOT="$VCPKG_ROOT"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export STRATA_OUTPUT_DIR="$STRATA_OUTPUT_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
EOF
echo "== done. environment saved to $ENV_FILE =="
echo "Run stages with, e.g.:  source $ENV_FILE && uv run python kaggle/run_stages.py stage2"
