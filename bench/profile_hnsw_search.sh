#!/bin/bash
# Profile Strata's HNSW search path with macOS `sample`: build the index once, then sample the
# process only while queries run, so build time does not dilute the profile.
#
#   bench/profile_hnsw_search.sh [DATASET] [EF] [SECONDS]
#   bench/profile_hnsw_search.sh sift1m-200k-q1000 40 6
#
# Writes results/profiles/<dataset>-ef<EF>-<commit>.sample.txt (the raw `sample` report: call
# tree plus "Sort by top of stack" self-time summary) and the harness's JSON next to it.
# macOS only. Refuses to start, and reports after, if `pmset -g therm` shows any thermal or
# performance warning: the M2 development machine is fanless (see CLAUDE.md, Machines).
set -euo pipefail

DATASET=${1:-sift1m-200k-q1000}
EF=${2:-40}
SECONDS_TO_SAMPLE=${3:-6}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HARNESS=$ROOT/build/release/bench/strata_search
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
OUT=$ROOT/results/profiles/$DATASET-ef$EF-$COMMIT
mkdir -p "$(dirname "$OUT")"

thermal_warnings() { pmset -g therm | grep -v "^Note: No" || true; }
if [[ -n "$(thermal_warnings)" ]]; then
  echo "thermal warning present; not starting:" >&2
  thermal_warnings >&2
  exit 1
fi
[[ -x $HARNESS ]] || { echo "build first: cmake --build --preset release" >&2; exit 1; }
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!results')" ]]; then
  echo "warning: uncommitted changes; the profile may not match $COMMIT" >&2
fi

# Enough timed passes at one ef that the search phase outlasts the sampling window.
"$HARNESS" --data "$ROOT/data/$DATASET" --index hnsw --M 16 --ef-construction 200 \
  --ef-search "$EF" --runs 60 > "$OUT.json" 2> "$OUT.stderr" &
PID=$!
until grep -q "built in" "$OUT.stderr" 2>/dev/null; do
  kill -0 $PID 2>/dev/null || { echo "harness exited before searching; see $OUT.stderr" >&2; exit 1; }
  sleep 1
done
grep "built in" "$OUT.stderr" >&2
sample $PID "$SECONDS_TO_SAMPLE" 1 -mayDie -file "$OUT.sample.txt" > /dev/null 2>&1
wait $PID
rm -f "$OUT.stderr"

echo "wrote ${OUT#"$ROOT"/}.sample.txt" >&2
grep -A12 "Sort by top of stack" "$OUT.sample.txt" | sed 's/  (in [^)]*)//' | cut -c1-140 >&2
if [[ -n "$(thermal_warnings)" ]]; then
  echo "THERMAL WARNING after the run:" >&2
  thermal_warnings >&2
  exit 3
fi
