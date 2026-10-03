#!/usr/bin/env bash
# Copy results and logs from the running stage's instances back into this repo. Safe to run any
# time, any number of times (rsync copies only what changed). AWS outputs use names the Mac
# results do not, so nothing local is overwritten; check `git status` before committing.
source "$(dirname "$0")/lib.sh"

stage=$(tf_out stage)
dest_logs="$REPO_ROOT/results/aws/logs"
mkdir -p "$dest_logs"
case $stage in
  main)
    host=$(tf_out main_public_ip)
    pull "$host" "$REMOTE_REPO/results/" "$REPO_ROOT/results/"
    pull "$host" "logs/" "$dest_logs/main/"
    ;;
  cluster)
    host=$(tf_out client_public_ip)
    pull "$host" "$REMOTE_REPO/results/" "$REPO_ROOT/results/"
    pull "$host" "logs/" "$dest_logs/client/"
    ;;
  *)
    die "no stage is up (stage=$stage)"
    ;;
esac
log "collected results from the $stage stage into results/ (logs: results/aws/logs/)"

# The comparison tables written on the instance by commits before 2026-10-03 mixed in the Mac
# records committed in the repo (and took their hardware label). Regenerate them here from the
# collected records, restricted to the AWS instance type. Harmless when they are already right.
if [[ $stage == main ]] && command -v uv >/dev/null; then
  machine=$(sed -n '/variable "main_instance_type"/,/}/s/.*default *= *"\(.*\)"/\1/p' "$TF_DIR/variables.tf")
  report() { (cd "$REPO_ROOT" && env -u PYTHONPATH uv run python bench/run_hnsw_curves.py \
    --report-only --machine "$machine" "$@" >/dev/null 2>&1) || true; }
  report --datasets sift1m glove100 --simd avx2 --name hnsw_vs_reference_x86
  report --datasets sift1m glove100 --simd native --name hnsw_vs_reference_x86_avx512
  report --datasets bigann10m --build-threads 16 --simd avx2 --name hnsw_vs_reference_10m
  report --datasets bigann10m --build-threads 16 --simd native --name hnsw_vs_reference_10m_avx512
  log "regenerated the comparison tables from $machine records only"
fi

# Guard: nothing secret may reach results/, which gets committed to a public repo. The instances
# have no IAM role (so no AWS credentials exist on them), and no script writes keys or the
# cluster token into results; this check makes that a verified fact rather than an assumption.
# A hit is moved to aws/.quarantine/ (gitignored) and the script fails.
patterns='BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|aws_secret_access_key|aws_session_token|sk-ant-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|X-aws-ec2-metadata-token: [A-Za-z0-9_=-]{20,}'
hits=$(grep -rIlE "$patterns" "$REPO_ROOT/results" 2>/dev/null || true)
if [[ -f "$AWS_DIR/.certs/token" ]]; then
  hits+=$'\n'$(grep -rIlF "$(tr -d '[:space:]' < "$AWS_DIR/.certs/token")" "$REPO_ROOT/results" 2>/dev/null || true)
fi
hits=$(printf '%s\n' "$hits" | sed '/^$/d' | sort -u)
if [[ -n "$hits" ]]; then
  mkdir -p "$AWS_DIR/.quarantine"
  while read -r f; do
    mv "$f" "$AWS_DIR/.quarantine/$(echo "${f#"$REPO_ROOT"/}" | tr / _)"
    log "SECRET-LIKE CONTENT: moved ${f#"$REPO_ROOT"/} to aws/.quarantine/"
  done <<< "$hits"
  die "collected files looked like they held credentials (moved out of results/); inspect before committing anything"
fi
