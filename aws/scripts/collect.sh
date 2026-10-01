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
