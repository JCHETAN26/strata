#!/usr/bin/env bash
# Main stage, from the laptop: set up the c7i.8xlarge, then run parts A-E one after another,
# collecting results after each (so a later failure never loses an earlier part's results).
#
#   aws/scripts/run_main.sh                 # setup + all parts
#   aws/scripts/run_main.sh ann threads     # setup (skipped if done) + only these parts
#
# Each step runs in tmux on the instance; if the laptop disconnects, rerun the same command: setup
# is idempotent and finished parts are skipped (a ~/logs/<part>.exit of 0 marks them done).
source "$(dirname "$0")/lib.sh"

[[ "$(tf_out stage)" == main ]] || die "the main stage is not up (aws/scripts/up.sh main --apply)"
host=$(tf_out main_public_ip)
commit=$(pinned_commit)
parts=("$@")
(( ${#parts[@]} )) || parts=(ann 10m threads filter perf)

wait_for_ssh "$host"
push "$host" "$AWS_DIR/scripts/setup_machine.sh" "$AWS_DIR/scripts/run_part.sh" "~/"
if ! remote "$host" 'test "$(cat ~/logs/setup.exit 2>/dev/null)" = 0'; then
  start_job "$host" setup "~/setup_machine.sh main $commit"
  wait_job "$host" setup
fi
mkdir -p "$ARTIFACTS"
scp -q "${ssh_opts[@]}" "ubuntu@$host:server-binaries.tar.gz" "$ARTIFACTS/"
log "saved server binaries for the cluster stage in $ARTIFACTS"

for part in "${parts[@]}"; do
  if remote "$host" "test \"\$(cat ~/logs/$part.exit 2>/dev/null)\" = 0"; then
    log "part $part already done; skipping"
    continue
  fi
  start_job "$host" "$part" "~/run_part.sh $part"
  wait_job "$host" "$part"
  "$AWS_DIR/scripts/collect.sh"
done
log "main stage done. Check results/, then: aws/scripts/teardown.sh"
