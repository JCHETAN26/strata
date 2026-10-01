# Shared helpers for the aws/scripts/*.sh entry points (sourced, not run).
# Everything runs from the operator's laptop except setup_machine.sh and run_part.sh, which these
# helpers copy to an instance and start there inside tmux, so a dropped laptop connection does
# not kill a two-hour benchmark.

set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AWS_DIR/.." && pwd)"
TF_DIR="$AWS_DIR/terraform"
SSH_KEY="$AWS_DIR/.ssh/strata-bench"
KNOWN_HOSTS="$AWS_DIR/.ssh/known_hosts"
ARTIFACTS="$AWS_DIR/.artifacts"
REMOTE_REPO="strata"  # ~/strata on every instance

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

tf() { terraform -chdir="$TF_DIR" "$@"; }

# A terraform output as plain text (lists as one value per line).
tf_out() {
  tf output -json "$1" | python3 -c '
import json, sys
v = json.load(sys.stdin)
if isinstance(v, list):
    print("\n".join(map(str, v)))
elif v is not None:
    print(v)'
}

ssh_opts=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS"
          -o ServerAliveInterval=30 -o ConnectTimeout=10)

# remote HOST COMMAND...: run a command on an instance (as ubuntu).
remote() {
  local host=$1; shift
  ssh "${ssh_opts[@]}" "ubuntu@$host" "$@"
}

# push HOST SRC... DEST: copy files to an instance.
push() {
  local host=$1; shift
  local dest=${*: -1}
  scp -q "${ssh_opts[@]}" "${@:1:$#-1}" "ubuntu@$host:$dest"
}

# pull HOST SRC DEST: copy a remote directory tree back (rsync: only what changed).
pull() {
  rsync -a -e "ssh ${ssh_opts[*]}" "ubuntu@$1:$2" "$3"
}

wait_for_ssh() {
  local host=$1
  for _ in $(seq 60); do
    if remote "$host" true 2>/dev/null; then return 0; fi
    sleep 10
  done
  die "no SSH to $host after 10 minutes"
}

# start_job HOST NAME COMMAND: run COMMAND in a detached tmux session on HOST, logging to
# ~/logs/NAME.log and writing its exit code to ~/logs/NAME.exit when it finishes.
start_job() {
  local host=$1 name=$2 cmd=$3
  remote "$host" "mkdir -p ~/logs && rm -f ~/logs/$name.exit && \
    tmux new-session -d -s $name \"bash -lc '$cmd' > ~/logs/$name.log 2>&1; echo \\\$? > ~/logs/$name.exit\""
  log "started $name on $host (log: ~/logs/$name.log; attach: ssh ... tmux attach -t $name)"
}

# wait_job HOST NAME: poll until the job finishes; fail if it failed. Prints the log tail while
# waiting, so progress is visible.
wait_job() {
  local host=$1 name=$2
  while ! remote "$host" "test -f ~/logs/$name.exit" 2>/dev/null; do
    remote "$host" "tail -n 2 ~/logs/$name.log" 2>/dev/null | sed "s/^/  [$name] /" >&2 || true
    sleep 60
  done
  local code
  code=$(remote "$host" "cat ~/logs/$name.exit")
  [[ "$code" == 0 ]] || { remote "$host" "tail -n 40 ~/logs/$name.log" >&2; die "$name failed (exit $code)"; }
  log "$name finished"
}

# The commit every instance checks out: the local HEAD, which must be pushed and clean, so each
# result's recorded commit is one anyone can check out.
pinned_commit() {
  git -C "$REPO_ROOT" diff --quiet && git -C "$REPO_ROOT" diff --cached --quiet \
    || die "uncommitted changes: commit and push first, so results record a real commit"
  local commit
  commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
  git -C "$REPO_ROOT" branch -r --contains "$commit" | grep -q . \
    || die "HEAD $commit is not on any remote branch: push it first"
  echo "$commit"
}
