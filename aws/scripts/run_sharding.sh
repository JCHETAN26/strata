#!/usr/bin/env bash
# Part F, from the laptop: multi-machine sharding scaling and tail latency on the cluster stage.
#
#   aws/scripts/run_sharding.sh                     # 1, 2, 4 shards on the cluster stage
#   LOCAL=1 DATASET=siftsmall LABEL=local-smoke aws/scripts/run_sharding.sh
#                                                   # the same flow on this machine, for testing
#
# For each shard count N: fresh shards on N machines (HNSW, --sync none for the bulk load), a
# coordinator in front of them, the dataset loaded through the coordinator by the load client
# (its own machine), then bench/run_sharding_bench.py measure. All traffic uses TLS and the shared
# token (certificates made here, naming every machine's private address). Then the report.
# Needs the server binaries from the main stage (run_main.sh saves them in aws/.artifacts/).
source "$(dirname "$0")/lib.sh"

DATASET=${DATASET:-sift1m}
LABEL=${LABEL:-aws}
SHARD_COUNTS=${SHARD_COUNTS:-"1 2 4"}
DIM=${DIM:-128}
LOCAL=${LOCAL:-0}
CERTS="$AWS_DIR/.certs"

if (( LOCAL )); then
  # Everything on this machine: processes instead of instances, one port per shard.
  W=$(mktemp -d "${TMPDIR:-/tmp}/strata-sharding.XXXXXX")
  BIN="$REPO_ROOT/build/server-release/server"
  shard_hosts=(127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1)
  shard_private=("${shard_hosts[@]}")
  coord_host=127.0.0.1 coord_private=127.0.0.1 client_host=127.0.0.1
  cluster_description="local smoke test on $(uname -sm), not a measurement"
  on() { local _h=$1; shift; bash -c "$*"; }
  send() { local _h=$1; shift; cp "${@:1:$#-1}" "${*: -1}"; }
  PY="env -u PYTHONPATH uv run python"
  CLIENT_REPO=$REPO_ROOT
else
  [[ "$(tf_out stage)" == cluster ]] || die "the cluster stage is not up (aws/scripts/up.sh cluster --apply)"
  [[ -f "$ARTIFACTS/server-binaries.tar.gz" ]] || die "no server binaries: run the main stage first"
  W='~'
  BIN='~/bin'
  # (No mapfile: macOS ships bash 3.2.)
  shard_hosts=()
  while IFS= read -r line; do [[ -n "$line" ]] && shard_hosts+=("$line"); done < <(tf_out shard_public_ips)
  shard_private=()
  while IFS= read -r line; do [[ -n "$line" ]] && shard_private+=("$line"); done < <(tf_out shard_private_ips)
  (( ${#shard_hosts[@]} == ${#shard_private[@]} && ${#shard_hosts[@]} > 0 )) \
    || die "terraform reported ${#shard_hosts[@]} shard public and ${#shard_private[@]} private addresses"
  coord_host=$(tf_out coordinator_public_ip) coord_private=$(tf_out coordinator_private_ip)
  client_host=$(tf_out client_public_ip)
  cluster_description=$(tf_out cluster_description)
  on() { remote "$@"; }
  send() { push "$@"; }
  PY=".venv/bin/python"
  CLIENT_REPO='~/strata'
fi
all_hosts=("${shard_hosts[@]}" "$coord_host" "$client_host")

# --- Machines, binaries, certificates -------------------------------------------------------------
if (( ! LOCAL )); then
  commit=$(pinned_commit)
  for h in "${all_hosts[@]}"; do wait_for_ssh "$h"; done
  for h in "${all_hosts[@]}"; do push "$h" "$AWS_DIR/scripts/setup_machine.sh" "~/"; done
  start_job "$client_host" setup "~/setup_machine.sh client $commit"
  for h in "${shard_hosts[@]}" "$coord_host"; do on "$h" "bash ~/setup_machine.sh node"; done
  for h in "${all_hosts[@]}"; do
    push "$h" "$ARTIFACTS/server-binaries.tar.gz" "~/"
    on "$h" "mkdir -p ~/bin && tar -xzf ~/server-binaries.tar.gz -C ~/bin"
  done
  wait_job "$client_host" setup
fi

sans=(DNS:localhost IP:127.0.0.1)
for ip in "${shard_private[@]}" "$coord_private"; do sans+=("IP:$ip"); done
rm -rf "$CERTS"
"$REPO_ROOT/scripts/make_dev_certs.sh" "$CERTS" $(printf '%s\n' "${sans[@]}" | sort -u) >/dev/null
for h in $(printf '%s\n' "${all_hosts[@]}" | sort -u); do
  on "$h" "mkdir -p $W/certs $W/logs"
  send "$h" "$CERTS/ca.pem" "$CERTS/server.pem" "$CERTS/server.key" "$CERTS/token" "$W/certs/"
done
tls="--tls-cert $W/certs/server.pem --tls-key $W/certs/server.key --token-file $W/certs/token"
client_tls="--ca $W/certs/ca.pem --token-file $W/certs/token"

stop_all() {
  for h in $(printf '%s\n' "${all_hosts[@]}" | sort -u); do
    on "$h" "pkill -x strata_shard; pkill -x strata_coordinator; true"
  done
  sleep 2
}

# wait_listening HOST LOG: until the server prints its startup line.
wait_listening() {
  for _ in $(seq 60); do
    on "$1" "grep -q listening $2" 2>/dev/null && return 0
    sleep 1
  done
  on "$1" "cat $2" >&2 || true
  die "server on $1 did not start (log $2)"
}

info="$W/cluster.json"
on "$client_host" "printf '%s' '{\"description\": \"$cluster_description\"}' > $info"

# --- One shard count at a time -----------------------------------------------------------------
for n in $SHARD_COUNTS; do
  (( n <= ${#shard_hosts[@]} )) || die "$n shards requested, ${#shard_hosts[@]} available"
  log "=== $n shard(s)"
  stop_all
  targets=()
  for (( i = 0; i < n; i++ )); do
    port=$((50051 + i))
    on "${shard_hosts[$i]}" "rm -rf $W/shard-data-$i; nohup $BIN/strata_shard --dir $W/shard-data-$i \
      --dim $DIM --index hnsw --sync none --listen 0.0.0.0 --port $port $tls \
      > $W/logs/shard-$i.log 2>&1 < /dev/null &"
    wait_listening "${shard_hosts[$i]}" "$W/logs/shard-$i.log"
    targets+=(--shard "${shard_private[$i]}:$port")
  done
  on "$coord_host" "nohup $BIN/strata_coordinator --dim $DIM ${targets[*]} \
    --shard-ca $W/certs/ca.pem --shard-token-file $W/certs/token \
    --listen 0.0.0.0 --port 50050 $tls > $W/logs/coordinator.log 2>&1 < /dev/null &"
  wait_listening "$coord_host" "$W/logs/coordinator.log"

  out="results/server/sharding/$LABEL"
  on "$client_host" "cd $CLIENT_REPO && mkdir -p $out && $BIN/strata_load insert \
    --target $coord_private:50050 --data data/$DATASET --ids-out $W/ids-$n.u32 $client_tls \
    > $out/insert-n$n.json"
  on "$client_host" "cat $CLIENT_REPO/$out/insert-n$n.json" >&2
  on "$client_host" "cd $CLIENT_REPO && $PY bench/run_sharding_bench.py measure --label $LABEL \
    --shards $n --target $coord_private:50050 --dataset $DATASET --ids $W/ids-$n.u32 \
    $client_tls --cluster-info $info --load-binary $BIN/strata_load ${MEASURE_ARGS:-}"
done
stop_all
on "$client_host" "cd $CLIENT_REPO && $PY bench/run_sharding_bench.py report --label $LABEL"
if (( ! LOCAL )); then
  "$AWS_DIR/scripts/collect.sh"
  log "cluster stage done. Then: aws/scripts/teardown.sh"
else
  rm -rf "$W"
fi
