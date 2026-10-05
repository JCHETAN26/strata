# Strata in Docker Compose: three shards and a coordinator

Runs `strata_shard` ×3 and `strata_coordinator` on one machine, on a private Docker network.
Every hop uses TLS and the shared token: the servers refuse to listen beyond loopback without
them. Each shard keeps its data (snapshot and write-ahead log) in a named volume, so it survives
restarts. The coordinator is published on the host's loopback only (`127.0.0.1:50050`).

## Run

From the repository root, on Linux (tested on Ubuntu 22.04, ARM):

```sh
# 1. Build the server binaries (gRPC is linked statically) and the runtime image
cmake --preset linux-server-release && cmake --build --preset linux-server-release
docker build -f deploy/docker/Dockerfile -t strata-server build/linux-server-release/server

# 2. Certificates naming the compose services, and the token (key and token: mode 600, yours)
scripts/make_dev_certs.sh deploy/docker/certs DNS:shard0 DNS:shard1 DNS:shard2 DNS:coordinator \
  DNS:localhost IP:127.0.0.1

# 3. Start, running the containers as you, so they can read the key and token
export STRATA_UID=$(id -u) STRATA_GID=$(id -g)
docker compose -f deploy/docker/docker-compose.yml up -d
docker compose -f deploy/docker/docker-compose.yml logs | grep listening   # 4 lines
```

A client connects to `127.0.0.1:50050` with the CA and the token, for example the load client:

```sh
build/linux-server-release/server/strata_load insert --target 127.0.0.1:50050 \
  --data data/siftsmall --ids-out /tmp/ids.u32 \
  --ca deploy/docker/certs/ca.pem --token-file deploy/docker/certs/token
```

## Operating notes

- **The shard order is part of every id.** Each shard's position in the coordinator's `--shard`
  list is encoded into the ids it returns, so never reorder or remove one. Adding shards means a
  new deployment.
- **Restarts are safe.** A restarted shard recovers from its volume: the snapshot, then a replay
  of the write-ahead log.
- **Stop:** `docker compose -f deploy/docker/docker-compose.yml down` (add `-v` to delete the
  data volumes).
- `deploy/docker/certs/` is gitignored. It holds the private key and the token.
