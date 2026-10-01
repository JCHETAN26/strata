# AWS benchmark session: runbook

What runs, why these instances, and what it costs: [`docs/aws-plan.md`](../docs/aws-plan.md).
About $20 on-demand for the whole session. Everything is created by Terraform in a dedicated VPC
in us-east-2, tagged `Project=strata-bench`, and removed by `teardown.sh`, which then verifies
that nothing remains.

## Prerequisites (laptop)

- AWS CLI configured (`aws sts get-caller-identity` works), Terraform ≥ 1.6, `rsync`, `ssh`.
- The commit to benchmark pushed to GitHub, with a clean working tree. The scripts refuse
  otherwise, so every result records a commit anyone can check out.
- Optional: an AWS Budgets alert (e.g. $40) set up in the console.

## Steps

```sh
# 0. See what would be created (no cost): checks credentials, vCPU quota, and your IP, then plans
aws/scripts/up.sh main

# 1. Main stage: one c7i.8xlarge, parts A-E (~11 h, ~$16)
aws/scripts/up.sh main --apply          # asks you to type "main"
aws/scripts/run_main.sh                 # setup, then ann, 10m, threads, filter, perf
#    (or a subset: aws/scripts/run_main.sh ann threads). Rerunnable: finished parts are skipped.
aws/scripts/teardown.sh                 # collects results, destroys, verifies nothing remains

# 2. Cluster stage: 4 shards + coordinator + client, part F (~2.5 h, ~$3)
aws/scripts/up.sh cluster --apply       # asks you to type "cluster"
aws/scripts/run_sharding.sh             # 1, 2, 4 shards; TLS + token everywhere
aws/scripts/teardown.sh

# Any time: is anything still running?
aws/scripts/check_clean.sh
```

Results land in `results/` locally, with logs in `results/aws/logs/`. Check `git status`, then
commit them.

## While it runs

- Every long step runs in tmux on the instance. If the laptop disconnects, nothing stops: rerun
  the same script to reattach to the progress polling.
- To watch a step directly:
  `ssh -i aws/.ssh/strata-bench ubuntu@<ip> tmux attach -t <part>` (detach with `Ctrl-b d`).
- `aws/scripts/collect.sh` copies results back at any point.
- Each instance terminates itself 14 h after boot (`max_lifetime_hours`) as a safety net.

## Files

| file | runs on | does |
|---|---|---|
| `terraform/` | laptop | VPC, subnet, security group (SSH from your IP only; all traffic within the group), key pair, instances for the chosen stage, cluster placement group |
| `scripts/up.sh` | laptop | checks (identity, quota, other stage down), creates the SSH key, `terraform plan`, and `apply` only with `--apply` plus typed confirmation |
| `scripts/setup_machine.sh` | instance | `main`: toolchain, vcpkg, builds + tests, Python env from `uv.lock`, datasets. `client`: repo, Python env, SIFT1M. `node`: binaries directory |
| `scripts/run_part.sh` | main instance | parts A–E (`ann`, `10m`, `threads`, `filter`, `perf`) |
| `scripts/run_main.sh` | laptop | main stage end to end, collecting after each part |
| `scripts/run_sharding.sh` | laptop | cluster stage end to end. `LOCAL=1` runs the same flow on this machine as a smoke test |
| `scripts/collect.sh` | laptop | rsync results and logs back |
| `scripts/teardown.sh` | laptop | collect, `terraform destroy`, `check_clean.sh` |
| `scripts/check_clean.sh` | laptop | read-only: lists any tagged instance, volume, VPC, security group, key pair, placement group, EIP, or spot request still present; exits non-zero if so |

Local, gitignored state: `aws/.ssh/` (session key), `aws/.certs/` (TLS for the cluster),
`aws/.artifacts/` (server binaries from the main stage), and `terraform/*.tfstate`.
