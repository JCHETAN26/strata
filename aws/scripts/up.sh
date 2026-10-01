#!/usr/bin/env bash
# Plan (and, only with --apply, create) one stage of the benchmark session.
#
#   aws/scripts/up.sh main              # checks + terraform plan; creates nothing
#   aws/scripts/up.sh main --apply      # the same, then asks you to type the stage name to apply
#   aws/scripts/up.sh cluster --apply   # after `teardown.sh` of the main stage (vCPU quota)
#
# Checks before planning: AWS identity, terraform installed, no instances from the other stage
# still running, and the region's vCPU quota covers the stage. Your current public IP becomes the
# only address allowed to SSH in.
source "$(dirname "$0")/lib.sh"

stage=${1:-}
apply=${2:-}
[[ "$stage" == main || "$stage" == cluster ]] || die "usage: up.sh main|cluster [--apply]"

command -v terraform >/dev/null || die "terraform is not installed (https://developer.hashicorp.com/terraform/install)"
command -v aws >/dev/null || die "the AWS CLI is not installed"
identity=$(aws sts get-caller-identity --query Arn --output text) || die "AWS credentials not configured"
log "AWS identity: $identity"

region=$(sed -n '/variable "region"/,/}/s/.*default *= *"\(.*\)"/\1/p' "$TF_DIR/variables.tf")
case $stage in
  main) need=32 ;;     # c7i.8xlarge
  cluster) need=28 ;;  # 4 x c7i.xlarge + c7i.xlarge + c7i.2xlarge
esac
quota=$(aws service-quotas get-service-quota --region "$region" --service-code ec2 \
  --quota-code L-1216C47A --query 'Quota.Value' --output text)
running=$(aws ec2 describe-instances --region "$region" \
  --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].CpuOptions.[CoreCount,ThreadsPerCore]' --output text \
  | awk '{s += $1 * $2} END {print s + 0}')
log "vCPUs: stage needs $need, $running already running in $region, quota ${quota%.*}"
(( need + running <= ${quota%.*} )) \
  || die "not enough vCPU quota: tear down the other stage first, or request an increase (Service Quotas > EC2 > Running On-Demand Standard instances)"

if [[ ! -f "$SSH_KEY" ]]; then
  mkdir -p "$(dirname "$SSH_KEY")"
  ssh-keygen -q -t ed25519 -N "" -C strata-bench -f "$SSH_KEY"
  log "created SSH key $SSH_KEY"
fi

my_ip=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
[[ "$my_ip" =~ ^[0-9.]+$ ]] || die "could not determine your public IPv4 address"
log "SSH will be allowed from $my_ip/32 only"

tf init -input=false >/dev/null
tf plan -input=false -out=stage.tfplan -var "stage=$stage" -var "operator_cidr=$my_ip/32"

if [[ "$apply" != --apply ]]; then
  log "plan only (nothing created). Rerun with --apply to create the '$stage' stage."
  exit 0
fi
log "Applying creates billable instances (see docs/aws-plan.md for costs)."
read -r -p "Type '$stage' to apply: " answer
[[ "$answer" == "$stage" ]] || die "not confirmed; nothing created"
tf apply -input=false stage.tfplan
log "created. Next: aws/scripts/run_main.sh (main) or aws/scripts/run_sharding.sh (cluster)"
