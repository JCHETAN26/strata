#!/usr/bin/env bash
# Verify nothing from the benchmark session is left in the account: no instances (in any state but
# terminated), volumes, VPCs, security groups, key pairs, placement groups, or Elastic IPs tagged or
# named strata-bench. Exits non-zero, listing what remains, if anything does. Read-only.
#
#   aws/scripts/check_clean.sh [region]    # default: the region in terraform/variables.tf
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
region=${1:-$(sed -n '/variable "region"/,/}/s/.*default *= *"\(.*\)"/\1/p' "$here/../terraform/variables.tf")}
tag=(Name=tag:Project,Values=strata-bench)
left=0

check() {
  local what=$1 found=$2
  if [[ -n "$found" && "$found" != "None" ]]; then
    echo "LEFT: $what: $found"
    left=1
  fi
}

check "instances" "$(aws ec2 describe-instances --region "$region" --filters "${tag[@]}" \
  Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text)"
check "volumes" "$(aws ec2 describe-volumes --region "$region" --filters "${tag[@]}" \
  --query 'Volumes[].VolumeId' --output text)"
# Unattached volumes of any tag, as a warning only (they could belong to another project).
loose=$(aws ec2 describe-volumes --region "$region" --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output text)
[[ -z "$loose" ]] || echo "WARNING: unattached volumes in $region (billing; check they are not ours): $loose"
check "VPCs" "$(aws ec2 describe-vpcs --region "$region" --filters "${tag[@]}" \
  --query 'Vpcs[].VpcId' --output text)"
check "security groups" "$(aws ec2 describe-security-groups --region "$region" \
  --filters Name=group-name,Values=strata-bench --query 'SecurityGroups[].GroupId' --output text)"
check "key pairs" "$(aws ec2 describe-key-pairs --region "$region" \
  --filters Name=key-name,Values=strata-bench --query 'KeyPairs[].KeyName' --output text 2>/dev/null)"
check "placement groups" "$(aws ec2 describe-placement-groups --region "$region" \
  --filters Name=group-name,Values=strata-bench-cluster --query 'PlacementGroups[].GroupName' --output text)"
check "Elastic IPs" "$(aws ec2 describe-addresses --region "$region" --filters "${tag[@]}" \
  --query 'Addresses[].PublicIp' --output text)"
check "spot requests" "$(aws ec2 describe-spot-instance-requests --region "$region" --filters "${tag[@]}" \
  Name=state,Values=open,active --query 'SpotInstanceRequests[].SpotInstanceRequestId' --output text)"

if (( left )); then
  echo "Resources remain in $region (above). Run aws/scripts/teardown.sh, or delete them in the console."
  exit 1
fi
echo "clean: no strata-bench resources remain in $region"
