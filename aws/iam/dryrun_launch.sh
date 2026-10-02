#!/usr/bin/env bash
# Asks AWS whether the strata-terraform user may launch the main-stage instance exactly as
# Terraform would, without launching anything (EC2 --dry-run). The IAM simulator only evaluates
# the context you give it; this evaluates the context AWS really builds (e.g. a Canonical AMI's
# ec2:Owner is "amazon", not Canonical's account ID, which the simulator could not have told us).
#
#   AWS_PROFILE=strata aws/iam/dryrun_launch.sh               # needs the network stage applied
#   AWS_PROFILE=strata DECODE_PROFILE=default aws/iam/dryrun_launch.sh
#
# Uses the subnet, security group, key pair, and AMI from Terraform's state (a partial or full
# apply of the main stage). On refusal it decodes the authorization message with DECODE_PROFILE
# (an identity allowed sts:DecodeAuthorizationMessage; the strata-terraform user deliberately is
# not) and prints the failing resource and the condition keys AWS evaluated. Fix, update the
# user's policy, and rerun until it says the launch would succeed. Read-only either way.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tf_dir="$here/../terraform"
decode_profile=${DECODE_PROFILE:-default}

state_attr() {  # state_attr RESOURCE ATTRIBUTE
  terraform -chdir="$tf_dir" state show -no-color "$1" \
    | awk -v k="$2" '$1 == k { gsub(/"/, "", $3); print $3; exit }'
}
subnet=$(state_attr aws_subnet.bench id)
sg=$(state_attr aws_security_group.bench id)
ami=$(state_attr data.aws_ami.ubuntu id)
root=$(state_attr data.aws_ami.ubuntu root_device_name)
[[ -n "$subnet" && -n "$sg" && -n "$ami" ]] || { echo "no network in Terraform state: apply the network first"; exit 2; }
type=$(sed -n '/variable "main_instance_type"/,/}/s/.*default *= *"\(.*\)"/\1/p' "$tf_dir/variables.tf")
size=$(sed -n '/variable "main_volume_gb"/,/}/s/.*default *= *\([0-9]*\).*/\1/p' "$tf_dir/variables.tf")
echo "as $(aws sts get-caller-identity --query Arn --output text): $type, $ami ($root), ${size} GB gp3, $subnet, $sg"

set +e
err=$(aws ec2 run-instances --dry-run --region us-east-2 \
  --image-id "$ami" --instance-type "$type" --subnet-id "$subnet" --security-group-ids "$sg" \
  --key-name strata-bench --metadata-options HttpTokens=required \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=$root,Ebs={VolumeSize=$size,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Project,Value=strata-bench},{Key=ManagedBy,Value=terraform},{Key=Name,Value=strata-bench-main},{Key=Role,Value=main}]' \
    'ResourceType=volume,Tags=[{Key=Project,Value=strata-bench},{Key=ManagedBy,Value=terraform},{Key=Name,Value=strata-bench}]' \
  2>&1 >/dev/null)
set -e

if [[ "$err" == *DryRunOperation* ]]; then
  echo "OK: the launch would succeed (DryRunOperation). Safe to rerun the apply."
  exit 0
fi
echo "REFUSED: ${err%%Encoded authorization failure message*}"
msg=$(grep -o 'Encoded authorization failure message: [^ ]*' <<< "$err" | awk '{print $NF}')
[[ -n "$msg" ]] || exit 1
AWS_PROFILE=$decode_profile aws sts decode-authorization-message --encoded-message "$msg" \
  --query DecodedMessage --output text | python3 -c '
import json, sys
d = json.load(sys.stdin)
c = d["context"]
print("failing resource:", c["resource"], "| explicit deny:", d.get("explicitDeny"))
for e in c["conditions"]["items"]:
    print("  ", e["key"], "=", [v["value"] for v in e["values"]["items"]])'
exit 1
