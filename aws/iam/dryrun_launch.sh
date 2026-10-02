#!/usr/bin/env bash
# Asks AWS itself, with EC2 --dry-run (nothing is created, changed, or deleted), whether the
# strata-terraform user may make every call of the benchmark session: the launch, everything
# Terraform does after it, the reads, and the whole teardown. And whether calls the policy must
# refuse are refused. The IAM simulator only evaluates the context you give it; a dry run
# evaluates the context AWS really builds.
#
#   AWS_PROFILE=strata aws/iam/dryrun_launch.sh
#   AWS_PROFILE=strata DECODE_PROFILE=default aws/iam/dryrun_launch.sh
#
# Needs the network applied (it uses the real VPC, subnet, security group, and key pair from
# Terraform's state). Section B also needs an instance in the state: run it after a launch and
# before destroying, or it is skipped (and says so). Unexpected outcomes are decoded with
# DECODE_PROFILE, an identity allowed sts:DecodeAuthorizationMessage (the strata user is not).
#
# Where the calls come from: Terraform AWS provider v5.100.0 (aws/terraform/.terraform.lock.hcl),
# internal/service/ec2/ec2_instance.go. After RunInstances, create waits (DescribeInstances), tags
# the root volume, then runs the update function on the new instance, in which two blocks fire
# with this configuration: instance_initiated_shutdown_behavior (ModifyInstanceAttribute) and
# root_block_device tags (CreateTags/DeleteTags on the volume). Read uses DescribeInstances,
# DescribeInstanceAttribute, DescribeVolumes, DescribeInstanceTypes, DescribeTags, DescribeImages,
# DescribeVpcs. Delete turns termination protection off (ModifyInstanceAttribute) and then calls
# TerminateInstances. Re-derive this list when the provider version changes.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tf_dir="$here/../terraform"
decode_profile=${DECODE_PROFILE:-default}
region=us-east-2
ok=0
bad=0
skipped=0

# --- Resource IDs from Terraform's state -----------------------------------------------------------
state_json=$(terraform -chdir="$tf_dir" show -json 2>/dev/null) || { echo "cannot read Terraform state"; exit 2; }
eval "$(python3 -c '
import json, shlex, sys
d = json.loads(sys.stdin.read())
res = {}
for r in d.get("values", {}).get("root_module", {}).get("resources", []):
    res[r["address"]] = r["values"]
def g(addr, key):
    return (res.get(addr) or {}).get(key) or ""
inst = res.get("aws_instance.main[0]") or {}
root = (inst.get("root_block_device") or [{}])[0]
out = {
    "vpc": g("aws_vpc.bench", "id"), "subnet": g("aws_subnet.bench", "id"),
    "sg": g("aws_security_group.bench", "id"), "igw": g("aws_internet_gateway.bench", "id"),
    "rtb": g("aws_route_table.bench", "id"), "rtb_assoc": g("aws_route_table_association.bench", "id"),
    "key": g("aws_key_pair.bench", "key_name"), "ami": g("data.aws_ami.ubuntu", "id"),
    "root_dev": g("data.aws_ami.ubuntu", "root_device_name"),
    "instance": inst.get("id") or "", "root_vol": root.get("volume_id") or "",
}
for k, v in out.items():
    print(f"{k}={shlex.quote(v)}")
' <<< "$state_json")"
[[ -n "$vpc" && -n "$subnet" && -n "$sg" && -n "$ami" ]] || { echo "no network in Terraform state: apply first"; exit 2; }
# A tainted (partly created) instance has no root volume ID in the state; ask EC2.
if [[ -n "$instance" && -z "$root_vol" ]]; then
  root_vol=$(aws ec2 describe-volumes --region "$region" \
    --filters "Name=attachment.instance-id,Values=$instance" --query 'Volumes[0].VolumeId' \
    --output text 2>/dev/null)
  [[ "$root_vol" == None ]] && root_vol=""
fi
var_default() { sed -n "/variable \"$1\"/,/}/s/.*default *= *\"*\([^\"]*\)\"*/\1/p" "$tf_dir/variables.tf" | head -n 1; }
main_type=$(var_default main_instance_type)
main_gb=$(var_default main_volume_gb)
shard_type=$(var_default shard_instance_type)
client_type=$(var_default client_instance_type)
cluster_gb=$(var_default cluster_volume_gb)
echo "as $(aws sts get-caller-identity --query Arn --output text 2>&1)"
echo "vpc $vpc subnet $subnet sg $sg igw $igw rtb $rtb key $key ami $ami instance ${instance:-none} root volume ${root_vol:-none}"

# --- Checking one call ------------------------------------------------------------------------------
# check EXPECT LABEL AWS-ARGS...    EXPECT: allow (DryRunOperation or success) | deny (refused)
decode() {
  local msg
  msg=$(grep -o 'Encoded authorization failure message: [^ ]*' <<< "$1" | awk '{print $NF}')
  [[ -n "$msg" ]] || return 0
  AWS_PROFILE=$decode_profile aws sts decode-authorization-message --encoded-message "$msg" \
    --query DecodedMessage --output text 2>/dev/null | python3 -c '
import json, sys
c = json.load(sys.stdin)["context"]
print("        resource:", c["resource"])
for e in c["conditions"]["items"]:
    k = e["key"]
    if k.startswith(("ec2:Attribute", "ec2:Instance", "ec2:Volume", "ec2:Owner", "aws:Request",
                     "aws:ResourceTag/Project", "ec2:ResourceTag/Project", "aws:ResourceAccount",
                     "ec2:Metadata", "aws:TagKeys", "ec2:CreateAction")):
        print("       ", k, "=", [v["value"] for v in e["values"]["items"]])' || true
}

check() {
  local expect=$1 label=$2
  shift 2
  local out got
  out=$(aws "$@" --region "${CHECK_REGION:-$region}" 2>&1 >/dev/null)
  local code=$?
  if [[ "$out" == *DryRunOperation* || ( $code -eq 0 && -z "$out" ) ]]; then
    got=allow
  elif [[ "$out" == *UnauthorizedOperation* || "$out" == *AccessDenied* ]]; then
    got=deny
  else
    got="error: $(tr '\n' ' ' <<< "$out" | cut -c1-160)"
  fi
  if [[ "$got" == "$expect" ]]; then
    ok=$((ok + 1))
    printf '  OK      %-5s %s\n' "$expect" "$label"
  else
    bad=$((bad + 1))
    printf '  PROBLEM %-5s %s -> %s\n' "$expect" "$label" "$got"
    [[ "$got" == deny ]] && decode "$out"
  fi
}

skip() { skipped=$((skipped + 1)); printf '  SKIPPED       %s (%s)\n' "$1" "$2"; }

tags_instance='ResourceType=instance,Tags=[{Key=Project,Value=strata-bench},{Key=ManagedBy,Value=terraform},{Key=Name,Value=strata-bench-main},{Key=Role,Value=main}]'
tags_volume='ResourceType=volume,Tags=[{Key=Project,Value=strata-bench},{Key=ManagedBy,Value=terraform},{Key=Name,Value=strata-bench}]'
launch() {  # launch TYPE GB [extra args]: sets L, the arguments of a dry-run RunInstances
  local type=$1 gb=$2
  shift 2
  L=(ec2 run-instances --dry-run --image-id "$ami" --instance-type "$type"
     --subnet-id "$subnet" --security-group-ids "$sg" --key-name "$key"
     --metadata-options HttpTokens=required --instance-initiated-shutdown-behavior terminate
     "--block-device-mappings=DeviceName=${root_dev},Ebs={VolumeSize=${gb},VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}"
     --tag-specifications "$tags_instance" "$tags_volume" "$@")
}

echo "A. Launch"
launch "$main_type" "$main_gb"
check allow "RunInstances main ($main_type, ${main_gb} GB)" "${L[@]}"
launch "$shard_type" "$cluster_gb"
check allow "RunInstances shard ($shard_type, ${cluster_gb} GB; placement group checked below)" "${L[@]}"
launch "$client_type" "$cluster_gb"
check allow "RunInstances client ($client_type, ${cluster_gb} GB)" "${L[@]}"
check allow "CreatePlacementGroup (cluster stage), tagged" ec2 create-placement-group --dry-run \
  --group-name strata-bench-cluster --strategy cluster \
  --tag-specifications 'ResourceType=placement-group,Tags=[{Key=Project,Value=strata-bench}]'
launch c7i.16xlarge "$main_gb"
check deny "RunInstances a type outside the plan (c7i.16xlarge)" "${L[@]}"
launch "$main_type" "$main_gb" --instance-market-options MarketType=spot
check deny "RunInstances on spot" "${L[@]}"
launch "$main_type" 500
check deny "RunInstances with a 500 GB root volume" "${L[@]}"

echo "B. After launch (create and update paths)"
if [[ -n "$instance" ]]; then
  check allow "ModifyInstanceAttribute shutdown behavior = terminate" ec2 modify-instance-attribute \
    --dry-run --instance-id "$instance" --instance-initiated-shutdown-behavior Value=terminate
  check deny "ModifyInstanceAttribute shutdown behavior = stop" ec2 modify-instance-attribute \
    --dry-run --instance-id "$instance" --instance-initiated-shutdown-behavior Value=stop
  check deny "ModifyInstanceAttribute termination protection on" ec2 modify-instance-attribute \
    --dry-run --instance-id "$instance" --disable-api-termination
  check deny "ModifyInstanceAttribute user data" ec2 modify-instance-attribute \
    --dry-run --instance-id "$instance" --attribute userData --value dGVzdA==
  check allow "CreateTags on the instance (tag updates)" ec2 create-tags --dry-run \
    --resources "$instance" --tags Key=Name,Value=strata-bench-main
  if [[ -n "$root_vol" ]]; then
    check allow "CreateTags on the root volume (root_block_device tags)" ec2 create-tags --dry-run \
      --resources "$root_vol" --tags Key=Name,Value=strata-bench
    check allow "DeleteTags on the root volume (a tag other than Project)" ec2 delete-tags --dry-run \
      --resources "$root_vol" --tags Key=Name
  else
    skip "root volume tag calls" "no root volume ID in state"
  fi
  check deny "DeleteTags Project on the instance" ec2 delete-tags --dry-run \
    --resources "$instance" --tags Key=Project
else
  skip "post-launch calls" "no instance in Terraform state; rerun after a launch, before destroying"
fi

echo "C. Reads (Terraform refresh, plan, waits; up.sh; check_clean.sh)"
for a in describe-instances describe-volumes describe-instance-types describe-tags \
         describe-vpcs describe-subnets describe-security-groups describe-route-tables \
         describe-internet-gateways describe-key-pairs describe-network-interfaces \
         describe-network-acls describe-placement-groups describe-account-attributes \
         describe-availability-zones describe-addresses describe-spot-instance-requests \
         describe-instance-credit-specifications describe-instance-type-offerings \
         describe-security-group-rules; do
  check allow "$a" ec2 "$a" --dry-run
done
check allow "describe-images (the AMI)" ec2 describe-images --dry-run --image-ids "$ami"
check allow "describe-vpc-attribute" ec2 describe-vpc-attribute --dry-run --vpc-id "$vpc" \
  --attribute enableDnsHostnames
if [[ -n "$instance" ]]; then
  for attr in instanceInitiatedShutdownBehavior disableApiTermination disableApiStop userData; do
    check allow "describe-instance-attribute $attr" ec2 describe-instance-attribute --dry-run \
      --instance-id "$instance" --attribute "$attr"
  done
fi
check allow "sts get-caller-identity (up.sh)" sts get-caller-identity
check allow "service-quotas get-service-quota (up.sh)" service-quotas get-service-quota \
  --service-code ec2 --quota-code L-1216C47A

echo "D. Teardown (terraform destroy)"
if [[ -n "$instance" ]]; then
  check allow "ModifyInstanceAttribute termination protection off (before terminate)" \
    ec2 modify-instance-attribute --dry-run --instance-id "$instance" --no-disable-api-termination
  check allow "TerminateInstances" ec2 terminate-instances --dry-run --instance-ids "$instance"
else
  skip "instance teardown calls" "no instance in Terraform state"
fi
check allow "DeleteSecurityGroup" ec2 delete-security-group --dry-run --group-id "$sg"
[[ -n "$rtb_assoc" ]] && check allow "DisassociateRouteTable" ec2 disassociate-route-table \
  --dry-run --association-id "$rtb_assoc"
check allow "DeleteRouteTable" ec2 delete-route-table --dry-run --route-table-id "$rtb"
check allow "DeleteSubnet" ec2 delete-subnet --dry-run --subnet-id "$subnet"
check allow "DetachInternetGateway" ec2 detach-internet-gateway --dry-run \
  --internet-gateway-id "$igw" --vpc-id "$vpc"
check allow "DeleteInternetGateway" ec2 delete-internet-gateway --dry-run --internet-gateway-id "$igw"
check allow "DeleteKeyPair" ec2 delete-key-pair --dry-run --key-name "$key"
check allow "DeleteVpc" ec2 delete-vpc --dry-run --vpc-id "$vpc"

echo "E. Must be refused (scope)"
default_vpc=$(aws ec2 describe-vpcs --region "$region" --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
if [[ -z "$default_vpc" || "$default_vpc" == None ]]; then
  skip "default-VPC scope checks" "this region has no default VPC"
else
  check deny "DeleteVpc on the account's default VPC" ec2 delete-vpc --dry-run --vpc-id "$default_vpc"
  check deny "CreateSubnet in the default VPC" ec2 create-subnet --dry-run --vpc-id "$default_vpc" \
    --cidr-block 172.31.250.0/24 \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Project,Value=strata-bench}]'
fi
other_key=$(aws ec2 describe-key-pairs --region "$region" \
  --query "KeyPairs[?KeyName!='strata-bench'] | [0].KeyName" --output text 2>/dev/null)
if [[ -n "$other_key" && "$other_key" != None ]]; then
  check deny "DeleteKeyPair on another project's key pair ($other_key)" ec2 delete-key-pair \
    --dry-run --key-name "$other_key"
fi
check deny "CreateVpc without the Project tag" ec2 create-vpc --dry-run --cidr-block 10.99.0.0/16
CHECK_REGION=us-west-2 check deny "DescribeInstances in us-west-2" ec2 describe-instances --dry-run

echo
echo "$ok as expected, $bad problems, $skipped skipped"
if ((bad)); then
  echo "Fix the policy for each PROBLEM, update the user's policy, and rerun before applying."
  exit 1
fi
((skipped == 0)) || echo "Some checks were skipped (see SKIPPED lines above)."
echo "OK: every checked call behaves as the session needs."
