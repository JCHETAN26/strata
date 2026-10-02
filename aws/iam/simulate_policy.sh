#!/usr/bin/env bash
# Checks strata-terraform-policy.json with the IAM policy simulator: what the benchmark session
# needs is allowed, and the things it must not be able to do are denied. Read-only (the simulator
# evaluates the JSON; nothing is attached or created). Rerun after any edit to the policy.
#
#   aws/iam/simulate_policy.sh        # exits non-zero if any case does not match
#
# The simulator caps each policy document at 2,000 characters, so the policy is passed one
# statement per document. Every statement is an Allow, so their union is the policy.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

split=$(mktemp)
trap 'rm -f "$split"' EXIT
python3 -c '
import json, sys
policy = json.load(open(sys.argv[1]))
assert all(s["Effect"] == "Allow" for s in policy["Statement"]), "split is only valid for Allow-only"
for s in policy["Statement"]:
    doc = json.dumps({"Version": policy["Version"], "Statement": [s]}, separators=(",", ":"))
    assert len(doc) <= 2000, "statement " + s["Sid"] + " is too long for the simulator"
    print(doc)
' "$here/strata-terraform-policy.json" > "$split"
docs=()
while IFS= read -r line; do docs+=("$line"); done < "$split"

A=arn:aws:ec2:us-east-2:123456789012
R=aws:RequestedRegion=us-east-2
T=aws:RequestTag/Project=strata-bench
RT=aws:ResourceTag/Project=strata-bench
pass=0
fail=0

# case EXPECTED LABEL ACTION RESOURCE_ARN [CONTEXT_KEY=VALUE ...]
case_() {
  local expect=$1 label=$2 action=$3 arn=$4
  shift 4
  local ctx=() kv k v t
  for kv in "$@"; do
    k=${kv%%=*}
    v=${kv#*=}
    t=string
    [[ $k == ec2:VolumeSize ]] && t=numeric
    ctx+=("ContextKeyName=$k,ContextKeyValues=$v,ContextKeyType=$t")
  done
  local args=(--policy-input-list "${docs[@]}" --action-names "$action" --resource-arns "$arn"
              --query 'EvaluationResults[0].EvalDecision' --output text)
  if ((${#ctx[@]})); then args+=(--context-entries "${ctx[@]}"); fi
  local got
  got=$(aws iam simulate-custom-policy "${args[@]}" 2>&1 | head -n 1)
  if [[ "$got" == "$expect" ]]; then
    pass=$((pass + 1)); printf 'PASS %-13s %s\n' "$got" "$label"
  else
    fail=$((fail + 1)); printf 'FAIL %-13s %s (expected %s)\n' "$got" "$label" "$expect"
  fi
}

launch=(ec2:InstanceMarketType=on-demand ec2:MetadataHttpTokens=required)
case_ allowed      "launch c7i.8xlarge (main), on-demand, IMDSv2, tagged" ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=c7i.8xlarge "${launch[@]}"
case_ allowed      "launch c7i.xlarge (shard)"                ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=c7i.xlarge "${launch[@]}"
case_ allowed      "launch c7i.2xlarge (client)"              ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=c7i.2xlarge "${launch[@]}"
case_ implicitDeny "launch p5.48xlarge (not in the plan)"     ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=p5.48xlarge "${launch[@]}"
case_ implicitDeny "launch on spot"                           ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=c7i.8xlarge ec2:InstanceMarketType=spot ec2:MetadataHttpTokens=required
case_ implicitDeny "launch without IMDSv2"                    ec2:RunInstances "$A:instance/*" $R $T ec2:InstanceType=c7i.8xlarge ec2:InstanceMarketType=on-demand ec2:MetadataHttpTokens=optional
case_ implicitDeny "launch untagged"                          ec2:RunInstances "$A:instance/*" $R ec2:InstanceType=c7i.8xlarge "${launch[@]}"
case_ implicitDeny "launch in us-east-1"                      ec2:RunInstances "arn:aws:ec2:us-east-1:123456789012:instance/*" aws:RequestedRegion=us-east-1 $T ec2:InstanceType=c7i.8xlarge "${launch[@]}"
case_ allowed      "root volume gp3, 150 GB, tagged"          ec2:RunInstances "$A:volume/*" $R $T ec2:VolumeType=gp3 ec2:VolumeSize=150
case_ implicitDeny "root volume 2000 GB"                      ec2:RunInstances "$A:volume/*" $R $T ec2:VolumeType=gp3 ec2:VolumeSize=2000
case_ implicitDeny "root volume io2"                          ec2:RunInstances "$A:volume/*" $R $T ec2:VolumeType=io2 ec2:VolumeSize=100
# Images: the context below is what AWS actually evaluated for the Canonical AMI (decoded from the
# UnauthorizedOperation of the first apply, 2026-10-02). For a verified provider's public image,
# ec2:Owner is the alias "amazon", not the owner's account ID; the account is aws:ResourceAccount.
IMG=arn:aws:ec2:us-east-2::image/ami-0fa99aa8f97f9e30b
case_ allowed      "Canonical image, real context (owner alias amazon)" ec2:RunInstances "$IMG" $R ec2:Owner=amazon aws:ResourceAccount=099720109477 ec2:Public=true ec2:ImageType=machine ec2:RootDeviceType=ebs
case_ implicitDeny "Amazon Linux image (owner amazon, Amazon's account)" ec2:RunInstances "arn:aws:ec2:us-east-2::image/ami-al2023" $R ec2:Owner=amazon aws:ResourceAccount=137112412989 ec2:Public=true
case_ implicitDeny "image in Canonical's account, owner not amazon"   ec2:RunInstances "$IMG" $R ec2:Owner=099720109477 aws:ResourceAccount=099720109477
case_ implicitDeny "someone else's image"                     ec2:RunInstances "arn:aws:ec2:us-east-2::image/ami-123" $R ec2:Owner=111122223333 aws:ResourceAccount=111122223333
case_ allowed      "launch into the bench subnet"             ec2:RunInstances "$A:subnet/subnet-1" $R $RT
case_ implicitDeny "launch into another subnet"               ec2:RunInstances "$A:subnet/subnet-2" $R aws:ResourceTag/Project=other
case_ allowed      "terminate a strata-bench instance"        ec2:TerminateInstances "$A:instance/i-1" $R $RT
case_ implicitDeny "terminate another project's instance"     ec2:TerminateInstances "$A:instance/i-2" $R aws:ResourceTag/Project=kafka
case_ implicitDeny "stop an instance (not needed)"            ec2:StopInstances "$A:instance/i-1" $R $RT
# ModifyInstanceAttribute: contexts as AWS evaluated them for a dry run on a live instance
# (2026-10-02). Terraform sets the shutdown behavior after launch and turns termination
# protection off before terminating; nothing else.
MIA=ec2:ModifyInstanceAttribute
case_ allowed      "set shutdown behavior to terminate (create)"  $MIA "$A:instance/i-1" $R $RT ec2:Attribute=InstanceInitiatedShutdownBehavior ec2:Attribute/InstanceInitiatedShutdownBehavior=terminate
case_ implicitDeny "set shutdown behavior to stop"                $MIA "$A:instance/i-1" $R $RT ec2:Attribute=InstanceInitiatedShutdownBehavior ec2:Attribute/InstanceInitiatedShutdownBehavior=stop
case_ allowed      "turn termination protection off (destroy)"    $MIA "$A:instance/i-1" $R $RT ec2:Attribute=DisableApiTermination ec2:Attribute/DisableApiTermination=false
case_ implicitDeny "turn termination protection on"               $MIA "$A:instance/i-1" $R $RT ec2:Attribute=DisableApiTermination ec2:Attribute/DisableApiTermination=true
case_ implicitDeny "change instance type"                         $MIA "$A:instance/i-1" $R $RT ec2:Attribute=InstanceType ec2:Attribute/InstanceType=p5.48xlarge
case_ implicitDeny "change user data"                             $MIA "$A:instance/i-1" $R $RT ec2:Attribute=UserData
case_ implicitDeny "set shutdown behavior on an untagged instance" $MIA "$A:instance/i-9" $R aws:ResourceTag/Project=other ec2:Attribute/InstanceInitiatedShutdownBehavior=terminate
case_ allowed      "delete the bench VPC"                     ec2:DeleteVpc "$A:vpc/vpc-1" $R $RT
case_ implicitDeny "delete an untagged VPC"                   ec2:DeleteVpc "$A:vpc/vpc-2" $R
case_ implicitDeny "delete another project's key pair"        ec2:DeleteKeyPair "$A:key-pair/kafka" $R
case_ allowed      "create a tagged VPC"                      ec2:CreateVpc "$A:vpc/*" $R $T
case_ implicitDeny "create an untagged VPC"                   ec2:CreateVpc "$A:vpc/*" $R
case_ allowed      "create subnet: the new subnet is tagged"  ec2:CreateSubnet "$A:subnet/*" $R $T
case_ allowed      "create subnet: inside the bench VPC"      ec2:CreateSubnet "$A:vpc/vpc-1" $R $RT
case_ implicitDeny "create subnet in another VPC"             ec2:CreateSubnet "$A:vpc/vpc-9" $R aws:ResourceTag/Project=other
case_ allowed      "tag on create (RunInstances)"             ec2:CreateTags "$A:instance/i-1" $R ec2:CreateAction=RunInstances
case_ implicitDeny "tag another project's instance"           ec2:CreateTags "$A:instance/i-9" $R aws:ResourceTag/Project=other
case_ allowed      "remove a Name tag"                        ec2:DeleteTags "$A:instance/i-1" $R $RT aws:TagKeys=Name
case_ implicitDeny "remove the Project tag"                   ec2:DeleteTags "$A:instance/i-1" $R $RT aws:TagKeys=Project
case_ allowed      "describe instances in us-east-2"          ec2:DescribeInstances "*" $R
case_ implicitDeny "describe instances in us-west-2"          ec2:DescribeInstances "*" aws:RequestedRegion=us-west-2
case_ allowed      "quota check"                              servicequotas:GetServiceQuota "*"
case_ allowed      "who am I"                                 sts:GetCallerIdentity "*"
case_ implicitDeny "create an IAM user"                       iam:CreateUser "*"
case_ implicitDeny "attach a policy to itself"                iam:AttachUserPolicy "*"
case_ implicitDeny "read S3"                                  s3:GetObject "*"

echo "passed $pass, failed $fail"
((fail == 0))
