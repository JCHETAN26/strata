# IAM for the benchmark session

`strata-terraform-policy.json` is the least-privilege policy for a dedicated IAM user,
`strata-terraform`, that runs `aws/terraform` and `aws/scripts`. Don't run the session as an admin
or root identity.

## What it allows

- **Read:**
  - `ec2:Describe*` calls that Terraform and `check_clean.sh` make, in **us-east-2 only**.
  - `sts:GetCallerIdentity`.
  - `servicequotas:GetServiceQuota` (the vCPU check in `up.sh`).
- **Create:** only resources tagged `Project=strata-bench` at creation. That covers the VPC,
  internet gateway, subnet, route table, security group, key pair, placement group, instances and
  their root volumes. Subnets, route tables and security groups can be created only inside a VPC
  that has that tag.
- **Launch, only:**
  - c7i.8xlarge, c7i.2xlarge, or c7i.xlarge (the plan's three types);
  - on-demand;
  - with IMDSv2 required;
  - from Canonical's images (owner 099720109477);
  - into the tagged subnet, security group, key pair, and placement group;
  - with gp3 root volumes of at most 150 GB.
- **Modify, delete, terminate:** only resources tagged `Project=strata-bench`.
- **Tags:** tagging is allowed at creation, and retagging only on tagged resources. The
  `Project` tag itself can never be removed, so nothing can escape these conditions.
- **Nothing else:** no IAM, S3, or other services, and no other regions.

The policy is about 5 KB, under the 6 KB managed-policy limit. Account IDs in the ARNs are
wildcards, so the file holds nothing account-specific.

## Checked

- **IAM Access Analyzer `validate-policy`:** no errors, warnings, or suggestions.
- **`simulate_policy.sh`:** 37 cases through the IAM policy simulator (read-only), all as
  expected. It covers everything the session must do, and a list of things it must not: other
  instance types, spot, other regions, untagged resources, another project's instances or key
  pair, removing the `Project` tag, IAM, and S3. Rerun it after any edit.

Not checkable without creating the user: whether every API call the Terraform AWS provider makes
is listed. The policy relies on the provider tagging resources *at creation* (`TagSpecifications`),
which provider v5 does for every resource type used here. If a call is missing, the error names
it (`UnauthorizedOperation ... ec2:SomeAction`). Add that one action to the matching statement and
rerun `simulate_policy.sh`. Run `aws/scripts/up.sh main` (plan only) first: it exercises every
read call before anything is created.

Spot is denied on purpose (the decision was on-demand). Switching `market` to `spot` would need
`ec2:InstanceMarketType` widened, plus `iam:CreateServiceLinkedRole` for `spot.amazonaws.com` on
first use.

## Setting it up (you run these; nothing here runs them)

```sh
# with an admin identity, once
aws iam create-user --user-name strata-terraform
aws iam put-user-policy --user-name strata-terraform --policy-name strata-bench \
  --policy-document file://aws/iam/strata-terraform-policy.json
aws iam create-access-key --user-name strata-terraform     # shown once

# store the key in a named profile: ~/.aws/credentials, never in this repo
aws configure --profile strata                             # region us-east-2

# every session command then runs as that user
export AWS_PROFILE=strata
aws sts get-caller-identity                                # should show user/strata-terraform

# after the session
aws iam delete-access-key --user-name strata-terraform --access-key-id <id>
```

The scripts and Terraform read credentials only from the standard AWS credential chain
(`AWS_PROFILE`, `~/.aws`). Nothing in `aws/` asks for, stores, or prints keys. The instances have
no IAM role, so no AWS credentials exist on them, and `collect.sh` scans everything copied back
for secrets before it reaches `results/`.
