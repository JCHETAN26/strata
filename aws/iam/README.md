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
  - from Canonical's images: `ec2:Owner = amazon` and `aws:ResourceAccount = 099720109477`.
    For a verified provider's public AMI, AWS evaluates `ec2:Owner` as the alias `amazon`, not
    the owner's account ID (decoded from a real refusal). The account condition then narrows
    "Amazon or verified providers" to Canonical alone;
  - into the tagged subnet, security group, key pair, and placement group;
  - with gp3 root volumes of at most 150 GB.
- **Modify, delete, terminate:** only resources tagged `Project=strata-bench`.
- **`ModifyInstanceAttribute`, for exactly two things Terraform does**, on tagged instances only
  (scoped with `ec2:Attribute/<Name>`, which AWS populates with the requested value):
  - after launch, set the shutdown behavior to `terminate`. Never `stop`, which keeps the
    auto-termination safety net.
  - before terminating, turn termination protection **off**. Never on, so nothing can block
    teardown.

  Instance type, user data, security groups, and other attributes stay denied.
- **Tags:** tagging is allowed at creation, and retagging only on tagged resources. The
  `Project` tag itself can never be removed, so nothing can escape these conditions.
- **Nothing else:** no IAM, S3, or other services, and no other regions.

The policy is about 5 KB, under the 6 KB managed-policy limit. Account IDs in the ARNs are
wildcards, so the file holds nothing account-specific.

## Checked

- **IAM Access Analyzer `validate-policy`:** no errors, warnings, or suggestions.
- **`simulate_policy.sh`:** 39 cases through the IAM policy simulator (read-only), all as
  expected. The image cases use the context AWS actually evaluated. Run it with a profile that may
  call the simulator; the strata user deliberately may not. It covers everything the session must do, and a list of things it must not: other
  instance types, spot, other regions, untagged resources, another project's instances or key
  pair, removing the `Project` tag, IAM, and S3. Rerun it after any edit.

**Run `dryrun_launch.sh` before every apply.** The simulator only evaluates the context you give
it. The first apply showed what that
misses: I had modeled the image's `ec2:Owner` as Canonical's account ID, and the simulator agreed
with my model. **`dryrun_launch.sh` checks the real thing**, for the whole session, through
`--dry-run` (nothing is created, changed, or deleted). It covers:

- **A.** The launches: main, shard, client, and the placement group. Also refusals for a type
  outside the plan, spot, and a 500 GB root volume.
- **B.** Everything Terraform does after launch: the attribute changes, and the instance and
  root-volume tags. Also refusals for shutdown `stop`, termination protection on, user data, and
  removing the `Project` tag.
- **C.** Every read Terraform and the scripts make.
- **D.** The whole teardown: termination protection off, terminate, and every network delete.
- **E.** Scope: another project's key pair, an untagged VPC, another region.

The call list comes from the pinned provider's source (v5.100.0, `ec2_instance.go`; see the
script's header) and must be re-derived when the provider version changes. Any unexpected result
is decoded into the condition values AWS evaluated. Sections B and D need an instance in
Terraform's state, so run it after a launch, before destroying.

```sh
AWS_PROFILE=strata aws/iam/dryrun_launch.sh      # needs the network applied; must end "OK: every checked call ..."
```

Also not checkable without running as the user: whether every API call the Terraform AWS provider
makes is listed. The policy relies on the provider tagging resources *at creation* (`TagSpecifications`),
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
