# Strata AWS benchmark session. See docs/aws-plan.md for what runs where, and aws/README.md for
# the step-by-step runbook. Nothing here is shared with anything else in the account: a dedicated
# VPC, tagged Project=strata-bench, removed by `terraform destroy`.

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical (verified: owns the noble 24.04 images)
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  main_count    = var.stage == "main" ? 1 : 0
  cluster_count = var.stage == "cluster" ? 1 : 0
  shard_count   = var.stage == "cluster" ? var.shard_count : 0

  # Auto-termination: shutdown -h after the lifetime; instance_initiated_shutdown_behavior turns
  # that shutdown into termination. Runs as cloud-init user data on first boot.
  user_data = <<-EOT
    #!/bin/bash
    shutdown -h +${var.max_lifetime_hours * 60} "strata-bench: max lifetime reached"
  EOT
}

# --- Network: one public subnet in one AZ ---------------------------------------------------------

resource "aws_vpc" "bench" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = "strata-bench" }
}

resource "aws_internet_gateway" "bench" {
  vpc_id = aws_vpc.bench.id
  tags   = { Name = "strata-bench" }
}

resource "aws_subnet" "bench" {
  vpc_id                  = aws_vpc.bench.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true
  tags                    = { Name = "strata-bench" }
}

resource "aws_route_table" "bench" {
  vpc_id = aws_vpc.bench.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.bench.id
  }
  tags = { Name = "strata-bench" }
}

resource "aws_route_table_association" "bench" {
  subnet_id      = aws_subnet.bench.id
  route_table_id = aws_route_table.bench.id
}

# SSH only from the operator's address; machines talk to each other freely (gRPC between
# coordinator, shards, and client, on private addresses). Nothing else is reachable.
resource "aws_security_group" "bench" {
  name        = "strata-bench"
  description = "strata-bench: SSH from the operator, all traffic within the group"
  vpc_id      = aws_vpc.bench.id

  ingress {
    description = "SSH from the operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.operator_cidr]
  }
  ingress {
    description = "Within the group (cluster gRPC)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }
  egress {
    description = "Package and dataset downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "strata-bench" }
}

resource "aws_key_pair" "bench" {
  key_name   = "strata-bench"
  public_key = file(var.ssh_public_key_path)
}

# Cluster stage: a cluster placement group puts the machines close together in the network, so
# measured latency is Strata's, not cross-rack distance.
resource "aws_placement_group" "cluster" {
  count    = local.cluster_count
  name     = "strata-bench-cluster"
  strategy = "cluster"
}

# --- Instances ------------------------------------------------------------------------------------

locals {
  common = {
    ami                    = data.aws_ami.ubuntu.id
    subnet_id              = aws_subnet.bench.id
    vpc_security_group_ids = [aws_security_group.bench.id]
    key_name               = aws_key_pair.bench.key_name
  }
}

resource "aws_instance" "main" {
  count                                = local.main_count
  instance_type                        = var.main_instance_type
  ami                                  = local.common.ami
  subnet_id                            = local.common.subnet_id
  vpc_security_group_ids               = local.common.vpc_security_group_ids
  key_name                             = local.common.key_name
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.user_data

  dynamic "instance_market_options" {
    for_each = var.market == "spot" ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.main_volume_gb
    encrypted             = true
    delete_on_termination = true
    # Root volumes don't inherit the provider's default_tags; tag them so check_clean.sh finds them.
    tags = { Project = "strata-bench", Name = "strata-bench" }
  }
  metadata_options {
    http_tokens = "required" # IMDSv2 only; bench/benchmeta.py reads the instance type with it
  }
  tags = { Name = "strata-bench-main", Role = "main" }
}

resource "aws_instance" "shard" {
  count                                = local.shard_count
  instance_type                        = var.shard_instance_type
  ami                                  = local.common.ami
  subnet_id                            = local.common.subnet_id
  vpc_security_group_ids               = local.common.vpc_security_group_ids
  key_name                             = local.common.key_name
  placement_group                      = aws_placement_group.cluster[0].id
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.user_data

  dynamic "instance_market_options" {
    for_each = var.market == "spot" ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.cluster_volume_gb
    encrypted             = true
    delete_on_termination = true
    # Root volumes don't inherit the provider's default_tags; tag them so check_clean.sh finds them.
    tags = { Project = "strata-bench", Name = "strata-bench" }
  }
  metadata_options {
    http_tokens = "required"
  }
  tags = { Name = "strata-bench-shard-${count.index}", Role = "shard" }
}

resource "aws_instance" "coordinator" {
  count                                = local.cluster_count
  instance_type                        = var.coordinator_instance_type
  ami                                  = local.common.ami
  subnet_id                            = local.common.subnet_id
  vpc_security_group_ids               = local.common.vpc_security_group_ids
  key_name                             = local.common.key_name
  placement_group                      = aws_placement_group.cluster[0].id
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.user_data

  dynamic "instance_market_options" {
    for_each = var.market == "spot" ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.cluster_volume_gb
    encrypted             = true
    delete_on_termination = true
    # Root volumes don't inherit the provider's default_tags; tag them so check_clean.sh finds them.
    tags = { Project = "strata-bench", Name = "strata-bench" }
  }
  metadata_options {
    http_tokens = "required"
  }
  tags = { Name = "strata-bench-coordinator", Role = "coordinator" }
}

resource "aws_instance" "client" {
  count                                = local.cluster_count
  instance_type                        = var.client_instance_type
  ami                                  = local.common.ami
  subnet_id                            = local.common.subnet_id
  vpc_security_group_ids               = local.common.vpc_security_group_ids
  key_name                             = local.common.key_name
  placement_group                      = aws_placement_group.cluster[0].id
  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = local.user_data

  dynamic "instance_market_options" {
    for_each = var.market == "spot" ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.cluster_volume_gb
    encrypted             = true
    delete_on_termination = true
    # Root volumes don't inherit the provider's default_tags; tag them so check_clean.sh finds them.
    tags = { Project = "strata-bench", Name = "strata-bench" }
  }
  metadata_options {
    http_tokens = "required"
  }
  tags = { Name = "strata-bench-client", Role = "client" }
}
