variable "region" {
  description = "AWS region. us-east-2 (Ohio): same on-demand price as us-east-1, cheaper spot, all needed types in every AZ."
  type        = string
  default     = "us-east-2"
}

variable "availability_zone" {
  description = "One AZ for everything, so cluster traffic stays inside it (free, low latency)."
  type        = string
  default     = "us-east-2a"
}

variable "stage" {
  description = <<-EOT
    What to run. The account's vCPU quota is 32 per region, so the stages never overlap:
      "none"    - network only (or nothing running); use between stages
      "main"    - one c7i.8xlarge (32 vCPU) for parts A-E
      "cluster" - 4 shards + coordinator + client (28 vCPU) for part F
  EOT
  type        = string
  default     = "none"
  validation {
    condition     = contains(["none", "main", "cluster"], var.stage)
    error_message = "stage must be none, main, or cluster."
  }
}

variable "operator_cidr" {
  description = "Your public IP as a /32: the only address allowed to SSH in. aws/scripts/up.sh fills it in."
  type        = string
  validation {
    condition     = can(cidrhost(var.operator_cidr, 0)) && endswith(var.operator_cidr, "/32")
    error_message = "operator_cidr must be a single address, like 203.0.113.7/32."
  }
}

variable "ssh_public_key_path" {
  description = "Public half of the key used to SSH in (aws/scripts/up.sh creates aws/.ssh/strata-bench)."
  type        = string
  default     = "../.ssh/strata-bench.pub"
}

variable "market" {
  description = "on-demand (recommended: an interruption mid-run costs more than spot saves) or spot."
  type        = string
  default     = "on-demand"
  validation {
    condition     = contains(["on-demand", "spot"], var.market)
    error_message = "market must be on-demand or spot."
  }
}

variable "main_instance_type" {
  description = "Parts A-E. 16 physical cores (32 vCPU, Sapphire Rapids, AVX2 + AVX-512), 64 GiB."
  type        = string
  default     = "c7i.8xlarge"
}

variable "main_volume_gb" {
  description = "Datasets (~8 GB), index snapshots (~10 GB), vcpkg build trees (~15 GB), results."
  type        = number
  default     = 150
}

variable "shard_count" {
  description = "Shard machines in the cluster stage (the run uses 1, 2, then all of them)."
  type        = number
  default     = 4
}

variable "shard_instance_type" {
  description = "2 physical cores, 8 GiB: SIFT1M (~0.7 GB in memory) fits on one."
  type        = string
  default     = "c7i.xlarge"
}

variable "coordinator_instance_type" {
  type    = string
  default = "c7i.xlarge"
}

variable "client_instance_type" {
  description = "The load generator: bigger than a shard so it is never the bottleneck."
  type        = string
  default     = "c7i.2xlarge"
}

variable "cluster_volume_gb" {
  type    = number
  default = 30
}

variable "max_lifetime_hours" {
  description = <<-EOT
    Safety net: each instance schedules its own shutdown this many hours after boot, and shutdown
    terminates it. Covers a forgotten teardown. Main stage plan: ~14.5 h with contingency (the
    references run twice, at AVX2 and at AVX-512); cluster: ~2.5 h.
  EOT
  type        = number
  default     = 18
}
