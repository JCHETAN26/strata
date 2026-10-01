terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
  # Local state (aws/terraform/terraform.tfstate, gitignored). One person, one short session:
  # a remote backend would be one more resource to create and remember to delete.
}

provider "aws" {
  region = var.region
  # Every resource carries these tags; aws/scripts/check_clean.sh finds leftovers by them.
  default_tags {
    tags = {
      Project   = "strata-bench"
      ManagedBy = "terraform"
    }
  }
}
