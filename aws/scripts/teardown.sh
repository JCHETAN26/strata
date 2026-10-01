#!/usr/bin/env bash
# Collect results, destroy everything Terraform created, and verify nothing remains.
#
#   aws/scripts/teardown.sh             # collect, destroy, check
#   aws/scripts/teardown.sh --no-collect
#
# Between stages (main, then cluster) this tears down the whole session, network included; the
# next up.sh recreates it in about a minute. Ends with check_clean.sh, whose exit code is this
# script's: non-zero means something is still running (and billing).
source "$(dirname "$0")/lib.sh"

if [[ "${1:-}" != --no-collect ]] && [[ "$(tf_out stage 2>/dev/null || true)" =~ ^(main|cluster)$ ]]; then
  "$AWS_DIR/scripts/collect.sh" || log "warning: collect failed; continuing with teardown"
fi

my_ip=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]' || echo 127.0.0.1)
tf destroy -input=false -auto-approve -var "stage=none" -var "operator_cidr=$my_ip/32"
rm -f "$TF_DIR/stage.tfplan"
"$AWS_DIR/scripts/check_clean.sh"
