#!/usr/bin/env bash
set -euo pipefail

# Terminates the tracked instance. This is what actually stops billing --
# run it as soon as you've fetched what you need with 04_fetch_results.sh.
# Termination deletes the root EBS volume too (DeleteOnTermination=true at
# launch), so make sure results are fetched first.

REGION="us-east-1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/.instance_id"
IP_FILE="$SCRIPT_DIR/.instance_ip"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "No tracked instance ($STATE_FILE missing) -- nothing to terminate."
  exit 0
fi

INSTANCE_ID=$(cat "$STATE_FILE")
echo "Terminating instance $INSTANCE_ID in $REGION..."
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
echo "Instance $INSTANCE_ID terminated. Billing for it stops now."

rm -f "$STATE_FILE" "$IP_FILE"
