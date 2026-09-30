#!/usr/bin/env bash
set -euo pipefail

# Launches a single GPU instance (default: g4dn.xlarge, NVIDIA T4, 16GB VRAM)
# to run the full baseline QG training (scripts/train_baseline_qg.py) at a
# scale that's impractical on the local 4GB GTX 1650 -- see the project
# conversation for the throughput numbers behind that call.
#
# Requires: the "Running On-Demand G and VT instances" service quota
# (ec2, L-DB2E81BA) to be >= 4 in this region. New/unused accounts default
# to 0. Check with:
#   aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1
# and request an increase with:
#   aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 --region us-east-1
#
# Cost: check current on-demand pricing before running -- this was roughly
# US$0.53/hour for g4dn.xlarge in us-east-1 at the time this script was
# written. A full training run was estimated at 6-10 hours.
#
# Safety net: in addition to --instance-initiated-shutdown-behavior=terminate,
# this schedules a hard auto-shutdown via user-data (default 12h) so a
# forgotten instance can't run up an unbounded bill. If a legitimate run
# needs more time, ssh in and run "sudo shutdown -c" before the deadline.

REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
SAFETY_SHUTDOWN_MINUTES="${SAFETY_SHUTDOWN_MINUTES:-720}"
KEY_NAME="ic-tcqg-training-key"
SECURITY_GROUP_NAME="ic-tcqg-ssh"
ROOT_VOLUME_GB=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_PATH="$SCRIPT_DIR/${KEY_NAME}.pem"
STATE_FILE="$SCRIPT_DIR/.instance_id"
IP_FILE="$SCRIPT_DIR/.instance_ip"

if [[ -f "$STATE_FILE" ]]; then
  echo "An instance is already tracked in $STATE_FILE ($(cat "$STATE_FILE"))." >&2
  echo "Terminate it first with 05_terminate_instance.sh, or remove the state file if it's stale." >&2
  exit 1
fi

echo "Checking GPU instance quota (Running On-Demand G and VT instances)..."
QUOTA=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region "$REGION" --query 'Quota.Value' --output text)
NEEDED=4
if ! python3 -c "exit(0 if float('$QUOTA') >= $NEEDED else 1)"; then
  echo "ERROR: on-demand G/VT vCPU quota is $QUOTA in $REGION, need >= $NEEDED for $INSTANCE_TYPE." >&2
  echo "Check for a pending increase request with:" >&2
  echo "  aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region $REGION" >&2
  exit 1
fi
echo "Quota OK ($QUOTA vCPUs available)."

echo "Resolving latest Deep Learning AMI (Ubuntu, GPU, PyTorch preinstalled)..."
AMI_INFO=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*" "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate)[-1].[ImageId,Name,RootDeviceName]' --output text)
AMI_ID=$(echo "$AMI_INFO" | cut -f1)
AMI_NAME=$(echo "$AMI_INFO" | cut -f2)
ROOT_DEVICE=$(echo "$AMI_INFO" | cut -f3)
echo "Using AMI: $AMI_ID ($AMI_NAME), root device $ROOT_DEVICE"

if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "Creating key pair $KEY_NAME..."
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
else
  echo "Key pair $KEY_NAME already exists in AWS; reusing it (make sure $KEY_PATH matches it locally)."
fi

MY_IP="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')/32"
echo "Your current public IP: $MY_IP"

VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query 'Subnets[0].SubnetId' --output text)
echo "Using default VPC $VPC_ID, subnet $SUBNET_ID"

SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  echo "Creating security group $SECURITY_GROUP_NAME..."
  SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SECURITY_GROUP_NAME" \
    --description "SSH access for IC T-CQG training instance" --vpc-id "$VPC_ID" --query 'GroupId' --output text)
fi

echo "Authorizing SSH from $MY_IP on $SG_ID (safe to ignore a 'Duplicate' error)..."
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null 2>&1 || true

USERDATA_FILE="$(mktemp)"
cat > "$USERDATA_FILE" <<EOF
#!/bin/bash
# Hard safety net: shuts the instance down after ${SAFETY_SHUTDOWN_MINUTES} minutes
# regardless of training status, so a forgotten instance can't run up an
# unbounded AWS bill. To extend or cancel for a legitimate run: ssh in and
# run "sudo shutdown -c" before the deadline.
shutdown -h +${SAFETY_SHUTDOWN_MINUTES}
EOF

echo "Launching $INSTANCE_TYPE in $REGION (safety auto-shutdown in ${SAFETY_SHUTDOWN_MINUTES} min)..."
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=$ROOT_DEVICE,Ebs={VolumeSize=$ROOT_VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
  --user-data "file://$USERDATA_FILE" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ic-tcqg-training}]' \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$USERDATA_FILE"

echo "$INSTANCE_ID" > "$STATE_FILE"
echo "Instance $INSTANCE_ID launching. Waiting for it to enter 'running' state..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$PUBLIC_IP" > "$IP_FILE"

echo ""
echo "Instance running: $INSTANCE_ID at $PUBLIC_IP"
echo "SSH: ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
echo "(it may take a minute or two more for SSH to actually accept connections)"
echo ""
echo "Next: run ./02_deploy_and_train.sh"
echo ""
echo "IMPORTANT: this instance bills by the hour while running. A ${SAFETY_SHUTDOWN_MINUTES}-minute"
echo "safety auto-shutdown is scheduled, but don't rely on it -- run ./05_terminate_instance.sh"
echo "yourself as soon as you have what you need."
