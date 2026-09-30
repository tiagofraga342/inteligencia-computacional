#!/usr/bin/env bash
set -euo pipefail

# Pulls the trained checkpoint and logs back from the instance. Kept in a
# separate directory (baseline_qg_full_from_aws) so it never clobbers
# whatever local checkpoint already exists at outputs/checkpoints/baseline_qg.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_PATH="$SCRIPT_DIR/ic-tcqg-training-key.pem"
IP_FILE="$SCRIPT_DIR/.instance_ip"

if [[ ! -f "$IP_FILE" ]]; then
  echo "No tracked instance ($IP_FILE missing)." >&2
  exit 1
fi
HOST="ubuntu@$(cat "$IP_FILE")"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=accept-new)

DEST="$PROJECT_ROOT/outputs/checkpoints/baseline_qg_full_from_aws"
LOG_DEST="$PROJECT_ROOT/outputs/logs_from_aws"
mkdir -p "$DEST" "$LOG_DEST"

echo "Fetching checkpoint from $HOST..."
rsync -az --progress -e "ssh ${SSH_OPTS[*]}" \
  "$HOST:~/inteligencia-computacional/outputs/checkpoints/baseline_qg_full/" "$DEST/"

echo "Fetching logs..."
rsync -az --progress -e "ssh ${SSH_OPTS[*]}" \
  "$HOST:~/inteligencia-computacional/outputs/logs/" "$LOG_DEST/"

echo ""
echo "Done. Checkpoint at $DEST"
echo "Next: python scripts/evaluate_qg.py --checkpoint $DEST/final"
