#!/usr/bin/env bash
set -euo pipefail

# Quick health check of the remote training run: GPU utilization, whether the
# process is still alive, and the tail of its log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_PATH="$SCRIPT_DIR/ic-tcqg-training-key.pem"
IP_FILE="$SCRIPT_DIR/.instance_ip"

if [[ ! -f "$IP_FILE" ]]; then
  echo "No tracked instance ($IP_FILE missing)." >&2
  exit 1
fi
HOST="ubuntu@$(cat "$IP_FILE")"

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=accept-new "$HOST" bash -s <<'REMOTE'
cd ~/inteligencia-computacional
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo ""
echo "--- training process ---"
pgrep -af train_baseline_qg.py || echo "(not running)"
echo ""
echo "--- last 20 log lines ---"
tail -n 20 outputs/logs/train_full.log 2>/dev/null || echo "(no log yet)"
REMOTE
