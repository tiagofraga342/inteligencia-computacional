#!/usr/bin/env bash
set -euo pipefail

# Syncs this project to the running instance (see 01_launch_instance.sh),
# sets up the same Python environment as README.md's local Setup section,
# and starts the full baseline training in the background (nohup, survives
# SSH disconnect) with the full official SQuAD train split
# (--max-train-samples -1), which is the whole point of running this on a
# GPU with real VRAM headroom instead of the local GTX 1650.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KEY_PATH="$SCRIPT_DIR/ic-tcqg-training-key.pem"
IP_FILE="$SCRIPT_DIR/.instance_ip"
SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=accept-new)

if [[ ! -f "$IP_FILE" ]]; then
  echo "No running instance found ($IP_FILE missing). Run 01_launch_instance.sh first." >&2
  exit 1
fi
HOST="ubuntu@$(cat "$IP_FILE")"

echo "Waiting for SSH to be ready on $HOST..."
until ssh "${SSH_OPTS[@]}" "$HOST" true 2>/dev/null; do sleep 5; done
echo "SSH is up."

echo "Syncing project files (excluding .venv, data/cache, outputs/checkpoints, .git)..."
rsync -az --progress -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude 'data/cache' --exclude 'outputs/checkpoints' --exclude '.git' \
  "$PROJECT_ROOT/" "$HOST:~/inteligencia-computacional/"

echo "Setting up the Python environment on the instance..."
ssh "${SSH_OPTS[@]}" "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/inteligencia-computacional
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate sentencepiece evaluate rouge-score sacrebleu nltk pandas matplotlib
REMOTE

echo "Starting full training in the background..."
ssh "${SSH_OPTS[@]}" "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/inteligencia-computacional
source .venv/bin/activate
mkdir -p outputs/checkpoints outputs/logs
nohup python scripts/train_baseline_qg.py \
  --max-train-samples -1 \
  --output-dir outputs/checkpoints/baseline_qg_full \
  > outputs/logs/train_full.log 2>&1 &
disown
echo "Started training with PID $!"
REMOTE

echo ""
echo "Training launched on $HOST. Check progress with ./03_check_status.sh."
