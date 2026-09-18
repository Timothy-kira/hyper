#!/usr/bin/env bash
# Kaggle caps concurrent GPU sessions at 2. A stopped loop does not stop its
# kernel, so a slot can stay held for a while after the orchestrator is gone.
set -u
DIR="$1"; SLUG="$2"
for i in $(seq 1 90); do
  out=$(kaggle kernels push -p "$DIR" 2>&1)
  if ! grep -q "session count" <<<"$out"; then
    echo "[$(date -u +%H:%M:%S)] pushed: $(tail -1 <<<"$out")"
    exit 0
  fi
  echo "[$(date -u +%H:%M:%S)] slots full; retrying in 60s"
  sleep 60
done
echo "gave up waiting for a free session slot"
exit 1
