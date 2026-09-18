#!/usr/bin/env bash
# Hand the loop over to the widened search space without losing measured nodes.
#
# The running loop holds the old candidate space in memory (no lda3/bandsel, no
# augmentation, no YOLO26 track). Restarting is only safe once its rollout has
# written the tree, so this waits for the rollout to end, stops the loop before
# it spends GPU on another old-space iteration, and brings up both tracks.
set -u
cd /home/user/hyper

echo "[$(date -u +%H:%M:%S)] waiting for the rollout to persist its tree"
for _ in $(seq 1 180); do
  if grep -q "dreaming over" runs/loop.log 2>/dev/null || ls runs/rsi/tree_*.json >/dev/null 2>&1; then
    break
  fi
  sleep 30
done

nodes=$(python3 - <<'PY' 2>/dev/null || echo 0
import json, glob
print(sum(len(json.load(open(f))["nodes"]) for f in glob.glob("runs/rsi/tree_*.json")))
PY
)
echo "[$(date -u +%H:%M:%S)] tree persisted with ${nodes} nodes; stopping the old loop"
pkill -f "dream_rsi.run" 2>/dev/null
sleep 5

# Both tracks share one quota. Today's allowance expires at the refresh, so the
# guard lets each spend what is left rather than reserving against it; the
# quota itself is the cap.
for track in transformer yolo26; do
  echo "[$(date -u +%H:%M:%S)] starting ${track} track"
  setsid nohup python3 -m dream_rsi.run \
    --executor kaggle --slug "xishengfeng/hod26-${track}" \
    --track "${track}" \
    --iterations 4 --workers 3 --max-rounds 3 --versions 40 \
    --proxy-train 300 --proxy-val 120 \
    --reserve-hours 7 --state-dir "runs/rsi_${track}" \
    > "runs/loop_${track}.log" 2>&1 < /dev/null &
  disown
  sleep 10
done
echo "[$(date -u +%H:%M:%S)] handoff complete"
