#!/usr/bin/env bash
# Pull the competition archive, backing off through Kaggle's rate limiting.
# Per-file downloads get throttled to ~1.5 files/s, which is hours for 7000
# files; the bulk archive is one stream, so it is worth waiting out a 429.
set -u
OUT="${1:-data/raw}"
COMP=hyperspectral-object-detection-challenge-2026
mkdir -p "$OUT"
DELAY=60
for attempt in $(seq 1 12); do
  echo "[$(date +%H:%M:%S)] attempt $attempt (backoff ${DELAY}s on failure)"
  if kaggle competitions download -c "$COMP" -p "$OUT" 2>&1 | tee -a runs/bulk.log | tail -2; then
    if ls "$OUT"/*.zip >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] archive present: $(du -sh "$OUT" | cut -f1)"
      exit 0
    fi
  fi
  echo "[$(date +%H:%M:%S)] not yet; sleeping ${DELAY}s"
  sleep "$DELAY"
  DELAY=$(( DELAY < 480 ? DELAY * 2 : 600 ))
done
echo "gave up after 12 attempts"
exit 1
