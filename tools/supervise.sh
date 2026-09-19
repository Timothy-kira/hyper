#!/bin/bash
# Keep the Phase B orchestrator alive.
#
# It died once already, silently: nohup'd from a shell the harness later reaped,
# leaving session 1 running on Kaggle with nothing left locally to push session
# 2. Nothing would have reported that -- the run would simply have stopped.
# This restarts it with --auto-continue, which rejoins whatever session is
# furthest along rather than starting over.
cd /home/user/hyper
LOG=runs/phase_b.log
while true; do
  if ! pgrep -f "final_runs.py --tracks" > /dev/null 2>&1; then
    if grep -q "^  transformer" "$LOG" 2>/dev/null; then
      echo "[$(date -u +%H:%M)] run finished; supervisor stopping" >> runs/supervisor.log
      exit 0
    fi
    echo "[$(date -u +%H:%M)] orchestrator not running, restarting" >> runs/supervisor.log
    setsid nohup python -u tools/final_runs.py --tracks transformer --epochs 27 \
      --session-hours 10.5 --timeout-hours 11.9 --max-sessions 4 --auto-continue \
      >> "$LOG" 2>&1 < /dev/null &
    disown
  fi
  sleep 300
done
