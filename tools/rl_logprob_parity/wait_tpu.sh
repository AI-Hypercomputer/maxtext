#!/bin/bash
# usage: [ENV=...] wait_tpu.sh <log-name> <run_maxtext.sh|run_vllm.sh> <script.py> [args]
# waits until no process holds /dev/vfio/*, then runs the script; retries if the TPU was grabbed in between.
L=/mnt/disks/persist/pr4925_repro; D="$(cd "$(dirname "$0")" && pwd)"; LOG=$L/$1; shift
for attempt in $(seq 1 12); do
  while [ -n "$(sudo lsof -t /dev/vfio/* 2>/dev/null)" ]; do sleep 10; done
  sleep 5; echo "attempt $attempt start $(date)" > $LOG
  bash $D/$1 "${@:2}" >> $LOG 2>&1
  if ! grep -q "libtpu_lockfile\|Device or resource busy\|already in use" $LOG; then echo "attempt $attempt ran" >> $LOG; exit 0; fi
  echo "attempt $attempt: TPU taken; retry" >> $LOG; sleep 20
done
