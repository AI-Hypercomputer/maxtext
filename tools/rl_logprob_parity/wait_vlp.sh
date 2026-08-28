#!/bin/bash
for attempt in $(seq 1 12); do
  while [ -n "$(sudo lsof -t /dev/vfio/* 2>/dev/null)" ]; do sleep 10; done
  sleep 5; echo "attempt $attempt start $(date)" > /mnt/disks/persist/pr4925_repro/vlp.log
  bash /home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/run_vlp.sh >> /mnt/disks/persist/pr4925_repro/vlp.log 2>&1
  if ! grep -q "libtpu_lockfile\|Device or resource busy\|already in use" /mnt/disks/persist/pr4925_repro/vlp.log; then echo "attempt $attempt ran" >> /mnt/disks/persist/pr4925_repro/vlp.log; exit 0; fi
  echo "attempt $attempt: TPU taken; retry" >> /mnt/disks/persist/pr4925_repro/vlp.log; sleep 20
done
