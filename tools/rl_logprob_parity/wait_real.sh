#!/bin/bash
while kill -0 1665431 2>/dev/null; do sleep 15; done
echo "holder exited at $(date)" > /mnt/disks/persist/pr4925_repro/real35b.log
for attempt in 1 2 3 4 5 6; do sleep 15; bash /home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/run_real.sh >> /mnt/disks/persist/pr4925_repro/real35b.log 2>&1
  if ! grep -q "Device or resource busy\|already in use\|libtpu_lockfile" /mnt/disks/persist/pr4925_repro/real35b.log; then echo "attempt $attempt ran" >> /mnt/disks/persist/pr4925_repro/real35b.log; exit 0; fi
  echo "attempt $attempt: busy" >> /mnt/disks/persist/pr4925_repro/real35b.log; sleep 160; done
