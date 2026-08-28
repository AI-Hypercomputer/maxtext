#!/bin/bash
while kill -0 3491526 2>/dev/null; do sleep 20; done; sleep 10
L=/mnt/disks/persist/pr4925_repro; T=/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp
rm -f $L/t397.log; ATTN_DP=4 $T/wait_t397.sh; cp $L/t397.log $L/t397_run2.log; sleep 5
rm -f $L/m397.log; MODE=replay $T/wait_m397.sh; cp $L/m397.log $L/m397_replay.log
echo CHAIN3_DONE > $L/chain3_done.flag
