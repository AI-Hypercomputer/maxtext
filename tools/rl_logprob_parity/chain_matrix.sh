#!/bin/bash
# after the 397B chain (3439499) and the 35B full-model chain (3469950): A) native bf16 NL=40 own/replay, B) native FP8 NL=8 + NL=40 own/replay
while kill -0 3469950 2>/dev/null; do sleep 20; done; sleep 10
L=/mnt/disks/persist/pr4925_repro; T=/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp
# A: torchax bf16 NL=40 -> maxtext NL=40 own+replay
rm -f $L/tgen.log; ATTN_DP=4 NL=40 MODEL=Qwen/Qwen3.5-35B-A3B $T/wait_tgen.sh; cp $L/tgen.log $L/tgen_bf16_nl40.log
rm -f $L/mgen.log; NL=40 MODES=own,replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl40.npz SUFFIX=_bf16 $T/wait_mgen.sh; cp $L/mgen.log $L/mgen_bf16_nl40.log
# B: torchax FP8 NL=8 and NL=40 -> maxtext replay with fp8 indices
rm -f $L/tgen.log; ATTN_DP=4 NL=8 MODEL=Qwen/Qwen3.5-35B-A3B-FP8 $T/wait_tgen.sh; cp $L/tgen.log $L/tgen_fp8_nl8.log
rm -f $L/mgen.log; NL=8 MODES=replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl8_fp8.npz SUFFIX=_fp8replay $T/wait_mgen.sh; cp $L/mgen.log $L/mgen_fp8_nl8.log
rm -f $L/tgen.log; ATTN_DP=4 NL=40 MODEL=Qwen/Qwen3.5-35B-A3B-FP8 $T/wait_tgen.sh; cp $L/tgen.log $L/tgen_fp8_nl40.log
rm -f $L/mgen.log; NL=40 MODES=replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl40_fp8.npz SUFFIX=_fp8replay $T/wait_mgen.sh; cp $L/mgen.log $L/mgen_fp8_nl40.log
echo MATRIX_DONE > $L/matrix_done.flag
