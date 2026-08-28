#!/bin/bash
while kill -0 3487041 2>/dev/null; do sleep 20; done; sleep 10
L=/mnt/disks/persist/pr4925_repro; T=/home/wenxindong_google_com/.claude/jobs/7ea88918/tmp
# C: MaxText-in-vLLM adapter engine prompt_logprobs (bf16, attn_dp=4, MoE TP-8)
rm -f $L/adapter.log; ATTN_DP=4 $T/wait_adapter.sh; cp $L/adapter.log $L/adapter_dp4.log
# D: FP8 (qwix) trainer, NL=8: own routing + replay with FP8-sampler indices
rm -f $L/mgen.log; NL=8 MODES=own,replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl8_fp8.npz SUFFIX=_fp8trainer EXTRA='{"quantization": "fp8", "use_qwix_quantization": true}' $T/wait_mgen.sh; cp $L/mgen.log $L/mgen_fp8trainer_nl8.log
echo CHAIN2_DONE > $L/chain2_done.flag
