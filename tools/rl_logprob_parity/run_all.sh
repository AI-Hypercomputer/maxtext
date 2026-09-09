#!/bin/bash
# Exact invocation for every cell of the 35B / 397B tables, in dependency order. Each step needs the whole TPU.
set -e; L=/mnt/disks/persist/pr4925_repro; D="$(cd "$(dirname "$0")" && pwd)"; W=$D/wait_tpu.sh
# 0. real tokens + layer-3 hidden state (real35b_L3.npz) — input to everything below
$W real35b.log run_maxtext.sh run_35b_real.py
# 1. 35B bf16/bf16, layers 4 and 8 (sampler prefix also captures the expert ids used for replay)
ATTN_DP=4 $W tprefix.log run_vllm.sh torchax_prefix.py
MODES=own,replay $W mprefix.log run_maxtext.sh maxtext_prefix.py
# 2. 35B bf16/bf16, 40 layers (own + replay)
ATTN_DP=4 NL=40 MODEL=Qwen/Qwen3.5-35B-A3B $W tgen.log run_vllm.sh torchax_prefix_gen.py
NL=40 MODES=own,replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl40.npz SUFFIX=_bf16 $W mgen.log run_maxtext.sh maxtext_prefix_gen.py
# 3. 35B bf16 trainer / FP8 sampler, layers 4, 8 and 40 (own rows reuse the bf16 trainer outputs of steps 1-2; replay rows re-run the trainer with the FP8 sampler's expert ids)
ATTN_DP=4 NL=8  MODEL=Qwen/Qwen3.5-35B-A3B-FP8 $W tgen.log run_vllm.sh torchax_prefix_gen.py
NL=8  MODES=replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl8_fp8.npz  SUFFIX=_fp8replay $W mgen.log run_maxtext.sh maxtext_prefix_gen.py
ATTN_DP=4 NL=40 MODEL=Qwen/Qwen3.5-35B-A3B-FP8 $W tgen.log run_vllm.sh torchax_prefix_gen.py
NL=40 MODES=replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl40_fp8.npz SUFFIX=_fp8replay $W mgen.log run_maxtext.sh maxtext_prefix_gen.py
# 4. 35B FP8 trainer (qwix fp8_full, dynamic) / FP8 sampler, 40 layers (whole-model __call__ is required for qwix)
FULLCALL=1 NL=40 MODES=own,replay TX_NPZ=$L/torchax_prefix_dp4tp2_ep1_nl40_fp8.npz SUFFIX=_fp8trainer_full EXTRA='{"quantization": "fp8_full", "use_qwix_quantization": true}' $W mgen.log run_maxtext.sh maxtext_prefix_gen.py
# 5. 35B output tokens (decode path): engine rollouts, then trainer teacher-forcing (own routing; replay of the decode-position routing)
$W gen.log run_vllm.sh vllm_generate.py
TAG=bf16 $W score.log run_maxtext.sh maxtext_score.py
TAG=bf16 GEN_ONLY_REPLAY=1 SUFFIX=_genreplay2 $W score.log run_maxtext.sh maxtext_score.py
# 6. 35B MaxText-in-vLLM sampler (real engine prompt_logprobs) vs MaxText trainer full-model logprobs
ATTN_DP=4 ADAPTER=1 $W adapter.log run_vllm.sh adapter_prompt_logprobs.py
$W mfull.log run_maxtext.sh maxtext_full_logprobs.py
# 7. 397B bf16 trainer (8-layer prefix) / FP8 sampler, layers 4 and 8
ATTN_DP=4 $W t397.log run_vllm.sh torchax_prefix_397b.py
MODE=own    $W m397.log run_maxtext.sh maxtext_prefix_397b.py
MODE=replay $W m397.log run_maxtext.sh maxtext_prefix_397b.py
# 8. tables
python $D/build_tables.py > $L/tis_tables.md
python $D/compare_output.py bf16 _genreplay2
