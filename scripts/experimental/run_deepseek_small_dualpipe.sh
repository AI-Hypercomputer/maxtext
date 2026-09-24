#!/usr/bin/env bash
# Keep the regular DeepSeek preset with test size, topology, GA, schedule, and full remat.
# Explicitly select FP8 current scaling for dense GEMMs and MXFP8 for expert GEMMs.
# Keep PGLE disabled for a matched schedule comparison.
# Add --dry-run to inspect generated files without submitting a SLURM job.
set -euo pipefail

: "${LAUNCHER_DIR:?Set LAUNCHER_DIR to your maxtext-launcher checkout}"
: "${CLUSTER:?Set CLUSTER to a cluster configured in your launcher}"
MAXTEXT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 "${LAUNCHER_DIR}/launcher.py" deepseek-v3-671b \
  --cluster "${CLUSTER}" --nodes 1 --ntasks-per-node 4 \
  --code-dir "${MAXTEXT_DIR}" \
  --no-pgle \
  --ici-dp 1 --ici-fsdp 2 --ici-tp 1 --ici-expert 2 \
  --dcn-dp 1 --dcn-fsdp 1 --dcn-tp 1 --dcn-expert 1 \
  --maxtext-arg override_model_config=true \
  --maxtext-arg gradient_accumulation_schedule=dual_pipe \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg remat_policy=full \
  --maxtext-arg quantization=te_fp8_currentscaling \
  --maxtext-arg te_gmm_quantization=te_mxfp8 \
  --maxtext-arg base_num_decoder_layers=6 \
  --maxtext-arg first_num_dense_layers=1 \
  --maxtext-arg num_experts=16 \
  --tag deepseek-small-dualpipe-ga3 \
  "$@"
