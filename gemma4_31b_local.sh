#!/bin/bash
# Local (no-xpk) smoke + A/B test for the gemma4 scan/remat rewrite on a single TPU host (v4-8/v5p-8).
#
# What it validates (before the cluster run in gemma4_31b_xpk_test.sh):
#   * The new Gemma4ScannableBlock path compiles & trains end-to-end:
#       - inner local-layer scan with per-layer jax.checkpoint,
#       - global-layer scan with its own jax.checkpoint (symmetric with locals),
#       - block-level remat skipped for gemma4 (skip_block_remat=True), and
#       - the outer block scan unrolled by base_num_decoder_layers/inhomogeneous_layer_cycle_interval.
#   * Numerics: UNROLLED vs ROLLED block scan produce matching loss curves (identical math; unroll is
#     only an XLA scheduling/liveness hint). The unroll is toggled purely via the config flag, no code
#     change, exercising the derived-unroll formula both ways:
#       inhomogeneous_layer_cycle_interval=6           -> block_unroll = NUM_LAYERS/6  (fully unrolled)
#       inhomogeneous_layer_cycle_interval=NUM_LAYERS  -> block_unroll = 1             (fully rolled)
#
# v4-8 reports 4 megacore devices -> a 2x2 fsdp x fsdp_transpose mesh (engages 2-axis weight sharding
# like the cluster). gemma4-31b is cut to a few blocks to fit one host. Perf here is meaningless; this
# is a correctness/compile smoke test only.
#
# Usage:
#   ./gemma4_31b_local.sh                        # runs unrolled + rolled, compares loss
#   NUM_LAYERS=18 STEPS=10 ./gemma4_31b_local.sh # NUM_LAYERS must be a multiple of 6 (block = 5 local + 1 global)
#   REMAT=save_out_proj ./gemma4_31b_local.sh    # exercise a non-full (per-layer) remat policy
#   MODE=unrolled ./gemma4_31b_local.sh          # run only one side (unrolled|rolled|both)
#   ./gemma4_31b_local.sh model_name=gemma4-26b  # extra args are appended to both runs

set -euo pipefail

# --- Tunables (override via env) ---
: "${OUTDIR:=/mnt/disks/pd2/gemma4_31b_local_test}"  # outside the repo to keep the tree clean
: "${NUM_LAYERS:=12}"   # multiple of 6; 12 -> 2 blocks, each 5 local + 1 global
: "${STEPS:=8}"
: "${PDBS:=1}"          # per_device_batch_size
: "${SEQ:=1024}"        # max_target_length
: "${FSDP:=2}"
: "${FSDP_T:=2}"
: "${REMAT:=full}"      # full | save_out_proj | save_qkv_proj | qkv_proj_offloaded | minimal_offloaded | ...
: "${MODE:=both}"       # both | unrolled | rolled

if (( NUM_LAYERS % 6 != 0 )); then
  echo "WARNING: NUM_LAYERS=${NUM_LAYERS} is not a multiple of 6; trailing layers go through the"
  echo "         unscanned remainder path (still valid, but not what this test targets)."
fi

export PYTHONPATH="./src/:.${PYTHONPATH:+:$PYTHONPATH}"

# attention=dot_product keeps the tiny local run robust (splash/flash tuning is a cluster concern and
# is orthogonal to the scan/remat structure under test). Two-axis FSDP is preserved via the 2x2 mesh.
COMMON_ARGS="\
  model_name=gemma4-31b \
  override_model_config=True \
  base_num_decoder_layers=${NUM_LAYERS} \
  per_device_batch_size=${PDBS} \
  max_target_length=${SEQ} \
  ici_fsdp_parallelism=${FSDP} \
  ici_fsdp_transpose_parallelism=${FSDP_T} \
  dataset_type=synthetic \
  attention=dot_product \
  pure_nnx_decoder=True \
  scan_layers=True \
  remat_policy=${REMAT} \
  enable_checkpointing=False \
  async_checkpointing=False \
  gcs_metrics=False \
  steps=${STEPS} \
  $*"

run_one() {
  # $1 = tag ; $2 = inhomogeneous_layer_cycle_interval (controls block_unroll = NUM_LAYERS/$2)
  local tag="$1"
  local cycle="$2"
  local run_out="${OUTDIR}/${tag}"
  local hlo_dir="${run_out}/hlo"
  rm -rf "${run_out}"
  mkdir -p "${hlo_dir}"

  echo "============================================================"
  echo ">>> ${tag}: inhomogeneous_layer_cycle_interval=${cycle}  (block_unroll=$((NUM_LAYERS / cycle)))"
  echo ">>> output: ${run_out}"
  echo "============================================================"

  # Dump optimized HLO so the scan structure can be inspected (unrolled block loop vs one while-body).
  XLA_FLAGS="--xla_dump_to=${hlo_dir}" \
  python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
    ${COMMON_ARGS} \
    inhomogeneous_layer_cycle_interval=${cycle} \
    run_name="${tag}" \
    base_output_directory="${OUTDIR}" \
    2>&1 | tee "${run_out}/run.log"
}

[[ "${MODE}" == "both" || "${MODE}" == "unrolled" ]] && run_one "unrolled" 6
[[ "${MODE}" == "both" || "${MODE}" == "rolled"   ]] && run_one "rolled"   "${NUM_LAYERS}"

echo
echo "############################################################"
echo "# RESULT SUMMARY"
echo "############################################################"
last_step_log() { grep -E "completed step: $((STEPS - 1))" "${1}/run.log" 2>/dev/null | tail -1 || true; }
if [[ "${MODE}" == "both" ]]; then
  echo "Last-step log (UNROLLED): $(last_step_log "${OUTDIR}/unrolled")"
  echo "Last-step log (ROLLED  ): $(last_step_log "${OUTDIR}/rolled")"
  echo "  -> losses should match closely (identical math; unroll only changes XLA structure)."
else
  echo "Last-step log (${MODE}): $(last_step_log "${OUTDIR}/${MODE}")"
fi
echo
echo "Checks:"
echo "  * Both runs reach step $((STEPS - 1)) with finite, decreasing loss and no OOM / UnexpectedTracerError."
echo "  * HLO dumps: ${OUTDIR}/{unrolled,rolled}/hlo/*after_optimizations*.txt"
echo "      - rolled:   expect the block loop as a single while-body (repeated ${NUM_LAYERS}/${NUM_LAYERS}=1)."
echo "      - unrolled: expect the block body replicated $((NUM_LAYERS / 6))x straight-line."
