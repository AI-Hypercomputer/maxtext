#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AOT memory check of the e2e trainer program (Tunix GRPO loss, packed micro-batch with router replay,
# fwd_bwd / accumulate / update) with maxtext.training_engine.maxtext_engine_compile.
#
#   LAUNCH_RENDER=<file> [IMAGE=<image>] [TRAINER_SLICE=4x8x8] [COMPILE_TOPOLOGY=tpu7x-512]
#   [EXTRA_MAXTEXT_FLAGS=... REPLACE_MAXTEXT_FLAGS=... LIBTPU_EXTRA=... LIBTPU_DROP=... COMPUTE_LOGPS_CHUNK_SIZE=...]
#   [LOG=<file>] bash run_trainer_compile.sh [extra MaxText key=value ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
: "${LAUNCH_RENDER:?set LAUNCH_RENDER to the DRY_RUN output of your e2e launch}"
_aot_env=$("${PYTHON:-python3}" "$SCRIPT_DIR/render_trainer_jobset.py" --print-aot-env --launch-render "$LAUNCH_RENDER" \
  --extra-maxtext-flags "${EXTRA_MAXTEXT_FLAGS:-}" --replace-maxtext-flags "${REPLACE_MAXTEXT_FLAGS:-}" \
  --libtpu-extra="${LIBTPU_EXTRA:-}" --libtpu-drop="${LIBTPU_DROP:-}" --trainer-slice "${TRAINER_SLICE:-}" \
  --compute-logps-chunk-size "${COMPUTE_LOGPS_CHUNK_SIZE:-}")
eval "$_aot_env"

_run_name="$(date -u +%Y%m%dT%H%M%SZ)"
_out_dir="/tmp/${_run_name}"
mkdir -p "$_out_dir"
_ga=$(( GLOBAL_BATCH / TRAIN_MICRO_BATCH_SIZE ))
_ndev=${NUM_DEVICES:-$(( MESH_FSDP * MESH_TP * MESH_EXPERT * MESH_CONTEXT ))}
_pdb=$("${PYTHON:-python3}" -c "print((${MESH_FSDP} * ${MESH_EXPERT}) / ${_ndev})")
_grpo="{num_generations: 16, beta: 0.0, epsilon: 0.2, epsilon_high: 0.28, use_rollout_logps: false, exact_token_continuity: true, loss_agg_mode: token-mean, advantage_estimator: grpo-loo, seq_logprob_error_threshold: 2.0, sampler_is: token, sampler_rs: geometric, sampler_rs_min: 0.999, sampler_rs_max: 1.002, temperature: 1.0}"
read -r -a _extra_flags <<< "${MAXTEXT_EXTRA_FLAGS:-}"
LOG=${LOG:-logs/${_run_name}.log}
mkdir -p "$(dirname "$LOG")"
echo "log: $LOG"
{ echo "=== $(date -u -Is) aot_bench RUN_NAME=$_run_name TOPOLOGY=$COMPILE_TOPOLOGY MODEL=$MODEL_NAME FSDP=$MESH_FSDP TP=$MESH_TP EXPERT=$MESH_EXPERT CONTEXT=$MESH_CONTEXT IMAGE=${IMAGE:-local}"
  echo "MAXTEXT_EXTRA_FLAGS=$MAXTEXT_EXTRA_FLAGS"; echo "LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS"; } | tee -a "$LOG"

if [[ -n "${IMAGE:-}" ]]; then
  _cmd=(docker run --rm --entrypoint python
    -v "$REPO_ROOT:/workspace" -w /workspace
    -e JAX_PLATFORMS=cpu,tpu -e LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS" -e TF_CPP_MIN_LOG_LEVEL=1
    "$IMAGE")
  _base_yml="src/maxtext/configs/base.yml"
else
  _cmd=(env JAX_PLATFORMS=cpu,tpu LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS" TF_CPP_MIN_LOG_LEVEL=1
    "${PYTHON:-python3}")
  _base_yml=$("${PYTHON:-python3}" -c 'import os, maxtext.configs.pyconfig as p; print(os.path.join(os.path.dirname(p.__file__), "base.yml"))' 2>/dev/null || echo "$REPO_ROOT/src/maxtext/configs/base.yml")
fi

"${_cmd[@]}" -m maxtext.training_engine.maxtext_engine_compile \
  "$_base_yml" \
  "model_name=$MODEL_NAME" "run_name=$_run_name" "base_output_directory=$_out_dir" \
  enable_checkpointing=False "profiler=''" profiler_steps=0 skip_first_n_steps_for_profiler=-1 \
  adam_b1=0.9 adam_b2=0.999 adam_eps=1e-8 adam_weight_decay=0.0 gradient_clipping_threshold=0.0625 \
  scan_layers=True convert_checkpoint_if_possible=False skip_jax_distributed_system=True \
  allow_split_physical_axes=True "per_device_batch_size=$_pdb" "gradient_accumulation_steps=$_ga" \
  "max_target_length=$MAX_TARGET_LENGTH" attention=flash use_tokamax_gmm=true use_gmm_v2=true \
  "ici_fsdp_parallelism=$MESH_FSDP" "ici_tensor_parallelism=$MESH_TP" \
  "ici_expert_parallelism=$MESH_EXPERT" "ici_context_parallelism=$MESH_CONTEXT" \
  override_model_config=True context_parallel_load_balance=False \
  learning_rate=2e-6 warmup_steps_fraction=0.0 dtype=bfloat16 weight_dtype=float32 grad_dtype=float32 \
  enable_tensorboard=False record_internal_nn_metrics=False init_weights_seed=42 \
  prefuse_moe_weights=True use_weight_converter=True rollout_tensor_parallelism=1 \
  checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false \
  "trainable_parameters_mask=^(?!.*routed_experts/gate/kernel).*" remat_policy=custom \
  learning_rate_final_fraction=1.0 skip_step_on_spikes=False abort_on_nan_loss=True \
  float32_gate_logits=True float32_logits=True ${_extra_flags[@]+"${_extra_flags[@]}"} \
  "compile_topology=$COMPILE_TOPOLOGY" compile_topology_num_slices=1 loss=grpo \
  "grpo_config=$_grpo" "logps_chunk_size=${LOGPS_CHUNK_SIZE:-512}" \
  "max_seq_token_per_tpu=$MAX_SEQ_TOKEN_PER_TPU" \
  "max_segments_per_packed_row=$MAX_SEGMENTS_PER_PACKED_ROW" \
  "router_replay=$ROUTER_REPLAY" \
  "max_prefill_predict_length=$MAX_PREFILL_PREDICT_LENGTH" \
  decode_sampling_temperature=1.0 \
  "$@" 2>&1 | tee -a "$LOG"
