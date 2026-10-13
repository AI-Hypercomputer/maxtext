#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AOT compile (CPU, no TPU) of the Tunix RL MaxText trainer program for a virtual TPU topology.

Builds the MaxText argv as the Tunix trainer node does (`tunix.utils.maxtext_utils.build_maxtext_config`, with
MAXTEXT_EXTRA_FLAGS from the environment), appends the `compile_*` keys and calls MaxText's
`training_engine/maxtext_engine_compile.py`, which compiles fwd_bwd, accumulate and update with Tunix's GRPO loss on a
sequence-packed RLTrainerPayload with router replay, and reports each kernel's memory.

The recipe constants below are those of the Qwen3.5-397B-A17B GRPO recipe the bench was built for (prompt and
response lengths, optimizer, GRPO config). The mesh, the logps chunk and the topology come from the environment
(`render_e2e.py --print-aot-env` derives the first two from an e2e DRY_RUN render); edit the rest for another
recipe so that the compile matches your trainer. Run with `aot_e2e.sh`, inside an image whose MaxText and Tunix
match the e2e run.

Environment: MAXTEXT_EXTRA_FLAGS (required), LIBTPU_INIT_ARGS (--xla_* flags), AOT_MESH_FSDP (32), AOT_MESH_TP (1),
AOT_MESH_EXPERT (2), AOT_MESH_CONTEXT (4), AOT_NUM_DEVICES (fsdp x tp x expert x context), AOT_COMPILE_TOPOLOGY
(tpu7x-256), AOT_LOGPS_CHUNK_SIZE (512), AOT_TRAIN_MICRO_BATCH_SIZE (64), AOT_GLOBAL_BATCH (1024).
Extra command-line arguments are appended to the MaxText argv as key=value overrides.
"""

import os
import sys


def main():
  extra = sys.argv[1:]
  os.environ.setdefault("TRAINER_MAXTEXT_ATTENTION", "flash")
  os.environ.setdefault("ENABLE_PATHWAYS_PERSISTENCE", "1")
  os.environ.setdefault("CHECKPOINT_ASYNC", "true")
  assert os.environ.get("MAXTEXT_EXTRA_FLAGS"), "MAXTEXT_EXTRA_FLAGS must be set"

  from maxtext.configs import pyconfig
  from tunix.utils import maxtext_utils as tunix_mu

  captured = {}
  real_initialize = pyconfig.initialize

  def capture(argv):
    captured["argv"] = list(argv)
    return None

  pyconfig.initialize = capture
  gbs = int(os.environ.get("AOT_GLOBAL_BATCH", str(64 * 16)))  # prompts x generations per optimizer step
  micro = int(os.environ.get("AOT_TRAIN_MICRO_BATCH_SIZE", "64"))
  fsdp, tp, ep, cp = (int(os.environ.get(f"AOT_MESH_{k}", d)) for k, d in
                      (("FSDP", "32"), ("TP", "1"), ("EXPERT", "2"), ("CONTEXT", "4")))
  try:
    tunix_mu.build_maxtext_config(
        model_name="qwen3.5-397b-a17b",
        worker_id="aot",
        train_micro_batch_size=micro,
        mesh_fsdp=fsdp, mesh_tp=tp, mesh_expert=ep, mesh_context=cp,
        num_devices=int(os.environ.get("AOT_NUM_DEVICES", str(fsdp * tp * ep * cp))),
        max_prompt_length=8192, max_response_length=57344,
        learning_rate=2e-6,
        warmup_steps_fraction=0.0,
        load_parameters_path="",
        padded_moe_mlp_dim=0,
        base_output_directory="/tmp/aot-maxtext-out",
        gradient_accumulation_steps=gbs // micro,
        checkpointing_options=type("C", (), {"save_interval_steps": 0, "max_to_keep": 10})(),
        rollout_mesh_tp=1,
        prefuse_moe_weights=True,
        use_weight_converter=True,
        max_seq_token_per_tpu=65536,
        trainable_parameters_mask="^(?!.*routed_experts/gate/kernel).*",
        base_num_kv_heads=2,
        attention="flash",
        remat_policy="custom",
        learning_rate_final_fraction=1.0,
        adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8, adam_weight_decay=0.0,
        gradient_clipping_threshold=0.0625,
        skip_step_on_spikes=False, skip_step_on_nan=True,
        float32_gate_logits=True, float32_logits=True,
    )
  finally:
    pyconfig.initialize = real_initialize

  argv = captured["argv"]
  grpo = (
      "{num_generations: 16, beta: 0.0, epsilon: 0.2, epsilon_high: 0.28, use_rollout_logps: false,"
      " exact_token_continuity: true, loss_agg_mode: token-mean, advantage_estimator: grpo-loo,"
      " overlong_loss_masking: true, seq_logprob_error_threshold: 2.0,"
      " truncated_importance_sampling_type: seq-mask-tis,"
      " truncated_importance_sampling_ratio_min: 0.999, truncated_importance_sampling_ratio: 1.002,"
      " temperature: 1.0}"
  )
  argv += [
      # tpu7x-256 is a 4x4x8 slice (128 chips, 256 devices); tpu7x-512 is 4x8x8 (MaxText accelerator_to_spec_map).
      f"compile_topology={os.environ.get('AOT_COMPILE_TOPOLOGY', 'tpu7x-256')}",
      "compile_topology_num_slices=1",
      "compile_engine_loss=grpo",
      f"compile_engine_grpo_config={grpo}",
      # Match the trainer's --compute_logps_chunk_size.
      f"compile_engine_logps_chunk_size={int(os.environ.get('AOT_LOGPS_CHUNK_SIZE', '512'))}",
      "compile_engine_max_seq_token_per_tpu=65536",
      "compile_engine_max_segments_per_packed_row=16",
      "compile_engine_router_replay=true",
      "max_prefill_predict_length=8192",
      "decode_sampling_temperature=1.0",
  ] + extra
  print("AOT argv:", argv, flush=True)
  import maxtext
  print("maxtext from:", maxtext.__file__, flush=True)

  from maxtext.training_engine import maxtext_engine_compile
  maxtext_engine_compile.main(argv)


if __name__ == "__main__":
  sys.exit(main())
