# Copyright 2023-2026 Google LLC
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

"""Verify end-to-end OLMo 3.5 rollout-vs-trainer logprob & sync parity on TPU Pathways."""

import argparse
import json
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import pathwaysutils
from transformers import AutoTokenizer

import maxtext.integration.vllm.maxtext_vllm_adapter as adapter

adapter.register()

from flax import nnx  # noqa: E402
from maxtext.configs import types  # noqa: E402
from maxtext.integration.vllm.maxtext_vllm_rollout import (  # noqa: E402
    MaxTextVllmRollout,
    validate_direct_sync_layer_coverage,
)
from maxtext.trainers.post_train.rl import train_rl  # noqa: E402
from maxtext.utils import model_creation_utils  # noqa: E402
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # noqa: E402
from tunix.rl.rollout import base_rollout  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-name", default="olmo35-tiny-e256k8-sh2x")
  parser.add_argument(
      "--hf-config-dir",
      default=os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "olmo35_rl", "hf_config_tiny_e256"),
  )
  parser.add_argument("--tokenizer-path", default="allenai/Olmo-3-7B-Instruct")
  parser.add_argument("--rollout-tp", type=int, default=4)
  parser.add_argument("--rollout-dp", type=int, default=2)
  parser.add_argument("--rollout-ep", type=int, default=1)
  parser.add_argument("--max-diff-threshold", type=float, default=0.08)
  parser.add_argument("--json-out", default="")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  pathwaysutils.initialize()
  devices = jax.devices()
  print(f"[PARITY] Connected to {len(devices)} devices", flush=True)

  out_dir = tempfile.mkdtemp(prefix="olmo35_rl_parity_")
  vllm_additional = json.dumps({
      "maxtext_config": {
          "model_name": args.model_name,
          "model_call_mode": "inference",
          "override_model_config": True,
          "enable_dp_attention": False,
          "allow_split_physical_axes": True,
          "log_config": False,
          "weight_dtype": "bfloat16",
          "prefuse_moe_weights": True,
          "emo_enabled": False,
          "load_balance_loss_weight": 0.0,
      }
  })
  argv = [
      "",
      os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "rl.yml"),
      f"model_name={args.model_name}",
      f"tokenizer_path={args.tokenizer_path}",
      "run_name=olmo35_rl_parity",
      f"base_output_directory={out_dir}",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "load_checkpoint_only_once=True",
      "scan_layers=False",
      "override_model_config=True",
      "emo_enabled=False",
      "load_balance_loss_weight=0.0",
      "use_pathways=True",
      "chips_per_vm=4",
      "trainer_devices_fraction=1.0",
      "sampler_devices_fraction=1.0",
      f"rollout_tensor_parallelism={args.rollout_tp}",
      f"rollout_data_parallelism={args.rollout_dp}",
      f"rollout_expert_parallelism={args.rollout_ep}",
      "enable_dp_attention=False",
      "hbm_utilization_vllm=0.40",
      "max_prefill_predict_length=128",
      "max_target_length=256",
      "kv_cache_buffer=64",
      "batch_size=4",
      "max_num_seqs=16",
      "max_num_batched_tokens=1024",
      "async_scheduling=False",
      "allow_split_physical_axes=True",
      "log_weight_sync_time=True",
      f"vllm_hf_config_path={args.hf_config_dir}",
      'vllm_hf_overrides={"architectures":["MaxTextForCausalLM"]}',
      f"vllm_additional_config={vllm_additional}",
  ]

  trainer_cfg, sampler_cfg, trainer_devs, sampler_devs = model_creation_utils.setup_configs_and_devices(
      argv, {}, config_class=types.RLConfig
  )
  tok = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path)
  train_rl.configure_tokenizer_chat_template(tok, trainer_cfg)

  t0 = time.perf_counter()
  ref_model, _, actor_model, actor_mesh, rollout_mesh = model_creation_utils.create_models_and_meshes(
      trainer_cfg, sampler_cfg, trainer_devs, sampler_devs, tokenizer_pad_id=tok.pad_token_id
  )
  print(f"[PARITY] Models created in {time.perf_counter() - t0:.2f}s", flush=True)

  vllm_add = (
      json.loads(trainer_cfg.vllm_additional_config)
      if isinstance(trainer_cfg.vllm_additional_config, str)
      else trainer_cfg.vllm_additional_config
  )
  rollout_cfg = base_rollout.RolloutConfig(
      max_tokens_to_generate=32,
      max_prompt_length=128,
      kv_cache_size=320,
      temperature=0.0,
      top_p=1.0,
      top_k=1,
      rollout_vllm_model_version=trainer_cfg.tokenizer_path,
      rollout_vllm_hbm_utilization=trainer_cfg.hbm_utilization_vllm,
      rollout_vllm_tpu_backend_type="maxtext",
      rollout_vllm_hf_config_path=trainer_cfg.vllm_hf_config_path,
      rollout_vllm_additional_config=vllm_add,
      rollout_vllm_init_with_random_weights=True,
      rollout_vllm_enable_dp_attention=False,
      rollout_vllm_max_num_batched_tokens=1024,
      rollout_vllm_max_num_seqs=16,
      rollout_vllm_async_scheduling=False,
      return_logprobs=True,
      rollout_vllm_kwargs={
          "hf_overrides": trainer_cfg.vllm_hf_overrides,
          "enable_expert_parallel": args.rollout_ep > 1,
          "enable_prefix_caching": False,
          "dtype": "bfloat16",
      },
      **model_creation_utils.get_rollout_kwargs_for_parallelism(sampler_cfg, len(sampler_devs)),
  )

  t1 = time.perf_counter()
  rollout = MaxTextVllmRollout(
      rollout_actor=actor_model,
      tokenizer=tok,
      mesh=rollout_mesh,
      rollout_config=rollout_cfg,
      maxtext_config=trainer_cfg,
  )
  print(f"[PARITY] MaxTextVllmRollout initialized + synced in {time.perf_counter() - t1:.2f}s", flush=True)

  actor_params = nnx.state(actor_model, nnx.Param)
  n_covered = validate_direct_sync_layer_coverage(actor_params, rollout._sampler.transformer_state)
  print(f"[PARITY] Direct-sync layer coverage verified: {n_covered} target layer tensors", flush=True)

  # Confirm reference_model buffers were not deleted by colocated weight sync.
  ref_leaves = jax.tree_util.tree_leaves(nnx.state(ref_model.base, nnx.Param))
  assert all(not x.is_deleted() for x in ref_leaves), "Reference model buffers deleted during weight sync!"

  prompts = [
      "What is 17 + 25? Think step by step.",
      "A store sells 4 apples for $3. How much do 12 apples cost?",
      "Solve for x: 3x + 7 = 22.",
      "Explain why 2 * 3 * 5 = 30.",
  ]
  gen_out = rollout.generate(prompts, rollout_cfg)

  pad_id = tok.pad_token_id
  left_padded = np.asarray(gen_out.left_padded_prompt_tokens, dtype=np.int32)
  prompt_lens = list(map(int, gen_out.prompt_lengths))
  prompt_ids_list = [
      left_padded[i, left_padded.shape[1] - prompt_lens[i] :].tolist() for i in range(len(prompts))
  ]
  comp_ids_list = [list(map(int, t)) for t in gen_out.tokens]
  rollout_lps_list = [list(map(float, lp)) for lp in gen_out.logprobs]

  max_p = max(len(p) for p in prompt_ids_list)
  max_c = max(len(c) for c in comp_ids_list)
  total_len = ((max_p + max_c + 63) // 64) * 64

  input_ids = np.full((len(prompts), total_len), pad_id, dtype=np.int32)
  positions = np.zeros((len(prompts), total_len), dtype=np.int32)
  seg_ids = np.zeros((len(prompts), total_len), dtype=np.int32)
  for i, (p_ids, c_ids) in enumerate(zip(prompt_ids_list, comp_ids_list)):
    full = p_ids + c_ids
    start = max_p - len(p_ids)
    input_ids[i, start : start + len(full)] = full
    positions[i, start : start + len(full)] = np.arange(len(full), dtype=np.int32)
    seg_ids[i, start : start + len(full)] = 1

  @nnx.jit
  def _trainer_logprobs(model, ids, pos, segs):
    logits, _ = model(ids, pos, None, None, decoder_segment_ids=segs)
    return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

  with jax.set_mesh(actor_mesh):
    log_probs_np = np.asarray(
        _trainer_logprobs(
            actor_model,
            jnp.asarray(input_ids),
            jnp.asarray(positions),
            jnp.asarray(seg_ids),
        )
    )

  diffs = []
  for i, (_, c_ids, r_lps) in enumerate(zip(prompt_ids_list, comp_ids_list, rollout_lps_list)):
    for j, tok_id in enumerate(c_ids):
      t_lp = float(log_probs_np[i, max_p + j - 1, tok_id])
      diffs.append(abs(t_lp - float(r_lps[j])))

  summary = {
      "model_name": args.model_name,
      "num_devices": len(devices),
      "direct_sync_tensors": int(n_covered),
      "ref_unchanged": True,
      "num_tokens": len(diffs),
      "mean_abs_logprob_diff": float(np.mean(diffs)),
      "p95_abs_logprob_diff": float(np.percentile(diffs, 95)),
      "max_abs_logprob_diff": float(np.max(diffs)),
      "passed": bool(np.all(np.isfinite(diffs)) and float(np.mean(diffs)) < args.max_diff_threshold),
  }
  print(f"[PARITY SUMMARY] {json.dumps(summary)}", flush=True)
  if args.json_out:
    with open(args.json_out, "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
  assert summary["passed"], f"Parity check failed: {summary}"
  print("[PARITY] PASSED!", flush=True)


if __name__ == "__main__":
  main()
