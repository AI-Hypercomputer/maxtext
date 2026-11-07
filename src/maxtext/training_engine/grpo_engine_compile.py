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

"""Tunix GRPO engine and synthetic micro-batch builders for `maxtext_engine_compile`."""

import math
from typing import Any

import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
import numpy as np

# Default pad and EOS token IDs used for AOT compilation so no tokenizer is loaded.
GRPO_PAD_ID = 0
GRPO_EOS_ID = GRPO_PAD_ID


def get_rl_micro_batch(
    config: pyconfig.HyperParameters, algo: Any, mesh: jax.sharding.Mesh
) -> abstract_engine.RLTrainerPayload:
  """Builds a synthetic `RLTrainerPayload` micro-batch matching Tunix's orchestrator layout.

  Constructs synthetic rollouts with `max_prefill_predict_length` prompt tokens
  and
  `max_target_length - max_prefill_predict_length` completion tokens, converts
  them
  into payloads via `algo.create_trainer_payloads`, and assembles one
  micro-batch
  using Tunix's `create_batch_assembler`:
  - When `config.max_seq_token_per_tpu > 0`, uses `SequencePackedBatchAssembler`
    with one packed row per data-parallel shard (`fsdp * data * fsdp_transpose *
    expert`)
    and at most `config.max_segments_per_packed_row` sequences per row.
  - Otherwise, uses `PaddedBatchAssembler` with `micro_batch_size_to_train_on`
    padded sequences.

  Populates `routed_experts` when `config.router_replay=True` and reference
  log-probabilities when `algo.requires_reference_kl=True`.
  """
  from tunix.dist_rl.common import datatypes  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from tunix.dist_rl.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from tunix.rl import packing  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  prompt_length = config.max_prefill_predict_length
  completion_length = config.max_target_length - prompt_length
  if prompt_length <= 0 or completion_length <= 0:
    raise ValueError(
        "loss=grpo splits max_target_length"
        f" ({config.max_target_length}) into max_prefill_predict_length"
        f" ({prompt_length}) prompt tokens and the rest as completion tokens,"
        " so it needs 0 < max_prefill_predict_length < max_target_length."
    )
  if config.router_replay and config.num_experts <= 1:
    raise ValueError(
        "router_replay replays the rollouts' MoE routing, but"
        f" num_experts={config.num_experts} leaves this model without routed"
        " experts."
    )
  num_data_shards = math.prod(mesh.shape.get(axis, 1) for axis in ("fsdp", "data", "fsdp_transpose", "expert"))
  assembler = batch_assembly.create_batch_assembler(
      num_generations=algo.num_generations,
      mini_batch_size=1,
      train_micro_batch_size=int(config.micro_batch_size_to_train_on),
      batch_config=batch_assembly.BatchConfig(
          pad_id=GRPO_PAD_ID,
          max_prompt_length=prompt_length,
          max_response_length=completion_length,
          max_seq_token_per_tpu=config.max_seq_token_per_tpu or None,
          max_segments_per_packed_row=config.max_segments_per_packed_row or None,
          num_data_shards=num_data_shards,
      ),
  )
  rollout_length = prompt_length + completion_length
  rollouts_per_row = 1
  if isinstance(assembler, batch_assembly.SequencePackedBatchAssembler):
    max_segments = packing.effective_max_segments(assembler.max_packed_len, assembler.max_segments_per_packed_row)
    rollouts_per_row = min(assembler.max_packed_len // rollout_length, max_segments)
  num_rollouts = assembler.batch_size * rollouts_per_row

  token = GRPO_PAD_ID + 1
  rollout = {
      "prompt_tokens": np.full(prompt_length, token, np.int32),
      "conversation_tokens": np.full(completion_length, token, np.int32),
      "conversation_masks": np.ones(completion_length, np.float32),
      "old_logprobs": np.zeros(completion_length, np.float32),
      "status": "SUCCEEDED",
  }
  if config.router_replay:
    top_k = config.num_experts_per_tok
    experts = np.arange(top_k, dtype=np.int16)
    experts[-1] = config.num_experts - 1
    rollout["routed_experts"] = np.broadcast_to(experts, (rollout_length, config.num_decoder_layers, top_k))
  group_size = algo.num_generations
  payloads = []
  for prompt_index in range(-(-num_rollouts // group_size)):
    group = [
        datatypes.TrajectoryItem(
            prompt_id=f"prompt_{prompt_index}",
            group_index=index,
            traj=dict(rollout),
        )
        for index in range(group_size)
    ]
    payloads.extend(algo.create_trainer_payloads(group, rewards=[float(index % 2) for index in range(group_size)]))
  batches = [*assembler.feed(payloads[:num_rollouts]), *assembler.flush()]
  if len(batches) != 1:
    raise RuntimeError(
        f"Tunix's {type(assembler).__name__} assembled {num_rollouts} rollouts"
        f" into {len(batches)} micro-batches, not one."
    )
  micro_batch = batches[0].payload
  if algo.requires_reference_kl:
    micro_batch = batch_assembly.with_ref_per_token_logps(
        micro_batch, np.zeros(np.shape(micro_batch.completion_ids), np.float32)
    )
  return micro_batch


def get_shaped_rl_micro_batch(
    config: pyconfig.HyperParameters, algo: Any, mesh: jax.sharding.Mesh
) -> abstract_engine.RLTrainerPayload:
  """Returns `get_rl_micro_batch` converted to `jax.ShapeDtypeStruct` leaves with empty metadata."""
  micro_batch = get_rl_micro_batch(config, algo, mesh)
  return jax.tree.map(
      lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
      micro_batch.replace(metadata={}),
  )


def describe_rl_micro_batch(
    config: pyconfig.HyperParameters,
    micro_batch: abstract_engine.RLTrainerPayload,
) -> str:
  """Formats a human-readable summary of the RL micro-batch being compiled."""
  prompt_length = config.max_prefill_predict_length
  completion_length = config.max_target_length - prompt_length
  if micro_batch.segment_ids is None:
    lines = [
        "Compiling Tunix's GRPO loss, for micro-batches of"
        f" {micro_batch.prompt_ids.shape[0]} sequences of {prompt_length}"
        f" prompt and {completion_length} completion tokens."
    ]
  else:
    rows, tokens = micro_batch.completion_ids.shape
    lines = [
        f"Compiling Tunix's GRPO loss, for packed micro-batches of {rows} x"
        f" {tokens} tokens, each row holding up to"
        f" {micro_batch.num_segments - 1} sequences of up to {prompt_length}"
        f" prompt and {completion_length} completion tokens."
    ]
  if micro_batch.routed_experts is not None:
    if micro_batch.routed_experts.ndim == 5:
      _, _, layers, groups, _ = micro_batch.routed_experts.shape
      top_k = groups * 4
    else:
      _, _, layers, top_k = micro_batch.routed_experts.shape
    lines.append(
        f"The trainer replays the rollouts' routing: each token's {top_k}" f" experts in each of {layers} layers."
    )
  elif config.router_replay:
    lines.append(
        "router_replay is set, but Tunix's batch assembler drops"
        " routed_experts from this micro-batch, so the compiled trainer routes"
        " every token itself."
    )
  return "\n".join(lines)


def build_grpo_engine(
    config: pyconfig.HyperParameters,
    topology_mesh: jax.sharding.Mesh,
    engine_cls: Any,
) -> tuple[Any, abstract_engine.RLTrainerPayload]:
  """Builds an abstract engine configured for Tunix's GRPO loss and its shaped micro-batch.

  Constructs `GRPOConfig` from `config.grpo_config` (defaulting `temperature` to
  `config.decode_sampling_temperature`), wraps the abstract model with
  `TunixMaxTextAdapter`, and attaches `GRPOAdapter`'s loss and model input
  mapping (including `compute_logps_chunk_size` when `config.logps_chunk_size >
  0`).
  """
  from tunix.dist_rl.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from tunix.rl import algorithm_config  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  grpo_kwargs = dict(config.grpo_config)
  temperature = grpo_kwargs.pop("temperature", config.decode_sampling_temperature)
  algo_cfg = algorithm_config.GRPOConfig(**grpo_kwargs)
  algo_cfg.temperature = temperature
  algo = algorithm_adapter.GRPOAdapter(algo_cfg)
  micro_batch = get_shaped_rl_micro_batch(config, algo, topology_mesh)
  engine = engine_cls(
      config,
      topology_mesh,
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=GRPO_PAD_ID,
  )
  engine.with_loss_fn(algo.loss_fn(), has_aux=True)
  gen_model_input_fn = algo.build_gen_model_input_fn(pad_id=GRPO_PAD_ID, eos_id=GRPO_EOS_ID)
  chunk_size = config.logps_chunk_size
  if chunk_size > 0:
    base_fn = gen_model_input_fn

    def chunked_gen_model_input_fn(payload: Any) -> dict[str, Any]:
      inputs = dict(base_fn(payload))
      inputs.setdefault("compute_logps_chunk_size", chunk_size)
      return inputs

    gen_model_input_fn = chunked_gen_model_input_fn

  engine.with_gen_model_input_fn(gen_model_input_fn)
  return engine, micro_batch
