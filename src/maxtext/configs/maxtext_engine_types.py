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

"""Pydantic configuration types specific to `maxtext_engine`."""

from typing import Any, Literal

from maxtext.configs import types as pydantic_types
import pydantic


class CompileConfig(pydantic_types.MaxTextConfig):
  """`MaxTextConfig` extended with `maxtext_engine_compile.py`-only options."""

  loss: Literal["maxtext", "grpo"] = pydantic.Field(
      "maxtext",
      description=(
          "Loss function to compile MaxTextTrainingEngine's training kernels"
          " with in maxtext_engine_compile.py. 'maxtext' compiles MaxText's"
          " standard pre-training cross-entropy loss on a shaped pre-training"
          " micro-batch. 'grpo' wraps the model with TunixMaxTextAdapter and"
          " compiles Tunix's GRPO loss on an RLTrainerPayload built from"
          " rollouts of max_prefill_predict_length prompt tokens and"
          " (max_target_length - max_prefill_predict_length) completion tokens."
      ),
  )
  grpo_config: dict[str, Any] = pydantic.Field(
      default_factory=dict,
      description=(
          "Keyword arguments forwarded to Tunix's GRPOConfig when"
          " loss='grpo' (for example, {'beta': 0.0,"
          " 'use_rollout_logps': False, 'sampler_is': 'token'}). Controls which"
          " optional payload fields (such as reference log-probabilities, old"
          " policy log-probabilities, and importance-sampling weights) are"
          " included in the compiled micro-batch and loss computation. Defaults"
          " temperature to decode_sampling_temperature and all other fields to"
          " GRPOConfig's defaults."
      ),
  )
  logps_chunk_size: int = pydantic.Field(
      0,
      ge=0,
      description=(
          "Token chunk size (Tunix's compute_logps_chunk_size) passed to the"
          " model input mapping when loss='grpo'. When positive,"
          " the adapter skips the full-sequence LM head projection during the"
          " decoder pass and computes per-token log-probabilities in chunks of"
          " this many tokens to reduce peak HBM usage. Set to 0 to disable"
          " chunking and materialize full-sequence logits at once."
      ),
  )
  max_seq_token_per_tpu: int = pydantic.Field(
      0,
      ge=0,
      description=(
          "Maximum number of tokens per packed row (Tunix's"
          " max_seq_token_per_tpu) when loss='grpo'. When 0,"
          " compiles against a padded micro-batch of"
          " micro_batch_size_to_train_on sequences built by Tunix's"
          " PaddedBatchAssembler. When positive, compiles against a"
          " sequence-packed micro-batch built by Tunix's"
          " SequencePackedBatchAssembler, with one row of this many tokens per"
          " data-parallel shard across the mesh's data, fsdp, fsdp_transpose,"
          " and expert axes."
      ),
  )
  max_segments_per_packed_row: int = pydantic.Field(
      0,
      ge=0,
      description=(
          "Maximum number of rollout sequences that can be packed into a single"
          " row (Tunix's max_segments_per_packed_row) when"
          " max_seq_token_per_tpu > 0. Determines the static"
          " segment-reduction dimension (num_segments) compiled into the GRPO"
          " loss. Set to 0 to leave the per-row sequence cap unset, allowing as"
          " many sequences as fit within the row's token budget."
      ),
  )
  router_replay: bool = pydantic.Field(
      False,
      description=(
          "Whether to compile with MoE router replay enabled (Tunix's"
          " return_routed_experts) when loss='grpo'. When True,"
          " synthetic rollout trajectories include per-token, per-layer top-k"
          " expert routing indices (routed_experts) so the compiled training"
          " step replays the rollout's expert assignments instead of routing"
          " tokens dynamically."
      ),
  )

  @pydantic.model_validator(mode="after")
  def validate_grpo_options(self) -> "CompileConfig":
    """Rejects GRPO options that would be silently ignored."""
    rl_options = (
        "grpo_config",
        "logps_chunk_size",
        "max_seq_token_per_tpu",
        "max_segments_per_packed_row",
        "router_replay",
    )
    set_options = [option for option in rl_options if getattr(self, option)]
    if self.loss != "grpo" and set_options:
      raise ValueError(
          f"loss={self.loss} does not compile Tunix's GRPO loss, so"
          f" {', '.join(set_options)} would be ignored: set loss=grpo."
      )
    if self.max_segments_per_packed_row and not self.max_seq_token_per_tpu:
      raise ValueError(
          "max_segments_per_packed_row limits the sequences in a packed row, so"
          " it needs max_seq_token_per_tpu, which turns packing on."
      )
    return self
