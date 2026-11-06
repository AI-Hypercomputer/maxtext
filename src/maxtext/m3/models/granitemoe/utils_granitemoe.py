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

"""Per-device training TFLOPs and `MaxTextTrainingEngine` helpers for GraniteMoe (`m3`)."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from maxtext.m3.models.granitemoe import modeling_granitemoe

if TYPE_CHECKING:
  from maxtext.training_engine import abstract_engine

_GRANITEMOE_LOSS_BATCH_KEYS = (
    "inputs",
    "targets",
    "targets_segmentation",
    "targets_position",
    "inputs_segmentation",
    "inputs_position",
    "attention_mask",
)


@dataclasses.dataclass(frozen=True)
class _FallbackWeightedMetric:
  """Fallback WeightedMetric container used when the post-training engine bundle is absent."""

  unreduced_sum: jax.Array
  denominator: jax.Array


def _resolve_weighted_metric_cls() -> Any:
  """Resolves canonical Tunix/MaxText `WeightedMetric` class (falling back cleanly in base-only installs)."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  try:
    from maxtext.training_engine import abstract_engine

    return abstract_engine.WeightedMetric
  except ImportError:
    return _FallbackWeightedMetric


def calculate_granitemoe_tflops_per_device(
    config: Any,
    log: bool = True,
) -> tuple[float, float, float]:
  """Calculates per-device training TFLOPs for GraniteMoe (`(total_tflops, weight_tflops, attention_tflops)`).

  Accounts for:
  - GQA attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) across all `num_hidden_layers`
  - Causal dot-product attention (`Q @ K^T` and `P @ V`, `0.5` causal triangle factor)
  - MoE router gate (`hidden_size -> num_local_experts`) + `num_experts_per_tok` active SwiGLU experts
    (`gate`, `up`, `down` -> `6 * B * S * hidden_size * intermediate_size` per active expert)
  - Output vocabulary unembedding projection (`hidden_size -> vocab_size`)
  - `3x` forward + backward multiplier and `gradient_accumulation_steps` (plus `use_dpo` reference forward if set)
  """
  gm_cfg = modeling_granitemoe.GraniteMoeConfig.from_config(config)

  b_dim = float(getattr(config, "per_device_batch_size", 1.0))
  s_dim = int(getattr(config, "max_target_length", 2048))
  grad_accum = int(getattr(config, "gradient_accumulation_steps", 1))
  use_dpo = bool(getattr(config, "use_dpo", False))

  batch_len = b_dim * s_dim
  h_dim = gm_cfg.hidden_size
  i_dim = gm_cfg.intermediate_size
  v_dim = gm_cfg.vocab_size
  num_layers = gm_cfg.num_hidden_layers
  num_q_heads = gm_cfg.num_attention_heads
  num_kv_heads = gm_cfg.num_key_value_heads
  head_dim = gm_cfg.head_dim
  num_experts = gm_cfg.num_local_experts
  top_k = gm_cfg.num_experts_per_tok

  # 1. Output vocabulary unembedding projection (embedding lookup is a gather with 0 matmul FLOPs).
  unembedding_flops = 2.0 * batch_len * h_dim * v_dim

  # 2. Per-layer GQA projection FLOPs: q_proj + k_proj + v_proj + o_proj.
  q_dim = num_q_heads * head_dim
  kv_dim = num_kv_heads * head_dim
  attn_weight_flops_per_layer = 2.0 * batch_len * h_dim * (2 * q_dim + 2 * kv_dim)

  # 3. Per-layer causal attention QK^T + PV FLOPs (half-matrix causal factor 0.5 * 4 = 2).
  attn_core_flops_per_layer = 2.0 * batch_len * s_dim * num_q_heads * head_dim

  # 4. Per-layer routed SwiGLU MoE FLOPs: router gate + top_k active experts (gate + up + down).
  router_flops_per_layer = 2.0 * batch_len * h_dim * num_experts
  routed_experts_flops_per_layer = top_k * (6.0 * batch_len * h_dim * i_dim)
  moe_weight_flops_per_layer = router_flops_per_layer + routed_experts_flops_per_layer

  total_weight_flops_fwd = unembedding_flops + num_layers * (attn_weight_flops_per_layer + moe_weight_flops_per_layer)
  total_attn_flops_fwd = num_layers * attn_core_flops_per_layer

  learnable_weight_tflops = (3.0 * total_weight_flops_fwd / 1e12) * grad_accum
  attention_tflops = (3.0 * total_attn_flops_fwd / 1e12) * grad_accum

  if use_dpo:
    ref_weight_tflops = learnable_weight_tflops / 3.0
    ref_attn_tflops = attention_tflops / 3.0
    attention_tflops = attention_tflops + ref_attn_tflops
  else:
    ref_weight_tflops = 0.0

  total_tflops = learnable_weight_tflops + attention_tflops + ref_weight_tflops

  if log:
    print(
        f"GraniteMoe per train step:\nTotal TFLOPs: {total_tflops:.4f}\n"
        f"split as {100.0 * learnable_weight_tflops / max(total_tflops, 1e-12):.2f}% learnable weight flops "
        f"and {100.0 * attention_tflops / max(total_tflops, 1e-12):.2f}% attention flops"
    )
  return total_tflops, learnable_weight_tflops, attention_tflops


def granitemoe_engine_loss_fn(
    model: modeling_granitemoe.GraniteMoeModel,
    config: Any,
    data: Mapping[str, Any],
    dropout_rng: Any = None,
    params: Any = None,
    *,
    is_train: bool,
) -> tuple[abstract_engine.WeightedMetric, dict[str, jax.Array]]:
  """`MaxTextTrainingEngine` loss adapter folding Switch `moe_lb_loss` into primary scalar loss."""
  metric_cls = _resolve_weighted_metric_cls()

  del params
  filtered_batch = {k: data[k] for k in _GRANITEMOE_LOSS_BATCH_KEYS if k in data}
  coef_override: float | None = None
  if config is not None:
    for key in ("router_aux_loss_coef", "load_balance_loss_weight"):
      if hasattr(config, key) and getattr(config, key) is not None:
        coef_override = float(getattr(config, key))
        break

  total_loss, aux = modeling_granitemoe.compute_granitemoe_loss(
      model=model,
      tokens=filtered_batch["inputs"],
      decoder_target_tokens=filtered_batch["targets"],
      decoder_target_mask=filtered_batch.get("targets_segmentation"),
      decoder_segment_ids=filtered_batch.get("inputs_segmentation"),
      decoder_positions=filtered_batch.get("inputs_position"),
      attention_mask=filtered_batch.get("attention_mask"),
      router_aux_loss_coef=coef_override,
      is_train=is_train,
      dropout_rng=dropout_rng,
  )
  return (
      metric_cls(
          unreduced_sum=(total_loss * aux["total_weights"]).astype(jnp.float32),
          denominator=aux["total_weights"],
      ),
      aux,
  )


def configure_granitemoe_engine(engine: Any) -> Any:
  """Registers `granitemoe_engine_loss_fn` on `MaxTextTrainingEngine` via `with_loss_fn(..., has_aux=True)`."""
  engine.with_loss_fn(granitemoe_engine_loss_fn, has_aux=True)
  engine.with_gen_model_input_fn(None)
  return engine
