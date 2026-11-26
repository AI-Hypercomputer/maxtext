# Copyright 2023–2026 Google LLC
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

"""DeepSeek-V3 in m3 format.

What sets DeepSeek-V3 apart from a Qwen3-style decoder is defined in this file:

  * **Multi-head latent attention (MLA).** Queries and keys/values go through
    low-rank latents (`q_lora_rank`, `kv_lora_rank`), and only a small
    `qk_rope_head_dim` slice of each head carries position, via YaRN RoPE.
  * **Sigmoid routing with a balancing bias and expert groups.** Router scores
    are sigmoids; a per-expert bias (`RouterBias`, not a parameter) steers
    load balancing without an auxiliary loss; with `n_routing_groups` set,
    top-k is restricted to the best `topk_routing_group` groups of experts.
  * **Shared experts** alongside the routed ones, and
    `first_num_dense_layers` ordinary MLP layers before the MoE layers.

The router lives here; dispatching tokens to the selected experts and running
them is `maxtext.m3.core.moe`. The softmax attention arithmetic is
`maxtext.m3.core.kernels`.

Parameter names follow the legacy checkpoint layout, with two differences that
are translated where legacy weights are loaded rather than here: legacy's
`DeepSeekMoeBlock_0.MoeBlock_0` nesting is `moe_block.routed_experts`, and
legacy numbers its dense and MoE layers separately (`dense_layers_i`,
`moe_layers_j`) where this model keeps both kinds in one `layers` container, in
order. Under `scan_layers` they become two scanned runs (`maxtext.m3.infra.scan`).

Not implemented, and refused by name when a config asks for it: the balancing
bias update (`routed_bias_update_rate`), the load-balance loss
(`load_balance_loss_weight`), multi-token prediction (`mtp_num_layers`),
DeepSeek-V3.2/V4 extensions (`use_indexer`, `mhc_expansion_rate`,
`mlp_activations_limit`, `engram_layers`), and non-zero dropout.
"""

import dataclasses
import math
from typing import Any, Optional

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core import kernels
from maxtext.m3.core import moe
from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType, check_mesh, param_metadata

# Mesh axes this model has been run on. `models.MODEL_SUPPORTED_MESH_AXES` refuses the rest.
DEEPSEEK_V3_MESH_AXES = ("data", "fsdp", "tensor", "expert")


@dataclasses.dataclass(frozen=True)
class DeepSeekV3Sharding(DefaultSharding):
  """Default rules plus MLA's latent axes.

  The up-projections (`wq_b`, `wkv_b`) have no `embed` dimension, so their
  latent dimension takes its place and is sharded the same way. In the
  down-projections (`wq_a`, `wkv_a`) `embed` is already sharded and the latent
  dimension stays replicated. Legacy does the same with `q_lora` / `kv_lora` and
  `q_lora_up_proj` / `kv_lora_up_proj`.
  """

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    if axis in ("q_lora", "kv_lora"):
      return super().map_axis("embed", tensor_name, tensor_type)
    if axis in ("q_lora_up_proj", "kv_lora_up_proj"):
      return None
    return super().map_axis(axis, tensor_name, tensor_type)


class RouterBias(nnx.Variable):  # pylint: disable=abstract-method
  """Per-expert routing bias for auxiliary-loss-free load balancing.

  Deliberately not an `nnx.Param`: it is adjusted by a counting rule, not by
  gradients, and the training engine differentiates `nnx.Param` only.
  """


def unsupported_settings(config: Config) -> dict[str, Any]:
  """DeepSeek-V3 settings this model does not implement, as `{field: value}`."""
  unsupported = {}
  if config.load_balance_loss_weight > 0:
    unsupported["load_balance_loss_weight"] = config.load_balance_loss_weight
  if config.routed_bias_update_rate > 0:
    unsupported["routed_bias_update_rate"] = config.routed_bias_update_rate
  if config.mtp_num_layers > 0:
    unsupported["mtp_num_layers"] = config.mtp_num_layers
  if config.use_indexer:
    unsupported["use_indexer"] = config.use_indexer
  if config.mhc_expansion_rate > 1:
    unsupported["mhc_expansion_rate"] = config.mhc_expansion_rate
  if config.mlp_activations_limit > 0:
    unsupported["mlp_activations_limit"] = config.mlp_activations_limit
  if config.engram_layers:
    unsupported["engram_layers"] = config.engram_layers
  if config.enable_dropout and config.dropout_rate > 0:
    unsupported["dropout_rate"] = config.dropout_rate
  return unsupported


def norm_kwargs(config: Config, sharding: Sharding, name: str, rngs: nnx.Rngs) -> dict[str, Any]:
  """Keyword arguments shared by every `nnx.RMSNorm` in the model, with the scale's sharding."""
  return {
      "epsilon": config.normalization_layer_epsilon,
      "dtype": config.dtype,
      "param_dtype": config.weight_dtype,
      "scale_metadata": param_metadata(sharding, name, ("norm",)),
      "rngs": rngs,
  }


# ----------------------------------------------------------------------------
# YaRN RoPE
# ----------------------------------------------------------------------------


def yarn_inv_freqs(config: Config, dim: int) -> jax.Array:
  """Inverse frequencies for YaRN-extended RoPE, shape `(dim // 2,)`, float32.

  Base RoPE frequencies, scaled down by `rope_factor` in the dimensions that
  rotate slowly (long wavelengths, which extrapolate badly) and left alone in
  the fast ones, with a linear ramp in between. The ramp's end points come
  from `beta_fast` / `beta_slow` rotation counts over the original context
  length, floored and ceiled when `rope_truncate` is set. Same float32
  arithmetic as legacy `YarnRotaryEmbedding`.
  """
  half = dim // 2
  theta, factor = config.rope_max_timescale, config.rope_factor
  freqs = 1.0 / (theta ** (2.0 * jnp.arange(half, dtype=jnp.float32) / dim))

  def correction_dim(num_rotations):
    return dim * math.log(config.original_max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

  low, high = correction_dim(config.beta_fast), correction_dim(config.beta_slow)
  if config.rope_truncate:
    low, high = math.floor(low), math.ceil(high)
  low, high = max(low, 0), min(high, dim - 1)
  if low == high:
    high += 0.001
  ramp = jnp.clip((jnp.arange(half, dtype=jnp.float32) - low) / (high - low), 0, 1)
  smooth = 1 - ramp
  return freqs / factor * (1 - smooth) + freqs * smooth


def yarn_attention_scaling(config: Config) -> float:
  """YaRN's output scale, `0.1 * ln(rope_factor) + 1`, when `rope_attention_scaling` asks for it."""
  if config.rope_attention_scaling and config.rope_factor > 1:
    return 0.1 * math.log(config.rope_factor) + 1.0
  return 1.0


def apply_yarn_rope(
    x: jax.Array, positions: jax.Array, inv_freqs: jax.Array, *, interleave: bool, scale: float = 1.0
) -> jax.Array:
  """Rotates `x` of shape `(batch, length, heads, dim)` by `positions * inv_freqs`.

  Pairs are read interleaved (`[re0, im0, re1, im1, ...]`, the DeepSeek
  checkpoint layout) or concatenated, and always written concatenated
  (`[re..., im...]`). Attention scores do not depend on the layout as long as
  queries and keys agree. Complex arithmetic in float32, as in legacy.
  """
  angles = positions.astype(jnp.float32)[:, :, None, None] * inv_freqs[None, None, None, :]
  freqs = jax.lax.complex(jnp.cos(angles), jnp.sin(angles))
  if interleave:
    pairs = jnp.reshape(x, x.shape[:-1] + (x.shape[-1] // 2, 2))
    first, second = pairs[..., 0], pairs[..., 1]
  else:
    first, second = jnp.split(x, 2, axis=-1)
  rotated = (first + 1j * second) * freqs
  out = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
  if scale != 1.0:
    out = out * scale
  return out.astype(x.dtype)


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


class DeepSeekV3Attention(nnx.Module):
  """Multi-head latent attention.

  Submodule names (`wq_a`, `q_norm`, `wq_b`, `wkv_a`, `kv_norm`, `wkv_b`,
  `out`; `query` when `q_lora_rank == 0`) match the checkpoint layout.
  """

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    s = sharding
    self.attention_kernel = kernels.attention_from_config(config, s, mesh=mesh)
    self.num_heads = config.num_query_heads
    self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.rope_scale = yarn_attention_scaling(config)
    # DeepSeek folds YaRN's attention temperature into the softmax scale.
    self.softmax_scale = self.qk_head_dim**-0.5
    if config.max_position_embeddings > config.original_max_position_embeddings:
      mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
      self.softmax_scale *= mscale * mscale

    linear_kw = {"use_bias": False, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    einsum_kw = {"bias_shape": None, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    emb, heads = config.emb_dim, self.num_heads
    if config.q_lora_rank == 0:
      self.query = nnx.Einsum(
          "...e,ehd->...hd",
          (emb, heads, self.qk_head_dim),
          kernel_metadata=param_metadata(s, "query", ("embed", "heads", "head_dim")),
          **einsum_kw,
      )
    else:
      self.wq_a = nnx.Linear(
          emb,
          config.q_lora_rank,
          kernel_metadata=param_metadata(s, "wq_a", ("embed", "q_lora_up_proj")),
          **linear_kw,
      )
      self.q_norm = nnx.RMSNorm(config.q_lora_rank, **norm_kwargs(config, s, "q_norm", rngs))
      self.wq_b = nnx.Einsum(
          "...r,rhd->...hd",
          (config.q_lora_rank, heads, self.qk_head_dim),
          kernel_metadata=param_metadata(s, "wq_b", ("q_lora", "heads", "head_dim")),
          **einsum_kw,
      )
    self.wkv_a = nnx.Linear(
        emb,
        config.kv_lora_rank + config.qk_rope_head_dim,
        kernel_metadata=param_metadata(s, "wkv_a", ("embed", "kv_lora_up_proj")),
        **linear_kw,
    )
    self.kv_norm = nnx.RMSNorm(config.kv_lora_rank, **norm_kwargs(config, s, "kv_norm", rngs))
    self.wkv_b = nnx.Einsum(
        "...r,rhd->...hd",
        (config.kv_lora_rank, heads, config.qk_nope_head_dim + config.v_head_dim),
        kernel_metadata=param_metadata(s, "wkv_b", ("kv_lora", "heads", "head_dim")),
        **einsum_kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (heads, config.v_head_dim, emb),
        kernel_metadata=param_metadata(s, "out", ("heads", "head_dim", "embed")),
        **einsum_kw,
    )

  def _rope(self, x: jax.Array, positions: jax.Array) -> jax.Array:
    cfg = self.config
    inv_freqs = yarn_inv_freqs(cfg, cfg.qk_rope_head_dim)
    return apply_yarn_rope(x, positions, inv_freqs, interleave=cfg.rope_interleave, scale=self.rope_scale)

  def _query(self, x: jax.Array, positions: jax.Array) -> jax.Array:
    """Low-rank query projection, RoPE on the rotary slice, softmax scale folded in."""
    cfg = self.config
    if cfg.q_lora_rank == 0:
      q = self.query(x)
    else:
      latent = checkpoint_name(self.wq_a(x), "query_wa_proj")
      latent = checkpoint_name(self.q_norm(latent), "mla_q")
      q = self.wq_b(latent)
    q = self.sharding.constrain(q, "query", ("batch", "length", "heads", "head_dim"))
    q_nope, q_rope = jnp.split(q, [cfg.qk_nope_head_dim], axis=-1)
    q_rope = self._rope(q_rope, positions)
    return jnp.concatenate([q_nope, q_rope], axis=-1) * self.softmax_scale

  def _key_value(self, x: jax.Array, positions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Keys `(nope | shared rotary slice)` and values from the compressed latent."""
    cfg = self.config
    latent = checkpoint_name(self.wkv_a(x), "kv_wa_proj")
    latent, k_rope = jnp.split(latent, [cfg.kv_lora_rank], axis=-1)
    latent = checkpoint_name(self.kv_norm(latent), "mla_kv")
    # One rotary key slice is shared by every head.
    k_rope = self._rope(k_rope[:, :, None, :], positions)
    kv = self.sharding.constrain(self.wkv_b(latent), "key_value", ("batch", "length", "heads", "head_dim"))
    k_nope, value = jnp.split(kv, [cfg.qk_nope_head_dim], axis=-1)
    k_rope = jnp.broadcast_to(k_rope, k_nope.shape[:-1] + (cfg.qk_rope_head_dim,))
    return jnp.concatenate([k_nope, k_rope], axis=-1), value

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    s = self.sharding
    is_default_positions = decoder_positions is None
    inputs = s.constrain(inputs, "attn_in", ("batch", "length", "embed"))
    if decoder_positions is None:
      batch, length = inputs.shape[:2]
      decoder_positions = s.constrain(
          jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32)[None, :], (batch, length)),
          "decoder_positions",
          ("batch", "length"),
      )
    query = checkpoint_name(self._query(inputs, decoder_positions), "query_proj")
    key, value = self._key_value(inputs, decoder_positions)
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")
    # As in Qwen3: positions reach the kernel only for packed inputs, which keeps
    # unpacked training on the kernel's plain causal path.
    use_mask_positions = not is_default_positions and decoder_segment_ids is not None
    out = self.attention_kernel(
        query,
        key,
        value,
        segment_ids=decoder_segment_ids,
        positions=decoder_positions if use_mask_positions else None,
    )
    out = s.constrain(out.astype(self.config.dtype), "attention_out", ("batch", "length", "heads", "head_dim"))
    out = checkpoint_name(out, "attention_out")
    return checkpoint_name(self.out(out), "out_proj")


# ----------------------------------------------------------------------------
# Feed-forward: dense MLP, router, routed and shared experts
# ----------------------------------------------------------------------------


class DeepSeekV3MLP(nnx.Module):
  """SwiGLU MLP; also the shared experts, with `intermediate_dim` their combined width."""

  def __init__(self, config: Config, intermediate_dim: int, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    s = sharding
    kw = {"use_bias": False, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    emb = config.emb_dim
    self.wi_0 = nnx.Linear(emb, intermediate_dim, kernel_metadata=param_metadata(s, "wi_0", ("embed", "mlp")), **kw)
    self.wi_1 = nnx.Linear(emb, intermediate_dim, kernel_metadata=param_metadata(s, "wi_1", ("embed", "mlp")), **kw)
    self.wo = nnx.Linear(intermediate_dim, emb, kernel_metadata=param_metadata(s, "wo", ("mlp", "embed")), **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = checkpoint_name(self.wi_0(x), "mlpwi_0")
    up = checkpoint_name(self.wi_1(x), "mlpwi_1")
    hidden = (jax.nn.silu(gate) * up).astype(self.config.dtype)
    hidden = self.sharding.constrain(hidden, "mlp_hidden", ("batch", "length", "mlp"))
    return checkpoint_name(self.wo(hidden), "mlpwo")


class DeepSeekV3Router(nnx.Module):
  """Sigmoid router with a balancing bias; attribute name `gate` in the checkpoint."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    self.dtype = config.dtype
    init = nnx.initializers.variance_scaling(config.dense_init_scale, "fan_in", "truncated_normal")
    # The expert dimension is replicated (legacy: `kernel_axes=("embed", None)`):
    # every device needs every expert's score to route.
    self.kernel = nnx.Param(
        init(rngs.params(), (config.emb_dim, config.num_experts), config.weight_dtype),
        **param_metadata(sharding, "gate", ("embed", None)),
    )
    if config.routed_bias:
      self.bias = RouterBias(
          jnp.zeros((config.num_experts,), config.weight_dtype), **param_metadata(sharding, "gate_bias", (None,))
      )
    else:
      self.bias = None

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns `(scores_for_selection, scores_for_weights)`: the latter excludes the bias."""
    logits = jnp.dot(x.astype(self.dtype), self.kernel[...].astype(self.dtype))
    logits = self.sharding.constrain(logits, "router_logits", ("batch", "length", None))
    match self.config.routed_score_func:
      case "sigmoid":
        scores = jax.nn.sigmoid(logits)
      case "":
        scores = logits
      case other:
        raise ValueError(f"Unsupported routed_score_func {other!r}; expected 'sigmoid' or ''.")
    if self.bias is None:
      return scores, scores
    return scores + self.bias[...].astype(scores.dtype), scores


def group_limited_mask(scores: jax.Array, *, n_routing_groups: int, topk_routing_group: int) -> jax.Array:
  """1.0 for experts in the top `topk_routing_group` groups, 0.0 elsewhere.

  A group's score is the sum of its two best expert scores (DeepSeek-V3's
  device-limited routing). Groups are contiguous expert index ranges.
  """
  num_experts = scores.shape[-1]
  grouped = jnp.reshape(scores, scores.shape[:-1] + (n_routing_groups, num_experts // n_routing_groups))
  top2, _ = jax.lax.top_k(grouped, k=2)
  group_scores = jnp.sum(top2.astype(jnp.float32), axis=-1)
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)
  group_mask = jnp.sum(jax.nn.one_hot(group_idx, n_routing_groups, dtype=jnp.float32), axis=-2)
  mask = jnp.broadcast_to(group_mask[..., None], group_mask.shape + (num_experts // n_routing_groups,))
  return jnp.reshape(mask, scores.shape)


class DeepSeekV3RoutedExperts(nnx.Module):
  """Router plus routed experts (legacy checkpoint name: `MoeBlock_0`)."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    s = sharding
    self.executor = moe.moe_from_config(config, s)
    self.gate = DeepSeekV3Router(config, rngs=rngs, sharding=s)
    num_experts, emb, mlp = config.num_experts, config.emb_dim, config.moe_mlp_dim
    # Legacy's axes for the expert kernels (`layers/moe.py`: in_axis 0, out_axis 1), kept so that
    # one config initializes both stacks from the same distribution.
    init = nnx.initializers.variance_scaling(config.dense_init_scale, "fan_in", "truncated_normal", in_axis=0, out_axis=1)
    wi_axes = ("experts", "expert_embed", "expert_mlp")
    self.wi_0 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype), **param_metadata(s, "wi_0", wi_axes)
    )
    self.wi_1 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype), **param_metadata(s, "wi_1", wi_axes)
    )
    self.wo = nnx.Param(
        init(rngs.params(), (num_experts, mlp, emb), config.weight_dtype),
        **param_metadata(s, "wo", ("experts", "expert_mlp", "expert_embed")),
    )

  def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns `(weights, indices)` of the experts each token is sent to.

    Experts are *selected* on the biased scores and *weighted* by the unbiased
    ones, so the bias moves load without changing what a chosen expert
    contributes. Sigmoid weights are normalized to sum to one before the
    `routed_scaling_factor`.
    """
    cfg = self.config
    selection, unbiased = self.gate(x)
    if cfg.n_routing_groups > 0:
      mask = group_limited_mask(
          selection, n_routing_groups=cfg.n_routing_groups, topk_routing_group=cfg.topk_routing_group
      )
      selection = jnp.where(mask > 0, selection, -jnp.inf)
    _, indices = jax.lax.top_k(selection, k=cfg.num_experts_per_tok)
    weights = jnp.take_along_axis(unbiased, indices, axis=-1)
    if cfg.routed_score_func == "sigmoid":
      weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return weights.astype(cfg.dtype), indices

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.astype(self.config.dtype)
    weights, indices = self.route(x)
    out = self.executor(x, weights, indices, self.wi_0[...], self.wi_1[...], self.wo[...])
    return checkpoint_name(out, "mlpwo")


class DeepSeekV3MoE(nnx.Module):
  """Routed experts plus shared experts (legacy checkpoint name: `DeepSeekMoeBlock_0`)."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs, sharding: Sharding):
    self.routed_experts = DeepSeekV3RoutedExperts(config, rngs=rngs, sharding=sharding)
    self.shared_experts = (
        DeepSeekV3MLP(config, config.shared_experts * config.moe_mlp_dim, rngs=rngs, sharding=sharding)
        if config.shared_experts > 0
        else None
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    out = self.routed_experts(x)
    if self.shared_experts is not None:
      out = out + self.shared_experts(x)
    return out


# ----------------------------------------------------------------------------
# Decoder layers and model
# ----------------------------------------------------------------------------


class _DeepSeekV3Layer(nnx.Module):
  """Shared skeleton: pre-norm MLA, then a pre-norm feed-forward block set by the subclass."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = DeepSeekV3Attention(config, mesh, rngs=rngs, sharding=sharding)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "post_self_attention_layer_norm", rngs)
    )

  def feed_forward(self, x: jax.Array) -> jax.Array:
    raise NotImplementedError

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
  ) -> jax.Array:
    inputs = self.sharding.constrain(inputs, "decoder_layer_input", ("batch", "length", "embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    attn_out = self.self_attention(
        self.pre_self_attention_layer_norm(inputs),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    x = (inputs + attn_out).astype(self.config.dtype)
    return (x + self.feed_forward(self.post_self_attention_layer_norm(x))).astype(self.config.dtype)


class DeepSeekV3DenseLayer(_DeepSeekV3Layer):
  """The first `first_num_dense_layers` layers: MLA plus an ordinary MLP."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, sharding: Sharding):
    super().__init__(config, mesh, rngs=rngs, sharding=sharding)
    self.mlp = DeepSeekV3MLP(config, config.mlp_dim, rngs=rngs, sharding=sharding)

  def feed_forward(self, x: jax.Array) -> jax.Array:
    return self.mlp(x)


class DeepSeekV3MoeLayer(_DeepSeekV3Layer):
  """The remaining layers: MLA plus routed and shared experts."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, sharding: Sharding):
    super().__init__(config, mesh, rngs=rngs, sharding=sharding)
    self.moe_block = DeepSeekV3MoE(config, rngs=rngs, sharding=sharding)

  def feed_forward(self, x: jax.Array) -> jax.Array:
    return self.moe_block(x)


class DeepSeekV3Decoder(nnx.Module):
  """Dense layers, then MoE layers, final norm and the untied output head."""

  def __init__(self, config: Config, mesh: Optional[Mesh] = None, *, rngs: nnx.Rngs, sharding: Sharding):
    self.config = config
    self.sharding = sharding
    s = sharding
    num_dense = min(config.first_num_dense_layers, config.num_decoder_layers)
    kw = {"rngs": rngs, "sharding": s}
    self.layers = nnx.Sequential(
        *(DeepSeekV3DenseLayer(config, mesh, **kw) for _ in range(num_dense)),
        *(DeepSeekV3MoeLayer(config, mesh, **kw) for _ in range(config.num_decoder_layers - num_dense)),
    )
    self.decoder_norm = nnx.RMSNorm(config.emb_dim, **norm_kwargs(config, s, "decoder_norm", rngs))
    self.logits_dense = nnx.Linear(
        config.emb_dim,
        config.vocab_size,
        use_bias=False,
        dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
        param_dtype=config.weight_dtype,
        kernel_metadata=param_metadata(s, "logits_dense", ("embed", "vocab")),
        rngs=rngs,
    )

  def __call__(
      self,
      y: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    s = self.sharding
    y = self.layers(y, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions)
    y = s.constrain(y, "pre_logits", ("batch", "length", "embed"))
    logits = self.logits_dense(self.decoder_norm(y))
    logits = s.constrain(logits, "logits", ("batch", "length", "vocab"))
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits


class DeepSeekV3Model(nnx.Module):
  """DeepSeek-V3 language model."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Optional[Sharding] = None,
      **kwargs,
  ):
    """Builds the model.

    Args:
      config: Model configuration.
      mesh: Device mesh of the driver, stored for drivers that read `model.mesh`.
      rngs: NNX random number generators.
      sharding: Resolves logical axis names to the mesh; `DeepSeekV3Sharding` if omitted.
        Parameters are placed from it at construction, under the mesh in scope.
      **kwargs: Ignored; accepted so every registered constructor takes the same call.
    """
    del kwargs
    if config.logits_via_embedding:
      raise ValueError("DeepSeek-V3 has an untied output head; set logits_via_embedding=false.")
    unsupported = unsupported_settings(config)
    if unsupported:
      raise ValueError(f"m3 DeepSeek-V3 does not implement {unsupported} yet.")
    ambient = jax.sharding.get_abstract_mesh()
    if sharding is None and not ambient.empty:
      check_mesh(ambient, DEEPSEEK_V3_MESH_AXES)
    self.config = config
    self.mesh = mesh
    self.sharding = sharding or DeepSeekV3Sharding()
    self.token_embedder = nnx.Embed(
        config.vocab_size,
        config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata=param_metadata(self.sharding, "token_embedder", ("vocab", "embed")),
        rngs=rngs,
    )
    self.decoder = DeepSeekV3Decoder(config, mesh, rngs=rngs, sharding=self.sharding)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      **kwargs,
  ) -> jax.Array:
    """Returns logits of shape `(batch, length, vocab_size)`. Other keyword arguments are ignored."""
    del kwargs
    s = self.sharding
    decoder_input_tokens = s.constrain(decoder_input_tokens, "decoder_input_tokens", ("batch", "length"))
    return self.decoder(
        self.token_embedder(decoder_input_tokens),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )


def create_deepseek_v3_model(
    config: Config,
    mesh: Optional[Mesh] = None,
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> DeepSeekV3Model:
  """Registry entry point; see `maxtext.m3.models.create_model`."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return DeepSeekV3Model(config, mesh, rngs=rngs, **kwargs)
