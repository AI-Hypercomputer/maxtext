# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V3 in m3 format.

Three things set DeepSeek-V3 apart from a Qwen3-style decoder, and all three
are defined in this file:

  * **Multi-head latent attention (MLA).** Queries and keys/values go through
    low-rank latents (`q_lora_rank`, `kv_lora_rank`), and only a small
    `qk_rope_head_dim` slice of each head carries position via YaRN RoPE.
  * **Sigmoid routing with a balancing bias and expert groups.** Router scores
    are sigmoids; a per-expert bias (`RouterBias`, not a parameter) steers
    load balancing without an auxiliary loss; with `n_routing_groups` set,
    top-k is restricted to the best `topk_routing_group` groups of experts.
  * **Shared experts** alongside the routed ones, and
    `first_num_dense_layers` ordinary MLP layers before the MoE layers.

The expert matmuls come from `maxtext.m3.core.moe`; the softmax attention
arithmetic from `maxtext.m3.core.attention`. Parameter names match the legacy
checkpoint layout, including its historical `DeepSeekMoeBlock_0.MoeBlock_0`
nesting. The dense and MoE layers share one `Layers` container, in order;
under `scan_layers` they become two scanned runs (see `maxtext.m3.infra.scan`).

Not here yet: the loss-free bias *update* (`routed_bias_update_rate`),
multi-token prediction, and the DeepSeek-V3.2 indexer.
"""

import math

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core import moe
from maxtext.m3.core import kernels
from maxtext.m3.core.attention import dot_product_attention
from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.core.layers import Layers
from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType

Carry = tuple[jax.Array, jax.Array]


class DeepSeekV3Sharding(DefaultSharding):
  """Default rules plus MLA's latent axes, which are small and stay replicated."""

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    if axis in ("q_lora", "kv_lora"):
      return None
    return super().map_axis(axis, tensor_name, tensor_type)


class RouterBias(nnx.Variable):  # pylint: disable=abstract-method
  """Per-expert routing bias for auxiliary-loss-free load balancing.

  Deliberately not an `nnx.Param`: it is adjusted by a counting rule, not by
  gradients, and the training engine differentiates `nnx.Param` only.
  """


def norm_kwargs(config: Config, sharding: Sharding, name: str, rngs: nnx.Rngs, axis: str = "norm") -> dict:
  return {
      "epsilon": config.normalization_layer_epsilon,
      "dtype": config.dtype,
      "param_dtype": config.weight_dtype,
      "scale_metadata": {"out_sharding": sharding(name, (axis,))},
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
  length, floored and ceiled when `rope_truncate` is set. Follows the
  DeepSeek-V3 reference implementation, with the same float32 device
  arithmetic as legacy `YarnRotaryEmbedding` so the two stay bit-comparable.
  Cheap enough to recompute per call; XLA folds it into a constant.
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
  queries and keys agree. Complex arithmetic in float32, as in legacy, so the
  two stacks stay bit-comparable.
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

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.num_heads = config.num_query_heads
    self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.rope_scale = yarn_attention_scaling(config)
    self.attention_kernel = kernels.attention_from_config(config, sharding)
    # DeepSeek folds YaRN's attention temperature into the softmax scale.
    self.softmax_scale = self.qk_head_dim**-0.5
    if config.max_position_embeddings > config.original_max_position_embeddings:
      mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
      self.softmax_scale *= mscale * mscale

    kw = {"dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    emb, heads = config.emb_dim, self.num_heads
    if config.q_lora_rank == 0:
      self.query = nnx.Einsum(
          "...e,ehd->...hd",
          (emb, heads, self.qk_head_dim),
          bias_shape=None,
          kernel_metadata={"out_sharding": sharding("query", ("embed", "heads", "head_dim"))},
          **kw,
      )
    else:
      self.wq_a = nnx.Linear(
          emb,
          config.q_lora_rank,
          use_bias=False,
          kernel_metadata={"out_sharding": sharding("wq_a", ("embed", "q_lora"))},
          **kw,
      )
      self.q_norm = nnx.RMSNorm(config.q_lora_rank, **norm_kwargs(config, sharding, "q_norm", rngs))
      self.wq_b = nnx.Einsum(
          "...r,rhd->...hd",
          (config.q_lora_rank, heads, self.qk_head_dim),
          bias_shape=None,
          kernel_metadata={"out_sharding": sharding("wq_b", ("q_lora", "heads", "head_dim"))},
          **kw,
      )
    self.wkv_a = nnx.Linear(
        emb,
        config.kv_lora_rank + config.qk_rope_head_dim,
        use_bias=False,
        kernel_metadata={"out_sharding": sharding("wkv_a", ("embed", "kv_lora"))},
        **kw,
    )
    self.kv_norm = nnx.RMSNorm(config.kv_lora_rank, **norm_kwargs(config, sharding, "kv_norm", rngs))
    self.wkv_b = nnx.Einsum(
        "...r,rhd->...hd",
        (config.kv_lora_rank, heads, config.qk_nope_head_dim + config.v_head_dim),
        bias_shape=None,
        kernel_metadata={"out_sharding": sharding("wkv_b", ("kv_lora", "heads", "head_dim"))},
        **kw,
    )
    self.out = nnx.Einsum(
        "...hd,hde->...e",
        (heads, config.v_head_dim, emb),
        bias_shape=None,
        kernel_metadata={"out_sharding": sharding("out", ("heads", "head_dim", "embed"))},
        **kw,
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

  def __call__(self, x: jax.Array, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None):
    query = checkpoint_name(self._query(x, decoder_positions), "query_proj")
    key, value = self._key_value(x, decoder_positions)
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")
    out = dot_product_attention(
        query,
        key,
        value,
        positions=decoder_positions,
        segment_ids=decoder_segment_ids,
        float32_logits=self.config.float32_logits,
        kernel=self.attention_kernel,
    )
    out = checkpoint_name(
        self.sharding.constrain(out, "attention_out", ("batch", "length", "heads", "head_dim")), "attention_out"
    )
    return checkpoint_name(self.out(out), "out_proj")


# ----------------------------------------------------------------------------
# Feed-forward: dense MLP, router, routed and shared experts
# ----------------------------------------------------------------------------


class DeepSeekV3MLP(nnx.Module):
  """SwiGLU MLP; also the shared experts, with `intermediate_dim` their combined width."""

  def __init__(self, config: Config, intermediate_dim: int, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    kw = {"use_bias": False, "dtype": config.dtype, "param_dtype": config.weight_dtype, "rngs": rngs}
    emb = config.emb_dim
    self.wi_0 = nnx.Linear(
        emb, intermediate_dim, kernel_metadata={"out_sharding": sharding("wi_0", ("embed", "mlp"))}, **kw
    )
    self.wi_1 = nnx.Linear(
        emb, intermediate_dim, kernel_metadata={"out_sharding": sharding("wi_1", ("embed", "mlp"))}, **kw
    )
    self.wo = nnx.Linear(intermediate_dim, emb, kernel_metadata={"out_sharding": sharding("wo", ("mlp", "embed"))}, **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = checkpoint_name(self.wi_0(x), "mlpwi_0")
    up = checkpoint_name(self.wi_1(x), "mlpwi_1")
    hidden = (jax.nn.silu(gate) * up).astype(self.config.dtype)
    hidden = self.sharding.constrain(hidden, "mlp_hidden", ("batch", "length", "mlp"))
    return checkpoint_name(self.wo(hidden), "mlpwo")


class DeepSeekV3Router(nnx.Module):
  """Sigmoid router with a balancing bias; attribute name `gate` in the checkpoint."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.dtype = jnp.float32 if config.float32_gate_logits else config.dtype
    init = nnx.initializers.variance_scaling(config.dense_init_scale, "fan_in", "truncated_normal")
    # The router's expert dimension is replicated (legacy: `kernel_axes=("embed", None)`):
    # every device needs every expert's score to route.
    self.kernel = nnx.Param(
        init(rngs.params(), (config.emb_dim, config.num_experts), config.weight_dtype),
        out_sharding=sharding("gate", ("embed", None)),
    )
    if config.routed_bias:
      self.bias = RouterBias(jnp.zeros((config.num_experts,), self.dtype), out_sharding=sharding("gate_bias", (None,)))
    else:
      self.bias = None

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns `(scores_for_selection, scores_for_weights)`: the latter excludes the bias."""
    logits = jnp.dot(x.astype(self.dtype), self.kernel[...].astype(self.dtype))
    logits = self.sharding.constrain(logits, "router_logits", ("batch", "length", "experts"))
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
  """Router plus routed experts; attribute name `MoeBlock_0` in the checkpoint."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.executor = moe.executor_from_config(config, sharding)
    self.gate = DeepSeekV3Router(config, sharding=sharding, rngs=rngs)
    num_experts, emb, mlp = config.num_experts, config.emb_dim, config.moe_mlp_dim
    init = nnx.initializers.variance_scaling(config.dense_init_scale, "fan_in", "truncated_normal", in_axis=1, out_axis=2)
    self.wi_0 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype),
        out_sharding=sharding("wi_0", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wi_1 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype),
        out_sharding=sharding("wi_1", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wo = nnx.Param(
        init(rngs.params(), (num_experts, mlp, emb), config.weight_dtype),
        out_sharding=sharding("wo", ("experts", "expert_mlp", "expert_embed")),
    )

  def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns `(weights, indices, selection_scores)`.

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
    return weights.astype(cfg.dtype), indices, selection

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    cfg = self.config
    x = x.astype(cfg.dtype)
    weights, indices, selection = self.route(x)
    out = moe.routed_experts(x, weights, indices, self.wi_0[...], self.wi_1[...], self.wo[...], executor=self.executor)
    out = checkpoint_name(out, "mlpwo")
    if cfg.load_balance_loss_weight > 0:
      probs = jax.nn.softmax(selection.astype(jnp.float32), axis=-1).astype(cfg.dtype)
      aux = moe.load_balance_loss(indices, probs, weight=cfg.load_balance_loss_weight)
    else:
      aux = jnp.zeros((), jnp.float32)
    return out, aux


class DeepSeekV3MoE(nnx.Module):
  """Routed experts plus shared experts; attribute name `DeepSeekMoeBlock_0` in the checkpoint."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.MoeBlock_0 = DeepSeekV3RoutedExperts(config, sharding=sharding, rngs=rngs)  # pylint: disable=invalid-name
    self.shared_experts = DeepSeekV3MLP(config, config.shared_experts * config.moe_mlp_dim, sharding=sharding, rngs=rngs)

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    routed, aux = self.MoeBlock_0(x)
    return routed + self.shared_experts(x), aux


# ----------------------------------------------------------------------------
# Decoder layers and model
# ----------------------------------------------------------------------------


class _DeepSeekV3Layer(nnx.Module):
  """Shared skeleton: pre-norm MLA, then a pre-norm feed-forward block set by the subclass."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = DeepSeekV3Attention(config, sharding=sharding, rngs=rngs)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "post_self_attention_layer_norm", rngs)
    )

  def _attend(self, x: jax.Array, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None) -> jax.Array:
    x = checkpoint_name(
        self.sharding.constrain(x, "decoder_layer_input", ("batch", "length", "embed")), "decoder_layer_input"
    )
    attn_out = self.self_attention(
        self.pre_self_attention_layer_norm(x),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return (x + attn_out).astype(self.config.dtype)


class DeepSeekV3DenseLayer(_DeepSeekV3Layer):
  """The first `first_num_dense_layers` layers: MLA plus an ordinary MLP."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    super().__init__(config, sharding=sharding, rngs=rngs)
    self.mlp = DeepSeekV3MLP(config, config.mlp_dim, sharding=sharding, rngs=rngs)

  def __call__(
      self, carry: Carry, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None
  ) -> Carry:
    x, aux_loss = carry
    x = self._attend(x, decoder_positions, decoder_segment_ids)
    x = (x + self.mlp(self.post_self_attention_layer_norm(x))).astype(self.config.dtype)
    return x, aux_loss


class DeepSeekV3MoeLayer(_DeepSeekV3Layer):
  """The remaining layers: MLA plus routed and shared experts."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    super().__init__(config, sharding=sharding, rngs=rngs)
    self.DeepSeekMoeBlock_0 = DeepSeekV3MoE(config, sharding=sharding, rngs=rngs)  # pylint: disable=invalid-name

  def __call__(
      self, carry: Carry, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None
  ) -> Carry:
    x, aux_loss = carry
    x = self._attend(x, decoder_positions, decoder_segment_ids)
    moe_out, layer_aux = self.DeepSeekMoeBlock_0(self.post_self_attention_layer_norm(x))
    return (x + moe_out).astype(self.config.dtype), aux_loss + layer_aux


class DeepSeekV3Decoder(nnx.Module):
  """Dense layers, then MoE layers, final norm and the untied output head."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    num_moe = config.num_decoder_layers - config.first_num_dense_layers
    self.layers = Layers(
        [DeepSeekV3DenseLayer(config, sharding=sharding, rngs=rngs) for _ in range(config.first_num_dense_layers)]
        + [DeepSeekV3MoeLayer(config, sharding=sharding, rngs=rngs) for _ in range(num_moe)]
    )
    self.decoder_norm = nnx.RMSNorm(config.emb_dim, **norm_kwargs(config, sharding, "decoder_norm", rngs))
    self.logits_dense = nnx.Linear(
        config.emb_dim,
        config.vocab_size,
        use_bias=False,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        kernel_metadata={"out_sharding": sharding("logits_dense", ("embed", "vocab"))},
        rngs=rngs,
    )

  def __call__(self, y: jax.Array, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None):
    carry: Carry = (y, jnp.zeros((), jnp.float32))
    y, aux_loss = self.layers(carry, decoder_positions=decoder_positions, decoder_segment_ids=decoder_segment_ids)
    logits = self.logits_dense(self.decoder_norm(y))
    logits = self.sharding.constrain(logits, "logits", ("batch", "length", "vocab"))
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits, aux_loss


class DeepSeekV3Model(nnx.Module):
  """DeepSeek-V3 language model."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None = None,
      *,
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
      **unused_kwargs,
  ):
    del mesh, unused_kwargs  # See `Qwen3Model.__init__`.
    if config.logits_via_embedding:
      raise ValueError("DeepSeek-V3 has an untied output head; set logits_via_embedding=false.")
    self.config = config
    self.sharding = sharding or DeepSeekV3Sharding()
    self.token_embedder = nnx.Embed(
        config.vocab_size,
        config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata={"out_sharding": self.sharding("token_embedder", ("vocab", "embed"))},
        rngs=rngs,
    )
    self.decoder = DeepSeekV3Decoder(config, sharding=self.sharding, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      **unused_kwargs,
  ) -> CausalLMOutput:
    """See `Qwen3Model.__call__`. `aux_loss` is the mean load-balance loss over MoE layers (zero unless weighted)."""
    del unused_kwargs
    num_moe = self.config.num_decoder_layers - self.config.first_num_dense_layers
    logits, aux_loss = self.decoder(
        self.token_embedder(decoder_input_tokens),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    return CausalLMOutput(logits=logits, aux_loss=aux_loss / max(num_moe, 1))


def create_deepseek_v3_model(
    config: Config, mesh: Mesh | None = None, *, rngs: nnx.Rngs | None = None, **kwargs
) -> DeepSeekV3Model:
  """Registry entry point; see `maxtext.m3.models.create_model`."""
  return DeepSeekV3Model(config, mesh, rngs=rngs or nnx.Rngs(0), **kwargs)
