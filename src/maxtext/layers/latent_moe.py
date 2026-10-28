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

"""Latent MoE and SiTU-activated MLP layers for Kimi-K3."""

import math
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import Array, Config, DType, ShardMode, get_quant, get_weight_dtype
from maxtext.layers import moe
from maxtext.layers import mxfp4
from maxtext.layers.initializers import NdInitializer, nd_dense_init
from maxtext.layers.linears import DenseGeneral, situ_gate, situ_linear
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_logging


class KimiDenseMLP(nnx.Module):
  """Dense MLP with SiTU activations (Layer 0 and shared experts)."""

  def __init__(
      self,
      in_features: int,
      intermediate_dim: int,
      out_features: Optional[int] = None,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      quant: Optional[Quant] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
  ):
    out_features = out_features or in_features
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.out_features = out_features
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.wi_0 = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=intermediate_dim,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )
    self.wi_1 = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=intermediate_dim,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )
    self.wo = DenseGeneral(
        in_features_shape=intermediate_dim,
        out_features_shape=out_features,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      out_sharding: Optional[NamedSharding] = None,
  ) -> Array:
    """Forward pass of KimiDenseMLP."""
    gate = self.wi_0(x)
    up = self.wi_1(x)
    act = situ_gate(gate) * situ_linear(up)
    return self.wo(act.astype(self.dtype), out_sharding=out_sharding)


# Logical sharding axes (same names as `moe.RoutedMoE` / the DeepSeek gate).
ROUTER_KERNEL_AXES = ("embed_router", None)
# MXFP4 packed codes / E8M0 scales, wi_{0,1} [E, d/2|d/32, m], wo [E, m/2|m/32, d]: same axes as
# RoutedMoE's dense kernels, but the quantized contraction axis (-2) is left unsharded so
# 32-element MXFP4 groups never straddle devices (and d/32 need not divide the mesh).
EXPERT_PACKED_AXES = {
    "wi_0": ("exp", None, "mlp_moe"),
    "wi_1": ("exp", None, "mlp_moe"),
    "wo": ("exp", None, "embed_moe"),
}

ROUTED_EXPERT_WEIGHT_FORMATS = ("bf16", "mxfp4")


class KimiMxfp4RoutedExperts(nnx.Module):
  """Router + MXFP4-packed routed experts with SiTU activation (inference / parity only).

  The router is the same `moe.GateLogit` that `moe.RoutedMoE` builds (params `gate/kernel`,
  `gate/bias`), followed by RoutedMoE's DeepSeek-style routing: top-k on the biased sigmoid
  scores, weights = pre-bias scores renormalized and scaled by `routed_scaling_factor`. Only
  the expert weights differ: they stay in the released MXFP4 form, in MaxText orientation and
  quantized along each kernel's contraction axis (-2):
    `wi_{0,1}_packed: uint8 [E, d/2, m]`, `wi_{0,1}_scale: uint8 [E, d/32, m]`,
    `wo_packed: uint8 [E, m/2, d]`, `wo_scale: uint8 [E, m/32, d]`.
  The packed bytes are gathered per token and only the selected top-k experts are
  dequantized, so a dense copy of all experts never exists. The per-token gather is meant for
  CPU parity runs, not for sharded TPU execution; the uint8 params are not trainable (the
  config rejects `routed_experts_weight_format=mxfp4` outside `model_call_mode=inference`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh],
      num_experts: int,
      num_experts_per_tok: int,
      intermediate_dim: int,
      kernel_init: NdInitializer,
      kernel_axes: tuple,
      dtype: DType = jnp.float32,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    if config.routed_score_func != "sigmoid" or not config.routed_bias or config.n_routing_groups != -1:
      raise NotImplementedError(
          "mxfp4 routed experts implement sigmoid + bias routing without routing groups "
          f"(got routed_score_func={config.routed_score_func!r}, routed_bias={config.routed_bias}, "
          f"n_routing_groups={config.n_routing_groups})."
      )
    if config.hardware != "cpu":
      max_logging.warning(
          "routed_experts_weight_format=mxfp4 gathers and dequantizes per-token expert kernels; it is "
          "meant for CPU inference/parity runs and is slow and memory-hungry on accelerators."
      )
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.routed_scaling_factor = config.routed_scaling_factor
    self.dtype = dtype

    gate_weight_dtype = get_weight_dtype(config, "gate")
    self.gate = moe.GateLogit(
        in_features_shape=config.moe_router_input_dim,
        out_features_shape=num_experts,
        mesh=mesh,
        model_name=config.model_name,
        dtype=jnp.float32 if config.float32_gate_logits else dtype,
        weight_dtype=gate_weight_dtype,
        quant=get_quant(config, "gate", quant),
        kernel_init=kernel_init,
        kernel_axes=kernel_axes,
        use_bias=True,
        bias_dtype=jnp.float32 if config.float32_gate_logits else gate_weight_dtype,
        score_func="sigmoid",
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    in_features = config.moe_expert_input_dim
    shapes = {
        "wi_0": ((num_experts, in_features, intermediate_dim), 1.0 / (in_features**0.5)),
        "wi_1": ((num_experts, in_features, intermediate_dim), 1.0 / (in_features**0.5)),
        "wo": ((num_experts, intermediate_dim, in_features), 1.0 / (intermediate_dim**0.5)),
    }
    for name, (shape, init_scale) in shapes.items():
      packed, scale = self._init_mxfp4(rngs.params(), shape, init_scale)
      setattr(self, f"{name}_packed", nnx.Param(packed, out_sharding=EXPERT_PACKED_AXES[name]))
      setattr(self, f"{name}_scale", nnx.Param(scale, out_sharding=EXPERT_PACKED_AXES[name]))

  @staticmethod
  def _init_mxfp4(key, shape, init_scale):
    """Random packed init with roughly the same magnitude as a dense init.

    Codes are uniform over the 16 E2M1 values (std ~2.9); the shared E8M0 exponent is
    chosen so the decoded std is ~`init_scale`. Only used for from-scratch init; real
    runs restore the released packed bytes.
    """
    if shape[-2] % mxfp4.MXFP4_GROUP_SIZE:
      raise ValueError(f"contraction dim {shape[-2]} must be a multiple of {mxfp4.MXFP4_GROUP_SIZE} for mxfp4")
    packed_shape = shape[:-2] + (shape[-2] // 2, shape[-1])
    scale_shape = shape[:-2] + (shape[-2] // mxfp4.MXFP4_GROUP_SIZE, shape[-1])
    packed = jax.random.randint(key, packed_shape, 0, 256, dtype=jnp.int32).astype(jnp.uint8)
    exponent = int(round(math.log2(init_scale / 2.9)))
    scale = jnp.full(scale_shape, mxfp4.E8M0_BIAS + exponent, dtype=jnp.uint8)
    return packed, scale

  def route(self, gate_inputs: Array) -> tuple[Array, Array]:
    """RoutedMoE's DeepSeek routing (no groups): `gate_inputs [N, H]` -> (idx, weights) `[N, k]`."""
    biased_scores, scores = self.gate(gate_inputs)
    _, topk_idx = jax.lax.top_k(biased_scores, self.num_experts_per_tok)
    topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
    topk_weight = topk_weight / (topk_weight.sum(-1, keepdims=True) + 1e-20) * self.routed_scaling_factor
    return topk_idx, topk_weight.astype(self.dtype)

  def _selected(self, name: str, topk_idx: Array) -> Array:
    """Returns the gathered, dequantized kernel `[N, top_k, ...]` for `name`."""
    packed = getattr(self, f"{name}_packed").value[topk_idx]
    scale = getattr(self, f"{name}_scale").value[topk_idx]
    return mxfp4.dequantize_mxfp4(packed, scale, axis=-2, dtype=self.dtype)

  def apply_experts(self, x: Array, topk_idx: Array, topk_weight: Array) -> Array:
    """Weighted sum of the selected experts: `x [N, d]`, `topk_* [N, k]` -> `[N, d]`."""
    wi_0_selected = self._selected("wi_0", topk_idx)  # [N, top_k, in_features, intermediate_dim]
    wi_1_selected = self._selected("wi_1", topk_idx)  # [N, top_k, in_features, intermediate_dim]
    wo_selected = self._selected("wo", topk_idx)  # [N, top_k, intermediate_dim, in_features]

    gate = jnp.einsum("nd,nkdm->nkm", x, wi_0_selected)
    up = jnp.einsum("nd,nkdm->nkm", x, wi_1_selected)
    act = situ_gate(gate) * situ_linear(up)

    expert_out = jnp.einsum("nkm,nkmd->nkd", act, wo_selected)
    routed_out = jnp.einsum("nk,nkd->nd", topk_weight, expert_out)
    return routed_out.astype(self.dtype)

  def __call__(self, inputs: Array, gate_inputs: Array) -> Array:
    """Routes on `gate_inputs [N, H]` and runs the experts on the latent `inputs [N, d]`."""
    topk_idx, topk_weight = self.route(gate_inputs)
    return self.apply_experts(inputs, topk_idx, topk_weight)


class KimiLatentMoEBlock(nnx.Module):
  """Kimi-K3 Latent MoE Block with routed and shared experts.

  The router reads the full hidden state; the routed experts run on a down-projected latent
  (7168 -> 3584), followed by an RMSNorm and an up-projection; shared experts read the full
  hidden state.

  With `routed_experts_weight_format="bf16"` the router and routed experts are a
  `moe.RoutedMoE` (sorted/grouped dispatch: megablox / tokamax GMM / `ragged_dot`, expert
  parallelism), which needs `config` and `mesh`. The config must set
  `moe_expert_input_dim=routed_expert_hidden_size` and `moe_router_input_dim=hidden_size`;
  the latent is the expert input and the hidden state is passed as `gate_inputs`.

  With `"mxfp4"` they are a `KimiMxfp4RoutedExperts`: the same `GateLogit` router and routing,
  with the released packed expert bytes gathered and dequantized per token (CPU inference /
  parity only until MXFP4 lands in RoutedMoE).
  """

  def __init__(
      self,
      hidden_size: int = 7168,
      num_experts: int = 896,
      top_k: int = 16,
      routed_expert_hidden_size: int = 3584,
      moe_intermediate_size: int = 3072,
      num_shared_experts: int = 2,
      shared_intermediate_size: Optional[int] = None,
      routed_scaling_factor: float = 1.0,
      moe_renormalize: bool = True,
      latent_moe_use_norm: bool = True,
      rms_norm_eps: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      quant: Optional[Quant] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      mesh: Optional[Mesh] = None,
      routed_experts_weight_format: str = "bf16",
      config: Optional[Config] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.hidden_size = hidden_size
    self.num_experts = num_experts
    self.top_k = top_k
    self.routed_expert_hidden_size = routed_expert_hidden_size
    self.moe_intermediate_size = moe_intermediate_size
    self.latent_moe_use_norm = latent_moe_use_norm
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    # Routed down projection: 7168 -> 3584
    self.routed_expert_down_proj = DenseGeneral(
        in_features_shape=hidden_size,
        out_features_shape=routed_expert_hidden_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        # The 3584-wide latent axis is deliberately unsharded (replicated); `embed` carries the sharding.
        kernel_axes=("embed", None),
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )

    # 896 Routed Experts: 3584 -> 3072 -> 3584 (router on the 7168 hidden state)
    if config is None or mesh is None:
      raise ValueError("KimiLatentMoEBlock's routed experts need `config` and `mesh`.")
    expected = {
        "moe_expert_input_dim": routed_expert_hidden_size,
        "moe_router_input_dim": hidden_size,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "routed_scaling_factor": routed_scaling_factor,
    }
    mismatched = {k: (getattr(config, k), v) for k, v in expected.items() if getattr(config, k) != v}
    if mismatched:
      raise ValueError(f"config disagrees with KimiLatentMoEBlock args (config, arg): {mismatched}")
    if not moe_renormalize:
      raise ValueError("RoutedMoE always renormalizes sigmoid top-k weights; moe_renormalize=False is unsupported.")
    # This block returns only the MoE output (the decoder layer expects a single array), so the RoutedMoE
    # load-balance loss would be dropped; fail fast instead of training without it. (Router-bias updates,
    # `routed_bias_update_rate > 0`, are already rejected by config validation for non-DeepSeek decoder blocks.)
    if config.load_balance_loss_weight > 0.0:
      raise NotImplementedError(
          "KimiLatentMoEBlock does not propagate the MoE load-balance loss yet;"
          f" got load_balance_loss_weight={config.load_balance_loss_weight}. Set it to 0."
      )
    if routed_experts_weight_format == "bf16":
      self.routed_experts = moe.RoutedMoE(
          config=config,
          num_experts=num_experts,
          num_experts_per_tok=top_k,
          mesh=mesh,
          kernel_init=kernel_init,
          kernel_axes=ROUTER_KERNEL_AXES,
          intermediate_dim=moe_intermediate_size,
          weight_dtype=weight_dtype,
          dtype=dtype,
          quant=quant,
          rngs=rngs,
      )
    elif routed_experts_weight_format == "mxfp4":
      self.routed_experts = KimiMxfp4RoutedExperts(
          config=config,
          mesh=mesh,
          num_experts=num_experts,
          num_experts_per_tok=top_k,
          intermediate_dim=moe_intermediate_size,
          kernel_init=kernel_init,
          kernel_axes=ROUTER_KERNEL_AXES,
          dtype=dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      raise ValueError(
          f"routed_experts_weight_format must be one of {ROUTED_EXPERT_WEIGHT_FORMATS}, "
          f"got {routed_experts_weight_format!r}"
      )

    # Post-expert RMSNorm: 3584
    if self.latent_moe_use_norm:
      self.routed_expert_norm = RMSNorm(
          num_features=routed_expert_hidden_size,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=("norm",),
          shard_mode=shard_mode,
          rngs=rngs,
      )
    else:
      self.routed_expert_norm = None

    # Routed up projection: 3584 -> 7168
    self.routed_expert_up_proj = DenseGeneral(
        in_features_shape=routed_expert_hidden_size,
        out_features_shape=hidden_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=(None, "embed"),  # latent axis unsharded, as in the down projection.
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )

    # Shared Experts: 7168 -> 6144 -> 7168
    if num_shared_experts > 0:
      shared_dim = shared_intermediate_size or (moe_intermediate_size * num_shared_experts)
      self.shared_expert = KimiDenseMLP(
          in_features=hidden_size,
          intermediate_dim=shared_dim,
          out_features=hidden_size,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_init=kernel_init,
          quant=quant,
          shard_mode=shard_mode,
          matmul_precision=matmul_precision,
          mesh=mesh,
          rngs=rngs,
      )
    else:
      self.shared_expert = None

  def __call__(
      self,
      hidden_states: Array,
      out_sharding: Optional[NamedSharding] = None,
  ) -> Array:
    """Forward pass of KimiLatentMoEBlock.

    Args:
      hidden_states: Input tensor of shape [..., hidden_size].
      out_sharding: Optional sharding specification.

    Returns:
      Output tensor of shape [..., hidden_size].
    """
    identity = hidden_states
    orig_shape = hidden_states.shape

    if isinstance(self.routed_experts, moe.RoutedMoE):
      # Down-project to the latent (7168 -> 3584); the router reads the full hidden state.
      x_latent = self.routed_expert_down_proj(hidden_states)
      y_routed, _, _ = self.routed_experts(x_latent, gate_inputs=hidden_states)
    else:
      x_flat = hidden_states.reshape(-1, orig_shape[-1])
      x_latent = self.routed_expert_down_proj(x_flat)
      y_routed = self.routed_experts(x_latent, gate_inputs=x_flat)

    # Post-expert RMSNorm
    if self.routed_expert_norm is not None:
      y_routed = self.routed_expert_norm(y_routed)

    # Up-project back to model dimension: 3584 -> 7168
    y_routed = self.routed_expert_up_proj(y_routed, out_sharding=out_sharding)
    y_routed = y_routed.reshape(orig_shape)

    # Shared experts addition
    if self.shared_expert is not None:
      y_shared = self.shared_expert(identity, out_sharding=out_sharding)
      y_routed = y_routed + y_shared

    return y_routed.astype(self.dtype)
