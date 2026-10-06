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

from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import Array, DType, ShardMode
from maxtext.layers.initializers import NdInitializer, nd_dense_init
from maxtext.layers.linears import DenseGeneral, situ_gate, situ_linear
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant


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
ROUTER_KERNEL_AXES = ("embed", None)
# Dense routed-expert kernels: wi_{0,1} [E, d, m], wo [E, m, d].
EXPERT_KERNEL_AXES = {
    "wi_0": ("exp", "embed_moe", "mlp_moe"),
    "wi_1": ("exp", "embed_moe", "mlp_moe"),
    "wo": ("exp", "mlp_moe", "embed_moe"),
}


class KimiMoERouter(nnx.Module):
  """Sigmoid router with auxiliary-loss-free bias correction."""

  def __init__(
      self,
      hidden_size: int,
      num_experts: int,
      top_k: int = 16,
      routed_scaling_factor: float = 1.0,
      moe_renormalize: bool = True,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    self.hidden_size = hidden_size
    self.num_experts = num_experts
    self.top_k = top_k
    self.routed_scaling_factor = routed_scaling_factor
    self.moe_renormalize = moe_renormalize
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.kernel = nnx.Param(
        jax.random.normal(rngs.params(), (hidden_size, num_experts), dtype=weight_dtype) * (1.0 / (hidden_size**0.5)),
        out_sharding=ROUTER_KERNEL_AXES,
    )
    self.e_score_correction_bias = nnx.Param(
        jnp.zeros((num_experts,), dtype=weight_dtype),
        out_sharding=(None,),
    )

  def __call__(self, hidden_states: Array) -> tuple[Array, Array]:
    """Calculates top-k expert indices and normalized routing weights.

    Args:
      hidden_states: Input tensor of shape [..., hidden_size].

    Returns:
      topk_idx: Expert indices of shape [..., top_k].
      topk_weight: Normalized routing weights of shape [..., top_k].
    """
    orig_shape = hidden_states.shape
    x_flat = hidden_states.reshape(-1, orig_shape[-1]).astype(jnp.float32)
    logits = jnp.matmul(x_flat, self.kernel.value.astype(jnp.float32))
    scores = jax.nn.sigmoid(logits)

    scores_for_choice = scores + self.e_score_correction_bias.value.astype(jnp.float32)
    _, topk_idx = jax.lax.top_k(scores_for_choice, self.top_k)

    topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)

    if self.top_k > 1 and self.moe_renormalize:
      denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
      topk_weight = topk_weight / denominator

    topk_weight = topk_weight * self.routed_scaling_factor

    batch_shape = orig_shape[:-1]
    topk_idx = topk_idx.reshape(batch_shape + (self.top_k,))
    topk_weight = topk_weight.reshape(batch_shape + (self.top_k,)).astype(self.dtype)

    return topk_idx, topk_weight


class KimiRoutedExperts(nnx.Module):
  """Routed experts with SiTU activation.

  Keeps dense float kernels `wi_0/wi_1: [E, d, m]`, `wo: [E, m, d]`
  (the dense dtype is `weight_dtype`).
  """

  def __init__(
      self,
      num_experts: int,
      in_features: int,
      intermediate_dim: int,
      hidden_size: int,
      top_k: int = 16,
      routed_scaling_factor: float = 1.0,
      moe_renormalize: bool = True,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.gate = KimiMoERouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        moe_renormalize=moe_renormalize,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )

    scale_in = 1.0 / (in_features**0.5)
    scale_inter = 1.0 / (intermediate_dim**0.5)
    shapes = {
        "wi_0": ((num_experts, in_features, intermediate_dim), scale_in),
        "wi_1": ((num_experts, in_features, intermediate_dim), scale_in),
        "wo": ((num_experts, intermediate_dim, in_features), scale_inter),
    }
    for name, (shape, init_scale) in shapes.items():
      kernel = jax.random.normal(rngs.params(), shape, dtype=weight_dtype) * init_scale
      setattr(self, name, nnx.Param(kernel, out_sharding=EXPERT_KERNEL_AXES[name]))

  def _selected(self, name: str, topk_idx: Array) -> Array:
    """Returns the gathered kernel `[N, top_k, ...]` for `name`."""
    return getattr(self, name).value[topk_idx]

  def __call__(
      self,
      x: Array,
      topk_idx: Array,
      topk_weight: Array,
  ) -> Array:
    """Computes routed expert output.

    Args:
      x: Latent input tensor of shape [N, in_features].
      topk_idx: Expert indices of shape [N, top_k].
      topk_weight: Expert weights of shape [N, top_k].

    Returns:
      Routed output of shape [N, in_features].
    """
    wi_0_selected = self._selected("wi_0", topk_idx)  # [N, top_k, in_features, intermediate_dim]
    wi_1_selected = self._selected("wi_1", topk_idx)  # [N, top_k, in_features, intermediate_dim]
    wo_selected = self._selected("wo", topk_idx)  # [N, top_k, intermediate_dim, in_features]

    gate = jnp.einsum("nd,nkdm->nkm", x, wi_0_selected)
    up = jnp.einsum("nd,nkdm->nkm", x, wi_1_selected)
    act = situ_gate(gate) * situ_linear(up)

    expert_out = jnp.einsum("nkm,nkmd->nkd", act, wo_selected)
    routed_out = jnp.einsum("nk,nkd->nd", topk_weight, expert_out)

    return routed_out.astype(self.dtype)


class KimiLatentMoEBlock(nnx.Module):
  """Kimi-K3 Latent MoE Block with routed and shared experts."""

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
        kernel_axes=("embed", "latent_moe"),
        quant=quant,
        shard_mode=shard_mode,
        matmul_precision=matmul_precision,
        mesh=mesh,
        rngs=rngs,
    )

    # 896 Routed Experts: 3584 -> 3072 -> 3584
    self.routed_experts = KimiRoutedExperts(
        num_experts=num_experts,
        in_features=routed_expert_hidden_size,
        intermediate_dim=moe_intermediate_size,
        hidden_size=hidden_size,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        moe_renormalize=moe_renormalize,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
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
        kernel_axes=("latent_moe", "embed"),
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
    x_flat = hidden_states.reshape(-1, orig_shape[-1])

    # Router gating (runs on original hidden size)
    topk_idx, topk_weight = self.routed_experts.gate(x_flat)

    # Down-project to latent dimension: 7168 -> 3584
    x_latent = self.routed_expert_down_proj(x_flat)

    # Routed expert computation
    y_routed = self.routed_experts(x_latent, topk_idx, topk_weight)

    # Post-expert RMSNorm
    if self.routed_expert_norm is not None:
      y_routed = self.routed_expert_norm(y_routed)

    # Up-project back to model dimension: 3584 -> 7168
    y_routed = self.routed_expert_up_proj(y_routed)
    y_routed = y_routed.reshape(orig_shape)

    # Shared experts addition
    if self.shared_expert is not None:
      y_shared = self.shared_expert(identity)
      y_routed = y_routed + y_shared

    return y_routed.astype(self.dtype)
