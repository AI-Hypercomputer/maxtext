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
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import Array, Config, DType, ShardMode
from maxtext.layers import moe
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
ROUTER_KERNEL_AXES = ("embed_router", None)


class KimiLatentMoEBlock(nnx.Module):
  """Kimi-K3 Latent MoE Block with routed and shared experts.

  The router reads the full hidden state; the routed experts run on a down-projected latent
  (7168 -> 3584), followed by an RMSNorm and an up-projection; shared experts read the full
  hidden state.

  The router and routed experts are a `moe.RoutedMoE` (sorted/grouped dispatch: megablox /
  tokamax GMM / `ragged_dot`, expert parallelism), which needs `config` and `mesh`. The config
  must set `moe_expert_input_dim=routed_expert_hidden_size` and
  `moe_router_input_dim=hidden_size`; the latent is the expert input and the hidden state is
  passed as `gate_inputs`.
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
      raise ValueError("KimiLatentMoEBlock's routed experts are a moe.RoutedMoE, which needs `config` and `mesh`.")
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

    # Down-project to the latent (7168 -> 3584); the router reads the full hidden state.
    x_latent = self.routed_expert_down_proj(hidden_states)
    y_routed, _, _ = self.routed_experts(x_latent, gate_inputs=hidden_states)

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
