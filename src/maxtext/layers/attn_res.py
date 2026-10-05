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

"""Attention Residuals (AttnRes) Highway Subsystem for Kimi-K3.

Kimi-K3 utilizes an Attention Residuals (AttnRes) highway architecture where
intermediate block representations are checkpointed every `attn_res_block_size`
layers (default: 12 layers). Before self-attention, before MLP, and before the
final layer norm / LM head, a learned dynamic softmax pooling operation
aggregates across the checkpointed block residuals and the running prefix sum:

  V = [block_residuals[:b], prefix_sum]
  k = RMSNorm(V, eps=epsilon)  (unscaled RMSNorm)
  w_score = w_norm * w_proj
  scores = sum(k * w_score, dim=-1)
  alpha = Softmax(scores, dim=-1)
  x_in = sum(alpha_i * V_i)
"""

from typing import Any, Optional, Tuple, Union

import flax.struct
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp

from maxtext.common.common_types import DType, ShardMode
from maxtext.layers import initializers, nnx_wrappers
from maxtext.layers.initializers import default_bias_init, nd_dense_init, variable_to_logically_partitioned


@flax.struct.dataclass
class AttnResState:
  """State container for Attention Residuals (AttnRes) Highway.

  Holds the static residual buffer of checkpointed blocks and the running prefix sum.

  Attributes:
    block_residuals: Checkpointed block residuals buffer of shape [B, S, max_blocks, D]
                     (e.g., [B, S, 8, 7168]).
    prefix_sum: Running prefix sum accumulator of shape [B, S, D] (or None).
    num_blocks: Number of active checkpointed blocks (0 <= num_blocks <= max_blocks).
    max_blocks: Maximum number of checkpointed block slots (default 8 for 93 layers / 12).
  """

  block_residuals: jnp.ndarray
  prefix_sum: Optional[jnp.ndarray] = None
  num_blocks: Union[int, jnp.ndarray] = 0
  max_blocks: int = 8

  @classmethod
  def create(
      cls,
      batch_size: int,
      seq_len: int,
      hidden_size: int = 7168,
      max_blocks: int = 8,
      dtype: Any = jnp.float32,
  ) -> "AttnResState":
    """Initializes an empty AttnResState with zeroed block buffer."""
    return cls(
        block_residuals=jnp.zeros((batch_size, seq_len, max_blocks, hidden_size), dtype=dtype),
        prefix_sum=None,
        num_blocks=0,
        max_blocks=max_blocks,
    )

  def add_block(self, block: jnp.ndarray) -> "AttnResState":
    """Appends a new checkpointed block to the residual buffer and resets prefix_sum.

    Args:
      block: Tensor of shape [B, S, D] to checkpoint as a new block.

    Returns:
      Updated AttnResState with new block checkpointed and prefix_sum reset to None.
    """
    new_residuals = self.block_residuals.at[:, :, self.num_blocks, :].set(block)
    return self.replace(
        block_residuals=new_residuals,
        prefix_sum=None,
        num_blocks=self.num_blocks + 1,
    )

  def update_prefix_sum(self, delta: jnp.ndarray) -> "AttnResState":
    """Adds a sublayer output delta to the running prefix sum.

    Args:
      delta: Tensor of shape [B, S, D] to add.

    Returns:
      Updated AttnResState with accumulated prefix_sum.
    """
    if self.prefix_sum is None:
      new_prefix_sum = delta
    else:
      new_prefix_sum = self.prefix_sum + delta
    return self.replace(prefix_sum=new_prefix_sum)

  def reset_prefix_sum(self) -> "AttnResState":
    """Resets prefix_sum accumulator to None."""
    return self.replace(prefix_sum=None)

  def get_active_blocks(self) -> jnp.ndarray:
    """Returns the slice of active checkpointed blocks: shape [B, S, num_blocks, D]."""
    return self.block_residuals[:, :, : self.num_blocks, :]


def _apply_attn_res(
    prefix_sum: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    epsilon: float = 1e-5,
    num_blocks: Optional[Union[int, jnp.ndarray]] = None,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Applies Attention Residual pooling over block residuals and prefix sum.

  Equivalent to PyTorch reference `_apply_attn_res`:
    V = [block_residual, prefix_sum]
    k = RMSNorm(V, eps=epsilon)  (unscaled RMSNorm)
    w_score = norm_weight * proj_weight
    scores = sum(k * w_score, dim=-1)
    probs = softmax(scores, dim=-1)
    hidden_states = sum(probs * V, dim=-2)

  Args:
    prefix_sum: Running prefix sum of shape [..., D].
    block_residual: Optional checkpointed block residuals of shape [..., K, D].
      If None or K=0, returns prefix_sum directly.
    proj_weight: Linear projection weight/kernel of shape [D, 1] or [1, D] or [D].
    norm_weight: Normalization scale of shape [D].
    epsilon: Variance epsilon for RMSNorm.
    num_blocks: Optional number of active blocks if block_residual is statically padded.
    mask: Optional boolean mask of shape [..., K+1] where True indicates active slots.

  Returns:
    Pooled hidden states of shape [..., D], matching prefix_sum.dtype.
  """
  # Fast path: if no block residuals are available and no static buffer is provided
  if block_residual is None or (num_blocks is None and block_residual.shape[-2] == 0):
    return prefix_sum

  orig_dtype = prefix_sum.dtype
  prefix_sum_expanded = jnp.expand_dims(prefix_sum, axis=-2)  # [..., 1, D]

  # Case 1: Static shape padding with active block count `num_blocks`
  if num_blocks is not None and block_residual.shape[-2] > 0:
    max_blocks = block_residual.shape[-2]
    # Static candidate buffer: shape [..., max_blocks + 1, D]
    # Initialize with [block_residual, zeros]
    zeros_slot = jnp.zeros_like(prefix_sum_expanded)
    v_init = jnp.concatenate([block_residual, zeros_slot], axis=-2)
    # Place prefix_sum at active slot index `num_blocks`
    v = v_init.at[..., num_blocks, :].set(prefix_sum)

    # Active mask: slots 0..num_blocks are True, slots > num_blocks are False
    if mask is None:
      slot_indices = jnp.arange(max_blocks + 1)
      mask = slot_indices <= num_blocks  # shape [max_blocks + 1]

  # Case 2: Dynamically sliced or explicit active block tensor [..., K, D]
  else:
    v = jnp.concatenate([block_residual, prefix_sum_expanded], axis=-2)  # [..., K+1, D]

  # RMSNorm without scale over last dimension in float32
  v_float = v.astype(jnp.float32)
  variance = jnp.mean(jnp.square(v_float), axis=-1, keepdims=True)
  k = v_float * lax.rsqrt(variance + epsilon)

  # Learned attention query weight: w_norm * w_proj (shape: [D])
  norm_w = norm_weight.astype(jnp.float32).reshape(-1)
  proj_w = proj_weight.astype(jnp.float32).reshape(-1)
  score_weight = norm_w * proj_w

  # Attention logits: dot product along D
  scores = jnp.sum(k * score_weight, axis=-1)  # [..., K+1]

  # Apply mask for static padding if present
  if mask is not None:
    scores = jnp.where(mask, scores, -1e9)

  # Softmax pooling weights
  probs = jax.nn.softmax(scores, axis=-1)  # [..., K+1]

  # Weighted sum: sum(probs * V)
  hidden_states = jnp.sum(jnp.expand_dims(probs, axis=-1) * v_float, axis=-2)

  return hidden_states.astype(orig_dtype)


def _apply_output_attn_res(
    hidden_states: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    epsilon: float = 1e-5,
    num_blocks: Optional[Union[int, jnp.ndarray]] = None,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Applies final output Attention Residual pooling before final RMSNorm and LM Head.

  In Kimi-K3 backbone, before the final RMSNorm:
    hidden_states = _apply_output_attn_res(
        hidden_states=final_prefix_sum,
        block_residual=block_residuals,
        proj_weight=output_attn_res_proj,
        norm_weight=output_attn_res_norm,
    )
  """
  return _apply_attn_res(
      prefix_sum=hidden_states,
      block_residual=block_residual,
      proj_weight=proj_weight,
      norm_weight=norm_weight,
      epsilon=epsilon,
      num_blocks=num_blocks,
      mask=mask,
  )


class KimiAttnResLayer(nnx.Module):
  """Attention Residuals (AttnRes) pooling layer for Kimi-K3.

  Encapsulates the normalization scale parameter and projection kernel parameter
  used for learned softmax pooling over residual highway representations.

  Attributes:
    norm_scale: Normalization scale parameter of shape [hidden_size].
    proj_kernel: Linear projection kernel parameter of shape [hidden_size, 1].
  """

  def __init__(
      self,
      hidden_size: int = 7168,
      epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_blocks: int = 8,
      scale_init: initializers.Initializer = default_bias_init,
      kernel_init: initializers.NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.hidden_size = hidden_size
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.max_blocks = max_blocks
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes

    if rngs is None:
      rngs = nnx.Rngs(0)

    # Normalization scale: shape [hidden_size]
    self.norm_scale = nnx.Param(
        scale_init(rngs.params(), (hidden_size,), weight_dtype),
        out_sharding=kernel_axes,
    )
    # Projection kernel: shape [hidden_size, 1] in MaxText convention
    self.proj_kernel = nnx.Param(
        kernel_init(rngs.params(), (hidden_size, 1), weight_dtype, in_axis=0, out_axis=1),
        out_sharding=kernel_axes,
    )

  @property
  def scale(self) -> nnx.Param:
    """Alias for norm_scale parameter matching RMSNorm attribute name."""
    return self.norm_scale

  @property
  def kernel(self) -> nnx.Param:
    """Alias for proj_kernel parameter matching DenseGeneral attribute name."""
    return self.proj_kernel

  def pool(
      self,
      block_residuals: Optional[jnp.ndarray],
      prefix_sum: jnp.ndarray,
      num_blocks: Optional[Union[int, jnp.ndarray]] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Dynamic Softmax Pooling over checkpoint blocks and prefix sum."""
    norm_w = self.norm_scale.get_value() if isinstance(self.norm_scale, nnx.Param) else self.norm_scale
    proj_w = self.proj_kernel.get_value() if isinstance(self.proj_kernel, nnx.Param) else self.proj_kernel

    return _apply_attn_res(
        prefix_sum=prefix_sum,
        block_residual=block_residuals,
        proj_weight=proj_w,
        norm_weight=norm_w,
        epsilon=self.epsilon,
        num_blocks=num_blocks,
        mask=mask,
    )

  def apply_to_state(self, state: AttnResState) -> jnp.ndarray:
    """Applies dynamic pooling directly to an AttnResState."""
    if state.prefix_sum is None:
      raise ValueError("AttnResState.prefix_sum cannot be None when applying AttnRes pooling.")
    if state.num_blocks == 0:
      return state.prefix_sum
    active_blocks = state.get_active_blocks()
    return self.pool(active_blocks, state.prefix_sum)

  def __call__(
      self,
      prefix_sum: jnp.ndarray,
      block_residuals: Optional[jnp.ndarray] = None,
      num_blocks: Optional[Union[int, jnp.ndarray]] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Computes pooled representation for given prefix sum and block residuals."""
    return self.pool(
        block_residuals=block_residuals,
        prefix_sum=prefix_sum,
        num_blocks=num_blocks,
        mask=mask,
    )


KimiAttnResLayerLinen = nnx_wrappers.to_linen_class(
    KimiAttnResLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)
