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

"""Nemotron-H model components."""

import functools
from typing import Any, Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh
from flax import linen as nn
from flax import nnx
from maxtext.common.common_types import Array, Config, DType, ShardMode, MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers import normalizations
from maxtext.layers import attentions
from maxtext.layers import quantizations
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.inference import kvcache


# Helper functions from Bonsai Mamba-2

def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
  """Pad zeros at the end of the sequence dimension (axis=1)."""
  if pad_size == 0:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[1] = (0, pad_size)
  return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
  """Stable segment sum calculation. Input: (..., T) -> Output: (..., T, T)."""
  T = x.shape[-1]
  x_cumsum = jnp.cumsum(x, axis=-1)
  x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
  x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
  return x_segsum


def ssd_forward(
    x: jnp.ndarray,  # (B, L, H, P)
    dt: jnp.ndarray,  # (B, L, H)
    A: jnp.ndarray,  # (H,)
    B_mat: jnp.ndarray,  # (B, L, H, N)
    C_mat: jnp.ndarray,  # (B, L, H, N)
    chunk_size: int,
    D: jnp.ndarray,  # (H,)
    dt_bias: jnp.ndarray,  # (H,)
    dt_min: float,
    dt_max: float,
    initial_states: jnp.ndarray | None = None,
    return_final_states: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """SSD (State Space Duality) forward pass with chunked computation."""
  _B_size, seq_len, num_heads, _head_dim = x.shape
  pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

  # Apply dt bias with softplus and clamp
  dt = jax.nn.softplus(dt + dt_bias)
  dt = jnp.clip(dt, dt_min, dt_max)

  # Pad tensors along sequence dimension
  x_padded = _pad_seq_dim(x, pad_size)
  dt_padded = _pad_seq_dim(dt, pad_size)
  B_padded = _pad_seq_dim(B_mat, pad_size)
  C_padded = _pad_seq_dim(C_mat, pad_size)

  # D residual connection
  D_residual = D.reshape(1, 1, num_heads, 1) * x_padded

  # Discretize x and A
  x_disc = x_padded * dt_padded[..., None]
  A_disc = A.astype(x_disc.dtype) * dt_padded

  # Chunk everything
  def chunk_tensor(t):
    b, cl, *remaining = t.shape
    return t.reshape(b, cl // chunk_size, chunk_size, *remaining)

  x_blk = chunk_tensor(x_disc)
  A_blk = chunk_tensor(A_disc)
  B_blk = chunk_tensor(B_padded)
  C_blk = chunk_tensor(C_padded)

  # A cumsum over intra-chunk time dimension
  A_blk2 = jnp.transpose(A_blk, (0, 3, 1, 2))
  A_cumsum = jnp.cumsum(A_blk2, axis=-1)

  # 1. Intra-chunk (diagonal blocks)
  L_mat = jnp.exp(segsum(A_blk2))
  Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blk, B_blk, L_mat, x_blk)

  # 2. States within each chunk
  decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B_blk, decay_states, x_blk)

  # 3. Inter-chunk recurrence
  if initial_states is None:
    initial_states = jnp.zeros_like(states[:, :1, ...])
  elif initial_states.ndim == 4:
    initial_states = jnp.expand_dims(initial_states, axis=1)
  states = jnp.concatenate([initial_states, states], axis=1)

  A_end = A_cumsum[..., -1]
  A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))
  decay_chunk = jnp.exp(segsum(A_end_padded))
  new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
  states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

  # 4. Convert states -> outputs
  state_decay_out = jnp.exp(A_cumsum)
  Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C_blk, states, state_decay_out)

  y = Y_diag + Y_off
  b, c, l, h, p = y.shape
  y = y.reshape(b, c * l, h, p)
  y = y + D_residual

  # Remove padding
  if pad_size > 0:
    y = y[:, :seq_len, :, :]

  return (y, final_state) if return_final_states else (y, None)


class DepthwiseConv1d(nnx.Module):
  """Depthwise causal 1D convolution with state caching."""

  def __init__(self, features: int, kernel_size: int, use_bias: bool = True, *, rngs: nnx.Rngs):
    self.features = features
    self.kernel_size = kernel_size
    self.conv = nnx.Conv(
        in_features=features,
        out_features=features,
        kernel_size=(kernel_size,),
        padding=((0, 0),),
        feature_group_count=features,
        use_bias=use_bias,
        rngs=rngs,
    )

  def __call__(
      self,
      x: jnp.ndarray,
      conv_state: jnp.ndarray | None = None,
      decoder_segment_ids: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    cache_len = self.kernel_size - 1

    if conv_state is None:
      x_padded = jnp.pad(x, ((0, 0), (cache_len, 0), (0, 0)), mode="constant", constant_values=0.0)
    else:
      x_padded = jnp.concatenate([jnp.transpose(conv_state, (0, 2, 1)), x], axis=1)

    output = self.conv(x_padded)

    if decoder_segment_ids is not None:
      valid_lens = jnp.sum(decoder_segment_ids != 0, axis=1)
      def extract_state(c_padded, v_len):
        return jax.lax.dynamic_slice_in_dim(c_padded, v_len, cache_len, axis=0)
      new_conv_state_transposed = jax.vmap(extract_state)(x_padded, valid_lens)
      new_conv_state = jnp.transpose(new_conv_state_transposed, (0, 2, 1))
    else:
      new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))

    return output, new_conv_state


class Zamba2RMSNormGated(nnx.Module):
  """RMSNorm with gating applied before normalization."""

  def __init__(
      self,
      num_features: int,
      group_size: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[str | None, ...] = (),
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.group_size = group_size
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes

    self.scale = nnx.Param(
        nnx.initializers.ones(rngs.params(), (num_features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, hidden_states: Array, gate: Array | None = None, out_sharding=None) -> Array:
    hidden_states = jnp.asarray(hidden_states, jnp.float32)

    if gate is not None:
      hidden_states = hidden_states * jax.nn.silu(jnp.asarray(gate, jnp.float32))

    prefix_dims = hidden_states.shape[:-1]
    last_dim = hidden_states.shape[-1]
    group_count = last_dim // self.group_size

    # Reshape to group
    hidden_states_group = hidden_states.reshape((*prefix_dims, group_count, self.group_size))

    # Compute variance over group_size dim
    variance = jnp.mean(jnp.square(hidden_states_group), axis=-1, keepdims=True)
    hidden_states_group = hidden_states_group * jax.lax.rsqrt(variance + self.epsilon)

    # Reshape back
    hidden_states = hidden_states_group.reshape((*prefix_dims, last_dim))

    # Apply scale
    scale = jnp.asarray(self.scale.get_value(), self.dtype)
    hidden_states = hidden_states.astype(self.dtype)

    # Sharding constraint if explicit
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    return jnp.einsum("...k,k->...k", hidden_states, scale, out_sharding=out_sharding)


class NemotronHMamba2Mixer(nnx.Module):
  """Nemotron-H Mamba-2 Mixer layer."""

  def __init__(
      self,
      config: Config,
      dtype: DType = jnp.float32,
      model_mode: str = MODEL_MODE_TRAIN,
      layer_idx: int | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.dtype = dtype
    self.model_mode = model_mode
    self.layer_idx = layer_idx

    self.hidden_size = config.emb_dim
    self.ssm_state_size = config.ssm_state_size
    self.conv_kernel_size = config.conv_kernel
    self.n_groups = config.n_groups
    self.head_dim = config.mamba_head_dim
    self.num_heads = config.mamba_num_heads
    self.chunk_size = config.mamba_chunk_size

    self.intermediate_size = self.num_heads * self.head_dim
    self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

    projection_size = self.intermediate_size + self.conv_dim + self.num_heads

    self.in_proj = nnx.Linear(
        in_features=self.hidden_size,
        out_features=projection_size,
        use_bias=config.use_bias,
        dtype=dtype,
        rngs=rngs,
    )

    self.conv1d = DepthwiseConv1d(
        features=self.conv_dim,
        kernel_size=self.conv_kernel_size,
        use_bias=config.use_conv_bias,
        rngs=rngs,
    )

    self.dt_bias = nnx.Param(
        jnp.ones((self.num_heads,), dtype=dtype),
    )

    A_init = jnp.log(jnp.arange(1, self.num_heads + 1, dtype=jnp.float32))
    self.A_log = nnx.Param(A_init.astype(dtype))

    self.norm = Zamba2RMSNormGated(
        num_features=self.intermediate_size,
        group_size=self.intermediate_size // self.n_groups,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        rngs=rngs,
    )

    self.D = nnx.Param(
        jnp.ones((self.num_heads,), dtype=dtype),
    )

    self.out_proj = nnx.Linear(
        in_features=self.intermediate_size,
        out_features=self.hidden_size,
        use_bias=config.use_bias,
        dtype=dtype,
        rngs=rngs,
    )

    if self.model_mode != MODEL_MODE_TRAIN:
      self.cache = kvcache.NemotronHMamba2Cache(
          batch=config.micro_batch_size_to_train_on,
          num_heads=self.num_heads,
          head_dim=self.head_dim,
          ssm_state_size=self.ssm_state_size,
          conv_kernel_size=self.conv_kernel_size,
          conv_dim=self.conv_dim,
          dtype=dtype,
      )

  def __call__(self, hidden_states: Array, decoder_segment_ids: Array | None = None) -> Array:
    batch_size, seq_len, _ = hidden_states.shape

    projected = self.in_proj(hidden_states)

    gate, conv_input, dt = jnp.split(
        projected,
        [self.intermediate_size, self.intermediate_size + self.conv_dim],
        axis=-1,
    )

    if self.model_mode == MODEL_MODE_AUTOREGRESSIVE:
      conv_state = self.cache.conv_state[...]
      if conv_state.shape[0] != batch_size:
        if conv_state.shape[0] == 1:
          conv_state = jnp.broadcast_to(conv_state, (batch_size,) + conv_state.shape[1:])
        else:
          conv_state = conv_state[:batch_size]

      conv_out, new_conv_state = self.conv1d(conv_input, conv_state, decoder_segment_ids)
      self.cache.conv_state.set_value(new_conv_state)
    elif self.model_mode == MODEL_MODE_PREFILL:
      conv_out, new_conv_state = self.conv1d(conv_input, None, decoder_segment_ids)
      self.cache.conv_state.set_value(new_conv_state)
    else:
      conv_out, _ = self.conv1d(conv_input, decoder_segment_ids=decoder_segment_ids)

    conv_out = jax.nn.silu(conv_out)

    if decoder_segment_ids is not None:
      mask = decoder_segment_ids != 0
      conv_out = jnp.where(mask[..., None], conv_out, 0.0)

    ssm_input, B, C = jnp.split(
        conv_out,
        [self.intermediate_size, self.intermediate_size + self.n_groups * self.ssm_state_size],
        axis=-1,
    )

    B = B.reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))
    C = C.reshape((batch_size, seq_len, self.n_groups, self.ssm_state_size))

    rep_factor = self.num_heads // self.n_groups
    B = jnp.repeat(B, rep_factor, axis=2)
    C = jnp.repeat(C, rep_factor, axis=2)

    ssm_input = ssm_input.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

    A = -jnp.exp(self.A_log.get_value().astype(jnp.float32))

    if decoder_segment_ids is not None:
      mask = decoder_segment_ids != 0
      dt = jnp.where(mask[..., None], dt, 0.0)

    if self.model_mode != MODEL_MODE_TRAIN and seq_len == 1:
      dt_sq = dt[:, 0, :]
      dt_sq = jax.nn.softplus(dt_sq + self.dt_bias.get_value())
      dt_sq = jnp.clip(dt_sq, self.config.time_step_min, self.config.time_step_max)

      dt_expanded = jnp.expand_dims(dt_sq, axis=-1)
      dt_expanded = jnp.broadcast_to(dt_expanded, (batch_size, self.num_heads, self.head_dim))

      A_expanded = A[:, None, None]
      dA = jnp.exp(dt_expanded[..., None] * A_expanded)

      B_sq = B[:, 0, :, :]
      dB = dt_expanded[..., None] * B_sq[:, :, None, :]

      ssm_input_sq = ssm_input[:, 0, :, :]
      dBx = dB * ssm_input_sq[..., None]

      recurrent_state = self.cache.recurrent_state[...]
      if recurrent_state.shape[0] != batch_size:
        if recurrent_state.shape[0] == 1:
          recurrent_state = jnp.broadcast_to(recurrent_state, (batch_size,) + recurrent_state.shape[1:])
        else:
          recurrent_state = recurrent_state[:batch_size]

      new_recurrent_state = recurrent_state * dA + dBx
      self.cache.recurrent_state.set_value(new_recurrent_state)

      C_sq = C[:, 0, :, :]
      y = jnp.sum(new_recurrent_state * C_sq[:, :, None, :], axis=-1)

      D_expanded = self.D.get_value()[:, None]
      y = y + ssm_input_sq * D_expanded

      y = y.reshape((batch_size, 1, self.intermediate_size))
    else:
      recurrent_state = None
      if self.model_mode != MODEL_MODE_TRAIN:
        recurrent_state = self.cache.recurrent_state[...]
        if recurrent_state.shape[0] != batch_size:
          if recurrent_state.shape[0] == 1:
            recurrent_state = jnp.broadcast_to(recurrent_state, (batch_size,) + recurrent_state.shape[1:])
          else:
            recurrent_state = recurrent_state[:batch_size]

      ssd_out = ssd_forward(
          x=ssm_input,
          dt=dt,
          A=A,
          B_mat=B,
          C_mat=C,
          chunk_size=self.chunk_size,
          D=self.D.get_value(),
          dt_bias=self.dt_bias.get_value(),
          dt_min=self.config.time_step_min,
          dt_max=self.config.time_step_max,
          initial_states=recurrent_state,
          return_final_states=(self.model_mode != MODEL_MODE_TRAIN),
      )

      if self.model_mode != MODEL_MODE_TRAIN:
        y, final_state = ssd_out
        self.cache.recurrent_state.set_value(final_state)
      else:
        y, _ = ssd_out

      y = y.reshape((batch_size, seq_len, self.intermediate_size))

    y = self.norm(y, gate)
    out = self.out_proj(y)

    return out


NemotronHMamba2MixerToLinen = nnx_wrappers.to_linen_class(NemotronHMamba2Mixer)


class NemotronHMLP(nnx.Module):
  """Standard MLP for Nemotron-H (non-gated)."""

  def __init__(self, config: Config, intermediate_size: int | None = None, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.config = config
    self.hidden_size = config.emb_dim
    self.intermediate_size = intermediate_size or config.mlp_dim

    self.up_proj = nnx.Linear(
        in_features=self.hidden_size,
        out_features=self.intermediate_size,
        use_bias=config.mlp_bias,
        dtype=dtype,
        rngs=rngs,
    )
    self.down_proj = nnx.Linear(
        in_features=self.intermediate_size,
        out_features=self.hidden_size,
        use_bias=config.mlp_bias,
        dtype=dtype,
        rngs=rngs,
    )

  def __call__(self, x: Array) -> Array:
    h = self.up_proj(x)
    h = jnp.square(jax.nn.relu(h))
    return self.down_proj(h)


class NemotronHTopkRouter(nnx.Module):
  """Router for Nemotron-H MoE."""

  def __init__(self, config: Config, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.config = config
    self.n_routed_experts = config.num_experts
    self.hidden_size = config.emb_dim

    self.weight = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (self.n_routed_experts, self.hidden_size), dtype),
    )
    self.e_score_correction_bias = nnx.Param(
        jnp.zeros((self.n_routed_experts,), dtype=dtype),
    )

  def __call__(self, hidden_states: Array) -> Array:
    x = hidden_states.astype(jnp.float32)
    w = self.weight.get_value().astype(jnp.float32)
    router_logits = jnp.einsum("...h,eh->...e", x, w)
    return router_logits


class NemotronHExperts(nnx.Module):
  """Collection of expert weights stored as 3D tensors, vectorized execution."""

  def __init__(self, config: Config, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.num_experts = config.num_experts
    self.hidden_dim = config.emb_dim
    self.intermediate_dim = config.moe_mlp_dim

    input_dim = self.hidden_dim

    self.up_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (self.num_experts, self.intermediate_dim, input_dim), dtype),
    )
    self.down_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (self.num_experts, input_dim, self.intermediate_dim), dtype),
    )

  def __call__(self, hidden_states: Array, top_k_index: Array, top_k_weights: Array) -> Array:
    N, D = hidden_states.shape
    _, K = top_k_index.shape

    up_w = self.up_proj.get_value()[top_k_index]
    down_w = self.down_proj.get_value()[top_k_index]

    x = jnp.expand_dims(hidden_states, axis=1)
    x = jnp.broadcast_to(x, (N, K, D))

    h = jnp.einsum("nkd,nked->nke", x, up_w)
    h = jnp.square(jax.nn.relu(h))
    out = jnp.einsum("nke,nkde->nkd", h, down_w)

    out = out * top_k_weights[..., None]
    final_hidden_states = jnp.sum(out, axis=1)

    return final_hidden_states


class NemotronHMoE(nnx.Module):
  """Mixture-of-Experts (MoE) module for NemotronH."""

  def __init__(self, config: Config, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.config = config
    self.dtype = dtype

    self.experts = NemotronHExperts(config, dtype, rngs=rngs)
    self.gate = NemotronHTopkRouter(config, dtype, rngs=rngs)

    self.shared_experts = NemotronHMLP(
        config=config,
        intermediate_size=config.moe_shared_expert_intermediate_size,
        dtype=dtype,
        rngs=rngs,
    )

    self.n_routed_experts = config.num_experts
    self.n_group = config.n_routing_groups
    self.topk_group = config.topk_routing_group
    self.norm_topk_prob = config.norm_topk_prob
    self.routed_scaling_factor = config.routed_scaling_factor
    self.top_k = config.num_experts_per_tok

    self.use_latent_proj = hasattr(config, 'moe_latent_size') and config.moe_latent_size is not None

    if self.use_latent_proj:
      self.fc1_latent_proj = nnx.Linear(config.emb_dim, config.moe_latent_size, use_bias=config.mlp_bias, dtype=dtype, rngs=rngs)
      self.fc2_latent_proj = nnx.Linear(config.moe_latent_size, config.emb_dim, use_bias=config.mlp_bias, dtype=dtype, rngs=rngs)

  def route_tokens_to_experts(self, router_logits: Array) -> tuple[Array, Array]:
    router_probs = jax.nn.sigmoid(router_logits)
    bias = self.gate.e_score_correction_bias.get_value()
    router_probs_for_choice = router_probs + bias

    N, E = router_probs_for_choice.shape
    scores_grouped = router_probs_for_choice.reshape((N, self.n_group, E // self.n_group))
    
    top2_vals, _ = jax.lax.top_k(scores_grouped, k=2)
    group_scores = jnp.sum(top2_vals, axis=-1)
    
    _, group_idx = jax.lax.top_k(group_scores, k=self.topk_group)
    
    group_mask = jax.nn.one_hot(group_idx, num_classes=self.n_group)
    group_mask = jnp.sum(group_mask, axis=-2)
    
    score_mask = jnp.expand_dims(group_mask, axis=-1)
    score_mask = jnp.broadcast_to(score_mask, (N, self.n_group, E // self.n_group))
    score_mask = score_mask.reshape((N, E))
    
    scores_for_choice = jnp.where(score_mask > 0, router_probs_for_choice, -jnp.inf)
    
    _, topk_indices = jax.lax.top_k(scores_for_choice, k=self.top_k)
    
    topk_weights = jnp.take_along_axis(router_probs, topk_indices, axis=-1)
    
    if self.norm_topk_prob:
      denominator = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
      topk_weights /= denominator
      
    topk_weights = topk_weights * self.routed_scaling_factor
    
    return topk_indices, topk_weights

  def __call__(self, hidden_states: Array) -> Array:
    residuals = hidden_states
    orig_shape = hidden_states.shape
    
    hidden_states_2d = hidden_states.reshape((-1, orig_shape[-1]))
    
    router_logits = self.gate(hidden_states_2d)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    
    x = hidden_states_2d
    if self.use_latent_proj:
      x = self.fc1_latent_proj(x)
      
    x = self.experts(x, topk_indices, topk_weights)
    
    if self.use_latent_proj:
      x = self.fc2_latent_proj(x)
      
    x = x.reshape(orig_shape)
    
    shared_out = self.shared_experts(residuals)
    
    return x + shared_out


NemotronHMoEToLinen = nnx_wrappers.to_linen_class(NemotronHMoE)


class NemotronHDecoderLayer(nnx.Module):
  """Decoder layer for Nemotron-H."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    self.norm = normalizations.RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    self.block_type = config.layers_block_type[layer_idx]

    if self.block_type == "mamba":
      self.mixer = NemotronHMamba2Mixer(
          config=cfg,
          dtype=cfg.dtype,
          model_mode=model_mode,
          layer_idx=layer_idx,
          rngs=rngs,
      )
    elif self.block_type == "moe":
      self.mixer = NemotronHMoE(
          config=cfg,
          dtype=cfg.dtype,
          rngs=rngs,
      )
    elif self.block_type == "attention":
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
      
      self.mixer = attentions.Attention(
          config=cfg,
          num_query_heads=cfg.num_query_heads,
          num_kv_heads=cfg.num_kv_heads,
          head_dim=cfg.head_dim,
          max_target_length=cfg.max_target_length,
          max_prefill_predict_length=cfg.max_prefill_predict_length,
          attention_kernel=cfg.attention,
          inputs_q_shape=dummy_inputs_shape,
          inputs_kv_shape=dummy_inputs_shape,
          mesh=mesh,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          dropout_rate=cfg.dropout_rate,
          name="self_attention",
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(cfg),
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      raise ValueError(f"Unknown block type: {self.block_type}")

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ) -> tuple[jnp.ndarray, None | jnp.ndarray]:
    
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    residual = inputs

    hidden_states = self.norm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    new_kv_cache = None
    if self.block_type == "mamba":
      hidden_states = self.mixer(hidden_states, decoder_segment_ids)
    elif self.block_type == "moe":
      hidden_states = self.mixer(hidden_states)
    elif self.block_type == "attention":
      hidden_states, new_kv_cache = self.mixer(
          inputs_q=hidden_states,
          inputs_kv=hidden_states,
          inputs_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )

    layer_output = residual + hidden_states
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    return layer_output, new_kv_cache


NemotronHDecoderLayerToLinen = nnx_wrappers.to_linen_class(NemotronHDecoderLayer)
