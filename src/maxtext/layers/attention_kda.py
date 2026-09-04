# Copyright 2026 Ant Group. All Rights Reserved.
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

"""Kimi Delta Attention (KDA) Layer Implementation.

KDA is a linear attention mechanism with Delta Rule correction, featuring:
  - Depthwise causal 1D convolution for local dependency modeling
  - Optional numerically safe gate (sigmoid lower-bound) mechanism
  - Optional Q/K L2 normalization

The layer computes Q/K/V projections, short convolutions, gate/beta
projections, and delegates the chunk-parallel Delta Rule recurrence to
``tokamax._src.ops.experimental.kda.api.kimi_delta_attention`` via
``maxtext.kernels.kda.chunk_kda``.
"""


import functools
import math
import warnings

from flax import nnx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from maxtext.kernels.kda import chunk_kda

# KDA depends on tokamax at runtime, but import should succeed because
# tokamax is a mandatory dependency for KDA models.
try:
  from tokamax._src.ops.experimental.kda.cp_utils import (
      ContextParallelMetadata as TokamaxContextParallelMetadata,
  )
except ImportError:
  TokamaxContextParallelMetadata = None

from maxtext.common.common_types import Config, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import linears
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils.sharding import logical_to_mesh_axes


# Sequence is padded to a multiple of this size before the KDA kernel, so
# TPU-friendly fixed shapes are used (matching the Megatron chunk convention).
_KDA_CHUNK_SIZE = 64


def _l2_normalize(x, axis=-1, eps=1e-6):
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=axis, keepdims=True) + eps)
  return (x_f * rstd).astype(x.dtype)


def _has_named_axis(axis_name: str) -> bool:
  """Check whether *axis_name* is bound in the current shard_map / mesh scope."""
  try:
    jax.lax.axis_index(axis_name)
    return True
  except NameError:
    return False


def halo_exchange_for_conv(
    x: jax.Array,
    halo_size: int,
    axis_name: str = "context",
    seq_axis: int = 1,
) -> jax.Array:
  """Prepend ``halo_size`` tokens from the previous CP rank for causal conv.

  KDA's ``ShortConvolution`` is the only user today; the helper lives in this
  module accordingly. The caller receives ``[halo_size + T_local, …]`` so the
  per-tap loop naturally reads the correct context window. Halos are fetched
  via a forward-ring ``ppermute``: rank *i* sends its last ``halo_size``
  tokens to rank *i+1*; rank 0 receives zeros (sequence start).

  When no CP axis is in scope or ``cp_size == 1`` the function degrades to
  left zero-padding, which is the correct causal-convolution boundary for a
  single-device / no-CP run.

  Constraint: the exchange only reads from the immediately preceding rank,
  so ``halo_size`` must not exceed the local sequence length. A larger
  receptive field (kernel_size - 1 > T_local) would need tokens from
  multiple previous ranks, which is not implemented; a ``ValueError`` is
  raised instead of silently reading the wrong context.

  Args:
    x: Tensor shaped ``[B, T, …]`` (seq_axis = 1).
    halo_size: Number of tokens to pull from the previous rank.
    axis_name: Mesh axis along which the sequence is sharded.
    seq_axis: The sequence dimension index (default 1).

  Returns:
    ``x`` with ``halo_size`` context tokens prepended along *seq_axis*.
  """
  if halo_size <= 0:
    return x

  # Left zero-pad — works correctly for both no-CP and CP.
  pad_width = [(0, 0)] * x.ndim
  pad_width[seq_axis] = (halo_size, 0)
  zero_padded = jnp.pad(x, pad_width)

  if not _has_named_axis(axis_name):
    return zero_padded

  cp_size = jax.lax.psum(1, axis_name=axis_name)
  if cp_size == 1:
    return zero_padded

  t_local = x.shape[seq_axis]
  if halo_size > t_local:
    raise ValueError(
        f"halo_exchange_for_conv: halo_size ({halo_size}) exceeds the local "
        f"sequence length ({t_local}) on the '{axis_name}' axis. The causal "
        "convolution receptive field would span multiple CP ranks, which is "
        "not implemented. Use a smaller linear_conv_kernel_dim, a longer "
        "sequence, or a smaller CP size."
    )

  # Forward ring: each rank sends its tail to the next rank.
  tail = jax.lax.dynamic_slice_in_dim(x, x.shape[seq_axis] - halo_size, halo_size, axis=seq_axis)
  perm = [(i, (i + 1) % cp_size) for i in range(cp_size)]
  halo = jax.lax.ppermute(tail, axis_name=axis_name, perm=perm)

  cp_rank = jax.lax.axis_index(axis_name)
  halo = jnp.where(cp_rank == 0, jnp.zeros_like(halo), halo)

  return jnp.concatenate([halo, x], axis=seq_axis)


class ShortConvolution(nnx.Module):
  """Depthwise causal 1D convolution for local dependency modeling in KDA.

  Each channel is convolved independently (no cross-channel mixing),
  matching Megatron's Conv1d with groups=in_channels. Position i can
  only attend to positions <= i (causal). When segment_ids is provided,
  cross-segment contributions are masked to prevent leakage across
  document boundaries (matches Megatron causal_conv1d_fn seq_idx).
  """

  def __init__(
      self,
      kernel_size: int,
      features: int,
      *,
      dtype: jnp.dtype = jnp.bfloat16,
      weight_dtype: jnp.dtype = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.kernel_size = kernel_size
    self.features = features
    self.dtype = dtype

    self.kernel = nnx.Param(
        nnx.initializers.lecun_normal()(
            rngs.params(),
            (kernel_size, features),
            weight_dtype,
        )
    )

  def __call__(
      self,
      x: jnp.ndarray,
      segment_ids: jnp.ndarray | None = None,
      cp_axis_name: str = "context",
  ) -> jnp.ndarray:
    B, T, F = x.shape
    if F != self.features:
      raise ValueError(f"Input features {F} != {self.features}")

    x_padded = halo_exchange_for_conv(x, self.kernel_size - 1, axis_name=cp_axis_name)

    if segment_ids is not None:
      seg_padded = halo_exchange_for_conv(segment_ids, self.kernel_size - 1, axis_name=cp_axis_name)
      # Stack per-tap masks once so the loop body has no tap-dependent
      # broadcasts beyond the slice itself.
      masks = [
          (seg_padded[:, k : k + T] == segment_ids).astype(x.dtype)[:, :, None]
          for k in range(self.kernel_size - 1, -1, -1)
      ]

    output = jnp.zeros((B, T, F), dtype=x.dtype)
    for k in range(self.kernel_size):
      offset = self.kernel_size - 1 - k
      x_slice = x_padded[:, offset : offset + T, :]
      if segment_ids is not None:
        x_slice = x_slice * masks[k]
      output = output + x_slice * self.kernel[k]

    return output.astype(self.dtype)


class KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) layer implementation.

    KDA is a linear attention mechanism that uses the Delta Rule for state
  correction:
      S' = S * exp(g_t)
      residual = v_t - k_t^T @ S'
      S = S' + beta_t * k_t (x) residual
      o_t = scale * q_t^T @ S

    Attributes:
      config: Model configuration containing KDA parameters.
      layer_idx: Index of this layer in the decoder stack.
      mesh: JAX device mesh for sharding.
  """

  def __init__(
      self,
      config: Config,
      layer_idx: int,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.mesh = mesh

    cfg = self.config

    # KDA head dimensions derived from global config (matching Megatron convention):
    #   key_head_dim = value_head_dim = config.head_dim (kv_channels)
    #   num_key_heads = num_value_heads = config.base_num_query_heads (num_attention_heads)
    self.key_head_dim = cfg.head_dim
    self.value_head_dim = cfg.head_dim
    self.num_key_heads = cfg.base_num_query_heads
    self.num_value_heads = cfg.base_num_query_heads
    self.num_query_heads = self.num_key_heads

    # Short convolution for local dependency modeling
    if cfg.linear_conv_kernel_dim > 0:
      self.q_conv = ShortConvolution(
          kernel_size=cfg.linear_conv_kernel_dim,
          features=self.num_query_heads * self.key_head_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=rngs,
      )
      self.k_conv = ShortConvolution(
          kernel_size=cfg.linear_conv_kernel_dim,
          features=self.num_key_heads * self.key_head_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=rngs,
      )
      self.v_conv = ShortConvolution(
          kernel_size=cfg.linear_conv_kernel_dim,
          features=self.num_value_heads * self.value_head_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=rngs,
      )
    else:
      self.q_conv = None
      self.k_conv = None
      self.v_conv = None

    # QKV projections
    # Separate projections for Q, K, V (not fused) to allow independent conv
    self.q_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_query_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.k_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.v_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_value_heads, self.value_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Output projection
    self.o_proj = linears.DenseGeneral(
        in_features_shape=(self.num_value_heads, self.value_head_dim),
        out_features_shape=cfg.base_emb_dim,
        axis=(-2, -1),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("heads", "kv", "embed"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Gate projection for generating g (log-space gate)
    # g has shape [B, T, H, K] - per-head, per-dim gate
    self.g_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Beta projection for generating beta (Delta rule mixing coefficient)
    # beta has shape [B, T, H] - per-head scalar
    self.b_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads,),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads"),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Q/K L2 normalization is applied outside chunk_kda (matching Megatron)

    # Output gate projection: gate shape [B, T, H, V] (matching Megatron no_kda_lora path)
    self.gate_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_value_heads, self.value_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Output norm (per-head RMSNorm, applied before gating)
    self.out_norm = RMSNorm(
        num_features=self.value_head_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Gate parameters (matching Megatron kda.py:299-330). Params keep the
    # Megatron names (A_log / dt_bias) but are passed to tokamax as `a_log`
    # and `delta_time_bias` inside the shard_map below.
    # A_log: [num_key_heads] — log of diagonal decay matrix
    A_init_range = (1.0, 16.0)
    A = jax.random.uniform(
        rngs.params(),
        shape=(self.num_key_heads,),
        minval=A_init_range[0],
        maxval=A_init_range[1],
    )
    self.A_log = nnx.Param(jnp.log(A))

    # dt_bias: [num_key_heads * key_head_dim] — gate bias
    # Initialize via inverse softplus of uniform(dt_min, dt_max)
    dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
    dt = jnp.exp(
        jax.random.uniform(
            rngs.params(),
            shape=(self.num_key_heads * self.key_head_dim,),
        )
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = jnp.clip(dt, min=dt_init_floor)
    # Inverse softplus: x = dt + log(-expm1(-dt))
    inv_dt = dt + jnp.log(-jnp.expm1(-dt))
    self.dt_bias = nnx.Param(inv_dt)

    # Axis names for shard_map (tokamax kernels cannot be auto-partitioned).
    self.qkv_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_heads",
        "activation_kv",
    )
    self.beta_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_heads",
    )

  def _logical_to_mesh_axes(self, logical_name):
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=self.config.logical_axis_rules)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      decoder_positions: jnp.ndarray | None = None,
      deterministic: bool = True,
      model_mode: str = "train",
      *,
      layer_idx: int | None = None,
      decoder_segment_ids: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, None]:
    """Forward pass for KDA attention.

    Args:
      hidden_states: Input tensor of shape [B, T, emb_dim].
      decoder_positions: Position indices for RoPE (not used in KDA).
      deterministic: Whether to use deterministic mode.
      model_mode: Model mode (train/prefill/autoregressive).
      layer_idx: Optional layer index override.
      decoder_segment_ids: Optional segment IDs for packed sequences.

    Returns:
      Tuple of (output, None) where output has shape [B, T, emb_dim].
    """
    del decoder_positions  # KDA doesn't use RoPE
    del deterministic  # No dropout in KDA currently
    del layer_idx  # Not used

    cfg = self.config

    # Context-parallel size derived from the mesh (the CP axis name is
    # cfg.context_sharding, default "context"; it may also be "expert" for
    # expert-as-context). This mirrors attention_op.py.
    cp_axis_name = cfg.context_sharding
    cp_size = self.mesh.shape.get(cp_axis_name, 1)

    # KDA Delta Rule relies on sequential recurrent state S_t = f(S_{t-1}, ...).
    # load_balance's DUAL_CHUNK_SWAP reorder breaks token order, invalidating
    # the sequential dependency. Reject this combination up front.
    if cp_size > 1 and getattr(cfg, "context_parallel_load_balance", False):
      raise ValueError(
          "KDA CP does not support context_parallel_load_balance. "
          "Recurrent state S depends on exact token order; DUAL_CHUNK_SWAP "
          "reorder breaks the sequential dependency. Set "
          "context_parallel_load_balance=false when using KDA with CP."
      )

    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise NotImplementedError("KDA autoregressive mode not yet implemented.")

    # Packed/varlen execution: tokamax requires a static positive
    # max_num_segments whenever segment_ids are supplied without
    # initial_state (see tokamax .../kda/api.py). maxtext surfaces this as
    # `max_segments_per_seq`, which defaults to -1 (unset) — fail fast with
    # a config-level message instead of erroring deep in kernel binding.
    if decoder_segment_ids is not None and cfg.max_segments_per_seq <= 0:
      raise ValueError(
          "KDA with packed sequences (decoder_segment_ids) requires "
          f"`max_segments_per_seq` to be a positive integer, got "
          f"{cfg.max_segments_per_seq}. Set `max_segments_per_seq` in your "
          "config to a static upper bound on the number of packed segments "
          "per sequence."
      )

    def _inject_cp_axis_on_T(pspec, t_axis=1):
      """Overwrite the T axis of *pspec* with the CP axis name.

      logical_to_mesh_axes may map the LENGTH logical axis to a different
      mesh axis, or to None, because the activation_norm_length rules do not
      cover every CP strategy (notably expert-as-context). Overwrite
      unconditionally so shard_map always sees the correct per-rank sequence
      shards on the axis the collectives (halo exchange, CP state merge) use.
      """
      spec = list(pspec)
      spec[t_axis] = cp_axis_name
      return jax.sharding.PartitionSpec(*spec)

    B, T_orig, _ = hidden_states.shape
    T = T_orig

    if T % _KDA_CHUNK_SIZE != 0:
      pad_len = _KDA_CHUNK_SIZE - (T % _KDA_CHUNK_SIZE)
      hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
      if decoder_segment_ids is not None:
        decoder_segment_ids = jnp.pad(decoder_segment_ids, ((0, 0), (0, pad_len)), constant_values=0)
      T = hidden_states.shape[1]

    # QKV projections
    with jax.named_scope("qkv_proj"):
      q = self.q_proj(hidden_states)  # [B, T, H, K]
      k = self.k_proj(hidden_states)  # [B, T, H, K]
      v = self.v_proj(hidden_states)  # [B, T, H, V]

      # Names must match decoders.minimal_policy so remat policies save these.
      q = checkpoint_name(q, "query_proj")
      k = checkpoint_name(k, "key_proj")
      v = checkpoint_name(v, "value_proj")

    # Apply short convolution if enabled (before activation, matching Megatron)
    if self.q_conv is not None:
      with jax.named_scope("short_conv"):
        # Reshape for conv: [B, T, H*D] -> conv -> [B, T, H*D] -> reshape back
        q_flat = q.reshape(B, T, -1)
        k_flat = k.reshape(B, T, -1)
        v_flat = v.reshape(B, T, -1)

        # Under CP, ShortConvolution needs to pull kernel_size-1 tokens
        # of left context from the previous CP rank via ppermute — which
        # is a collective and so must run inside a shard_map that exposes
        # the CP mesh axis (cfg.context_sharding). Without this wrap,
        # halo_exchange_for_conv would silently degrade to zero-pad
        # (causal-conv at every CP shard boundary would be wrong). This
        # applies to all CP strategies.
        if cp_size > 1:
          conv_flat_pspec = _inject_cp_axis_on_T(
              self._logical_to_mesh_axes(("activation_batch", "activation_norm_length", None))
          )
          conv_seg_pspec = (
              _inject_cp_axis_on_T(self._logical_to_mesh_axes(("activation_batch", "activation_norm_length")))
              if decoder_segment_ids is not None
              else None
          )
          q_conv_mod, k_conv_mod, v_conv_mod = (
              self.q_conv,
              self.k_conv,
              self.v_conv,
          )

          @functools.partial(
              jax.shard_map,
              mesh=self.mesh,
              in_specs=(
                  conv_flat_pspec,
                  conv_flat_pspec,
                  conv_flat_pspec,
                  conv_seg_pspec,
              ),
              out_specs=(conv_flat_pspec, conv_flat_pspec, conv_flat_pspec),
              check_vma=False,
          )
          def _conv_with_halo(qf, kf, vf, seg):
            qf = q_conv_mod(qf, segment_ids=seg, cp_axis_name=cp_axis_name)
            kf = k_conv_mod(kf, segment_ids=seg, cp_axis_name=cp_axis_name)
            vf = v_conv_mod(vf, segment_ids=seg, cp_axis_name=cp_axis_name)
            return qf, kf, vf

          q_flat, k_flat, v_flat = _conv_with_halo(q_flat, k_flat, v_flat, decoder_segment_ids)
        else:
          q_flat = self.q_conv(q_flat, segment_ids=decoder_segment_ids, cp_axis_name=cp_axis_name)
          k_flat = self.k_conv(k_flat, segment_ids=decoder_segment_ids, cp_axis_name=cp_axis_name)
          v_flat = self.v_conv(v_flat, segment_ids=decoder_segment_ids, cp_axis_name=cp_axis_name)

        q = q_flat.reshape(B, T, self.num_query_heads, self.key_head_dim)
        k = k_flat.reshape(B, T, self.num_key_heads, self.key_head_dim)
        v = v_flat.reshape(B, T, self.num_value_heads, self.value_head_dim)

    # Apply SiLU activation after conv on q, k, v (matching Megatron)
    q = jax.nn.silu(q)
    k = jax.nn.silu(k)
    v = jax.nn.silu(v)

    # Apply L2 normalization to Q/K outside the kernel (matching Megatron
    # kda.py:824-828). Always on for KDA: the Delta-Rule recurrence is
    # numerically unstable with unbounded q/k (training diverges to NaN in
    # bf16), and QK L2-norm is part of the KDA architecture in the reference.
    # This is independent of the shared `use_qk_norm` flag, which belongs to
    # dot-product attention.
    q = _l2_normalize(q)
    k = _l2_normalize(k)

    # Generate gate g (raw projection, gate transform done inside kernel)
    with jax.named_scope("gate_proj"):
      g = self.g_proj(hidden_states)  # [B, T, H, K]

    # Generate output gate (for gated norm after KDA kernel)
    with jax.named_scope("output_gate_proj"):
      output_gate = self.gate_proj(hidden_states)  # [B, T, H, V]

    # Generate beta (Delta rule mixing coefficient)
    with jax.named_scope("beta_proj"):
      beta = self.b_proj(hidden_states)  # [B, T, H]
      beta = beta.astype(jnp.float32)
      beta = jax.nn.sigmoid(beta)  # Ensure (0, 1) range, in fp32

    scale = self.key_head_dim**-0.5
    safe_gate = cfg.use_kda_safe_gate
    lower_bound = cfg.kda_lower_bound if safe_gate else None
    if not safe_gate and cfg.kda_lower_bound != 0.0:
      warnings.warn(
          f"kda_lower_bound={cfg.kda_lower_bound} is ignored because use_kda_safe_gate=False. "
          "Set use_kda_safe_gate=True to enable lower_bound clamping.",
          stacklevel=2,
      )
    n_max = cfg.max_segments_per_seq if cfg.max_segments_per_seq > 0 else None
    if cp_size > 1 and decoder_segment_ids is None:
      # Under CP the tokamax kernel derives per-rank cu_seqlens / chain
      # metadata from segment_ids, so a seg tensor must always be present.
      # Without user segmentation, synthesize a single all-ones segment —
      # done outside shard_map so a real array is sharded through.
      decoder_segment_ids = jnp.ones((B, T), dtype=jnp.int32)
      n_max = 1

    # Call KDA kernel via shard_map (tokamax kernels cannot be auto-partitioned).
    with jax.named_scope("kda_kernel"):
      qkv_pspec = self._logical_to_mesh_axes(self.qkv_axis_names)
      beta_pspec = self._logical_to_mesh_axes(self.beta_axis_names)
      a_log_pspec = self._logical_to_mesh_axes(("activation_heads",))
      delta_time_bias_2d_pspec = self._logical_to_mesh_axes(("activation_heads", "activation_kv"))
      seg_pspec = self._logical_to_mesh_axes(("activation_batch", "activation_norm_length"))

      # Reshape the dt_bias param from [H*K] to [H, K] for head-dim sharding.
      # (Tokamax's argument name is delta_time_bias; the nnx param is named
      # dt_bias to match the Megatron reference.)
      delta_time_bias_2d = self.dt_bias.value.reshape(self.num_key_heads, self.key_head_dim)

      # Force the CP axis onto the T axis of every pspec so the shard_map
      # sees the correct per-rank shard layout and the collectives run on
      # the axis the sequence is actually sharded over.
      if cp_size > 1:
        qkv_pspec = _inject_cp_axis_on_T(qkv_pspec)
        beta_pspec = _inject_cp_axis_on_T(beta_pspec)
        seg_pspec = _inject_cp_axis_on_T(seg_pspec)

        def _wsc(x, pspec):
          return jax.lax.with_sharding_constraint(x, jax.sharding.NamedSharding(self.mesh, pspec))

        q, k, v, g = (
            _wsc(q, qkv_pspec),
            _wsc(k, qkv_pspec),
            _wsc(v, qkv_pspec),
            _wsc(g, qkv_pspec),
        )
        beta = _wsc(beta, beta_pspec)
        if decoder_segment_ids is not None:
          decoder_segment_ids = _wsc(decoder_segment_ids, seg_pspec)

      # Under CP a seg tensor is always present (synthesized above when
      # the user supplies none), so the kernel can derive cu_seqlens / chain
      # fields via one small all_gather.
      has_seg = decoder_segment_ids is not None
      base_in_specs = (
          qkv_pspec,
          qkv_pspec,
          qkv_pspec,
          qkv_pspec,
          beta_pspec,
          a_log_pspec,
          delta_time_bias_2d_pspec,
      )
      in_specs = base_in_specs + ((seg_pspec,) if has_seg else ())

      # ContextParallelMetadata lives outside shard_map — it is a frozen
      # dataclass that holds the mesh identity. tokamax's chunk_kda derives
      # the per-rank chain fields (cu_seqlens, is_first_rank, …) internally
      # from segment_ids, then passes the completed metadata to the kernel.
      cp_ctx = None
      if cp_size > 1:
        if TokamaxContextParallelMetadata is None:
          raise ImportError(
              "KDA context parallelism requires "
              "tokamax._src.ops.experimental.kda.cp_utils.ContextParallelMetadata, "
              "but it failed to import. Refusing to run: CP would silently "
              "break recurrent state across ranks."
          )
        cp_ctx = TokamaxContextParallelMetadata(mesh=self.mesh, axis_name=cp_axis_name)

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=in_specs,
          out_specs=qkv_pspec,
          check_vma=False,
      )
      def _shard_map_chunk_kda(*args):
        q, k, v, g, beta, a_log, delta_time_bias_2d, *rest = args
        seg = rest[0] if rest else None

        delta_time_bias_flat = delta_time_bias_2d.reshape(-1)
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            a_log=a_log,
            delta_time_bias=delta_time_bias_flat,
            segment_ids=seg,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm=False,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            max_num_segments=n_max,
            context_parallel_metadata=cp_ctx,
        )
        return o

      kda_args = (q, k, v, g, beta, self.A_log.value, delta_time_bias_2d)
      if has_seg:
        kda_args = kda_args + (decoder_segment_ids,)
      o = _shard_map_chunk_kda(*kda_args)

    # Analogous to MLA's `context` (see attention_op.py); a remat boundary
    # right after the KDA kernel so its result survives `minimal_with_context`.
    o = checkpoint_name(o, "context")

    # Output gated norm (matching Megatron _apply_gated_norm):
    # per-head RMSNorm over the value dim, then sigmoid gate.
    with jax.named_scope("output_gated_norm"):
      o_dtype = o.dtype
      o_normed = self.out_norm(o)
      o = (o_normed * jax.nn.sigmoid(output_gate.astype(jnp.float32))).astype(o_dtype)

    # Output projection
    with jax.named_scope("o_proj"):
      output = self.o_proj(o)
      output = checkpoint_name(output, "out_proj")

    return output[:, :T_orig, :], None
