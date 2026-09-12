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

"""Model-facing API and custom VJP boundary for GDN backward kernel."""

import functools
from typing import Optional, Tuple

import jax

try:
  from jax.ad_checkpoint import checkpoint_name
except ImportError:
  try:
    from jax._src.ad_checkpoint import checkpoint_name
  except ImportError:

    def checkpoint_name(x, name):
      del name
      return x


from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import wrapper as local_gdn_wrapper
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import wrapper as local_gdn_wrapper
  except (ImportError, ModuleNotFoundError):
    from .. import wrapper as local_gdn_wrapper

try:
  from maxtext.models.kernels.gdn.gdn_bwd.compute_conv1d_bwd import conv1d_silu_bwd, conv1d_silu_fwd
  from maxtext.models.kernels.gdn.gdn_bwd.compute_bwd_gdn import _compute_forward_conv_and_states, pure_jax_decoupled_conv1d_gdn
  from maxtext.models.kernels.gdn.gdn_bwd.pallas_mosaic_tpu_bwd import pallas_gdn_bwd_kernel
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.compute_conv1d_bwd import conv1d_silu_bwd, conv1d_silu_fwd
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.compute_bwd_gdn import _compute_forward_conv_and_states, pure_jax_decoupled_conv1d_gdn
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.pallas_mosaic_tpu_bwd import pallas_gdn_bwd_kernel
  except (ImportError, ModuleNotFoundError):
    from .compute_conv1d_bwd import conv1d_silu_bwd, conv1d_silu_fwd
    from .compute_bwd_gdn import _compute_forward_conv_and_states, pure_jax_decoupled_conv1d_gdn
    from .pallas_mosaic_tpu_bwd import pallas_gdn_bwd_kernel


def decoupled_conv1d_gdn_bwd_kernel(
    pre_conv_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array] = None,
    t_inv: Optional[jax.Array] = None,
    qkv: Optional[jax.Array] = None,
    seq_lens: Optional[jax.Array] = None,
    *,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int = 4,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
    head_tile: Optional[int] = None,
    segment_ids: Optional[jax.Array] = None,
    interpret: bool | pltpu.InterpretParams | None = None,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Optional[jax.Array],
    jax.Array,
    jax.Array,
]:
  """Decoupled Conv1D + GDN backward combining Pallas GDN bwd and JAX Conv1D bwd."""
  del seq_lens, qkv
  _, qkv_conv = conv1d_silu_fwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      kernel_size=kernel_size,
  )

  dy_conv, d_b, d_a, d_a_log, d_dt_bias = pallas_gdn_bwd_kernel(
      qkv_conv=qkv_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=do,
      chunk_states=chunk_states,
      t_inv=t_inv,
      num_v_heads=num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      vmem_limit_mb=vmem_limit_mb,
      head_tile=head_tile,
      segment_ids=segment_ids,
      interpret=interpret,
  )

  d_pre_conv_qkv, d_conv_weight, d_conv_bias = conv1d_silu_bwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      dy=dy_conv,
      kernel_size=kernel_size,
  )

  return (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight,
      d_conv_bias,
      d_a_log,
      d_dt_bias,
  )


decoupled_conv1d_gdn_bwd_computation = decoupled_conv1d_gdn_bwd_kernel
pallas_fused_conv1d_gdn_bwd_kernel = decoupled_conv1d_gdn_bwd_kernel
pallas_fused_conv1d_gdn_bwd_computation = decoupled_conv1d_gdn_bwd_kernel


def _run_local_gdn_decoupled_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
) -> Tuple[
    Tuple[jax.Array, Tuple[jax.Array, jax.Array]],
    Optional[jax.Array],
    Optional[jax.Array],
]:
  """Runs local GDN forward pass on TPU returning (t_inv, chunk_states), or pure JAX on CPU."""
  if jax.extend.backend.get_backend().platform == "cpu":
    out, states = pure_jax_decoupled_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
    )
    _, chunk_states, t_inv = _compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
    )
    return (out, states), t_inv, chunk_states

  batch_size, seq_len, dim_size = qkv.shape
  num_seqs = batch_size
  num_chunks = seq_len // chunk_size

  qkv_flat = qkv.reshape(-1, dim_size)
  b_flat = b.reshape(-1, b.shape[-1])
  a_flat = a.reshape(-1, a.shape[-1])
  tokamax_conv_weight = jnp.swapaxes(conv_weight, 0, 2)

  query_start_loc = jnp.arange(0, (num_seqs + 1) * seq_len, seq_len, dtype=jnp.int32)
  state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
  seq_lens = jnp.full((num_seqs,), seq_len, dtype=jnp.int32)
  distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

  if conv_state is None:
    tokamax_conv_state = jnp.zeros((num_seqs + 1, conv_kernel_size - 1, dim_size), dtype=qkv.dtype)
  elif conv_state.shape[0] == num_seqs:
    tokamax_conv_state = jnp.pad(conv_state, ((1, 0), (0, 0), (0, 0)))
  else:
    tokamax_conv_state = conv_state

  if recurrent_state is None:
    tokamax_recurrent_state = jnp.zeros((num_seqs + 1, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
  elif recurrent_state.shape[0] == num_seqs:
    tokamax_recurrent_state = jnp.pad(recurrent_state.astype(jnp.float32), ((1, 0), (0, 0), (0, 0), (0, 0)))
  else:
    tokamax_recurrent_state = recurrent_state.astype(jnp.float32)

  (
      core_attn_out_flat,
      (new_conv_state, new_recurrent_state),
      t_inv_raw,
      chunk_states_raw,
  ) = local_gdn_wrapper.fused_conv1d_gdn(
      qkv=qkv_flat,
      b=b_flat,
      a=a_flat,
      conv_state=tokamax_conv_state,
      recurrent_state=tokamax_recurrent_state,
      conv_weight=tokamax_conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      query_start_loc=query_start_loc,
      state_indices=state_indices,
      distribution=distribution,
      seq_lens=seq_lens,
      n_kq=num_k_heads,
      n_v=num_v_heads,
      d_k=head_k_dim,
      d_v=head_v_dim,
      kernel_size=conv_kernel_size,
      compute_precision=jnp.dtype(jnp.float32),
      mixed_tile_size=chunk_size,
      is_prefill_only=True,
  )

  core_attn_out = core_attn_out_flat.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
  t_inv = t_inv_raw.astype(jnp.float32).reshape(batch_size, num_chunks, num_v_heads, chunk_size, chunk_size)
  chunk_states = chunk_states_raw.astype(jnp.float32).reshape(batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim)
  return (
      (
          core_attn_out.astype(qkv.dtype),
          (
              new_conv_state[1:].astype(qkv.dtype),
              new_recurrent_state[1:].astype(jnp.float32),
          ),
      ),
      t_inv,
      chunk_states,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
def gdn_decoupled_conv1d(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Decoupled Conv1D + GDN with Pallas backward pass."""
  (out, states), _, _ = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
  )
  return out, states


def _gdn_decoupled_conv1d_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
):
  """Forward rule for custom_vjp registration of gdn_decoupled_conv1d."""
  qkv = checkpoint_name(qkv, "gdn_fwd_conv")
  (out, states), t_inv, chunk_states = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
  )
  out = checkpoint_name(out, "gdn_core_attn_out")
  residuals = (
      checkpoint_name(qkv, "gdn_qkv") if qkv is not None else None,
      checkpoint_name(b, "gdn_b") if b is not None else None,
      checkpoint_name(a, "gdn_a") if a is not None else None,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      checkpoint_name(t_inv, "gdn_t_inv") if t_inv is not None else None,
      checkpoint_name(chunk_states, "gdn_chunk_states") if chunk_states is not None else None,
  )
  return (out, states), residuals


def _gdn_decoupled_conv1d_bwd(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    residuals: tuple,
    cotangents: tuple,
):
  """Backward rule for custom_vjp registration of gdn_decoupled_conv1d."""
  if len(residuals) == 11:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
        chunk_states,
    ) = residuals
  else:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
    ) = residuals
    chunk_states = None

  d_out, d_states = cotangents
  d_conv_state, d_recurrent_state = d_states
  del d_conv_state, d_recurrent_state

  # Recompute forward chunk states and t_inv if not cached in residuals
  if chunk_states is None or t_inv_fwd is None:
    qkv_conv, chunk_states_recomputed, t_inv_recomputed = _compute_forward_conv_and_states(
        qkv=pre_conv_qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        cached_t_inv=t_inv_fwd,
    )
    if chunk_states is None:
      chunk_states = chunk_states_recomputed
    if t_inv_fwd is None:
      t_inv_fwd = t_inv_recomputed
  else:
    _, qkv_conv = conv1d_silu_fwd(
        qkv=pre_conv_qkv,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        kernel_size=conv_kernel_size,
    )
  t_inv = t_inv_fwd

  dy_conv, d_b, d_a, d_a_log, d_dt_bias = pallas_gdn_bwd_kernel(
      qkv_conv=qkv_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=d_out,
      chunk_states=chunk_states,
      t_inv=t_inv,
      num_v_heads=num_v_heads,
      kq_head_dim=head_k_dim,
      v_head_dim=head_v_dim,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
  )

  d_pre_conv_qkv, d_conv_weight, d_conv_bias = conv1d_silu_bwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      dy=dy_conv,
      kernel_size=conv_kernel_size,
  )

  d_conv_state_out = None if conv_state is None else jnp.zeros_like(conv_state)
  d_recurrent_state_out = None if recurrent_state is None else jnp.zeros_like(recurrent_state)
  return (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight,
      d_conv_bias,
      d_a_log,
      d_dt_bias,
      d_conv_state_out,
      d_recurrent_state_out,
  )


gdn_decoupled_conv1d.defvjp(
    _gdn_decoupled_conv1d_fwd,
    _gdn_decoupled_conv1d_bwd,
)

# Backward compatibility aliases
gdn_fused_conv1d = gdn_decoupled_conv1d
gdn_kernel = gdn_decoupled_conv1d
_gdn_fused_conv1d_fwd = _gdn_decoupled_conv1d_fwd
_gdn_fused_conv1d_bwd = _gdn_decoupled_conv1d_bwd
_run_local_gdn_fused_fwd = _run_local_gdn_decoupled_fwd
