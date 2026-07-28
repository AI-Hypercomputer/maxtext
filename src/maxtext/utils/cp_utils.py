"""Context Parallelism utilities — halo exchange for ShortConvolution."""

import jax
import jax.numpy as jnp


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

  The caller (ShortConvolution) receives ``[halo_size + T_local, …]`` so the
  per-tap loop naturally reads the correct context window.  Halos are fetched
  via a forward-ring ``ppermute``: rank *i* sends its last ``halo_size``
  tokens to rank *i+1*; rank 0 receives zeros (sequence start).

  When no CP axis is in scope or ``cp_size == 1`` the function degrades to
  left zero-padding, which is the correct causal-convolution boundary for a
  single-device / no-CP run.

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

  # Forward ring: each rank sends its tail to the next rank.
  tail = jax.lax.dynamic_slice_in_dim(x, x.shape[seq_axis] - halo_size, halo_size, axis=seq_axis)
  perm = [(i, (i + 1) % cp_size) for i in range(cp_size)]
  halo = jax.lax.ppermute(tail, axis_name=axis_name, perm=perm)

  cp_rank = jax.lax.axis_index(axis_name)
  halo = jnp.where(cp_rank == 0, jnp.zeros_like(halo), halo)

  return jnp.concatenate([halo, x], axis=seq_axis)
