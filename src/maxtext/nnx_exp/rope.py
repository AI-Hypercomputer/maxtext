"""Rotary positional embedding helpers."""

import jax.numpy as jnp


def rope_frequencies(head_dim, max_timescale=10_000.0):
  half_dim = head_dim // 2
  return 1.0 / (max_timescale ** (jnp.arange(half_dim, dtype=jnp.float32) / half_dim))


def rope_factors(positions, frequencies, ndim, dtype=jnp.float32):
  angles = positions[..., None] * frequencies
  extra_dims = ndim - angles.ndim
  if extra_dims > 0:
    angles = angles.reshape(angles.shape[:-1] + (1,) * extra_dims + angles.shape[-1:])
  cos = jnp.cos(angles).astype(dtype)
  sin = jnp.sin(angles).astype(dtype)
  return cos, sin


def apply_rope_factors(x, cos, sin):
  half_dim = x.shape[-1] // 2
  x1, x2 = x[..., :half_dim], x[..., half_dim:]
  out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return out.astype(x.dtype)


def apply_rope(x, positions, max_timescale=10_000.0):
  cos, sin = rope_factors(positions, rope_frequencies(x.shape[-1], max_timescale), x.ndim, x.dtype)
  return apply_rope_factors(x, cos, sin)
