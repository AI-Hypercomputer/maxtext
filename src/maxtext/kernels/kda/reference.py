# Copyright 2023-2026 Google LLC
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

"""Pure-JAX references for the KDA kernels (per-token recurrence), for tests and as executable documentation."""

import jax
import jax.numpy as jnp

from maxtext.kernels.kda import api


def kda_recurrent_reference(q, k, v, log_decay, beta, *, segment_ids=None, resets=None):
  """The KDA recurrence scanned token by token in f32 (`[B, T, H, d]` in, `[B, T, H, dv]` f32 out).

  Same contract as `api.chunk_kda`: q, k L2-normalized (q also scaled), log_decay <= 0, beta `[B, T, H]`; the
  state is zeroed at every document start.
  """
  f32 = jnp.float32
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  if resets is None:
    resets = api.segment_resets(segment_ids, batch, seq_len)

  def step(state, xs):
    q_t, k_t, v_t, ld_t, beta_t, reset_t = xs
    state = jnp.where(reset_t[:, None, None, None], 0.0, state)
    state = state * jnp.exp(ld_t)[..., None]  # forget: Diag(alpha_t) S
    prediction = jnp.einsum("bhkv,bhk->bhv", state, k_t, precision=jax.lax.Precision.HIGHEST)
    delta = (v_t - prediction) * beta_t[..., None]
    state = state + jnp.einsum("bhk,bhv->bhkv", k_t, delta, precision=jax.lax.Precision.HIGHEST)
    return state, jnp.einsum("bhkv,bhk->bhv", state, q_t, precision=jax.lax.Precision.HIGHEST)

  init = jnp.zeros((batch, heads, dk, dv), f32)
  xs = tuple(jnp.swapaxes(t.astype(f32) if t.dtype != bool else t, 0, 1) for t in (q, k, v, log_decay, beta, resets))
  _, out = jax.lax.scan(step, init, xs)
  return jnp.swapaxes(out, 0, 1)
