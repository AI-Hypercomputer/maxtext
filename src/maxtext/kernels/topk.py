# Copyright 2023–2026 Google LLC
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

"""Top-k indices over a small last axis, as a Pallas TPU kernel.

XLA lowers `lax.top_k` on TPU to a sort, which costs about 1.2 ms for router
logits of shape [24576, 512] on tpu7x against a 55 us HBM roofline. This kernel
runs k rounds of max-and-mask in VMEM instead. Experts sit on sublanes and
tokens on lanes, so each round's max over experts is an elementwise reduction
across vregs rather than a cross-lane one.

The result matches `lax.top_k` indices exactly: descending by value, ties
broken by the lower index, -0.0 below +0.0, -inf entries taken in index order once the finite
ones run out.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _topk_kernel(x_ref, o_ref, *, k: int):
  v = x_ref[...].astype(jnp.float32)  # [E, bn]
  bits = jax.lax.bitcast_convert_type(v, jnp.int32)
  key = bits ^ ((bits >> 31) & 0x7FFFFFFF)  # int order = float order; -inf stays above INT_MIN
  num = v.shape[0]
  iota = jax.lax.broadcasted_iota(jnp.int32, v.shape, 0)
  floor = jnp.int32(jnp.iinfo(jnp.int32).min)  # marks taken entries
  for j in range(k):
    m = jnp.max(key, axis=0, keepdims=True)
    idx = jnp.min(jnp.where(key == m, iota, num), axis=0, keepdims=True)
    o_ref[pl.ds(j, 1), :] = idx
    key = jnp.where(iota == idx, floor, key)


@functools.partial(jax.jit, static_argnames=("k", "block_tokens", "interpret"))
def topk_indices(x: jax.Array, k: int, block_tokens: int = 256, interpret: bool = False) -> jax.Array:
  """Indices of the k largest entries along the last axis, as `lax.top_k(x, k)[1]`."""
  lead, num = x.shape[:-1], x.shape[-1]
  rows = 1
  for d in lead:
    rows *= d
  xt = x.reshape(rows, num).T  # [E, rows]
  pad = (-rows) % block_tokens
  if pad:
    xt = jnp.pad(xt, ((0, 0), (0, pad)))
  out = pl.pallas_call(
      functools.partial(_topk_kernel, k=k),
      out_shape=jax.ShapeDtypeStruct((k, rows + pad), jnp.int32),
      grid=((rows + pad) // block_tokens,),
      in_specs=[pl.BlockSpec((num, block_tokens), lambda i: (0, i))],
      out_specs=pl.BlockSpec((k, block_tokens), lambda i: (0, i)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=interpret,
  )(xt)
  return out[:, :rows].T.reshape(*lead, k)
