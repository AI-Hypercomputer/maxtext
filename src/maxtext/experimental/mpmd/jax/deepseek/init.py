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

"""A test init for the DeepSeek-V3 port's parameter trees.

`init_params` draws a tree -- the full model's `model.param_avals(config)`, or
any subtree of it -- from its aval tree, straight onto the devices
`shardings` names, so the full model never has to exist on one host.  It is a
*test* init, not MaxText's: `normal / sqrt(fan_in)` with the norm scales at
one.  MaxText's own weights come through `convert.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _fan_in(path, shape):
  """Elements contracted by the kernel at `path` -- not simply `shape[-2]`.

  `shape[-2]` is right for an `[in, out]` kernel and wrong for MLA's 3-D ones:
  `query`, `wq_b` and `wkv_b` are `[in, heads, head_dim]`, so `shape[-2]` is the
  head count, and `out` is `[heads, v_head_dim, emb]`, which contracts two axes.
  Taking `heads` as the fan-in made queries 4x and attention logits ~16x too
  large at test widths, and a model that sharp amplifies fp32 rounding by
  three orders of magnitude -- enough to swamp every reassociation floor the
  correctness tests are built on.
  """
  if "token_embedder" in path:
    return 1  # embeddings are unit-scale
  if len(shape) < 2:
    return shape[-1]
  name = path[-2] if len(path) >= 2 else ""
  if name in ("query", "wq_b", "wkv_b"):
    return shape[-3]
  if name == "out" and "self_attention" in path:
    return shape[-3] * shape[-2]
  return shape[-2]


def _init_leaf(path, shape, dtype, key):
  """One leaf: `normal / sqrt(fan_in)`, norm scales at one.

  Biases and zero-sized leaves start at zero.  What a test init has to get
  right is activations that stay O(1) through the model, because a model with
  logits orders of magnitude too sharp amplifies fp32 rounding past the
  reassociation floors the correctness tests are built on (see `_fan_in`).
  """
  if int(np.prod(shape)) == 0:
    return jnp.zeros(shape, dtype)
  name = path[-1] if path else ""
  if name == "scale":  # RMSNorm gains start at one
    return jnp.ones(shape, dtype)
  if name == "bias":
    return jnp.zeros(shape, dtype)
  v = jax.random.normal(key, shape, jnp.float32) * (_fan_in(path, shape) ** -0.5)
  return v.astype(dtype)


def _dict_path(path):
  """A jax key path -> the tuple of its dict keys."""
  return tuple(k.key for k in path if isinstance(k, jax.tree_util.DictKey))


def init_params(avals, key, shardings=None):
  """Materialise the parameter tree `avals` describes (`model.param_avals`, or
  a subtree of it) straight onto `shardings` (None: jit's default placement).

  Driven by the aval tree, so the full model never exists anywhere.  Each
  leaf draws from its own key, split off `key` by the leaf's position in the
  tree, so the result is deterministic in `key` and the tree.
  """
  leaves, treedef = jax.tree_util.tree_flatten_with_path(avals)
  keys = jax.random.split(key, max(1, len(leaves)))

  def build():
    out = []
    for i, (path, a) in enumerate(leaves):
      shape, dt = tuple(int(d) for d in a.shape), a.dtype
      out.append(_init_leaf(_dict_path(path), shape, dt, keys[i]))
    return jax.tree.unflatten(treedef, out)

  return jax.jit(build, out_shardings=shardings)()
