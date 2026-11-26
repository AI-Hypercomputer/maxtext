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

"""Parameter init for the Qwen3.5 port, by MaxText's rules where they shape the math.

`init_params` materialises a parameter tree (`model.param_avals(config)`, or a
subtree of it) straight onto its devices; `leaf_rule` says how each leaf is
drawn.  `ds_ep.qwen_init_rule` (the MPMD pipeline's `StagePrograms.init`)
holds a second copy of the zeros / ones / a_log choice and of `A_LOG_RANGE`
(`ds_ep_qwen_test.QwenInit` holds the two to each other).

Not in `model.py`: `qwen35_model_test.CallSiteTest` holds the model modules to
making no fresh array (`jnp.zeros` / `jnp.ones`) of their own.
"""

from __future__ import annotations

import math
import zlib

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.qwen35.model import LAYERS


def _dict_path(path):
  """A tuple of dict keys, or a jax key path -> its dict keys."""
  if path and not isinstance(path[0], str):
    return tuple(k.key for k in path if isinstance(k, jax.tree_util.DictKey))
  return tuple(path)


def is_stacked(path) -> bool:
  """True for a leaf under `decoder/layers`, whose axis 0 is the block axis.

  `path` is a tuple of dict keys (or a jax key path).  DeepSeek's stacks are
  `dense_layers` / `moe_layers`; a rule that looks for those would treat every
  Qwen leaf as unstacked.
  """
  keys = _dict_path(path)
  return len(keys) >= 2 and keys[0] == "decoder" and keys[1] == LAYERS


def fan_in(path, shape) -> int:
  """Elements the kernel at `path` contracts.  Stacked leaves carry the block
  axis first, so everything is read from the right.

    [.., in, out]       2-D kernels (projections, gates, experts, head): shape[-2]
    [.., E, heads, D]   query / key / value: shape[-3]
    [.., K, 1, C]       the depthwise conv: K taps x 1 input channel = shape[-3]
    embedding           1 (unit-scale rows, MaxText's normal(1.0))
  """
  keys = _dict_path(path)
  shape = tuple(int(d) for d in shape)
  if "token_embedder" in keys:
    return 1
  if len(shape) < 2:
    return max(1, shape[-1]) if shape else 1
  parent = keys[-2] if len(keys) >= 2 else ""
  if parent in ("query", "key", "value", "conv1d"):
    return shape[-3]
  return shape[-2]


def _leaf_key(key, keys):
  """`key` folded with a stable hash of the leaf's path (not its tree position)."""
  return jax.random.fold_in(key, zlib.crc32("/".join(keys).encode()) & 0x7FFFFFFF)


# `A_log = log A` with A ~ U(A_LOG_RANGE): qwen3.py's A ~ U(0, 16], kept off 0.
A_LOG_RANGE = (1e-9, 16.0)
LEAF_KINDS = ("zeros", "ones", "a_log", "normal")


def leaf_rule(path, shape) -> tuple[str, int]:
  """`(kind, fan_in)`: how the init draws the leaf at `path`, of (per-block) `shape`.

    'zeros'   the zero-centred norm scales (input / post-attention / final,
              q / k norms) and any empty leaf
    'ones'    the GDN output (gated) norm scale, `dt_bias`
    'a_log'   `A_log` = log U(A_LOG_RANGE)
    'normal'  normal / sqrt(fan_in): every kernel and the embedding

  `fan_in` is 0 unless the kind is 'normal'.  `init_leaf` draws by it and
  refuses a kind it does not know, so a leaf given a rule of its own here
  needs a draw there too.  `ds_ep.qwen_init_rule` holds a second copy of the
  zeros / ones / a_log choice, and of A_LOG_RANGE, but draws 'normal' as
  N(0, 0.02^2) rather than by fan_in.
  """
  keys = _dict_path(path)
  shape = tuple(int(d) for d in shape)
  name = keys[-1] if keys else ""
  if math.prod(shape) == 0:
    return "zeros", 0
  if name == "scale":
    gated = len(keys) >= 3 and keys[-3:-1] == ("norm", "rms_norm")
    return ("ones" if gated else "zeros"), 0
  if name == "A_log":
    return "a_log", 0
  if name == "dt_bias":
    return "ones", 0
  return "normal", fan_in(keys, shape)


def _init_one(keys, shape, dtype, key):
  kind, fan = leaf_rule(keys, shape)
  if kind == "zeros":
    return jnp.zeros(shape, dtype)
  if kind == "ones":
    return jnp.ones(shape, dtype)
  if kind == "a_log":
    lo, hi = A_LOG_RANGE
    return jnp.log(jax.random.uniform(key, shape, jnp.float32, minval=lo, maxval=hi)).astype(dtype)
  if kind == "normal":
    return (jax.random.normal(key, shape, jnp.float32) * (fan**-0.5)).astype(dtype)
  raise ValueError(f'no init for leaf kind {kind!r} at {"/".join(keys)}')


def init_leaf(path, shape, dtype, key):
  """One parameter, by MaxText's Qwen3.5 rules where they matter for the math
  (`leaf_rule`):

    zero-centred norm scales (input / post-attention / final, q / k norms)  0
    the GDN output (gated) norm scale                                    1
    A_log                                    log U(1e-9, 16)   (qwen3.py, = log A, A ~ U(0, 16])
    dt_bias                                                              1
    every kernel and the embedding     normal / sqrt(fan_in)   (MaxText: truncated normal)

  The norm rules are not cosmetic: a zero-centred norm initialised at 1
  doubles its gain.  `key` is the leaf's own key (`init_params` folds the
  leaf's path into the model key).  A stacked (`decoder/layers`) leaf draws
  block b's slice from `fold_in(key, b)`.
  """
  keys = _dict_path(path)
  shape, dtype = tuple(int(d) for d in shape), jnp.dtype(dtype)
  if not is_stacked(keys) or not shape:
    return _init_one(keys, shape, dtype, key)
  if not shape[0]:
    return jnp.zeros(shape, dtype)
  # A loop (lax.map), not a vmap: each block's draw is then exactly its
  # unbatched draw for ANY key implementation -- the RngBitGenerator ones
  # ('rbg') give batch-size-dependent bits under vmap, threefry does not.
  return jax.lax.map(
      lambda b: _init_one(keys, shape[1:], dtype, jax.random.fold_in(key, b)), jnp.arange(shape[0], dtype=jnp.uint32)
  )


def init_params(avals, key, shardings=None):
  """Materialise a parameter tree (`model.param_avals(config)`, or a subtree
  of it) straight onto `shardings` (None: jit's default placement).

  Driven by the aval tree, so the full model never exists anywhere.  Every
  leaf draws from `key` folded with its path, so a leaf's values do not
  depend on what else is in the tree.
  """
  leaves, treedef = jax.tree_util.tree_flatten_with_path(avals)

  def build():
    out = []
    for path, a in leaves:
      keys = _dict_path(path)
      out.append(init_leaf(keys, a.shape, a.dtype, _leaf_key(key, keys)))
    return jax.tree.unflatten(treedef, out)

  return jax.jit(build, out_shardings=shardings)()
