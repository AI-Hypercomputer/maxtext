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

"""MaxText NNX state -> this port's parameter tree.

The main layout difference is the scanned block axis: MaxText stores every leaf
under `decoder/layers` with the block axis at `param_scan_axis` (1 by default,
e.g. `in_proj_qkvz [E, NB, 2Kd+2Vd]`, `wi_0 [X, NB, E, F]`); this port keeps it
at 0, natural for `jax.lax.scan`.  Names are MaxText's NNX path names
verbatim, so nothing else is renamed (cf. `src/maxtext/experimental/mpmd/jax/deepseek/convert.py`).

MaxText holds a block's layers as its nested scan: `local_layers`, the
block's cycle - 1 Gated DeltaNet layers scanned inside the block, each leaf
with their axis right after the block axis (`in_proj_qkvz [E, NB, cycle - 1,
...]`), and `global_layer`, its full-attention layer (`[E, NB, ...]`).  This
port keeps one subtree a layer, `layer_0 .. layer_{cycle-1}`, the
full-attention layer last: the layer order unchanged.  Leaves already in the
port's layout pass through as they are.
Imports no MaxText or Flax: it only walks `state.flat_state()`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.mpmd.jax.qwen35 import model as q_model


def _value(var):
  if hasattr(var, "get_value"):
    return var.get_value()
  return getattr(var, "value", var)


def _leaves(tree):
  return {
      jax.tree_util.keystr(p): (tuple(int(d) for d in a.shape), jnp.dtype(a.dtype))
      for p, a in jax.tree_util.tree_flatten_with_path(tree)[0]
  }


def check_params(params, config) -> None:
  """Raises ValueError unless `params` has exactly `model.param_avals(config)`'s
  leaves, shapes and dtypes.

  A MaxText setting that adds parameters (`mlp_bias`'s expert biases, say)
  would otherwise convert into leaves the forward never reads -- a different
  model, silently.
  """
  got, want = _leaves(params), _leaves(q_model.param_avals(config))
  extra = sorted(set(got) - set(want))
  missing = sorted(set(want) - set(got))
  wrong = sorted(f"{k}: {got[k]} != {want[k]}" for k in set(got) & set(want) if got[k] != want[k])
  if extra or missing or wrong:

    def head(items):
      return items[:6] + ([f"... {len(items) - 6} more"] if len(items) > 6 else [])

    raise ValueError(
        f"parameter tree is not param_avals(config): {len(extra)} extra leaves "
        f"{head(extra)}, {len(missing)} missing {head(missing)}, {len(wrong)} with "
        f"another shape/dtype {head(wrong)}"
    )


def from_nnx_state(state, param_scan_axis: int = 1, params_only: bool = True, config=None):
  """`nnx.split(model)[1]` of MaxText's `Qwen3_5` model -> nested dict of arrays.

  Keeps `nnx.Param` leaves only (no RNG state, no intermediates) and moves the
  block axis of every `decoder/layers` leaf from `param_scan_axis` to 0.
  `params_only=False` keeps every array leaf, for a state that is already
  parameters only, e.g. the gradient of a loss w.r.t. `nnx.split(model,
  nnx.Param, ...)[1]`.  With `config` (the `from_maxtext` of the same
  pyconfig) the result must be exactly `model.param_avals(config)`'s tree
  (`check_params`).
  """
  items = []
  for path, var in state.flat_state():
    parts = [str(getattr(p, "key", p)) for p in path]
    if "rngs" in parts or (params_only and type(var).__name__ != "Param"):
      continue
    if not hasattr(_value(var), "shape"):
      continue
    items.append((parts, _value(var)))
  return from_flat(items, param_scan_axis, config=config)


def from_flat(items, param_scan_axis: int = 1, config=None, asarray=jnp.asarray):
  """MaxText's parameters as (path parts, array) pairs -- `from_nnx_state`'s, or
  a params checkpoint's flattened -- -> this port's tree (the module docstring:
  the block axis to 0, the nested scan's layers to layer_0..).  `asarray`: what each
  array becomes (np.asarray keeps host arrays on the host, views where it can).
  With `config`, `check_params`."""
  params, global_layer = {}, {}
  moveaxis = jnp.moveaxis if asarray is jnp.asarray else np.moveaxis

  def put(tree, parts, val):
    node = tree
    for key in parts[:-1]:
      node = node.setdefault(key, {})
    node[parts[-1]] = val

  for parts, val in items:
    parts = [str(p) for p in parts]
    val = asarray(val)
    if "layers" in parts and param_scan_axis != 0:
      val = moveaxis(val, param_scan_axis, 0)
    nested = parts[:2] == ["decoder", "layers"] and len(parts) > 2 and parts[2] in ("local_layers", "global_layer")
    if nested and parts[2] == "local_layers":
      # [NB, ..., cycle - 1, ...] (the block axis now at 0, the local layers' axis still right after
      # param_scan_axis) -> layer_0 .. layer_{cycle-2}, by basic indexing: a view of a host array
      axis = param_scan_axis + 1
      for i in range(val.shape[axis]):
        put(params, ["decoder", "layers", f"layer_{i}"] + parts[3:], val[(slice(None),) * axis + (i,)])
    elif nested:
      put(global_layer, parts[3:], val)
    else:
      put(params, parts, val)
  if global_layer:
    # the block's full-attention layer comes last: layer_{cycle-1}, cycle - 1 being local_layers' length
    layers = params.setdefault("decoder", {}).setdefault("layers", {})
    layers[f'layer_{sum(k.startswith("layer_") for k in layers)}'] = global_layer
  if config is not None:
    check_params(params, config)
  return params
