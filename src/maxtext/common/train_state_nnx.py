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
"""The NNX Unified TrainState."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


class TrainStateNNX(nnx.Module):
  """A unified container for NNX models and optimizers.

  This replaces Linen's TrainState for checkpointing.

  Linen TrainState pytree:
    {"params": {...}, "opt_state": {}...}
  TrainStateNNX state pytree:
    {"model": {...}, "optimizer": {"opt_state": {...}}}
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None,
  ):
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any, **kwargs):
    """Mimics the Linen apply_gradients function.

    Updates the optimizer state, applies updates to parameters, and increments
    the step counter. Only updates `self.model`. Extra kwargs (e.g. loss/grad_norm
    for the skip-step-on-spikes optimizer) are forwarded to the optax update.
    """
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without"
          " an optimizer. This usually happens when the state was created for"
          " inference only.")
    self.optimizer.update(self.model, grads, **kwargs)

  def to_pure_dict(self):
    """Returns the pure dict representation of the NNX state."""
    return nnx.state(self).to_pure_dict()


# On-disk checkpoint format.
#
# A pure_nnx run saves in the same on-disk layout as a Linen run, so the two are
# interchangeable. The NNX state pure dict differs from Linen's in three ways, all
# reshaped below at save time:
#   1. top-level keys: {model, optimizer:{step, opt_state}} -> {params:{params:...}, step, opt_state}
#   2. weights: model/... -> params/params/... (Linen `params` collection)
#   3. opt_state: int-keyed dict (empty entries skipped) -> list with None for EmptyState,
#      and mu/nu wrapped under the `params` collection
# Persistent NNX state Linen's params/opt_state/step layout can't hold -- rngs/dropout,
# batch stats, and any custom non-Param variable -- is split out by Variable type and stored
# under an `nnx_aux` subtree of `items` (see `to_checkpoint_dict`), so it survives a resume
# instead of resetting.

_NNX_RNG_STATE_KEYS = ("rngs", "dropout")


def _cast_step(step, dtype):
  """Casts the step's dtype, handling both concrete arrays and abstract ShapeDtypeStruct.

  NNX stores step as uint32 and Linen as int32; this also runs on the abstract
  (SDS) state when building a restore target, so it can't assume concrete
  values.
  """
  if isinstance(step, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(step.shape,
                                dtype,
                                sharding=getattr(step, "sharding", None))
  return jnp.asarray(step, dtype=dtype)


def _strip_rng_state(tree):
  """Removes the NNX-only 'rngs'/'dropout' subtrees that Linen doesn't carry.

  Subtrees that become empty after stripping are dropped too, so the result has
  no keys Linen wouldn't also have.
  """
  if not isinstance(tree, dict):
    return tree
  out = {}
  for k, v in tree.items():
    if k in _NNX_RNG_STATE_KEYS:
      continue
    stripped = _strip_rng_state(v)
    if isinstance(stripped, dict) and not stripped:
      continue
    out[k] = stripped
  return out


def _wrap_mu_nu_with_params(state):
  """Wraps mu/nu under an inner 'params' key (the Linen collection)."""
  if not isinstance(state, dict):
    return state
  return {
      k: {
          "params": v
      } if k in ("mu", "nu") and isinstance(v, dict) else
         v for k, v in state.items()
  }


def _as_chain_index(key):
  """Returns the int index for an int or digit-string key, else None."""
  if isinstance(key, int):
    return key
  if isinstance(key, str) and key.isdigit():
    return int(key)
  return None


def _opt_state_to_linen(opt_state):
  """Reshapes the NNX opt_state to Linen's on-disk layout.

  A chain (optax.adamw) is an int-keyed dict -> list-with-None. An un-chained
  optimizer (adam_pax) is flat (count/mu/nu) and stays a dict.
  """
  if not isinstance(opt_state, dict):
    return opt_state
  indices = [_as_chain_index(k) for k in opt_state.keys()]
  if not indices or any(i is None for i in indices):
    return _wrap_mu_nu_with_params(opt_state)
  chain = [None] * (max(indices) + 1)
  for key, idx in zip(opt_state.keys(), indices):
    chain[idx] = _wrap_mu_nu_with_params(opt_state[key])
  return chain


def _rename_nnx_to_linen_layers(d):
  """Converts NNX nested layer dicts ('scanned_blocks/layers/0' or 'layers_remainder/layers/0')
  to Linen-style flat layer keys ('layers_0', 'layers_1') for on-disk checkpoint compatibility.
  """
  if not isinstance(d, dict):
    return d
  res = {}
  num_scanned = 0

  def _count_scanned(d_sub):
    if not isinstance(d_sub, dict):
      return 0
    if "scanned_blocks" in d_sub and isinstance(d_sub["scanned_blocks"], dict):
      sb = d_sub["scanned_blocks"]
      if "layers" in sb and isinstance(sb["layers"], dict):
        return len(sb["layers"])
      return len(
          [k for k in sb.keys() if k.startswith("layers_") or str(k).isdigit()])
    for v_sub in d_sub.values():
      if isinstance(v_sub, dict):
        c = _count_scanned(v_sub)
        if c > 0:
          return c
    return 0

  num_scanned = _count_scanned(d)

  for k, v in d.items():
    if k == "scanned_blocks" and isinstance(v, dict):
      if "layers" in v and isinstance(v["layers"], dict):
        for l_idx, l_val in v["layers"].items():
          layer_key = f"layers_{l_idx}" if str(l_idx).isdigit() else str(l_idx)
          res[layer_key] = _rename_nnx_to_linen_layers(l_val)
      else:
        for sub_k, sub_v in v.items():
          layer_key = sub_k if sub_k.startswith("layers_") else (
              f"layers_{sub_k}" if str(sub_k).isdigit() else sub_k)
          res[layer_key] = _rename_nnx_to_linen_layers(sub_v)
    elif k == "layers_remainder" and isinstance(v, dict):
      if "layers" in v and isinstance(v["layers"], dict):
        for l_idx, l_val in v["layers"].items():
          idx_int = int(l_idx) if str(l_idx).isdigit() else 0
          global_idx = num_scanned + idx_int
          layer_key = f"layers_{global_idx}"
          res[layer_key] = _rename_nnx_to_linen_layers(l_val)
      else:
        for sub_k, sub_v in v.items():
          if sub_k.startswith("layers_") and sub_k[7:].isdigit():
            idx_int = int(sub_k[7:])
          elif str(sub_k).isdigit():
            idx_int = int(sub_k)
          else:
            idx_int = 0
          global_idx = num_scanned + idx_int
          layer_key = f"layers_{global_idx}"
          res[layer_key] = _rename_nnx_to_linen_layers(sub_v)
    else:
      res[k] = _rename_nnx_to_linen_layers(v)
  return res


def to_linen_checkpoint_dict(nnx_pure_dict):
  """Reshapes a TrainStateNNX pure dict ({model, optimizer}) into the Linen on-disk layout."""
  if not isinstance(nnx_pure_dict, dict):
    return nnx_pure_dict
  result = {}
  if "model" in nnx_pure_dict:
    model_dict = _rename_nnx_to_linen_layers(
        _strip_rng_state(nnx_pure_dict["model"]))
    if isinstance(model_dict,
                  dict) and "params" in model_dict and len(model_dict) == 1:
      result["params"] = model_dict
    else:
      result["params"] = {"params": model_dict}
  optimizer = nnx_pure_dict.get("optimizer")
  if isinstance(optimizer, dict):
    if "step" in optimizer:
      # NNX stores step as uint32; Linen uses int32.
      result["step"] = _cast_step(optimizer["step"], jnp.int32)
    if "opt_state" in optimizer:
      result["opt_state"] = _opt_state_to_linen(optimizer["opt_state"])
  return result


def _strip_mu_nu_params(state):
  """Inverse of `_wrap_mu_nu_with_params`: removes the inner 'params' wrap from mu/nu."""
  if not isinstance(state, dict):
    return state
  return {
      k: (v["params"] if k in ("mu", "nu") and isinstance(v, dict) and
          "params" in v else v) for k, v in state.items()
  }


def _opt_state_from_linen(opt_state):
  """Inverse of `_opt_state_to_linen`: Linen layout -> NNX opt_state.

  A chain (optax.adamw) is a list -> int-keyed dict. An un-chained state (adam_pax:
  `{count, mu, nu}`) stays flat -- re-wrapping it under a `0` index would not line
  up with the model's opt_state.
  """
  if isinstance(opt_state, list):
    return {
        i: _strip_mu_nu_params(e)
        for i, e in enumerate(opt_state)
        if isinstance(e, dict)
    }
  if not isinstance(opt_state, dict):
    return opt_state
  return _strip_mu_nu_params(opt_state)


def from_linen_checkpoint_dict(linen_pure_dict):
  """Inverse of `to_linen_checkpoint_dict`: Linen on-disk layout -> NNX layout.

  Doesn't restore NNX-only rngs/dropout (absent from Linen); callers fill those.
  """
  if not isinstance(linen_pure_dict, dict):
    return linen_pure_dict
  result = {}
  params = linen_pure_dict.get("params")
  if isinstance(params, dict) and "params" in params:
    result["model"] = params["params"]
  elif params is not None:
    result["model"] = params
  optimizer = {}
  if "step" in linen_pure_dict:
    # Linen stores step as int32; NNX uses uint32.
    optimizer["step"] = _cast_step(linen_pure_dict["step"], jnp.uint32)
  if "opt_state" in linen_pure_dict:
    optimizer["opt_state"] = _opt_state_from_linen(linen_pure_dict["opt_state"])
  if optimizer:
    result["optimizer"] = optimizer
  return result


def split_for_checkpoint(state: nnx.State):
  """Partition an NNX State into on-disk checkpoint collections.

  When LoRA parameters (nnx.LoRAParam) are present in the state, only LoRAParam
  is saved in `params` (adapter-only saving), and frozen base parameters are
  excluded to avoid duplicating base weights or saving quantized arrays.
  """
  has_lora = bool(nnx.filter_state(state, nnx.LoRAParam))
  flat = state.flat_state()
  if not has_lora:
    has_lora = any(
        any(
            isinstance(k, str) and ("lora_a" in k or "lora_b" in k)
            for k in path)
        for path, _ in flat)

  if has_lora and not bool(nnx.filter_state(state, nnx.LoRAParam)):
    # Abstract/unboxed state where variable wrappers were stripped to ShapeDtypeStruct
    params_flat = []
    custom_flat = []
    opt_flat = []
    aux_flat = []
    for path, val in flat:
      if path and path[0] == "optimizer":
        opt_flat.append((path, val))
      elif path and path[0] == "model":
        if any(isinstance(k, str) and k in ("rngs", "dropout") for k in path):
          aux_flat.append((path, val))
        elif any(
            isinstance(k, str) and ("lora_a" in k or "lora_b" in k)
            for k in path):
          params_flat.append((path, val))
        else:
          custom_flat.append((path, val))
      else:
        aux_flat.append((path, val))
    params = nnx.State.from_flat_path(params_flat)
    optimizer = nnx.State.from_flat_path(opt_flat)
    custom = nnx.State.from_flat_path(custom_flat)
    aux = nnx.State.from_flat_path(aux_flat)
    linen_state = nnx.merge_state(params, optimizer)
    aux_state = nnx.merge_state(aux, custom)
    return linen_state, aux_state, nnx.State({})

  param_type = nnx.LoRAParam if has_lora else nnx.Param

  params, rng_state, batch_stats, caches, intermediates, rest = nnx.split_state(
      state, param_type, nnx.RngState, nnx.BatchStat, nnx.Cache,
      nnx.Intermediate, ...)
  optimizer = nnx.State({"optimizer": rest["optimizer"]
                        }) if "optimizer" in rest else nnx.State({})
  custom = nnx.State({"model": rest["model"]
                     }) if ("model" in rest and not has_lora) else nnx.State({})
  linen_state = nnx.merge_state(params, optimizer)

  aux = nnx.merge_state(rng_state, batch_stats, custom)
  ephemeral = nnx.merge_state(caches, intermediates)
  return linen_state, aux, ephemeral


def to_checkpoint_dict(state: Any):
  """Reshapes an nnx.State or TrainStateNNX into the on-disk checkpoint layout.

  Weights (nnx.Param / nnx.LoRAParam) map to the Linen `params` collection and the optimizer to
  opt_state/step, so pure_nnx and Linen checkpoints stay interchangeable. Everything else that
  must persist -- rngs/dropout, batch stats, and any custom variable -- goes under an `nnx_aux`
  subtree. Works on a concrete state (save) or an abstract state (restore target).
  """
  nnx_state = state if isinstance(state, nnx.State) else nnx.state(state)
  linen_state, aux_state, _ = split_for_checkpoint(nnx_state)
  pure = linen_state.to_pure_dict()
  linen_dict = to_linen_checkpoint_dict({
      "model": pure.get("model", {}),
      "optimizer": pure.get("optimizer", {})
  })
  aux = aux_state.to_pure_dict()
  if aux:
    linen_dict["nnx_aux"] = aux
  return linen_dict
