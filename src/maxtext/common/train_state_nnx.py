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

  For DPO (Direct Preference Optimization), an optional `reference_model`
  carries a frozen copy of the same architecture used to compute reference
  log-probabilities. Only `model` is updated by `apply_gradients`; the
  reference is held alongside so it is sharded, jit-traced, and checkpointed
  with the rest of the train state.
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None,
      reference_model: nnx.Module | None = None,
  ):
    self.model = model
    self.optimizer = optimizer
    if reference_model is not None:
      self.reference_model = reference_model

  def apply_gradients(self, grads: Any):
    """Mimics the Linen apply_gradients function.

    Updates the optimizer state, applies updates to parameters, and increments
    the step counter. Only updates `self.model`; `self.reference_model` (if
    present) is left untouched.
    """
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without"
          " an optimizer. This usually happens when the state was created for"
          " inference only."
      )
    self.optimizer.update(self.model, grads)


# On-disk checkpoint format.
#
# A pure_nnx run saves in the same on-disk layout as a Linen run, so the two are
# interchangeable. The NNX state pure dict differs from Linen's in three ways,
# all
# reshaped below at save time:
#   1. top-level keys: {model, optimizer:{step, opt_state}} -> {params:{params:...}, step, opt_state}
#   2. weights: model/... -> params/params/... (Linen `params` collection)
#   3. opt_state: int-keyed dict (empty entries skipped) -> list with None for EmptyState,
#      and mu/nu wrapped under the `params` collection
# NNX-only rngs/dropout state is dropped (Linen never had it).

_NNX_RNG_STATE_KEYS = ("rngs", "dropout")


def _cast_step(step, dtype):
  """Casts the step's dtype, handling both concrete arrays and abstract ShapeDtypeStruct.

  NNX stores step as uint32 and Linen as int32; this also runs on the abstract
  (SDS) state when building a restore target, so it can't assume concrete
  values.
  """
  if isinstance(step, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(
        step.shape, dtype, sharding=getattr(step, "sharding", None)
    )
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
      k: {"params": v} if k in ("mu", "nu") and isinstance(v, dict) else v
      for k, v in state.items()
  }


def _as_chain_index(key):
  """Returns the int index for an int or digit-string key, else None."""
  if isinstance(key, int):
    return key
  if isinstance(key, str) and key.isdigit():
    return int(key)
  return None


def _opt_state_to_linen(opt_state):
  """Reshapes the NNX optax-chain opt_state to Linen's list-with-None layout.

  NNX serializes the chain as an int-keyed dict, skipping empty entries; Linen
  uses a list with `None` for each `EmptyState`. A single-element chain is
  returned unwrapped to match Linen's un-chained optimizers (e.g. adam_pax).
  """
  if not isinstance(opt_state, dict):
    return opt_state
  indices = [_as_chain_index(k) for k in opt_state.keys()]
  if not indices or any(i is None for i in indices):
    return _wrap_mu_nu_with_params(opt_state)
  chain = [None] * (max(indices) + 1)
  for key, idx in zip(opt_state.keys(), indices):
    chain[idx] = _wrap_mu_nu_with_params(opt_state[key])
  return chain[0] if len(chain) == 1 else chain


def to_linen_checkpoint_dict(nnx_pure_dict):
  """Reshapes a TrainStateNNX pure dict ({model, optimizer}) into the Linen on-disk layout."""
  if not isinstance(nnx_pure_dict, dict):
    return nnx_pure_dict
  result = {}
  if "model" in nnx_pure_dict:
    result["params"] = {"params": _strip_rng_state(nnx_pure_dict["model"])}
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
      k: (
          v["params"]
          if k in ("mu", "nu") and isinstance(v, dict) and "params" in v
          else v
      )
      for k, v in state.items()
  }


def _opt_state_from_linen(opt_state):
  """Inverse of `_opt_state_to_linen`: Linen list-with-None -> NNX int-keyed dict."""
  if isinstance(opt_state, list):
    return {
        i: _strip_mu_nu_params(e)
        for i, e in enumerate(opt_state)
        if isinstance(e, dict)
    }
  if not isinstance(opt_state, dict):
    return opt_state
  return {0: _strip_mu_nu_params(opt_state)}


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
