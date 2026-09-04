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

"""Flax Linen and NNX interceptors for automatic tensor distribution debugging.

Provides non-intrusive hooks to monitor layer activations, auxiliary outputs,
and MoE routing statistics in MaxText without modifying model code.

Key Mechanisms:
  - Flax Linen: Intercepts module calls via `nn.intercept_methods` to capture
    forward outputs and auxiliary tensors (e.g., KV cache, MoE loss).
  - Flax NNX: Traverses module hierarchies to wrap `__call__` and MoE routing
    methods (`get_topk`, `reshape_and_update_weights`).
  - Scoping: `debug_telemetry_scope` activates capture for configured steps and
    layer filters, incurring zero overhead when disabled.

Key Entry Points:
  - `debug_telemetry_scope`: Context manager activating telemetry and Linen
    interception.
  - `wrap_nnx_module_for_debug`: Traverses and instruments an NNX module tree.
  - `is_debug_telemetry_active`: Checks if telemetry is currently active.
  - `get_active_telemetry_context`: Returns active config and step.
"""

import contextlib
import dataclasses
import threading
from typing import Any
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.utils.debug_tensor_utils import debug_tensor
from maxtext.utils.debug_tensor_utils import should_debug_tensor
import numpy as np


@dataclasses.dataclass
class _TelemetryState:
  active: bool = False
  config: Any = None
  step: int | jax.Array = 0


_TELEMETRY_STATE = threading.local()


def _get_telemetry_state() -> _TelemetryState:
  if not hasattr(_TELEMETRY_STATE, "state"):
    _TELEMETRY_STATE.state = _TelemetryState()
  return _TELEMETRY_STATE.state


def is_debug_telemetry_active() -> bool:
  """Returns True if debug telemetry scope is currently active and enabled."""
  state = _get_telemetry_state()
  return state.active and getattr(state.config, "debug_tensor_distribution", False)


def get_active_telemetry_context() -> tuple[Any, int | jax.Array]:
  """Returns (config, step) of currently active debug telemetry scope."""
  state = _get_telemetry_state()
  return state.config, state.step


def _instrument_output(
    out: Any,
    primary_tag: str,
    aux_tag: str | None,
    step: int | jax.Array,
) -> Any:
  """Instruments a tensor or tuple of tensors with primary and optional aux debug tags."""
  if isinstance(out, (jax.Array, np.ndarray, jnp.ndarray)):
    return debug_tensor(out, primary_tag, step=step, enabled=True)
  if isinstance(out, tuple) and len(out) > 0:
    items = list(out)
    if isinstance(items[0], (jax.Array, np.ndarray, jnp.ndarray)):
      items[0] = debug_tensor(items[0], primary_tag, step=step, enabled=True)
    if aux_tag and len(items) > 1 and items[1] is not None and isinstance(items[1], (jax.Array, np.ndarray, jnp.ndarray)):
      items[1] = debug_tensor(items[1], aux_tag, step=step, enabled=True)
    return tuple(items)
  return out


def linen_interceptor_fn(next_fun, args, kwargs, context: nn.module.InterceptorContext):
  """Flax Linen method interceptor that automatically instruments module outputs."""
  out = next_fun(*args, **kwargs)
  if not is_debug_telemetry_active():
    return out

  config, step = get_active_telemetry_context()
  path_tuple = getattr(context.module, "path", ())
  if not path_tuple:
    return out

  path_tag = "/".join(str(p) for p in path_tuple)
  if not should_debug_tensor(config, path_tag, step):
    return out

  tag_lower = path_tag.lower()
  if any(k in tag_lower for k in ("moe", "router", "expert", "mhc")):
    aux_tag = f"{path_tag}/load_balancing_loss"
  elif any(k in tag_lower for k in ("attn", "attention")):
    aux_tag = f"{path_tag}/kv_cache"
  else:
    aux_tag = f"{path_tag}/aux_output"

  return _instrument_output(out, path_tag, aux_tag, step)


_DECODER_LAYER_NAMES = (
    "DecoderLayer",
    "NNXDecoderLayer",
    "LlamaDecoderLayer",
    "MixtralDecoderLayer",
    "DeepSeekDenseLayer",
    "DeepSeekMoELayer",
    "Gemma4DecoderLayer",
    "Qwen3NextDecoderLayer",
)


def _get_nnx_tags(node: nnx.Module, tag: str) -> tuple[str, str | None]:
  """Determines primary and auxiliary output tags for an NNX module."""
  name = type(node).__name__
  if name == "GateLogit":
    return f"{tag}/gate_logits", None
  if name == "RoutedAndSharedMoE":
    return f"{tag}/combined_outputs", f"{tag}/load_balancing_loss"
  if (
      name in ("RoutedMoE", "MoeBlock")
      or name.endswith(("MoE", "Moe", "MoeBlock", "SparseMoeBlock", "MoELayer"))
      or hasattr(node, "get_topk")
  ):
    return f"{tag}/expert_outputs", f"{tag}/load_balancing_loss"
  if name.endswith("DecoderLayer") or name in _DECODER_LAYER_NAMES:
    return f"{tag}/layer_output", None
  return tag, None


def _is_telemetry_active_for_tag(tag: str) -> tuple[bool, int | jax.Array]:
  """Checks if debug telemetry is active and enabled for the given tag."""
  if not is_debug_telemetry_active():
    return False, 0
  config, step = get_active_telemetry_context()
  return should_debug_tensor(config, tag, step), step


def _wrap_call(fn, primary_tag: str, aux_tag: str | None, node_tag: str):
  """Wraps an NNX module's __call__ method to instrument its output."""

  def wrapped_call(self, *args, **kwargs):
    out = fn(self, *args, **kwargs)
    active, step = _is_telemetry_active_for_tag(node_tag)
    if not active:
      return out
    return _instrument_output(out, primary_tag, aux_tag, step)

  return wrapped_call


def _wrap_topk(fn, node_tag: str):
  """Wraps an MoE module's get_topk method to instrument router weights."""
  tag = f"{node_tag}/router_weights"

  def wrapped_topk(self, *args, **kwargs):
    res = fn(self, *args, **kwargs)
    active, step = _is_telemetry_active_for_tag(tag)
    if active and isinstance(res, tuple) and len(res) >= 2:
      weights = debug_tensor(res[0], tag, step=step, enabled=True)
      return (weights, *res[1:])
    return res

  return wrapped_topk


def _wrap_reshape_and_update_weights(fn, node_tag: str):
  """Wraps an MoE module's reshape_and_update_weights to instrument combine weights."""
  tag = f"{node_tag}/combine_weights"

  def wrapped_reshape(self, *args, **kwargs):
    weights = fn(self, *args, **kwargs)
    active, step = _is_telemetry_active_for_tag(tag)
    if active and isinstance(weights, (jax.Array, np.ndarray, jnp.ndarray)):
      return debug_tensor(weights, tag, step=step, enabled=True)
    return weights

  return wrapped_reshape


def _instrument_nnx_node(node: nnx.Module, node_tag: str) -> None:
  """Subclasses an NNX module node in-place to wrap methods with telemetry."""
  primary_tag, aux_tag = _get_nnx_tags(node, node_tag)
  methods = {
      "__call__": _wrap_call(node.__class__.__call__, primary_tag, aux_tag, node_tag),
      "_is_debug_wrapped": True,
  }
  if hasattr(node, "get_topk"):
    methods["get_topk"] = _wrap_topk(node.__class__.get_topk, node_tag)
  if hasattr(node, "reshape_and_update_weights"):
    methods["reshape_and_update_weights"] = _wrap_reshape_and_update_weights(
        node.__class__.reshape_and_update_weights, node_tag
    )
  node.__class__ = type(node.__class__.__name__, (node.__class__,), methods)


def _format_node_tag(parent_path: str, path: tuple[Any, ...]) -> str:
  """Formats hierarchical tag string from parent path and graph path."""
  subpath = "/".join(str(p) for p in path) if path else ""
  if parent_path and subpath:
    return f"{parent_path}/{subpath}"
  return parent_path or subpath


def wrap_nnx_module_for_debug(
    module: nnx.Module,
    parent_path: str = "",
    step: int | jax.Array = 0,
    config: Any = None,
) -> nnx.Module:
  """Centralized NNX wrapping hook to instrument NNX module outputs with hierarchical tags."""
  if config is None:
    config, step = get_active_telemetry_context()

  if not getattr(config, "debug_tensor_distribution", False):
    return module

  for path, node in nnx.iter_graph(module):
    if not isinstance(node, nnx.Module) or getattr(node, "_is_debug_wrapped", False):
      continue

    node_tag = _format_node_tag(parent_path, path)
    if not node_tag:
      continue

    _instrument_nnx_node(node, node_tag)

  return module


@contextlib.contextmanager
def debug_telemetry_scope(config: Any = None, step: int | jax.Array = 0):
  """Scope context manager activating Flax Linen & NNX unified interceptors.

  When config is None and no active scope exists, or debug_tensor_distribution
  is False,
  this yields immediately with zero overhead at trace time.
  """
  state = _get_telemetry_state()
  active_config = config if config is not None else state.config
  if active_config is None or not getattr(active_config, "debug_tensor_distribution", False):
    old_step = state.step
    state.step = step
    try:
      yield
    finally:
      state.step = old_step
    return

  old_active, old_config, old_step = state.active, state.config, state.step
  state.active = True
  state.config = active_config
  state.step = step

  try:
    with nn.intercept_methods(linen_interceptor_fn):
      yield
  finally:
    state.active = old_active
    state.config = old_config
    state.step = old_step
