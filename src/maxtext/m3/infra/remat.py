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

"""Path-transparent activation checkpointing (`nnx.remat`) for m3 decoder stacks.

Applied post-construction and prior to `maybe_apply_scan` (`construct -> remat -> scan -> optimizer`)
so each layer in an `nnx.Sequential` retains its exact parameter path AND shares a cached class and
`nnx.GraphDef` with identical sibling layers.
"""

from collections.abc import Callable
import functools
from typing import Any

from flax import nnx
import jax


def _resolve_checkpoint_policy(
    policy_name: str,
    save_names: tuple[str, ...] = (),
) -> Callable[..., Any] | None:
  """Maps a policy name (`full`, `minimal`, `save_qkv_proj`, `custom`, ...) to a JAX checkpoint policy."""
  normalized = policy_name.strip().lower()
  if normalized == "full":
    return jax.checkpoint_policies.nothing_saveable
  if normalized in ("minimal", "dots_saveable", "minimal_offloaded"):
    return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
  if normalized in ("save_qkv_proj", "qkv_proj_offloaded"):
    return jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj", "value_proj", "qkv_proj")
  if normalized == "save_out_proj":
    return jax.checkpoint_policies.save_only_these_names("out_proj")
  if normalized in (
      "save_dot_except_mlp",
      "save_dot_except_mlpwi",
      "save_dot_with_context_except_mlp",
  ):
    return jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj", "value_proj", "out_proj")
  if normalized == "custom":
    if save_names:
      return jax.checkpoint_policies.save_only_these_names(*save_names)
    return jax.checkpoint_policies.nothing_saveable
  raise ValueError(
      f"Unsupported m3 remat_policy {policy_name!r}; expected 'none', 'full', 'minimal', 'save_qkv_proj', "
      "'save_out_proj', 'save_dot_except_mlp', or 'custom'."
  )


def _is_tracer_or_array(value: Any) -> bool:
  return isinstance(value, (jax.Array, jax.core.Tracer))


@functools.cache
def _rematted_layer_class(
    layer_cls: type[nnx.Module],
    policy_name: str,
    save_names: tuple[str, ...],
    prevent_cse: bool,
) -> type[nnx.Module]:
  """Creates and caches a path-transparent subclass wrapping `layer_cls.__call__` in `nnx.remat`."""
  jax_policy = _resolve_checkpoint_policy(policy_name, save_names)

  class _RemattedLayer(layer_cls):  # type: ignore[valid-type,misc]
    """Path-transparent `nnx.remat` layer subclass."""

    _is_m3_rematted = True

    def __call__(self, carry: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
      # Partition array/tracer kwargs from static Python scalars (e.g. `deterministic=False`)
      # so `nnx.remat` never attempts to trace Python booleans or strings as dynamic JAX arrays.
      dyn_kwargs = {k: v for k, v in kwargs.items() if _is_tracer_or_array(v)}
      static_kwargs = {k: v for k, v in kwargs.items() if not _is_tracer_or_array(v)}

      def _call_impl(module: nnx.Module, x: Any, dyn_kw: dict[str, Any]) -> Any:
        return layer_cls.__call__(module, x, **dyn_kw, **static_kwargs)

      remat_call = nnx.remat(_call_impl, prevent_cse=prevent_cse, policy=jax_policy)
      return remat_call(self, carry, dyn_kwargs)

  _RemattedLayer.__name__ = f"Rematted{layer_cls.__name__}"
  _RemattedLayer.__qualname__ = f"Rematted{layer_cls.__qualname__}"
  return _RemattedLayer


def apply_remat(
    model: nnx.Module,
    *,
    policy: str = "full",
    save_names: tuple[str, ...] = (),
    prevent_cse: bool = False,
) -> nnx.Module:
  """Wraps every layer inside each `nnx.Sequential` in `model` with `nnx.remat` in place."""
  normalized = policy.strip().lower()
  if normalized in ("", "none"):
    return model
  # Validate policy eagerly even if the model has zero sequential layers.
  _resolve_checkpoint_policy(normalized, tuple(save_names))

  for _, node in nnx.iter_graph(model):
    if isinstance(node, nnx.Sequential):
      for layer in node.layers:
        if getattr(type(layer), "_is_m3_rematted", False):
          continue
        base_cls = type(layer)
        layer.__class__ = _rematted_layer_class(base_cls, normalized, tuple(save_names), prevent_cse)
  return model


def maybe_apply_remat(model: nnx.Module, config: Any) -> nnx.Module:
  """Applies `nnx.remat` to `model`'s decoder layers according to `config`."""
  transforms = getattr(config, "transforms", None)
  raw_policy = getattr(transforms, "remat_policy", None)
  if raw_policy is None:
    raw_policy = getattr(config, "remat_policy", "none")
  policy_str = str(getattr(raw_policy, "value", raw_policy)).strip().lower()
  if policy_str in ("", "none"):
    return model

  scan_enabled = getattr(transforms, "scan_layers", None)
  if scan_enabled is None:
    scan_enabled = getattr(config, "scan_layers", False)

  save_names: tuple[str, ...] = ()
  if transforms is not None and hasattr(transforms, "remat_save_names"):
    save_names = tuple(transforms.remat_save_names)
  elif hasattr(config, "tensors_on_device"):
    save_names = tuple(getattr(config, "tensors_on_device") or ())

  # Inside `jax.lax.scan`, `prevent_cse` must be False; without scan, `prevent_cse=True` stops XLA from
  # hoisting recomputed ops across unrolled layer boundaries.
  return apply_remat(
      model,
      policy=policy_str,
      save_names=save_names,
      prevent_cse=not bool(scan_enabled),
  )
