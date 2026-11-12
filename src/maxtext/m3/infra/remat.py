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

"""Path-transparent rematerialization (gradient checkpointing) for m3 models.

Applies activation checkpointing after model construction without inserting
wrapper modules into parameter paths (`nnx.state(model, nnx.Param)` stays
identical across `none` / `full` / `minimal` / `custom`, scanned or unscanned,
regardless of whether `maybe_apply_remat` runs before or after `maybe_apply_scan`).
"""

from collections.abc import Callable, Iterator, MutableMapping, Sequence
import dataclasses
import inspect
from typing import Any, Optional

from flax import nnx
import jax
from jax import checkpoint_policies as cp
import jax.extend as jex

from maxtext.m3.infra import scan as m3_scan

_TRAIN_MODE = "train"

_PROJECTIONS = ("query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj")
_MINIMAL = _PROJECTIONS + ("out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo")
_MINIMAL_WITH_CONTEXT = _MINIMAL + ("context",)
_HIDDEN_OFFLOAD = ("hidden_states", "self_attn", "cross_attn")

# Supported `remat_policy` config values (case-insensitive).
REMAT_POLICIES = (
    "none",
    "full",
    "minimal",
    "minimal_with_context",
    "minimal_flash",
    "save_qkv_proj",
    "save_dot_except_mlp",
    "save_dot_except_mlpwi",
    "save_out_proj",
    "qkv_proj_offloaded",
    "minimal_offloaded",
    "matmul_without_batch",
    "offload_matmul_without_batch",
    "custom",
    "hidden_state_with_offload",
)


@dataclasses.dataclass(frozen=True)
class Remat:
  """Resolved checkpointing policy and expected activation tag names."""

  policy: Callable[..., Any] | None = cp.nothing_saveable
  names: tuple[str, ...] | None = None


def policy_callable(remat: Remat | Callable[..., Any] | None) -> Callable[..., Any] | None:
  """Extracts the `jax.checkpoint` policy callable from `remat`, or `None` when remat is disabled."""
  if remat is None:
    return None
  if isinstance(remat, Remat):
    return cp.nothing_saveable if remat.policy is None else remat.policy
  if callable(remat):
    return remat
  raise TypeError(f"Expected Remat, callable policy, or None; got {remat!r}.")


def _policy_names(
    key: str,
    saved: tuple[str, ...],
    offloaded: tuple[str, ...],
) -> tuple[str, ...] | None:
  """Returns the `checkpoint_name` tags targeted by `key`, or `None` if tag-free."""
  match key:
    case "full" | "matmul_without_batch" | "offload_matmul_without_batch":
      return None
    case "minimal" | "minimal_flash" | "minimal_offloaded":
      return _MINIMAL
    case "minimal_with_context":
      return _MINIMAL_WITH_CONTEXT
    case "save_qkv_proj" | "qkv_proj_offloaded":
      return _PROJECTIONS
    case "save_dot_except_mlp":
      return _PROJECTIONS + ("out_proj",)
    case "save_dot_except_mlpwi":
      return _PROJECTIONS + ("out_proj", "mlpwo")
    case "save_out_proj":
      return ("out_proj",)
    case "hidden_state_with_offload":
      return _HIDDEN_OFFLOAD
    case "custom":
      return saved + offloaded
    case _:
      return None


def remat_policy(
    name: Optional[str],
    names_which_can_be_saved: Optional[Sequence[str]] = None,
    names_which_can_be_offloaded: Optional[Sequence[str]] = None,
) -> Optional[Callable[..., Any]]:
  """Returns the `jax.checkpoint` policy callable for `name`, or `None` for `'none'`."""
  if name is None:
    return None
  if not isinstance(name, str):
    raise ValueError(f"Unknown remat_policy {name!r}; expected one of {REMAT_POLICIES}.")
  key = name.lower()
  saved = list(names_which_can_be_saved or [])
  offloaded = list(names_which_can_be_offloaded or [])
  match key:
    case "none":
      return None
    case "full":
      return cp.nothing_saveable
    case "minimal" | "minimal_flash":
      return cp.save_only_these_names(*_MINIMAL)
    case "minimal_with_context":
      return cp.save_only_these_names(*_MINIMAL_WITH_CONTEXT)
    case "save_qkv_proj":
      return cp.save_only_these_names(*_PROJECTIONS)
    case "save_dot_except_mlp":
      return cp.save_only_these_names(*_PROJECTIONS, "out_proj")
    case "save_dot_except_mlpwi":
      return cp.save_only_these_names(*_PROJECTIONS, "out_proj", "mlpwo")
    case "save_out_proj":
      return cp.save_only_these_names("out_proj")
    case "qkv_proj_offloaded":
      return cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=list(_PROJECTIONS),
          offload_src="device",
          offload_dst="pinned_host",
      )
    case "minimal_offloaded":
      return cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=list(_MINIMAL),
          offload_src="device",
          offload_dst="pinned_host",
      )
    case "matmul_without_batch":
      return cp.checkpoint_dots_with_no_batch_dims
    case "offload_matmul_without_batch":
      return cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host")
    case "custom":
      return cp.save_and_offload_only_these_names(
          names_which_can_be_saved=saved,
          names_which_can_be_offloaded=offloaded,
          offload_src="device",
          offload_dst="pinned_host",
      )
    case "hidden_state_with_offload":
      return cp.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=list(_HIDDEN_OFFLOAD),
          offload_src="device",
          offload_dst="pinned_host",
      )
  raise ValueError(f"Unknown remat_policy {name!r}; expected one of {REMAT_POLICIES}.")


def resolve_policy(config: Any) -> Remat | None:
  """Resolves `config.remat_policy` (and custom save/offload lists) into a `Remat` or `None`."""
  raw = getattr(config, "remat_policy", "none")
  if raw is None or raw == "":
    return None
  if not isinstance(raw, str):
    raise ValueError(f"Unknown remat_policy {raw!r}; expected one of {REMAT_POLICIES}.")
  key = raw.lower()
  if key == "none":
    return None
  on_device = tuple(getattr(config, "tensors_on_device", None) or getattr(config, "names_which_can_be_saved", None) or ())
  to_offload = tuple(
      getattr(config, "tensors_to_offload", None) or getattr(config, "names_which_can_be_offloaded", None) or ()
  )
  pol = remat_policy(
      key,
      names_which_can_be_saved=on_device,
      names_which_can_be_offloaded=to_offload,
  )
  assert pol is not None
  return Remat(policy=pol, names=_policy_names(key, on_device, to_offload))


def _filter_kwargs(fn: Callable[..., Any], num_pos_args: int, kwargs: dict[str, Any]) -> dict[str, Any]:
  """Filters `kwargs` to parameters accepted by `fn`, matching `nnx.Sequential`."""
  sig = inspect.signature(fn)
  params = list(sig.parameters.values())
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
    return dict(kwargs)
  pos_consumed = 0
  accepted: set[str] = set()
  for p in params:
    if p.kind == inspect.Parameter.POSITIONAL_ONLY:
      pos_consumed += 1
    elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
      if pos_consumed < num_pos_args:
        pos_consumed += 1
      else:
        accepted.add(p.name)
    elif p.kind == inspect.Parameter.VAR_POSITIONAL:
      pos_consumed = num_pos_args
    elif p.kind == inspect.Parameter.KEYWORD_ONLY:
      accepted.add(p.name)
  return {k: v for k, v in kwargs.items() if k in accepted}


def _is_traced_value(val: Any) -> bool:
  return isinstance(val, jax.Array)


class RematSequential(nnx.Sequential):
  """Path-transparent `nnx.Sequential` subclass applying per-layer `jax.checkpoint`."""

  def __init__(self, *fns: Callable[..., Any], remat: Remat | None = None):
    super().__init__(*fns)
    self.remat = remat

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    policy = policy_callable(self.remat)
    if policy is None or len(self.layers) == 0:
      return super().__call__(*args, **kwargs)

    output: Any = None
    for i, f in enumerate(self.layers):
      if not callable(f):
        raise TypeError(f"Sequence[{i}] is not callable: {f}")
      if i > 0:
        if isinstance(output, tuple):
          args = output
          new_kwargs = None
        elif isinstance(output, dict):
          args = ()
          new_kwargs = dict(output)
        else:
          args = (output,)
          new_kwargs = None
      else:
        new_kwargs = None

      layer_kwargs = _filter_kwargs(f, len(args), kwargs) if kwargs else {}
      if new_kwargs is not None:
        if repeated_kwargs := set(new_kwargs).intersection(layer_kwargs):
          raise TypeError(f"Got multiple values for arguments {repeated_kwargs} in Sequential[{i}]")
        layer_kwargs = {**new_kwargs, **layer_kwargs}

      if isinstance(f, m3_scan.ScannedLayers):
        # `ScannedLayers` already checkpoints each layer inside its scan body.
        output = f(*args, **layer_kwargs)
      elif isinstance(f, nnx.Module):
        graphdef, params, rest = nnx.split(f, nnx.Param, ...)
        traced_kw = {k: v for k, v in layer_kwargs.items() if _is_traced_value(v)}
        static_kw = {k: v for k, v in layer_kwargs.items() if k not in traced_kw}

        def _run_mod(p_st, r_st, pos_args, dyn_kw, gd=graphdef, st_kw=static_kw):
          mod = nnx.merge(gd, p_st, r_st)
          out_val = mod(*pos_args, **dyn_kw, **st_kw)
          _, _, updated_r = nnx.split(mod, nnx.Param, ...)
          return out_val, updated_r

        output, new_rest = jax.checkpoint(_run_mod, policy=policy, prevent_cse=True)(params, rest, args, traced_kw)
        if jax.tree.leaves(new_rest):
          nnx.update(f, new_rest)
      else:
        traced_kw = {k: v for k, v in layer_kwargs.items() if _is_traced_value(v)}
        static_kw = {k: v for k, v in layer_kwargs.items() if k not in traced_kw}

        def _run_fn(pos_args, dyn_kw, fn=f, st_kw=static_kw):
          return fn(*pos_args, **dyn_kw, **st_kw)

        output = jax.checkpoint(_run_fn, policy=policy, prevent_cse=True)(args, traced_kw)

    return output


def _replace_child(model: nnx.Module, path: tuple[Any, ...], replacement: nnx.Module) -> None:
  parent: Any = model
  for key in path[:-1]:
    parent = parent[key] if isinstance(key, int) or isinstance(parent, MutableMapping) else getattr(parent, key)
  key = path[-1]
  if isinstance(key, int) or isinstance(parent, MutableMapping):
    parent[key] = replacement
  else:
    setattr(parent, key, replacement)


def apply_remat(model: nnx.Module, remat: Remat | None) -> nnx.Module:
  """Applies `remat` across `ScannedLayers`, `nnx.Sequential`, and `set_remat` hooks in place."""
  hook_nodes: list[Any] = []
  scanned_nodes: list[m3_scan.ScannedLayers] = []
  seq_containers: list[tuple[tuple[Any, ...], nnx.Sequential]] = []

  for path, node in nnx.iter_graph(model):
    if isinstance(node, m3_scan.ScannedLayers):
      scanned_nodes.append(node)
    elif isinstance(node, nnx.Sequential):
      seq_containers.append((path, node))
    elif isinstance(node, nnx.Module) and callable(getattr(node, "set_remat", None)):
      hook_nodes.append(node)

  for node in hook_nodes:
    node.set_remat(remat)
  for node in scanned_nodes:
    node.remat = remat
  for path, container in seq_containers:
    if container.layers and all(isinstance(child, m3_scan.ScannedLayers) for child in container.layers):
      continue
    if isinstance(container, RematSequential):
      container.remat = remat
    elif remat is not None:
      replacement = RematSequential(*container.layers, remat=remat)
      replacement.layers = container.layers
      if not path:
        model = replacement
      else:
        _replace_child(model, path, replacement)
  return model


def maybe_apply_remat(
    model: nnx.Module,
    config: Any,
    model_mode: str = _TRAIN_MODE,
) -> nnx.Module:
  """Applies `config.remat_policy` during training and disables remat in non-train modes."""
  if model_mode != _TRAIN_MODE:
    return apply_remat(model, None)
  return apply_remat(model, resolve_policy(config))


def _jaxprs_in(value: Any) -> Iterator[jex.core.Jaxpr]:
  """Yields nested jaxprs inside equation parameters (`scan`, `remat2`, `pjit`, `shard_map`, ...)."""
  if isinstance(value, jex.core.ClosedJaxpr):
    yield value.jaxpr
  elif isinstance(value, jex.core.Jaxpr):
    yield value
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _jaxprs_in(item)
  elif isinstance(value, dict):
    for item in value.values():
      yield from _jaxprs_in(item)


def _names_in(jaxpr: jex.core.Jaxpr) -> Iterator[str]:
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "name":
      yield eqn.params["name"]
    for value in eqn.params.values():
      for sub in _jaxprs_in(value):
        yield from _names_in(sub)


def checkpoint_names(model: nnx.Module, *args: Any, **kwargs: Any) -> frozenset[str]:
  """Traces one forward pass `model(*args, **kwargs)` and returns all emitted `checkpoint_name` tags."""
  graphdef, state = nnx.split(model)
  traced_kw = {k: v for k, v in kwargs.items() if isinstance(v, (jax.ShapeDtypeStruct, jax.Array))}
  static_kw = {k: v for k, v in kwargs.items() if k not in traced_kw}

  def forward(state_in, pos_args, dyn_kw):
    return nnx.merge(graphdef, state_in)(*pos_args, **dyn_kw, **static_kw)

  return frozenset(_names_in(jax.make_jaxpr(forward)(state, args, traced_kw).jaxpr))


def check_tags(
    model: nnx.Module,
    config: Any,
    *example_args: Any,
    **example_kwargs: Any,
) -> frozenset[str]:
  """Raises `ValueError` if `config.remat_policy` targets tags absent from `model`'s forward trace."""
  remat = resolve_policy(config)
  if remat is None or remat.names is None:
    return frozenset()
  found = checkpoint_names(model, *example_args, **example_kwargs)
  if not found & set(remat.names):
    raise ValueError(
        f"remat_policy={config.remat_policy!r} saves or offloads {sorted(remat.names)}, "
        f"but the model tags {sorted(found)}; it would save nothing. "
        "Tag the activations with those names or choose another policy."
    )
  return found
