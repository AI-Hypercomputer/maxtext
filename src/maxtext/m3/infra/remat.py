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
import math
from typing import Any, Optional
import warnings

from flax import nnx
import jax
from jax import ad_checkpoint
from jax import checkpoint_policies as cp
import jax.extend as jex
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import scan as m3_scan

_TRAIN_MODE = "train"

_PROJECTIONS = ("query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj")
_OFFLOAD_PROJECTIONS = ("query_proj", "value_proj", "key_proj", "kv_proj")
_MINIMAL = _PROJECTIONS + ("out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo")
_MINIMAL_WITH_CONTEXT = _MINIMAL + ("context",)
_MINIMAL_WITH_QUANTIZATION = _MINIMAL + ("quantization",)
_MINIMAL_WITH_CONTEXT_AND_QUANTIZATION = _MINIMAL_WITH_CONTEXT + ("quantization",)
_DOT_EXCEPT_MLP = _PROJECTIONS + ("out_proj",)
_DOT_WITH_CONTEXT_EXCEPT_MLP = _PROJECTIONS + ("context", "out_proj")
_DOT_EXCEPT_MLPWI = _PROJECTIONS + ("out_proj", "mlpwo")
_HIDDEN_OFFLOAD = ("hidden_states", "self_attn", "cross_attn")


@dataclasses.dataclass(frozen=True)
class _PolicySpec:
  """Single-source-of-truth specification for one static `remat_policy` preset."""

  build: Callable[[], Callable[..., Any]]
  names: tuple[str, ...] | None


def _offload_names_policy(names: Sequence[str]) -> Callable[..., Any]:
  return cp.save_and_offload_only_these_names(
      names_which_can_be_saved=[],
      names_which_can_be_offloaded=list(names),
      offload_src="device",
      offload_dst="pinned_host",
  )


# Single policy table shared by `remat_policy`, `resolve_policy`, `check_tags`, and models.
_STATIC_POLICY_SPECS: dict[str, _PolicySpec] = {
    "full": _PolicySpec(lambda: cp.nothing_saveable, None),
    "minimal": _PolicySpec(lambda: cp.save_only_these_names(*_MINIMAL), _MINIMAL),
    "minimal_with_context": _PolicySpec(lambda: cp.save_only_these_names(*_MINIMAL_WITH_CONTEXT), _MINIMAL_WITH_CONTEXT),
    # Legacy MaxText maps `minimal_flash` to `minimal_with_context` so Splash `context` is saved.
    "minimal_flash": _PolicySpec(lambda: cp.save_only_these_names(*_MINIMAL_WITH_CONTEXT), _MINIMAL_WITH_CONTEXT),
    "minimal_with_quantization": _PolicySpec(lambda: cp.save_only_these_names(*_MINIMAL_WITH_QUANTIZATION), _MINIMAL),
    "minimal_with_context_and_quantization": _PolicySpec(
        lambda: cp.save_only_these_names(*_MINIMAL_WITH_CONTEXT_AND_QUANTIZATION), _MINIMAL_WITH_CONTEXT
    ),
    "save_dot_with_context_except_mlp": _PolicySpec(
        lambda: cp.save_only_these_names(*_DOT_WITH_CONTEXT_EXCEPT_MLP), _DOT_WITH_CONTEXT_EXCEPT_MLP
    ),
    "save_dot_except_mlpwi": _PolicySpec(lambda: cp.save_only_these_names(*_DOT_EXCEPT_MLPWI), _DOT_EXCEPT_MLPWI),
    "save_dot_except_mlp": _PolicySpec(lambda: cp.save_only_these_names(*_DOT_EXCEPT_MLP), _DOT_EXCEPT_MLP),
    "save_qkv_proj": _PolicySpec(lambda: cp.save_only_these_names(*_PROJECTIONS), _PROJECTIONS),
    "save_out_proj": _PolicySpec(lambda: cp.save_only_these_names("out_proj"), ("out_proj",)),
    "qkv_proj_offloaded": _PolicySpec(lambda: _offload_names_policy(_OFFLOAD_PROJECTIONS), _PROJECTIONS),
    "minimal_offloaded": _PolicySpec(lambda: _offload_names_policy(_MINIMAL), _MINIMAL),
    "matmul_without_batch": _PolicySpec(lambda: cp.checkpoint_dots_with_no_batch_dims, None),
    "offload_matmul_without_batch": _PolicySpec(
        lambda: cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host"), None
    ),
    "hidden_state_with_offload": _PolicySpec(lambda: _offload_names_policy(_HIDDEN_OFFLOAD), _HIDDEN_OFFLOAD),
}

# Supported `remat_policy` config values (case-insensitive).
REMAT_POLICIES: tuple[str, ...] = ("none", *_STATIC_POLICY_SPECS.keys(), "custom")


@dataclasses.dataclass(frozen=True)
class Remat:
  """Resolved checkpointing policy and expected activation tag names."""

  policy: Callable[..., Any] | None = cp.nothing_saveable
  names: tuple[str, ...] | None = None
  is_custom: bool = False
  saved_names: tuple[str, ...] = ()
  offloaded_names: tuple[str, ...] = ()


def policy_callable(remat: Remat | Callable[..., Any] | None) -> Callable[..., Any] | None:
  """Extracts the `jax.checkpoint` policy callable from `remat`, or `None` when remat is disabled."""
  if remat is None:
    return None
  if isinstance(remat, Remat):
    return cp.nothing_saveable if remat.policy is None else remat.policy
  if callable(remat):
    return remat
  raise TypeError(f"Expected Remat, callable policy, or None; got {remat!r}.")


def scan_safe_policy(remat: Remat | Callable[..., Any] | None) -> Callable[..., Any] | None:
  """Returns a policy callable keeping offloaded residuals on device across trip-count-one scans."""
  pol = policy_callable(remat)
  if pol is None or pol is cp.nothing_saveable:
    return pol

  def _safe_policy(prim: Any, *args: Any, **params: Any) -> Any:
    decision = pol(prim, *args, **params)
    if isinstance(decision, ad_checkpoint.Offloadable):
      return ad_checkpoint.Saveable
    return decision

  return _safe_policy


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
  if key == "none":
    return None
  if key == "custom":
    saved = list(names_which_can_be_saved or [])
    offloaded = list(names_which_can_be_offloaded or [])
    return cp.save_and_offload_only_these_names(
        names_which_can_be_saved=saved,
        names_which_can_be_offloaded=offloaded,
        offload_src="device",
        offload_dst="pinned_host",
    )
  spec = _STATIC_POLICY_SPECS.get(key)
  if spec is not None:
    return spec.build()
  raise ValueError(f"Unknown remat_policy {name!r}; expected one of {REMAT_POLICIES}.")


def resolve_policy(config: Any) -> Remat | None:
  """Resolves `config.remat_policy` (and custom save/offload lists) into a `Remat` or `None`."""
  raw = getattr(config, "remat_policy", "none")
  if raw is None or raw == "":
    return None
  if callable(raw):
    return Remat(policy=raw, names=None)
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
  if key == "custom":
    return Remat(
        policy=pol,
        names=on_device + to_offload,
        is_custom=True,
        saved_names=on_device,
        offloaded_names=to_offload,
    )
  return Remat(policy=pol, names=_STATIC_POLICY_SPECS[key].names)


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


def _call_remat_layer(
    f: Callable[..., Any],
    policy: Callable[..., Any],
    args: tuple[Any, ...],
    layer_kwargs: dict[str, Any],
) -> Any:
  """Executes one callable `f` under `jax.checkpoint(..., policy=policy, prevent_cse=True)`."""
  if isinstance(f, m3_scan.ScannedLayers):
    # `ScannedLayers` already checkpoints each layer inside its scan body.
    return f(*args, **layer_kwargs)
  if isinstance(f, nnx.Module):
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
    return output
  traced_kw = {k: v for k, v in layer_kwargs.items() if _is_traced_value(v)}
  static_kw = {k: v for k, v in layer_kwargs.items() if k not in traced_kw}

  def _run_fn(pos_args, dyn_kw, fn=f, st_kw=static_kw):
    return fn(*pos_args, **dyn_kw, **st_kw)

  return jax.checkpoint(_run_fn, policy=policy, prevent_cse=True)(args, traced_kw)


class RematSequential(nnx.Sequential):
  """Path-transparent `nnx.Sequential` subclass applying per-layer `jax.checkpoint`."""

  def __init__(self, *fns: Callable[..., Any], remat: Remat | None = None):
    super().__init__(*fns)
    self.remat = remat

  def __len__(self) -> int:
    return len(self.layers)

  def __iter__(self) -> Iterator[Callable[..., Any]]:
    policy = policy_callable(self.remat)
    if policy is None:
      yield from self.layers
      return
    for f in self.layers:
      yield lambda *a, _f=f, _pol=policy, **kw: _call_remat_layer(_f, _pol, a, kw)

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

      output = _call_remat_layer(f, policy, args, layer_kwargs)

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
  if remat is None:
    return frozenset()
  if remat.is_custom:
    if not remat.saved_names and not remat.offloaded_names:
      warnings.warn(
          "remat_policy='custom' has empty save and offload lists; treating as 'full'.",
          UserWarning,
          stacklevel=2,
      )
      return frozenset()
    found = checkpoint_names(model, *example_args, **example_kwargs)
    missing_saved = set(remat.saved_names) - found
    missing_offloaded = set(remat.offloaded_names) - found
    if "decoder_layer_input" in missing_saved:
      warnings.warn(
          "remat_policy='custom' lists 'decoder_layer_input' on device, which this model does not tag; "
          "per-layer checkpointing already saves layer inputs on device "
          "(treating unmatched 'decoder_layer_input' as 'full').",
          UserWarning,
          stacklevel=2,
      )
      missing_saved.discard("decoder_layer_input")
    missing = missing_saved | missing_offloaded
    if missing:
      raise ValueError(
          f"remat_policy='custom' requests unknown checkpoint tag(s) {sorted(missing)}; "
          f"the model tags {sorted(found)}. "
          "Fix the tensor names or tag those activations in the model."
      )
    return found
  if remat.names is None:
    return frozenset()
  found = checkpoint_names(model, *example_args, **example_kwargs)
  if not found & set(remat.names):
    raise ValueError(
        f"remat_policy={config.remat_policy!r} saves or offloads {sorted(remat.names)}, "
        f"but the model tags {sorted(found)}; it would save nothing. "
        "Tag the activations with those names or choose another policy."
    )
  return found


def check_model_tags(
    model: nnx.Module,
    config: Any,
    *,
    mesh: Any = None,
    sharding: m3_sharding.Sharding | None = None,
) -> frozenset[str]:
  """Builds abstract token inputs for `model` and validates `config.remat_policy` tags."""
  remat = resolve_policy(config)
  if remat is None or (not remat.is_custom and remat.names is None):
    return frozenset()
  seq_len = int(getattr(config, "max_target_length", 0) or 128)
  ambient = jax.sharding.get_abstract_mesh()
  in_sharding = None
  min_batch = 1
  if mesh is not None and not getattr(mesh, "empty", False):
    sharding_obj = sharding or m3_sharding.sharding_from_config(config)
    raw_spec = sharding_obj("input_tokens", ("batch", "length"), m3_sharding.TensorType.ACTIVATION)
    spec = m3_sharding.project_spec_to_mesh(raw_spec, mesh)
    for dim_axes in spec:
      if dim_axes is None:
        continue
      axes_tup = (dim_axes,) if isinstance(dim_axes, str) else tuple(dim_axes)
      min_batch = max(min_batch, math.prod(int(mesh.shape[a]) for a in axes_tup))
    target_mesh = ambient if not ambient.empty else mesh
    try:
      in_sharding = NamedSharding(mesh, spec)
    except (TypeError, ValueError):
      in_sharding = NamedSharding(target_mesh, spec)
  cfg_batch = int(
      getattr(config, "global_batch_size_to_train_on", 0) or getattr(config, "micro_batch_size_to_train_on", 0) or 0
  )
  batch = cfg_batch if (cfg_batch >= min_batch and cfg_batch % min_batch == 0) else min_batch
  in_shape = (batch, seq_len)
  toks = jax.ShapeDtypeStruct(in_shape, jnp.int32, sharding=in_sharding)
  pos = jax.ShapeDtypeStruct(in_shape, jnp.int32, sharding=in_sharding)
  return check_tags(model, config, toks, pos)
