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

"""Quantized training via qwix, applied after the model is built.

qwix intercepts `dot_general` / `einsum` / `ragged_dot` while the model runs
and quantizes them by module path, so model files don't change and params stay
`nnx.Param` with the same paths.

We quantize the NNX projection layers (`nnx.Linear`, `nnx.LinearGeneral`,
`nnx.Einsum`) under `decoder/layers`, plus MoE routers unless
`quantize_router_proj` is off (as in legacy). The rule lists their exact paths:
a prefix like `decoder/layers.*` would also catch einsums a block writes
inline, e.g. the attention scores. The embedding, LM head and norms aren't
quantized, same as legacy.

`maybe_quantize` raises instead of silently skipping: on weights under the
stack it can't quantize (e.g. a hand-rolled `DenseGeneral`), on legacy knobs m3
doesn't implement, and if any planned projection does no quantized matmul.
Scales are dynamic absmax.
"""

from collections.abc import Iterator
import collections
import dataclasses
import re
from typing import Any

from absl import logging
from flax import nnx
import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np
import qwix

from maxtext.m3.infra import scan as m3_scan

# NNX layers that do exactly one weight matmul.
_PROJECTIONS = (nnx.Linear, nnx.LinearGeneral, nnx.Einsum)
# The decoder stack (`layers/<i>`, `layers_<i>`, or `layers` when scanned) and everything under it.
_LAYER_STACK = re.compile(r"decoder/layers(_\d+)?(/|$)")
# MoE router attribute names (HF and legacy both use `gate`).
_ROUTERS = frozenset({"gate"})
# `jax.lax.ragged_dot` binds `ragged_dot_general`.
_MATMULS = frozenset({"dot_general", "ragged_dot_general"})
# Legacy knobs m3 doesn't implement, with their "off" value. Setting one raises.
_UNSUPPORTED_WHEN_SET = {
    "quantize_logits_proj": False,
    "quantize_mtp": False,
    "sa_qproj_bwd_bf16": False,
    "qwix_bwd_weight_grad_tile_count": 1,
    "weight_quantization_calibration_method": "absmax",
    "act_quantization_calibration_method": "absmax",
    "bwd_quantization_calibration_method": "absmax",
    "drhs_grad_quantization_calibration_method": None,
    "weight_sparsity_n": None,
    "weight_sparsity_m": None,
    "use_batch_split_schedule": False,
    "weight_block_size": None,
    "unquantized_modules": [],
}


def unsupported_settings(config) -> dict[str, Any]:
  """Legacy knobs `config` turns on that m3 doesn't implement. Missing fields count as off."""
  return {
      field: getattr(config, field)
      for field, off in _UNSUPPORTED_WHEN_SET.items()
      if getattr(config, field, off) not in (off, None)
  }


@dataclasses.dataclass(frozen=True)
class Recipe:
  """Dtypes of one quantized-training recipe."""

  forward: Any  # weights and activations
  gradient: Any  # incoming gradient in the backward matmuls


RECIPES: dict[str, Recipe] = {
    "int8": Recipe(jnp.int8, jnp.int8),
    "fp8": Recipe(jnp.float8_e4m3fn, jnp.float8_e4m3fn),  # same as legacy qwix
    "fp8_e4m3": Recipe(jnp.float8_e4m3fn, jnp.float8_e4m3fn),
    "fp8_e5m2": Recipe(jnp.float8_e5m2, jnp.float8_e5m2),
    "fp8_full": Recipe(jnp.float8_e4m3fn, jnp.float8_e5m2),
}


@dataclasses.dataclass(frozen=True)
class Plan:
  """Paths to quantize, routers left alone, and weights we can't quantize."""

  quantized: tuple[str, ...]
  routers: tuple[str, ...]
  uncovered: tuple[str, ...]

  def describe(self) -> str:
    leaves = sorted({path.rsplit("/", 1)[-1] for path in self.quantized})
    text = f"{len(self.quantized)} projections under decoder/layers ({', '.join(leaves)})"
    if self.routers:
      text += f"; {len(self.routers)} routers kept in full precision"
    return text


def recipe(name: str) -> Recipe:
  """The recipe for `quantization=name`."""
  if name not in RECIPES:
    raise ValueError(f"m3 supports quantization in {sorted(RECIPES)}; got {name!r}.")
  return RECIPES[name]


def _owns_weights(module: nnx.Module, layer_axes: int = 0) -> bool:
  """Whether `module` directly holds a 2D+ param (a matmul weight, not a norm scale).

  `layer_axes` discounts the layer axis scan adds to every param.
  """
  return any(isinstance(value, nnx.Param) and value[...].ndim - layer_axes >= 2 for value in vars(module).values())


def plan(model: nnx.Module, quantize_routers: bool = True) -> Plan:
  """Splits the modules under `decoder/layers` into quantized, routers left alone, and uncovered.

  Uncovered are modules holding weights that aren't NNX projections, e.g. a
  hand-rolled `DenseGeneral` or router. Call after scan.
  """
  modules = [("/".join(map(str, path)), path, module) for path, module in nnx.iter_modules(model)]
  scanned = tuple(name + "/" for name, _, module in modules if isinstance(module, m3_scan.ScannedLayers))
  quantized, routers, uncovered = [], [], []
  for name, path, module in modules:
    if not _LAYER_STACK.match(name):
      continue
    if not quantize_routers and _ROUTERS.intersection(map(str, path)):
      if path[-1] in _ROUTERS:
        routers.append(name)
      continue  # A router, or a layer inside one (e.g. `gate/linear`).
    if isinstance(module, _PROJECTIONS):
      quantized.append(name)
    elif _owns_weights(module, layer_axes=int(isinstance(module, m3_scan.ScannedLayers) or name.startswith(scanned))):
      uncovered.append(name)
  return Plan(tuple(quantized), tuple(routers), tuple(uncovered))


def rules(the_plan: Plan, the_recipe: Recipe) -> list[qwix.QtRule]:
  """One qwix rule for exactly the planned paths (qwix uses `re.fullmatch`, so parents never match)."""
  return [
      qwix.QtRule(
          module_path="|".join(map(re.escape, the_plan.quantized)),
          op_names=("dot_general", "einsum"),
          weight_qtype=the_recipe.forward,
          act_qtype=the_recipe.forward,
          bwd_qtype=the_recipe.gradient,
      )
  ]


def example_inputs(config) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
  """Shapes of `(tokens, positions)` for one training step; only used for tracing."""
  shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  return jax.ShapeDtypeStruct(shape, jnp.int32), jax.ShapeDtypeStruct(shape, jnp.int32)


def _jaxprs_in(value: Any) -> Iterator[jex.core.Jaxpr]:
  """Jaxprs inside an equation param (scan, checkpoint, pjit, custom_vjp, ...)."""
  if isinstance(value, jex.core.ClosedJaxpr):
    yield value.jaxpr
  elif isinstance(value, jex.core.Jaxpr):
    yield value
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _jaxprs_in(item)


def _eqns(jaxpr: jex.core.Jaxpr) -> Iterator[jex.core.JaxprEqn]:
  for eqn in jaxpr.eqns:
    yield eqn
    for value in eqn.params.values():
      for sub in _jaxprs_in(value):
        yield from _eqns(sub)


def matmul_dtypes(fn: Any, *args: Any, **kwargs: Any) -> collections.Counter:
  """Counts the matmuls in a trace of `fn`, keyed by `(lhs dtype, rhs dtype)`, including nested bodies."""
  jaxpr = jax.make_jaxpr(fn)(*args, **kwargs).jaxpr
  return collections.Counter(
      (eqn.invars[0].aval.dtype, eqn.invars[1].aval.dtype) for eqn in _eqns(jaxpr) if eqn.primitive.name in _MATMULS
  )


def _forward_jaxpr(model: nnx.Module, args: tuple, kwargs: dict) -> jex.core.Jaxpr:
  """Jaxpr of `model(*args, **kwargs)`. Array kwargs are traced; the rest (flags) stay static."""
  graphdef, state = nnx.split(model)
  traced = {k: v for k, v in kwargs.items() if isinstance(v, (jax.ShapeDtypeStruct, jax.Array, np.ndarray))}
  static = {k: v for k, v in kwargs.items() if k not in traced}

  def forward(state, args, traced):
    return nnx.merge(graphdef, state)(*args, **traced, **static)

  return jax.make_jaxpr(forward)(state, args, traced).jaxpr


def quantized_matmuls(model: nnx.Module, dtype: Any, *args: Any, **kwargs: Any) -> int:
  """Number of matmuls with both operands in `dtype` in one forward pass."""
  dtype = jnp.dtype(dtype)
  return sum(
      eqn.primitive.name in _MATMULS and eqn.invars[0].aval.dtype == dtype == eqn.invars[1].aval.dtype
      for eqn in _eqns(_forward_jaxpr(model, args, kwargs))
  )


_MARKER = "m3_quantization_census_"


def _marked(cls: type, marker: str) -> type:
  """Subclass of `cls` whose `__call__` runs under `jax.named_scope(marker)`."""

  def __call__(self, *args, **kwargs):  # pylint: disable=invalid-name
    with jax.named_scope(marker):
      return cls.__call__(self, *args, **kwargs)

  return type(cls.__name__, (cls,), {"__call__": __call__})


def quantized_matmuls_by_module(
    model: nnx.Module, paths: tuple[str, ...], dtype: Any, *args: Any, **kwargs: Any
) -> dict[str, int]:
  """Like `quantized_matmuls`, but per module in `paths`.

  Traces a clone where each of those modules runs under a marker scope.
  """
  probe = nnx.clone(model)
  modules = {"/".join(map(str, path)): module for path, module in nnx.iter_modules(probe)}
  for i, path in enumerate(paths):
    # `object.__setattr__` because NNX would otherwise treat `__class__` as state.
    object.__setattr__(modules[path], "__class__", _marked(type(modules[path]), f"{_MARKER}{i}"))
  jaxpr = _forward_jaxpr(probe, args, kwargs)
  dtype = jnp.dtype(dtype)
  counts = collections.Counter()
  for eqn, stack in _eqns_with_scope(jaxpr):
    if eqn.primitive.name in _MATMULS and eqn.invars[0].aval.dtype == dtype == eqn.invars[1].aval.dtype:
      for index in re.findall(rf"{_MARKER}(\d+)", stack):
        counts[paths[int(index)]] += 1
  return {path: counts[path] for path in paths}


def _eqns_with_scope(jaxpr: jex.core.Jaxpr, outer: str = "") -> Iterator[tuple[jex.core.JaxprEqn, str]]:
  """Like `_eqns`, plus each equation's full name scope.

  Nested jaxprs have scopes relative to their caller. qwix wraps each quantized
  matmul in a `custom_vjp`, so the marker is on the call, not the inner matmul.
  """
  for eqn in jaxpr.eqns:
    scope = f"{outer}/{eqn.source_info.name_stack}"
    yield eqn, scope
    for value in eqn.params.values():
      for sub in _jaxprs_in(value):
        yield from _eqns_with_scope(sub, scope)


def _variable_paths(model: nnx.Module) -> list[tuple]:
  return [path for path, _ in nnx.to_flat_state(nnx.state(model))]


def maybe_quantize(model: nnx.Module, config, *example_args: Any, **example_kwargs: Any) -> nnx.Module:
  """Returns `model` quantized per `config.quantization` (or unchanged if unset).

  Call after scan and before creating the optimizer. `example_args` are shapes
  from `example_inputs`, used only for tracing.
  """
  if not config.quantization:
    return model
  # pyconfig passes a `str` Enum.
  name = getattr(config.quantization, "value", config.quantization)
  the_recipe = recipe(name)
  if not getattr(config, "use_qwix_quantization", True):
    raise ValueError(
        f"quantization={name!r} with use_qwix_quantization=false means AQT in legacy MaxText; "
        "m3 only supports qwix. Set use_qwix_quantization=true."
    )
  unsupported = unsupported_settings(config)
  if unsupported:
    raise ValueError(f"quantization={name!r}: m3 doesn't support {unsupported} (see m3/README.md).")
  quantize_routers = bool(getattr(config, "quantize_router_proj", True))
  the_plan = plan(model, quantize_routers=quantize_routers)
  if the_plan.uncovered:
    routers = [path for path in the_plan.uncovered if path.rsplit("/", 1)[-1] in _ROUTERS]
    hint = f" Set quantize_router_proj=false to keep the routers {routers} in full precision." if routers else ""
    raise ValueError(
        f"quantization={name!r}: these modules hold weights but aren't nnx.Linear / LinearGeneral / Einsum, "
        f"so they can't be quantized: {the_plan.uncovered}.{hint}"
    )
  if not the_plan.quantized:
    raise ValueError(f"quantization={name!r}: {type(model).__name__} has no projections under decoder/layers.")
  provider = qwix.QtProvider(rules(the_plan, the_recipe))
  before = _variable_paths(model)
  # Dynamic scales have no state, so skip qwix's eager init pass (a full forward step).
  quantized = qwix.quantize_model(model, provider, *example_args, skip_nnx_init=True, **example_kwargs)
  after = _variable_paths(quantized)
  if before != after:
    raise RuntimeError(f"qwix changed the variable tree: {sorted(set(before) ^ set(after))[:5]} ...")
  per_projection = quantized_matmuls_by_module(
      quantized, the_plan.quantized, the_recipe.forward, *example_args, **example_kwargs
  )
  missing = [path for path, count in per_projection.items() if count == 0]
  if missing:
    raise RuntimeError(
        f"quantization={name!r}: these projections do no {jnp.dtype(the_recipe.forward).name} matmul: {missing}"
    )
  logging.info(
      "m3 quantization=%s: %s; %s forward, %s gradients.",
      name,
      the_plan.describe(),
      jnp.dtype(the_recipe.forward).name,
      jnp.dtype(the_recipe.gradient).name,
  )
  return quantized
