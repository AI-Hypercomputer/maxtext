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

"""Quantized training via qwix, applied after construction.

qwix works by interception: `qwix.quantize_model` wraps the model's `__call__`
so that, while it runs, `jax.lax.dot_general`, `jnp.einsum` and
`jax.lax.ragged_dot` are rewritten to quantize their operands on the fly, under
rules keyed on the path of the module making the call. Nothing in a model file
changes: no `dot_general` injection, no quantized layer classes, parameters
stay `nnx.Param` and keep their paths. That is what keeps rule 4 of
`m3/README.md` intact.

What gets quantized is decided structurally rather than by a path regex.
`plan` walks the model and picks out the NNX projection layers -- `nnx.Linear`,
`nnx.LinearGeneral`, `nnx.Einsum`, each of which performs exactly one weight
matmul -- under the decoder stack, leaving out MoE routers. The qwix rule is
the exact list of those paths. A rule written as a prefix (`decoder/layers.*`)
with `einsum` in its op names also catches every `jnp.einsum` a parent module
writes in its own `__call__`: qwix quantizes a matmul whose operands are both
activations with `act_qtype` on both sides, so attention scores and the MoE
combine would be quantized too. Measured on the draft's rules with int8: 4 of
18 quantized matmuls in a 2-layer Qwen3 were the attention scores. Exact leaf
paths cannot match anything a parent does.

The embedding, the LM head, norms and vision towers are outside the decoder
stack and are never matched. Legacy keeps the same set in full precision.

The plan is also checked for what it would miss. A module under the stack
that holds a weight itself without being one of those layers -- Qwen3-VL's
hand-rolled `DenseGeneral`, stacked MoE expert weights -- would otherwise be
left in full precision with nothing to show for it; `maybe_quantize` refuses
such a model and names the modules. Routers are the one exemption.

`quantization` names a recipe (`RECIPES`): one dtype for weights and
activations in the forward pass, one for the gradients in the backward pass.
`fp8` is `float8_e4m3fn` forward and `float8_e5m2` backward -- the design doc's
rule and legacy's `fp8_full`, which is accepted as a spelling of the same
thing. Calibration is absmax with dynamic scales. No other legacy quantization
field is read -- and a config that turns one of them on (`quantize_logits_proj`,
a calibration method, weight-gradient tiling, sparsity, ...) is refused, since
it would otherwise train with a different recipe than it names.

`maybe_quantize` checks its own work. qwix's eager initialization pass is
skipped (there is nothing to initialize for dynamic scales); instead one
forward pass is traced abstractly, and the trace must contain at least one
matmul in the recipe's forward dtype per planned projection. A model whose
decoder blocks do not live under `decoder/layers` plans nothing and is
refused rather than trained unquantized.
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
import qwix

from maxtext.m3.infra import scan as m3_scan

# The NNX layers that perform one weight matmul each.
_PROJECTIONS = (nnx.Linear, nnx.LinearGeneral, nnx.Einsum)
# Module paths of the decoder stack, in m3's layouts: `decoder/layers/<i>/...`
# (a container), `decoder/layers_<i>/...` (flat attributes) and
# `decoder/layers/...` (scanned). The same prefix m3 guarantees for checkpoints.
_LAYER_STACK = re.compile(r"decoder/layers(_\d+)?/")
# Attribute names of projections that route rather than project. `gate` is the
# MoE router's name in Hugging Face checkpoints and in legacy MaxText; routing
# decides which experts run, so it stays in full precision.
_ROUTERS = frozenset({"gate"})
# Primitives a quantized projection lowers to.
_MATMULS = frozenset({"dot_general", "ragged_dot"})
# Legacy quantization knobs m3 does not implement, with the value that means
# "off". A config that turns one on would otherwise train with a different
# recipe than it names, so `maybe_quantize` refuses it. Absent fields (a slim
# m3 config) count as off. `quantize_router_proj` is not here: legacy's
# default is to quantize the router and m3's is not, a documented divergence
# rather than a knob; `use_qwix_quantization` is moot, m3 is qwix only.
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
}


def unsupported_settings(config) -> dict[str, Any]:
  """The legacy quantization knobs `config` turns on that m3 does not implement, with their values."""
  return {
      field: getattr(config, field)
      for field, off in _UNSUPPORTED_WHEN_SET.items()
      if getattr(config, field, off) not in (off, None)
  }


@dataclasses.dataclass(frozen=True)
class Recipe:
  """Dtypes of one quantized-training recipe."""

  forward: Any  # weights and activations
  gradient: Any  # the incoming gradient in both backward matmuls


RECIPES: dict[str, Recipe] = {
    "int8": Recipe(jnp.int8, jnp.int8),
    "fp8": Recipe(jnp.float8_e4m3fn, jnp.float8_e5m2),
    "fp8_full": Recipe(jnp.float8_e4m3fn, jnp.float8_e5m2),  # legacy's name for `fp8`
    "fp8_e4m3": Recipe(jnp.float8_e4m3fn, jnp.float8_e4m3fn),  # legacy's bare qwix `fp8`
    "fp8_e5m2": Recipe(jnp.float8_e5m2, jnp.float8_e5m2),
}


@dataclasses.dataclass(frozen=True)
class Plan:
  """Module paths of the projections to quantize, the routers kept in full precision, and weight owners that are neither."""

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
  """The recipe `config.quantization=name` selects. Raises for a name m3 does not define."""
  if name not in RECIPES:
    raise ValueError(f"m3 supports quantization in {sorted(RECIPES)}; got {name!r}.")
  return RECIPES[name]


def _owns_weights(module: nnx.Module, layer_axes: int = 0) -> bool:
  """Whether `module` itself holds a parameter with two or more axes -- a matmul operand, not a norm scale or bias.

  `layer_axes` is the number of stacking axes scan has added to every
  parameter of the module; under a scanned stack a norm scale is
  `(layers, features)` and must still count as one-dimensional.
  """
  return any(isinstance(value, nnx.Param) and value[...].ndim - layer_axes >= 2 for value in vars(module).values())


def plan(model: nnx.Module) -> Plan:
  """Finds the projection layers under the decoder stack of `model`.

  Also finds what the plan would miss: a module under the stack that holds a
  weight itself without being one of the NNX projection layers (a hand-rolled
  `DenseGeneral`, stacked MoE expert weights). Those are reported as
  `uncovered` rather than left in full precision without a word.

  Runs after scan (`train/loop.py::setup_model`): a scanned stack is one
  `ScannedLayers` whose parameters carry a layer axis, which the weight test
  discounts.
  """
  modules = [("/".join(map(str, path)), path, module) for path, module in nnx.iter_modules(model)]
  scanned = tuple(name + "/" for name, _, module in modules if isinstance(module, m3_scan.ScannedLayers))
  quantized, routers, uncovered = [], [], []
  for name, path, module in modules:
    if not _LAYER_STACK.match(name):
      continue
    if path[-1] in _ROUTERS:
      routers.append(name)
    elif isinstance(module, _PROJECTIONS):
      quantized.append(name)
    elif _owns_weights(module, layer_axes=int(isinstance(module, m3_scan.ScannedLayers) or name.startswith(scanned))):
      uncovered.append(name)
  return Plan(tuple(quantized), tuple(routers), tuple(uncovered))


def rules(the_plan: Plan, the_recipe: Recipe) -> list[qwix.QtRule]:
  """One qwix rule matching exactly the planned module paths.

  `op_names` lists `einsum` for `nnx.Einsum` and `dot_general` for the two
  Linear layers. qwix matches with `re.fullmatch`, so a leaf's path cannot
  match the ops of its parent.
  """
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
  """`(tokens, positions)` of the training shape, as shapes only: the traces here never run."""
  shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  return jax.ShapeDtypeStruct(shape, jnp.int32), jax.ShapeDtypeStruct(shape, jnp.int32)


def _jaxprs_in(value: Any) -> Iterator[jex.core.Jaxpr]:
  """The jaxprs held by one equation parameter: `scan`, `checkpoint`, `pjit`, `shard_map`, ..."""
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
  """Counts the matmuls `fn(*args, **kwargs)` performs, keyed by `(lhs dtype, rhs dtype)`.

  Traces the call (no compilation, no execution), looking inside `jax.checkpoint`,
  `jax.lax.scan` and `jit` bodies, so it sees the same ops whether or not remat
  and scan have been applied.
  """
  jaxpr = jax.make_jaxpr(fn)(*args, **kwargs).jaxpr
  return collections.Counter(
      (eqn.invars[0].aval.dtype, eqn.invars[1].aval.dtype) for eqn in _eqns(jaxpr) if eqn.primitive.name in _MATMULS
  )


def quantized_matmuls(model: nnx.Module, dtype: Any, *args: Any, **kwargs: Any) -> int:
  """How many matmuls with both operands in `dtype` one forward pass `model(*args, **kwargs)` performs."""
  graphdef, state = nnx.split(model)

  def forward(state, *args):
    return nnx.merge(graphdef, state)(*args, **kwargs)

  dtype = jnp.dtype(dtype)
  return matmul_dtypes(forward, state, *args)[(dtype, dtype)]


def _variable_paths(model: nnx.Module) -> list[tuple]:
  return [path for path, _ in nnx.to_flat_state(nnx.state(model))]


def maybe_quantize(model: nnx.Module, config, *example_args: Any, **example_kwargs: Any) -> nnx.Module:
  """Returns `model` quantized per `config.quantization`, or `model` itself when that is unset.

  `example_args` / `example_kwargs` are the shapes the model will be called
  with (`example_inputs`); they are only traced. Must run after transforms that
  restructure the model (scan), since qwix records module paths, and before the
  optimizer, which must see the returned model. Raises if the decoder stack
  holds weights the plan cannot cover, if nothing would be quantized, if the
  traced forward pass has fewer quantized matmuls than planned projections, or
  if qwix changed the variable tree -- every path-keyed consumer downstream
  (checkpointing, HF mappings) would otherwise break quietly.
  """
  if not config.quantization:
    return model
  # pyconfig hands over a `str` Enum; its value is the name users wrote.
  name = getattr(config.quantization, "value", config.quantization)
  the_recipe = recipe(name)
  unsupported = unsupported_settings(config)
  if unsupported:
    raise ValueError(
        f"quantization={name!r}: m3 does not implement {unsupported}. It reads `quantization` only; "
        "calibration is absmax with dynamic scales, weight gradients are not tiled, and the LM head, "
        "MTP block and sparsity are not quantized (see m3/README.md)."
    )
  the_plan = plan(model)
  if the_plan.uncovered:
    raise ValueError(
        f"quantization={name!r}: these modules under decoder/layers hold weights but are not NNX "
        f"projection layers, so they would stay in full precision: {the_plan.uncovered}. m3 projections are "
        f"nnx.Linear / nnx.LinearGeneral / nnx.Einsum (README rule 1)."
    )
  if not the_plan.quantized:
    raise ValueError(
        f"quantization={name!r}: {type(model).__name__} has no projection layers under "
        f"decoder/layers, so nothing would be quantized. m3 models keep their decoder blocks there."
    )
  provider = qwix.QtProvider(rules(the_plan, the_recipe))
  before = _variable_paths(model)
  # `skip_nnx_init`: dynamic scales have no state to initialize, and the eager
  # forward pass qwix would otherwise run costs a full step at setup.
  quantized = qwix.quantize_model(model, provider, *example_args, skip_nnx_init=True, **example_kwargs)
  after = _variable_paths(quantized)
  if before != after:
    raise RuntimeError(f"qwix changed the variable tree: {sorted(set(before) ^ set(after))[:5]} ...")
  found = quantized_matmuls(quantized, the_recipe.forward, *example_args, **example_kwargs)
  if found < len(the_plan.quantized):
    raise RuntimeError(
        f"quantization={name!r}: planned {len(the_plan.quantized)} projections but the forward "
        f"pass has {found} matmuls in {jnp.dtype(the_recipe.forward).name}. Projections: {the_plan.quantized}"
    )
  logging.info(
      "m3 quantization=%s: %s; %s forward, %s gradients.",
      name,
      the_plan.describe(),
      jnp.dtype(the_recipe.forward).name,
      jnp.dtype(the_recipe.gradient).name,
  )
  return quantized
