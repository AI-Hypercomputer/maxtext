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

"""Model-trainer boundary contracts, registered nondiff variables, and invariant verification."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


class QuantStat(nnx.Variable):  # pylint: disable=abstract-method
  """Registered non-differentiable quantization calibration statistic."""


class RouterBias(nnx.Variable):  # pylint: disable=abstract-method
  """Registered non-differentiable auxiliary-loss-free MoE router bias."""


class StaticBuffer(nnx.Variable):  # pylint: disable=abstract-method
  """Registered non-differentiable precomputed buffer (e.g., RoPE tables)."""


REGISTERED_NONDIFF_VARIABLE_TYPES: tuple[type[nnx.Variable], ...] = (
    QuantStat,
    RouterBias,
    StaticBuffer,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CausalLMOutput:
  """Explicit PyTree output returned by m3 causal language models."""

  logits: jax.Array | None
  hidden_states: jax.Array | None = None
  aux_loss: jax.Array | None = None
  router_token_counts: jax.Array | None = None
  aux_metrics: dict[str, jax.Array] = dataclasses.field(default_factory=dict)


class M3ContractViolationError(TypeError):
  """Raised when an m3 model violates the pure-NNX model-trainer contract."""


def _format_path(path: tuple[Any, ...]) -> str:
  return "/".join(str(part) for part in path)


def _check_decoder_layer_container(model: nnx.Module) -> None:
  """Verifies decoder layers use `nnx.Sequential` or `ScannedLayers`, never flat `layers_<i>` attributes."""
  decoder = getattr(model, "decoder", None)
  if decoder is None:
    return
  flat_layer_attrs = sorted(name for name in vars(decoder) if name.startswith("layers_") and name[7:].isdecimal())
  if flat_layer_attrs:
    raise M3ContractViolationError(
        f"Decoder {type(decoder).__name__} defines flat layer attributes {flat_layer_attrs[:4]}; "
        "m3 decoders must store repeated blocks in `self.layers = nnx.Sequential(...)`."
    )
  if hasattr(decoder, "layers"):
    from maxtext.m3.infra.scan import ScannedLayers  # pylint: disable=import-outside-toplevel

    layers_container = decoder.layers
    if not isinstance(layers_container, (nnx.Sequential, ScannedLayers)):
      raise M3ContractViolationError(
          f"Decoder `{type(decoder).__name__}.layers` must be `nnx.Sequential` or `ScannedLayers`, "
          f"got {type(layers_container).__name__}."
      )


def _validate_state_variables(state: nnx.State, mesh: Mesh) -> None:
  """Validates variable types and `out_sharding` metadata across `state`."""
  allowed_types = (nnx.Param, *REGISTERED_NONDIFF_VARIABLE_TYPES)
  flat = nnx.to_flat_state(state)
  for path, var in flat:
    path_str = _format_path(path)
    if isinstance(var, nnx.Intermediate):
      raise M3ContractViolationError(
          f"Variable at {path_str!r} is `nnx.Intermediate`; m3 models must return auxiliary values "
          "explicitly in `CausalLMOutput` instead of mutating module state via `sow()`."
      )
    if not isinstance(var, allowed_types):
      allowed_names = ", ".join(cls.__name__ for cls in allowed_types)
      raise M3ContractViolationError(
          f"Variable at {path_str!r} has unregistered type {type(var).__name__}; " f"expected one of ({allowed_names})."
      )
    metadata = var.get_metadata()
    if "sharding" in metadata:
      raise M3ContractViolationError(
          f"Variable at {path_str!r} uses deprecated metadata key 'sharding'; use 'out_sharding' instead."
      )
    if isinstance(var, nnx.Param):
      out_sharding = metadata.get("out_sharding")
      if not isinstance(out_sharding, P):
        raise M3ContractViolationError(
            f"Parameter at {path_str!r} is missing `PartitionSpec` metadata under 'out_sharding'; "
            f"got {out_sharding!r}."
        )
    raw_val = var.get_value()
    val_sharding = getattr(raw_val, "sharding", None)
    if isinstance(val_sharding, NamedSharding) and val_sharding.mesh != mesh:
      raise M3ContractViolationError(f"Variable at {path_str!r} is bound to mesh {val_sharding.mesh}, expected {mesh}.")


def verify_m3_model_invariants(
    model_fn: Callable[[], nnx.Module],
    mesh: Mesh,
    sample_spec: Mapping[str, jax.ShapeDtypeStruct] | None = None,
    *,
    batch_size: int = 2,
    seq_len: int = 8,
) -> CausalLMOutput:
  """Verifies all m3 model-trainer contract invariants via abstract `jax.eval_shape` (< 200ms on CPU).

  Checks:
    1. Ambient mesh includes the `'data'` axis required by Zero-1 and data-parallel reductions.
    2. Every `nnx.Variable` in model state is `nnx.Param` or a registered nondiff variable
       (`QuantStat`, `RouterBias`, `StaticBuffer`), with zero `nnx.Intermediate` mutations.
    3. Every `nnx.Param` carries a `PartitionSpec` under `'out_sharding'`, uses zero deprecated
       `'sharding'` metadata keys, and binds to `mesh`.
    4. Decoder stacks use `nnx.Sequential` or `ScannedLayers`, never flat `layers_<i>` attributes.
    5. Forward execution preserves the exact `nnx.State` treedef, returns `CausalLMOutput` when
       `return_output=True`, and skips dense vocabulary projection (`logits is None`) when
       `return_logits=False`.

  Args:
    model_fn: Zero-argument factory constructing the `nnx.Module` inside `jax.set_mesh(mesh)`.
    mesh: Target JAX `Mesh`.
    sample_spec: Optional abstract batch specification mapping `'inputs'`, `'positions'`, and
      `'segment_ids'` to `jax.ShapeDtypeStruct`.
    batch_size: Default abstract batch size when `sample_spec` is omitted.
    seq_len: Default abstract sequence length when `sample_spec` is omitted.

  Returns:
    Abstract `CausalLMOutput` shape structure from the `return_logits=True` trace.

  Raises:
    M3ContractViolationError: If any contract invariant is violated.
  """
  if "data" not in mesh.axis_names:
    raise M3ContractViolationError(
        f"Mesh axis_names {mesh.axis_names} must contain 'data' for data-parallel / Zero-1 compatibility."
    )

  if sample_spec is None:
    token_spec = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)
    inputs_spec = token_spec
    positions_spec = token_spec
    segment_ids_spec = token_spec
  else:
    inputs_spec = sample_spec["inputs"]
    positions_spec = sample_spec["positions"]
    segment_ids_spec = sample_spec.get("segment_ids", inputs_spec)

  def _trace_invariants(inputs: jax.Array, positions: jax.Array, segment_ids: jax.Array) -> CausalLMOutput:
    model = model_fn()
    _check_decoder_layer_container(model)
    state_before = nnx.state(model)
    _validate_state_variables(state_before, mesh)
    treedef_before = jax.tree_util.tree_structure(state_before)

    out_full = model(
        inputs,
        positions,
        decoder_segment_ids=segment_ids,
        return_output=True,
        return_logits=True,
    )
    if not isinstance(out_full, CausalLMOutput):
      raise M3ContractViolationError(
          f"Model forward with `return_output=True` must return `CausalLMOutput`, got {type(out_full).__name__}."
      )
    if out_full.logits is None:
      raise M3ContractViolationError("`CausalLMOutput.logits` must not be None when `return_logits=True`.")

    state_after_full = nnx.state(model)
    _validate_state_variables(state_after_full, mesh)
    if jax.tree_util.tree_structure(state_after_full) != treedef_before:
      raise M3ContractViolationError(
          "Model forward pass mutated module state PyTree structure; m3 forward passes must be pure."
      )

    out_skip = model(
        inputs,
        positions,
        decoder_segment_ids=segment_ids,
        return_output=True,
        return_logits=False,
    )
    if not isinstance(out_skip, CausalLMOutput):
      raise M3ContractViolationError(
          f"Model forward with `return_logits=False` must return `CausalLMOutput`, got {type(out_skip).__name__}."
      )
    if out_skip.logits is not None:
      raise M3ContractViolationError(
          "`CausalLMOutput.logits` must be None when `return_logits=False` so vocab-tiled training never "
          "materializes dense logits."
      )
    if out_skip.hidden_states is None:
      raise M3ContractViolationError("`CausalLMOutput.hidden_states` must be populated when `return_logits=False`.")

    state_after_skip = nnx.state(model)
    if jax.tree_util.tree_structure(state_after_skip) != treedef_before:
      raise M3ContractViolationError(
          "Model forward pass with `return_logits=False` mutated module state PyTree structure."
      )
    return out_full

  with jax.set_mesh(mesh):
    return jax.eval_shape(_trace_invariants, inputs_spec, positions_spec, segment_ids_spec)
