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

"""Model-independent pipeline parallelism over the `stage` mesh axis, applied after construction.

Replaces the decoder's `nnx.Sequential` of layers with `PipelinedLayers`: the
layers are grouped into stage bodies, the bodies are stacked on a leading
`[stages]` axis (`[repeats, stages]` for a circular pipeline) that is sharded
over the `stage` mesh axis, and each call runs the GPipe (one repeat) or
circular (several repeats) schedule of legacy `layers/pipeline.py`:

  * the batch is split into `num_pipeline_microbatches` microbatches, which
    flow through the stages one pipeline iteration at a time;
  * every iteration runs all stages at once, as one `jax.vmap` over the stage
    axis of the stacked bodies, each stage on its own microbatch;
  * between iterations each stage's output is shifted to the next stage with a
    `ppermute` over `stage` (`shift`), new microbatches enter stage 0 from and
    finished ones return to a per-stage buffer (`state_io`), and a circular
    pipeline with more microbatches than stages parks the last stage's output
    until its next repeat (`circ_storage`);
  * a circular pipeline hands each stage the weights of the repeat it is in,
    so a stage runs layer `s` in repeat 0, `stages + s` in repeat 1, and so on;
  * with `pipeline_fsdp_ag_once` the stage weights are all-gathered over `fsdp`
    once, before the first iteration, instead of once per iteration inside the
    stage bodies (the gradient is reduce-scattered once in the backward pass).

The schedule runs `num_microbatches * num_repeats + (num_stages - 1)`
iterations (twice the bubble with `pipeline_delay_activation_forwarding`),
as a `jax.lax.scan` by default (`scan_pipeline_iterations`), each iteration
rematerialized in the backward pass from its inputs by default
(`set_remat_policy_on_pipeline_iterations`).

Parameter paths retain one stage body's layout under a leading stage axis
(and repeat axis), as legacy pipelined checkpoints do; with `scan_layers` the
body is the stage's layers scanned (`infra.scan`), so the layer axis sits
after them. Models hold their repeated blocks in an `nnx.Sequential` and run
them with `carry = self.layers(carry, **kwargs)` (see `infra.scan`); array
keyword arguments whose leading dimension is the batch (positions, segment
ids) are split into microbatches alongside the carry, anything else is passed
to every stage unchanged.

Nothing here is model specific: the stage bodies are whatever layer class the
model built, and the only mesh axis named is `stage`. The microbatch dimension
of the pipeline's buffers follows the strategy's activation rule for `batch`,
so the stages shard their batch the way the layers inside them expect.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from flax import nnx
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType
from maxtext.m3.infra import scan as m3_scan

# The mesh axis the stages are laid out over (`core.sharding.MESH_AXIS_NAMES`).
STAGE_AXIS = "stage"

# Legacy pipeline knobs this module does not implement, with the value that
# means "off". A config that turns one on asks for a schedule m3 cannot run, so
# it is refused rather than silently run as another one.
_UNSUPPORTED_PIPELINE_FIELDS = {
    # Per-repeat weight prefetch through a two-slot buffer (legacy `NNXCircularPipeline`), and its sub-options.
    "pipeline_fsdp_ag_per_repeat": False,
    "pipeline_bsw_phase_split": False,
    "pipeline_bsw_in_compute_dtype": False,
    "pipeline_unreduced_weight_grads": False,
}


@dataclasses.dataclass(frozen=True)
class PipelineSchedule:
  """The shape of one pipelined forward pass, with legacy's arithmetic for it.

  `num_stages * num_repeats * layers_per_stage` layers are pipelined over
  `num_stages` stages; `num_microbatches` microbatches flow through them.
  Frozen so a `PipelinedLayers` can keep it as a static attribute.
  """

  num_stages: int
  num_repeats: int
  layers_per_stage: int
  num_microbatches: int
  delay_activation_forwarding: bool = False
  fsdp_ag_once: bool = False
  scan_iterations: bool = True
  remat_iterations: bool = True

  def __post_init__(self):
    for name in ("num_stages", "num_repeats", "layers_per_stage", "num_microbatches"):
      if getattr(self, name) < 1:
        raise ValueError(f"PipelineSchedule.{name} must be >= 1; got {getattr(self, name)}.")
    if self.num_microbatches % self.num_stages:
      raise ValueError(
          f"num_pipeline_microbatches ({self.num_microbatches}) must be a multiple of the number of stages "
          f"({self.num_stages})."
      )
    if self.num_microbatches < self.num_stages * self.forwarding_delay:
      raise ValueError(
          f"num_pipeline_microbatches ({self.num_microbatches}) must be at least {self.forwarding_delay} x the number "
          f"of stages ({self.num_stages}) with pipeline_delay_activation_forwarding={self.delay_activation_forwarding}."
      )

  @property
  def num_layers(self) -> int:
    return self.num_stages * self.num_repeats * self.layers_per_stage

  @property
  def forwarding_delay(self) -> int:
    """Iterations between a stage producing an output and the next stage consuming it."""
    return 2 if self.delay_activation_forwarding else 1

  @property
  def microbatches_per_stage(self) -> int:
    return self.num_microbatches // self.num_stages

  @property
  def use_circ_storage(self) -> bool:
    """Whether the last stage's output must be parked before its next repeat (it cannot flow straight into stage 0)."""
    return self.num_repeats > 1 and self.num_microbatches > self.num_stages * self.forwarding_delay

  @property
  def bubble_iterations(self) -> int:
    """Iterations for microbatch 0 to finish one repeat: the fill (and drain) bubble."""
    return self.forwarding_delay * (self.num_stages - 1)

  @property
  def iterations_to_complete_first_microbatch(self) -> int:
    """Iterations for microbatch 0 to leave the last stage of the last repeat."""
    return self.num_microbatches * (self.num_repeats - 1) + self.bubble_iterations

  @property
  def total_iterations(self) -> int:
    """Every microbatch through every repeat, plus the bubble."""
    return self.num_microbatches * self.num_repeats + self.bubble_iterations


def unsupported_pipeline_settings(config) -> dict[str, Any]:
  """The legacy weight-prefetch settings `config` turns on that m3 does not implement, with their values."""
  return {
      field: getattr(config, field, off)
      for field, off in _UNSUPPORTED_PIPELINE_FIELDS.items()
      if getattr(config, field, off) not in (off, None)
  }


def schedule_from_config(config, num_stages: int, num_layers: int) -> PipelineSchedule:
  """The schedule a legacy-surface config describes for `num_layers` layers over `num_stages` stages.

  Reads legacy's field names (`num_layers_per_pipeline_stage`,
  `num_pipeline_repeats`, `num_pipeline_microbatches`,
  `pipeline_delay_activation_forwarding`, `pipeline_fsdp_ag_once`,
  `scan_pipeline_iterations`, `set_remat_policy_on_pipeline_iterations`) with
  legacy's defaults, so pyconfig and a slim config read the same. `-1` repeats
  means "as many as the layers allow", `-1` microbatches means one per stage
  (two with delayed forwarding), as in pyconfig.

  Raises:
    ValueError: If the layers do not divide into `num_stages x repeats x
      layers_per_stage`, or if a legacy knob m3 does not implement is on.
  """
  unsupported = unsupported_pipeline_settings(config)
  # Pipelining a prefix of the stack and running the rest data-parallel over `stage`; pyconfig resolves the
  # default `-1` to the layer count, so that value is "off" too.
  partial = getattr(config, "pipeline_parallel_layers", -1)
  if partial not in (-1, None, num_layers):
    unsupported["pipeline_parallel_layers"] = partial
  if unsupported:
    raise ValueError(
        f"m3 pipelines the whole decoder stack with the GPipe / circular schedule of legacy `NNXPipeline`; "
        f"it does not implement {unsupported} (see m3/README.md)."
    )
  layers_per_stage = getattr(config, "num_layers_per_pipeline_stage", 1)
  num_repeats = getattr(config, "num_pipeline_repeats", -1)
  if num_repeats == -1:
    num_repeats, remainder = divmod(num_layers, num_stages * layers_per_stage)
    if remainder or num_repeats < 1:
      raise ValueError(
          f"{num_layers} layers do not divide into {num_stages} stages x {layers_per_stage} layers per stage."
      )
  elif num_stages * num_repeats * layers_per_stage != num_layers:
    raise ValueError(
        f"{num_stages} stages x {num_repeats} repeats x {layers_per_stage} layers per stage is not the "
        f"{num_layers} layers of the stack."
    )
  delay = bool(getattr(config, "pipeline_delay_activation_forwarding", False))
  num_microbatches = getattr(config, "num_pipeline_microbatches", -1)
  if num_microbatches == -1:
    num_microbatches = num_stages * (2 if delay else 1)
  return PipelineSchedule(
      num_stages=num_stages,
      num_repeats=num_repeats,
      layers_per_stage=layers_per_stage,
      num_microbatches=num_microbatches,
      delay_activation_forwarding=delay,
      fsdp_ag_once=bool(getattr(config, "pipeline_fsdp_ag_once", False)),
      scan_iterations=bool(getattr(config, "scan_pipeline_iterations", True)),
      remat_iterations=bool(getattr(config, "set_remat_policy_on_pipeline_iterations", True)),
  )


def _is_variable(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def _mesh_axes_without(entry: MeshAxes, axis: str) -> MeshAxes:
  """`entry` of a `PartitionSpec` with mesh axis `axis` removed."""
  if entry is None or entry is P.UNCONSTRAINED:
    return entry
  if isinstance(entry, str):
    return None if entry == axis else entry
  kept = tuple(name for name in entry if name != axis)
  return kept or None


def _stack_stage_variables(variables: Sequence[nnx.Variable], leading: tuple[int, ...]) -> nnx.Variable:
  """Stacks one variable from each stage body under `leading` axes, the last of them sharded over `stage`.

  Placed explicitly, as `infra.scan._stack_variables` does: `jnp.stack` of
  sharded arrays leaves the result's placement to XLA. Placement is skipped
  when the bodies were not built under a mesh.
  """
  first = variables[0]
  values = [v[...] for v in variables]
  value = jnp.stack(values).reshape(leading + values[0].shape)
  spec = first.get_metadata().get("out_sharding")
  changes = {}
  if isinstance(spec, P):
    if any(STAGE_AXIS in ((e,) if isinstance(e, str) else (e or ())) for e in spec):
      raise ValueError(f"A pipelined parameter is already sharded over {STAGE_AXIS!r}: {spec}.")
    spec = P(*([None] * (len(leading) - 1)), STAGE_AXIS, *spec)
    changes["out_sharding"] = spec
    sharding = getattr(first.get_value(), "sharding", None)
    if isinstance(sharding, NamedSharding):
      value = jax.device_put(value, NamedSharding(sharding.mesh, spec))
  return first.replace(value, **changes)


def _strip_leading_axes(state: Any, count: int) -> Any:
  """Returns `state` with the `count` leading entries dropped from each sharding spec, for one stage's body."""

  def strip(variable):
    spec = variable.get_metadata().get("out_sharding")
    if not isinstance(spec, P):
      return variable
    return variable.replace(variable[...], out_sharding=P(*list(spec)[count:]))

  return jax.tree.map(strip, state, is_leaf=_is_variable)


class PipelinedLayers(nnx.Module):
  """Structurally identical stage bodies with stacked state, run as a pipeline over `stage`.

  `pipeline_run` combines this mixin with the stage body's class, preserving
  parameter paths and the body's helper methods. Each iteration rebuilds one
  body per stage from the stacked state (the current repeat's slice, in a
  circular pipeline) and runs it under a `jax.vmap` over the stages.

  Stage bodies hold only `nnx.Param` state. Other state (RNG streams, counters)
  would have to be carried through the iteration loop per stage and repeat;
  m3 models keep such state out of the decoder stack (README invariants), so
  it is refused rather than threaded.
  """

  def __init__(  # pylint: disable=super-init-not-called
      self, stages: Sequence[nnx.Module], *, schedule: PipelineSchedule, batch_axes: MeshAxes = None
  ):
    if len(stages) != schedule.num_stages * schedule.num_repeats:
      raise ValueError(
          f"Got {len(stages)} stage bodies for {schedule.num_repeats} repeats x {schedule.num_stages} stages."
      )
    graphdef = nnx.graphdef(stages[0])
    for i, stage in enumerate(stages[1:], start=1):
      if nnx.graphdef(stage) != graphdef:
        raise ValueError(f"Stage body {i} is not structurally identical to stage body 0, so they cannot share one vmap.")
    states = []
    for i, stage in enumerate(stages):
      params, rest = nnx.state(stage, nnx.Param, ...)
      if jax.tree.leaves(rest):
        kinds = sorted({type(v).__name__ for v in jax.tree.leaves(rest, is_leaf=_is_variable) if _is_variable(v)})
        raise ValueError(f"Stage body {i} holds non-parameter state {kinds}, which pipelined layers do not carry.")
      states.append(params)
    leading = (schedule.num_repeats, schedule.num_stages) if schedule.num_repeats > 1 else (schedule.num_stages,)
    stacked = jax.tree.map(  # pylint: disable=no-value-for-parameter
        lambda *vs: _stack_stage_variables(vs, leading), *states, is_leaf=_is_variable
    )
    stacked_stage = nnx.merge(graphdef, stacked)
    # Adopt the stacked body's attributes, so this module's parameter tree is
    # one body's tree under the stage (and repeat) axes -- no wrapper segment.
    for name, value in vars(stacked_stage).items():
      if not name.startswith(("_pytree__", "_object__")):
        setattr(self, name, value)
    self._schedule = schedule
    self._batch_axes = batch_axes
    self._num_leading = len(leading)

  @property
  def schedule(self) -> PipelineSchedule:
    return self._schedule

  # -- Layouts -----------------------------------------------------------------

  def _stage_spec(self, trailing: int) -> P:
    """`[stages, microbatch, ...]`: stages over `stage`, the microbatch as the strategy lays out a batch."""
    return P(STAGE_AXIS, self._batch_axes, *([None] * (trailing - 1)))

  def _state_io_spec(self, trailing: int) -> P:
    """`[stages, microbatches per stage, microbatch, ...]`; the microbatch index is looped over, not sharded."""
    return P(STAGE_AXIS, None, self._batch_axes, *([None] * (trailing - 1)))

  @staticmethod
  def _constrain(x: jax.Array, spec: P) -> jax.Array:
    return jax.lax.with_sharding_constraint(x, spec)

  def _by_stage(self, x: jax.Array, stage_dim: int = 0) -> jax.Array:
    """Pins `stage_dim` of `x` to `stage`, leaving the other dimensions to XLA."""
    spec = [P.UNCONSTRAINED] * x.ndim
    spec[stage_dim] = STAGE_AXIS
    return self._constrain(x, P(*spec))

  def _check_mesh(self) -> jax.sharding.AbstractMesh:
    """The auto-sharded mesh in scope, which must carry the schedule's number of stages."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
      raise ValueError("Pipelined layers must run under the mesh they were built with (jax.set_mesh).")
    if mesh.shape.get(STAGE_AXIS, 1) != self._schedule.num_stages:
      raise ValueError(
          f"Pipelined layers were built for {self._schedule.num_stages} stages, but the mesh in scope has "
          f"{STAGE_AXIS}={mesh.shape.get(STAGE_AXIS, 1)}: {dict(mesh.shape)}."
      )
    if any(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types):
      raise ValueError("Pipelined layers run under auto sharding only; the mesh in scope has explicit axes.")
    return mesh

  # -- Weights -----------------------------------------------------------------

  def _all_gather_over_fsdp(self, raw: Any, params: Any) -> Any:
    """Constrains each stacked weight to its placement with `fsdp` removed: one all-gather, before the loop."""

    def gather(x, variable):
      spec = variable.get_metadata().get("out_sharding")
      if not isinstance(spec, P):
        return x
      return self._constrain(x, P(*(_mesh_axes_without(entry, "fsdp") for entry in spec)))

    # `params` is flattened up to `raw`'s leaves, so its Variables pair with the raw arrays.
    return jax.tree.map(gather, raw, params)

  def _repeat_weights(self, raw: Any, repeat_ids: jax.Array) -> Any:
    """Each stage's slice of `[repeats, stages, ...]` at its own repeat: `[stages, ...]`."""

    def take(w_stage, repeat_id):
      return jax.lax.dynamic_index_in_dim(w_stage, repeat_id, axis=0, keepdims=False)

    def gather(w):
      return self._by_stage(jax.vmap(take, in_axes=(1, 0), out_axes=0)(w, repeat_ids))

    return jax.tree.map(gather, raw)

  # -- Schedule ----------------------------------------------------------------

  def _microbatch_and_repeat_ids(self, loop_iteration: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Which microbatch, of which repeat, each stage runs at `loop_iteration`; stage s is s x delay iterations behind."""
    s = self._schedule
    processed = jnp.maximum(loop_iteration - s.forwarding_delay * jnp.arange(s.num_stages), 0)
    processed = self._constrain(processed, P(STAGE_AXIS))
    return processed % s.num_microbatches, processed // s.num_microbatches

  def _gather_microbatches(self, x: jax.Array, microbatch_ids: jax.Array) -> jax.Array:
    """`x[microbatch_ids[s]]` for every stage s: `[microbatches, ...]` to `[stages, ...]`."""
    out = jax.vmap(lambda xs, i: jax.lax.dynamic_index_in_dim(xs, i, axis=0, keepdims=False), in_axes=(None, 0))(
        x, microbatch_ids
    )
    return self._constrain(out, self._stage_spec(x.ndim - 1))

  def _init_loop_state(self, inputs: jax.Array) -> dict[str, Any]:
    """The buffers of the schedule, from `inputs` of shape `[microbatches, microbatch, ...]`.

    `shift` carries each stage's output to the next; `state_io` holds the
    microbatches not yet entered and, by the end, the finished ones, per
    stage; `circ_storage` parks the last stage's outputs until stage 0 starts
    their next repeat, with `circ_storage_mover` buffering one iteration of
    delay; `prev_outputs` is the one-iteration delay of
    `pipeline_delay_activation_forwarding`.
    """
    s = self._schedule
    trailing = inputs.ndim - 1
    shift = self._constrain(jnp.zeros((s.num_stages,) + inputs.shape[1:], inputs.dtype), self._stage_spec(trailing))
    state_io = inputs.reshape((s.num_stages, s.microbatches_per_stage) + inputs.shape[1:])
    state_io = self._constrain(state_io, self._state_io_spec(trailing))
    circ_storage = None
    if s.use_circ_storage:
      # One slot per microbatch on every stage, though only stage 0 reads it: the
      # storage is `num_stages` times what is needed, as in legacy (b/347603101).
      circ_storage = jnp.zeros((s.num_stages,) + inputs.shape, inputs.dtype)
      circ_storage = self._constrain(circ_storage, self._state_io_spec(trailing))
    return {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": shift if s.use_circ_storage else None,
        "prev_outputs": shift if s.delay_activation_forwarding else None,
        "loop_iteration": jnp.zeros((), jnp.int32),
    }

  def _iteration_inputs(self, loop_state: Mapping[str, Any]) -> jax.Array:
    """What each stage runs this iteration: the rotated previous outputs, except stage 0, which takes a new microbatch.

    Stage 0 reads `state_io` while microbatches are still entering, then the
    last stage's parked output (`circ_storage`) or, when the last stage flows
    straight into the first, `shift`. In the drain iterations stage 0 runs on
    stale data whose result never reaches `state_io`.
    """
    s = self._schedule
    loop_iteration = loop_state["loop_iteration"]
    trailing = loop_state["shift"].ndim - 1
    shift = self._constrain(loop_state["shift"], self._stage_spec(trailing))
    state_io_slice = jax.lax.dynamic_index_in_dim(
        loop_state["state_io"], loop_iteration % s.microbatches_per_stage, axis=1, keepdims=False
    )
    if s.use_circ_storage:
      circular_in = jax.lax.dynamic_index_in_dim(
          loop_state["circ_storage"], loop_iteration % s.num_microbatches, axis=1, keepdims=False
      )
    else:
      circular_in = shift
    first_stage_in = jnp.where(loop_iteration < s.num_microbatches, state_io_slice, circular_in)
    first_stage_in = self._constrain(first_stage_in, self._stage_spec(trailing))
    stage_index = jax.lax.broadcasted_iota(jnp.int32, shift.shape, 0)
    return self._constrain(jnp.where(stage_index == 0, first_stage_in, shift), self._stage_spec(trailing))

  def _next_loop_state(
      self, mesh: jax.sharding.AbstractMesh, output: jax.Array, loop_state: Mapping[str, Any]
  ) -> dict[str, Any]:
    """Advances the buffers by one iteration given this iteration's stage outputs.

    `shift` takes the outputs moved one stage down (the last stage's output
    wraps to stage 0 only when it flows straight into the next repeat);
    `circ_storage` takes the previous iteration's outputs at the rotating
    microbatch slot; `state_io` moves one stage up at the current slot, the
    last stage's row refilled with the output that just finished.
    """
    s = self._schedule
    loop_iteration = loop_state["loop_iteration"]
    trailing = output.ndim - 1
    stage_spec, state_io_spec = self._stage_spec(trailing), self._state_io_spec(trailing)

    def rotate(x, offset):
      size = jax.lax.axis_size(STAGE_AXIS)
      return jax.lax.ppermute(x, STAGE_AXIS, [(i, (i + offset) % size) for i in range(size)])

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=stage_spec, out_specs=stage_spec)
    def rotate_down(x):
      return rotate(x, 1)

    @functools.partial(jax.shard_map, mesh=mesh, in_specs=stage_spec, out_specs=stage_spec)
    def shift_down(x):
      return jnp.where(jax.lax.axis_index(STAGE_AXIS) == 0, jnp.zeros_like(x), rotate(x, 1))

    @functools.partial(
        jax.shard_map, mesh=mesh, in_specs=(state_io_spec, stage_spec, stage_spec, P()), out_specs=state_io_spec
    )
    def update_state_io(state_io, stream_slice, output, slot):
      last = jax.lax.axis_index(STAGE_AXIS) == jax.lax.axis_size(STAGE_AXIS) - 1
      stream_slice = jnp.where(last, output, rotate(stream_slice, -1))
      return jax.lax.dynamic_update_slice_in_dim(state_io, stream_slice[:, None], slot, axis=1)

    forwarded = loop_state["prev_outputs"] if s.delay_activation_forwarding else output
    if s.num_repeats == 1 or s.use_circ_storage:
      new_shift = shift_down(forwarded)
    else:
      new_shift = rotate_down(forwarded)

    new_circ_storage = new_circ_storage_mover = None
    if s.use_circ_storage:
      # The previous iteration's output, so that the transfer overlaps this iteration's compute; the slot rotates so
      # that microbatch 0 lands in slot 0.
      slot = (loop_iteration - s.bubble_iterations - 1) % s.num_microbatches
      rotated = rotate_down(loop_state["circ_storage_mover"])[:, None]
      new_circ_storage = jax.lax.dynamic_update_slice_in_dim(loop_state["circ_storage"], rotated, slot, axis=1)
      new_circ_storage_mover = output

    slot = loop_iteration % s.microbatches_per_stage
    stream_slice = jax.lax.dynamic_index_in_dim(loop_state["state_io"], slot, axis=1, keepdims=False)
    new_state_io = update_state_io(loop_state["state_io"], stream_slice, output, slot)

    return {
        "state_io": new_state_io,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "prev_outputs": output if s.delay_activation_forwarding else None,
        "loop_iteration": loop_iteration + 1,
    }

  def _finished_microbatches(self, state_io: jax.Array) -> jax.Array:
    """`state_io` at the end, with its slots rotated back so microbatch 0 is first: `[microbatches, microbatch, ...]`.

    Microbatch 0 finishes `iterations_to_complete_first_microbatch` iterations
    in and lands in that slot (modulo the slots per stage); the rest follow.
    """
    s = self._schedule
    first = s.iterations_to_complete_first_microbatch % s.microbatches_per_stage
    permutation = (np.arange(s.microbatches_per_stage) + first) % s.microbatches_per_stage
    return state_io[:, permutation].reshape((s.num_microbatches,) + state_io.shape[2:])

  # -- Forward -----------------------------------------------------------------

  def __call__(self, carry: jax.Array, **kwargs: Any) -> jax.Array:
    s = self._schedule
    mesh = self._check_mesh()
    batch = carry.shape[0]
    if batch % s.num_microbatches:
      raise ValueError(f"Batch {batch} does not split into {s.num_microbatches} microbatches.")
    microbatch = batch // s.num_microbatches
    batch_axes = (self._batch_axes,) if isinstance(self._batch_axes, str) else tuple(self._batch_axes or ())
    batch_shards = math.prod(mesh.shape[axis] for axis in batch_axes)
    if microbatch % batch_shards:
      raise ValueError(
          f"Microbatches of {microbatch} (batch {batch} over {s.num_microbatches} microbatches) do not split over "
          f"the {batch_shards} shards of the batch axes {batch_axes}; use fewer microbatches or a larger batch."
      )

    graphdef, params, _ = nnx.split(self, nnx.Param, ...)
    # The graphdef the stage bodies are rebuilt from: one stage's metadata, without the stacked axes.
    stage_graphdef = nnx.graphdef(nnx.merge(graphdef, _strip_leading_axes(params, self._num_leading)))
    raw = jax.tree.map(lambda v: v[...], params, is_leaf=_is_variable)
    if s.fsdp_ag_once:
      raw = self._all_gather_over_fsdp(raw, params)

    def is_batched(value):
      return isinstance(value, (jax.Array, np.ndarray)) and value.ndim >= 1 and value.shape[0] == batch

    batched = {k: v.reshape((s.num_microbatches, microbatch) + v.shape[1:]) for k, v in kwargs.items() if is_batched(v)}
    static = {k: v for k, v in kwargs.items() if not is_batched(v)}
    inputs = carry.reshape((s.num_microbatches, microbatch) + carry.shape[1:])

    def stage_fn(stage_params, stage_inputs, stage_kwargs):
      stage = nnx.merge(stage_graphdef, stage_params)
      # The stage body's `__call__`: next in the MRO after this mixin.
      return super(PipelinedLayers, stage).__call__(stage_inputs, **stage_kwargs, **static)

    run_stages = jax.vmap(stage_fn, in_axes=(0, 0, 0), spmd_axis_name=STAGE_AXIS)

    def iteration(loop_state):
      microbatch_ids, repeat_ids = self._microbatch_and_repeat_ids(loop_state["loop_iteration"])
      stage_inputs = ad_checkpoint.checkpoint_name(self._iteration_inputs(loop_state), "iteration_input")
      stage_kwargs = {k: self._gather_microbatches(v, microbatch_ids) for k, v in batched.items()}
      stage_params = raw if s.num_repeats == 1 else self._repeat_weights(raw, repeat_ids)
      output = run_stages(stage_params, stage_inputs, stage_kwargs)
      return self._next_loop_state(mesh, output, loop_state)

    if s.remat_iterations:
      iteration = jax.checkpoint(
          iteration,
          policy=jax.checkpoint_policies.save_only_these_names("iteration_input"),
          prevent_cse=not s.scan_iterations,
      )

    loop_state = self._init_loop_state(inputs)
    if s.scan_iterations:
      loop_state, _ = jax.lax.scan(lambda state, _: (iteration(state), None), loop_state, None, length=s.total_iterations)
    else:
      for _ in range(s.total_iterations):
        loop_state = iteration(loop_state)

    return self._finished_microbatches(loop_state["state_io"]).reshape((batch,) + carry.shape[1:])


@functools.cache
def _pipelined_class(body_cls: type[nnx.Module]) -> type[PipelinedLayers]:
  """Cache the path-transparent wrapper type so models share graphdefs and JIT caches."""
  return type(f"Pipelined{body_cls.__name__}", (PipelinedLayers, body_cls), {})


def pipeline_run(
    stages: Sequence[nnx.Module], *, schedule: PipelineSchedule, batch_axes: MeshAxes = None
) -> PipelinedLayers:
  """Stacks `stages` (all of one class, repeat-major) into a `Pipelined<Body>` instance."""
  if not stages:
    raise ValueError("Cannot pipeline an empty run of stages.")
  return _pipelined_class(type(stages[0]))(stages, schedule=schedule, batch_axes=batch_axes)


def pipeline_layers(
    layers: nnx.Sequential,
    *,
    schedule: PipelineSchedule,
    batch_axes: MeshAxes = None,
    scan_layers: bool = False,
    param_scan_axis: int = 1,
) -> PipelinedLayers:
  """Returns the pipelined replacement for one `nnx.Sequential` container.

  Consecutive runs of `layers_per_stage` layers form the stage bodies, in
  repeat-major order: stage `s` of repeat `r` holds layers
  `(r * num_stages + s) * layers_per_stage` onwards. A body is the layer itself
  for one layer per stage, an `nnx.Sequential` of the layers otherwise, and
  with `scan_layers` the layers scanned (`infra.scan.scan_run`) -- also for a
  single layer, so the parameter layout does not depend on the stage size.
  """
  layers = list(layers)
  if len(layers) != schedule.num_layers:
    raise ValueError(
        f"The stack has {len(layers)} layers, but the schedule pipelines {schedule.num_layers} "
        f"({schedule.num_stages} stages x {schedule.num_repeats} repeats x {schedule.layers_per_stage} per stage)."
    )
  n = schedule.layers_per_stage
  bodies = []
  for i in range(schedule.num_stages * schedule.num_repeats):
    group = layers[i * n : (i + 1) * n]
    if scan_layers:
      bodies.append(m3_scan.scan_run(group, param_scan_axis=param_scan_axis))
    elif n == 1:
      bodies.append(group[0])
    else:
      bodies.append(nnx.Sequential(*group))
  return pipeline_run(bodies, schedule=schedule, batch_axes=batch_axes)


def _replace_child(model: nnx.Module, path: tuple, replacement: nnx.Module) -> None:
  parent: Any = model
  for key in path[:-1]:
    parent = parent[key] if isinstance(key, int) or isinstance(parent, MutableMapping) else getattr(parent, key)
  key = path[-1]
  if isinstance(key, int) or isinstance(parent, MutableMapping):
    parent[key] = replacement
  else:
    setattr(parent, key, replacement)


def the_stack(model: nnx.Module) -> tuple[tuple, nnx.Sequential]:
  """The path and container of the model's one `nnx.Sequential`: the decoder stack.

  A model with several containers is refused: which one is the stack to
  pipeline would be a guess, and the others would run unpipelined in silence.
  """
  containers = [(path, node) for path, node in nnx.iter_graph(model) if isinstance(node, nnx.Sequential)]
  if len(containers) != 1:
    raise ValueError(
        f"Pipelining expects exactly one nnx.Sequential (the decoder stack); found {len(containers)}: "
        f"{['/'.join(map(str, path)) for path, _ in containers]}."
    )
  return containers[0]


def apply_pipeline(
    model: nnx.Module,
    *,
    schedule: PipelineSchedule,
    sharding: Sharding,
    scan_layers: bool = False,
    param_scan_axis: int = 1,
) -> nnx.Module:
  """Replaces the model's decoder stack (`the_stack`) with its pipelined form, in place.

  `sharding` lays out the microbatch dimension of the pipeline's buffers: its
  activation rule for `batch`, which is also what the layers inside the stages
  constrain their batch to.
  """
  path, container = the_stack(model)
  (batch_axes,) = sharding("pipeline_microbatch", ("batch",), TensorType.ACTIVATION)
  if batch_axes is not None and STAGE_AXIS in ((batch_axes,) if isinstance(batch_axes, str) else batch_axes):
    raise ValueError(f"The strategy shards the batch over {STAGE_AXIS!r}, which the pipeline needs for its stages.")
  replacement = pipeline_layers(
      container, schedule=schedule, batch_axes=batch_axes, scan_layers=scan_layers, param_scan_axis=param_scan_axis
  )
  if not path:
    return replacement
  _replace_child(model, path, replacement)
  return model


def maybe_apply_pipeline(model: nnx.Module, config, mesh: jax.sharding.Mesh, sharding: Sharding | None = None):
  """Pipelines the model's layers over the mesh's `stage` axis when it has more than one stage.

  The mesh decides, as it does for every other axis: `mesh_from_config` sizes
  `stage` from `ici_pipeline_parallelism x dcn_pipeline_parallelism`. The
  schedule comes from the config (`schedule_from_config`); the microbatch
  layout from `sharding`, `DefaultSharding` unless given, as in the loop.
  """
  num_stages = dict(mesh.shape).get(STAGE_AXIS, 1)
  if num_stages == 1:
    return model
  _, container = the_stack(model)
  return apply_pipeline(
      model,
      schedule=schedule_from_config(config, num_stages=num_stages, num_layers=len(list(container))),
      sharding=sharding or DefaultSharding(),
      scan_layers=bool(getattr(config, "scan_layers", False)),
      param_scan_axis=getattr(config, "param_scan_axis", 1),
  )
