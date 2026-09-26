# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ahead-of-time (XAOT) compilation of `MaxTextTrainingEngine`'s training step.

`trainers/pre_train/train_compile.py` does this for `train.py`'s single fused `train_step`.
The engine splits the same work across three kernels -- one forward/backward for the first
micro-batch of an update, an accumulating one for every later micro-batch, and the optimizer
update -- so this compiles all three and reports the cost and memory of each.

Nothing is materialized: the weights, the optimizer moments and the batch are all
`jax.ShapeDtypeStruct`s, and the device mesh is a topology description rather than hardware.
So a v5e-256 configuration can be compiled from a workstation, and an out-of-memory one
reports the same `RESOURCE_EXHAUSTED` it would report on the target -- before the target is
booked.

Example, qwen3-0.6b on four v6e chips:

  python3 -m maxtext.training_engine.maxtext_engine_compile src/maxtext/configs/base.yml \
    model_name=qwen3-0.6b run_name=engine_aot_qwen3 \
    compile_topology=v6e-4 compile_topology_num_slices=1 \
    per_device_batch_size=4 max_target_length=2048 \
    ici_fsdp_parallelism=4 attention=flash enable_checkpointing=false

Add `compiled_trainstep_file=/tmp/engine_qwen3.pickle` to serialize the executables; each
kernel is written to its own file, suffixed with the kernel name.

Each kernel's raw `memory_analysis()` is printed, followed by a per-device table built from it.
The table corrects for two things the raw numbers hide: a donated buffer is counted in both the
arguments and the outputs, and the forward/backward kernels are not passed the optimizer state,
so it appears in neither of their analyses although it stays in HBM. The table ends with the
TRAIN-KERNEL DEVICE PEAK: the largest, over the three kernels, of what the kernel holds plus the
train state it runs alongside without being passed.

That peak covers the three training kernels only. It excludes weight-sync staging (reported on
its own line when it can be estimated; see `weight_sync_staging`), `fwd_only` / `model_scope`
scoring, the eval kernel, generated code (as `max_utils.print_compiled_memory_stats` does), and
anything the caller holds. It includes `fwd_bwd_accum` even though that kernel never runs at one
micro-batch per update.
"""

import dataclasses
import math
import os
from typing import Any, Sequence

from absl import app
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import maxtext_engine
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils

# Re-exported: which kernels there are is the engine's to say, and this module reports on
# whatever it lowers.
KERNEL_NAMES = maxtext_engine.KERNEL_NAMES

# Both `dump_hlo` filters default to `jit_train_step`, `train.py`'s fused step; the engine's
# kernels lower as `jit_<name>` for each of `KERNEL_NAMES` (`jit_fwd_bwd`, `jit_fwd_bwd_accum`,
# `jit_update`), so on those defaults the dump comes back empty.
HLO_DUMP_DEFAULTS = {
    "dump_hlo_local_module_name": f"jit_({'|'.join(KERNEL_NAMES)})",
    "dump_hlo_module_name": "jit_",
}

_GIB = 2**30

# Labels of the report's summary lines. The peak is named for what it covers, the three training
# kernels, because a full step can need more; see the module docstring.
PEAK_LABEL = "TRAIN-KERNEL DEVICE PEAK"
STAGING_LABEL = "WEIGHT-SYNC STAGING"

# Memory kinds that live in host RAM. Anything else, including `None` (an aval with no memory kind
# yet), is counted as device memory.
_HOST_MEMORY_KINDS = frozenset({"pinned_host", "unpinned_host"})

# Top-level keys of the train state's pure form, `nnx.split(TrainStateNNX(...))`.
_MODEL_KEY = maxtext_engine._MODEL_STATE_KEY  # pylint: disable=protected-access
_OPTIMIZER_KEY = maxtext_engine._OPTIMIZER_STATE_KEY  # pylint: disable=protected-access

# Per kernel, the top-level train-state subtrees that stay allocated while it runs but are not
# among its arguments, so its `memory_analysis()` cannot see them. From the signatures
# `_compile_for_batch` jits: the forward/backward kernels take the model state, a batch and, for
# `fwd_bwd_accum`, the gradient sum, but never the optimizer; `update` takes the whole state.
STATE_NOT_PASSED = {
    "fwd_bwd": (_OPTIMIZER_KEY,),
    "fwd_bwd_accum": (_OPTIMIZER_KEY,),
    "update": (),
}


@dataclasses.dataclass(frozen=True)
class KernelMemory:
  """One kernel's memory, per device, in bytes.

  `argument` through `host_temp` come from `memory_analysis()`; the `not_passed` fields are
  what it cannot see.

  Attributes:
    kernel: The kernel's name, one of `KERNEL_NAMES`.
    argument: Device bytes of the arguments.
    output: Device bytes of the outputs.
    alias: Device bytes an argument shares with an output -- a donated buffer written in place,
      which `argument` and `output` both include.
    temp: Device scratch the kernel allocates while it runs.
    host_argument: Host bytes of the arguments, e.g. a leaf on `pinned_host`. XLA:CPU has a single
      memory space, so there it reports zero and counts such a leaf in `argument`.
    host_output: Host bytes of the outputs, with the same caveat.
    host_alias: `alias` for host memory -- under `optimizer_memory_host_offload` the update
      donates the offloaded optimizer state, so it is in both `host_argument` and `host_output`.
    host_temp: Host scratch the kernel allocates while it runs, e.g. activations offloaded under
      `remat_policy=custom` with `decoder_layer_input=offload`. Host RAM, so not in `resident`.
    not_passed: The train-state subtrees resident while this kernel runs but not passed to it.
    device_not_passed: Their bytes in device memory.
    host_not_passed: Their bytes in host memory.
  """

  kernel: str
  argument: int
  output: int
  alias: int
  temp: int
  host_argument: int
  host_output: int
  host_alias: int
  host_temp: int
  not_passed: tuple[str, ...]
  device_not_passed: int
  host_not_passed: int

  @property
  def resident(self) -> int:
    """Device bytes the kernel holds: `argument + (output - alias) + temp`.

    The formula `max_utils.print_compiled_memory_stats` uses, which also excludes generated code.
    """
    return self.argument + self.output - self.alias + self.temp

  @property
  def device_total(self) -> int:
    """Device bytes in use while this kernel runs: its own, plus the train state it is not passed."""
    return self.resident + self.device_not_passed


def per_device_bytes(leaf: Any) -> tuple[int, int]:
  """Returns the `(device, host)` bytes one device holds of `leaf`; at most one is nonzero.

  A leaf with no sharding is counted whole, as a replicated one is. Uses `dtype.itemsize` rather
  than `np.dtype(...).itemsize`, which rejects the PRNG key dtype in the model's RNG state.
  """
  if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
    return 0, 0
  sharding = getattr(leaf, "sharding", None)
  shard_shape = sharding.shard_shape(leaf.shape) if sharding is not None else leaf.shape
  nbytes = math.prod(shard_shape) * leaf.dtype.itemsize
  if getattr(sharding, "memory_kind", None) in _HOST_MEMORY_KINDS:
    return 0, nbytes
  return nbytes, 0


def _tree_bytes_per_device(tree: Any) -> tuple[int, int]:
  """Returns the `(device, host)` bytes one device holds of every leaf in `tree`."""
  device, host = 0, 0
  for leaf in jax.tree.leaves(tree):
    leaf_device, leaf_host = per_device_bytes(leaf)
    device += leaf_device
    host += leaf_host
  return device, host


def memory_report(compiled: dict[str, Any], state: Any) -> list[KernelMemory]:
  """Returns each kernel's per-device memory, including the train state it runs without being passed.

  Args:
    compiled: `{kernel name: jax.stages.Compiled}`, as `compile_engine_kernels` returns it.
    state: The train state's pure form -- arrays or avals, each on its sharding -- on the layouts
      the kernels were compiled against; see `AbstractMaxTextEngine.train_state_avals`.

  Returns:
    One `KernelMemory` per kernel, in `compiled`'s order.

  Raises:
    ValueError: If a kernel is not in `STATE_NOT_PASSED`, has no memory analysis, or runs
      alongside a subtree `state` lacks, or if `state` has a top-level subtree that is neither the
      model nor in `STATE_NOT_PASSED`. Each would otherwise understate the peak.
  """
  # The model is passed to every kernel. Any other subtree must be classified, or it would be
  # resident while a kernel runs yet counted in neither its arguments nor `+state`.
  known = {_MODEL_KEY}.union(*STATE_NOT_PASSED.values())
  unknown = sorted(str(key) for key in state if key not in known)
  if unknown:
    raise ValueError(
        f"The train state has top-level subtrees {unknown} that STATE_NOT_PASSED does not classify. Add them, "
        "based on the kernel signatures in `_compile_for_batch`."
    )
  rows = []
  for name, executable in compiled.items():
    if name not in STATE_NOT_PASSED:
      raise ValueError(
          f"STATE_NOT_PASSED has no entry for kernel {name!r}. Add one, based on its signature in "
          "`_compile_for_batch`."
      )
    stats = executable.memory_analysis()
    if stats is None:
      raise ValueError(f"{name} has no memory analysis on this backend, so there is nothing to report.")
    not_passed = STATE_NOT_PASSED[name]
    missing = [key for key in not_passed if key not in state]
    if missing:
      raise ValueError(f"The train state has no {missing} subtree, which {name} is expected to run alongside.")
    device_not_passed, host_not_passed = _tree_bytes_per_device([state[key] for key in not_passed])
    rows.append(
        KernelMemory(
            kernel=name,
            argument=stats.argument_size_in_bytes,
            output=stats.output_size_in_bytes,
            alias=stats.alias_size_in_bytes,
            temp=stats.temp_size_in_bytes,
            host_argument=stats.host_argument_size_in_bytes,
            host_output=stats.host_output_size_in_bytes,
            host_alias=stats.host_alias_size_in_bytes,
            host_temp=stats.host_temp_size_in_bytes,
            not_passed=not_passed,
            device_not_passed=device_not_passed,
            host_not_passed=host_not_passed,
        )
    )
  return rows


@dataclasses.dataclass(frozen=True)
class WeightSyncStaging:
  """What `MaxTextTrainingEngine.prepare_weight_sync` holds on device, per device, in bytes.

  Attributes:
    held_copy: Device bytes of the staged parameter copy, which lives until `release_weight_sync`
      (or the next `prepare_weight_sync`) and is not in any kernel's analysis. None when
      `use_weight_converter` is set: the converter stages the rollout's layout, which is not
      known here.
    model: Device bytes of the model state, resident throughout.
    optimizer: Device bytes of the optimizer state, resident throughout unless offloaded.
  """

  held_copy: int | None
  model: int
  optimizer: int

  @property
  def device_total(self) -> int | None:
    """Device bytes while the copy is held: the train state plus the copy, before any temporaries."""
    if self.held_copy is None:
      return None
    return self.model + self.optimizer + self.held_copy


def _path_keys(path: Sequence[Any]) -> tuple[Any, ...]:
  """Returns a `tree_flatten_with_path` path as the keys `flax.traverse_util.flatten_dict` would give."""
  return tuple(getattr(key, "key", getattr(key, "name", getattr(key, "idx", key))) for key in path)


def weight_sync_staging(
    state: Any,
    *,
    scan_layers: bool,
    scan_axis: int,
    use_weight_converter: bool,
    layer_container: str = "layers",
) -> WeightSyncStaging:
  """Returns the device bytes `prepare_weight_sync` holds beyond the train state.

  Mirrors its path without `use_weight_converter`: every floating `nnx.Param` is cast to bfloat16
  and, under `scan_layers`, the scanned layers are sliced into one array per layer
  (`raiden_unscan.unscan_layers`). A cast to a different dtype or a slice makes a new buffer, so an
  unscanned bfloat16 model is staged in place and costs nothing. The copy is held until
  `release_weight_sync`, alongside the model and optimizer state (the gradient accumulator is
  released by `update`).

  With `use_weight_converter` (the config default) the copy is in the rollout's layout, which is
  not known at compile time, so `held_copy` is None.

  Excludes the conversion's temporaries: with `scan_layers` and weights that are not bfloat16, the
  scanned bfloat16 cast is alive alongside the per-layer slices until the unscan returns, up to one
  more copy of the scanned parameters. Assumes the scan axis is not sharded.

  Args:
    state: The train state's pure form, as `memory_report` takes it.
    scan_layers: `config.scan_layers`.
    scan_axis: `config.param_scan_axis`.
    use_weight_converter: `config.use_weight_converter`.
    layer_container: The key `unscan_layers` splits; `prepare_weight_sync` uses its default.

  Returns:
    The staged copy's device bytes, next to the train state it is held alongside.
  """
  model, _ = _tree_bytes_per_device(state[_MODEL_KEY])
  optimizer, _ = _tree_bytes_per_device(state[_OPTIMIZER_KEY])
  if use_weight_converter:
    return WeightSyncStaging(held_copy=None, model=model, optimizer=optimizer)

  held_copy = 0
  params = jax.tree_util.tree_flatten_with_path(state[_MODEL_KEY], is_leaf=lambda node: isinstance(node, nnx.Variable))[0]
  for path, variable in params:
    if not isinstance(variable, nnx.Param):
      continue
    in_scanned_layer = scan_layers and layer_container in _path_keys(path)
    for leaf in jax.tree.leaves(variable):
      if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        continue
      floating = jnp.issubdtype(leaf.dtype, jnp.floating)
      staged_dtype = jnp.dtype(jnp.bfloat16) if floating else leaf.dtype
      cast_copies = floating and leaf.dtype != staged_dtype
      slice_copies = in_scanned_layer and len(leaf.shape) > scan_axis
      if cast_copies or slice_copies:
        staged = jax.ShapeDtypeStruct(leaf.shape, staged_dtype, sharding=getattr(leaf, "sharding", None))
        held_copy += sum(per_device_bytes(staged))
  return WeightSyncStaging(held_copy=held_copy, model=model, optimizer=optimizer)


def device_peak(rows: Sequence[KernelMemory]) -> KernelMemory:
  """Returns the kernel that sets the train-kernel device peak: the largest `device_total`.

  Not the largest `resident`: `update` is passed the optimizer state, so its own numbers usually
  look largest, but a forward/backward kernel holds that same state on top of its activations.
  """
  if not rows:
    raise ValueError("There are no kernels to take a peak over.")
  return max(rows, key=lambda row: row.device_total)


# The table's columns between the kernel name and the subtrees not passed to it: (header, width,
# bytes). The header and the cell come from one entry, so they cannot drift apart.
_TABLE_COLUMNS = (
    ("arg", 9, lambda row: row.argument),
    ("out", 9, lambda row: row.output),
    ("alias", 9, lambda row: row.alias),
    ("temp", 9, lambda row: row.temp),
    ("resident", 10, lambda row: row.resident),
    ("+state", 9, lambda row: row.device_not_passed),
    ("total", 9, lambda row: row.device_total),
    ("host arg", 12, lambda row: row.host_argument),
    ("host out", 9, lambda row: row.host_output),
    ("host alias", 11, lambda row: row.host_alias),
    ("host temp", 10, lambda row: row.host_temp),
    ("host state", 11, lambda row: row.host_not_passed),
)


def _table_line(first: str, cells: Any, last: str) -> str:
  """Returns one table line: `first` left-aligned, each `(header, width, text)` right-aligned, then `last`."""
  return f"{first:<15}" + "".join(f"{text:>{width}}" for _, width, text in cells) + f"   {last}"


def format_memory_report(rows: Sequence[KernelMemory], staging: WeightSyncStaging | None = None) -> str:
  """Returns the report `main` prints: a per-device table, then a line naming the peak.

  Args:
    rows: `memory_report`'s rows.
    staging: `weight_sync_staging`'s estimate. When given, it is printed on its own line before
      the peak, which does not include it.
  """

  def gib(num_bytes: int) -> str:
    return f"{num_bytes / _GIB:.2f}"

  lines = [
      "Engine memory per device, GiB (2^30). resident = arg + out - alias + temp: a donated buffer is in",
      "both arg and out, and alias is what they share. +state = train state in device memory while the",
      "kernel runs that is not one of its arguments, so its memory analysis cannot see it. The host",
      "columns are the same quantities in host memory (pinned_host), which XLA:CPU counts as device.",
      "The peak below is these three kernels' only. NOT included: weight-sync staging (a copy of the",
      "parameters in the rollout's layout, held until release_weight_sync), fwd_only/model_scope scoring,",
      "the eval kernel, generated code, and anything the caller holds. It assumes gradient accumulation:",
      "at one micro-batch per update fwd_bwd_accum never runs.",
      _table_line("kernel", ((name, width, name) for name, width, _ in _TABLE_COLUMNS), "state not passed"),
  ]
  for row in rows:
    cells = ((name, width, gib(cell(row))) for name, width, cell in _TABLE_COLUMNS)
    lines.append(_table_line(row.kernel, cells, ", ".join(row.not_passed) or "-"))

  peak = device_peak(rows)
  if peak.not_passed:
    state_name = " + ".join(f"{key} state" for key in peak.not_passed)
    because = f"{peak.kernel} {gib(peak.resident)} + {state_name} {gib(peak.device_not_passed)} not passed to it"
    if peak.host_not_passed:
      because += f"; {gib(peak.host_not_passed)} GiB more of it is on host"
  else:
    because = f"{peak.kernel}, which is passed all the train state it runs with"

  if staging is not None:
    train_state = (
        f"{gib(staging.model + staging.optimizer)} GiB of train state "
        f"(model {gib(staging.model)} + optimizer {gib(staging.optimizer)})"
    )
    if staging.device_total is None:
      lines.append(
          f"{STAGING_LABEL}, outside the peak below: not estimated -- use_weight_converter stages the rollout's "
          f"layout, which this tool does not see. It is held until release_weight_sync on top of {train_state}."
      )
    else:
      above = f"; ABOVE the {PEAK_LABEL.lower()}" if staging.device_total > peak.device_total else ""
      lines.append(
          f"{STAGING_LABEL}, outside the peak below: prepare_weight_sync holds a {gib(staging.held_copy)} GiB copy of "
          f"the parameters until release_weight_sync, on top of {train_state}: {gib(staging.device_total)} GiB before "
          f"the conversion's temporaries{above}"
      )
  lines.append(f"{PEAK_LABEL}: {gib(peak.device_total)} GiB ({because})")
  return "\n".join(lines)


def _propagation_mesh(mesh: jax.sharding.Mesh) -> jax.sharding.Mesh:
  """Returns a stand-in for `mesh` that `jax.eval_shape` will propagate shardings across.

  Nothing runs on it. `jax.eval_shape` carries a value's layout through an operation only on
  `Explicit` axes, so under `shard_mode=auto` -- where every axis is `Auto` -- the moments would
  all come back replicated. Marking the axes `Explicit` is how the layouts are *observed*; they
  are re-homed onto the real mesh afterwards, and its axis types decide what actually runs.
  """
  axis_types = getattr(mesh, "axis_types", None)
  if axis_types is not None and all(axis_type == jax.sharding.AxisType.Explicit for axis_type in axis_types):
    return mesh
  return jax.sharding.Mesh(
      mesh.devices,
      mesh.axis_names,
      axis_types=(jax.sharding.AxisType.Explicit,) * len(mesh.axis_names),
  )


def _rehome_aval(aval: Any, mesh: jax.sharding.Mesh) -> Any:
  """Returns `aval` with its sharding spec re-expressed on `mesh`.

  The engine's `_mesh_sharding` compares meshes by equality, so a spec that is right but homed
  on the trace's mesh would be silently replaced by a replicated one.
  """
  if not hasattr(aval, "shape") or not hasattr(aval, "dtype"):
    return aval
  spec = getattr(getattr(aval, "sharding", None), "spec", None)
  target = jax.sharding.NamedSharding(mesh, spec) if spec is not None else None
  return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=target)


class AbstractMaxTextEngine(maxtext_engine.MaxTextTrainingEngine):
  """A `MaxTextTrainingEngine` whose weights and moments are shapes rather than arrays.

  Enough to trace and compile every kernel, which is all `compile_kernels()` needs, while nothing
  is allocated and no checkpoint, tokenizer or network is touched. Nothing can be executed.
  """

  def __init__(
      self,
      training_config: pyconfig.HyperParameters,
      mesh: jax.sharding.Mesh,
      wrap_with_tunix_adapter: bool = False,
      tokenizer_pad_id: int | None = None,
  ) -> None:
    """Initializes an engine that can be lowered but not run.

    Args:
      training_config: MaxText HyperParameters configuration instance.
      mesh: The mesh to compile against, typically a topology this host does not own.
      wrap_with_tunix_adapter: As `MaxTextTrainingEngine`'s. Needed to compile a Tunix loss
        (`algo_core.grpo_loss_fn`), which calls the model with Tunix's signature.
      tokenizer_pad_id: As `MaxTextTrainingEngine`'s; required with the adapter.

    Raises:
      ValueError: If `mesh` is None. With no weights there is no device set to read one off.
    """
    if mesh is None:
      raise ValueError(
          "AbstractMaxTextEngine requires a mesh: with no weights there is nothing to read a device set "
          "off, and the point of the abstract path is to compile against a mesh this host does not own -- "
          "build one with `trainers.pre_train.train_compile.get_topology_mesh`."
      )
    super().__init__(
        training_config,
        mesh=mesh,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )

  def _build_model(self, wrap_with_tunix_adapter: bool, tokenizer_pad_id: int | None) -> Any:
    """Returns the model with `jax.ShapeDtypeStruct` weights on their real shardings.

    The same `create_nnx_abstract_model` call `from_pretrained` makes before it materializes
    anything, minus the checkpoint load -- so no weights, no HF token and no network. The
    adapter wrap matches `from_pretrained`'s, so a compiled Tunix loss sees the same module as a
    live engine's.
    """
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(
        model_creation_utils.verify_and_sync_scan_layers(self._config),
        self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
    )
    if wrap_with_tunix_adapter:
      with self._mesh:
        abstract_model = TunixMaxTextAdapter(
            base_model=abstract_model,
            use_no_op_mappings="maxtext_config" in self._config.vllm_additional_config,
            pad_id=tokenizer_pad_id,
        )
        abstract_model.config = None  # pyrefly: ignore[missing-attribute]
    return abstract_model

  def _build_optimizer(self, tx: Any) -> Any:
    """Installs the traced train state and returns the optimizer inside it.

    `self._model` is rebound to the model inside that state so the two stay one graph.
    """
    self._state = self._trace_train_state(tx)
    self._model = self._state.model
    return self._state.optimizer

  def _trace_train_state(self, tx: Any) -> Any:
    """Returns the `TrainStateNNX` for this model, moments included, as avals.

    `nnx.Optimizer` allocates the moments eagerly with `zeros_like`, so they are traced instead.
    Two traces, because neither alone answers both questions: `nnx.eval_shape` gives the module
    graph but drops shardings, and `jax.eval_shape` under `_propagation_mesh` gives the
    layouts, because that is where JAX carries a parameter's sharding through the `zeros_like`
    inside `tx.init` into the moment allocated from it. The result is merged back onto the real
    mesh, whose axis types -- not the stand-in's -- decide what the compiled kernels do.

    The layout trace runs under the engine's own `_sharding_ctx`, so the rules the MaxText layers
    are written against are the live ones rather than a second copy that can drift from them.

    The graph trace runs with no mesh in context. Given one, `nnx.eval_shape` re-derives every
    variable's sharding from its logical names through flax's `get_var_pspec`, which, unlike
    MaxText's lookup, raises on a mesh axis two dimensions both map to or on a logical name the
    rules leave unmapped. Only the graph is kept from this trace, so skipping that loses nothing.
    """
    model_graphdef, model_pure = nnx.split(self._model)

    def build(model_state):
      model = nnx.merge(model_graphdef, model_state)
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    propagation_mesh = _propagation_mesh(self._mesh)
    state_graphdef, _ = nnx.split(nnx.eval_shape(build, model_pure))
    with self._sharding_ctx():
      # Displaces the real mesh for this trace only: the graph trace above needs none, and a
      # stand-in set around it collides with the config's own AbstractMesh under `shard_mode=auto`.
      with jax.set_mesh(propagation_mesh):
        state_pure = jax.eval_shape(
            lambda model_state: nnx.split(build(model_state))[1],
            jax.tree.map(lambda aval: _rehome_aval(aval, propagation_mesh), model_pure),
        )
    return nnx.merge(state_graphdef, jax.tree.map(lambda aval: _rehome_aval(aval, self._mesh), state_pure))

  def train_state_avals(self) -> Any:
    """Returns the train state's pure form, as avals on the layouts the kernels were compiled against.

    `memory_report` needs these because the optimizer state is not an argument of either
    forward/backward kernel, so neither kernel's `memory_analysis()` includes it.

    Only valid after `compile_kernels()`, which moves the optimizer state onto its Zero-1 layout
    (`_shard_optimizer_state_over_data`); read earlier, the moments may come back replicated and
    overstate the peak. Each call returns a new tree, so a caller cannot modify the engine's own.

    Raises:
      RuntimeError: If no kernel has been compiled yet.
    """
    if not self._compiled:
      raise RuntimeError(
          "train_state_avals() is only meaningful after compile_kernels(): compiling is what places the "
          "optimizer state on the layout it runs on."
      )
    return jax.tree.map(maxtext_engine._to_aval, self._read_state_pure())  # pylint: disable=protected-access

  def _checkpoint_dir(self) -> str:
    """Returns no directory: Orbax creates whatever it is given, and this engine can never save."""
    return ""

  def _place_leaf(self, leaf: Any, target: jax.sharding.Sharding) -> Any:
    """Restates the aval on `target`: there is nothing to move, and `device_put` takes no aval."""
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=target)

  def _cannot_run(self, operation: str) -> RuntimeError:
    """Returns the error every execution entry point raises instead of running."""
    return RuntimeError(
        f"AbstractMaxTextEngine.{operation}() needs real weights, and this engine has only shapes. It "
        "exists to be traced and compiled: call `compile_kernels()`, or build a MaxTextTrainingEngine instead."
    )

  def fwd_bwd(self, payload: Any, **kwargs: Any) -> None:
    raise self._cannot_run("fwd_bwd")

  def update(self, **kwargs: Any) -> int:
    raise self._cannot_run("update")

  def fwd_only(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
    raise self._cannot_run("fwd_only")

  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    raise self._cannot_run("save_checkpoint")

  def restore_checkpoint(self, **kwargs: Any) -> Any:
    raise self._cannot_run("restore_checkpoint")


def with_engine_hlo_dump_defaults(argv: Sequence[str]) -> list[str]:
  """Returns `argv` with the HLO dump filters pointed at the kernels, if a dump was asked for.

  On `argv` rather than the config, and only under `dump_hlo`: `pyconfig.initialize` bakes the
  regex into `XLA_FLAGS` whether or not a dump was asked for, so widening it unconditionally
  would leave every compile writing a dump nobody asked for.
  """
  given = dict(arg.split("=", 1) for arg in argv if "=" in arg)
  if given.get("dump_hlo", "").strip().lower() not in ("true", "1"):
    return list(argv)
  return list(argv) + [f"{key}={value}" for key, value in HLO_DUMP_DEFAULTS.items() if key not in given]


def get_shaped_micro_batch(config: pyconfig.HyperParameters) -> dict[str, jax.ShapeDtypeStruct]:
  """Returns the abstract batch one `fwd_bwd` call is given.

  `maxtext_utils.get_shaped_batch` shapes the *global* batch, because `train.py`'s fused step
  folds gradient accumulation inside itself. The engine's caller drives one `fwd_bwd` per
  micro-batch, so compiling against the global batch would size every activation by the
  accumulation factor and report a peak memory no step ever reaches.
  """
  shaped_batch = maxtext_utils.get_shaped_batch(config)
  micro_batch_size = int(config.micro_batch_size_to_train_on)

  def to_micro_batch(aval: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    if not aval.shape or aval.shape[0] == micro_batch_size:
      return aval
    return jax.ShapeDtypeStruct((micro_batch_size,) + aval.shape[1:], aval.dtype)

  return {key: to_micro_batch(aval) for key, aval in shaped_batch.items()}


def compile_engine(
    config: pyconfig.HyperParameters, topology_mesh: jax.sharding.Mesh
) -> tuple[AbstractMaxTextEngine, dict[str, Any]]:
  """Lowers and compiles every kernel the engine runs, on `topology_mesh`, and keeps the engine.

  The engine is returned too because it knows the train state's layout, which the memory report
  needs and the executables do not carry; see `AbstractMaxTextEngine.train_state_avals`.

  Returns:
    `(engine, {kernel name: jax.stages.Compiled})`, the dict keyed by `KERNEL_NAMES`.
  """
  engine = AbstractMaxTextEngine(config, topology_mesh)
  return engine, engine.compile_kernels(get_shaped_micro_batch(config))


def compile_engine_kernels(config: pyconfig.HyperParameters, topology_mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """Lowers and compiles every kernel the engine runs, on `topology_mesh`.

  Returns:
    `{kernel name: jax.stages.Compiled}`, keyed by `KERNEL_NAMES`.
  """
  return compile_engine(config, topology_mesh)[1]


def kernel_save_path(compiled_trainstep_file: str, kernel_name: str) -> str:
  """Returns where one kernel's executable goes: `/tmp/engine.pickle` -> `/tmp/engine_fwd_bwd.pickle`."""
  stem, extension = os.path.splitext(compiled_trainstep_file)
  return f"{stem}_{kernel_name}{extension}"


def main(argv: Sequence[str]) -> None:
  """Compiles the engine's kernels for `compile_topology` and reports what they cost."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting training_engine/maxtext_engine_compile.py...", flush=True)

  config = pyconfig.initialize(with_engine_hlo_dump_defaults(argv))
  pre_train_compile.validate_config(config)
  if config.enable_diloco:
    raise NotImplementedError(
        "enable_diloco is not supported here: MaxTextTrainingEngine has no DiLoCo outer step, so the "
        "numbers reported would describe a different computation."
    )

  topology_mesh = pre_train_compile.get_topology_mesh(config)

  # After the topology is built, so this does not initialize the local backend first.
  max_utils.print_system_information()

  print("Jitting and compiling the engine's kernels...", flush=True)
  engine, compiled = compile_engine(config, topology_mesh)
  print("Jitting and compilation complete!", flush=True)

  # Saved first, so the executables are kept even if the report below raises.
  if config.compiled_trainstep_file != "":
    for name in KERNEL_NAMES:
      save_path = kernel_save_path(config.compiled_trainstep_file, name)
      pre_train_compile.save_compiled(compiled[name], save_path)
      print(f"Successfully saved compiled {name} kernel as {save_path}")

  for name in KERNEL_NAMES:
    print(f"--- {name} ---")
    print(f"Cost analysis: {compiled[name].cost_analysis()}")
    print(f"Memory analysis: {compiled[name].memory_analysis()}")

  # The raw analyses above see only each kernel's arguments. The report adds the train state each
  # kernel runs alongside without being passed, and names the peak.
  state = engine.train_state_avals()
  staging = weight_sync_staging(
      state,
      scan_layers=config.scan_layers,
      scan_axis=config.param_scan_axis,
      use_weight_converter=config.use_weight_converter,
  )
  print(format_memory_report(memory_report(compiled, state), staging), flush=True)

  print("Finished training_engine/maxtext_engine_compile.py successfully!", flush=True)

  if config.dump_hlo:
    # `upload_dump` deletes what it uploaded; say which filter was too narrow rather than raise
    # from the rmtree of a directory XLA never wrote.
    if not os.path.isdir(config.dump_hlo_local_dir):
      modules = ", ".join(f"jit_{name}" for name in KERNEL_NAMES)
      raise FileNotFoundError(
          f"dump_hlo is set but XLA wrote nothing to {config.dump_hlo_local_dir}: "
          f"dump_hlo_local_module_name={config.dump_hlo_local_module_name!r} matched none of the engine's "
          f"kernels ({modules})."
      )
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


if __name__ == "__main__":
  app.run(main)
