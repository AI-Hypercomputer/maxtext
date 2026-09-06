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
"""

import os
from typing import Any, Sequence

from absl import app
from flax import nnx
import jax
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
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
# kernels lower as `jit_first_kernel`, `jit_accum_kernel` and `jit__update_kernel`, so on those
# defaults the dump comes back empty.
HLO_DUMP_DEFAULTS = {
    "dump_hlo_local_module_name": "jit_.*kernel",
    "dump_hlo_module_name": "kernel",
}


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

  def __init__(self, training_config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh) -> None:
    """Initializes an engine that can be lowered but not run.

    Args:
      training_config: MaxText HyperParameters configuration instance.
      mesh: The mesh to compile against, typically a topology this host does not own.

    Raises:
      ValueError: If `mesh` is None. With no weights there is no device set to read one off.
    """
    if mesh is None:
      raise ValueError(
          "AbstractMaxTextEngine requires a mesh: with no weights there is nothing to read a device set "
          "off, and the point of the abstract path is to compile against a mesh this host does not own -- "
          "build one with `trainers.pre_train.train_compile.get_topology_mesh`."
      )
    super().__init__(training_config, mesh=mesh)

  def _build_model(self, wrap_with_tunix_adapter: bool, tokenizer_pad_id: int | None) -> Any:
    """Returns the model with `jax.ShapeDtypeStruct` weights on their real shardings.

    The same `create_nnx_abstract_model` call `from_pretrained` makes before it materializes
    anything, minus the checkpoint load -- so no weights, no HF token and no network.
    """
    del wrap_with_tunix_adapter, tokenizer_pad_id  # `__init__` accepts neither.
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(
        model_creation_utils.verify_and_sync_scan_layers(self._config),
        self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
    )
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

    Both run under the engine's own `_sharding_ctx`, so the rules the MaxText layers are written
    against are the live ones rather than a second copy that can drift from them.
    """
    model_graphdef, model_pure = nnx.split(self._model)

    def build(model_state):
      model = nnx.merge(model_graphdef, model_state)
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    propagation_mesh = _propagation_mesh(self._mesh)
    with self._sharding_ctx():
      state_graphdef, _ = nnx.split(nnx.eval_shape(build, model_pure))
      # Displaces the real mesh for this trace only: the one above needs no propagation, and a
      # stand-in set around it collides with the config's own AbstractMesh under `shard_mode=auto`.
      with jax.set_mesh(propagation_mesh):
        state_pure = jax.eval_shape(
            lambda model_state: nnx.split(build(model_state))[1],
            jax.tree.map(lambda aval: _rehome_aval(aval, propagation_mesh), model_pure),
        )
    return nnx.merge(state_graphdef, jax.tree.map(lambda aval: _rehome_aval(aval, self._mesh), state_pure))

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


def compile_engine_kernels(config: pyconfig.HyperParameters, topology_mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """Lowers and compiles every kernel the engine runs, on `topology_mesh`.

  Returns:
    `{kernel name: jax.stages.Compiled}`, keyed by `KERNEL_NAMES`.
  """
  return AbstractMaxTextEngine(config, topology_mesh).compile_kernels(get_shaped_micro_batch(config))


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
  compiled = compile_engine_kernels(config, topology_mesh)
  print("Jitting and compilation complete!", flush=True)

  for name in KERNEL_NAMES:
    print(f"--- {name} ---")
    print(f"Cost analysis: {compiled[name].cost_analysis()}")
    print(f"Memory analysis: {compiled[name].memory_analysis()}")

  if config.compiled_trainstep_file != "":
    for name in KERNEL_NAMES:
      save_path = kernel_save_path(config.compiled_trainstep_file, name)
      pre_train_compile.save_compiled(compiled[name], save_path)
      print(f"Successfully saved compiled {name} kernel as {save_path}")

  print("Finished training_engine/maxtext_engine_compile.py successfully!", flush=True)

  if config.dump_hlo:
    # `upload_dump` deletes what it uploaded; say which filter was too narrow rather than raise
    # from the rmtree of a directory XLA never wrote.
    if not os.path.isdir(config.dump_hlo_local_dir):
      raise FileNotFoundError(
          f"dump_hlo is set but XLA wrote nothing to {config.dump_hlo_local_dir}: "
          f"dump_hlo_local_module_name={config.dump_hlo_local_module_name!r} matched none of the engine's "
          f"kernels (jit_first_kernel, jit_accum_kernel, jit__update_kernel)."
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
