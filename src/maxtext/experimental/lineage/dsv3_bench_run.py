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

r"""Runs Lineage DSv3 training steps on real TPUs and times them.

Shares the configuration, model, sharding and train step of
`dsv3_aot_compile`, so its step time is directly comparable with the AOT
compiled program. The weights are randomly initialized and the tokens are
synthetic.

Usage (on every host of a TPU slice):

  python3 -m maxtext.experimental.lineage.dsv3_bench_run \
      src/maxtext/configs/base.yml model_name=deepseek3-671b-lineage \
      use_lineage=True ... steps=20 base_output_directory=gs://<bucket>/run \
      profiler=xplane profiler_steps=1 skip_first_n_steps_for_profiler=8

Prints the time of every step and the median of the steps after the first
`skip_first_n_steps_for_profiler`. With `profiler=xplane`, process 0 captures an
XProf trace of `profiler_steps` steps after them, writes it locally and copies
it to `base_output_directory/tensorboard`; profiling errors are only logged.
"""

from collections.abc import Sequence
import functools
import os
import statistics
import time
from typing import Any

from absl import app
from absl import flags
from flax.core import spmd
import jax
import jax.numpy as jnp
from maxtext.experimental.lineage import dsv3_aot_compile
from maxtext.experimental.lineage import dsv3_router
from maxtext.experimental.lineage import dsv3_sharding
from maxtext.experimental.lineage import dsv3_sparse_layer

_SPLIT_SMALL_W_SCATTERS = flags.DEFINE_bool(
    "split_small_w_scatters",
    True,
    "See `dsv3_sparse_layer.set_split_small_w_scatters`.",
)
_REUSE_INITIAL_STATE = flags.DEFINE_bool(
    "reuse_initial_state",
    True,
    "Run every step on the initial state (the updated state is discarded, so"
    " the state is not donated). The synthetic random state diverges after one"
    " update, and NaN weights make the routing degenerate (all tokens go to a"
    " few experts), so updating it times load imbalance, not the model.",
)

_VOCAB_KEY = "vocab_size"


def _create_device_mesh(
    config: dsv3_aot_compile.AOTConfig,
) -> jax.sharding.Mesh:
  """Builds the mesh of `config` over the devices of this job."""
  devices = jax.devices()
  num_slices = int(config.num_slices)
  ici_parallelism = dsv3_aot_compile.fill_unspecified_mesh_axes(config.ici_parallelism, len(devices) // num_slices, "ICI")
  dcn_parallelism = (
      dsv3_aot_compile.fill_unspecified_mesh_axes(config.dcn_parallelism, num_slices, "DCN") if num_slices > 1 else None
  )
  devices_array = dsv3_sharding.create_device_mesh(
      ici_parallelism,
      devices,
      dcn_parallelism=dcn_parallelism,
      allow_split_physical_axes=bool(config.allow_split_physical_axes),
      log_fn=print,
  )
  return jax.sharding.Mesh(
      devices_array,
      config.mesh_axes,
      axis_types=(jax.sharding.AxisType.Explicit,) * len(config.mesh_axes),
  )


def _synthetic_batch(
    config: dsv3_aot_compile.AOTConfig,
    data_sharding: jax.sharding.NamedSharding,
    seed: int,
) -> dict[str, jax.Array]:
  """A batch of random tokens, one unpacked sequence per row."""
  shape = (
      int(config.global_batch_size_to_train_on),
      int(config.max_target_length),
  )

  @functools.partial(jax.jit, out_shardings=data_sharding)
  def make_tokens(key):
    return jax.random.randint(key, shape, 0, int(getattr(config, _VOCAB_KEY)), dtype=jnp.int32)

  @functools.partial(jax.jit, out_shardings=data_sharding)
  def make_positions():
    return jnp.broadcast_to(jnp.arange(shape[1], dtype=jnp.int32)[None], shape)

  @functools.partial(jax.jit, out_shardings=data_sharding)
  def make_ones():
    return jnp.ones(shape, jnp.int32)

  key = jax.random.key(seed)
  inputs = make_tokens(key)
  targets = make_tokens(jax.random.fold_in(key, 1))
  positions = make_positions()
  segmentation = make_ones()
  return {
      "inputs": inputs,
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets": targets,
      "targets_position": positions,
      "targets_segmentation": segmentation,
  }


def _init_state(abstract_state: Any, state_shardings: Any, compile_only: bool = False) -> Any:
  """Synthetic train state with the shapes and shardings of `abstract_state`.

  Initializing the model's real initializers in one program materializes
  unsharded temporaries far beyond HBM, so every leaf is generated directly in
  its sharding instead: matrices are N(0, 0.02^2), vectors named `*scale*` are
  ones, and the other vectors (biases, router bias), the optimizer state and
  integer leaves are zeros. This is enough to time the steps.

  Args:
    abstract_state: `nnx.State` of `jax.ShapeDtypeStruct` leaves.
    state_shardings: The matching `NamedSharding`s.
    compile_only: Only compile the initialization (e.g. for a
      `compile_topology`) and return the compiled program.

  Returns:
    The state, or the compiled initialization with `compile_only`.
  """

  def make_leaf(index, path, leaf):
    name = jax.tree_util.keystr(path).lower()
    sharding = leaf.sharding
    if not jnp.issubdtype(leaf.dtype, jnp.floating) or "optimizer" in name:
      return jnp.zeros(leaf.shape, leaf.dtype, out_sharding=sharding)
    if len(leaf.shape) >= 2:
      # Threefry (unlike the default `unsafe_rbg`) partitions, so each device
      # only generates its own shard.
      key = jax.random.fold_in(jax.random.key(0, impl="threefry2x32"), index)
      return jax.random.normal(key, leaf.shape, leaf.dtype, out_sharding=sharding) * jnp.asarray(0.02, leaf.dtype)
    if "scale" in name:
      return jnp.ones(leaf.shape, leaf.dtype, out_sharding=sharding)
    return jnp.zeros(leaf.shape, leaf.dtype, out_sharding=sharding)

  @functools.partial(jax.jit, out_shardings=state_shardings)
  def init():
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(abstract_state)
    leaves = [make_leaf(i, path, leaf) for i, (path, leaf) in enumerate(paths_and_leaves)]
    return jax.tree_util.tree_unflatten(treedef, leaves)

  if compile_only:
    return init.lower().compile()
  return init()


def _profiler_options(
    config: dsv3_aot_compile.AOTConfig,
) -> jax.profiler.ProfileOptions:
  """XProf options, as MaxText's `enable_tpu_profiling_options`.

  Tracing every chip and SparseCore tile of a host makes multi-GB traces that
  are slow to load, so with `enable_tpu_profiling_options` (the default here)
  only `tpu_num_chips_to_profile_per_task` chips per host and
  `tpu_num_sparse_cores_to_trace` SparseCores with
  `tpu_num_sparse_core_tiles_to_trace` tiles each are traced.

  Args:
    config: The run config.

  Returns:
    The profiler options.
  """
  options = jax.profiler.ProfileOptions()
  if bool(getattr(config, "enable_tpu_profiling_options", True)):
    options.advanced_configuration = {
        "tpu_num_chips_to_profile_per_task": int(config.tpu_num_chips_to_profile_per_task),
        "tpu_num_sparse_core_tiles_to_trace": int(config.tpu_num_sparse_core_tiles_to_trace),
        "tpu_num_sparse_cores_to_trace": int(config.tpu_num_sparse_cores_to_trace),
    }
  return options


def _upload_dir(local_dir: str, dest: str) -> None:
  """Uploads `local_dir` recursively to `dest` (gs://bucket/prefix or a local path)."""
  if not dest.startswith("gs://"):
    import shutil  # pylint: disable=g-import-not-at-top

    shutil.copytree(local_dir, dest, dirs_exist_ok=True)
    return
  from google.cloud import storage  # pylint: disable=g-import-not-at-top

  bucket_name, _, prefix = dest[len("gs://") :].partition("/")
  bucket = storage.Client().bucket(bucket_name)
  for root, _, files in os.walk(local_dir):
    for name in files:
      path = os.path.join(root, name)
      blob = prefix.rstrip("/") + "/" + os.path.relpath(path, local_dir)
      bucket.blob(blob).upload_from_filename(path)
      print(f"Uploaded gs://{bucket_name}/{blob}", flush=True)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  jax.config.update("jax_threefry_partitionable", True)
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  print("Starting dsv3_bench_run...", flush=True)
  config = dsv3_aot_compile.load_aot_config(argv)
  if not config.use_lineage:
    raise ValueError("dsv3_bench_run requires use_lineage=True.")
  dsv3_aot_compile.validate_supported(config)
  dsv3_sparse_layer.set_split_small_w_scatters(_SPLIT_SMALL_W_SCATTERS.value)
  dsv3_router.set_random_routing(bool(getattr(config, "use_random_routing", False)))
  # With `compile_topology`, only compile the programs for that topology (a
  # memory check of the initialization and step on CPU).
  compile_only = bool(config.using_compiled_topology)
  mesh = dsv3_aot_compile.create_topology_mesh(config) if compile_only else _create_device_mesh(config)
  is_main = jax.process_index() == 0
  if is_main:
    print(
        f"JAX {jax.__version__}: {jax.process_count()} processes," f" {jax.device_count()} devices, mesh {dict(mesh.shape)}",
        flush=True,
    )

  (
      (abstract_state, _),
      _,
      state_shardings,
      graphdef,
  ) = dsv3_aot_compile.get_shaped_inputs_single_pass(mesh, config)
  data_sharding = dsv3_aot_compile.get_input_data_sharding(config, mesh)

  with jax.set_mesh(mesh), spmd.logical_axis_rules(config.logical_axis_rules):
    if compile_only:
      init_compiled = _init_state(abstract_state, state_shardings, compile_only=True)
      print(f"Init memory analysis: {init_compiled.memory_analysis()}")
      return
    t0 = time.perf_counter()
    state = _init_state(abstract_state, state_shardings)
    jax.block_until_ready(state)
    if is_main:
      print(f"Initialized state in {time.perf_counter() - t0:.1f} s", flush=True)

    batch = _synthetic_batch(config, data_sharding, seed=0)
    step_fn = functools.partial(dsv3_aot_compile.train_step, graphdef, config)
    setattr(step_fn, "__name__", "train_step")
    jitted = jax.jit(
        step_fn,
        in_shardings=(state_shardings, data_sharding),
        out_shardings=(state_shardings, None),
        donate_argnums=() if _REUSE_INITIAL_STATE.value else 0,
    )
    t0 = time.perf_counter()
    compiled = jitted.lower(state, batch).compile(
        compiler_options=dsv3_aot_compile.parse_libtpu_flags_to_dict(str(getattr(config, "compile_xla_flags", "") or ""))
    )
    if is_main:
      print(f"Compiled train step in {time.perf_counter() - t0:.1f} s")
      print(f"Memory analysis: {compiled.memory_analysis()}", flush=True)

    num_steps = int(config.steps)
    skip = int(config.skip_first_n_steps_for_profiler)
    profile_steps = int(config.profiler_steps)
    profile = str(config.profiler or "") == "xplane" and profile_steps > 0
    profile_dir = os.path.join(str(config.base_output_directory), "tensorboard")
    local_profile_dir = "/tmp/lineage_xprof/tensorboard"
    step_times = []
    for step in range(num_steps):
      # Only process 0 traces, to keep the profile small.
      if profile and is_main and step == skip:
        try:
          jax.profiler.start_trace(local_profile_dir, profiler_options=_profiler_options(config))
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(f"WARNING: start_trace failed: {e!r}", flush=True)
          profile = False
      t0 = time.perf_counter()
      new_state, metrics = compiled(state, batch)
      jax.block_until_ready((new_state, metrics))
      if not _REUSE_INITIAL_STATE.value:
        state = new_state
      del new_state
      step_time = time.perf_counter() - t0
      step_times.append(step_time)
      if profile and is_main and step == skip + profile_steps - 1:
        try:
          jax.profiler.stop_trace()
          print(f"Wrote local XProf trace to {local_profile_dir}", flush=True)
          _upload_dir(local_profile_dir, profile_dir)
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(f"WARNING: profile stop/upload failed: {e!r}", flush=True)
      if is_main:
        loss = float(metrics["scalar"]["learning/loss"])
        print(f"step {step}: {step_time:.4f} s, loss {loss:.4f}", flush=True)

    timed = step_times[skip:] or step_times
    if is_main:
      print(
          f"Median step time over steps {skip}..{num_steps - 1}:"
          f" {statistics.median(timed):.4f} s (min {min(timed):.4f} s,"
          f" max {max(timed):.4f} s)",
          flush=True,
      )


if __name__ == "__main__":
  app.run(main)
