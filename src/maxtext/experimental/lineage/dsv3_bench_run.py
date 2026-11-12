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
from maxtext.experimental.lineage import dsv3_embed
from maxtext.experimental.lineage import dsv3_router
from maxtext.experimental.lineage import dsv3_sharding
from maxtext.experimental.lineage import dsv3_mla
from maxtext.experimental.lineage import dsv3_sparse_layer

_FUSED_LINEAR_COMBINE = flags.DEFINE_bool(
    "fused_linear_combine",
    False,
    "See `dsv3_sparse_layer.set_fused_linear_combine`.",
)
_EP_RAGGED_A2A = flags.DEFINE_bool(
    "ep_ragged_a2a",
    False,
    "Dispatch and combine the EP tokens with ragged all-to-alls (forward and"
    " backward). See `dsv3_sparse_layer.set_ep_ragged_a2a`.",
)
_EP_RAGGED_A2A_SC = flags.DEFINE_bool(
    "ep_ragged_a2a_sc",
    False,
    "With --ep_ragged_a2a, run the ragged all-to-alls on SparseCore 0.",
)
_OFFSET_SPLASH_FWD_BKVC = flags.DEFINE_integer(
    "offset_splash_fwd_block_kv_compute",
    0,
    "If > 0, forward kv compute block of the offset-causal splash kernels." " See `dsv3_mla.set_offset_splash_fwd_tuning`.",
)
_OFFSET_SPLASH_DKV_BKVC = flags.DEFINE_integer(
    "offset_splash_dkv_block_kv_compute",
    0,
    "If > 0, backward kv compute block of the offset-causal splash kernels.",
)
_OFFSET_SPLASH_FWD_SCHED = flags.DEFINE_bool(
    "offset_splash_fwd_experimental_scheduler",
    False,
    "See `dsv3_mla.set_offset_splash_fwd_tuning`.",
)
_FUSED_GATHER_GATE = flags.DEFINE_bool(
    "fused_gather_gate",
    False,
    "With --fused_linear_combine, also fuse permute_chunk + gate GMM + SiLU."
    " See `dsv3_sparse_layer.set_fused_gather_gate`.",
)
_FUSED_MOE_BWD = flags.DEFINE_bool(
    "fused_moe_bwd",
    False,
    "Fuse weight-only fp8 MoE backward gather+SiLU and gate dx+combine." " See `dsv3_sparse_layer.set_fused_moe_bwd`.",
)
_EP_COMBINE_A2A = flags.DEFINE_bool(
    "ep_combine_a2a",
    False,
    "Reduce-scatter the EP tokens (forward combine and token-gradient"
    " dispatch transpose) as a bf16 all-to-all and a local f32 sum. See"
    " `dsv3_router.set_ep_combine_a2a`.",
)
_EP_COMBINE_A2A_BWD = flags.DEFINE_bool(
    "ep_combine_a2a_bwd",
    True,
    "With --ep_combine_a2a, also use it for the token-gradient dispatch" " transpose in the backward pass.",
)
_EP_COMBINE_A2A_SC = flags.DEFINE_bool(
    "ep_combine_a2a_sc",
    False,
    "With --ep_combine_a2a, run the all-to-all on SparseCore 0 (dropping the"
    " forced-false SparseCore dense all-to-all offload compile flag so it"
    " resolves to auto) instead of on the TensorCore.",
)
_FSDP_GRAD_A2A = flags.DEFINE_bool(
    "fsdp_grad_a2a",
    False,
    "Reduce-scatter the routed expert weight gradients as a bf16 SparseCore"
    " all-to-all and a local f32 sum. See"
    " `dsv3_sparse_layer.set_fsdp_grad_a2a`.",
)
_EP_COMBINE_HIER = flags.DEFINE_enum(
    "ep_combine_hier",
    "",
    ["", "a2a_first", "rs_first", "rs_first_fused", "rs_first_ppermute"],
    "With --ep_combine_a2a, reduce-scatter in two SparseCore stages: a bf16"
    " all-to-all over the outer expert axis (z) and an f32 reduce-scatter over"
    " the inner one (core), in this order. See"
    " `dsv3_router.set_ep_combine_a2a`.",
)
_EP_COMBINE_KERNEL = flags.DEFINE_bool(
    "ep_combine_kernel",
    False,
    "Reduce-scatter the f32 EP tokens with the SparseCore Pallas kernel"
    " `ops.reduce_scatter_sc0` (f32 D2D/accumulation, bf16 ring sends). See"
    " `dsv3_router.set_ep_combine_a2a(kernel=True)`.",
)
_EP_COMBINE_KERNEL_MB = flags.DEFINE_integer(
    "ep_combine_kernel_microbatches",
    16,
    "num_microbatches of the --ep_combine_kernel kernel.",
)
_EMBED_GRAD_SC = flags.DEFINE_bool(
    "embed_grad_sc",
    False,
    "Reduce the f32 embedding table gradient on SparseCore 1. See" " `dsv3_embed.set_grad_sparse_core`.",
)
_HEAD_GRAD_SC = flags.DEFINE_bool(
    "head_grad_sc",
    False,
    "Reduce the f32 LM head kernel gradient on SparseCore 1 (overlapping the"
    " decoder backward). See `dsv3_embed.set_grad_sparse_core`.",
)
_ULYSSES_A2A_SC = flags.DEFINE_integer(
    "ulysses_a2a_sc",
    -1,
    "If >= 0, run the GQA Ulysses all-to-alls on this SparseCore. See" " `dsv3_mla.set_ulysses_a2a_sparse_core`.",
)
_SEQ_SPLIT_KV_GATHER = flags.DEFINE_bool(
    "seq_split_kv_gather",
    False,
    "With lineage_microbatch_split=seq, use KV all-gather (zigzag) CP instead"
    " of Ulysses. See `dsv3_mla.set_seq_split_kv_gather`.",
)
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
  dsv3_sparse_layer.set_fused_linear_combine(_FUSED_LINEAR_COMBINE.value)
  dsv3_embed.set_grad_sparse_core(_EMBED_GRAD_SC.value, _HEAD_GRAD_SC.value)
  dsv3_mla.set_seq_split_kv_gather(_SEQ_SPLIT_KV_GATHER.value)
  dsv3_mla.set_ulysses_a2a_sparse_core(_ULYSSES_A2A_SC.value if _ULYSSES_A2A_SC.value >= 0 else None)
  dsv3_sparse_layer.set_ep_ragged_a2a(_EP_RAGGED_A2A.value, sparse_core=_EP_RAGGED_A2A_SC.value)
  dsv3_sparse_layer.set_fused_gather_gate(_FUSED_GATHER_GATE.value)
  dsv3_sparse_layer.set_fused_moe_bwd(_FUSED_MOE_BWD.value)
  dsv3_sparse_layer.set_fsdp_grad_a2a(_FSDP_GRAD_A2A.value)
  if _ULYSSES_A2A_SC.value >= 0:
    # The SparseCore Ulysses all-to-alls need the dense a2a offload enabled.
    dsv3_router.request_sparse_core_dense_a2a(True)
  dsv3_mla.set_offset_splash_fwd_tuning(
      _OFFSET_SPLASH_FWD_BKVC.value or None,
      _OFFSET_SPLASH_FWD_SCHED.value,
      _OFFSET_SPLASH_DKV_BKVC.value or None,
  )
  ep_combine = _EP_COMBINE_A2A.value or _EP_COMBINE_KERNEL.value
  dsv3_sparse_layer.set_ep_combine_a2a(
      ep_combine,
      ep_combine and _EP_COMBINE_A2A_BWD.value,
      sparse_core=_EP_COMBINE_A2A_SC.value,
      hierarchical=_EP_COMBINE_HIER.value or None,
      kernel=_EP_COMBINE_KERNEL.value,
      kernel_microbatches=_EP_COMBINE_KERNEL_MB.value,
  )
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
        compiler_options=dsv3_router.adjust_compiler_options(
            dsv3_aot_compile.parse_libtpu_flags_to_dict(str(getattr(config, "compile_xla_flags", "") or ""))
        )
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
