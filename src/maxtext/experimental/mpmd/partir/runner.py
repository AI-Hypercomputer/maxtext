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
"""Assembles a runnable MPMD pipeline from hyperparameters.

``train.py``, ``train_compile.py``, ``parity.py`` and the unit tests all build
their model, data and training step through :func:`build`, and lower it with
:meth:`Runnable.lower`, so a lowering and the run it is supposed to predict
describe the same program.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

from flax import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.mpmd.partir import config as config_lib
from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import models
from maxtext.experimental.mpmd.partir import pipeline as pipeline_lib
from maxtext.experimental.mpmd.partir import train_step as train_step_lib


def make_synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    abstract: bool = False,
) -> dict[str, Any]:
  """Builds a deterministic synthetic training batch.

  The same `seed` always yields the same tokens, which is what allows the MPMD
  run and the SPMD reference to be compared value-for-value.

  Args:
    batch_size: Global batch size (across one stage mesh).
    seq_len: Sequence length.
    vocab_size: Vocabulary size to sample token ids from.
    seed: Seed for the token sampler.
    abstract: When True return ``jax.ShapeDtypeStruct`` placeholders instead of
      concrete arrays, for ahead-of-time lowering.

  Returns:
    A dict with the keys listed in :data:`train_step.BATCH_KEYS`.
  """
  if abstract:
    spec = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)
    return {key: spec for key in train_step_lib.BATCH_KEYS}

  rng = np.random.default_rng(seed)
  # `seq_len + 1` tokens so that targets are the inputs shifted by one.
  tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len + 1), dtype=np.int32)
  positions = np.tile(np.arange(seq_len + 1, dtype=np.int32), (batch_size, 1))
  # Every position is a real token: segment id 1, no padding.
  segmentation = np.ones((batch_size, seq_len), dtype=np.int32)
  return {
      "inputs": jnp.asarray(tokens[:, :-1]),
      "inputs_position": jnp.asarray(positions[:, :-1]),
      "inputs_segmentation": jnp.asarray(segmentation),
      "targets": jnp.asarray(tokens[:, 1:]),
      "targets_position": jnp.asarray(positions[:, 1:]),
      "targets_segmentation": jnp.asarray(segmentation),
  }


shard_tree_on_fsdp = models.shard_tree_on_fsdp


def place_on_stage_meshes(inputs: Any, input_specs: Any) -> Any:
  """Commits `inputs` onto the meshes an MPMD executable expects.

  ``Lowered.function_named_shardings.input_specs`` is a *flat* sequence with
  one :class:`jax.sharding.NamedSharding` per flattened argument leaf, and
  different leaves generally live on different stage meshes. It is therefore
  not a pytree prefix of the arguments: handing the pair
  ``(state, batch)`` and the flat specs to :func:`jax.device_put` in one go
  produces arrays the executable rejects with::

      INVALID_ARGUMENT: Input signature does not match the signature of the
      compiled function.

  Args:
    inputs: The pytree of arguments, in the order they were lowered with.
    input_specs: The flat shardings reported by the lowering.

  Returns:
    `inputs` with every leaf committed to its mesh.

  Raises:
    ValueError: If the number of leaves and shardings disagree, which means
      the arguments are not the ones that were lowered.
  """
  leaves, treedef = jax.tree_util.tree_flatten(inputs)
  specs = jax.tree_util.tree_leaves(input_specs)
  if len(leaves) != len(specs):
    raise ValueError(
        f"Lowering expects {len(specs)} input arrays but {len(leaves)} were"
        " supplied; the arguments differ from the ones lowered."
    )
  return jax.tree_util.tree_unflatten(
      treedef,
      [jax.device_put(leaf, spec) for leaf, spec in zip(leaves, specs)],
  )


@dataclasses.dataclass(frozen=True)
class Runnable:
  """A fully assembled, not-yet-lowered MPMD training program.

  Attributes:
    hparams: The hyperparameters this was built from.
    maxtext_config: The materialised MaxText config.
    plan: The MPMD pipeline plan (topology, schedule, mesh assignment).
    graphdef: Static model structure.
    state: Initial optimizer state (concrete, or abstract when ``abstract=True``
      was requested).
    batch: A training batch (concrete or abstract).
    train_step: The function to lower, ``(state, batch) -> (state, loss)``.
    in_shardings: Input shardings for ``mpmd.jit``.
    batch_size: Global batch size.
  """

  hparams: hparams_lib.Hparams
  maxtext_config: Any
  plan: pipeline_lib.PipelinePlan
  graphdef: models.PipelineGraphDef
  state: train_state.TrainState
  batch: dict[str, Any]
  train_step: Any
  in_shardings: tuple[Any, Any]
  batch_size: int

  @property
  def out_shardings(self) -> tuple[Any, None]:
    """Output shardings matching ``(new_state, loss)``."""
    return (self.in_shardings[0], None)

  def lower(self) -> Any:
    """Lowers :attr:`train_step` with ``mpmd.jit`` and pins its transfers.

    The state is donated and its shardings are pinned on both sides, so that
    step N's state is what step N+1 accepts (see ``pipeline._make_mpmd_config``);
    :func:`pipeline.pin_ifrt_transfers_to_producers` keeps each microbatch's
    transfers apart.

    Returns:
      The ``MpmdLowered`` program, to compile on Pathways.
    """
    lowered = (
        pipeline_lib.shardy_mpmd()
        .jit(
            self.train_step,
            mpmd_config=self.plan.mpmd_config,
            in_shardings=self.in_shardings,
            out_shardings=self.out_shardings,
            donate_argnums=(0,),
        )
        .lower(self.state, self.batch)
    )
    return pipeline_lib.pin_ifrt_transfers_to_producers(lowered)

  def describe(self) -> str:
    """Returns a multi-line summary of model, pipeline and batch."""
    return "\n".join(
        [
            "MaxText configuration:",
            config_lib.describe(self.maxtext_config, self.hparams),
            "Pipeline plan:",
            self.plan.describe(),
            (
                f"  global batch   : {self.batch_size}"
                f" ({self.batch_size // self.hparams.num_microbatches} per"
                " microbatch)"
            ),
        ]
    )


def build(
    hparams: hparams_lib.Hparams,
    *,
    devices: Sequence[Any] | None = None,
    mode: train_step_lib.ExecutionMode = train_step_lib.ExecutionMode.MPMD,
    abstract: bool = False,
    config_overrides: dict[str, Any] | None = None,
    check_microbatches: bool = False,
) -> Runnable:
  """Builds everything needed to lower or run one training step.

  Args:
    hparams: Hyperparameters.
    devices: Devices for the stage meshes. Defaults to ``jax.devices()``.
    mode: Whether to build the MPMD (pipelined) or SPMD (reference) step.
    abstract: Build abstract (shape-only) state and data, for AOT compilation on
      a machine with no accelerators.
    config_overrides: Extra MaxText settings, for tests.
    check_microbatches: Refuse a microbatch that does not split over a stage's
      batch-sharding devices (:func:`hparams.check_microbatch_split`), which only
      the TPUs would otherwise reject. The entry points ask for it; tests build
      such configurations on CPU.

  Returns:
    The assembled :class:`Runnable`.

  Raises:
    ValueError: If the global batch is not divisible by the microbatch count,
      or (with `check_microbatches`) a microbatch not by the stage's
      batch-sharding devices.
  """
  maxtext_config = config_lib.build_maxtext_config(hparams, overrides=config_overrides)
  plan = pipeline_lib.build_plan(hparams, devices=devices)
  if mode is train_step_lib.ExecutionMode.MPMD:
    # Builds the Shardy MPMD configuration now, so that a missing shardy-mpmd
    # or a configuration Shardy refuses fails before the model is initialised.
    _ = plan.mpmd_config
  if check_microbatches:
    # Before any parameter or optimizer state exists: a large model takes
    # minutes to initialise.
    hparams_lib.check_microbatch_split(
        hparams,
        devices_per_stage=plan.topology.devices_per_stage,
        data_parallel_size=plan.topology.data_parallel_size,
    )

  # The batch is sharded across a *single* stage's devices: every microbatch
  # traverses all stages, so the data-parallel width is the width of one stage,
  # not of the whole slice.
  batch_size = int(maxtext_config.per_device_batch_size * plan.topology.devices_per_stage)
  if batch_size % hparams.num_microbatches:
    raise ValueError(
        f"Global batch size ({batch_size} = per_device_batch_size"
        f" {maxtext_config.per_device_batch_size} x"
        f" {plan.topology.devices_per_stage} devices per stage) must be"
        f" divisible by num_microbatches ({hparams.num_microbatches})."
    )

  batch = make_synthetic_batch(
      batch_size=batch_size,
      seq_len=maxtext_config.max_target_length,
      vocab_size=maxtext_config.vocab_size,
      seed=hparams.seed,
      abstract=abstract,
  )

  fsdp_size = plan.topology.fsdp_size
  if abstract:

    def _init_abstract():
      model = models.PipelinedTransformer(
          config=maxtext_config,
          # `layer_mesh`, not `sharding_mesh`: building DeepSeek's MLA
          # attention reads `mesh.devices` to choose an attention kernel,
          # which an abstract mesh cannot answer.
          mesh=plan.layer_mesh,
          rngs=nnx.Rngs(hparams.seed),
          remat_policy=hparams.mpmd_remat_policy,
          stage_layer_counts=hparams.stage_layer_counts,
      )
      gdef, params = model.split()
      st = train_step_lib.create_train_state(params, hparams.learning_rate)
      return (
          gdef.embed_def,
          gdef.embed_rest,
          gdef.layer_defs,
          gdef.layer_rests,
          gdef.unembed_def,
          gdef.unembed_rest,
          st,
      )

    (
        embed_def,
        embed_rest,
        layer_defs,
        layer_rests,
        unembed_def,
        unembed_rest,
        state,
    ) = jax.eval_shape(_init_abstract)

    try:
      cpu_device = jax.devices("cpu")[0]
    except RuntimeError:
      cpu_device = None

    def _materialize_rest(leaf):
      if isinstance(leaf, jax.ShapeDtypeStruct):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
          if cpu_device is not None:
            with jax.default_device(cpu_device):
              return jax.random.key(0)
          return jax.random.key(0)
        return np.zeros(leaf.shape, dtype=leaf.dtype)
      return leaf

    is_deepseek = maxtext_config.decoder_block == models.DecoderBlockType.DEEPSEEK
    graphdef = models.PipelineGraphDef(
        config=maxtext_config,
        embed_def=embed_def,
        embed_rest=jax.tree.map(_materialize_rest, embed_rest),
        layer_defs=tuple(layer_defs),
        layer_rests=tuple(jax.tree.map(_materialize_rest, layer_rests)),
        unembed_def=unembed_def,
        unembed_rest=jax.tree.map(_materialize_rest, unembed_rest),
        is_deepseek=is_deepseek,
        fsdp_size=fsdp_size,
        mesh=plan.layer_mesh,
        remat_policy=hparams.mpmd_remat_policy,
        stage_layer_counts=hparams.stage_layer_counts,
    )
  else:
    unit_shard_fn = None
    init_mesh = plan.layer_mesh
    if mode is train_step_lib.ExecutionMode.MPMD:

      def _shard_unit(unit_name: str, unit_params: nnx.State) -> nnx.State:
        stage_mesh_name = plan.name_to_mesh[unit_name]
        stage_mesh = plan.topology.meshes[stage_mesh_name]
        return shard_tree_on_fsdp(stage_mesh, unit_params, fsdp_size, device_put=True)

      unit_shard_fn = _shard_unit
    else:
      # In SPMD reference mode (`parity.py`), `spmd_state` and `spmd_batch` are
      # unsharded single-device arrays on device 0. Using a 1-device mesh with
      # the same axis names ensures any internal `jax.shard_map(mesh=self.mesh)`
      # (such as Qwen3.5 GatedDeltaNet or Splash Attention) executes on that
      # same single device rather than demanding a multi-device stage mesh.
      single_dev = np.asarray(plan.layer_mesh.devices).flat[:1].reshape((1,) * len(plan.layer_mesh.axis_names))
      init_mesh = jax.sharding.Mesh(single_dev, plan.layer_mesh.axis_names)

    graphdef, params = models.PipelinedTransformer.init_split(
        config=maxtext_config,
        mesh=init_mesh,
        rngs=nnx.Rngs(hparams.seed),
        unit_shard_fn=unit_shard_fn,
        remat_policy=hparams.mpmd_remat_policy,
        stage_layer_counts=hparams.stage_layer_counts,
    )
    state = train_step_lib.create_train_state(params, hparams.learning_rate)

  state_shardings = shard_tree_on_fsdp(plan.sharding_mesh, state, fsdp_size, device_put=False)
  batch_shardings = shard_tree_on_fsdp(plan.sharding_mesh, batch, fsdp_size, device_put=False)

  train_step = train_step_lib.make_train_step(
      graphdef,
      vocab_size=maxtext_config.vocab_size,
      num_microbatches=hparams.num_microbatches,
      mode=mode,
  )

  return Runnable(
      hparams=hparams,
      maxtext_config=maxtext_config,
      plan=plan,
      graphdef=graphdef,
      state=state,
      batch=batch,
      train_step=train_step,
      in_shardings=(state_shardings, batch_shardings),
      batch_size=batch_size,
  )
