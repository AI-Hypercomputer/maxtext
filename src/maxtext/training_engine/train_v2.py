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

"""Training loop and data loading for pre-training using MaxTextTrainingEngine (train_v2).

This module implements the production V2 pre-training driver on top of
`MaxTextTrainingEngine`. It provides:
- Dataloader initialization with context parallelism and ramp-up batch support
- Pre-training loop iteration delegating fwd_bwd, update, and step metric recording to MaxTextTrainingEngine
- Periodic evaluation, checkpointing, profiling, and HLO dumping
- Standard top-level execution entry points (initialize, run, get_train_func, main) with elastic retry support
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib
import functools
import os
from typing import Any

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import pathwaysutils
import tensorflow as tf

from maxtext.common.common_types import ReorderStrategy
from maxtext.common.data_loader import create_dataloader
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.configs import pyconfig
from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import (
    elastic_utils,
    exceptions,
    max_logging,
    max_utils,
    maxtext_utils,
    sharding,
    train_utils,
)
from maxtext.utils.rampup_batch import create_rampup_manager

VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


class _StepHolder:
  """Helper class to provide latest_step() for RampupBatchManager."""

  def __init__(self, step: int):
    self._step = step

  def latest_step(self) -> int | None:
    return self._step if self._step > 0 else None


def setup_dataloaders(
    config: pyconfig.HyperParameters,
    mesh: jax.sharding.Mesh,
    goodput_recorder: Any = None,
    checkpoint_manager: Any | None = None,
    start_step: int | None = None,
) -> tuple[Any, Any, Any]:
  """Sets up pre-training dataloaders, data iterators, and rampup manager.

  Args:
    config: Hyperparameters configuration instance.
    mesh: The SPMD device mesh.
    goodput_recorder: Optional GoodputRecorder.
    checkpoint_manager: Optional CheckpointManager for checking start step in rampup.
    start_step: Optional start step integer for rampup calculation.

  Returns:
    A tuple of (train_data_loader, eval_data_iterator, rampup_manager).

  Raises:
    ValueError: If sequence packing is used with synthetic dataset under
      context parallelism, or if an invalid context parallelism strategy is used.
  """
  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    if start_step is not None:
      step_holder = _StepHolder(start_step)
    else:
      step_holder = getattr(checkpoint_manager, "_checkpoint_manager", checkpoint_manager)
    rampup_manager = create_rampup_manager(config, step_holder)

    context_parallel_size = mesh.shape.get(getattr(config, "context_sharding", "context"), 1) if mesh else 1
    context_parallel_strategy = getattr(config, "context_parallel_strategy", "").lower()

    if context_parallel_size > 1 and getattr(config, "packing", False):
      if getattr(config, "dataset_type", "") == "synthetic":
        raise ValueError(
            "Context parallelism with sequence packing is not supported with synthetic data. "
            "Please disable sequence packing (set packing=False)."
        )
      if context_parallel_strategy not in ("all_gather", "ring"):
        raise ValueError(
            "Context parallelism with sequence packing supports context_parallel_strategy='all_gather' or 'ring'."
        )
      if (
          getattr(config, "hardware", "") in ("gpu", "gpu_multiprocess")
          and getattr(config, "attention", "") == "cudnn_flash_te"
          and not (context_parallel_strategy == "ring" and getattr(config, "context_parallel_load_balance", False))
      ):
        raise ValueError("Packing is only supported for load balanced ring attention with context parallelism for GPU.")

    with jax.set_mesh(mesh) if mesh else contextlib.nullcontext():
      if context_parallel_size > 1 and getattr(config, "context_parallel_load_balance", False):
        reorder_strategy_cfg = getattr(
            config,
            "context_parallel_reorder_strategy",
            ReorderStrategy.AUTO,
        )
        if reorder_strategy_cfg == ReorderStrategy.AUTO:
          reorder_strategy = (
              ReorderStrategy.STRIPED
              if (
                  getattr(config, "packing", False)
                  and context_parallel_strategy == "ring"
                  and getattr(config, "hardware", "") in ("gpu", "gpu_multiprocess")
              )
              else ReorderStrategy.DUAL_CHUNK_SWAP
          )
        else:
          reorder_strategy = reorder_strategy_cfg

        reorder_fn = maxtext_utils.get_reorder_callable(
            context_parallel_size,
            getattr(config, "shard_mode", None),
            reorder_strategy,
            getattr(config, "hardware", ""),
        )
        data_iterator = map(reorder_fn, data_iterator)
        if eval_data_iterator:
          eval_data_iterator = map(reorder_fn, eval_data_iterator)

    train_data_loader = create_dataloader(config, mesh, data_iterator, goodput_recorder, rampup_manager)

  return train_data_loader, eval_data_iterator, rampup_manager


def load_next_batch(
    data_loader: Any,
    rampup_manager: Any = None,
    goodput_recorder: Any = None,
) -> Any:
  """Loads the next batch from the data loader and optionally records goodput.

  Args:
    data_loader: The DataLoader instance.
    rampup_manager: Optional RampupBatchManager instance.
    goodput_recorder: Optional GoodputRecorder.
    step: Optional current training step integer.

  Returns:
    The next pre-training data batch dictionary.
  """
  with maybe_record_goodput(goodput_recorder, GoodputEvent.DATA_LOADING):
    return data_loader.load_next_batch(rampup_manager=rampup_manager)


def run_evaluation(
    engine: maxtext_engine.MaxTextTrainingEngine,
    config: pyconfig.HyperParameters,
    mesh: jax.sharding.Mesh | None,
    eval_data_iterator: Iterator[Any] | None,
    step: int,
) -> list[abstract_engine.MetricsBuffer]:
  """Executes evaluation loop over `eval_data_iterator` using `engine`.

  Args:
    engine: Instance of MaxTextTrainingEngine.
    config: Training hyperparameters configuration.
    mesh: Optional SPMD device mesh.
    eval_data_iterator: Iterator yielding evaluation batches.
    step: Current training step integer.

  Returns:
    List of MetricsBuffer produced during evaluation steps.
  """
  eval_steps = getattr(config, "eval_steps", 0)
  if not eval_data_iterator or eval_steps <= 0:
    return []

  if hasattr(eval_data_iterator, "reset"):
    eval_data_iterator.reset()

  max_logging.log(f"Starting eval after train step {step}")
  history: list[abstract_engine.MetricsBuffer] = []
  eval_step_count = 0
  logical_axis_rules_for_eval = getattr(
      config,
      "logical_axis_rules_for_eval",
      getattr(config, "logical_axis_rules", ()),
  )

  for eval_batch in eval_data_iterator:
    if 0 < eval_steps <= eval_step_count:
      break
    if mesh is not None:
      eval_sharding = sharding.get_input_data_sharding(config, mesh, rules=logical_axis_rules_for_eval)
      eval_batch = jax.device_put(eval_batch, eval_sharding)
    with (
        jax.set_mesh(mesh) if mesh else contextlib.nullcontext(),
        nn_partitioning.axis_rules(logical_axis_rules_for_eval),
    ):
      engine.eval_step(eval_batch, step=step)
    step_metrics = engine.get_metrics(clear_cache=True)
    history.append(step_metrics)
    eval_step_count += 1

  max_logging.log(f"Completed eval after train step {step}")
  return history


run_eval = run_evaluation


# pylint: disable=too-many-positional-arguments
def training_loop_iteration(
    engine: maxtext_engine.MaxTextTrainingEngine,
    config: pyconfig.HyperParameters,
    mesh: jax.sharding.Mesh | None,
    data_loader: Any,
    rampup_manager: Any = None,
    eval_data_iterator: Any = None,
    goodput_recorder: Any = None,
    step: int = 0,
    start_step: int = 0,
) -> None:
  """Executes a single pre-training iteration using MaxTextTrainingEngine.

  Args:
    engine: The MaxTextTrainingEngine instance.
    config: Hyperparameters configuration instance.
    mesh: Optional SPMD device mesh.
    data_loader: The DataLoader instance.
    rampup_manager: Optional RampupBatchManager instance.
    eval_data_iterator: Optional Iterator yielding evaluation batches.
    goodput_recorder: Optional GoodputRecorder.
    step: Current training step integer.
    start_step: Initial training step index.

  Returns:
    None.
  """
  logical_axis_rules = getattr(config, "logical_axis_rules", ())

  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = load_next_batch(data_loader, rampup_manager=rampup_manager, goodput_recorder=goodput_recorder)
    with (
        jax.set_mesh(mesh) if mesh else contextlib.nullcontext(),
        nn_partitioning.axis_rules(logical_axis_rules),
    ):
      engine.fwd_bwd(example_batch)

    engine.update()

  new_step = engine.train_step

  # Periodic checkpoint saving
  checkpoint_period = getattr(config, "checkpoint_period", 0)
  if checkpoint_period > 0 and new_step > start_step and (new_step - start_step) % checkpoint_period == 0:
    engine.save_checkpoint(
        metadata={"step": new_step, "source": "train_v2"},
        step=new_step,
    )

  # Periodic evaluation
  eval_interval = getattr(config, "eval_interval", 0)
  eval_start_step = getattr(config, "eval_start_step", 0)
  if (
      eval_interval > 0
      and new_step >= start_step
      and new_step >= eval_start_step
      and (new_step - eval_start_step) % eval_interval == 0
  ):
    run_evaluation(
        engine,
        config,
        mesh,
        eval_data_iterator,
        step=new_step,
    )


def train_loop(
    config: pyconfig.HyperParameters,
    goodput_recorder: Any = None,
    engine: maxtext_engine.MaxTextTrainingEngine | None = None,
) -> maxtext_engine.MaxTextTrainingEngine:
  """Main pre-training loop driver delegating to MaxTextTrainingEngine.

  Args:
    config: Hyperparameters configuration instance.
    goodput_recorder: Optional GoodputRecorder.
    engine: Optional pre-initialized MaxTextTrainingEngine instance.

  Returns:
    The trained MaxTextTrainingEngine instance.
  """
  mesh = maxtext_utils.get_mesh_from_config(config)

  if engine is None:
    engine = maxtext_engine.MaxTextTrainingEngine(training_config=config, mesh=mesh, goodput_recorder=goodput_recorder)

  train_utils.maybe_apply_dcn_throttling(config)

  _ = engine.restore_checkpoint()
  start_step = engine.train_step
  train_utils.validate_completed_steps(start_step, getattr(config, "steps", 0))

  data_loader, eval_data_iterator, rampup_manager = setup_dataloaders(
      config=config,
      mesh=mesh,
      goodput_recorder=goodput_recorder,
      start_step=start_step,
  )

  data_sharding = sharding.get_input_data_sharding(config, mesh)
  shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
  engine.compile(shaped_batch)

  elastic_utils.record_elastic_reinit_end()

  if start_step == 0:
    max_utils.print_mem_stats("After params initialized")

  job_completed_gracefully = False
  try:
    while engine.train_step < getattr(config, "steps", 0):
      step = engine.train_step
      training_loop_iteration(
          engine=engine,
          config=config,
          mesh=mesh,
          data_loader=data_loader,
          rampup_manager=rampup_manager,
          eval_data_iterator=eval_data_iterator,
          goodput_recorder=goodput_recorder,
          step=step,
          start_step=start_step,
      )

    if getattr(config, "save_checkpoint_on_completion", False):
      engine.save_checkpoint(
          metadata={"step": engine.train_step, "source": "train_v2"},
          step=engine.train_step,
      )

    job_completed_gracefully = True
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
    job_completed_gracefully = True
  finally:
    if job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
    engine.close()
    train_utils.maybe_cleanup_dcn_throttling(config)

  return engine


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any]:
  """Initializes hyperparameters and system utilities for pre-training."""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  tf.config.set_visible_devices([], "GPU")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_utils.validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  jax.config.update(
      "jax_remove_size_one_mesh_axis_from_type",
      config.remove_size_one_mesh_axis_from_type,
  )
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if getattr(config, "use_te_comm_gemm_overlap", False):
    max_utils.bootstrap_transformer_engine_cgemm(config)

  goodput_recorder = create_goodput_recorder(config)
  return config, goodput_recorder


def run(
    config: pyconfig.HyperParameters,
    goodput_recorder: Any = None,
    engine: maxtext_engine.MaxTextTrainingEngine | None = None,
) -> maxtext_engine.MaxTextTrainingEngine:
  """Runs the pre-training job given hyperparameters and utilities."""
  with (max_utils.maybe_get_transformer_engine_context(config),):
    return train_loop(config, goodput_recorder, engine=engine)


def get_train_func(config: pyconfig.HyperParameters, goodput_recorder: Any, argv: Sequence[str]) -> Any:
  """Returns the train function, wrapping in elastic_retry if elastic training is enabled."""
  if getattr(config, "elastic_enabled", False):
    max_logging.log("Elastic utils: Elastic training enabled.")

    def on_elastic_event():
      elastic_utils.record_elastic_event_start(goodput_recorder, config)

    def on_slices_ready():
      elastic_utils.record_elastic_wait_end_and_reinit_start(goodput_recorder)

    def elastic_train_wrapper(argv: Sequence[str]) -> None:
      elastic_config, elastic_goodput_recorder = initialize(argv)
      run(elastic_config, elastic_goodput_recorder)

    return elastic_utils.elastic_retry(
        config,
        callback_fn=on_elastic_event,
        pre_callback_fn=on_slices_ready,
    )(functools.partial(elastic_train_wrapper, argv=argv))
  else:

    def train_func():
      run(config, goodput_recorder)

    return train_func


def main(argv: Sequence[str]) -> None:
  config, goodput_recorder = initialize(argv)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, goodput_recorder, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
