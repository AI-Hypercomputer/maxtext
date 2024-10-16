# A stateful version of MaxText trainer.

from typing import Any, Mapping, Optional
import logging
import sys
import os

import jax
from jax.experimental.compilation_cache import compilation_cache as cc

from train import create_goodput_recorder, record_goodput
from train import setup_train_loop, validate_train_config, train_step
from train import get_first_step, load_next_batch, check_example_batch
from train import record_scalar_metrics, save_checkpoint, write_metrics
from train import clear_buffered_metrics

import pyconfig
from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration
import datetime
import os
import sys
from etils import epath
import functools
import time

from typing import Sequence, Optional
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import profiler
import pyconfig
from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring


class MaxTextTrainer:

  def __init__(self, config: Mapping[str, Any]):
    self._config = config

  def process_config(self, run_name: str, config: Optional[Mapping[str, Any]] = None):
    # Some "tricks" to process the correct path...
    if "maxtext" not in sys.path[0]:
      extended_path = os.path.join(sys.path[0], "maxtext", "MaxText")
      sys.path.insert(0, extended_path)

    if not config:
      config = self._config
    base_yaml_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "base.yml")

    # Builds the argv from dict as MaxText currently only supports
    # an argv route
    argv = [
        "MaxText.py",
        base_yaml_fpath,
        f"run_name={run_name}",
    ]
    argv.extend([f"{k}={v}" for k, v in config.items()])
    pyconfig.initialize(argv)
    max_utils.print_system_information()
    config = pyconfig.config
    validate_train_config(config)
    os.environ["TFDS_DATA_DIR"] = config.dataset_path
    debug_config = debug_configuration.DebugConfig(
        stack_trace_config=stack_trace_configuration.StackTraceConfig(
            collect_stack_trace=config.collect_stack_trace,
            stack_trace_to_cloud=config.stack_trace_to_cloud,
            stack_trace_interval_seconds=config.stack_trace_interval_seconds,
        )
    )
    self.diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
    self.config = config

  def initialize(self, run_name: str, config: Optional[Mapping[str, Any]] = None):
    """Initializes the MaxText trainer."""
    logging.info("Initializing MaxText")
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    self.process_config(run_name=run_name, config=config)
    if jax.__version__ <= "0.4.23":
      cc.initialize_cache(os.path.expanduser(self.config.jax_cache_dir))
    else:
      cc.set_cache_dir(os.path.expanduser(self.config.jax_cache_dir))
    self.recorder = create_goodput_recorder(self.config)
    record_goodput(self.recorder, self.config, self.recorder.record_job_start_time if self.recorder else None)
    (
        self.init_rng,
        self.writer,
        self.checkpoint_manager,
        self.state_mesh_annotations,
        self.model,
        self.mesh,
        self.learning_rate_schedule,
        self.data_iterator,
        self.eval_data_iterator,
        self.state,
    ) = setup_train_loop(self.config)
    (
        self.functional_train,
        self.in_shard_train,
        self.out_shard_train,
        self.static_argnums_train,
        self.donate_argnums_train,
    ) = maxtext_utils.get_functional_train_with_signature(
        train_step, self.mesh, self.state_mesh_annotations, self.model, self.config
    )
    if self.eval_data_iterator:
      # pylint: disable=line-too-long
      (
          self.functional_eval,
          self.in_shard_eval,
          self.out_shard_eval,
          self.static_argnums_eval,
          self.donate_argnums_eval,
      ) = maxtext_utils.get_functional_eval_with_signature(
          self.eval_step, self.mesh, self.state_mesh_annotations, self.model, self.config
      )

    num_model_parameters = max_utils.calculate_num_params_from_pytree(self.state.params)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")

    # Write train config params, num model params, and XLA flags to tensorboard
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], self.writer)
    max_utils.add_config_to_summary_writer(self.config, self.writer)

    # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
    if self.config.compiled_trainstep_file != "":
      logging.info("Loading the compiled function...")
      # Need to pass train signature and state to determine i/o shapes of train_state for now.
      self.p_train_step = maxtext_utils.load_compiled(self.config, self.functional_train, self.state)
      # TODO: p_eval_step is not yet supported in load_compiled
      self.p_eval_step = None
      print("Loaded compiled function!", flush=True)
    else:
      self.p_train_step = jax.jit(
          self.functional_train,
          in_shardings=self.in_shard_train,
          out_shardings=self.out_shard_train,
          static_argnums=self.static_argnums_train,
          donate_argnums=self.donate_argnums_train,
      )

      if self.eval_data_iterator:
        self.p_eval_step = jax.jit(
            self.functional_eval,
            in_shardings=self.in_shard_eval,
            out_shardings=self.out_shard_eval,
            static_argnums=self.static_argnums_eval,
            donate_argnums=self.donate_argnums_eval,
        )
      else:
        self.p_eval_step = None

    self.local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    self.running_gcs_metrics = [] if self.config.gcs_metrics else None

  def train(self, num_steps: Optional[int] = None) -> int:
    """Runs training for a specific number of steps."""
    logging.info("Running training workload...")
    start_step = get_first_step(self.state)  # this is the start_step for training
    stop_step = min(start_step + num_steps, self.config.steps)
    logging.info("Running training workload from step %d to %d", start_step, stop_step)

    per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
    per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(self.config)
    record_goodput(self.recorder, self.config, self.recorder.record_tpu_init_start_time if self.recorder else None)
    with diagnostic.diagnose(self.diagnostic_config):
      first_profiling_step = start_step + self.config.skip_first_n_steps_for_profiler
      if self.config.profiler != "" and first_profiling_step >= self.config.steps:
        raise ValueError("Profiling requested but initial profiling step set past training final step")
      last_profiling_step = np.clip(
          first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.steps - 1
      )

      example_batch = None
      last_step_completion = datetime.datetime.now()
      prof = profiler.Profiler(self.config)
      for step in np.arange(start_step, stop_step):
        if step == first_profiling_step:
          prof.activate()

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          record_goodput(self.recorder, self.config, self.recorder.record_data_loading_start_time if self.recorder else None)
          example_batch = load_next_batch(self.data_iterator, example_batch, self.config)
          record_goodput(self.recorder, self.config, self.recorder.record_data_loading_end_time if self.recorder else None)
          check_example_batch(self.config, example_batch=example_batch)
          nextrng = jax.jit(jax.random.fold_in)(self.init_rng, step)
          record_goodput(self.recorder, self.config, self.recorder.record_step_start_time if self.recorder else None, step)
          with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            self.state, metrics = self.p_train_step(self.state, example_batch, nextrng)

        new_time = datetime.datetime.now()
        record_scalar_metrics(
            metrics, new_time - last_step_completion, per_device_tflops, self.learning_rate_schedule(step), per_device_tokens
        )
        last_step_completion = new_time

        if self.checkpoint_manager is not None:
          if save_checkpoint(
              self.checkpoint_manager, int(step), self.state, self.config.dataset_type, self.data_iterator, self.config
          ):
            max_logging.log(f"saved a checkpoint at step {step}")

          # Upon preemption, exit when and only when all ongoing saves are complete.
          if self.checkpoint_manager.reached_preemption(step):
            logging.info("Reached preemption, checkpointing now.")
            self.checkpoint_manager.wait_until_finished()
            sys.exit()

        write_metrics(self.writer, self.local_metrics_file, self.running_gcs_metrics, metrics, step, self.config)

        if self.config.eval_interval > 0 and step > start_step and step % self.config.eval_interval == 0:
          assert self.eval_data_iterator
          cumulative_eval_metrics = {"total_loss": 0.0, "total_weights": 0.0, "moe_lb_loss": 0.0}
          eval_step_count = 0
          for eval_batch in self.eval_data_iterator:
            if self.config.eval_steps > 0 and eval_step_count >= self.config.eval_steps:
              break
            with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
              eval_metrics = self.p_eval_step(self.state, eval_batch, nextrng)
            cumulative_eval_metrics["total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
            cumulative_eval_metrics["total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
            cumulative_eval_metrics["moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
            max_logging.log(f"Completed eval step {eval_step_count}")
            eval_step_count += 1
          eval_loss = (
              cumulative_eval_metrics["total_loss"] / (cumulative_eval_metrics["total_weights"] + EPS)
              + cumulative_eval_metrics["moe_lb_loss"] / eval_step_count
          )
          max_logging.log(
              f"average loss after {step=}: {eval_step_count=}, {eval_loss=}, total_weights={cumulative_eval_metrics['total_weights']}"
          )
          if eval_loss <= self.config.target_eval_loss:
            max_logging.log(f"Early stop and exit loop after reaching {self.config.target_eval_loss=}")
            prof.deactivate()
            break

        if step == last_profiling_step:
          prof.deactivate()

      if self.checkpoint_manager is not None:
        self.checkpoint_manager.wait_until_finished()
      write_metrics(
          self.writer, self.local_metrics_file, self.running_gcs_metrics, metrics, stop_step, self.config
      )  # final step metrics
      max_utils.close_summary_writer(self.writer)
      record_goodput(self.recorder, self.config, self.recorder.record_job_end_time if self.recorder else None)
      clear_buffered_metrics()
      return stop_step
