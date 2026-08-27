# Copyright 2023–2026 Google LLC
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Standalone checkpointer - only saves and restores checkpoints at regular intervals, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
from functools import partial
import os
import time
from typing import Sequence

from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import numpy as jnp
from maxtext.configs import pyconfig
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.models import models
from maxtext.trainers.pre_train.train import get_first_step
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils import train_utils
from maxtext.utils.model_creation_utils import from_config
import numpy as np

Transformer = models.transformer_as_linen


def checkpoint_loop(config, state=None):
  """Save/restore exerciser.

  Builds an abstract train state, restores or initializes it, perturbs the
  optimizer moments via `add_entropy_to_checkpoint`, then writes checkpoints
  on the configured cadence. Works on both Linen and NNX state shapes.
  """
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  if config.pure_nnx:
    mesh = maxtext_utils.get_mesh_from_config(config)
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=init_rng)
    model = from_config(config, mesh=mesh, rngs=rngs)
    _, tx = train_utils.create_training_optimizer(config, model)
    _create_model_partial, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

    def init_state_fn():
      nnx_model = _create_model_partial()
      wrt = (
          getattr(nnx, "LoRAParam", nnx.Param)
          if getattr(getattr(config, "lora", None), "enable_lora", False)
          else nnx.Param
      )
      optimizer = nnx.Optimizer(nnx_model, tx, wrt=wrt)
      return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

  else:
    model = from_config(config)
    mesh = model.mesh
    _, tx = train_utils.create_training_optimizer(config, model)
    init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, init_rng)

  checkpoint_manager = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)

  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")

  state = None

  if config.standalone_checkpointer_start_from_checkpoint:
    unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, is_training=True)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      loaded_state, _ = checkpointing.load_state_if_possible(
          checkpoint_manager,
          None,
          config.load_parameters_path,
          config.load_full_state_path,
          config.checkpoint_storage_concurrent_gb,
          unboxed_abstract_state,
          config.enable_single_replica_ckpt_restoring,
          config.dataset_type,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
          enable_orbax_v1=config.enable_orbax_v1,
          checkpoint_conversion_fn=config.checkpoint_conversion_fn,
          source_checkpoint_layout=config.source_checkpoint_layout,
          expansion_factor_real_data=config.expansion_factor_real_data,
          maxtext_config=config,
      )
      if loaded_state:
        state = loaded_state.get("items", loaded_state)

  if state is None:
    # Delegate checkpoint restoration or state initialization to setup_training_state
    state, _, _, _, _ = maxtext_utils.setup_training_state(model, config, mesh, checkpoint_manager, init_state_fn)

  jax.block_until_ready(state)

  state = add_entropy_to_checkpoint(state)

  start_step = get_first_step(model, state)  # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      state_to_save = train_state_nnx.to_linen_checkpoint_dict(state.to_pure_dict()) if config.pure_nnx else state
      if checkpointing.save_checkpoint(checkpoint_manager, int(step), state_to_save):
        checkpointing.wait_until_finished(checkpoint_manager)
        end_time = datetime.datetime.now()
        if jax.process_index() == 0:
          max_logging.log(
              "STANDALONE CHECKPOINTER : Checkpoint saved in" f" {end_time - start_time} ,step {step}, on host 0"
          )
          elapsed_time = datetime.datetime.now() - start_time
          time_to_wait = config.standalone_checkpointer_per_step_interval - elapsed_time.total_seconds()
          if time_to_wait > 0:
            time.sleep(time_to_wait)
        jax.experimental.multihost_utils.sync_global_devices("Barrier after step")

        if config.standalone_checkpointer_enable_restore_in_loop:
          # Optional OS Page Cache Eviction (for Checkpointing Benchmarks):
          # When saving a checkpoint to storage and immediately restoring it on the same host,
          # the Linux kernel OS page cache holds the newly written blocks in RAM.
          # Without dropping the cache, the restore operation will read from host RAM rather
          # than actual backing storage (e.g., GCS / Lustre / persistent disk), which artificially
          # inflates restore speeds and distorts storage benchmark metrics.
          #
          # NOTE: Executing this command requires `sudo` privileges on Linux:
          # `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`.
          # It defaults to False for compatibility with standard non-sudo MaxText environments,
          # and should only be enabled in dedicated benchmarking environments.
          if jax.process_index() == 0 and config.standalone_checkpointer_drop_page_cache_before_restore:
            max_logging.log("STANDALONE CHECKPOINTER : Dropping OS page cache before restore...")
          if config.standalone_checkpointer_drop_page_cache_before_restore:
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")

          restore_start = datetime.datetime.now()
          restored_state = checkpoint_manager.restore(int(step))
          if restored_state:
            restored_state = restored_state.get("items", restored_state)
          jax.block_until_ready(restored_state)
          restore_end = datetime.datetime.now()
          if jax.process_index() == 0:
            max_logging.log(
                f"STANDALONE CHECKPOINTER : Checkpoint restored in {restore_end - restore_start} ,step {step}, on host 0"
            )

  return state


def add_entropy_to_checkpoint(state):
  """Replace adam mu/nu with cos/sin of params.

  Stand-in for real training when exercising checkpoint save/restore. Handles
  three shapes:
    * Linen `TrainState`: `state.params` + `state.opt_state` (tuple).
    * NNX `TrainStateNNX` (Module): `state.model` is an `nnx.Module`; the
      optimizer's `opt_state` is the optax tuple of NamedTuples.
    * NNX `nnx.State` (post-split, what `setup_training_state` returns under
      `pure_nnx`): `state.model` and `state.optimizer.opt_state` are sub-States;
      `opt_state[0].mu`/`nu` are themselves States that can be reassigned.
  """
  if hasattr(state, "model"):
    if isinstance(state, nnx.Module):
      params = nnx.state(state.model, nnx.Param)
    else:
      params = state.model.filter(nnx.Param) if hasattr(state.model, "filter") else state.model
    new_mu = jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), params)
    new_nu = jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), params)

    if isinstance(state, nnx.Module):
      opt = state.optimizer
      opt.opt_state = (opt.opt_state[0]._replace(mu=new_mu, nu=new_nu),) + tuple(opt.opt_state[1:])
    else:
      state.optimizer.opt_state[0].mu = new_mu
      state.optimizer.opt_state[0].nu = new_nu
    return state

  opt_0 = state.opt_state[0]
  opt_0 = opt_0._replace(mu=jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), state.params))
  opt_0 = opt_0._replace(nu=jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), state.params))
  new_opt = [opt_0] + list(state.opt_state[1:])
  return state.replace(opt_state=new_opt)


def main(argv: Sequence[str]) -> None:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  train_utils.validate_train_config(config)
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  checkpoint_loop(config)


if __name__ == "__main__":
  app.run(main)
