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
from typing import Sequence

from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import numpy as jnp
from maxtext.configs import pyconfig
from maxtext.common import checkpointing
from maxtext.layers import train_state_nnx
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
      optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

  else:
    model = from_config(config)
    mesh = model.mesh
    _, tx = train_utils.create_training_optimizer(config, model)
    init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, init_rng)
  checkpoint_manager = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)

  unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, is_training=True)
  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")
  checkpoint_load_start = datetime.datetime.now()
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, _ = checkpointing.load_state_if_possible(
        checkpoint_manager,
        None,
        config.load_parameters_path,
        config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb,
        unboxed_abstract_state,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    if state:
      state = state["items"]

  jax.block_until_ready(state)
  checkpoint_load_end = datetime.datetime.now()
  if state is not None:  # Checkpoint was available for restore
    if jax.process_index() == 0:
      max_logging.log(
          "STANDALONE CHECKPOINTER : Checkpoint restored in :" f" {checkpoint_load_end - checkpoint_load_start}"
      )
  else:  # Checkpoint was unavailable, state needs to be initialized
    state, _, _, _ = maxtext_utils.setup_training_state(None, config, mesh, checkpoint_manager, init_state_fn)
  state = add_entropy_to_checkpoint(state)

  start_step = get_first_step(model, state)  # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      if checkpointing.save_checkpoint(checkpoint_manager, int(step), state):
        checkpoint_manager.wait_until_finished()
        end_time = datetime.datetime.now()
        if jax.process_index() == 0:
          max_logging.log(
              "STANDALONE CHECKPOINTER : Checkpoint saved in" f" {end_time - start_time} ,step {step}, on host 0"
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
