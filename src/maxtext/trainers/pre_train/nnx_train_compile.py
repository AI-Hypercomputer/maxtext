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

"""
Save a Cross Ahead of Time Compiled (XAOT) version of nnx_train.py's train step.

Mirrors train_compile.py but uses the Flax NNX API throughout, in contrast to
train_compile.py which relies on Linen's TrainState.

Key differences from train_compile.py
--------------------------------------
- No Linen TrainState.  State lives in two separate pytrees:
    model_state   – nnx.State for the model parameters
    opt_state     – nnx.State for the optimizer (optax state + step counter)
- nnx.eval_shape creates abstract shapes without materialising parameters, so the
  whole compilation is done without ever touching real hardware memory.
- Graphdefs (model_graphdef, opt_graphdef) are baked into the partial and are
  Python-static across the JIT boundary; they are therefore not listed in
  static_argnums.
- in_shardings / out_shardings follow the NNX train_step signature:
    in:  (model_state, opt_state, batch, rng)
    out: ((model_state, opt_state), metrics)

Entry point:
  python -m maxtext.trainers.pre_train.nnx_train_compile <config> [overrides…]
"""

import functools
import os
from typing import Callable, Sequence

import jax
from absl import app
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import nnx_train
from maxtext.trainers.pre_train.train_compile import get_topology_mesh, jit_and_compile, save_compiled, validate_config
from maxtext.utils import gcs_utils, max_utils, maxtext_utils, model_creation_utils, sharding


def create_nnx_rngs(
    config: pyconfig.HyperParameters, is_training: bool = True, rng_key: jax.Array | None = None
) -> nnx.Rngs:
  """
  Create NNX Rngs

  Args:
    config: the configuration
    is_training: if the Rngs are for training
    rng_key: the Rng key

  Returns:
    The NNX Rngs
  """
  if rng_key is None:
    rng_key = jax.random.PRNGKey(config.init_weights_seed)

  if is_training:
    return nnx.Rngs(
        params=jax.random.fold_in(rng_key, 0), dropout=jax.random.fold_in(rng_key, 1), aqt=jax.random.fold_in(rng_key, 2)
    )
  return nnx.Rngs(params=rng_key)  # disable dropout RNG and aqt for inference


# ---------------------------------------------------------------------------
# Shaped inputs (NNX version)
# ---------------------------------------------------------------------------


def get_shaped_inputs_nnx(topology_mesh, config):
  """Build abstract (shape-only) versions of nnx_train.train_step's inputs.

  Uses nnx.eval_shape to trace through model and optimizer construction so that
  no actual parameters are allocated.  The returned abstract states have
  ShapeDtypeStruct leaves and can be passed directly to jax.jit.lower().

  Returns:
    model_graphdef:   Static NNX graph definition for the model.
    opt_graphdef:     Static NNX graph definition for the optimizer.
    abstract_model_state: Abstract model parameter pytree.
    abstract_opt_state:   Abstract optimizer state pytree.
    model_shardings:  Partition specs mapped to mesh shardings for model_state.
    opt_shardings:    Partition specs mapped to mesh shardings for opt_state.
    data_sharding:    Input-batch sharding.
    shaped_batch:     Shaped batch dict (ShapeDtypeStruct leaves).
    shaped_rng:       Shaped RNG key.
    learning_rate_schedule: LR schedule (baked into the compiled object).
  """
  # rng_key = jax.random.PRNGKey(config.init_weights_seed)
  # rngs = nnx.Rngs(params=rng_key, dropout=1)

  # ------------------------------------------------------------------
  # 1. Abstract model via nnx.eval_shape — no parameters materialised.
  # ------------------------------------------------------------------

  def get_nnx_create_model_fn(config, mesh=None, devices=None) -> Callable:
    """Creates the function for NNX model creation."""

    def _create_model():
      # is_training = model_mode == MODEL_MODE_TRAIN
      # rngs = maxtext_utils_nnx.create_nnx_rngs(config, is_training=is_training, rng_key=rng_key)
      rng_key = jax.random.PRNGKey(config.init_weights_seed)
      rngs = create_nnx_rngs(config, True, rng_key)
      return model_creation_utils.from_config(config, devices, mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

    return _create_model

  with nn.logical_axis_rules(config.logical_axis_rules):
    create_model_fn = get_nnx_create_model_fn(config, topology_mesh)
    abstract_model = nnx.eval_shape(create_model_fn)
    model_graphdef, abstract_model_state = nnx.split(abstract_model)

  # ------------------------------------------------------------------
  # 2. Abstract optimizer via nnx.eval_shape.
  # ------------------------------------------------------------------
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # get_optimizer may inspect the model structure (e.g. for Muon); the abstract
  # model has the same tree structure as the real one, so this is safe.
  tx = optimizers.get_optimizer(config, learning_rate_schedule, abstract_model)

  def _build_optimizer():
    return nnx.Optimizer(abstract_model, tx, wrt=nnx.Param)

  abstract_optimizer = nnx.eval_shape(_build_optimizer)
  opt_graphdef, abstract_opt_state = nnx.split(abstract_optimizer)

  # ------------------------------------------------------------------
  # 3. Partition specs → mesh shardings.
  # ------------------------------------------------------------------
  with nn.logical_axis_rules(config.logical_axis_rules):
    model_shardings = nn.logical_to_mesh_sharding(
        nnx.get_partition_spec(abstract_model_state), topology_mesh, config.logical_axis_rules
    )
    opt_shardings = nn.logical_to_mesh_sharding(
        nnx.get_partition_spec(abstract_opt_state), topology_mesh, config.logical_axis_rules
    )

  # ------------------------------------------------------------------
  # 4. Shaped batch and RNG.
  # ------------------------------------------------------------------
  data_sharding = sharding.get_input_data_sharding(config, topology_mesh)
  shaped_batch = maxtext_utils.get_shaped_batch(config)

  _, example_rng = jax.random.split(jax.random.PRNGKey(0), 2)
  shaped_rng = jax.ShapeDtypeStruct(example_rng.shape, example_rng.dtype)

  return (
      model_graphdef,
      opt_graphdef,
      abstract_model_state,
      abstract_opt_state,
      model_shardings,
      opt_shardings,
      data_sharding,
      shaped_batch,
      shaped_rng,
      learning_rate_schedule,
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting nnx_train_compile.py...", flush=True)

  # Parse and validate configuration
  config = pyconfig.initialize(argv)
  validate_config(config)

  # Create target mesh
  topology_mesh = get_topology_mesh(config)

  # Print system information after building the compile topology to avoid
  # prematurely initialising the backend.
  max_utils.print_system_information()

  # Get shaped inputs
  (
      model_graphdef,
      opt_graphdef,
      abstract_model_state,
      abstract_opt_state,
      model_shardings,
      opt_shardings,
      data_sharding,
      shaped_batch,
      shaped_rng,
      _,  # _learning_rate_schedule,
  ) = get_shaped_inputs_nnx(topology_mesh, config)

  # Build the partial that matches what _build_jit_steps produces in nnx_train.
  # graphdefs are static (captured in the Python closure) so they do not appear
  # in static_argnums.
  func_to_compile = functools.partial(nnx_train.train_step, model_graphdef, opt_graphdef, config=config)
  func_to_compile.__name__ = "nnx_train_step"

  shaped_train_args = (abstract_model_state, abstract_opt_state, shaped_batch, shaped_rng)
  shaped_train_kwargs = {}

  in_shard = (model_shardings, opt_shardings, data_sharding, None)
  out_shard = ((model_shardings, opt_shardings), None)
  static_argnums = ()
  donate_argnums = (0, 1)

  # Compile
  print("Jitting and compiling NNX train step...", flush=True)
  compiled = jit_and_compile(
      func_to_compile,
      shaped_train_args,
      shaped_train_kwargs,
      topology_mesh,
      in_shard,
      out_shard,
      static_argnums,
      donate_argnums,
      config,
      nn_partitioning.axis_rules(config.logical_axis_rules),
  )
  print("Jitting and compilation complete!", flush=True)

  # Serialize and save the compiled object
  if config.compiled_trainstep_file != "":
    print("Saving compiled object...")
    save_compiled(compiled, config.compiled_trainstep_file)
    print(f"Successfully saved compiled object as {config.compiled_trainstep_file}")
  print("Finished nnx_train_compile.py successfully!", flush=True)
  print(f"Cost analysis: {compiled.cost_analysis()}")
  print(f"Memory analysis: {compiled.memory_analysis()}")

  # Dump HLO if requested
  if config.dump_hlo:
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


if __name__ == "__main__":
  app.run(main)
