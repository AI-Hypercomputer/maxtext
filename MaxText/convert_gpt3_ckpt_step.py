"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=line-too-long
"""change ckpt step

python MaxText/convert_gpt3_ckpt_step.py \
  --maxtext-model-name=gpt3-175b \
  --run-name=$RUN_NAME \
  --base-output-directory=$BASE_OUTPUT_DIR\
  --load-full-state-path=$LOAD_FULL_STATE_PATH\
  --overwrite-ckpt-step=$OVERWRITE_CKPT_STEP
"""
import max_utils
import optimizers
import pyconfig
from jax import random
from jax.sharding import Mesh
from layers.models import Transformer
import checkpointing

import sys
import jax
import max_logging
import argparse
import jax.numpy as jnp

def convert(maxtext_model_name, base_output_directory, run_name, load_full_state_path, overwrite_ckpt_step):
  """convert ckpt."""

  base_args = [
    '', 'MaxText/configs/base.yml',  # base arg
    'per_device_batch_size=1',
    'ici_fsdp_parallelism=-1', 'ici_tensor_parallelism=1',
    f'model_name={maxtext_model_name}',
    f'run_name={run_name}', f'base_output_directory={base_output_directory}',
    f'load_full_state_path={load_full_state_path}',
    'checkpoint_period=1',
    'async_checkpointing=false',
  ]
  pyconfig.initialize(base_args)
  cfg = pyconfig.config
  init_rng, _ = random.split(random.PRNGKey(cfg.init_weights_seed), 2)
  devices_array = max_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  model = Transformer(config=cfg, mesh=mesh)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(cfg)
  tx = optimizers.get_optimizer(cfg, learning_rate_schedule)

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    cfg.checkpoint_dir,
    cfg.enable_checkpointing,
    cfg.async_checkpointing,
    cfg.checkpoint_period,
  )

  # load_full_state_path
  state, _ = max_utils.setup_training_state(model, tx, cfg, init_rng, mesh, None)
  max_logging.log("start")

   # hack overwrite state
  def map_fn(key_path, value):
    key_path_str = jax.tree_util.keystr(key_path)
    if key_path_str in  (".step", ".opt_state[0].count", ".opt_state[1].count", ".opt_state.count", ".opt_state[<flat index 0>]"):
      max_logging.log(f"overwrite step: {key_path_str}")
      shape = value.shape
      sharding = value.sharding
      result = jax.make_array_from_single_device_arrays(
        shape,
        sharding,
        [jax.device_put(jnp.array(overwrite_ckpt_step, dtype=value.dtype), d)
        for d, index in sharding.addressable_devices_indices_map(shape).items()],
        )
      return result
    else:
      return value

  converted_state = jax.tree_util.tree_map_with_path(map_fn, state)
  max_logging.log("converted state finished")

  if checkpoint_manager.save(converted_state.step, converted_state):
    max_logging.log(f"saved a checkpoint at step {converted_state.step}")
  # Upon preemption, exit when and only when all ongoing saves are complete.
  if checkpoint_manager.reached_preemption(converted_state.step):
    checkpoint_manager.wait_until_finished()
    sys.exit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--maxtext-model-name', choices=['gpt3-175b', 'gpt3-52k'],  type=str, required=True)
  parser.add_argument('--base-output-directory', type=str, required=True)
  parser.add_argument('--run-name', type=str, required=True)
  parser.add_argument('--load-full-state-path', type=str, required=True)
  parser.add_argument('--overwrite-ckpt-step', type=int, required=True)

  args = parser.parse_args()

  convert(args.maxtext_model_name, args.base_output_directory, args.run_name, args.load_full_state_path, args.overwrite_ckpt_step)
