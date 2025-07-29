"""
Copyright 2025 Google LLC

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

import os
import jax
import jax.numpy as jnp

from MaxText import pyconfig
from MaxText import maxengine
from MaxText import checkpointing
from MaxText import max_logging
from typing import Sequence
from absl import app

from flax.training import train_state

def main(argv: Sequence[str]) -> None:
  # Initialize config
  config = pyconfig.initialize(argv)

  # Set up output directory if needed
  output_path = config.base_output_directory
  os.makedirs(output_path, exist_ok=True)

  # Initialize model and random params
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_params = jax.random.split(rng)
  params = engine.load_params(rng_params)
  max_logging.log(f"Initialized random parameters for model {config.model_name}")
  interest_weight = params['params']['vision_encoder']['Gemma3VisionEncoderLayer_0']['embedding']['kernel']
  max_logging.log(f"Weight mean: {interest_weight.mean()}\nWeight shape: {interest_weight.shape}")

  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1 

  # Create checkpoint manager and save
  state = train_state.TrainState(
      step=0, apply_fn=None, params={"params": params['params']}, tx=None, opt_state={}  # type: ignore
  )
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      output_path, enable_checkpointing, async_checkpointing, save_interval_steps
  )
  checkpointing.save_checkpoint(checkpoint_manager, 0, state)
  max_logging.log(f"Saved random-initialized checkpoint to {output_path}/0")


if __name__ == "__main__":
  app.run(main)
