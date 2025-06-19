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
"""
Convert orbax Gemma checkpoint to MaxText compatible checkpoint.
"""

from typing import Any
import argparse
import copy
import sys

import numpy as np

import jax
import jax.numpy as jnp

from flax.training import train_state

import orbax

from MaxText import checkpointing
from MaxText import max_logging
from MaxText.train import save_checkpoint

jax.config.update("jax_platform_name", "cpu")

from MaxText.checkpoint_conversion_utils import convert_gemma_weights, nest_params

Params = dict[str, Any]


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  args = parser.parse_args(raw_args)
  if args.model_size not in ("2b", "7b", "9b"):
    raise NotImplementedError(args.model_size)

  print("Loading checkpoint")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  source_params = checkpointer.restore(args.base_model_path)

  jax_weights = convert_gemma_weights(source_params, args.model_size)

  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      args.maxtext_model_path, enable_checkpointing, async_checkpointing, save_interval_steps
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, 0, state_new):
      max_logging.log("saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()


if __name__ == "__main__":
  main()
