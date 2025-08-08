# SPDX-License-Identifier: Apache-2.0

"""CLI utility for loading and quantizing a checkpoint."""

import os
from typing import Sequence

from absl import app

import jax

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  validate_config(config)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)

  # load_params will load a checkpoint and quantize if the following parameters are set:
  # quantization=$valid_quantization_type \
  # save_quantized_params_path=$gsbucket_path \
  # checkpoint_is_quantized=false (default)
  engine.load_params(rng_load_params)


def validate_config(config):
  assert config.load_full_state_path == "", "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
