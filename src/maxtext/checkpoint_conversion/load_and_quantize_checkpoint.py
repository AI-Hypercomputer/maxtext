# Copyright 2023–2025 Google LLC
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

"""CLI utility for loading and quantizing a checkpoint."""

import os
from typing import Sequence

from absl import app

import jax

from MaxText import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils


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
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
