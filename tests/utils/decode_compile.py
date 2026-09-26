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

"""A command-line tool to compile and analyze MaxText inference (prefill and decode).

This script performs ahead-of-time (AOT) or live compilation for inference routines:
  1. Prefill step (`prefill_aot`): Processes prompt tokens and populates the KV cache.
  2. Autoregressive decode step (`generate_aot`): Single-step token generation.
  3. KV Cache & Inference Memory Analysis: Analyzes parameters, KV cache footprint,
     and peak execution memory.

Compilation Modes:
  - Simulated Mesh (AOT): Set `compile_topology=<topology>` (e.g., `v5p-128`) on CPU/VM
    to test sharding, compilation, KV cache allocation, and memory bounds without physical hardware.
  - Active Hardware: Omit `compile_topology` when running directly on a TPU VM to compile for
    attached physical TPU chips.

Usage Examples:
  [AOT Compilation on Simulated Mesh]
    python tests/utils/decode_compile.py \
        src/maxtext/configs/base.yml \
        model_name=llama2-7b \
        compile_topology=v5p-8 \
        per_device_batch_size=1 \
        max_prefill_predict_length=1024 \
        max_target_length=2048

  [Compilation on Active Hardware]
    python tests/utils/decode_compile.py \
        src/maxtext/configs/base.yml \
        model_name=llama2-7b \
        per_device_batch_size=1 \
        max_prefill_predict_length=1024 \
        max_target_length=2048

Arguments:
  Positional / Key-Value:
    Standard MaxText configuration options (e.g., model_name, compile_topology,
    per_device_batch_size, max_prefill_predict_length, max_target_length, etc.).
"""

import os
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp

# Import MaxText modules
from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import max_utils


def _validate_decode_config(config):
  assert config.load_full_state_path == "", (
      "Decode compilation operates on parameters, not full training state. "
      "Ensure load_full_state_path is empty."
  )


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  _validate_decode_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)

  topology_mesh = train_compile.get_topology_mesh(config)
  max_utils.print_system_information()

  print("Initializing MaxEngine...", flush=True)
  engine = maxengine.MaxEngine(config)

  rng = jax.random.PRNGKey(1234)
  rng, rng_prefill, rng_init_decode, rng_generate = jax.random.split(rng, 4)

  # Abstract shapes for parameters and decode state (KV Cache)
  abstract_params = engine.abstract_params
  abstract_decode_state = jax.eval_shape(engine.init_decode_state, rng_init_decode)

  # Dummy prefill inputs
  prefill_length = config.max_prefill_predict_length
  padded_tokens = jnp.zeros((prefill_length,), dtype=jnp.int32)
  abstract_tokens = jax.ShapeDtypeStruct(padded_tokens.shape, padded_tokens.dtype)
  true_length = prefill_length

  compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)

  print("\n==================================================", flush=True)
  print("1. Compiling Prefill Phase (Prompt Processing)...", flush=True)
  print("==================================================", flush=True)
  with jax.set_mesh(topology_mesh), topology_mesh:
    prefill_lowered = jax.jit(engine.prefill_aot).lower(
        abstract_params, abstract_tokens, true_length, rng_prefill
    )
    compiled_prefill = prefill_lowered.compile(compiler_options=compiler_options)

  print("Prefill compilation complete!", flush=True)
  print(f"Prefill Memory Analysis:\n{compiled_prefill.memory_analysis()}")

  print("\n==================================================", flush=True)
  print("2. Compiling Generate Phase (Single Autoregressive Step)...", flush=True)
  print("==================================================", flush=True)
  with jax.set_mesh(topology_mesh), topology_mesh:
    generate_lowered = jax.jit(engine.generate_aot).lower(
        abstract_params, abstract_decode_state, rng_generate
    )
    compiled_generate = generate_lowered.compile(compiler_options=compiler_options)

  print("Generate compilation complete!", flush=True)
  print(f"Generate Memory Analysis:\n{compiled_generate.memory_analysis()}")

  print("\n==================================================", flush=True)
  print("Inference Memory Summary", flush=True)
  print("==================================================", flush=True)
  print(f"Model Name:                 {config.model_name}")
  print(f"Per-Device Batch Size:      {config.per_device_batch_size}")
  print(f"Max Prefill Length:         {config.max_prefill_predict_length}")
  print(f"Max Target (Total) Length:  {config.max_target_length}")
  print("==================================================\n", flush=True)


if __name__ == "__main__":
  app.run(main)
