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

"""A command-line tool to compile and analyze the forward pass of a MaxText model.

This script performs ahead-of-time (AOT) or live compilation of only the model's forward
pass (and logit gathering across data-parallel groups) without running backward passes
or optimizer step updates. It outputs JAX memory analysis upon completion.

Compilation Modes:
  - Simulated Mesh (AOT): Use `compile_topology=<topology>` (e.g., `v5p-128`) on CPU/VM
    to test sharding, compilation, and HBM memory footprint without requiring physical TPU hardware.
  - Active Hardware: Omit `compile_topology` when running directly on a TPU VM to compile for
    attached physical TPU chips.

Usage Examples:
  [AOT Compilation on Simulated Mesh]
    python tests/utils/forward_compile.py \
        src/maxtext/configs/base.yml \
        model_name=llama2-7b \
        compile_topology=v5p-128 \
        compile_topology_num_slices=1

  [Compilation on Active Hardware]
    python tests/utils/forward_compile.py \
        src/maxtext/configs/base.yml \
        model_name=llama2-7b

Arguments:
  Positional / Key-Value:
    Standard MaxText configuration options (e.g., model_name, compile_topology,
    ici_fsdp_parallelism, per_device_batch_size, etc.).
"""

import functools
import os
import sys
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning

# Import MaxText modules
from maxtext.configs import pyconfig
from maxtext.utils import max_utils
from maxtext.trainers.pre_train import train_compile

def forward_and_gather(model, config, state, batch, init_rng):
    # Depending on pure_nnx or linen, state might just be the params
    if config.pure_nnx:
        # NNX logic mocking if needed, but the original script uses linen mostly
        pass
    
    ids = batch.get('inputs')
    decoder_positions = batch.get('inputs_position')
    decoder_segment_ids = batch.get('inputs_segmentation')
    encoder_images = batch.get('images')

    params_dict = state.params
    if "params" in getattr(params_dict, "keys", lambda: [])():
        params_dict = params_dict["params"]

    full_train_logits = model.apply(
        {"params": params_dict},
        ids,
        decoder_positions,
        decoder_segment_ids,
        encoder_images,
        enable_dropout=False,
        rngs={"aqt": init_rng},
    )

    data_parallelism = max(config.ici_fsdp_parallelism, 1) * max(config.ici_data_parallelism, 1) * max(config.dcn_data_parallelism, 1) * max(config.dcn_fsdp_parallelism, 1)
    if data_parallelism > 1:
        full_train_logits = jax.experimental.multihost_utils.process_allgather(full_train_logits, tiled=True)
    return full_train_logits


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    
    config = pyconfig.initialize(argv)
    train_compile.validate_config(config)

    topology_mesh = train_compile.get_topology_mesh(config)
    max_utils.print_system_information()

    (
        shaped_train_args,
        shaped_train_kwargs,
        state_mesh_shardings,
        logical_annotations,
        model,
    ) = train_compile.get_shaped_inputs(topology_mesh, config)

    # shaped_train_args is usually (abstract_state, shaped_batch, shaped_rng)
    abstract_state = shaped_train_args[0]
    shaped_batch = shaped_train_args[1]
    shaped_rng = shaped_train_args[2] if len(shaped_train_args) > 2 else jax.random.PRNGKey(0)

    # Sharding for inputs
    from maxtext.utils import sharding
    data_sharding = sharding.get_input_data_sharding(config, topology_mesh)
    
    # We don't bother zero1 sharding updates because this is just a forward pass
    in_shard = (state_mesh_shardings, data_sharding, None)
    out_shard = None  # We don't strictly care about the output sharding constraint here

    # Wrap function
    func_to_compile = functools.partial(forward_and_gather, model, config)
    
    print("Jitting and compiling forward_and_gather...", flush=True)
    
    with jax.set_mesh(topology_mesh), topology_mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        jitted = jax.jit(
            func_to_compile,
            in_shardings=in_shard,
            out_shardings=out_shard,
            static_argnums=(),
            donate_argnums=(),
        )
        lowered = jitted.lower(abstract_state, shaped_batch, shaped_rng)
    
    compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
    compiled = lowered.compile(compiler_options=compiler_options)
    
    print("Jitting and compilation complete!", flush=True)
    print(f"Memory analysis: {compiled.memory_analysis()}")

if __name__ == "__main__":
    app.run(main)
