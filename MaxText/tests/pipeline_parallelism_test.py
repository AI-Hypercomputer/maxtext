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

# pylint: disable=missing-module-docstring, missing-function-docstring
import sys
import numpy as np
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec

import tensorflow as tf
import unittest
import pytest

import pyconfig
import multihost_dataloading


def assert_same_output_and_grad(f1,f2, targets, *inputs, f1_extra_inputs=[], f2_extra_inputs=[],f1_name="regular",f2_name="pipeline"):
  f1_inputs = (*inputs, *f1_extra_inputs)
  f2_inputs = (*inputs, *f2_extra_inputs)
  def f1_loss(*f1_inputs):
    return jnp.linalg.norm(f1(*f1_inputs) - targets)

  def f2_loss(*f2_inputs):
    return jnp.linalg.norm(f2(*f2_inputs) - targets)

  def print_norms(a,b,f1_name="regular",f2_name="pipeline",diff_name="diff"):
    a_norm = jnp.linalg.norm(a)
    b_norm = jnp.linalg.norm(b)
    diff_norm = jnp.linalg.norm(a-b)

    print(f"{diff_name} norm of {diff_norm}")
    print(f"{f1_name} norm of {a_norm}")
    print(f"{f2_name} norm of {b_norm}")

  def my_ravel(pytree):
    ravelled_tree = jax.tree_map(jnp.ravel, pytree)
    ravelled_leaves, _ = jax.tree_util.tree_flatten(ravelled_tree)
    return jnp.concatenate(ravelled_leaves)

  f1_value = f1(*f1_inputs)
  f2_value = f2(*f2_inputs)
  _, f1_grad = jax.value_and_grad(f1_loss)(*f1_inputs)
  _, f2_grad = jax.value_and_grad(f2_loss)(*f2_inputs)

  f1_grad = my_ravel(f1_grad)
  f2_grad = my_ravel(f2_grad)
  print(f"{f1_grad.shape=}")

  print_norms(f1_value, f2_value, f1_name=f"{f1_name} output", f2_name=f"{f2_name} output", diff_name="Output difference")
  print_norms(f1_grad, f2_grad, f1_name=f"{f1_name} grad", f2_name=f"{f2_name} grad", diff_name="Gradient difference")


class PipelineParallelismTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    batch_size = 4
    # pyconfig.initialize(
    #     [sys.argv[0], "configs/base.yml"],
    #     per_device_batch_size=1,
    #     run_name="test",
    #     mesh_axes=["data"],
    #     logical_axis_rules=[["batch", "data"]],
    #     data_sharding=["data"],
    #     base_output_directory="gs://max-experiments/",
    #     dataset_path="gs://maxtext-dataset/",
    #     enable_checkpointing=False,
    # )
    pyconfig.initialize(sys.argv)
    config = pyconfig.config
    def get_inputs(batch_size, sequence, features, n_layers):
        '''Get random inputs, and random targets
            Returns
                inputs: [global_batch, sequence, features]
                targets: [global_batch, sequence, features]
        '''

        # we pass in input with global batch, its up to the pipeline function to reshape to microbatches
        input_shape = [batch_size, sequence, features]
        k = jax.random.PRNGKey(2)
        inputs = jax.random.normal(k,input_shape, dtype=jnp.float32)

        # dummy targets same shape as inputs to use for a dummy loss function to check gradient correctness
        k = jax.random.PRNGKey(3)
        dummy_targets = jax.random.normal(k,input_shape, dtype=jnp.float32)

        inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
        inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)

        return inputs, dummy_targets, inputs_position, inputs_segmentation

  BlockLayer = llama2.LlamaDecoderLayer
  
  @pytest.mark.tpu
  def test_pipeline_parallelism_same_output_and_grad(self):
    first_batch = next(self.multihost_gen)
    sec_batch = next(self.multihost_gen)
    self.assertTrue(not np.array_equal(first_batch, sec_batch, equal_nan=True))


if __name__ == "__main__":
  unittest.main()





############# Pipeline Parallelism Test #########

import jax
from jax import numpy as jnp
from jax import tree_map
from jax.sharding import Mesh
from typing import Sequence
from absl import app
import os
#from layers import simple_decoder_layer
import common_types
import pyconfig
import functools
import max_utils
from layers import llama2
from flax.core import meta

import jax.numpy as jnp
import timing_util
from flax import linen as nn

def main(argv: Sequence[str]) -> None:
  # TODO: Reformat this test into the same format as other MaxText tests

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


  num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
  layers_per_stage = config.num_decoder_layers / (num_stages * config.num_pipeline_repeats)

  _, inputs, targets, inputs_position, inputs_segmentation = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)
  deterministic = False
  model_mode = common_types.MODEL_MODE_TRAIN

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)


  #decoder_layer_instance = simple_decoder_layer.SimpleDecoderLayer(config=config, mesh=mesh, name="layers")
  #decoder_layer_instance = llama2.LlamaDecoderLayer(config=config, mesh=mesh, name="layers")
  BlockLayer = llama2.LlamaDecoderLayer
  if config.num_layers_per_pipeline_stage == 1:
    pipeline_stage = BlockLayer(config=config, mesh=mesh)
  else:
    params_spec = 0
    cache_spec = 1
    pipeline_stage = nn.scan(
      BlockLayer,
      variable_axes={
          "params": params_spec,
          "cache": cache_spec,
          "intermediates": 0,
          "aqt": 0,
          "_overwrite_with_gradient": 0,
      },
      split_rngs={
          "params": True,
          "dropout": config.enable_dropout,
      },
      in_axes=(
          nn.broadcast,
          nn.broadcast,
          nn.broadcast,
          nn.broadcast,
      ),
      length=config.num_layers_per_pipeline_stage,
      metadata_params={nn.PARTITION_NAME: "layers_per_stage"},
  )(config=config, mesh=mesh, name="layers")






  from layers import pipeline_flax
  my_pipeline = pipeline_flax.Pipeline(
    config=config,
    layers=pipeline_stage,
    mesh=mesh
  )


  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
  pretty_print_pytree(init_pipeline_params)
  #pipeline_out = my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)


  def run_regular_pipeline(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode):
    decoder_layer_instance = BlockLayer(config=config, mesh=mesh)
    reg_layer_activations = inputs

    def get_cur_layer_params(params, layer_idx):
      circular_metadata_params={
        nn.PARTITION_NAME: "circular_repeats",
        'sub_weight_split_dims_mapping': (None,), #(None,), # Maybe -1? 
        "is_initializing": True,
        "x_times": config.num_pipeline_repeats,
        'optimizer_dims_mapping': None,
      }
      stage_metadata_params={
        nn.PARTITION_NAME: "layers",
        'sub_weight_split_dims_mapping': (None,), #(None,), # Maybe -1? 
        "is_initializing": True,
        "x_times": config.ici_pipeline_parallelism,
        'optimizer_dims_mapping': None,
      }
      def get_cur_layer_params_arr(leaf):
        if config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage == 1:
          new_shape = (leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]
          leaf = jnp.reshape(leaf, new_shape)
        elif config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage > 1:
          new_shape = (leaf.shape[0] * leaf.shape[1] * leaf.shape[2],) + leaf.shape[3:]
          leaf = jnp.reshape(leaf, new_shape)
        elif config.num_pipeline_repeats == 1 and config.num_layers_per_pipeline_stage > 1:
          new_shape = (leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]
          leaf = jnp.reshape(leaf, new_shape)
        return leaf[layer_idx]
      return jax.tree.map(get_cur_layer_params_arr, params)

    old=False
    for layer in range(config.num_decoder_layers):
      if old:
        cur_layer_params = params['params'][f'layers_{layer}']
        cur_layer_params = {'params':cur_layer_params}
      else:
        cur_layer_params = get_cur_layer_params(params, layer)
        cur_layer_params['params'] = cur_layer_params['params']['layers']
      if config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage > 1:
        cur_layer_params['params'] = meta.remove_axis(cur_layer_params['params'], 0, {nn.PARTITION_NAME:"circular_repeats"})
        cur_layer_params['params'] = meta.remove_axis(cur_layer_params['params'], 0, {nn.PARTITION_NAME:"layers"})
      reg_layer_activations, _ = decoder_layer_instance.apply(cur_layer_params, reg_layer_activations, inputs_position, inputs_segmentation, deterministic, model_mode)
    return reg_layer_activations


  
  reg_layers = run_regular_pipeline
  pipeline_func = my_pipeline.apply

  # pipeline_func(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
  # assert_same_output_and_grad runs non-jitted functions, which in particular will probably fail on multihost with "non-addresable" array error
  assert_same_output_and_grad(reg_layers,pipeline_func, targets, init_pipeline_params, inputs, inputs_segmentation, inputs_position, deterministic, model_mode)

  partial_pipeline_func = functools.partial(pipeline_func, deterministic=deterministic, model_mode=model_mode)

  jit_pipeline_func = jax.jit(partial_pipeline_func)
  #timing_util.simple_timeit(jit_pipeline_func, init_pipeline_params, inputs, inputs_segmentation, inputs_position, tries = 3, task = 'basic_pp')


if __name__ == "__main__":
  app.run(main)
  # Circular
  # python3 MaxText/pipeline_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=8 scan_layers=True num_pipeline_microbatches=12 num_pipeline_repeats=2
  # Non-circular
  # python3 MaxText/pipeline_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=4 scan_layers=True num_pipeline_microbatches=12 num_pipeline_repeats=1
  # Multiple layers per stage + circular (24 layers = 4 stages * 3 repeats * 2 layers per stage) !! Takes 2 min to run
  # python3 MaxText/pipeline_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=24  scan_layers=True num_pipeline_microbatches=4 num_pipeline_repeats=3 num_layers_per_pipeline_stage=2 per_device_batch_size=1
  # For timing:
  # python3 MaxText/pipeline_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False ici_pipeline_parallelism=4 base_num_decoder_layers=4 scan_layers=True num_pipeline_microbatches=4 num_pipeline_repeats=1 base_emb_dim=2560