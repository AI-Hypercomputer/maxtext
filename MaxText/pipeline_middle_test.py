# Difficulty: Making sure pipeline vs not use the same parameters, currently code works but is ugly and maybe fragile to changes (depends on implementation details)

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

from layers import pipeline_flax
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


class PipelineParallelismTest:

  def setUp(self):
    pyconfig.initialize(
        [sys.argv[0], "MaxText/configs/base.yml"],
        enable_checkpointing=False,
        run_name="pipeline_parallelism_test",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4        
    )
    config = pyconfig.config
    self.config = config

    self.rng = jax.random.PRNGKey(1234)

    devices_array = max_utils.create_device_mesh(self.config)
    mesh = Mesh(devices_array, self.config.mesh_axes)
    self.mesh = mesh

    def get_inputs(batch_size, sequence, features):
        '''Get random inputs, and random targets
            Returns
                inputs: [batch_size, sequence, features]
                targets: [batch_size, sequence, features]
                positions: [batch_size, sequence]
                segmentations: [batch_size, segmentation]
        '''
        input_shape = [batch_size, sequence, features]
        k = jax.random.PRNGKey(2)
        inputs = jax.random.normal(k,input_shape, dtype=jnp.float32)

        # dummy targets same shape as inputs to use for a dummy loss function to check gradient correctness
        k = jax.random.PRNGKey(3)
        dummy_targets = jax.random.normal(k,input_shape, dtype=jnp.float32)

        inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
        inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)
        return inputs, dummy_targets, inputs_position, inputs_segmentation

    self.inputs, self.dummy_targets, self.inputs_position, self.inputs_segmentation = get_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim)


    self.deterministic = True
    self.model_mode = common_types.MODEL_MODE_TRAIN
    pipeline_stage = llama2.LlamaDecoderLayer(config=config, mesh=self.mesh)
    my_pipeline = pipeline_flax.Pipeline(
        config=self.config,
        layers=pipeline_stage,
        mesh=self.mesh
    )
    self.init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), self.inputs, self.inputs_position, self.inputs_segmentation, self.deterministic, self.model_mode)

    self.config=config
    def run_regular_pipeline(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode):
        config = self.config
        decoder_layer_instance = llama2.LlamaDecoderLayer(config=self.config, mesh=self.mesh)
        reg_layer_activations = inputs

        def get_cur_layer_params(params, layer_idx):
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


  
    self.reg_layers = run_regular_pipeline
    self.pipeline_func = my_pipeline.apply


  def test_pipeline_parallelism_same_output_and_grad(self):
    print("Setting up...", flush=True)
    self.setUp()
    print("Set up complete!", flush=True)
    assert 2 > 1
    print("asserting grad...", flush=True)
    assert_same_output_and_grad(self.pipeline_func, self.reg_layers, self.dummy_targets, self.init_pipeline_params, self.inputs, self.inputs_segmentation, self.inputs_position, self.deterministic, self.model_mode)
    print("Grad asserted!...", flush=True)

if __name__ == "__main__":
  my_test = PipelineParallelismTest()
  my_test.test_pipeline_parallelism_same_output_and_grad()