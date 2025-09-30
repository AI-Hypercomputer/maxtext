# Copyright 2025 Google LLC
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

""" Tests for verifying losses and gradients match using/without using vocab tiling."""

import unittest
import pytest
import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.core import freeze

from MaxText import maxtext_utils
from MaxText.maxtext_utils import compute_loss_linen
from MaxText.common_types import Config
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models
from MaxText.layers import quantizations


class LossAndGradientCorrectnessTest(unittest.TestCase):
  """
  Unit tests for verifying loss and gradient correctness of vocabulary tiling.
  """

  def setUp(self):
    """
    Set up common configurations and dummy data for the tests.
    """
    self.base_config = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")]
    self.rng = jax.random.PRNGKey(1234)
    self.batch_size = 1
    self.seq_len = 64
    self.dummy_inputs = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    self.rtol = 1e-2

  def get_grads(self, cfg: Config, params, data):
    """
    Computes and returns the gradients for a given configuration and set of parameters.
    """
    quant = quantizations.configure_quantization(cfg)
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

    @jax.jit
    def grad_fn(p, d):
      def loss_fn(train_params):
        logits, intermediate_outputs = model.apply(
            train_params,
            decoder_input_tokens=self.dummy_inputs,
            decoder_positions=self.dummy_inputs,
            mutable=['intermediates']
        )
        return compute_loss_linen(
            intermediate_outputs, logits, d, cfg, model, train_params, is_train=True
        )
      return jax.value_and_grad(loss_fn)(p)

    return grad_fn(params, data)

  @pytest.mark.tpu_only
  def test_tiling_gradient_non_tied_embedding(self):
    """
    Tests loss and gradient correctness for a model with non-tied embeddings (FSDP).
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="grad_test_non_tied_no_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN)
    
    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_tiling.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_tiling.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_tiling, grads_non_tiling = self.get_grads(cfg_non_tiling, params, data)

    cfg_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_non_tied_with_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(
      loss_non_tiling,
      loss_tiling,
      rtol=self.rtol
    ), "Losses do not match for non-tied embeddings."

    # Gradient correctness test
    assert jnp.allclose(
        grads_non_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        grads_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        rtol=self.rtol
    ), "Gradients of embedding table do not match for non-tied embeddings."

  @pytest.mark.tpu_only
  def test_tiling_gradient_tied_embedding(self):
    """
    Tests loss and gradient correctness for a model with tied embeddings (FSDP).
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_tied_no_tiling",
        enable_checkpointing=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=True,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=1,
    )

    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN)

    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_tiling.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_tiling.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_tiling, grads_non_tiling = self.get_grads(cfg_non_tiling, params, data)

    cfg_tiling = pyconfig.initialize(
        self.base_config,
        run_name="grad_test_tied_with_tiling",
        enable_checkpointing=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=True,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)

    assert jnp.allclose(
      loss_non_tiling,
      loss_tiling,
      rtol=self.rtol
    ), "Losses do not match for tied embeddings."

    assert jnp.allclose(
        grads_non_tiling['params']['token_embedder']['embedding'].unbox(),
        grads_tiling['params']['token_embedder']['embedding'].unbox(),
        rtol=self.rtol
    ), "Gradients of embedding table do not match for tied embeddings."


  @pytest.mark.tpu_only
  def test_tiling_gradient_data_parallelism(self):
    """
    Tests loss and gradient correctness for data parallelism sharding.
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_dp_non_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        ici_data_parallelism=4,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN)
    
    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_tiling.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_tiling.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_tiling, grads_non_tiling = self.get_grads(cfg_non_tiling, params, data)

    cfg_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_dp_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        ici_data_parallelism=4,
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(
      loss_non_tiling,
      loss_tiling,
      rtol=self.rtol
    ), "Losses do not match for data parallelism."

    # Gradient correctness test
    assert jnp.allclose(
        grads_non_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        grads_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        rtol=self.rtol
    ), "Gradients of embedding table do not match for data parallelism."


  @pytest.mark.tpu_only
  def test_tiling_gradient_tensor_parallelism(self):
    """
    Tests loss and gradient correctness for tensor parallelism sharding.
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_tp_non_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        ici_tensor_parallelism=4,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN)
    
    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_tiling.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_tiling.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_tiling, grads_non_tiling = self.get_grads(cfg_non_tiling, params, data)

    cfg_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_tp_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype='float32',
        matmul_precision='high',
        ici_tensor_parallelism=4,
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(
      loss_non_tiling,
      loss_tiling,
      rtol=self.rtol
    ), "Losses do not match for tensor parallelism."

    # Gradient correctness test
    assert jnp.allclose(
        grads_non_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        grads_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        rtol=self.rtol
    ), "Gradients of embedding table do not match for tensor parallelism."

  @pytest.mark.tpu_only
  def test_tiling_gradient_context_parallelism(self):
    """
    Tests loss and gradient correctness for context parallelism sharding.
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_cp_non_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        ici_context_parallelism=4,
        base_num_decoder_layers=0,
        dataset_type='synthetic',
        packing=False,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN)
    
    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_tiling.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_tiling.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_tiling, grads_non_tiling = self.get_grads(cfg_non_tiling, params, data)

    cfg_tiling = pyconfig.initialize(
        self.base_config,
        run_name="value_and_grad_test_cp_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        ici_context_parallelism=4,
        dataset_type='synthetic',
        packing=False,
        dtype='float32',
        matmul_precision='high',
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)

    # Loss correctness test
    assert jnp.allclose(
      loss_non_tiling,
      loss_tiling,
      rtol=self.rtol
    ), "Losses do not match for context parallelism."

    # Gradient correctness test
    assert jnp.allclose(
        grads_non_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        grads_tiling['params']['decoder']['logits_dense']['kernel'].unbox(),
        rtol=self.rtol
    ), "Gradients of embedding table do not match for context parallelism."
