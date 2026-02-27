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

"""
Tests for verifying losses and gradients match using/without using tiling methods:
- Gradient accumulation (GA)
- Vocabulary tiling (VT)
"""

import unittest
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText import pyconfig
from maxtext.common.common_types import Config
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.vocabulary_tiling import vocab_tiling_linen_loss
from tests.utils.test_helpers import get_test_config_path
import pytest


def compute_loss_linen(intermediate_outputs, logits, data, config, model, params, is_train):
  """
  A loss function wrapper that deals with both vocab tiling or non-vocab tiling cases
  """
  if config.num_vocab_tiling > 1:
    hidden_state_key = ("intermediates", "decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    total_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
  else:
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
    xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
    # Mask out paddings at the end of each example.
    xent = xent * (data["targets_segmentation"] != 0)
    total_loss = jnp.sum(xent)
  return total_loss


class LossAndGradientCorrectnessTest(unittest.TestCase):
  """
  Unit tests for verifying loss and gradient correctness of:
  - Gradient accumulation (GA)
  - Vocabulary tiling (VT)
  """

  def setUp(self):
    """
    Set up common configurations and dummy data for the tests.
    """
    self.base_config = [None, get_test_config_path()]
    self.rng = jax.random.PRNGKey(1234)
    self.batch_size = 1
    self.seq_len = 64
    self.dummy_inputs = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    self.rtol = 1e-2
    self.atol = 1e-2

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
            mutable=["intermediates"],
        )
        return compute_loss_linen(intermediate_outputs, logits, d, cfg, model, train_params, is_train=True)

      return jax.value_and_grad(loss_fn)(p)

    return grad_fn(params, data)

  def assert_pytrees_all_close(self, tree1, tree2, error_message=None):
    """Manually assert that two pytrees are all-close."""

    # Map jnp.allclose to every leaf
    # This creates a new pytree of the same structure, but with boolean leaves
    leaves_are_close = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=self.rtol, atol=self.atol),
        tree1,
        tree2,
    )

    # Get a flat list of all boolean leaves
    all_leaves = jax.tree_util.tree_leaves(leaves_are_close)

    # Check if every single leaf was True
    all_are_close = all(all_leaves)

    # Assert the final result
    if not all_are_close:
      # For a better error, you could find the first False leaf
      print("Pytrees are not all-close. Mismatched leaves:")
      # This part is more complex, which is why chex is preferred
      mismatches = jax.tree_util.tree_map(
          lambda x, y, z: "MISMATCH" if not z else "MATCH", tree1, tree2, leaves_are_close
      )
      print(mismatches)
      raise AssertionError(error_message)

  @pytest.mark.tpu_only
  def test_gradient_accumulation(self):
    """
    Tests GA loss and gradient correctness.
    """
    cfg_non_ga = pyconfig.initialize(
        self.base_config,
        run_name="non_GA_grad_test",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=4,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        gradient_accumulation_steps=1,
    )
    quant_non_ga = quantizations.configure_quantization(cfg_non_ga)
    devices_array_non_ga = maxtext_utils.create_device_mesh(cfg_non_ga)
    mesh_non_ga = Mesh(devices_array_non_ga, cfg_non_ga.mesh_axes)
    model_non_ga = models.transformer_as_linen(
        cfg_non_ga, mesh=mesh_non_ga, quant=quant_non_ga, model_mode=MODEL_MODE_TRAIN
    )

    rng_model, rng_targets = jax.random.split(self.rng)

    params = model_non_ga.init(
        {"params": rng_model, "dropout": rng_model},
        self.dummy_inputs,
        self.dummy_inputs,
    )

    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg_non_ga.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    loss_non_ga, grads_non_ga = self.get_grads(cfg_non_ga, params, data)

    cfg_ga = pyconfig.initialize(
        self.base_config,
        run_name="GA_grad_test",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=1,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        gradient_accumulation_steps=4,
    )
    loss_ga, grads_ga = self.get_grads(cfg_ga, params, data)
    # Loss correctness test
    assert jnp.allclose(loss_non_ga, loss_ga, rtol=self.rtol), "Losses do not match for gradient accumulation test."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_ga,
        grads_ga,
        "Gradients of embedding table do not match for GA.",
    )

  @pytest.mark.tpu_only
  def test_vocab_tiling_gradient_non_tied_embedding(self):
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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(
        cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN
    )

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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match for non-tied embeddings."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_tiling,
        grads_tiling,
        "Gradients of embedding table do not match for non-tied embeddings.",
    )

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_vocab_tiling_gradient_tied_embedding(self):
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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
    )

    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(
        cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN
    )

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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)

    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match for tied embeddings."

    self.assert_pytrees_all_close(
        grads_non_tiling, grads_tiling, "Gradients of embedding table do not match for tied embeddings."
    )

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_vocab_tiling_gradient_data_parallelism(self):
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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(
        cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN
    )

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
        dtype="float32",
        matmul_precision="high",
        ici_data_parallelism=4,
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match for data parallelism."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_tiling, grads_tiling, "Gradients of embedding table do not match for data parallelism."
    )

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_vocab_tiling_gradient_tensor_parallelism(self):
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
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(
        cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN
    )

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
        dtype="float32",
        matmul_precision="high",
        ici_tensor_parallelism=4,
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)
    # Loss correctness test
    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match for tensor parallelism."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_tiling, grads_tiling, "Gradients of embedding table do not match for tensor parallelism."
    )

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_vocab_tiling_gradient_context_parallelism(self):
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
        dataset_type="synthetic",
        packing=False,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
    )
    quant_non_tiling = quantizations.configure_quantization(cfg_non_tiling)
    devices_array_non_tiling = maxtext_utils.create_device_mesh(cfg_non_tiling)
    mesh_non_tiling = Mesh(devices_array_non_tiling, cfg_non_tiling.mesh_axes)
    model_non_tiling = models.transformer_as_linen(
        cfg_non_tiling, mesh=mesh_non_tiling, quant=quant_non_tiling, model_mode=MODEL_MODE_TRAIN
    )

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
        dataset_type="synthetic",
        packing=False,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=4,
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)

    # Loss correctness test
    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match for context parallelism."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_tiling, grads_tiling, "Gradients of embedding table do not match for context parallelism."
    )
