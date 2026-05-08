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
import pytest

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.common.common_types import Config
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils.vocabulary_tiling import vocab_tiling_linen_loss, vocab_tiling_nnx_loss

from tests.utils.test_helpers import get_test_config_path


def compute_loss_linen(intermediate_outputs, logits, data, config, model, params, is_train):
  """
  A loss function wrapper that deals with both vocab tiling or non-vocab tiling cases
  """
  if config.num_vocab_tiling > 1:
    hidden_state_key = ("intermediates", "decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    total_loss, _ = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
  else:
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)
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
    # vocab_tiling on the Linen path uses transformer_as_linen + model.apply,
    # so this class must stay on Linen even when NNX defaults are flipped to
    # True. The NNX-side equivalents live in VocabTilingNNXTest below.
    self.base_config = [
        None,
        get_test_config_path(),
        "base_emb_dim=32",
        "vocab_size=128",
        "enable_nnx=False",
        "pure_nnx=False",
        "pure_nnx_decoder=False",
    ]
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
  def test_vocab_tiling_gradient_with_z_loss(self):
    """
    Tests loss and gradient correctness when z-loss is enabled, comparing
    standard computation vs. vocabulary tiling computation.
    """
    cfg_non_tiling = pyconfig.initialize(
        self.base_config,
        run_name="grad_test_z_loss_no_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=1,
        z_loss_multiplier=1e-4,  # Enable z-loss
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
        run_name="grad_test_z_loss_with_tiling",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=4,
        z_loss_multiplier=1e-4,  # Enable z-loss
    )
    loss_tiling, grads_tiling = self.get_grads(cfg_tiling, params, data)

    # Loss correctness test
    assert jnp.allclose(loss_non_tiling, loss_tiling, rtol=self.rtol), "Losses do not match when z-loss is enabled."

    # Gradient correctness test
    self.assert_pytrees_all_close(
        grads_non_tiling,
        grads_tiling,
        "Gradients do not match for vocab tiling when z-loss is enabled.",
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


class VocabTilingNNXTest(unittest.TestCase):
  """Loss + gradient parity for the NNX vocab-tiling `custom_vjp` path.

  Compares two computations against the same NNX model:
    - reference: full-vocab `model.logits_from_hidden_states(...)` then xent over the whole vocab.
    - tiled: `vocab_tiling_nnx_loss(...)` which scans over `num_vocab_tiling` chunks
      and uses the new `custom_vjp` for the backward.

  Both paths share the same params; the test checks that loss values and parameter
  gradients match within tolerance, exercising both forward and backward.
  """

  def setUp(self):
    self.base_config = [None, get_test_config_path()]
    self.rng = jax.random.PRNGKey(1234)
    # Global batch must divide fsdp axis (= jax.device_count() by default), so the
    # batch sharding constraints inside vocab_tiling_nnx_loss are satisfied.
    self.batch_size = jax.device_count()
    self.seq_len = 64
    self.rtol = 1e-2
    self.atol = 1e-2

  def _build_cfg_and_model(
      self,
      *,
      num_vocab_tiling=4,
      logits_via_embedding=False,
      z_loss_multiplier=1e-4,
  ):
    """Build a pyconfig + matching NNX `Transformer` for the test."""
    cfg = pyconfig.initialize(
        self.base_config,
        run_name=f"vt_nnx_n{num_vocab_tiling}_emb{logits_via_embedding}_z{z_loss_multiplier}",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=1,
        logits_via_embedding=logits_via_embedding,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=num_vocab_tiling,
        z_loss_multiplier=z_loss_multiplier,
        pure_nnx=True,
        enable_nnx=True,
        pure_nnx_decoder=True,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
    return cfg, model

  def _make_inputs(self, cfg, *, dtype=jnp.float32, pad_half=False):
    """Synthetic hidden_states/labels/segmentation; `pad_half=True` zeros the back half of seg."""
    rng_hidden, rng_targets = jax.random.split(self.rng)
    hidden_states = jax.random.normal(rng_hidden, (self.batch_size, self.seq_len, cfg.emb_dim), dtype=dtype)
    labels = jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg.vocab_size)
    if pad_half:
      half = self.seq_len // 2
      segmentation = jnp.concatenate(
          [
              jnp.ones((self.batch_size, half), dtype=jnp.int32),
              jnp.zeros((self.batch_size, self.seq_len - half), dtype=jnp.int32),
          ],
          axis=1,
      )
    else:
      segmentation = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    return hidden_states, labels, segmentation

  def _reference_loss_fn(self, cfg, graphdef, rest, hidden_states, labels, segmentation):
    """Full-vocab xent loss closure (params, hidden_states) -> scalar loss."""

    def loss_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      logits = local_model.logits_from_hidden_states(h, True, "train")
      one_hot = jax.nn.one_hot(labels, cfg.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=cfg.z_loss_multiplier)
      return jnp.sum(xent * (segmentation != 0))

    return loss_fn

  def _tiled_loss_fn(self, cfg, graphdef, rest, hidden_states, labels, segmentation):
    """vocab_tiling_nnx_loss closure (params, hidden_states) -> scalar loss."""
    # hidden_states unused at the closure boundary (it comes via h), but kept in the
    # signature so the two closures are callable interchangeably.
    del hidden_states
    data = {"targets": labels, "targets_segmentation": segmentation}

    def loss_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      total_loss, _ = vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)
      return total_loss

    return loss_fn

  def _split_and_axes(self, cfg, model):
    """Common boilerplate: split the model and bind the logical axis rules."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    return graphdef, params, rest

  def _assert_pytrees_close(self, ref, tiled, msg, *, rtol=None, atol=None):
    rtol = self.rtol if rtol is None else rtol
    atol = self.atol if atol is None else atol
    leaves_close = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), ref, tiled)
    if not all(jax.tree_util.tree_leaves(leaves_close)):
      raise AssertionError(msg)

  @staticmethod
  def _vg(fn, *, argnums=0):
    """jit-wrapped value_and_grad. Required because eager-mode value_and_grad
    triggers a reshape-resharding `IndivisibleError` under recent JAX versions
    when bridging `P('fsdp', None, None)` inputs to `P(None, 'fsdp', None)`
    reshape outputs; XLA's SPMD compiler handles the same resharding cleanly
    under jit. Production train.py jit-wraps its train_step too, so this
    matches the real call site."""
    return jax.jit(jax.value_and_grad(fn, argnums=argnums))

  @staticmethod
  def _g(fn, *, argnums=0):
    """jit-wrapped grad — see `_vg`."""
    return jax.jit(jax.grad(fn, argnums=argnums))

  def _run_parity(self, *, logits_via_embedding):
    """Compare full-vocab xent loss/grads against the tiled custom_vjp path."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4, logits_via_embedding=logits_via_embedding)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hidden_states)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)

    assert jnp.allclose(
        ref_loss, tile_loss, rtol=self.rtol, atol=self.atol
    ), f"Losses differ: ref={ref_loss} tiled={tile_loss}"
    self._assert_pytrees_close(ref_grads, tile_grads, "Param gradients differ between full-vocab and tiled paths.")

  # ---------- Original parity tests (params gradient under both embedding modes) ----------

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_non_tied_embedding(self):
    """custom_vjp parity for non-tied embedding (separate logits_dense)."""
    self._run_parity(logits_via_embedding=False)

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_tied_embedding(self):
    """custom_vjp parity when logits share the input embedding table."""
    self._run_parity(logits_via_embedding=True)

  # ---------- Coverage expansion ----------

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_total_z_loss_value_parity(self):
    """The second tuple element (total_z_loss) must match the full-vocab reference."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)
    data = {"targets": labels, "targets_segmentation": segmentation}

    def _ref(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      logits = local_model.logits_from_hidden_states(h, True, "train")
      one_hot = jax.nn.one_hot(labels, cfg.vocab_size)
      xent_ref, z_ref = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=cfg.z_loss_multiplier)
      return jnp.sum(xent_ref * (segmentation != 0)), jnp.sum(z_ref * (segmentation != 0))

    def _tile(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      return vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_total_loss, ref_total_z_loss = jax.jit(_ref)(params, hidden_states)
      tile_total_loss, tile_total_z_loss = jax.jit(_tile)(params, hidden_states)

    assert jnp.allclose(ref_total_loss, tile_total_loss, rtol=self.rtol, atol=self.atol)
    assert jnp.allclose(
        ref_total_z_loss, tile_total_z_loss, rtol=self.rtol, atol=self.atol
    ), f"total_z_loss differs: ref={ref_total_z_loss} tiled={tile_total_z_loss}"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_padded_segmentation(self):
    """Half-padded segmentation: mask actually changes the loss, and parity holds."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)

    # Compare unpadded vs padded loss to confirm the mask is wired through.
    hs, labels, full_seg = self._make_inputs(cfg, pad_half=False)
    _, _, pad_seg = self._make_inputs(cfg, pad_half=True)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    def _tile_loss_only(p, h, seg):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      total, _ = vocab_tiling_nnx_loss(
          local_model, h, {"targets": labels, "targets_segmentation": seg}, cfg, is_train=True
      )
      return total

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      full_loss = jax.jit(_tile_loss_only)(params, hs, full_seg)
      pad_loss = jax.jit(_tile_loss_only)(params, hs, pad_seg)
    assert float(pad_loss) < float(
        full_loss
    ), f"Padded loss should be strictly smaller (fewer tokens contribute). full={full_loss} pad={pad_loss}"

    # Now check parity against the full-vocab reference using the padded mask.
    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hs, labels, pad_seg)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hs, labels, pad_seg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hs)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hs)
    assert jnp.allclose(ref_loss, tile_loss, rtol=self.rtol, atol=self.atol)
    self._assert_pytrees_close(ref_grads, tile_grads, "Padded-segmentation gradients differ.")

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_grad_over_hidden_states(self):
    """Differentiate w.r.t. hidden_states (argnums=1): the second-primal cotangent path of custom_vjp."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_grad_h = self._g(ref_loss_fn, argnums=1)(params, hidden_states)
      tile_grad_h = self._g(tile_loss_fn, argnums=1)(params, hidden_states)

    assert ref_grad_h.shape == hidden_states.shape
    assert tile_grad_h.shape == hidden_states.shape
    assert ref_grad_h.dtype == hidden_states.dtype
    assert tile_grad_h.dtype == hidden_states.dtype
    assert jnp.allclose(ref_grad_h, tile_grad_h, rtol=self.rtol, atol=self.atol), "grad_hidden_states diverged"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_bf16_hidden_states(self):
    """bf16 hidden_states: the bwd dtype-cast (`y.astype(x.dtype)`) preserves parity at lower precision."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg, dtype=jnp.bfloat16)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grad_h = self._vg(ref_loss_fn, argnums=1)(params, hidden_states)
      tile_loss, tile_grad_h = self._vg(tile_loss_fn, argnums=1)(params, hidden_states)

    # bf16 has ~3 decimal digits — loosen tolerance.
    assert jnp.allclose(ref_loss, tile_loss, rtol=5e-2, atol=5e-2)
    assert tile_grad_h.dtype == jnp.bfloat16, f"grad cast to primal dtype expected bf16, got {tile_grad_h.dtype}"
    assert jnp.allclose(ref_grad_h, tile_grad_h, rtol=5e-2, atol=5e-2)

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_z_loss_zero(self):
    """z_loss=0: total_z_loss is exactly zero; loss/grad parity still holds."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4, z_loss_multiplier=0.0)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)
    data = {"targets": labels, "targets_segmentation": segmentation}

    def _tile_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      return vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      total_loss, total_z_loss = jax.jit(_tile_fn)(params, hidden_states)
    assert float(total_z_loss) == 0.0, f"z_loss=0 but tile path returned {total_z_loss}"
    assert float(total_loss) > 0.0  # cross-entropy on random logits should be positive

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hidden_states)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)
    assert jnp.allclose(ref_loss, tile_loss, rtol=self.rtol, atol=self.atol)
    self._assert_pytrees_close(ref_grads, tile_grads, "z_loss=0 gradients differ.")

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_other_params_get_zero_grad(self):
    """Output-head carve-out invariant: every non-head nnx.Param leaf gets exactly zero grad.

    The output-head carve-out splits the model into head_params (used by
    `logits_from_hidden_states`) vs. other_params (transformer layers, etc.),
    threading other_params through the custom_vjp as a non-differentiated primal
    whose bwd cotangent is `tree_map(jnp.zeros_like, ...)`. This test asserts the
    contract: the gradient at every non-head path is exactly 0, and at least one
    head path has a non-zero gradient (so it isn't trivially passing because some
    bug zeroed everything).
    """
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      _, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)

    head_keywords = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")
    head_nonzero_seen = False
    for path, leaf in jax.tree_util.tree_leaves_with_path(tile_grads):
      path_str = jax.tree_util.keystr(path)
      is_head = any(kw in path_str for kw in head_keywords)
      if is_head:
        if jnp.any(leaf != 0):
          head_nonzero_seen = True
      else:
        assert jnp.all(leaf == 0), f"non-head leaf {path_str} has non-zero grad — carve-out is wrong"
    assert head_nonzero_seen, "expected at least one head leaf with non-zero grad; got all zeros"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_num_vocab_tiling_variants(self):
    """Different num_vocab_tiling values (2, 4, 8) all produce identical loss + grads."""
    losses = []
    grads_list = []
    for n in (2, 4, 8):
      cfg, model = self._build_cfg_and_model(num_vocab_tiling=n)
      hidden_states, labels, segmentation = self._make_inputs(cfg)
      graphdef, params, rest = self._split_and_axes(cfg, model)
      tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        loss, grads = self._vg(tile_loss_fn)(params, hidden_states)
      losses.append(loss)
      grads_list.append(grads)

    base_loss = losses[0]
    base_grads = grads_list[0]
    for n, loss, grads in zip((2, 4, 8), losses, grads_list):
      assert jnp.allclose(
          loss, base_loss, rtol=self.rtol, atol=self.atol
      ), f"num_vocab_tiling={n}: loss diverges from n=2 baseline ({loss} vs {base_loss})"
      self._assert_pytrees_close(base_grads, grads, f"num_vocab_tiling={n}: grads diverge from n=2 baseline.")
