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

"""Tests for GPT3."""

import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest


def init_random_model_vars(model, rng, example_batch):
  """Initialize random model vars."""
  model_vars = model.init(
      {"params": rng, "aqt": rng},
      example_batch["inputs"],
      example_batch["inputs_position"],
      enable_dropout=False,
  )

  def _replace_initialization(key, value):
    keystr = jax.tree_util.keystr(key)
    # replace zero initializer to ensure strong test cases
    #   including Gpt3LayerNorm scale, Gpt3LayerNorm bias, and dense_general bias
    if "scale" in keystr or "bias" in keystr:
      value = jax.nn.initializers.normal(1.0)(rng, value.shape, dtype=value.dtype)
    return value

  model_vars = jax.tree_util.tree_map_with_path(_replace_initialization, model_vars)
  return model_vars


class GPT3(unittest.TestCase):
  """Numerical tests for GPT3."""

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test",
        enable_checkpointing=False,
        model_name="gpt3-52k",
        dtype="float32",
        per_device_batch_size=1.0 / jax.device_count(),
    )
    self.rng = jax.random.PRNGKey(1234)

    devices_array = maxtext_utils.create_device_mesh(self.cfg, devices=[jax.devices()[0]])
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    self.model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    self.example_batch = {
        "inputs": jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        "inputs_position": jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        "inputs_segmentation": jnp.array([[1, 1, 1, 1, 1]], dtype=jnp.int32),
        "targets": jnp.array([[12, 13, 14, 15, 1]], dtype=jnp.int32),
        "targets_position": jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        "targets_segmentation": jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }
    self.model_vars = init_random_model_vars(self.model, self.rng, self.example_batch)

  @pytest.mark.skip(reason="Numerical differences large on jax>0.5.0")
  @pytest.mark.tpu_only
  def test_logits_numerically(self):
    # ground truth values are calculated from paxml after loading above model_vars
    # note we expect all xents are the same except the padding one since:
    #    paxml applies padding in mlp layer
    #    while maxtext implementation applies padding in attention mask instead
    # the two implementation are equivalent in valid non-padding tokens
    per_example_xent_truth = jnp.array([[31.976467, 25.806253, 17.311134, 45.362663, 0.0]], dtype=jnp.float32)
    logits, _ = self.model.apply(
        self.model_vars,
        self.example_batch["inputs"],
        self.example_batch["inputs_position"],
        decoder_segment_ids=self.example_batch["inputs_segmentation"],
        enable_dropout=self.cfg.enable_dropout,
        rngs={"dropout": self.rng, "aqt": self.rng},
        mutable="intermediates",
    )

    one_hot_targets = jax.nn.one_hot(self.example_batch["targets"], self.cfg.vocab_size)
    per_example_xent = -jnp.sum(jax.nn.log_softmax(logits) * one_hot_targets, axis=-1, dtype=jnp.float32)
    # Mask out paddings at the end of each example.
    per_example_xent = per_example_xent * (self.example_batch["targets_segmentation"] != 0)

    self.assertTrue(
        jax.numpy.allclose(per_example_xent, per_example_xent_truth, rtol=1e-03, atol=1e-03),
        msg=f"per_example_xent:\n{per_example_xent}\n\nper_example_xent_truth:\n{per_example_xent_truth}",
    )

  @pytest.mark.tpu_only
  def test_prefill_and_autoregress(self):
    """Verifies that GPT-3 attention correctly initializes and updates the KV cache during decoding."""
    PREFILL_RANGE = 2
    devices_array = maxtext_utils.create_device_mesh(self.cfg, devices=[jax.devices()[0]])
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    prefill_model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_PREFILL)

    # 0. Compute baseline full training logits for numerical equivalence comparison.
    full_train_logits, _ = self.model.apply(
        self.model_vars,
        self.example_batch["inputs"],
        self.example_batch["inputs_position"],
        decoder_segment_ids=self.example_batch["inputs_segmentation"],
        enable_dropout=False,
        rngs={"dropout": self.rng, "aqt": self.rng},
        mutable="intermediates",
    )

    # 1. Initialize model variables and KV cache structures in prefill mode.
    prefill_transformer_vars = prefill_model.init(
        {"params": self.rng, "aqt": self.rng},
        self.example_batch["inputs"],
        self.example_batch["inputs_position"],
        model_mode=MODEL_MODE_PREFILL,
        decoder_segment_ids=self.example_batch["inputs_segmentation"],
        enable_dropout=False,
    )
    # Replace zero initializers with normal distribution to ensure strong numerical test cases
    prefill_transformer_vars = init_random_model_vars(prefill_model, self.rng, self.example_batch)

    # 2. Execute a partial prefill pass to populate the KV cache.
    partial_prefill_logits, partial_cache = prefill_model.apply(
        prefill_transformer_vars,
        self.example_batch["inputs"][:, :PREFILL_RANGE],
        self.example_batch["inputs_position"][:, :PREFILL_RANGE],
        model_mode=MODEL_MODE_PREFILL,
        decoder_segment_ids=self.example_batch["inputs_segmentation"][:, :PREFILL_RANGE],
        enable_dropout=False,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    # Verify partial prefill exactly matches full training logits
    np.testing.assert_allclose(
        full_train_logits[:, :PREFILL_RANGE, :],
        partial_prefill_logits,
        rtol=1e-01,
        atol=1e-01,
    )

    # 3. Perform an autoregressive decoding step using the updated KV cache.
    idx = PREFILL_RANGE
    ids_idx = self.example_batch["inputs"][:, idx : idx + 1]
    decoder_positions_idx = self.example_batch["inputs_position"][:, idx : idx + 1]
    prefill_transformer_vars.update(partial_cache)
    ar_logits, _ = prefill_model.apply(
        prefill_transformer_vars,
        ids_idx,
        decoder_positions_idx,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        enable_dropout=False,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    # Verify autoregressive decoding exactly matches full training logits at the decoded position
    np.testing.assert_allclose(
        full_train_logits[:, idx : idx + 1, :],
        ar_logits,
        rtol=1e-01,
        atol=1e-01,
    )
