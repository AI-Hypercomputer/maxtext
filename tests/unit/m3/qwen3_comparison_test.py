# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comparison, unit, and parity tests for M3 Qwen3 implementation.

Covers:
(1) Model routing and construction via model_creation_utils
(2) Forward pass execution under different batch sizes, sequence lengths, and modes
(3) Parameter tree, paths, shapes, and dtypes parity against Flax Linen
(4) Numerical parity of forward pass logits
(5) Numerical parity of cross-entropy loss and backward gradients
"""

import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import (
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Model, create_qwen3_model
from maxtext.utils import max_utils, model_creation_utils


def _get_base_config_args(use_m3: bool, reduced: bool = True, tp: int = 1):
  """Generates CLI config arguments for Qwen3 comparison tests."""
  args = [
      "",
      "src/maxtext/configs/base.yml",
      "model_name=qwen3-0.6b",
      f"use_m3_model={str(use_m3).lower()}",
      "scan_layers=false",
      "weight_dtype=bfloat16",
      "per_device_batch_size=1",
      "max_target_length=32",
      "max_prefill_predict_length=16",
  ]
  if reduced:
    args.extend(
        [
            "override_model_config=true",
            "num_decoder_layers=2",
            "emb_dim=64",
            "mlp_dim=128",
            "num_query_heads=4",
            "num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
        ]
    )
  if tp > 1:
    args.append(f"ici_tensor_parallelism={tp}")
  return args


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class Qwen3ComparisonTest(unittest.TestCase):
  """Compares legacy Qwen3 (use_m3_model=false) vs modern M3 Qwen3 (use_m3_model=true)."""

  def setUp(self):
    """Sets up device allocations for comparison tests."""
    super().setUp()
    self.num_devices = min(len(jax.devices()), 4)
    self.devices = jax.devices()[: self.num_devices]

  def _make_mesh(self, tp: int = 1):
    """Creates a device mesh."""
    cfg = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=True, tp=tp))
    mesh_shape = [1] * len(cfg.mesh_axes)
    tp_axis_idx = cfg.mesh_axes.index("tensor")
    mesh_shape[tp_axis_idx] = tp
    mesh_array = np.array(self.devices[:tp]).reshape(mesh_shape)
    return Mesh(mesh_array, cfg.mesh_axes)

  def test_model_creation_utils_routing(self):
    """Verifies that model_creation_utils.create_model routes to Qwen3Model when use_m3_model=true."""
    cfg = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)
    with jax.set_mesh(mesh):
      model = model_creation_utils.create_model(cfg, mesh)
      self.assertIsInstance(model, Qwen3Model)

  def test_qwen3_forward_pass_train(self):
    """Verifies train forward pass executes and produces correct logits shape."""
    cfg = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)
    with jax.set_mesh(mesh):
      model = create_qwen3_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN)
      batch_size, seq_len = 2, 8
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN)
      self.assertEqual(logits.shape, (batch_size, seq_len, cfg.vocab_size))

  def test_qwen3_forward_pass_variable_length(self):
    """Verifies forward pass executes on different sequence lengths and batch sizes."""
    cfg = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)
    with jax.set_mesh(mesh):
      model = create_qwen3_model(cfg, mesh)
      batch_size, seq_len = 1, 4
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      logits = model(tokens, positions)
      self.assertEqual(logits.shape, (batch_size, seq_len, cfg.vocab_size))

  def test_parameter_shapes_and_paths_match(self):
    """Verifies that all parameter paths, shapes, and dtypes match 100% with Flax Linen."""
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=False, tp=1))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=False, tp=1))
    mesh = self._make_mesh(tp=1)

    _, abs_old = model_creation_utils.create_nnx_abstract_model(cfg_old, mesh)
    _, abs_new = model_creation_utils.create_nnx_abstract_model(cfg_new, mesh)

    p_old = nnx.state(abs_old, nnx.Param)
    p_new = nnx.state(abs_new, nnx.Param)

    flat_old = dict(p_old.flat_state())
    flat_new = dict(p_new.flat_state())

    self.assertEqual(
        len(flat_old), len(flat_new), f"Parameter counts differ: old has {len(flat_old)}, new has {len(flat_new)}"
    )
    self.assertEqual(set(flat_old.keys()), set(flat_new.keys()), "Parameter keys / paths do not match")

    for key in sorted(flat_old.keys()):
      val_old = flat_old[key]
      val_new = flat_new[key]

      self.assertEqual(val_old.shape, val_new.shape, f"Shape mismatch for {key}: {val_old.shape} vs {val_new.shape}")
      self.assertEqual(val_old.dtype, val_new.dtype, f"Dtype mismatch for {key}: {val_old.dtype} vs {val_new.dtype}")

  def test_forward_numerical_parity(self):
    """Verifies numerical equivalence of forward pass logits under identical parameters."""
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=True))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)

    with jax.set_mesh(mesh):
      rngs = nnx.Rngs(42)
      model_old = model_creation_utils.create_model(cfg_old, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
      model_new = model_creation_utils.create_model(cfg_new, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

      p_old = nnx.state(model_old, nnx.Param)
      nnx.update(model_new, p_old)

      batch_size, seq_len = 2, 8
      tokens = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, cfg_old.vocab_size)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))

      logits_old = model_old(tokens, positions, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
      logits_new = model_new(tokens, positions, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)

      max_diff = float(jnp.max(jnp.abs(logits_old - logits_new)))
      self.assertLess(max_diff, 1e-4, f"Max logits diff too high: {max_diff}")

  def test_loss_and_gradients_parity(self):
    """Verifies numerical parity of cross-entropy loss and parameter gradients."""
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=True))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)

    with jax.set_mesh(mesh):
      rngs = nnx.Rngs(42)
      model_old = model_creation_utils.create_model(cfg_old, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
      model_new = model_creation_utils.create_model(cfg_new, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

      p_old = nnx.state(model_old, nnx.Param)
      nnx.update(model_new, p_old)

      batch_size, seq_len = 2, 8
      tokens = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, cfg_old.vocab_size)
      targets = jax.random.randint(jax.random.PRNGKey(2), (batch_size, seq_len), 0, cfg_old.vocab_size)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

      def loss_fn(m):
        """Calculates mean cross-entropy loss over random tokens."""
        logits = m(tokens, positions, decoder_segment_ids=segment_ids, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
        one_hot = jax.nn.one_hot(targets, cfg_old.vocab_size)
        xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot)
        return jnp.mean(xent)

      # Test direct loss evaluation parity
      loss_direct_old = float(loss_fn(model_old))
      loss_direct_new = float(loss_fn(model_new))
      self.assertAlmostEqual(
          loss_direct_old, loss_direct_new, places=3, msg=f"Direct loss mismatch: {loss_direct_old} vs {loss_direct_new}"
      )

      # Test backward pass gradients parity
      loss_old, grads_old = nnx.value_and_grad(loss_fn)(model_old)
      loss_new, grads_new = nnx.value_and_grad(loss_fn)(model_new)

      rel_loss_diff = abs(float(loss_old) - float(loss_new)) / (float(loss_old) + 1e-6)
      self.assertLess(rel_loss_diff, 0.01, f"Relative loss diff in value_and_grad too high: {rel_loss_diff}")

      flat_g_old = dict(nnx.state(grads_old, nnx.Param).flat_state())
      flat_g_new = dict(nnx.state(grads_new, nnx.Param).flat_state())

      for key in sorted(flat_g_old.keys()):
        g_old = flat_g_old[key][...]
        g_new = flat_g_new[key][...]
        diff = float(jnp.max(jnp.abs(g_old - g_new)))
        norm = float(jnp.max(jnp.abs(g_old)))
        rel_diff = diff / (norm + 1e-6)
        self.assertTrue(
            diff < 0.05 or rel_diff < 0.10,
            f"Grad diff too high for {key}: rel_diff={rel_diff:.4f}, diff={diff:.4f}, norm={norm:.4f}",
        )


if __name__ == "__main__":
  absltest.main()
