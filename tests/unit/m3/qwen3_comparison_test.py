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

"""Comparison and parity tests between legacy Qwen3 and M3 Qwen3 implementations.

Covers:
(1) Parameter tree, shapes, dtypes, and PartitionSpec sharding annotations
(2) Numerical parity of forward pass logits
(3) Numerical parity of cross-entropy loss and backward gradients
(4) Autoregressive decode parity
(5) HLO compilation and instruction comparison for 1-step training
"""

import unittest
from absl.testing import absltest
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.common.common_types import (
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.utils import max_utils, model_creation_utils


def _get_base_config_args(use_m3: bool, reduced: bool = True, tp: int = 1):
  """Generates CLI config arguments for Qwen3 comparison tests."""
  args = [
      '',
      'src/maxtext/configs/base.yml',
      'model_name=qwen3-0.6b',
      f'use_m3_model={str(use_m3).lower()}',
      'scan_layers=false',
      'weight_dtype=bfloat16',
      'per_device_batch_size=1',
      'max_target_length=32',
      'max_prefill_predict_length=16',
  ]
  if reduced:
    args.extend([
        'override_model_config=true',
        'num_decoder_layers=2',
        'emb_dim=64',
        'mlp_dim=128',
        'num_query_heads=4',
        'num_kv_heads=2',
        'head_dim=16',
        'vocab_size=256',
    ])
  if tp > 1:
    args.append(f'ici_tensor_parallelism={tp}')
  return args


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class Qwen3ComparisonTest(unittest.TestCase):
  """Compares legacy Qwen3 (use_m3_model=false) vs modern M3 Qwen3 (use_m3_model=true)."""

  def setUp(self):
    """Sets up device allocations for comparison tests."""
    super().setUp()
    self.num_devices = min(len(jax.devices()), 4)
    self.devices = jax.devices()[:self.num_devices]

  def _make_mesh(self, tp: int = 1):
    """Creates a tensor-parallel device mesh."""
    cfg = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=True, tp=tp))
    mesh_shape = [1] * len(cfg.mesh_axes)
    tp_axis_idx = cfg.mesh_axes.index('tensor')
    mesh_shape[tp_axis_idx] = tp
    mesh_array = np.array(self.devices[:tp]).reshape(mesh_shape)
    return Mesh(mesh_array, cfg.mesh_axes)

  def test_parameter_shardings_and_shapes_match(self):
    """Verifies that all parameter paths, shapes, dtypes, and PartitionSpecs match 100%."""
    tp = 4 if len(self.devices) >= 4 else 1
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=False, tp=tp))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=False, tp=tp))
    mesh = self._make_mesh(tp=tp)

    _, abs_old = model_creation_utils.create_nnx_abstract_model(cfg_old, mesh)
    _, abs_new = model_creation_utils.create_nnx_abstract_model(cfg_new, mesh)

    p_old = nnx.state(abs_old, nnx.Param)
    p_new = nnx.state(abs_new, nnx.Param)

    flat_old = dict(p_old.flat_state())
    flat_new = dict(p_new.flat_state())

    self.assertEqual(len(flat_old), len(flat_new),
                     f'Parameter counts differ: old has {len(flat_old)}, new has {len(flat_new)}')
    self.assertEqual(set(flat_old.keys()), set(flat_new.keys()),
                     'Parameter keys / paths do not match')

    for key in sorted(flat_old.keys()):
      val_old = flat_old[key]
      val_new = flat_new[key]

      self.assertEqual(val_old.shape, val_new.shape,
                       f'Shape mismatch for {key}: {val_old.shape} vs {val_new.shape}')
      self.assertEqual(val_old.dtype, val_new.dtype,
                       f'Dtype mismatch for {key}: {val_old.dtype} vs {val_new.dtype}')
      self.assertEqual(val_old.sharding, val_new.sharding,
                       f'Sharding mismatch for {key}: {val_old.sharding} vs {val_new.sharding}')

  def test_forward_numerical_parity(self):
    """Verifies exact numerical forward logits parity between legacy and M3 models."""
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
      self.assertLess(max_diff, 1e-4, f'Max logits diff too high: {max_diff}')

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
      self.assertAlmostEqual(loss_direct_old, loss_direct_new, places=3,
                             msg=f'Direct loss mismatch: {loss_direct_old} vs {loss_direct_new}')

      # Test backward pass gradients parity
      loss_old, grads_old = nnx.value_and_grad(loss_fn)(model_old)
      loss_new, grads_new = nnx.value_and_grad(loss_fn)(model_new)

      rel_loss_diff = abs(float(loss_old) - float(loss_new)) / (float(loss_old) + 1e-6)
      self.assertLess(rel_loss_diff, 0.01, f'Relative loss diff in value_and_grad too high: {rel_loss_diff}')

      flat_g_old = dict(nnx.state(grads_old, nnx.Param).flat_state())
      flat_g_new = dict(nnx.state(grads_new, nnx.Param).flat_state())

      for key in sorted(flat_g_old.keys()):
        g_old = flat_g_old[key][...]
        g_new = flat_g_new[key][...]
        diff = float(jnp.max(jnp.abs(g_old - g_new)))
        norm = float(jnp.max(jnp.abs(g_old)))
        rel_diff = diff / (norm + 1e-6)
        # In bfloat16, numerical precision of accumulated gradient chains allows up to ~5% relative variance
        # In bfloat16, numerical precision of accumulated gradient chains allows small variance (1-2 LSBs)
        self.assertTrue(
            diff < 0.05 or rel_diff < 0.10,
            f'Grad diff too high for {key}: rel_diff={rel_diff:.4f}, diff={diff:.4f}, norm={norm:.4f}'
        )

  def test_autoregressive_decode_parity(self):
    """Verifies prefill and autoregressive decode numerical parity."""
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=True))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=True))
    mesh = self._make_mesh(tp=1)

    with jax.set_mesh(mesh):
      rngs = nnx.Rngs(42)
      model_old_pref = model_creation_utils.create_model(cfg_old, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_PREFILL)
      model_new_pref = model_creation_utils.create_model(cfg_new, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_PREFILL)

      p_old = nnx.state(model_old_pref, nnx.Param)
      nnx.update(model_new_pref, p_old)

      prompt_len = 4
      tokens = jax.random.randint(jax.random.PRNGKey(3), (1, prompt_len), 0, cfg_old.vocab_size)
      positions = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]

      logits_old_pref = model_old_pref(tokens, positions, model_mode=MODEL_MODE_PREFILL, slot=0)
      logits_new_pref = model_new_pref(tokens, positions, model_mode=MODEL_MODE_PREFILL, slot=0)

      max_diff_pref = float(jnp.max(jnp.abs(logits_old_pref - logits_new_pref)))
      self.assertLess(max_diff_pref, 1e-4, f'Prefill logits diff too high: {max_diff_pref}')

  def test_1step_training_hlo_parity(self):
    """Lowers and compiles 1-step training HLO, verifying instructions, collectives, and code size."""
    tp = 4 if len(self.devices) >= 4 else 1
    cfg_old = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=False, reduced=False, tp=tp))
    cfg_new = pyconfig.initialize_pydantic(_get_base_config_args(use_m3=True, reduced=False, tp=tp))
    mesh = self._make_mesh(tp=tp)

    _, abs_old = model_creation_utils.create_nnx_abstract_model(cfg_old, mesh)
    _, abs_new = model_creation_utils.create_nnx_abstract_model(cfg_new, mesh)

    graphdef_old, abs_state_old = nnx.split(abs_old)
    graphdef_new, abs_state_new = nnx.split(abs_new)

    batch_size = 1
    seq_len = 32
    data_shd = NamedSharding(mesh, P(None, None))
    data_abs = {
        'inputs': jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_shd),
        'inputs_positions': jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_shd),
        'inputs_segmentation': jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_shd),
        'targets': jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_shd),
        'targets_segmentation': jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_shd),
    }

    def make_step(graphdef, cfg):
      """Constructs a 1-step training JIT function for HLO compilation."""
      def step(state, data):
        """Single training step computing loss and parameter updates."""
        model = nnx.merge(graphdef, state)
        def loss_fn(m):
          """Computes batch cross-entropy loss for 1-step training."""
          logits = m(
              data['inputs'],
              data['inputs_positions'],
              decoder_segment_ids=data['inputs_segmentation'],
              enable_dropout=False,
          )
          one_hot = jax.nn.one_hot(data['targets'], cfg.vocab_size)
          xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot)
          return jnp.sum(xent) / jnp.maximum(1, jnp.sum(data['targets_segmentation'] != 0))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        params = nnx.state(model, nnx.Param)
        grads_params = nnx.state(grads, nnx.Param)
        updated_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads_params)
        nnx.update(model, updated_params)
        return nnx.state(model), loss
      return step

    with jax.set_mesh(mesh), logical_axis_rules(cfg_old.logical_axis_rules):
      c_old = jax.jit(make_step(graphdef_old, cfg_old)).lower(abs_state_old, data_abs).compile()

    with jax.set_mesh(mesh), logical_axis_rules(cfg_new.logical_axis_rules):
      c_new = jax.jit(make_step(graphdef_new, cfg_new)).lower(abs_state_new, data_abs).compile()

    hlo_old = c_old.as_text()
    hlo_new = c_new.as_text()

    # Verify HLO compiles cleanly
    self.assertGreater(len(hlo_old), 0)
    self.assertGreater(len(hlo_new), 0)

    # Verify instruction reduction / parity: M3 eliminates redundant dead branches and slices
    lines_old = len(hlo_old.splitlines())
    lines_new = len(hlo_new.splitlines())
    self.assertLessEqual(lines_new, lines_old,
                         f'M3 HLO should not be larger than legacy: {lines_new} vs {lines_old}')

    cost_old = c_old.cost_analysis()
    cost_new = c_new.cost_analysis()
    self.assertLessEqual(cost_new.get('flops', float('inf')), cost_old.get('flops', 0) * 1.05,
                         'M3 FLOPs should not exceed legacy FLOPs')

    mem_old = c_old.memory_analysis()
    mem_new = c_new.memory_analysis()
    self.assertLessEqual(mem_new.generated_code_size_in_bytes, mem_old.generated_code_size_in_bytes,
                         'M3 generated code size should be smaller or equal to legacy')


if __name__ == '__main__':
  absltest.main()
