# Copyright 2026 Google LLC
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

"""Tests for the nested scans inside Qwen3NextScannableBlock."""

import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_CONFIG_OVERRIDES = {
    "run_name": "qwen3_next_scannable_block_test",
    "model_name": "qwen3-next-80b-a3b",
    "enable_checkpointing": False,
    "per_device_batch_size": 1.0,
    "max_target_length": 8,
    "max_prefill_predict_length": 4,
    "attention": "dot_product",
    "base_emb_dim": 64,
    "base_num_decoder_layers": 4,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 32,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_chunk_size": 4,
    "sparse_matmul": True,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "scan_layers": False,
}


def _make_config(**overrides):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()], override_model_config=True, **{**_CONFIG_OVERRIDES, **overrides}
  )


class Qwen3NextScannableBlockTest(unittest.TestCase):
  """The block's nested scans must reproduce a plain sequential unroll."""

  def _build(self, **overrides):
    cfg = _make_config(**overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    block = qwen3.Qwen3NextScannableBlock(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(0),
    )
    return cfg, mesh, block

  def test_block_splits_cycle_into_local_stack_plus_one_global(self):
    """A block covers one attention period: cycle-1 linear layers and one full-attention layer."""
    cfg, _, block = self._build()
    self.assertEqual(block.num_local, cfg.inhomogeneous_layer_cycle_interval - 1)
    self.assertEqual(block.num_global, 1)
    self.assertIsNotNone(block.local_layers)
    self.assertIsNotNone(block.global_layer)

  def test_local_params_are_stacked(self):
    """The linear-attention layers are stacked along param_scan_axis, not stored per layer."""
    cfg, _, block = self._build()
    _, params, _ = nnx.split(block.local_layers, nnx.Param, ...)
    leaves = [v.value for _, v in params.flat_state()]
    self.assertTrue(leaves)
    for leaf in leaves:
      self.assertEqual(leaf.shape[cfg.param_scan_axis], block.num_local)

  def test_nested_scan_matches_sequential_unroll(self):
    """Scanning the local layers then the global layer equals applying them one by one."""
    cfg, _, block = self._build()
    inputs = jax.random.normal(jax.random.PRNGKey(1), (1, cfg.max_target_length, cfg.emb_dim), dtype=jnp.float32)
    positions = jnp.arange(cfg.max_target_length)[None, :]
    segment_ids = jnp.ones((1, cfg.max_target_length), dtype=jnp.int32)

    scanned = block(inputs, segment_ids, positions, True, MODEL_MODE_TRAIN)

    # Reference: pull each stacked local layer out by index and run it, then the global layer.
    graphdef, params, rest = nnx.split(block.local_layers, nnx.Param, ...)
    if cfg.param_scan_axis != 0:
      params = jax.tree.map(lambda x: jnp.moveaxis(x, cfg.param_scan_axis, 0), params)
    y = inputs
    for i in range(block.num_local):
      layer = nnx.merge(
          graphdef,
          jax.tree.map(lambda x, i=i: x[i], params),
          jax.tree.map(lambda x, i=i: x[i], rest),
      )
      y = layer(y, segment_ids, positions, True, MODEL_MODE_TRAIN)[0]
    expected = block.global_layer(y, segment_ids, positions, True, MODEL_MODE_TRAIN)[0]

    np.testing.assert_allclose(np.asarray(scanned), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_rejects_block_whose_global_layer_is_not_last(self):
    """The local scan runs before the global layer, so any other ordering must be refused."""
    with self.assertRaisesRegex(ValueError, "full-attention layer last"):
      self._build(full_attention_layer_offset=0)


if __name__ == "__main__":
  unittest.main()
