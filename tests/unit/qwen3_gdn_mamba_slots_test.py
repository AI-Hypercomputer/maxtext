# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests mamba slot resolution from the block table under vLLM prefix caching."""

# pylint: disable=protected-access

import sys
import types
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.models import qwen3
from maxtext.models.qwen3 import Qwen3NextGatedDeltaNet
from tests.utils.test_helpers import get_test_config_path

_STUB_MODULES = (
    "tpu_inference",
    "tpu_inference.layers",
    "tpu_inference.layers.common",
    "tpu_inference.layers.common.gdn_attention",
    "tpu_inference.layers.common.sharding",
    "tpu_inference.layers.common.utils",
    "tpu_inference.utils",
)


def _gdn_config(**overrides):
  """A minimal Qwen3-Next config sized for a single-device CPU mesh."""
  argv = [
      None,
      get_test_config_path(),
      "run_name=qwen3_gdn_align_test",
      "dtype=float32",
      "weight_dtype=float32",
      "decoder_block=qwen3_next",
      "attention=dot_product",
      "base_emb_dim=64",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "head_dim=32",
      "gdn_num_value_heads=1",
      "gdn_num_key_heads=1",
      "gdn_key_head_dim=32",
      "gdn_value_head_dim=32",
      "gdn_conv_kernel_dim=4",
      "gdn_chunk_size=64",
  ]
  argv += [f"{k}={v}" for k, v in overrides.items()]
  return pyconfig.initialize(argv)


def _install_tpu_inference_stubs(captured):
  """Puts the handful of tpu_inference symbols the paged branch imports on sys.modules."""

  def run_jax_gdn_attention(mixed_qkv, b, a, conv_state, recurrent_state, *args, **kwargs):
    del b, a, args
    captured["state_indices"] = kwargs.get("state_indices")
    captured["read_state_indices"] = kwargs.get("read_state_indices")
    num_tokens = mixed_qkv.shape[0]
    out = jnp.zeros((num_tokens, recurrent_state.shape[1] * recurrent_state.shape[-1]), dtype=mixed_qkv.dtype)
    return (conv_state, recurrent_state), out

  def _positional_aware(*args, **kwargs):
    # state_indices and read_state_indices are the 10th positional arg and a
    # keyword respectively; normalise so the stub sees both by name.
    if len(args) > 9:
      kwargs.setdefault("state_indices", args[9])
    return run_jax_gdn_attention(*args[:5], **kwargs)

  for name in _STUB_MODULES:
    sys.modules[name] = types.ModuleType(name)
  sys.modules["tpu_inference.layers.common.gdn_attention"].run_jax_gdn_attention = _positional_aware
  sys.modules["tpu_inference.layers.common.sharding"].ShardingAxisName = types.SimpleNamespace(
      ATTN_DATA=None, ATTN_HEAD=None, MODEL=None
  )
  sys.modules["tpu_inference.layers.common.utils"].reorder_concatenated_tensor_for_sharding = (
      lambda tensor, sizes, tp_size, axis: tensor
  )
  sys.modules["tpu_inference.layers.common.utils"].truncate_sharded_tensor = lambda tensor, length, dp_size: tensor[
      :length
  ]
  sys.modules["tpu_inference.utils"].get_mesh_shape_product = lambda mesh, axis: 1


def _metadata(seq_lens, block_tables):
  return types.SimpleNamespace(
      seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
      block_tables=jnp.asarray(block_tables, dtype=jnp.int32),
  )


class Qwen3GdnMambaSlotsTest(unittest.TestCase):
  """`_mamba_block_table_slots` maps each request to its read/write mamba block."""

  def test_read_and_write_slots_differ_across_a_block_boundary(self):
    # One DP rank, two requests, 4 tokens per mamba block.
    #   req 0: 5 computed tokens, 4 new -> reads block (5-1)//4 = 1,
    #                                      writes block (9-1)//4 = 2
    #   req 1: 0 computed tokens, 3 new -> reads block 0, writes block 0
    block_tables = [[10, 11, 12, 13], [20, 21, 22, 23]]
    write, read = Qwen3NextGatedDeltaNet._mamba_block_table_slots(
        _metadata(seq_lens=[9, 3], block_tables=block_tables),
        seq_lens=jnp.asarray([9, 3], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 4, 7], dtype=jnp.int32),
        mamba_block_size=4,
        padded_num_reqs_per_dp=2,
        dp_size=1,
        local_rows=64,
    )
    np.testing.assert_array_equal(np.asarray(read), [11, 20])
    np.testing.assert_array_equal(np.asarray(write), [12, 20])

  def test_slots_are_clamped_into_the_rank_local_range(self):
    # A block id past the rank's shard must not be handed to the kernel, which
    # DMAs these ids with bounds checks disabled.
    write, read = Qwen3NextGatedDeltaNet._mamba_block_table_slots(
        _metadata(seq_lens=[4], block_tables=[[99, 99]]),
        seq_lens=jnp.asarray([4], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 4], dtype=jnp.int32),
        mamba_block_size=4,
        padded_num_reqs_per_dp=1,
        dp_size=1,
        local_rows=8,
    )
    np.testing.assert_array_equal(np.asarray(read), [7])
    np.testing.assert_array_equal(np.asarray(write), [7])

  def test_each_dp_rank_reads_its_own_slice_of_the_block_table(self):
    # Two ranks, one request each; rank 1's row must not leak into rank 0.
    block_tables = [[10, 11], [20, 21]]
    write, read = Qwen3NextGatedDeltaNet._mamba_block_table_slots(
        _metadata(seq_lens=[1, 5], block_tables=block_tables),
        seq_lens=jnp.asarray([1, 5], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 1, 0, 5], dtype=jnp.int32),
        mamba_block_size=4,
        padded_num_reqs_per_dp=1,
        dp_size=2,
        local_rows=64,
    )
    # req 0 (rank 0): 1 token, no prefix -> block 0 of its own row.
    # req 1 (rank 1): 5 tokens, no prefix -> writes block (5-1)//4 = 1.
    np.testing.assert_array_equal(np.asarray(read), [10, 20])
    np.testing.assert_array_equal(np.asarray(write), [10, 21])


class Qwen3GdnAlignCallTest(unittest.TestCase):
  """`__call__` routes align-mode requests through the block table.

  The kernel and its sharding helpers live in tpu_inference, which is not a
  MaxText dependency, so they are stubbed. What is under test is MaxText's own
  plumbing: that `mamba_state_indices=None` plus a block table still selects the
  paged path, and that the resulting read slot trails the write slot.
  """

  def setUp(self):
    super().setUp()
    self.captured = {}
    self._saved = {name: sys.modules.get(name) for name in _STUB_MODULES}
    _install_tpu_inference_stubs(self.captured)

  def tearDown(self):
    for name, module in self._saved.items():
      if module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = module
    super().tearDown()

  def test_align_mode_passes_a_trailing_read_slot(self):
    cfg = _gdn_config(gdn_mamba_block_size=4)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
    layer = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh, rngs=nnx.Rngs(0), inputs_shape=(1, 8, cfg.emb_dim))

    # One request: 5 tokens already computed, 4 new -> reads block (5-1)//4 = 1,
    # writes block (9-1)//4 = 2, i.e. block ids 11 and 12.
    metadata = types.SimpleNamespace(
        seq_lens=jnp.asarray([9], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 4], dtype=jnp.int32),
        block_tables=jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        padded_num_reqs=1,
        mamba_state_indices=None,
    )
    kv_cache = (
        jnp.zeros((16, cfg.gdn_conv_kernel_dim - 1, 3 * 32), dtype=jnp.float32),
        jnp.zeros((16, cfg.gdn_num_value_heads, 32, 32), dtype=jnp.float32),
    )

    layer(
        jnp.ones((1, 8, cfg.emb_dim), dtype=cfg.dtype),
        kv_cache=kv_cache,
        attention_metadata=metadata,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )

    self.assertIn("state_indices", self.captured, "paged path was not taken")
    np.testing.assert_array_equal(np.asarray(self.captured["state_indices"]), [12])
    np.testing.assert_array_equal(np.asarray(self.captured["read_state_indices"]), [11])

  def test_resident_slot_mode_reads_and_writes_the_same_slot(self):
    # Without prefix caching vLLM still passes mamba_state_indices, and the
    # block table must not be consulted: one resident slot per request, read
    # and written in place.
    cfg = _gdn_config(gdn_mamba_block_size=0)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
    layer = qwen3.Qwen3NextGatedDeltaNet(config=cfg, mesh=mesh, rngs=nnx.Rngs(0), inputs_shape=(1, 8, cfg.emb_dim))

    metadata = types.SimpleNamespace(
        seq_lens=jnp.asarray([9], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 4], dtype=jnp.int32),
        block_tables=jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        padded_num_reqs=1,
        mamba_state_indices=jnp.asarray([3], dtype=jnp.int32),
    )
    kv_cache = (
        jnp.zeros((16, cfg.gdn_conv_kernel_dim - 1, 3 * 32), dtype=jnp.float32),
        jnp.zeros((16, cfg.gdn_num_value_heads, 32, 32), dtype=jnp.float32),
    )

    layer(
        jnp.ones((1, 8, cfg.emb_dim), dtype=cfg.dtype),
        kv_cache=kv_cache,
        attention_metadata=metadata,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )

    np.testing.assert_array_equal(np.asarray(self.captured["state_indices"]), [3])
    np.testing.assert_array_equal(np.asarray(self.captured["read_state_indices"]), [3])


if __name__ == "__main__":
  unittest.main()
