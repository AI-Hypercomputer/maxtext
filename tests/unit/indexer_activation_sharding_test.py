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

"""Tests for DeepSeek-V4 CSA indexer activation sharding."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

from maxtext.configs import pyconfig
from maxtext.common.common_types import DEFAULT_MASK_VALUE, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import initializers
from maxtext.layers.attention_compressed import DeepseekV4CSACompressor
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

BATCH = 2
SEQ = 64
RATE = 4
HEADS = 4


def make_config(**overrides):
  """Creates a small DeepSeek-V4 config."""
  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "indexer_activation_sharding_test",
      "enable_checkpointing": False,
      "max_target_length": 128,
      "base_emb_dim": 64,
      "head_dim": 64,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "q_lora_rank": 16,
      "indexer_n_heads": HEADS,
      "indexer_head_dim": 64,
      "indexer_topk": 8,
      "sliding_window_size": 8,
      "compress_ratios": [0, 0, 4, 128],
  }
  config_arguments.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **config_arguments)


def make_csa_compressor(config, mesh=None, seed=0):
  rotary = DeepSeekV4RotaryEmbedding(
      head_dim=config.head_dim,
      partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
      rope_theta=config.compressed_rope_max_timescale,
      fprop_dtype=config.dtype,
  )
  return DeepseekV4CSACompressor(
      config=config,
      compress_ratio=RATE,
      rotary_embedding=rotary,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs=nnx.Rngs(seed),
      mesh=mesh,
  )


def make_inputs(config, seed=0):
  rng = np.random.default_rng(seed)
  hidden = jnp.array(rng.normal(size=(BATCH, SEQ, config.emb_dim)), dtype=jnp.float32)
  q_latent = jnp.array(rng.normal(size=(BATCH, SEQ, config.q_lora_rank)), dtype=jnp.float32)
  positions = jnp.broadcast_to(jnp.arange(SEQ)[None, :], (BATCH, SEQ))
  return hidden, q_latent, positions


def packed_segment_mask(n_segments=3):
  ids = np.zeros((BATCH, SEQ), dtype=np.int32)
  bounds = np.linspace(0, SEQ, n_segments + 1).astype(int)
  for i in range(n_segments):
    ids[:, bounds[i] : bounds[i + 1]] = i + 1
  segment_ids = jnp.array(ids)
  same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
  return jnp.where(same_segment, 0.0, DEFAULT_MASK_VALUE)[:, :, ::RATE]


def constraint_specs(jaxpr):
  """Returns shapes and specs for all nested sharding constraints."""
  found = []

  def walk(inner):
    for eqn in inner.eqns:
      if eqn.primitive.name in ("sharding_constraint", "reshard"):
        sharding = eqn.params.get("sharding")
        found.append((eqn.invars[0].aval.shape, getattr(sharding, "spec", sharding)))
      for param in eqn.params.values():
        for sub in jax.tree_util.tree_leaves(param, is_leaf=lambda x: hasattr(x, "jaxpr") or hasattr(x, "eqns")):
          if hasattr(sub, "jaxpr"):
            walk(sub.jaxpr)
          elif hasattr(sub, "eqns"):
            walk(sub)

  walk(jaxpr.jaxpr)
  return found


def indexer_jaxpr(indexer, hidden, q_latent, positions, model_mode=None, rules=None):
  """Traces the indexer under logical-axis rules."""
  graphdef, state = nnx.split(indexer)
  kwargs = {} if model_mode is None else {"model_mode": model_mode}
  rules = indexer.config.logical_axis_rules if rules is None else rules
  with nn_partitioning.axis_rules(rules):
    return jax.make_jaxpr(lambda s, h, q, p: nnx.merge(graphdef, s)(h, q, p, **kwargs))(
        state, hidden, q_latent, positions
    )


class IndexerActivationShardingTest(unittest.TestCase):

  def setUp(self):
    tensor = 4 if jax.device_count() >= 8 else 1
    self.mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count() // tensor, tensor)), axis_names=("data", "tensor"))

  def test_flag_defaults_off_and_emits_no_constraints(self):
    cfg = make_config()
    self.assertFalse(cfg.shard_indexer_acts, "the flag must default to off")
    indexer = make_csa_compressor(cfg, mesh=self.mesh).indexer
    self.assertFalse(indexer.shard_indexer_acts)
    self.assertEqual(constraint_specs(indexer_jaxpr(indexer, *make_inputs(cfg))), [])

  def test_ambient_rules_win_over_config_rules(self):
    cfg = make_config(shard_indexer_acts=True)
    indexer = make_csa_compressor(cfg, mesh=self.mesh).indexer
    eval_rules = [(name, [] if name == "activation_heads" else axes) for name, axes in cfg.logical_axis_rules]
    jaxpr = indexer_jaxpr(indexer, *make_inputs(cfg), rules=eval_rules)
    head_axes = {spec[1] for _, spec in constraint_specs(jaxpr) if len(spec) == 4}
    self.assertEqual(head_axes, {None})

  def test_decode_is_not_constrained(self):
    cfg = make_config(shard_indexer_acts=True)
    indexer = make_csa_compressor(cfg, mesh=self.mesh).indexer
    self.assertTrue(indexer.shard_indexer_acts)
    jaxpr = indexer_jaxpr(indexer, *make_inputs(cfg), model_mode=MODEL_MODE_AUTOREGRESSIVE)
    self.assertEqual(constraint_specs(jaxpr), [])

  def test_flag_on_resolves_expected_specs(self):
    if jax.device_count() < 8:
      self.skipTest("needs XLA_FLAGS=--xla_force_host_platform_device_count=8 for non-degenerate mesh axes")
    cfg = make_config(shard_indexer_acts=True)
    indexer = make_csa_compressor(cfg, mesh=self.mesh).indexer
    self.assertTrue(indexer.shard_indexer_acts)
    jaxpr = indexer_jaxpr(indexer, *make_inputs(cfg))
    n_windows = SEQ // RATE
    head_dim = cfg.indexer_head_dim
    self.assertEqual(
        constraint_specs(jaxpr),
        [
            ((BATCH, HEADS, SEQ, head_dim), PartitionSpec("data", "tensor", None, None)),
            ((BATCH, HEADS, n_windows, head_dim), PartitionSpec("data", "tensor", None, None)),
            ((BATCH, HEADS, SEQ, n_windows), PartitionSpec("data", "tensor", None, None)),
            ((BATCH, SEQ, n_windows), PartitionSpec("data", None, None)),
        ],
    )

  def _assert_selection_unchanged(self, attention_mask):
    cfg_off = make_config()
    cfg_on = make_config(shard_indexer_acts=True)
    comp_off = make_csa_compressor(cfg_off, mesh=self.mesh)
    comp_on = make_csa_compressor(cfg_on, mesh=self.mesh)
    hidden, q_latent, positions = make_inputs(cfg_off)

    with jax.set_mesh(self.mesh):
      sel_off = jax.jit(lambda h, q, p, m: comp_off.indexer(h, q, p, m))(hidden, q_latent, positions, attention_mask)
      sel_on = jax.jit(lambda h, q, p, m: comp_on.indexer(h, q, p, m))(hidden, q_latent, positions, attention_mask)
      kv_off, mask_off = jax.jit(lambda h, q, p, m: comp_off(h, q, p, m))(hidden, q_latent, positions, attention_mask)
      kv_on, mask_on = jax.jit(lambda h, q, p, m: comp_on(h, q, p, m))(hidden, q_latent, positions, attention_mask)

    self.assertEqual(sel_off.dtype, sel_on.dtype)
    np.testing.assert_array_equal(np.array(sel_off), np.array(sel_on))
    np.testing.assert_array_equal(np.array(mask_off), np.array(mask_on))
    # Constraints can change pooling fusion and reassociate float sums by about 1 ulp.
    np.testing.assert_allclose(np.array(kv_off), np.array(kv_on), rtol=1e-5, atol=1e-6)

  def test_output_bitwise_unchanged_unpacked(self):
    self._assert_selection_unchanged(None)

  def test_output_bitwise_unchanged_packed(self):
    self._assert_selection_unchanged(packed_segment_mask())


if __name__ == "__main__":
  unittest.main()
