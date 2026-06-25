"""Tests for compressed attention."""

import unittest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from maxtext.configs.pyconfig import initialize
from maxtext.layers.attention_compressed import CompressedAttention
from tests.utils.test_helpers import get_test_config_path


class CompressedAttentionTest(unittest.TestCase):
  """Tests for Compressed Attention."""

  def setUp(self):
    self.config = initialize(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek4-284b",
            "attention=dot_product",
            "qk_rope_head_dim=16",
            "v_head_dim=16",
            "qk_nope_head_dim=16",
        ]
    )
    self.mesh = Mesh(jax.devices(), ("data",))

  def test_compressed_attention_jaxpr_tag_counts(self):
    layer = CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=4,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

    q = jnp.ones((1, 32, 4096))
    kv = jnp.ones((1, 32, 4096))
    pos = jnp.arange(32)[None, :]
    seg = jnp.zeros((1, 32), dtype=jnp.int32)

    graphdef, state = nnx.split(layer)

    def forward(state, q, kv, seg, pos):
      layer = nnx.merge(graphdef, state)
      return layer(q, kv, seg, pos, deterministic=True)

    jaxpr = jax.make_jaxpr(forward)(state, q, kv, seg, pos)
    jaxpr_str = str(jaxpr)

    self.assertEqual(jaxpr_str.count("name=query_proj"), 1)
    self.assertEqual(jaxpr_str.count("name=kv_proj"), 1)
    self.assertEqual(jaxpr_str.count("name=attention_out"), 1)
    self.assertEqual(jaxpr_str.count("name=out_proj"), 1)

  def test_compressed_attention_no_double_tagging(self):
    layer = CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=4,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

    q = jnp.ones((1, 32, 4096))
    kv = jnp.ones((1, 32, 4096))
    pos = jnp.arange(32)[None, :]
    seg = jnp.zeros((1, 32), dtype=jnp.int32)

    graphdef, state = nnx.split(layer)

    def forward(state, q, kv, seg, pos):
      layer = nnx.merge(graphdef, state)
      return layer(q, kv, seg, pos, deterministic=True)

    jaxpr = jax.make_jaxpr(forward)(state, q, kv, seg, pos)
    jaxpr_str = str(jaxpr)

    self.assertEqual(jaxpr_str.count("name=key_proj"), 0)
    self.assertEqual(jaxpr_str.count("name=value_proj"), 0)


if __name__ == "__main__":
  unittest.main()
