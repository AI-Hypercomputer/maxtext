# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape and smoke tests for the MaxText GLM-5.3-Flash layer components."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import HyperConnectionType
from maxtext.configs import pyconfig
from maxtext.layers import mhc
from maxtext.models.glm5_next import Glm5NextAttention, Glm5NextDecoderLayer, Glm5NextDenseMLP
import numpy as np


class Glm53FlashComponentsTest(unittest.TestCase):
  """Tests individual MaxText GLM-5.3-Flash components."""

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize_pydantic(
        [
            "src/maxtext/configs/base.yml",
            "model_name=glm5.3-flash",
            "base_num_decoder_layers=1",
            "base_emb_dim=16",
            "base_mlp_dim=32",
            "base_num_query_heads=2",
            "base_num_kv_heads=2",
            "head_dim=4",
            "linear_num_heads=2",
            "linear_head_dim=4",
            "vocab_size=32",
            "mhc_expansion_rate=2",
            "scan_layers=false",
            "override_model_config=true",
            "dtype=float32",
            "weight_dtype=float32",
            "per_device_batch_size=1",
            "max_target_length=64",
        ]
    )
    self.mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
    self.rngs = nnx.Rngs(0)
    np.random.seed(42)

  def test_mhc_layer(self):
    """Tests the mHC output shape and numerical sanity."""
    mhc_jax = mhc.ManifoldConstrainedHyperConnections(
        config=self.config,
        dim=self.config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    b, s, k, d = 2, 4, self.config.mhc_expansion_rate, self.config.emb_dim
    x_np = np.random.randn(b, s, k, d).astype(np.float32)

    def dummy_norm(inp):
      return inp

    def dummy_branch(inputs_q, inputs_kv=None, **kwargs):
      return inputs_q * 0.5, None

    out_jax, _ = mhc_jax(
        norm_fn=dummy_norm,
        branch_fn=dummy_branch,
        x=jnp.array(x_np),
        mhc_type=HyperConnectionType.ATTENTION,
    )
    self.assertEqual(out_jax.shape, (b, s, k, d))
    self.assertFalse(np.isnan(np.asarray(out_jax)).any())

  def test_dense_mlp_layer(self):
    """Tests MaxText Glm5NextDenseMLP with SwiGLU clamping."""
    mlp_jax = Glm5NextDenseMLP(
        config=self.config,
        mesh=self.mesh,
        in_features=self.config.emb_dim,
        intermediate_dim=self.config.mlp_dim,
        model_mode="train",
        rngs=self.rngs,
    )
    b, s, d = 2, 4, self.config.emb_dim
    x_np = np.random.randn(b, s, d).astype(np.float32)
    out_jax = mlp_jax(jnp.array(x_np))
    self.assertEqual(out_jax.shape, (b, s, d))
    self.assertFalse(np.isnan(np.asarray(out_jax)).any())

  def test_kda_attention_layer(self):
    """Tests the KDA attention output shape and numerical sanity."""
    attn_jax = Glm5NextAttention(
        config=self.config,
        mesh=self.mesh,
        model_mode="train",
        rngs=self.rngs,
    )
    b, s, d = 2, 4, self.config.emb_dim
    x_np = np.random.randn(b, s, d).astype(np.float32)
    out_jax, _ = attn_jax(inputs_q=jnp.array(x_np))
    self.assertEqual(out_jax.shape, (b, s, d))
    self.assertFalse(np.isnan(np.asarray(out_jax)).any())

  def test_decoder_layer(self):
    """Tests full Glm5NextDecoderLayer forward pass."""
    layer_jax = Glm5NextDecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode="train",
        layer_idx=0,
        rngs=self.rngs,
    )
    b, s, k, d = 2, 4, self.config.mhc_expansion_rate, self.config.emb_dim
    x_np = np.random.randn(b, s, k, d).astype(np.float32)
    out_jax, _ = layer_jax(jnp.array(x_np))
    self.assertEqual(out_jax.shape, (b, s, k, d))
    self.assertFalse(np.isnan(np.asarray(out_jax)).any())


if __name__ == "__main__":
  unittest.main()
