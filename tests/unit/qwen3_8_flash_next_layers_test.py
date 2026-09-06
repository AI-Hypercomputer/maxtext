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

"""Unit tests for Qwen3.8-Flash-Next layers."""

import unittest
from flax import nnx
import jax
from jax import numpy as jnp
from maxtext.configs import pyconfig
from maxtext.models import qwen3_8_flash_next


class Qwen3_8FlashNextLayersTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [
            None,
            "src/maxtext/configs/base.yml",
            "model_name=qwen3.8-flash-next",
            "run_name=test",
            "override_model_config=true",
            "base_num_decoder_layers=4",
            "ngram_vocab_size_base=1000",
            "scan_layers=false",
            "enable_dropout=false",
        ]
    )
    self.rngs = nnx.Rngs(0)
    devices = jax.devices()
    self.mesh = jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh((len(devices),), devices),
        ("data",),
    )

  def test_rmsnorm(self):
    norm = qwen3_8_flash_next.Qwen3_8FlashNextRMSNorm(
        num_features=2560 * 4,
        group_size=2560,
        rngs=self.rngs,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2560 * 4))
    y = norm(x)
    self.assertEqual(y.shape, (2, 4, 2560 * 4))

  def test_hyperconnection(self):
    hc = qwen3_8_flash_next.Qwen3_8FlashNextHyperConnection(
        config=self.config,
        mesh=self.mesh,
        use_combine=True,
        rngs=self.rngs,
    )
    hyper_input = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2560 * 4))
    mixed_input, orig_hyper, inject_weights = hc(hyper_input)
    self.assertEqual(mixed_input.shape, (2, 4, 2560))
    self.assertEqual(orig_hyper.shape, (2, 4, 2560 * 4))
    self.assertEqual(inject_weights.shape, (2, 4, 4))

    # Test mixer mode (use_combine=False)
    mixer = qwen3_8_flash_next.Qwen3_8FlashNextHyperConnection(
        config=self.config,
        mesh=self.mesh,
        use_combine=False,
        rngs=self.rngs,
    )
    mixed_only = mixer(hyper_input)
    self.assertEqual(mixed_only.shape, (2, 4, 2560))

  def test_ple_layer(self):
    ple = qwen3_8_flash_next.Qwen3_8FlashNextPLELayer(
        config=self.config,
        layer_idx=1,
        ple_layer_index=0,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2560 * 4))
    token_ids = jnp.array([[40, 2854, 310, 100], [10, 20, 30, 40]], dtype=jnp.int64)
    out = ple(hidden_states, decoder_input_tokens=token_ids)
    self.assertEqual(out.shape, (2, 4, 2560 * 4))

  def test_decoder_layers(self):
    # Test linear attention layer (layer 0)
    layer0 = qwen3_8_flash_next.Qwen3_8FlashNextDecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode="train",
        layer_idx=0,
        rngs=self.rngs,
    )
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2560 * 4))
    out, _ = layer0(hidden_states)
    self.assertEqual(out.shape, (2, 4, 2560 * 4))

    # Test full attention layer (layer 3)
    layer3 = qwen3_8_flash_next.Qwen3_8FlashNextDecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode="train",
        layer_idx=3,
        rngs=self.rngs,
    )
    positions = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    out, _ = layer3(hidden_states, decoder_positions=positions)
    self.assertEqual(out.shape, (2, 4, 2560 * 4))


if __name__ == "__main__":
  unittest.main()
