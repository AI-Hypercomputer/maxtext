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

"""Unit tests for canonical Gemma 4 attention scaling and weight initialization."""

import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.common import common_types
from maxtext.models import gemma4, gemma4_small
from maxtext.utils.globals import MAXTEXT_REPO_ROOT


class Gemma4CanonicalAttentionTest(unittest.TestCase):
  """Tests that Gemma 4 models follow canonical attention scaling and unscaled weight init."""

  def setUp(self):
    super().setUp()
    self.base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

  def test_gemma4_26b_attention_config(self):
    config = pyconfig.initialize(
        ["", self.base_config_path],
        model_name="gemma4-26b",
        enable_dropout=False,
    )
    self.assertTrue(config.use_qk_norm, "gemma4-26b should enable use_qk_norm in config")

    mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
    rngs = nnx.Rngs(0)
    layer = gemma4.Gemma4DecoderLayer(
        config=config,
        mesh=mesh,
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs=rngs,
        attention_type=gemma4.AttentionType.LOCAL_SLIDING,
        layer_idx=0,
    )
    # Canonical Gemma 4 attention uses query_pre_attn_scalar = 1.0 (unscaled logits)
    self.assertEqual(layer.self_attention.query_pre_attn_scalar, 1.0)
    self.assertTrue(layer.self_attention.use_qk_norm)

    # Initial query weights should NOT be divided by sqrt(head_dim) = 16.0
    # Expected standard deviation for fan_in=2816 is 1/sqrt(2816) ~= 0.0188
    q_kernel = layer.self_attention.query.kernel[...]
    std_q = float(jnp.std(q_kernel))
    self.assertGreater(std_q, 0.01, f"Query kernel std ({std_q}) should not be divided by depth_scaling")

  def test_gemma4_small_attention_config(self):
    for model_name in ["gemma4-e2b", "gemma4-e4b"]:
      config = pyconfig.initialize(
          ["", self.base_config_path],
          model_name=model_name,
          enable_dropout=False,
      )
      self.assertTrue(config.use_qk_norm, f"{model_name} should enable use_qk_norm in config")

      mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
      rngs = nnx.Rngs(0)
      layer = gemma4_small.Gemma4SmallDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=common_types.MODEL_MODE_PREFILL,
          layer_idx=0,
          rngs=rngs,
      )
      self.assertEqual(layer.self_attention.query_pre_attn_scalar, 1.0)
      self.assertTrue(layer.self_attention.use_qk_norm)

      q_kernel = layer.self_attention.query.kernel[...]
      std_q = float(jnp.std(q_kernel))
      self.assertGreater(std_q, 0.01, f"Query kernel std ({std_q}) should not be divided by depth_scaling")


if __name__ == "__main__":
  unittest.main()
