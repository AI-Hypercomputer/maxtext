# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke test for m3 model registry routing and forward pass under Blaze."""

import re

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Model
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from maxtext.utils import model_creation_utils
import numpy as np


class M3SmokeTest(absltest.TestCase):
  """Smoke test verifying m3 registry routing and forward execution under Blaze."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()[:1]
    self.mesh = Mesh(np.array(devices), ("data",))

  def test_unregistered_model_raises_value_error(self):
    """Verifies model_name not in m3 registry raises ValueError."""
    config_args = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=default",
        "use_m3_model=true",
        "skip_jax_distributed_system=true",
    ]
    cfg = pyconfig.initialize_pydantic(config_args)
    expected_models = ", ".join(sorted(m3_models.MODEL_REGISTRY))
    with self.assertRaisesRegex(
        ValueError,
        r"Model 'default' does not support the m3 backend"
        r" \(use_m3_model=True\)\. Registered models:"
        rf" {re.escape(expected_models)}\.",
    ):
      model_creation_utils.create_model(cfg, self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

  def test_qwen3_routing_and_forward_pass(self):
    """Verifies use_m3_model=True constructs Qwen3Model and executes forward pass."""
    config_args = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=qwen3-0.6b",
        "use_m3_model=true",
        "scan_layers=false",
        "weight_dtype=bfloat16",
        "per_device_batch_size=1",
        "max_target_length=32",
        "max_prefill_predict_length=16",
        "override_model_config=true",
        "num_decoder_layers=2",
        "emb_dim=64",
        "mlp_dim=128",
        "num_query_heads=4",
        "num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
    ]
    cfg = pyconfig.initialize_pydantic(config_args)
    with jax.set_mesh(self.mesh):
      rngs = nnx.Rngs(0)
      model = model_creation_utils.create_model(cfg, self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
      self.assertIsInstance(model, Qwen3Model)

      batch_size, seq_len = 2, 8
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN)
      self.assertEqual(logits.shape, (batch_size, seq_len, cfg.vocab_size))

  def test_qwen3_vl_routing_and_forward_pass(self):
    """Verifies use_m3_model=True constructs Qwen3VLModel and executes forward pass."""
    config_args = [
        "",
        "src/maxtext/configs/base.yml",
        "model_name=qwen3-vl-2b",
        "use_m3_model=true",
        "use_multimodal=true",
        "scan_layers=false",
        "per_device_batch_size=1",
        "dtype=float32",
        "weight_dtype=float32",
        "skip_jax_distributed_system=true",
        "override_model_config=true",
        "base_num_decoder_layers=2",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=151936",
        "num_hidden_layers_for_vit=2",
        "deepstack_visual_indexes_for_vit=[1]",
        "hidden_size_for_vit=16",
        "spatial_merge_size_for_vit=2",
        "out_hidden_size_for_vit=64",
        "intermediate_size_for_vit=64",
        "num_attention_heads_for_vit=4",
        "num_position_embeddings_for_vit=64",
    ]
    cfg = pyconfig.initialize_pydantic(config_args)
    with jax.set_mesh(self.mesh):
      rngs = nnx.Rngs(0)
      model = model_creation_utils.create_model(cfg, self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
      self.assertIsInstance(model, Qwen3VLModel)

      batch_size, seq_len = 1, 4
      tokens = jnp.array([[1, 151655, 2, 3]], dtype=jnp.int32)
      images = jnp.zeros((batch_size, 3, 2, 32, 32), dtype=jnp.float32)
      grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)
      logits = model(
          decoder_input_tokens=tokens,
          encoder_images=images,
          encoder_video_grid_thw=grid_thw,
      )
      self.assertEqual(logits.shape, (batch_size, seq_len, cfg.vocab_size))
      self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))


if __name__ == "__main__":
  absltest.main()
