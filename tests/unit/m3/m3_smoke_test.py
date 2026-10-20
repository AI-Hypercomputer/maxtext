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

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Model
from maxtext.utils import model_creation_utils
import numpy as np


class M3SmokeTest(absltest.TestCase):
  """Smoke test verifying m3 registry routing and forward execution under Blaze."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()[:1]
    self.mesh = Mesh(np.array(devices), ("data",))

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


if __name__ == "__main__":
  absltest.main()
