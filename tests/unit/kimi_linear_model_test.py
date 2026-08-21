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

"""Unit tests for Kimi K3 Linear Model Backbone in MaxText."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.models.kimi_linear import KimiDecoderLayer, KimiLinearModel


def test_kimi_decoder_layer_kda_and_mla():
  """Test KimiDecoderLayer for both KDA (layer 0) and MLA (layer 3) in kimi-k3-tiny."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
  ])

  rngs = nnx.Rngs(0)
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

  # Layer 0 (KDA layer, 1-indexed layer 1)
  layer0 = KimiDecoderLayer(cfg, mesh, layer_idx=0, rngs=rngs)
  assert layer0.is_kda is True

  # Layer 3 (MLA layer, 1-indexed layer 4)
  layer3 = KimiDecoderLayer(cfg, mesh, layer_idx=3, rngs=rngs)
  assert layer3.is_kda is False

  x = jnp.ones((2, 4, cfg.emb_dim))
  positions = jnp.arange(4)[None, :].repeat(2, axis=0)

  # Forward pass on Layer 0
  out0, kda_state0 = layer0(x)
  assert out0.shape == (2, 4, cfg.emb_dim)
  assert kda_state0 is not None
  assert not jnp.isnan(out0).any()

  # Forward pass on Layer 3
  out3, kda_state3 = layer3(x, inputs_positions=positions)
  assert out3.shape == (2, 4, cfg.emb_dim)
  assert kda_state3 is None
  assert not jnp.isnan(out3).any()


def test_kimi_linear_model_end_to_end():
  """Test KimiLinearModel end-to-end forward pass on kimi-k3-tiny."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
  ])

  rngs = nnx.Rngs(0)
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

  model = KimiLinearModel(cfg, mesh, rngs=rngs)

  # Input IDs: (batch=2, seq_len=4)
  input_ids = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
  inputs_positions = jnp.arange(4)[None, :].repeat(2, axis=0)

  logits, kda_states = model(input_ids, inputs_positions=inputs_positions)

  # Verify shapes and non-NaN
  assert logits.shape == (2, 4, cfg.vocab_size)
  assert len(kda_states) == cfg.num_decoder_layers
  assert not jnp.isnan(logits).any()

  # Verify KDA states: layers 0, 1, 2 should be not None, layer 3 should be None
  assert kda_states[0] is not None
  assert kda_states[1] is not None
  assert kda_states[2] is not None
  assert kda_states[3] is None


def test_kimi_linear_model_with_initial_kda_state():
  """Test KimiLinearModel with pre-populated initial KDA states."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
  ])

  rngs = nnx.Rngs(0)
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

  model = KimiLinearModel(cfg, mesh, rngs=rngs)

  # Create dummy initial KDA state for layer 0: (batch=2, num_heads=4, head_dim=64, head_dim=64)
  init_kda_state0 = jnp.ones((2, cfg.num_query_heads, cfg.head_dim, cfg.head_dim))
  initial_kda_states = [init_kda_state0, None, None, None]

  input_ids = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
  inputs_positions = jnp.arange(4)[None, :].repeat(2, axis=0)

  logits, kda_states = model(
      input_ids,
      inputs_positions=inputs_positions,
      initial_kda_states=initial_kda_states,
  )

  assert logits.shape == (2, 4, cfg.vocab_size)
  assert not jnp.isnan(logits).any()
