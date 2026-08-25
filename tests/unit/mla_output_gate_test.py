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

"""Unit tests for MLA with Output Gate in MaxText."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


from maxtext.configs import pyconfig
from maxtext.layers.attention_mla import MLA


def test_mla_output_gate_initialization_and_forward():
  """Test that MLA with mla_use_output_gate=True initializes and executes a forward pass."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
      "mla_use_output_gate=True",
  ])

  rngs = nnx.Rngs(0)
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

  mla = MLA(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      mesh=mesh,
      attention_kernel=cfg.attention,
      inputs_q_shape=(2, 8, cfg.emb_dim),
      inputs_kv_shape=(2, 8, cfg.emb_dim),
      rngs=rngs,
  )

  assert hasattr(mla, "g_a_proj")
  assert hasattr(mla, "g_b_proj")
  assert hasattr(mla, "o_norm")

  x = jnp.ones((2, 8, cfg.emb_dim))
  out, _ = mla(
      inputs_q=x,
      inputs_kv=x,
      inputs_positions=jnp.arange(8)[None, :].repeat(2, axis=0),
  )

  assert out.shape == (2, 8, cfg.emb_dim)
  assert not jnp.isnan(out).any()
