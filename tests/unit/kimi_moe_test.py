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

"""Unit tests for Kimi K3 896-expert MoE in MaxText."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.moe import RoutedAndSharedMoE



def test_kimi_moe_initialization_and_forward():
  """Test that RoutedAndSharedMoE with Kimi K3 896-expert config initializes and executes."""
  cfg = pyconfig.initialize([
      "",
      "src/maxtext/configs/models/kimi-k3-tiny.yml",
      "run_name=test",
      "steps=1",
      "log_config=False",
      "skip_jax_distributed_system=True",
      "latent_moe_use_norm=True",
  ])

  rngs = nnx.Rngs(0)
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", "model"))

  moe = RoutedAndSharedMoE(
      config=cfg,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=("embed_moe", None),
      rngs=rngs,
  )


  assert hasattr(moe, "routed_expert_norm")
  assert moe.routed_expert_norm is not None

  x = jnp.ones((2, 4, cfg.emb_dim))
  out, _, _ = moe(x)

  assert out.shape == (2, 4, cfg.emb_dim)
  assert not jnp.isnan(out).any()
