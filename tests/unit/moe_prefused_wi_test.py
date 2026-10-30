# Copyright 2023–2026 Google LLC
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

"""RoutedMoE with `prefuse_moe_weights=True` and `moe_unsliced_prefused_wi=True`
matches the separate wi_0 / wi_1 weights in forward and backward."""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class PrefusedWiSparseMatmulTest(unittest.TestCase):
  """prefuse_moe_weights=True + moe_unsliced_prefused_wi=True vs False under sparse_matmul + gmm_v2."""

  def _make_model(self, prefuse):
    """Small mixtral-style RoutedMoE (gmm_v2, experts sharded on fsdp) with fused or separate wi."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"moe_prefused_wi_{prefuse}",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=512,
        base_mlp_dim=1024,
        base_moe_mlp_dim=1024,
        dtype="bfloat16",
        weight_dtype="float32",
        sparse_matmul=True,
        megablox=False,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        prefuse_moe_weights=prefuse,
        moe_unsliced_prefused_wi=prefuse,
        shard_exp_on_fsdp=True,
        ici_fsdp_parallelism=jax.device_count(),
        per_device_batch_size=1,
        max_target_length=64,
        log_config=False,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )
    return cfg, mesh, model

  @pytest.mark.tpu_only
  def test_prefused_matches_separate_weights(self):
    cfg, mesh, split_model = self._make_model(prefuse=False)
    _, _, fused_model = self._make_model(prefuse=True)
    fused_model.wi = nnx.Param(
        jnp.concatenate([split_model.wi_0[...], split_model.wi_1[...]], axis=-1),
        out_sharding=fused_model.wi_kernel_axes,
    )
    fused_model.wo = split_model.wo
    fused_model.gate = split_model.gate

    inputs = jax.random.normal(jax.random.PRNGKey(1), (jax.device_count(), 64, cfg.emb_dim), jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.PRNGKey(2), inputs.shape, jnp.float32)

    def loss(model, x):
      out, _, _ = model(x)
      return jnp.sum(out.astype(jnp.float32) * cotangent)

    with mesh:
      run = nnx.jit(nnx.value_and_grad(loss, argnums=(0, 1)))
      loss_split, (grads_split, dx_split) = run(split_model, inputs)
      loss_fused, (grads_fused, dx_fused) = run(fused_model, inputs)

    np.testing.assert_allclose(float(loss_fused), float(loss_split), rtol=1e-3)
    np.testing.assert_allclose(np.asarray(dx_fused, np.float32), np.asarray(dx_split, np.float32), rtol=2e-2, atol=1e-2)
    dwi_split = jnp.concatenate([grads_split["wi_0"][...], grads_split["wi_1"][...]], axis=-1)
    np.testing.assert_allclose(
        np.asarray(grads_fused["wi"][...], np.float32), np.asarray(dwi_split, np.float32), rtol=2e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(grads_fused["wo"][...], np.float32),
        np.asarray(grads_split["wo"][...], np.float32),
        rtol=2e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  unittest.main()
