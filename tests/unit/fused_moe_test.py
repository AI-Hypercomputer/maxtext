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
"""Tests for fused_moe_matmul (vllm_rpa path) in RoutedMoE."""

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
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides


def make_moe(cfg, mesh):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      dtype=cfg.dtype,
      rngs=nnx.Rngs(params=0),
  )


def copy_weights(src_model, dst_model):
  """Copy wi_0, wi_1, wo, and gate weights from src to dst."""
  dst_model.wi_0 = src_model.wi_0
  dst_model.wi_1 = src_model.wi_1
  dst_model.wo = src_model.wo
  dst_model.gate = src_model.gate


# fused_moe_func requires num_tokens * topk % 16 == 0.
# B=1, S=16, topk=2 -> T*topk = 32, divisible by 16.
_B = 1
_S = 16


class FusedMoeTPUTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

    # Dense reference config (no vllm, einsum-based)
    extra_args = get_decoupled_parallelism_overrides()
    self.dense_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_dense_ref",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=False,
        megablox=False,
        max_target_length=_S,
        per_device_batch_size=_B,
        **extra_args,
    )
    dense_devices = maxtext_utils.create_device_mesh(self.dense_cfg)
    self.dense_mesh = Mesh(dense_devices, self.dense_cfg.mesh_axes)
    self.dense_model = make_moe(self.dense_cfg, self.dense_mesh)

    # vllm_rpa fused config
    self.fused_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml"), get_test_config_path()],
        run_name="fused_moe_vllm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        max_target_length=_S,
        per_device_batch_size=_B,
    )
    fused_devices = maxtext_utils.create_device_mesh(self.fused_cfg)
    self.fused_mesh = Mesh(fused_devices, self.fused_cfg.mesh_axes)
    self.fused_model = make_moe(self.fused_cfg, self.fused_mesh)
    copy_weights(self.dense_model, self.fused_model)

  def _inputs(self):
    return jax.random.normal(self.rng, (_B, _S, self.dense_cfg.base_emb_dim), dtype=jnp.bfloat16)

  def test_fused_vs_dense_softmax(self):
    """fused_moe_matmul agrees with dense_matmul under softmax routing."""
    inputs = self._inputs()

    dense_out, _, _ = self.dense_model(inputs)
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    np.testing.assert_allclose(
        np.array(dense_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_vs_sparse_softmax(self):
    """fused_moe_matmul agrees with sparse_matmul (Megablox) under softmax routing."""
    extra_args = get_decoupled_parallelism_overrides()
    sparse_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_sparse_ref",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=True,
        megablox=True,
        max_target_length=_S,
        per_device_batch_size=_B,
        **extra_args,
    )
    sparse_devices = maxtext_utils.create_device_mesh(sparse_cfg)
    sparse_mesh = Mesh(sparse_devices, sparse_cfg.mesh_axes)
    sparse_model = make_moe(sparse_cfg, sparse_mesh)
    copy_weights(self.dense_model, sparse_model)

    inputs = self._inputs()
    sparse_out, _, _ = sparse_model(inputs)
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    np.testing.assert_allclose(
        np.array(sparse_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_output_shape_and_dtype(self):
    """Output shape is (B, S, D), dtype matches cfg.dtype, and losses are None."""
    inputs = self._inputs()
    fused_out, lb_loss, bias_updates = self.fused_model(inputs)

    expected_shape = (_B, _S, self.fused_cfg.base_emb_dim)
    self.assertEqual(fused_out.shape, expected_shape)
    self.assertEqual(fused_out.dtype, self.fused_cfg.dtype)
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)

  def test_fused_vs_dense_renormalize(self):
    """fused_moe_matmul agrees with dense_matmul when norm_topk_prob=True."""
    extra_args = get_decoupled_parallelism_overrides()
    dense_renorm_cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="fused_moe_dense_renorm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        sparse_matmul=False,
        megablox=False,
        norm_topk_prob=True,
        max_target_length=_S,
        per_device_batch_size=_B,
        **extra_args,
    )
    dense_renorm_devices = maxtext_utils.create_device_mesh(dense_renorm_cfg)
    dense_renorm_mesh = Mesh(dense_renorm_devices, dense_renorm_cfg.mesh_axes)
    dense_renorm_model = make_moe(dense_renorm_cfg, dense_renorm_mesh)

    fused_renorm_cfg = pyconfig.initialize(
        [None, get_test_config_path("inference/vllm.yml"), get_test_config_path()],
        run_name="fused_moe_vllm_renorm",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        norm_topk_prob=True,
        max_target_length=_S,
        per_device_batch_size=_B,
    )
    fused_renorm_devices = maxtext_utils.create_device_mesh(fused_renorm_cfg)
    fused_renorm_mesh = Mesh(fused_renorm_devices, fused_renorm_cfg.mesh_axes)
    fused_renorm_model = make_moe(fused_renorm_cfg, fused_renorm_mesh)
    copy_weights(dense_renorm_model, fused_renorm_model)

    inputs = self._inputs()
    dense_out, _, _ = dense_renorm_model(inputs)
    fused_out, lb_loss, bias_updates = fused_renorm_model(inputs)

    np.testing.assert_allclose(
        np.array(dense_out, dtype=np.float32),
        np.array(fused_out, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    self.assertIsNone(lb_loss)
    self.assertIsNone(bias_updates)


if __name__ == "__main__":
  unittest.main()
