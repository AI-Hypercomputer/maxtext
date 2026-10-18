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

"""gdn_in_proj_kernel_order: in_proj_qkvz computed directly in the GDN kernel's q|k|v order gives the same values.

With the key on, models/qwen3.py permutes the in_proj_qkvz kernel's columns from the per-key-head [q | k | v | z] order
into [q all | k all | v all | z all] and computes qkv (straight to the Pallas kernel path, model_runner qkv_precomputed)
and z with two matmuls under a custom VJP whose input gradient is one float32 sum of the two partial products.

  - Config: the key defaults to False (types.py, base.yml, the constructed config); the validator accepts it with the GDN
    kernel and sequence-sharded GDN context parallelism and refuses it without the kernel, with gdn_cp_mode='head' and
    without a Gated DeltaNet decoder, naming the key.
  - The column permutation equals a NumPy reference.
  - The real Qwen3NextGatedDeltaNet in training (use_gdn_kernel=True; the kernels' CPU path), key on vs off, packed and
    unpacked: the loss and every parameter gradient bit for bit; the input gradient to float32 summation order
    (relative 1e-3; measured ~7e-5 / ~7e-4). With the key on the layer calls the kernel-order matmul, with it off it
    does not.
  - Positive controls through the same comparison: the q and k column blocks swapped, and the input gradient summed as
    two separately rounded bf16 products, each break it, so the comparison can fail.
"""

import os
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_SEQ, _EMB = 128, 64
_HK, _DK, _HV, _DV = 2, 128, 4, 128  # key heads, key head dim, value heads, value head dim
_VPK = _HV // _HK * _DV  # value columns per key head
_DX_RTOL = 1e-3
_ORIG = {
    n: getattr(qwen3, n)
    for n in ("_gdn_in_proj_kernel_order_columns", "_gdn_in_proj_kernel_order_dx", "_gdn_in_proj_kernel_order_dot")
}


def _cfg(order):
  return pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=gdn_in_proj_kernel_order_test",
          "dtype=bfloat16",
          "weight_dtype=float32",
          "decoder_block=qwen3_next",
          "attention=dot_product",
          f"base_emb_dim={_EMB}",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "head_dim=32",
          f"gdn_num_key_heads={_HK}",
          f"gdn_num_value_heads={_HV}",
          f"gdn_key_head_dim={_DK}",
          f"gdn_value_head_dim={_DV}",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          f"max_target_length={_SEQ}",
          "use_gdn_kernel=True",
          "gdn_out_proj_token_minor=True",
          "gdn_gated_norm_layout=token_minor",
          f"gdn_in_proj_kernel_order={order}",
      ],
      skip_jax_distributed_system=True,
  )


def _grads(order, packed):
  """Loss, input gradient and every parameter gradient of the real layer (float64 copies, in a fixed order)."""
  cfg = _cfg(order)
  assert cfg.gdn_in_proj_kernel_order is order
  mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
  layer = qwen3.Qwen3NextGatedDeltaNet(
      config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, inputs_shape=(1, _SEQ, cfg.emb_dim), rngs=nnx.Rngs(0)
  )
  graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
  rng = np.random.default_rng(7)
  hidden = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32)).astype(jnp.bfloat16)
  dy = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32))
  segs = jnp.asarray(np.concatenate([np.ones(80), 2 * np.ones(_SEQ - 80)]).astype(np.int32))[None] if packed else None

  def loss(p, h):
    with mesh:
      out = nnx.merge(graphdef, p, rest)(h, decoder_segment_ids=segs, model_mode=MODEL_MODE_TRAIN)[0]
    return jnp.sum(out.astype(jnp.float32) * dy)

  value, (g_params, g_hidden) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(params, hidden)
  named = [("loss", value), ("d_input", g_hidden)]
  named += [(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(g_params)]
  return [(n, np.asarray(jax.device_get(v), dtype=np.float64)) for n, v in named]


def _violations(ref, got):
  """The comparison's failures: any loss / parameter-gradient bit, or an input gradient beyond float32 order."""
  bad = []
  for (name, x), (name_got, y) in zip(ref, got, strict=True):
    assert name == name_got, (name, name_got)
    if name == "d_input":
      rel = float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(x))), 1e-30)
      if rel > _DX_RTOL:
        bad.append(f"d_input rel {rel:.3e}")
    elif not np.array_equal(x, y):
      bad.append(f"{name} not bit-identical")
  return bad


def _qkswap_columns(kernel, num_k_heads, head_k_dim, v_dim_per_k_head):
  k = _ORIG["_gdn_in_proj_kernel_order_columns"](kernel, num_k_heads, head_k_dim, v_dim_per_k_head)
  n = num_k_heads * head_k_dim
  return jnp.concatenate([k[..., n : 2 * n], k[..., :n], k[..., 2 * n :]], axis=-1)


def _bf16_dx(d_qkv, d_z, w_qkv, w_z, precision):
  dims = (((d_qkv.ndim - 1,), (1,)), ((), ()))
  return jax.lax.dot_general(d_qkv, w_qkv, dims, precision=precision) + jax.lax.dot_general(
      d_z, w_z, dims, precision=precision
  )


class LayerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ref = {packed: _grads(False, packed) for packed in (True, False)}

  @parameterized.parameters(True, False)
  def test_same_values(self, packed):
    self.assertEqual(_violations(self.ref[packed], _grads(True, packed)), [])

  def test_key_selects_the_kernel_order_matmul(self):
    for order, expect_calls in ((True, True), (False, False)):
      jax.clear_caches()
      with mock.patch.object(qwen3, "_gdn_in_proj_kernel_order_dot", wraps=_ORIG["_gdn_in_proj_kernel_order_dot"]) as dot:
        _grads(order, True)
      self.assertEqual(dot.called, expect_calls, f"gdn_in_proj_kernel_order={order}")

  @parameterized.named_parameters(
      ("qk_blocks_swapped", "_gdn_in_proj_kernel_order_columns", _qkswap_columns),
      ("input_gradient_in_bf16", "_gdn_in_proj_kernel_order_dx", _bf16_dx),
  )
  def test_positive_control(self, name, impl):
    jax.clear_caches()
    with mock.patch.object(qwen3, name, impl):
      got = _grads(True, True)
    jax.clear_caches()
    self.assertNotEqual(_violations(self.ref[True], got), [])


class ColumnsTest(absltest.TestCase):

  def test_matches_numpy_reference(self):
    rng = np.random.default_rng(3)
    kernel = rng.standard_normal((_EMB, _HK * (2 * _DK + 2 * _VPK))).astype(np.float32)
    per_head = kernel.reshape(_EMB, _HK, 2 * _DK + 2 * _VPK)
    q, k = per_head[..., :_DK], per_head[..., _DK : 2 * _DK]
    v, z = per_head[..., 2 * _DK : 2 * _DK + _VPK], per_head[..., 2 * _DK + _VPK :]
    want = np.concatenate([x.reshape(_EMB, -1) for x in (q, k, v, z)], axis=-1)
    got = np.asarray(qwen3._gdn_in_proj_kernel_order_columns(jnp.asarray(kernel), _HK, _DK, _VPK))  # pylint: disable=protected-access
    np.testing.assert_array_equal(got, want)


def _validator_cfg(**overrides):
  fields = {
      "decoder_block": config_types.DecoderBlockType.QWEN3_5,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_context_usp_ulysses_parallelism": 1,
      "dcn_context_usp_ulysses_parallelism": 1,
      "use_gdn_kernel": True,
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "gdn_chunk_size": 64,
      "gdn_key_head_dim": 128,
      "gdn_value_head_dim": 128,
      "gdn_num_key_heads": 16,
      "gdn_num_value_heads": 64,
      "use_qk_norm_in_gdn": True,
      "max_target_length": 4096,
      "gdn_conv1d_input_in_qkv_dtype": False,
      "gdn_conv1d_dense_token_stack": False,
      "gdn_gated_norm_layout": "auto",
      "gdn_cp_mode": "seq",
      "gdn_cp_pass1_states_and_bf16_qkv": False,
      "dtype": "bfloat16",
      "packing": True,
      "gdn_in_proj_kernel_order": True,
  }
  fields.update(overrides)
  return SimpleNamespace(**fields)


class ConfigTest(parameterized.TestCase):

  def test_default_off(self):
    self.assertIs(config_types.Qwen3Next.model_fields["gdn_in_proj_kernel_order"].default, False)
    with open(os.path.join(os.path.dirname(config_types.__file__), "base.yml"), encoding="utf-8") as f:
      self.assertIn("\ngdn_in_proj_kernel_order: false\n", f.read())

  def test_accepted(self):
    config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(), is_rl_config=False)

  @parameterized.named_parameters(
      ("without_the_kernel", {"use_gdn_kernel": False}, "requires use_gdn_kernel=True"),
      ("head_sharded_cp", {"gdn_cp_mode": "head"}, "gdn_cp_mode='head'"),
      ("without_gdn", {"decoder_block": config_types.DecoderBlockType.LLAMA2}, "Gated DeltaNet"),
  )
  def test_rejected(self, overrides, pattern):
    with self.assertRaisesRegex(ValueError, "gdn_in_proj_kernel_order=True") as ctx:
      config_types.MaxTextConfig.validate_gdn_config(_validator_cfg(**overrides), is_rl_config=False)
    self.assertRegex(str(ctx.exception), pattern)


if __name__ == "__main__":
  absltest.main()
