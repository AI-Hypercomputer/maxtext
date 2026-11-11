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

"""Unit tests verifying decoupled GMM backend dispatch and quantization composition."""

import dataclasses
import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.kernels import gmm as gmm_lib
from maxtext.kernels import megablox
from maxtext.kernels.gmm import backends as gmm_backends
from maxtext.kernels.gmm import quantize as gmm_quantize
from maxtext.kernels.megablox import backend as megablox_backend
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2_kernel
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2_kernel
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils
from tests.utils import test_helpers
import numpy as np
import qwix
import qwix.pallas as qpl


def _ref_grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
  """Reference per-group matrix multiplication handling empty groups and trailing padding."""
  m = lhs.shape[0]
  n = rhs.shape[2]
  out = jnp.zeros((m, n), dtype=jnp.float32)
  offset = 0
  for g in range(rhs.shape[0]):
    size = int(group_sizes[g])
    if size > 0:
      out = out.at[offset : offset + size].set(
          lhs[offset : offset + size].astype(jnp.float32) @ rhs[g].astype(jnp.float32)
      )
      offset += size
  return out


def _make_sparse_inputs(
    key: jax.Array,
    *,
    num_groups: int = 4,
    group_sizes_list: tuple[int, ...] = (3, 0, 5, 2),
    m_pad: int = 12,
    k: int = 16,
    n: int = 8,
    dtype=jnp.float32,
):
  """Generates deterministic sparse MoE inputs with empty and padded groups."""
  k1, k2 = jax.random.split(key)
  lhs = jax.random.normal(k1, (m_pad, k), dtype=dtype) * 0.5
  rhs = jax.random.normal(k2, (num_groups, k, n), dtype=dtype) * 0.5
  group_sizes = jnp.array(group_sizes_list, dtype=jnp.int32)
  return lhs, rhs, group_sizes


class DecouplingArchitectureTest(parameterized.TestCase):
  """Tests module boundaries, backend immutability, and capability traits."""

  def test_megablox_and_moe_do_not_couple_tokamax(self):
    self.assertFalse(hasattr(megablox, "tokamax"))
    self.assertFalse(hasattr(megablox_ops, "tokamax"))
    self.assertFalse(hasattr(megablox_backend, "tokamax"))
    self.assertFalse(hasattr(moe, "tokamax"))

  def test_backends_module_does_not_import_quantizations_or_tokamax_eagerly(self):
    self.assertNotIn("quantizations", vars(gmm_backends))
    self.assertNotIn("tokamax", vars(gmm_backends))
    self.assertNotIn("transformer_engine", vars(gmm_backends))

  @parameterized.parameters(
      (gmm_backends.JaxRaggedDotBackend(),),
      (gmm_backends.MegabloxBackend(),),
      (gmm_backends.TokamaxV1Backend(),),
      (gmm_backends.TokamaxV2Backend(),),
      (gmm_backends.TeGroupedGemmBackend(),),
  )
  def test_backends_are_frozen_and_hashable(self, backend: gmm_backends.GmmBackend):
    _ = hash(backend)
    self.assertEqual(backend, backend)
    attr = "is_tpu" if isinstance(backend, gmm_backends.JaxRaggedDotBackend) else None
    if attr is not None:
      with self.assertRaises(dataclasses.FrozenInstanceError):
        setattr(backend, attr, True)

  def test_backend_capability_traits(self):
    jax_rd = gmm_backends.JaxRaggedDotBackend()
    self.assertFalse(jax_rd.requires_custom_vjp)
    self.assertTrue(jax_rd.supports_native_qwix_interception)
    self.assertFalse(jax_rd.supports_3d_lhs)
    self.assertFalse(jax_rd.supports_bias)

    mblx = gmm_backends.MegabloxBackend()
    self.assertTrue(mblx.requires_custom_vjp)
    self.assertTrue(mblx.accumulate_existing_out_in_bwd)
    self.assertFalse(mblx.supports_native_lhs_quant)

    tv1 = gmm_backends.TokamaxV1Backend()
    self.assertFalse(tv1.requires_custom_vjp)
    self.assertTrue(tv1.supports_qag)
    self.assertFalse(tv1.supports_tc_unsort)

    tv2 = gmm_backends.TokamaxV2Backend()
    self.assertTrue(tv2.requires_custom_vjp)
    self.assertTrue(tv2.supports_qag)
    self.assertTrue(tv2.supports_native_lhs_quant)
    self.assertTrue(tv2.dlhs_needs_repeated_rhs_scale)
    self.assertTrue(tv2.supports_3d_lhs)
    self.assertTrue(tv2.supports_partial_sum)
    self.assertTrue(tv2.supports_tc_unsort)
    self.assertTrue(tv2.supports_flat_fsdp_shard)

    te = gmm_backends.TeGroupedGemmBackend()
    self.assertFalse(te.requires_custom_vjp)
    self.assertTrue(te.supports_bias)
    self.assertTrue(te.supports_partial_sum)


class BackendSelectionTest(parameterized.TestCase):
  """Tests `select_gmm_backend` routing and invalid flag combination checks."""

  def test_select_default_jax_ragged_dot(self):
    backend = gmm_lib.select_gmm_backend(hardware="tpu", is_tpu=True)
    self.assertEqual(backend, gmm_lib.JaxRaggedDotBackend(is_tpu=True))

  def test_select_megablox(self):
    backend = gmm_lib.select_gmm_backend(megablox=True, hardware="tpu")
    self.assertEqual(backend, gmm_lib.MegabloxBackend())

  def test_select_tokamax_v1(self):
    backend = gmm_lib.select_gmm_backend(use_tokamax_gmm=True, hardware="tpu")
    self.assertEqual(backend, gmm_lib.TokamaxV1Backend())

  def test_select_tokamax_v2_forwards_kernel_flags(self):
    backend = gmm_lib.select_gmm_backend(
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        use_gmm_v2_heuristic_tiling=True,
        hardware="tpu",
        use_dlhs_transpose_rhs=True,
        fuse_dlhs_scale=True,
        fuse_3d_dlhs_ps=True,
    )
    self.assertEqual(
        backend,
        gmm_lib.TokamaxV2Backend(
            heuristic_tiling=True,
            dlhs_transpose_rhs=True,
            fuse_dlhs_scale=True,
            fuse_3d_dlhs_ps=True,
        ),
    )

  def test_select_te_grouped_gemm(self):
    backend = gmm_lib.select_gmm_backend(use_te_grouped_gemm=True, hardware="gpu")
    self.assertEqual(backend, gmm_lib.TeGroupedGemmBackend())

  def test_cpu_forces_jax_ragged_dot_over_pallas_or_tokamax(self):
    backend = gmm_lib.select_gmm_backend(
        megablox=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        hardware="cpu",
        is_tpu=False,
    )
    self.assertEqual(backend, gmm_lib.JaxRaggedDotBackend(is_tpu=False))

  @parameterized.named_parameters(
      ("v2_without_tokamax", {"use_gmm_v2": True, "use_tokamax_gmm": False, "hardware": "tpu"}),
      ("te_with_tokamax", {"use_te_grouped_gemm": True, "use_tokamax_gmm": True, "hardware": "gpu"}),
      ("te_with_megablox", {"use_te_grouped_gemm": True, "megablox": True, "hardware": "gpu"}),
  )
  def test_invalid_backend_combinations_raise(self, kwargs):
    with self.assertRaises(ValueError):
      gmm_lib.select_gmm_backend(**kwargs)


class QuantizedGmmCpuExecutionTest(parameterized.TestCase):
  """Tests real CPU execution of `grouped_matmul` and `quantized_gmm` across quantization modes."""

  def test_unquantized_native_vs_custom_vjp_parity_with_empty_and_padded_groups(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(0))

    def ref_fn(l, r):
      return jnp.sum(_ref_grouped_matmul(l, r, group_sizes) ** 2)

    def native_fn(l, r):
      """Computes squared sum loss with JaxRaggedDotBackend native execution."""
      return jnp.sum(
          gmm_lib.grouped_matmul(
              l,
              r,
              group_sizes,
              backend=gmm_lib.JaxRaggedDotBackend(is_tpu=False),
              preferred_element_type=jnp.float32,
          )
          ** 2
      )

    def vjp_fn(l, r):
      """Computes squared sum loss via quantized_gmm custom VJP."""
      return jnp.sum(
          gmm_lib.quantized_gmm(
              l,
              r,
              group_sizes,
              backend=gmm_lib.JaxRaggedDotBackend(is_tpu=False),
              preferred_element_type=jnp.float32,
          )
          ** 2
      )

    ref_loss, (ref_dlhs, ref_drhs) = jax.value_and_grad(ref_fn, argnums=(0, 1))(lhs, rhs)
    nat_loss, (nat_dlhs, nat_drhs) = jax.value_and_grad(native_fn, argnums=(0, 1))(lhs, rhs)
    vjp_loss, (vjp_dlhs, vjp_drhs) = jax.value_and_grad(vjp_fn, argnums=(0, 1))(lhs, rhs)

    np.testing.assert_allclose(nat_loss, ref_loss, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(nat_dlhs, ref_dlhs, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(nat_drhs, ref_drhs, rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(vjp_loss, ref_loss, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vjp_dlhs, ref_dlhs, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vjp_drhs, ref_drhs, rtol=1e-5, atol=1e-5)

    # Active rows slice check: trailing rows [10:12] must stay zero.
    out = gmm_lib.grouped_matmul(
        lhs,
        rhs,
        group_sizes,
        backend=gmm_lib.JaxRaggedDotBackend(is_tpu=False),
        weight_gather_axes=[],  # Empty list must NOT force custom VJP.
    )
    np.testing.assert_array_equal(out[10:], jnp.zeros((2, 8), dtype=jnp.float32))

  @parameterized.named_parameters(
      ("same_dtype_float32", jnp.float32, 1e-5),
      ("mismatched_dtype_bfloat16", jnp.bfloat16, 2e-2),
  )
  def test_existing_out_fwd_and_bwd_including_dtype_mismatch(self, existing_dtype, tol):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(3))
    eo = jax.random.normal(jax.random.PRNGKey(4), (lhs.shape[0], rhs.shape[2]), dtype=existing_dtype)
    backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)

    def ref_fn(l, r, existing):
      out = _ref_grouped_matmul(l, r, group_sizes) + existing.astype(jnp.float32)
      return jnp.sum(out**2)

    def grouped_fn(l, r, existing):
      """Computes squared sum loss via grouped_matmul with existing_out."""
      out = gmm_lib.grouped_matmul(
          l,
          r,
          group_sizes,
          backend=backend,
          preferred_element_type=jnp.float32,
          existing_out=existing,
      )
      return jnp.sum(out**2)

    def vjp_fn(l, r, existing):
      """Computes squared sum loss via quantized_gmm with existing_out."""
      out = gmm_lib.quantized_gmm(
          l,
          r,
          group_sizes,
          backend=backend,
          preferred_element_type=jnp.float32,
          existing_out=existing,
      )
      return jnp.sum(out**2)

    ref_out = _ref_grouped_matmul(lhs, rhs, group_sizes) + eo.astype(jnp.float32)
    nat_out = gmm_lib.grouped_matmul(
        lhs, rhs, group_sizes, backend=backend, preferred_element_type=jnp.float32, existing_out=eo
    )
    vjp_out = gmm_lib.quantized_gmm(
        lhs, rhs, group_sizes, backend=backend, preferred_element_type=jnp.float32, existing_out=eo
    )
    self.assertEqual(nat_out.dtype, jnp.float32)
    self.assertEqual(vjp_out.dtype, jnp.float32)
    np.testing.assert_allclose(nat_out, ref_out, rtol=tol, atol=tol)
    np.testing.assert_allclose(vjp_out, ref_out, rtol=tol, atol=tol)

    ref_loss, (ref_dlhs, ref_drhs, ref_deo) = jax.value_and_grad(ref_fn, argnums=(0, 1, 2))(lhs, rhs, eo)
    nat_loss, (nat_dlhs, nat_drhs, nat_deo) = jax.value_and_grad(grouped_fn, argnums=(0, 1, 2))(lhs, rhs, eo)
    vjp_loss, (vjp_dlhs, vjp_drhs, vjp_deo) = jax.value_and_grad(vjp_fn, argnums=(0, 1, 2))(lhs, rhs, eo)

    for loss, dlhs, drhs, deo in (
        (nat_loss, nat_dlhs, nat_drhs, nat_deo),
        (vjp_loss, vjp_dlhs, vjp_drhs, vjp_deo),
    ):
      np.testing.assert_allclose(loss, ref_loss, rtol=tol, atol=tol)
      np.testing.assert_allclose(dlhs, ref_dlhs, rtol=tol, atol=tol)
      np.testing.assert_allclose(drhs, ref_drhs, rtol=tol, atol=tol)
      self.assertEqual(deo.dtype, existing_dtype)
      np.testing.assert_allclose(deo, ref_deo, rtol=tol, atol=tol)

    # Direct jax.vjp with existing_out=None must return None cotangent (no tangent mismatch).
    _, pull_none = jax.vjp(
        lambda l, r, existing: gmm_lib.quantized_gmm(
            l, r, group_sizes, backend=backend, preferred_element_type=jnp.float32, existing_out=existing
        ),
        lhs,
        rhs,
        None,
    )
    _, _, deo_none = pull_none(jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32))
    self.assertIsNone(deo_none)

    # Backends with accumulate_existing_out_in_bwd=False do not differentiate existing_out.
    self.assertFalse(gmm_lib.TokamaxV1Backend().accumulate_existing_out_in_bwd)
    with mock.patch.object(gmm_backends.JaxRaggedDotBackend, "accumulate_existing_out_in_bwd", False):
      _, pull_no_eo = jax.vjp(
          lambda l, r, existing: gmm_lib.quantized_gmm(
              l, r, group_sizes, backend=backend, preferred_element_type=jnp.float32, existing_out=existing
          ),
          lhs,
          rhs,
          eo,
      )
      _, _, deo_disabled = pull_no_eo(jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32))
      np.testing.assert_array_equal(deo_disabled, jnp.zeros_like(eo))

  @parameterized.named_parameters(
      ("int8_dynamic", jnp.int8, jnp.int8, jnp.int8, "absmax", "absmax", "absmax", False),
      ("int8_weight_only", None, jnp.int8, None, None, "absmax", None, False),
      (
          "fp8_dynamic_channelwise",
          jnp.float8_e4m3fn,
          jnp.float8_e4m3fn,
          jnp.float8_e5m2,
          "absmax",
          "absmax",
          "absmax",
          False,
      ),
      (
          "fp8_static_per_tensor",
          jnp.float8_e4m3fn,
          jnp.float8_e4m3fn,
          jnp.float8_e5m2,
          "fixed,-2.0,2.0",
          "fixed,-2.0,2.0",
          "fixed,-2.0,2.0",
          True,
      ),
  )
  def test_qwix_rule_quantized_fwd_and_bwd_on_cpu_backend(
      self,
      act_qtype,
      weight_qtype,
      bwd_qtype,
      act_cal,
      weight_cal,
      bwd_cal,
      disable_channelwise_axes,
  ):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(42))
    rule = qwix.QtRule(
        act_qtype=act_qtype,
        weight_qtype=weight_qtype,
        bwd_qtype=bwd_qtype,
        act_calibration_method=act_cal,
        weight_calibration_method=weight_cal,
        bwd_calibration_method=bwd_cal,
        disable_channelwise_axes=disable_channelwise_axes,
    )
    backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)

    def loss_fn(l, r):
      """Computes quantized grouped_matmul squared sum loss."""
      y = gmm_lib.grouped_matmul(
          l,
          r,
          group_sizes,
          backend=backend,
          preferred_element_type=jnp.float32,
          use_qwix_quantization=True,
          qwix_rule=rule,
      )
      return jnp.sum(y**2)

    def ref_loss_fn(l, r):
      return jnp.sum(_ref_grouped_matmul(l, r, group_sizes) ** 2)

    ref_loss, (ref_dlhs, ref_drhs) = jax.value_and_grad(ref_loss_fn, argnums=(0, 1))(lhs, rhs)
    q_loss, (q_dlhs, q_drhs) = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)

    self.assertTrue(jnp.isfinite(q_loss))
    self.assertTrue(jnp.all(jnp.isfinite(q_dlhs)))
    self.assertTrue(jnp.all(jnp.isfinite(q_drhs)))
    # Quantized forward and gradients should track the unquantized reference closely.
    loss_tol = 0.25 if act_qtype == jnp.float8_e4m3fn else 0.12
    np.testing.assert_allclose(q_loss, ref_loss, rtol=loss_tol, atol=0.1)
    if act_qtype == jnp.float8_e4m3fn:
      # e4m3fn/e5m2 have 2-3 mantissa bits; verify relative L2 norm rather than strict entrywise tolerance.
      for actual, ref in ((q_dlhs, ref_dlhs), (q_drhs, ref_drhs)):
        rel_l2 = float(jnp.linalg.norm(actual - ref) / jnp.linalg.norm(ref))
        self.assertLess(rel_l2, 0.15)
    else:
      np.testing.assert_allclose(q_dlhs, ref_dlhs, rtol=0.12, atol=0.1)
      np.testing.assert_allclose(q_drhs, ref_drhs, rtol=0.12, atol=0.1)

  def test_lhs_and_rhs_quantize_dtypes(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(7))
    backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)

    def loss_fn(l, r):
      """Computes grouped_matmul loss with explicit int8 input dtypes."""
      return jnp.sum(
          gmm_lib.grouped_matmul(
              l,
              r,
              group_sizes,
              backend=backend,
              lhs_quantize_dtype=jnp.int8,
              rhs_quantize_dtype=jnp.int8,
          )
      )

    loss, (dlhs, drhs) = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    self.assertTrue(jnp.isfinite(loss))
    self.assertEqual(dlhs.shape, lhs.shape)
    self.assertEqual(drhs.shape, rhs.shape)

  @parameterized.named_parameters(
      ("scalar_rhs_scale", ()),
      ("per_tensor_3d_rhs_scale", (1, 1, 1)),
      ("per_channel_rhs_scale", (4, 1, 8)),
  )
  def test_prequantized_qarray_operands_including_scalar_scale(self, scale_shape):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(11))
    lhs_q = qpl.quantize(lhs, jnp.int8, channelwise_axes=[0])
    if not scale_shape:
      rhs_scale = jnp.max(jnp.abs(rhs)) / 127.0
      rhs_qval = jnp.round(rhs / rhs_scale).astype(jnp.int8)
      rhs_q = qpl.QArray(qvalue=rhs_qval, scale=rhs_scale, zero_point=None, qtype=jnp.int8)
    elif scale_shape == (1, 1, 1):
      rhs_scale = jnp.max(jnp.abs(rhs), keepdims=True) / 127.0
      rhs_qval = jnp.round(rhs / rhs_scale).astype(jnp.int8)
      rhs_q = qpl.QArray(qvalue=rhs_qval, scale=rhs_scale, zero_point=None, qtype=jnp.int8)
    else:
      rhs_q = qpl.quantize(rhs, jnp.int8, channelwise_axes=[0, 2])

    backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)
    rhs_deq = rhs_q.qvalue.astype(jnp.float32) * gmm_lib.scale_like(rhs_q.scale, rhs_q.qvalue, jnp.float32)
    out = gmm_lib.grouped_matmul(lhs_q, rhs_q, group_sizes, backend=backend)
    ref = _ref_grouped_matmul(qwix.dequantize(lhs_q), rhs_deq, group_sizes)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    # Also exercise backward pass wrt float lhs (verifying dlhs_scale_grad_by_rhs_scale across scale shapes).
    lhs_f = qwix.dequantize(lhs_q)
    dlhs = jax.grad(lambda l: jnp.sum(gmm_lib.grouped_matmul(l, rhs_q, group_sizes, backend=backend) ** 2))(lhs_f)
    ref_dlhs = jax.grad(lambda l: jnp.sum(_ref_grouped_matmul(l, rhs_deq, group_sizes) ** 2))(lhs_f)
    np.testing.assert_allclose(dlhs, ref_dlhs, rtol=1e-5, atol=1e-5)

  def test_manual_quantization_flag_with_qarray_operands(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(13))
    lhs_q = qpl.quantize(lhs, jnp.int8, channelwise_axes=[0])
    rhs_q = qpl.quantize(rhs, jnp.int8, channelwise_axes=[0, 2])
    out = gmm_lib.grouped_matmul(
        lhs_q,
        rhs_q,
        group_sizes,
        backend=gmm_lib.JaxRaggedDotBackend(is_tpu=False),
        use_manual_quantization=True,
    )
    ref = _ref_grouped_matmul(qwix.dequantize(lhs_q), qwix.dequantize(rhs_q), group_sizes)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  def test_native_qwix_interception_uses_direct_call_and_full_k_tiling(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(17), k=32, n=16)
    backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)
    with mock.patch.object(gmm_backends.JaxRaggedDotBackend, "call", wraps=backend.call) as call_spy:
      out = gmm_lib.grouped_matmul(
          lhs,
          rhs,
          group_sizes,
          backend=backend,
          tiling=(8, 8, 8, 8, 8, 8, 8, 8, 8),
          use_qwix_quantization=True,
          weight_gather_axes=[],  # Empty list from RoutedMoE must still allow native interception.
          is_quantized_config=True,
      )
      call_spy.assert_called_once()
      self.assertTrue(call_spy.call_args.kwargs["use_full_k_tiling"])
    np.testing.assert_allclose(out, _ref_grouped_matmul(lhs, rhs, group_sizes), rtol=1e-5, atol=1e-5)


class BackwardsCompatFacadeAndMockedBackendsTest(parameterized.TestCase):
  """Tests legacy `megablox.ops` facade, helper aliases, error guards, and mocked backends."""

  def test_helper_aliases_and_scale_helpers(self):
    # pylint: disable=protected-access
    self.assertIs(megablox_ops.TcUnsortCfg, gmm_quantize.TcUnsortCfg)
    self.assertIs(megablox_ops._fwd_prepare_lhs_scale, gmm_quantize.prepare_lhs_scale)
    self.assertIs(megablox_ops._check_tc_unsort_bwd_supported, gmm_quantize.check_tc_unsort_bwd_supported)
    self.assertIs(megablox_ops._lhs_rhs_share_fixed_scale, gmm_quantize.lhs_rhs_share_fixed_scale)

    fixed_rule = qwix.QtRule(
        act_qtype=jnp.float8_e4m3fn,
        act_calibration_method="fixed,-448.0,448.0",
        weight_qtype=jnp.float8_e4m3fn,
        weight_calibration_method="fixed,-448.0,448.0",
    )
    lhs_scale = megablox_ops._fwd_prepare_lhs_scale(fixed_rule)
    assert lhs_scale is not None
    self.assertEqual(lhs_scale.shape, (1, 1))
    dummy_q = qpl.QArray(
        qvalue=jnp.zeros((2, 2), dtype=jnp.float8_e4m3fn),
        scale=jnp.array(1.0, dtype=jnp.float32),
        zero_point=None,
        qtype=jnp.float8_e4m3fn,
    )
    self.assertTrue(megablox_ops._lhs_rhs_share_fixed_scale(fixed_rule, dummy_q, dummy_q))

    dyn_rule = qwix.QtRule(act_qtype=jnp.float8_e4m3fn, act_calibration_method="absmax")
    self.assertIsNone(megablox_ops._fwd_prepare_lhs_scale(dyn_rule))
    self.assertFalse(megablox_ops._lhs_rhs_share_fixed_scale(dyn_rule, dummy_q, dummy_q))
    # pylint: enable=protected-access

  def test_megablox_ops_gmm_routes_to_megablox_backend(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(21))
    tiling = (16, 16, 8, 8, 16, 16, 8, 8, 16)

    def fake_gmm(l, r, gs, preferred_element_type, t, group_offset, existing_out=None, **kwargs):
      del preferred_element_type, t, group_offset, existing_out
      if kwargs.get("transpose_rhs"):
        return _ref_grouped_matmul(l, jnp.swapaxes(r, 1, 2), gs)
      return _ref_grouped_matmul(l, r, gs)

    def fake_tgmm(l_t, drhs_dout, gs, preferred_element_type, t, group_offset, num_actual_groups, **kwargs):
      """Computes reference transposed grouped matmul for mocked calls."""
      del preferred_element_type, t, group_offset, num_actual_groups, kwargs
      l = l_t.swapaxes(0, 1)
      grads = []
      offset = 0
      for g in range(rhs.shape[0]):
        size = int(gs[g])
        if size > 0:
          grads.append(l[offset : offset + size].T @ drhs_dout[offset : offset + size])
          offset += size
        else:
          grads.append(jnp.zeros((l.shape[1], drhs_dout.shape[1]), dtype=jnp.float32))
      return jnp.stack(grads, axis=0)

    with (
        mock.patch.object(megablox_backend, "gmm", side_effect=fake_gmm) as mock_gmm,
        mock.patch.object(megablox_backend, "tgmm", side_effect=fake_tgmm) as mock_tgmm,
    ):
      loss, (dlhs, drhs) = jax.value_and_grad(
          lambda l, r: jnp.sum(
              megablox_ops.gmm(
                  l,
                  r,
                  group_sizes,
                  tiling=tiling,
                  use_tokamax_backend=False,
              )
              ** 2
          ),
          argnums=(0, 1),
      )(lhs, rhs)

    self.assertEqual(mock_gmm.call_count, 2)  # forward + dlhs
    self.assertEqual(mock_gmm.call_args_list[0].args[4], (16, 16, 8))
    self.assertEqual(mock_gmm.call_args_list[1].args[4], (8, 16, 16))
    mock_tgmm.assert_called_once()
    self.assertEqual(mock_tgmm.call_args.args[4], (8, 8, 16))
    self.assertTrue(jnp.isfinite(loss))
    self.assertEqual(dlhs.shape, lhs.shape)
    self.assertEqual(drhs.shape, rhs.shape)

  def test_megablox_ops_gmm_routes_to_tokamax_v2_and_dlhs_alias(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(23))
    fake_fwd_ps = jnp.ones((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32) * 3.0
    fake_bwd_ps = jnp.ones_like(rhs)

    def fake_v2_gmm_ret(lhs, rhs, group_sizes, **kwargs):
      """Computes reference v2 grouped matmul with optional partial_sum."""
      l_f = qwix.dequantize(lhs) if isinstance(lhs, qpl.QArray) else lhs
      r_f = qwix.dequantize(rhs) if isinstance(rhs, qpl.QArray) else rhs
      if kwargs.get("transpose_rhs"):
        res = _ref_grouped_matmul(l_f, jnp.swapaxes(r_f, 1, 2), group_sizes)
      else:
        res = _ref_grouped_matmul(l_f, r_f, group_sizes)
      if kwargs.get("partial_sum") is not None:
        res = res + kwargs["partial_sum"]
      return res

    expected_drhs_shape = rhs.shape

    def fake_v2_tgmm(*, lhs, rhs, group_sizes, **kwargs):
      del lhs, rhs, group_sizes
      if kwargs.get("partial_sum") is not None:
        return kwargs["partial_sum"] * 2.0
      return jnp.ones(expected_drhs_shape, dtype=jnp.float32)

    with (
        mock.patch.object(gmm_v2_kernel, "gmm_v2", side_effect=fake_v2_gmm_ret) as mock_v2_gmm,
        mock.patch.object(tgmm_v2_kernel, "tgmm_v2", side_effect=fake_v2_tgmm) as mock_v2_tgmm,
    ):

      def loss_fn(l, r, fwd_ps):
        """Computes mocked tokamax v2 forward and partial_sum loss."""
        out_val, rhs_pass = megablox_ops.gmm(
            l,
            r,
            group_sizes,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_gmm_v2_heuristic_tiling=True,
            use_dlhs_transpose_rhs=True,
            partial_sum=fwd_ps,
            return_rhs=True,
        )
        return jnp.sum(out_val) + jnp.sum(rhs_pass * fake_bwd_ps)

      out, (dlhs, drhs, d_fwd_ps) = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(lhs, rhs, fake_fwd_ps)
      # Legacy alias _dlhs_run_tokamax_v2 still forwards cleanly.
      dlhs_alias, applied = megablox_ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
          jnp.ones((lhs.shape[0], rhs.shape[2])),
          rhs,
          group_sizes,
          None,
          jnp.float32,
          (128,) * 9,
          use_gmm_v2_heuristic_tiling=True,
          use_dlhs_transpose_rhs=True,
      )

    self.assertGreaterEqual(mock_v2_gmm.call_count, 2)
    mock_v2_tgmm.assert_called_once()
    self.assertTrue(jnp.isfinite(out))
    self.assertEqual(dlhs.shape, lhs.shape)
    np.testing.assert_allclose(d_fwd_ps, jnp.ones_like(fake_fwd_ps))
    np.testing.assert_allclose(drhs, fake_bwd_ps * 2.0)
    self.assertFalse(applied)
    self.assertEqual(dlhs_alias.shape, lhs.shape)

  def test_tokamax_v1_backend_native_and_quantized_dispatch_with_mocked_module(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(29))
    calls = []

    class FakeGroupSizes:

      def __init__(self, gs, m):
        self.group_sizes = gs
        self.m = m

    def fake_ragged_dot(*, lhs, rhs, group_sizes, precision, preferred_element_type, implementation, group_offset, **kw):
      del precision, preferred_element_type, group_offset
      calls.append((implementation, type(group_sizes).__name__, kw))
      gs = group_sizes.group_sizes if isinstance(group_sizes, FakeGroupSizes) else group_sizes
      l_f = qwix.dequantize(lhs) if isinstance(lhs, qpl.QArray) else lhs
      r_f = qwix.dequantize(rhs) if isinstance(rhs, qpl.QArray) else rhs
      return _ref_grouped_matmul(l_f, r_f, gs)

    fake_tokamax = types.ModuleType("tokamax")
    fake_tokamax.RaggedDotGroupSizes = FakeGroupSizes  # type: ignore[attr-defined]
    fake_tokamax.ragged_dot = fake_ragged_dot  # type: ignore[attr-defined]

    backend = gmm_lib.TokamaxV1Backend()
    with mock.patch.dict(sys.modules, {"tokamax": fake_tokamax}):
      # Unquantized native call uses RaggedDotGroupSizes wrapper.
      out_native = gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=backend)
      # Quantized call routes through custom VJP + tokamax.ragged_dot.
      rule = qwix.QtRule(weight_qtype=jnp.int8, act_qtype=jnp.int8)
      out_quant = gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=backend, qwix_rule=rule)

    self.assertEqual(calls[0][:2], ("mosaic", "FakeGroupSizes"))
    self.assertEqual(calls[1][:2], ("mosaic", "ArrayImpl"))
    np.testing.assert_allclose(out_native, _ref_grouped_matmul(lhs, rhs, group_sizes), rtol=1e-5, atol=1e-5)
    self.assertEqual(out_quant.shape, (lhs.shape[0], rhs.shape[2]))

  def test_te_grouped_gemm_backend_dispatch_bias_partial_sum_and_group_offset(self):
    lhs, rhs, _ = _make_sparse_inputs(
        jax.random.PRNGKey(31),
        num_groups=2,
        group_sizes_list=(2, 3),
        m_pad=10,
        k=8,
        n=6,
        dtype=jnp.bfloat16,
    )
    bias = jnp.ones((2, 6), dtype=jnp.float32) * 0.25
    partial_sum = jnp.ones((10, 6), dtype=jnp.float32) * 2.0
    full_group_sizes = jnp.array([2, 3, 2, 1], dtype=jnp.int32)
    recorded = {}

    def fake_grouped_dense(l, r, gs, *, contracting_dims, bias=None, preferred_element_type=None):
      """Records call arguments and computes reference grouped matmul."""
      recorded["lhs_dtype"] = l.dtype
      recorded["gs"] = gs
      recorded["contracting_dims"] = contracting_dims
      recorded["bias_dtype"] = bias.dtype if bias is not None else None
      recorded["pref_dtype"] = preferred_element_type
      out = _ref_grouped_matmul(l, r, gs)
      if bias is not None:
        offset = 0
        for g in range(r.shape[0]):
          sz = int(gs[g])
          if sz > 0:
            out = out.at[offset : offset + sz].add(bias[g].astype(jnp.float32))
            offset += sz
      return out.astype(preferred_element_type)

    fake_te = types.ModuleType("transformer_engine")
    fake_te_jax = types.ModuleType("transformer_engine.jax")
    fake_te_dense = types.ModuleType("transformer_engine.jax.dense")
    fake_te_dense.grouped_dense = fake_grouped_dense  # type: ignore[attr-defined]
    fake_te_jax.dense = fake_te_dense  # type: ignore[attr-defined]
    fake_te.jax = fake_te_jax  # type: ignore[attr-defined]

    te_backend = gmm_lib.TeGroupedGemmBackend()
    with mock.patch.dict(
        sys.modules,
        {
            "transformer_engine": fake_te,
            "transformer_engine.jax": fake_te_jax,
            "transformer_engine.jax.dense": fake_te_dense,
        },
    ):
      out = gmm_lib.grouped_matmul(
          lhs,
          rhs,
          full_group_sizes,
          backend=te_backend,
          preferred_element_type=jnp.float32,
          group_offset=jnp.array(1, dtype=jnp.int32),
          bias=bias,
          partial_sum=partial_sum,
      )

    self.assertEqual(recorded["contracting_dims"], ((1,), (1,)))
    self.assertEqual(recorded["lhs_dtype"], jnp.bfloat16)
    self.assertEqual(recorded["bias_dtype"], jnp.bfloat16)
    self.assertEqual(recorded["pref_dtype"], jnp.bfloat16)
    np.testing.assert_array_equal(recorded["gs"], jnp.array([3, 2], dtype=jnp.int32))
    self.assertEqual(out.dtype, jnp.float32)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_capability_guardrails_reject_unsupported_backend_options(self):
    lhs, rhs, group_sizes = _make_sparse_inputs(jax.random.PRNGKey(37))
    jax_backend = gmm_lib.JaxRaggedDotBackend(is_tpu=False)
    te_backend = gmm_lib.TeGroupedGemmBackend()

    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, bias=jnp.zeros((4, 8)))
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, out_is_3d=True)
    with self.assertRaises(ValueError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, return_lhs=True)
    with self.assertRaises(ValueError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, return_rhs=True)
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, partial_sum=jnp.zeros_like(rhs))
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=jax_backend, flat_fsdp_shard=(1, 1, 1))
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(
          lhs,
          rhs,
          group_sizes,
          backend=gmm_lib.MegabloxBackend(),
          flat_fsdp_shard=(1, 1, 1),
      )
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=gmm_lib.TokamaxV1Backend(), flat_fsdp_shard=(1, 1, 1))
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=te_backend, return_rhs=True)
    with self.assertRaises(NotImplementedError):
      gmm_lib.grouped_matmul(lhs, rhs, group_sizes, backend=te_backend, weight_gather_axes=[("fsdp", 0)])


class RoutedMoeDecoupledDispatchTest(parameterized.TestCase):
  """Tests `RoutedMoE.sparse_matmul` using decoupled `kernels.gmm` on CPU."""

  def _make_routed_moe(
      self,
      *,
      sparse_matmul: bool,
      quantization: str = "",
      quant_mode: str = "train",
      use_qwix_quantization: bool = False,
      use_megablox: bool = False,
      use_tokamax_gmm: bool = False,
      moe_flat_fsdp_weights: bool = False,
      prefetch_fsdp_weights: bool = False,
      single_device_mesh: bool = False,
  ):
    """Constructs a configured RoutedMoE module and sample input tensor."""
    cfg = pyconfig.initialize(
        [None, test_helpers.get_test_config_path()],
        run_name="gmm_dispatch_moe_test",
        override_model_config=True,
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        hardware="cpu",
        model_name="mixtral-8x7b",
        dtype="float32",
        weight_dtype="float32",
        sparse_matmul=sparse_matmul,
        megablox=use_megablox,
        use_tokamax_gmm=use_tokamax_gmm,
        moe_flat_fsdp_weights=moe_flat_fsdp_weights,
        prefetch_fsdp_weights=prefetch_fsdp_weights,
        quantization=quantization,
        use_qwix_quantization=use_qwix_quantization,
        num_experts=4,
        num_experts_per_tok=2,
        base_emb_dim=32,
        base_mlp_dim=64,
        base_moe_mlp_dim=64,
        max_target_length=4,
        per_device_batch_size=1,
    )
    rngs = nnx.Rngs(params=0)
    if single_device_mesh:
      devices_array = np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes))
    else:
      devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    quant = quantizations.configure_quantization(cfg, quant_mode)
    layer = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=quant,
        rngs=rngs,
    )
    return cfg, mesh, layer

  def test_sparse_matmul_matches_dense_einsum_fwd_and_bwd(self):
    _, _, dense_moe = self._make_routed_moe(sparse_matmul=False)
    _, _, sparse_moe = self._make_routed_moe(sparse_matmul=True)

    # Copy exact weights so both paths evaluate identical parameters.
    nnx.update(sparse_moe, nnx.state(dense_moe))
    x = jax.random.normal(jax.random.PRNGKey(99), (1, 4, 32), dtype=jnp.float32)

    def fwd_loss(model, inputs):
      out = model(inputs)[0]
      return jnp.sum(out**2)

    with (
        mock.patch.object(gmm_lib, "select_gmm_backend", wraps=gmm_lib.select_gmm_backend) as select_spy,
        mock.patch.object(gmm_lib, "grouped_matmul", wraps=gmm_lib.grouped_matmul) as gmm_spy,
    ):
      sparse_out = sparse_moe(x)[0]
      dense_out = dense_moe(x)[0]
      sparse_loss, sparse_grads = nnx.value_and_grad(fwd_loss, argnums=(0, 1))(sparse_moe, x)
      dense_loss, dense_grads = nnx.value_and_grad(fwd_loss, argnums=(0, 1))(dense_moe, x)

    # Backend is selected cleanly once per sparse_matmul invocation while grouped_matmul runs for w0, w1, wo.
    self.assertGreaterEqual(select_spy.call_count, 1)
    self.assertEqual(gmm_spy.call_count, select_spy.call_count * 3)
    for call in gmm_spy.call_args_list:
      self.assertIsInstance(call.kwargs["backend"], gmm_lib.JaxRaggedDotBackend)

    np.testing.assert_allclose(sparse_out, dense_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sparse_loss, dense_loss, rtol=1e-4, atol=1e-4)
    for g_s, g_d in zip(jax.tree.leaves(sparse_grads), jax.tree.leaves(dense_grads)):
      np.testing.assert_allclose(g_s, g_d, rtol=1e-4, atol=1e-4)

  def test_sparse_matmul_with_qwix_provider_composes_same_backend_selector(self):
    _, _, base_moe = self._make_routed_moe(sparse_matmul=True)
    x = jax.random.normal(jax.random.PRNGKey(101), (1, 4, 32), dtype=jnp.float32)

    rules = [
        qwix.QtRule(
            module_path=".*",
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            op_names=("gmm", "ragged_dot"),
        )
    ]
    q_moe = qwix.quantize_model(base_moe, qwix.QtProvider(rules), x)
    base_out = base_moe(x)[0]

    with (
        mock.patch.object(gmm_lib, "select_gmm_backend", wraps=gmm_lib.select_gmm_backend) as select_spy,
        mock.patch.object(gmm_lib, "grouped_matmul", wraps=gmm_lib.grouped_matmul) as gmm_spy,
    ):
      q_out = q_moe(x)[0]

    self.assertEqual(select_spy.call_count, 1)
    self.assertEqual(gmm_spy.call_count, 3)
    self.assertTrue(jnp.all(jnp.isfinite(q_out)))
    np.testing.assert_allclose(q_out, base_out, rtol=0.15, atol=0.15)

  def test_aqt_qtensor_serve_mode_avoids_double_quantization(self):
    _, _, serve_moe = self._make_routed_moe(sparse_matmul=True, quantization="int8", quant_mode="serve")
    _, _, ref_moe = self._make_routed_moe(sparse_matmul=True, quantization="", quant_mode="train")

    def make_qtensor(key, shape):
      """Builds a sample per-channel int8 QTensor for inference testing."""
      k1, k2 = jax.random.split(key)
      qval = jax.random.randint(k1, shape, minval=-12, maxval=12, dtype=jnp.int8)
      scale = jax.random.uniform(k2, (shape[0], 1, shape[2]), minval=0.05, maxval=0.25, dtype=jnp.float32)
      # `utils.flax_slots_kw_only_dataclass` obscures keyword args from static Pyrefly signatures.
      qtensor_cls: Any = aqt_tensor.QTensor
      return qtensor_cls(
          qvalue=qval,
          scale=[scale],
          scale_t=None,
          bias=[],
          dequant_dtype=jnp.float32,
      )

    w0_qt = make_qtensor(jax.random.PRNGKey(201), (4, 32, 64))
    w1_qt = make_qtensor(jax.random.PRNGKey(202), (4, 32, 64))
    wo_qt = make_qtensor(jax.random.PRNGKey(203), (4, 64, 32))

    x = jax.random.normal(jax.random.PRNGKey(204), (1, 4, 32), dtype=jnp.float32)
    gate_logits = jax.random.normal(jax.random.PRNGKey(205), (1, 4, 4), dtype=jnp.float32)
    ref_out, _, _ = ref_moe.sparse_matmul(
        x, gate_logits, None, w0_qt.dequant(), w1_qt.dequant(), wo_qt.dequant(), None, None, None
    )

    with mock.patch.object(gmm_lib, "grouped_matmul", wraps=gmm_lib.grouped_matmul) as gmm_spy:
      serve_out, _, _ = serve_moe.sparse_matmul(x, gate_logits, None, w0_qt, w1_qt, wo_qt, None, None, None)

    self.assertEqual(gmm_spy.call_count, 3)
    for call in gmm_spy.call_args_list:
      self.assertIsNone(call.kwargs["lhs_quantize_dtype"])
      self.assertIsNone(call.kwargs["rhs_quantize_dtype"])
      self.assertEqual(call.kwargs["rhs"].dtype, jnp.int8)
      self.assertTrue(call.kwargs["is_quantized_config"])

    np.testing.assert_allclose(serve_out, ref_out, rtol=1e-4, atol=1e-4)

  def test_sparse_matmul_flat_fsdp_weights_no_gather_axes_supports_non_flat_backends(self):
    _, _, ref_moe = self._make_routed_moe(
        sparse_matmul=True,
        use_megablox=False,
        use_tokamax_gmm=False,
        moe_flat_fsdp_weights=False,
        single_device_mesh=True,
    )
    _, _, ragged_flat_moe = self._make_routed_moe(
        sparse_matmul=True,
        use_megablox=False,
        use_tokamax_gmm=False,
        moe_flat_fsdp_weights=True,
        prefetch_fsdp_weights=True,
        single_device_mesh=True,
    )
    _, _, megablox_flat_moe = self._make_routed_moe(
        sparse_matmul=True,
        use_megablox=True,
        use_tokamax_gmm=False,
        moe_flat_fsdp_weights=True,
        prefetch_fsdp_weights=True,
        single_device_mesh=True,
    )

    w0_3d = jax.random.normal(jax.random.PRNGKey(301), (4, 32, 64), dtype=jnp.float32) * 0.1
    w1_3d = jax.random.normal(jax.random.PRNGKey(302), (4, 32, 64), dtype=jnp.float32) * 0.1
    wo_3d = jax.random.normal(jax.random.PRNGKey(303), (4, 64, 32), dtype=jnp.float32) * 0.1
    w0_flat = w0_3d.reshape(-1, w0_3d.shape[-1])
    w1_flat = w1_3d.reshape(-1, w1_3d.shape[-1])
    wo_flat = wo_3d.reshape(-1, wo_3d.shape[-1])

    x = jax.random.normal(jax.random.PRNGKey(304), (1, 4, 32), dtype=jnp.float32)
    gate_logits = jax.random.normal(jax.random.PRNGKey(305), (1, 4, 4), dtype=jnp.float32)

    ref_out, _, _ = ref_moe.sparse_matmul(x, gate_logits, None, w0_3d, w1_3d, wo_3d, None, None, None)

    with mock.patch.object(gmm_lib, "grouped_matmul", wraps=gmm_lib.grouped_matmul) as ragged_spy:
      ragged_flat_out, _, _ = ragged_flat_moe.sparse_matmul(
          x, gate_logits, None, w0_flat, w1_flat, wo_flat, None, None, None
      )

    self.assertEqual(ragged_spy.call_count, 3)
    for call, w_3d in zip(ragged_spy.call_args_list, (w0_3d, w1_3d, wo_3d)):
      self.assertIsInstance(call.kwargs["backend"], gmm_lib.JaxRaggedDotBackend)
      self.assertIsNone(call.kwargs["flat_fsdp_shard"])
      self.assertEqual(call.kwargs["rhs"].shape, w_3d.shape)
    np.testing.assert_allclose(ragged_flat_out, ref_out, rtol=1e-4, atol=1e-4)

    def fake_megablox_gmm(
        lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset, existing_out=None, **kwargs
    ):
      del tiling, group_offset, existing_out
      self.assertEqual(rhs.ndim, 3)
      effective_rhs = jnp.swapaxes(rhs, 1, 2) if kwargs.get("transpose_rhs") else rhs
      return jax.lax.ragged_dot(lhs, effective_rhs, group_sizes, preferred_element_type=preferred_element_type)

    with (
        mock.patch.object(gmm_lib, "select_gmm_backend", return_value=gmm_lib.MegabloxBackend()),
        mock.patch.object(megablox_backend, "gmm", side_effect=fake_megablox_gmm),
        mock.patch.object(gmm_lib, "grouped_matmul", wraps=gmm_lib.grouped_matmul) as mblx_spy,
    ):
      mblx_flat_out, _, _ = megablox_flat_moe.sparse_matmul(
          x, gate_logits, None, w0_flat, w1_flat, wo_flat, None, None, None
      )

    self.assertEqual(mblx_spy.call_count, 3)
    for call, w_3d in zip(mblx_spy.call_args_list, (w0_3d, w1_3d, wo_3d)):
      self.assertIsInstance(call.kwargs["backend"], gmm_lib.MegabloxBackend)
      self.assertIsNone(call.kwargs["flat_fsdp_shard"])
      self.assertEqual(call.kwargs["rhs"].shape, w_3d.shape)
    np.testing.assert_allclose(mblx_flat_out, ref_out, rtol=1e-4, atol=1e-4)

  def test_sparse_matmul_accepts_minimal_stub_config_and_mesh(self):
    cfg, mesh, ref_moe = self._make_routed_moe(sparse_matmul=True, single_device_mesh=True)
    _, _, stub_moe = self._make_routed_moe(sparse_matmul=True, single_device_mesh=True)
    nnx.update(stub_moe, nnx.state(ref_moe))

    # Intentionally omit kernel/quant/attention flags and `mesh.devices` as lightweight unit stubs do.
    omitted_attrs = {
        "megablox",
        "use_tokamax_gmm",
        "use_tokamax_gmm_v2",
        "use_gmm_v2",
        "moe_megablox_use_v2",
        "moe_quantize_token_all_gather",
        "moe_flat_fsdp_weights",
        "use_qwix_quantization",
        "attention",
        "quantization",
        "hardware",
    }
    stub_moe.config = types.SimpleNamespace(**{k: v for k, v in cfg.get_keys().items() if k not in omitted_attrs})

    x = jax.random.normal(jax.random.PRNGKey(306), (1, 4, 32), dtype=jnp.float32)
    gate_logits = jax.random.normal(jax.random.PRNGKey(307), (1, 4, 4), dtype=jnp.float32)
    w0 = jax.random.normal(jax.random.PRNGKey(308), (4, 32, 64), dtype=jnp.float32) * 0.1
    w1 = jax.random.normal(jax.random.PRNGKey(309), (4, 32, 64), dtype=jnp.float32) * 0.1
    wo = jax.random.normal(jax.random.PRNGKey(310), (4, 64, 32), dtype=jnp.float32) * 0.1

    # Verify tracing with an AbstractMesh (which disallows `.devices`) and concrete execution with stubbed config.
    stub_moe.mesh = jax.sharding.AbstractMesh(tuple(mesh.shape.values()), mesh.axis_names)
    eval_out, _, _ = nnx.eval_shape(
        lambda m: m.sparse_matmul(x, gate_logits, None, w0, w1, wo, None, None, None),
        stub_moe,
    )
    self.assertEqual(eval_out.shape, x.shape)

    stub_moe.mesh = mesh
    stub_out, _, _ = stub_moe.sparse_matmul(x, gate_logits, None, w0, w1, wo, None, None, None)
    ref_out, _, _ = ref_moe.sparse_matmul(x, gate_logits, None, w0, w1, wo, None, None, None)
    np.testing.assert_allclose(stub_out, ref_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
