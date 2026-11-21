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

"""Unit tests for `muse_spark_fp8_moe`, the FP8 expert matmuls of `MuseSparkMoE`."""

from __future__ import annotations

import functools
import types

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import DecoderBlockType
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.layers import muse_spark_fp8_moe as fp8_moe
from maxtext.layers import quantizations
import numpy as np

_DEFAULT_TILING = (512, 1024, 1024, 512, 1024, 1024, 512, 1024, 1024)


def _sqnr_db(ref: np.ndarray, cand: np.ndarray) -> float:
  ref_f64 = np.asarray(ref, dtype=np.float64)
  cand_f64 = np.asarray(cand, dtype=np.float64)
  sig = float(np.sum(np.square(ref_f64)))
  err = float(np.sum(np.square(ref_f64 - cand_f64)))
  if err == 0.0:
    return float("inf")
  return 10.0 * float(np.log10(sig / max(err, 1e-30)))


def _bf16_ref_gmm(lhs: jax.Array, rhs_full: jax.Array, grp: jax.Array, is_tpu: bool) -> jax.Array:
  """BF16 reference grouped matmul (`gmm_v2` / `tgmm_v2` on TPU, where XLA `ragged_dot` runs out of VMEM)."""
  if not is_tpu:
    return jax.lax.ragged_dot(lhs, rhs_full, grp, precision=jax.lax.Precision.HIGHEST).astype(jnp.bfloat16)

  g_num, k_dim, n_dim = rhs_full.shape
  t_fwd = gmm_v2.TileSizes(256, min(4096, k_dim), min(2048, n_dim))
  t_dlhs = gmm_v2.TileSizes(256, min(4096, n_dim), min(2048, k_dim))
  t_drhs = gmm_v2.TileSizes(256, min(2048, k_dim), min(2048, n_dim))

  @jax.custom_vjp
  def _op(l_in, r_in):
    return gmm_v2.gmm_v2(l_in, r_in, grp, tile_info=t_fwd, preferred_element_type=jnp.bfloat16)

  def _fwd(l_in, r_in):
    return _op(l_in, r_in), (l_in, r_in)

  def _bwd(res, g_in):
    l_in, r_in = res
    dl = gmm_v2.gmm_v2(g_in, r_in, grp, tile_info=t_dlhs, preferred_element_type=jnp.bfloat16, transpose_rhs=True)
    dr = tgmm_v2.tgmm_v2(
        l_in, g_in, grp, num_actual_groups=g_num, tile_info=t_drhs, preferred_element_type=jnp.bfloat16
    )
    return dl, dr

  _op.defvjp(_fwd, _bwd)
  return _op(lhs, rhs_full)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
  a_f64 = np.asarray(a, dtype=np.float64).ravel()
  b_f64 = np.asarray(b, dtype=np.float64).ravel()
  denom = float(np.linalg.norm(a_f64) * np.linalg.norm(b_f64))
  if denom == 0.0:
    return 1.0
  return float(np.dot(a_f64, b_f64) / denom)


class MuseSparkFp8MoeTest(parameterized.TestCase):

  def test_config_gating_and_qwix_rule_exclusions(self):
    base_stub = types.SimpleNamespace(
        decoder_block=DecoderBlockType.MUSE_SPARK,
        quantization="fp8_full",
        weight_quantization_calibration_method="absmax",
        act_quantization_calibration_method="absmax",
        bwd_quantization_calibration_method="absmax",
        drhs_grad_quantization_calibration_method="",
        weight_sparsity_n=0,
        weight_sparsity_m=0,
        quantize_logits_proj=False,
        quantize_mtp=False,
        sa_qproj_bwd_bf16=False,
    )
    self.assertTrue(fp8_moe.is_fp8_moe_active(base_stub))
    bf16_stub = types.SimpleNamespace(**{**base_stub.__dict__, "quantization": ""})
    self.assertFalse(fp8_moe.is_fp8_moe_active(bf16_stub))

    rules = quantizations.get_quantization_rule(base_stub)
    self.assertNotEmpty(rules)
    # The layer rule quantizes the grouped matmuls; the router exclusion rule only covers dot_general/einsum.
    self.assertTrue(any("gmm" in (r.op_names or ()) and r.weight_qtype == jnp.float8_e4m3fn for r in rules))
    self.assertTrue(any(r.weight_qtype is None and "gmm" not in (r.op_names or ()) for r in rules))
    # The unquantized rules come first: Qwix uses the first rule that matches.
    self.assertIsNone(rules[0].weight_qtype)
    self.assertIsNotNone(rules[-1].weight_qtype)

  def test_tile_selection(self):
    # `tm=512` needs at least 1792 active rows per expert; smaller buffers use `tm=256`.
    for size_m in (65536, 98304):
      self.assertEqual(
          fp8_moe._fp8_clamped_gmm_tiles(512, 1024, 1024, size_m, 32, 4096, 4096),
          gmm_v2.TileSizes(512, 4096, 2048),
      )
      self.assertEqual(
          fp8_moe._fp8_clamped_tgmm_tiles(512, 1024, 1024, size_m, 32, 4096, 4096),
          gmm_v2.TileSizes(512, 2048, 2048),
      )
    for size_m in (8192, 49152):
      self.assertEqual(
          fp8_moe._fp8_clamped_gmm_tiles(512, 1024, 1024, size_m, 32, 4096, 4096),
          gmm_v2.TileSizes(256, 4096, 2048),
      )
      self.assertEqual(
          fp8_moe._fp8_clamped_tgmm_tiles(512, 1024, 1024, size_m, 32, 4096, 4096),
          gmm_v2.TileSizes(256, 2048, 2048),
      )
    # A half-filled buffer (the overflow loop's capacity is twice the expected load) is sized by its active rows.
    self.assertEqual(
        fp8_moe._fp8_clamped_gmm_tiles(512, 1024, 1024, 65536, 32, 4096, 4096, expected_active_rows=32768),
        gmm_v2.TileSizes(256, 4096, 2048),
    )
    self.assertEqual(
        fp8_moe._fp8_clamped_tgmm_tiles(512, 1024, 1024, 131072, 32, 4096, 4096, expected_active_rows=65536),
        gmm_v2.TileSizes(512, 2048, 2048),
    )
    # Explicit tiles are kept, except that `tn=4096` is clamped to 2048.
    self.assertEqual(
        fp8_moe._fp8_clamped_gmm_tiles(512, 1024, 4096, 65536, 32, 4096, 4096),
        gmm_v2.TileSizes(512, 1024, 2048),
    )

  @parameterized.named_parameters(
      ("wo_fsdp_axis1_true", True),
      ("wo_fsdp_axis1_false", False),
  )
  def test_quantize_before_fsdp_all_gather_is_exact(self, wo_fsdp_axis1: bool):
    """Quantizing each FSDP weight shard and then gathering must equal gathering BF16 first."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices; got {len(devices)}")
    is_tpu = devices[0].platform == "tpu"
    fsdp_size = 2
    ep_size = len(devices) // fsdp_size
    mesh = Mesh(np.array(devices).reshape(fsdp_size, ep_size), ("fsdp", "ep"))

    g_local, k_dim, n_dim, m_rows = 4, 512, 512, 256
    key = jax.random.PRNGKey(101 if wo_fsdp_axis1 else 202)
    k0, k1, k2, kx = jax.random.split(key, 4)

    wi0_global = (jax.random.normal(k0, (ep_size * g_local, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(
        jnp.bfloat16
    )
    wi1_global = (jax.random.normal(k1, (ep_size * g_local, k_dim, n_dim), dtype=jnp.float32) * 0.015).astype(
        jnp.bfloat16
    )
    wo_global = (jax.random.normal(k2, (ep_size * g_local, n_dim, k_dim), dtype=jnp.float32) * 0.004).astype(
        jnp.bfloat16
    )
    lhs_global = jax.random.normal(kx, (ep_size, m_rows, k_dim), dtype=jnp.bfloat16)
    grp_single = jnp.array([64, 64, 64, 64], dtype=jnp.int32)
    grp_global = jnp.broadcast_to(grp_single[None, :], (ep_size, g_local))

    wo_pspec = P("ep", "fsdp", None) if wo_fsdp_axis1 else P("ep", None, "fsdp")
    wo_gather_axis = 1 if wo_fsdp_axis1 else 2

    @functools.partial(
        jax.jit,
        out_shardings=jax.sharding.NamedSharding(mesh, P()),
    )
    def run_comparison(wi0_in, wi1_in, wo_in, lhs_in, grp_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(
              P("ep", "fsdp", None),
              P("ep", "fsdp", None),
              wo_pspec,
              P("ep", None, None),
              P("ep", None),
          ),
          out_specs=P(),
          check_vma=False,
      )
      def _per_shard(wi0_s, wi1_s, wo_s, x_s, grp_s):
        x_2d = x_s.squeeze(0)
        grp_1d = grp_s.squeeze(0)
        s0, s1, so = fp8_moe.precompute_expert_weight_scales(
            (wi0_s, wi1_s, wo_s), (1, 1, wo_gather_axis), "fsdp"
        )
        # Post-AG reference: gather BF16 first, then quantize locally.
        wi0_bf16_full = jax.lax.all_gather(wi0_s, "fsdp", axis=1, tiled=True)
        wo_bf16_full = jax.lax.all_gather(wo_s, "fsdp", axis=wo_gather_axis, tiled=True)
        (s0_post, so_post) = fp8_moe.precompute_expert_weight_scales(
            (wi0_bf16_full, wo_bf16_full), (1, 1), None
        )
        out_pre = fp8_moe.ragged_dot_fp8(
            x_2d,
            wi0_s,
            s0,
            grp_1d,
            fsdp_axis="fsdp",
            gather_axis=1,
            tiling=_DEFAULT_TILING,
            use_gmm_v2=is_tpu,
        )
        out_post = fp8_moe.ragged_dot_fp8(
            x_2d,
            wi0_bf16_full,
            s0_post,
            grp_1d,
            fsdp_axis=None,
            gather_axis=1,
            tiling=_DEFAULT_TILING,
            use_gmm_v2=is_tpu,
        )

        wo_pre = fp8_moe.ragged_dot_fp8(
            x_2d,
            wo_s,
            so,
            grp_1d,
            fsdp_axis="fsdp",
            gather_axis=wo_gather_axis,
            tiling=_DEFAULT_TILING,
            use_gmm_v2=is_tpu,
        )
        wo_post = fp8_moe.ragged_dot_fp8(
            x_2d,
            wo_bf16_full,
            so_post,
            grp_1d,
            fsdp_axis=None,
            gather_axis=1,
            tiling=_DEFAULT_TILING,
            use_gmm_v2=is_tpu,
        )
        s0_diff = jnp.max(jnp.abs(s0 - s0_post))
        out_diff = jnp.max(jnp.abs(out_pre.astype(jnp.float32) - out_post.astype(jnp.float32)))
        wo_diff = jnp.max(jnp.abs(wo_pre.astype(jnp.float32) - wo_post.astype(jnp.float32)))
        return jax.lax.pmax(jnp.stack([s0_diff, out_diff, wo_diff]), axis_name=("fsdp", "ep"))

      return _per_shard(wi0_in, wi1_in, wo_in, lhs_in, grp_in)

    diffs = np.asarray(run_comparison(wi0_global, wi1_global, wo_global, lhs_global, grp_global))
    np.testing.assert_array_equal(diffs, np.zeros_like(diffs))

    # The weights must cross the FSDP all-gather as `f8e4m3fn`.
    @jax.jit
    def qag_only(wi0_in, lhs_in, grp_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P("ep", "fsdp", None), P("ep", None, None), P("ep", None)),
          out_specs=P("ep", None),
          check_vma=False,
      )
      def _fn(w_s, x_s, g_s):
        (sc,) = fp8_moe.precompute_expert_weight_scales((w_s,), (1,), "fsdp")
        return fp8_moe.ragged_dot_fp8(
            x_s.squeeze(0), w_s, sc, g_s.squeeze(0), fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu
        )

      return _fn(wi0_in, lhs_in, grp_in)

    stablehlo_txt = qag_only.lower(wi0_global, lhs_global, grp_global).as_text()
    self.assertIn("all_gather", stablehlo_txt)
    self.assertIn("f8E4M3FN", stablehlo_txt)

  @parameterized.named_parameters(
      ("2d_tokens", False),
      ("3d_tokens", True),
  )
  def test_ep_token_all_gather(self, use_3d: bool):
    """The FP8 token all-gather must match a BF16 all-gather feeding the FP8 matmul."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices; got {len(devices)}")
    is_tpu = devices[0].platform == "tpu"
    ep_size = len(devices)
    mesh = Mesh(np.array(devices), ("ep",))

    t_local, k_dim, n_dim, g_local = 64, 512, 512, 4
    t_ep = t_local * ep_size
    key = jax.random.PRNGKey(303 if not use_3d else 304)
    kx, kw, kg = jax.random.split(key, 3)

    x_shape = (ep_size * t_local, k_dim // 128, 128) if use_3d else (ep_size * t_local, k_dim)
    x_global = (jax.random.normal(kx, x_shape, dtype=jnp.float32) * 1.7).astype(jnp.bfloat16)
    w_global = (jax.random.normal(kw, (ep_size * g_local, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    out_grad = jax.random.normal(kg, (t_ep, n_dim // 128, 128) if use_3d else (t_ep, n_dim), dtype=jnp.bfloat16)
    grp = jnp.full((g_local,), t_ep // g_local, dtype=jnp.int32)

    x_pspec = P("ep", None, None) if use_3d else P("ep", None)

    @jax.jit
    def run_l3(x_in, w_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(x_pspec, P("ep", None, None)),
          out_specs=(P(), x_pspec, x_pspec, x_pspec, x_pspec),
          check_vma=False,
      )
      def _fn(x_loc, w_loc):
        (sw,) = fp8_moe.precompute_expert_weight_scales((w_loc,), (1,), None)

        def loss_fp8_ag(xl):
          ep_x = fp8_moe.fp8_ep_token_all_gather(xl, "ep")
          y = fp8_moe.ragged_dot_fp8(
              ep_x, w_loc, sw, grp, fsdp_axis=None, gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, out_is_3d=use_3d
          )
          return jnp.sum(y.astype(jnp.float32) * out_grad.astype(jnp.float32)), y

        def loss_bf16_ag(xl):
          ep_x = jax.lax.all_gather(xl, "ep", axis=0, tiled=True)
          y = fp8_moe.ragged_dot_fp8(
              ep_x, w_loc, sw, grp, fsdp_axis=None, gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, out_is_3d=use_3d
          )
          return jnp.sum(y.astype(jnp.float32) * out_grad.astype(jnp.float32)), y

        (_, y_l3), dx_l3 = jax.value_and_grad(loss_fp8_ag, has_aux=True)(x_loc)
        (_, y_ref), dx_ref = jax.value_and_grad(loss_bf16_ag, has_aux=True)(x_loc)
        max_fwd_diff = jax.lax.pmax(
            jnp.max(jnp.abs(y_l3.astype(jnp.float32) - y_ref.astype(jnp.float32))), axis_name="ep"
        )
        return max_fwd_diff, y_l3[:t_local], y_ref[:t_local], dx_l3, dx_ref

      return _fn(x_in, w_in)

    max_fwd_diff, y_l3, y_ref, dx_l3, dx_ref = run_l3(x_global, w_global)
    if not is_tpu:
      self.assertEqual(float(max_fwd_diff), 0.0)
    else:
      # On TPU the kernel computes its block scale in bf16, so the two paths differ by quantization noise.
      self.assertGreaterEqual(_sqnr_db(np.asarray(y_ref), np.asarray(y_l3)), 33.0)
    dx_l3_np = np.asarray(dx_l3, dtype=np.float32)
    dx_ref_np = np.asarray(dx_ref, dtype=np.float32)
    self.assertGreater(float(np.linalg.norm(dx_l3_np)), 0.0)
    self.assertGreaterEqual(_sqnr_db(dx_ref_np, dx_l3_np), 29.0)

  @parameterized.named_parameters(
      ("2d_scale_1", False, 1.0),
      ("3d_scale_tiny", True, 1e-6),
  )
  def test_ep_combine_psum_scatter(self, use_3d: bool, grad_scale: float):
    """The combine is an exact `psum_scatter` forward; its backward all-gathers FP8 cotangents."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices; got {len(devices)}")
    ep_size = len(devices)
    mesh = Mesh(np.array(devices), ("ep",))

    t_local, d_dim = 32, 512
    t_ep = t_local * ep_size
    key = jax.random.PRNGKey(404)
    kx, kg = jax.random.split(key)
    ep_shape = (ep_size, t_ep, d_dim // 128, 128) if use_3d else (ep_size, t_ep, d_dim)
    loc_shape = (ep_size * t_local, d_dim // 128, 128) if use_3d else (ep_size * t_local, d_dim)
    combined_in = jax.random.normal(kx, ep_shape, dtype=jnp.bfloat16)
    cotangent = (jax.random.normal(kg, loc_shape, dtype=jnp.float32) * grad_scale).astype(jnp.bfloat16)

    in_spec = P("ep", None, None, None) if use_3d else P("ep", None, None)
    loc_spec = P("ep", None, None) if use_3d else P("ep", None)

    @jax.jit
    def run_l4(c_in, g_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(in_spec, loc_spec),
          out_specs=(P(), in_spec, in_spec),
          check_vma=False,
      )
      def _fn(c_ep, g_loc):
        c_s = c_ep.squeeze(0)
        y_l4, vjp_l4 = jax.vjp(lambda z: fp8_moe.fp8_ep_combine_psum_scatter(z, "ep"), c_s)
        y_ref, vjp_ref = jax.vjp(
            lambda z: jax.lax.psum_scatter(z, axis_name="ep", scatter_dimension=0, tiled=True), c_s
        )
        (dc_l4,) = vjp_l4(g_loc)
        (dc_ref,) = vjp_ref(g_loc)
        fwd_diff = jax.lax.pmax(jnp.max(jnp.abs(y_l4.astype(jnp.float32) - y_ref.astype(jnp.float32))), axis_name="ep")
        return fwd_diff, dc_l4[None, ...], dc_ref[None, ...]

      return _fn(c_in, g_in)

    fwd_diff, dc_l4, dc_ref = run_l4(combined_in, cotangent)
    self.assertEqual(float(fwd_diff), 0.0)
    self.assertGreaterEqual(_sqnr_db(np.asarray(dc_ref), np.asarray(dc_l4)), 30.5)

  @parameterized.named_parameters(
      ("both_2d", True, True, False),
      ("lhs_only_3d", True, False, True),
      ("rhs_only_2d", False, True, False),
  )
  def test_invalid_tail_rows_do_not_poison_results(self, need_lhs: bool, need_rhs: bool, use_3d: bool):
    """Rows past the valid count hold garbage in production; NaN/Inf there must not leak anywhere."""
    is_tpu = jax.devices()[0].platform == "tpu"
    g_num, m_cap, valid_count, k_dim, n_dim = 4, 512, 300, 256, 256
    grp = jnp.array([100, 80, 70, 50], dtype=jnp.int32)
    valid_mask = jnp.arange(m_cap) < valid_count

    key = jax.random.PRNGKey(505)
    kx, kw, kg = jax.random.split(key, 3)
    x_shape = (m_cap, k_dim // 128, 128) if use_3d else (m_cap, k_dim)
    g_shape = (m_cap, n_dim // 128, 128) if use_3d else (m_cap, n_dim)

    x_clean = jnp.where(
        valid_mask.reshape(m_cap, *([1] * (len(x_shape) - 1))),
        jax.random.normal(kx, x_shape, dtype=jnp.bfloat16),
        jnp.bfloat16(0.0),
    )
    g_clean = jnp.where(
        valid_mask.reshape(m_cap, *([1] * (len(g_shape) - 1))),
        jax.random.normal(kg, g_shape, dtype=jnp.bfloat16),
        jnp.bfloat16(0.0),
    )
    # Poison `[valid_count : m_cap]` with NaNs and Infs.
    poison_x = jnp.full(x_shape, jnp.bfloat16(jnp.nan))
    poison_g = jnp.full(g_shape, jnp.bfloat16(jnp.inf))
    x_poisoned = jnp.where(valid_mask.reshape(m_cap, *([1] * (len(x_shape) - 1))), x_clean, poison_x)
    g_poisoned = jnp.where(valid_mask.reshape(m_cap, *([1] * (len(g_shape) - 1))), g_clean, poison_g)

    w = (jax.random.normal(kw, (g_num, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    (sw,) = fp8_moe.precompute_expert_weight_scales((w,), (1,), None)

    def eval_fn(x_in, w_in, g_cot):
      def fwd(l_arg, r_arg):
        y = fp8_moe.ragged_dot_fp8(
            l_arg,
            r_arg,
            sw,
            grp,
            fsdp_axis=None,
            gather_axis=1,
            tiling=_DEFAULT_TILING,
            use_gmm_v2=is_tpu,
            valid_mask=valid_mask,
            out_is_3d=use_3d,
            need_lhs_grad=need_lhs,
            need_rhs_grad=need_rhs,
        )
        # The model masks the forward output's invalid rows the same way.
        y_valid = jnp.where(valid_mask.reshape(m_cap, *([1] * (y.ndim - 1))), y, jnp.bfloat16(0.0))
        return y_valid

      y_out, vjp_fn = jax.vjp(fwd, x_in, w_in)
      dl, dr = vjp_fn(g_cot)
      return y_out, dl, dr

    y_ref, dl_ref, dr_ref = jax.jit(eval_fn)(x_clean, w, g_clean)
    y_pos, dl_pos, dr_pos = jax.jit(eval_fn)(x_poisoned, w, g_poisoned)

    self.assertFalse(bool(jnp.any(jnp.isnan(y_pos))))
    np.testing.assert_array_equal(np.asarray(y_pos[:valid_count]), np.asarray(y_ref[:valid_count]))
    if need_lhs:
      self.assertFalse(bool(jnp.any(jnp.isnan(dl_pos))))
      np.testing.assert_array_equal(np.asarray(dl_pos), np.asarray(dl_ref))
    if need_rhs:
      self.assertFalse(bool(jnp.any(jnp.isnan(dr_pos))))
      np.testing.assert_array_equal(np.asarray(dr_pos), np.asarray(dr_ref))

  def test_2d_vs_3d_layout_and_chunked_gradient_accumulation(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Requires >= 2 devices")
    is_tpu = devices[0].platform == "tpu"
    mesh = Mesh(np.array(devices[:2]), ("fsdp",))

    g_num, m_tot, k_dim, n_dim = 4, 512, 512, 512
    key = jax.random.PRNGKey(606)
    kx, kw0, kw1, kg = jax.random.split(key, 4)
    x_2d = jax.random.normal(kx, (m_tot, k_dim), dtype=jnp.bfloat16)
    x_3d = x_2d.reshape(m_tot, k_dim // 128, 128)
    wi0_global = (jax.random.normal(kw0, (g_num, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    wi1_global = (jax.random.normal(kw1, (g_num, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    dy_2d = jax.random.normal(kg, (m_tot, n_dim), dtype=jnp.bfloat16)
    grp_full = jnp.array([128, 128, 128, 128], dtype=jnp.int32)
    grp_half = jnp.array([64, 64, 64, 64], dtype=jnp.int32)

    # 1. 2D vs 3D (`out_is_3d=True`) + `return_lhs=True` dlhs accumulation parity.
    def run_wi_pair(x_in, w0_in, w1_in, is_3d: bool):
      (s0, s1) = fp8_moe.precompute_expert_weight_scales((w0_in, w1_in), (1, 1), None)
      g0, x_pass = fp8_moe.ragged_dot_fp8(
          x_in, w0_in, s0, grp_full, fsdp_axis=None, gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, out_is_3d=is_3d, return_lhs=True
      )
      u1 = fp8_moe.ragged_dot_fp8(
          x_pass, w1_in, s1, grp_full, fsdp_axis=None, gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, out_is_3d=is_3d
      )
      act = jax.nn.silu(g0.astype(jnp.float32)).astype(jnp.bfloat16) * u1
      dy = dy_2d.reshape(act.shape)
      return jnp.sum(act.astype(jnp.float32) * dy.astype(jnp.float32)), act

    def run_wi_pair_f32_ref(x_in, w0_in, w1_in):
      g0 = jax.lax.ragged_dot(x_in.astype(jnp.float32), w0_in.astype(jnp.float32), grp_full, precision=jax.lax.Precision.HIGHEST)
      u1 = jax.lax.ragged_dot(x_in.astype(jnp.float32), w1_in.astype(jnp.float32), grp_full, precision=jax.lax.Precision.HIGHEST)
      act = jax.nn.silu(g0) * u1
      return jnp.sum(act * dy_2d.astype(jnp.float32)), act

    (_, act_ref), (dx_ref, dw0_ref, dw1_ref) = jax.jit(jax.value_and_grad(run_wi_pair_f32_ref, argnums=(0, 1, 2), has_aux=True))(
        x_2d, wi0_global, wi1_global
    )
    (_, act_2d), (dx_2d, dw0_2d, dw1_2d) = jax.jit(jax.value_and_grad(functools.partial(run_wi_pair, is_3d=False), argnums=(0, 1, 2), has_aux=True))(
        x_2d, wi0_global, wi1_global
    )
    (_, act_3d), (dx_3d, dw0_3d, dw1_3d) = jax.jit(jax.value_and_grad(functools.partial(run_wi_pair, is_3d=True), argnums=(0, 1, 2), has_aux=True))(
        x_3d, wi0_global, wi1_global
    )
    act_3d_2d = np.asarray(act_3d.reshape(m_tot, n_dim))
    dx_3d_2d = np.asarray(dx_3d.reshape(m_tot, k_dim))
    np.testing.assert_allclose(act_3d_2d, np.asarray(act_2d), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(dx_3d_2d, np.asarray(dx_2d), rtol=2e-2, atol=2e-2)
    # Verify both 2D and 3D match exact float32 ragged_dot reference (>= 22 dB) and agree closely (>= 30 dB, cos >= 0.999;
    # note 1 BF16 ULP in 3D vs 2D `silu(g0)` can flip 2-mantissa-bit `f8e5m2` rounding boundaries in `drhs_q`).
    for arr_2d, arr_3d, arr_ref in ((act_2d, act_3d_2d, act_ref), (dx_2d, dx_3d_2d, dx_ref), (dw0_2d, dw0_3d, dw0_ref), (dw1_2d, dw1_3d, dw1_ref)):
      self.assertGreaterEqual(_sqnr_db(np.asarray(arr_ref), np.asarray(arr_2d)), 22.0)
      self.assertGreaterEqual(_sqnr_db(np.asarray(arr_ref), np.asarray(arr_3d)), 22.0)
      self.assertGreaterEqual(_sqnr_db(np.asarray(arr_2d), np.asarray(arr_3d)), 30.0)
      self.assertGreaterEqual(_cosine_sim(np.asarray(arr_2d), np.asarray(arr_3d)), 0.999)

    # 2. Two token chunks that share one gathered FP8 weight (`return_rhs=True`) accumulate one weight gradient.
    @jax.jit
    def run_chunk_parity(w0_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P(None, "fsdp", None),),
          out_specs=(P(), P(None, "fsdp", None), P(None, "fsdp", None)),
          check_vma=False,
      )
      def _fn(w_s):
        (sc,) = fp8_moe.precompute_expert_weight_scales((w_s,), (1,), "fsdp")
        x0, x1 = x_2d[:256], x_2d[256:]
        dy0, dy1 = dy_2d[:256], dy_2d[256:]

        def separate_chunks(w):
          y0 = fp8_moe.ragged_dot_fp8(x0, w, sc, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu)
          y1 = fp8_moe.ragged_dot_fp8(x1, w, sc, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu)
          y = jnp.concatenate([y0, y1], axis=0)
          return jnp.sum(y.astype(jnp.float32) * dy_2d.astype(jnp.float32)), y

        def return_rhs_chunks(w):
          y0, w_q = fp8_moe.ragged_dot_fp8(
              x0, w, sc, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, return_rhs=True, scatter_drhs=True
          )
          y1 = fp8_moe.ragged_dot_fp8(
              x1, jax.lax.stop_gradient(w), sc, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu, pre_gathered_rhs_q=w_q, scatter_drhs=False
          )
          y = jnp.concatenate([y0, y1], axis=0)
          return jnp.sum(y.astype(jnp.float32) * jnp.concatenate([dy0, dy1], axis=0).astype(jnp.float32)), y

        (_, y_sep), dw_sep = jax.value_and_grad(separate_chunks, has_aux=True)(w_s)
        (_, y_ret), dw_ret = jax.value_and_grad(return_rhs_chunks, has_aux=True)(w_s)
        max_y_diff = jax.lax.pmax(jnp.max(jnp.abs(y_ret - y_sep)).astype(jnp.float32), axis_name="fsdp")
        return max_y_diff, dw_sep, dw_ret

      return _fn(w0_in)

    max_y_diff, dw_sep, dw_ret = run_chunk_parity(wi0_global)
    self.assertEqual(float(max_y_diff), 0.0)
    dw_sep_f32 = np.asarray(dw_sep, dtype=np.float32)
    dw_ret_f32 = np.asarray(dw_ret, dtype=np.float32)
    # `dw_ret` sums the chunks in f32 before one bf16 `psum_scatter`, so it may differ by one bf16 ULP.
    self.assertGreaterEqual(_sqnr_db(dw_sep_f32, dw_ret_f32), 45.0)
    np.testing.assert_allclose(dw_ret_f32, dw_sep_f32, rtol=2e-2, atol=0.25)

  def test_swiglu_block_accuracy_vs_bf16(self):
    """FP8 forward and gradients stay within quantization noise of the BF16 expert block."""
    devices = jax.devices()
    is_tpu = devices[0].platform == "tpu"

    g_local, m_rows, k_dim, n_dim = 4, 512, 512, 512
    mesh = Mesh(np.array(devices[:2]).reshape(2, 1), ("fsdp", "ep"))
    ep_size = mesh.shape["ep"]
    key = jax.random.PRNGKey(707)
    kx, k0, k1, ko, kg = jax.random.split(key, 5)

    x_global = jax.random.normal(kx, (ep_size, m_rows, k_dim), dtype=jnp.bfloat16)
    wi0_global = (jax.random.normal(k0, (ep_size * g_local, k_dim, n_dim), dtype=jnp.float32) / np.sqrt(k_dim)).astype(
        jnp.bfloat16
    )
    wi1_global = (jax.random.normal(k1, (ep_size * g_local, k_dim, n_dim), dtype=jnp.float32) / np.sqrt(k_dim)).astype(
        jnp.bfloat16
    )
    wo_global = (jax.random.normal(ko, (ep_size * g_local, n_dim, k_dim), dtype=jnp.float32) / np.sqrt(n_dim)).astype(
        jnp.bfloat16
    )
    dy_global = jax.random.normal(kg, (ep_size, m_rows, k_dim), dtype=jnp.bfloat16)
    grp_1 = jnp.full((g_local,), m_rows // g_local, dtype=jnp.int32)
    grp_global = jnp.broadcast_to(grp_1[None, :], (ep_size, g_local))

    def build_step(use_fp8: bool, tiling: tuple[int, ...] = _DEFAULT_TILING):
      @jax.jit
      def step_fn(x_in, w0_in, w1_in, wo_in, dy_in, grp_in):
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(
                P("ep", None, None),
                P("ep", "fsdp", None),
                P("ep", "fsdp", None),
                P("ep", "fsdp", None),
                P("ep", None, None),
                P("ep", None),
            ),
            out_specs=(
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", "fsdp", None),
                P("ep", "fsdp", None),
                P("ep", "fsdp", None),
            ),
            check_vma=False,
        )
        def _shard(x_s, w0_s, w1_s, wo_s, dy_s, grp_s):
          x_2d = x_s.squeeze(0)
          dy_2d = dy_s.squeeze(0)
          grp_loc = grp_s.squeeze(0)

          def fwd_swiglu(xl, a0, a1, ao):
            if use_fp8:
              s0, s1, so = fp8_moe.precompute_expert_weight_scales((a0, a1, ao), (1, 1, 1), "fsdp")
              h0, xl_pass = fp8_moe.ragged_dot_fp8(
                  xl, a0, s0, grp_loc, fsdp_axis="fsdp", gather_axis=1, tiling=tiling, use_gmm_v2=is_tpu, return_lhs=True
              )
              h1 = fp8_moe.ragged_dot_fp8(
                  xl_pass, a1, s1, grp_loc, fsdp_axis="fsdp", gather_axis=1, tiling=tiling, use_gmm_v2=is_tpu
              )
              act = jax.nn.silu(h0.astype(jnp.float32)).astype(jnp.bfloat16) * h1
              y = fp8_moe.ragged_dot_fp8(
                  act, ao, so, grp_loc, fsdp_axis="fsdp", gather_axis=1, tiling=tiling, use_gmm_v2=is_tpu
              )
            else:
              w0_f = jax.lax.all_gather(a0, "fsdp", axis=1, tiled=True)
              w1_f = jax.lax.all_gather(a1, "fsdp", axis=1, tiled=True)
              wo_f = jax.lax.all_gather(ao, "fsdp", axis=1, tiled=True)
              h0 = _bf16_ref_gmm(xl, w0_f, grp_loc, is_tpu)
              h1 = _bf16_ref_gmm(xl, w1_f, grp_loc, is_tpu)
              act = jax.nn.silu(h0.astype(jnp.float32)).astype(jnp.bfloat16) * h1
              y = _bf16_ref_gmm(act, wo_f, grp_loc, is_tpu)
            # Post-expert RMSNorm (`eps=1e-8`) and the residual add.
            y_f32 = y.astype(jnp.float32)
            rstd = jax.lax.rsqrt(jnp.mean(jnp.square(y_f32), axis=-1, keepdims=True) + 1e-8)
            y_norm = (y_f32 * rstd).astype(jnp.bfloat16)
            y_res = xl + y_norm
            loss = jnp.sum(y_res.astype(jnp.float32) * dy_2d.astype(jnp.float32))
            return loss, (y_res, y_norm)

          (_, (y_res_out, y_norm_out)), (dx, dw0, dw1, dwo) = jax.value_and_grad(
              fwd_swiglu, argnums=(0, 1, 2, 3), has_aux=True
          )(x_2d, w0_s, w1_s, wo_s)
          return y_res_out[None, ...], y_norm_out[None, ...], dx[None, ...], dw0, dw1, dwo

        return _shard(x_in, w0_in, w1_in, wo_in, dy_in, grp_in)

      return step_fn

    # 1. A single `ragged_dot_fp8`: forward >= 28 dB, both gradients >= 24 dB.
    @jax.jit
    def single_op_eval(x_in, w0_in, dy_in, grp_in):
      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(P("ep", None, None), P("ep", "fsdp", None), P("ep", None, None), P("ep", None)),
          out_specs=(P("ep", None, None), P("ep", None, None), P("ep", "fsdp", None), P("ep", None, None), P("ep", None, None), P("ep", "fsdp", None)),
          check_vma=False,
      )
      def _op(x_s, w_s, dy_s, g_s):
        xl, dyl, gl = x_s.squeeze(0), dy_s.squeeze(0), g_s.squeeze(0)
        (sw,) = fp8_moe.precompute_expert_weight_scales((w_s,), (1,), "fsdp")

        def fp8_f(l, r):
          y = fp8_moe.ragged_dot_fp8(l, r, sw, gl, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu)
          return jnp.sum(y.astype(jnp.float32) * dyl.astype(jnp.float32)), y

        def ref_f(l, r):
          rf = jax.lax.all_gather(r, "fsdp", axis=1, tiled=True)
          y = _bf16_ref_gmm(l, rf, gl, is_tpu)
          return jnp.sum(y.astype(jnp.float32) * dyl.astype(jnp.float32)), y

        (_, yf), (dxf, dwf) = jax.value_and_grad(fp8_f, argnums=(0, 1), has_aux=True)(xl, w_s)
        (_, yr), (dxr, dwr) = jax.value_and_grad(ref_f, argnums=(0, 1), has_aux=True)(xl, w_s)
        return yf[None, ...], dxf[None, ...], dwf, yr[None, ...], dxr[None, ...], dwr

      return _op(x_in, w0_in, dy_in, grp_in)

    yf_1, dxf_1, dwf_1, yr_1, dxr_1, dwr_1 = single_op_eval(x_global, wi0_global, dy_global, grp_global)
    op_fwd_sqnr = _sqnr_db(np.asarray(yr_1), np.asarray(yf_1))
    op_dlhs_sqnr = _sqnr_db(np.asarray(dxr_1), np.asarray(dxf_1))
    op_drhs_sqnr = _sqnr_db(np.asarray(dwr_1), np.asarray(dwf_1))
    self.assertGreaterEqual(op_fwd_sqnr, 28.0, msg=f"Per-op fwd SQNR {op_fwd_sqnr:.2f} dB < 28.0 dB")
    self.assertGreaterEqual(op_dlhs_sqnr, 24.0, msg=f"Per-op dlhs SQNR {op_dlhs_sqnr:.2f} dB < 24.0 dB")
    self.assertGreaterEqual(op_drhs_sqnr, 24.0, msg=f"Per-op drhs SQNR {op_drhs_sqnr:.2f} dB < 24.0 dB")

    # 2. The full 3-matmul SwiGLU + RMSNorm + residual block.
    ref_outs = build_step(use_fp8=False)(x_global, wi0_global, wi1_global, wo_global, dy_global, grp_global)
    fp8_step = build_step(use_fp8=True)
    fp8_outs = fp8_step(x_global, wi0_global, wi1_global, wo_global, dy_global, grp_global)

    y_sqnr = _sqnr_db(np.asarray(ref_outs[0]), np.asarray(fp8_outs[0]))
    y_branch_sqnr = _sqnr_db(np.asarray(ref_outs[1]), np.asarray(fp8_outs[1]))
    dx_sqnr = _sqnr_db(np.asarray(ref_outs[2]), np.asarray(fp8_outs[2]))
    dw0_sqnr = _sqnr_db(np.asarray(ref_outs[3]), np.asarray(fp8_outs[3]))
    dw1_sqnr = _sqnr_db(np.asarray(ref_outs[4]), np.asarray(fp8_outs[4]))
    dwo_sqnr = _sqnr_db(np.asarray(ref_outs[5]), np.asarray(fp8_outs[5]))
    dx_cos = _cosine_sim(np.asarray(ref_outs[2]), np.asarray(fp8_outs[2]))

    # Residual stream at the 62-layer decoder's branch scale (`x + y_norm / sqrt(62)`).
    x_np = np.asarray(x_global, dtype=np.float32)
    y_depth62_ref = x_np + np.asarray(ref_outs[1], dtype=np.float32) / np.sqrt(62.0)
    y_depth62_fp8 = x_np + np.asarray(fp8_outs[1], dtype=np.float32) / np.sqrt(62.0)
    y_depth62_sqnr = _sqnr_db(y_depth62_ref, y_depth62_fp8)

    self.assertGreaterEqual(y_depth62_sqnr, 38.0, msg=f"62L decoder residual fwd SQNR {y_depth62_sqnr:.2f} dB < 38.0 dB")
    self.assertGreaterEqual(y_sqnr, 26.0, msg=f"Unweighted 3-matmul residual fwd SQNR {y_sqnr:.2f} dB < 26.0 dB")
    self.assertGreaterEqual(y_branch_sqnr, 23.0, msg=f"3-matmul SwiGLU branch SQNR {y_branch_sqnr:.2f} dB < 23.0 dB")
    self.assertGreaterEqual(dx_sqnr, 21.5, msg=f"5-quantization SwiGLU+RMSNorm bwd dx SQNR {dx_sqnr:.2f} dB < 21.5 dB")
    self.assertGreaterEqual(dw0_sqnr, 19.5, msg=f"3-matmul bwd dwi_0 SQNR {dw0_sqnr:.2f} dB < 19.5 dB")
    self.assertGreaterEqual(dw1_sqnr, 19.5, msg=f"3-matmul bwd dwi_1 SQNR {dw1_sqnr:.2f} dB < 19.5 dB")
    self.assertGreaterEqual(dwo_sqnr, 21.0, msg=f"3-matmul bwd dwo SQNR {dwo_sqnr:.2f} dB < 21.0 dB")
    self.assertGreaterEqual(dx_cos, 0.996, msg=f"Backward dx cosine {dx_cos:.6f} < 0.996")

  def test_scan_checkpoint_shard_map_has_no_tracer_leak(self):
    """`jax.lax.scan` + `jax.checkpoint` + `jax.shard_map` must trace cleanly under `jax.checking_leaks()`."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices; got {len(devices)}")
    is_tpu = devices[0].platform == "tpu"
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ("fsdp", "ep"))

    n_layers, g_local, k_dim, n_dim, t_local = 2, 2, 512, 512, 64
    key = jax.random.PRNGKey(808)
    kx, kw0, kw1, kwo = jax.random.split(key, 4)
    x_in = jax.random.normal(kx, (2, t_local, k_dim), dtype=jnp.bfloat16)
    w0_in = (jax.random.normal(kw0, (n_layers, 2 * g_local, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    w1_in = (jax.random.normal(kw1, (n_layers, 2 * g_local, k_dim, n_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)
    wo_in = (jax.random.normal(kwo, (n_layers, 2 * g_local, n_dim, k_dim), dtype=jnp.float32) * 0.02).astype(jnp.bfloat16)

    def _make_scanned_step():
      @jax.jit
      def scanned_step(x_arg, w0_arg, w1_arg, wo_arg):
        def loss_fn(x0, w0s, w1s, wos):
          @jax.checkpoint
          def layer_fn(carry_x, layer_ws):
            lw0, lw1, lwo = layer_ws

            @functools.partial(
                jax.shard_map,
                mesh=mesh,
                in_specs=(P("ep", None, None), P("ep", "fsdp", None), P("ep", "fsdp", None), P("ep", None, "fsdp")),
                out_specs=P("ep", None, None),
                check_vma=False,
            )
            def _shard_layer(xl, sw0, sw1, swo):
              x_2d = xl.squeeze(0)
              s0, s1, so = fp8_moe.precompute_expert_weight_scales((sw0, sw1, swo), (1, 1, 2), "fsdp")
              ep_x = fp8_moe.fp8_ep_token_all_gather(x_2d, "ep")
              m_ep = ep_x.shape[0]
              half_m = m_ep // 2
              grp_half = jnp.full((g_local,), half_m // g_local, dtype=jnp.int32)
              vmask = jnp.arange(half_m) < (half_m - 4)

              # Exercise the shared pre-gathered weight, `bwd_lhs_override`, and all three VJP modes.
              c0, x0_pass, w0_q = fp8_moe.ragged_dot_fp8(
                  ep_x[:half_m], sw0, s0, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING,
                  use_gmm_v2=is_tpu, valid_mask=vmask, return_lhs=True, return_rhs=True, scatter_drhs=True,
              )
              u0, w1_q = fp8_moe.ragged_dot_fp8(
                  x0_pass, sw1, s1, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING,
                  use_gmm_v2=is_tpu, valid_mask=vmask, return_rhs=True, scatter_drhs=True,
                  bwd_lhs_override=ep_x[:half_m],
              )
              c1 = fp8_moe.ragged_dot_fp8(
                  ep_x[half_m:], sw0, s0, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING,
                  use_gmm_v2=is_tpu, pre_gathered_rhs_q=w0_q, scatter_drhs=False, need_lhs_grad=True, need_rhs_grad=False,
              )
              u1 = fp8_moe.ragged_dot_fp8(
                  ep_x[half_m:], sw1, s1, grp_half, fsdp_axis="fsdp", gather_axis=1, tiling=_DEFAULT_TILING,
                  use_gmm_v2=is_tpu, pre_gathered_rhs_q=w1_q, scatter_drhs=False, need_lhs_grad=False, need_rhs_grad=True,
                  bwd_lhs_override=ep_x[half_m:],
              )
              act = jax.nn.silu(jnp.concatenate([c0, c1], axis=0).astype(jnp.float32)).astype(jnp.bfloat16) * jnp.concatenate([u0, u1], axis=0)
              grp_full = jnp.full((g_local,), m_ep // g_local, dtype=jnp.int32)
              y_ep = fp8_moe.ragged_dot_fp8(
                  act, swo, so, grp_full, fsdp_axis="fsdp", gather_axis=2, tiling=_DEFAULT_TILING, use_gmm_v2=is_tpu,
              )
              y_loc = fp8_moe.fp8_ep_combine_psum_scatter(y_ep, "ep")
              return (x_2d + y_loc)[None, ...]

            return _shard_layer(carry_x, lw0, lw1, lwo), None

          y_final, _ = jax.lax.scan(layer_fn, x0, (w0s, w1s, wos))
          return jnp.mean(jnp.square(y_final.astype(jnp.float32)))

        return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(x_arg, w0_arg, w1_arg, wo_arg)

      return scanned_step

    # Warm up JAX's and Pallas's one-time global caches, then trace a brand-new `scanned_step` under
    # `jax.checking_leaks()`.
    jax.block_until_ready(_make_scanned_step()(x_in, w0_in, w1_in, wo_in))

    with jax.checking_leaks():
      fresh_scanned_step = _make_scanned_step()
      loss_val, grads = fresh_scanned_step(x_in, w0_in, w1_in, wo_in)
      jax.block_until_ready((loss_val, grads))

    self.assertFalse(bool(jnp.isnan(loss_val)))
    for g in grads:
      self.assertFalse(bool(jnp.any(jnp.isnan(g))))


if __name__ == "__main__":
  absltest.main()
