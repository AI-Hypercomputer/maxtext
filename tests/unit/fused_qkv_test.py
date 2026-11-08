# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity tests of the fused fp8 QKV projection Pallas kernel.

The kernel runs in Pallas interpret mode on CPU and is compared with
`fused_qkv_proj_reference` (same op structure and rounding points, exact in
the f32 mode) and with `fused_qkv_proj_maxtext_literal` (MaxText's un-fused
ops with per-op compute-dtype rounding, matched up to bf16 noise).
"""

import dataclasses
import functools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.kernels import fused_qkv
import numpy as np


@dataclasses.dataclass(frozen=True)
class ModelDims:
  emb: int = 4096
  num_q_heads: int = 64
  num_kv_heads: int = 4
  head_dim: int = 128
  eps: float = 1e-6
  rope_max_timescale: float = 5e6


QWEN3_235B = ModelDims()
SMALL = ModelDims(emb=512, num_q_heads=16, num_kv_heads=2)


def _errors(actual, expected):
  """Max abs error normalized by max |expected|, and the relative RMS error."""
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  diff = actual - expected
  scale = max(np.max(np.abs(expected)), 1e-30)
  return (
      float(np.max(np.abs(diff)) / scale),
      float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-30)),
      float(np.mean(diff != 0)),
  )


class FusedQkvProjTest(parameterized.TestCase):

  def _inputs(self, dims: ModelDims, n_tokens: int, dtype, quantize: bool, seed: int = 0):
    """Returns random `(x, w_qkv, q_norm_scale, k_norm_scale, positions)` for `dims`."""
    k_x, k_q, k_k, k_v, k_qs, k_ks, k_pos = jax.random.split(jax.random.PRNGKey(seed), 7)
    x = jax.random.normal(k_x, (n_tokens, dims.emb), jnp.float32) * 2.0
    # MaxText kernels are f32 masters; `fuse_qkv_weights` casts them to the
    # compute dtype like `DenseGeneral` does before the quantized dot.
    std = dims.emb**-0.5
    wq = jax.random.normal(k_q, (dims.emb, dims.num_q_heads, dims.head_dim), jnp.float32) * std
    wk = jax.random.normal(k_k, (dims.emb, dims.num_kv_heads, dims.head_dim), jnp.float32) * std
    wv = jax.random.normal(k_v, (dims.emb, dims.num_kv_heads, dims.head_dim), jnp.float32) * std
    q_norm_scale = 1.0 + 0.1 * jax.random.normal(k_qs, (dims.head_dim,), jnp.float32)
    k_norm_scale = 1.0 + 0.1 * jax.random.normal(k_ks, (dims.head_dim,), jnp.float32)
    positions = jax.random.randint(k_pos, (n_tokens,), 0, 4096, jnp.int32)
    w_qkv = fused_qkv.fuse_qkv_weights(wq, wk, wv, compute_dtype=dtype, qtype=jnp.float8_e4m3fn if quantize else None)
    return x.astype(dtype), w_qkv, q_norm_scale, k_norm_scale, positions

  def _kwargs(self, dims: ModelDims, quantize: bool):
    return {
        "num_q_heads": dims.num_q_heads,
        "num_kv_heads": dims.num_kv_heads,
        "head_dim": dims.head_dim,
        "eps": dims.eps,
        "rope_max_timescale": dims.rope_max_timescale,
        "act_qtype": jnp.float8_e4m3fn if quantize else None,
    }

  @parameterized.named_parameters(
      ("qwen3_235b_two_row_tiles", QWEN3_235B, 256, 128, 8, False),
      ("small_16_heads_per_tile", SMALL, 512, 256, 16, False),
  )
  def test_f32_mode_is_exact(self, dims, n_tokens, block_m, heads_per_tile, k_seq_minor):
    """Without quantization the kernel reproduces the jitted reference to the ulp.

    `fused_qkv_proj` is jitted, so it is held to the jitted reference: XLA:CPU evaluates
    `positions / timescale` differently under jit than eagerly, and the large RoPE angles turn
    that into ~1e-4 differences in q and k (the eager comparison is only logged). The only
    admissible deviations are few-ulp effects of FMA contraction and reduction order. RoPE
    subtracts two O(1) products, so a small output keeps the absolute rounding error of its
    O(1) inputs; `atol` allows about 4 f32 ulps at 1.0, since CPUs differ in how they round
    a few of those. Anything structural (a missing scale, a wrong rotation, ...) is orders of
    magnitude larger.
    """
    x, w_qkv, q_scale, k_scale, positions = self._inputs(dims, n_tokens, jnp.float32, quantize=False)
    kwargs = self._kwargs(dims, quantize=False)
    expected = fused_qkv.fused_qkv_proj_reference(
        x, w_qkv, q_scale, k_scale, positions, k_seq_minor=k_seq_minor, **kwargs
    )
    expected_jit = jax.jit(functools.partial(fused_qkv.fused_qkv_proj_reference, k_seq_minor=k_seq_minor, **kwargs))(
        x, w_qkv, q_scale, k_scale, positions
    )
    actual = fused_qkv.fused_qkv_proj(
        x,
        w_qkv,
        q_scale,
        k_scale,
        positions,
        block_m=block_m,
        heads_per_tile=heads_per_tile,
        k_seq_minor=k_seq_minor,
        interpret=True,
        **kwargs,
    )
    for name, a, e, ej in zip("qkv", actual, expected, expected_jit):
      self.assertEqual(a.shape, e.shape, name)
      self.assertEqual(a.dtype, jnp.float32, name)
      max_err, rms_err, frac = _errors(a, ej)
      _, _, frac_eager = _errors(a, e)
      logging.info(
          "f32 %s: max_err=%.3e rms_err=%.3e mismatching=%.2e (vs jitted reference),"
          " mismatching=%.2e (vs eager reference)",
          name,
          max_err,
          rms_err,
          frac,
          frac_eager,
      )
      np.testing.assert_allclose(np.asarray(a), np.asarray(ej), rtol=1e-6, atol=5e-7, err_msg=name)

  @parameterized.named_parameters(
      ("qwen3_235b_two_row_tiles", QWEN3_235B, 256, 128, 8, False),
  )
  def test_bf16_fp8_mode_matches_reference(self, dims, n_tokens, block_m, heads_per_tile, k_seq_minor):
    """bf16 activations, fp8 e4m3 operands with the fixed,-224,224 scales."""
    x, w_qkv, q_scale, k_scale, positions = self._inputs(dims, n_tokens, jnp.bfloat16, quantize=True)
    self.assertEqual(w_qkv.dtype, jnp.float8_e4m3fn)
    kwargs = self._kwargs(dims, quantize=True)
    expected = fused_qkv.fused_qkv_proj_reference(
        x, w_qkv, q_scale, k_scale, positions, k_seq_minor=k_seq_minor, **kwargs
    )
    actual = fused_qkv.fused_qkv_proj(
        x,
        w_qkv,
        q_scale,
        k_scale,
        positions,
        block_m=block_m,
        heads_per_tile=heads_per_tile,
        k_seq_minor=k_seq_minor,
        interpret=True,
        **kwargs,
    )
    for name, a, e in zip("qkv", actual, expected):
      self.assertEqual(a.shape, e.shape, name)
      self.assertEqual(a.dtype, jnp.bfloat16, name)
      max_err, rms_err, frac = _errors(a, e)
      logging.info("bf16/fp8 %s vs reference: max_err=%.3e rms_err=%.3e mismatching=%.2e", name, max_err, rms_err, frac)
      self.assertLessEqual(max_err, 1e-2, name)
      self.assertLessEqual(rms_err, 2e-3, name)

  def test_static_quantize_matches_qwix_fixed_calibration(self):
    """`clip(x / 0.5, -448, 448)` in bf16 then e4m3, as Qwix evaluates it."""
    from maxtext.layers import quantizations  # pylint: disable=import-outside-toplevel

    x = jnp.asarray(np.linspace(-300, 300, 4097, dtype=np.float32), jnp.bfloat16)
    expected = jnp.clip(x / jnp.asarray(0.5, jnp.bfloat16), -448, 448).astype(jnp.float8_e4m3fn)
    actual = fused_qkv.static_quantize(x, jnp.float8_e4m3fn, 0.5)
    np.testing.assert_array_equal(np.asarray(actual, np.float32), np.asarray(expected, np.float32))
    self.assertEqual(quantizations.get_static_scale(jnp.float8_e4m3fn, "fixed,-224,224"), 0.5)
    self.assertEqual(quantizations.get_static_scale(jnp.float8_e4m3fn, "fixed,224"), 0.5)

  @parameterized.named_parameters(
      ("bf16_seq_minor", False),
      ("fp8_qk_seq_minor", True),
  )
  def test_4d_splash_layout_and_fp8_qk(self, fp8_qk: bool):
    """4D batched `[B, H, D, S]` SEQ_MINOR Q/K and `[B, Hkv, S, D]` V match reference."""
    dims = SMALL
    batch, seq_len, block_m = 2, 256, 128
    x, w_qkv, q_scale, k_scale, positions = self._inputs(dims, batch * seq_len, jnp.bfloat16, quantize=True)
    qk_dtype = jnp.float8_e4m3fn if fp8_qk else None
    kwargs = {
        **self._kwargs(dims, quantize=True),
        "seq_len": seq_len,
        "q_seq_minor": True,
        "k_seq_minor": True,
        "q_out_dtype": qk_dtype,
        "k_out_dtype": qk_dtype,
    }
    # Jitted like the kernel wrapper, so both see the same XLA:CPU RoPE tables (see `test_f32_mode_is_exact`).
    expected = jax.jit(functools.partial(fused_qkv.fused_qkv_proj_reference, **kwargs))(
        x, w_qkv, q_scale, k_scale, positions
    )
    actual = fused_qkv.fused_qkv_proj(
        x, w_qkv, q_scale, k_scale, positions, block_m=block_m, heads_per_tile=8, interpret=True, **kwargs
    )
    self.assertEqual(actual[0].shape, (batch, dims.num_q_heads, dims.head_dim, seq_len))
    self.assertEqual(actual[1].shape, (batch, dims.num_kv_heads, dims.head_dim, seq_len))
    self.assertEqual(actual[2].shape, (batch, dims.num_kv_heads, seq_len, dims.head_dim))
    if fp8_qk:
      self.assertEqual(actual[0].dtype, jnp.float8_e4m3fn)
      self.assertEqual(actual[1].dtype, jnp.float8_e4m3fn)
    self.assertEqual(actual[2].dtype, jnp.bfloat16)
    for name, a, e in zip("qkv", actual, expected):
      max_err, rms_err, _ = _errors(a, e)
      self.assertLessEqual(max_err, 5e-2 if fp8_qk else 1e-2, name)
      self.assertLessEqual(rms_err, 1e-2 if fp8_qk else 2e-3, name)


class FusedQkvVmemAndConfigTest(parameterized.TestCase):
  """Tests the `fused_qkv_proj` VMEM row-tile fit and the config-time validation of the fused kernel flags."""

  _MIB = 1024 * 1024

  def _footprint_fn(self, *, emb, num_kv_heads, heads_per_tile, head_dim=128, save_pre_norm=True):
    # The training call: bf16 activations quantized in-kernel to fp8, fp8 weights, fp8 Q/K and bf16 V outputs.
    return functools.partial(
        fused_qkv._fused_qkv_vmem_footprint,  # pylint: disable=protected-access
        emb=emb,
        tn=heads_per_tile * head_dim,
        kv_cols=2 * num_kv_heads * head_dim,
        heads_per_tile=heads_per_tile,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        x_itemsize=2,
        itemsize=2,
        q_itemsize=1,
        k_itemsize=1,
        w_itemsize=1,
        need_xq_scratch=True,
        save_pre_norm=save_pre_norm,
    )

  def test_shrink_block_m_to_fit(self):
    shrink = fused_qkv._shrink_block_m_to_fit  # pylint: disable=protected-access
    linear = lambda bm: bm * self._MIB  # pylint: disable=unnecessary-lambda-assignment
    self.assertEqual(shrink(1024, linear, 2048 * self._MIB), 1024)  # Fits: unchanged.
    self.assertEqual(shrink(1024, linear, 600 * self._MIB), 512)
    self.assertEqual(shrink(1024, linear, 10 * self._MIB), 128)  # Stops at a multiple of 128.
    self.assertEqual(shrink(384, linear, 10 * self._MIB), 384)  # Halving would no longer divide 384.

  def test_qwen3_235b_tile_unchanged_and_8_kv_head_tile_shrinks(self):
    budget = 60 * self._MIB  # TPU7x: 64 MiB VMEM minus the 4 MiB headroom used by `fused_qkv_proj`.
    shrink = fused_qkv._shrink_block_m_to_fit  # pylint: disable=protected-access
    # Qwen3-235B-A22B (64 query / 4 KV heads, 8 heads per tile): the 1024-row tile fits and is kept.
    fn_235b = self._footprint_fn(emb=4096, num_kv_heads=4, heads_per_tile=8)
    self.assertLessEqual(fn_235b(1024), budget)
    self.assertEqual(shrink(1024, fn_235b, budget), 1024)
    # 8 KV heads (16 heads per tile), e.g. Qwen3-32B: 1024 rows overflow, so the tile halves to 512.
    fn_kv8 = self._footprint_fn(emb=5120, num_kv_heads=8, heads_per_tile=16)
    self.assertGreater(fn_kv8(1024), budget)
    self.assertEqual(shrink(1024, fn_kv8, budget), 512)
    self.assertLessEqual(fn_kv8(512), budget)

  def test_fused_kernel_flag_config_validation(self):
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    def init_cfg(**extra):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="fused_kernel_flag_validation_test",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          **extra,
      )

    fused = {"use_fused_qkv_kernel": True, "use_fused_out_kernel": True, "defer_norm_grad_allreduce": True}
    # The flags of the best Qwen3-235B-A22B run pass.
    cfg = init_cfg(model_name="qwen3-235b-a22b", override_model_config=True, **fused)
    self.assertTrue(cfg.use_fused_qkv_kernel and cfg.use_fused_out_kernel and cfg.defer_norm_grad_allreduce)
    init_cfg(decoder_block="qwen3", use_qk_norm=True, **fused)
    # The fused kernels are not tied to Qwen3: another decoder with per-head QK RMSNorm passes config validation.
    cfg = init_cfg(decoder_block="llama2", use_qk_norm=True, use_fused_qkv_kernel=True, use_fused_out_kernel=True)
    self.assertTrue(cfg.use_fused_qkv_kernel and cfg.use_fused_out_kernel)

    qk = {"decoder_block": "qwen3", "use_qk_norm": True}
    rejected = [
        ({"decoder_block": "llama2", "use_fused_qkv_kernel": True}, "requires use_qk_norm=True"),
        ({**qk, "use_fused_qkv_kernel": True, "attention_type": "mla"}, "attention_type='mla'"),
        ({**qk, "use_fused_out_kernel": True, "attention_type": "mla"}, "attention_type='mla'"),
        ({**qk, "use_fused_qkv_kernel": True, "fused_qkv": True}, "fused_qkv=True"),
        ({**qk, "use_fused_qkv_kernel": True, "share_kv_projections": True}, "share_kv_projections"),
    ]
    for overrides, msg in rejected:
      with self.subTest(**{k: str(v) for k, v in overrides.items()}):
        with self.assertRaisesRegex(ValueError, msg):
          init_cfg(**overrides)

    # The deferral is a no-op by design where the post-scan all-reduce does not run: those configs still load
    # (keeping the regular per-layer all-reduce) and only log a warning.
    for overrides in ({"scan_layers": False}, {"prefetch_fsdp_weights": True}):
      with self.subTest(**{k: str(v) for k, v in overrides.items()}):
        cfg = init_cfg(decoder_block="qwen3", defer_norm_grad_allreduce=True, **overrides)
        self.assertTrue(cfg.defer_norm_grad_allreduce)


class FusedQkvAttentionModuleTest(parameterized.TestCase):
  """Tests `Attention` with `use_fused_qkv_kernel=True` vs `False` across a 4-device FSDP mesh."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("needs 4 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")

  def _run_attention_pair(
      self,
      dtype: str,
      quantized: bool,
      tol_y: float,
      tol_grad: float,
      fused_out_flag: bool = False,
      bwd_calib: str = "fixed,-224,224",
      data_par: int = 1,
      fsdp_par: int = 4,
  ):
    """Runs Qwen3 attention with and without the fused QKV/out kernels and compares outputs and grads."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel
    import qwix  # pylint: disable=import-outside-toplevel

    emb, seq_len, nq, nkv, hd = 512, 128, 8, 2, 128

    def make_cfg(fused_qkv_flag: bool, cfg_dtype: str, cfg_quant: bool):
      kw = {
          "run_name": "fused_qkv_attn_test",
          "enable_checkpointing": False,
          "skip_jax_distributed_system": True,
          "hardware": "cpu",
          "decoder_block": "qwen3",
          "attention": "dot_product",
          "base_emb_dim": emb,
          "base_num_query_heads": nq,
          "base_num_kv_heads": nkv,
          "head_dim": hd,
          "use_qk_norm": True,
          "rope_max_timescale": 1_000_000.0,
          "max_target_length": seq_len,
          "per_device_batch_size": 1,
          "ici_data_parallelism": data_par,
          "ici_fsdp_parallelism": fsdp_par,
          "dtype": cfg_dtype,
          "weight_dtype": "float32",
          "matmul_precision": "highest",
          "use_fused_qkv_kernel": fused_qkv_flag,
          "use_fused_out_kernel": (fused_qkv_flag and fused_out_flag),
          "sa_q_layout": "SEQ_MINOR",
          "sa_k_layout": "SEQ_MINOR",
      }
      if cfg_quant:
        kw.update(
            quantization="fp8_full",
            use_qwix_quantization=True,
            weight_quantization_calibration_method="fixed,-224,224",
            act_quantization_calibration_method="fixed,-224,224",
            bwd_quantization_calibration_method=bwd_calib,
        )
      return pyconfig.initialize([None, get_test_config_path()], **kw)

    cfg_ref = make_cfg(False, dtype, quantized)
    cfg_fused = make_cfg(True, dtype, quantized)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    batch = int(cfg_ref.per_device_batch_size) * jax.device_count()
    x_shape = (batch, seq_len, emb)

    def build(cfg, quant_flag: bool):
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        attn = attentions.Attention(
            config=cfg,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            max_target_length=seq_len,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=x_shape,
            inputs_kv_shape=x_shape,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            use_qk_norm=True,
            query_pre_attn_scalar=hd**-0.5,
            rngs=nnx.Rngs(params=0),
        )
        if quant_flag:
          rules = [
              qwix.QtRule(
                  module_path=".*",
                  weight_qtype=jnp.float8_e4m3fn,
                  act_qtype=jnp.float8_e4m3fn,
                  bwd_qtype=jnp.float8_e5m2,
                  weight_calibration_method=cfg.weight_quantization_calibration_method,
                  act_calibration_method=cfg.act_quantization_calibration_method,
                  bwd_calibration_method=cfg.bwd_quantization_calibration_method,
                  op_names=("dot_general",),
              )
          ]
          pos_dummy = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
          dummy = jnp.zeros(x_shape, cfg.dtype)
          attn = qwix.quantize_model(attn, qwix.QtProvider(rules), dummy, dummy, pos_dummy)
      return attn

    attn_ref = build(cfg_ref, quantized)
    attn_fused = build(cfg_fused, quantized)
    kx, kc = jax.random.split(jax.random.PRNGKey(3))
    x = jax.random.normal(kx, x_shape, jnp.float32)
    cot = jax.random.normal(kc, x_shape, jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))

    def eval_and_grad(cfg, model, x_in):
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)

      def loss_fn(p_, x_):
        m = nnx.merge(graphdef, p_, rest)
        y, _ = m(x_, x_, inputs_positions=pos)
        return jnp.sum(y.astype(jnp.float32) * cot), y

      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        (_, y), (gp, gx) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x_in)
      flat_gp = {
          jax.tree_util.keystr(k): np.asarray(v, np.float32) for k, v in jax.tree_util.tree_flatten_with_path(gp)[0]
      }
      return np.asarray(y, np.float32), flat_gp, np.asarray(gx, np.float32)

    y_ref, gp_ref, gx_ref = eval_and_grad(cfg_ref, attn_ref, x.astype(cfg_ref.dtype))
    y_fused, gp_fused, gx_fused = eval_and_grad(cfg_fused, attn_fused, x.astype(cfg_fused.dtype))
    _, rms_y, _ = _errors(y_fused, y_ref)
    _, rms_gx, _ = _errors(gx_fused, gx_ref)
    self.assertLess(rms_y, tol_y, f"y rms_err={rms_y}")
    self.assertLess(rms_gx, tol_grad, f"gx rms_err={rms_gx}")
    self.assertEqual(set(gp_ref.keys()), set(gp_fused.keys()))
    for k in gp_ref:
      _, rms_g, _ = _errors(gp_fused[k], gp_ref[k])
      self.assertLess(rms_g, tol_grad, f"grad {k} rms_err={rms_g}")

  def test_attention_f32_fsdp_parity(self):
    self._run_attention_pair("float32", quantized=False, tol_y=1e-4, tol_grad=1e-4)

  def test_attention_fp8_fsdp_parity(self):
    self._run_attention_pair("bfloat16", quantized=True, tol_y=1.5e-2, tol_grad=2.5e-2)
    # e5m2 has 2 mantissa bits (~0.072 RMS per-ULP shift between pow2 per-tensor and Qwix per-channel absmax).
    self._run_attention_pair("bfloat16", quantized=True, tol_y=1.5e-2, tol_grad=9e-2, bwd_calib="absmax")
    # 2D (data=2, fsdp=2) mesh exercises extra_w_psum_axes without double-psum!
    self._run_attention_pair(
        "bfloat16",
        quantized=True,
        tol_y=1.5e-2,
        tol_grad=2.5e-2,
        fused_out_flag=True,
        data_par=2,
        fsdp_par=2,
    )

  def test_fused_paths_reject_unsupported_additional_qt_config(self):
    """The fused QKV / out projections raise for Qwix settings they would otherwise drop silently."""
    import types  # pylint: disable=import-outside-toplevel

    from flax import nnx  # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel
    import qwix  # pylint: disable=import-outside-toplevel

    emb, seq_len, nq, nkv, hd = 512, 128, 8, 2, 128
    bwd_calib = "fixed,-224,224"

    def make_rule(extra):
      return qwix.QtRule(
          module_path=".*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method="fixed,-224,224",
          act_calibration_method="fixed,-224,224",
          bwd_calibration_method=bwd_calib,
          additional_qt_config=extra,
          op_names=("dot_general",),
      )

    # 1. The helper accepts what `get_fp8_full_qwix_rule_w_sparsity` passes when both features are off (or when the
    # drhs calibration equals the bwd one) and rejects an active sparsity rule or a different drhs calibration.
    active_sparsity = types.SimpleNamespace(weight_sparsity_n=2, weight_sparsity_m=4)
    inactive_sparsity = types.SimpleNamespace(weight_sparsity_n=None, weight_sparsity_m=None)
    for ok_extra in (
        None,
        {"sparsity_rule": None},
        {"sparsity_rule": inactive_sparsity},
        {"sparsity_rule": None, "drhs_grad_calibration_method": bwd_calib},
    ):
      attentions.Attention._check_fused_qt_rule(make_rule(ok_extra), "feature")  # pylint: disable=protected-access
    with self.assertRaisesRegex(NotImplementedError, "feature does not support weight sparsity"):
      attentions.Attention._check_fused_qt_rule(  # pylint: disable=protected-access
          make_rule({"sparsity_rule": active_sparsity}), "feature"
      )
    with self.assertRaisesRegex(NotImplementedError, "drhs_grad_calibration_method='absmax'"):
      attentions.Attention._check_fused_qt_rule(  # pylint: disable=protected-access
          make_rule({"sparsity_rule": None, "drhs_grad_calibration_method": "absmax"}), "feature"
      )

    # 2. Both fused paths call it: quantizing `Attention` with such a rule fails in the fused QKV projection, and
    # with only the fused out projection enabled it fails there.
    def quantize_attention(use_fused_qkv: bool, extra):
      cfg = pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="fused_kernel_qt_guard_test",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          decoder_block="qwen3",
          attention="dot_product",
          base_emb_dim=emb,
          base_num_query_heads=nq,
          base_num_kv_heads=nkv,
          head_dim=hd,
          use_qk_norm=True,
          max_target_length=seq_len,
          per_device_batch_size=1,
          ici_fsdp_parallelism=4,
          dtype="bfloat16",
          weight_dtype="float32",
          use_fused_qkv_kernel=use_fused_qkv,
          use_fused_out_kernel=True,
          quantization="fp8_full",
          use_qwix_quantization=True,
          weight_quantization_calibration_method="fixed,-224,224",
          act_quantization_calibration_method="fixed,-224,224",
          bwd_quantization_calibration_method=bwd_calib,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      x_shape = (jax.device_count(), seq_len, emb)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        attn = attentions.Attention(
            config=cfg,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            max_target_length=seq_len,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=x_shape,
            inputs_kv_shape=x_shape,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            use_qk_norm=True,
            query_pre_attn_scalar=hd**-0.5,
            rngs=nnx.Rngs(params=0),
        )
        dummy = jnp.zeros(x_shape, cfg.dtype)
        pos_dummy = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), x_shape[:2])
        return qwix.quantize_model(attn, qwix.QtProvider([make_rule(extra)]), dummy, dummy, pos_dummy)

    quantize_attention(True, {"sparsity_rule": None})
    unsupported = {"sparsity_rule": None, "drhs_grad_calibration_method": "absmax"}
    with self.assertRaisesRegex(NotImplementedError, "use_fused_qkv_kernel quantizes one cotangent"):
      quantize_attention(True, unsupported)
    with self.assertRaisesRegex(NotImplementedError, "use_fused_out_kernel quantizes one cotangent"):
      quantize_attention(False, unsupported)

  def test_fp8_qk_with_bf16_splash_cotangents(self):
    """Regression test: Tokamax Splash custom_vjp returns bf16 dq/dk for fp8_e4m3fn q/k."""
    base_p = fused_qkv.FusedQkvTrainParams(
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        eps=1e-6,
        rope_max_timescale=1e6,
        rope_min_timescale=1.0,
        query_pre_attn_scalar=(128**-0.5) * float(np.log2(np.e)),
        compute_dtype=jnp.bfloat16,
        act_qtype=jnp.float8_e4m3fn,
        weight_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        act_scale=0.5,
        weight_scale=0.5,
        bwd_scale=224.0 / 57344.0,
        q_seq_minor=True,
        k_seq_minor=True,
        q_out_dtype=jnp.float8_e4m3fn,
        k_out_dtype=jnp.float8_e4m3fn,
        block_m=128,
        heads_per_tile=4,
        quant_rows=128,
        use_reference=True,
    )

    @jax.custom_vjp
    def fake_splash(q, k, v):
      qf = q.astype(jnp.float32)
      kf = jnp.repeat(k.astype(jnp.float32), 2, axis=2)
      vf = jnp.repeat(v.astype(jnp.float32), 2, axis=2)
      return (qf * kf + vf).astype(jnp.bfloat16)

    def fake_splash_fwd(q, k, v):
      return fake_splash(q, k, v), (q, k, v)

    def fake_splash_bwd(res, do):
      q, k, _ = res
      qf = q.astype(jnp.float32)
      kf = jnp.repeat(k.astype(jnp.float32), 2, axis=2)
      dof = do.astype(jnp.float32)
      dq = (dof * kf).astype(jnp.bfloat16)
      dk = jnp.sum((dof * qf).reshape(q.shape[0], q.shape[1], 2, 2, 128), axis=3).astype(jnp.bfloat16)
      dv = jnp.sum(dof.reshape(q.shape[0], q.shape[1], 2, 2, 128), axis=3).astype(jnp.bfloat16)
      return dq, dk, dv

    fake_splash.defvjp(fake_splash_fwd, fake_splash_bwd)

    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(11), 5)
    x = jax.random.normal(k0, (2, 128, 256), jnp.bfloat16)
    wq = jax.random.normal(k1, (256, 4, 128), jnp.float32) * 0.05
    wk = jax.random.normal(k2, (256, 2, 128), jnp.float32) * 0.05
    wv = jax.random.normal(k3, (256, 2, 128), jnp.float32) * 0.05
    qns = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(12), (128,), jnp.float32)
    kns = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(13), (128,), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32), (2, 128))
    cot = jax.random.normal(k4, (2, 128, 4, 128), jnp.bfloat16)

    def loss_unfused_f32(x_, wq_, wk_, wv_, qns_, kns_):
      w_qkv = fused_qkv.fuse_qkv_weights(wq_, wk_, wv_, compute_dtype=jnp.float32, qtype=None)
      q, k, v = fused_qkv.fused_qkv_proj_maxtext_literal(
          x_.astype(jnp.float32).reshape(-1, 256),
          w_qkv,
          qns_,
          kns_,
          pos.reshape(-1),
          num_q_heads=4,
          num_kv_heads=2,
          head_dim=128,
          eps=1e-6,
          rope_max_timescale=1e6,
          query_pre_attn_scalar=base_p.query_pre_attn_scalar,
          act_qtype=None,
      )
      q = q.reshape(2, 128, 4, 128).astype(jnp.float8_e4m3fn)
      k = k.reshape(2, 128, 2, 128).astype(jnp.float8_e4m3fn)
      v = v.reshape(2, 128, 2, 128).astype(jnp.bfloat16)
      return jnp.sum(fake_splash(q, k, v).astype(jnp.float32) * cot.astype(jnp.float32))

    val_exp, grads_exp = jax.jit(jax.value_and_grad(loss_unfused_f32, argnums=(0, 1, 2, 3, 4, 5)))(
        x, wq, wk, wv, qns, kns
    )
    for p in (base_p, dataclasses.replace(base_p, use_reference=False, interpret=True)):

      def loss_fused(x_, wq_, wk_, wv_, qns_, kns_, p_=p):
        q, k, v = fused_qkv.fused_qkv_local(p_, x_, wq_, wk_, wv_, qns_, kns_, pos)
        self.assertEqual(q.dtype, jnp.float8_e4m3fn)
        self.assertEqual(k.dtype, jnp.float8_e4m3fn)
        return jnp.sum(fake_splash(q, k, v).astype(jnp.float32) * cot.astype(jnp.float32))

      val, grads = jax.jit(jax.value_and_grad(loss_fused, argnums=(0, 1, 2, 3, 4, 5)))(x, wq, wk, wv, qns, kns)
      self.assertLess(abs(float(val) - float(val_exp)) / (abs(float(val_exp)) + 1e-6), 0.05)
      for name, g, ge in zip(("dx", "dwq", "dwk", "dwv", "dqns", "dkns"), grads, grads_exp):
        self.assertLess(_errors(g, ge)[1], 0.12, f"use_ref={p.use_reference} {name}")

  @parameterized.named_parameters(
      ("f32_exact", jnp.float32, False, None, 1e-5),
      ("bf16_fp8_absmax", jnp.bfloat16, True, None, 3.5e-2),
      ("bf16_fp8_static", jnp.bfloat16, True, 224.0 / 57344.0, 3.5e-2),
  )
  def test_pallas_remat_and_bwd_post_parity(self, dtype, quantize: bool, bwd_scale, tol_rms: float):
    """Verifies `optimize_remat=True`, `save_pre_norm`, and `fused_qkv_bwd_post` under `jax.remat`."""
    batch, seq_len, emb, nq, nkv, hd = 2, 256, 256, 8, 2, 128
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.PRNGKey(29), 7)
    x = jax.random.normal(k0, (batch, seq_len, emb), dtype)
    wq = jax.random.normal(k1, (emb, nq, hd), jnp.float32) * 0.05
    wk = jax.random.normal(k2, (emb, nkv, hd), jnp.float32) * 0.05
    wv = jax.random.normal(k3, (emb, nkv, hd), jnp.float32) * 0.05
    qns = 1.0 + 0.1 * jax.random.normal(k4, (hd,), jnp.float32)
    kns = 1.0 + 0.1 * jax.random.normal(k5, (hd,), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
    cot = jax.random.normal(k6, (batch, seq_len, nq, hd), dtype)

    @jax.custom_vjp
    def fake_splash(q, k, v):
      qf = q.astype(jnp.float32)
      kf = jnp.repeat(k.astype(jnp.float32), nq // nkv, axis=2)
      vf = jnp.repeat(v.astype(jnp.float32), nq // nkv, axis=2)
      return (qf * kf + vf).astype(dtype)

    def fake_splash_fwd(q, k, v):
      return fake_splash(q, k, v), (q, k, v)

    def fake_splash_bwd(res, do):
      q, k, _ = res
      rep = nq // nkv
      qf = q.astype(jnp.float32)
      kf = jnp.repeat(k.astype(jnp.float32), rep, axis=2)
      dof = do.astype(jnp.float32)
      dq = (dof * kf).astype(dtype)
      dk = jnp.sum((dof * qf).reshape(batch, seq_len, nkv, rep, hd), axis=3).astype(dtype)
      dv = jnp.sum(dof.reshape(batch, seq_len, nkv, rep, hd), axis=3).astype(dtype)
      return dq, dk, dv

    fake_splash.defvjp(fake_splash_fwd, fake_splash_bwd)

    def make_params(use_ref: bool):
      return fused_qkv.FusedQkvTrainParams(
          num_q_heads=nq,
          num_kv_heads=nkv,
          head_dim=hd,
          eps=1e-6,
          rope_max_timescale=1e6,
          rope_min_timescale=1.0,
          query_pre_attn_scalar=(hd**-0.5) * (float(np.log2(np.e)) if quantize else 1.0),
          compute_dtype=dtype,
          act_qtype=jnp.float8_e4m3fn if quantize else None,
          bwd_qtype=jnp.float8_e5m2 if quantize else None,
          act_scale=0.5,
          weight_scale=0.5,
          bwd_scale=bwd_scale,
          q_seq_minor=True,
          k_seq_minor=True,
          q_out_dtype=jnp.float8_e4m3fn if quantize else None,
          k_out_dtype=jnp.float8_e4m3fn if quantize else None,
          block_m=128,
          heads_per_tile=4,
          quant_rows=128,
          use_reference=use_ref,
          interpret=not use_ref,
      )

    p_ref = make_params(True)
    p_pal = make_params(False)

    def make_loss(p):
      @jax.remat
      def layer(x_, wq_, wk_, wv_, qns_, kns_):
        q, k, v = fused_qkv.fused_qkv_local(p, x_, wq_, wk_, wv_, qns_, kns_, pos)
        return fake_splash(q, k, v)

      def loss(x_, wq_, wk_, wv_, qns_, kns_):
        y = layer(x_, wq_, wk_, wv_, qns_, kns_)
        return jnp.sum(y.astype(jnp.float32) * cot.astype(jnp.float32))

      return loss

    p_pal_nointerp = dataclasses.replace(p_pal, interpret=False)
    jaxpr_str = str(
        jax.make_jaxpr(jax.value_and_grad(make_loss(p_pal_nointerp), argnums=(0, 1, 2, 3, 4, 5)))(x, wq, wk, wv, qns, kns)
    )
    # Verify outer fwd uses fused_qkv_fwd, remat fwd uses fused_qkv_remat_fwd,
    # and backward uses fused_qkv_bwd_post!
    self.assertIn("fused_qkv_fwd", jaxpr_str)
    self.assertIn("fused_qkv_remat_fwd", jaxpr_str)
    self.assertIn("fused_qkv_bwd_post", jaxpr_str)

    fn_ref = jax.jit(jax.value_and_grad(make_loss(p_ref), argnums=(0, 1, 2, 3, 4, 5)))
    fn_pal = jax.jit(jax.value_and_grad(make_loss(p_pal), argnums=(0, 1, 2, 3, 4, 5)))
    hlo_pal = fn_pal.lower(x, wq, wk, wv, qns, kns).as_text()
    self.assertIn("@fused_qkv_proj", hlo_pal)
    self.assertIn("@fused_qkv_bwd_post", hlo_pal)

    val_ref, grads_ref = fn_ref(x, wq, wk, wv, qns, kns)
    val_pal, grads_pal = fn_pal(x, wq, wk, wv, qns, kns)
    _, val_rms, _ = _errors(val_pal, val_ref)
    self.assertLessEqual(val_rms, tol_rms)
    for name, gp, gr in zip(("dx", "dwq", "dwk", "dwv", "dqns", "dkns"), grads_pal, grads_ref):
      max_err, rms_err, _ = _errors(gp, gr)
      logging.info("remat %s: max_err=%.3e rms_err=%.3e", name, max_err, rms_err)
      self.assertLessEqual(rms_err, tol_rms, f"{name} rms_err={rms_err}")

  def test_bwd_post_tpu_aot_lowering(self):
    """Verifies `fused_qkv_proj` (`save_pre_norm` False/True) and `fused_qkv_bwd_post` lower cleanly to TPU7x Mosaic."""
    tpu7x = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=tpu7x)):
      for save_pre_norm in (False, True):
        aot_fwd = (
            jax.jit(
                functools.partial(
                    fused_qkv.fused_qkv_proj,
                    num_q_heads=64,
                    num_kv_heads=4,
                    head_dim=128,
                    seq_len=512,
                    eps=1e-6,
                    rope_max_timescale=5e6,
                    query_pre_attn_scalar=128**-0.5,
                    act_qtype=jnp.float8_e4m3fn,
                    act_scale=0.5,
                    weight_scale=0.5,
                    q_seq_minor=True,
                    k_seq_minor=True,
                    q_out_dtype=jnp.float8_e4m3fn,
                    k_out_dtype=jnp.float8_e4m3fn,
                    block_m=512,
                    heads_per_tile=8,
                    save_pre_norm=save_pre_norm,
                    interpret=False,
                )
            )
            .trace(
                jax.ShapeDtypeStruct((1024, 4096), jnp.bfloat16),
                jax.ShapeDtypeStruct((4096, 72 * 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((128,), jnp.float32),
                jax.ShapeDtypeStruct((128,), jnp.float32),
                jax.ShapeDtypeStruct((1024,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        self.assertIn("tpu_custom_call", aot_fwd)
      aot_mlir = (
          jax.jit(
              functools.partial(
                  fused_qkv.fused_qkv_bwd_post,
                  eps=1e-6,
                  rope_max_timescale=5e6,
                  query_pre_attn_scalar=128**-0.5,
                  q_out_dtype=jnp.float8_e4m3fn,
                  out_qtype=jnp.float8_e5m2,
                  out_scale=224.0 / 57344.0,
                  block_m=512,
                  interpret=False,
              )
          )
          .trace(
              jax.ShapeDtypeStruct((2, 64, 512, 128), jnp.bfloat16),
              jax.ShapeDtypeStruct((2, 64, 512, 128), jnp.bfloat16),
              jax.ShapeDtypeStruct((2, 4, 512, 128), jnp.bfloat16),
              jax.ShapeDtypeStruct((2, 4, 512, 128), jnp.bfloat16),
              jax.ShapeDtypeStruct((2, 4, 512, 128), jnp.bfloat16),
              jax.ShapeDtypeStruct((128,), jnp.float32),
              jax.ShapeDtypeStruct((128,), jnp.float32),
              jax.ShapeDtypeStruct((1024,), jnp.int32),
          )
          .lower(lowering_platforms=("tpu",))
          .as_text()
      )
    self.assertIn("tpu_custom_call", aot_mlir)
    self.assertIn("fused_qkv_bwd_post", aot_mlir)

  def test_attention_f32_fused_out_fsdp_parity(self):
    self._run_attention_pair("float32", quantized=False, tol_y=1e-4, tol_grad=1e-4, fused_out_flag=True)

  def test_prequantized_x_matches_in_kernel_quantization(self):
    """`x_is_quantized=True` (fp8 `x` quantized outside the kernel) is bit-exact with in-kernel quantization."""
    batch, seq_len, emb, nq, nkv, hd = 1, 1024, 256, 8, 2, 128
    kx, kwq, _, _, kqn, kkn = jax.random.split(jax.random.PRNGKey(901), 6)
    x_2d = jax.random.normal(kx, (batch * seq_len, emb), jnp.bfloat16)
    act_scale = 224.0 / 448.0
    x_q = fused_qkv.static_quantize(x_2d, jnp.float8_e4m3fn, act_scale)
    w_qkv_q = jax.random.normal(kwq, (emb, (nq + 2 * nkv) * hd), jnp.float32).astype(jnp.float8_e4m3fn)
    qns = 1.0 + 0.1 * jax.random.normal(kqn, (hd,), jnp.float32)
    kns = 1.0 + 0.1 * jax.random.normal(kkn, (hd,), jnp.float32)
    pos_1d = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len)).reshape(-1)
    common_proj = {
        "num_q_heads": nq,
        "num_kv_heads": nkv,
        "head_dim": hd,
        "seq_len": seq_len,
        "eps": 1e-6,
        "rope_max_timescale": 1e6,
        "rope_min_timescale": 1.0,
        "query_pre_attn_scalar": hd**-0.5,
        "act_qtype": jnp.float8_e4m3fn,
        "act_scale": act_scale,
        "weight_scale": 0.5,
        "q_seq_minor": True,
        "k_seq_minor": True,
        "q_out_dtype": jnp.float8_e4m3fn,
        "k_out_dtype": jnp.float8_e4m3fn,
        "block_m": 256,
        "heads_per_tile": 4,
        "quant_rows": 128,
        "save_pre_norm": False,
        "interpret": True,
    }
    q0, k0, v0 = fused_qkv.fused_qkv_proj(x_2d, w_qkv_q, qns, kns, pos_1d, x_is_quantized=False, **common_proj)
    q1, k1, v1 = fused_qkv.fused_qkv_proj(
        x_q, w_qkv_q, qns, kns, pos_1d, x_is_quantized=True, compute_dtype=jnp.bfloat16, **common_proj
    )
    np.testing.assert_array_equal(np.asarray(q1, np.float32), np.asarray(q0, np.float32))
    np.testing.assert_array_equal(np.asarray(k1, np.float32), np.asarray(k0, np.float32))
    np.testing.assert_array_equal(np.asarray(v1, np.float32), np.asarray(v0, np.float32))

  def test_fused_out_subnormal_cotangent_is_finite(self):
    """Regression (#26): the dynamic absmax cotangent scale of `fused_out_local` must not produce NaN or Inf."""
    nq, hd, emb = 8, 128, 256
    ctx = jax.random.normal(jax.random.PRNGKey(905), (1, 128, nq, hd), jnp.bfloat16)
    w_o = jax.random.normal(jax.random.PRNGKey(906), (nq, hd, emb), jnp.float32) * 0.5
    p_sub = fused_qkv.FusedOutTrainParams(
        compute_dtype=jnp.bfloat16,
        act_qtype=jnp.float8_e4m3fn,
        weight_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        act_scale=0.5,
        weight_scale=0.5,
        bwd_scale=None,
        ctx_dtype=jnp.bfloat16,
        wo_dtype=jnp.float32,
    )
    _, sub_vjp = jax.vjp(lambda c_, w_: fused_qkv.fused_out_local(p_sub, c_, w_), ctx, w_o)
    dc_sub, dw_sub = sub_vjp(jnp.full((1, 128, emb), 1e-36, dtype=jnp.bfloat16))
    self.assertTrue(np.all(np.isfinite(np.asarray(dc_sub, np.float32))))
    self.assertTrue(np.all(np.isfinite(np.asarray(dw_sub, np.float32))))

  def test_deferred_norm_grad_allreduce_scan_parity(self):
    """Verifies `defer_norm_grad_allreduce=True` matches `False` across a 2-layer scanned block."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel
    from maxtext.layers import normalizations  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    emb, seq_len, nq, nkv, hd = 512, 128, 8, 2, 128

    def make_cfg(defer_flag: bool, cfg_dtype: str = "bfloat16"):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="defer_norm_test",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          decoder_block="qwen3",
          attention="dot_product",
          scan_layers=True,
          base_num_decoder_layers=2,
          base_emb_dim=emb,
          base_mlp_dim=512,
          base_num_query_heads=nq,
          base_num_kv_heads=nkv,
          head_dim=hd,
          use_qk_norm=True,
          rope_max_timescale=1_000_000.0,
          max_target_length=seq_len,
          per_device_batch_size=1,
          ici_fsdp_parallelism=4,
          dtype=cfg_dtype,
          weight_dtype="float32",
          matmul_precision="highest",
          use_fused_qkv_kernel=True,
          use_fused_out_kernel=True,
          defer_norm_grad_allreduce=defer_flag,
      )

    cfg_base = make_cfg(False, "bfloat16")
    cfg_defer = make_cfg(True, "bfloat16")
    cfg_f32_base = make_cfg(False, "float32")
    cfg_f32_defer = make_cfg(True, "float32")
    self.assertTrue(cfg_defer.defer_norm_grad_allreduce)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
    batch = int(cfg_base.per_device_batch_size) * jax.device_count()
    x_shape = (batch, seq_len, emb)
    kx, kc = jax.random.split(jax.random.PRNGKey(1301))
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
    x_emb_f32 = jax.random.normal(kx, x_shape, jnp.float32)
    cot = jax.random.normal(kc, x_shape, jnp.float32)

    class TwoBlockLayer(nnx.Module):
      """Norm -> attention -> norm block with optionally deferred norm-scale gradient all-reduces."""

      def __init__(self, cfg, rngs):
        defer_ar = bool(normalizations.get_deferred_norm_psum_axes(cfg, mesh))
        self.pre_self_attention_layer_norm = normalizations.RMSNorm(
            num_features=emb,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            shard_mode=cfg.shard_mode,
            kernel_axes=("norm",),
            epsilon=cfg.normalization_layer_epsilon,
            mesh=mesh,
            defer_scale_grad_allreduce=defer_ar,
            rngs=rngs,
        )
        self.self_attention = attentions.Attention(
            config=cfg,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            max_target_length=seq_len,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=x_shape,
            inputs_kv_shape=x_shape,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            use_qk_norm=True,
            query_pre_attn_scalar=hd**-0.5,
            model_mode=MODEL_MODE_TRAIN,
            defer_qk_norm_grad_allreduce=defer_ar,
            rngs=rngs,
        )
        self.post_self_attention_layer_norm = normalizations.RMSNorm(
            num_features=emb,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            shard_mode=cfg.shard_mode,
            kernel_axes=("norm",),
            epsilon=cfg.normalization_layer_epsilon,
            mesh=mesh,
            defer_scale_grad_allreduce=defer_ar,
            rngs=rngs,
        )

      def __call__(self, h):
        attn_in = self.pre_self_attention_layer_norm(h)
        attn_out, _ = self.self_attention(attn_in, attn_in, inputs_positions=pos, model_mode=MODEL_MODE_TRAIN)
        h = h + attn_out
        return h + self.post_self_attention_layer_norm(h)

    def run_scanned(cfg):
      x_in = x_emb_f32.astype(cfg.dtype)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        defer_axes = normalizations.get_deferred_norm_psum_axes(cfg, mesh)
        l0 = TwoBlockLayer(cfg, nnx.Rngs(params=0))
        l1 = TwoBlockLayer(cfg, nnx.Rngs(params=1))
        graphdef, p0, rest0 = nnx.split(l0, nnx.Param, ...)
        _, p1, _ = nnx.split(l1, nnx.Param, ...)

        def _stack_and_perturb(a, b):
          va, vb = a.get_value(), b.get_value()
          if va.ndim == 1:
            ramp = jnp.linspace(0.75, 1.25, va.shape[0], dtype=va.dtype)
            va, vb = va * ramp, vb * (2.0 - ramp)
          return a.replace(jnp.stack([va, vb], axis=0))

        stacked_params = jax.tree.map(
            _stack_and_perturb,
            p0,
            p1,
            is_leaf=lambda v: isinstance(v, nnx.Variable),
        )
        defer_targets = {
            ("pre_self_attention_layer_norm", "scale"),
            ("post_self_attention_layer_norm", "scale"),
            ("self_attention", "query_norm", "scale"),
            ("self_attention", "key_norm", "scale"),
        }

        def loss_fn(sp, x_):
          if defer_axes:

            def _wrap(path, leaf):
              if not isinstance(leaf, nnx.Variable):
                return leaf
              keys = tuple(getattr(k, "key", getattr(k, "name", str(k))) for k in path)
              if keys in defer_targets:
                return leaf.replace(normalizations.post_scan_psum_grad(leaf.get_value(), mesh, defer_axes))
              return leaf

            sp = jax.tree_util.tree_map_with_path(_wrap, sp, is_leaf=lambda v: isinstance(v, nnx.Variable))

          @jax.checkpoint
          def body(carry, layer_p):
            m = nnx.merge(graphdef, layer_p, rest0)
            return m(carry), None

          y_out, _ = jax.lax.scan(body, x_, sp)
          return jnp.sum(y_out.astype(jnp.float32) * cot), y_out

        jit_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
        hlo = jit_fn.lower(stacked_params, x_in).compile().as_text()
        (_, y_out), (gp, gx) = jit_fn(stacked_params, x_in)
      flat_gp = {
          jax.tree_util.keystr(k): np.asarray(v, np.float32) for k, v in jax.tree_util.tree_flatten_with_path(gp)[0]
      }
      return np.asarray(y_out, np.float32), flat_gp, np.asarray(gx, np.float32), hlo

    # 1. Exact float32 parity across all activations + all 4 norm scale gradients (`1e-5`).
    yf0, gpf0, gxf0, _ = run_scanned(cfg_f32_base)
    yf1, gpf1, gxf1, _ = run_scanned(cfg_f32_defer)
    np.testing.assert_allclose(yf1, yf0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gxf1, gxf0, rtol=1e-5, atol=1e-5)
    for k in gpf0:
      np.testing.assert_allclose(gpf1[k], gpf0[k], rtol=1e-5, atol=1e-5, err_msg=k)

    # 2. bfloat16 parity (`_scale_mul_deferred_grad` accumulates `ds_l` in fp32 rather than bf16, matching f32 closer!).
    y0, gp0, gx0, hlo0 = run_scanned(cfg_base)
    y1, gp1, gx1, hlo1 = run_scanned(cfg_defer)
    self.assertNotEmpty(normalizations.get_deferred_norm_psum_axes(cfg_defer, mesh))
    np.testing.assert_allclose(y1, y0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gx1, gx0, rtol=1e-4, atol=1e-4)
    for k in gp0:
      if "attention_layer_norm" in k:
        self.assertLess(_errors(gp1[k], gp0[k])[1], 5e-3, msg=k)
        self.assertLess(_errors(gp1[k], gpf0[k])[1], _errors(gp0[k], gpf0[k])[1] + 1e-3, msg=k)
      else:
        np.testing.assert_allclose(gp1[k], gp0[k], rtol=1e-4, atol=1e-4, err_msg=k)
    # Baseline emits per-layer `f32[512]` / `f32[128]` all-reduces inside the scan body (combined by XLA);
    # deferred mode eliminates both from the scan body and emits post-scan `[2, D]` all-reduces instead!
    self.assertRegex(hlo0, r"f32\[512\]\{0\}.*all-reduce\(")
    self.assertNotRegex(hlo1, r"f32\[512\]\{0\}.*all-reduce\(")
    self.assertNotRegex(hlo1, r"f32\[128\]\{0\}.*all-reduce\(")
    self.assertNotRegex(hlo0, r"f32\[2,512\]\{1,0\}.*all-reduce\(")
    self.assertRegex(hlo1, r"f32\[2,512\]\{1,0\}.*all-reduce\(")
    self.assertRegex(hlo1, r"f32\[2,128\]\{1,0\}.*all-reduce\(")

  @parameterized.named_parameters(("unfused_qkv", False), ("fused_qkv", True))
  def test_nnx_decoder_defer_norm_grad_allreduce_parity(self, use_fused_qkv: bool):
    """`defer_norm_grad_allreduce` on vs. off through a scanned `NNXDecoder`: same output and gradients.

    Differentiates through the decoder itself, so it covers what the flag needs in order to be correct: the
    `layer_fn` overrides that switch the layers to shard-local scale gradients, the key paths under which
    `NNXDecoder` finds and wraps every deferred scale, and the psum axes. It also shows that the comparison notices
    a missing wrapper, and that `NNXDecoder` refuses a deferred scale it cannot find.
    """
    from unittest import mock  # pylint: disable=import-outside-toplevel

    from flax import linen as nn  # pylint: disable=import-outside-toplevel
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import nnx_decoders  # pylint: disable=import-outside-toplevel
    from maxtext.layers import normalizations  # pylint: disable=import-outside-toplevel
    from maxtext.layers.embeddings import Embed  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    emb, seq_len, nq, nkv, hd, vocab = 512, 128, 8, 2, 128, 128

    def make_cfg(defer_flag: bool):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="defer_norm_decoder_test",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          decoder_block="qwen3",
          attention="dot_product",
          scan_layers=True,
          base_num_decoder_layers=2,
          base_emb_dim=emb,
          base_mlp_dim=512,
          base_num_query_heads=nq,
          base_num_kv_heads=nkv,
          head_dim=hd,
          use_qk_norm=True,
          rope_max_timescale=1_000_000.0,
          max_target_length=seq_len,
          per_device_batch_size=1,
          ici_fsdp_parallelism=4,
          vocab_size=vocab,
          dtype="float32",
          weight_dtype="float32",
          matmul_precision="highest",
          use_fused_qkv_kernel=use_fused_qkv,
          use_fused_out_kernel=use_fused_qkv,
          defer_norm_grad_allreduce=defer_flag,
      )

    cfg_base, cfg_defer = make_cfg(False), make_cfg(True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
    batch = int(cfg_base.per_device_batch_size) * jax.device_count()
    k_tok, k_cot = jax.random.split(jax.random.PRNGKey(1305))
    toks = jax.random.randint(k_tok, (batch, seq_len), 0, vocab, jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
    seg = jnp.ones((batch, seq_len), jnp.int32)
    cot = jax.random.normal(k_cot, (batch, seq_len, vocab), jnp.float32)
    norm_scale_paths = [
        ("pre_self_attention_layer_norm", "scale"),
        ("post_self_attention_layer_norm", "scale"),
        ("self_attention", "query_norm", "scale"),
        ("self_attention", "key_norm", "scale"),
    ]

    def eval_decoder(cfg):
      with mesh, nn.logical_axis_rules(cfg.logical_axis_rules):
        shared_emb = Embed(
            num_embeddings=vocab,
            num_features=emb,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            config=cfg,
            mesh=mesh,
            rngs=nnx.Rngs(params=1),
        )
        dec = nnx_decoders.NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
        nnx.pop(dec, nnx.Intermediate)
        graphdef, params, rest = nnx.split(dec, nnx.Param, ...)

        def loss_fn(p_):
          m = nnx.merge(graphdef, p_, rest)
          y, _, _ = m(shared_emb, toks, pos, decoder_segment_ids=seg, deterministic=True, model_mode=MODEL_MODE_TRAIN)
          return jnp.sum(y.astype(jnp.float32) * cot), y

        (_, y), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
      flat = {
          jax.tree_util.keystr(k): np.asarray(v, np.float32) for k, v in jax.tree_util.tree_flatten_with_path(grads)[0]
      }
      return np.asarray(y, np.float32), flat, (dec, params)

    def scale_keys(grads):
      return [k for k in grads if any(all(name in k for name in path) for path in norm_scale_paths)]

    y_base, g_base, _ = eval_decoder(cfg_base)
    y_defer, g_defer, (dec_defer, params_defer) = eval_decoder(cfg_defer)

    # The deferred mode is really on: active psum axes, and (below) shard-local scale gradients once the post-scan
    # all-reduce is removed. The layers themselves are built with the deferral off; `_apply_layers_sequentially`
    # turns it on per call.
    defer_axes = normalizations.get_deferred_norm_psum_axes(cfg_defer, mesh)
    self.assertNotEmpty(defer_axes)
    self.assertEmpty(normalizations.get_deferred_norm_psum_axes(cfg_base, mesh))
    self.assertFalse(dec_defer.layers.pre_self_attention_layer_norm.defer_scale_grad_allreduce)
    self.assertFalse(dec_defer.layers.post_self_attention_layer_norm.defer_scale_grad_allreduce)

    self.assertEqual(set(g_base), set(g_defer))
    self.assertLen(scale_keys(g_base), 4)
    self.assertLess(_errors(y_defer, y_base)[1], 1e-5)
    for k in g_base:
      self.assertLess(_errors(g_defer[k], g_base[k])[1], 1e-4, msg=k)

    # Without the post-scan wrapper the deferred scale gradients are per-shard, and the comparison sees it. Only
    # the scales that the layers actually defer (the QK-norm ones need the fused QKV path) are affected.
    with mock.patch.object(normalizations, "post_scan_psum_grad", side_effect=lambda w, *_: w):
      _, g_unwrapped, _ = eval_decoder(cfg_defer)
    deferred = [k for k in scale_keys(g_base) if use_fused_qkv or "attention_layer_norm" in k]
    self.assertLen(deferred, 4 if use_fused_qkv else 2)
    for k in deferred:
      self.assertGreater(_errors(g_unwrapped[k], g_base[k])[1], 0.1, msg=k)

    # `NNXDecoder` raises instead of leaving a deferred scale unwrapped when its key path is not in the params.
    with self.assertRaisesRegex(ValueError, "no parameter found for the deferred norm scale"):
      nnx_decoders._wrap_deferred_norm_scales(  # pylint: disable=protected-access
          params_defer["layers"], {("no_such_layer_norm", "scale")}, mesh, defer_axes
      )

  def test_attention_fp8_qk_log2e_and_config_guards(self):
    """Verifies `experimental_sa_quant_q/k_fp8`, `sa_q_prescaled_log2e`, and TP guard."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    emb, seq_len, nq, nkv, hd = 256, 128, 4, 2, 128

    def init_cfg(**extra):
      return pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="fp8_qk_guard_test",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          decoder_block="qwen3",
          base_emb_dim=emb,
          base_num_query_heads=nq,
          base_num_kv_heads=nkv,
          head_dim=hd,
          use_qk_norm=True,
          max_target_length=seq_len,
          max_prefill_predict_length=seq_len,
          per_device_batch_size=1,
          dtype="bfloat16",
          weight_dtype="float32",
          **extra,
      )

    # 2. `use_fused_qkv_kernel=True` with `ici_tensor_parallelism > 1` raises `NotImplementedError`.
    cfg_tp = init_cfg(
        attention="dot_product", ici_fsdp_parallelism=2, ici_tensor_parallelism=2, use_fused_qkv_kernel=True
    )
    mesh_tp = Mesh(maxtext_utils.create_device_mesh(cfg_tp), cfg_tp.mesh_axes)
    with mesh_tp, nn_partitioning.axis_rules(cfg_tp.logical_axis_rules):
      attn_tp = attentions.Attention(
          config=cfg_tp,
          num_query_heads=nq,
          num_kv_heads=nkv,
          head_dim=hd,
          max_target_length=seq_len,
          mesh=mesh_tp,
          attention_kernel="dot_product",
          inputs_q_shape=(4, seq_len, emb),
          inputs_kv_shape=(4, seq_len, emb),
          dtype=cfg_tp.dtype,
          weight_dtype=cfg_tp.weight_dtype,
          use_qk_norm=True,
          query_pre_attn_scalar=hd**-0.5,
          rngs=nnx.Rngs(params=0),
      )
      x_tp = jnp.zeros((4, seq_len, emb), jnp.bfloat16)
      with self.assertRaisesRegex(NotImplementedError, "requires TP=1"):
        attn_tp(
            x_tp,
            x_tp,
            inputs_positions=jnp.zeros((4, seq_len), jnp.int32),
            model_mode=MODEL_MODE_TRAIN,
        )

    # 2b. `defer_qk_norm_grad_allreduce=True` without the fused QKV path fails fast in training (the standard path
    # would reduce the QK-norm scale gradients itself and the post-scan all-reduce would count them twice).
    cfg_nofused = init_cfg(attention="dot_product", ici_fsdp_parallelism=4, use_fused_qkv_kernel=False)
    mesh_nofused = Mesh(maxtext_utils.create_device_mesh(cfg_nofused), cfg_nofused.mesh_axes)
    with mesh_nofused, nn_partitioning.axis_rules(cfg_nofused.logical_axis_rules):
      attn_nofused = attentions.Attention(
          config=cfg_nofused,
          num_query_heads=nq,
          num_kv_heads=nkv,
          head_dim=hd,
          max_target_length=seq_len,
          mesh=mesh_nofused,
          attention_kernel="dot_product",
          inputs_q_shape=(4, seq_len, emb),
          inputs_kv_shape=(4, seq_len, emb),
          dtype=cfg_nofused.dtype,
          weight_dtype=cfg_nofused.weight_dtype,
          use_qk_norm=True,
          query_pre_attn_scalar=hd**-0.5,
          defer_qk_norm_grad_allreduce=True,
          rngs=nnx.Rngs(params=0),
      )
      x_nofused = jnp.zeros((4, seq_len, emb), jnp.bfloat16)
      with self.assertRaisesRegex(ValueError, "requires use_fused_qkv_kernel=True"):
        attn_nofused(
            x_nofused,
            x_nofused,
            inputs_positions=jnp.zeros((4, seq_len), jnp.int32),
            model_mode=MODEL_MODE_TRAIN,
        )

    # 3. `experimental_sa_quant_q/k_fp8` and `sa_q_prescaled_log2e` parity between fused and unfused `Attention`,
    # plus `MODEL_MODE_PREFILL` `log2(e)` prescaling!
    q_unfused_train = k_unfused_train = v_unfused_train = None
    for fused_flag in (False, True):
      cfg_qk = init_cfg(
          attention="flash",
          use_tokamax_splash=True,
          sa_use_base2_exp=True,
          sa_q_prescaled_log2e=True,
          experimental_sa_quant_q_fp8=True,
          experimental_sa_quant_k_fp8=True,
          sa_q_layout="SEQ_MINOR",
          sa_k_layout="SEQ_MINOR",
          use_fused_qkv_kernel=fused_flag,
          ici_fsdp_parallelism=4,
      )
      mesh_qk = Mesh(maxtext_utils.create_device_mesh(cfg_qk), cfg_qk.mesh_axes)
      with mesh_qk, nn_partitioning.axis_rules(cfg_qk.logical_axis_rules):
        attn_qk = attentions.Attention(
            config=cfg_qk,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_dim=hd,
            max_target_length=seq_len,
            max_prefill_predict_length=seq_len,
            mesh=mesh_qk,
            attention_kernel="flash",
            inputs_q_shape=(4, seq_len, emb),
            inputs_kv_shape=(4, seq_len, emb),
            dtype=cfg_qk.dtype,
            weight_dtype=cfg_qk.weight_dtype,
            use_qk_norm=True,
            query_pre_attn_scalar=hd**-0.5,
            rngs=nnx.Rngs(params=0),
        )
        captured = {}
        attn_qk.update_kv_caches = lambda *a, **kw: [None, None]
        attn_qk.attention_op = lambda q, k, v, *a, _cap=captured, **kw: (
            _cap.update(q=q, k=k, v=v) or v.repeat(2, axis=2)
        )
        x_in = jax.random.normal(jax.random.PRNGKey(55), (4, seq_len, emb), jnp.bfloat16)
        pos_in = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (4, seq_len))
        attn_qk(x_in, x_in, inputs_positions=pos_in, model_mode=MODEL_MODE_TRAIN)
        self.assertEqual(captured["q"].dtype, jnp.float8_e4m3fn)
        self.assertEqual(captured["k"].dtype, jnp.float8_e4m3fn)
        if not fused_flag:
          q_unfused_train = np.asarray(captured["q"], np.float32)
          k_unfused_train = np.asarray(captured["k"], np.float32)
          v_unfused_train = np.asarray(captured["v"], np.float32)
          attn_qk(x_in, x_in, inputs_positions=pos_in, model_mode=MODEL_MODE_PREFILL)
          q_unfused_prefill = np.asarray(captured["q"], np.float32)
          # Prefill must also include the log2(e) prescale (matching train up to fp8 vs bf16 rounding)!
          self.assertLess(_errors(q_unfused_prefill, q_unfused_train)[1], 0.05)
          attn_qk.config = init_cfg(
              attention="flash",
              use_tokamax_splash=True,
              ici_fsdp_parallelism=4,
              experimental_sa_quant_q_fp8=True,
              experimental_sa_quant_k_fp8=True,
              sa_q_prescaled_log2e=False,
              use_fused_qkv_kernel=False,
          )
          attn_qk(x_in, x_in, inputs_positions=pos_in, model_mode=MODEL_MODE_PREFILL)
          q_unscaled_prefill = np.asarray(captured["q"], np.float32)
          self.assertLess(_errors(q_unfused_prefill, q_unscaled_prefill * float(np.log2(np.e)))[1], 3e-2)
        else:
          assert q_unfused_train is not None
          self.assertLess(_errors(captured["q"], q_unfused_train)[1], 0.03)
          self.assertLess(_errors(captured["k"], k_unfused_train)[1], 0.03)
          self.assertLess(_errors(captured["v"], v_unfused_train)[1], 1e-2)


if __name__ == "__main__":
  absltest.main()
