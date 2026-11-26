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

"""Unit tests for GatedDeltaNet deferred GQA repeat, projection/norm layouts, and FP8 projections."""

import os
import subprocess
import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import yaml

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.attention import gdn_fp8_projections
from maxtext.models.qwen3 import (
    Qwen3NextGatedDeltaNet,
    jax_ar_gated_delta_rule,
    jax_chunk_gated_delta_rule,
)
from maxtext.utils.sharding import nnx_construct_named_sharding

_BASE_YML = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  af = np.asarray(a, dtype=np.float64).ravel()
  bf = np.asarray(b, dtype=np.float64).ravel()
  denom = max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-30)
  return float(np.dot(af, bf) / denom)


def _eval_gdn_layer(cfg, x, seg_ids=None, mesh=None):
  """Evaluates forward output and parameter/input gradients of Qwen3NextGatedDeltaNet."""
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    layer = Qwen3NextGatedDeltaNet(
        config=cfg,
        mesh=mesh,
        dtype=jnp.float32,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(13),
    )
    graphdef, state = nnx.split(layer)
    if mesh is not None:
      state_shardings = nnx_construct_named_sharding(state, mesh)
      state = jax.device_put(state, state_shardings)

    def loss_fn(st, inp):
      mod = nnx.merge(graphdef, st)
      out, _ = mod(inp, decoder_segment_ids=seg_ids, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(out * jnp.sin(inp)), out

    (_, out), (grad_state, grad_x) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(state, x)
    return (
        jax.device_get(out),
        jax.tree_util.tree_map(jax.device_get, grad_state),
        jax.device_get(grad_x),
    )


class GdnLayerLayoutsTest(unittest.TestCase):
  """Tests deferred GQA repeat, layout constraints, kernel-order in_proj, FP8 projections, and config guards."""

  def _make_cfg(self, **overrides):
    """Builds a minimal Qwen3Next MaxTextConfig for GDN layer layout tests."""
    defaults = {
        "run_name": "gdn_layer_layouts_test",
        "decoder_block": "qwen3_next",
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "base_emb_dim": 32,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 8,
        "gdn_key_head_dim": 16,
        "gdn_value_head_dim": 16,
        "gdn_conv_kernel_dim": 4,
        "gdn_chunk_size": 4,
        "use_qk_norm_in_gdn": True,
        "context_parallel_load_balance": False,
        "enable_checkpointing": False,
    }
    defaults.update(overrides)
    return pyconfig.initialize([None, _BASE_YML], **defaults)

  def test_deferred_gqa_repeat_equivalence(self):
    batch, seq, h_k, h_v, d_k, d_v = 2, 10, 2, 8, 16, 16
    repeats = h_v // h_k
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    q = jax.random.normal(keys[0], (batch, seq, h_k, d_k), jnp.float32) * 0.2
    k = jax.random.normal(keys[1], (batch, seq, h_k, d_k), jnp.float32) * 0.2
    v = jax.random.normal(keys[2], (batch, seq, h_v, d_v), jnp.float32) * 0.2
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq, h_v), jnp.float32))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq, h_v), jnp.float32))
    h0 = jax.random.normal(keys[5], (batch, h_v, d_k, d_v), jnp.float32) * 0.1
    seg_ids = jnp.array([[1, 1, 1, 1, 2, 2, 2, 2, 0, 0], [0, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=jnp.int32)

    for seg in (None, seg_ids):

      def deferred_loss(q_in, k_in, v_in, g_in, b_in, h_in, seg_arg=seg):
        out, st = jax_chunk_gated_delta_rule(
            q_in,
            k_in,
            v_in,
            g_in,
            b_in,
            chunk_size=4,
            initial_state=h_in,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
            segment_ids=seg_arg,
        )
        return jnp.sum(out**2) + jnp.sum(st**2), (out, st)

      def repeated_loss(q_in, k_in, v_in, g_in, b_in, h_in, seg_arg=seg):
        out, st = jax_chunk_gated_delta_rule(
            jnp.repeat(q_in, repeats, axis=2),
            jnp.repeat(k_in, repeats, axis=2),
            v_in,
            g_in,
            b_in,
            chunk_size=4,
            initial_state=h_in,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
            segment_ids=seg_arg,
        )
        return jnp.sum(out**2) + jnp.sum(st**2), (out, st)

      (_, (out_def, st_def)), grads_def = jax.jit(
          jax.value_and_grad(deferred_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)
      )(q, k, v, g, beta, h0)
      (_, (out_rep, st_rep)), grads_rep = jax.jit(
          jax.value_and_grad(repeated_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)
      )(q, k, v, g, beta, h0)

      np.testing.assert_allclose(jax.device_get(out_def), jax.device_get(out_rep), atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(jax.device_get(st_def), jax.device_get(st_rep), atol=1e-5, rtol=1e-5)
      for gd, gr in zip(grads_def, grads_rep):
        np.testing.assert_allclose(jax.device_get(gd), jax.device_get(gr), atol=1e-5, rtol=1e-5)

    # Autoregressive seq_len == 1 deferred GQA check.
    ar_def_out, ar_def_st = jax.jit(
        lambda: jax_ar_gated_delta_rule(
            q[:, :1],
            k[:, :1],
            v[:, :1],
            g[:, :1],
            beta[:, :1],
            initial_state=h0,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
        )
    )()
    ar_rep_out, ar_rep_st = jax.jit(
        lambda: jax_ar_gated_delta_rule(
            jnp.repeat(q[:, :1], repeats, axis=2),
            jnp.repeat(k[:, :1], repeats, axis=2),
            v[:, :1],
            g[:, :1],
            beta[:, :1],
            initial_state=h0,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
        )
    )()
    np.testing.assert_allclose(jax.device_get(ar_def_out), jax.device_get(ar_rep_out), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(jax.device_get(ar_def_st), jax.device_get(ar_rep_st), atol=1e-5, rtol=1e-5)

  def test_in_proj_kernel_order_matches_split_path(self):
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 32), jnp.float32) * 0.2
    seg_ids = jnp.array([[1, 1, 1, 1, 2, 2, 2, 0], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32)
    cfg_split = self._make_cfg(gdn_in_proj_kernel_order=False)
    cfg_korder = self._make_cfg(gdn_in_proj_kernel_order=True)

    out_split, gs_split, gx_split = _eval_gdn_layer(cfg_split, x, seg_ids)
    out_korder, gs_korder, gx_korder = _eval_gdn_layer(cfg_korder, x, seg_ids)

    np.testing.assert_allclose(out_korder, out_split, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(gx_korder, gx_split, atol=1e-5, rtol=1e-5)
    for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_korder), jax.tree_util.tree_leaves(gs_split)):
      np.testing.assert_allclose(g_got, g_ref, atol=1e-5, rtol=1e-5)

  def test_out_proj_token_minor_and_gated_norm_layouts(self):
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 8, 32), jnp.float32) * 0.2
    cfg_base = self._make_cfg()
    out_ref, gs_ref, gx_ref = _eval_gdn_layer(cfg_base, x)

    for norm_layout in ("auto", "natural", "natural_fwd_bwd", "token_minor"):
      cfg_layout = self._make_cfg(gdn_out_proj_token_minor=True, gdn_gated_norm_layout=norm_layout)
      out_got, gs_got, gx_got = _eval_gdn_layer(cfg_layout, x)
      np.testing.assert_allclose(out_got, out_ref, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(gx_got, gx_ref, atol=1e-5, rtol=1e-5)
      for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_got), jax.tree_util.tree_leaves(gs_ref)):
        np.testing.assert_allclose(g_got, g_ref, atol=1e-5, rtol=1e-5)

  def test_fp8_projections_and_pallas_transpose_quant(self):
    # 1. Pallas interpret transpose_quant vs XLA fallback (exact match on 128-divisible dims).
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    src = jax.random.normal(keys[0], (128, 256), jnp.bfloat16)
    row_s = [jax.random.uniform(keys[1], (128, 1), jnp.float32, minval=0.5, maxval=2.0)]
    col_s = [jax.random.uniform(keys[2], (256,), jnp.float32, minval=0.5, maxval=2.0)]

    tq_xla = gdn_fp8_projections.transpose_quant(src, row_s, col_s, [gdn_fp8_projections.E4M3], use_pallas=False)[0]
    tq_pallas = gdn_fp8_projections.transpose_quant(
        src, row_s, col_s, [gdn_fp8_projections.E4M3], use_pallas=True, interpret=True
    )[0]
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(tq_pallas.astype(jnp.float32))),
        np.asarray(jax.device_get(tq_xla.astype(jnp.float32))),
    )

    # 2. Standalone narrow (K <= N) and wide (K > N) fp8_projection forward + backward vs float32 matmul.
    for k_dim, n_dim, splits in ((32, 96, (64,)), (96, 32, ())):
      x_mat = jax.random.normal(jax.random.PRNGKey(6 + k_dim), (4, 8, k_dim), jnp.float32) * 0.5
      w_mat = jax.random.normal(jax.random.PRNGKey(7 + n_dim), (k_dim, n_dim), jnp.float32) * 0.2

      def f32_loss(xm, wm, sp=splits):
        bounds = (0,) + sp + (wm.shape[1],)
        outs = [xm @ wm[:, lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])]
        return sum(jnp.sum(o * jnp.arange(1, o.shape[-1] + 1, dtype=jnp.float32)) for o in outs), outs

      def fp8_loss(xm, wm, sp=splits):
        outs = gdn_fp8_projections.fp8_projection(xm, wm, sp)
        return sum(jnp.sum(o * jnp.arange(1, o.shape[-1] + 1, dtype=jnp.float32)) for o in outs), outs

      (_, outs_ref), (dx_ref, dw_ref) = jax.jit(jax.value_and_grad(f32_loss, argnums=(0, 1), has_aux=True))(x_mat, w_mat)
      (_, outs_fp8), (dx_fp8, dw_fp8) = jax.jit(jax.value_and_grad(fp8_loss, argnums=(0, 1), has_aux=True))(x_mat, w_mat)
      for o_got, o_ref in zip(outs_fp8, outs_ref):
        self.assertGreater(_cosine_similarity(jax.device_get(o_got), jax.device_get(o_ref)), 0.995)
        self.assertLess(float(jnp.linalg.norm(o_got - o_ref) / jnp.linalg.norm(o_ref)), 0.1)
      self.assertGreater(_cosine_similarity(jax.device_get(dx_fp8), jax.device_get(dx_ref)), 0.995)
      self.assertLess(float(jnp.linalg.norm(dx_fp8 - dx_ref) / jnp.linalg.norm(dx_ref)), 0.1)
      self.assertGreater(_cosine_similarity(jax.device_get(dw_fp8), jax.device_get(dw_ref)), 0.995)
      self.assertLess(float(jnp.linalg.norm(dw_fp8 - dw_ref) / jnp.linalg.norm(dw_ref)), 0.1)

    # 2b. Standalone col_perm_groups forward + backward (unpermute_grouped_columns).
    perm_groups = (2, (16, 16, 16, 16))
    perm_layout = gdn_fp8_projections.ProjectionLayout(col_perm_groups=perm_groups)
    x_perm = jax.random.normal(jax.random.PRNGKey(41), (4, 8, 32), jnp.float32) * 0.5
    w_perm = jax.random.normal(jax.random.PRNGKey(42), (32, 128), jnp.float32) * 0.2

    def _perm_ref_loss(xm, wm):
      wp = gdn_fp8_projections.permute_grouped_columns(wm, *perm_groups)
      return jnp.sum((xm @ wp[:, :96]) * 1.5) + jnp.sum((xm @ wp[:, 96:]) * 0.7), (xm @ wp[:, :96], xm @ wp[:, 96:])

    def _perm_fp8_loss(xm, wm):
      outs = gdn_fp8_projections.fp8_projection(xm, wm, (96,), perm_layout)
      return jnp.sum(outs[0] * 1.5) + jnp.sum(outs[1] * 0.7), outs

    (_, outs_perm_ref), (dx_perm_ref, dw_perm_ref) = jax.jit(
        jax.value_and_grad(_perm_ref_loss, argnums=(0, 1), has_aux=True)
    )(x_perm, w_perm)
    (_, outs_perm_fp8), (dx_perm_fp8, dw_perm_fp8) = jax.jit(
        jax.value_and_grad(_perm_fp8_loss, argnums=(0, 1), has_aux=True)
    )(x_perm, w_perm)
    for o_got, o_ref in zip(outs_perm_fp8, outs_perm_ref):
      self.assertGreater(_cosine_similarity(jax.device_get(o_got), jax.device_get(o_ref)), 0.995)
      self.assertLess(float(jnp.linalg.norm(o_got - o_ref) / jnp.linalg.norm(o_ref)), 0.1)
    self.assertGreater(_cosine_similarity(jax.device_get(dx_perm_fp8), jax.device_get(dx_perm_ref)), 0.995)
    self.assertLess(float(jnp.linalg.norm(dx_perm_fp8 - dx_perm_ref) / jnp.linalg.norm(dx_perm_ref)), 0.1)
    self.assertGreater(_cosine_similarity(jax.device_get(dw_perm_fp8), jax.device_get(dw_perm_ref)), 0.995)
    self.assertLess(float(jnp.linalg.norm(dw_perm_fp8 - dw_perm_ref) / jnp.linalg.norm(dw_perm_ref)), 0.1)

    # 2c. Offline XAOT compilation on CPU host targeting TPU selects Pallas via pltpu.is_tpu_device().
    from unittest import mock  # pylint: disable=import-outside-toplevel

    with (
        mock.patch.object(gdn_fp8_projections.pltpu, "is_tpu_device", return_value=True),
        mock.patch.object(
            gdn_fp8_projections, "_tq_pallas_2d", wraps=gdn_fp8_projections._tq_pallas_2d  # pylint: disable=protected-access
        ) as spy_tq,
    ):
      tq_xaot = gdn_fp8_projections.transpose_quant(
          src, row_s, col_s, [gdn_fp8_projections.E4M3], use_pallas=None, interpret=True
      )[0]
      self.assertEqual(spy_tq.call_count, 1)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(tq_xaot.astype(jnp.float32))),
        np.asarray(jax.device_get(tq_xla.astype(jnp.float32))),
    )

    # 3. End-to-end Qwen3NextGatedDeltaNet with gdn_fp8_projections (plain and kernel-order).
    x_layer = jax.random.normal(jax.random.PRNGKey(9), (2, 8, 32), jnp.float32) * 0.3
    out_f32, gs_f32, gx_f32 = _eval_gdn_layer(self._make_cfg(), x_layer)
    for korder in (False, True):
      cfg_fp8 = self._make_cfg(gdn_fp8_projections=True, gdn_in_proj_kernel_order=korder)
      out_fp8, gs_fp8, gx_fp8 = _eval_gdn_layer(cfg_fp8, x_layer)
      self.assertGreater(_cosine_similarity(out_fp8, out_f32), 0.99)
      self.assertGreater(_cosine_similarity(gx_fp8, gx_f32), 0.98)
      for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_fp8), jax.tree_util.tree_leaves(gs_f32)):
        self.assertGreater(_cosine_similarity(g_got, g_ref), 0.98)

  def test_embed_gdn_and_multidevice_sharding(self):
    cfg_default = self._make_cfg()
    layer_default = Qwen3NextGatedDeltaNet(config=cfg_default, dtype=jnp.float32, rngs=nnx.Rngs(0))
    self.assertEqual(layer_default.in_proj_qkvz.kernel_axes, ("embed_attn", "gdn_head"))
    self.assertEqual(layer_default.in_proj_ba.kernel_axes, ("embed_attn", "gdn_head"))
    self.assertEqual(layer_default.out_proj.kernel_axes, ("gdn_head", "embed_attn"))

    with open(_BASE_YML, "r", encoding="utf-8") as f:
      base_rules = yaml.safe_load(f)["logical_axis_rules"]
    custom_rules = list(base_rules) + [["embed_gdn", "fsdp"]]
    cfg_embed_gdn = self._make_cfg(logical_axis_rules=custom_rules)
    layer_embed_gdn = Qwen3NextGatedDeltaNet(config=cfg_embed_gdn, dtype=jnp.float32, rngs=nnx.Rngs(0))
    self.assertEqual(layer_embed_gdn.in_proj_qkvz.kernel_axes, ("embed_gdn", "gdn_head"))
    self.assertEqual(layer_embed_gdn.in_proj_ba.kernel_axes, ("embed_gdn", "gdn_head"))
    self.assertEqual(layer_embed_gdn.out_proj.kernel_axes, ("gdn_head", "embed_gdn"))

    devices = jax.devices()
    if len(devices) == 4 and devices[0].platform == "cpu":
      _run_multidevice_child()
      return
    env = os.environ.copy()
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, __file__, "--multidevice-child"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    self.assertIn("GDN_LAYER_LAYOUTS_SHARDING_OK", result.stdout)

  def test_tier2_config_validations(self):
    with open(_BASE_YML, "r", encoding="utf-8") as f:
      base_data = yaml.safe_load(f)
    for tier2_key in (
        "gdn_pin_in_proj_output_layout",
        "gdn_in_proj_kernel_order",
        "gdn_out_proj_token_minor",
        "gdn_gated_norm_layout",
        "gdn_fp8_projections",
    ):
      self.assertNotIn(tier2_key, base_data)

    with self.assertRaisesRegex(ValueError, "require a GatedDeltaNet decoder_block"):
      pyconfig.initialize(
          [None, _BASE_YML],
          run_name="non_gdn_bad",
          decoder_block="llama2",
          gdn_in_proj_kernel_order=True,
          enable_checkpointing=False,
      )

    with self.assertRaisesRegex(ValueError, "requires gdn_out_proj_token_minor=True"):
      self._make_cfg(gdn_gated_norm_layout="natural", gdn_out_proj_token_minor=False)

    with self.assertRaisesRegex(ValueError, "require shard_mode='auto'"):
      config_types.MaxTextConfig(
          run_name="explicit_bad",
          decoder_block="qwen3_next",
          shard_mode="explicit",
          gdn_fp8_projections=True,
          enable_checkpointing=False,
      )

    with self.assertRaisesRegex(ValueError, "incompatible with gdn_cp_mode='head'"):
      self._make_cfg(ici_context_parallelism=2, gdn_cp_mode="head", gdn_in_proj_kernel_order=True)

    with self.assertRaisesRegex(ValueError, "incompatible with gdn_cp_mode='head'"):
      config_types.RLConfig(
          run_name="rl_bad_korder",
          decoder_block="qwen3_next",
          ici_context_parallelism=2,
          context_parallel_load_balance=False,
          gdn_cp_mode="head",
          gdn_in_proj_kernel_order=True,
          enable_checkpointing=False,
      )

    for quant_kwargs in ({"quantization": "fp8_full"}, {"quantization": "int8", "use_qwix_quantization": False}):
      with self.assertRaisesRegex(ValueError, "incompatible with quantized in_proj_qkvz"):
        self._make_cfg(gdn_in_proj_kernel_order=True, gdn_fp8_projections=False, **quant_kwargs)
      with self.assertRaisesRegex(ValueError, "incompatible with quantized in_proj_qkvz"):
        config_types.RLConfig(
            run_name="rl_bad_quant_korder",
            decoder_block="qwen3_next",
            gdn_in_proj_kernel_order=True,
            gdn_fp8_projections=False,
            enable_checkpointing=False,
            **quant_kwargs,
        )
    self._make_cfg(gdn_in_proj_kernel_order=True, gdn_fp8_projections=True, quantization="fp8_full")
    self._make_cfg(
        gdn_in_proj_kernel_order=True,
        gdn_fp8_projections=False,
        quantization="int8",
        use_qwix_quantization=False,
        unquantized_modules=["in_proj_qkvz"],
    )

  def test_fp8_projections_coexist_with_qwix_and_non_qwix_quantization(self):
    # pylint: disable=import-outside-toplevel
    from unittest import mock
    import qwix
    from maxtext.layers import quantizations

    class _DecoderPathWrapper(nnx.Module):
      """Places GDN under `decoder.layers` so MaxText Qwix module_path rules (`decoder/.*layers.*`) match."""

      def __init__(self, gdn):
        self.decoder = nnx.Module()
        self.decoder.layers = gdn

      def __call__(self, inp):
        return self.decoder.layers(inp, model_mode=MODEL_MODE_TRAIN)[0]

    x = jax.random.normal(jax.random.PRNGKey(31), (2, 8, 32), jnp.bfloat16) * 0.3

    def eval_wrapped(wrapped_mod):
      gdef, st = nnx.split(wrapped_mod)

      def _wrapped_loss(s, inp):
        y = nnx.merge(gdef, s)(inp)
        return jnp.sum(y.astype(jnp.float32) * jnp.sin(inp.astype(jnp.float32))), y

      (_, out), (gs, gx) = jax.jit(jax.value_and_grad(_wrapped_loss, argnums=(0, 1), has_aux=True))(st, x)
      return np.asarray(jax.device_get(out)), gs, np.asarray(jax.device_get(gx))

    for korder in (False, True):
      cfg_base = self._make_cfg(
          dtype="bfloat16", weight_dtype="bfloat16", gdn_fp8_projections=True, gdn_in_proj_kernel_order=korder
      )
      base_mod = _DecoderPathWrapper(
          Qwen3NextGatedDeltaNet(config=cfg_base, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13))
      )
      out_base, gs_base, gx_base = eval_wrapped(base_mod)
      pure_base = nnx.to_pure_dict(gs_base)["decoder"]["layers"]

      # Positive control: without `without_qwix_interception`, Qwix `fp8_gpu` crashes on `_fp8_dot`.
      cfg_ctrl = self._make_cfg(
          dtype="bfloat16",
          weight_dtype="bfloat16",
          gdn_fp8_projections=True,
          gdn_in_proj_kernel_order=korder,
          quantization="fp8_gpu",
          use_qwix_quantization=True,
      )
      with mock.patch.object(quantizations, "without_qwix_interception", side_effect=lambda fn: fn):
        ctrl_mod = _DecoderPathWrapper(
            Qwen3NextGatedDeltaNet(config=cfg_ctrl, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13))
        )
        with self.assertRaises(Exception):
          qwix.quantize_model(ctrl_mod, quantizations.get_qt_provider(cfg_ctrl), x)

      # With in_proj_ba excluded via a Qwix rule, Qwix fp8_full leaves GDN FP8 outputs & grads bit-identical.
      cfg_q_exact = self._make_cfg(
          dtype="bfloat16",
          weight_dtype="bfloat16",
          gdn_fp8_projections=True,
          gdn_in_proj_kernel_order=korder,
          quantization="fp8_full",
          use_qwix_quantization=True,
      )
      q_exact_mod = _DecoderPathWrapper(
          Qwen3NextGatedDeltaNet(config=cfg_q_exact, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13))
      )
      no_ba_rule = [qwix.QtRule(module_path=r".*in_proj_ba.*", weight_qtype=None, act_qtype=None, bwd_qtype=None)]
      base_rules = quantizations.get_quantization_rule(cfg_q_exact)
      q_exact_mod = qwix.quantize_model(q_exact_mod, qwix.QtProvider(no_ba_rule + list(base_rules)), x)
      nnx.pop(q_exact_mod, nnx.Intermediate)
      out_q_exact, gs_q_exact, gx_q_exact = eval_wrapped(q_exact_mod)
      np.testing.assert_array_equal(out_q_exact, out_base)
      np.testing.assert_array_equal(gx_q_exact, gx_base)
      pure_q = nnx.to_pure_dict(gs_q_exact)["decoder"]["layers"]
      for proj_name in ("in_proj_qkvz", "out_proj"):
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(pure_q[proj_name]["kernel"])),
            np.asarray(jax.device_get(pure_base[proj_name]["kernel"])),
        )

      # With in_proj_ba also quantized by Qwix, full layer forward + backward coexists cleanly.
      q_coexist_mod = _DecoderPathWrapper(
          Qwen3NextGatedDeltaNet(config=cfg_q_exact, dtype=jnp.bfloat16, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13))
      )
      q_coexist_mod = qwix.quantize_model(q_coexist_mod, quantizations.get_qt_provider(cfg_q_exact), x)
      nnx.pop(q_coexist_mod, nnx.Intermediate)
      out_coexist, _, gx_coexist = eval_wrapped(q_coexist_mod)
      self.assertGreater(_cosine_similarity(out_coexist, out_base), 0.98)
      self.assertGreater(_cosine_similarity(gx_coexist, gx_base), 0.97)

      # Non-Qwix quant objects leave in_proj_qkvz and out_proj with quant=None in train while quantizing in_proj_ba.
      cfg_nq = self._make_cfg(
          dtype="bfloat16",
          weight_dtype="bfloat16",
          gdn_fp8_projections=True,
          gdn_in_proj_kernel_order=korder,
          quantization="jax_mxfp8",
          use_qwix_quantization=False,
      )
      quant_obj = quantizations.configure_quantization(cfg_nq)
      layer_nq = Qwen3NextGatedDeltaNet(
          config=cfg_nq,
          dtype=jnp.bfloat16,
          model_mode=MODEL_MODE_TRAIN,
          quant=quant_obj,
          rngs=nnx.Rngs(13),
      )
      self.assertIsNone(layer_nq.in_proj_qkvz.quant)
      self.assertIsNone(layer_nq.out_proj.quant)
      self.assertIsNotNone(layer_nq.in_proj_ba.quant)


def _run_multidevice_child():
  """Runs 4-device GSPMD sharded verification of embed_gdn, kernel-order in_proj, and token-minor out_proj."""
  devices = np.array(jax.devices())
  assert len(devices) == 4, jax.devices()
  mesh = Mesh(devices.reshape(2, 2), ("fsdp", "context"))
  x = jax.random.normal(jax.random.PRNGKey(19), (2, 16, 32), jnp.float32) * 0.2
  seg_ids = jnp.array(
      [
          [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
      ],
      dtype=jnp.int32,
  )

  common = {
      "run_name": "gdn_layouts_sharded",
      "decoder_block": "qwen3_next",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": 32,
      "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 8,
      "gdn_key_head_dim": 16,
      "gdn_value_head_dim": 16,
      "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 4,
      "use_qk_norm_in_gdn": True,
      "context_parallel_load_balance": False,
      "enable_checkpointing": False,
  }
  cfg_ref = pyconfig.initialize([None, _BASE_YML], **common)
  out_ref, gs_ref, gx_ref = _eval_gdn_layer(cfg_ref, x, seg_ids, mesh=None)

  cfg_sharded = pyconfig.initialize(
      [None, _BASE_YML],
      ici_fsdp_parallelism=2,
      ici_context_parallelism=2,
      gdn_cp_mode="seq",
      gdn_in_proj_kernel_order=True,
      gdn_out_proj_token_minor=True,
      gdn_gated_norm_layout="natural_fwd_bwd",
      **common,
  )
  out_got, gs_got, gx_got = _eval_gdn_layer(cfg_sharded, x, seg_ids, mesh=mesh)
  np.testing.assert_allclose(out_got, out_ref, atol=3e-4, rtol=2e-3)
  np.testing.assert_allclose(gx_got, gx_ref, atol=5e-4, rtol=3e-3)
  for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_got), jax.tree_util.tree_leaves(gs_ref)):
    np.testing.assert_allclose(g_got, g_ref, atol=5e-4, rtol=3e-3)

  cfg_sharded_fp8 = pyconfig.initialize(
      [None, _BASE_YML],
      ici_fsdp_parallelism=2,
      ici_context_parallelism=2,
      gdn_cp_mode="seq",
      gdn_in_proj_kernel_order=True,
      gdn_fp8_projections=True,
      **common,
  )
  out_fp8, gs_fp8, gx_fp8 = _eval_gdn_layer(cfg_sharded_fp8, x, seg_ids, mesh=mesh)
  assert _cosine_similarity(out_fp8, out_ref) > 0.99
  assert _cosine_similarity(gx_fp8, gx_ref) > 0.98
  for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_fp8), jax.tree_util.tree_leaves(gs_ref)):
    assert _cosine_similarity(g_got, g_ref) > 0.98

  # Exercise sharded `jax.shard_map` + Pallas (`interpret=True`) path in `transpose_quant` with `B_local=2 > 1`
  # and verify `fp8_projection` wgrad emits no cross-rank `all-to-all` / `collective-permute` across `context`.
  tq_keys = jax.random.split(jax.random.PRNGKey(29), 4)
  src_3d = jax.device_put(
      jax.random.normal(tq_keys[0], (4, 256, 128), jnp.bfloat16),
      NamedSharding(mesh, P("fsdp", "context", None)),
  )
  row_s_3d = [
      jax.device_put(
          jax.random.uniform(tq_keys[1], (4, 256, 1), jnp.float32, minval=0.5, maxval=2.0),
          NamedSharding(mesh, P("fsdp", "context", None)),
      )
  ]
  col_s_1d = [jax.random.uniform(tq_keys[2], (128,), jnp.float32, minval=0.5, maxval=2.0)]
  sharded_layout = gdn_fp8_projections.ProjectionLayout(
      gathered_sharding=NamedSharding(mesh, P(None, None)),
      mesh=mesh,
      token_spec=P("fsdp", "context"),
  )
  tq_ref = gdn_fp8_projections.transpose_quant(
      src_3d, row_s_3d, col_s_1d, [gdn_fp8_projections.E4M3], sharded_layout, use_pallas=False
  )[0]
  tq_sharded_pallas = gdn_fp8_projections.transpose_quant(
      src_3d, row_s_3d, col_s_1d, [gdn_fp8_projections.E4M3], sharded_layout, use_pallas=True, interpret=True
  )[0]
  assert tq_sharded_pallas.shape == (128, 4, 256), tq_sharded_pallas.shape
  assert tq_sharded_pallas.sharding.spec == P(None, "fsdp", "context"), tq_sharded_pallas.sharding
  np.testing.assert_array_equal(
      np.asarray(jax.device_get(tq_sharded_pallas.astype(jnp.float32))),
      np.asarray(jax.device_get(tq_ref.astype(jnp.float32))),
  )

  w_fp8 = jax.device_put(
      jax.random.normal(tq_keys[3], (128, 256), jnp.bfloat16) * 0.2,
      NamedSharding(mesh, P("fsdp", None)),
  )

  def _sharded_fp8_loss(xm, wm):
    y0, y1 = gdn_fp8_projections.fp8_projection(xm, wm, (192,), sharded_layout)
    return jnp.sum(y0.astype(jnp.float32)) + 0.5 * jnp.sum(y1.astype(jnp.float32))

  fp8_grad_fn = jax.jit(
      jax.value_and_grad(_sharded_fp8_loss, argnums=(0, 1)),
      in_shardings=(NamedSharding(mesh, P("fsdp", "context", None)), NamedSharding(mesh, P("fsdp", None))),
      out_shardings=(None, (NamedSharding(mesh, P("fsdp", "context", None)), NamedSharding(mesh, P("fsdp", None)))),
  )
  fp8_hlo = fp8_grad_fn.lower(src_3d, w_fp8).compile().as_text()
  assert "all-to-all" not in fp8_hlo and "collective-permute" not in fp8_hlo, fp8_hlo
  assert "f8e4m3fn" in fp8_hlo and "f8e5m2" in fp8_hlo, fp8_hlo

  # Verify `_is_gdn_head_sharded` stays False under default `embed_attn` rules (where `embed_attn` claims
  # `fsdp_transpose` before `gdn_head`) and flips True when `embed_gdn -> fsdp` shards `in_proj_qkvz.kernel`
  # across `('fsdp', 'tensor')`.
  tp_mesh = Mesh(devices.reshape(2, 2), ("fsdp", "tensor"))
  cfg_default_tp = pyconfig.initialize(
      [None, _BASE_YML],
      ici_fsdp_parallelism=2,
      ici_tensor_parallelism=2,
      gdn_in_proj_kernel_order=True,
      **common,
  )
  with nn_partitioning.axis_rules(cfg_default_tp.logical_axis_rules):
    layer_default_tp = Qwen3NextGatedDeltaNet(
        config=cfg_default_tp, mesh=tp_mesh, dtype=jnp.float32, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13)
    )
    assert not layer_default_tp._is_gdn_head_sharded(  # pylint: disable=protected-access
        layer_default_tp.in_proj_qkvz.kernel[...].shape
    )

  with open(_BASE_YML, "r", encoding="utf-8") as f:
    tp_rules = list(yaml.safe_load(f)["logical_axis_rules"]) + [["embed_gdn", "fsdp"]]
  cfg_tp = pyconfig.initialize(
      [None, _BASE_YML],
      ici_fsdp_parallelism=2,
      ici_tensor_parallelism=2,
      logical_axis_rules=tp_rules,
      gdn_in_proj_kernel_order=True,
      **common,
  )
  with nn_partitioning.axis_rules(cfg_tp.logical_axis_rules):
    layer_tp = Qwen3NextGatedDeltaNet(
        config=cfg_tp, mesh=tp_mesh, dtype=jnp.float32, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(13)
    )
    assert layer_tp._is_gdn_head_sharded(layer_tp.in_proj_qkvz.kernel[...].shape)  # pylint: disable=protected-access
  out_tp, gs_tp, gx_tp = _eval_gdn_layer(cfg_tp, x, seg_ids, mesh=tp_mesh)
  np.testing.assert_allclose(out_tp, out_ref, atol=3e-4, rtol=2e-3)
  np.testing.assert_allclose(gx_tp, gx_ref, atol=5e-4, rtol=3e-3)
  for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_tp), jax.tree_util.tree_leaves(gs_ref)):
    np.testing.assert_allclose(g_got, g_ref, atol=5e-4, rtol=3e-3)

  cfg_tp_fp8 = pyconfig.initialize(
      [None, _BASE_YML],
      ici_fsdp_parallelism=2,
      ici_tensor_parallelism=2,
      logical_axis_rules=tp_rules,
      gdn_in_proj_kernel_order=True,
      gdn_fp8_projections=True,
      **common,
  )
  out_tp_fp8, gs_tp_fp8, gx_tp_fp8 = _eval_gdn_layer(cfg_tp_fp8, x, seg_ids, mesh=tp_mesh)
  assert _cosine_similarity(out_tp_fp8, out_ref) > 0.99
  assert _cosine_similarity(gx_tp_fp8, gx_ref) > 0.98
  for g_got, g_ref in zip(jax.tree_util.tree_leaves(gs_tp_fp8), jax.tree_util.tree_leaves(gs_ref)):
    assert _cosine_similarity(g_got, g_ref) > 0.98
  print("GDN_LAYER_LAYOUTS_SHARDING_OK")


if __name__ == "__main__":
  if "--multidevice-child" in sys.argv:
    _run_multidevice_child()
  else:
    unittest.main()
