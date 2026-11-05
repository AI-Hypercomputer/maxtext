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
from jax.sharding import Mesh
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
      self.assertGreater(_cosine_similarity(jax.device_get(dx_fp8), jax.device_get(dx_ref)), 0.995)
      self.assertGreater(_cosine_similarity(jax.device_get(dw_fp8), jax.device_get(dw_ref)), 0.995)

    # 3. End-to-end Qwen3NextGatedDeltaNet with gdn_fp8_projections (plain and kernel-order).
    x_layer = jax.random.normal(jax.random.PRNGKey(9), (2, 8, 32), jnp.float32) * 0.3
    out_f32, _, gx_f32 = _eval_gdn_layer(self._make_cfg(), x_layer)
    for korder in (False, True):
      cfg_fp8 = self._make_cfg(gdn_fp8_projections=True, gdn_in_proj_kernel_order=korder)
      out_fp8, _, gx_fp8 = _eval_gdn_layer(cfg_fp8, x_layer)
      self.assertGreater(_cosine_similarity(out_fp8, out_f32), 0.99)
      self.assertGreater(_cosine_similarity(gx_fp8, gx_f32), 0.98)

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
  out_fp8, _, gx_fp8 = _eval_gdn_layer(cfg_sharded_fp8, x, seg_ids, mesh=mesh)
  assert _cosine_similarity(out_fp8, out_ref) > 0.99
  assert _cosine_similarity(gx_fp8, gx_ref) > 0.98
  print("GDN_LAYER_LAYOUTS_SHARDING_OK")


if __name__ == "__main__":
  if "--multidevice-child" in sys.argv:
    _run_multidevice_child()
  else:
    unittest.main()
