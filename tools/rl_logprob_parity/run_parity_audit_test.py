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
"""Unit tests for `tools/rl_logprob_parity/run_parity_audit.py`."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import qwix

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOL_DIR))
for _p in (_TOOL_DIR, os.path.join(_REPO_ROOT, "src"), os.path.join(os.path.dirname(_REPO_ROOT), "tunix")):
  if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

# Stub optional runtime deps (`aqt`, `qwix.sparsity`, `orbax.step`) for lightweight CPU unit test envs.
try:
  import aqt.jax.v2.aqt_tensor  # noqa: F401
except ModuleNotFoundError:
  for _m in (
      "aqt",
      "aqt.jax",
      "aqt.jax.v2",
      "aqt.jax.v2.config",
      "aqt.jax.v2.aqt_tensor",
      "aqt.jax.v2.flax",
      "aqt.jax.v2.flax.aqt_flax",
      "aqt.jax.v2.tiled_dot_general",
      "aqt.jax.v2.calibration",
  ):
    sys.modules.setdefault(_m, mock.MagicMock())
try:
  from qwix._src import core as _qwix_core

  if not hasattr(_qwix_core, "sparsity"):
    setattr(_qwix_core, "sparsity", mock.MagicMock())
except ImportError:
  pass
try:
  import orbax.checkpoint.experimental.v1 as _ocp_v1

  if hasattr(_ocp_v1, "path") and not hasattr(_ocp_v1.path, "step"):
    setattr(_ocp_v1.path, "step", mock.MagicMock())
except ImportError:
  pass
for _opt_mod in ("omegaconf", "metrax", "metrax.logging"):
  try:
    __import__(_opt_mod)
  except ModuleNotFoundError:
    sys.modules.setdefault(_opt_mod, mock.MagicMock())
if "tunix" not in sys.modules:
  try:
    import tunix  # noqa: F401
  except Exception:
    _tunix_dir = os.path.join(os.path.dirname(_REPO_ROOT), "tunix", "tunix")
    if os.path.isdir(_tunix_dir):
      _pkg = types.ModuleType("tunix")
      _pkg.__path__ = [_tunix_dir]
      sys.modules["tunix"] = _pkg

import run_parity_audit as audit_mod  # noqa: E402
from maxtext.layers import quantizations  # noqa: E402
from tunix.rl import algo_core, common as rl_common  # noqa: E402
from tunix.utils import maxtext_utils as tunix_maxtext_utils  # noqa: E402

pytestmark = [pytest.mark.post_training]


def _make_linear_stub(shape, dtype=jnp.bfloat16, scale_shape=None, serve_fp8=False):
  rng = np.random.default_rng(0)
  scale = 50.0 if "float8" in str(jnp.dtype(dtype)) else 0.05
  arr = jnp.asarray((rng.standard_normal(shape) * scale).astype(np.float32), dtype=dtype)
  return types.SimpleNamespace(
      kernel=nnx.Param(arr),
      kernel_scale=nnx.Param(jnp.ones(scale_shape, dtype=jnp.float32)) if scale_shape is not None else None,
      quant=quantizations.ServeFp8WeightQuantization() if serve_fp8 else None,
  )


def _make_mock_decoder(
    *,
    attention="flash",
    weight_dtype=jnp.bfloat16,
    prefuse_moe=True,
    fp8_moe=False,
    quantization="",
    moe_scale_shape=None,
    dense_scale_shape=None,
    serve_fp8_moe=False,
    serve_fp8_dense=False,
):
  rng = np.random.default_rng(123)
  w_scale_factor = 50.0 if ("float8" in str(jnp.dtype(weight_dtype)) or "int8" in str(jnp.dtype(weight_dtype))) else 0.03

  def _w(shape):
    return nnx.Param(jnp.asarray((rng.standard_normal(shape) * w_scale_factor).astype(np.float32), dtype=weight_dtype))

  def _s(shape):
    return nnx.Param(jnp.full(shape, 0.02, dtype=jnp.float32)) if shape is not None else None

  class _MockRoutedMoE(nnx.Module):

    def __init__(self):
      self.is_hash_routing = False
      self.weight_dtype = weight_dtype
      self.quant = quantizations.ServeFp8WeightQuantization() if serve_fp8_moe else None
      self.wi_kernel_axes = ("expert", None, "mlp")
      self.wo_kernel_axes = ("expert", "mlp", None)
      self.wo = _w((4, 256, 256))
      self.wo_scale = _s(moe_scale_shape)
      self.wi = _w((4, 256, 256)) if prefuse_moe else None
      self.wi_scale = _s(moe_scale_shape) if prefuse_moe else None
      self.wi_0 = None if prefuse_moe else _w((4, 256, 256))
      self.wi_1 = None if prefuse_moe else _w((4, 256, 256))
      self.wi_0_scale = None if prefuse_moe else _s(moe_scale_shape)
      self.wi_1_scale = None if prefuse_moe else _s(moe_scale_shape)

  routed0 = _MockRoutedMoE()
  shared0 = types.SimpleNamespace(
      wi_0=_make_linear_stub((256, 128), dtype=weight_dtype, scale_shape=dense_scale_shape, serve_fp8=serve_fp8_dense)
  )
  gdn0 = types.SimpleNamespace(
      in_proj_qkvz=_make_linear_stub((256, 512), dtype=weight_dtype, scale_shape=dense_scale_shape, serve_fp8=serve_fp8_dense)
  )
  attn3 = types.SimpleNamespace(
      query=_make_linear_stub(
          (256, 4, 64), dtype=weight_dtype, scale_shape=(2, 1, 1) if dense_scale_shape else None, serve_fp8=serve_fp8_dense
      )
  )
  cfg = types.SimpleNamespace(
      model_name="qwen3.5-35b-a3b",
      attention=attention,
      weight_dtype=weight_dtype,
      dtype=jnp.bfloat16,
      quantization=quantization,
      fp8_moe=fp8_moe,
      prefuse_moe_weights=prefuse_moe,
      sparse_matmul=True,
      use_gmm_v2=True,
      num_decoder_layers=4,
  )
  decoder = types.SimpleNamespace(
      config=cfg,
      layers_0=types.SimpleNamespace(
          attention=gdn0,
          mlp=types.SimpleNamespace(routed_experts=routed0, shared_expert=shared0),
      ),
      layers_1=None,
      layers_2=None,
      layers_3=types.SimpleNamespace(attention=types.SimpleNamespace(attention=attn3), mlp=None),
  )
  return types.SimpleNamespace(decoder=decoder)


class QuantizationAndScaleTest(unittest.TestCase):

  def test_quantize_moe_fp8_matches_maxtext_fused_moe_quantizer(self):
    for qtype in (jnp.float8_e4m3fn, jnp.int8):
      for smode, tile_size, expected_shape in [
          ("per_channel", None, (4, 1, 256)),
          ("subchannel128", 128, (4, 2, 256)),
      ]:
        model = _make_mock_decoder(weight_dtype=jnp.bfloat16, prefuse_moe=True)
        routed = model.decoder.layers_0.mlp.routed_experts
        orig_wi = jnp.asarray(routed.wi[...])
        audit_mod.quantize_moe_fp8(model, scale_mode=smode, weight_qtype=qtype)
        qw, qs = routed.wi[...], routed.wi_scale[...]
        ref_qw, ref_qs_4d = quantizations.quantize_weight_for_fused_moe(
            orig_wi, qwix.QtRule(weight_qtype=qtype, tile_size=tile_size)
        )
        self.assertEqual(qw.dtype, jnp.dtype(qtype))
        self.assertEqual(qs.shape, expected_shape)
        np.testing.assert_array_equal(np.asarray(qw), np.asarray(ref_qw))
        np.testing.assert_array_equal(np.asarray(qs), np.asarray(jnp.squeeze(ref_qs_4d, axis=2)))
        np.testing.assert_array_equal(np.asarray(quantizations.prepare_fused_gmm_scale(qs, qw.shape)), np.asarray(ref_qs_4d))

  def test_quantize_moe_fp8_roundtrip_all_scale_modes(self):
    for smode, expected_s_shape in [
        ("per_channel", (4, 1, 256)),
        ("subchannel128", (4, 2, 256)),
        ("block128", (4, 2, 2)),
    ]:
      model = _make_mock_decoder(weight_dtype=jnp.bfloat16, prefuse_moe=True)
      routed = model.decoder.layers_0.mlp.routed_experts
      w_np = np.asarray(routed.wi[...], dtype=np.float32)
      audit_mod.quantize_moe_fp8(model, scale_mode=smode)
      qw, qs = routed.wi[...], routed.wi_scale[...]
      self.assertEqual(qw.dtype, jnp.float8_e4m3fn)
      self.assertEqual(qs.shape, expected_s_shape)
      w_deq = np.asarray(quantizations.dequantize_weight(qw, qs, jnp.float32))
      cos = float(np.sum(w_np * w_deq) / (np.linalg.norm(w_np) * np.linalg.norm(w_deq)))
      self.assertGreater(cos, 0.999)
      self.assertEqual(quantizations.prepare_fused_gmm_scale(qs, qw.shape).shape, (4, expected_s_shape[1], 1, 256))

  def test_quantize_moe_fp8_rejects_invalid_scale_mode(self):
    model = _make_mock_decoder(weight_dtype=jnp.bfloat16, prefuse_moe=True)
    with self.assertRaises(ValueError):
      audit_mod.quantize_moe_fp8(model, scale_mode="unknown_mode")


class InPlaceMoeQuantizationAndModelAuditTest(unittest.TestCase):

  def test_audit_bf16_sampler_and_trainer(self):
    with tempfile.TemporaryDirectory() as tmp:
      sa_model = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      sa_rep = audit_mod.audit_model(sa_model, role="sampler", expected_mode="bf16", out_dir=tmp)
      self.assertEqual(sa_rep["execution_paths"]["routed_moe"], "FUSED_MOE_BF16 (tpu_inference)")
      self.assertEqual(sa_rep["execution_paths"]["dense_linear"], "DENSE_PURE_BF16 (BF16 dot_general)")

      tr_model = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      tr_rep = audit_mod.audit_model(tr_model, role="trainer", expected_mode="bf16", out_dir=tmp)
      self.assertEqual(tr_rep["execution_paths"]["routed_moe"], "GMM_V2_PURE_BF16 (BF16 gmm_v2)")
      self.assertEqual(tr_rep["execution_paths"]["dense_linear"], "DENSE_PURE_BF16 (BF16 dot_general)")
      self.assertTrue(os.path.isfile(os.path.join(tmp, "audit_sampler.json")))
      self.assertTrue(os.path.isfile(os.path.join(tmp, "audit_trainer.json")))

  def test_quantize_moe_in_place_and_audit_fp8_moe_and_native(self):
    with tempfile.TemporaryDirectory() as tmp:
      sa_model = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      self.assertEqual(audit_mod.quantize_moe_fp8(sa_model, scale_mode="per_channel"), 1)
      sa_rep = audit_mod.audit_model(sa_model, role="sampler", expected_mode="fp8_moe", out_dir=tmp)
      self.assertIn("FUSED_MOE_NATIVE_FP8", sa_rep["execution_paths"]["routed_moe"])
      self.assertEqual(sa_rep["execution_paths"]["dense_linear"], "DENSE_PURE_BF16 (BF16 dot_general)")

      tr_deq = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      audit_mod.quantize_moe_fp8(tr_deq, scale_mode="per_channel", set_serve_quant=False)
      tr_deq_rep = audit_mod.audit_model(tr_deq, role="trainer", expected_mode="fp8_moe", out_dir=tmp)
      self.assertIn("GMM_V2_DEQUANT_TO_BF16", tr_deq_rep["execution_paths"]["routed_moe"])

      tr_nat = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=False)
      audit_mod.quantize_moe_fp8(tr_nat, scale_mode="per_channel", set_serve_quant=True)
      tr_nat_rep = audit_mod.audit_model(tr_nat, role="trainer", expected_mode="fp8_moe_native", out_dir=tmp)
      self.assertIn("GMM_V2_NATIVE_FP8", tr_nat_rep["execution_paths"]["routed_moe"])

  def test_audit_fp8_ckpt_and_fp8_serve_and_int8_moe(self):
    with tempfile.TemporaryDirectory() as tmp:
      ckpt_tr = _make_mock_decoder(
          attention="flash",
          weight_dtype=jnp.float8_e4m3fn,
          prefuse_moe=False,
          fp8_moe=True,
          moe_scale_shape=(4, 2, 2),
          dense_scale_shape=(2, 4),
      )
      rep_ckpt = audit_mod.audit_model(ckpt_tr, role="trainer", expected_mode="fp8_ckpt", out_dir=tmp)
      self.assertIn("GMM_V2_DEQUANT_TO_BF16", rep_ckpt["execution_paths"]["routed_moe"])
      self.assertIn("DENSE_DEQUANT_TO_BF16", rep_ckpt["execution_paths"]["dense_linear"])
      rep_fp8 = audit_mod.audit_model(ckpt_tr, role="trainer", expected_mode="fp8", out_dir=tmp)
      self.assertEqual(rep_fp8["expected_mode"], "fp8")

      fp8moe_sa = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      audit_mod.quantize_moe_fp8(fp8moe_sa, scale_mode="per_channel")
      rep_fp8moe = audit_mod.audit_model(fp8moe_sa, role="sampler", expected_mode="fp8moe", out_dir=tmp)
      self.assertIn("FUSED_MOE_NATIVE_FP8", rep_fp8moe["execution_paths"]["routed_moe"])

      serve_tr = _make_mock_decoder(
          attention="flash",
          weight_dtype=jnp.float8_e4m3fn,
          prefuse_moe=False,
          fp8_moe=True,
          quantization="serve_fp8_weight",
          moe_scale_shape=(4, 1, 256),
          dense_scale_shape=(2, 4),
          serve_fp8_moe=True,
          serve_fp8_dense=True,
      )
      rep_serve = audit_mod.audit_model(serve_tr, role="trainer", expected_mode="fp8_serve", out_dir=tmp)
      self.assertIn("GMM_V2_NATIVE_FP8", rep_serve["execution_paths"]["routed_moe"])
      self.assertIn("DENSE_NATIVE_FP8", rep_serve["execution_paths"]["dense_linear"])

      int8_tr = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      audit_mod.quantize_moe_fp8(int8_tr, scale_mode="per_channel", weight_qtype=jnp.int8)
      rep_int8 = audit_mod.audit_model(int8_tr, role="trainer", expected_mode="int8_moe", out_dir=tmp)
      self.assertIn("GMM_V2_DEQUANT_TO_BF16", rep_int8["execution_paths"]["routed_moe"])
      self.assertEqual(rep_int8["modules"]["layers_0.mlp.routed_experts.wi"]["weight"]["dtype"], "int8")

  def test_audit_model_fails_fast_on_mode_mismatch(self):
    bf16_model = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
    with self.assertRaises(AssertionError):
      audit_mod.audit_model(bf16_model, role="trainer", expected_mode="fp8_moe", out_dir="")

    fp8_no_scale = _make_mock_decoder(attention="flash", weight_dtype=jnp.float8_e4m3fn, prefuse_moe=True, moe_scale_shape=None)
    with self.assertRaises(AssertionError):
      audit_mod.audit_model(fp8_no_scale, role="trainer", expected_mode="fp8_ckpt", out_dir="")

    tr_prefused_fp8 = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
    audit_mod.quantize_moe_fp8(tr_prefused_fp8, scale_mode="per_channel", set_serve_quant=True)
    with self.assertRaises(AssertionError):
      audit_mod.audit_model(tr_prefused_fp8, role="trainer", expected_mode="fp8_moe_native", out_dir="")


class SamplerConfigBuilderTest(unittest.TestCase):

  def test_tunix_build_vllm_maxtext_additional_config(self):
    self.assertEqual(tunix_maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES, {"architectures": ["MaxTextForCausalLM"]})
    add_bf16 = tunix_maxtext_utils.build_vllm_maxtext_additional_config(
        model_name=audit_mod.MODEL_MAXTEXT_BF16,
        attention="vllm_rpa",
        prefuse_moe_weights=True,
        return_routed_experts=True,
        float32_gate_logits=True,
        float32_logits=True,
    )
    mt_bf16 = add_bf16["maxtext_config"]
    self.assertEqual(mt_bf16["model_name"], "qwen3.5-35b-a3b")
    self.assertEqual(mt_bf16["weight_dtype"], "bfloat16")
    self.assertTrue(mt_bf16["prefuse_moe_weights"])
    self.assertTrue(mt_bf16["return_routed_experts"])


class RouterReplayAndLogprobAlignmentTest(unittest.TestCase):

  def test_router_replay_terminal_padding_and_tunix_alignment(self):
    rng = np.random.default_rng(99)
    P, G, pad_len, L, K = 16, 8, 8, 4, 2
    captured = rng.integers(0, 64, size=(3, P + G - 1, L, K), dtype=np.int16)
    padded = [
        np.concatenate([c, np.full((1, L, K), rl_common.UNSET_ROUTED_EXPERT, dtype=np.int16)], axis=0)
        for c in captured
    ]
    aligned = rl_common.align_routed_experts(
        padded, completion_lengths=[G] * 3, prompt_width=P, completion_width=G + pad_len
    )
    self.assertEqual(aligned.shape, (3, P + G + pad_len, L, K))
    np.testing.assert_array_equal(aligned[:, : P + G - 1], captured)
    self.assertTrue(np.all(aligned[:, P + G - 1 :] == rl_common.UNSET_ROUTED_EXPERT))

  def test_selective_log_softmax_shift_and_temperature_scaling(self):
    rng = np.random.default_rng(19)
    B, P, G, V = 2, 12, 6, 32
    seq_len = P + G
    logits = jnp.asarray(rng.standard_normal((B, seq_len, V)), dtype=jnp.float32)
    tokens = rng.integers(0, V, size=(B, seq_len), dtype=np.int32)
    nxt = jnp.asarray(np.roll(tokens, -1, axis=1))
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (B, seq_len))

    gen_temp = 0.7
    temp_scale = jnp.where(pos >= (P - 1), jnp.float32(gen_temp), jnp.float32(1.0))[..., None]
    tr_lp = np.asarray(rl_common.selective_log_softmax(logits / temp_scale, nxt))

    ref_prompt_lse = jax.nn.log_softmax(logits, axis=-1)
    ref_decode_lse = jax.nn.log_softmax(logits / gen_temp, axis=-1)
    sa_prompt_lp = np.zeros((B, P), dtype=np.float32)
    for t in range(1, P):
      sa_prompt_lp[:, t] = np.asarray(ref_prompt_lse[np.arange(B), t - 1, tokens[:, t]])
    sa_gen_lp = np.zeros((B, G), dtype=np.float32)
    for j in range(G):
      sa_gen_lp[:, j] = np.asarray(ref_decode_lse[np.arange(B), P - 1 + j, tokens[:, P + j]])

    np.testing.assert_allclose(tr_lp[:, : P - 1], sa_prompt_lp[:, 1:P], atol=1e-6)
    np.testing.assert_allclose(tr_lp[:, P - 1 : P + G - 1], sa_gen_lp, atol=1e-6)


class OobScoringAndStageCompareTest(unittest.TestCase):

  def test_tunix_algo_core_sequence_loss_mask_and_tis_oob(self):
    rng = np.random.default_rng(2026)
    sa_lp = rng.uniform(-3.0, -0.05, size=(8, 64)).astype(np.float32)
    delta = np.zeros_like(sa_lp)
    delta[:4] = rng.normal(0.0002, 0.0002, size=(4, 64)).astype(np.float32)
    delta[4:] = 0.01
    tr_lp = sa_lp + delta
    mask = np.ones_like(sa_lp, dtype=np.float32)

    raw_jnp = jnp.asarray(tr_lp - sa_lp, dtype=jnp.float32)
    mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
    log_is_jnp = jnp.nan_to_num(raw_jnp, nan=0.0, posinf=0.0, neginf=0.0)
    seq_mask_res = algo_core.sequence_loss_mask(mask_jnp, log_is_raw=raw_jnp, mult_prob_error_threshold=2.0)
    seq_geomean_jnp, seq_valid_jnp = algo_core.sequence_geomean_ratio(log_is_jnp, mask_jnp)
    _, expected_oob = algo_core.truncated_importance_weights(
        raw_jnp, seq_geomean_jnp, seq_valid_jnp, seq_mask_res.sample_mask, audit_mod.RATIO_MIN, audit_mod.RATIO_MAX
    )
    self.assertAlmostEqual(float(expected_oob), 0.5, places=6)
    self.assertEqual(float(algo_core.masked_mean(seq_mask_res.sample_mask, seq_valid_jnp)), 1.0)

  def test_tunix_sequence_loss_mask_nan_rejection_and_masked_nan_handling(self):
    sa_lp = np.full((3, 16), -0.5, dtype=np.float32)
    tr_lp = np.full((3, 16), -0.5, dtype=np.float32)
    sa_lp[0, 5] = np.nan
    raw_jnp = jnp.asarray(tr_lp - sa_lp, dtype=jnp.float32)

    unmasked = algo_core.sequence_loss_mask(jnp.ones_like(raw_jnp), log_is_raw=raw_jnp, mult_prob_error_threshold=2.0)
    np.testing.assert_array_equal(np.asarray(unmasked.sample_mask), np.array([0.0, 1.0, 1.0]))

    masked = algo_core.sequence_loss_mask(jnp.isfinite(raw_jnp).astype(jnp.float32), log_is_raw=raw_jnp, mult_prob_error_threshold=2.0)
    np.testing.assert_array_equal(np.asarray(masked.sample_mask), np.array([1.0, 1.0, 1.0]))

  def test_stage_compare_end_to_end_output_format(self):
    rng = np.random.default_rng(42)
    B, P, G = 4, 16, 16
    tokens = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    full_tokens = rng.integers(10, 100, size=(B, P + G), dtype=np.int32)
    full_tokens[:, :P] = tokens

    sa_prompt_lp = rng.uniform(-2.0, -0.1, size=(B, P)).astype(np.float32)
    sa_prompt_lp[:, 0] = np.nan
    sa_gen_lp = rng.uniform(-1.5, -0.05, size=(B, G)).astype(np.float32)

    tr_full_lp = np.zeros((B, P + G), dtype=np.float32)
    tr_full_lp[:, : P - 1] = sa_prompt_lp[:, 1:P] + 0.0001
    tr_full_lp[:, P - 1 : P + G - 1] = sa_gen_lp - 0.0001

    sa_top1 = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    sa_gen_top1 = rng.integers(10, 100, size=(B, G), dtype=np.int32)
    tr_full_top1 = np.zeros((B, P + G), dtype=np.int32)
    tr_full_top1[:, : P - 1] = sa_top1[:, 1:P]
    tr_full_top1[:, P - 1 : P + G - 1] = sa_gen_top1

    with tempfile.TemporaryDirectory() as tmp:
      for role, moe_path in [
          ("sampler", "FUSED_MOE_BF16 (tpu_inference)"),
          ("trainer", "GMM_V2_PURE_BF16 (BF16 gmm_v2)"),
      ]:
        with open(os.path.join(tmp, f"audit_{role}.json"), "w", encoding="utf-8") as f:
          json.dump(
              {
                  "expected_mode": "bf16",
                  "execution_paths": {
                      "routed_moe": moe_path,
                      "dense_linear": "DENSE_PURE_BF16 (BF16 dot_general)",
                  },
              },
              f,
          )

      np.savez(
          os.path.join(tmp, "sampler_logprobs.npz"),
          logp=sa_prompt_lp,
          top1=sa_top1,
          tokens=tokens,
          gen_ids=full_tokens[:, P:],
          gen_logp=sa_gen_lp,
          gen_top1=sa_gen_top1,
          gen_lens=np.full(B, G, dtype=np.int32),
      )
      np.savez(
          os.path.join(tmp, "trainer_logprobs.npz"),
          logp=tr_full_lp,
          top1=tr_full_top1,
          tokens=tokens,
          full_tokens=full_tokens,
          gen_logp=tr_full_lp[:, P - 1 : P + G - 1],
          gen_top1=tr_full_top1[:, P - 1 : P + G - 1],
      )

      buf = io.StringIO()
      with redirect_stdout(buf):
        res = audit_mod.main(["--sampler-mode", "bf16", "--trainer-mode", "bf16", "--out-dir", tmp, "--stage", "compare"])
      out = buf.getvalue()

      self.assertIn("[SAMPLER AUDIT] mode=bf16 | MoE=FUSED_MOE_BF16 (tpu_inference)", out)
      self.assertIn("[TRAINER AUDIT] mode=bf16 | MoE=GMM_V2_PURE_BF16 (BF16 gmm_v2)", out)
      self.assertIn("### Sampler=BF16 vs Trainer=BF16 — PROMPT tokens (prefill)", out)
      self.assertIn("### Sampler=BF16 vs Trainer=BF16 — OUTPUT tokens (decode)", out)
      self.assertIn(">>> is_oob_ratio (seq-mask-tis)", out)
      self.assertIn("Top-5 Worst Token Divergences:", out)
      self.assertEqual(res["prompt"]["is_oob_ratio"], 0.0)
      self.assertEqual(res["decode"]["is_oob_ratio"], 0.0)
      self.assertEqual(res["prompt"]["top1_agree"], 1.0)
      self.assertEqual(res["decode"]["top1_agree"], 1.0)

      buf2 = io.StringIO()
      with redirect_stdout(buf2):
        audit_mod.main(["--sampler-mode", "fp8moe", "--trainer-mode", "fp8", "--out-dir", tmp, "--stage", "compare"])
      self.assertIn("### Sampler=FP8MOE vs Trainer=FP8 — OUTPUT tokens (decode)", buf2.getvalue())

  def test_build_algo_and_batch_variable_length_routed_experts_and_safe_pad_id(self):
    rng = np.random.default_rng(7)
    B, P, max_G, L, K = 3, 12, 10, 4, 2
    comp_lens = np.array([6, 10, 4], dtype=np.int32)
    prompts = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    comp_ids = rng.integers(10, 100, size=(B, max_G), dtype=np.int32)
    # Include tokenizer pad_id (248044) as an active generated token in seq 0
    comp_ids[0, 2] = 248044
    logps = np.full((B, max_G), np.nan, dtype=np.float32)
    routed = np.full((B, P + max_G, L, K), rl_common.UNSET_ROUTED_EXPERT, dtype=np.int16)
    for i, c_len in enumerate(comp_lens):
      logps[i, :c_len] = rng.uniform(-1.5, -0.1, size=(c_len,)).astype(np.float32)
      routed[i, : P + c_len - 1] = rng.integers(0, 32, size=(P + c_len - 1, L, K), dtype=np.int16)

    eff_pad_id = audit_mod._safe_pad_id(248044, prompts, comp_ids)
    self.assertNotEqual(eff_pad_id, 248044)

    _, batch = audit_mod.build_algo_and_batch(
        prompts,
        comp_ids,
        logps,
        completion_lens=comp_lens,
        routed_experts=routed,
        pad_id=eff_pad_id,
    )
    # Double-assembly re-invocation (as done in stage_trainer) must preserve exact routed_experts alignment
    _, batch2 = audit_mod.build_algo_and_batch(
        batch.prompt_ids[:, :P],
        batch.completion_ids[:, :max_G],
        np.where(batch.completion_mask[:, :max_G] > 0, batch.rollout_per_token_logps[:, :max_G], np.nan),
        completion_lens=comp_lens,
        routed_experts=batch.routed_experts[:, : P + max_G],
        pad_id=eff_pad_id,
    )
    for i, c_len in enumerate(comp_lens):
      self.assertEqual(int(batch2.completion_mask[i].sum()), int(c_len))
      np.testing.assert_array_equal(
          batch2.routed_experts[i, : P + c_len - 1],
          routed[i, : P + c_len - 1],
      )
      self.assertEqual(int(batch2.routed_experts[i, P + c_len - 1, 0, 0]), rl_common.UNSET_ROUTED_EXPERT)

  def test_sequence_packing_batch_and_compare(self):
    rng = np.random.default_rng(11)
    B, P, G = 4, 16, 12
    comp_lens = np.array([12, 8, 10, 6], dtype=np.int32)
    prompts = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    comp_ids = rng.integers(10, 100, size=(B, G), dtype=np.int32)
    sa_gen_lp = np.full((B, G), np.nan, dtype=np.float32)
    for i, c_len in enumerate(comp_lens):
      sa_gen_lp[i, :c_len] = rng.uniform(-1.5, -0.1, size=(c_len,)).astype(np.float32)
    tr_gen_lp = sa_gen_lp + 0.0002

    _, packed_batch = audit_mod.build_algo_and_batch(
        prompts,
        comp_ids,
        sa_gen_lp,
        completion_lens=comp_lens,
        max_seq_token_per_tpu=1024,
    )
    self.assertIsNotNone(packed_batch.segment_ids)
    self.assertIsNotNone(packed_batch.segment_positions)
    self.assertEqual(int(packed_batch.completion_mask.sum()), int(comp_lens.sum()))

    # Verify _unpack_trainer_logps round-trips packed trainer per_token_logps back to [B, G]
    unpacked = audit_mod._unpack_trainer_logps(
        packed_batch.rollout_per_token_logps, packed_batch, comp_lens, B, G
    )
    for i, c_len in enumerate(comp_lens):
      np.testing.assert_allclose(unpacked[i, :c_len], sa_gen_lp[i, :c_len], atol=1e-6)
      self.assertTrue(np.all(np.isnan(unpacked[i, c_len:])))

    with tempfile.TemporaryDirectory() as tmp:
      np.savez(
          os.path.join(tmp, "sampler_logprobs.npz"),
          tokens=prompts,
          gen_ids=comp_ids,
          gen_logp=sa_gen_lp,
          gen_lens=comp_lens,
      )
      np.savez(
          os.path.join(tmp, "trainer_logprobs.npz"),
          tokens=prompts,
          gen_ids=comp_ids,
          gen_logp=tr_gen_lp,
          gen_lens=comp_lens,
      )
      buf = io.StringIO()
      with redirect_stdout(buf):
        res_packed = audit_mod.main([
            "--sampler-mode", "bf16",
            "--trainer-mode", "bf16",
            "--out-dir", tmp,
            "--stage", "compare",
            "--pack-sequences",
            "--max-seq-token-per-tpu", "1024",
        ])
      self.assertEqual(res_packed["decode"]["is_oob_ratio"], 0.0)
      self.assertEqual(res_packed["decode"]["sample_mask/kept_frac"], 1.0)

  def test_multi_chunk_sequence_packing_preserves_all_trajectories(self):
    rng = np.random.default_rng(23)
    B, P, G = 4, 200, 100
    comp_lens = np.array([100, 90, 80, 70], dtype=np.int32)
    prompts = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    comp_ids = rng.integers(10, 100, size=(B, G), dtype=np.int32)
    sa_gen_lp = np.full((B, G), np.nan, dtype=np.float32)
    for i, c_len in enumerate(comp_lens):
      sa_gen_lp[i, :c_len] = rng.uniform(-1.5, -0.1, size=(c_len,)).astype(np.float32)

    # With pack_size=1 and budget=712 (each trajectory is 200..300 tokens), 4 trajectories spill across multiple chunks
    _, packed_batch = audit_mod.build_algo_and_batch(
        prompts,
        comp_ids,
        sa_gen_lp,
        completion_lens=comp_lens,
        max_seq_token_per_tpu=512,
        pack_size=1,
    )
    self.assertGreater(packed_batch.segment_ids.shape[0], 1)
    self.assertEqual(len(packed_batch.metadata["trajectory_ids"]), B)
    self.assertEqual(int(packed_batch.completion_mask.sum()), int(comp_lens.sum()))

    unpacked = audit_mod._unpack_trainer_logps(
        packed_batch.rollout_per_token_logps, packed_batch, comp_lens, B, G
    )
    for i, c_len in enumerate(comp_lens):
      np.testing.assert_allclose(unpacked[i, :c_len], sa_gen_lp[i, :c_len], atol=1e-6)
      self.assertTrue(np.all(np.isnan(unpacked[i, c_len:])))

  def test_scanned_decoder_quantize_and_audit(self):
    base_model = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=False)
    routed = base_model.decoder.layers_0.mlp.routed_experts
    # Give routed experts 4D scanned weights (scan_blocks=2, E=4, K=256, N=256)
    rng = np.random.default_rng(55)
    for attr in ("wo", "wi_0", "wi_1"):
      arr = jnp.asarray((rng.standard_normal((2, 4, 256, 256)) * 0.03).astype(np.float32), dtype=jnp.bfloat16)
      setattr(routed, attr, nnx.Param(arr))
    routed.wi_kernel_axes = ("layers", "expert", None, "mlp")
    routed.wo_kernel_axes = ("layers", "expert", "mlp", None)

    scanned_decoder = types.SimpleNamespace(
        config=base_model.decoder.config,
        layers_0=None,
        layers_3=None,
        layers=types.SimpleNamespace(
            local_layers=[base_model.decoder.layers_0],
            global_layer=base_model.decoder.layers_3,
        ),
    )
    scanned_model = types.SimpleNamespace(decoder=scanned_decoder)
    self.assertEqual(audit_mod.quantize_moe_fp8(scanned_model, scale_mode="per_channel", set_serve_quant=True), 1)
    self.assertEqual(routed.wi_0[...].shape, (2, 4, 256, 256))
    self.assertEqual(routed.wi_0_scale[...].shape, (2, 4, 1, 256))
    rep = audit_mod.audit_model(scanned_model, role="trainer", expected_mode="fp8_moe_native", out_dir="")
    self.assertIn("GMM_V2_NATIVE_FP8", rep["execution_paths"]["routed_moe"])

  def test_stage_trainer_no_optimizer_recursion_and_packed_flags(self):
    import optax

    rng = np.random.default_rng(77)
    B, P, G = 4, 16, 16
    tokens = rng.integers(10, 100, size=(B, P), dtype=np.int32)
    gen_ids = rng.integers(10, 100, size=(B, G), dtype=np.int32)
    sa_gen_lp = rng.uniform(-1.5, -0.1, size=(B, G)).astype(np.float32)

    raw_decoder = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=False)

    class _MockNNXModel(nnx.Module):

      def __init__(self):
        self.decoder = raw_decoder.decoder
        self.dummy_param = nnx.Param(jnp.zeros((1,), dtype=jnp.float32))

    mock_model = _MockNNXModel()

    class _FakeMaxTextEngine:

      def __init__(self):
        self._model, self._quant = self._build_model()
        self._optimizer = self._build_optimizer(optax.identity())

      def _build_model(self):
        return mock_model, None

      def _build_optimizer(self, tx):
        return nnx.Optimizer(self._model, tx, wrt=nnx.Param)

    def _fake_create_trainer_factory(trainer_args):
      # Verify packed max_response_length is per-trajectory padded length (512 - P = 496), NOT max_seq_token_per_tpu (2048)
      self.assertEqual(trainer_args.max_response_length, 512 - P)
      self.assertEqual(trainer_args.max_seq_token_per_tpu, 2048)

      def _factory():
        fp32_cls = tunix_maxtext_utils._build_fp32_master_optimizer_cls()

        class _Fp32SubEngine(_FakeMaxTextEngine):

          def _build_optimizer(self, tx):
            orig = nnx.Optimizer
            nnx.Optimizer = fp32_cls
            try:
              return super()._build_optimizer(tx)
            finally:
              nnx.Optimizer = orig

        return _Fp32SubEngine()

      mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
      return _factory, mesh

    class _FakeTrainerWorker:

      def __init__(self, trainer_factory, worker_id, logps_chunk_size, logps_micro_batch_size, execution_context):
        self._trainer = trainer_factory()
        self.logps_micro_batch_size = logps_micro_batch_size

      def per_token_logps(self, req):
        _ = req.micro_batch_size
        return types.SimpleNamespace(per_token_logps=np.zeros_like(req.completion_tokens, dtype=np.float32) - 0.25)

    fake_maxtext_engine_mod = types.SimpleNamespace(MaxTextTrainingEngine=_FakeMaxTextEngine, nnx=nnx)
    with tempfile.TemporaryDirectory() as tmp:
      np.savez(os.path.join(tmp, "tokens.npz"), tokens=tokens)
      np.savez(
          os.path.join(tmp, "sampler_logprobs.npz"),
          tokens=tokens,
          gen_ids=gen_ids,
          gen_logp=sa_gen_lp,
          gen_lens=np.full(B, G, dtype=np.int32),
      )
      with (
          mock.patch.object(audit_mod, "get_tokenizer", return_value=(types.SimpleNamespace(eos_token_id=2), "dummy-tok")),
          mock.patch.object(tunix_maxtext_utils, "get_tokenizer_pad_id", return_value=0),
          mock.patch.object(tunix_maxtext_utils, "maxtext_modules", return_value=(None, fake_maxtext_engine_mod, None)),
          mock.patch("tunix.experimental.examples.common.run_trainer_node._create_maxtext_trainer_factory", side_effect=_fake_create_trainer_factory),
          mock.patch("tunix.experimental.worker.trainer_worker.TrainerWorker", side_effect=_FakeTrainerWorker),
      ):
        audit_mod.main([
            "--stage", "trainer",
            "--trainer-mode", "fp8_moe_native",
            "--out-dir", tmp,
            "--pack-sequences",
            "--max-seq-token-per-tpu", "2048",
        ])
      extra = os.environ.get("MAXTEXT_EXTRA_FLAGS", "")
      self.assertIn("logits_dot_in_fp32=True", extra)
      self.assertIn("use_tokamax_splash=True", extra)
      self.assertIn("sa_use_base2_exp=False", extra)
      self.assertIn("sa_fuse_reciprocal=True", extra)
      self.assertIn("megablox=True", extra)
      self.assertIn("wi_tile_fwd_batch_seq=256", extra)
      self.assertIn("wi_tile_fwd_embed_dim=128", extra)
      saved = np.load(os.path.join(tmp, "trainer_logprobs.npz"))
      self.assertEqual(saved["gen_logp"].shape, (B, G))
      np.testing.assert_allclose(saved["gen_logp"], -0.25, atol=1e-6)

  def test_stage_sampler_in_place_moe_and_cleanup_and_stage_all_subprocesses(self):
    B, P, G, L, K = 2, 8, 4, 4, 2
    tokens = np.arange(B * P, dtype=np.int32).reshape(B, P) + 10
    raw_decoder = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)

    class _MockNNXModel(nnx.Module):

      def __init__(self):
        self.decoder = raw_decoder.decoder
        self.routed0 = raw_decoder.decoder.layers_0.mlp.routed_experts

    mock_inner_model = _MockNNXModel()
    cleanup_calls = []
    sampler_call_kwargs = []
    vllm_configs = []

    class _FakeAdapter(nnx.Module):

      def __init__(self):
        self.mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
        self.maxtext_config = types.SimpleNamespace(logical_axis_rules=[])
        self.model = None

      def load_weights(self, rng):
        del rng
        self.model = nnx.data(mock_inner_model)

      def get_mrope_input_positions(self, input_tokens, mm_features=None):
        del mm_features
        return np.zeros((3, len(input_tokens)), dtype=np.int32), 0

    class _FakeVllmSampler:

      def __init__(self, tokenizer, config):
        del tokenizer
        vllm_configs.append(config)
        adapter = _FakeAdapter()
        adapter.load_weights(jax.random.PRNGKey(0))
        state = nnx.state(adapter)
        self._model_runner = types.SimpleNamespace(
            model=adapter,
            state=state,
            state_leaves=jax.tree.leaves(state),
            uses_mrope=True,
        )

      def refresh_state_leaves(self):
        self._model_runner.state_leaves = jax.tree.leaves(self._model_runner.state)

      def delete_cache(self):
        cleanup_calls.append("delete_cache")

      def stop(self):
        cleanup_calls.append("stop")

      def __call__(self, prompt_token_ids, max_generation_steps, **kwargs):
        sampler_call_kwargs.append(kwargs)
        n = len(prompt_token_ids)
        return types.SimpleNamespace(
            tokens=[[101 + j for j in range(max_generation_steps)] for _ in range(n)],
            logprobs=[[-0.1 * (j + 1) for j in range(max_generation_steps)] for _ in range(n)],
            routed_experts=[np.ones((len(prompt_token_ids[i]) + max_generation_steps - 1, L, K), dtype=np.int16) for i in range(n)],
        )

    fake_adapter_mod = types.SimpleNamespace(MaxTextForCausalLM=_FakeAdapter, register=lambda: None)
    fake_vllm_sampler_mod = types.SimpleNamespace(
        VllmConfig=lambda **kw: types.SimpleNamespace(**kw),
        VllmSampler=_FakeVllmSampler,
    )
    fake_generate_pkg = types.SimpleNamespace(vllm_sampler=fake_vllm_sampler_mod)
    with tempfile.TemporaryDirectory() as tmp:
      np.savez(os.path.join(tmp, "tokens.npz"), tokens=tokens)
      with (
          mock.patch.dict(
              sys.modules,
              {
                  "maxtext_vllm_adapter": fake_adapter_mod,
                  "tunix.generate": fake_generate_pkg,
                  "tunix.generate.vllm_sampler": fake_vllm_sampler_mod,
              },
          ),
          mock.patch.object(audit_mod, "get_tokenizer", return_value=(types.SimpleNamespace(eos_token_id=2), "dummy-tok")),
          mock.patch.object(tunix_maxtext_utils, "get_tokenizer_pad_id", return_value=0),
      ):
        audit_mod.main([
            "--stage", "sampler",
            "--sampler-mode", "fp8moe",
            "--gen-tokens", str(G),
            "--out-dir", tmp,
        ])
      self.assertEqual(cleanup_calls, ["delete_cache", "stop"])
      self.assertEqual(vllm_configs[-1].eos_tokens, [])
      self.assertTrue(sampler_call_kwargs[-1]["ignore_eos"])
      self.assertEqual(sampler_call_kwargs[-1]["stop_token_ids"], [])
      self.assertIsNone(sampler_call_kwargs[-1]["_eos_token_id"])
      self.assertEqual(sampler_call_kwargs[-1]["min_tokens"], G)
      sa = np.load(os.path.join(tmp, "sampler_logprobs.npz"))
      self.assertEqual(sa["gen_ids"].shape, (B, G))
      self.assertTrue(os.path.isfile(os.path.join(tmp, "router_indices.npz")))

      # Verify --stage all launches isolated sampler & trainer subprocesses by default
      subproc_cmds = []
      with (
          mock.patch.object(audit_mod, "stage_tokenize", return_value=tokens),
          mock.patch.object(audit_mod.subprocess, "run", side_effect=lambda cmd, check: subproc_cmds.append(cmd)),
          mock.patch.object(audit_mod, "stage_compare", return_value={"is_oob_ratio": 0.0}),
      ):
        audit_mod.main(["--stage", "all", "--out-dir", tmp])
      self.assertEqual(len(subproc_cmds), 2)
      self.assertEqual(subproc_cmds[0][-2:], ["--stage", "sampler"])
      self.assertEqual(subproc_cmds[1][-2:], ["--stage", "trainer"])


# --------------------------------------------------------- Hybrid Qwen3.5 Toy Model & Module Divergence Probe Tests

import module_divergence_probe as mdp


class _ToyLinear(nnx.Module):

  def __init__(self, kernel: np.ndarray, dtype=jnp.bfloat16):
    self.dtype = dtype
    self.kernel = nnx.Param(jnp.asarray(kernel, dtype=dtype))
    self.kernel_scale = None

  def __call__(self, x):
    x_in = x.astype(self.dtype).astype(jnp.float32)
    w = self.kernel[...].astype(jnp.float32)
    return jnp.matmul(x_in, w).astype(self.dtype)


class _ToyRMSNorm(nnx.Module):

  def __init__(self, scale: np.ndarray, eps: float = 1e-6, dtype=None):
    self.dtype = dtype
    self.scale = nnx.Param(jnp.asarray(scale, dtype=jnp.float32))
    self.eps = eps

  def __call__(self, x):
    xf = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
    out_dtype = self.dtype if self.dtype is not None else x.dtype
    return (xf / rms * self.scale[...]).astype(out_dtype)


class _ToyGDN(nnx.Module):

  def __init__(self, w_qkvz: np.ndarray, w_ba: np.ndarray, w_out: np.ndarray):
    self.in_proj_qkvz = _ToyLinear(w_qkvz)
    self.in_proj_ba = _ToyLinear(w_ba)
    self.out_proj = _ToyLinear(w_out)

  def __call__(self, inputs, decoder_segment_ids=None, model_mode="train", **kwargs):
    del decoder_segment_ids, model_mode, kwargs
    qkvz = self.in_proj_qkvz(inputs)
    ba = self.in_proj_ba(inputs)
    d = inputs.shape[-1]
    mixed = jax.nn.silu(qkvz[..., :d].astype(jnp.float32)) * jax.nn.sigmoid(ba[..., :1].astype(jnp.float32))
    return self.out_proj(mixed.astype(inputs.dtype))


class _ToyFullAttnInner(nnx.Module):

  def __init__(self, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray):
    self.query = _ToyLinear(w_q)
    self.key = _ToyLinear(w_k)
    self.value = _ToyLinear(w_v)


class _ToyFullAttention(nnx.Module):

  def __init__(self, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray, w_o: np.ndarray):
    self.attention = _ToyFullAttnInner(w_q, w_k, w_v)
    self.out = _ToyLinear(w_o)

  def __call__(self, inputs, decoder_segment_ids=None, decoder_positions=None, deterministic=True, model_mode="train", **kwargs):
    del decoder_segment_ids, decoder_positions, deterministic, model_mode, kwargs
    q = self.attention.query(inputs)
    k = self.attention.key(inputs)
    v = self.attention.value(inputs)
    gate = jax.nn.sigmoid(jnp.mean((q.astype(jnp.float32) + k.astype(jnp.float32)), axis=-1, keepdims=True))
    return self.out((v.astype(jnp.float32) * gate).astype(inputs.dtype)), None


class _ToySharedExpert(nnx.Module):

  def __init__(self, wi_0: np.ndarray, wi_1: np.ndarray, wo: np.ndarray):
    self.wi_0 = _ToyLinear(wi_0)
    self.wi_1 = _ToyLinear(wi_1)
    self.wo = _ToyLinear(wo)

  def __call__(self, x):
    x_bf16 = x.astype(jnp.bfloat16)
    h0 = jax.nn.silu(self.wi_0(x_bf16).astype(jnp.float32))
    h1 = self.wi_1(x_bf16).astype(jnp.float32)
    return self.wo((h0 * h1).astype(jnp.bfloat16))


class _ToyRoutedMoE(nnx.Module):

  def __init__(self, gate_w: np.ndarray, wi_0: np.ndarray, wi_1: np.ndarray, wo: np.ndarray, num_experts_per_tok: int = 2, prefuse: bool = False):
    self.config = types.SimpleNamespace(num_experts_per_tok=num_experts_per_tok, float32_weight_sum=True)
    self.gate = _ToyLinear(gate_w, dtype=jnp.float32)
    self.weight_dtype = jnp.bfloat16
    self.quant = None
    self.is_hash_routing = False
    self.wi_kernel_axes = ("expert", "embed", "mlp")
    self.wo_kernel_axes = ("expert", "mlp", "embed")
    if prefuse:
      wi_cat = np.concatenate([wi_0, wi_1], axis=-1)
      self.wi = nnx.Param(jnp.asarray(wi_cat, dtype=jnp.bfloat16))
      self.wi_scale = None
      self.wi_0 = None
      self.wi_1 = None
      self.wi_0_scale = None
      self.wi_1_scale = None
    else:
      self.wi = None
      self.wi_scale = None
      self.wi_0 = nnx.Param(jnp.asarray(wi_0, dtype=jnp.bfloat16))
      self.wi_1 = nnx.Param(jnp.asarray(wi_1, dtype=jnp.bfloat16))
      self.wi_0_scale = None
      self.wi_1_scale = None
    self.wo = nnx.Param(jnp.asarray(wo, dtype=jnp.bfloat16))
    self.wo_scale = None

  def _dequant(self, param, scale_param):
    w = param[...].astype(jnp.float32)
    if scale_param is not None:
      s = scale_param[...].astype(jnp.float32)
      if s.ndim == w.ndim - 1:
        s = s[:, :, None, :] if s.shape[-1] == w.shape[-1] else s[..., None]
      if s.shape[-2] > 1 and w.shape[-2] % s.shape[-2] == 0:
        s = jnp.repeat(s, w.shape[-2] // s.shape[-2], axis=-2)
      if s.shape[-1] > 1 and w.shape[-1] % s.shape[-1] == 0:
        s = jnp.repeat(s, w.shape[-1] // s.shape[-1], axis=-1)
      w = w * s
    return w

  def __call__(self, inputs, forced_routed_experts=None, **kwargs):
    del kwargs
    orig_shape = inputs.shape
    d = orig_shape[-1]
    x_flat = jnp.reshape(inputs.astype(jnp.bfloat16).astype(jnp.float32), (-1, d))
    gate_logits = self.gate(jnp.reshape(inputs, orig_shape)).astype(jnp.float32)
    probs = jax.nn.softmax(jnp.reshape(gate_logits, (-1, gate_logits.shape[-1])), axis=-1)
    k = self.config.num_experts_per_tok
    if forced_routed_experts is not None:
      top_idx = jnp.reshape(jnp.asarray(forced_routed_experts, dtype=jnp.int32), (-1, k))
      top_w = jnp.take_along_axis(probs, top_idx, axis=-1)
    else:
      top_w, top_idx = jax.lax.top_k(probs, k)
    top_w = top_w / jnp.maximum(jnp.sum(top_w, axis=-1, keepdims=True), 1e-9)

    if self.wi is not None:
      wi_full = self._dequant(self.wi, self.wi_scale)
      mid = wi_full.shape[-1] // 2
      w0_all, w1_all = wi_full[..., :mid], wi_full[..., mid:]
    else:
      w0_all = self._dequant(self.wi_0, self.wi_0_scale)
      w1_all = self._dequant(self.wi_1, self.wi_1_scale)
    wo_all = self._dequant(self.wo, self.wo_scale)

    w0_sel = jnp.take(w0_all, top_idx, axis=0)
    w1_sel = jnp.take(w1_all, top_idx, axis=0)
    wo_sel = jnp.take(wo_all, top_idx, axis=0)
    h0 = jax.nn.silu(jnp.einsum("td,tkdm->tkm", x_flat, w0_sel))
    h1 = jnp.einsum("td,tkdm->tkm", x_flat, w1_sel)
    exp_out = jnp.einsum("tkm,tkmd->tkd", h0 * h1, wo_sel)
    combined = jnp.sum(exp_out * top_w[:, :, None], axis=1)
    return jnp.reshape(combined.astype(jnp.bfloat16), orig_shape), gate_logits


class _ToySparseMoeBlock(nnx.Module):

  def __init__(self, routed: _ToyRoutedMoE, shared: _ToySharedExpert, shared_gate_w: np.ndarray):
    self.routed_experts = routed
    self.shared_expert = shared
    self.shared_expert_gate = _ToyLinear(shared_gate_w, dtype=jnp.float32)

  def __call__(self, inputs, deterministic=False, forced_routed_experts=None, **kwargs):
    del deterministic, kwargs
    routed_out, _ = self.routed_experts(inputs, forced_routed_experts=forced_routed_experts)
    shared_out = self.shared_expert(inputs)
    gate_val = jax.nn.sigmoid(self.shared_expert_gate(inputs).astype(jnp.float32))
    return (routed_out.astype(jnp.float32) + gate_val * shared_out.astype(jnp.float32)).astype(jnp.bfloat16)


class _ToyDecoderLayer(nnx.Module):

  def __init__(self, layer_idx: int, in_ln: _ToyRMSNorm, attn: nnx.Module, post_ln: _ToyRMSNorm, mlp: _ToySparseMoeBlock, layer_type: str):
    self.layer_idx = layer_idx
    self.layer_type = layer_type
    self.input_layernorm = in_ln
    self.attention = attn
    self.post_attention_layernorm = post_ln
    self.mlp = mlp

  def __call__(
      self,
      inputs,
      decoder_segment_ids=None,
      decoder_positions=None,
      deterministic=True,
      model_mode="train",
      kv_caches=None,
      attention_metadata=None,
      forced_routed_experts=None,
      **kwargs,
  ):
    del kv_caches, attention_metadata, kwargs
    x = inputs[0] if isinstance(inputs, tuple) else inputs
    norm1 = self.input_layernorm(x)
    if self.layer_type == "linear_attention":
      attn_out = self.attention(norm1, decoder_segment_ids=decoder_segment_ids, model_mode=model_mode)
    else:
      attn_out, _ = self.attention(norm1, decoder_segment_ids, decoder_positions, deterministic, model_mode)
    hidden = (x.astype(jnp.float32) + attn_out.astype(jnp.float32)).astype(x.dtype)
    norm2 = self.post_attention_layernorm(hidden)
    mlp_out = self.mlp(norm2, deterministic=deterministic, forced_routed_experts=forced_routed_experts)
    out = (hidden.astype(jnp.float32) + mlp_out.astype(jnp.float32)).astype(x.dtype)
    return (out,) if isinstance(inputs, tuple) else out


class _ToyScannableBlock(nnx.Module):

  def __init__(self, layers_list: list[_ToyDecoderLayer]):
    for i, lyr in enumerate(layers_list):
      setattr(self, f"layer_{i}", lyr)
    self._num = len(layers_list)

  def __call__(self, carry, _unused_scan_in, decoder_segment_ids=None, decoder_positions=None, deterministic=True, model_mode="train"):
    h = carry
    for i in range(self._num):
      lyr = getattr(self, f"layer_{i}")
      h = lyr(h, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, deterministic=deterministic, model_mode=model_mode)
    return h, None


def _build_hybrid_qwen3_5_toy_model(
    seed: int = 123,
    *,
    attention: str = "flash",
    prefuse_moe: bool = False,
    scanned: bool = False,
    num_layers: int = 4,
    emb_dim: int = 128,
    mlp_dim: int = 128,
    vocab_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
):
  rng = np.random.default_rng(seed)
  emb_table = (rng.standard_normal((vocab_size, emb_dim)) * 0.08).astype(np.float32)
  dec_norm_w = (1.0 + rng.standard_normal((emb_dim,)) * 0.05).astype(np.float32)
  logits_w = (rng.standard_normal((emb_dim, vocab_size)) * 0.08).astype(np.float32)

  layers_list = []
  for i in range(num_layers):
    in_ln = _ToyRMSNorm((1.0 + rng.standard_normal((emb_dim,)) * 0.05).astype(np.float32))
    post_ln = _ToyRMSNorm((1.0 + rng.standard_normal((emb_dim,)) * 0.05).astype(np.float32), dtype=jnp.float32)
    if i < num_layers - 1:
      ltype = "linear_attention"
      w_qkvz = (rng.standard_normal((emb_dim, emb_dim * 2)) * 0.06).astype(np.float32)
      w_ba = (rng.standard_normal((emb_dim, 4)) * 0.06).astype(np.float32)
      w_out = (rng.standard_normal((emb_dim, emb_dim)) * 0.06).astype(np.float32)
      attn = _ToyGDN(w_qkvz, w_ba, w_out)
    else:
      ltype = "full_attention"
      w_q = (rng.standard_normal((emb_dim, emb_dim)) * 0.06).astype(np.float32)
      w_k = (rng.standard_normal((emb_dim, emb_dim)) * 0.06).astype(np.float32)
      w_v = (rng.standard_normal((emb_dim, emb_dim)) * 0.06).astype(np.float32)
      w_o = (rng.standard_normal((emb_dim, emb_dim)) * 0.06).astype(np.float32)
      attn = _ToyFullAttention(w_q, w_k, w_v, w_o)

    gate_w = (rng.standard_normal((emb_dim, num_experts)) * 0.2).astype(np.float32)
    wi_0 = (rng.standard_normal((num_experts, emb_dim, mlp_dim)) * 0.12).astype(np.float32)
    wi_1 = (rng.standard_normal((num_experts, emb_dim, mlp_dim)) * 0.12).astype(np.float32)
    wo = (rng.standard_normal((num_experts, mlp_dim, emb_dim)) * 0.12).astype(np.float32)
    routed = _ToyRoutedMoE(gate_w, wi_0, wi_1, wo, num_experts_per_tok=top_k, prefuse=prefuse_moe)

    s_wi_0 = (rng.standard_normal((emb_dim, mlp_dim)) * 0.06).astype(np.float32)
    s_wi_1 = (rng.standard_normal((emb_dim, mlp_dim)) * 0.06).astype(np.float32)
    s_wo = (rng.standard_normal((mlp_dim, emb_dim)) * 0.06).astype(np.float32)
    shared = _ToySharedExpert(s_wi_0, s_wi_1, s_wo)
    s_gate = (rng.standard_normal((emb_dim, 1)) * 0.1).astype(np.float32)
    mlp = _ToySparseMoeBlock(routed, shared, s_gate)
    layers_list.append(_ToyDecoderLayer(i, in_ln, attn, post_ln, mlp, ltype))

  cfg = types.SimpleNamespace(
      model_name="qwen3.5-35b-a3b",
      attention=attention,
      dtype=jnp.bfloat16,
      weight_dtype=jnp.bfloat16,
      quantization="",
      fp8_moe=False,
      sparse_matmul=True,
      use_gmm_v2=True,
      prefuse_moe_weights=prefuse_moe,
      num_decoder_layers=num_layers,
      inhomogeneous_layer_cycle_interval=num_layers,
      scan_layers=scanned,
      param_scan_axis=0,
      float32_gate_logits=True,
      float32_weight_sum=True,
      logits_dot_in_fp32=True,
  )

  class _ToyEmbedder(nnx.Module):

    def __init__(self, table: np.ndarray):
      self.embedding = nnx.Param(jnp.asarray(table, dtype=jnp.bfloat16))

    def __call__(self, ids):
      return jnp.take(self.embedding[...], ids, axis=0)

  class _ToyDecoder(nnx.Module):

    def __init__(self):
      self.config = cfg
      self.decoder_norm = _ToyRMSNorm(dec_norm_w)
      self.logits_dense = _ToyLinear(logits_w)
      if scanned:
        self.layers = _ToyScannableBlock(layers_list)
      else:
        for i, lyr in enumerate(layers_list):
          setattr(self, f"layers_{i}", lyr)

    def apply_output_head(self, token_embedder, y, deterministic, model_mode):
      del token_embedder, deterministic, model_mode
      h = self.decoder_norm(y)
      return self.logits_dense(h).astype(jnp.float32)

  class _ToyModel(nnx.Module):

    def __init__(self):
      self.config = cfg
      self.token_embedder = _ToyEmbedder(emb_table)
      self.decoder = _ToyDecoder()

    def forward_sampler_step(self, token_ids_1d: jax.Array, query_start_loc: jax.Array, seq_lens: jax.Array):
      """Simulates vLLM Sampler flattened `[num_batched_tokens, 1, D]` execution with `attention_metadata`."""
      h = self.token_embedder(token_ids_1d[:, None])  # [N_tok, 1, D]
      meta = types.SimpleNamespace(query_start_loc=query_start_loc, seq_lens=seq_lens)
      for i in range(num_layers):
        lyr = getattr(self.decoder, f"layers_{i}")
        h = lyr(h, model_mode="prefill", attention_metadata=meta)
      return h

    def forward_trainer(self, token_ids_2d: jax.Array, segment_ids: jax.Array | None = None):
      """Simulates TrainerWorker `[B, S, D]` forward pass (unscanned or scanned)."""
      h = self.token_embedder(token_ids_2d)  # [B, S, D]
      if scanned:
        h, _ = self.decoder.layers(h, None, decoder_segment_ids=segment_ids, model_mode="train")
      else:
        for i in range(num_layers):
          lyr = getattr(self.decoder, f"layers_{i}")
          h = lyr(h, decoder_segment_ids=segment_ids, model_mode="train")
      return h

  return _ToyModel()


class ModuleDivergenceProbeTest(unittest.TestCase):

  def test_select_probe_positions_and_parse_layers(self):
    pos_all = mdp.select_probe_positions(prompt_len=8, gen_len=6, max_tokens=64)
    np.testing.assert_array_equal(pos_all, np.arange(14, dtype=np.int32))

    pos_sub = mdp.select_probe_positions(prompt_len=100, gen_len=100, max_tokens=16)
    self.assertLessEqual(len(pos_sub), 16)
    # Must include boundaries: first prompt (0), last prompt (99), first decode (100), last decode (199)
    for boundary in (0, 99, 100, 199):
      self.assertIn(boundary, pos_sub.tolist())

    self.assertIsNone(mdp.parse_probe_layers("all"))
    self.assertIsNone(mdp.parse_probe_layers(None))
    self.assertEqual(mdp.parse_probe_layers("0, 2-4, 7"), {0, 2, 3, 4, 7})
    self.assertEqual(mdp.parse_probe_layers([1, 3]), {1, 3})

  def test_bf16_sampler_vs_unscanned_packed_and_scanned_trainer_zero_divergence(self):
    rng = np.random.default_rng(2026)
    P, G = 6, 4
    seq0_tokens = rng.integers(1, 60, size=(P + G,), dtype=np.int32)
    seq1_tokens = rng.integers(1, 60, size=(P + G,), dtype=np.int32)
    probe_pos = mdp.select_probe_positions(P, G, max_tokens=64)
    prompt_tokens_probe = seq0_tokens[probe_pos]
    target_next_tokens = seq0_tokens[np.minimum(probe_pos + 1, P + G - 1)]

    sampler_model = _build_hybrid_qwen3_5_toy_model(seed=99, attention="vllm_rpa", prefuse_moe=True, scanned=False)
    with mdp.ModuleProbeTap(
        sampler_model,
        role="sampler",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
    ) as sa_tap:
      # Chunked prefill step 1: seq0[0:3] + seq1[0:2] batched together in flattened [5, 1, D]
      chunk1 = jnp.asarray(np.concatenate([seq0_tokens[:3], seq1_tokens[:2]]), dtype=jnp.int32)
      sampler_model.forward_sampler_step(
          chunk1,
          query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32),
          seq_lens=jnp.array([3, 2], dtype=jnp.int32),
      )
      # Chunked prefill step 2: seq0[3:6] + seq1[2:6] batched together in flattened [7, 1, D]
      chunk2 = jnp.asarray(np.concatenate([seq0_tokens[3:P], seq1_tokens[2:P]]), dtype=jnp.int32)
      sampler_model.forward_sampler_step(
          chunk2,
          query_start_loc=jnp.array([0, 3, 7], dtype=jnp.int32),
          seq_lens=jnp.array([P, P], dtype=jnp.int32),
      )
      # Token-by-token decode steps for G tokens: [2, 1, D] per step
      for step in range(G):
        cur_len = P + step + 1
        dec_toks = jnp.array([seq0_tokens[P + step], seq1_tokens[P + step]], dtype=jnp.int32)
        sampler_model.forward_sampler_step(
            dec_toks,
            query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
            seq_lens=jnp.array([cur_len, cur_len], dtype=jnp.int32),
        )

    sa_probes = sa_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )

    # 1. Compare against Unscanned 2D Trainer
    trainer_2d = _build_hybrid_qwen3_5_toy_model(seed=99, attention="flash", prefuse_moe=False, scanned=False)
    batch_2d = jnp.asarray(np.stack([seq0_tokens, seq1_tokens], axis=0), dtype=jnp.int32)
    with mdp.ModuleProbeTap(
        trainer_2d,
        role="trainer",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
    ) as tr_2d_tap:
      trainer_2d.forward_trainer(batch_2d)
    tr_2d_probes = tr_2d_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )
    iso_2d_probes = mdp.run_isolated_trainer_replay(
        trainer_2d,
        sa_probes,
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
      rep_2d = mdp.compare_module_probes(sa_probes, tr_2d_probes, iso_2d_probes, tag="BF16 vs BF16 (2D)")
    for rec in rep_2d["records"]:
      self.assertLess(rec["cumulative"]["all"]["rel_l2"], 1e-6, msg=f"Non-zero cumulative rel_l2 on {rec['key']}")
      if rec["isolated"] is not None:
        self.assertLess(rec["isolated"]["all"]["rel_l2"], 1e-6, msg=f"Non-zero isolated rel_l2 on {rec['key']}")
      if rec["router"] is not None:
        self.assertEqual(rec["router"]["cum"]["top1_agree"], 1.0)
        self.assertEqual(rec["router"]["cum"]["topk_jaccard"], 1.0)

    # 2. Compare against 1D Sequence-Packed Trainer where seq 0 is in microbatch 1, row 1, offset by 5 tokens
    packed_width = 28
    mb0_pack = jnp.ones((2, packed_width), dtype=jnp.int32)
    mb1_row0 = jnp.ones((packed_width,), dtype=jnp.int32)
    # Place seq1 at [0:10] (segment 1), padding at [10:14] (segment 0), and seq0 at [14:24] (segment 2) in row 1 of microbatch 1
    mb1_row1_np = np.zeros((packed_width,), dtype=np.int32)
    mb1_row1_np[: P + G] = seq1_tokens
    seq0_packed_cols = np.arange(14, 14 + P + G, dtype=np.int32)
    mb1_row1_np[seq0_packed_cols] = seq0_tokens
    mb1_pack = jnp.asarray(np.stack([mb1_row0, mb1_row1_np], axis=0), dtype=jnp.int32)

    with mdp.ModuleProbeTap(
        trainer_2d,
        role="trainer",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
        packed_seq0_row=1,
        packed_seq0_positions=seq0_packed_cols,
        trainer_microbatch_idx=1,
    ) as tr_pack_tap:
      trainer_2d.forward_trainer(mb0_pack)  # Microbatch 0 (ignored by trainer_microbatch_idx=1)
      trainer_2d.forward_trainer(mb1_pack)  # Microbatch 1, row 1 (captured!)

    tr_pack_probes = tr_pack_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )
    with redirect_stdout(io.StringIO()):
      rep_pack = mdp.compare_module_probes(sa_probes, tr_pack_probes, iso_2d_probes, tag="BF16 vs BF16 (1D Packed)")
    for rec in rep_pack["records"]:
      self.assertLess(rec["cumulative"]["all"]["rel_l2"], 1e-6, msg=f"1D packed mismatch on {rec['key']}")

    # 3. Compare against Scanned Trainer (`Qwen3_5ScannableBlock`)
    trainer_scanned = _build_hybrid_qwen3_5_toy_model(seed=99, attention="flash", prefuse_moe=False, scanned=True)
    with mdp.ModuleProbeTap(
        trainer_scanned,
        role="trainer",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
    ) as tr_scan_tap:
      trainer_scanned.forward_trainer(batch_2d)
    tr_scan_probes = tr_scan_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )
    iso_scan_probes = mdp.run_isolated_trainer_replay(
        trainer_scanned,
        sa_probes,
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )
    with redirect_stdout(io.StringIO()):
      rep_scan = mdp.compare_module_probes(sa_probes, tr_scan_probes, iso_scan_probes, tag="BF16 vs BF16 (Scanned)")
    for rec in rep_scan["records"]:
      self.assertLess(rec["cumulative"]["all"]["rel_l2"], 1e-6, msg=f"Scanned mismatch on {rec['key']}")
      if rec["isolated"] is not None:
        self.assertLess(rec["isolated"]["all"]["rel_l2"], 1e-6, msg=f"Scanned isolated mismatch on {rec['key']}")

  def test_fp8_moe_sampler_vs_bf16_trainer_pinpoints_routed_experts_only(self):
    rng = np.random.default_rng(404)
    P, G = 6, 4
    seq0_tokens = rng.integers(1, 60, size=(P + G,), dtype=np.int32)
    probe_pos = mdp.select_probe_positions(P, G, max_tokens=64)
    prompt_tokens_probe = seq0_tokens[probe_pos]
    target_next_tokens = seq0_tokens[np.minimum(probe_pos + 1, P + G - 1)]

    sampler_fp8moe = _build_hybrid_qwen3_5_toy_model(seed=321, attention="vllm_rpa", prefuse_moe=True, scanned=False)
    trainer_bf16 = _build_hybrid_qwen3_5_toy_model(seed=321, attention="flash", prefuse_moe=False, scanned=False)

    # Quantize ONLY RoutedMoE on the Sampler to FP8
    n_q = audit_mod.quantize_moe_fp8(sampler_fp8moe, scale_mode="per_channel")
    self.assertEqual(n_q, 4)

    with mdp.ModuleProbeTap(
        sampler_fp8moe,
        role="sampler",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
    ) as sa_tap:
      sampler_fp8moe.forward_sampler_step(
          jnp.asarray(seq0_tokens, dtype=jnp.int32),
          query_start_loc=jnp.array([0, P + G], dtype=jnp.int32),
          seq_lens=jnp.array([P + G], dtype=jnp.int32),
      )
    sa_probes = sa_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )

    with mdp.ModuleProbeTap(
        trainer_bf16,
        role="trainer",
        probe_positions=probe_pos,
        prompt_len=P,
        gen_len=G,
    ) as tr_tap:
      trainer_bf16.forward_trainer(jnp.asarray(seq0_tokens[None, :], dtype=jnp.int32))
    tr_probes = tr_tap.get_probes(
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )

    iso_probes = mdp.run_isolated_trainer_replay(
        trainer_bf16,
        sa_probes,
        target_next_tokens=target_next_tokens,
        prompt_tokens_probe=prompt_tokens_probe,
        temperature=1.0,
    )

    with tempfile.TemporaryDirectory() as tmp:
      buf = io.StringIO()
      with redirect_stdout(buf):
        report = mdp.compare_module_probes(sa_probes, tr_probes, iso_probes, tag="Sampler=FP8MOE vs Trainer=BF16", out_dir=tmp)
      out_text = buf.getvalue()
      self.assertIn("MODULE-BY-MODULE VALUE DIVERGENCE", out_text)
      self.assertTrue(os.path.isfile(os.path.join(tmp, "module_divergence_report.json")))
      self.assertTrue(os.path.isfile(os.path.join(tmp, "module_divergence_metrics.npz")))

    by_key = {r["key"]: r for r in report["records"]}

    # In Layer 0, everything before mlp.routed_experts has 0 cumulative divergence
    for clean_l0 in (
        "layer_0.input_layernorm",
        "layer_0.attn.in_proj_qkvz",
        "layer_0.attn.in_proj_ba",
        "layer_0.attn.out",
        "layer_0.post_attn_residual",
        "layer_0.post_attention_layernorm",
        "layer_0.mlp.gate_logits",
        "layer_0.mlp.shared_expert",
        "layer_0.mlp.shared_expert_gate",
    ):
      self.assertLess(by_key[clean_l0]["cumulative"]["all"]["rel_l2"], 1e-6, msg=f"Expected 0 cum_rel_l2 at {clean_l0}")

    # Layer 0 mlp.routed_experts is where divergence first originates!
    self.assertGreater(by_key["layer_0.mlp.routed_experts"]["isolated"]["all"]["rel_l2"], 1e-3)
    self.assertGreater(by_key["layer_0.mlp.routed_experts"]["cumulative"]["all"]["rel_l2"], 1e-3)

    # In Layers 1, 2, 3: cumulative divergence is > 0 everywhere (due to Layer 0 drift),
    # BUT isolated divergence is 0 (< 1e-6) on all non-MoE submodules and > 1e-3 ONLY on mlp.routed_experts (and mlp.out / layer_out)!
    for lyr in (1, 2, 3):
      for sub in ("input_layernorm", "post_attention_layernorm", "mlp.gate_logits", "mlp.shared_expert", "mlp.shared_expert_gate"):
        key = f"layer_{lyr}.{sub}"
        self.assertGreater(by_key[key]["cumulative"]["all"]["rel_l2"], 1e-4, msg=f"Expected >0 cum_rel_l2 at {key}")
        self.assertLess(by_key[key]["isolated"]["all"]["rel_l2"], 1e-6, msg=f"Expected 0 iso_rel_l2 at {key}")
      routed_key = f"layer_{lyr}.mlp.routed_experts"
      self.assertGreater(by_key[routed_key]["isolated"]["all"]["rel_l2"], 1e-3)

    # Top isolated bottleneck must be mlp.routed_experts (or mlp.out)
    top_iso_mods = {b["module"] for b in report["top_isolated_bottlenecks"][:4]}
    self.assertIn("mlp.routed_experts", top_iso_mods)
    self.assertGreater(
        report["family_summary"]["mlp.routed_experts"]["iso_rel_l2_mean"],
        1000.0 * (report["family_summary"]["mlp.shared_expert"]["iso_rel_l2_mean"] + 1e-12),
    )

  def test_sync_trainer_weights_to_sampler_and_mlperf_v5p_cli_integration(self):
    trainer_model = _build_hybrid_qwen3_5_toy_model(seed=111, attention="flash", prefuse_moe=False, scanned=False)
    sampler_model = _build_hybrid_qwen3_5_toy_model(seed=222, attention="vllm_rpa", prefuse_moe=True, scanned=False)

    # Before sync, weights differ
    w_tr_0 = np.asarray(trainer_model.decoder.layers_0.mlp.routed_experts.wi_0[...], dtype=np.float32)
    w_tr_1 = np.asarray(trainer_model.decoder.layers_0.mlp.routed_experts.wi_1[...], dtype=np.float32)
    w_sa_before = np.asarray(sampler_model.decoder.layers_0.mlp.routed_experts.wi[...], dtype=np.float32)
    self.assertFalse(np.allclose(np.concatenate([w_tr_0, w_tr_1], axis=-1), w_sa_before))

    # Sync Trainer -> Sampler via transfer_state_directly (fusing wi_0 + wi_1 -> wi)
    mdp.sync_trainer_weights_to_sampler(trainer_model, sampler_model)
    w_sa_after = np.asarray(sampler_model.decoder.layers_0.mlp.routed_experts.wi[...], dtype=np.float32)
    np.testing.assert_allclose(np.concatenate([w_tr_0, w_tr_1], axis=-1), w_sa_after, atol=1e-6)

    # Verify CLI --mlperf-v5p and --probe-modules integration in stage_compare
    with tempfile.TemporaryDirectory() as tmp:
      B, P, G = 2, 6, 4
      tokens = np.arange(1, B * P + 1, dtype=np.int32).reshape(B, P)
      gen_ids = np.arange(1, B * G + 1, dtype=np.int32).reshape(B, G)
      gen_lp = np.full((B, G), -0.3, dtype=np.float32)
      np.savez(os.path.join(tmp, "sampler_logprobs.npz"), tokens=tokens, gen_ids=gen_ids, gen_logp=gen_lp, gen_lens=np.full(B, G, dtype=np.int32))
      np.savez(os.path.join(tmp, "trainer_logprobs.npz"), tokens=tokens, gen_ids=gen_ids, gen_logp=gen_lp, gen_lens=np.full(B, G, dtype=np.int32))

      probe_pos = mdp.select_probe_positions(P, G, max_tokens=16)
      seq0_full = np.concatenate([tokens[0], gen_ids[0]])
      with mdp.ModuleProbeTap(sampler_model, role="sampler", probe_positions=probe_pos, prompt_len=P, gen_len=G) as sa_tap:
        sampler_model.forward_sampler_step(
            jnp.asarray(seq0_full, dtype=jnp.int32),
            query_start_loc=jnp.array([0, P + G], dtype=jnp.int32),
            seq_lens=jnp.array([P + G], dtype=jnp.int32),
        )
      sa_probes = sa_tap.get_probes(target_next_tokens=seq0_full, prompt_tokens_probe=seq0_full)
      np.savez(os.path.join(tmp, "sampler_module_probes.npz"), **sa_probes)

      with mdp.ModuleProbeTap(trainer_model, role="trainer", probe_positions=probe_pos, prompt_len=P, gen_len=G) as tr_tap:
        trainer_model.forward_trainer(jnp.asarray(seq0_full[None, :], dtype=jnp.int32))
      tr_probes = tr_tap.get_probes(target_next_tokens=seq0_full, prompt_tokens_probe=seq0_full)
      np.savez(os.path.join(tmp, "trainer_module_probes.npz"), **tr_probes)

      iso_probes = mdp.run_isolated_trainer_replay(trainer_model, sa_probes, target_next_tokens=seq0_full, prompt_tokens_probe=seq0_full)
      np.savez(os.path.join(tmp, "trainer_isolated_probes.npz"), **iso_probes)

      buf = io.StringIO()
      with redirect_stdout(buf):
        res = audit_mod.main(["--stage", "compare", "--mlperf-v5p", "--probe-modules", "--probe-layers", "0,3", "--out-dir", tmp, "--max-seq-token-per-tpu", "512"])
      out_str = buf.getvalue()
      self.assertIn("module_divergence", res)
      self.assertIn("  00 | input_layernorm", out_str)
      self.assertIn("  03 | input_layernorm", out_str)
      self.assertNotIn("  01 | input_layernorm", out_str)
      self.assertEqual(res["module_divergence"]["logprob_divergence"]["top1_agree"], 1.0)
      for rec in res["module_divergence"]["records"]:
        self.assertLess(rec["cumulative"]["all"]["rel_l2"], 1e-6)


if __name__ == "__main__":
  unittest.main()

