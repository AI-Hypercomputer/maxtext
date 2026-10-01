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

  routed0 = types.SimpleNamespace(
      is_hash_routing=False,
      weight_dtype=weight_dtype,
      quant=quantizations.ServeFp8WeightQuantization() if serve_fp8_moe else None,
      wi_kernel_axes=("expert", None, "mlp"),
      wo_kernel_axes=("expert", "mlp", None),
      wo=_w((4, 256, 256)),
      wo_scale=_s(moe_scale_shape),
      wi=_w((4, 256, 256)) if prefuse_moe else None,
      wi_scale=_s(moe_scale_shape) if prefuse_moe else None,
      wi_0=None if prefuse_moe else _w((4, 256, 256)),
      wi_1=None if prefuse_moe else _w((4, 256, 256)),
      wi_0_scale=None if prefuse_moe else _s(moe_scale_shape),
      wi_1_scale=None if prefuse_moe else _s(moe_scale_shape),
  )
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


if __name__ == "__main__":
  unittest.main()
