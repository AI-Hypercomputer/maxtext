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
"""Unit tests for `tools/rl_logprob_parity/run_parity_audit.py` and `module_divergence_probe.py`."""

import contextlib
from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import qwix

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOL_DIR))
for _p in (_TOOL_DIR, os.path.join(_REPO_ROOT, "src"), os.path.join(os.path.dirname(_REPO_ROOT), "tunix")):
  if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

# Stub optional runtime deps (`aqt`, `qwix.sparsity`, `orbax.step`, `omegaconf`, `metrax`) for CPU unit test envs.
for _mod_name in (
    "aqt",
    "aqt.jax",
    "aqt.jax.v2",
    "aqt.jax.v2.config",
    "aqt.jax.v2.aqt_tensor",
    "aqt.jax.v2.flax",
    "aqt.jax.v2.flax.aqt_flax",
    "aqt.jax.v2.tiled_dot_general",
    "aqt.jax.v2.calibration",
    "omegaconf",
    "metrax",
    "metrax.logging",
):
  try:
    __import__(_mod_name)
  except ModuleNotFoundError:
    sys.modules.setdefault(_mod_name, mock.MagicMock())

with contextlib.suppress(ImportError):
  from qwix._src import core as _qwix_core

  if not hasattr(_qwix_core, "sparsity"):
    setattr(_qwix_core, "sparsity", mock.MagicMock())

with contextlib.suppress(ImportError):
  import orbax.checkpoint.experimental.v1 as _ocp_v1

  if hasattr(_ocp_v1, "path") and not hasattr(_ocp_v1.path, "step"):
    setattr(_ocp_v1.path, "step", mock.MagicMock())

if "tunix" not in sys.modules:
  try:
    import tunix  # noqa: F401
  except Exception:
    _tunix_dir = os.path.join(os.path.dirname(_REPO_ROOT), "tunix", "tunix")
    if os.path.isdir(_tunix_dir):
      _pkg = types.ModuleType("tunix")
      _pkg.__path__ = [_tunix_dir]
      sys.modules["tunix"] = _pkg

from maxtext.layers import quantizations  # noqa: E402
import module_divergence_probe as mdp  # noqa: E402
import run_parity_audit as audit_mod  # noqa: E402
from tunix.rl import algo_core, common as rl_common  # noqa: E402
from tunix.utils import maxtext_utils as tunix_maxtext_utils  # noqa: E402

pytestmark = [pytest.mark.post_training]


# --------------------------------------------------------- Shared Mock & Toy Model Builders


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
  dtype_str = str(jnp.dtype(weight_dtype))
  w_scale = 50.0 if ("float8" in dtype_str or "int8" in dtype_str) else 0.03
  _w = lambda shape: nnx.Param(jnp.asarray((rng.standard_normal(shape) * w_scale).astype(np.float32), dtype=weight_dtype))
  _s = lambda shape: nnx.Param(jnp.full(shape, 0.02, dtype=jnp.float32)) if shape is not None else None

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
  q_scale_shape = (2, 1, 1) if dense_scale_shape else None
  attn3 = types.SimpleNamespace(
      query=_make_linear_stub((256, 4, 64), dtype=weight_dtype, scale_shape=q_scale_shape, serve_fp8=serve_fp8_dense)
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


class _ToyLinear(nnx.Module):

  def __init__(self, kernel: np.ndarray, dtype=jnp.bfloat16):
    self.kernel = nnx.Param(jnp.asarray(kernel, dtype=dtype))
    self.kernel_scale = None

  def __call__(self, x):
    return jnp.matmul(x.astype(jnp.float32), self.kernel[...].astype(jnp.float32)).astype(self.kernel[...].dtype)


class _ToyRMSNorm(nnx.Module):

  def __init__(self, scale: np.ndarray, eps: float = 1e-6, dtype=None):
    self.scale = nnx.Param(jnp.asarray(scale, dtype=jnp.float32))
    self.eps = eps
    self.dtype = dtype

  def __call__(self, x):
    xf = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
    return (xf / rms * self.scale[...]).astype(self.dtype if self.dtype is not None else x.dtype)


class _ToyGDN(nnx.Module):

  def __init__(self, w_qkvz: np.ndarray, w_ba: np.ndarray, w_out: np.ndarray):
    self.in_proj_qkvz = _ToyLinear(w_qkvz)
    self.in_proj_ba = _ToyLinear(w_ba)
    self.out_proj = _ToyLinear(w_out)

  def __call__(self, inputs, decoder_segment_ids=None, model_mode="train", **kwargs):
    del decoder_segment_ids, model_mode, kwargs
    qkvz, ba = self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
    d = inputs.shape[-1]
    mixed = jax.nn.silu(qkvz[..., :d].astype(jnp.float32)) * jax.nn.sigmoid(ba[..., :1].astype(jnp.float32))
    return self.out_proj(mixed.astype(inputs.dtype))


class _ToyFullAttention(nnx.Module):

  def __init__(self, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray, w_o: np.ndarray):
    self.attention = types.SimpleNamespace(query=_ToyLinear(w_q), key=_ToyLinear(w_k), value=_ToyLinear(w_v))
    self.out = _ToyLinear(w_o)

  def __call__(self, inputs, decoder_segment_ids=None, decoder_positions=None, deterministic=True, model_mode="train", **kwargs):
    del decoder_segment_ids, decoder_positions, deterministic, model_mode, kwargs
    q, k, v = self.attention.query(inputs), self.attention.key(inputs), self.attention.value(inputs)
    gate = jax.nn.sigmoid(jnp.mean((q.astype(jnp.float32) + k.astype(jnp.float32)), axis=-1, keepdims=True))
    return self.out((v.astype(jnp.float32) * gate).astype(inputs.dtype)), None


class _ToySharedExpert(nnx.Module):

  def __init__(self, wi_0: np.ndarray, wi_1: np.ndarray, wo: np.ndarray):
    self.wi_0, self.wi_1, self.wo = _ToyLinear(wi_0), _ToyLinear(wi_1), _ToyLinear(wo)

  def __call__(self, x):
    h0 = jax.nn.silu(self.wi_0(x).astype(jnp.float32))
    h1 = self.wi_1(x).astype(jnp.float32)
    return self.wo((h0 * h1).astype(jnp.bfloat16))


class _ToyRoutedMoE(nnx.Module):

  def __init__(
      self,
      gate_w: np.ndarray,
      wi_0: np.ndarray,
      wi_1: np.ndarray,
      wo: np.ndarray,
      num_experts_per_tok: int = 2,
      prefuse: bool = False,
  ):
    self.config = types.SimpleNamespace(num_experts_per_tok=num_experts_per_tok, float32_weight_sum=True)
    self.gate = _ToyLinear(gate_w, dtype=jnp.float32)
    self.weight_dtype = jnp.bfloat16
    self.quant = None
    self.is_hash_routing = False
    self.wi_kernel_axes = ("expert", "embed", "mlp")
    self.wo_kernel_axes = ("expert", "mlp", "embed")
    self.wi_scale = self.wi_0_scale = self.wi_1_scale = self.wo_scale = None
    if prefuse:
      self.wi = nnx.Param(jnp.asarray(np.concatenate([wi_0, wi_1], axis=-1), dtype=jnp.bfloat16))
      self.wi_0 = self.wi_1 = None
    else:
      self.wi = None
      self.wi_0 = nnx.Param(jnp.asarray(wi_0, dtype=jnp.bfloat16))
      self.wi_1 = nnx.Param(jnp.asarray(wi_1, dtype=jnp.bfloat16))
    self.wo = nnx.Param(jnp.asarray(wo, dtype=jnp.bfloat16))

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
    x_flat = jnp.reshape(inputs.astype(jnp.bfloat16).astype(jnp.float32), (-1, orig_shape[-1]))
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

    h0 = jax.nn.silu(jnp.einsum("td,tkdm->tkm", x_flat, jnp.take(w0_all, top_idx, axis=0)))
    h1 = jnp.einsum("td,tkdm->tkm", x_flat, jnp.take(w1_all, top_idx, axis=0))
    exp_out = jnp.einsum("tkm,tkmd->tkd", h0 * h1, jnp.take(wo_all, top_idx, axis=0))
    combined = jnp.sum(exp_out * top_w[:, :, None], axis=1)
    return jnp.reshape(combined.astype(jnp.bfloat16), orig_shape), gate_logits


class _ToySparseMoeBlock(nnx.Module):

  def __init__(self, routed: _ToyRoutedMoE, shared: _ToySharedExpert, shared_gate_w: np.ndarray):
    self.routed_experts = routed
    self.shared_expert = shared
    self.shared_expert_gate = _ToyLinear(shared_gate_w)

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
      h = getattr(self, f"layer_{i}")(
          h, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, deterministic=deterministic, model_mode=model_mode
      )
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
  _randn = lambda shape, scale=0.06, bias=0.0: (bias + rng.standard_normal(shape) * scale).astype(np.float32)
  emb_table, dec_norm_w, logits_w = _randn((vocab_size, emb_dim), 0.08), _randn((emb_dim,), 0.05, bias=1.0), _randn((emb_dim, vocab_size), 0.08)

  layers_list = []
  for i in range(num_layers):
    in_ln = _ToyRMSNorm(_randn((emb_dim,), 0.05, bias=1.0))
    post_ln = _ToyRMSNorm(_randn((emb_dim,), 0.05, bias=1.0), dtype=jnp.float32)
    if i < num_layers - 1:
      ltype = "linear_attention"
      attn = _ToyGDN(_randn((emb_dim, emb_dim * 2)), _randn((emb_dim, 4)), _randn((emb_dim, emb_dim)))
    else:
      ltype = "full_attention"
      attn = _ToyFullAttention(_randn((emb_dim, emb_dim)), _randn((emb_dim, emb_dim)), _randn((emb_dim, emb_dim)), _randn((emb_dim, emb_dim)))
    routed = _ToyRoutedMoE(
        _randn((emb_dim, num_experts), 0.2),
        _randn((num_experts, emb_dim, mlp_dim), 0.12),
        _randn((num_experts, emb_dim, mlp_dim), 0.12),
        _randn((num_experts, mlp_dim, emb_dim), 0.12),
        num_experts_per_tok=top_k,
        prefuse=prefuse_moe,
    )
    shared = _ToySharedExpert(_randn((emb_dim, mlp_dim)), _randn((emb_dim, mlp_dim)), _randn((mlp_dim, emb_dim)))
    mlp = _ToySparseMoeBlock(routed, shared, _randn((emb_dim, 1), 0.1))
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
      return self.logits_dense(self.decoder_norm(y)).astype(jnp.float32)

  class _ToyModel(nnx.Module):

    def __init__(self):
      self.config = cfg
      self.token_embedder = _ToyEmbedder(emb_table)
      self.decoder = _ToyDecoder()

    def forward_sampler_step(self, token_ids_1d: jax.Array, query_start_loc: jax.Array, seq_lens: jax.Array):
      h = self.token_embedder(token_ids_1d[:, None])
      meta = types.SimpleNamespace(query_start_loc=query_start_loc, seq_lens=seq_lens)
      for i in range(num_layers):
        h = getattr(self.decoder, f"layers_{i}")(h, model_mode="prefill", attention_metadata=meta)
      return h

    def forward_trainer(self, token_ids_2d: jax.Array, segment_ids: jax.Array | None = None):
      h = self.token_embedder(token_ids_2d)
      if scanned:
        h, _ = self.decoder.layers(h, None, decoder_segment_ids=segment_ids, model_mode="train")
      else:
        for i in range(num_layers):
          h = getattr(self.decoder, f"layers_{i}")(h, decoder_segment_ids=segment_ids, model_mode="train")
      return h

  return _ToyModel()


# --------------------------------------------------------- Unit Test Suites


class QuantizationAndScaleTest(unittest.TestCase):

  def test_quantize_moe_fp8_matches_maxtext_and_roundtrips_all_scale_modes(self):
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
        self.assertEqual((qw.dtype, qs.shape), (jnp.dtype(qtype), expected_shape))
        np.testing.assert_array_equal(np.asarray(qw), np.asarray(ref_qw))
        np.testing.assert_array_equal(np.asarray(qs), np.asarray(jnp.squeeze(ref_qs_4d, axis=2)))
        np.testing.assert_array_equal(
            np.asarray(quantizations.prepare_fused_gmm_scale(qs, qw.shape)), np.asarray(ref_qs_4d)
        )

    for smode, expected_s_shape in [("per_channel", (4, 1, 256)), ("subchannel128", (4, 2, 256)), ("block128", (4, 2, 2))]:
      model = _make_mock_decoder(weight_dtype=jnp.bfloat16, prefuse_moe=True)
      routed = model.decoder.layers_0.mlp.routed_experts
      w_np = np.asarray(routed.wi[...], dtype=np.float32)
      audit_mod.quantize_moe_fp8(model, scale_mode=smode)
      qw, qs = routed.wi[...], routed.wi_scale[...]
      w_deq = np.asarray(quantizations.dequantize_weight(qw, qs, jnp.float32))
      cos = float(np.sum(w_np * w_deq) / (np.linalg.norm(w_np) * np.linalg.norm(w_deq)))
      self.assertGreater(cos, 0.999)
      self.assertEqual(quantizations.prepare_fused_gmm_scale(qs, qw.shape).shape, (4, expected_s_shape[1], 1, 256))

  def test_quantize_moe_fp8_rejects_invalid_scale_mode(self):
    with self.assertRaises(ValueError):
      audit_mod.quantize_moe_fp8(_make_mock_decoder(), scale_mode="unknown_mode")


class InPlaceMoeQuantizationAndModelAuditTest(unittest.TestCase):

  def _assert_audit(self, model, role: str, mode: str, exp_moe: str, exp_dense: str | None = None, out_dir: str = ""):
    with redirect_stdout(io.StringIO()):
      rep = audit_mod.audit_model(model, role=role, expected_mode=mode, out_dir=out_dir)
    self.assertIn(exp_moe, rep["execution_paths"]["routed_moe"])
    if exp_dense is not None:
      self.assertIn(exp_dense, rep["execution_paths"]["dense_linear"])
    return rep

  def test_audit_bf16_and_in_place_moe_modes(self):
    with tempfile.TemporaryDirectory() as tmp:
      sa_bf16 = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      self._assert_audit(sa_bf16, "sampler", "bf16", "FUSED_MOE_BF16", "DENSE_PURE_BF16", out_dir=tmp)

      tr_bf16 = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      self._assert_audit(tr_bf16, "trainer", "bf16", "GMM_V2_PURE_BF16", "DENSE_PURE_BF16", out_dir=tmp)
      self.assertTrue(os.path.isfile(os.path.join(tmp, "audit_sampler.json")))
      self.assertTrue(os.path.isfile(os.path.join(tmp, "audit_trainer.json")))

      sa_fp8 = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      self.assertEqual(audit_mod.quantize_moe_fp8(sa_fp8, scale_mode="per_channel"), 1)
      self._assert_audit(sa_fp8, "sampler", "fp8_moe", "FUSED_MOE_NATIVE_FP8", "DENSE_PURE_BF16", out_dir=tmp)
      self._assert_audit(sa_fp8, "sampler", "fp8moe", "FUSED_MOE_NATIVE_FP8", out_dir=tmp)

      tr_deq = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      audit_mod.quantize_moe_fp8(tr_deq, scale_mode="per_channel", set_serve_quant=False)
      self._assert_audit(tr_deq, "trainer", "fp8_moe", "GMM_V2_DEQUANT_TO_BF16", out_dir=tmp)

      tr_nat = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=False)
      audit_mod.quantize_moe_fp8(tr_nat, scale_mode="per_channel", set_serve_quant=True)
      self._assert_audit(tr_nat, "trainer", "fp8_moe_native", "GMM_V2_NATIVE_FP8", out_dir=tmp)

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
      self._assert_audit(ckpt_tr, "trainer", "fp8_ckpt", "GMM_V2_DEQUANT_TO_BF16", "DENSE_DEQUANT_TO_BF16", out_dir=tmp)
      self._assert_audit(ckpt_tr, "trainer", "fp8", "GMM_V2_DEQUANT_TO_BF16", out_dir=tmp)

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
      self._assert_audit(serve_tr, "trainer", "fp8_serve", "GMM_V2_NATIVE_FP8", "DENSE_NATIVE_FP8", out_dir=tmp)

      int8_tr = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=True)
      audit_mod.quantize_moe_fp8(int8_tr, scale_mode="per_channel", weight_qtype=jnp.int8)
      rep_int8 = self._assert_audit(int8_tr, "trainer", "int8_moe", "GMM_V2_DEQUANT_TO_BF16", out_dir=tmp)
      self.assertEqual(rep_int8["modules"]["layers_0.mlp.routed_experts.wi"]["weight"]["dtype"], "int8")

  def test_audit_model_fails_fast_on_mode_mismatch(self):
    with self.assertRaises(AssertionError):
      audit_mod.audit_model(_make_mock_decoder(), role="trainer", expected_mode="fp8_moe", out_dir="")
    with self.assertRaises(AssertionError):
      audit_mod.audit_model(
          _make_mock_decoder(weight_dtype=jnp.float8_e4m3fn, moe_scale_shape=None),
          role="trainer",
          expected_mode="fp8_ckpt",
          out_dir="",
      )
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
    p_len, g_len, pad_len, num_l, k_exp = 16, 8, 8, 4, 2
    captured = rng.integers(0, 64, size=(3, p_len + g_len - 1, num_l, k_exp), dtype=np.int16)
    padded = [
        np.concatenate([c, np.full((1, num_l, k_exp), rl_common.UNSET_ROUTED_EXPERT, dtype=np.int16)], axis=0)
        for c in captured
    ]
    aligned = rl_common.align_routed_experts(
        padded, completion_lengths=[g_len] * 3, prompt_width=p_len, completion_width=g_len + pad_len
    )
    self.assertEqual(aligned.shape, (3, p_len + g_len + pad_len, num_l, k_exp))
    np.testing.assert_array_equal(aligned[:, : p_len + g_len - 1], captured)
    self.assertTrue(np.all(aligned[:, p_len + g_len - 1 :] == rl_common.UNSET_ROUTED_EXPERT))

  def test_selective_log_softmax_shift_and_temperature_scaling(self):
    rng = np.random.default_rng(19)
    b_sz, p_len, g_len, vocab = 2, 12, 6, 32
    seq_len = p_len + g_len
    logits = jnp.asarray(rng.standard_normal((b_sz, seq_len, vocab)), dtype=jnp.float32)
    tokens = rng.integers(0, vocab, size=(b_sz, seq_len), dtype=np.int32)
    nxt = jnp.asarray(np.roll(tokens, -1, axis=1))
    pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (b_sz, seq_len))

    gen_temp = 0.7
    temp_scale = jnp.where(pos >= (p_len - 1), jnp.float32(gen_temp), jnp.float32(1.0))[..., None]
    tr_lp = np.asarray(rl_common.selective_log_softmax(logits / temp_scale, nxt))

    ref_prompt_lse = jax.nn.log_softmax(logits, axis=-1)
    ref_decode_lse = jax.nn.log_softmax(logits / gen_temp, axis=-1)
    sa_prompt_lp = np.zeros((b_sz, p_len), dtype=np.float32)
    for t in range(1, p_len):
      sa_prompt_lp[:, t] = np.asarray(ref_prompt_lse[np.arange(b_sz), t - 1, tokens[:, t]])
    sa_gen_lp = np.zeros((b_sz, g_len), dtype=np.float32)
    for j in range(g_len):
      sa_gen_lp[:, j] = np.asarray(ref_decode_lse[np.arange(b_sz), p_len - 1 + j, tokens[:, p_len + j]])

    np.testing.assert_allclose(tr_lp[:, : p_len - 1], sa_prompt_lp[:, 1:p_len], atol=1e-6)
    np.testing.assert_allclose(tr_lp[:, p_len - 1 : p_len + g_len - 1], sa_gen_lp, atol=1e-6)


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

    masked = algo_core.sequence_loss_mask(
        jnp.isfinite(raw_jnp).astype(jnp.float32), log_is_raw=raw_jnp, mult_prob_error_threshold=2.0
    )
    np.testing.assert_array_equal(np.asarray(masked.sample_mask), np.array([1.0, 1.0, 1.0]))

  def test_stage_compare_end_to_end_output_format(self):
    rng = np.random.default_rng(42)
    b_sz, p_len, g_len = 4, 16, 16
    tokens = rng.integers(10, 100, size=(b_sz, p_len), dtype=np.int32)
    full_tokens = rng.integers(10, 100, size=(b_sz, p_len + g_len), dtype=np.int32)
    full_tokens[:, :p_len] = tokens
    sa_gen_lp = rng.uniform(-1.5, -0.05, size=(b_sz, g_len)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
      for role, moe_path in [
          ("sampler", "FUSED_MOE_BF16 (tpu_inference)"),
          ("trainer", "GMM_V2_PURE_BF16 (BF16 gmm_v2)"),
      ]:
        with open(os.path.join(tmp, f"audit_{role}.json"), "w", encoding="utf-8") as f:
          json.dump(
              {
                  "expected_mode": "bf16",
                  "execution_paths": {"routed_moe": moe_path, "dense_linear": "DENSE_PURE_BF16 (BF16 dot_general)"},
              },
              f,
          )

      np.savez(
          os.path.join(tmp, "sampler_logprobs.npz"),
          tokens=tokens,
          gen_ids=full_tokens[:, p_len:],
          gen_logp=sa_gen_lp,
          gen_lens=np.full(b_sz, g_len, dtype=np.int32),
      )
      np.savez(
          os.path.join(tmp, "trainer_logprobs.npz"),
          tokens=tokens,
          full_tokens=full_tokens,
          gen_logp=sa_gen_lp - 0.0001,
      )

      buf = io.StringIO()
      with redirect_stdout(buf):
        res = audit_mod.main(["--sampler-mode", "bf16", "--trainer-mode", "bf16", "--out-dir", tmp, "--stage", "compare"])
      out = buf.getvalue()

      self.assertIn("[SAMPLER AUDIT] mode=bf16 | MoE=FUSED_MOE_BF16 (tpu_inference)", out)
      self.assertIn("[TRAINER AUDIT] mode=bf16 | MoE=GMM_V2_PURE_BF16 (BF16 gmm_v2)", out)
      self.assertIn("### Sampler=BF16 vs Trainer=BF16 — OUTPUT tokens (decode)", out)
      self.assertIn(">>> is_oob_ratio (seq-mask-tis)", out)
      self.assertIn("Top-5 Worst Token Divergences:", out)
      self.assertEqual(res["decode"]["is_oob_ratio"], 0.0)

      buf2 = io.StringIO()
      with redirect_stdout(buf2):
        audit_mod.main(["--sampler-mode", "fp8moe", "--trainer-mode", "fp8", "--out-dir", tmp, "--stage", "compare"])
      self.assertIn("### Sampler=FP8MOE vs Trainer=FP8 — OUTPUT tokens (decode)", buf2.getvalue())

  def test_build_algo_and_batch_variable_length_routed_experts_and_safe_pad_id(self):
    rng = np.random.default_rng(7)
    b_sz, p_len, max_g, num_l, k_exp = 3, 12, 10, 4, 2
    comp_lens = np.array([6, 10, 4], dtype=np.int32)
    prompts = rng.integers(10, 100, size=(b_sz, p_len), dtype=np.int32)
    comp_ids = rng.integers(10, 100, size=(b_sz, max_g), dtype=np.int32)
    comp_ids[0, 2] = 248044
    logps = np.full((b_sz, max_g), np.nan, dtype=np.float32)
    routed = np.full((b_sz, p_len + max_g, num_l, k_exp), rl_common.UNSET_ROUTED_EXPERT, dtype=np.int16)
    for i, c_len in enumerate(comp_lens):
      logps[i, :c_len] = rng.uniform(-1.5, -0.1, size=(c_len,)).astype(np.float32)
      routed[i, : p_len + c_len - 1] = rng.integers(0, 32, size=(p_len + c_len - 1, num_l, k_exp), dtype=np.int16)

    eff_pad_id = audit_mod._safe_pad_id(248044, prompts, comp_ids)
    self.assertNotEqual(eff_pad_id, 248044)

    _, batch = audit_mod.build_algo_and_batch(
        prompts, comp_ids, logps, completion_lens=comp_lens, routed_experts=routed, pad_id=eff_pad_id
    )
    _, batch2 = audit_mod.build_algo_and_batch(
        batch.prompt_ids[:, :p_len],
        batch.completion_ids[:, :max_g],
        np.where(batch.completion_mask[:, :max_g] > 0, batch.rollout_per_token_logps[:, :max_g], np.nan),
        completion_lens=comp_lens,
        routed_experts=batch.routed_experts[:, : p_len + max_g],
        pad_id=eff_pad_id,
    )
    for i, c_len in enumerate(comp_lens):
      self.assertEqual(int(batch2.completion_mask[i].sum()), int(c_len))
      np.testing.assert_array_equal(batch2.routed_experts[i, : p_len + c_len - 1], routed[i, : p_len + c_len - 1])
      self.assertEqual(int(batch2.routed_experts[i, p_len + c_len - 1, 0, 0]), rl_common.UNSET_ROUTED_EXPERT)

  def _make_variable_len_rollout(self, seed: int, b_sz: int, p_len: int, g_len: int, comp_lens: list[int]):
    rng = np.random.default_rng(seed)
    c_lens = np.array(comp_lens, dtype=np.int32)
    prompts = rng.integers(10, 100, size=(b_sz, p_len), dtype=np.int32)
    comp_ids = rng.integers(10, 100, size=(b_sz, g_len), dtype=np.int32)
    sa_gen_lp = np.full((b_sz, g_len), np.nan, dtype=np.float32)
    for i, c_len in enumerate(c_lens):
      sa_gen_lp[i, :c_len] = rng.uniform(-1.5, -0.1, size=(c_len,)).astype(np.float32)
    return prompts, comp_ids, sa_gen_lp, c_lens

  def _assert_unpacked_matches(self, packed_batch, sa_gen_lp: np.ndarray, comp_lens: np.ndarray):
    b_sz, g_len = sa_gen_lp.shape
    self.assertEqual(int(packed_batch.completion_mask.sum()), int(comp_lens.sum()))
    unpacked = audit_mod._unpack_trainer_logps(packed_batch.rollout_per_token_logps, packed_batch, comp_lens, b_sz, g_len)
    for i, c_len in enumerate(comp_lens):
      np.testing.assert_allclose(unpacked[i, :c_len], sa_gen_lp[i, :c_len], atol=1e-6)
      self.assertTrue(np.all(np.isnan(unpacked[i, c_len:])))

  def test_sequence_packing_batch_and_compare(self):
    prompts, comp_ids, sa_gen_lp, comp_lens = self._make_variable_len_rollout(11, 4, 16, 12, [12, 8, 10, 6])
    _, packed_batch = audit_mod.build_algo_and_batch(
        prompts, comp_ids, sa_gen_lp, completion_lens=comp_lens, max_seq_token_per_tpu=1024
    )
    self.assertIsNotNone(packed_batch.segment_ids)
    self.assertIsNotNone(packed_batch.segment_positions)
    self._assert_unpacked_matches(packed_batch, sa_gen_lp, comp_lens)

    with tempfile.TemporaryDirectory() as tmp:
      for fname, lp in [("sampler_logprobs.npz", sa_gen_lp), ("trainer_logprobs.npz", sa_gen_lp + 0.0002)]:
        np.savez(os.path.join(tmp, fname), tokens=prompts, gen_ids=comp_ids, gen_logp=lp, gen_lens=comp_lens)
      with redirect_stdout(io.StringIO()):
        res_packed = audit_mod.main([
            "--sampler-mode", "bf16", "--trainer-mode", "bf16", "--out-dir", tmp,
            "--stage", "compare", "--pack-sequences", "--max-seq-token-per-tpu", "1024",
        ])
      self.assertEqual(res_packed["decode"]["is_oob_ratio"], 0.0)
      self.assertEqual(res_packed["decode"]["sample_mask/kept_frac"], 1.0)

  def test_multi_chunk_sequence_packing_preserves_all_trajectories(self):
    prompts, comp_ids, sa_gen_lp, comp_lens = self._make_variable_len_rollout(23, 4, 200, 100, [100, 90, 80, 70])
    _, packed_batch = audit_mod.build_algo_and_batch(
        prompts, comp_ids, sa_gen_lp, completion_lens=comp_lens, max_seq_token_per_tpu=512, pack_size=1
    )
    self.assertGreater(packed_batch.segment_ids.shape[0], 1)
    self.assertEqual(len(packed_batch.metadata["trajectory_ids"]), 4)
    self._assert_unpacked_matches(packed_batch, sa_gen_lp, comp_lens)

  def test_scanned_decoder_quantize_and_audit(self):
    base_model = _make_mock_decoder(attention="flash", weight_dtype=jnp.bfloat16, prefuse_moe=False)
    routed = base_model.decoder.layers_0.mlp.routed_experts
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
        layers=types.SimpleNamespace(local_layers=[base_model.decoder.layers_0], global_layer=base_model.decoder.layers_3),
    )
    scanned_model = types.SimpleNamespace(decoder=scanned_decoder)
    self.assertEqual(audit_mod.quantize_moe_fp8(scanned_model, scale_mode="per_channel", set_serve_quant=True), 1)
    self.assertEqual(routed.wi_0[...].shape, (2, 4, 256, 256))
    self.assertEqual(routed.wi_0_scale[...].shape, (2, 4, 1, 256))
    rep = audit_mod.audit_model(scanned_model, role="trainer", expected_mode="fp8_moe_native", out_dir="")
    self.assertIn("GMM_V2_NATIVE_FP8", rep["execution_paths"]["routed_moe"])

  def test_stage_trainer_no_optimizer_recursion_and_packed_flags(self):
    prompts, gen_ids, sa_gen_lp, comp_lens = self._make_variable_len_rollout(77, 4, 16, 16, [16, 16, 16, 16])
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
      self.assertEqual((trainer_args.max_response_length, trainer_args.max_seq_token_per_tpu), (512 - 16, 2048))
      mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
      return _FakeMaxTextEngine, mesh

    class _FakeTrainerWorker:

      def __init__(self, trainer_factory, worker_id, logps_chunk_size, logps_micro_batch_size, execution_context):
        del worker_id, logps_chunk_size, execution_context
        self._trainer = trainer_factory()
        self.logps_micro_batch_size = logps_micro_batch_size

      def per_token_logps(self, req):
        _ = req.micro_batch_size
        return types.SimpleNamespace(per_token_logps=np.zeros_like(req.completion_tokens, dtype=np.float32) - 0.25)

    fake_maxtext_engine_mod = types.SimpleNamespace(MaxTextTrainingEngine=_FakeMaxTextEngine, nnx=nnx)
    with tempfile.TemporaryDirectory() as tmp:
      np.savez(os.path.join(tmp, "tokens.npz"), tokens=prompts)
      np.savez(os.path.join(tmp, "sampler_logprobs.npz"), tokens=prompts, gen_ids=gen_ids, gen_logp=sa_gen_lp, gen_lens=comp_lens)
      with (
          mock.patch.object(audit_mod, "get_tokenizer", return_value=(types.SimpleNamespace(eos_token_id=2), "dummy-tok")),
          mock.patch.object(tunix_maxtext_utils, "get_tokenizer_pad_id", return_value=0),
          mock.patch.object(tunix_maxtext_utils, "maxtext_modules", return_value=(None, fake_maxtext_engine_mod, None)),
          mock.patch(
              "tunix.experimental.examples.common.run_trainer_node._create_maxtext_trainer_factory",
              side_effect=_fake_create_trainer_factory,
          ),
          mock.patch("tunix.experimental.worker.trainer_worker.TrainerWorker", side_effect=_FakeTrainerWorker),
      ):
        audit_mod.main([
            "--stage", "trainer", "--trainer-mode", "fp8_moe_native", "--out-dir", tmp,
            "--pack-sequences", "--max-seq-token-per-tpu", "2048",
        ])
      extra = os.environ.get("MAXTEXT_EXTRA_FLAGS", "")
      for flag in ("logits_dot_in_fp32=True", "use_tokamax_splash=True", "sa_use_base2_exp=False", "megablox=True", "wi_tile_fwd_batch_seq=256"):
        self.assertIn(flag, extra)
      with np.load(os.path.join(tmp, "trainer_logprobs.npz")) as saved:
        self.assertEqual(saved["gen_logp"].shape, (4, 16))
        np.testing.assert_allclose(saved["gen_logp"], -0.25, atol=1e-6)

  def test_stage_sampler_in_place_moe_and_cleanup_and_stage_all_subprocesses(self):
    b_sz, p_len, g_len, num_l, k_exp = 2, 8, 4, 4, 2
    tokens = np.arange(b_sz * p_len, dtype=np.int32).reshape(b_sz, p_len) + 10
    raw_decoder = _make_mock_decoder(attention="vllm_rpa", weight_dtype=jnp.bfloat16, prefuse_moe=True)

    class _MockNNXModel(nnx.Module):

      def __init__(self):
        self.decoder = raw_decoder.decoder
        self.routed0 = raw_decoder.decoder.layers_0.mlp.routed_experts

    mock_inner_model = _MockNNXModel()
    cleanup_calls, sampler_call_kwargs, vllm_configs = [], [], []

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
            model=adapter, state=state, state_leaves=jax.tree.leaves(state), uses_mrope=True
        )

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
            routed_experts=[
                np.ones((len(prompt_token_ids[i]) + max_generation_steps - 1, num_l, k_exp), dtype=np.int16)
                for i in range(n)
            ],
        )

    fake_adapter_mod = types.SimpleNamespace(MaxTextForCausalLM=_FakeAdapter, register=lambda: None)
    fake_vllm_sampler_mod = types.SimpleNamespace(
        VllmConfig=lambda **kw: types.SimpleNamespace(**kw), VllmSampler=_FakeVllmSampler
    )
    with tempfile.TemporaryDirectory() as tmp:
      np.savez(os.path.join(tmp, "tokens.npz"), tokens=tokens)
      with (
          mock.patch.dict(
              sys.modules,
              {
                  "maxtext_vllm_adapter": fake_adapter_mod,
                  "tunix.generate": types.SimpleNamespace(vllm_sampler=fake_vllm_sampler_mod),
                  "tunix.generate.vllm_sampler": fake_vllm_sampler_mod,
              },
          ),
          mock.patch.object(audit_mod, "get_tokenizer", return_value=(types.SimpleNamespace(eos_token_id=2), "dummy-tok")),
          mock.patch.object(tunix_maxtext_utils, "get_tokenizer_pad_id", return_value=0),
      ):
        audit_mod.main(["--stage", "sampler", "--sampler-mode", "fp8moe", "--gen-tokens", str(g_len), "--out-dir", tmp])
      self.assertEqual(cleanup_calls, ["delete_cache", "stop"])
      self.assertEqual(vllm_configs[-1].eos_tokens, [])
      self.assertTrue(sampler_call_kwargs[-1]["ignore_eos"])
      self.assertEqual(sampler_call_kwargs[-1]["stop_token_ids"], [])
      self.assertIsNone(sampler_call_kwargs[-1]["_eos_token_id"])
      self.assertEqual(sampler_call_kwargs[-1]["min_tokens"], g_len)
      with np.load(os.path.join(tmp, "sampler_logprobs.npz")) as sa:
        self.assertEqual(sa["gen_ids"].shape, (b_sz, g_len))
      self.assertTrue(os.path.isfile(os.path.join(tmp, "router_indices.npz")))

      subproc_cmds = []
      with (
          mock.patch.object(audit_mod, "stage_tokenize", return_value=tokens),
          mock.patch.object(audit_mod.subprocess, "run", side_effect=lambda cmd, check: subproc_cmds.append(cmd)),
          mock.patch.object(audit_mod, "stage_compare", return_value={"is_oob_ratio": 0.0}),
      ):
        audit_mod.main(["--stage", "all", "--out-dir", tmp])
      self.assertEqual([c[-2:] for c in subproc_cmds], [["--stage", "sampler"], ["--stage", "trainer"]])


class ModuleDivergenceProbeTest(unittest.TestCase):

  def _capture_chunked_sampler_probes(
      self, sampler_model, seq0_tokens: np.ndarray, seq1_tokens: np.ndarray, probe_pos: np.ndarray, p_len: int, g_len: int
  ) -> dict[str, np.ndarray]:
    """Runs 2-chunk prefill + G decode steps on `sampler_model` under `ModuleProbeTap` and returns probes."""
    prompt_tokens_probe = seq0_tokens[probe_pos]
    target_next_tokens = seq0_tokens[np.minimum(probe_pos + 1, p_len + g_len - 1)]
    with mdp.ModuleProbeTap(sampler_model, role="sampler", probe_positions=probe_pos, prompt_len=p_len, gen_len=g_len) as sa_tap:
      chunk1 = jnp.asarray(np.concatenate([seq0_tokens[:3], seq1_tokens[:2]]), dtype=jnp.int32)
      sampler_model.forward_sampler_step(
          chunk1, query_start_loc=jnp.array([0, 3, 5], dtype=jnp.int32), seq_lens=jnp.array([3, 2], dtype=jnp.int32)
      )
      chunk2 = jnp.asarray(np.concatenate([seq0_tokens[3:p_len], seq1_tokens[2:p_len]]), dtype=jnp.int32)
      sampler_model.forward_sampler_step(
          chunk2, query_start_loc=jnp.array([0, 3, 7], dtype=jnp.int32), seq_lens=jnp.array([p_len, p_len], dtype=jnp.int32)
      )
      for step in range(g_len):
        cur_len = p_len + step + 1
        dec_toks = jnp.array([seq0_tokens[p_len + step], seq1_tokens[p_len + step]], dtype=jnp.int32)
        sampler_model.forward_sampler_step(
            dec_toks, query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32), seq_lens=jnp.array([cur_len, cur_len], dtype=jnp.int32)
        )
    return sa_tap.get_probes(target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe, temperature=1.0)

  def _capture_trainer_probes(
      self,
      trainer_model,
      batches: list[jnp.ndarray],
      probe_pos: np.ndarray,
      p_len: int,
      g_len: int,
      target_next_tokens: np.ndarray,
      prompt_tokens_probe: np.ndarray,
      **tap_kwargs,
  ) -> dict[str, np.ndarray]:
    with mdp.ModuleProbeTap(
        trainer_model, role="trainer", probe_positions=probe_pos, prompt_len=p_len, gen_len=g_len, **tap_kwargs
    ) as tr_tap:
      for b in batches:
        trainer_model.forward_trainer(b)
    return tr_tap.get_probes(target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe, temperature=1.0)

  def _assert_zero_divergence(self, report: dict, check_isolated: bool = True, check_router: bool = False):
    for rec in report["records"]:
      self.assertLess(rec["cumulative"]["rel_l2"], 1e-6, msg=f"Non-zero cumulative rel_l2 on {rec['key']}")
      if check_isolated and rec["isolated"] is not None:
        self.assertLess(rec["isolated"]["rel_l2"], 1e-6, msg=f"Non-zero isolated rel_l2 on {rec['key']}")
      if check_router and rec["router"] is not None:
        self.assertEqual(rec["router"]["cum"]["top1_agree"], 1.0)
        self.assertEqual(rec["router"]["cum"]["topk_jaccard"], 1.0)

  def test_select_probe_positions_and_parse_layers(self):
    pos_all = mdp.select_probe_positions(prompt_len=8, gen_len=6, max_tokens=64)
    np.testing.assert_array_equal(pos_all, np.arange(14, dtype=np.int32))

    pos_sub = mdp.select_probe_positions(prompt_len=100, gen_len=100, max_tokens=16)
    self.assertLessEqual(len(pos_sub), 16)
    for boundary in (0, 99, 100, 199):
      self.assertIn(boundary, pos_sub.tolist())

    self.assertIsNone(mdp.parse_probe_layers("all"))
    self.assertIsNone(mdp.parse_probe_layers(None))
    self.assertEqual(mdp.parse_probe_layers("0, 2-4, 7"), {0, 2, 3, 4, 7})
    self.assertEqual(mdp.parse_probe_layers([1, 3]), {1, 3})

  def test_bf16_sampler_vs_unscanned_packed_and_scanned_trainer_zero_divergence(self):
    rng = np.random.default_rng(2026)
    p_len, g_len = 6, 4
    seq0_tokens = rng.integers(1, 60, size=(p_len + g_len,), dtype=np.int32)
    seq1_tokens = rng.integers(1, 60, size=(p_len + g_len,), dtype=np.int32)
    probe_pos = mdp.select_probe_positions(p_len, g_len, max_tokens=64)
    prompt_tokens_probe = seq0_tokens[probe_pos]
    target_next_tokens = seq0_tokens[np.minimum(probe_pos + 1, p_len + g_len - 1)]

    sampler_model = _build_hybrid_qwen3_5_toy_model(seed=99, attention="vllm_rpa", prefuse_moe=True, scanned=False)
    sa_probes = self._capture_chunked_sampler_probes(sampler_model, seq0_tokens, seq1_tokens, probe_pos, p_len, g_len)

    # 1. Compare against Unscanned 2D Trainer
    trainer_2d = _build_hybrid_qwen3_5_toy_model(seed=99, attention="flash", prefuse_moe=False, scanned=False)
    batch_2d = jnp.asarray(np.stack([seq0_tokens, seq1_tokens], axis=0), dtype=jnp.int32)
    tr_2d_probes = self._capture_trainer_probes(
        trainer_2d, [batch_2d], probe_pos, p_len, g_len, target_next_tokens, prompt_tokens_probe
    )
    iso_2d_probes = mdp.run_isolated_trainer_replay(
        trainer_2d, sa_probes, target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe
    )
    with redirect_stdout(io.StringIO()):
      rep_2d = mdp.compare_module_probes(sa_probes, tr_2d_probes, iso_2d_probes, tag="BF16 vs BF16 (2D)")
    self._assert_zero_divergence(rep_2d, check_isolated=True, check_router=True)

    # 2. Compare against 1D Sequence-Packed Trainer (seq 0 in microbatch 1, row 1, offset 14)
    packed_width = 28
    mb0_pack = jnp.ones((2, packed_width), dtype=jnp.int32)
    mb1_row0 = np.ones((packed_width,), dtype=jnp.int32)
    mb1_row1_np = np.zeros((packed_width,), dtype=np.int32)
    mb1_row1_np[: p_len + g_len] = seq1_tokens
    seq0_packed_cols = np.arange(14, 14 + p_len + g_len, dtype=np.int32)
    mb1_row1_np[seq0_packed_cols] = seq0_tokens
    mb1_pack = jnp.asarray(np.stack([mb1_row0, mb1_row1_np], axis=0), dtype=jnp.int32)

    tr_pack_probes = self._capture_trainer_probes(
        trainer_2d,
        [mb0_pack, mb1_pack],
        probe_pos,
        p_len,
        g_len,
        target_next_tokens,
        prompt_tokens_probe,
        packed_seq0_row=1,
        packed_seq0_positions=seq0_packed_cols,
        trainer_microbatch_idx=1,
    )
    with redirect_stdout(io.StringIO()):
      rep_pack = mdp.compare_module_probes(sa_probes, tr_pack_probes, iso_2d_probes, tag="BF16 vs BF16 (1D Packed)")
    self._assert_zero_divergence(rep_pack, check_isolated=False)

    # 3. Compare against Scanned Trainer (`Qwen3_5ScannableBlock`)
    trainer_scanned = _build_hybrid_qwen3_5_toy_model(seed=99, attention="flash", prefuse_moe=False, scanned=True)
    tr_scan_probes = self._capture_trainer_probes(
        trainer_scanned, [batch_2d], probe_pos, p_len, g_len, target_next_tokens, prompt_tokens_probe
    )
    iso_scan_probes = mdp.run_isolated_trainer_replay(
        trainer_scanned, sa_probes, target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe
    )
    with redirect_stdout(io.StringIO()):
      rep_scan = mdp.compare_module_probes(sa_probes, tr_scan_probes, iso_scan_probes, tag="BF16 vs BF16 (Scanned)")
    self._assert_zero_divergence(rep_scan, check_isolated=True)

  def test_fp8_moe_sampler_vs_bf16_trainer_pinpoints_routed_experts_only(self):
    rng = np.random.default_rng(404)
    p_len, g_len = 6, 4
    seq0_tokens = rng.integers(1, 60, size=(p_len + g_len,), dtype=np.int32)
    probe_pos = mdp.select_probe_positions(p_len, g_len, max_tokens=64)
    prompt_tokens_probe = seq0_tokens[probe_pos]
    target_next_tokens = seq0_tokens[np.minimum(probe_pos + 1, p_len + g_len - 1)]

    sampler_fp8moe = _build_hybrid_qwen3_5_toy_model(seed=321, attention="vllm_rpa", prefuse_moe=True, scanned=False)
    trainer_bf16 = _build_hybrid_qwen3_5_toy_model(seed=321, attention="flash", prefuse_moe=False, scanned=False)

    self.assertEqual(audit_mod.quantize_moe_fp8(sampler_fp8moe, scale_mode="per_channel"), 4)

    with mdp.ModuleProbeTap(
        sampler_fp8moe, role="sampler", probe_positions=probe_pos, prompt_len=p_len, gen_len=g_len
    ) as sa_tap:
      sampler_fp8moe.forward_sampler_step(
          jnp.asarray(seq0_tokens, dtype=jnp.int32),
          query_start_loc=jnp.array([0, p_len + g_len], dtype=jnp.int32),
          seq_lens=jnp.array([p_len + g_len], dtype=jnp.int32),
      )
    sa_probes = sa_tap.get_probes(
        target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe, temperature=1.0
    )

    tr_probes = self._capture_trainer_probes(
        trainer_bf16,
        [jnp.asarray(seq0_tokens[None, :], dtype=jnp.int32)],
        probe_pos,
        p_len,
        g_len,
        target_next_tokens,
        prompt_tokens_probe,
    )
    iso_probes = mdp.run_isolated_trainer_replay(
        trainer_bf16, sa_probes, target_next_tokens=target_next_tokens, prompt_tokens_probe=prompt_tokens_probe
    )

    with tempfile.TemporaryDirectory() as tmp:
      buf = io.StringIO()
      with redirect_stdout(buf):
        report = mdp.compare_module_probes(
            sa_probes, tr_probes, iso_probes, tag="Sampler=FP8MOE vs Trainer=BF16", out_dir=tmp
        )
      self.assertIn("MODULE-BY-MODULE VALUE DIVERGENCE", buf.getvalue())
      self.assertTrue(os.path.isfile(os.path.join(tmp, "module_divergence_report.json")))

    by_key = {r["key"]: r for r in report["records"]}
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
      self.assertLess(by_key[clean_l0]["cumulative"]["rel_l2"], 1e-6, msg=f"Expected 0 cum_rel_l2 at {clean_l0}")

    self.assertGreater(by_key["layer_0.mlp.routed_experts"]["isolated"]["rel_l2"], 1e-3)
    self.assertGreater(by_key["layer_0.mlp.routed_experts"]["cumulative"]["rel_l2"], 1e-3)

    non_moe_subs = (
        "input_layernorm",
        "post_attention_layernorm",
        "mlp.gate_logits",
        "mlp.shared_expert",
        "mlp.shared_expert_gate",
    )
    for lyr in (1, 2, 3):
      for sub in non_moe_subs:
        key = f"layer_{lyr}.{sub}"
        self.assertGreater(by_key[key]["cumulative"]["rel_l2"], 1e-4, msg=f"Expected >0 cum_rel_l2 at {key}")
        self.assertLess(by_key[key]["isolated"]["rel_l2"], 1e-6, msg=f"Expected 0 iso_rel_l2 at {key}")
      self.assertGreater(by_key[f"layer_{lyr}.mlp.routed_experts"]["isolated"]["rel_l2"], 1e-3)

    top_iso_mods = {b["module"] for b in report["top_isolated_bottlenecks"][:4]}
    self.assertIn("mlp.routed_experts", top_iso_mods)
    self.assertGreater(
        report["family_summary"]["mlp.routed_experts"]["iso_rel_l2_mean"],
        1000.0 * (report["family_summary"]["mlp.shared_expert"]["iso_rel_l2_mean"] + 1e-12),
    )

  def test_mlperf_v5p_and_probe_layers_cli_integration(self):
    trainer_model = _build_hybrid_qwen3_5_toy_model(seed=111, attention="flash", prefuse_moe=False, scanned=False)
    sampler_model = _build_hybrid_qwen3_5_toy_model(seed=111, attention="vllm_rpa", prefuse_moe=True, scanned=False)

    with tempfile.TemporaryDirectory() as tmp:
      b_sz, p_len, g_len = 2, 6, 4
      tokens = np.arange(1, b_sz * p_len + 1, dtype=np.int32).reshape(b_sz, p_len)
      gen_ids = np.arange(1, b_sz * g_len + 1, dtype=np.int32).reshape(b_sz, g_len)
      gen_lp = np.full((b_sz, g_len), -0.3, dtype=np.float32)
      for fname in ("sampler_logprobs.npz", "trainer_logprobs.npz"):
        np.savez(
            os.path.join(tmp, fname),
            tokens=tokens,
            gen_ids=gen_ids,
            gen_logp=gen_lp,
            gen_lens=np.full(b_sz, g_len, dtype=np.int32),
        )

      probe_pos = mdp.select_probe_positions(p_len, g_len, max_tokens=16)
      seq0_full = np.concatenate([tokens[0], gen_ids[0]])
      with mdp.ModuleProbeTap(
          sampler_model, role="sampler", probe_positions=probe_pos, prompt_len=p_len, gen_len=g_len
      ) as sa_tap:
        sampler_model.forward_sampler_step(
            jnp.asarray(seq0_full, dtype=jnp.int32),
            query_start_loc=jnp.array([0, p_len + g_len], dtype=jnp.int32),
            seq_lens=jnp.array([p_len + g_len], dtype=jnp.int32),
        )
      sa_probes = sa_tap.get_probes(target_next_tokens=seq0_full, prompt_tokens_probe=seq0_full)
      np.savez(os.path.join(tmp, "sampler_module_probes.npz"), **sa_probes)

      tr_probes = self._capture_trainer_probes(
          trainer_model,
          [jnp.asarray(seq0_full[None, :], dtype=jnp.int32)],
          probe_pos,
          p_len,
          g_len,
          seq0_full,
          seq0_full,
      )
      np.savez(os.path.join(tmp, "trainer_module_probes.npz"), **tr_probes)

      iso_probes = mdp.run_isolated_trainer_replay(
          trainer_model, sa_probes, target_next_tokens=seq0_full, prompt_tokens_probe=seq0_full
      )
      np.savez(os.path.join(tmp, "trainer_isolated_probes.npz"), **iso_probes)

      with redirect_stdout(io.StringIO()):
        res = audit_mod.main(
            ["--stage", "compare", "--mlperf-v5p", "--probe-modules", "--out-dir", tmp, "--max-seq-token-per-tpu", "512"]
        )
      self.assertIn("module_divergence", res)
      self.assertEqual(res["module_divergence"]["logprob_divergence"]["top1_agree"], 1.0)
      self._assert_zero_divergence(res["module_divergence"], check_isolated=True)

      with redirect_stdout(io.StringIO()):
        res_sub = audit_mod.main([
            "--stage",
            "compare",
            "--mlperf-v5p",
            "--probe-modules",
            "--probe-layers",
            "0,2",
            "--out-dir",
            tmp,
            "--max-seq-token-per-tpu",
            "512",
        ])
      self.assertEqual({m["layer"] for m in res_sub["module_divergence"]["records"] if m["layer"] is not None}, {0, 2})


if __name__ == "__main__":
  unittest.main()
