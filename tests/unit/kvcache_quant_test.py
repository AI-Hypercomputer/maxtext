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

"""Tests for Qwix KV-cache quantization (`quantize_kvcache=True`)."""

import sys
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import qwix

from maxtext.common.common_types import CACHE_BATCH
from maxtext.common.common_types import CACHE_HEADS
from maxtext.common.common_types import CACHE_KV
from maxtext.common.common_types import CACHE_SEQUENCE
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.inference import kvcache
from maxtext.layers import attention_op
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_LAYER_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "kvcache_quant_test",
    "enable_checkpointing": False,
    "base_emb_dim": 512,
    "base_num_query_heads": 8,
    "base_num_kv_heads": 4,
    "head_dim": 64,
    "max_prefill_predict_length": 64,
    "max_target_length": 128,
}

_LAYOUTS = {
    "default": {},
    # The layout used by the MaxText MLPerf inference scripts.
    "mlperf": {
        "prefill_cache_axis_order": "0,2,1,3",
        "ar_cache_axis_order": "0,2,1,3",
        "compute_axis_order": "0,2,1,3",
    },
}

_QTYPES = {"int8": jnp.int8, "int4": jnp.int4, "fp8": jnp.float8_e4m3fn}
_BOUNDS = {"int8": 127.5, "int4": 7.5, "fp8": 448.0}

# Upper bounds for the relative L2 error of the 3 autoregressive steps against the full-precision
# full-sequence (train mode) output.
_AR_REL_L2_TOL = {"int8": 2.5e-2, "int4": 2.5e-1, "fp8": 7e-2}

_AR_STEPS = 3


def _init_config(**overrides):
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **overrides)


def _kv_overrides(kv_dtype, kv_axis, layout):
  overrides: dict[str, Any] = dict(_LAYOUTS[layout])
  if kv_dtype == "bf16":
    overrides["quantize_kvcache"] = False
  else:
    overrides.update(quantize_kvcache=True, kv_quant_dtype=kv_dtype, kv_quant_axis=kv_axis)
  return overrides


def _axis_order(order: str) -> tuple[int, ...]:
  return tuple(int(i) for i in order.split(","))


def _rel_l2(x, ref):
  x, ref = np.asarray(x, np.float32), np.asarray(ref, np.float32)
  return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


class _LayerRunner:
  """Builds an `Attention` layer like the decoder layers do and runs train, prefill and AR steps."""

  def __init__(self, kv_dtype, kv_axis, layout):
    cfg = _init_config(**_LAYER_CONFIG, **_kv_overrides(kv_dtype, kv_axis, layout))
    self.cfg = cfg
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    b, t, e = cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim
    self.prefill_len = cfg.max_prefill_predict_length
    dummy = jnp.ones((b, t, e))
    self.attn = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=t,
        max_prefill_predict_length=self.prefill_len,
        inputs_q_shape=dummy.shape,
        inputs_kv_shape=dummy.shape,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=cfg.attention_type,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=_axis_order(cfg.prefill_cache_axis_order),
        ar_cache_axis_order=_axis_order(cfg.ar_cache_axis_order),
        compute_axis_order=_axis_order(cfg.compute_axis_order),
    )
    self.inputs = jax.random.normal(jax.random.PRNGKey(0), (b, t, e), dtype=cfg.dtype)
    self.positions = jnp.stack([jnp.arange(t, dtype=jnp.int32) for _ in range(b)])
    self.segment_ids = jnp.zeros((b, t), jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR

  def train(self, attn=None):
    """Runs a full-sequence forward pass in train mode (no KV cache)."""
    attn = attn or self.attn
    out, _ = attn(
        self.inputs,
        self.inputs,
        decoder_segment_ids=self.segment_ids,
        inputs_positions=self.positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return out

  def prefill(self, attn=None):
    """Runs prefill on the first `prefill_len` tokens, populating the prefill KV cache."""
    attn = attn or self.attn
    p = self.prefill_len
    out, _ = attn(
        self.inputs[:, :p],
        self.inputs[:, :p],
        decoder_segment_ids=self.segment_ids[:, :p],
        inputs_positions=self.positions[:, :p],
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )
    return out

  def ar_step(self, idx, attn=None):
    attn = attn or self.attn
    out, _ = attn(
        self.inputs[:, idx : idx + 1],
        self.inputs[:, idx : idx + 1],
        inputs_positions=self.positions[:, idx : idx + 1],
        deterministic=True,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
    return out

  def ar(self, attn=None):
    p = self.prefill_len
    return jnp.concatenate([self.ar_step(idx, attn) for idx in range(p, p + _AR_STEPS)], axis=1)


def _sub_jaxprs(param):
  if hasattr(param, "eqns"):  # Jaxpr
    yield param
  elif hasattr(getattr(param, "jaxpr", None), "eqns"):  # ClosedJaxpr
    yield param.jaxpr
  elif isinstance(param, (tuple, list)):
    for p in param:
      yield from _sub_jaxprs(p)


def _iter_eqns(jaxpr):
  """Yields the equations of `jaxpr` and of all nested jaxprs."""
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in _sub_jaxprs(param):
        yield from _iter_eqns(sub)


class KVQuantTest(parameterized.TestCase):
  """Tests `kvcache.KVQuant.quantize`."""

  @parameterized.product(kv_dtype=("int8", "int4", "fp8"), kv_axis=("dkv", "heads_and_dkv"))
  def test_quantize_scale_and_error(self, kv_dtype, kv_axis):
    cfg = _init_config(**_LAYER_CONFIG, quantize_kvcache=True, kv_quant_dtype=kv_dtype, kv_quant_axis=kv_axis)
    kv_quant = kvcache.KVQuant(cfg)
    # Physical layout of the default AR cache ("1,2,0,3"): [sequence, heads, batch, kv].
    axis_names = (CACHE_SEQUENCE, CACHE_HEADS, CACHE_BATCH, CACHE_KV)
    kv = jax.random.normal(jax.random.PRNGKey(1), (16, 4, 2, 64), jnp.float32) * 3.0
    kv = kv.at[3].set(0.0).astype(jnp.bfloat16)  # An all-zero token must not produce NaN/Inf.

    qvalue, scale = kv_quant.quantize(kv, axis_names)

    self.assertEqual(qvalue.dtype, _QTYPES[kv_dtype])
    self.assertEqual(qvalue.shape, kv.shape)
    self.assertEqual(scale.dtype, kvcache.KV_SCALE_DTYPE)
    reduced = (1, 3) if kv_axis == "heads_and_dkv" else (3,)
    self.assertEqual(scale.shape, tuple(1 if i in reduced else n for i, n in enumerate(kv.shape)))

    kv32 = np.asarray(kv, np.float32)
    absmax = np.max(np.abs(kv32), axis=reduced, keepdims=True)
    scale32 = np.asarray(scale, np.float32)
    nonzero = absmax > 0
    # Qwix symmetric absmax: scale = absmax / bound (bf16 rounding allowed).
    np.testing.assert_allclose(scale32[nonzero], (absmax / _BOUNDS[kv_dtype])[nonzero], rtol=1e-2)

    dequant = np.asarray(qwix.dequantize(qwix.QArray(qvalue, scale.astype(jnp.float32))))
    self.assertTrue(np.all(np.isfinite(dequant)))
    np.testing.assert_array_equal(dequant[3], 0.0)
    if kv_dtype == "fp8":
      # float8_e4m3fn has 3 mantissa bits: relative rounding error <= 2**-4 (plus bf16 rounding).
      max_error = np.abs(kv32) * 0.07 + scale32 * 2.0**-9 + 1e-6
      np.testing.assert_array_less(np.abs(dequant - kv32), max_error)
      return
    # Bound: 0.5 * scale from integer rounding, plus bf16 division rounding on `x / scale`.
    max_error = np.broadcast_to(scale32 * (0.5 + _BOUNDS[kv_dtype] * 2.0**-7) + 1e-6, kv.shape)
    np.testing.assert_array_less(np.abs(dequant - kv32), max_error)
    # Check that Qwix matches or improves on the previous AQT scale formula.
    absmax_bf16 = jnp.max(jnp.abs(kv), axis=reduced, keepdims=True)
    qmax = np.floor(_BOUNDS[kv_dtype])
    legacy_q = jnp.clip(jnp.rint(kv * (_BOUNDS[kv_dtype] / absmax_bf16)), -qmax - 1, qmax)
    legacy = np.asarray(legacy_q, np.float32) * np.asarray(absmax_bf16, np.float32) / _BOUNDS[kv_dtype]
    legacy = np.where(nonzero, legacy, 0.0)
    self.assertLessEqual(_rel_l2(dequant, kv32), 1.05 * _rel_l2(legacy, kv32))


class KVCacheQuantLayerTest(parameterized.TestCase):
  """Prefill + autoregressive decoding of an `Attention` layer with a quantized KV cache."""

  _reference = {}

  def _bf16(self, layout):
    if layout not in self._reference:
      runner = _LayerRunner("bf16", None, layout)
      full = runner.train()
      prefill = runner.prefill()
      ar = runner.ar()
      self._reference[layout] = (np.asarray(full), np.asarray(prefill), np.asarray(ar))
    return self._reference[layout]

  @parameterized.product(kv_dtype=("int8", "int4", "fp8"), kv_axis=("dkv", "heads_and_dkv"), layout=("default", "mlperf"))
  def test_prefill_then_autoregressive(self, kv_dtype, kv_axis, layout):
    ref_full, ref_prefill, ref_ar = self._bf16(layout)
    runner = _LayerRunner(kv_dtype, kv_axis, layout)
    full = np.asarray(runner.train())
    prefill = np.asarray(runner.prefill())

    # Train and prefill attention use the full-precision K/V.
    np.testing.assert_array_equal(full, ref_full)
    np.testing.assert_array_equal(prefill, ref_prefill)

    caches = {"/".join(str(k) for k in path): v for path, v in nnx.to_flat_state(nnx.state(runner.attn, nnx.Cache))}
    for name in ("cached_prefill_key", "cached_prefill_value"):
      (cache,) = [v for k, v in caches.items() if k.endswith(name)]
      self.assertEqual(cache.value.dtype, _QTYPES[kv_dtype], name)
      (scale,) = [v for k, v in caches.items() if k.endswith(name + "_scale")]
      self.assertEqual(scale.value.dtype, kvcache.KV_SCALE_DTYPE, name)

    ar = np.asarray(runner.ar())
    p = runner.prefill_len
    expected = ref_full[:, p : p + _AR_STEPS]
    ar_err = _rel_l2(ar, expected)
    ref_err = _rel_l2(ref_ar, expected)
    logging.info("%s/%s/%s: AR rel_l2 %.3e (bf16 cache: %.3e)", kv_dtype, kv_axis, layout, ar_err, ref_err)
    self.assertLess(ar_err, _AR_REL_L2_TOL[kv_dtype])
    # The cache is actually quantized: the error exceeds that of the bf16 cache.
    self.assertGreater(ar_err, 1.5 * ref_err)

  @parameterized.product(kv_dtype=("int8", "int4", "fp8"), layout=("default", "mlperf"))
  def test_autoregressive_dot_products_consume_quantized_cache(self, kv_dtype, layout):
    """Checks that AR QK^T and PV dot_general ops take quantized K/V directly without dequantizing first."""
    runner = _LayerRunner(kv_dtype, "heads_and_dkv", layout)
    runner.prefill()
    graphdef, state = nnx.split(runner.attn)
    idx = runner.prefill_len

    def ar_step(state):
      return runner.ar_step(idx, nnx.merge(graphdef, state))

    jaxpr = jax.make_jaxpr(ar_step)(state)
    qtype = jnp.dtype(_QTYPES[kv_dtype])
    quantized_dots = [
        eqn
        for eqn in _iter_eqns(jaxpr.jaxpr)
        if eqn.primitive.name == "dot_general" and any(v.aval.dtype == qtype for v in eqn.invars)
    ]
    # QK^T and PV over both the prefill and the AR cache.
    self.assertLen(quantized_dots, 4, jaxpr)


class KVCacheQuantInterceptionTest(absltest.TestCase):
  """Checks that model-wide Qwix interception (`use_qwix_quantization=True`) leaves KV-cache einsums untouched."""

  def _quantize_attention_op(self, runner):
    """Returns `runner.attn` with Qwix interception quantizing every `dot_general` in `attention_op`."""
    provider = qwix.QtProvider(
        [
            qwix.QtRule(
                module_path="attention_op",
                weight_qtype=jnp.int8,
                act_qtype=jnp.int8,
                op_names=("dot_general",),
            )
        ]
    )
    p = runner.prefill_len
    return qwix.quantize_model(
        runner.attn,
        provider,
        runner.inputs[:, :p],
        runner.inputs[:, :p],
        decoder_segment_ids=runner.segment_ids[:, :p],
        inputs_positions=runner.positions[:, :p],
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

  def test_autoregressive_output_unchanged_by_interception(self):
    runner = _LayerRunner("int8", "heads_and_dkv", "default")
    intercepted = self._quantize_attention_op(runner)
    runner.prefill()
    runner.prefill(intercepted)
    expected = np.asarray(runner.ar())
    np.testing.assert_array_equal(np.asarray(runner.ar(intercepted)), expected)

  def test_unwrapped_einsum_is_intercepted(self):
    """Confirms that without `without_qwix_interception`, model-wide interception alters the AR output."""
    runner = _LayerRunner("int8", "heads_and_dkv", "default")
    intercepted = self._quantize_attention_op(runner)
    runner.prefill()
    runner.prefill(intercepted)
    expected = np.asarray(runner.ar())
    with mock.patch.object(attention_op, "_qwix_einsum", qwix.einsum):
      actual = np.asarray(runner.ar(intercepted))
    self.assertFalse(np.array_equal(actual, expected))


class KVCacheBatchBroadcastTest(parameterized.TestCase):
  """Checks that `KVCache.update_ar_key_value` grows a batch-1 AR cache to the batch size of the new token."""

  _BATCH = 3
  _HEADS = 4
  _HEAD_DIM = 16
  _PREFILL_LEN = 4
  _TARGET_LEN = 12

  def _make_cache(self, cfg, batch):
    return kvcache.KVCache(
        max_prefill_length=self._PREFILL_LEN,
        max_target_length=self._TARGET_LEN,
        batch=batch,
        key_seq_len=1,
        value_seq_len=1,
        key_heads=self._HEADS,
        value_heads=self._HEADS,
        key_head_size=self._HEAD_DIM,
        value_head_size=self._HEAD_DIM,
        dtype=jnp.bfloat16,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=_axis_order(cfg.prefill_cache_axis_order),
        ar_cache_axis_order=_axis_order(cfg.ar_cache_axis_order),
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )

  def _write_token(self, cache, key, value, index):
    cache.update_ar_key_value(
        key,
        value,
        cache.ar_key_vars,
        cache.ar_value_vars,
        one_hot_indices=jnp.array([index], jnp.int32),
        lengths=jnp.zeros((key.shape[0],), jnp.int32),
        use_ragged_attention=False,
    )

  @parameterized.product(kv_dtype=("bf16", "int8"), layout=("default", "mlperf"))
  def test_update_ar_key_value_broadcasts_batch_one_cache(self, kv_dtype, layout):
    """A batch-1 cache updated with a batch-3 token must equal a cache allocated with batch 3."""
    # `dkv` keeps a `heads` axis in the scale caches, so that no two cache axes have the same size.
    cfg = _init_config(**_LAYER_CONFIG, **_kv_overrides(kv_dtype, "dkv", layout))
    # Tokens are in logical [batch, sequence, heads, d_kv] order.
    batch_one_shape = (1, 1, self._HEADS, self._HEAD_DIM)
    full_batch_shape = (self._BATCH, 1, self._HEADS, self._HEAD_DIM)
    first_key = jax.random.normal(jax.random.PRNGKey(0), batch_one_shape, jnp.bfloat16)
    first_value = jax.random.normal(jax.random.PRNGKey(1), batch_one_shape, jnp.bfloat16)
    second_key = jax.random.normal(jax.random.PRNGKey(2), full_batch_shape, jnp.bfloat16)
    second_value = jax.random.normal(jax.random.PRNGKey(3), full_batch_shape, jnp.bfloat16)

    cache = self._make_cache(cfg, batch=1)
    self._write_token(cache, first_key, first_value, index=0)
    self._write_token(cache, second_key, second_value, index=1)

    reference = self._make_cache(cfg, batch=self._BATCH)
    self._write_token(
        reference, jnp.repeat(first_key, self._BATCH, axis=0), jnp.repeat(first_value, self._BATCH, axis=0), index=0
    )
    self._write_token(reference, second_key, second_value, index=1)

    names = ["cached_ar_key", "cached_ar_value"]
    if cfg.quantize_kvcache:
      names += ["cached_ar_key_scale", "cached_ar_value_scale"]
    for name in names:
      with self.subTest(name):
        np.testing.assert_array_equal(
            np.asarray(getattr(cache, name).get_value(), np.float32),
            np.asarray(getattr(reference, name).get_value(), np.float32),
        )


class KVCacheQuantConfigTest(parameterized.TestCase):
  """Config validation for `quantize_kvcache`."""

  def test_fp8_accepted(self):
    cfg = _init_config(**_LAYER_CONFIG, quantize_kvcache=True, kv_quant_dtype="fp8")
    self.assertEqual(quantizations.configure_kv_quant(cfg).dtype, jnp.float8_e4m3fn)

  @parameterized.named_parameters(
      ("chunked_prefill", {"use_chunked_prefill": True}, "use_chunked_prefill"),
      ("ragged", {"use_ragged_attention": True}, "use_ragged_attention"),
      ("mla_latent", {"attention_type": "mla", "mla_naive_kvcache": False}, "mla_naive_kvcache=False"),
      ("moba", {"moba": True, "attention": "dot_product"}, "moba"),
      ("compressed", {"attention_type": "compressed"}, "compressed"),
      ("float16", {"dtype": "float16"}, "float16"),
  )
  def test_unsupported_combinations_rejected(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      _init_config(**_LAYER_CONFIG, quantize_kvcache=True, **overrides)

  def test_unsupported_combination_allowed_without_kv_quant(self):
    _init_config(**_LAYER_CONFIG, quantize_kvcache=False, use_ragged_attention=True)

  def test_vllm_rpa_warns(self):
    with self.assertLogs(config_types.logger, "WARNING") as logs:
      _init_config(**_LAYER_CONFIG, quantize_kvcache=True, attention="vllm_rpa")
    self.assertTrue(any("vllm_rpa" in line for line in logs.output), logs.output)


if __name__ == "__main__":
  absltest.main()
