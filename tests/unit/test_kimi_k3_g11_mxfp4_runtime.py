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

"""Kimi-K3 on-the-fly MXFP4 routed experts (g11).

The full Kimi-K3 model (~2.78T params, ~98% routed experts) does not fit in bf16 on
any single host, so the routed experts are kept in their released MXFP4 form
(uint8 packed E2M1 nibbles + uint8 E8M0 group scales) and dequantized inside the
forward pass, after the top-k expert gather.

Goalposts:
  * Phase 1 -- `maxtext.layers.mxfp4.dequantize_mxfp4`, a jit-able JAX decoder that is
    bit-exact with the NumPy converter codec (`checkpoint_conversion/utils/mxfp4.py`).
  * Phase 2 -- `KimiRoutedExperts(weight_format="mxfp4")` stores packed uint8 params in
    MaxText orientation, matches a dense module holding the dequantized weights, never
    materializes a full dequantized `[E, d, m]` tensor, and is selectable through the
    `routed_experts_weight_format` config flag.

Run:
  PYTHONPATH=./src JAX_PLATFORMS=cpu pytest tests/unit/test_kimi_k3_g11_mxfp4_runtime.py -q
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core
import ml_dtypes
import numpy as np
import torch
from flax import nnx
from safetensors import safe_open

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import mxfp4 as np_mxfp4
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.common import checkpointing
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import latent_moe
from maxtext.layers import mxfp4 as jax_mxfp4
from maxtext.models import kimi_k3
from maxtext.utils import model_creation_utils
from tests.unit.test_kimi_k3_g9_mxfp4 import _write_fake_hf_release
from tests.utils import kimi_k3_real_ckpt
from tests.utils import kimi_k3_torch_mxfp4 as torch_mxfp4
from tests.utils.kimi_k3_parity_utils import (
    TinyKimiK3Spec,
    init_hf_uninitialized_params,
    load_hf_reference,
    make_hf_config,
    make_maxtext_config,
    make_mesh,
    positions_and_segments,
    unpatch_hf_reference,
)
from tests.utils.test_helpers import get_test_config_path


def _np_dequant_axis(packed, scale, axis):
  """NumPy reference decode along an arbitrary axis (the codec only packs the last axis)."""
  packed_last = np.moveaxis(packed, axis, -1)
  scale_last = np.moveaxis(scale, axis, -1)
  out = np_mxfp4.dequantize_mxfp4_packed(packed_last, scale_last, dtype=np.float32)
  return np.moveaxis(out, -1, axis)


def _quantize_maxtext_layout(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Quantizes a MaxText-orientation weight `[E, in, out]` along its contraction axis (-2).

  Returns `(packed [E, in/2, out], scale [E, in/32, out], dequantized [E, in, out] float32)`.
  Mirrors the HF release: HF stores `[E, out, in]` packed along `in`; MaxText just transposes.
  """
  hf = np.swapaxes(w, -1, -2)
  packed_hf, scale_hf = np_mxfp4.quantize_mxfp4_reference(hf)
  deq_hf = np_mxfp4.dequantize_mxfp4_packed(packed_hf, scale_hf, dtype=np.float32)
  return np.swapaxes(packed_hf, -1, -2), np.swapaxes(scale_hf, -1, -2), np.swapaxes(deq_hf, -1, -2)


# -----------------------------------------------------------------------------
# Phase 1: JAX MXFP4 primitive
# -----------------------------------------------------------------------------
class Phase1JaxMxfp4PrimitiveTest(unittest.TestCase):
  """`dequantize_mxfp4` must be a bit-exact, jit-able twin of the NumPy codec."""

  def test_all_byte_values_and_edge_scales_match_numpy(self):
    # 16 rows x 16 bytes = every byte value once; each row is one 32-wide group.
    # Scales start at 2: codes 0/1 yield fp32 denormal products, which XLA:CPU flushes
    # to zero (FTZ) -- a platform property, not a codec one, and never used by real weights.
    # 255 decodes to +inf in both codecs (so 0 * inf = NaN in the same slots).
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scale = np.array([2, 3, 50, 100, 120, 125, 126, 127, 128, 129, 130, 140, 200, 250, 254, 255], dtype=np.uint8)
    scale = scale.reshape(16, 1)
    want = np_mxfp4.dequantize_mxfp4_packed(packed, scale, dtype=np.float32)
    got = np.asarray(jax_mxfp4.dequantize_mxfp4(jnp.asarray(packed), jnp.asarray(scale), dtype=jnp.float32))
    self.assertEqual(got.dtype, np.float32)
    self.assertEqual(got.shape, (16, 32))
    np.testing.assert_array_equal(got, want)

  def test_contraction_axis_minus2_matches_transposed_numpy(self):
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(3, 64, 48), dtype=np.uint8)  # [E, in/2, out]
    scale = rng.integers(110, 140, size=(3, 4, 48), dtype=np.uint8)  # [E, in/32, out]
    want = _np_dequant_axis(packed, scale, axis=-2)
    got = np.asarray(jax_mxfp4.dequantize_mxfp4(packed, scale, axis=-2, dtype=jnp.float32))
    self.assertEqual(got.shape, (3, 128, 48))
    np.testing.assert_array_equal(got, want)

  def test_jit_bf16_is_exact(self):
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(4, 5, 64), dtype=np.uint8)
    scale = rng.integers(100, 150, size=(4, 5, 4), dtype=np.uint8)
    fn = jax.jit(lambda p, s: jax_mxfp4.dequantize_mxfp4(p, s, dtype=jnp.bfloat16))
    got = fn(packed, scale)
    self.assertEqual(got.dtype, jnp.bfloat16)
    want = np_mxfp4.dequantize_mxfp4_packed(packed, scale, dtype=ml_dtypes.bfloat16)
    np.testing.assert_array_equal(np.asarray(got).astype(np.float32), want.astype(np.float32))

  def test_rejects_bad_inputs(self):
    packed = jnp.zeros((4, 16), jnp.uint8)
    with self.assertRaises(ValueError):  # group size 16*2/2 = 16 != 32
      jax_mxfp4.dequantize_mxfp4(packed, jnp.zeros((4, 2), jnp.uint8))
    with self.assertRaises(ValueError):  # leading dims disagree
      jax_mxfp4.dequantize_mxfp4(packed, jnp.zeros((3, 1), jnp.uint8))
    with self.assertRaises(TypeError):  # packed must be uint8
      jax_mxfp4.dequantize_mxfp4(packed.astype(jnp.int32), jnp.zeros((4, 1), jnp.uint8))


# -----------------------------------------------------------------------------
# Phase 2: MXFP4-packed KimiRoutedExperts
# -----------------------------------------------------------------------------
_E, _D, _M, _K, _N = 8, 64, 96, 2, 5  # experts, latent in, intermediate, top-k, tokens


def _make_experts(weight_format, dtype=jnp.float32):
  return latent_moe.KimiRoutedExperts(
      num_experts=_E,
      in_features=_D,
      intermediate_dim=_M,
      hidden_size=_D,
      top_k=_K,
      dtype=dtype,
      weight_dtype=jnp.float32,
      weight_format=weight_format,
      rngs=nnx.Rngs(0),
  )


def _load_matching_weights(dense_mod, mx_mod, seed=0):
  """Quantizes random dense weights into `mx_mod` and the dequantized copy into `dense_mod`."""
  rng = np.random.default_rng(seed)
  shapes = {"wi_0": (_E, _D, _M), "wi_1": (_E, _D, _M), "wo": (_E, _M, _D)}
  for name, shape in shapes.items():
    w = rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[1])
    packed, scale, deq = _quantize_maxtext_layout(w)
    getattr(mx_mod, f"{name}_packed").value = jnp.asarray(packed)
    getattr(mx_mod, f"{name}_scale").value = jnp.asarray(scale)
    getattr(dense_mod, name).value = jnp.asarray(deq)


def _inputs(seed=0):
  rng = np.random.default_rng(seed)
  x = jnp.asarray(rng.standard_normal((_N, _D)).astype(np.float32))
  idx = jnp.asarray(rng.integers(0, _E, size=(_N, _K)).astype(np.int32))
  w = jnp.asarray(rng.random((_N, _K)).astype(np.float32))
  return x, idx, w


def _iter_eqns(jaxpr):
  for eqn in jaxpr.eqns:
    yield eqn
    for sub in jax_core.jaxprs_in_params(eqn.params):
      yield from _iter_eqns(sub)


class Phase2PackedRoutedExpertsTest(unittest.TestCase):
  """Packed routed experts: storage contract, numerics and dequantize-after-gather."""

  def test_packed_param_layout(self):
    mod = _make_experts("mxfp4")
    expected = {
        "wi_0_packed": (_E, _D // 2, _M),
        "wi_0_scale": (_E, _D // 32, _M),
        "wi_1_packed": (_E, _D // 2, _M),
        "wi_1_scale": (_E, _D // 32, _M),
        "wo_packed": (_E, _M // 2, _D),
        "wo_scale": (_E, _M // 32, _D),
    }
    for name, shape in expected.items():
      param = getattr(mod, name)
      self.assertEqual(param.value.shape, shape, name)
      self.assertEqual(param.value.dtype, jnp.uint8, name)
    for dense_name in ("wi_0", "wi_1", "wo"):
      self.assertFalse(hasattr(mod, dense_name), f"mxfp4 module must not also hold dense {dense_name}")

  def test_default_format_is_dense(self):
    mod = latent_moe.KimiRoutedExperts(
        num_experts=_E, in_features=_D, intermediate_dim=_M, hidden_size=_D, top_k=_K, rngs=nnx.Rngs(0)
    )
    self.assertEqual(mod.wi_0.value.shape, (_E, _D, _M))
    self.assertFalse(hasattr(mod, "wi_0_packed"))

  def test_random_init_is_finite_and_sane(self):
    mod = _make_experts("mxfp4")
    x, idx, w = _inputs()
    out = np.asarray(mod(x, idx, w))
    self.assertTrue(np.all(np.isfinite(out)))
    self.assertGreater(np.abs(out).max(), 0.0)
    self.assertLess(np.abs(out).max(), 1e3)

  def test_matches_dense_module_with_dequantized_weights_fp32(self):
    dense_mod, mx_mod = _make_experts("bf16"), _make_experts("mxfp4")
    _load_matching_weights(dense_mod, mx_mod)
    x, idx, w = _inputs()
    np.testing.assert_allclose(np.asarray(mx_mod(x, idx, w)), np.asarray(dense_mod(x, idx, w)), rtol=1e-6, atol=1e-6)

  def test_matches_dense_module_with_dequantized_weights_bf16(self):
    dense_mod, mx_mod = _make_experts("bf16", jnp.bfloat16), _make_experts("mxfp4", jnp.bfloat16)
    _load_matching_weights(dense_mod, mx_mod, seed=3)
    x, idx, w = _inputs(seed=3)
    got = np.asarray(mx_mod(x.astype(jnp.bfloat16), idx, w)).astype(np.float32)
    want = np.asarray(dense_mod(x.astype(jnp.bfloat16), idx, w)).astype(np.float32)
    self.assertEqual(mx_mod(x.astype(jnp.bfloat16), idx, w).dtype, jnp.bfloat16)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)

  def test_never_materializes_full_dequantized_experts(self):
    """Dequantization must run on the gathered top-k experts, not on all E experts."""
    mod = _make_experts("mxfp4")
    x, idx, w = _inputs()
    graphdef, state = nnx.split(mod)

    def fwd(state, x, idx, w):
      return nnx.merge(graphdef, state)(x, idx, w)

    closed = jax.make_jaxpr(fwd)(state, x, idx, w)
    full_shapes = {(_E, _D, _M), (_E, _M, _D)}
    for eqn in _iter_eqns(closed.jaxpr):
      for var in eqn.outvars:
        aval = var.aval
        if hasattr(aval, "shape") and jnp.issubdtype(aval.dtype, jnp.floating):
          self.assertNotIn(tuple(aval.shape), full_shapes, f"{eqn.primitive} materialized all experts")

  def test_latent_moe_block_threads_weight_format(self):
    block = latent_moe.KimiLatentMoEBlock(
        hidden_size=_D,
        num_experts=_E,
        top_k=_K,
        routed_expert_hidden_size=_D,
        moe_intermediate_size=_M,
        num_shared_experts=1,
        shared_intermediate_size=_M,
        routed_experts_weight_format="mxfp4",
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(block.routed_experts.wi_0_packed.value.dtype, jnp.uint8)
    out = block(jnp.ones((1, 3, _D), jnp.float32))
    self.assertEqual(out.shape, (1, 3, _D))
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))


def _toy_config(**overrides):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      model_name="kimi-k3",
      override_model_config=True,
      base_emb_dim=128,
      base_num_decoder_layers=2,
      base_num_query_heads=4,
      base_num_kv_heads=4,
      head_dim=32,
      vocab_size=256,
      q_lora_rank=32,
      kv_lora_rank=16,
      qk_nope_head_dim=16,
      qk_rope_head_dim=8,
      v_head_dim=16,
      base_mlp_dim=64,
      base_moe_mlp_dim=64,
      moe_intermediate_size=64,
      shared_intermediate_size=64,
      routed_expert_hidden_size=64,
      num_experts=8,
      num_experts_per_tok=2,
      max_target_length=8,
      max_position_embeddings=8,
      original_max_position_embeddings=8,
      max_prefill_predict_length=4,
      per_device_batch_size=1,
      scan_layers=False,
      enable_checkpointing=False,
      attn_res_block_size=2,
      **overrides,
  )


class Phase2ConfigFlagTest(unittest.TestCase):
  """`routed_experts_weight_format` selects the packed experts end to end."""

  def test_default_is_bf16(self):
    self.assertEqual(_toy_config().routed_experts_weight_format, "bf16")

  def test_invalid_value_rejected(self):
    with self.assertRaises(Exception):
      _toy_config(routed_experts_weight_format="int4")

  def test_decoder_layer_uses_packed_experts(self):
    cfg = _toy_config(routed_experts_weight_format="mxfp4")
    layer = kimi_k3.KimiK3DecoderLayer(
        config=cfg,
        model_mode=MODEL_MODE_TRAIN,
        mesh=make_mesh(cfg),
        rngs=nnx.Rngs(0),
        layer_idx=1,
        is_linear_attn=True,
        is_moe=True,
    )
    experts = layer.mlp.routed_experts
    self.assertEqual(experts.wi_0_packed.value.dtype, jnp.uint8)
    self.assertEqual(experts.wo_packed.value.shape, (8, 64 // 2, 64))


# -----------------------------------------------------------------------------
# Phase 3: checkpoint conversion keeps the packed bytes
# -----------------------------------------------------------------------------
_HF_EXPERT_NAMES = (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2"))


def _read_hf_raw(hf_dir: str) -> dict[str, np.ndarray]:
  """Reads every tensor of a (sharded) safetensors release exactly as stored."""
  with open(os.path.join(hf_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
    shards = sorted(set(json.load(f)["weight_map"].values()))
  out = {}
  for shard in shards:
    with safe_open(os.path.join(hf_dir, shard), framework="pt", device="cpu") as f:
      for k in f.keys():
        t = f.get_tensor(k)
        out[k] = t.view(torch.uint8).numpy() if t.dtype == torch.uint8 else t.float().numpy()
  return out


class Phase3Mxfp4ConversionTest(unittest.TestCase):
  """Fake MXFP4 HF release -> `to_maxtext.main(routed_experts_weight_format=mxfp4)` -> packed MaxText."""

  @classmethod
  def setUpClass(cls):
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)
    cls.spec = dataclasses.replace(TinyKimiK3Spec(), routed_expert_hidden_size=64, moe_intermediate_size=64)
    torch.manual_seed(0)
    cls.pt_model = init_hf_uninitialized_params(
        cls.hf_model_mod.KimiLinearForCausalLM(make_hf_config(cls.spec, cls.hf_config_mod))
    ).eval()
    with torch.no_grad():
      for p in cls.pt_model.parameters():
        p.copy_(p.to(torch.bfloat16).to(torch.float32))
    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    cls.hf_dir = os.path.join(cls.tmp.name, "hf")
    _write_fake_hf_release(cls.pt_model, cls.hf_dir)
    cls.hf_raw = _read_hf_raw(cls.hf_dir)
    hf_kwargs = {k: v for k, v in cls.spec.hf_config_kwargs().items() if not k.startswith("_")}
    cls.tiny_hf_config = hf_model_configs.KimiK3Config(**hf_kwargs)
    common = {"hardware": "cpu", "skip_jax_distributed_system": True}
    cls.cfg_mx = make_maxtext_config(cls.spec, routed_experts_weight_format="mxfp4", **common)
    cls.cfg_bf = make_maxtext_config(cls.spec, **common)
    B, S = cls.spec.batch_size, cls.spec.seq_len
    cls.tokens = np.random.RandomState(0).randint(0, cls.spec.vocab_size, size=(B, S)).astype(np.int32)
    with torch.no_grad():
      cls.logits_pt = cls.pt_model(input_ids=torch.from_numpy(cls.tokens).long()).logits.numpy()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()
    unpatch_hf_reference()

  # --- helpers ----------------------------------------------------------------
  def _convert(self, name: str, fmt: str, **main_kwargs) -> str:
    """Converts the HF model to a MaxText checkpoint."""
    out_dir = os.path.join(self.tmp.name, name)
    args = [sys.argv[0], get_test_config_path(), "model_name=kimi-k3", f"base_output_directory={out_dir}"]
    args += ["run_name=kimi_k3_g11", "hardware=cpu", "skip_jax_distributed_system=True"]
    args += [f"{k}={v}" for k, v in self.spec.maxtext_overrides().items()]
    args += [f"routed_experts_weight_format={fmt}"]
    with mock.patch.dict(HF_MODEL_CONFIGS, {"kimi-k3": self.tiny_hf_config}):
      to_maxtext.main(args, hf_model_path=self.hf_dir, save_dtype="float32", simulated_cpu_devices_count=1, **main_kwargs)
    items = os.path.join(out_dir, "0", "items")
    self.assertTrue(os.path.isdir(items), f"no checkpoint at {items}")
    return items

  def _restore(self, items: str, cfg):
    """Restores model parameters from a checkpoint items path."""
    model = model_creation_utils.from_config(cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    abstract = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=getattr(a, "sharding", None)), params
    )
    restored = checkpointing.load_params_from_path(
        items,
        abstract,
        cfg.checkpoint_storage_concurrent_gb,
        cfg.checkpoint_storage_use_ocdbt,
        cfg.checkpoint_storage_use_zarr3,
    )
    nnx.update(model, restored)
    return model

  def _logits(self, model) -> np.ndarray:
    positions, _ = positions_and_segments(self.spec.batch_size, self.spec.seq_len)
    return np.asarray(model(jnp.asarray(self.tokens), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))

  def _assert_packed_bytes_match_release(self, model):
    """Asserts packed weights match reference bytes."""
    for i in range(self.spec.num_layers):
      if not self.spec.is_moe(i):
        continue
      experts = getattr(model.decoder, f"layers_{i}").mlp.routed_experts
      for mt_name, hf_name in _HF_EXPERT_NAMES:
        for suffix in ("packed", "scale"):
          got = np.asarray(getattr(experts, f"{mt_name}_{suffix}")[...])
          self.assertEqual(got.dtype, np.uint8, f"{mt_name}_{suffix} L{i}")
          for e in range(self.spec.num_experts):
            hf_key = f"language_model.model.layers.{i}.block_sparse_moe.experts.{e}.{hf_name}.weight_{suffix}"
            np.testing.assert_array_equal(got[e], self.hf_raw[hf_key].T, err_msg=f"{mt_name}_{suffix} L{i} E{e}")

  # --- tests ------------------------------------------------------------------
  def test_mapping_emits_packed_keys(self):
    mapping_mx = PARAM_MAPPING["kimi-k3"](self.spec.mapping_config(), self.cfg_mx, False)
    mapping_bf = PARAM_MAPPING["kimi-k3"](self.spec.mapping_config(), self.cfg_bf, False)
    mapping_none = PARAM_MAPPING["kimi-k3"](self.spec.mapping_config(), None, False)
    prefix = "params-decoder-layers_1-mlp-routed_experts"
    for mt_name, hf_name in _HF_EXPERT_NAMES:
      for suffix in ("packed", "scale"):
        self.assertEqual(
            mapping_mx[f"{prefix}-{mt_name}_{suffix}"],
            [
                f"language_model.model.layers.1.block_sparse_moe.experts.{e}.{hf_name}.weight_{suffix}"
                for e in range(self.spec.num_experts)
            ],
        )
      self.assertNotIn(f"{prefix}-{mt_name}", mapping_mx)
      self.assertIn(f"{prefix}-{mt_name}", mapping_bf)
      self.assertIn(f"{prefix}-{mt_name}", mapping_none)  # no MaxText config -> dense (default)
      self.assertNotIn(f"{prefix}-{mt_name}_packed", mapping_bf)

  def test_eager_conversion_keeps_packed_bytes_and_matches_logits(self):
    items_mx = self._convert("mx_eager", "mxfp4", lazy_load_tensors=False, eager_load_method="safetensors")
    model_mx = self._restore(items_mx, self.cfg_mx)
    self._assert_packed_bytes_match_release(model_mx)

    # No dense routed-expert kernels anywhere in the packed checkpoint.
    E, d, m = self.spec.num_experts, self.spec.routed_expert_hidden_size, self.spec.moe_intermediate_size
    for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(model_mx, nnx.Param).to_pure_dict()):
      key = jax.tree_util.keystr(path)
      if "routed_experts" in key and "gate" not in key:
        self.assertEqual(np.asarray(leaf).dtype, np.uint8, key)
        self.assertNotIn(tuple(np.shape(leaf)), {(E, d, m), (E, m, d)}, key)

    # Same bytes decoded in-graph vs decoded by the converter (bf16 path): same logits.
    model_bf = self._restore(
        self._convert("bf_eager", "bf16", lazy_load_tensors=False, eager_load_method="safetensors"), self.cfg_bf
    )
    logits_mx, logits_bf = self._logits(model_mx), self._logits(model_bf)
    np.testing.assert_allclose(logits_mx, logits_bf, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logits_mx, self.logits_pt, rtol=1e-3, atol=1e-3)

  def test_lazy_conversion_keeps_packed_bytes(self):
    items = self._convert("mx_lazy", "mxfp4", lazy_load_tensors=True)
    self._assert_packed_bytes_match_release(self._restore(items, self.cfg_mx))


# -----------------------------------------------------------------------------
# Phase 4a: MXFP4-aware PyTorch oracle
# -----------------------------------------------------------------------------
class Phase4TorchMxfp4OracleTest(unittest.TestCase):
  """The HF reference keeps experts packed and decodes them in forward, bit-identical to dense."""

  @classmethod
  def setUpClass(cls):
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)
    cls.spec = dataclasses.replace(TinyKimiK3Spec(), routed_expert_hidden_size=64, moe_intermediate_size=64)
    cls.hf_cfg = make_hf_config(cls.spec, cls.hf_config_mod)
    torch.manual_seed(0)
    cls.pt_model = init_hf_uninitialized_params(cls.hf_model_mod.KimiLinearForCausalLM(cls.hf_cfg)).eval()
    with torch.no_grad():
      for p in cls.pt_model.parameters():
        p.copy_(p.to(torch.bfloat16).to(torch.float32))
    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    cls.hf_dir = os.path.join(cls.tmp.name, "hf")
    _write_fake_hf_release(cls.pt_model, cls.hf_dir)  # also writes dequantized experts back into pt_model
    cls.hf_raw = _read_hf_raw(cls.hf_dir)
    B, S = cls.spec.batch_size, cls.spec.seq_len
    cls.tokens = torch.from_numpy(np.random.RandomState(0).randint(0, cls.spec.vocab_size, size=(B, S))).long()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()
    unpatch_hf_reference()

  def _load(self, keep_mxfp4: bool, dtype=torch.float32):
    return kimi_k3_real_ckpt.load_torch_oracle(
        self.hf_dir,
        self.hf_cfg,
        self.hf_model_mod,
        num_layers=self.spec.num_layers,
        target_pt_dtype=dtype,
        keep_mxfp4=keep_mxfp4,
    )

  def test_torch_dequant_bit_exact_all_bytes(self):
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scale = np.arange(110, 126, dtype=np.uint8).reshape(16, 1)
    want = np_mxfp4.dequantize_mxfp4_packed(packed, scale, dtype=np.float32)
    got = torch_mxfp4.dequantize_mxfp4_torch(torch.from_numpy(packed), torch.from_numpy(scale), torch.float32)
    np.testing.assert_array_equal(got.numpy(), want)
    got_bf16 = torch_mxfp4.dequantize_mxfp4_torch(torch.from_numpy(packed), torch.from_numpy(scale), torch.bfloat16)
    self.assertEqual(got_bf16.dtype, torch.bfloat16)
    np.testing.assert_array_equal(got_bf16.float().numpy(), want)  # E2M1 x 2^k is exact in bf16

  def test_mxfp4_linear_matches_dense_linear(self):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((48, 64)).astype(np.float32)
    packed, scale = np_mxfp4.quantize_mxfp4_reference(w)
    deq = torch.from_numpy(np_mxfp4.dequantize_mxfp4_packed(packed, scale, np.float32))
    lin = torch_mxfp4.Mxfp4Linear(64, 48)
    lin.load_state_dict({"weight_packed": torch.from_numpy(packed), "weight_scale": torch.from_numpy(scale)})
    x = torch.from_numpy(rng.standard_normal((5, 64)).astype(np.float32))
    for dt in (torch.float32, torch.bfloat16):
      want = torch.nn.functional.linear(x.to(dt), deq.to(dt))  # pylint: disable=not-callable
      torch.testing.assert_close(lin(x.to(dt)), want, rtol=0, atol=0)

  def test_patch_replaces_expert_weights_with_packed_buffers(self):
    model = self.hf_model_mod.KimiLinearForCausalLM(self.hf_cfg)
    n = torch_mxfp4.patch_experts_mxfp4(model)
    E = self.spec.num_experts
    self.assertEqual(n, 3 * E * sum(self.spec.is_moe(i) for i in range(self.spec.num_layers)))
    keys = model.state_dict()
    for k, v in keys.items():
      if ".experts." in k:
        self.assertTrue(k.endswith((".weight_packed", ".weight_scale")), k)
        self.assertEqual(v.dtype, torch.uint8, k)
    k = "model.layers.1.block_sparse_moe.experts.0.w2.weight_packed"
    self.assertEqual(tuple(keys[k].shape), (self.spec.routed_expert_hidden_size, self.spec.moe_intermediate_size // 2))

  def test_oracle_loader_mxfp4_matches_dense_loader(self):
    dense = self._load(keep_mxfp4=False)
    packed = self._load(keep_mxfp4=True)
    # The packed oracle holds the release bytes verbatim.
    sd = packed.state_dict()
    for k, v in self.hf_raw.items():
      if k.endswith((".weight_packed", ".weight_scale")):
        np.testing.assert_array_equal(sd[k[len("language_model.") :]].numpy(), v, err_msg=k)
    with torch.no_grad():
      logits_dense = dense(input_ids=self.tokens).logits
      logits_packed = packed(input_ids=self.tokens).logits
      logits_ref = self.pt_model(input_ids=self.tokens).logits
    torch.testing.assert_close(logits_packed, logits_dense, rtol=0, atol=0)
    torch.testing.assert_close(logits_packed, logits_ref, rtol=1e-5, atol=1e-5)

  def test_oracle_loader_mxfp4_bf16_matches_dense_bf16(self):
    with torch.no_grad():
      a = self._load(keep_mxfp4=False, dtype=torch.bfloat16)(input_ids=self.tokens).logits
      b = self._load(keep_mxfp4=True, dtype=torch.bfloat16)(input_ids=self.tokens).logits
    torch.testing.assert_close(b, a, rtol=0, atol=0)

  def test_packed_oracle_expert_storage_is_small(self):
    packed = self._load(keep_mxfp4=True)
    E, d, m = self.spec.num_experts, self.spec.routed_expert_hidden_size, self.spec.moe_intermediate_size
    n_moe = sum(self.spec.is_moe(i) for i in range(self.spec.num_layers))
    expert_bytes = sum(t.numel() * t.element_size() for k, t in packed.state_dict().items() if ".experts." in k)
    self.assertEqual(expert_bytes, n_moe * E * 3 * (d * m // 2 + d * m // 32))
    self.assertFalse(any(".experts." in k for k, _ in packed.named_parameters()))

  def test_hf_reference_init_is_deterministic_on_a_dirty_allocator(self):
    """The HF reference leaves `dt_bias` / `e_score_correction_bias` as `torch.empty`.

    Regression guard for an order-dependent flake: after a memory-heavy test, those
    tensors held NaN and every parity fixture built on them produced NaN logits.
    """
    for _ in range(4):  # dirty the allocator's free lists with NaN-filled blocks
      torch.full((1 << 16,), float("nan")).sum()
    outs = []
    for _ in range(2):
      m = init_hf_uninitialized_params(self.hf_model_mod.KimiLinearForCausalLM(self.hf_cfg)).eval()
      for name, p in m.named_parameters():
        if name.endswith(("dt_bias", "e_score_correction_bias")):
          self.assertTrue(torch.isfinite(p).all(), name)
      outs.append({n: p.detach().clone() for n, p in m.named_parameters() if n.endswith("dt_bias")})
    for n, v in outs[0].items():
      torch.testing.assert_close(outs[1][n], v, rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main()
