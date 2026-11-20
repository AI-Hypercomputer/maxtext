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

"""Tests for on-the-fly FP8 and MXFP4 -> BF16 dequantization in safetensors_dynamic."""

import os

# Simulate 4 devices on CPU so axis-0 load shardings and target shardings are exercised.
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

# pylint: disable=wrong-import-position
import functools
import json
import math
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax.traverse_util
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.checkpoint_conversion.utils import dequantize
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import safetensors_reader
import numpy as np
import pytest

pytestmark = [pytest.mark.decoupled_target]

_FP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
)

_SAFETENSORS_DTYPE_NAMES = {
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.uint8): "U8",
    np.dtype(np.int8): "I8",
    np.dtype(np.int16): "I16",
    np.dtype(np.uint16): "U16",
    np.dtype(np.int32): "I32",
    np.dtype(np.uint32): "U32",
    np.dtype(np.int64): "I64",
    np.dtype(np.uint64): "U64",
    np.dtype(np.float16): "F16",
    np.dtype(jnp.bfloat16): "BF16",
    np.dtype(np.float32): "F32",
    np.dtype(np.float64): "F64",
    np.dtype(jnp.float8_e4m3fn): "F8_E4M3",
    np.dtype(jnp.float8_e5m2): "F8_E5M2",
    np.dtype(jnp.float8_e8m0fnu): "F8_E8M0",
}


def _save_raw_safetensors(tensors: dict[str, np.ndarray | jax.Array], path: str):
  """Writes a SafeTensors file directly, including `F8_E8M0` and `F8_E4M3` dtypes."""
  header = {}
  chunks = []
  offset = 0
  for name in sorted(tensors):
    arr = np.asarray(tensors[name])
    dt = np.dtype(arr.dtype)
    raw = np.ascontiguousarray(arr).view(np.uint8).tobytes()
    header[name] = {
        "dtype": _SAFETENSORS_DTYPE_NAMES[dt],
        "shape": list(arr.shape),
        "data_offsets": [offset, offset + len(raw)],
    }
    chunks.append(raw)
    offset += len(raw)
  header_bytes = json.dumps(header).encode("utf-8")
  with open(path, "wb") as f:
    f.write(len(header_bytes).to_bytes(8, "little"))
    f.write(header_bytes)
    for chunk in chunks:
      f.write(chunk)


def _ref_scale_fp32(scale: np.ndarray) -> np.ndarray:
  if np.dtype(scale.dtype) == np.dtype(np.uint8):
    scale = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(scale), jnp.float8_e8m0fnu))
  return np.asarray(jnp.asarray(scale).astype(jnp.float32), dtype=np.float32)


def _ref_dequant_fp8(weight: np.ndarray, scale: np.ndarray, block_size: int | tuple[int, int] = 128) -> np.ndarray:
  """Reference CPU implementation matching `standalone_scripts/deepseek_dequantize.py:dequantize_fp8`."""
  m, n = weight.shape
  bm_cfg, bn_cfg = (block_size, block_size) if isinstance(block_size, int) else tuple(block_size)
  x = np.asarray(jnp.asarray(weight).astype(jnp.float32))
  s_fp32 = _ref_scale_fp32(scale)
  if m % bm_cfg != 0 or n % bn_cfg != 0:
    s_exp = np.repeat(np.repeat(s_fp32, bm_cfg, axis=0), bn_cfg, axis=1)[:m, :n]
    out = x * s_exp
  else:
    bm, bn = m // scale.shape[0], n // scale.shape[1]
    out = (x.reshape(scale.shape[0], bm, scale.shape[1], bn) * s_fp32[:, None, :, None]).reshape(m, n)
  return np.asarray(jnp.asarray(out).astype(jnp.bfloat16))


def _ref_dequant_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """Reference CPU implementation matching `standalone_scripts/deepseek_dequantize.py:dequantize_mxfp4`."""
  u8 = np.ascontiguousarray(packed).view(np.uint8)
  unpacked = np.stack([_FP4_LUT[u8 & 0xF], _FP4_LUT[(u8 >> 4) & 0xF]], axis=-1)
  q_f32 = unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])
  rows, cols = q_f32.shape[-2:]
  sr, sc = scale.shape[-2:]
  bm, bn = rows // sr, cols // sc
  s_fp32 = _ref_scale_fp32(scale)
  scaled = q_f32.reshape(-1, sr, bm, sc, bn) * s_fp32.reshape(-1, sr, 1, sc, 1)
  return np.asarray(jnp.asarray(scaled).astype(jnp.bfloat16)).reshape(q_f32.shape)


def _mesh():
  devices = jax.devices()[:4]
  grid = (2, len(devices) // 2) if len(devices) % 2 == 0 else (1, len(devices))
  return Mesh(np.array(devices).reshape(grid), ("fsdp", "tensor"))


def _spec(shape, mesh):
  spec, dim = [None] * len(shape), 0
  for axis in ("fsdp", "tensor"):
    while dim < len(shape) and shape[dim] % mesh.shape[axis]:
      dim += 1
    if dim < len(shape):
      spec[dim] = axis
      dim += 1
  return P(*spec)


class DequantizeUnitTest(parameterized.TestCase):
  """Unit tests for `dequantize.py` numerical parity, sharding preservation, and edge cases."""

  @parameterized.named_parameters(
      ("e4m3_f32_divisible", jnp.float8_e4m3fn, "f32", ((64, 64), (2, 2))),
      ("e4m3_bf16_padded", jnp.float8_e4m3fn, "bf16", ((76, 48), (3, 2))),
      ("e4m3_e8m0_single_row_block", jnp.float8_e4m3fn, "e8m0", ((32, 64), (1, 2))),
      ("e5m2_u8_padded", jnp.float8_e5m2, "u8", ((76, 48), (3, 2))),
      ("e5m2_f32_single_row_block", jnp.float8_e5m2, "f32", ((32, 64), (1, 2))),
      ("e5m2_e8m0_divisible", jnp.float8_e5m2, "e8m0", ((64, 64), (2, 2))),
  )
  def test_fp8_matches_reference_bit_for_bit(self, w_dtype, s_kind, shape_and_scale):
    w_shape, s_shape = shape_and_scale
    rng = np.random.default_rng(42)
    w = jnp.asarray(rng.standard_normal(w_shape, dtype=np.float32)).astype(w_dtype)
    if s_kind == "f32":
      s = jnp.asarray(rng.uniform(0.01, 2.0, size=s_shape).astype(np.float32))
    elif s_kind == "bf16":
      s = jnp.asarray(rng.uniform(0.01, 2.0, size=s_shape).astype(np.float32)).astype(jnp.bfloat16)
    elif s_kind == "e8m0":
      raw_u8 = rng.integers(120, 134, size=s_shape, dtype=np.uint8)
      s = jax.lax.bitcast_convert_type(jnp.asarray(raw_u8), jnp.float8_e8m0fnu)
    else:
      s = jnp.asarray(rng.integers(120, 134, size=s_shape, dtype=np.uint8))

    got = dequantize.dequantize_weight(w, s, block_size=32)
    want = _ref_dequant_fp8(np.asarray(w), np.asarray(s), block_size=32)
    self.assertEqual(got.dtype, jnp.bfloat16)
    self.assertEqual(got.shape, w_shape)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

  @parameterized.parameters("e8m0", "u8", "f32")
  def test_mxfp4_all_256_byte_patterns_match_reference(self, s_kind):
    # 16 rows x 16 packed bytes = all 256 possible int8 byte patterns -> unpacked (16, 32).
    packed = np.arange(256, dtype=np.uint8).view(np.int8).reshape(16, 16)
    rng = np.random.default_rng(7)
    if s_kind == "e8m0":
      s = jax.lax.bitcast_convert_type(
          jnp.asarray(rng.integers(122, 132, size=(16, 1), dtype=np.uint8)), jnp.float8_e8m0fnu
      )
    elif s_kind == "u8":
      s = jnp.asarray(rng.integers(122, 132, size=(16, 1), dtype=np.uint8))
    else:
      s = jnp.asarray(rng.uniform(0.25, 4.0, size=(16, 1)).astype(np.float32))

    got = dequantize.dequantize_weight(jnp.asarray(packed), s)
    want = _ref_dequant_mxfp4(packed, np.asarray(s))
    self.assertEqual(got.dtype, jnp.bfloat16)
    self.assertEqual(got.shape, (16, 32))
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

  def test_e8m0_uint8_matches_float8_e8m0fnu_across_all_256_bytes(self):
    """Sweeps all 256 E8M0 byte values (`0..255`), including `u8=1` (`2**-126`) and `u8=255` (`NaN`)."""
    w = jnp.ones((256, 16), dtype=jnp.float8_e4m3fn)
    s_u8 = jnp.arange(256, dtype=jnp.uint8).reshape(256, 1)
    s_e8m0 = jax.lax.bitcast_convert_type(s_u8, jnp.float8_e8m0fnu)

    got_u8 = dequantize.dequantize_weight(w, s_u8, block_size=16)
    got_e8m0 = dequantize.dequantize_weight(w, s_e8m0, block_size=16)
    np.testing.assert_array_equal(np.asarray(got_u8, np.float32), np.asarray(got_e8m0, np.float32))

    # u8 = 1 encodes 2**-126 (smallest normal BF16/FP32 value, strictly positive in BF16).
    self.assertGreater(float(got_u8[1, 0]), 0.0)
    self.assertEqual(float(got_u8[1, 0]), float(2.0**-126))
    # u8 = 255 encodes NaN in E8M0 (not +inf).
    self.assertTrue(bool(jnp.all(jnp.isnan(got_u8[255]))))
    self.assertTrue(bool(jnp.all(jnp.isnan(got_e8m0[255]))))

  def test_axis0_sharded_dequantization_crosses_block_boundaries_and_preserves_sharding(self):
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices.")
    rng = np.random.default_rng(11)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("process", "local"))
    w_sharding = NamedSharding(mesh, P(("process", "local")))
    s_sharding = NamedSharding(mesh, P())

    # Case 1: 36 rows across 4 chips = 9 rows/chip, crossing 16-row block boundaries, with non-divisible edge block!
    w_np = np.asarray(jnp.asarray(rng.standard_normal((36, 40), dtype=np.float32)).astype(jnp.float8_e4m3fn))
    s_np = rng.uniform(0.1, 1.5, size=(3, 3)).astype(np.float32)  # ceil(36/16)=3, ceil(40/16)=3
    got = dequantize.dequantize_weight(jax.device_put(w_np, w_sharding), jax.device_put(s_np, s_sharding), block_size=16)
    want = _ref_dequant_fp8(w_np, s_np, block_size=16)
    self.assertEqual(got.sharding, w_sharding)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    # Case 2: scale_rows=1 < num_devices=4 (like DeepSeek-V3.2 indexer.wk.weight [128, 7168] with scale [1, 56]).
    w1_np = np.asarray(jnp.asarray(rng.standard_normal((16, 32), dtype=np.float32)).astype(jnp.float8_e4m3fn))
    s1_np = rng.uniform(0.1, 1.5, size=(1, 2)).astype(np.float32)
    got1 = dequantize.dequantize_weight(
        jax.device_put(w1_np, w_sharding), jax.device_put(s1_np, s_sharding), block_size=16
    )
    want1 = _ref_dequant_fp8(w1_np, s1_np, block_size=16)
    self.assertEqual(got1.sharding, w_sharding)
    np.testing.assert_array_equal(np.asarray(got1, np.float32), np.asarray(want1, np.float32))

    # Case 3: 3D expert stack (4, 8, 16) packed MXFP4 sharded along axis 0 -> (4, 8, 32).
    w3_np = rng.integers(-128, 127, size=(4, 8, 16), dtype=np.int8)
    s3_np = rng.integers(124, 130, size=(4, 8, 1), dtype=np.uint8)
    got3 = dequantize.dequantize_weight(jax.device_put(w3_np, w_sharding), jax.device_put(s3_np, s_sharding))
    want3 = _ref_dequant_mxfp4(w3_np, s3_np)
    self.assertEqual(got3.sharding, w_sharding)
    self.assertEqual(got3.shape, (4, 8, 32))
    np.testing.assert_array_equal(np.asarray(got3, np.float32), np.asarray(want3, np.float32))

  def test_needs_dequantization_respects_native_fp8_and_mapped_scales(self):
    meta = {
        "layer.q.weight": jax.ShapeDtypeStruct((128, 128), jnp.float8_e4m3fn),
        "layer.q.weight_scale_inv": jax.ShapeDtypeStruct((1, 1), jnp.float32),
        "layer.w1.weight": jax.ShapeDtypeStruct((128, 64), jnp.int8),
        "layer.w1.scale": jax.ShapeDtypeStruct((128, 4), jnp.float8_e8m0fnu),
        "layer.norm.weight": jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        "layer.gate.tid2eid": jax.ShapeDtypeStruct((128, 6), jnp.int64),
    }
    self.assertEqual(dequantize.needs_dequantization("layer.q.weight", jnp.bfloat16, meta), "layer.q.weight_scale_inv")
    self.assertEqual(dequantize.needs_dequantization("layer.w1.weight", jnp.float32, meta), "layer.w1.scale")
    # Native FP8 target leaf keeps FP8 weights as-is.
    self.assertIsNone(dequantize.needs_dequantization("layer.q.weight", jnp.float8_e4m3fn, meta))
    # Explicitly mapped kernel_scale keeps scale separate.
    self.assertIsNone(dequantize.needs_dequantization("layer.q.weight", jnp.bfloat16, meta, {"layer.q.weight_scale_inv"}))
    # Non-quantized BF16 and integer leaves are untouched.
    self.assertIsNone(dequantize.needs_dequantization("layer.norm.weight", jnp.bfloat16, meta))
    self.assertIsNone(dequantize.needs_dequantization("layer.gate.tid2eid", jnp.int32, meta))
    # Untyped string mappings (`{k: k}`) are never classified as companion scales.
    self.assertFalse(dequantize.is_companion_scale_key("layer.q.weight_scale_inv", {k: k for k in meta}))

  def test_leading_axis_shardings_replicates_only_companion_scales(self):
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices.")
    abstract = {
        "layers.0.w.weight": jax.ShapeDtypeStruct((8, 16), jnp.float8_e4m3fn),
        "layers.0.w.scale": jax.ShapeDtypeStruct((8, 4), jnp.float8_e8m0fnu),
        "layers.0.q.weight": jax.ShapeDtypeStruct((8, 16), jnp.float8_e4m3fn),
        "layers.0.q.weight_scale_inv": jax.ShapeDtypeStruct((8, 4), jnp.float32),
        "layers.0.router.scale": jax.ShapeDtypeStruct((8, 4), jnp.bfloat16),  # No `router.weight`: standalone weight.
        "layers.0.norm.weight": jax.ShapeDtypeStruct((8, 16), jnp.bfloat16),  # Non-quantized BF16 `.weight`...
        "layers.0.norm.scale": jax.ShapeDtypeStruct((8, 4), jnp.bfloat16),  # ...so `norm.scale` is NOT a quant scale!
    }
    self.assertFalse(dequantize.is_companion_scale_key("layers.0.norm.scale", abstract))
    shardings = load_dynamic._leading_axis_shardings(abstract, jax.devices()[:4], num_groups=2)  # pylint: disable=protected-access
    self.assertEqual(shardings["layers.0.w.weight"].spec, P(("process", "local")))
    self.assertEqual(shardings["layers.0.w.scale"].spec, P())
    self.assertEqual(shardings["layers.0.q.weight_scale_inv"].spec, P())
    self.assertEqual(shardings["layers.0.router.scale"].spec, P(("process", "local")))
    self.assertEqual(shardings["layers.0.norm.scale"].spec, P(("process", "local")))

  def test_invalid_inputs_raise(self):
    with self.assertRaisesRegex(ValueError, "Unsupported quantized weight dtype"):
      dequantize.dequantize_weight(jnp.ones((4, 4), jnp.bfloat16), jnp.ones((1, 1), jnp.float32))
    with self.assertRaisesRegex(ValueError, "Incompatible quantized weight shape"):
      dequantize.dequantize_weight(jnp.ones((4, 4), jnp.int8), jnp.ones((4,), jnp.float32))
    with self.assertRaisesRegex(ValueError, "incompatible with scale rows"):
      dequantize.dequantize_weight(jnp.ones((10, 16), jnp.float8_e4m3fn), jnp.ones((3, 1), jnp.float32), block_size=16)
    with self.assertRaisesRegex(ValueError, "block_size must be"):
      dequantize.dequantize_weight(jnp.ones((16, 16), jnp.float8_e4m3fn), jnp.ones((1, 1), jnp.float32), block_size=0)

  def test_resolve_fp8_block_size_handles_mock_and_none_configs(self):
    self.assertEqual(dequantize.resolve_fp8_block_size(None), (128, 128))
    self.assertEqual(dequantize.resolve_fp8_block_size(types.SimpleNamespace(model_name=None)), (128, 128))
    self.assertEqual(
        dequantize.resolve_fp8_block_size(types.SimpleNamespace(weight_block_size=True, model_name="")), (128, 128)
    )
    self.assertEqual(dequantize.resolve_fp8_block_size(mock.Mock(model_name="llama3.1-8b")), (128, 128))
    self.assertEqual(
        dequantize.resolve_fp8_block_size(None, {"quantization_config": {"weight_block_size": [64, 32]}}), (64, 32)
    )


def _build_tiny_deepseek_v3_trees(scan_layers: bool, target_dtype, mesh: Mesh):
  """Builds tiny DeepSeek-V3 quantized + offline-BF16 HF tensors (via `hf_shape`) and MaxText target tree."""
  # 1 dense layer (0) + 1 MoE layer (1); 2 routed experts; non-divisible 160-row FP8 weight (ceil(160/128)=2).
  emb, vocab, q_lora, kv_lora, qk_nope, qk_rope, v_dim, heads, mlp, moe_mlp = 16, 32, 16, 14, 76, 4, 8, 2, 32, 16
  hf_cfg = {
      "hidden_size": emb,
      "num_hidden_layers": 2,
      "vocab_size": vocab,
      "q_lora_rank": q_lora,
      "kv_lora_rank": kv_lora,
      "num_attention_heads": heads,
      "qk_nope_head_dim": qk_nope,
      "qk_rope_head_dim": qk_rope,
      "v_head_dim": v_dim,
      "intermediate_size": mlp,
      "moe_intermediate_size": moe_mlp,
      "n_routed_experts": 2,
      "n_shared_experts": 1,
      "first_k_dense_replace": 1,
  }
  mt_cfg = types.SimpleNamespace(
      model_name="deepseek3-671b",
      scan_layers=scan_layers,
      param_scan_axis=1,
      num_decoder_layers=2,
      first_num_dense_layers=1,
      num_experts=2,
      mtp_num_layers=0,
  )
  rng = np.random.default_rng(123)
  quant_hf, bf16_hf = {}, {}
  for name, shape_list in hf_shape.DEEPSEEK_HF_WEIGHTS_TO_SHAPE(hf_cfg).items():
    shape = tuple(shape_list)
    if name.endswith("e_score_correction_bias"):
      quant_hf[name] = bf16_hf[name] = rng.standard_normal(shape, dtype=np.float32)
    elif len(shape) == 2 and not any(s in name for s in ("embed_tokens", "lm_head", "mlp.gate.weight")):
      w = np.asarray(jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(jnp.float8_e4m3fn))
      s = rng.uniform(0.05, 1.5, size=(math.ceil(shape[0] / 128), math.ceil(shape[1] / 128))).astype(np.float32)
      quant_hf[name] = w
      quant_hf[f"{name}_scale_inv"] = s
      bf16_hf[name] = _ref_dequant_fp8(w, s, block_size=128)
    else:
      quant_hf[name] = bf16_hf[name] = np.asarray(
          jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(jnp.bfloat16)
      )

  attn_mt = {
      "pre_self_attention_layer_norm.scale": (emb,),
      "post_self_attention_layer_norm.scale": (emb,),
      "self_attention.kv_norm.scale": (kv_lora,),
      "self_attention.q_norm.scale": (q_lora,),
      "self_attention.wkv_a.kernel": (emb, kv_lora + qk_rope),
      "self_attention.wkv_b.kernel": (kv_lora, heads, qk_nope + v_dim),
      "self_attention.wq_a.kernel": (emb, q_lora),
      "self_attention.wq_b.kernel": (q_lora, heads, qk_nope + qk_rope),
      "self_attention.out.kernel": (heads, v_dim, emb),
  }
  dense_mt = attn_mt | {
      "mlp.wi_0.kernel": (emb, mlp),
      "mlp.wi_1.kernel": (emb, mlp),
      "mlp.wo.kernel": (mlp, emb),
  }
  moe_mt = attn_mt | {
      "DeepSeekMoeBlock_0.shared_experts.wi_0.kernel": (emb, moe_mlp),
      "DeepSeekMoeBlock_0.shared_experts.wi_1.kernel": (emb, moe_mlp),
      "DeepSeekMoeBlock_0.shared_experts.wo.kernel": (moe_mlp, emb),
      "DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel": (emb, 2),
  }
  moe_exp_mt = {
      "DeepSeekMoeBlock_0.MoeBlock_0.wi_0": (emb, moe_mlp),
      "DeepSeekMoeBlock_0.MoeBlock_0.wi_1": (emb, moe_mlp),
      "DeepSeekMoeBlock_0.MoeBlock_0.wo": (moe_mlp, emb),
  }

  flat_shapes = {
      "token_embedder.embedding": (vocab, emb),
      "decoder.decoder_norm.scale": (emb,),
      "decoder.logits_dense.kernel": (emb, vocab),
  }
  if scan_layers:
    for k, s in dense_mt.items():
      flat_shapes[f"decoder.dense_layers.{k}"] = s[:1] + (1,) + s[1:]
    for k, s in moe_mt.items():
      flat_shapes[f"decoder.moe_layers.{k}"] = s[:1] + (1,) + s[1:]
    for k, s in moe_exp_mt.items():
      flat_shapes[f"decoder.moe_layers.{k}"] = (2, 1) + s
    flat_shapes["decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"] = (1, 2)
  else:
    for k, s in dense_mt.items():
      flat_shapes[f"decoder.dense_layers_0.{k}"] = s
    for k, s in moe_mt.items():
      flat_shapes[f"decoder.moe_layers_0.{k}"] = s
    for k, s in moe_exp_mt.items():
      flat_shapes[f"decoder.moe_layers_0.{k}"] = (2,) + s
    flat_shapes["decoder.moe_layers_0.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"] = (2,)

  flat_target = {
      name: jax.ShapeDtypeStruct(
          shape,
          jnp.float32 if name.endswith(".gate.bias") else target_dtype,
          sharding=NamedSharding(mesh, _spec(shape, mesh)),
      )
      for name, shape in flat_shapes.items()
  }
  target_tree = flax.traverse_util.unflatten_dict(flat_target, sep=".")
  pmap = param_mapping.DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=scan_layers)
  hmap = param_mapping.DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=False)
  return quant_hf, bf16_hf, target_tree, pmap, hmap, mt_cfg


def _build_tiny_deepseek_v4_trees(scan_layers: bool, target_dtype, mesh: Mesh):
  """Builds tiny DeepSeek-V4 hybrid FP8 + MXFP4 (`F8_E8M0` `.scale`) + mHC + MTP trees via `hf_shape`."""
  # 3 layers: layer 0 is hash-routed prefix, layers 1 (HCA) & 2 (CSA) form 1 scanned block; plus MTP on scanned.
  emb, vocab, k_mhc, heads, head_dim, q_lora, o_groups, moe_mlp = 32, 32, 2, 2, 16, 32, 2, 32
  mtp_layers = 1 if scan_layers else 0
  kv_dim = 2 * head_dim
  hf_cfg = {
      "hidden_size": emb,
      "vocab_size": vocab,
      "num_hidden_layers": 3,
      "q_lora_rank": q_lora,
      "o_lora_rank": q_lora,
      "o_groups": o_groups,
      "num_attention_heads": heads,
      "num_key_value_heads": 2,
      "head_dim": head_dim,
      "moe_intermediate_size": moe_mlp,
      "n_routed_experts": 2,
      "num_experts_per_tok": 2,
      "hc_mult": k_mhc,
      "num_hash_layers": 1,
      "index_n_heads": heads,
      "index_head_dim": head_dim,
      "compress_ratios": [0, 128, 4],
      "mtp_num_layers": mtp_layers,
  }
  mt_cfg = types.SimpleNamespace(
      model_name="deepseek4-284b",
      scan_layers=scan_layers,
      param_scan_axis=1,
      base_num_decoder_layers=3,
      num_experts=2,
      first_num_hash_layers=1,
      compress_ratios=[0, 128, 4],
      mtp_num_layers=mtp_layers,
      o_groups=o_groups,
  )
  rng = np.random.default_rng(999)
  quant_hf, bf16_hf = {}, {}
  for name, shape_list in hf_shape.DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE(hf_cfg).items():
    if not isinstance(name, str) or name == "layers.0.ffn.gate.bias":
      continue
    shape = tuple(shape_list)
    if name.endswith(".tid2eid"):
      # Include expert IDs > 256 (e.g. 257..383 in DeepSeek-V4-Pro) so accidental BF16 rounding would fail.
      quant_hf[name] = bf16_hf[name] = rng.integers(257, 384, size=shape, dtype=np.int64)
    elif any(s in name for s in ("_fn", "_base", "_scale", "attn_sink", ".ape", "gate.bias")):
      quant_hf[name] = bf16_hf[name] = rng.standard_normal(shape, dtype=np.float32)
    elif any(s in name for s in (".ffn.experts.", ".ffn.shared_experts.", ".e_proj.weight")):
      # Packed MXFP4 (`I8`) + `F8_E8M0` `.scale` (including `mtp.0.e_proj.weight` in the MTP tuple!).
      rows, cols = shape[0], shape[1]
      packed = rng.integers(-128, 127, size=(rows, cols // 2), dtype=np.int8)
      s_u8 = rng.integers(124, 130, size=(rows, cols // 32), dtype=np.uint8)
      s_e8m0 = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(s_u8), jnp.float8_e8m0fnu))
      quant_hf[name] = packed
      quant_hf[f"{name[:-len('.weight')]}.scale"] = s_e8m0
      bf16_hf[name] = _ref_dequant_mxfp4(packed, s_e8m0)
    elif len(shape) == 2 and not any(
        s in name for s in ("embed.weight", "head.weight", "ffn.gate.weight", "weights_proj.weight")
    ):
      w = np.asarray(jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(jnp.float8_e4m3fn))
      s_u8 = rng.integers(124, 130, size=(1, 1), dtype=np.uint8)
      s_e8m0 = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(s_u8), jnp.float8_e8m0fnu))
      quant_hf[name] = w
      quant_hf[f"{name[:-len('.weight')]}.scale"] = s_e8m0
      bf16_hf[name] = _ref_dequant_fp8(w, s_e8m0, block_size=128)
    else:
      quant_hf[name] = bf16_hf[name] = np.asarray(
          jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(jnp.bfloat16)
      )

  def layer_mt_shapes(comp: str | None):
    d = {
        "pre_self_attention_layer_norm.scale": (emb,),
        "post_self_attention_layer_norm.scale": (emb,),
        "mhc_attention.mhc_norm.scale": (k_mhc * emb,),
        "mhc_mlp.mhc_norm.scale": (k_mhc * emb,),
        "self_attention.q_norm.scale": (q_lora,),
        "self_attention.kv_norm.scale": (head_dim,),
        "self_attention.wq_a.kernel": (emb, q_lora),
        "self_attention.wq_b.kernel": (q_lora, heads, head_dim),
        "self_attention.wkv.kernel": (emb, 2, head_dim),
        "self_attention.sinks": (heads,),
        "self_attention.o_a_proj.kernel": (o_groups, (heads * head_dim) // o_groups, q_lora),
        "self_attention.o_b_proj.kernel": (o_groups * q_lora, emb),
        "mlp.MoeBlock_0.gate.kernel": (emb, 2),
        "mlp.shared_experts.wi_0.kernel": (emb, moe_mlp),
        "mlp.shared_experts.wi_1.kernel": (emb, moe_mlp),
        "mlp.shared_experts.wo.kernel": (moe_mlp, emb),
    }
    for mod in ("mhc_attention", "mhc_mlp"):
      d[f"{mod}.pre_alpha"] = (k_mhc * emb, k_mhc)
      d[f"{mod}.post_alpha"] = (k_mhc * emb, k_mhc)
      d[f"{mod}.res_alpha"] = (k_mhc * emb, k_mhc * k_mhc)
      d[f"{mod}.pre_beta"] = (k_mhc,)
      d[f"{mod}.post_beta"] = (k_mhc,)
      d[f"{mod}.res_beta"] = (k_mhc, k_mhc)
      d[f"{mod}.pre_alpha_scale"] = (1,)
      d[f"{mod}.post_alpha_scale"] = (1,)
      d[f"{mod}.res_alpha_scale"] = (1,)
    if comp == "hca":
      d |= {
          "self_attention.hca_compressor.gate_proj.kernel": (emb, head_dim),
          "self_attention.hca_compressor.kv_proj.kernel": (emb, head_dim),
          "self_attention.hca_compressor.kv_norm.scale": (head_dim,),
          "self_attention.hca_compressor.position_bias": (128, head_dim),
      }
    if comp == "csa":
      d |= {
          "self_attention.csa_compressor.gate_proj.kernel": (emb, kv_dim),
          "self_attention.csa_compressor.kv_proj.kernel": (emb, kv_dim),
          "self_attention.csa_compressor.kv_norm.scale": (head_dim,),
          "self_attention.csa_compressor.position_bias": (4, kv_dim),
          "self_attention.csa_compressor.indexer.gate_proj.kernel": (emb, kv_dim),
          "self_attention.csa_compressor.indexer.kv_proj.kernel": (emb, kv_dim),
          "self_attention.csa_compressor.indexer.kv_norm.scale": (head_dim,),
          "self_attention.csa_compressor.indexer.position_bias": (4, kv_dim),
          "self_attention.csa_compressor.indexer.weights_proj.kernel": (emb, heads),
          "self_attention.csa_compressor.indexer.q_proj.kernel": (q_lora, heads, head_dim),
      }
    return d

  exp_mt = {
      "mlp.MoeBlock_0.wi_0": (emb, moe_mlp),
      "mlp.MoeBlock_0.wi_1": (emb, moe_mlp),
      "mlp.MoeBlock_0.wo": (moe_mlp, emb),
  }

  flat_shapes = {
      "token_embedder.embedding": (vocab, emb),
      "decoder.decoder_norm.scale": (emb,),
      "decoder.logits_dense.kernel": (emb, vocab),
      "decoder.hc_head.hc_fn": (k_mhc * emb, k_mhc),
      "decoder.hc_head.hc_base": (k_mhc,),
      "decoder.hc_head.hc_scale": (1,),
  }

  def add_unrolled(prefix: str, is_hash: bool, comp: str | None):
    for k, s in layer_mt_shapes(comp).items():
      flat_shapes[f"{prefix}.{k}"] = s
    for k, s in exp_mt.items():
      flat_shapes[f"{prefix}.{k}"] = (2,) + s
    if is_hash:
      flat_shapes[f"{prefix}.mlp.MoeBlock_0.tid2eid"] = (vocab, 2)
    else:
      flat_shapes[f"{prefix}.mlp.MoeBlock_0.gate.bias"] = (2,)

  if not scan_layers:
    add_unrolled("decoder.layers_0", is_hash=True, comp=None)
    add_unrolled("decoder.layers_1", is_hash=False, comp="hca")
    add_unrolled("decoder.layers_2", is_hash=False, comp="csa")
  else:
    add_unrolled("decoder.layers_0", is_hash=True, comp=None)
    for b_idx, comp in ((0, "hca"), (1, "csa")):
      b_prefix = f"decoder.scanned_blocks.layers_{b_idx}"
      for k, s in layer_mt_shapes(comp).items():
        flat_shapes[f"{b_prefix}.{k}"] = s[:1] + (1,) + s[1:]
      for k, s in exp_mt.items():
        flat_shapes[f"{b_prefix}.{k}"] = (2, 1) + s
      flat_shapes[f"{b_prefix}.mlp.MoeBlock_0.gate.bias"] = (1, 2)

  if mtp_layers:
    mtp_p = "mtp_block.mtp_layer_1"
    flat_shapes[f"{mtp_p}.mtp_1_embedding_norm.scale"] = (emb,)
    flat_shapes[f"{mtp_p}.mtp_1_hidden_state_norm.scale"] = (emb,)
    flat_shapes[f"{mtp_p}.mtp_1_final_norm.scale"] = (emb,)
    flat_shapes[f"{mtp_p}.mtp_1_projection.kernel"] = (2 * emb, emb)
    flat_shapes[f"{mtp_p}.hc_head.hc_fn"] = (k_mhc * emb, k_mhc)
    flat_shapes[f"{mtp_p}.hc_head.hc_base"] = (k_mhc,)
    flat_shapes[f"{mtp_p}.hc_head.hc_scale"] = (1,)
    add_unrolled(f"{mtp_p}.mtp_1_transformer_layer", is_hash=False, comp=None)

  flat_target = {}
  for name, shape in flat_shapes.items():
    if name.endswith((".tid2eid", ".gate.bias")):
      dt = jnp.float32
    else:
      dt = target_dtype
    flat_target[name] = jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, _spec(shape, mesh)))
  target_tree = flax.traverse_util.unflatten_dict(flat_target, sep=".")
  pmap = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=scan_layers)
  hmap = param_mapping.DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=False
  )
  return quant_hf, bf16_hf, target_tree, pmap, hmap, mt_cfg


class QuantizedStreamingEndToEndTest(parameterized.TestCase):
  """End-to-end tests proving on-the-fly quantized load matches offline BF16 load bit for bit."""

  def _write_sharded_files(self, directory: str, tensors: dict, num_files: int = 2):
    keys = sorted(tensors)
    for i in range(num_files):
      _save_raw_safetensors(
          {k: tensors[k] for k in keys[i::num_files]},
          os.path.join(directory, f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"),
      )

  @parameterized.named_parameters(
      ("scanned_bf16_per_layer", True, jnp.bfloat16, 0, 0),
      ("unscanned_fp32_batched", False, jnp.float32, None, safetensors_reader.MIN_PIECE_BYTES),
  )
  def test_deepseek_v3_fp8_streaming_and_transform_match_offline_bf16(
      self, scan_layers, target_dtype, read_bytes_per_host, min_piece_bytes
  ):
    mesh = _mesh()
    quant_hf, bf16_hf, target_tree, pmap, hmap, cfg = _build_tiny_deepseek_v3_trees(scan_layers, target_dtype, mesh)
    quant_dir = self.create_tempdir().full_path
    bf16_dir = self.create_tempdir().full_path
    self._write_sharded_files(quant_dir, quant_hf)
    self._write_sharded_files(bf16_dir, bf16_hf)

    got_stream_quant = hf_streaming_load.load_hf_params_streaming(
        quant_dir,
        target_tree,
        pmap,
        hmap,
        cfg,
        min_bytes_to_split=0,
        read_bytes_per_host=read_bytes_per_host,
        min_piece_bytes=min_piece_bytes,
    )
    want_stream_bf16 = hf_streaming_load.load_hf_params_streaming(
        bf16_dir,
        target_tree,
        pmap,
        hmap,
        cfg,
        min_bytes_to_split=0,
        read_bytes_per_host=read_bytes_per_host,
        min_piece_bytes=min_piece_bytes,
    )
    if scan_layers and len(jax.devices()) >= 4:
      # Multi-process Orbax `load_sharded_hf_state(..., num_groups=2)` path:
      # `(18, 16)` `kv_a_proj_with_mqa.weight` splits across `"process"` (`18 % 2 == 0`, `18 % 4 != 0`),
      # `(160, 16)` `q_b_proj.weight` splits across `("process", "local")`, and `_scale_inv` stays replicated.
      quant_state = load_dynamic.load_sharded_hf_state(quant_dir, num_groups=2)
      self.assertEqual(quant_state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].sharding.spec, P("process"))
      self.assertEqual(quant_state["model.layers.0.self_attn.q_b_proj.weight"].sharding.spec, P(("process", "local")))
      self.assertEqual(quant_state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv"].sharding.spec, P())
    else:
      # Single-process Orbax `load_sharded_hf_state` path (`safetensors_dynamic_streaming=False` default):
      # `(18, 16)` `kv_a_proj_with_mqa.weight` must NOT split axis 1 across `(2, 2)`, and `_scale_inv` stays replicated.
      quant_state = load_dynamic.load_sharded_hf_state(quant_dir)
      kv_spec = quant_state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].sharding.spec
      self.assertTrue(all(axis is None for axis in kv_spec[1:]), f"Quantized inner axis was sharded: {kv_spec}")
      self.assertEqual(quant_state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv"].sharding.spec, P())
    got_transform_quant = load_dynamic.transform_hf_state_to_mt_state(quant_state, target_tree, pmap, hmap, cfg)
    self.assertEmpty(quant_state)

    flat_got = flax.traverse_util.flatten_dict(got_stream_quant["params"], sep=".")
    flat_want = flax.traverse_util.flatten_dict(want_stream_bf16["params"], sep=".")
    flat_transform = flax.traverse_util.flatten_dict(got_transform_quant["params"], sep=".")
    flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")

    self.assertEqual(set(flat_got), set(flat_target))
    self.assertEqual(set(flat_transform), set(flat_target))
    got_np, want_np, transform_np = jax.device_get((flat_got, flat_want, flat_transform))
    for name, target in flat_target.items():
      with self.subTest(name):
        self.assertIsInstance(flat_got[name], jax.Array)
        self.assertEqual(flat_got[name].dtype, target.dtype)
        self.assertEqual(flat_got[name].sharding, target.sharding)
        np.testing.assert_array_equal(got_np[name].astype(np.float32), want_np[name].astype(np.float32))
        np.testing.assert_array_equal(got_np[name].astype(np.float32), transform_np[name].astype(np.float32))

  @parameterized.named_parameters(
      ("scanned_bf16", True, jnp.bfloat16),
      ("unscanned_fp32", False, jnp.float32),
  )
  def test_deepseek_v4_hybrid_fp8_mxfp4_streaming_and_dynamic_state_match_offline_bf16(self, scan_layers, target_dtype):
    mesh = _mesh()
    quant_hf, bf16_hf, target_tree, pmap, hmap, cfg = _build_tiny_deepseek_v4_trees(scan_layers, target_dtype, mesh)
    quant_dir = self.create_tempdir().full_path
    bf16_dir = self.create_tempdir().full_path
    self._write_sharded_files(quant_dir, quant_hf, num_files=3)
    self._write_sharded_files(bf16_dir, bf16_hf, num_files=3)

    if not scan_layers and len(jax.devices()) >= 4:
      # Multi-process `SafetensorsReader` fallback path (`F8_E8M0` header with `num_groups=2`).
      mp_state = load_dynamic.load_sharded_hf_state(quant_dir, num_groups=2)
      self.assertEqual(mp_state["layers.0.attn.wo_a.weight"].sharding.spec, P(("process", "local")))
      self.assertEqual(mp_state["layers.0.attn.wo_a.scale"].sharding.spec, P())
      mp_mt = load_dynamic.transform_hf_state_to_mt_state(mp_state, target_tree, pmap, hmap, cfg)
      self.assertEmpty(mp_state)
    else:
      mp_mt = None

    real_fetch = safetensors_reader.SafetensorsReader.fetch
    stacked_counts = []

    def spy_fetch(reader, request, *args, **kwargs):
      fetched = real_fetch(reader, request, *args, **kwargs)
      stacked_counts.append(sum(len(s.names) for s in fetched.stacks))
      return fetched

    with (
        mock.patch.object(load_dynamic, "get_hf_config_and_mappings", return_value=(pmap, hmap)),
        mock.patch.object(safetensors_reader.SafetensorsReader, "fetch", autospec=True, side_effect=spy_fetch),
    ):
      # 1. Default `safetensors_dynamic_streaming=False` (`load_sharded_hf_state` F8_E8M0 fallback + transform).
      cfg.safetensors_dynamic_streaming = False
      _, got_default = load_dynamic.load_safetensors_dynamic_state(quant_dir, target_tree, cfg)
      # 2. `safetensors_dynamic_streaming=True` (`hf_streaming_load` + SafetensorsReader stacked fetch/unpack).
      cfg.safetensors_dynamic_streaming = True
      _, got_streaming = load_dynamic.load_safetensors_dynamic_state(quant_dir, target_tree, cfg)
      _, want_dynamic = load_dynamic.load_safetensors_dynamic_state(bf16_dir, target_tree, cfg)

    if len(jax.devices()) > 1:
      self.assertTrue(any(c > 0 for c in stacked_counts), f"Expected stacked whole-tensor reads, got {stacked_counts}")

    flat_default = flax.traverse_util.flatten_dict(got_default["params"], sep=".")
    flat_streaming = flax.traverse_util.flatten_dict(got_streaming["params"], sep=".")
    flat_want = flax.traverse_util.flatten_dict(want_dynamic["params"], sep=".")
    flat_mp = flax.traverse_util.flatten_dict(mp_mt["params"], sep=".") if mp_mt is not None else None
    flat_target = flax.traverse_util.flatten_dict(target_tree, sep=".")
    self.assertEqual(set(flat_default), set(flat_target))
    self.assertEqual(set(flat_streaming), set(flat_target))
    default_np, streaming_np, want_np = jax.device_get((flat_default, flat_streaming, flat_want))
    mp_np = jax.device_get(flat_mp) if flat_mp is not None else None
    for name, target in flat_target.items():
      with self.subTest(name):
        self.assertIsInstance(flat_streaming[name], jax.Array)
        self.assertEqual(flat_streaming[name].dtype, target.dtype)
        self.assertEqual(flat_streaming[name].sharding, target.sharding)
        self.assertEqual(flat_default[name].dtype, target.dtype)
        self.assertEqual(flat_default[name].sharding, target.sharding)
        np.testing.assert_array_equal(streaming_np[name], want_np[name])
        np.testing.assert_array_equal(default_np[name], want_np[name])
        if mp_np is not None:
          np.testing.assert_array_equal(mp_np[name], want_np[name])

  def test_non_128_weight_block_size_edge_padded_fp8(self):
    """Non-128 `weight_block_size` (e.g. `[64, 64]`) disambiguates edge-padded FP8 tensors (`96 % 2 == 0` -> 64, not 48)."""
    mesh = _mesh()
    rng = np.random.default_rng(2026)
    # 96 rows with 64-row blocks -> 2 scale rows. Without `weight_block_size=[64, 64]`, `96 % 2 == 0` would infer 48!
    w = np.asarray(jnp.asarray(rng.standard_normal((96, 256), dtype=np.float32)).astype(jnp.float8_e4m3fn))
    s = np.array([[0.25, 0.5, 1.0, 2.0], [4.0, 8.0, 0.125, 1.5]], dtype=np.float32)
    ckpt_dir = self.create_tempdir().full_path
    _save_raw_safetensors({"w.weight": w, "w.weight_scale_inv": s}, os.path.join(ckpt_dir, "model.safetensors"))

    target = {"w": jax.ShapeDtypeStruct((96, 256), jnp.bfloat16, sharding=NamedSharding(mesh, _spec((96, 256), mesh)))}
    pmap = {"params-w": "w.weight"}
    cfg = types.SimpleNamespace(scan_layers=False, param_scan_axis=1, weight_block_size=[64, 64])
    want = _ref_dequant_fp8(w, s, block_size=(64, 64))

    got_stream = hf_streaming_load.load_hf_params_streaming(ckpt_dir, target, pmap, {}, cfg, min_bytes_to_split=0)
    got_transform = load_dynamic.transform_hf_state_to_mt_state(
        {"w.weight": w, "w.weight_scale_inv": s}, target, pmap, {}, cfg
    )
    np.testing.assert_array_equal(np.asarray(got_stream["params"]["w"], np.float32), want.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(got_transform["params"]["w"], np.float32), want.astype(np.float32))

  def test_unscaled_fp8_guard_and_scaled_tuple_dequantization(self):
    """Unscaled FP8 `str` or `tuple` sources targeting >=2B floats raise; scaled tuples, native FP8, & mapped scales pass."""
    mesh = _mesh()
    sharding4 = NamedSharding(mesh, _spec((4, 4), mesh))
    sharding8 = NamedSharding(mesh, _spec((8, 4), mesh))
    cfg = types.SimpleNamespace(scan_layers=False, param_scan_axis=1)
    rng = np.random.default_rng(321)
    concat_hook = {"params-w": lambda pair, shape: jnp.concatenate([pair[0], pair[1]], axis=0).reshape(shape)}

    for fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2):
      w0 = np.asarray(jnp.asarray(rng.standard_normal((4, 4), dtype=np.float32)).astype(fp8_dtype))
      w1 = np.asarray(jnp.asarray(rng.standard_normal((4, 4), dtype=np.float32)).astype(fp8_dtype))
      unscaled_dir = self.create_tempdir().full_path
      _save_raw_safetensors({"w0.weight": w0, "w1.weight": w1}, os.path.join(unscaled_dir, "model.safetensors"))

      # 1. Lone `str` AND `tuple` FP8 sources targeting BF16 without companion scales fail before streaming reads.
      bf16_4 = {"w": jax.ShapeDtypeStruct((4, 4), jnp.bfloat16, sharding=sharding4)}
      bf16_8 = {"w": jax.ShapeDtypeStruct((8, 4), jnp.bfloat16, sharding=sharding8)}
      with mock.patch.object(safetensors_reader.SafetensorsReader, "fetch", autospec=True) as mock_fetch:
        with self.assertRaisesRegex(ValueError, r"w0\.weight.*no companion"):
          hf_streaming_load.load_hf_params_streaming(unscaled_dir, bf16_4, {"params-w": "w0.weight"}, {}, cfg)
        with self.assertRaisesRegex(ValueError, r"w0\.weight.*no companion"):
          hf_streaming_load.load_hf_params_streaming(
              unscaled_dir, bf16_8, {"params-w": ("w0.weight", "w1.weight")}, concat_hook, cfg
          )
        mock_fetch.assert_not_called()
      with self.assertRaisesRegex(ValueError, r"w0\.weight.*no companion"):
        load_dynamic.transform_hf_state_to_mt_state({"w0.weight": w0}, bf16_4, {"params-w": "w0.weight"}, {}, cfg)
      with self.assertRaisesRegex(ValueError, r"w0\.weight.*no companion"):
        load_dynamic.transform_hf_state_to_mt_state(
            {"w0.weight": w0, "w1.weight": w1}, bf16_8, {"params-w": ("w0.weight", "w1.weight")}, concat_hook, cfg
        )

      # 2. Native FP8 target leaf loads unscaled FP8 as-is.
      fp8_target = {"w": jax.ShapeDtypeStruct((4, 4), fp8_dtype, sharding=sharding4)}
      got_fp8 = hf_streaming_load.load_hf_params_streaming(unscaled_dir, fp8_target, {"params-w": "w0.weight"}, {}, cfg)
      self.assertEqual(got_fp8["params"]["w"].dtype, fp8_dtype)
      np.testing.assert_array_equal(np.asarray(got_fp8["params"]["w"]), w0)

      # 3. Scaled FP8 tuple `("w0.weight", "w1.weight")` dequantizes both inputs before the fusion hook;
      #    explicitly mapped `kernel_scale` leaves also pass without error.
      s0 = np.array([[0.5]], dtype=np.float32)
      s1 = np.array([[1.5]], dtype=np.float32)
      scaled_tensors = {"w0.weight": w0, "w0.weight_scale_inv": s0, "w1.weight": w1, "w1.weight_scale_inv": s1}
      scaled_dir = self.create_tempdir().full_path
      _save_raw_safetensors(scaled_tensors, os.path.join(scaled_dir, "model.safetensors"))

      got_tuple_stream = hf_streaming_load.load_hf_params_streaming(
          scaled_dir, bf16_8, {"params-w": ("w0.weight", "w1.weight")}, concat_hook, cfg
      )
      got_tuple_transform = load_dynamic.transform_hf_state_to_mt_state(
          dict(scaled_tensors), bf16_8, {"params-w": ("w0.weight", "w1.weight")}, concat_hook, cfg
      )
      want_concat = np.concatenate([_ref_dequant_fp8(w0, s0), _ref_dequant_fp8(w1, s1)], axis=0)
      np.testing.assert_array_equal(
          np.asarray(got_tuple_stream["params"]["w"], np.float32), want_concat.astype(np.float32)
      )
      np.testing.assert_array_equal(
          np.asarray(got_tuple_transform["params"]["w"], np.float32), want_concat.astype(np.float32)
      )

      mapped_scale_target = {
          "w": jax.ShapeDtypeStruct((4, 4), jnp.bfloat16, sharding=sharding4),
          "w_scale": jax.ShapeDtypeStruct((1, 1), jnp.float32, sharding=NamedSharding(mesh, P())),
      }
      mapped_pmap = {"params-w": "w0.weight", "params-w_scale": "w0.weight_scale_inv"}
      got_mapped = hf_streaming_load.load_hf_params_streaming(scaled_dir, mapped_scale_target, mapped_pmap, {}, cfg)
      np.testing.assert_array_equal(np.asarray(got_mapped["params"]["w_scale"]), s0)

  def test_rename_legacy_hf_keys_renames_companion_scales(self):
    """Legacy multimodal prefixes (`language_model.model.*`) rename companion `*_scale_inv` / `*.scale` keys too."""
    mesh = _mesh()
    cfg = types.SimpleNamespace(scan_layers=False, param_scan_axis=1)
    rng = np.random.default_rng(77)

    q_fp8 = np.asarray(jnp.asarray(rng.standard_normal((8, 16), dtype=np.float32)).astype(jnp.float8_e4m3fn))
    q_scale = np.array([[0.75]], dtype=np.float32)
    w1_packed = rng.integers(-128, 127, size=(8, 16), dtype=np.int8)
    w1_scale = np.asarray(jax.lax.bitcast_convert_type(jnp.full((8, 1), 126, dtype=jnp.uint8), jnp.float8_e8m0fnu))
    norm_bf16 = np.ones((16,), dtype=np.float32)
    unrelated_scale = np.full((16,), 3.0, dtype=np.float32)

    legacy_hf = {
        "language_model.model.layers.0.q.weight": q_fp8,
        "language_model.model.layers.0.q.weight_scale_inv": q_scale,
        "language_model.model.layers.0.w1.scale": w1_scale,  # Alphabetically precedes `w1.weight`.
        "language_model.model.layers.0.w1.weight": w1_packed,
        "language_model.model.layers.0.norm.weight": jnp.asarray(norm_bf16, dtype=jnp.bfloat16),
        "language_model.model.layers.0.norm.scale": jnp.asarray(unrelated_scale, dtype=jnp.bfloat16),
    }
    pmap = {
        "params-q": "model.language_model.layers.0.q.weight",
        "params-w1": "model.language_model.layers.0.w1.weight",
        "params-norm": "model.language_model.layers.0.norm.weight",
    }
    target = {
        "q": jax.ShapeDtypeStruct((8, 16), jnp.bfloat16, sharding=NamedSharding(mesh, _spec((8, 16), mesh))),
        "w1": jax.ShapeDtypeStruct((8, 32), jnp.bfloat16, sharding=NamedSharding(mesh, _spec((8, 32), mesh))),
        "norm": jax.ShapeDtypeStruct((16,), jnp.bfloat16, sharding=NamedSharding(mesh, _spec((16,), mesh))),
    }

    # Both array-valued `hf_state` and string-valued `stored = {k: k}` keep unmapped `norm.scale` untouched.
    renamed_stored = load_dynamic.rename_legacy_hf_keys({k: k for k in legacy_hf}, pmap)
    self.assertIn("language_model.model.layers.0.norm.scale", renamed_stored)

    renamed = load_dynamic.rename_legacy_hf_keys(dict(legacy_hf), pmap)
    self.assertIn("model.language_model.layers.0.q.weight_scale_inv", renamed)
    self.assertIn("model.language_model.layers.0.w1.scale", renamed)
    self.assertIn("language_model.model.layers.0.norm.scale", renamed)

    got_transform = load_dynamic.transform_hf_state_to_mt_state(renamed, target, pmap, {}, cfg)
    ckpt_dir = self.create_tempdir().full_path
    _save_raw_safetensors(legacy_hf, os.path.join(ckpt_dir, "model.safetensors"))
    got_stream = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir,
        target,
        pmap,
        {},
        cfg,
        rename_hf_keys=functools.partial(load_dynamic.rename_legacy_hf_keys, param_map_mt_to_hf=pmap),
    )
    np.testing.assert_array_equal(
        np.asarray(got_transform["params"]["q"], np.float32), _ref_dequant_fp8(q_fp8, q_scale).astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(got_transform["params"]["w1"], np.float32),
        _ref_dequant_mxfp4(w1_packed, w1_scale).astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(got_stream["params"]["q"], np.float32), np.asarray(got_transform["params"]["q"], np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(got_stream["params"]["w1"], np.float32), np.asarray(got_transform["params"]["w1"], np.float32)
    )

  def test_load_sharded_hf_state_wraps_pytree_metadata_in_no_gcs_read_ahead(self):
    """`ocp_v1.pytree_metadata` runs inside `no_gcs_read_ahead()` so Orbax header reads never over-read +5 MiB."""
    ckpt_dir = self.create_tempdir().full_path
    _save_raw_safetensors({"w": np.ones((4, 4), dtype=np.float32)}, os.path.join(ckpt_dir, "model.safetensors"))
    fake_fs = type("GCSFileSystem", (), {"default_block_size": 5 << 20})
    fake_gcsfs = types.SimpleNamespace(GCSFileSystem=fake_fs)
    seen = []
    real_meta = load_dynamic.ocp_v1.pytree_metadata

    def spy_meta(path):
      seen.append(fake_fs.default_block_size)
      return real_meta(path)

    with (
        mock.patch.dict(sys.modules, {"gcsfs": fake_gcsfs}),
        mock.patch.object(load_dynamic.ocp_v1, "pytree_metadata", side_effect=spy_meta),
    ):
      load_dynamic.load_sharded_hf_state(ckpt_dir)
      self.assertEqual(fake_fs.default_block_size, 5 << 20)
    self.assertEqual(seen, [0])

  def test_fallback_safetensors_reader_batches_reads_under_byte_cap(self):
    """`_load_sharded_via_safetensors_reader` splits large checkpoints into bounded lockstep `reader.read` batches."""
    ckpt_dir = self.create_tempdir().full_path
    tensors = {
        "a.weight": np.arange(64, dtype=np.int8).reshape(8, 8),
        "a.scale": np.ones((1, 1), dtype=jnp.float8_e8m0fnu),
        "b.weight": np.arange(64, dtype=np.int8).reshape(8, 8),
        "b.scale": np.ones((1, 1), dtype=jnp.float8_e8m0fnu),
        "c.weight": np.arange(64, dtype=np.int8).reshape(8, 8),
        "c.scale": np.ones((1, 1), dtype=jnp.float8_e8m0fnu),
    }
    _save_raw_safetensors(tensors, os.path.join(ckpt_dir, "model.safetensors"))
    batch_sizes = []
    real_read = safetensors_reader.SafetensorsReader.read

    def spy_read(reader_self, req):
      batch_sizes.append(len(req))
      return real_read(reader_self, req)

    with mock.patch.object(safetensors_reader.SafetensorsReader, "read", autospec=True, side_effect=spy_read):
      got = load_dynamic._load_sharded_via_safetensors_reader(  # pylint: disable=protected-access
          ckpt_dir, jax.devices(), num_groups=2, max_bytes_per_host=40
      )
    self.assertGreaterEqual(len(batch_sizes), 3)
    self.assertEqual(sum(batch_sizes), len(tensors))
    for k in ("a.weight", "b.weight", "c.weight"):
      np.testing.assert_array_equal(np.asarray(got[k]), tensors[k])


if __name__ == "__main__":
  absltest.main()
