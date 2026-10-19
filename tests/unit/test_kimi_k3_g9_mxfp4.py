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

"""Kimi-K3 MXFP4 checkpoint path (g9).

The Kimi-K3 Hugging Face release stores its routed experts in the `compressed-tensors`
`mxfp4-pack-quantized` format: `X.weight_packed` (uint8, two E2M1 codes per byte) plus
`X.weight_scale` (uint8 E8M0, one per 32 input columns) instead of `X.weight`. MaxText
has no sub-byte weight format, so `to_maxtext` dequantizes these pairs on the fly.

This file checks that path at two levels:

  1. Codec unit tests for `maxtext.checkpoint_conversion.utils.mxfp4` -- known bytes,
     agreement with the `transformers` / DeepSeek-V4 torch reference decoders, a
     provable round-trip bound, and the converter's `.weight -> pair` resolution.
  2. End to end: a tiny `KimiLinearForCausalLM` is written out as a fake HF release
     (bf16 tensors, MXFP4-packed routed experts, sharded safetensors + index.json,
     `language_model.` key prefix), converted through the real `to_maxtext.main`,
     restored from the resulting Orbax checkpoint into a MaxText `Transformer`, checked
     weight-for-weight against the in-memory conversion path used by the g8 parity
     tests, and its logits compared with PyTorch. Both the lazy (just-in-time HF reads)
     and eager loaders are driven and must produce identical checkpoints.

Run:
  PYTHONPATH=./src JAX_PLATFORMS=cpu pytest tests/unit/test_kimi_k3_g9_mxfp4.py -q
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from flax import nnx
from safetensors.torch import save_file

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import mxfp4
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.common import checkpointing
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import model_creation_utils
from tests.utils.kimi_k3_parity_utils import (
    TinyKimiK3Spec,
    convert_hf_model_state_dict,
    init_hf_uninitialized_params,
    load_hf_reference,
    unpatch_hf_reference,
    load_params_into_nnx,
    make_hf_config,
    make_maxtext_config,
    positions_and_segments,
)
from tests.utils.test_helpers import get_test_config_path

# The e2e test pins every weight exactly against the in-memory conversion path, so the
# logits check is a forward-pass sanity bound rather than the primary assertion.
_E2E_RTOL = 1e-3
_E2E_ATOL = 1e-3

# Routed experts are the only MXFP4-quantized modules in the release
# (`quantization_config.ignore` excludes attention, shared experts, dense MLP, lm_head).
_EXPERT_WEIGHT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$")


# =============================================================================
# 1. Codec
# =============================================================================
class Mxfp4CodecTest(unittest.TestCase):
  """Unit tests for the NumPy MXFP4 codec."""

  def test_known_bytes(self):
    # 0x21: low nibble 1 -> 0.5, high nibble 2 -> 1.0.  0x8F: low 0xF -> -6.0, high 0x8 -> -0.0.
    packed = np.array([[0x21, 0x8F]], dtype=np.uint8)
    np.testing.assert_array_equal(mxfp4.unpack_e2m1(packed), np.array([[0.5, 1.0, -6.0, -0.0]], dtype=np.float32))
    # E8M0 128 -> 2**1 multiplier over a full group of 32.
    packed = np.full((1, 16), 0x21, dtype=np.uint8)
    scale = np.array([[128]], dtype=np.uint8)
    out = mxfp4.dequantize_mxfp4_packed(packed, scale, np.float32)
    self.assertEqual(out.shape, (1, 32))
    np.testing.assert_array_equal(out[0, :4], [1.0, 2.0, 1.0, 2.0])
    self.assertEqual(out.dtype, np.float32)

  def test_e2m1_lut_matches_transformers(self):
    try:
      from transformers.integrations.mxfp4 import FP4_VALUES  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - depends on installed transformers
      self.skipTest("transformers.integrations.mxfp4 unavailable")
    np.testing.assert_array_equal(mxfp4.E2M1_LUT, np.asarray(FP4_VALUES, dtype=np.float32))

  def test_matches_deepseek_torch_reference(self):
    """Byte-for-byte agreement with the DeepSeek-V4 torch decoder on random packed data."""
    from maxtext.checkpoint_conversion.standalone_scripts import (  # pylint: disable=import-outside-toplevel
        deepseek_dequantize,
    )

    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(6, 48), dtype=np.uint8)  # in = 96 -> 3 groups
    scale = rng.integers(120, 134, size=(6, 3), dtype=np.uint8)
    ours = mxfp4.dequantize_mxfp4_packed(packed, scale, ml_dtypes.bfloat16)
    ref = deepseek_dequantize.dequantize_mxfp4(torch.from_numpy(packed).view(torch.int8), torch.from_numpy(scale))
    np.testing.assert_array_equal(ours.astype(np.float32), ref.float().numpy())

  def test_round_trip_bound_and_exactness(self):
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 128)).astype(np.float32) * rng.uniform(1e-2, 1e2, (16, 1)).astype(np.float32)
    packed, scale = mxfp4.quantize_mxfp4_reference(w)
    self.assertEqual(packed.shape, (16, 64))
    self.assertEqual(scale.shape, (16, 4))
    self.assertEqual(packed.dtype, np.uint8)
    self.assertEqual(scale.dtype, np.uint8)
    deq = mxfp4.dequantize_mxfp4_packed(packed, scale, np.float32)
    factor = np.repeat(mxfp4.e8m0_to_float32(scale), mxfp4.MXFP4_GROUP_SIZE, axis=-1)
    # Widest E2M1 gap is 4 -> 6, so nearest rounding is off by at most one scale unit.
    self.assertTrue(np.all(np.abs(deq - w) <= factor))
    # Values already on the grid survive unchanged (incl. sign and zero).
    on_grid = mxfp4.E2M1_LUT[rng.integers(0, 16, (4, 64))] * np.exp2(rng.integers(-3, 3, (4, 2))).repeat(32, axis=-1)
    p2, s2 = mxfp4.quantize_mxfp4_reference(on_grid.astype(np.float32))
    np.testing.assert_array_equal(mxfp4.dequantize_mxfp4_packed(p2, s2, np.float32), on_grid)

  def test_batched_leading_dims(self):
    rng = np.random.default_rng(2)
    w = rng.standard_normal((3, 4, 64)).astype(np.float32)
    packed, scale = mxfp4.quantize_mxfp4_reference(w)
    deq = mxfp4.dequantize_mxfp4_packed(packed, scale)
    for e in range(3):
      p_e, s_e = mxfp4.quantize_mxfp4_reference(w[e])
      np.testing.assert_array_equal(deq[e], mxfp4.dequantize_mxfp4_packed(p_e, s_e))

  def test_rejects_wrong_dtype_or_group(self):
    with self.assertRaises(TypeError):
      mxfp4.unpack_e2m1(np.zeros((2, 4), dtype=np.int8))
    with self.assertRaises(TypeError):
      mxfp4.e8m0_to_float32(np.zeros((2, 1), dtype=np.float32))
    with self.assertRaises(ValueError):  # 16 codes per scale, not 32
      mxfp4.dequantize_mxfp4_packed(np.zeros((2, 8), np.uint8), np.zeros((2, 1), np.uint8))
    with self.assertRaises(ValueError):
      mxfp4.quantize_mxfp4_reference(np.zeros((2, 48), np.float32))

  def test_resolve_pair(self):
    keys = {"a.w1.weight_packed", "a.w1.weight_scale", "b.w2.weight_packed", "c.weight"}
    has = keys.__contains__
    self.assertEqual(mxfp4.resolve_mxfp4_pair("a.w1.weight", has), ("a.w1.weight_packed", "a.w1.weight_scale"))
    self.assertIsNone(mxfp4.resolve_mxfp4_pair("c.weight", has))  # dense weight, no sidecars
    self.assertIsNone(mxfp4.resolve_mxfp4_pair("a.w1.bias", has))  # not a `.weight`
    with self.assertRaises(KeyError):  # only one half present
      mxfp4.resolve_mxfp4_pair("b.w2.weight", has)
    self.assertTrue(mxfp4.is_mxfp4_sidecar_key("a.w1.weight_scale"))
    self.assertFalse(mxfp4.is_mxfp4_sidecar_key("a.w1.weight"))

  def test_converter_hook_dtypes(self):
    """`to_maxtext._maybe_load_mxfp4_weight` honours save_dtype and accepts torch tensors."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    packed, scale = mxfp4.quantize_mxfp4_reference(w)
    store = {"x.weight_packed": torch.from_numpy(packed), "x.weight_scale": torch.from_numpy(scale)}
    expected = mxfp4.dequantize_mxfp4_packed(packed, scale, np.float32)

    out_f32 = to_maxtext._maybe_load_mxfp4_weight(  # pylint: disable=protected-access
        "x.weight", store.__contains__, store.__getitem__, "float32"
    )
    self.assertEqual(out_f32.dtype, np.float32)
    np.testing.assert_array_equal(out_f32, expected)

    out_bf16 = to_maxtext._maybe_load_mxfp4_weight(  # pylint: disable=protected-access
        "x.weight", store.__contains__, store.__getitem__, "bfloat16"
    )
    self.assertEqual(out_bf16.dtype, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(out_bf16.astype(np.float32), expected.astype(ml_dtypes.bfloat16).astype(np.float32))

    self.assertIsNone(
        to_maxtext._maybe_load_mxfp4_weight(  # pylint: disable=protected-access
            "y.weight", store.__contains__, store.__getitem__, "bfloat16"
        )
    )


# =============================================================================
# 2. End to end through `to_maxtext.main`
# =============================================================================
def _write_fake_hf_release(pt_model: torch.nn.Module, hf_dir: str, num_shards: int = 2) -> dict[str, np.ndarray]:
  """Serialises `pt_model` like the Kimi-K3 HF release into `hf_dir`.

  * All keys get the `language_model.` prefix.
  * Routed-expert `w1/w2/w3` weights are MXFP4-quantized into `weight_packed` +
    `weight_scale`; every other tensor is stored as bf16.
  * Tensors are spread over `num_shards` safetensors files with an index.json.

  The model is modified in place so its expert weights hold the *dequantized* values
  (the numbers MaxText will end up with); returns those dequantized expert weights
  keyed by HF name for direct checks.
  """
  tensors: dict[str, torch.Tensor] = {}
  dequantized: dict[str, np.ndarray] = {}
  params = dict(pt_model.named_parameters())
  for name, param in params.items():
    key = "language_model." + name
    if _EXPERT_WEIGHT_RE.search(name):
      packed, scale = mxfp4.quantize_mxfp4_reference(param.detach().numpy())
      base = key[: -len(".weight")]
      tensors[base + mxfp4.MXFP4_PACKED_SUFFIX] = torch.from_numpy(packed)
      tensors[base + mxfp4.MXFP4_SCALE_SUFFIX] = torch.from_numpy(scale)
      deq = mxfp4.dequantize_mxfp4_packed(packed, scale, np.float32)
      dequantized[key] = deq
      with torch.no_grad():
        param.copy_(torch.from_numpy(deq))
    elif name.endswith("self_attn.A_log"):
      # The real release stores A_log as [head_dim] (num_heads values + zero padding); mimic
      # that so the converter's `slice_a_log` hook is exercised end to end.
      head_dim = pt_model.config.linear_attn_config["head_dim"]
      padded = torch.zeros(head_dim, dtype=torch.float32)
      padded[: param.shape[0]] = param.detach()
      tensors[key] = padded
    else:
      tensors[key] = param.detach().to(torch.bfloat16).contiguous()

  os.makedirs(hf_dir, exist_ok=True)
  names = sorted(tensors)
  weight_map: dict[str, str] = {}
  total = 0
  for s in range(num_shards):
    shard = f"model-{s + 1:05d}-of-{num_shards:05d}.safetensors"
    chunk = {n: tensors[n] for n in names[s::num_shards]}
    save_file(chunk, os.path.join(hf_dir, shard), metadata={"format": "pt"})
    for n, t in chunk.items():
      weight_map[n] = shard
      total += t.numel() * t.element_size()
  with open(os.path.join(hf_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
  return dequantized


def _abstract_like(params_state: nnx.State) -> nnx.State:
  """ShapeDtypeStruct twin of a concrete NNX param state (what Orbax restore wants)."""
  return jax.tree.map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=getattr(a, "sharding", None)),
      params_state,
  )


class KimiK3Mxfp4EndToEndTest(unittest.TestCase):
  """Fake MXFP4 HF release -> `to_maxtext.main` -> Orbax -> MaxText logits vs PyTorch."""

  @classmethod
  def setUpClass(cls):
    cls.hf_config_mod, cls.hf_model_mod = load_hf_reference(patch_kda=True)
    # Two 32-column groups per expert row so the group axis is really exercised.
    cls.spec = dataclasses.replace(TinyKimiK3Spec(), routed_expert_hidden_size=64, moe_intermediate_size=64)

    torch.manual_seed(0)
    cls.pt_model = init_hf_uninitialized_params(
        cls.hf_model_mod.KimiLinearForCausalLM(make_hf_config(cls.spec, cls.hf_config_mod))
    ).eval()
    # The release stores bf16; round every weight to bf16 first so PyTorch and the
    # checkpoint agree exactly and the comparison isolates the conversion path.
    with torch.no_grad():
      for p in cls.pt_model.parameters():
        p.copy_(p.to(torch.bfloat16).to(torch.float32))

    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    cls.hf_dir = os.path.join(cls.tmp.name, "hf")
    cls.dequantized_experts = _write_fake_hf_release(cls.pt_model, cls.hf_dir)

    # `to_maxtext` reads the HF architecture from the registry keyed by model_name; swap
    # in the tiny architecture (same 1-indexed `linear_attn_config` shape as the real entry).
    hf_kwargs = {k: v for k, v in cls.spec.hf_config_kwargs().items() if not k.startswith("_")}
    cls.tiny_hf_config = hf_model_configs.KimiK3Config(**hf_kwargs)

    cls.mt_cfg = make_maxtext_config(cls.spec, hardware="cpu", skip_jax_distributed_system=True)

    B, S = cls.spec.batch_size, cls.spec.seq_len
    cls.tokens = np.random.RandomState(0).randint(0, cls.spec.vocab_size, size=(B, S)).astype(np.int32)
    with torch.no_grad():
      cls.logits_pt = cls.pt_model(input_ids=torch.from_numpy(cls.tokens).long()).logits.numpy()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()
    # The KDA patch is process-global; restore it so collection order cannot decide
    # which kernel another test file ends up running against.
    unpatch_hf_reference()

  # --- helpers ----------------------------------------------------------------
  def _convert(self, out_dir: str, **main_kwargs) -> str:
    """Runs `to_maxtext.main` for the tiny model; returns the Orbax items path."""
    args = [sys.argv[0], get_test_config_path(), "model_name=kimi-k3", f"base_output_directory={out_dir}"]
    args += ["run_name=kimi_k3_mxfp4_test", "hardware=cpu", "skip_jax_distributed_system=True"]
    args += [f"{k}={v}" for k, v in self.spec.maxtext_overrides().items()]
    with mock.patch.dict(HF_MODEL_CONFIGS, {"kimi-k3": self.tiny_hf_config}):
      to_maxtext.main(
          args,
          hf_model_path=self.hf_dir,
          save_dtype="float32",
          simulated_cpu_devices_count=1,
          **main_kwargs,
      )
    items = os.path.join(out_dir, "0", "items")
    self.assertTrue(os.path.isdir(items), f"no checkpoint at {items}")
    return items

  def _restore(self, items: str):
    """Builds the tiny MaxText model and loads the converted checkpoint into it."""
    mt_model = model_creation_utils.from_config(
        self.mt_cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0)
    )
    params = nnx.state(mt_model, nnx.Param)
    restored = checkpointing.load_params_from_path(
        items,
        _abstract_like(params),
        self.mt_cfg.checkpoint_storage_concurrent_gb,
        self.mt_cfg.checkpoint_storage_use_ocdbt,
        self.mt_cfg.checkpoint_storage_use_zarr3,
    )
    nnx.update(mt_model, restored)
    return mt_model

  def _mt_logits(self, mt_model) -> np.ndarray:
    positions, _ = positions_and_segments(self.spec.batch_size, self.spec.seq_len)
    return np.asarray(mt_model(jnp.asarray(self.tokens), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False))

  # --- tests ------------------------------------------------------------------
  def test_lazy_loader_dequantizes_experts(self):
    """`LazyHFLoader.get_tensor` materialises `X.weight` from the packed/scale pair."""
    loader = to_maxtext.LazyHFLoader(self.hf_dir, token=None, save_dtype="float32")
    expert_key = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
    self.assertNotIn(expert_key, loader.shard_map)  # only the sidecars exist on disk
    out = loader.get_tensor(expert_key)
    self.assertEqual(out.dtype, np.float32)
    np.testing.assert_array_equal(out, self.dequantized_experts[expert_key])
    # Dense tensors still come through the normal path (bf16 on disk -> f32 here).
    dense = loader.get_tensor("language_model.model.layers.0.mlp.gate_proj.weight")
    self.assertEqual(dense.dtype, np.float32)
    expected = dict(self.pt_model.named_parameters())["model.layers.0.mlp.gate_proj.weight"].detach().numpy()
    np.testing.assert_array_equal(dense, expected)
    with self.assertRaises(ValueError):
      loader.get_tensor("language_model.model.layers.0.does_not_exist.weight")
    # bf16 save mode rounds the dequantized values (they are exactly representable).
    loader_bf16 = to_maxtext.LazyHFLoader(self.hf_dir, token=None, save_dtype="bfloat16")
    out_bf16 = loader_bf16.get_tensor(expert_key)
    self.assertEqual(out_bf16.dtype, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(out_bf16.astype(np.float32), self.dequantized_experts[expert_key])

  def test_eager_conversion_roundtrip(self):
    """Fake release -> `to_maxtext.main` (eager) -> Orbax -> MaxText; exact vs in-memory path."""
    items = self._convert(
        os.path.join(self.tmp.name, "mt_eager"), lazy_load_tensors=False, eager_load_method="safetensors"
    )
    mt_model = self._restore(items)

    # (a) Direct check of the MXFP4 path: the stacked expert kernels must equal the
    #     dequantized HF weights (transposed [out, in] -> [in, out] by the hook).
    for i in range(self.spec.num_layers):
      if not self.spec.is_moe(i):
        continue
      layer = getattr(mt_model.decoder, f"layers_{i}")
      for mt_name, hf_name in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
        kernel = np.asarray(getattr(layer.mlp.routed_experts, mt_name)[...])
        for e in range(self.spec.num_experts):
          hf_key = f"language_model.model.layers.{i}.block_sparse_moe.experts.{e}.{hf_name}.weight"
          np.testing.assert_array_equal(kernel[e], self.dequantized_experts[hf_key].T, err_msg=f"{mt_name} L{i} E{e}")

    # (b) Every restored parameter must be bit-identical to the g8 in-memory path
    #     (same PARAM_MAPPING/HOOK_FNS applied to the torch state_dict directly).
    mem_model = model_creation_utils.from_config(
        self.mt_cfg, devices=jax.devices(), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1)
    )
    load_params_into_nnx(mem_model, convert_hf_model_state_dict(self.pt_model, self.spec), prefix="params")
    restored_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(mt_model, nnx.Param).to_pure_dict())
    memory_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(mem_model, nnx.Param).to_pure_dict())
    self.assertEqual(len(restored_leaves), len(memory_leaves))
    self.assertGreater(len(restored_leaves), 0)
    for (path_r, a), (path_m, b) in zip(restored_leaves, memory_leaves):
      self.assertEqual(path_r, path_m)
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path_r))

    # (c) Logits sanity vs PyTorch. (b) already pins the weights exactly, so this only
    #     guards the forward pass; the tolerance is looser than g8's because this spec
    #     runs wider (64-column, two-group) experts on bf16-rounded weights.
    logits_mt = self._mt_logits(mt_model)
    self.assertEqual(logits_mt.shape, self.logits_pt.shape)
    np.testing.assert_allclose(logits_mt, self.logits_pt, rtol=_E2E_RTOL, atol=_E2E_ATOL)

  def test_lazy_conversion_matches_eager(self):
    """Lazy loader (just-in-time HF reads, `LazyTensor` leaves) yields the same checkpoint."""
    lazy_items = self._convert(os.path.join(self.tmp.name, "mt_lazy"), lazy_load_tensors=True)
    eager_items = self._convert(
        os.path.join(self.tmp.name, "mt_eager2"), lazy_load_tensors=False, eager_load_method="safetensors"
    )
    lazy_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(self._restore(lazy_items), nnx.Param).to_pure_dict())
    eager_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(self._restore(eager_items), nnx.Param).to_pure_dict())
    self.assertEqual(len(lazy_leaves), len(eager_leaves))
    self.assertGreater(len(lazy_leaves), 0)
    for (path_l, a), (path_e, b) in zip(lazy_leaves, eager_leaves):
      self.assertEqual(path_l, path_e)
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=jax.tree_util.keystr(path_l))


if __name__ == "__main__":
  unittest.main()
