# Copyright 2023–2026 Google LLC
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

"""Unit tests for Inkling 2 Mini checkpoint conversion (`to_maxtext` and `to_huggingface`).

Core tests are strictly torch-free and CPU-only:
  1. Registration in `HF_MODEL_CONFIGS`, `HF_SHAPE`, `PARAM_MAPPING`, `HOOK_FNS`, and FP32 suffixes.
  2. 100% key bijection against the actual `Inkling2` MaxText abstract parameter tree for both
     `scan_layers=False` and `scan_layers=True` (and both `param_scan_axis=0` and `1`).
  3. Exact NVFP4 and MXFP8 dequantization + interleaved `w13` de-interleaving (`[0::2]`/`[1::2]`)
     against independent reference math (including non-multiple-of-8 expert counts).
  4. Strict FP32 preservation under `save_dtype=bfloat16` matching HF `_keep_in_fp32_modules_strict`.
  5. Lossless `MT -> HF -> MT` round-trip (`max_abs_diff == 0.0`) for both `scan_layers=False`
     and `scan_layers=True` (`param_scan_axis=0` and `1`) across eager and lazy loaders.

Additional `HAS_TORCH`-guarded tests verify `_convert_tensor_to_numpy(dtype="raw")`,
`LazyHFLoader` optional keys on both indexed and single-file safetensors, and (when the local
`tiny-inkling2-mini` fixture is present) bit-exact parity against Hugging Face
`Inkling2LoopForCausalLM.from_pretrained` plus end-to-end `to_maxtext` -> Orbax -> `to_huggingface`
-> `AutoModelForCausalLM.from_pretrained` round-trip.
"""

import copy
from functools import partial
import json
import os
import sys
import tempfile
import unittest

from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from safetensors.numpy import save_file as safetensors_save_file

from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS, inkling2_mini_dict
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE, INKLING2_HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    OptionalHFKey,
    PARAM_MAPPING,
    _dequantize_inkling2_experts,
)
from maxtext.checkpoint_conversion.utils.tensor_handling import single_stack_axis
from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

try:
  import torch  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  HAS_TORCH = False

try:
  import transformers  # pylint: disable=g-import-not-at-top

  HAS_INKLING2_HF = HAS_TORCH and hasattr(transformers, "Inkling2LoopForCausalLM")
except ImportError:
  transformers = None
  HAS_INKLING2_HF = False

_TINY_INKLING2_DIR = os.environ.get("INKLING2_TINY_CKPT_DIR", "")
HAS_TINY_INKLING2_CKPT = (
    bool(_TINY_INKLING2_DIR)
    and os.path.isdir(_TINY_INKLING2_DIR)
    and os.path.isfile(os.path.join(_TINY_INKLING2_DIR, "model.safetensors"))
)

_E2M1_REF = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _synthetic_hf_config(
    *,
    num_layers: int = 12,
    block_size: int = 2,
    core_start: int = 4,
    core_end: int = 8,
    hidden_size: int = 64,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    d_rel: int = 8,
    rel_extent: int = 16,
    conv_kernel: int = 4,
    gate_rank: int = 16,
    num_experts: int = 4,
    moe_inter: int = 32,
    vocab_size: int = 128,
    unpadded_vocab_size: int = 100,
) -> dict:
  """Builds a small non-square Inkling 2 HF text config for CPU unit tests."""
  cfg = copy.deepcopy(inkling2_mini_dict)
  cfg.update(
      vocab_size=vocab_size,
      unpadded_vocab_size=unpadded_vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_layers,
      num_attention_heads=num_q_heads,
      num_key_value_heads=num_kv_heads,
      head_dim=head_dim,
      swa_num_attention_heads=num_q_heads,
      swa_num_key_value_heads=num_kv_heads,
      swa_head_dim=head_dim,
      sliding_window_size=16,
      d_rel=d_rel,
      rel_extent=rel_extent,
      log_scaling_n_floor=32,
      conv_kernel_size=conv_kernel,
      moe_intermediate_size=moe_inter,
      n_routed_experts=num_experts,
      num_experts_per_tok=2,
      n_layers_per_attn_res_block=block_size,
      layer_types=["hybrid" if (i + 1) % block_size == 0 else "hybrid_sliding" for i in range(num_layers)],
      loop_layer_index_start=core_start,
      loop_layer_index_end=core_end,
      loop_num_passes=3,
      loop_gate_rank=gate_rank,
  )
  return cfg


def _synthetic_maxtext_config(
    hf_cfg: dict,
    *,
    scan_layers: bool = False,
    param_scan_axis: int = 1,
    weight_dtype: str = "float32",
):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      model_name="inkling2-mini",
      override_model_config=True,
      base_emb_dim=hf_cfg["hidden_size"],
      base_num_query_heads=hf_cfg["num_attention_heads"],
      base_num_kv_heads=hf_cfg["num_key_value_heads"],
      head_dim=hf_cfg["head_dim"],
      base_mlp_dim=hf_cfg["moe_intermediate_size"],
      base_moe_mlp_dim=hf_cfg["moe_intermediate_size"],
      base_num_decoder_layers=hf_cfg["num_hidden_layers"],
      vocab_size=hf_cfg["vocab_size"],
      unpadded_vocab_size=hf_cfg["unpadded_vocab_size"],
      attn_res_block_size=hf_cfg["n_layers_per_attn_res_block"],
      inkling2_core_start=hf_cfg["loop_layer_index_start"],
      inkling2_core_end=hf_cfg["loop_layer_index_end"],
      inkling2_num_passes=hf_cfg["loop_num_passes"],
      inkling2_message_gate_rank=hf_cfg["loop_gate_rank"],
      inkling2_d_rel=hf_cfg["d_rel"],
      inkling2_rel_extent=hf_cfg["rel_extent"],
      sliding_window_size=hf_cfg["sliding_window_size"],
      inkling2_log_scaling_n_floor=float(hf_cfg["log_scaling_n_floor"]),
      short_conv_kernel_size=hf_cfg["conv_kernel_size"],
      num_experts=hf_cfg["n_routed_experts"],
      num_experts_per_tok=hf_cfg["num_experts_per_tok"],
      max_target_length=16,
      per_device_batch_size=1,
      dtype="float32",
      weight_dtype=weight_dtype,
      matmul_precision="highest",
      packing=False,
      enable_checkpointing=False,
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
  )


def _flatten_mt_keys(keys) -> list[str]:
  out = []
  for k in keys:
    if isinstance(k, tuple):
      out.extend(k)
    else:
      out.append(k)
  return out


def _flatten_hf_primary_keys(hf_value) -> list[str]:
  """Extracts all required (non-OptionalHFKey) HF keys from a mapping value."""
  if isinstance(hf_value, list):
    return [k for item in hf_value for k in _flatten_hf_primary_keys(item)]
  if isinstance(hf_value, tuple):
    return [str(k) for k in hf_value if not isinstance(k, OptionalHFKey)]
  return [str(hf_value)]


def _make_synthetic_hf_state(hf_cfg: dict, *, quantized: bool = False, seed: int = 0) -> dict[str, np.ndarray]:
  """Builds a synthetic `model.llm.*` state dict (optionally mixed NVFP4 + MXFP8 experts)."""
  rng = np.random.default_rng(seed)
  shapes = INKLING2_HF_SHAPE(hf_cfg)
  num_layers = hf_cfg["num_hidden_layers"]
  nvfp4_cutoff = max(1, num_layers - hf_cfg["n_layers_per_attn_res_block"])
  state = {}

  for key, shape in shapes.items():
    if isinstance(key, tuple):
      continue
    shape_t = tuple(shape)
    if quantized and key.endswith((".w13_weight", ".w2_weight")):
      layer_idx = int(key.split(".")[3])
      e, rows, cols = shape_t[0], shape_t[1], shape_t[2]
      if layer_idx < nvfp4_cutoff:
        state[key] = rng.integers(0, 256, size=(e, rows, cols // 2), dtype=np.uint8)
        state[f"{key}_scale"] = rng.uniform(0.25, 2.0, size=(e, rows, cols // 16)).astype(ml_dtypes.float8_e4m3fn)
        state[f"{key}_amax"] = rng.uniform(0.5, 4.0, size=(e,)).astype(np.float32)
        state[f"{key}_outer_scale_denominator"] = np.array([2688.0], dtype=np.float32)
      else:
        state[key] = rng.uniform(-2.0, 2.0, size=(e, rows, cols)).astype(ml_dtypes.float8_e4m3fn)
        state[f"{key}_scale"] = rng.integers(123, 131, size=(e, rows, cols // 32), dtype=np.uint8)
    else:
      state[key] = (rng.standard_normal(shape_t) * 0.1).astype(np.float32)
  return state


def _convert_hf_state_in_memory(
    hf_state: dict[str, np.ndarray],
    hf_cfg: dict,
    mt_cfg,
    abstract_dict: dict[str, tuple[int, tuple[int, ...]]],
    *,
    save_dtype: str = "float32",
    lazy: bool = False,
) -> dict[str, np.ndarray]:
  """Runs `to_maxtext`'s exact per-parameter loading & hook pipeline in memory."""
  param_map = PARAM_MAPPING["inkling2-mini"](hf_cfg, mt_cfg, mt_cfg.scan_layers)
  hook_map = HOOK_FNS["inkling2-mini"](hf_cfg, mt_cfg, mt_cfg.scan_layers, saving_to_hf=False)
  filtered_keys = conversion_utils.validate_and_filter_param_map_keys(param_map.keys(), set(abstract_dict.keys()))

  def _getter(key, dtype=None):
    if key is None:
      return None
    str_key = str(key)
    if str_key not in hf_state:
      if isinstance(key, OptionalHFKey):
        return None
      raise ValueError(f"Missing HF key: {key}")
    return to_maxtext._convert_tensor_to_numpy(hf_state[str_key], dtype or save_dtype)  # pylint: disable=protected-access

  final_weights = [None] * len(abstract_dict)
  for mt_key in filtered_keys:
    hf_keys = param_map[mt_key]
    hook_fn = hook_map.get(mt_key)
    idx_or_indices, shape_or_shapes = to_maxtext._get_maxtext_indices_and_shapes(  # pylint: disable=protected-access
        mt_key, abstract_dict
    )
    param_dtype = to_maxtext._param_save_dtype(mt_key, save_dtype, "inkling2-mini")  # pylint: disable=protected-access
    if to_maxtext._needs_raw_hf_tensors(hf_keys):  # pylint: disable=protected-access
      getter = partial(_getter, dtype="raw")
      target_np_dtype = to_maxtext._target_numpy_dtype(param_dtype)  # pylint: disable=protected-access

      def _cast_hook(weight, target_shape, _hook=hook_fn, _dtype=target_np_dtype):
        out = to_maxtext.apply_hook_fns(weight, target_shape, _hook)
        return out.astype(_dtype) if getattr(out, "dtype", None) != _dtype else out

      eff_hook = _cast_hook
    else:
      getter = _getter if param_dtype == save_dtype else partial(_getter, dtype=param_dtype)
      eff_hook = hook_fn

    load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
        hf_keys, getter, eff_hook, shape_or_shapes, mt_cfg, mt_key
    )
    to_maxtext._get_maxtext_weight(  # pylint: disable=protected-access
        load_fn, idx_or_indices, shape_or_shapes, mt_key, final_weights, param_dtype, lazy
    )

  idx_to_key = {idx: k for k, (idx, _) in abstract_dict.items()}
  return {idx_to_key[i]: np.array(w) for i, w in enumerate(final_weights)}


def _export_mt_state_in_memory(
    mt_state: dict[str, np.ndarray],
    hf_cfg: dict,
    mt_cfg,
) -> dict[str, np.ndarray]:
  """Runs `to_huggingface`'s exact `process_maxtext_param` pipeline in memory."""
  mappings = to_huggingface._get_model_mappings(  # pylint: disable=protected-access
      "inkling2-mini", mt_cfg.scan_layers, hf_cfg, mt_cfg
  )
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]
  hook_map = mappings["hook_fn_mapping"]
  filtered = conversion_utils.validate_and_filter_param_map_keys(param_map, set(mt_state.keys()))

  exported = {}
  for key in filtered:
    if isinstance(key, tuple):
      weight = [jnp.asarray(mt_state[sub]) for sub in key]
    else:
      weight = jnp.asarray(mt_state[key])
    for hf_k, hf_v in conversion_utils.process_maxtext_param(key, weight, param_map, hook_map, shape_map, mt_cfg):
      exported[hf_k] = hf_v
  return exported


class Inkling2CheckpointConversionTest(parameterized.TestCase):
  """Torch-free tests for Inkling 2 Mini checkpoint conversion mappings, hooks, and round-trips."""

  def test_registrations_and_full_model_shape_schema(self):
    self.assertIn("inkling2-mini", HF_MODEL_CONFIGS)
    self.assertIn("inkling2-mini", HF_SHAPE)
    self.assertIn("inkling2-mini", PARAM_MAPPING)
    self.assertIn("inkling2-mini", HOOK_FNS)
    self.assertIn("inkling2-mini", to_maxtext._FULL_PRECISION_SUFFIXES_BY_MODEL)  # pylint: disable=protected-access

    # HF_SHAPE accepts PreTrainedConfig, flat dict, and nested {"text_config": ...} dict identically.
    shapes_from_obj = INKLING2_HF_SHAPE(HF_MODEL_CONFIGS["inkling2-mini"])
    shapes_from_flat = INKLING2_HF_SHAPE(inkling2_mini_dict)
    shapes_from_nested = INKLING2_HF_SHAPE({"text_config": inkling2_mini_dict})
    self.assertEqual(shapes_from_obj, shapes_from_flat)
    self.assertEqual(shapes_from_obj, shapes_from_nested)

    # 24L Inkling 2 Mini: 694 unquantized string tensor keys (excl. composite tuple keys).
    string_keys = [k for k in shapes_from_obj if isinstance(k, str)]
    self.assertLen(string_keys, 694)
    self.assertEqual(shapes_from_obj["model.llm.attn_res.attn_queries.0"], [2304])
    self.assertEqual(shapes_from_obj["model.llm.attn_res.attn_queries.4"], [3, 2304])
    self.assertEqual(shapes_from_obj["model.llm.attn_res.attn_queries.22"], [3, 2304])
    self.assertNotIn("model.llm.attn_res.attn_queries.23", shapes_from_obj)
    self.assertEqual(shapes_from_obj["model.llm.middle_cycle_cross_pass_queries"], [24, 2304])
    self.assertEqual(shapes_from_obj["model.llm.middle_cycle_cross_pass_kv_queries"], [24, 2304])

  def test_nvfp4_mxfp8_dequant_and_w13_deinterleave_exact(self):
    rng = np.random.default_rng(42)
    # Non-multiple of _INKLING2_EXPERTS_PER_CHUNK (8) to exercise partial tail chunking.
    num_experts, rows, cols = 11, 12, 64

    # 1. NVFP4 covering all 16 E2M1 codes in both low and high nibbles.
    codes = rng.integers(0, 256, size=(num_experts, rows, cols // 2), dtype=np.uint8)
    codes[0, 0, :16] = np.arange(16, dtype=np.uint8)
    codes[0, 1, :16] = (np.arange(16, dtype=np.uint8) << 4) | np.arange(15, -1, -1, dtype=np.uint8)
    scale_fp8 = rng.uniform(0.125, 4.0, size=(num_experts, rows, cols // 16)).astype(ml_dtypes.float8_e4m3fn)
    amax = rng.uniform(0.5, 5.0, size=(num_experts,)).astype(np.float32)
    denom = np.array([2688.0], dtype=np.float32)

    got_nvfp4 = _dequantize_inkling2_experts(codes, scale_fp8, amax, denom)
    # Independent reference implementation
    lo = _E2M1_REF[codes & 0xF]
    hi = _E2M1_REF[codes >> 4]
    ref_vals = np.stack([lo, hi], axis=-1).reshape(num_experts, rows, cols // 16, 16)
    ref_nvfp4 = (ref_vals * scale_fp8.astype(np.float32)[..., None] * (amax / denom)[:, None, None, None]).reshape(
        num_experts, rows, cols
    )
    np.testing.assert_array_equal(got_nvfp4, ref_nvfp4)

    # 2. MXFP8 reference verification.
    w_fp8 = rng.uniform(-3.0, 3.0, size=(num_experts, rows, cols)).astype(ml_dtypes.float8_e4m3fn)
    scale_u8 = rng.integers(120, 135, size=(num_experts, rows, cols // 32), dtype=np.uint8)
    got_mxfp8 = _dequantize_inkling2_experts(w_fp8, scale_u8, None, None)
    ref_mxfp8 = (
        w_fp8.astype(np.float32).reshape(num_experts, rows, cols // 32, 32)
        * np.exp2(scale_u8.astype(np.float32) - 127.0)[..., None]
    ).reshape(num_experts, rows, cols)
    np.testing.assert_array_equal(got_mxfp8, ref_mxfp8)

    # 3. Unquantized pass-through preserves dtype & exact values.
    w_bf16 = rng.standard_normal((num_experts, rows, cols)).astype(ml_dtypes.bfloat16)
    got_raw = _dequantize_inkling2_experts(w_bf16, None, None, None)
    self.assertEqual(got_raw.dtype, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(got_raw, w_bf16)

    # 4. w13 de-interleave [0::2] -> wi_0, [1::2] -> wi_1 and re-interleave round-trip.
    hf_cfg = _synthetic_hf_config(num_layers=6, block_size=2, core_start=2, core_end=4, num_experts=num_experts)
    hooks_to_mt = HOOK_FNS["inkling2-mini"](hf_cfg, None, scan_layers=False, saving_to_hf=False)
    hooks_to_hf = HOOK_FNS["inkling2-mini"](hf_cfg, None, scan_layers=False, saving_to_hf=True)
    w13_key = ("params-decoder-layers_0-mlp-wi_0", "params-decoder-layers_0-mlp-wi_1")
    stacked = hooks_to_mt[w13_key]((codes, scale_fp8, amax, denom), None)
    wi_0, wi_1 = stacked[..., 0], stacked[..., 1]
    np.testing.assert_array_equal(wi_0, ref_nvfp4[:, 0::2, :].transpose(0, 2, 1))
    np.testing.assert_array_equal(wi_1, ref_nvfp4[:, 1::2, :].transpose(0, 2, 1))
    # Confirm contiguous split would NOT match interleaved layout.
    self.assertFalse(np.array_equal(wi_0, ref_nvfp4[:, : rows // 2, :].transpose(0, 2, 1)))
    (reconstructed_w13,) = hooks_to_hf[w13_key]([wi_0, wi_1], None)
    np.testing.assert_array_equal(reconstructed_w13, ref_nvfp4)

  @parameterized.named_parameters(
      ("unscanned_kcore2", False, 1, (12, 2, 4, 8)),
      ("scanned_axis1_kcore2", True, 1, (12, 2, 4, 8)),
      ("scanned_axis0_kcore2", True, 0, (12, 2, 4, 8)),
      ("scanned_axis1_kcore1", True, 1, (6, 2, 2, 4)),
  )
  def test_key_bijection_and_lossless_roundtrip(self, scan_layers, param_scan_axis, layout_spec):
    num_layers, block_size, core_start, core_end = layout_spec
    hf_cfg = _synthetic_hf_config(num_layers=num_layers, block_size=block_size, core_start=core_start, core_end=core_end)
    mt_cfg = _synthetic_maxtext_config(hf_cfg, scan_layers=scan_layers, param_scan_axis=param_scan_axis)

    # 1. 100% key bijection against the actual Inkling2 MaxText abstract parameter tree.
    abstract_dict, _ = to_maxtext.get_maxtext_model_info(mt_cfg)
    param_map = PARAM_MAPPING["inkling2-mini"](hf_cfg, mt_cfg, scan_layers=scan_layers)
    hooks_to_mt = HOOK_FNS["inkling2-mini"](hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=False)
    hooks_to_hf = HOOK_FNS["inkling2-mini"](hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=True)

    mapped_flat = _flatten_mt_keys(param_map.keys())
    self.assertEqual(len(mapped_flat), len(set(mapped_flat)), "Duplicate MaxText keys in mapping")
    self.assertEqual(sorted(set(abstract_dict) - set(mapped_flat)), [], "Model params missing from mapping")
    self.assertEqual(sorted(set(mapped_flat) - set(abstract_dict)), [], "Extra mapping keys not in model")
    self.assertEqual(sorted(set(hooks_to_mt) - set(param_map)), [], "HF->MT hooks for unmapped keys")
    self.assertEqual(sorted(set(hooks_to_hf) - set(param_map)), [], "MT->HF hooks for unmapped keys")

    # Layer 0 has no attention_input; layers >= 1 map to attn_queries.{i - 1}.
    self.assertNotIn("params-decoder-layers_0-attention_input-query", param_map)
    self.assertEqual(param_map["params-decoder-layers_1-attention_input-query"], "model.llm.attn_res.attn_queries.0")
    self.assertEqual(param_map["params-decoder-layers_0-mlp_input-query"], "model.llm.attn_res.mlp_queries.0")

    # Every primary HF tensor is mapped exactly once.
    primary_hf = [k for v in param_map.values() for k in _flatten_hf_primary_keys(v)]
    self.assertEqual(len(primary_hf), len(set(primary_hf)), "Duplicate HF primary tensor in mapping")
    expected_hf = {k for k in INKLING2_HF_SHAPE(hf_cfg) if isinstance(k, str)}
    self.assertEqual(set(primary_hf), expected_hf)

    # 2. Mixed NVFP4 + MXFP8 HF -> MaxText (eager & lazy agree bit-for-bit).
    quant_hf_state = _make_synthetic_hf_state(hf_cfg, quantized=True, seed=7)
    mt_eager = _convert_hf_state_in_memory(quant_hf_state, hf_cfg, mt_cfg, abstract_dict, lazy=False)
    mt_lazy = _convert_hf_state_in_memory(quant_hf_state, hf_cfg, mt_cfg, abstract_dict, lazy=True)
    self.assertEqual(set(mt_eager), set(abstract_dict))
    for k, (_, expected_shape) in abstract_dict.items():
      self.assertEqual(mt_eager[k].shape, expected_shape, k)
      np.testing.assert_array_equal(mt_eager[k], mt_lazy[k], err_msg=f"eager vs lazy mismatch on {k}")

    # 3. Lossless MT -> HF -> MT round-trip (max_abs_diff == 0.0).
    exported_hf = _export_mt_state_in_memory(mt_eager, hf_cfg, mt_cfg)
    self.assertEqual(set(exported_hf), expected_hf)
    mt_roundtrip_eager = _convert_hf_state_in_memory(exported_hf, hf_cfg, mt_cfg, abstract_dict, lazy=False)
    mt_roundtrip_lazy = _convert_hf_state_in_memory(exported_hf, hf_cfg, mt_cfg, abstract_dict, lazy=True)
    for k in abstract_dict:
      np.testing.assert_array_equal(mt_roundtrip_eager[k], mt_eager[k], err_msg=f"eager round-trip mismatch on {k}")
      np.testing.assert_array_equal(mt_roundtrip_lazy[k], mt_eager[k], err_msg=f"lazy round-trip mismatch on {k}")

  @parameterized.named_parameters(("axis1", 1), ("axis0", 0))
  def test_scanned_conversion_matches_restacked_unscanned(self, param_scan_axis):
    hf_cfg = _synthetic_hf_config(num_layers=12, block_size=2, core_start=4, core_end=8)
    mt_unscanned_cfg = _synthetic_maxtext_config(hf_cfg, scan_layers=False, param_scan_axis=param_scan_axis)
    mt_scanned_cfg = _synthetic_maxtext_config(hf_cfg, scan_layers=True, param_scan_axis=param_scan_axis)
    abs_unscanned, _ = to_maxtext.get_maxtext_model_info(mt_unscanned_cfg)
    abs_scanned, _ = to_maxtext.get_maxtext_model_info(mt_scanned_cfg)

    quant_hf = _make_synthetic_hf_state(hf_cfg, quantized=True, seed=11)
    unscanned = _convert_hf_state_in_memory(quant_hf, hf_cfg, mt_unscanned_cfg, abs_unscanned, lazy=False)
    scanned = _convert_hf_state_in_memory(quant_hf, hf_cfg, mt_scanned_cfg, abs_scanned, lazy=True)

    checked = set()
    for key, val in scanned.items():
      if "-scanned_blocks-sub_layer_" not in key:
        np.testing.assert_array_equal(val, unscanned[key], err_msg=key)
        checked.add(key)
        continue
      prefix, rest = key.split("-scanned_blocks-sub_layer_", 1)
      sub_str, suffix = rest.split("-", 1)
      sub_idx = int(sub_str)
      axis = single_stack_axis(key, mt_scanned_cfg, val.shape)
      self.assertEqual(val.shape[axis], 2, key)
      for b in range(2):
        phys_layer = 4 + b * 2 + sub_idx
        ref_key = f"{prefix}-layers_{phys_layer}-{suffix}"
        np.testing.assert_array_equal(np.take(val, b, axis=axis), unscanned[ref_key], err_msg=f"{key}[{b}]")
        checked.add(ref_key)
    self.assertEqual(checked, set(unscanned))

  @parameterized.named_parameters(("unscanned", False), ("scanned", True))
  def test_strict_fp32_preservation_under_bfloat16(self, scan_layers):
    hf_cfg = _synthetic_hf_config(num_layers=12, block_size=2, core_start=4, core_end=8)
    mt_cfg = _synthetic_maxtext_config(hf_cfg, scan_layers=scan_layers, weight_dtype="bfloat16")
    abstract_dict, _ = to_maxtext.get_maxtext_model_info(mt_cfg)
    param_map = PARAM_MAPPING["inkling2-mini"](hf_cfg, mt_cfg, scan_layers=scan_layers)

    expected_fp32_suffixes = (
        "-mlp-gate-kernel",
        "-mlp-gate-bias",
        "-post_attention_convolution-kernel",
        "-post_mlp_convolution-kernel",
        "-k_sconv-kernel",
        "-v_sconv-kernel",
        "-attention_input-query",
        "-mlp_input-query",
        "-final_input-query",
        "-message_input-query",
        "-forget_gate_proj-kernel",
        "-message_gate_proj-kernel",
    )

    for mt_key in param_map:
      got_dtype = to_maxtext._param_save_dtype(mt_key, "bfloat16", "inkling2-mini")  # pylint: disable=protected-access
      leaf_keys = mt_key if isinstance(mt_key, tuple) else (mt_key,)
      should_be_fp32 = any(k.endswith(expected_fp32_suffixes) for k in leaf_keys)
      self.assertEqual(got_dtype, "float32" if should_be_fp32 else "bfloat16", f"Wrong dtype for {mt_key}")

    # Verify materialized dtypes from conversion match _param_save_dtype.
    quant_hf = _make_synthetic_hf_state(hf_cfg, quantized=True, seed=13)
    converted = _convert_hf_state_in_memory(quant_hf, hf_cfg, mt_cfg, abstract_dict, save_dtype="bfloat16", lazy=True)
    for k, arr in converted.items():
      if k.endswith(expected_fp32_suffixes):
        self.assertEqual(arr.dtype, np.float32, f"{k} must stay float32")
      else:
        self.assertEqual(arr.dtype, ml_dtypes.bfloat16, f"{k} must be bfloat16")

  def test_to_huggingface_validate_or_update_architecture(self):
    # Full Inkling 2 Mini config passes strict validation without override.
    full_mt_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="inkling2-mini",
        max_target_length=16,
        per_device_batch_size=1,
        enable_checkpointing=False,
    )
    hf_cfg_copy = copy.deepcopy(HF_MODEL_CONFIGS["inkling2-mini"])
    to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
        hf_cfg_copy, full_mt_cfg, override=False
    )

    # Reduced config fails without override, and updates all Inkling2 fields cleanly with override=True.
    small_hf_dict = _synthetic_hf_config(num_layers=12, block_size=2, core_start=4, core_end=8)
    small_mt_cfg = _synthetic_maxtext_config(small_hf_dict, scan_layers=False)
    hf_cfg_override = copy.deepcopy(HF_MODEL_CONFIGS["inkling2-mini"])
    with self.assertRaises(ValueError):
      to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
          hf_cfg_override, small_mt_cfg, override=False
      )
    to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
        hf_cfg_override, small_mt_cfg, override=True
    )
    self.assertEqual(hf_cfg_override.num_hidden_layers, 12)
    self.assertEqual(hf_cfg_override.n_layers_per_attn_res_block, 2)
    self.assertEqual(hf_cfg_override.loop_layer_index_start, 4)
    self.assertEqual(hf_cfg_override.loop_layer_index_end, 8)
    self.assertEqual(hf_cfg_override.layer_types, ["hybrid_sliding", "hybrid"] * 6)
    self.assertEqual(hf_cfg_override.d_rel, 8)
    self.assertEqual(hf_cfg_override.rel_extent, 16)

    # Custom positive final_logits_soft_cap propagates under override=True.
    custom_cap_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="inkling2-mini",
        override_model_config=True,
        final_logits_soft_cap=15.0,
        max_target_length=16,
        per_device_batch_size=1,
        enable_checkpointing=False,
    )
    hf_cfg_cap = copy.deepcopy(HF_MODEL_CONFIGS["inkling2-mini"])
    to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
        hf_cfg_cap, custom_cap_cfg, override=True
    )
    self.assertEqual(hf_cfg_cap.final_logit_softcapping, 15.0)

    # final_logits_soft_cap=0.0 (normalized to None by pyconfig) raises a clear ValueError
    # rather than TypeError from float(None) or StrictDataclassFieldValidationError / NaN in HF.
    uncapped_mt_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="inkling2-mini",
        override_model_config=True,
        final_logits_soft_cap=0.0,
        max_target_length=16,
        per_device_batch_size=1,
        enable_checkpointing=False,
    )
    self.assertIsNone(uncapped_mt_cfg.final_logits_soft_cap)
    for override_flag in (False, True):
      hf_cfg_uncapped = copy.deepcopy(HF_MODEL_CONFIGS["inkling2-mini"])
      with self.assertRaisesRegex(ValueError, "final_logit_softcapping"):
        to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
            hf_cfg_uncapped, uncapped_mt_cfg, override=override_flag
        )

  def test_convert_tensor_to_numpy_preserves_integer_dtypes_under_bfloat16(self):
    # Integer / bool NumPy arrays must never be silently downcast to bfloat16.
    for arr in (
        np.array([0, 300, 100000], dtype=np.int32),
        np.array([-5, 500000], dtype=np.int64),
        np.arange(16, dtype=np.uint8),
        np.array([True, False, True], dtype=np.bool_),
    ):
      out = to_maxtext._convert_tensor_to_numpy(arr, "bfloat16")  # pylint: disable=protected-access
      self.assertEqual(out.dtype, arr.dtype, f"Integer/bool dtype {arr.dtype} must be preserved")
      np.testing.assert_array_equal(out, arr)

    # Floating-point and float8_e4m3fn NumPy arrays cast to bfloat16; bfloat16 stays bfloat16.
    for arr in (
        np.array([1.25, -2.5], dtype=np.float32),
        np.array([1.25, -2.5], dtype=np.float64),
        np.array([1.25, -2.5], dtype=np.float16),
        np.array([0.5, -1.5], dtype=ml_dtypes.float8_e4m3fn),
        np.array([1.25, -2.5], dtype=ml_dtypes.bfloat16),
    ):
      out = to_maxtext._convert_tensor_to_numpy(arr, "bfloat16")  # pylint: disable=protected-access
      self.assertEqual(out.dtype, ml_dtypes.bfloat16, f"Float dtype {arr.dtype} must become bfloat16")
      np.testing.assert_array_equal(out, arr.astype(np.float32).astype(ml_dtypes.bfloat16))

  def test_to_maxtext_rejects_transformers_eager_backend(self):
    with self.assertRaisesRegex(ValueError, "eager_load_method='safetensors'"):
      to_maxtext.main(
          args=[
              "to_maxtext.py",
              get_test_config_path(),
              "model_name=inkling2-mini",
              "hardware=cpu",
              "skip_jax_distributed_system=True",
              "max_target_length=16",
              "per_device_batch_size=1",
          ],
          lazy_load_tensors=False,
          eager_load_method="transformers",
          hf_model_path="/nonexistent",
      )


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for torch tensor & safetensors loader tests")
class Inkling2TorchConversionTest(parameterized.TestCase):
  """Tests requiring PyTorch (raw dtype pass-through, LazyHFLoader, and tiny-inkling2-mini parity)."""

  def test_convert_tensor_to_numpy_raw_and_uint8_preservation(self):
    all_bytes = torch.arange(256, dtype=torch.uint8)
    raw_u8 = to_maxtext._convert_tensor_to_numpy(all_bytes, "raw")  # pylint: disable=protected-access
    bf16_u8 = to_maxtext._convert_tensor_to_numpy(all_bytes, "bfloat16")  # pylint: disable=protected-access
    self.assertEqual(raw_u8.dtype, np.uint8)
    self.assertEqual(bf16_u8.dtype, np.uint8)
    np.testing.assert_array_equal(raw_u8, np.arange(256, dtype=np.uint8))
    np.testing.assert_array_equal(bf16_u8, np.arange(256, dtype=np.uint8))

    fp8_t = torch.tensor([0.5, -1.5, 4.0, 240.0], dtype=torch.float8_e4m3fn)
    raw_fp8 = to_maxtext._convert_tensor_to_numpy(fp8_t, "raw")  # pylint: disable=protected-access
    self.assertEqual(raw_fp8.dtype, ml_dtypes.float8_e4m3fn)
    np.testing.assert_array_equal(raw_fp8.astype(np.float32), fp8_t.float().numpy())

    bf16_t = torch.tensor([1.0 + 2**-7, -3.25], dtype=torch.bfloat16)
    raw_bf16 = to_maxtext._convert_tensor_to_numpy(bf16_t, "raw")  # pylint: disable=protected-access
    self.assertEqual(raw_bf16.dtype, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(raw_bf16.astype(np.float32), bf16_t.float().numpy())

    f32_t = torch.tensor([1.2345679, -9.876543], dtype=torch.float32)
    raw_f32 = to_maxtext._convert_tensor_to_numpy(f32_t, "raw")  # pylint: disable=protected-access
    self.assertEqual(raw_f32.dtype, np.float32)
    np.testing.assert_array_equal(raw_f32, f32_t.numpy())

  @parameterized.named_parameters(("single_file", False), ("indexed", True))
  def test_lazy_hf_loader_optional_keys(self, indexed):
    with tempfile.TemporaryDirectory() as tmpdir:
      tensors = {
          "model.llm.embed.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
          "model.llm.layers.0.mlp.experts.w2_weight": np.arange(256, dtype=np.uint8).reshape(2, 8, 16),
      }
      safetensors_save_file(tensors, os.path.join(tmpdir, "model.safetensors"))
      if indexed:
        with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
          json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, f)

      loader = to_maxtext.LazyHFLoader(tmpdir, token=None, save_dtype="bfloat16")
      got_u8 = loader.get_tensor("model.llm.layers.0.mlp.experts.w2_weight", dtype="raw")
      self.assertEqual(got_u8.dtype, np.uint8)
      np.testing.assert_array_equal(got_u8, tensors["model.llm.layers.0.mlp.experts.w2_weight"])
      self.assertIsNone(loader.get_tensor(OptionalHFKey("model.llm.layers.0.mlp.experts.w2_weight_scale"), dtype="raw"))
      with self.assertRaises(ValueError):
        loader.get_tensor("model.llm.does_not_exist")

  @unittest.skipUnless(
      HAS_INKLING2_HF and HAS_TINY_INKLING2_CKPT,
      "Requires transformers with Inkling2LoopForCausalLM and staged tiny-inkling2-mini checkpoint",
  )
  @parameterized.named_parameters(("fp32", "float32"), ("bf16", "bfloat16"))
  def test_tiny_inkling2_real_dequant_matches_hf_bit_exact(self, save_dtype):
    with open(os.path.join(_TINY_INKLING2_DIR, "config.json"), "r", encoding="utf-8") as f:
      tiny_hf_cfg = json.load(f)
    tcfg = tiny_hf_cfg["text_config"]
    tiny_mt_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="inkling2-mini",
        override_model_config=True,
        base_emb_dim=tcfg["hidden_size"],
        base_num_query_heads=tcfg["num_attention_heads"],
        base_num_kv_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        base_mlp_dim=tcfg["intermediate_size"],
        base_moe_mlp_dim=tcfg["intermediate_size"],
        base_num_decoder_layers=tcfg["num_hidden_layers"],
        vocab_size=tcfg["vocab_size"],
        unpadded_vocab_size=tcfg["unpadded_vocab_size"],
        attn_res_block_size=tcfg["n_layers_per_attn_res_block"],
        inkling2_core_start=tcfg["loop_config"]["core_start"],
        inkling2_core_end=tcfg["loop_config"]["core_end"],
        inkling2_num_passes=tcfg["loop_config"]["num_passes"],
        inkling2_message_gate_rank=tcfg["loop_config"]["gate_rank"],
        inkling2_d_rel=tcfg["d_rel"],
        inkling2_rel_extent=tcfg["rel_extent"],
        sliding_window_size=tcfg["sliding_window_size"],
        inkling2_log_scaling_n_floor=float(tcfg["log_scaling_n_floor"]),
        short_conv_kernel_size=tcfg["sconv_kernel_size"],
        num_experts=tcfg["n_routed_experts"],
        num_experts_per_tok=tcfg["num_experts_per_tok"],
        max_target_length=16,
        per_device_batch_size=1,
        dtype="float32",
        weight_dtype=save_dtype,
        enable_checkpointing=False,
        scan_layers=False,
    )

    # Load HF in FP32 so `amax` / `outer_scale_denominator` retain full FP32 precision
    # (in BF16 mode, transformers' generic loader pre-casts FP32 `amax` to BF16 before dequant,
    # whereas `to_maxtext` keeps raw FP32 `amax` and casts only the final dequantized weight).
    hf_model = transformers.Inkling2LoopForCausalLM.from_pretrained(_TINY_INKLING2_DIR, dtype=torch.float32)
    hf_sd = hf_model.state_dict()

    loader = to_maxtext.LazyHFLoader(_TINY_INKLING2_DIR, token=None, save_dtype=save_dtype)
    param_map = PARAM_MAPPING["inkling2-mini"](tiny_hf_cfg, tiny_mt_cfg, scan_layers=False)
    hook_map = HOOK_FNS["inkling2-mini"](tiny_hf_cfg, tiny_mt_cfg, scan_layers=False, saving_to_hf=False)
    target_np_dtype = to_maxtext._target_numpy_dtype(save_dtype)  # pylint: disable=protected-access

    # Verify message_input.query merge bit-exactness.
    msg_keys = param_map["params-decoder-message_input-query"]
    msg_mt = hook_map["params-decoder-message_input-query"](
        tuple(loader.get_tensor(k, dtype="float32") for k in msg_keys), None
    )
    np.testing.assert_array_equal(msg_mt, hf_sd["model.message_input.query"].float().numpy())

    # Verify ALL 18 layers (NVFP4 L0..13 AND MXFP8 L14..17) experts match HF bit-for-bit!
    for i in range(tcfg["num_hidden_layers"]):
      w13_mt_key = (f"params-decoder-layers_{i}-mlp-wi_0", f"params-decoder-layers_{i}-mlp-wi_1")
      w13_raw = tuple(loader.get_tensor(k, dtype="raw") for k in param_map[w13_mt_key])
      stacked = hook_map[w13_mt_key](w13_raw, None).astype(target_np_dtype)
      wi_0, wi_1 = stacked[..., 0], stacked[..., 1]

      hf_gate_up = hf_sd[f"model.layers.{i}.mlp.experts.gate_up_proj"]
      hf_gate, hf_up = hf_gate_up.chunk(2, dim=1)
      hf_gate_np = to_maxtext._convert_tensor_to_numpy(hf_gate, save_dtype).swapaxes(-1, -2)  # pylint: disable=protected-access
      hf_up_np = to_maxtext._convert_tensor_to_numpy(hf_up, save_dtype).swapaxes(-1, -2)  # pylint: disable=protected-access
      np.testing.assert_array_equal(wi_0, hf_gate_np, err_msg=f"Layer {i} wi_0 mismatch ({save_dtype})")
      np.testing.assert_array_equal(wi_1, hf_up_np, err_msg=f"Layer {i} wi_1 mismatch ({save_dtype})")

      w2_mt_key = f"params-decoder-layers_{i}-mlp-wo"
      w2_raw = tuple(loader.get_tensor(k, dtype="raw") for k in param_map[w2_mt_key])
      wo = hook_map[w2_mt_key](w2_raw, None).astype(target_np_dtype)
      hf_down = hf_sd[f"model.layers.{i}.mlp.experts.down_proj"]
      hf_down_np = to_maxtext._convert_tensor_to_numpy(hf_down, save_dtype).swapaxes(-1, -2)  # pylint: disable=protected-access
      np.testing.assert_array_equal(wo, hf_down_np, err_msg=f"Layer {i} wo mismatch ({save_dtype})")

  @unittest.skipUnless(
      HAS_INKLING2_HF and HAS_TINY_INKLING2_CKPT,
      "Requires transformers with Inkling2LoopForCausalLM and staged tiny-inkling2-mini checkpoint",
  )
  @parameterized.named_parameters(
      ("lazy_unscanned", True, False),
      ("lazy_scanned", True, True),
      ("eager_unscanned", False, False),
  )
  def test_end_to_end_to_maxtext_orbax_and_hf_reload(self, lazy_load_tensors, scan_layers):
    with open(os.path.join(_TINY_INKLING2_DIR, "config.json"), "r", encoding="utf-8") as f:
      tiny_hf_cfg = json.load(f)
    tcfg = tiny_hf_cfg["text_config"]

    with tempfile.TemporaryDirectory() as tmpdir:
      orbax_dir = os.path.join(tmpdir, "mt_ckpt")
      hf_export_dir = os.path.join(tmpdir, "hf_export")

      cli_args = [
          "to_maxtext.py",
          get_test_config_path(),
          "model_name=inkling2-mini",
          "override_model_config=True",
          f"base_emb_dim={tcfg['hidden_size']}",
          f"base_num_query_heads={tcfg['num_attention_heads']}",
          f"base_num_kv_heads={tcfg['num_key_value_heads']}",
          f"head_dim={tcfg['head_dim']}",
          f"base_mlp_dim={tcfg['intermediate_size']}",
          f"base_moe_mlp_dim={tcfg['intermediate_size']}",
          f"base_num_decoder_layers={tcfg['num_hidden_layers']}",
          f"vocab_size={tcfg['vocab_size']}",
          f"unpadded_vocab_size={tcfg['unpadded_vocab_size']}",
          f"attn_res_block_size={tcfg['n_layers_per_attn_res_block']}",
          f"inkling2_core_start={tcfg['loop_config']['core_start']}",
          f"inkling2_core_end={tcfg['loop_config']['core_end']}",
          f"inkling2_num_passes={tcfg['loop_config']['num_passes']}",
          f"inkling2_message_gate_rank={tcfg['loop_config']['gate_rank']}",
          f"inkling2_d_rel={tcfg['d_rel']}",
          f"inkling2_rel_extent={tcfg['rel_extent']}",
          f"sliding_window_size={tcfg['sliding_window_size']}",
          f"inkling2_log_scaling_n_floor={float(tcfg['log_scaling_n_floor'])}",
          f"short_conv_kernel_size={tcfg['sconv_kernel_size']}",
          f"num_experts={tcfg['n_routed_experts']}",
          f"num_experts_per_tok={tcfg['num_experts_per_tok']}",
          f"base_output_directory={orbax_dir}",
          f"scan_layers={scan_layers}",
          "hardware=cpu",
          "skip_jax_distributed_system=True",
          "max_target_length=16",
          "per_device_batch_size=1",
      ]

      to_maxtext.main(
          args=cli_args,
          lazy_load_tensors=lazy_load_tensors,
          eager_load_method="safetensors",
          hf_model_path=_TINY_INKLING2_DIR,
          save_dtype="bfloat16",
          simulated_cpu_devices_count=1,
      )

      items_dir = os.path.join(orbax_dir, "0", "items")
      self.assertTrue(os.path.isdir(items_dir), f"Expected Orbax items dir at {items_dir}")

      # Verify exact stored Orbax dtypes on disk (FP32 for strict-FP32 params & MoEBiasVar, BF16 for the rest).
      mt_cfg = pyconfig.initialize(
          cli_args,
          load_parameters_path=items_dir,
          weight_dtype="bfloat16",
          enable_checkpointing=False,
      )
      ckptr = ocp.Checkpointer(
          ocp.PyTreeCheckpointHandler(
              use_ocdbt=mt_cfg.checkpoint_storage_use_ocdbt,
              use_zarr3=mt_cfg.checkpoint_storage_use_zarr3,
          )
      )
      meta_tree = ckptr.metadata(epath.Path(items_dir)).item_metadata.tree["params"]
      for col_name, col_tree in meta_tree.items():
        for path_tuple, meta_leaf in jax.tree_util.tree_leaves_with_path(col_tree):
          mt_key = f"{col_name}-" + "-".join(conversion_utils.param_key_parts_from_path(path_tuple))
          expected = to_maxtext._param_save_dtype(mt_key, "bfloat16", "inkling2-mini")  # pylint: disable=protected-access
          self.assertEqual(str(meta_leaf.dtype), expected, f"Orbax stored dtype mismatch for {mt_key}")

      # Load restored Orbax checkpoint via to_huggingface helpers.
      ckpt_dict = conversion_utils.load_orbax_checkpoint(mt_cfg)
      mt_state = conversion_utils.detect_and_extract_checkpoint(ckpt_dict)

      # Verify MoEBiasVar collection survived Orbax save/restore.
      self.assertTrue(any(k.startswith("MoEBiasVar-") for k in mt_state))

      # Export to HF safetensors + config.json and verify AutoModelForCausalLM loads with zero missing/unexpected keys.
      hf_export_cfg = copy.deepcopy(HF_MODEL_CONFIGS["inkling2-mini"])
      to_huggingface._validate_or_update_architecture(  # pylint: disable=protected-access
          hf_export_cfg, mt_cfg, override=True
      )
      exported_weights = _export_mt_state_in_memory(mt_state, hf_export_cfg.to_dict(), mt_cfg)
      conversion_utils.save_model_files(
          exported_weights,
          hf_export_cfg,
          tokenizer=None,
          processor=None,
          output_dir=hf_export_dir,
      )

      reloaded, load_info = transformers.AutoModelForCausalLM.from_pretrained(
          hf_export_dir,
          output_loading_info=True,
          dtype=torch.bfloat16,
      )
      self.assertIsInstance(reloaded, transformers.Inkling2LoopForCausalLM)
      self.assertEqual(load_info["missing_keys"], set())
      self.assertEqual(load_info["unexpected_keys"], set())
      self.assertEqual(load_info["mismatched_keys"], set())


if __name__ == "__main__":
  unittest.main()
