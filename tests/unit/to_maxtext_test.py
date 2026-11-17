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

"""Tests for to_maxtext.py"""

import types
import unittest
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import pytest

try:
  import torch
except ImportError:
  torch = None

pytestmark = [pytest.mark.decoupled_target]

from maxtext.checkpoint_conversion.to_maxtext import (
    _convert_tensor_to_numpy,
    _extract_conversion_args,
    _param_save_dtype,
    resolve_scale_key,
)
from maxtext.checkpoint_conversion.utils.utils import load_orbax_checkpoint, process_maxtext_param
from maxtext.configs.types import DType


class ResolveScaleKeyTest(unittest.TestCase):

  def test_exact_match(self):
    container = {"model.layers.0.mlp.gate.weight": 1}
    self.assertEqual(resolve_scale_key("model.layers.0.mlp.gate.weight", container), "model.layers.0.mlp.gate.weight")

  def test_weight_scale_to_weight_scale_inv_fallback(self):
    container = {"model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": 1}
    resolved = resolve_scale_key("model.layers.0.mlp.experts.0.gate_proj.weight_scale", container)
    self.assertEqual(resolved, "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")

  def test_unmatched_key_returns_original(self):
    container = {}
    self.assertEqual(
        resolve_scale_key("model.layers.0.mlp.experts.0.gate_proj.weight_scale", container),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
    )


class ConvertTensorToNumpyTest(unittest.TestCase):

  def test_torch_float8_e4m3fn_to_numpy(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    arr = _convert_tensor_to_numpy(t, save_dtype="float8_e4m3fn")
    self.assertEqual(arr.dtype, ml_dtypes.float8_e4m3fn)
    self.assertEqual(arr.shape, (3,))

  def test_torch_float8_e4m3fn_to_float32(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    arr = _convert_tensor_to_numpy(t, save_dtype="float32")
    self.assertEqual(arr.dtype, np.float32)

  def test_torch_bfloat16_to_numpy(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    arr = _convert_tensor_to_numpy(t, save_dtype="bfloat16")
    self.assertEqual(arr.dtype, ml_dtypes.bfloat16)

  def test_numpy_passthrough(self):
    arr_in = np.array([1.0, 2.0], dtype=np.float32)
    arr_out = _convert_tensor_to_numpy(arr_in, save_dtype="float8_e4m3fn")
    self.assertTrue(np.array_equal(arr_in, arr_out))


class ParamSaveDtypeTest(unittest.TestCase):
  """DeepSeek-V4 router bias and tid2eid must not be rounded to bf16."""

  def test_router_state_saved_as_float32(self):
    self.assertEqual(_param_save_dtype("MoEBiasVar-decoder-layers_3-mlp-MoeBlock_0-gate-bias", "bfloat16"), "float32")
    self.assertEqual(_param_save_dtype("Tid2EidVar-decoder-layers_0-mlp-MoeBlock_0-tid2eid", "bfloat16"), "float32")

  def test_other_params_keep_save_dtype(self):
    self.assertEqual(_param_save_dtype("params-decoder-layers_3-mlp-MoeBlock_0-gate-kernel", "bfloat16"), "bfloat16")
    self.assertEqual(_param_save_dtype(("params-a", "params-b"), "bfloat16"), "bfloat16")

  def test_model_specific_float32_params(self):
    muse_fp32_keys = (
        "params-decoder-scanned_blocks-layers_0-moe_block-gate-kernel",
        "params-decoder-scanned_blocks-layers_0-moe_block-gate-bias",
        "params-decoder-scanned_blocks-layers_0-post_attention_residual_gate-gate",
        "params-decoder-layers_0-post_feedforward_residual_gate-gate",
    )
    for key in muse_fp32_keys:
      self.assertEqual(_param_save_dtype(key, "bfloat16", "muse-spark-816b-a42b"), "float32")
      self.assertEqual(_param_save_dtype(key, DType.BFLOAT16, "muse-spark-816b-a42b"), "float32")
      self.assertEqual(_param_save_dtype(key, jnp.bfloat16, "muse-spark-816b-a42b"), "float32")
      self.assertEqual(_param_save_dtype(key, "bfloat16", "qwen3-30b-a3b"), "bfloat16")
      self.assertEqual(_param_save_dtype(key, "bfloat16"), "bfloat16")

  def test_process_maxtext_param_preserves_full_precision_keys(self):
    arr = np.array([1.1234567, -0.9876543], dtype=np.float32)
    gate_key = "params-decoder-layers_0-post_attention_residual_gate-gate"
    proj_key = "params-decoder-layers_0-self_attention-query-kernel"
    param_map = {gate_key: "hf.gate", proj_key: "hf.proj"}
    hf_shape_map = {"hf.gate": [2], "hf.proj": [2]}
    cfg = types.SimpleNamespace(weight_dtype=DType.BFLOAT16, model_name="muse-spark-816b-a42b")
    gate_outputs = process_maxtext_param(gate_key, arr, param_map, {}, hf_shape_map, cfg)
    proj_outputs = process_maxtext_param(proj_key, arr, param_map, {}, hf_shape_map, cfg)
    self.assertEqual(len(gate_outputs), 1)
    self.assertEqual(len(proj_outputs), 1)
    gate_out, proj_out = gate_outputs[0][1], proj_outputs[0][1]
    self.assertEqual(gate_out.dtype, np.float32)
    np.testing.assert_array_equal(gate_out, arr)
    self.assertEqual(proj_out.dtype, ml_dtypes.bfloat16)

  def test_load_orbax_checkpoint_preserves_full_precision_keys(self):
    import tempfile  # pylint: disable=import-outside-toplevel

    fp32_vals = np.array([47.3456789, -16.061234], dtype=np.float32)
    ckpt_tree = {
        "params": {
            "params": {
                "decoder": {
                    "layers_0": {
                        "post_attention_residual_gate": {"gate": fp32_vals},
                        "moe_block": {"gate": {"kernel": fp32_vals}},
                        "self_attention": {"query": {"kernel": fp32_vals}},
                    }
                }
            },
            "MoEBiasVar": {"decoder": {"layers_0": {"moe_block": {"gate": {"bias": fp32_vals}}}}},
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = f"{tmpdir}/ckpt"
      ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
      ckptr.save(ckpt_dir, args=ocp.args.PyTreeSave(ckpt_tree))
      cfg = types.SimpleNamespace(
          load_parameters_path=ckpt_dir,
          lora=types.SimpleNamespace(lora_restore_path=""),
          checkpoint_storage_concurrent_gb=1,
          checkpoint_storage_use_ocdbt=True,
          checkpoint_storage_use_zarr3=False,
          weight_dtype=jnp.bfloat16,
          model_name="muse-spark-816b-a42b",
      )
      restored = load_orbax_checkpoint(cfg)
    layers_0 = restored["params"]["params"]["decoder"]["layers_0"]
    bias = restored["params"]["MoEBiasVar"]["decoder"]["layers_0"]["moe_block"]["gate"]["bias"]
    self.assertEqual(layers_0["post_attention_residual_gate"]["gate"].dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(layers_0["post_attention_residual_gate"]["gate"]), fp32_vals)
    self.assertEqual(layers_0["moe_block"]["gate"]["kernel"].dtype, jnp.float32)
    self.assertEqual(bias.dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(bias), fp32_vals)
    self.assertEqual(layers_0["self_attention"]["query"]["kernel"].dtype, jnp.bfloat16)

  def test_non_bf16_save_dtype_unchanged(self):
    self.assertEqual(_param_save_dtype("MoEBiasVar-x", "float8_e4m3fn"), "float8_e4m3fn")
    self.assertEqual(_param_save_dtype("Tid2EidVar-x", "float32"), "float32")

  def test_float32_path_is_exact(self):
    if torch is None:
      self.skipTest("torch not available")
    ids = torch.tensor([255, 257, 259, 383], dtype=torch.int64)
    bias = torch.tensor([47.3456789, 16.06], dtype=torch.float32)
    np.testing.assert_array_equal(_convert_tensor_to_numpy(ids, "float32"), [255.0, 257.0, 259.0, 383.0])
    np.testing.assert_array_equal(_convert_tensor_to_numpy(bias, "float32"), bias.numpy())
    # bf16 rounds both, which is the defect the float32 override avoids.
    self.assertFalse(np.array_equal(_convert_tensor_to_numpy(ids, "bfloat16").astype(np.float32), ids.numpy()))
    self.assertFalse(np.array_equal(_convert_tensor_to_numpy(bias, "bfloat16").astype(np.float32), bias.numpy()))


class DeepSeekDequantizeTest(unittest.TestCase):
  """Dequantization keeps non-quantized tensors in their source dtype."""

  def test_non_quantized_tensors_keep_dtype(self):
    if torch is None:
      self.skipTest("torch not available")
    import json  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel
    from safetensors.torch import load_file, save_file  # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion.standalone_scripts import deepseek_dequantize  # pylint: disable=import-outside-toplevel

    tensors = {
        "layers.0.ffn.w1.weight": torch.ones(128, 128).to(torch.float8_e4m3fn),
        "layers.0.ffn.w1.scale": torch.full((1, 1), 2.0, dtype=torch.float32),
        "layers.0.attention_norm.weight": torch.ones(128, dtype=torch.float32),
        "layers.3.ffn.gate.bias": torch.tensor([47.3456789, 16.06], dtype=torch.float32),
        "layers.0.ffn.gate.tid2eid": torch.tensor([[255, 257], [259, 383]], dtype=torch.int64),
    }
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
      save_file(tensors, os.path.join(src, "model-00001.safetensors"))
      with open(os.path.join(src, "model.safetensors.index.json"), "w", encoding="utf8") as f:
        json.dump({"weight_map": {k: "model-00001.safetensors" for k in tensors}}, f)
      # convert_model sets the global torch default dtype.
      self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
      deepseek_dequantize.convert_model(src, dst)
      out = load_file(os.path.join(dst, "model-00001.safetensors"))
    self.assertEqual(out["layers.0.ffn.w1.weight"].dtype, torch.bfloat16)
    self.assertTrue(torch.all(out["layers.0.ffn.w1.weight"] == 2.0))
    self.assertNotIn("layers.0.ffn.w1.scale", out)
    self.assertEqual(out["layers.0.attention_norm.weight"].dtype, torch.bfloat16)
    self.assertTrue(torch.equal(out["layers.3.ffn.gate.bias"], tensors["layers.3.ffn.gate.bias"]))
    self.assertTrue(torch.equal(out["layers.0.ffn.gate.tid2eid"], tensors["layers.0.ffn.gate.tid2eid"]))


class ExtractConversionArgsTest(unittest.TestCase):

  def test_extract_mixed_flags(self):
    args = [
        "base.yml",
        "model_name=qwen3.5-35b-a3b-fp8",
        "save_dtype=float8_e4m3fn",
        "--lazy_load_tensors=false",
        "eager_load_method=safetensors",
        "--simulated_cpu_devices_count=4",
        "steps=10",
    ]
    extracted, remaining = _extract_conversion_args(args)
    self.assertEqual(extracted["save_dtype"], "float8_e4m3fn")
    self.assertEqual(extracted["lazy_load_tensors"], False)
    self.assertEqual(extracted["eager_load_method"], "safetensors")
    self.assertEqual(extracted["simulated_cpu_devices_count"], 4)
    self.assertEqual(remaining, ["base.yml", "model_name=qwen3.5-35b-a3b-fp8", "steps=10"])


if __name__ == "__main__":
  unittest.main()
