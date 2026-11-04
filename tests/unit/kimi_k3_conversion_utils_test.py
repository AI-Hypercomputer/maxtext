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

"""Unit tests for `tests/utils/kimi_k3_conversion_utils.py` on a tiny fake state dict (no HF reference)."""

import unittest

from flax import nnx
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from tests.utils import kimi_k3_conversion_utils as conv

# 2 layers: layer 0 KDA + dense MLP, layer 1 MLA + latent MoE (0-indexed `full_attn_layers`).
_HEADS, _HEAD_DIM, _PADDED = 2, 4, 4
_TINY_HF_CONFIG = {
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "num_experts": 3,
    "full_attn_layers": [1],
    "linear_attn_config": {"num_heads": _HEADS, "head_dim": _HEAD_DIM},
}
_KDA = "params-decoder-layers_0-self_attention-"


def _fake_kda_state_dict(dtype=np.float32):
  """A KDA submodule state dict: keys are suffixes of the released `language_model.` names."""
  rng = np.random.default_rng(0)
  a_log = np.zeros(_PADDED, np.float32)
  a_log[:_HEADS] = [0.5, 1.5]  # the release pads A_log to [head_dim] with zeros
  return {
      "q_proj.weight": rng.standard_normal((_HEADS * _HEAD_DIM, 5)).astype(dtype),
      "f_b_proj.weight": rng.standard_normal((_HEADS * _HEAD_DIM, _HEAD_DIM)).astype(dtype),
      "b_proj.weight": rng.standard_normal((_HEADS, 5)).astype(dtype),
      "q_conv1d.weight": rng.standard_normal((_HEADS * _HEAD_DIM, 1, 3)).astype(np.float32),
      "A_log": a_log,
  }


class FindStateDictKeyTest(unittest.TestCase):

  def test_exact_prefix_stripped_and_suffix_matches(self):
    sd = {"language_model.lm_head.weight": 0, "model.norm.weight": 0, "b_proj.weight": 0, "f_b_proj.weight": 0}
    self.assertEqual(conv._find_state_dict_key("language_model.lm_head.weight", sd), "language_model.lm_head.weight")  # pylint: disable=protected-access
    self.assertEqual(conv._find_state_dict_key("language_model.model.norm.weight", sd), "model.norm.weight")  # pylint: disable=protected-access
    key = "language_model.model.layers.0.self_attn.f_b_proj.weight"
    self.assertEqual(conv._find_state_dict_key(key, sd), "f_b_proj.weight")  # pylint: disable=protected-access
    key = "language_model.model.layers.0.self_attn.b_proj.weight"
    self.assertEqual(conv._find_state_dict_key(key, sd), "b_proj.weight")  # pylint: disable=protected-access
    self.assertIsNone(conv._find_state_dict_key("language_model.model.layers.0.self_attn.g_proj.weight", sd))  # pylint: disable=protected-access


class ToNumpyTest(unittest.TestCase):

  def test_numpy_dtype_is_preserved(self):
    for dtype in (np.float32, ml_dtypes.bfloat16, np.int32):
      self.assertEqual(conv._to_numpy(np.ones(3, dtype)).dtype, dtype)  # pylint: disable=protected-access

  def test_torch_bf16_is_not_upcast(self):
    try:
      import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("torch not installed")
    t = torch.tensor([1.0 + 2**-7, -3.25, 1e-3], dtype=torch.bfloat16)
    y = conv._to_numpy(t)  # pylint: disable=protected-access
    self.assertEqual(y.dtype, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(y.astype(np.float32), t.float().numpy())
    self.assertEqual(conv._to_numpy(torch.ones(2, dtype=torch.float32)).dtype, np.float32)  # pylint: disable=protected-access


class ConvertPytorchModuleTest(unittest.TestCase):

  def test_kda_submodule_conversion(self):
    sd = _fake_kda_state_dict()
    out = conv.convert_pytorch_module_to_maxtext_params(sd, _TINY_HF_CONFIG, prefix_filter=_KDA)
    self.assertEqual(
        set(out), {_KDA + k for k in ("q_proj-kernel", "f_b_proj-kernel", "b_proj-kernel", "q_conv1d-kernel", "A_log")}
    )
    np.testing.assert_array_equal(out[_KDA + "q_proj-kernel"], sd["q_proj.weight"].T)
    np.testing.assert_array_equal(out[_KDA + "q_conv1d-kernel"], sd["q_conv1d.weight"].transpose(2, 1, 0))
    # No target_shapes: A_log is still sliced to the KDA num_heads from the HF config.
    np.testing.assert_array_equal(out[_KDA + "A_log"], [0.5, 1.5])

  def test_bf16_inputs_stay_bf16(self):
    sd = _fake_kda_state_dict(ml_dtypes.bfloat16)
    out = conv.convert_pytorch_module_to_maxtext_params(sd, _TINY_HF_CONFIG, prefix_filter=_KDA)
    self.assertEqual(out[_KDA + "q_proj-kernel"].dtype, ml_dtypes.bfloat16)
    self.assertEqual(out[_KDA + "A_log"].dtype, np.float32)

  def test_target_shapes_are_forwarded_and_experts_stacked(self):
    rng = np.random.default_rng(1)
    prefix = "params-decoder-layers_1-mlp-routed_experts-wi_0"
    sd = {f"block_sparse_moe.experts.{e}.w1.weight": rng.standard_normal((6, 4)).astype(np.float32) for e in range(3)}
    sd["A_log"] = np.array([1.0, 2.0, 0.0], np.float32)
    out = conv.convert_pytorch_module_to_maxtext_params(
        sd, _TINY_HF_CONFIG, target_shapes={_KDA + "A_log": (2,)}, prefix_filter="params-decoder-layers_"
    )
    self.assertEqual(out[prefix].shape, (3, 4, 6))
    np.testing.assert_array_equal(out[prefix][2], sd["block_sparse_moe.experts.2.w1.weight"].T)
    np.testing.assert_array_equal(out[_KDA + "A_log"], [1.0, 2.0])


class _TinyNorm(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.zeros((3,), jnp.float32))


class LoadParamsIntoNnxTest(unittest.TestCase):

  def test_loads_and_strictness(self):
    module = _TinyNorm()
    loaded = conv.load_params_into_nnx(module, {"p-scale": np.array([1.0, 2.0, 3.0])}, "p")
    self.assertEqual(loaded, ["p-scale"])
    np.testing.assert_array_equal(np.asarray(module.scale[...]), [1.0, 2.0, 3.0])
    with self.assertRaises(KeyError):
      conv.load_params_into_nnx(_TinyNorm(), {}, "p")
    with self.assertRaises(KeyError):
      conv.load_params_into_nnx(_TinyNorm(), {"p-scale": np.zeros(3), "p-extra": np.zeros(1)}, "p")
    with self.assertRaises(ValueError):
      conv.load_params_into_nnx(_TinyNorm(), {"p-scale": np.zeros(4)}, "p")


if __name__ == "__main__":
  unittest.main()
