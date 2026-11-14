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

"""Unit tests for the maxtext-logit-bisect skill utilities."""

import pathlib
import sys
import unittest
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BISECT_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "maxtext-logit-bisect" / "scripts"
sys.path.insert(0, str(BISECT_SCRIPTS))

import bisect_layer_divergence  # pylint: disable=wrong-import-position
import bisect_utils  # pylint: disable=wrong-import-position
import hf_cfg_audit  # pylint: disable=wrong-import-position


class _DummyLinearAdapter(bisect_utils.LayerAdapter):
  """Minimal custom adapter for testing pluggable LayerAdapter resolution."""

  def load_layer_weights(
      self, ckpt_items: str, layer: int, scanned: bool = False, scan_axis: int = 1
  ) -> dict[str, np.ndarray]:
    return {"scale": np.full((4,), float(layer + 1), dtype=np.float64)}

  def forward_layer_fp64(self, x: np.ndarray, weights: dict[str, np.ndarray], eps: float = 1e-6) -> np.ndarray:
    return x.astype(np.float64) * weights["scale"]


class MaxTextLogitBisectTest(unittest.TestCase):
  """Tests HF-to-MaxText config parity auditing and model-agnostic layer bisection."""

  def test_hf_cfg_audit_multimodal_and_autodiscover(self):
    hf_cfg = {
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "vocab_size": 128256,
            "sliding_window": 4096,
            "custom_shared_param": 42,
            "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
        }
    }
    mt_cfg = {
        "base_emb_dim": 4096,
        "base_num_decoder_layers": 32,
        "base_num_query_heads": 32,
        "base_num_kv_heads": 8,
        "base_mlp_dim": 14336,
        "normalization_layer_epsilon": 1e-5,
        "rope_max_timescale": 500000.0,
        "vocab_size": 128256,
        "sliding_window_size": 4096,
        "custom_shared_param": 42,
        "rope_factor": 8.0,
        "rope_low_freq_factor": 1.0,
        "rope_high_freq_factor": 4.0,
    }
    matches, mismatches = hf_cfg_audit.audit_configs(hf_cfg, mt_cfg)
    self.assertEqual(len(mismatches), 0)
    self.assertGreaterEqual(len(matches), 12)

    mt_cfg["rope_max_timescale"] = 10000.0
    _, mismatches_bad = hf_cfg_audit.audit_configs(hf_cfg, mt_cfg)
    self.assertEqual(len(mismatches_bad), 1)
    self.assertEqual(mismatches_bad[0][0], "rope_theta")

  def test_model_agnostic_activation_bisection(self):
    rng = np.random.default_rng(0)
    l0_in = rng.standard_normal((2, 8, 16))
    l0_out = l0_in * 1.5
    l1_in = l0_out
    l1_out_hf = l1_in * 2.0
    l1_out_mt = l1_out_hf.copy()
    l1_out_mt[:, 5, :] += 10.0  # Inject spike at layer 1, token position 5

    hf_npz = {
        "L00_in": l0_in,
        "L00_out": l0_out,
        "L01_in": l1_in,
        "L01_out": l1_out_hf,
        "L01_attn": l1_in,
    }
    mt_npz = {
        "L00_in": l0_in,
        "L00_out": l0_out,
        "L01_in": l1_in,
        "L01_out": l1_out_mt,
        "L01_attn": l1_in,
    }

    layers = bisect_layer_divergence.discover_layers(hf_npz, mt_npz, "L{layer:02d}_out")
    self.assertEqual(layers, [0, 1])

    rows, first_bad = bisect_layer_divergence.bisect_activations(
        hf_npz, mt_npz, layers, subkey_templates=["L{layer:02d}_attn"], rtol=1e-3
    )
    self.assertEqual(first_bad, 1)
    self.assertEqual(rows[0]["status"], "OK")
    self.assertEqual(rows[1]["status"], "DIVERGED")
    self.assertEqual(rows[1]["worst_pos"], 5)
    self.assertAlmostEqual(float(rows[1]["subkeys"]["L01_attn"]["rel_l2"]), 0.0, places=10)

    # Verify NaN activations are flagged as DIVERGED and not skipped (NaN > rtol is False in IEEE-754)
    l0_nan = l0_out.copy()
    l0_nan[0, 0, 0] = np.nan
    rows_nan, first_bad_nan = bisect_layer_divergence.bisect_activations(
        hf_npz,
        {"L00_in": l0_in, "L00_out": l0_nan, "L01_in": l1_in, "L01_out": l1_out_hf},
        layers,
        rtol=1e-3,
    )
    self.assertEqual(first_bad_nan, 0)
    self.assertEqual(rows_nan[0]["status"], "DIVERGED")

    # Verify tensor_diff_metrics rejects mismatched shapes instead of silently broadcasting
    with self.assertRaises(ValueError):
      bisect_utils.tensor_diff_metrics(np.ones((1, 8, 16)), np.ones((8, 16)))

  def test_fp64_primitives_and_adapters(self):
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    w = np.ones((4,), dtype=np.float32)
    normed = bisect_utils.rmsnorm(x, w, eps=1e-6)
    self.assertEqual(normed.dtype, np.float64)
    self.assertAlmostEqual(float(np.mean(normed * normed)), 1.0, places=5)

    normed_zc = bisect_utils.rmsnorm(x, np.zeros((4,), dtype=np.float32), eps=1e-6, zero_centered=True)
    np.testing.assert_allclose(normed, normed_zc, rtol=1e-12)

    adapter = bisect_utils.StandardSwiGLUMLPAdapter()
    weights = {
        "norm": np.ones((4,), dtype=np.float64),
        "wi_0": np.eye(4, dtype=np.float64),
        "wi_1": np.eye(4, dtype=np.float64),
        "wo": np.eye(4, dtype=np.float64),
    }
    out = adapter.forward_layer_fp64(x, weights, eps=1e-6)
    self.assertEqual(out.shape, (1, 4))

    custom = bisect_utils.resolve_adapter(f"{__name__}:_DummyLinearAdapter")
    w_custom = custom.load_layer_weights("/unused", layer=2, scanned=True)
    np.testing.assert_allclose(custom.forward_layer_fp64(x, w_custom), x.astype(np.float64) * 3.0)


if __name__ == "__main__":
  unittest.main()
