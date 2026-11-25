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

"""Unit tests for DeepSeek-V4 YaRN RoPE rescaling and per-layer theta selection."""

import math
from pathlib import Path
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
import yaml

try:
  import torch  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

from maxtext.configs.pyconfig import initialize
from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

# Mirrors the HF DeepSeek-V4-Flash config.json rope_scaling block, as pinned in
# src/maxtext/configs/models/deepseek4-284b.yml.
_HEAD_DIM = 512
_QK_ROPE_HEAD_DIM = 64
_PARTIAL_ROTARY_FACTOR = _QK_ROPE_HEAD_DIM / _HEAD_DIM
_COMPRESS_ROPE_THETA = 160000.0
_MAIN_ROPE_THETA = 10000.0
_ROPE_FACTOR = 16.0
_BETA_FAST = 32.0
_BETA_SLOW = 1.0
_ORIGINAL_MAX_POSITION_EMBEDDINGS = 65536

# YaRN correction range for the constants above, from
# find_correction_dim(beta) = dim * log(orig_max_pos / (beta * 2 * pi)) / (2 * log(theta))
# with dim = 64: floor(15.45) = 15 and ceil(24.71) = 25. Below `low` the ramp leaves the
# frequency untouched; at or above `high` it is fully divided by rope_factor.
_EXPECTED_LOW = 15
_EXPECTED_HIGH = 25


def _unscaled_inv_freq(rope_theta, dim):
  """Recomputes the pre-YaRN baseline exactly as DeepSeekV4RotaryEmbedding.inv_freq does."""
  half_dim = dim // 2
  fraction = 2 * jnp.arange(0, half_dim, dtype=jnp.float32) / dim
  return np.asarray(1.0 / (rope_theta**fraction))


def _rotary_embedding(rope_type, rope_theta=_COMPRESS_ROPE_THETA):
  return DeepSeekV4RotaryEmbedding(
      head_dim=_HEAD_DIM,
      partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
      rope_theta=rope_theta,
      rope_type=rope_type,
      rope_factor=_ROPE_FACTOR,
      beta_fast=_BETA_FAST,
      beta_slow=_BETA_SLOW,
      original_max_position_embeddings=_ORIGINAL_MAX_POSITION_EMBEDDINGS,
      truncate=True,
  )


class DeepSeekV4RotaryEmbeddingInvFreqTest(unittest.TestCase):
  """Locks in the inverse frequencies produced by DeepSeekV4RotaryEmbedding."""

  def test_default_rope_type_is_bit_identical_to_the_unscaled_baseline(self):
    for rope_theta in (_MAIN_ROPE_THETA, _COMPRESS_ROPE_THETA):
      with self.subTest(rope_theta=rope_theta):
        embedding = _rotary_embedding("default", rope_theta=rope_theta)
        np.testing.assert_array_equal(
            np.asarray(embedding.inv_freq),
            _unscaled_inv_freq(rope_theta, embedding.dim),
        )

  def test_yarn_leaves_high_frequency_bands_untouched(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)

    np.testing.assert_array_equal(scaled[: _EXPECTED_LOW + 1], baseline[: _EXPECTED_LOW + 1])

  def test_yarn_divides_low_frequency_bands_by_rope_factor(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)

    np.testing.assert_array_equal(scaled[_EXPECTED_HIGH:], baseline[_EXPECTED_HIGH:] / _ROPE_FACTOR)

  def test_yarn_ramp_interpolates_monotonically_between_the_two_regimes(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)
    ratio = scaled / baseline

    self.assertEqual(scaled.shape, (embedding.dim // 2,))
    # YaRN only ever stretches wavelengths, so every frequency shrinks or stays put.
    self.assertTrue(np.all(ratio <= 1.0 + 1e-6), f"ratio exceeded 1: {ratio}")
    self.assertTrue(np.all(ratio >= 1.0 / _ROPE_FACTOR - 1e-6), f"ratio below 1/factor: {ratio}")
    self.assertTrue(np.all(np.diff(ratio) <= 1e-7), f"ratio is not non-increasing: {ratio}")
    # The ramp region must be strictly interior, otherwise the test would pass vacuously.
    ramp = ratio[_EXPECTED_LOW + 1 : _EXPECTED_HIGH]
    self.assertTrue(np.all(ramp < 1.0) and np.all(ramp > 1.0 / _ROPE_FACTOR), f"degenerate ramp: {ramp}")

  def test_yarn_actually_changes_the_frequencies(self):
    baseline = np.asarray(_rotary_embedding("default").inv_freq)
    scaled = np.asarray(_rotary_embedding("yarn").inv_freq)

    self.assertFalse(np.allclose(scaled, baseline))


class CompressedAttentionRopeSelectionTest(unittest.TestCase):
  """Checks the per-layer rope_theta / rope_type selection in CompressedAttention."""

  def setUp(self):
    super().setUp()
    self.config = initialize(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek4-284b",
            "attention=dot_product",
            "qk_rope_head_dim=16",
            "v_head_dim=16",
            "qk_nope_head_dim=16",
            "override_model_config=True",
        ]
    )
    self.mesh = Mesh(jax.devices(), ("data",))

  def _build(self, compress_ratio):
    return CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=compress_ratio,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

  def test_model_config_pins_yarn_scaling(self):
    self.assertEqual(self.config.rope_type, "yarn")
    self.assertEqual(self.config.compressed_rope_max_timescale, 160000)
    self.assertEqual(self.config.rope_max_timescale, 10000)

  def test_compressed_layer_uses_compressed_theta_with_yarn(self):
    rotary = self._build(compress_ratio=4).rotary_embedding

    self.assertEqual(rotary.rope_theta, self.config.compressed_rope_max_timescale)
    self.assertEqual(rotary.rope_type, "yarn")

  def test_sliding_window_layer_uses_main_theta_without_yarn(self):
    rotary = self._build(compress_ratio=0).rotary_embedding

    self.assertEqual(rotary.rope_theta, self.config.rope_max_timescale)
    self.assertEqual(rotary.rope_type, "default")


def _load_ds4_model_rope_configs():
  """Loads the exact DeepSeek-V4 284B and 1600B RoPE configs from YAML."""
  models_dir = Path(MAXTEXT_CONFIGS_DIR) / "models"
  configs = []
  for model_name in ("deepseek4-284b", "deepseek4-1600b"):
    with (models_dir / f"{model_name}.yml").open("r", encoding="utf-8") as f:
      cfg = yaml.safe_load(f)
    common = {
        "head_dim": cfg["head_dim"],
        "partial_rotary_factor": cfg["qk_rope_head_dim"] / cfg["head_dim"],
        "rope_factor": float(cfg["rope_factor"]),
        "beta_fast": float(cfg["beta_fast"]),
        "beta_slow": float(cfg["beta_slow"]),
        "original_max_position_embeddings": int(cfg["original_max_position_embeddings"]),
        "truncate": bool(cfg.get("rope_truncate", True)),
    }
    configs.append(
        (f"{model_name}:main", {**common, "rope_theta": float(cfg["rope_max_timescale"]), "rope_type": "default"})
    )
    configs.append(
        (
            f"{model_name}:compressed",
            {**common, "rope_theta": float(cfg["compressed_rope_max_timescale"]), "rope_type": str(cfg["rope_type"])},
        )
    )
  return configs


def _yarn_correction_bounds(
    dim: int,
    original_max_position_embeddings: int,
    rope_theta: float,
    beta_fast: float,
    beta_slow: float,
    truncate: bool,
) -> tuple[float, float]:
  """Computes the YaRN (low, high) frequency-band correction bounds."""

  def _corr_dim(rotations: float) -> float:
    return dim * math.log(original_max_position_embeddings / (rotations * 2.0 * math.pi)) / (2.0 * math.log(rope_theta))

  low = _corr_dim(beta_fast)
  high = _corr_dim(beta_slow)
  if truncate:
    low = math.floor(low)
    high = math.ceil(high)
  low = max(low, 0)
  high = min(high, dim - 1)
  if low == high:
    high += 0.001
  return low, high


def _reference_pytorch_style_inv_freq_np(
    *,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    rope_type: str,
    rope_factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position_embeddings: int,
    truncate: bool,
) -> np.ndarray:
  """Torch-free NumPy implementation of the DeepSeek-V4 PyTorch inv_freq reference."""
  dim = int(head_dim * partial_rotary_factor)
  half_dim = dim // 2
  # PyTorch evaluates `base ** (torch.arange(0, dim, 2, dtype=float32) / dim)` to
  # accurately rounded float32, then computes 1.0 / pos_freqs and YaRN in float32.
  frac_f32 = (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)).astype(np.float64)
  pos_freqs = (np.float64(rope_theta) ** frac_f32).astype(np.float32)
  freqs = np.float32(1.0) / pos_freqs
  if rope_type == "yarn":
    low, high = _yarn_correction_bounds(dim, original_max_position_embeddings, rope_theta, beta_fast, beta_slow, truncate)
    ramp = np.clip(
        (np.arange(half_dim, dtype=np.float32) - np.float32(low)) / np.float32(high - low),
        np.float32(0.0),
        np.float32(1.0),
    )
    smooth = np.float32(1.0) - ramp
    freqs = freqs / np.float32(rope_factor) * (np.float32(1.0) - smooth) + freqs * smooth
  return freqs.astype(np.float32)


def _legacy_in_jax_inv_freq(
    *,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    rope_type: str,
    rope_factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position_embeddings: int,
    truncate: bool,
) -> jax.Array:
  """Previous in-JAX float32 formula used as a negative control."""
  dim = int(head_dim * partial_rotary_factor)
  half_dim = dim // 2
  fraction = 2 * jnp.arange(0, half_dim, dtype=jnp.float32) / dim
  freqs = 1.0 / (rope_theta**fraction)
  if rope_type != "yarn":
    return freqs

  low, high = _yarn_correction_bounds(dim, original_max_position_embeddings, rope_theta, beta_fast, beta_slow, truncate)
  ramp = jnp.clip((jnp.arange(half_dim, dtype=jnp.float32) - low) / (high - low), 0, 1)
  smooth = 1 - ramp
  return freqs / rope_factor * (1 - smooth) + freqs * smooth


def _ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return np.asarray(a, dtype=np.float32).view(np.int32) - np.asarray(b, dtype=np.float32).view(np.int32)


class DeepSeekV4RotaryEmbeddingHostPrecisionTest(unittest.TestCase):
  """Verifies host-computed inv_freq matches the PyTorch float32 reference bit-exactly."""

  def test_bit_exact_with_pytorch_reference_for_ds4_configs(self):
    for label, kwargs in _load_ds4_model_rope_configs():
      with self.subTest(config=label):
        rope = DeepSeekV4RotaryEmbedding(**kwargs)
        expected = _reference_pytorch_style_inv_freq_np(**kwargs)
        zero = jnp.zeros((expected.shape[0],), dtype=jnp.float32)

        actual_eager = np.asarray(rope.inv_freq, dtype=np.float32)
        actual_jit = np.asarray(jax.jit(lambda x, r=rope: r.inv_freq + x * 0.0)(zero), dtype=np.float32)

        np.testing.assert_array_equal(
            _ulp_diff(actual_eager, expected),
            np.zeros_like(expected, dtype=np.int32),
            err_msg=f"{label} eager inv_freq ULP drift vs reference",
        )
        np.testing.assert_array_equal(
            _ulp_diff(actual_jit, expected),
            np.zeros_like(expected, dtype=np.int32),
            err_msg=f"{label} jitted inv_freq ULP drift vs reference",
        )

  @unittest.skipUnless(HAS_TORCH, "Torch not available")
  def test_bit_exact_with_live_torch_reference_for_ds4_configs(self):
    for label, kwargs in _load_ds4_model_rope_configs():
      with self.subTest(config=label):
        rope = DeepSeekV4RotaryEmbedding(**kwargs)
        dim = int(kwargs["head_dim"] * kwargs["partial_rotary_factor"])
        base = kwargs["rope_theta"]
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if kwargs["rope_type"] == "yarn":
          low, high = _yarn_correction_bounds(
              dim,
              kwargs["original_max_position_embeddings"],
              base,
              kwargs["beta_fast"],
              kwargs["beta_slow"],
              kwargs["truncate"],
          )
          ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
          smooth = 1 - ramp
          freqs = freqs / kwargs["rope_factor"] * (1 - smooth) + freqs * smooth
        expected = freqs.numpy().astype(np.float32)
        zero = jnp.zeros((expected.shape[0],), dtype=jnp.float32)
        actual_jit = np.asarray(jax.jit(lambda x, r=rope: r.inv_freq + x * 0.0)(zero), dtype=np.float32)
        np.testing.assert_array_equal(_ulp_diff(actual_jit, expected), np.zeros_like(expected, dtype=np.int32))

  def test_compiled_hlo_has_no_runtime_power_op(self):
    for label, kwargs in _load_ds4_model_rope_configs():
      with self.subTest(config=label):
        rope = DeepSeekV4RotaryEmbedding(**kwargs)
        zero = jnp.zeros((rope.dim // 2,), dtype=jnp.float32)
        compiled = jax.jit(lambda x, r=rope: r.inv_freq + x * 0.0).lower(zero).compile()
        self.assertNotIn("power(", compiled.as_text(), f"{label} emitted runtime HLO power()")

  def test_negative_control_legacy_in_jax_fp32_fails_bit_exact_check(self):
    for label, kwargs in _load_ds4_model_rope_configs():
      with self.subTest(config=label):
        expected = _reference_pytorch_style_inv_freq_np(**kwargs)
        zero = jnp.zeros((expected.shape[0],), dtype=jnp.float32)
        compiled = jax.jit(lambda x, kw=kwargs: _legacy_in_jax_inv_freq(**kw) + x * 0.0).lower(zero).compile()
        self.assertIn("power(", compiled.as_text(), f"{label} legacy negative control should emit HLO power()")
        legacy_jit = np.asarray(compiled(zero), dtype=np.float32)
        nonzero_ulps = int(np.count_nonzero(_ulp_diff(legacy_jit, expected)))
        self.assertGreater(
            nonzero_ulps,
            0,
            f"{label} legacy in-JAX jitted inv_freq unexpectedly matched the reference bit-exactly",
        )


if __name__ == "__main__":
  unittest.main()
