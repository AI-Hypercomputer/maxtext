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

"""Unit and scheduled golden parity tests for forward_pass_velocity_checker."""

import os
import subprocess
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.configs import pyconfig
from maxtext.models import weaver
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import numpy as np
import pytest
from tests.utils import forward_pass_velocity_checker

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
_VELOCITY_GOLDEN_FILENAME = "weaver_velocity_golden_data.npz"


def _get_golden_data_path(filename: str) -> str | None:
  """Attempts to locate or download golden test asset from GCS bucket."""
  tmp_path = f"/tmp/{filename}"
  if os.path.exists(tmp_path):
    return tmp_path

  gcs_uri = f"gs://maxtext-test-assets/{filename}"
  try:
    ret = subprocess.call(
        ["gcloud", "storage", "cp", gcs_uri, tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if ret == 0 and os.path.exists(tmp_path):
      return tmp_path
  except (subprocess.SubprocessError, OSError):
    pass

  return None


def _run_velocity_checker(argv: list[str]) -> dict[str, float]:
  """Parses CLI arguments, initializes MaxText pyconfig, and invokes velocity checker."""
  test_args, remaining_args = forward_pass_velocity_checker.parse_velocity_checker_args(argv)
  cfg = pyconfig.initialize(["", *remaining_args])
  return forward_pass_velocity_checker.main(cfg, test_args)


@pytest.mark.cpu_only
class WeaverVelocityCheckerUnitTest(parameterized.TestCase):
  """Fast CPU presubmit unit tests for forward_pass_velocity_checker (no golden assets)."""

  def test_compute_velocity_metrics(self):
    """Verifies max_abs_diff, mean_abs_diff, rmse, and cosine_similarity calculations."""
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((2, 48, 2, 4, 4)).astype(np.float32)

    exact_metrics = forward_pass_velocity_checker.compute_velocity_metrics(ref, ref)
    self.assertAlmostEqual(exact_metrics["max_abs_diff"], 0.0, places=7)
    self.assertAlmostEqual(exact_metrics["mean_abs_diff"], 0.0, places=7)
    self.assertAlmostEqual(exact_metrics["rmse"], 0.0, places=7)
    self.assertAlmostEqual(exact_metrics["cosine_similarity"], 1.0, places=6)

    perturbed = ref.copy()
    perturbed[0, 0, 0, 0, 0] += 1e-4
    diff_metrics = forward_pass_velocity_checker.compute_velocity_metrics(perturbed, ref)
    self.assertAlmostEqual(diff_metrics["max_abs_diff"], 1e-4, places=6)
    self.assertGreater(diff_metrics["cosine_similarity"], 0.99999)

  def test_synthetic_1_layer_velocity_checker_pipeline(self):
    """Runs the full Orbax conversion + velocity checker pipeline on synthetic 1-layer weights."""
    cfg = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=weaver-mini-diffuser",
            "override_model_config=True",
            "base_num_decoder_layers=1",
            "base_emb_dim=64",
            "base_mlp_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
            "mrope_section=[4,2,2]",
            "scan_layers=False",
            "dtype=float32",
            "weight_dtype=float32",
            "skip_jax_distributed_system=True",
        ]
    )
    hf_cfg = forward_pass_velocity_checker.build_hf_config_from_maxtext_config(cfg)
    mt_cfg = weaver.WeaverConfig.from_maxtext_config(cfg)

    rng = np.random.default_rng(42)
    shape_map = hf_shape.WEAVER_HF_WEIGHTS_TO_SHAPE(hf_cfg)
    hf_weights = {}
    for hf_key, shape in shape_map.items():
      if hf_key == "lm_head.weight" or "k_norm_und_for_gen" in hf_key:
        continue
      hf_weights[hf_key] = (rng.standard_normal(size=tuple(shape)) * 0.02).astype(np.float32)

    ref_model = weaver.WeaverOmniTransformer(mt_cfg, rngs=nnx.Rngs(0))
    param_map = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=False)
    hooks = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, mt_cfg, scan_layers=False, saving_to_hf=False)

    pure_params = nnx.state(ref_model, nnx.Param).to_pure_dict()
    flat_leaves, treedef = jax.tree_util.tree_flatten_with_path({"params": pure_params})
    updated_leaves = []
    for path, leaf in flat_leaves:
      mt_key = "-".join(conversion_utils.param_key_parts_from_path(path))
      hf_key = param_map[mt_key]
      arr = hf_weights[hf_key]
      hook_fn = hooks.get(mt_key)
      if isinstance(hook_fn, list):
        for fn in hook_fn:
          arr = fn(arr, leaf.shape)
      elif callable(hook_fn):
        arr = hook_fn(arr, leaf.shape)
      updated_leaves.append(jnp.asarray(arr, dtype=jnp.float32))
    restored_tree = jax.tree_util.tree_unflatten(treedef, updated_leaves)
    nnx.update(ref_model, restored_tree["params"])

    input_ids = rng.integers(0, 256, size=(2, 6), dtype=np.int32)
    latents = rng.standard_normal((2, 48, 2, 4, 4)).astype(np.float32)
    timesteps = np.full((2,), 500.0, dtype=np.float32)

    mini_1layer_v_pred = np.asarray(
        ref_model(jnp.asarray(input_ids), jnp.asarray(latents), jnp.asarray(timesteps)),
        dtype=np.float32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      synthetic_npz_path = os.path.join(tmpdir, "synthetic_velocity.npz")
      archive = {
          "input_ids": input_ids,
          "latents": latents,
          "timesteps": timesteps,
          "mini_1layer_v_pred": mini_1layer_v_pred,
      }
      for hf_k, hf_v in hf_weights.items():
        archive[f"weights/{hf_k}"] = hf_v
      np.savez(synthetic_npz_path, **archive)

      metrics = _run_velocity_checker(
          [
              _BASE_CONFIG_PATH,
              "model_name=weaver-mini-diffuser",
              "override_model_config=True",
              "base_num_decoder_layers=1",
              "base_emb_dim=64",
              "base_mlp_dim=128",
              "base_num_query_heads=4",
              "base_num_kv_heads=2",
              "head_dim=16",
              "vocab_size=256",
              "mrope_section=[4,2,2]",
              "scan_layers=False",
              "dtype=float32",
              "weight_dtype=float32",
              "skip_jax_distributed_system=True",
              f"--golden_velocity_path={synthetic_npz_path}",
              "--atol=1e-5",
              "--min_cosine_sim=0.99999",
              "--timestep=500.0",
          ]
      )
      self.assertLessEqual(metrics["max_abs_diff"], 1e-5)
      self.assertGreaterEqual(metrics["cosine_similarity"], 0.99999)


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class WeaverVelocityCheckerGoldenParityTest(parameterized.TestCase):
  """Scheduled nightly golden parity tests for forward_pass_velocity_checker."""

  golden_path: str | None = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.golden_path = _get_golden_data_path(_VELOCITY_GOLDEN_FILENAME)

  def setUp(self):
    super().setUp()
    if self.golden_path is None:
      self.skipTest(
          f"Golden test asset {_VELOCITY_GOLDEN_FILENAME} not found locally under /tmp "
          "and could not be downloaded from gs://maxtext-test-assets/."
      )

  def _base_cli_args(self, num_layers: int, scan_layers: bool) -> list[str]:
    assert self.golden_path is not None
    return [
        _BASE_CONFIG_PATH,
        "model_name=weaver-mini-diffuser",
        "override_model_config=True",
        f"base_num_decoder_layers={num_layers}",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
        "mrope_section=[4,2,2]",
        f"scan_layers={scan_layers}",
        "dtype=float32",
        "weight_dtype=float32",
        "skip_jax_distributed_system=True",
        f"--golden_velocity_path={self.golden_path}",
        "--timestep=500.0",
    ]

  def test_mini_1_layer_velocity_golden_parity(self):
    """Validates mini 1-layer Orbax checkpoint against PyTorch golden (max_abs_diff <= 1e-4)."""
    args = self._base_cli_args(num_layers=1, scan_layers=False) + ["--atol=1e-4"]
    metrics = _run_velocity_checker(args)
    self.assertLessEqual(metrics["max_abs_diff"], 1e-4)

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_full_36_layer_velocity_golden_parity(self, scan_layers: bool):
    """Validates full 36-layer Orbax checkpoint against PyTorch golden (cos_sim >= 0.9999, max_abs_diff <= 5e-4)."""
    args = self._base_cli_args(num_layers=36, scan_layers=scan_layers) + [
        "--atol=5e-4",
        "--min_cosine_sim=0.9999",
    ]
    metrics = _run_velocity_checker(args)
    self.assertLessEqual(metrics["max_abs_diff"], 5e-4)
    self.assertGreaterEqual(metrics["cosine_similarity"], 0.9999)

  def test_mini_1_layer_live_hf_velocity_parity(self):
    """Validates mini 1-layer Orbax checkpoint against live PyTorch HF reference (--run_hf_model=True)."""
    if forward_pass_velocity_checker.torch is None:
      self.skipTest("PyTorch is not installed in this environment.")
    args = self._base_cli_args(num_layers=1, scan_layers=False) + [
        "--run_hf_model=True",
        "--atol=1e-4",
    ]
    metrics = _run_velocity_checker(args)
    self.assertLessEqual(metrics["max_abs_diff"], 1e-4)


if __name__ == "__main__":
  absltest.main()
