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

"""Unit tests for DeepSeek-V4.1 multimodal training E2E support and staged TPU script."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
import numpy as np
import optax
import pytest

from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.multimodal import processor as mm_processor
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils

try:
  import torch
  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  HAS_TORCH = False

from tests.utils import deepseek_v41_mm_testing as mm_testing

_REPO_ROOT = Path(__file__).resolve().parents[2]
_E2E_DIR = _REPO_ROOT / "tests" / "end_to_end" / "tpu" / "deepseek" / "v4.1-flash"
_STAGE_SCRIPT = (_E2E_DIR / "2_test_deepseek41_mm.sh").as_posix()
_AOT_CHECK_PY = (_E2E_DIR / "aot_check.py").as_posix()


def _load_aot_check_module() -> Any:
  """Loads `tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py` as a module."""
  spec = importlib.util.spec_from_file_location("ds41_aot_check_train_unit", _AOT_CHECK_PY)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_AOT_CHECK_PY}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _init_mini_mm_config(*, grid: tuple[int, int] | None = None, **overrides: Any) -> Any:
  """Initializes a 14-layer DeepSeek-V4.1-Flash mini multimodal config."""
  if grid is None:
    grid = multimodal.MINI_GRID_PADDED if multimodal is not None else (7, 5)
  defaults: dict[str, Any] = {
      "run_name": "ds41_mm_train_unit",
      "save_checkpoint_on_completion": False,
      "dataset_type": "synthetic",
      "per_device_batch_size": 1.0,
      "max_target_length": 64,
      "ici_fsdp_parallelism": 1,
      "ici_data_parallelism": -1,
      "ici_expert_parallelism": 1,
      "ici_tensor_parallelism": 1,
      **(
          {}
          if HAS_TORCH
          else {
              "tokenizer_path": mm_testing.OFFLINE_TOKENIZER_PATH,
              "ds41_engram_compressed_vocab_size": 0,
          }
      ),
  }
  defaults.update(overrides)
  return mm_testing.mini_vision_maxtext_config(grid=grid, **defaults)


def _read_metrics_by_step(metrics_path: str | Path) -> dict[int, dict[str, Any]]:
  """Reads JSONL training metrics keyed by integer `step`."""
  by_step: dict[int, dict[str, Any]] = {}
  for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    rec = json.loads(stripped)
    if "step" in rec:
      by_step[int(rec["step"])] = rec
  return by_step


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeekV41MMTrainE2ETest(unittest.TestCase):
  """CPU unit tests for multimodal training E2E support and staged TPU runner script."""

  tmp_dir: str = ""
  ckpt_dir: str = ""

  @classmethod
  def setUpClass(cls) -> None:
    """Initializes the shared 14-layer mini multimodal safetensors checkpoint."""
    super().setUpClass()
    cls.tmp_dir = tempfile.mkdtemp(prefix="ds41_mm_train_")
    if HAS_TORCH:
      cls.ckpt_dir = os.path.join(cls.tmp_dir, "mini_safetensors")
      args = multimodal.mini_model_args()
      ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
      multimodal.init_mini_reference(ref_mm, seed=91)
      offset = torch.tensor([-0.35, 0.35, -0.20, 0.20], dtype=torch.float32)
      with torch.no_grad():
        for layer in ref_mm.layers:
          layer.ffn.gate.bias_vl.copy_(layer.ffn.gate.bias + offset)
      mm_testing.write_mini_safetensors_checkpoint(cls.ckpt_dir, ref_mm, param_dtype=torch.float32)
      del ref_mm
      gc.collect()
    else:
      cls.ckpt_dir = ""

  @classmethod
  def tearDownClass(cls) -> None:
    """Cleans up the shared temporary directory."""
    if cls.tmp_dir:
      shutil.rmtree(cls.tmp_dir, ignore_errors=True)
      cls.tmp_dir = ""
    super().tearDownClass()

  def _run_two_step_cpu_train_loop(self, *, scan_layers: bool) -> None:
    """Runs a 2-step CPU training loop and checks finite loss decrease."""
    assert self.tmp_dir
    run_out_dir = os.path.join(self.tmp_dir, f"out_scan_{int(scan_layers)}")
    metrics_path = os.path.join(run_out_dir, "metrics.jsonl")
    os.makedirs(run_out_dir, exist_ok=True)

    if self.ckpt_dir:
      cfg = _init_mini_mm_config(
          grid=multimodal.MINI_GRID_PADDED,
          base_output_directory=run_out_dir,
          load_parameters_path=self.ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
          scan_layers=scan_layers,
          freeze_vision_encoder_params=True,
          log_moe_bias_norms=True,
          log_period=1,
          routed_bias=True,
          routed_bias_update_rate=1e-3,
          dataset_type="synthetic",
          max_num_images_per_example=1,
          max_target_length=32,
          per_device_batch_size=1.0,
          steps=2,
          learning_rate_schedule_steps=2,
          learning_rate=1e-3,
          warmup_steps_fraction=0.0,
          adam_weight_decay=0.1,
          metrics_file=metrics_path,
      )
    else:
      cfg = _init_mini_mm_config(
          grid=(7, 5),
          base_output_directory=run_out_dir,
          scan_layers=scan_layers,
          freeze_vision_encoder_params=True,
          log_moe_bias_norms=True,
          log_period=1,
          routed_bias=True,
          routed_bias_update_rate=1e-3,
          dataset_type="synthetic",
          max_num_images_per_example=1,
          max_target_length=32,
          per_device_batch_size=1.0,
          steps=2,
          learning_rate_schedule_steps=2,
          learning_rate=1e-3,
          warmup_steps_fraction=0.0,
          adam_weight_decay=0.1,
          metrics_file=metrics_path,
      )

    train.train_loop(cfg, recorder=None)

    metrics_by_step = _read_metrics_by_step(metrics_path)
    self.assertIn(0, metrics_by_step)
    self.assertIn(1, metrics_by_step)
    loss_0 = float(metrics_by_step[0]["learning/loss"])
    loss_1 = float(metrics_by_step[1]["learning/loss"])
    self.assertTrue(np.isfinite(loss_0) and np.isfinite(loss_1))
    self.assertLess(loss_1, loss_0, msg=f"Expected loss(1)={loss_1} < loss(0)={loss_0}")

  def test_dummy_image_shape_and_synthetic_iterator_n1_and_n4(self) -> None:
    """Verifies N=1/N=4 shapes, synthetic spans, target masking, and packing/vocab bounds."""
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices), 1), devices), ("data", "fsdp"))

    cases = [
        ((39, 39), -1, 1, 1521, 184, 512),
        ((39, 39), 1, 1, 1521, 184, 512),
        ((39, 39), 4, 4, 1521, 184, 1024),
        ((7, 5), 1, 1, 35, 11, 64),
        ((7, 5), 4, 4, 35, 11, 64),
        ((6, 9), 1, 1, 54, 10, 64),
        ((6, 9), 4, 4, 54, 10, 64),
    ]
    for grid, cfg_max_images, expected_n, expected_patches, expected_span_len, seq_len in cases:
      with self.subTest(grid=grid, cfg_max_images=cfg_max_images):
        cfg = _init_mini_mm_config(
            grid=grid,
            max_num_images_per_example=cfg_max_images,
            max_target_length=seq_len,
            per_device_batch_size=1.0,
            packing=False,
            max_segments_per_seq=1,
        )
        batch_size = int(cfg.global_batch_size_to_load)
        expected_img_shape = (batch_size, expected_n, expected_patches, 588)

        dummy_shape = mm_processor.get_dummy_image_shape_for_init(
            cfg.model_name,
            batch_size=batch_size,
            config=cfg,
        )
        self.assertEqual(dummy_shape, expected_img_shape)

        override_shape = mm_processor.get_dummy_image_shape_for_init(
            cfg.model_name,
            batch_size=batch_size,
            num_image_per_sequence=3,
            config=cfg,
        )
        self.assertEqual(override_shape, (batch_size, 3, expected_patches, 588))

        shaped = maxtext_utils.get_shaped_batch(cfg)
        self.assertIn("images", shaped)
        self.assertEqual(shaped["images"].shape, expected_img_shape)
        self.assertEqual(shaped["images"].dtype, jnp.float32)
        self.assertNotIn("image_masks", shaped)

        iterator = SyntheticDataIterator(cfg, mesh)
        batch = next(iterator)
        self.assertIn("images", batch)
        self.assertEqual(batch["images"].shape, expected_img_shape)
        self.assertEqual(batch["images"].dtype, jnp.float32)

        inputs = np.asarray(batch["inputs"])
        targets = np.asarray(batch["targets"])
        inputs_seg = np.asarray(batch["inputs_segmentation"])
        targets_seg = np.asarray(batch["targets_segmentation"])
        image_token_id = int(cfg.ds41_image_token_id)
        total_span = expected_n * expected_span_len

        self.assertEqual(inputs_seg.shape, (batch_size, seq_len))
        self.assertEqual(targets_seg.shape, (batch_size, seq_len))
        self.assertTrue(np.all(inputs[:, :8] != image_token_id))
        self.assertTrue(np.all(inputs[:, 8 : 8 + total_span] == image_token_id))
        self.assertTrue(np.all(inputs[:, 8 + total_span :] != image_token_id))
        self.assertTrue(np.all(inputs_seg == 1))

        self.assertTrue(np.all(targets[:, :7] != image_token_id))
        self.assertTrue(np.all(targets[:, 7 : 7 + total_span] == image_token_id))
        self.assertTrue(np.all(targets[:, 7 + total_span :] != image_token_id))
        self.assertTrue(np.all(targets_seg[:, :7] == 1))
        self.assertTrue(np.all(targets_seg[:, 7 : 7 + total_span] == 0))
        self.assertTrue(np.all(targets_seg[:, 7 + total_span :] == 1))
        np.testing.assert_array_equal(targets_seg == 0, targets == image_token_id)

    # DS4.1 get_shaped_batch always uses float32 for images while sizing axis 0 to global_batch_size.
    cfg_tfds = _init_mini_mm_config(
        grid=(7, 5),
        dataset_type="tfds",
        per_device_batch_size=0.5,
        max_target_length=64,
    )
    shaped_tfds = maxtext_utils.get_shaped_batch(cfg_tfds)
    self.assertEqual(shaped_tfds["images"].shape, (int(cfg_tfds.global_batch_size_to_load), 1, 35, 588))
    self.assertEqual(shaped_tfds["images"].dtype, jnp.float32)

    # Negative control 1: max_target_length <= 8 + N * span_len raises ValueError.
    for bad_seq_len in (8 + 4 * 11, 8 + 4 * 11 - 1):
      cfg_bad_len = _init_mini_mm_config(
          grid=(7, 5),
          max_num_images_per_example=4,
          max_target_length=bad_seq_len,
      )
      with self.assertRaisesRegex(ValueError, "max_target_length"):
        SyntheticDataIterator(cfg_bad_len, mesh)

    # Negative control 2: packing=True with max_segments_per_seq >= 2 raises ValueError (no fractured span).
    cfg_bad_pack = _init_mini_mm_config(
        grid=(7, 5),
        max_num_images_per_example=1,
        max_target_length=64,
        packing=False,
    )
    cfg_bad_pack.get_keys()["packing"] = True
    cfg_bad_pack.get_keys()["max_segments_per_seq"] = 2
    with self.assertRaisesRegex(ValueError, "max_segments_per_seq"):
      SyntheticDataIterator(cfg_bad_pack, mesh)

    # Negative control 3: out-of-range ds41_image_token_id (<= 0 or >= vocab_size=256) raises ValueError.
    cfg_tok = _init_mini_mm_config(
        grid=(7, 5),
        max_num_images_per_example=1,
        max_target_length=64,
        vocab_size=256,
        ds41_image_token_id=200,
    )
    ok_batch = next(SyntheticDataIterator(cfg_tok, mesh))
    self.assertTrue(bool(np.all(np.asarray(ok_batch["inputs"])[:, 8:19] == 200)))
    for bad_tok_id in (-1, 0, 256, 300):
      cfg_tok.get_keys()["ds41_image_token_id"] = bad_tok_id
      with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
        SyntheticDataIterator(cfg_tok, mesh)
    cfg_tok.get_keys()["ds41_image_token_id"] = 200

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_subbatch_per_device_0_5_no_indivisible_error_and_single_trace(self) -> None:
    """Verifies per_device_batch_size=0.5 across 4 CPU devices runs without IndivisibleError and traces train_step once."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices (e.g. XLA_FLAGS='--xla_force_host_platform_device_count=4')")
    run_out_dir = os.path.join(self.tmp_dir, "out_subbatch_0_5")
    metrics_path = os.path.join(run_out_dir, "metrics.jsonl")
    os.makedirs(run_out_dir, exist_ok=True)

    cfg = _init_mini_mm_config(
        grid=multimodal.MINI_GRID_PADDED,
        base_output_directory=run_out_dir,
        load_parameters_path=self.ckpt_dir,
        source_checkpoint_layout="safetensors_dynamic",
        scan_layers=True,
        freeze_vision_encoder_params=True,
        log_moe_bias_norms=True,
        log_period=1,
        dataset_type="synthetic",
        max_num_images_per_example=1,
        max_target_length=32,
        per_device_batch_size=0.5,
        ici_data_parallelism=1,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        ici_tensor_parallelism=1,
        steps=2,
        learning_rate_schedule_steps=2,
        learning_rate=1e-5,
        warmup_steps_fraction=0.0,
        adam_weight_decay=0.1,
        gradient_clipping_threshold=1.0,
        metrics_file=metrics_path,
    )
    self.assertEqual(int(cfg.global_batch_size_to_load), 4)
    self.assertEqual(int(cfg.micro_batch_size_to_train_on), 2)

    shaped = maxtext_utils.get_shaped_batch(cfg)
    self.assertEqual(shaped["images"].shape, (4, 1, 35, 588))
    self.assertEqual(shaped["images"].dtype, jnp.float32)
    self.assertNotIn("image_masks", shaped)

    # Verify tpu7x_64c_mm_train_adamw AOT preset carries expected schedule flags and compiles cleanly.
    aot_mod = _load_aot_check_module()
    spec_train = aot_mod.get_preset_spec("tpu7x_64c_mm_train_adamw")
    self.assertEqual(spec_train["overrides"]["learning_rate"], 1e-5)
    self.assertEqual(spec_train["overrides"]["warmup_steps_fraction"], 0.0)
    self.assertEqual(spec_train["overrides"]["adam_weight_decay"], 0.1)
    self.assertEqual(spec_train["overrides"]["gradient_clipping_threshold"], 1.0)
    self.assertEqual(spec_train["overrides"]["load_balance_loss_weight"], 0.0001)
    self.assertEqual(spec_train["overrides"]["routed_bias_update_rate"], 0.001)
    self.assertEqual(spec_train["overrides"]["steps"], 20)
    self.assertEqual(spec_train["overrides"]["learning_rate_schedule_steps"], 20)

    res_aot = aot_mod.run_train_aot_check(
        cfg,
        with_images=True,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
    )
    self.assertTrue(all(res_aot["vision_hlo_markers"].values()))
    self.assertGreater(res_aot["peak_hbm_gib"], 0.0)

    orig_train_step = train.train_step
    trace_count = 0

    def _counting_train_step(*args: Any, **kwargs: Any) -> Any:
      nonlocal trace_count
      trace_count += 1
      return orig_train_step(*args, **kwargs)

    with mock.patch.object(train, "train_step", side_effect=_counting_train_step):
      train.train_loop(cfg, recorder=None)

    # Exactly 1 XLA trace during AOT .lower(...).compile(); zero step-0 or step-1 cache-miss recompilations.
    self.assertEqual(trace_count, 1)
    metrics_by_step = _read_metrics_by_step(metrics_path)
    self.assertIn(0, metrics_by_step)
    self.assertIn(1, metrics_by_step)
    self.assertTrue(np.isfinite(float(metrics_by_step[0]["learning/loss"])))
    self.assertTrue(np.isfinite(float(metrics_by_step[1]["learning/loss"])))

  def test_two_step_cpu_train_loop_loss_descent_unscanned(self) -> None:
    """Verifies 2-step CPU train_loop loss reduction with scan_layers=False."""
    self._run_two_step_cpu_train_loop(scan_layers=False)

  def test_two_step_cpu_train_loop_loss_descent_scanned(self) -> None:
    """Verifies 2-step CPU train_loop loss reduction with scan_layers=True."""
    self._run_two_step_cpu_train_loop(scan_layers=True)

  def test_stage_script_bash_x_path_shim_and_failure_continuation(self) -> None:
    """Verifies 2_test_deepseek41_mm.sh flags, PYTHONPATH child imports, GCS error propagation, and stage continuation."""
    self.assertTrue(os.path.isfile(_STAGE_SCRIPT))
    self.assertTrue(os.access(_STAGE_SCRIPT, os.X_OK))

    with tempfile.TemporaryDirectory(prefix="ds41_mm_shim_") as tmp_dir:
      shim_bin = os.path.join(tmp_dir, "bin")
      os.makedirs(shim_bin, exist_ok=True)
      calls_log = os.path.join(tmp_dir, "python_calls.jsonl")
      gcloud_log = os.path.join(tmp_dir, "gcloud_calls.jsonl")

      shim_py = os.path.join(shim_bin, "python3")
      shim_code = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

if len(sys.argv) > 1 and sys.argv[1] == "-":
  os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])

with open({calls_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

out_dir = os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else ""
if out_dir:
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  (Path(out_dir) / "shim_marker.json").write_text("{{}}", encoding="utf-8")

for arg in sys.argv[2:]:
  if arg.startswith("--summary_json=") or arg.startswith("--output_json="):
    summary_dest = Path(arg.split("=", 1)[1])
    summary_dest.parent.mkdir(parents=True, exist_ok=True)
    summary_dest.write_text(json.dumps({{"passed": True, "written_by": "shim"}}), encoding="utf-8")

joined = " ".join(sys.argv[1:])
if "train.py" in joined:
  for arg in sys.argv[2:]:
    if arg.startswith("metrics_file="):
      mpath = Path(arg.split("=", 1)[1])
      mpath.parent.mkdir(parents=True, exist_ok=True)
      loss_base = float("nan") if os.environ.get("SHIM_NAN_LOSS") == "1" else (0.0 if os.environ.get("SHIM_ZERO_LOSS") == "1" else 10.0)
      loss_step = 0.0 if os.environ.get("SHIM_ZERO_LOSS") == "1" else 0.1
      lines = [json.dumps({{"step": s, "learning/loss": loss_base - loss_step * s}}) for s in range(20)]
      mpath.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

fail_match = os.environ["SHIM_FAIL_SUBSTR"] if "SHIM_FAIL_SUBSTR" in os.environ else ""
if fail_match and fail_match in joined:
  sys.exit(7)
sys.exit(0)
"""
      Path(shim_py).write_text(shim_code, encoding="utf-8")
      os.chmod(shim_py, stat.S_IRWXU)

      shim_gcloud = os.path.join(shim_bin, "gcloud")
      gcloud_code = f"""#!{sys.executable}
import json
import os
import sys

with open({gcloud_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

if "SHIM_GCLOUD_FAIL" in os.environ and os.environ["SHIM_GCLOUD_FAIL"] == "1":
  sys.exit(9)
sys.exit(0)
"""
      Path(shim_gcloud).write_text(gcloud_code, encoding="utf-8")
      os.chmod(shim_gcloud, stat.S_IRWXU)

      mm_dir = os.path.join(tmp_dir, "mm_goldens")
      os.makedirs(mm_dir, exist_ok=True)
      Path(mm_dir, "L1.npz").write_bytes(b"mock_L1")
      Path(mm_dir, "M1.npz").write_bytes(b"mock_M1")

      base_env = os.environ.copy()
      base_env["PATH"] = f"{shim_bin}:{base_env['PATH']}"
      base_env["HF_CKPT_PATH"] = os.path.join(tmp_dir, "hf_ckpt")
      base_env["OUTPUT_DIR"] = os.path.join(tmp_dir, "out")
      base_env["GOLDEN_TEXT_PKL"] = os.path.join(tmp_dir, "golden_full.pkl")
      base_env["GOLDEN_MM_DIR"] = mm_dir
      base_env["GOLDEN_PREFILL_TEXT_PKL"] = os.path.join(tmp_dir, "prefill_text.pkl")
      base_env["ORBAX_EXPORT_PREFIX"] = os.path.join(tmp_dir, "orbax")
      base_env["RUN_TEXT_PREFILL"] = "1"
      base_env["STAGES"] = "scanned_logits,unscanned_logits,masked_prefill,train_e2e"

      proc = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=base_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr} stdout={proc.stdout}")
      for stg in ("scanned_logits", "unscanned_logits", "masked_prefill", "train_e2e"):
        self.assertIn(f"=== STAGE {stg} START ", proc.stdout)
        self.assertIn(f"=== STAGE {stg} END ", proc.stdout)
      self.assertIn("rc=0 ===", proc.stdout)

      calls = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(calls), 4)
      hf_ckpt_arg = f"--hf_checkpoint_dir_for_audit={base_env['HF_CKPT_PATH']}"

      # Stage 1: scanned_logits via tests.utils.deepseek_v41.logit_runner
      self.assertEqual(calls[0][:2], ["-m", "tests.utils.deepseek_v41.logit_runner"])
      for req_scanned in (
          "run_name=ds41_mm_scanned_logits",
          "scan_layers=true",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--run_ds41_mm_audit=true",
          "--enable_long_text_gate=false",
          hf_ckpt_arg,
      ):
        self.assertIn(req_scanned, calls[0])
      self.assertTrue(any(a.startswith("--export_orbax_dir=") and a.endswith("/scanned") for a in calls[0]))

      # Stage 2: unscanned_logits via tests.utils.deepseek_v41.logit_runner
      self.assertEqual(calls[1][:2], ["-m", "tests.utils.deepseek_v41.logit_runner"])
      for req_unscanned in (
          "run_name=ds41_mm_unscanned_logits",
          "scan_layers=false",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--run_ds41_mm_audit=true",
          "--enable_long_text_gate=false",
          hf_ckpt_arg,
      ):
        self.assertIn(req_unscanned, calls[1])
      self.assertTrue(any(a.startswith("--export_orbax_dir=") and a.endswith("/unscanned") for a in calls[1]))

      # Stage 3: masked_prefill via tests.utils.deepseek_v41.masked_prefill_runner
      self.assertEqual(calls[2][:2], ["-m", "tests.utils.deepseek_v41.masked_prefill_runner"])
      for req_prefill in (
          "run_name=ds41_mm_masked_prefill",
          "scan_layers=true",
          "max_target_length=2048",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--max_kl_div=0.08",
          "--min_cos_sim=0.999",
          "--enforce_length_bucket_gates=true",
      ):
        self.assertIn(req_prefill, calls[2])
      expected_prefill_mm_dir = os.path.join(base_env["OUTPUT_DIR"], "masked_prefill_mm_m_only")
      self.assertIn(f"--mm_golden_dir={expected_prefill_mm_dir}", calls[2])
      self.assertNotIn(f"--mm_golden_dir={mm_dir}", calls[2])
      self.assertEqual(sorted(os.listdir(expected_prefill_mm_dir)), ["M1.npz"])
      self.assertNotIn("L1.npz", os.listdir(expected_prefill_mm_dir))
      self.assertEqual(
          Path(expected_prefill_mm_dir, "M1.npz").resolve(),
          Path(mm_dir, "M1.npz").resolve(),
      )
      self.assertTrue(any(a.startswith("--golden_logits_path=") for a in calls[2]))

      # Stage 4: direct train.py invocation and train_e2e_summary.json generation
      self.assertTrue(any("train.py" in a for a in calls[3]))
      for required_flag in (
          "run_name=ds41_mm_train_e2e",
          "override_model_config=true",
          "dtype_mm=bfloat16",
          "ds41_activation_quant=true",
          "indexer_sparse_training=true",
          "indexer_loss_scaling_factor=0.0",
          "use_multimodal=true",
          "vision_encoder_block=deepseek41",
          "freeze_vision_encoder_params=true",
          "dataset_type=synthetic",
          "max_num_images_per_example=1",
          "max_target_length=512",
          "per_device_batch_size=0.125",
          "log_moe_bias_norms=true",
          "log_moe_dropped_frac=true",
          "steps=20",
          "learning_rate_schedule_steps=20",
          "learning_rate=1e-5",
          "warmup_steps_fraction=0.0",
          "adam_weight_decay=0.1",
          "gradient_clipping_threshold=1.0",
          "load_balance_loss_weight=0.0001",
          "routed_bias_update_rate=0.001",
      ):
        self.assertIn(required_flag, calls[3])
      self.assertNotIn("ds41_routed_bias_vl_update_rate=", " ".join(calls[3]))
      train_summary = json.loads((Path(base_env["OUTPUT_DIR"]) / "train_e2e_summary.json").read_text(encoding="utf-8"))
      self.assertTrue(train_summary["passed"])
      self.assertEqual(train_summary["expected_steps"], 20)
      self.assertEqual(train_summary["recorded_steps"], 20)

      # GCS_LOG_PREFIX positive & negative controls (failed gcloud and missing gcloud on PATH must fail loudly).
      gcs_ok_env = dict(base_env)
      gcs_ok_env["STAGES"] = "scanned_logits"
      gcs_ok_env["GCS_LOG_PREFIX"] = "gs://example-bucket/ds41_mm_logs"
      proc_gcs_ok = subprocess.run(["bash", _STAGE_SCRIPT], env=gcs_ok_env, capture_output=True, text=True, check=False)
      self.assertEqual(proc_gcs_ok.returncode, 0, msg=f"stderr={proc_gcs_ok.stderr}")
      self.assertTrue(os.path.isfile(gcloud_log))

      gcs_fail_env = dict(gcs_ok_env)
      gcs_fail_env["SHIM_GCLOUD_FAIL"] = "1"
      proc_gcs_fail = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=gcs_fail_env, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_gcs_fail.returncode, 0)

      no_gcloud_bin = os.path.join(tmp_dir, "no_gcloud_bin")
      os.makedirs(no_gcloud_bin, exist_ok=True)
      os.symlink(shim_py, os.path.join(no_gcloud_bin, "python3"))
      for tool_name in ("dirname", "mkdir", "date", "tee"):
        resolved = shutil.which(tool_name)
        if resolved:
          os.symlink(resolved, os.path.join(no_gcloud_bin, tool_name))
      gcs_missing_bin_env = dict(gcs_ok_env)
      gcs_missing_bin_env["PATH"] = no_gcloud_bin
      proc_no_gcloud = subprocess.run(
          ["/bin/bash", _STAGE_SCRIPT], env=gcs_missing_bin_env, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_no_gcloud.returncode, 0)
      self.assertIn("gcloud was not found", proc_no_gcloud.stderr)

      # Setting TOKENIZER_PATH threads tokenizer_type=huggingface and tokenizer_path=... into all 4 stages.
      Path(calls_log).unlink()
      tok_env = dict(base_env)
      tok_path = os.path.join(tmp_dir, "custom_tokenizer")
      tok_env["TOKENIZER_PATH"] = tok_path
      proc_tok = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=tok_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_tok.returncode, 0, msg=f"stderr={proc_tok.stderr} stdout={proc_tok.stdout}")
      tok_calls = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(tok_calls), 4)
      for call_idx in range(4):
        self.assertIn("tokenizer_type=huggingface", tok_calls[call_idx])
        self.assertIn(f"tokenizer_path={tok_path}", tok_calls[call_idx])

      # Subset STAGES skips omitted stage; failure in unscanned_logits still runs train_e2e and exits non-zero.
      Path(calls_log).unlink()
      sub_env = dict(base_env)
      sub_env["STAGES"] = "scanned_logits,unscanned_logits,train_e2e"
      sub_env["SHIM_FAIL_SUBSTR"] = "run_name=ds41_mm_unscanned_logits"
      proc_fail = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=sub_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_fail.returncode, 0)
      self.assertNotIn("=== STAGE masked_prefill START ", proc_fail.stdout)
      self.assertIn("=== STAGE scanned_logits END ", proc_fail.stdout)
      self.assertIn("=== STAGE unscanned_logits END ", proc_fail.stdout)
      self.assertIn("rc=7 ===", proc_fail.stdout)
      self.assertIn("=== STAGE train_e2e START ", proc_fail.stdout)
      self.assertIn("=== STAGE train_e2e END ", proc_fail.stdout)

      # Negative control: STAGES=masked_prefill exits non-zero when GOLDEN_MM_DIR contains zero M*.npz files.
      mm_l_only_dir = os.path.join(tmp_dir, "mm_l_only")
      os.makedirs(mm_l_only_dir, exist_ok=True)
      Path(mm_l_only_dir, "L1.npz").write_bytes(b"mock_L1")
      prefill_no_m_env = dict(base_env)
      prefill_no_m_env["STAGES"] = "masked_prefill"
      prefill_no_m_env["GOLDEN_MM_DIR"] = mm_l_only_dir
      prefill_no_m_env["OUTPUT_DIR"] = os.path.join(tmp_dir, "out_prefill_no_m")
      proc_prefill_no_m = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=prefill_no_m_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_prefill_no_m.returncode, 0)
      self.assertIn("ERROR: Stage masked_prefill found zero M*.npz", proc_prefill_no_m.stderr)

      for bad_stages in ("unknown_stage", "scanned_logits, unscanned_logits"):
        bad_stage_env = dict(base_env)
        bad_stage_env["STAGES"] = bad_stages
        proc_bad_st = subprocess.run(
            ["bash", _STAGE_SCRIPT], env=bad_stage_env, capture_output=True, text=True, check=False
        )
        self.assertNotEqual(proc_bad_st.returncode, 0)
        self.assertIn("ERROR: Invalid stage token", proc_bad_st.stderr)

      # Missing required env var fails fast before starting any stage.
      bad_env = dict(base_env)
      del bad_env["GOLDEN_TEXT_PKL"]
      proc_missing = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=bad_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_missing.returncode, 0)
      self.assertNotIn("=== STAGE scanned_logits START ", proc_missing.stdout)

      nan_out_dir = os.path.join(tmp_dir, "out_nan")
      nan_env = dict(base_env)
      nan_env["OUTPUT_DIR"] = nan_out_dir
      nan_env["STAGES"] = "train_e2e"
      nan_env["SHIM_NAN_LOSS"] = "1"
      res_nan = subprocess.run(["bash", _STAGE_SCRIPT], env=nan_env, capture_output=True, text=True, check=False)
      self.assertNotEqual(res_nan.returncode, 0, msg=res_nan.stderr)
      nan_summary = json.loads((Path(nan_out_dir) / "train_e2e_summary.json").read_text(encoding="utf-8"))
      self.assertFalse(nan_summary["passed"])
      self.assertFalse(nan_summary["all_losses_finite"])

      # Negative control: non-positive loss fails train_e2e gate; TPU_WORKER_ID=1 stays non-rank0.
      zero_out_dir = os.path.join(tmp_dir, "out_zero_loss")
      res_zero = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env={**nan_env, "OUTPUT_DIR": zero_out_dir, "SHIM_NAN_LOSS": "0", "SHIM_ZERO_LOSS": "1"},
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(res_zero.returncode, 0)
      zero_summary = json.loads((Path(zero_out_dir) / "train_e2e_summary.json").read_text(encoding="utf-8"))
      self.assertFalse(zero_summary["passed"])

      w1_out_dir = os.path.join(tmp_dir, "out_tpu_worker_1")
      Path(gcloud_log).write_text("", encoding="utf-8")
      w1_env = dict(base_env)
      for k in ("WORKER_ID", "JOB_COMPLETION_INDEX"):
        w1_env.pop(k, None)
      w1_env["TPU_WORKER_ID"] = "1"
      w1_env["OUTPUT_DIR"] = w1_out_dir
      w1_env["STAGES"] = "scanned_logits,train_e2e"
      w1_env["GCS_LOG_PREFIX"] = "gs://example-bucket/ds41_w1_logs"
      res_w1 = subprocess.run(["bash", _STAGE_SCRIPT], env=w1_env, capture_output=True, text=True, check=False)
      self.assertEqual(res_w1.returncode, 0, msg=f"stderr={res_w1.stderr}")
      self.assertFalse((Path(w1_out_dir) / "train_e2e_summary.json").exists())
      w1_gcs_lines = Path(gcloud_log).read_text(encoding="utf-8").splitlines()
      w1_dsts = [json.loads(ln)[-1] for ln in w1_gcs_lines if ln.strip()]
      self.assertTrue(w1_dsts)
      self.assertTrue(
          all(Path(d).name.startswith("worker_1_") for d in w1_dsts),
          msg=f"Non-worker0 canonical upload: {w1_dsts}",
      )

  def test_stage_runner_resume_passed_stages_and_per_stage_gcs_sync(self) -> None:
    """Verifies `RESUME_PASSED_STAGES=1` skips passed stages, re-runs failed stages, and checks Orbax items."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      stub_bin = os.path.join(tmp_dir, "stub_bin")
      os.makedirs(stub_bin, exist_ok=True)
      calls_log = os.path.join(tmp_dir, "resume_calls.jsonl")
      gcloud_log = os.path.join(tmp_dir, "resume_gcloud.jsonl")

      stub_py = os.path.join(stub_bin, "python3")
      stub_py_code = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

if len(sys.argv) > 1 and sys.argv[1] == "-":
  os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])

with open({calls_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

for arg in sys.argv[2:]:
  if arg.startswith("--summary_json=") or arg.startswith("--output_json="):
    summary_dest = Path(arg.split("=", 1)[1])
    summary_dest.parent.mkdir(parents=True, exist_ok=True)
    summary_dest.write_text(json.dumps({{"passed": True, "written_by": "stub"}}), encoding="utf-8")
if "train.py" in " ".join(sys.argv[1:]):
  for arg in sys.argv[2:]:
    if arg.startswith("metrics_file="):
      mpath = Path(arg.split("=", 1)[1])
      mpath.parent.mkdir(parents=True, exist_ok=True)
      lines = [json.dumps({{"step": s, "learning/loss": 10.0 - 0.1 * s}}) for s in range(20)]
      mpath.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
sys.exit(0)
"""
      Path(stub_py).write_text(stub_py_code, encoding="utf-8")
      os.chmod(stub_py, stat.S_IRWXU)

      stub_gcloud = os.path.join(stub_bin, "gcloud")
      stub_gcloud_code = f"""#!{sys.executable}
import json
import sys

with open({gcloud_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)
sys.exit(0)
"""
      Path(stub_gcloud).write_text(stub_gcloud_code, encoding="utf-8")
      os.chmod(stub_gcloud, stat.S_IRWXU)

      mm_dir = os.path.join(tmp_dir, "mm_goldens")
      os.makedirs(mm_dir, exist_ok=True)
      Path(mm_dir, "L1.npz").write_bytes(b"mock_L1")
      Path(mm_dir, "M1.npz").write_bytes(b"mock_M1")

      resume_dir = Path(tmp_dir, "gcs_resume_summaries")
      resume_dir.mkdir(parents=True, exist_ok=True)
      (resume_dir / "scanned_logits_summary.json").write_text(
          json.dumps({"passed": True, "origin": "cached_scanned"}), encoding="utf-8"
      )
      (resume_dir / "unscanned_logits_summary.json").write_text(
          json.dumps({"passed": False, "origin": "failed_unscanned"}), encoding="utf-8"
      )

      out_logits = Path(tmp_dir, "out_logits_resume")
      env_logits = os.environ.copy()
      env_logits["PATH"] = f"{stub_bin}:{env_logits['PATH']}"
      env_logits["PYTHONPATH"] = f"{_REPO_ROOT.as_posix()}:{(_REPO_ROOT / 'src').as_posix()}"
      env_logits["HF_CKPT_PATH"] = os.path.join(tmp_dir, "hf_ckpt")
      env_logits["OUTPUT_DIR"] = out_logits.as_posix()
      env_logits["GOLDEN_TEXT_PKL"] = os.path.join(tmp_dir, "golden_full.pkl")
      env_logits["GOLDEN_MM_DIR"] = mm_dir
      env_logits["STAGES"] = "scanned_logits,unscanned_logits"
      env_logits["RESUME_PASSED_STAGES"] = "1"
      env_logits["GCS_RESUME_SUMMARY_PREFIX"] = resume_dir.as_posix()
      env_logits["GCS_LOG_PREFIX"] = "gs://example-bucket/ds41_resume_logs"

      proc_logits = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=env_logits,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_logits.returncode, 0, msg=f"stderr={proc_logits.stderr} stdout={proc_logits.stdout}")
      self.assertIn(
          f"[2_test_deepseek41_mm] Stage scanned_logits already passed in {resume_dir.as_posix()}; "
          "skipping (RESUME_PASSED_STAGES=1).",
          proc_logits.stdout,
      )
      self.assertNotIn("=== STAGE scanned_logits START ", proc_logits.stdout)
      self.assertIn("=== STAGE unscanned_logits START ", proc_logits.stdout)
      self.assertIn("=== STAGE unscanned_logits END ", proc_logits.stdout)

      copied_scanned = json.loads((out_logits / "scanned_logits_summary.json").read_text(encoding="utf-8"))
      self.assertEqual(copied_scanned, {"passed": True, "origin": "cached_scanned"})
      written_unscanned = json.loads((out_logits / "unscanned_logits_summary.json").read_text(encoding="utf-8"))
      self.assertEqual(written_unscanned, {"passed": True, "written_by": "stub"})

      calls_logits = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(calls_logits), 1, msg=f"Unexpected calls: {calls_logits}")
      self.assertEqual(calls_logits[0][:2], ["-m", "tests.utils.deepseek_v41.logit_runner"])
      self.assertIn("run_name=ds41_mm_unscanned_logits", calls_logits[0])

      # Per-stage sync_outputs_to_gcs ran after unscanned_logits (rc==0) AND at script exit.
      gcloud_calls = [json.loads(line) for line in Path(gcloud_log).read_text(encoding="utf-8").splitlines()]
      self.assertGreaterEqual(len(gcloud_calls), 2)

      # train_e2e with EXPORT_ORBAX=1 + ORBAX_EXPORT_PREFIX requires BOTH scanned/0/items and unscanned/0/items to skip.
      (resume_dir / "train_e2e_summary.json").write_text(
          json.dumps({"passed": True, "origin": "cached_train"}), encoding="utf-8"
      )
      orbax_dir = Path(tmp_dir, "orbax_exports")
      (orbax_dir / "scanned" / "0" / "items").mkdir(parents=True, exist_ok=True)

      Path(calls_log).unlink()
      env_train_incomplete = dict(env_logits)
      env_train_incomplete["STAGES"] = "train_e2e"
      env_train_incomplete["EXPORT_ORBAX"] = "1"
      env_train_incomplete["ORBAX_EXPORT_PREFIX"] = orbax_dir.as_posix()
      env_train_incomplete["OUTPUT_DIR"] = os.path.join(tmp_dir, "out_train_rerun")
      proc_train_rerun = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=env_train_incomplete, capture_output=True, text=True, check=False
      )
      self.assertEqual(
          proc_train_rerun.returncode, 0, msg=f"stderr={proc_train_rerun.stderr} stdout={proc_train_rerun.stdout}"
      )
      self.assertIn("=== STAGE train_e2e START ", proc_train_rerun.stdout)
      calls_train_rerun = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(calls_train_rerun), 1)
      self.assertTrue(any("train.py" in a for a in calls_train_rerun[0]))

      # Once unscanned/0/items also exists, train_e2e is skipped cleanly and summary copied to OUTPUT_DIR.
      (orbax_dir / "unscanned" / "0" / "items").mkdir(parents=True, exist_ok=True)
      Path(calls_log).unlink()
      out_train_skip = Path(tmp_dir, "out_train_skip")
      env_train_skip = dict(env_train_incomplete)
      env_train_skip["OUTPUT_DIR"] = out_train_skip.as_posix()
      proc_train_skip = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=env_train_skip, capture_output=True, text=True, check=False
      )
      self.assertEqual(
          proc_train_skip.returncode, 0, msg=f"stderr={proc_train_skip.stderr} stdout={proc_train_skip.stdout}"
      )
      self.assertIn("Stage train_e2e already passed", proc_train_skip.stdout)
      self.assertNotIn("=== STAGE train_e2e START ", proc_train_skip.stdout)
      self.assertFalse(Path(calls_log).exists())
      self.assertEqual(
          json.loads((out_train_skip / "train_e2e_summary.json").read_text(encoding="utf-8")),
          {"passed": True, "origin": "cached_train"},
      )

  def test_optimizer_masks_frozen_vit_with_masked_node_and_active_bias_vl(self) -> None:
    """freeze_vision_encoder_params masks ViT optimizer state via optax.multi_transform; bias_vl stays active."""
    with tempfile.TemporaryDirectory(prefix="ds41_opt_mask_") as tmp_dir:
      cfg = _init_mini_mm_config(
          grid=(7, 5),
          base_output_directory=tmp_dir,
          scan_layers=False,
          freeze_vision_encoder_params=True,
          routed_bias=True,
          routed_bias_update_rate=0.05,
          dataset_type="synthetic",
          max_num_images_per_example=1,
          max_target_length=32,
          per_device_batch_size=1.0,
          steps=1,
          learning_rate_schedule_steps=1,
          learning_rate=1e-2,
          warmup_steps_fraction=0.0,
          adam_weight_decay=0.5,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      tx = optimizers.get_optimizer(cfg, learning_rate_schedule=1e-2, model=model, mesh=mesh)
      state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))
      # Frozen branch in PartitionState is EmptyState (stripped by to_pure_dict), and ViT slots in Adam mu are MaskedNode!
      opt_pure = nnx.to_pure_dict(nnx.state(state.optimizer))
      inner_states = opt_pure["opt_state"]["inner_states"]
      self.assertNotIn("frozen", inner_states)
      trainable_chain = inner_states["trainable"]["inner_state"]
      adam_state = next(v for v in trainable_chain.values() if isinstance(v, dict) and "mu" in v)
      vit_mu = adam_state["mu"]["vision_encoder"]["DeepSeek41VisionEncoder_0"]["patch_embed"]["kernel"]
      proj_mu = adam_state["mu"]["vision_encoder"]["DeepSeek41VisionProjector_0"]["w1"]["kernel"]
      self.assertIsInstance(vit_mu, optax.MaskedNode)
      self.assertNotIsInstance(proj_mu, optax.MaskedNode)

      # Nonzero gradients: the optimizer itself (no post-step restore) keeps ViT bit-exact under AdamW decay.
      vit_before = np.array(
          state.model.vision_encoder.DeepSeek41VisionEncoder_0.patch_embed.kernel.get_value(), copy=True
      )
      proj_before = np.array(state.model.vision_encoder.DeepSeek41VisionProjector_0.w1.kernel.get_value(), copy=True)
      fake_grads = jax.tree_util.tree_map(jnp.ones_like, nnx.state(state.model, nnx.Param))
      state.apply_gradients(fake_grads)
      vit_after = np.array(state.model.vision_encoder.DeepSeek41VisionEncoder_0.patch_embed.kernel.get_value(), copy=True)
      proj_after = np.array(state.model.vision_encoder.DeepSeek41VisionProjector_0.w1.kernel.get_value(), copy=True)
      np.testing.assert_array_equal(vit_after, vit_before)
      self.assertFalse(np.array_equal(proj_after, proj_before))
      # Verify MoEBiasVar bias_vl is non-Param (excluded from Adam mu) and untouched by optimizer.apply_gradients.
      moe_layer = next(
          n for _, n in nnx.iter_graph(state.model.decoder) if type(n).__name__ == "GateLogit" and hasattr(n, "bias_vl")
      )
      self.assertIsNotNone(moe_layer.bias_vl)
      np.testing.assert_array_equal(
          np.asarray(moe_layer.bias_vl.get_value()), np.zeros_like(np.asarray(moe_layer.bias_vl.get_value()))
      )


if __name__ == "__main__":
  unittest.main()
