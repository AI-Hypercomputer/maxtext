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

"""Static verification tests for DeepSeek-V4 E2E test scripts."""

import importlib.util
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT_PATHS = {
    "v4_284b_1": REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4-284b/1_test_deepseek.sh",
    "v4_284b_2": REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4-284b/2_test_deepseek.sh",
    "v4_1600b_1": REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4-1600b/1_test_deepseek.sh",
    "v4_1600b_2": REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4-1600b/2_test_deepseek.sh",
}


class DeepSeekV4E2EScriptTest(unittest.TestCase):
  """Static assertions for DeepSeek-V4 284B and 1600B end-to-end scripts."""

  def test_scripts_exist(self):
    for key, path in SCRIPT_PATHS.items():
      self.assertTrue(path.exists(), f"Script {key} does not exist: {path}")
      self.assertTrue(path.is_file(), f"Script {key} is not a file: {path}")

  def test_bash_syntax_valid(self):
    for key, path in SCRIPT_PATHS.items():
      res = subprocess.run(
          ["bash", "-n", str(path)],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(res.returncode, 0, f"bash -n failed for {key} ({path}):\n{res.stderr}")

  def test_v4_1600b_no_284b_references(self):
    for key in ("v4_1600b_1", "v4_1600b_2"):
      path = SCRIPT_PATHS[key]
      content = path.read_text()
      self.assertNotIn(
          "deepseek4-284b",
          content,
          f"deepseek4-284b reference found in {key} ({path})",
      )

  def test_indexer_topk_defaults(self):
    content_284b = SCRIPT_PATHS["v4_284b_2"].read_text()
    match_284b = re.search(r"INDEXER_TOPK[:=].*?(\d+)", content_284b)
    self.assertIsNotNone(match_284b, "INDEXER_TOPK default not found in v4-284b/2_test_deepseek.sh")
    self.assertEqual(
        match_284b.group(1),
        "512",
        f"v4-284b INDEXER_TOPK should default to 512, got {match_284b.group(1)}",
    )

    content_1600b = SCRIPT_PATHS["v4_1600b_2"].read_text()
    match_1600b = re.search(r"INDEXER_TOPK[:=].*?(\d+)", content_1600b)
    self.assertIsNotNone(match_1600b, "INDEXER_TOPK default not found in v4-1600b/2_test_deepseek.sh")
    self.assertEqual(
        match_1600b.group(1),
        "1024",
        f"v4-1600b INDEXER_TOPK should default to 1024, got {match_1600b.group(1)}",
    )

  def test_v4_e2e_no_legacy_transformers_pin_and_284b_golden(self):
    for key in ("v4_284b_2", "v4_1600b_2"):
      content = SCRIPT_PATHS[key].read_text()
      self.assertNotIn(
          "transformers==4.57.3",
          content,
          f"Obsolete transformers==4.57.3 pin found in {key}; downgrades huggingface_hub and breaks datasets",
      )
    c2_284b = SCRIPT_PATHS["v4_284b_2"].read_text()
    self.assertIn("gs://maxtext-test-assets/golden_data_${MODEL_NAME}-topk512.jsonl", c2_284b)
    self.assertIn("Effective --golden_logits_path=", c2_284b)
    self.assertIn('--max_kl_div="${MAX_KL_DIV:-0.42}"', c2_284b)

  def test_no_comments_in_backslash_continuations(self):
    for key, path in SCRIPT_PATHS.items():
      lines = path.read_text().splitlines()
      in_continuation = False
      for i, line in enumerate(lines):
        stripped = line.strip()
        if in_continuation and stripped.startswith("#"):
          self.fail(f"Found comment line inside backslash continuation in {key} ({path}) at line {i+1}:\n{line}")
        in_continuation = stripped.endswith("\\")

  def test_v4_1600b_canonical_paths_and_features(self):
    c1 = SCRIPT_PATHS["v4_1600b_1"].read_text()
    self.assertIn("${BASE_OUTPUT_PATH:?", c1)
    self.assertIn("tid2eid.safetensors", c1)
    self.assertIn("deepseek-ai/DeepSeek-V4-Pro-0813", c1)
    self.assertIn("standalone_scripts.deepseek_dequantize", c1)
    self.assertNotIn("--checkpoint_storage_concurrent_gb", c1)
    self.assertIn("checkpoint_storage_concurrent_gb=256", c1)

    c2 = SCRIPT_PATHS["v4_1600b_2"].read_text()
    self.assertIn("gs://maxtext-deepseek/deepseek4-1600b/2026-10-03/scanned/0/items", c2)
    self.assertIn("gs://maxtext-deepseek/deepseek4-1600b/2026-10-03/unscanned/0/items", c2)
    # Tid2EidVar is restored from the checkpoint; there is no tid2eid_path config key.
    self.assertNotIn("tid2eid_path", c2)
    self.assertIn("RUN_SCANNED_LOGIT_CHECK", c2)
    # At most one DCN axis may be -1.
    self.assertIn('dcn_data_parallelism="${PRETRAIN_DCN_DP:-1}"', c2)
    self.assertEqual(c2.count('ici_expert_parallelism="${ICI_EXPERT_PARALLELISM:-16}"'), 5)
    self.assertEqual(c2.count('activations_in_float32="${LOGIT_ACTIVATIONS_IN_FLOAT32:-false}"'), 2)
    self.assertNotIn("activations_in_float32=true", c2)
    self.assertIn("gs://maxtext-test-assets/golden_data_deepseek4-1600b.jsonl", c2)
    self.assertIn("save_checkpoint_on_start=false", c2)
    self.assertIn("scan_layers=True", c2)
    self.assertIn("scan_layers=False", c2)
    self.assertIn("Effective --golden_logits_path=", c2)
    self.assertIn('steps="${STEPS:-20}"', c2)
    self.assertIn('max_target_length="${MAX_TARGET_LENGTH:-1024}"', c2)

  def test_decode_prefill_shorter_than_target(self):
    # pyconfig rejects max_target_length <= max_prefill_predict_length.
    def default(text, flag):
      m = re.search(flag + r'="?(?:\$\{[A-Z_]+:-)?(\d+)', text)
      self.assertIsNotNone(m, flag)
      return int(m.group(1))

    for name in ("v4_284b_2", "v4_1600b_2"):
      decode = SCRIPT_PATHS[name].read_text().split("maxtext.inference.decode", 1)[1]
      prefill = default(decode, "max_prefill_predict_length")
      target = default(decode, "max_target_length")
      self.assertLess(prefill, target, name)

  def test_decode_assert_defaults(self):
    # Official greedy prefixes for the shared BOS-first prompt (V4-Pro emits a leading "O").
    expected = {
        "v4_284b_2": "${DECODE_ASSERT:- mapping a query",
        "v4_1600b_2": "${DECODE_ASSERT:-O mapping a query",
    }
    for name, prefix in expected.items():
      decode = SCRIPT_PATHS[name].read_text().split("maxtext.inference.decode", 1)[1]
      self.assertIn(f'autoregressive_decode_assert="{prefix}', decode, name)

  def test_pretrain_extra_flags_override_hardcoded_keys(self):
    # pyconfig keeps the last duplicate key, so PRETRAIN_EXTRA_FLAGS must follow per_device_batch_size=1.
    for name in ("v4_284b_2", "v4_1600b_2"):
      cmd = SCRIPT_PATHS[name].read_text().split("maxtext.trainers.pre_train.train", 1)[1].split("tee /tmp/", 1)[0]
      self.assertGreater(cmd.index("${PRETRAIN_EXTRA_FLAGS:-}"), cmd.index("per_device_batch_size=1"), name)

  def test_v4_1600b_sft_stage(self):
    c2 = SCRIPT_PATHS["v4_1600b_2"].read_text()
    sft = c2.split("=== STAGE: sft ===", 1)[1].split("=== STAGE: decode ===", 1)[0]
    self.assertIn("maxtext.trainers.post_train.sft.train_sft_native", sft)
    self.assertIn("/post_train/sft.yml", sft)
    self.assertIn('model_name="${MODEL_NAME}"', sft)
    self.assertIn('load_parameters_path="${SCANNED_CKPT_PATH}"', sft)
    self.assertIn('ici_expert_parallelism="${ICI_EXPERT_PARALLELISM:-16}"', sft)
    self.assertIn("opt_type=sgd", sft)
    self.assertIn("learning_rate=1e-6", sft)
    self.assertIn("gs://maxtext-dataset/hf/ultrachat_200k/train_sft-*.parquet", sft)
    self.assertIn("tests/end_to_end/tpu/deepseek/sft_eos_probe.py", sft)
    self.assertLess(sft.index("sft_eos_probe.py"), sft.index("train_sft_native"))

  def test_v4_1600b_sft_loss_check_runs_under_awk(self):
    """Runs the SFT stage's loss check on sample logs, catching quoting or regex errors in the script."""
    if shutil.which("awk") is None:
      self.skipTest("awk not installed")
    stage = SCRIPT_PATHS["v4_1600b_2"].read_text().split("=== STAGE: sft ===", 1)[1]
    program = stage.split('if ! awk -F"loss: " \'', 1)[1].split("' /tmp/sft_output.log", 1)[0]
    cases = {
        "completed step: 0, seconds: 1.0, loss: 1.062, lm_loss: 1.0\ncompleted step: 1, loss: 1.061\n": 0,
        "completed step: 0, loss: nan\n": 1,
        "completed step: 0, loss: 0.000\n": 1,
        "completed step: 0, loss: -inf\n": 1,
        "no step lines\n": 1,
    }
    for log, expected_rc in cases.items():
      with self.subTest(log=log), tempfile.NamedTemporaryFile("w", suffix=".log") as f:
        f.write(log)
        f.flush()
        result = subprocess.run(["awk", "-F", "loss: ", program, f.name], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, expected_rc, result.stdout + result.stderr)

  def test_sft_eos_probe_counts_masked_eos(self):
    spec = importlib.util.spec_from_file_location(
        "sft_eos_probe", REPO_ROOT / "tests/end_to_end/tpu/deepseek/sft_eos_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    batch = {"targets": np.array([[5, 1, 2, 1]]), "targets_segmentation": np.array([[1, 1, 0, 0]])}
    self.assertEqual(probe.count_eos_targets(batch, eos_id=1), (2, 1))


if __name__ == "__main__":
  unittest.main()
