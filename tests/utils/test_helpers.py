# Copyright 2023-2026 Google LLC
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

"""Test helpers file for helper for test configuration path selection.

Provides helpers to return common test configuration values. When running in
decoupled mode (DECOUPLE_GCLOUD=TRUE), these helpers return local paths instead
of Google Cloud Storage paths.
"""

import os
import shutil
import subprocess
import unittest
import filelock
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR


def ensure_tokenizer_downloaded(
    tokenizer_name: str,
    target_path: str | None = None,
    skip_test_on_failure: bool = False,
) -> str:
  """Ensures a tokenizer directory exists locally and is non-empty, downloading from GCS if missing.

  Args:
    tokenizer_name: Name of the tokenizer folder in gs://maxtext-dataset/hf/ (e.g. 'llama2-chat-tokenizer').
    target_path: Optional local target path. Defaults to os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", tokenizer_name).
    skip_test_on_failure: If True, raises unittest.SkipTest on download failure instead of RuntimeError.

  Returns:
    The local path to the tokenizer directory.
  """
  if target_path is None:
    target_path = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", tokenizer_name)

  lock_path = target_path + ".lock"
  with filelock.FileLock(lock_path):
    if not os.path.exists(target_path) or not os.listdir(target_path):
      os.makedirs(os.path.dirname(target_path), exist_ok=True)
      exit_code = subprocess.call(
          [
              "gcloud",
              "storage",
              "cp",
              "--recursive",
              f"gs://maxtext-dataset/hf/{tokenizer_name}",
              os.path.join(os.path.dirname(target_path), ""),
          ]
      )
      if exit_code != 0:
        shutil.rmtree(target_path, ignore_errors=True)
        msg = f"Failed to download {tokenizer_name} from GCS with exit code {exit_code}"
        if skip_test_on_failure:
          raise unittest.SkipTest(f"Skipping test: {msg}")
        raise RuntimeError(msg)
  return target_path


def get_test_config_path(relative_path: str = "base.yml"):
  """Returns the absolute path for a test config.

  If `relative_path` is `base.yml`, applies the decoupled-mode logic and returns
  `decoupled_base_test.yml` when decoupled, otherwise `base.yml`.
  """
  if relative_path == "base.yml":
    base_cfg = "decoupled_base_test.yml" if is_decoupled() else "base.yml"
    return os.path.join(MAXTEXT_CONFIGS_DIR, base_cfg)
  return os.path.join(MAXTEXT_CONFIGS_DIR, relative_path)


def is_rocm_backend() -> bool:
  """Best-effort ROCm detection without internal JAX APIs."""
  try:
    import jax  # pylint: disable=import-outside-toplevel

    gpu = jax.devices("gpu")[0]
    return "rocm" in str(gpu).lower()
  except (ImportError, RuntimeError, IndexError):  # pragma: no cover - defensive
    return False


def get_post_train_test_config_path(sub_type="sft"):
  """Return absolute path to the chosen test config file.

  Returns `decoupled_base_test.yml` when decoupled, otherwise `base.yml`.
  """
  base_cfg = "rl.yml" if sub_type == "rl" else "sft.yml"
  return os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", base_cfg)


def get_test_dataset_path(cloud_path=None):
  """Return the dataset path for tests.

  Args:
    cloud_path: Optional custom GCS path to use in cloud mode.
                Defaults to "gs://maxtext-dataset" if not specified.

  Returns:
    Local minimal dataset path when decoupled, otherwise returns
    the specified cloud path or default GCS maxtext-dataset bucket.
  """
  if is_decoupled():
    return os.path.join("tests", "assets", "local_datasets", "c4_en_dataset_minimal")
  return cloud_path or "gs://maxtext-dataset"


def get_test_base_output_directory(cloud_path=None):
  """Return the base output directory for test logs and checkpoints.

  Args:
    cloud_path: Optional custom GCS path to use in cloud mode.
                Defaults to "gs://runner-maxtext-logs" if not specified.

  Returns:
    Local test logs directory when decoupled, otherwise returns
    the specified cloud path or default GCS runner-maxtext-logs bucket.
    The local path is absolute so Orbax (which rejects relative
    checkpoint paths) can write checkpoints into it.
  """
  if is_decoupled():
    return os.path.abspath(os.path.join("maxtext_local_output", "gcloud_decoupled_test_logs"))
  return cloud_path or "gs://runner-maxtext-logs"


__all__ = [
    "ensure_tokenizer_downloaded",
    "get_test_base_output_directory",
    "is_rocm_backend",
    "get_test_config_path",
    "get_post_train_test_config_path",
    "get_test_dataset_path",
]
