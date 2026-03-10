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
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


def get_test_config_path(relative_path: str = "base.yml"):
  """Returns the absolute path for a test config.

  If `relative_path` is `base.yml`, applies the decoupled-mode logic and returns
  `decoupled_base_test.yml` when decoupled, otherwise `base.yml`.
  """
  if relative_path == "base.yml":
    base_cfg = "decoupled_base_test.yml" if is_decoupled() else "base.yml"
    return os.path.join(MAXTEXT_CONFIGS_DIR, base_cfg)
  return os.path.join(MAXTEXT_CONFIGS_DIR, relative_path)


def get_decoupled_parallelism_overrides(
    *,
    fsdp_parallelism=None,
    include_mesh_defaults: bool = False,
    as_argv: bool = False,
):
  """Return decoupled-only overrides for ICI parallelism (kwargs or argv).

  - **kwargs mode** (`as_argv=False`): returns a dict suitable for `pyconfig.initialize(..., **overrides)`.
  - **argv mode** (`as_argv=True`): returns a list like `["ici_fsdp_parallelism=8"]` to append to argv.

  Args:
    fsdp_parallelism: If None, uses `jax.device_count()`; otherwise coerces to int.
    include_mesh_defaults: When True, also sets `mesh_axes=["data"]` and `ici_data_parallelism=-1`.
    as_argv: When True, return argv strings; otherwise return kwargs dict.
  """
  if not is_decoupled():
    return [] if as_argv else {}

  try:
    import jax  # pylint: disable=import-outside-toplevel

    overrides = {"ici_fsdp_parallelism": jax.device_count() if fsdp_parallelism is None else int(fsdp_parallelism)}
    if include_mesh_defaults:
      overrides.setdefault("mesh_axes", ["data"])
      overrides.setdefault("ici_data_parallelism", -1)

    if as_argv:
      return [f"{k}={v}" for k, v in overrides.items()]
    return overrides
  except (ImportError, ValueError, TypeError):  # pragma: no cover - defensive
    return [] if as_argv else {}


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
  """
  if is_decoupled():
    return os.path.join("maxtext_local_output", "gcloud_decoupled_test_logs")
  return cloud_path or "gs://runner-maxtext-logs"


__all__ = [
    "get_test_base_output_directory",
    "get_decoupled_parallelism_overrides",
    "is_rocm_backend",
    "get_test_config_path",
    "get_post_train_test_config_path",
    "get_test_dataset_path",
]
