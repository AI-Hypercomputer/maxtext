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

"""Tests for JAX compilation cache hits in train.py.

This test ensures that the `train_step` function is compiled only once.
It verifies that the Ahead-Of-Time (AOT) compilation signature (which uses
a dummy `shaped_batch` constructed in `train.py`) matches the runtime
compilation signature (which uses the actual `example_batch` from the data pipeline).

If this test fails, it likely means a regression was introduced where the AOT
batch sharding/shape does not match the runtime batch sharding/shape. This causes
JAX to recompile `train_step` at step 0, leading to a "double compilation"
and a very slow first step.

To debug:
1. Verify that `maxtext_utils.get_shaped_batch` in `train.py` is called with the
   correct `sharding` argument (matching the data pipeline sharding).
2. Check if there are differences in shapes or dtypes between the AOT dummy batch
   and the runtime batch.
"""

import os
import tempfile
import shutil
import pytest
import subprocess
import sys

from tests.utils.test_helpers import (
    get_test_config_path,
    get_test_base_output_directory,
)


@pytest.mark.cpu_only
def test_train_step_cache_hit():
  temp_dir = tempfile.mkdtemp()
  _base_output_directory = get_test_base_output_directory()

  try:
    small_model_overrides = [
        "base_emb_dim=16",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "base_mlp_dim=16",
        "base_num_decoder_layers=1",
        "head_dim=64",
        "max_target_length=64",
        "vocab_size=32",
        "sharding_tolerance=0.1",
    ]

    cmd = [
        sys.executable,
        "-m",
        "maxtext.trainers.pre_train.train",
        get_test_config_path(),
        f"base_output_directory={_base_output_directory}",
        "run_name=compile_cache_test_cpu",
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "dataset_type=synthetic",
        "hardware=cpu",
        "skip_jax_distributed_system=True",
        f"jax_cache_dir={temp_dir}",
    ] + small_model_overrides

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    env["JAX_COMPILATION_CACHE_DIR"] = temp_dir
    env["JAX_LOG_COMPILES"] = "1"

    print("Running CPU training subprocess:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    captured_logs = result.stderr

    # Print captured logs for debugging (will be shown by pytest if assert fails)
    print("=== Captured Subprocess Stderr ===")
    print(captured_logs)
    print("===================================")

    # Check if cache dir has files
    cache_files = os.listdir(temp_dir)
    train_step_cache_files = [f for f in cache_files if f.startswith("jit_train_step")]
    print("=== Cache Directory Content ===")
    print(f"Path: {temp_dir}")
    print(f"Files: {cache_files}")
    print(f"Train step cache files: {train_step_cache_files}")
    print("===============================")

    assert len(cache_files) > 0, (
        "JAX compilation cache directory is empty. This suggests the compilation "
        "cache was not writeable or the JAX cache configuration was ignored."
    )

    assert len(train_step_cache_files) == 1, (
        f"Expected exactly 1 jit_train_step JAX compilation cache file, but found "
        f"{len(train_step_cache_files)}: {train_step_cache_files}. "
        f"All cache files: {cache_files}. "
        "This indicates a cache miss where AOT compilation and runtime execution generated "
        "different keys, causing train_step to be compiled twice (double-compilation regression)."
    )

    assert "Persistent compilation cache hit for 'jit_train_step'" in captured_logs, (
        "Did not find 'Persistent compilation cache hit for 'jit_train_step'' in logs. "
        "This means the runtime execution of train_step did not hit the cache populated by the AOT compilation. "
        "Check if the AOT input batch signature (shape/dtype/sharding) matches the runtime input batch."
    )

  finally:
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)
