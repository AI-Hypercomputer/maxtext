# Copyright 2025 Google LLC
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

"""
These tests verify the Compile-Then-Load workflow.
It ensures that a model compiled via train_compile.py can be successfully
loaded and executed by train.py.
"""

import tempfile
import unittest
import pytest
import os
import shutil
import jax
from tests.utils.test_helpers import get_test_config_path
from MaxText import train_compile
from MaxText import train


class CompileThenLoadTest(unittest.TestCase):
  """Tests for the Split Compile and Train workflow"""

  def setUp(self):
    """Create a temporary directory for the compiled pickle file."""
    self.temp_dir = tempfile.mkdtemp()
    self.pickle_file = os.path.join(self.temp_dir, "test_compiled_train.pickle")

    # Ensure JAX cache doesn't interfere with clean test runs
    jax.config.update("jax_enable_compilation_cache", False)

  def tearDown(self):
    """Clean up the temporary directory."""
    if os.path.exists(self.temp_dir):
      shutil.rmtree(self.temp_dir)

  def get_device_user_facing_name(self):
    """Gets TPU device user facing name to generate correct AOT arguments."""
    devices = jax.devices()
    if not devices or "tpu" not in devices[0].platform.lower():
      pytest.skip("This test requires a TPU environment.")

    num_devices = len(devices)
    device_kind = devices[0].device_kind
    device_info = {
        "TPU v4": ("v4", 2 * num_devices),
        "TPU v5 lite": ("v5e", num_devices),
        "TPU v5": ("v5p", 2 * num_devices),
        "TPU v6": ("v6e", num_devices),
    }

    prefix, topology_devices = next((v for k, v in device_info.items() if k in device_kind), (None, None))
    if prefix is None:
      raise ValueError(f"Unsupported TPU device kind for AOT test: {device_kind}")

    return f"{prefix}-{topology_devices}"

  def run_compile_then_load(self, test_name, *extra_args):
    """
    Executes the compile step, checks for pickle existence,
    then executes the load/train step.
    """

    # Common arguments derived from your request
    shared_args = [
        "global_parameter_scale=1",
        "per_device_batch_size=4",
        "steps=1",
        "learning_rate=1e-3",
        "dataset_type=synthetic",
        "enable_checkpointing=False",
    ]

    if extra_args:
      shared_args.extend(extra_args)

    # Compilation
    topology = self.get_device_user_facing_name()

    compile_specific_args = [
        f"compile_topology={topology}",
        "compile_topology_num_slices=1",
        f"compiled_trainstep_file={self.pickle_file}",
    ]

    compile_argv = (None, get_test_config_path()) + tuple(shared_args) + tuple(compile_specific_args)

    print(f"\n--- Starting Compilation Step for {test_name} ---")
    # Clear caches before compile to ensure clean state
    jax.clear_caches()
    train_compile.main(compile_argv)

    # Assert the pickle file was actually created
    assert os.path.exists(self.pickle_file), f"Compilation failed: {self.pickle_file} was not created."

    load_specific_args = [
        "base_output_directory=gs://runner-maxtext-logs",
        f"run_name=compile_then_load_{test_name}",
        f"compiled_trainstep_file={self.pickle_file}",
    ]

    train_argv = (None, get_test_config_path()) + tuple(shared_args) + tuple(load_specific_args)

    print(f"\n--- Starting Load/Train Step for {test_name} ---")
    # Clear caches before train to ensure we are actually loading from the pickle
    jax.clear_caches()
    train.main(train_argv)

    print(f"Successfully compiled and loaded for test {test_name}!")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_default_compile_load(self):
    self.run_compile_then_load("default_run")
