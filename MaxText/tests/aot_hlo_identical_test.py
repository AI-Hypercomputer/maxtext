"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tempfile
import unittest
import os
import shutil
import hashlib
import sys
import subprocess
from MaxText.globals import PKG_DIR


class AotHloIdenticalTest(unittest.TestCase):

  def run_compile_and_real(self, aot_dump_dir, real_dump_dir, extra_config_args):
    """Runs AOT and real runs and returns the completed process object."""
    shared_args = [
        "MaxText/configs/base.yml",
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=compile_equivalent_test",
        "dataset_type=synthetic",
        "steps=1",
        "enable_checkpointing=False",
    ]
    if extra_config_args:
      shared_args.append(extra_config_args)

    try:
      # Step 1: Grab device info for train_compile
      device_info_command = "import jax; print(f'{jax.devices()[0].device_kind},{len(jax.devices())}')"
      completed_process = subprocess.run(
          ["python3", "-c", device_info_command],
          check=True,
          capture_output=True,
          text=True,
      )
      device_kind, num_devices_str = completed_process.stdout.strip().split(",")
      num_devices = int(num_devices_str)
    except (subprocess.CalledProcessError, RuntimeError, IndexError, ValueError) as e:
      self.fail(f"Failed to get device info: {e}")

    topology = ""
    if device_kind == "TPU v4":
      topology = f"v4-{num_devices * 2}"
    elif device_kind == "TPU v5 lite":
      topology = f"v5e-{num_devices}"
    elif device_kind == "TPU v5":
      topology = f"v5p-{num_devices * 2}"
    elif device_kind in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-40GB"):
      shared_args.append("attention=dot_product")
      topology = "a3"
    elif device_kind in ("cpu", "TFRT_CPU"):
      self.fail("AOT HLO comparison test is not intended for CPU.")
    else:
      self.fail(f"Unsupported device kind for dynamic topology: {device_kind}")

    aot_args = [f"compile_topology={topology}", "compile_topology_num_slices=1"]

    xla_dump_options = "--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_train_step"

    # Step 2: Run AOT compile
    compile_env = os.environ.copy()
    compile_env["XLA_FLAGS"] = f"--xla_dump_to={aot_dump_dir} {xla_dump_options}"
    compile_cmd = ["python3", "-m", "MaxText.train_compile"] + shared_args + aot_args

    print(f"Running AOT compile command: {' '.join(compile_cmd)}", flush=True)
    try:
      subprocess.run(
          compile_cmd,
          check=True,
          text=True,
          cwd=os.path.dirname(PKG_DIR),
          env=compile_env,
          stdout=sys.stdout,
          stderr=sys.stderr,
      )
    except subprocess.CalledProcessError as e:
      print(f"Error running AOT compile script: {e.returncode}", flush=True)
      return None

    # Step 3: Run real train
    train_env = os.environ.copy()
    train_env["XLA_FLAGS"] = f"--xla_dump_to={real_dump_dir} {xla_dump_options}"
    train_cmd = ["python3", "-m", "MaxText.train"] + shared_args

    print(f"Running real train command: {' '.join(train_cmd)}", flush=True)
    try:
      result_train = subprocess.run(
          train_cmd, check=True, text=True, cwd=os.path.dirname(PKG_DIR), env=train_env, stdout=sys.stdout, stderr=sys.stderr
      )
      return result_train
    except subprocess.CalledProcessError as e:
      print(f"Error running real train script: {e.returncode}", flush=True)
      return None

  def find_matched_files(self, compile_dump_dir, real_dump_dir):
    """
    Find all files in compile and train HLO graphs
    that share the same name and match specified patterns.
    The pattern is derived from HybridSim.
    """
    patterns = {
        ".after_optimizations_before_buffer_assignment.txt",
        ".execution_options.txt",
        ".target_arguments.txt",
        ".flagfile",
        ".tpu_comp_env.txt",
    }
    compile_files = set(os.listdir(compile_dump_dir))
    real_files = set(os.listdir(real_dump_dir))
    overlap_files = compile_files.intersection(real_files)
    return [f for f in overlap_files if any(f.endswith(p) for p in patterns)]

  def delete_dir(self, directory):
    if os.path.exists(directory):
      shutil.rmtree(directory)

  def check_large_files_equal(self, file_path1, file_path2):
    """Asserts that two potentially large text files have identical content."""
    h1 = hashlib.sha256()
    h2 = hashlib.sha256()

    with open(file_path1, "rb") as f1:
      for chunk in iter(lambda: f1.read(8192), b""):
        h1.update(chunk)

    with open(file_path2, "rb") as f2:
      for chunk in iter(lambda: f2.read(8192), b""):
        h2.update(chunk)

    return h1.hexdigest() == h2.hexdigest()

  def assert_compile_and_real_match_hlo(self, test_name, extra_config_args):
    """check that AOT compiled and trained HLO files are identical for a given test"""
    temp_dir = tempfile.gettempdir()
    compile_dump_dir = os.path.join(temp_dir, "compile_test_xla_dump", test_name, "aot", "")
    train_dump_dir = os.path.join(temp_dir, "compile_test_xla_dump", test_name, "real", "")
    self.delete_dir(compile_dump_dir)  # Ensure directories empty before use
    self.delete_dir(train_dump_dir)

    result = self.run_compile_and_real(compile_dump_dir, train_dump_dir, extra_config_args)
    if not result:
      self.fail("AOT compile and/or real train failed to run. Check logs for details.")

    matched_files = self.find_matched_files(compile_dump_dir, train_dump_dir)
    print(f"There are {len(matched_files)} matched HLO files for test {test_name}!")
    assert len(matched_files) > 0, f"No matched HLO files found for test {test_name}!"

    for file_name in matched_files:
      compile_file_path = os.path.join(compile_dump_dir, file_name)
      train_file_path = os.path.join(train_dump_dir, file_name)
      assert self.check_large_files_equal(
          compile_file_path, train_file_path
      ), f"HLO file is not identical for {file_name} for test {test_name}!"

    self.delete_dir(compile_dump_dir)  # Cleanup directories after use
    self.delete_dir(train_dump_dir)

    print("AOT Compiled and train HLO files are identical for test {test_name}!")

  def test_default_hlo_match(self):
    self.assert_compile_and_real_match_hlo("default_run", None)

  def test_int8_hlo_match(self):
    self.assert_compile_and_real_match_hlo("int8", "quantization=int8")
