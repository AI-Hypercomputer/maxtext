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
import re
import jax
from MaxText.globals import PKG_DIR
import MaxText.train_compile as train_compile
import MaxText.train as train


class AotHloIdenticalTestSimple(unittest.TestCase):

  def setUp(self):
    """
    Fix the dump dir and xla flags
    """
    jax.config.update("jax_enable_compilation_cache", False)
    temp_dir = tempfile.gettempdir()
    self.dump_dir = os.path.join(temp_dir, "aot_test_dump")
    xla_dump_options = "--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_train_step"
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={self.dump_dir} {xla_dump_options}"

  def find_HLO_files(self, compile_dump_dir, real_dump_dir):
    """
    Find the HLO file
    """
    pattern = re.compile(r"^.*\.jit_train_step\..*\.after_optimizations_after_buffer_assignment\.txt$")
    compile_files = set(os.listdir(compile_dump_dir))
    real_files = set(os.listdir(real_dump_dir))
    compile_hlo, real_hlo = None, None
    # HLO file satisfying above pattern should uniquely exist
    for file in compile_files:
      if pattern.search(file):
        compile_hlo = file
    for file in real_files:
      if pattern.search(file):
        real_hlo = file
    return compile_hlo, real_hlo

  def delete_dir(self, *directories):
    for directory in directories:
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
    shared_args = [
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=compile_equivalent_test",
        "dataset_type=synthetic",
        "steps=1",
        "enable_checkpointing=False",
    ]
    
    train_dump_dir = os.path.join(temp_dir, "compile_test_xla_dump", test_name, "real", "")
    train_argv = (None, os.path.join(PKG_DIR, "configs", "base.yml")) + tuple(shared_args)
    # TODO (chengnuojin) add supports on more TPU devices
    aot_args = [f"compile_topology=v4-8", "compile_topology_num_slices=1"]
    compile_argv = (None, os.path.join(PKG_DIR, "configs", "base.yml")) + tuple(shared_args) + tuple(aot_args)
    compile_dump_dir = os.path.join(temp_dir, "compile_test_xla_dump", test_name, "aot", "")

    # Cleanup directories before use
    self.delete_dir(*(self.dump_dir, compile_dump_dir, train_dump_dir))

    # Step 1: generate train.py HLO graphs
    try:
      train.main(train_argv)
    except Exception as e:
      self.fail(f"Error running real train script: {e}", flush=True)
    finally:
      shutil.move(self.dump_dir, train_dump_dir)
      jax.clear_caches()

    # Step 2: generate train_compile.py HL graphs
    try:
      train_compile.main(compile_argv)
    except Exception as e:
      self.fail(f"Error running AOT train script: {e}", flush=True)
    finally:
      shutil.move(self.dump_dir, compile_dump_dir)
      jax.clear_caches()

    # Step 3: specify the HLO files and check their homogeneity
    compile_hlo, real_hlo = self.find_HLO_files(compile_dump_dir, train_dump_dir)
    assert compile_hlo is not None, f"No HLO files found in train compile!"
    assert real_hlo is not None, f"No HLO files found in train!"

    compile_file_path = os.path.join(compile_dump_dir, compile_hlo)
    train_file_path = os.path.join(train_dump_dir, real_hlo)
    assert self.check_large_files_equal(
        compile_file_path, train_file_path
    ), f"HLO file is not identical for test {test_name}!"

    self.delete_dir(*(self.dump_dir, compile_dump_dir, train_dump_dir))

    print("AOT Compiled and train HLO files are identical for test {test_name}!")

  def test_default_hlo_match(self):
    self.assert_compile_and_real_match_hlo("default_run", None)

  def test_int8_hlo_match(self):
    self.assert_compile_and_real_match_hlo("int8", "quantization=int8")
