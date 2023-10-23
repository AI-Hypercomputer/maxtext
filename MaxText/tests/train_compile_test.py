"""
 Copyright 2023 Google LLC

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

""" Tests for the common Max Utils """
import jax
import unittest
from train_compile import main as train_compile_main
from train import main as train_main

jax.config.update('jax_platform_name', 'cpu')

class TrainCompile(unittest.TestCase):
  """Tests for the Ahead of Time Compilation functionality, train_compile.py"""

  def test_save_and_restore(self):
    compile_save_file='test_compiled.pickle'
    # 25 seconds to save
    train_compile_main((None, "configs/base.yml", f"compile_save_file={compile_save_file}", "compile_topology=v4-8"))
    # 17 seconds to run
    train_main((None, "configs/base.yml", f"compile_save_file={compile_save_file}",
      r"base_output_directory=gs://runner-maxtext-logs", r"dataset_path=gs://maxtext-dataset",
      "steps=2", "run_name=runner_compile_load_test", "enable_checkpointing=False", "assets_path=../assets"))
