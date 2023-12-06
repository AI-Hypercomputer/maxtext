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
import unittest
from train_compile import main as train_compile_main
from train import main as train_main


class TrainCompile(unittest.TestCase):
  """Tests for the Ahead of Time Compilation functionality, train_compile.py"""

  def test_save_compiled(self):
    compiled_trainstep_file='test_compiled.pickle'
    train_compile_main((None, "configs/base.yml", f"compiled_trainstep_file={compiled_trainstep_file}",
      "compile_topology=v4-8", "compile_topology_num_slices=1", "base_emb_dim=256", "base_mlp_dim=256",
      "base_num_decoder_layers=2"))
  
  def test_sequence_parallelism(self):
    compiled_trainstep_file='test_compiled.pickle'
    train_compile_main((None, "configs/base.yml", f"compiled_trainstep_file={compiled_trainstep_file}",
      "compile_topology=v5e-256", "use_iota_embed=true", "compile_topology_num_slices=1", 
      "ici_sequence_parallelism=16", "global_parameter_scale=32", "per_device_batch_size=0.125", "max_target_length=65536"))
