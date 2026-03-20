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

"""Tests for custom mesh and sharding rule configurations in model compilation.

This module verifies that `train_compile.py` correctly processes the
`custom_mesh_and_rule` flag. It ensures that user-defined hardware
meshes and parallelization strategies compile successfully prior to execution.
"""

import unittest

import pytest

from maxtext.trainers.pre_train.train_compile import main as train_compile_main
from tests.utils.test_helpers import get_test_config_path


@pytest.mark.tpu_backend
class CustomMeshAndRuleTest(unittest.TestCase):
  """Tests for custom_mesh functionality in train_compile.py"""

  @pytest.mark.cpu_only
  def test_pure_fsdp(self):
    """Test compiling with a pure FSDP custom mesh."""
    train_compile_main(
        (
            "",
            get_test_config_path(),
            "compile_topology=v4-8",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=1",
            "custom_mesh_and_rule=pure-fsdp",
        )
    )

  @pytest.mark.cpu_only
  def test_ds3_large_pp(self):
    """Test compiling deepseek3-tiny with the pipeline-large-moe custom mesh."""
    train_compile_main(
        (
            "",
            get_test_config_path(),
            "compile_topology=v5p-32",
            "compile_topology_num_slices=1",
            "ici_fsdp_transpose_parallelism=2",
            "ici_expert_parallelism=2",
            "model_name=deepseek3-tiny",
            "override_model_config=true",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=4",
            "custom_mesh_and_rule=pipeline-large-moe",
        )
    )
