# Copyright 2023–2026 Google LLC
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

"""Tests for distillation configuration."""

import unittest
import os
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

class TestDistillationConfig(unittest.TestCase):
    """Test class for distillation configuration."""

    def test_distillation_yaml_content(self):
        """Test the content of the distillation configuration YAML file."""
        config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train/distillation.yml")
        self.assertTrue(os.path.exists(config_path), f"Config file not found at {config_path}")
        
        # Initialize configuration using MaxText's standard pyconfig.initialize.
        # This ensures the configuration is validated against the MaxText Pydantic schema.
        cfg = pyconfig.initialize(
            [ "maxtext.trainers.post_train.distillation.train_distill", config_path ], 
            run_name="test_run"
        )
        
        # Student expectations
        student = cfg.student_overrides
        self.assertEqual(student['model_name'], "deepseek2-16b")
        self.assertEqual(student['num_experts'], 16)
        self.assertEqual(student['shared_experts'], 2)
        self.assertEqual(student['trainable_parameters_mask'], ['.*moe_layers.*'])
        self.assertEqual(student['base_emb_dim'], 2048)
        self.assertEqual(student['base_num_query_heads'], 16)
        self.assertEqual(student['base_num_kv_heads'], 16)
        self.assertEqual(student['base_mlp_dim'], 16384)
        self.assertEqual(student['base_moe_mlp_dim'], 4864)
        self.assertEqual(student['base_num_decoder_layers'], 28)
        self.assertEqual(student['first_num_dense_layers'], 14)

        # Teacher expectations
        teacher = cfg.teacher_overrides
        self.assertEqual(teacher['model_name'], "deepseek2-16b")
        self.assertEqual(teacher['num_experts'], 32)
        self.assertEqual(teacher['shared_experts'], 1)
        self.assertEqual(teacher['load_parameters_path'], 
                         "gs://yujiedeng-maxtext-dev/distillation/converted-from-hf-ds-v2-16b-fixed/0/items")

if __name__ == '__main__':
    unittest.main()
