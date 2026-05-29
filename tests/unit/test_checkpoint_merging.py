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

"""Unit tests for recursive checkpoint merging and load_orbax_checkpoint."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class CheckpointMergingTest(unittest.TestCase):
  """Tests the recursive_update and load_orbax_checkpoint functions to ensure we don't overwrite weights."""

  def test_recursive_update(self):
    from maxtext.checkpoint_conversion.utils.utils import recursive_update

    base = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel": np.ones((4, 4)),
                }
            }
        }
    }
    lora = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": np.ones((4, 2)),
                    "kernel_lora_b": np.ones((2, 4)),
                }
            }
        }
    }

    merged = {}
    recursive_update(merged, base)
    recursive_update(merged, lora)

    # Verify that both base and lora weights are present and not overwritten
    self.assertIn("kernel", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_a", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_b", merged["params"]["decoder"]["layers"])
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel"], np.ones((4, 4)))
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel_lora_a"], np.ones((4, 2)))
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel_lora_b"], np.ones((2, 4)))

  @patch("maxtext.checkpoint_conversion.utils.utils.ocp.Checkpointer")
  @patch("maxtext.checkpoint_conversion.utils.utils.epath.Path")
  @patch("maxtext.checkpoint_conversion.utils.utils.jax.devices")
  def test_load_orbax_checkpoint_recursive_merge(self, mock_jax_devices, mock_path, mock_checkpointer_cls):
    from maxtext.checkpoint_conversion.utils.utils import load_orbax_checkpoint

    # Mock jax devices
    mock_jax_devices.return_value = [MagicMock()]

    # Mock Orbax Checkpointer and its restore results
    mock_ckptr = MagicMock()
    mock_checkpointer_cls.return_value = mock_ckptr

    # Base checkpoint metadata and content
    base_metadata = MagicMock()
    base_metadata.item_metadata.tree = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel": MagicMock(shape=(4, 4))
                }
            }
        }
    }
    base_restore_content = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel": np.ones((4, 4))
                }
            }
        }
    }

    # LoRA checkpoint metadata and content
    lora_metadata = MagicMock()
    lora_metadata.item_metadata.tree = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": MagicMock(shape=(4, 2)),
                    "kernel_lora_b": MagicMock(shape=(2, 4)),
                }
            }
        }
    }
    lora_restore_content = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": np.ones((4, 2)),
                    "kernel_lora_b": np.ones((2, 4)),
                }
            }
        }
    }

    # Mock metadata and restore calls
    mock_ckptr.metadata.side_effect = [base_metadata, lora_metadata]
    mock_ckptr.restore.side_effect = [base_restore_content, lora_restore_content]

    # Create dummy config
    config = MagicMock()
    config.checkpoint_storage_concurrent_gb = 8
    config.checkpoint_storage_use_ocdbt = True
    config.checkpoint_storage_use_zarr3 = True
    config.load_parameters_path = "gs://base-bucket/checkpoints"
    config.lora.lora_restore_path = "gs://lora-bucket/checkpoints"

    # Load and merge
    merged = load_orbax_checkpoint(config)

    # Assert checkpointer was called twice and restored both
    self.assertEqual(mock_ckptr.restore.call_count, 2)

    # Verify that the keys are recursively merged correctly!
    self.assertIn("kernel", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_a", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_b", merged["params"]["decoder"]["layers"])


if __name__ == "__main__":
  unittest.main()
