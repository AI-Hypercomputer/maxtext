"""
Copyright 2024 Google LLC
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

import unittest
import pyconfig


class PyconfigTest(unittest.TestCase):
  """Tests for pyconfig.py"""

  def test_basic_override(self):
    raw_keys = {"megablox": None, "foo": ["bar", "baz"]}
    model_keys = {"foo": ["x", "y"]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(raw_keys, {"megablox": None, "foo": ["x", "y"]})

  def test_logical_axis_override(self):
    raw_keys = {
        "megablox": None,
        "foo": ["bar", "baz"],
        "logical_axis_rules": [["activation", ["data", "fsdp"]], ["norm", "tensor"]],
    }
    model_keys = {"logical_axis_rules": [["activation", ["data", "fsdp_transpose"]], ["norm", "fsdp"]]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(
        raw_keys,
        {
            "megablox": None,
            "foo": ["bar", "baz"],
            "logical_axis_rules": [("activation", ["data", "fsdp_transpose"]), ("norm", "fsdp")],
        },
    )

  def test_logical_axis_partial_override(self):
    raw_keys = {
        "megablox": None,
        "foo": ["bar", "baz"],
        "logical_axis_rules": [["activation", ["data", "fsdp"]], ["norm", "tensor"]],
    }
    model_keys = {"logical_axis_rules": [["norm", "fsdp"]]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(
        raw_keys,
        {
            "megablox": None,
            "foo": ["bar", "baz"],
            "logical_axis_rules": [("activation", ("data", "fsdp")), ("norm", "fsdp")],
        },
    )

  def test_multiple_unmodifiable_configs(self):
    config_train = pyconfig.initialize(
        ["train.py", "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        ici_tensor_parallelism=-1,
        ici_fsdp_parallelism=4,
    )
    config_inference = pyconfig.initialize(
        ["decode.py", "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        ici_tensor_parallelism=4,
        ici_fsdp_parallelism=-1,
    )
    self.assertNotEqual(
        config_train.ici_tensor_parallelism,
        config_inference.ici_tensor_parallelism,
    )
    with self.assertRaises(ValueError):
      config_inference.__setattr__("ici_fsdp_parallelism", 4)


if __name__ == "__main__":
  unittest.main()
