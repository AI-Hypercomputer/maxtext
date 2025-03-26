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

import subprocess
import unittest

import numpy as np
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from datasets import Dataset
import transformers

import pyconfig
from input_pipeline import _hf_data_processing
from input_pipeline import input_pipeline_interface

PROMPT_DATA = [
    [
        {"content": "example one question one", "role": "user"},
        {"content": "example one question two", "role": "user"},
    ],
    [
        {"content": "question two", "role": "user"},
    ],
    [
        {"content": "question three", "role": "user"},
    ],
    [
        {"content": "question four", "role": "user"},
    ],
]

COMPLETION_DATA = [
    [
        {"content": "example one answer one", "role": "assistant"},
        {"content": "example one answer two", "role": "assistant"},
    ],
    [
        {"content": "answer two", "role": "assistant"},
    ],
    [
        {"content": "answer three", "role": "assistant"},
    ],
    [
        {"content": "answer four", "role": "assistant"},
    ],
]

MESSAGES_DATA = [
    [
        {"content": "example one question one", "role": "user"},
        {"content": "example one answer one", "role": "assistant"},
        {"content": "example one question two", "role": "user"},
        {"content": "example one answer two", "role": "assistant"},
    ],
    [
        {"content": "question two", "role": "user"},
        {"content": "answer two", "role": "assistant"},
    ],
    [
        {"content": "question three", "role": "user"},
        {"content": "answer three", "role": "assistant"},
    ],
    [
        {"content": "question four", "role": "user"},
        {"content": "answer four", "role": "assistant"},
    ],
]


class SFTDataProcessingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    exit_code = subprocess.call(["gsutil", "cp", "-r", "gs://maxtext-dataset/hf/llama2-tokenizer", "../assets/"])
    if exit_code != 0:
      raise ValueError(f"Download tokenizer with gsutil cp failed with exit code: {exit_code}")

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        ["sft_trainer.py", "configs/sft.yml"],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        tokenizer_path="../assets/llama2-tokenizer",
        train_split="train",
        enable_checkpointing=False,
        use_sft=True,
        enable_data_shuffling=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )
    self.max_target_length = 44
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.config.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
    )

  def get_train_iterator(self, train_ds, data_columns):
    self.train_iter = _hf_data_processing.preprocessing_pipeline(
        dataloading_host_index=self.process_indices.index(jax.process_index()),
        dataloading_host_count=len(self.process_indices),
        global_mesh=self.mesh,
        dataset=train_ds,
        data_column_names=data_columns,
        tokenize=self.config.tokenize_train_data,
        tokenizer_path=self.config.tokenizer_path,
        hf_access_token=self.config.hf_access_token,
        global_batch_size=self.config.global_batch_size_to_load,
        max_target_length=self.max_target_length,
        shuffle=self.config.enable_data_shuffling,
        data_shuffle_seed=self.config.data_shuffle_seed,
        add_bos=self.config.add_bos,
        add_eos=self.config.add_eos,
        packing=self.config.packing,
        generate_padding_example=False,
        use_dpo=self.config.use_dpo,
        use_sft=self.config.use_sft,
        sft_train_on_completion_only=self.config.sft_train_on_completion_only,
        grain_worker_count=0,
    )

  def test_sft_format_with_messages(self):
    train_ds = Dataset.from_dict({"messages": MESSAGES_DATA * 2})
    data_columns = ["messages"]
    self.get_train_iterator(train_ds, data_columns)

    # exp1 is longer than max_target_length=44, testing truncation
    truncated_exp1_inputs = "<s> <user>example one question one</user> <assistant>example one answer one</assistant> <user>example one question two</user> <assistant>example one answer two</assistant"
    truncated_exp1_targets = "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer one</assistant><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer two</assistant<unk>"
    truncated_exp1_targets_predictable = "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer one</assistant><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer two</assistant<unk>"

    # exp2 is packed from 2nd and 3rd entries, testing packing
    packed_exp2_inputs = "<s> <user>question two</user> <assistant>answer two</assistant></s><s> <user>question three</user> <assistant>answer three</assistant></s><unk><unk><unk><unk>"
    packed_exp2_targets = "<unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer two</assistant></s><s><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer three</assistant></s><unk><unk><unk><unk><unk>"
    packed_exp2_targets_predictable = "<unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer two</assistant></s><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer three</assistant></s><unk><unk><unk><unk><unk>"

    batch = next(self.train_iter)
    self.assertEqual(self.tokenizer.decode(batch["inputs"][0]), truncated_exp1_inputs)
    self.assertEqual(self.tokenizer.decode(batch["targets"][0]), truncated_exp1_targets)
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][0] > 0, batch["inputs"][0], 0)), truncated_exp1_inputs
    )
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["targets_segmentation"][0] > 0, batch["targets"][0], 0)),
        truncated_exp1_targets_predictable,
    )
    self.assertEqual(self.tokenizer.decode(batch["inputs"][1]), packed_exp2_inputs)
    self.assertEqual(self.tokenizer.decode(batch["targets"][1]), packed_exp2_targets)
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][1] > 0, batch["inputs"][1], 0)), packed_exp2_inputs
    )
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["targets_segmentation"][1] > 0, batch["targets"][1], 0)),
        packed_exp2_targets_predictable,
    )

  def test_sft_format_with_prompt_completion(self):
    train_ds = Dataset.from_dict({"prompt": PROMPT_DATA * 2, "completion": COMPLETION_DATA * 2})
    data_columns = ["prompt", "completion"]
    self.get_train_iterator(train_ds, data_columns)

    # exp1 is longer than max_target_length=44, testing truncation
    truncated_exp1_inputs = "<s> <user>example one question one</user> <assistant>example one answer one</assistant> <user>example one question two</user> <assistant>example one answer two</assistant"
    truncated_exp1_targets = "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer one</assistant><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer two</assistant<unk>"
    truncated_exp1_targets_predictable = "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer one</assistant><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>example one answer two</assistant<unk>"

    # exp2 is packed from 2nd and 3rd entries, testing packing
    packed_exp2_inputs = "<s> <user>question two</user> <assistant>answer two</assistant></s><s> <user>question three</user> <assistant>answer three</assistant></s><unk><unk><unk><unk>"
    packed_exp2_targets = "<unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer two</assistant></s><s><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer three</assistant></s><unk><unk><unk><unk><unk>"
    packed_exp2_targets_predictable = "<unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer two</assistant></s><unk><unk><unk><unk><unk><unk><unk><unk><unk> <assistant>answer three</assistant></s><unk><unk><unk><unk><unk>"

    batch = next(self.train_iter)
    self.assertEqual(self.tokenizer.decode(batch["inputs"][0]), truncated_exp1_inputs)
    self.assertEqual(self.tokenizer.decode(batch["targets"][0]), truncated_exp1_targets)
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][0] > 0, batch["inputs"][0], 0)), truncated_exp1_inputs
    )
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["targets_segmentation"][0] > 0, batch["targets"][0], 0)),
        truncated_exp1_targets_predictable,
    )
    self.assertEqual(self.tokenizer.decode(batch["inputs"][1]), packed_exp2_inputs)
    self.assertEqual(self.tokenizer.decode(batch["targets"][1]), packed_exp2_targets)
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][1] > 0, batch["inputs"][1], 0)), packed_exp2_inputs
    )
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["targets_segmentation"][1] > 0, batch["targets"][1], 0)),
        packed_exp2_targets_predictable,
    )


if __name__ == "__main__":
  unittest.main()
