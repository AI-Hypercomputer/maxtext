# Copyright 2023–2025 Google LLC
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

"""Data processing tests for SFT."""
import subprocess
import unittest
import os.path
import pytest
import numpy as np
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from datasets import Dataset
import transformers
from parameterized import parameterized_class

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR, MAXTEXT_CONFIGS_DIR, MAXTEXT_ASSETS_ROOT
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline.hf_data_processing import _get_pad_id

PROMPT_DATA = [
    [
        {"content": "example one question one", "role": "user"},
        {"content": "example one question two", "role": "user"},
        {"content": "example one question three", "role": "user"},
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
    [
        {"content": "question five", "role": "user"},
    ],
]

COMPLETION_DATA = [
    [
        {"content": "example one answer one", "role": "assistant"},
        {"content": "example one answer two", "role": "assistant"},
        {"content": "example one answer three", "role": "assistant"},
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
    [
        {"content": "answer five", "role": "assistant"},
    ],
]

MESSAGES_DATA = [
    [
        {"content": "the system prompt", "role": "system"},
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
    [
        {"content": "question five", "role": "user"},
        {"content": "answer five", "role": "assistant"},
    ],
]

LLAMA2_DATA = {
    "tokenizer_path": None,
    "messages": {
        "truncated_exp1_inputs": (
            "<s> [INST] <<SYS>>\nthe system prompt\n<</SYS>>\n\nexample one question one [/INST] "
            "example one answer one </s>"
            "<s> [INST] example one question two [/INST] "
            "example one answer two"
        ),
        "truncated_exp1_targets": (
            "<unk>" * 27 + " " + "example one answer one </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer two<unk>"
        ),
        "truncated_exp1_targets_predictable": (
            "<unk>" * 27 + " " + "example one answer one </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer two<unk>"
        ),
        "packed_exp2_inputs": (
            "<s> [INST] question two [/INST] "
            "answer two </s>"
            "<s> [INST] question three [/INST] "
            "answer three </s>"
            "<s> [INST] question four [/INST] "
            "answer four </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
        "packed_exp2_targets": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer three </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer four </s><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
        "packed_exp2_targets_predictable": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer three </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer four </s><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
    },
    "prompt_completion": {
        "truncated_exp1_inputs": (
            "<s> [INST] example one question one [/INST] "
            "example one answer one </s>"
            "<s> [INST] example one question two [/INST] "
            "example one answer two </s>"
            "<s> [INST] example one question three [/INST] "
            "example one"
        ),
        "truncated_exp1_targets": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer one </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one<unk>"
        ),
        "truncated_exp1_targets_predictable": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer one </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "example one<unk>"
        ),
        "packed_exp2_inputs": (
            "<s> [INST] question two [/INST] "
            "answer two </s>"
            "<s> [INST] question three [/INST]"
            " answer three </s>"
            "<s> [INST] question four [/INST]"
            " answer four </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
        "packed_exp2_targets": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer three </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer four </s><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
        "packed_exp2_targets_predictable": (
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer two </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer three </s>"
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk> "
            "answer four </s><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        ),
    },
}

QWEN_DATA = {
    "tokenizer_path": "Qwen/Qwen3-4B",
    "messages": {
        "truncated_exp1_inputs": (
            "<|im_start|>system\nthe system prompt<|im_end|>\n"
            "<|im_start|>user\nexample one question one<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            "<|im_start|>user\nexample one question two<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two"
        ),
        "truncated_exp1_targets": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            + "<|endoftext|>" * 9
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two<|endoftext|>"
        ),
        "truncated_exp1_targets_predictable": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            + "<|endoftext|>" * 9
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two<|endoftext|>"
        ),
        "packed_exp2_inputs": (
            "<|im_start|>user\nquestion two<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|im_start|>user\nquestion three<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "!" * 14
        ),
        "packed_exp2_targets": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "!" * 14 + "<|endoftext|>"
        ),
        "packed_exp2_targets_predictable": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "<|endoftext|>" * 15
        ),
    },
    "prompt_completion": {
        "truncated_exp1_inputs": (
            "<|im_start|>user\nexample one question one<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            "<|im_start|>user\nexample one question two<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two<|im_end|>\n"
            "<|im_start|>user\nexample one question"
        ),
        "truncated_exp1_targets": (
            "<|endoftext|>" * 8
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            + "<|endoftext|>" * 9
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two<|im_end|>\n"
            + "<|endoftext|>" * 7
        ),
        "truncated_exp1_targets_predictable": (
            "<|endoftext|>" * 8
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer one<|im_end|>\n"
            + "<|endoftext|>" * 9
            + "<|im_start|>assistant\n<think>\n\n</think>\n\nexample one answer two<|im_end|>\n"
            + "<|endoftext|>" * 7
        ),
        "packed_exp2_inputs": (
            "<|im_start|>user\nquestion two<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|im_start|>user\nquestion three<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "!" * 14
        ),
        "packed_exp2_targets": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "!" * 14 + "<|endoftext|>"
        ),
        "packed_exp2_targets_predictable": (
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer two<|im_end|>\n"
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer three<|im_end|>\n" + "<|endoftext|>" * 15
        ),
    },
}


@parameterized_class(
    [
        {"test_data": LLAMA2_DATA},
        {"test_data": QWEN_DATA},
    ]
)
@pytest.mark.external_training  # Uses gcloud storage to pull tokenizer.
class SFTDataProcessingTest(unittest.TestCase):
  test_data = {}

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    exit_code = subprocess.call(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            "gs://maxtext-dataset/hf/llama2-chat-tokenizer",
            os.path.join(MAXTEXT_ASSETS_ROOT, ""),
        ]
    )
    if exit_code != 0:
      raise ValueError(f"Download tokenizer with gcloud storage cp failed with exit code: {exit_code}")

  def setUp(self):
    super().setUp()
    tokenizer_path = self.test_data.get("tokenizer_path")
    if tokenizer_path is None:
      tokenizer_path = os.path.join(MAXTEXT_ASSETS_ROOT, "llama2-chat-tokenizer")

    self.config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "sft_trainer"), os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
        per_device_batch_size=2,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        tokenizer_path=tokenizer_path,
        train_split="train",
        enable_checkpointing=False,
        use_sft=True,
        enable_data_shuffling=False,
        max_target_length=50,
        max_prefill_predict_length=16,
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
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.config.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
    )

  def get_data_iterator(self, train_ds, data_columns):
    """Get data iterator."""
    return hf_data_processing.preprocessing_pipeline(
        dataloading_host_index=self.process_indices.index(jax.process_index()),
        dataloading_host_count=len(self.process_indices),
        global_mesh=self.mesh,
        dataset=train_ds,
        data_column_names=data_columns,
        tokenize=self.config.tokenize_train_data,
        tokenizer_path=self.config.tokenizer_path,
        hf_access_token=self.config.hf_access_token,
        global_batch_size=self.config.global_batch_size_to_load,
        max_target_length=self.config.max_target_length,
        shuffle=self.config.enable_data_shuffling,
        data_shuffle_seed=self.config.data_shuffle_seed,
        add_bos=self.config.add_bos,
        add_eos=self.config.add_eos,
        packing=self.config.packing,
        generate_padding_batch=False,
        use_dpo=self.config.use_dpo,
        use_sft=self.config.use_sft,
        sft_train_on_completion_only=self.config.sft_train_on_completion_only,
        grain_worker_count=0,
    )

  def test_sft_format_with_messages(self):
    expected = self.test_data["messages"]
    dataset = Dataset.from_dict({"messages": MESSAGES_DATA * 4})
    data_columns = ["messages"]
    data_iter = self.get_data_iterator(dataset, data_columns)

    batch = next(data_iter)

    # Check Truncation
    self.assertEqual(self.tokenizer.decode(batch["inputs"][0]), expected["truncated_exp1_inputs"])
    self.assertEqual(self.tokenizer.decode(batch["targets"][0]), expected["truncated_exp1_targets"])
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][0] > 0, batch["inputs"][0], 0)),
        expected["truncated_exp1_inputs"],
    )
    self.assertEqual(
        self.tokenizer.decode(
            np.where(batch["targets_segmentation"][0] > 0, batch["targets"][0], _get_pad_id(self.tokenizer))
        ),
        expected["truncated_exp1_targets"],
    )

    # Check Packing
    self.assertEqual(self.tokenizer.decode(batch["inputs"][1]), expected["packed_exp2_inputs"])
    self.assertEqual(self.tokenizer.decode(batch["targets"][1]), expected["packed_exp2_targets"])
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][1] > 0, batch["inputs"][1], 0)),
        expected["packed_exp2_inputs"],
    )
    self.assertEqual(
        self.tokenizer.decode(
            np.where(batch["targets_segmentation"][1] > 0, batch["targets"][1], _get_pad_id(self.tokenizer))
        ),
        expected["packed_exp2_targets_predictable"],
    )

  def test_sft_format_with_prompt_completion(self):
    expected = self.test_data["prompt_completion"]

    dataset = Dataset.from_dict({"prompt": PROMPT_DATA * 4, "completion": COMPLETION_DATA * 4})
    data_columns = ["prompt", "completion"]
    data_iter = self.get_data_iterator(dataset, data_columns)

    batch = next(data_iter)

    # Check Truncation
    self.assertEqual(self.tokenizer.decode(batch["inputs"][0]), expected["truncated_exp1_inputs"])
    self.assertEqual(self.tokenizer.decode(batch["targets"][0]), expected["truncated_exp1_targets"])
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][0] > 0, batch["inputs"][0], 0)),
        expected["truncated_exp1_inputs"],
    )
    self.assertEqual(
        self.tokenizer.decode(
            np.where(batch["targets_segmentation"][0] > 0, batch["targets"][0], _get_pad_id(self.tokenizer))
        ),
        expected["truncated_exp1_targets_predictable"],
    )

    # Check Packing
    self.assertEqual(self.tokenizer.decode(batch["inputs"][1]), expected["packed_exp2_inputs"])
    self.assertEqual(self.tokenizer.decode(batch["targets"][1]), expected["packed_exp2_targets"])
    self.assertEqual(
        self.tokenizer.decode(np.where(batch["inputs_segmentation"][1] > 0, batch["inputs"][1], 0)),
        expected["packed_exp2_inputs"],
    )
    self.assertEqual(
        self.tokenizer.decode(
            np.where(batch["targets_segmentation"][1] > 0, batch["targets"][1], _get_pad_id(self.tokenizer))
        ),
        expected["packed_exp2_targets_predictable"],
    )

  def test_system_message_not_at_beginning(self):
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "You are a helpful assistant."},
                ]
            ]
        }
    )
    with self.assertRaisesRegex(ValueError, "System messages must be at index 0"):
      self.get_data_iterator(dataset, ["messages"])


if __name__ == "__main__":
  unittest.main()
