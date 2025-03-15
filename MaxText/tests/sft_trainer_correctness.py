# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT trainer correctness tests"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import torch
import unittest

from jax.sharding import Mesh
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

current_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_parent_dir = os.path.dirname(current_dir)
sys.path.append(maxtext_parent_dir)

import max_utils
import pyconfig
from input_pipeline import _input_pipeline_utils
from layers import models
from layers import quantizations

DATA = {
    "messages": [
        {"role": "user", "content": "Hello, what is your name?"},
        {"role": "assistant", "content": "I am a chatbot. How can I help?"},
    ],
}
os.environ["XLA_USE_SPMD"] = "1"


class SFTTrainerCorrectness(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [None, "configs/sft.yml"],
        run_name="test-sft-trainer-correctness",
        model_name="llama3.1-8b",
        tokenizer_path="meta-llama/Llama-3.1-8B",
        enable_checkpointing=True,
        load_parameters_path="gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
        max_target_length=64,
        per_device_batch_size=1,
        max_prefill_predict_length=32,
        dataset_type="synthetic",
        dtype="float32",
        matmul_precision="high",
        logits_dot_in_fp32=True,
        skip_jax_distributed_system=True,
    )
    self.hf_model = AutoModelForCausalLM.from_pretrained(
        self.config.tokenizer_path,
        torch_dtype=torch.float32,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.config.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        model_max_length=self.config.max_target_length,
    )

  def setup_maxtext_model(self):
    init_rng = jax.random.PRNGKey(self.config.init_weights_seed)
    init_rng, rng1 = jax.random.split(init_rng)
    quant = quantizations.configure_quantization(self.config)
    devices_array = max_utils.create_device_mesh(self.config)
    mesh = Mesh(devices_array, self.config.mesh_axes)
    maxtext_model = models.Transformer(config=self.config, mesh=mesh, quant=quant)
    state, _ = max_utils.setup_decode_state(maxtext_model, self.config, rng1, mesh, None)
    return maxtext_model, state, init_rng

  def setup_sft_trainer(self, data):
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "xla": True,
        "xla_fsdp_v2": True,
        "xla_fsdp_grad_ckpt": True,
    }
    global_batch_size = int(jax.device_count() * self.config.per_device_batch_size * self.config.gradient_accumulation_steps)
    training_args = TrainingArguments(
        per_device_train_batch_size=global_batch_size,
        bf16=True,
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    )
    return SFTTrainer(
        model=self.hf_model,
        processing_class=self.tokenizer,
        train_dataset=data,
        data_collator=None,
        args=SFTConfig(
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=self.config.max_target_length,
            **training_args.to_dict(),
        ),
    )

  def apply_chat_template(self, data):
    messages = []
    for message in data["messages"]:
      if message["role"] == "user":
        messages.append("<user>" + message["content"] + "</user>")
      elif message["role"] == "assistant":
        messages.append("<assistant>" + message["content"] + "</assistant>")
    return messages

  def get_input_ids(self, data):
    input_ids = []
    attention_mask = []
    for d in data:
      input_ids += d["input_ids"]
      attention_mask += d["attention_mask"]
    labels = input_ids + [self.tokenizer.eos_token_id] + [0]
    input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
    attention_mask = [1] + attention_mask + [1]
    global_batch_size = int(jax.device_count() * self.config.per_device_batch_size * self.config.gradient_accumulation_steps)
    return {
        "input_ids": torch.stack(
            [
                torch.tensor(input_ids[: self.config.max_target_length], dtype=torch.long).clone()
                for _ in range(global_batch_size)
            ]
        ),
        "labels": torch.stack(
            [
                torch.tensor(labels[: self.config.max_target_length], dtype=torch.long).clone()
                for _ in range(global_batch_size)
            ]
        ),
        "attention_mask": torch.stack(
            [
                torch.tensor(attention_mask[: self.config.max_target_length], dtype=torch.long).clone()
                for _ in range(global_batch_size)
            ]
        ),
    }

  def prepare_trl_inputs(self):
    trl_data = DATA.copy()
    data = self.apply_chat_template(trl_data)
    tokenized_data = [self.tokenizer(d) for d in data]
    processed_data = self.get_input_ids(tokenized_data)
    return processed_data

  def prepare_maxtext_inputs(self):
    maxtext_data = DATA.copy()
    data = _input_pipeline_utils.extract_messages_and_mask(maxtext_data, "messages")
    tokenized_data = _input_pipeline_utils.tokenization(
        data,
        hf_tokenizer=self.tokenizer,
        truncation=False,
        max_length=self.config.max_target_length,
        column_names=["messages"],
    )
    masked_inputs = _input_pipeline_utils.SFTPromptMasking(
        text_column_name="messages",
        completion_only=False,
        max_target_length=self.config.max_target_length,
        add_bos=True,
        add_eos=True,
        bos_id=self.tokenizer.bos_token_id,
        eos_id=self.tokenizer.eos_token_id,
        unk_id=self.tokenizer.unk_token_id,
    ).map(tokenized_data)

    global_batch_size = int(jax.device_count() * self.config.per_device_batch_size * self.config.gradient_accumulation_steps)
    inputs = jnp.stack([np.asarray(masked_inputs["inputs"], dtype=np.int32) for _ in range(global_batch_size)])
    inputs_segmentation = jnp.stack([(masked_inputs["inputs"] != 0).astype(np.int32) for _ in range(global_batch_size)])
    inputs_position = jnp.stack(
        [np.arange(masked_inputs["inputs"].shape[0], dtype=np.int32) for _ in range(global_batch_size)]
    )
    return inputs, inputs_segmentation, inputs_position

  def get_maxtext_logits(self, inputs, inputs_position, inputs_segmentation):
    maxtext_model, state, rng = self.setup_maxtext_model()
    maxtext_logits, _ = maxtext_model.apply(
        state.params,
        inputs,
        inputs_position,
        decoder_segment_ids=inputs_segmentation,
        enable_dropout=False,
        rngs={"aqt": rng},
        mutable="intermediates",
    )
    return maxtext_logits

  def get_kl_div(self, maxtext_logits, hf_logits):
    maxtext_probabilities = jax.nn.softmax(maxtext_logits, axis=-1)
    hf_probabilities = jax.nn.softmax(hf_logits, axis=-1)
    kl_div = jax.numpy.sum(jax.scipy.special.kl_div(hf_probabilities, maxtext_probabilities), axis=-1)
    return kl_div

  def test_maxtext_with_sft_in_trl(self):
    inputs, inputs_segmentation, inputs_position = self.prepare_maxtext_inputs()
    trl_data = self.prepare_trl_inputs()

    assert trl_data["input_ids"][0].tolist() == inputs[0].tolist()
    assert trl_data["attention_mask"][0].tolist() == inputs_segmentation[0].tolist()

    maxtext_logits = self.get_maxtext_logits(inputs, inputs_position, inputs_segmentation)

    trl_trainer = self.setup_sft_trainer(trl_data)
    trl_loss, trl_outputs = trl_trainer.compute_loss(self.hf_model, trl_data, return_outputs=True)
    trl_logits = trl_outputs.logits.detach().numpy()

    assert jax.numpy.allclose(
        maxtext_logits,
        trl_logits,
        rtol=1e-05,
        atol=0.06,
        equal_nan=False,
    )

    kl_div = self.get_kl_div(maxtext_logits, trl_logits)
    assert jax.numpy.all(kl_div < 7e-5)

  def test_maxtext_without_sft_in_trl(self):
    inputs, inputs_segmentation, inputs_position = self.prepare_maxtext_inputs()
    trl_data = self.prepare_trl_inputs()

    assert trl_data["input_ids"][0].tolist() == inputs[0].tolist()
    assert trl_data["attention_mask"][0].tolist() == inputs_segmentation[0].tolist()

    maxtext_logits = self.get_maxtext_logits(inputs, inputs_position, inputs_segmentation)

    hf_outputs = self.hf_model(input_ids=trl_data["input_ids"], attention_mask=trl_data["attention_mask"])
    hf_logits = hf_outputs.logits.detach().numpy()

    assert jax.numpy.allclose(
        maxtext_logits,
        hf_logits,
        rtol=1e-05,
        atol=0.06,
        equal_nan=False,
    )

    kl_div = self.get_kl_div(maxtext_logits, hf_logits)
    assert jax.numpy.all(kl_div < 7e-5)


if __name__ == "__main__":
  unittest.main()
