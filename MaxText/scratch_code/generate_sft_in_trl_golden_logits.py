#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Script to get golden data for SFTTrainer in TRL.

Usage:

python3 -m MaxText.scratch_code.generate_sft_in_trl_golden_logits --model-name=llama3.1-8b --tokenizer-path=meta-llama/Llama-3.1-8B --max-target-length=64
"""

import argparse
import jsonlines
import os
import sys
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DATA = {
    "messages": [
        {"role": "user", "content": "Hello, what is your name?"},
        {"role": "assistant", "content": "I am a chatbot. How can I help?"},
    ],
}


def get_hf_model(tokenizer_path):
  return AutoModelForCausalLM.from_pretrained(
      tokenizer_path,
      torch_dtype=torch.float32,
  )


def get_tokenizer(tokenizer_path, max_target_length):
  return AutoTokenizer.from_pretrained(
      tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      model_max_length=max_target_length,
  )


def setup_sft_trainer(data, hf_model, tokenizer, max_target_length):
  training_args = TrainingArguments(
      per_device_train_batch_size=1,
      bf16=True,
  )
  return SFTTrainer(
      model=hf_model,
      processing_class=tokenizer,
      train_dataset=data,
      data_collator=None,
      args=SFTConfig(
          dataset_kwargs={"skip_prepare_dataset": True},
          max_seq_length=max_target_length,
          **training_args.to_dict(),
      ),
  )


def apply_chat_template():
  messages = []
  for message in DATA["messages"]:
    if message["role"] == "user":
      messages.append("<user>" + message["content"] + "</user>")
    elif message["role"] == "assistant":
      messages.append("<assistant>" + message["content"] + "</assistant>")
  return messages


def get_input_ids(data, tokenizer, max_target_length):
  input_ids = []
  attention_mask = []
  for d in data:
    input_ids += d["input_ids"]
    attention_mask += d["attention_mask"]
  labels = input_ids + [tokenizer.eos_token_id] + [0]
  input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
  attention_mask = [1] + attention_mask + [1]
  return {
      "input_ids": torch.tensor(input_ids[:max_target_length], dtype=torch.long).unsqueeze(0),
      "labels": torch.tensor(labels[:max_target_length], dtype=torch.long).unsqueeze(0),
      "attention_mask": torch.tensor(attention_mask[:max_target_length], dtype=torch.long).unsqueeze(0),
  }


def prepare_trl_inputs(tokenizer, max_target_length):
  data = apply_chat_template()
  tokenized_data = [tokenizer(d) for d in data]
  processed_data = get_input_ids(tokenized_data, tokenizer, max_target_length)
  return processed_data


def save_golden_logits(config):
  hf_model = get_hf_model(config.tokenizer_path)
  tokenizer = get_tokenizer(config.tokenizer_path, config.max_target_length)
  trl_data = prepare_trl_inputs(tokenizer, config.max_target_length)
  trl_trainer = setup_sft_trainer(trl_data, hf_model, tokenizer, config.max_target_length)
  _, trl_outputs = trl_trainer.compute_loss(hf_model, trl_data, return_outputs=True)
  trl_logits = trl_outputs.logits.detach().numpy()

  data_to_save = {
      "data": DATA,
      "tokens": trl_data["input_ids"][0].tolist(),
      "attention_mask": trl_data["attention_mask"][0].tolist(),
      "trl_logits": trl_logits.tolist()[0],
  }

  model_output_path = os.path.join(os.getcwd(), "MaxText", "test_assets", f"golden_data_sft_{config.model_name}.jsonl")
  with jsonlines.open(model_output_path, "w") as f:
    f.write(data_to_save)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", type=str, required=False, default="llama3.1-8b")
  parser.add_argument("--tokenizer-path", type=str, required=False, default="meta-llama/Llama-3.1-8B")
  parser.add_argument("--max-target-length", type=int, required=False, default=64)
  config = parser.parse_args(sys.argv[1:])
  save_golden_logits(config)
