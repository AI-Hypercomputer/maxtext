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

python3 -m MaxText.scratch_code.generate_sft_in_trl_golden_logits --model-name=llama2-7b \
  --tokenizer-path=meta-llama/Llama-2-7b-chat-hf --max-target-length=32
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


def prepare_trl_inputs(tokenizer, max_target_length):
  """Get tokenized inputs."""
  data_in_chat_format = tokenizer.apply_chat_template(DATA["messages"], tokenize=False)
  tokenized_data = tokenizer(data_in_chat_format, max_length=max_target_length, return_tensors="pt")

  # masking prompt tokens in labels
  prompt = DATA["messages"][0]
  prompt_in_chat_template = tokenizer.apply_chat_template([prompt], tokenize=True)
  labels = tokenized_data["input_ids"].clone()
  labels[0][:len(prompt_in_chat_template)] = -100  # -100 is the masking value in Hugging Face

  return {
    "input_ids": tokenized_data["input_ids"],
    "attention_mask": tokenized_data["attention_mask"],
    "labels": labels,
  }


def save_golden_logits(conf):
  """Save golden logits."""
  hf_model = get_hf_model(conf.tokenizer_path)
  tokenizer = get_tokenizer(conf.tokenizer_path, conf.max_target_length)
  trl_data = prepare_trl_inputs(tokenizer, conf.max_target_length)
  trl_trainer = setup_sft_trainer(trl_data, hf_model, tokenizer, conf.max_target_length)
  _, trl_outputs = trl_trainer.compute_loss(hf_model, trl_data, return_outputs=True)
  trl_logits = trl_outputs.logits.detach().numpy()

  data_to_save = {
      "data": DATA,
      "tokens": trl_data["input_ids"][0].tolist(),
      "attention_mask": trl_data["attention_mask"][0].tolist(),
      "trl_logits": trl_logits.tolist()[0],
  }

  model_output_path = os.path.join(os.getcwd(), "MaxText", "test_assets", f"golden_data_sft_{conf.model_name}.jsonl")
  with jsonlines.open(model_output_path, "w") as f:
    f.write(data_to_save)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", type=str, required=False, default="llama2-7b")
  parser.add_argument("--tokenizer-path", type=str, required=False, default="meta-llama/Llama-2-7b-chat-hf")
  parser.add_argument("--max-target-length", type=int, required=False, default=32)
  config = parser.parse_args(sys.argv[1:])
  save_golden_logits(config)
