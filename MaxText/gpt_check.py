from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

max_target_length=4
global_batch_size_to_train_on=1

model_id="openai/gpt-oss-20b"
login(token="")

all_data_to_save = []
prompts = ["I love to"]
# prompts = ["I", "I love to", "Today is a", "What is the"]
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    # torch_dtype="bfloat16",
    torch_dtype="float32",
  )

for prompt in prompts:
  input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, :max_target_length]

  # Get the logits for the prompt + completion using Hugging Face model
  with torch.no_grad():
      outputs = hf_model(input_ids)
      logits = outputs.logits.cpu().numpy().astype("float32")

  print(logits)

  data_to_save = {
      "prompt": prompt,
      # "completion": out_data.text[prompt_index],
      "tokens": input_ids.tolist(),
      "logits": logits.tolist(),
  }
  all_data_to_save.append(data_to_save)

import jsonlines

output_path = "Llama-4-Scout-17B-16E.jsonl"
with jsonlines.open(output_path, "w") as f:
  f.write_all(all_data_to_save)

print(f"Data saved to {output_path}")
