"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Usage: python3 -m MaxText.scratch_code.golden_gemma3-4b-mm_export --model-id google/gemma-3-4b-it \
  --output-path MaxText/test_assets/golden_data_gemma3-4b-mm.jsonl
"""

import torch
import jsonlines
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse


# Save only the next 4 tokens to save space
NUM_TOKENS = 4
# For some reason, HF logits size is 262208 but GitHub is 262144 for gemma
NUM_LOGITS = 262144


def save_golden_logits(model_id, output_path):
  # Load the tokenizer and model from Hugging Face
  device = "cuda" if torch.cuda.is_available() else "cpu"
  processor = AutoProcessor.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.float32,
  )

  # Input image and prompt
  prompt_text = "<start_of_turn>user\nDescribe image <start_of_image><end_of_turn>\n<start_of_turn>model\n"
  image_path = "MaxText/test_assets/test_image.jpg"
  image = Image.open(image_path).convert("RGB")
  inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

  # Forward pass to get logits
  model.eval()
  with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=300,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

  # Decode the output
  generated_text = processor.tokenizer.decode(outputs[0][0], skip_special_tokens=True)
  print(generated_text)

  # Concatenate and clip the logits to the next 4 tokens with 262144 logits
  stacked_tensors = torch.stack(outputs.scores, dim=1)
  logits = stacked_tensors.numpy()
  logits = logits[:, :NUM_TOKENS, :NUM_LOGITS]

  all_data_to_save = []
  input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
  image_array = np.array(image)
  data_to_save = {
      "prompt": prompt_text,
      "tokens": input_ids.tolist()[0],
      "image_array": image_array.tolist(),  # Convert numpy array to list for JSON serialization
      "logits": logits.tolist()[0],  # Convert numpy array to list for JSON serialization
  }
  all_data_to_save.append(data_to_save)

  with jsonlines.open(output_path, "w") as f:
    f.write_all(all_data_to_save)


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", type=str, required=False, default="google/gemma-3-4b-it")
  parser.add_argument("--output-path", type=str, required=False, default="MaxText/test_assets/golden_data_gemma3-4b-mm.jsonl")
  args = parser.parse_args(raw_args)
  save_golden_logits(args.model_id, args.output_path)


if __name__ == "__main__":
  main()
