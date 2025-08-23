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
"""
Usage:

python3 -m MaxText.scratch_code.generate_hf_golden_logits --model-id=deepseek-ai/DeepSeek-V2-Lite \
     --output-path=golden_DeepSeek-V2-Lite.jsonl --prompts='I love to,Today is a,What is the' \
     --gcs-bucket=my-gcs-bucket

"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from google.cloud import storage

# Load the tokenizer and model from Hugging Face


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def save_golden_logits(model_id, output_path, prompt_texts, gcs_bucket):
  """save golden logits"""
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.float32,
      trust_remote_code=True,
  )

  all_data_to_save = []
  for prompt_text in prompt_texts:
    # Encode the prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Get the logits for the prompt + completion
    with torch.no_grad():
      outputs = model(input_ids)
      logits = outputs.logits.cpu().numpy().astype("float32")

      # Prepare data to be saved
      data_to_save = {
          "prompt": prompt_text,
          "tokens": input_ids.tolist()[0],
          "logits": logits.tolist()[0],  # Convert numpy array to list for JSON serialization
      }
      all_data_to_save.append(data_to_save)

  with jsonlines.open(output_path, "w") as f:
    f.write_all(all_data_to_save)
    print(f"File is stored locally at {output_path}.")

  if gcs_bucket:
    upload_blob(gcs_bucket, output_path, f"golden-logits/{model_id}/{output_path}")
    print(f"File is uploaded to gs://{gcs_bucket}/golden-logits/{model_id}/{output_path}.")


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", type=str, required=True, help="The identifier of the model to use.")
  parser.add_argument("--output-path", type=str, required=True, help="The path to save the generated golden logits.")
  parser.add_argument("--prompts", type=str, required=True, help="A comma-separated list of prompts.")
  parser.add_argument(
      "--gcs-bucket", type=str, required=False, default=None, help="A GCS bucket to store logits, without gs://."
  )
  args = parser.parse_args(raw_args)
  prompts = args.prompts.split(",")
  save_golden_logits(args.model_id, args.output_path, prompts, args.gcs_bucket)


if __name__ == "__main__":
  main()
