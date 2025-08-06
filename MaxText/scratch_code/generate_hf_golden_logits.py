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

Usage:

python3 -m MaxText.scratch_code.generate_hf_golden_logits --model-id=meta-llama/Llama-4-Scout-17B-16E \
     --output-path=golden_Llama-4-Scout-17B-16E_vision.jsonl --prompts='Describe this image.' \
     --gcs-bucket=aireenmei-multipod --checkpoint_path=/home/aireenmei_google_com/hf-checkpoint \
     --image_path=/home/aireenmei_google_com/maxtext/MaxText/test_assets/test_image.jpg
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import jsonlines
from google.cloud import storage
from PIL import Image

# Load the tokenizer and model from Hugging Face


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def save_golden_logits(model_id, output_path, prompt_texts, gcs_bucket, checkpoint_path, image_paths):
  """save golden logits"""
  if checkpoint_path is None:
     checkpoint_path = model_id
  tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
  model = AutoModelForCausalLM.from_pretrained(
      checkpoint_path,
      torch_dtype=torch.float32,
      trust_remote_code=True,
  )

  all_data_to_save = []
  for i, prompt_text in enumerate(prompt_texts):
    # Encode the prompt text
    if len(image_paths) > 0:
      try:
        image = Image.open(image_paths[i])
      except Exception as e:
        raise e
      image = image.convert("RGB")
      image = image.resize((336, 336))
      processor = AutoProcessor.from_pretrained(checkpoint_path)
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "image"},
                  {"type": "text", "text": prompt_text},
              ]
          },
      ]
      formatted_prompt = processor.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )
      inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
      data_to_save = {
        "prompt": prompt_text,
        "formatted_prompt": formatted_prompt,
        "input_ids": inputs["input_ids"].tolist()[0],
        "attention_mask": inputs["attention_mask"].tolist()[0],
        "image_path": image_paths[i],
        "pixel_values": inputs["pixel_values"].tolist()[0],
      }
    else:
      formatted_prompt = prompt_text
      inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
      # Prepare data to be saved
      data_to_save = {
          "prompt": prompt_text,
          "tokens": inputs.tolist()[0],
      }

    # Get the logits for the prompt + completion
    with torch.no_grad():
      outputs = model(inputs)
      logits = outputs.logits.cpu().numpy().astype("float32")
      data_to_save["logits"] = logits.tolist()[0] # Convert numpy array to list for JSON serialization

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
  parser.add_argument(
      "--checkpoint_path", type=str, required=False, default=None, help="local path to checkpoint if exists."
  )
  parser.add_argument(
      "--image_paths", type=str, required=False, default='', help="A comma-separated list of image_paths."
  )
  args = parser.parse_args(raw_args)
  prompts = args.prompts.split(",")
  image_paths = args.image_paths.split(",")
  if len(image_paths) > 0:
    assert len(image_paths) == len(prompts), "when image paths are provided, image_paths and prompts must have the same length."
  save_golden_logits(args.model_id, args.output_path, prompts, args.gcs_bucket, args.checkpoint_path, image_paths)


if __name__ == "__main__":
  main()
