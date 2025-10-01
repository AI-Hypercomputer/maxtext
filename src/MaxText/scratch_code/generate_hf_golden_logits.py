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
     --output-path=golden_DeepSeek-V2-Lite.jsonl --prompts='I love to;Today is a;What is the' \
     --gcs-bucket=my-gcs-bucket

For large models, you can use an m1 cpu. Calling the script directly instead of calling MaxText module \
can skip importing unnecessary dependencies.
For large Hugginface checkpoints, you can use pre-downloaded checkpoints with --hf-model-path argument.
For multimodal models, use --image-paths argument to provide image path(s),\
  use --apply-chat-template=true if use HF chat template to format image+prompt.\
  When using chat template, the prompt should not contain image placeholders.
  
More examples:
python3 -m MaxText.scratch_code.generate_hf_golden_logits --model-id=meta-llama/Llama-4-Scout-17B-16E \
     --output-path=golden_Llama-4-Scout-17B-16E_vision.jsonl --prompts='Describe this image.' \
     --apply-chat-template --gcs-bucket=<bucket> --hf-model-path=<hf_checkpoint_path> \
     --image-paths=src/MaxText/test_assets/test_image.jpg

python3 -m MaxText.scratch_code.generate_hf_golden_logits --model-id=google/gemma-3-4b-it \
     --output-path=golden_gemma-3-4b-it_vision.jsonl --prompts='<start_of_image>' \
     --gcs-bucket=<bucket> --hf-model-path=<hf_checkpoint_path> \
     --image-paths=src/MaxText/test_assets/test_image.jpg
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import jsonlines
from google.cloud import storage
from PIL import Image
import numpy as np
import pickle

# Load the tokenizer and model from Hugging Face


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def save_golden_logits(model_id, output_path, prompt_texts, apply_chat_template, gcs_bucket, hf_model_path, image_paths, format):
  """save golden logits"""
  if hf_model_path is None:
    hf_model_path = model_id
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if model_id.startswith("meta-llama/Llama-4"):
    from transformers import Llama4ForConditionalGeneration
    model = Llama4ForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
  else:  
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

  all_data_to_save = []
  for i, prompt_text in enumerate(prompt_texts):
    # Encode the prompt text
    if image_paths:
      try:
        image = Image.open(image_paths[i])
      except Exception as e:
        raise e
      image = image.convert("RGB")
      # TODO (aireenmei): remove this when Llama-4 supports dynamic image shapes.
      if model_id.startswith("meta-llama/Llama-4"):
        image = image.resize((336, 336))
      processor = AutoProcessor.from_pretrained(model_id, token=True)
      if apply_chat_template:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"{apply_chat_template=} {formatted_prompt=}")
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
      else:
        formatted_prompt = prompt_text
        print(f"{apply_chat_template=} {formatted_prompt=}")
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt", add_special_tokens=False)
      inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
      print(f"raw ids: {inputs['input_ids']}, shape {inputs['input_ids'].shape}")
      with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy().astype("float32")
      if format == "json":
        data_to_save = {
            "prompt": prompt_text,
            "formatted_prompt": formatted_prompt,
            "tokens": inputs["input_ids"].tolist()[0],
            "attention_mask": inputs["attention_mask"].tolist()[0],
            "image_path": image_paths[i],
            "pixel_values": inputs["pixel_values"].tolist()[0],
            "logits": logits.tolist()[0],
        }
      elif format == "pickle":
        data_to_save = {
            "prompt": prompt_text,
            "formatted_prompt": formatted_prompt,
            "tokens": np.array(inputs["input_ids"].cpu().numpy())[0],  # or np.array(tokens)
            "image_path": image_paths[i],
            "pixel_values": inputs["pixel_values"].cpu().numpy()[0],  # if not already numpy
            "attention_mask": np.array(inputs["attention_mask"].cpu().numpy())[0],
            "logits": logits[0],  # keep as np.ndarray
        }
    else:
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

  if format == "json":
    with jsonlines.open(output_path, "w") as f:
      f.write_all(all_data_to_save)
      print(f"File is stored locally at {output_path}.")
  elif format == "pickle":
    # Save all data to a single pickle file
    with open(output_path, "wb") as f:
      pickle.dump(all_data_to_save, f)
    print(f"File is stored locally at {output_path}.")

  if gcs_bucket:
    upload_blob(gcs_bucket, output_path, f"golden-logits/{model_id}/{output_path}")
    print(f"File is uploaded to gs://{gcs_bucket}/golden-logits/{model_id}/{output_path}.")


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", type=str, required=True, help="The identifier of the model to use.")
  parser.add_argument("--output-path", type=str, required=True, help="The path to save the generated golden logits.")
  parser.add_argument("--prompts", type=str, required=True, help="A semicolon-separated list of prompts.")
  parser.add_argument(
      "--apply-chat-template",
      action="store_true",
      help="Apply chat template from the HF processor. Used for image+text input. Pass this flag to enable; omit to disable.",
  )
  parser.add_argument(
      "--gcs-bucket", type=str, required=False, default=None, help="A GCS bucket to store logits, without gs://."
  )
  parser.add_argument("--hf-model-path", type=str, required=False, default=None, help="local path to checkpoint if exists.")
  parser.add_argument(
      "--image-paths", type=str, required=False, default=None, help="A semicolon-separated list of image_paths."
  )
  parser.add_argument(
      "--format", type=str, required=False, default="json", help="The output format for the golden logits. (json, pickle)"
  )
  args = parser.parse_args(raw_args)
  prompts = args.prompts.split(";")
  image_paths = args.image_paths.split(";") if args.image_paths else []
  if image_paths:
    assert len(image_paths) == len(
        prompts
    ), "when image paths are provided, image_paths and prompts must have the same length."
  if args.apply_chat_template:
    assert image_paths, "apply_chat_template is only used for image+text input, so image_paths must be provided."
  save_golden_logits(
      args.model_id, args.output_path, prompts, args.apply_chat_template, args.gcs_bucket, args.hf_model_path, image_paths, args.format
  )


if __name__ == "__main__":
  main()
