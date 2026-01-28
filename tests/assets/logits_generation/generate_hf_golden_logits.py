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

python3 -m tests.assets.logits_generation.generate_hf_golden_logits --model-id=deepseek-ai/DeepSeek-V2-Lite \
     --output-path=golden_DeepSeek-V2-Lite.jsonl --prompts='I love to;Today is a;What is the' \
     --gcs-bucket=my-gcs-bucket

For large models, you can use an m1 cpu. Calling the script directly instead of calling MaxText module \
can skip importing unnecessary dependencies.
For large Hugginface checkpoints, you can use pre-downloaded checkpoints with --hf-model-path argument.
For multimodal models, use --image-paths argument to provide image path(s),\
  use --apply-chat-template=true if use HF chat template to format image+prompt.\
  When using chat template, the prompt should not contain image placeholders.
For multimodal logits, since the model is not suppose to generate image, the logits correspond to images \
  tokens can be close to 0, using --output-format=pickle is recommended to preserve precision.

More examples:
python3 -m tests.assets.logits_generation.generate_hf_golden_logits --model-id=meta-llama/Llama-4-Scout-17B-16E \
     --output-path=golden_Llama-4-Scout-17B-16E_vision.jsonl --prompts='Describe this image.' \
     --apply-chat-template --gcs-bucket=<bucket> --hf-model-path=<hf_checkpoint_path> \
     --image-paths=tests/assets/test_image.jpg --output-format=pickle

python3 -m tests.assets.logits_generation.generate_hf_golden_logits --model-id=google/gemma-3-4b-it \
     --output-path=golden_gemma-3-4b-it_vision.jsonl --prompts='<start_of_image>' \
     --gcs-bucket=<bucket> --hf-model-path=<hf_checkpoint_path> \
     --image-paths=tests/assets/test_image.jpg
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoProcessor
import jsonlines
import pickle
import numpy as np
from google.cloud import storage
from PIL import Image
from maxtext.inference.inference_utils import str2bool

# Load the tokenizer and model from Hugging Face


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


def save_golden_logits(
    model_id,
    output_path,
    prompt_texts,
    apply_chat_template,
    gcs_bucket,
    hf_model_path,
    hf_load_dtype,
    trust_remote_code,
    image_paths,
    output_format,
):
  """save golden logits"""
  if hf_model_path is None:
    hf_model_path = model_id

  if model_id.startswith("meta-llama/Llama-4"):
    from transformers import Llama4ForConditionalGeneration  # pylint: disable=import-outside-toplevel

    model_class = Llama4ForConditionalGeneration
  else:
    from transformers import AutoModelForCausalLM  # pylint: disable=import-outside-toplevel

    model_class = AutoModelForCausalLM

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  print(f"loading model from {hf_model_path}")

  if hf_load_dtype == "float32":
    torch_dtype = torch.float32
  elif hf_load_dtype == "bfloat16":
    torch_dtype = torch.bfloat16
  else:
    raise ValueError

  model = model_class.from_pretrained(
      hf_model_path,
      dtype=torch_dtype,
      trust_remote_code=trust_remote_code,
  )

  all_data_to_save = []
  for i, prompt_text in enumerate(prompt_texts):
    # 1. Prepare inputs for the model and base data for saving
    data_to_save = {"prompt": prompt_text}
    if image_paths:
      image = Image.open(image_paths[i]).convert("RGB")
      if model_id.startswith("meta-llama/Llama-4"):
        image = image.resize((336, 336))
      processor = AutoProcessor.from_pretrained(model_id, token=True) if image_paths else None
      if apply_chat_template:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
      else:
        formatted_prompt = prompt_text
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt", add_special_tokens=False)

      inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
      print(f"{apply_chat_template=} {formatted_prompt=}")
      data_to_save.update({"formatted_prompt": formatted_prompt, "image_path": image_paths[i]})
    else:
      input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
      inputs = {"input_ids": input_ids}

    # 2. Run inference
    with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits.cpu().to(torch.float32).numpy()

    # 3. Populate final data dictionary with tensors from inputs and logits
    for key, value in inputs.items():
      new_key = "tokens" if key == "input_ids" else key
      data_to_save[new_key] = value.cpu().numpy()[0]
    data_to_save["logits"] = logits[0]

    print(f"Token length is {len(data_to_save['tokens'])} for prompt: {prompt_text}")
    print(f"raw ids: {data_to_save['tokens']}")

    # 4. Convert numpy arrays to lists if format is json
    if output_format == "json":
      for key, value in data_to_save.items():
        if isinstance(value, np.ndarray):
          data_to_save[key] = value.tolist()

    all_data_to_save.append(data_to_save)

  # 5. Save the collected data
  if output_format == "json":
    with jsonlines.open(output_path, "w") as f:
      f.write_all(all_data_to_save)
  elif output_format == "pickle":
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
      help="Apply chat template from the HF processor. Use for image+text input. Pass this flag to enable; omit to disable.",
  )
  parser.add_argument(
      "--gcs-bucket", type=str, required=False, default=None, help="A GCS bucket to store logits, without gs://."
  )
  parser.add_argument(
      "--hf-model-path", type=str, required=False, default=None, help="local path to checkpoint if exists."
  )
  parser.add_argument(
      "--hf-load-dtype",
      type=str,
      required=False,
      choices=["float32", "bfloat16"],
      default="float32",
      help="model_class.from_pretrained: dtype",
  )
  parser.add_argument(
      "--trust-remote-code",
      type=str2bool,
      required=False,
      default=True,
      help="model_class.from_pretrained: trust_remote_code",
  )
  parser.add_argument(
      "--image-paths", type=str, required=False, default=None, help="A semicolon-separated list of image_paths."
  )
  parser.add_argument(
      "--output-format",
      type=str,
      required=False,
      default="json",
      help="The output format for the golden logits. (json, pickle)",
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
      args.model_id,
      args.output_path,
      prompts,
      args.apply_chat_template,
      args.gcs_bucket,
      args.hf_model_path,
      args.hf_load_dtype,
      args.trust_remote_code,
      image_paths,
      args.output_format,
  )


if __name__ == "__main__":
  main()
