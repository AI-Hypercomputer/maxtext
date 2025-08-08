# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
from google.cloud import storage

# Load the tokenizer and model from Hugging Face

model_id = "meta-llama/Llama-2-70b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)


# Your prompt text
prompt_texts = ["I love to", "Today is a", "What is the"]
all_data_to_save = []

output_path = "golden_data_llama2-70b.jsonl"


for prompt_text in prompt_texts:
  # Encode the prompt text
  input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

  # Get the logits for the prompt + completion
  with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

    # Convert logits to fp32
    logits = logits.cpu().numpy().astype("float32")

    # Prepare data to be saved
    data_to_save = {
        "prompt": prompt_text,
        "tokens": input_ids.tolist()[0],
        "logits": logits.tolist()[0],  # Convert numpy array to list for JSON serialization
    }
    all_data_to_save.append(data_to_save)

with jsonlines.open(output_path, "w") as f:
  f.write_all(all_data_to_save)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)

  blob.upload_from_filename(source_file_name)


upload_blob("maxtext-llama", output_path, f"llama2-70b/golden-logits/{output_path}")
print("File", repr(output_path), "uploaded to", f"llama2-70b/golden-logits/{output_path}.")
