"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Usage: python3 golden_llama3-70b_export.py --model-id meta-llama/Meta-Llama-3-70B --output-path llama3-70b/golden_logits/golden_data_llama3-70b.jsonl
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

def convert(model_id, output_path):
    

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )

            
    # Your prompt text  
    prompt_texts = ["I love to"]
    all_data_to_save = []

    output_filename = output_path.split('/')[-1]

    for prompt_text in prompt_texts:
        # Encode the prompt text  
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt')  

        # Get the logits for the prompt + completion  
        with torch.no_grad():  
            outputs = model(input_ids)
            logits = outputs.logits  

            # Convert logits to fp32  
            logits = logits.cpu().numpy().astype('float32')  

            # Prepare data to be saved  
            data_to_save = {  
                "prompt": prompt_text,  
                "tokens": input_ids.tolist()[0],  
                "logits": logits.tolist()[0]  # Convert numpy array to list for JSON serialization  
            }  
            all_data_to_save.append(data_to_save)
        
    with jsonlines.open(output_filename,'w') as f:    
        f.write_all(all_data_to_save)

    
    
    upload_blob('maxtext-llama', output_filename, output_path)
    print('File {} uploaded to {}.'.format(
        output_filename,
        output_path))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    convert(args.model_id, args.output_path)
    

