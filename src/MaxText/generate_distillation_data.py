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
This script executes the data generation step for Response-based Knowledge Distillation.
Knowledge Distillation is a compression technique that transfers knowledge
from a larger (teacher) model to a smaller (student) model.
The script runs inference on a teacher model to create output samples.
This generated dataset can be used to fine-tune a student model.

Example command:
  python3 -m MaxText.generate_distillation_data \
    --dataset-path HuggingFaceH4/ultrachat_200k --data-split train_sft --data-columns messages \
    --tokenizer-path deepseek-ai/DeepSeek-V2-Lite-chat \
    --hf-access-token <access token> \
    --batch-size 1024 --num-batches 10 \
    --num-generations 2 \
    --max-prefill-length 256 --max-target-length 2048 \
    --use-chat-template --remove-local-dataset-files \
    upload-to-hf --hf-repo-id <hf repository id>

Running this command executes 10 processing steps.
In each step, it generates completions for a batch of 1024 prompts.
This results in inference running on 10240 prompts overall, producing 2 unique samples per prompt.
Some prompts may be filtered out if prompt tokens are longer than `max-prefill-length`.
`max-target-length` is the max length of prompt tokens and expected completion tokens.
Set `--remove-local-dataset-files` to remove dataset files created locally after uploading to Hugging Face or GCS.
`upload-to-hf` will upload the dataset to Hugging Face and `upload-to-gcs` will upload the dataset to GCS.
For more information, check out `python3 -m MaxText.generate_distillation_data --help`.
Note:
Make sure to run maxengine server in a new terminal before executing this command. Example command to run maxengine server:
  python3 -m MaxText.maxengine_server src/MaxText/configs/base.yml \
    model_name=deepseek2-16b tokenizer_path=deepseek-ai/DeepSeek-V2-Lite-chat tokenizer_type=huggingface \
    load_parameters_path=<unscanned checkpoint path> \
    max_target_length=2048 max_prefill_predict_length=256 \
    per_device_batch_size=10 multi_sampling=True ici_tensor_parallelism=4 \
    decode_sampling_strategy=weighted scan_layers=False
"""

import argparse
import asyncio
import grpc
import json
import os
import transformers

from datasets import Dataset
from huggingface_hub import create_repo, get_full_repo_name, repo_exists, upload_file

from MaxText import max_logging
from MaxText.input_pipeline import _distillation_data_processing
from MaxText.utils import gcs_utils

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
import tqdm.asyncio

_GRPC_KEEPALIVE_TIMEOUT_MS = 10000
_GRPC_MAX_ATTEMPTS = 5


async def get_request(input_requests):
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request


async def send_request(config, request, stub, tokenizer, progress_bar):  # pylint: disable=redefined-outer-name
  """Sends the request to JetStream server."""
  prompt = request.prompt
  prompt_token_ids = request.prompt_token_ids
  actual_completion = request.actual_completion

  decode_request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=prompt_token_ids),
      max_tokens=request.max_output_tokens,
      num_samples=config.num_generations,  # number of responses to generate for each request
      has_bos=True,
  )

  response = stub.Decode(decode_request)
  completion_tokens = [[] for _ in range(config.num_generations)]
  async for resp in response:
    for idx, sample in enumerate(resp.stream_content.samples):
      resp_tokens = sample.token_ids
      completion_tokens[idx].extend(resp_tokens)

  outputs = []
  for tokens in completion_tokens:
    completion = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    outputs.append(
        {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": completion}],
            "actual_completion": [{"role": "assistant", "content": actual_completion}],
        }
    )
  progress_bar.update(1)
  return outputs


async def run_inference(config, requests, tokenizer):  # pylint: disable=redefined-outer-name
  """Asynchronously runs inference on JetStream server."""
  progress_bar = tqdm.asyncio.tqdm(total=len(requests))
  progress_bar.set_description(f"Running inference on {len(requests)} prompts")

  server_url = f"localhost:{config.jetstream_server_port}"
  options = []
  options.append(("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS))
  options.append(("grpc.enable_retries", 1))
  service_config_json = json.dumps(
      {
          "methodConfig": [
              {
                  "name": [{}],
                  "retryPolicy": {
                      "maxAttempts": _GRPC_MAX_ATTEMPTS,
                      "initialBackoff": "0.2s",
                      "maxBackoff": "1s",
                      "backoffMultiplier": 2,
                      "retryableStatusCodes": ["UNAVAILABLE"],
                  },
              }
          ]
      }
  )
  options.append(("grpc.service_config", service_config_json))
  tasks = []
  async with grpc.aio.insecure_channel(server_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    async for request in get_request(requests):
      tasks.append(
          asyncio.create_task(
              send_request(
                  config=config,
                  request=request,
                  stub=stub,
                  tokenizer=tokenizer,
                  progress_bar=progress_bar,
              )
          )
      )
    outputs = await asyncio.gather(*tasks)
  progress_bar.close()
  return outputs


def generate_completions(config, requests, tokenizer):  # pylint: disable=redefined-outer-name
  """Generates num_generations of completions for each prompt in request."""
  outputs = asyncio.run(
      run_inference(
          config=config,
          requests=requests,
          tokenizer=tokenizer,
      ),
  )
  return [output for output_per_prompt_list in outputs for output in output_per_prompt_list]


def upload_data_to_hf(config, parquet_file_name, batch_num):  # pylint: disable=redefined-outer-name
  """Upload dataset to Hugging Face."""
  full_repo_name = get_full_repo_name(model_id=config.hf_repo_id, token=config.hf_access_token)
  if not repo_exists(repo_id=full_repo_name, repo_type="dataset", token=config.hf_access_token):
    max_logging.log("Repository doesn't exist on Hugging Face, creating a new one.")
    try:
      repo_url = create_repo(repo_id=config.hf_repo_id, repo_type="dataset", private=True, token=config.hf_access_token)
      max_logging.log(f"Successfully created repository on Hugging Face: {repo_url}.")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Error in creating repository on Hugging Face: {e}")
      raise e

  max_logging.log(f"Pushing dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  try:
    upload_file(
        repo_id=full_repo_name,
        repo_type="dataset",
        path_or_fileobj=parquet_file_name,
        path_in_repo=f"data/{parquet_file_name}",
        commit_message=f"Uploading dataset batch number {batch_num}",
        token=config.hf_access_token,
    )
    max_logging.log(f"Successfully pushed dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Error in pushing dataset to Hugging Face: {e}")
    raise e


def upload_data_to_gcs(config, source_file_name):  # pylint: disable=redefined-outer-name
  """Uploads dataset to Google Cloud Storage bucket."""
  data_path = gcs_utils.add_trailing_slash(config.gcs_data_path)
  destination_name = f"gs://{config.gcs_bucket}/{data_path}{source_file_name}"
  max_logging.log(f"Pushing dataset to GCS: {destination_name}")
  try:
    gcs_utils.upload_blob(destination_name, source_file_name)
    max_logging.log(f"Successfully pushed dataset to GCS: {destination_name}")
  except FileNotFoundError as e:
    max_logging.log(f"Error in pushing dataset to GCS: '{source_file_name}' not found during upload attempt.")
    raise e
  except Exception as e:
    max_logging.log(f"Error in pushing dataset to GCS: {e}")
    raise e


def upload_data(config, data, batch_num):  # pylint: disable=redefined-outer-name
  """Uploads dataset to Google Cloud Storage or Hugging Face."""
  distillation_dataset = Dataset.from_list(data)
  parquet_file_name = f"distillation-data-{batch_num}.parquet"
  distillation_dataset.to_parquet(parquet_file_name)
  if config.upload == "upload-to-hf":
    upload_data_to_hf(config, parquet_file_name, batch_num)
  elif config.upload == "upload-to-gcs":
    upload_data_to_gcs(config, parquet_file_name)
  # remove local dataset files after upload
  if config.remove_local_dataset_files and os.path.exists(parquet_file_name):
    try:
      os.remove(parquet_file_name)
    except OSError as e:
      max_logging.log(f"Unable to remove local dataset file {parquet_file_name}: {e}")


def generate_data(config):  # pylint: disable=redefined-outer-name
  """Generates data for distillation."""
  dataset = _distillation_data_processing.load_dataset(config)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      token=config.hf_access_token,
  )

  start_idx = 0
  distillation_data = []
  for batch_num in range(config.num_batches):
    data = dataset[start_idx : start_idx + config.batch_size]
    start_idx += config.batch_size
    sampled_dataset = Dataset.from_dict(data)
    sampled_dataset = _distillation_data_processing.process_dataset(config, sampled_dataset)
    requests = _distillation_data_processing.filter_dataset(config, sampled_dataset, tokenizer)
    distillation_data = generate_completions(config, requests, tokenizer)
    upload_data(config, distillation_data, batch_num)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--jetstream-server-port", type=str, default=9000, help="JetStream server port.")
  parser.add_argument("--dataset-type", type=str, default="huggingface", help="Type of dataset.")
  parser.add_argument("--dataset-path", type=str, required=True, help="Path to Hugging Face dataset.")
  parser.add_argument("--data-split", type=str, required=True, help="Subset of data to load, eg. train or test.")
  parser.add_argument("--data-columns", nargs="+", required=True, help="Columns names that contain relevant data.")
  parser.add_argument(
      "--hf-access-token", type=str, required=True, help="Access token used to load a tokenizer from Hugging Face."
  )
  parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to Hugging Face tokenizer.")
  parser.add_argument("--use-chat-template", action="store_true", help="Enable tokenizer to apply a chat template.")
  parser.add_argument("--max-prefill-length", type=int, default=256, help="The maximum prompt length.")
  parser.add_argument(
      "--max-target-length", type=int, default=2048, help="The maximum prompt length plus the output completion length."
  )
  parser.add_argument(
      "--num-generations", type=int, required=False, default=1, help="Number of samples to generate per prompt."
  )
  parser.add_argument("--batch-size", type=int, required=True, help="Number of prompts to process in a batch.")
  parser.add_argument("--num-batches", type=int, required=True, help="Total number of batches of prompts to process.")
  parser.add_argument(
      "--remove-local-dataset-files", action="store_true", help="Set to remove local dataset files after upload."
  )

  # Subparser for available upload commands (upload to GCS, upload to Hugging Face)
  subparsers = parser.add_subparsers(dest="upload", title="Available upload commands", required=True)

  # Subparser to upload dataset to Google Cloud Storage
  upload_to_gcs_parser = subparsers.add_parser("upload-to-gcs", help="Upload dataset to Google Cloud Storage.")
  upload_to_gcs_parser.add_argument(
      "--gcs-bucket", type=str, required=True, help="Name of GCS bucket to upload generated dataset."
  )
  upload_to_gcs_parser.add_argument("--gcs-data-path", type=str, required=True, help="Path to store dataset in GCS bucket.")

  # Subparser to upload dataset to Hugging Face
  upload_to_hf_parser = subparsers.add_parser(
      "upload-to-hf",
      help="Upload dataset to Hugging Face.",
  )
  upload_to_hf_parser.add_argument(
      "--hf-repo-id", type=str, required=True, help="Name of Hugging Face repository to upload generated dataset."
  )

  config = parser.parse_args()

  assert (
      config.max_prefill_length < config.max_target_length
  ), "Maximum length of prompt should be less than maximum target length."
  generate_data(config)
