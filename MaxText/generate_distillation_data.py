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
    --hf-access-token <access token> --hf-repo-id <hf repository id> \
    --batch-size 1024 --num-batches 100 \
    --num-generations 2 \
    --max-output-length 128 --max-target-length 256 --use-chat-template

Running this command executes 100 processing steps.
In each step, it generates completions for a batch of 40 prompts.
This results in inference running on 4000 prompts overall, producing 2 samples per prompt.
Note:
Make sure to run maxengine server in a new terminal before executing this command. Example command to run maxengine server:
  python3 -m MaxText.maxengine_server MaxText/configs/base.yml \
    model_name=deepseek2-16b tokenizer_path=deepseek-ai/DeepSeek-V2-Lite-chat tokenizer_type=huggingface \
    load_parameters_path=<unscanned checkpoint path> \
    per_device_batch_size=10 multi_sampling=True ici_tensor_parallelism=4 \
    decode_sampling_strategy=weighted scan_layers=False
"""

import argparse
import asyncio
import grpc
import json
import transformers

from datasets import Dataset
from huggingface_hub import create_repo, get_full_repo_name, repo_exists, upload_file

from MaxText import max_logging
from MaxText.input_pipeline import _distillation_data_processing

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from tqdm.asyncio import tqdm

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
  )

  response = stub.Decode(decode_request)
  completion_tokens = [[] for _ in range(config.num_generations)]
  async for resp in response:
    for idx, sample in enumerate(resp.stream_content.samples):
      resp_tokens = sample.token_ids
      completion_tokens[idx].extend(resp_tokens)

  outputs = []
  for tokens in completion_tokens:
    completion = tokenizer.decode(tokens, skip_special_tokens=True)
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
  progress_bar = tqdm(total=len(requests))
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


def upload_data_to_hf(distillation_data, batch_num, hf_repo_id, hf_access_token):
  """Upload dataset to Hugging Face."""
  full_repo_name = get_full_repo_name(model_id=hf_repo_id, token=hf_access_token)
  if not repo_exists(repo_id=full_repo_name, repo_type="dataset", token=hf_access_token):
    max_logging.log("Repository doesn't exist on Hugging Face, creating a new one.")
    try:
      repo_url = create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, token=hf_access_token)
      max_logging.log(f"Successfully created repository on Hugging Face: {repo_url}.")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"Error in creating repository on Hugging Face: {e}")
      raise e

  distillation_dataset = Dataset.from_list(distillation_data)
  max_logging.log(f"Pushing dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  try:
    parquet_file_name = f"distillation-data-{batch_num}.parquet"
    distillation_dataset.to_parquet(parquet_file_name)
    upload_file(
        repo_id=full_repo_name,
        repo_type="dataset",
        path_or_fileobj=parquet_file_name,
        path_in_repo=f"data/{parquet_file_name}",
        commit_message=f"Uploading dataset batch number {batch_num}",
        token=hf_access_token,
    )
    max_logging.log(f"Successfully pushed dataset to Hugging Face: https://huggingface.co/datasets/{full_repo_name}")
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Error in pushing dataset to Hugging Face: {e}")
    raise e


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
    upload_data_to_hf(distillation_data, batch_num, config.hf_repo_id, config.hf_access_token)


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
  parser.add_argument(
      "--max-output-length", type=int, required=True, help="The maximum completion tokens to generate for a prompt."
  )
  parser.add_argument(
      "--max-target-length", type=int, default=2048, help="The maximum prompt length plus the output completion length."
  )
  parser.add_argument(
      "--num-generations", type=int, required=False, default=1, help="Number of samples to generate per prompt."
  )
  parser.add_argument(
      "--hf-repo-id", type=str, required=True, help="Name of Hugging Face repository to upload generated dataset."
  )
  parser.add_argument("--batch-size", type=int, required=True, help="Number of prompts to process in a batch.")
  parser.add_argument("--num-batches", type=int, required=True, help="Total number of batches of prompts to process.")
  config, _ = parser.parse_known_args()

  assert (
      config.max_output_length < config.max_target_length
  ), "Maximum output length of completion should be less than maximum target length."
  generate_data(config)
