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
  python3 -m MaxText.generate_distillation_data MaxText/configs/base.yml \
  run_name=<run_name> base_output_directory=<base_output_directory> \
  model_name=llama2-7b tokenizer_path=meta-llama/Llama-2-7b-hf tokenizer_type=huggingface \
  dataset_type=hf hf_path='HuggingFaceH4/ultrachat_200k' train_split=train_sft train_data_columns=[messages] \
  load_parameters_path=<model checkpoint path> \
  per_device_batch_size=1 max_target_length=1024 max_prefill_predict_length=512 \
  steps=100 decode_sampling_strategy=greedy ici_tensor_parallelism=4 \
  --hf-repo-id="<Hugging Face repository id for dataset>" --num_generations=4
"""

import argparse
import asyncio
import grpc
import sys
import transformers

from dataclasses import dataclass
from datasets import Dataset

from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxengine_config
from MaxText import pyconfig
from MaxText.input_pipeline import _distillation_data_processing
from MaxText.train import load_next_batch

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core import server_lib
from tqdm.asyncio import tqdm

_GRPC_KEEPALIVE_TIMEOUT_MS = 10000


@dataclass
class InputRequest:
  prompt_tokens: str = ""
  prompt_true_length: int = 0


async def get_request(input_requests):
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request


async def send_grpc_request(request, server_url, num_generations):
  """Sends grpc synchronous request since the current grpc server is sync."""
  options = [("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS)]
  async with grpc.aio.insecure_channel(server_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    response = stub.Decode(request)
    completion_tokens = [[] for _ in range(num_generations)]
    async for resp in response:
      for idx, sample in enumerate(resp.stream_content.samples):
        resp_tokens = sample.token_ids
        completion_tokens[idx].extend(resp_tokens)
    return completion_tokens


async def send_request(config, parser_config, request, progress_bar):  # pylint: disable=redefined-outer-name
  """Sends the request to JetStream server."""
  prompt_tokens = request.prompt_tokens
  prompt_length = request.prompt_true_length

  decode_request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=prompt_tokens[:prompt_length]),
      max_tokens=config.max_target_length - config.max_prefill_predict_length,
      num_samples=parser_config.num_generations,  # number of responses to generate for each request
  )
  completion_tokens = await send_grpc_request(
      request=decode_request,
      server_url=f"localhost:{parser_config.jetstream_server_port}",
      num_generations=parser_config.num_generations,
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      legacy=False,
      token=config.hf_access_token,
  )

  outputs = []
  prompt = tokenizer.decode(prompt_tokens[:prompt_length])
  for tokens in completion_tokens:
    completion = tokenizer.decode(tokens)
    outputs.append(
        {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": completion}],
        }
    )
  progress_bar.update(1)
  return outputs


async def run_inference(config, parser_config, requests):  # pylint: disable=redefined-outer-name
  """Asynchronously runs inference on JetStream server."""
  progress_bar = tqdm(total=len(requests))
  progress_bar.set_description(f"Running inference on {len(requests)} prompts")

  tasks = []
  async for request in get_request(requests):
    tasks.append(
        asyncio.create_task(
            send_request(
                config=config,
                parser_config=parser_config,
                request=request,
                progress_bar=progress_bar,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)
  progress_bar.close()
  return outputs


def generate_completions(config, parser_config, data):  # pylint: disable=redefined-outer-name
  """Generates num_generations of completion for each prompt in data["prompt"]."""
  requests = []
  prompts = data["prompt"]
  prompts_true_length = data["prompt_true_length"]
  for idx in range(prompts.shape[0]):
    prompt_tokens = prompts[idx][0]
    prompt_true_length = prompts_true_length[idx][0]
    request = InputRequest(prompt_tokens, prompt_true_length)
    requests.append(request)
  outputs = asyncio.run(
      run_inference(
          config=config,
          parser_config=parser_config,
          requests=requests,
      ),
  )
  return [output for output_per_prompt_list in outputs for output in output_per_prompt_list]


def upload_data_to_hf(distillation_data, hf_repo_id, hf_access_token):
  """Uploads dataset to Hugging Face."""
  distillation_dataset = Dataset.from_list(distillation_data)
  max_logging.log(f"Pushing dataset to Hugging Face Hub: {hf_repo_id}")
  try:
    distillation_dataset.push_to_hub(repo_id=hf_repo_id, private=True, token=hf_access_token)
    max_logging.log(f"Successfully pushed dataset to Hugging Face: https://huggingface.co/datasets/{hf_repo_id}")
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Error in pushing dataset to Hugging Face: {e}")
    raise e


def generate_data(config, parser_config):  # pylint: disable=redefined-outer-name
  """Generates data for distillation."""
  assert config.tokenizer_type == "huggingface", "Please use tokenizer from Hugging Face."

  data_iterator = _distillation_data_processing.get_data_iterator(config)

  data_batch = None
  distillation_data = []
  for _ in range(config.steps):
    data_batch = load_next_batch(data_iterator, data_batch, config)
    prompts_completions = generate_completions(config, parser_config, data_batch)
    distillation_data.extend(prompts_completions)
  return distillation_data


def run_jetstream_server(config, parser_config):  # pylint: disable=redefined-outer-name
  """Runs JetStream server at localhost:{parser_config.jetstream_server_port}."""
  devices = server_lib.get_devices()
  server_config = maxengine_config.get_server_config(config.inference_server, config)

  jetstream_server = server_lib.run(
      threads=256,
      port=parser_config.jetstream_server_port,
      config=server_config,
      devices=devices,
      multi_sampling=True,
  )
  max_logging.log(f"Jetstream server is running on port: {parser_config.jetstream_server_port}")
  return jetstream_server


def main(config, parser_config):  # pylint: disable=redefined-outer-name
  jetstream_server = None
  try:
    jetstream_server = run_jetstream_server(config, parser_config)
    distillation_data = generate_data(config, parser_config)
    upload_data_to_hf(distillation_data, parser_config.hf_repo_id, config.hf_access_token)
  finally:
    max_logging.log("Shutting down Jetstream server.")
    if jetstream_server:
      jetstream_server.stop()
    max_logging.log("Jetstream server is shutdown.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--jetstream-server-port", type=str, default=9000)
  parser.add_argument("--num-generations", type=int, required=False, default=1)
  parser.add_argument("--hf-repo-id", type=str, required=True)
  parser_config, _ = parser.parse_known_args()

  model_args = sys.argv
  parser_args = ["--jetstream-server-port", "--num-generations", "--hf-repo-id"]
  for arg in parser_args:
    model_args = [s for s in model_args if not s.startswith(arg)]

  config = pyconfig.initialize(model_args)
  max_utils.print_system_information()

  main(config, parser_config)
