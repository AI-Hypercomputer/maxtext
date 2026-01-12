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
This script implements an OpenAI-compatible API server for MaxText models.

It uses FastAPI to create endpoints for `/v1/completions` and
`/v1/chat/completions`. The server runs in a multi-process JAX environment,
with the coordinator process (rank 0) managing the web server and all
processes participating in a batched inference loop to handle incoming
requests efficiently.
"""

from typing import Union
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid

import uvicorn

from fastapi import FastAPI, HTTPException

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from openai_harmony import (
  load_harmony_encoding,
  HarmonyEncodingName,
  Role,
)

from benchmarks.api_server.maxtext_generator import MaxTextGenerator
from benchmarks.api_server.server_models import (
  CompletionRequest,
  CompletionResponse,
  CompletionChoice,
  Usage,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChoice,
  ChatMessage,
)
from benchmarks.api_server import server_utils

# ----------------------------
# Init
# ----------------------------

# JAX distributed initialization must happen before any other JAX calls.
# We suppress the normal logger until after JAX is initialized.
logging.basicConfig(level=logging.WARNING)
print("Initializing MaxTextGenerator and JAX distributed system...")
LLM = MaxTextGenerator(sys.argv)
rank = jax.process_index()

# Now that JAX is initialized, we can get our rank-specific logger.
# The actual handler/formatter configuration will be done by Uvicorn.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure our logger passes INFO messages.
logger.info("MaxTextGenerator initialization complete.")

harmony_enc = None
try:
  harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
  logger.info("Harmony encoding for gpt-oss loaded successfully.")
except ImportError:
  logger.warning("openai_harmony not installed. GPT-OSS Harmony format will not be available.")
except (RuntimeError, ValueError) as e:
  logger.error("Failed to load Harmony encoding: %s", e, exc_info=True)


app = FastAPI()

# Global state for communication between threads.
request_queue = queue.Queue()
# A thread-safe dict to hold responses, keyed by request_id.
response_dict = {}
response_lock = threading.Lock()

# Batching configuration
BATCH_TIMEOUT_S = 0.1  # 100ms
# Timeout for a client waiting for a response.
REQUEST_TIMEOUT_S = int(os.environ.get("MAXTEXT_REQUEST_TIMEOUT_S", "36000"))


async def _queue_and_wait_for_response(request: Union[CompletionRequest, ChatCompletionRequest]):
  """
  Puts a request on the processing queue and waits for a response.

  This asynchronous function is the core of handling client requests. It generates
  a unique ID for the request, places it in a global queue to be processed by
  the batching loop, and then waits until the response is available in a
  shared dictionary or until a timeout occurs.

  Args:
      request: The incoming request object, either for a completion or a chat completion.

  Returns:
      The response data once it's available.

  Raises:
      HTTPException: If the request times out (504) or if an error occurs
                     during processing (500).
  """
  request_id = f"req_{uuid.uuid4().hex}"
  request_queue.put((request_id, request))

  start_time = time.time()
  while time.time() - start_time < REQUEST_TIMEOUT_S:
    with response_lock:
      if request_id in response_dict:
        response_data = response_dict.pop(request_id)
        if "error" in response_data:
          raise HTTPException(status_code=500, detail=response_data["error"])
        return response_data
    # Yield control to the event loop to allow other tasks to run.
    await asyncio.sleep(0.05)

  raise HTTPException(status_code=504, detail="Request timed out.")


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
  """Handles completion requests with dynamic batching."""
  return await _queue_and_wait_for_response(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
  """Handles chat completion requests with dynamic batching."""
  return await _queue_and_wait_for_response(request)


@app.get("/")
def health_check():
  """
  Provides a simple health check endpoint.

  Returns:
      A dictionary indicating the server status.
  """
  return {"status": "ok", "message": "MaxText API server is running."}


def run_server():
  """Runs the Uvicorn server in a separate thread."""
  uvicorn.run(app, host="0.0.0.0", port=8000)


# Define a maximum size for the request payload to be broadcasted.
# This avoids broadcasting variable-sized arrays, which can be complex.
MAX_REQUEST_SIZE = 65536 * 10


def _build_chat_completion_response(request, completion_result, llm):
  """Builds a ChatCompletionResponse from a single completion result."""
  text_out = completion_result.text
  if "gpt-oss" in request.model and harmony_enc:
    try:
      parsed_messages = harmony_enc.parse_messages_from_completion_tokens(completion_result.tokens, role=Role.ASSISTANT)
      user_visible = "".join(part.text for m in parsed_messages if m.channel == "final" for part in m.content)
      if user_visible:
        text_out = user_visible
      else:
        logger.warning(
          "Harmony parsing for gpt-oss did not yield content in the 'final' channel. Falling back to raw text."
        )
    except (ValueError, IndexError) as e:
      logger.error("Harmony parsing failed for gpt-oss: %s. Falling back to raw text.", e, exc_info=True)

  want_top_logprobs = (
    (request.top_logprobs or 0) > 0 if isinstance(request, ChatCompletionRequest) else (request.logprobs or 0) > 0
  )
  lp_payload = server_utils.to_openai_logprobs(
    getattr(completion_result, "logprobs", None), llm, want_top=want_top_logprobs
  )
  text_out, lp_payload, finish_reason = server_utils.apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
  if finish_reason is None:
    finish_reason = completion_result.finish_reason

  usage = Usage(
    prompt_tokens=completion_result.prompt_token_count,
    completion_tokens=completion_result.completion_token_count,
    total_tokens=completion_result.prompt_token_count + completion_result.completion_token_count,
  )
  return ChatCompletionResponse(
    model=request.model,
    choices=[
      ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text_out),
        finish_reason=finish_reason,
        logprobs=lp_payload,
      )
    ],
    usage=usage,
  )


def _build_completion_response(request, completions, prompts, llm):
  """Builds a CompletionResponse from a list of completion results."""
  choices = []
  prompt_tokens_total = 0
  completion_tokens_total = 0

  for idx, _ in enumerate(prompts):
    item = completions[idx]
    text_out = item.text
    lp_payload = server_utils.to_openai_logprobs(
      getattr(item, "logprobs", None), llm, want_top=(request.logprobs or 0) > 0
    )
    finish_reason = getattr(item, "finish_reason", "stop")
    text_out, lp_payload, stop_reason = server_utils.apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
    if stop_reason is not None:
      finish_reason = stop_reason

    prompt_tokens_total += item.prompt_token_count
    completion_tokens_total += item.completion_token_count

    choices.append(
      CompletionChoice(
        text=text_out,
        index=idx,
        logprobs=lp_payload,
        finish_reason=finish_reason,
      )
    )

  usage = Usage(
    prompt_tokens=prompt_tokens_total,
    completion_tokens=completion_tokens_total,
    total_tokens=prompt_tokens_total + completion_tokens_total,
  )
  return CompletionResponse(
    model=request.model,
    choices=choices,
    usage=usage,
  )


def _create_response(request, completions, prompts, is_chat, llm):
  """Creates either a CompletionResponse or ChatCompletionResponse."""
  if is_chat:
    # Chat API only ever processes one prompt at a time from the server's perspective.
    return _build_chat_completion_response(request, completions[0], llm)
  else:
    return _build_completion_response(request, completions, prompts, llm)


def _collect_batched_requests():
  """Waits for and collects a batch of requests from the queue."""
  batched_items = []
  start_time = time.time()
  while len(batched_items) < LLM.batch_size and (time.time() - start_time) < BATCH_TIMEOUT_S:
    try:
      item = request_queue.get(timeout=0.01)
      batched_items.append(item)
    except queue.Empty:
      if batched_items:
        break  # Process what we have if timeout is reached
  return batched_items


def _prepare_batch_for_broadcast(batched_items):
  """Prepares the batch payload and request map for broadcasting."""
  _, first_request = batched_items[0]

  logprobs_param = None
  if isinstance(first_request, ChatCompletionRequest):
    if first_request.logprobs:
      logprobs_param = first_request.top_logprobs if first_request.top_logprobs is not None else 1
  else:  # CompletionRequest
    logprobs_param = first_request.logprobs

  params = {
    "max_tokens": first_request.max_tokens,
    "logprobs": logprobs_param,
    "echo": getattr(first_request, "echo", False),
    "stop": first_request.stop,
    "temperature": first_request.temperature,
    "seed": first_request.seed,
    "top_k": first_request.top_k,
    "top_p": first_request.top_p,
  }

  all_prompts = []
  request_info_map = []
  for req_id, req in batched_items:
    is_chat = isinstance(req, ChatCompletionRequest)
    prompts_for_req = server_utils.get_prompts_for_request(req, LLM)
    all_prompts.extend(prompts_for_req)
    request_info_map.append((req_id, req, is_chat, len(prompts_for_req)))

  broadcast_payload = {"prompts": all_prompts, "params": params}
  payload_bytes = json.dumps(broadcast_payload).encode("utf-8")
  payload_len = len(payload_bytes)

  if payload_len > MAX_REQUEST_SIZE:
    logger.error("Batched request is too large (%d bytes > %d)", payload_len, MAX_REQUEST_SIZE)
    for req_id, _, _, _ in request_info_map:
      with response_lock:
        response_dict[req_id] = {"error": "Batched request payload is too large."}
    return 0, b"", []  # Signal other ranks to skip

  return payload_len, payload_bytes, request_info_map


def _process_results(completions, request_info_map, payload):
  """Processes completions and sends responses back to the waiting threads."""
  logger.info("Batched generation finished. Processing %d completions.", len(completions))
  completion_idx = 0
  for req_id, req, is_chat, num_prompts in request_info_map:
    completions_for_req = completions[completion_idx : completion_idx + num_prompts]
    prompts_for_req = payload["prompts"][completion_idx : completion_idx + num_prompts]
    completion_idx += num_prompts

    response = _create_response(req, completions_for_req, prompts_for_req, is_chat, LLM)
    with response_lock:
      response_dict[req_id] = response


def main_loop():
  """The main processing loop with dynamic batching for all JAX processes."""
  while True:
    payload_len, payload_bytes, request_info_map = 0, b"", []
    if jax.process_index() == 0:
      batched_items = _collect_batched_requests()
      if batched_items:
        payload_len, payload_bytes, request_info_map = _prepare_batch_for_broadcast(batched_items)

    # Broadcast the payload to all ranks
    data_to_broadcast = (
      jnp.array([payload_len], dtype=jnp.int32),
      jnp.pad(jnp.frombuffer(payload_bytes, dtype=jnp.uint8), (0, MAX_REQUEST_SIZE - payload_len)),
    )
    received_len_array, received_data_array = multihost_utils.broadcast_one_to_all(data_to_broadcast)

    received_len = int(received_len_array[0])
    if received_len == 0:
      time.sleep(0.01)
      continue

    broadcasted_data = received_data_array[:received_len].tobytes().decode("utf-8")
    payload = json.loads(broadcasted_data)

    try:
      if jax.process_index() == 0:
        logger.info(
          "Starting batched generation for %d prompts with params: %s", len(payload["prompts"]), payload["params"]
        )

      completions = LLM.generate_batch(prompts=payload["prompts"], **payload["params"])

      if jax.process_index() == 0:
        _process_results(completions, request_info_map, payload)

    except (ValueError, RuntimeError) as e:
      logger.error("Inference failed for batch: %s", e, exc_info=True)
      if jax.process_index() == 0:
        for req_id, _, _, _ in request_info_map:
          with response_lock:
            response_dict[req_id] = {"error": f"Inference failed: {e}"}


def main():
  server_thread = None
  server = None

  # The coordinator process (rank 0) runs the FastAPI server in a separate thread.
  if jax.process_index() == 0:
    # Define a Uvicorn-compatible logging config.
    LOGGING_CONFIG = {
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
        "default": {
          "()": "uvicorn.logging.DefaultFormatter",
          "fmt": f"%(levelprefix)s RANK {rank}: %(message)s",
          "use_colors": None,
        },
        "access": {
          "()": "uvicorn.logging.AccessFormatter",
          "fmt": f'%(levelprefix)s RANK {rank}: %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
      },
      "handlers": {
        "default": {
          "formatter": "default",
          "class": "logging.StreamHandler",
          "stream": "ext://sys.stderr",
        },
        "access": {
          "formatter": "access",
          "class": "logging.StreamHandler",
          "stream": "ext://sys.stdout",
        },
      },
      "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
      },
    }

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_config=LOGGING_CONFIG)
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run)

    logger.info("Starting Uvicorn server in a background thread on coordinator process %d...", jax.process_index())
    server_thread.start()

  try:
    # All processes (coordinator and workers) enter the main processing loop.
    logger.info("Process %d is entering the main processing loop.", jax.process_index())
    main_loop()
  except KeyboardInterrupt:
    logger.info("Process %d received KeyboardInterrupt. Shutting down.", jax.process_index())
  finally:
    if jax.process_index() == 0 and server is not None and server_thread is not None:
      logger.info("Stopping Uvicorn server...")
      server.should_exit = True
      server_thread.join()
      logger.info("Uvicorn server stopped.")

  logger.info("Process %d has exited.", jax.process_index())


if __name__ == "__main__":
  main()
