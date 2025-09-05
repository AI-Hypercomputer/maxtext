import os
import sys
import time
import uuid
import json
import signal
import asyncio
import threading
import queue
import logging
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils


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
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
)

# ----------------------------
# Init
# ----------------------------

# JAX distributed initialization must happen before any other JAX calls.
# We suppress the normal logger until after JAX is initialized.
logging.basicConfig(level=logging.WARNING)
print("Initializing MaxTextGenerator and JAX distributed system...")
llm = MaxTextGenerator(sys.argv)
rank = jax.process_index()

# Now that JAX is initialized, we can get our rank-specific logger.
# The actual handler/formatter configuration will be done by Uvicorn.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Ensure our logger passes INFO messages.
logger.info("MaxTextGenerator initialization complete.")

harmony_enc = None
try:
    harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    logger.info("Harmony encoding for gpt-oss loaded successfully.")
except ImportError:
    logger.warning("openai_harmony not installed. GPT-OSS Harmony format will not be available.")
except Exception as e:
    logger.error(f"Failed to load Harmony encoding: {e}")


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


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Handles completion requests with dynamic batching."""
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
        # Yield control to the event loop.
        await asyncio.sleep(0.05)

    raise HTTPException(status_code=504, detail="Request timed out.")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Handles chat completion requests with dynamic batching."""
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
        await asyncio.sleep(0.05)

    raise HTTPException(status_code=504, detail="Request timed out.")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "MaxText API server is running."}

def run_server():
    """Runs the Uvicorn server in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Define a maximum size for the request payload to be broadcasted.
# This avoids broadcasting variable-sized arrays, which can be complex.
MAX_REQUEST_SIZE = 65536 * 10

def _create_response(request, completions, prompts, is_chat, llm, formatted_prompt=None):
    """Creates either a CompletionResponse or ChatCompletionResponse."""
    if is_chat:
        result = completions[0]
        text_out = result.text
        if "gpt-oss" in request.model and harmony_enc:
            try:
                parsed_messages = harmony_enc.parse_messages_from_completion_tokens(result.tokens, role=Role.ASSISTANT)
                user_visible = "".join(
                    m.content for m in parsed_messages 
                    if m.role == Role.ASSISTANT and m.channel == "final"
                )
                if user_visible:
                    text_out = user_visible
                else:
                    logger.warning("Harmony parsing for gpt-oss did not yield content in the 'final' channel. Falling back to raw text.")
            except Exception as e:
                logger.error(f"Harmony parsing failed for gpt-oss: {e}. Falling back to raw text.")
        
        lp_payload = server_utils.to_openai_logprobs(
            getattr(result, "logprobs", None), llm, want_top=(request.top_logprobs or 0) > 0 if isinstance(request, ChatCompletionRequest) else (request.logprobs or 0) > 0
        )
        text_out, lp_payload, finish_reason = server_utils.apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
        if finish_reason is None:
            finish_reason = result.finish_reason

        usage = Usage(
            prompt_tokens=result.prompt_token_count,
            completion_tokens=result.completion_token_count,
            total_tokens=result.prompt_token_count + result.completion_token_count,
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
    else:
        choices: list[CompletionChoice] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0

        for idx, prompt in enumerate(prompts):
            item = completions[idx]
            text_out = item.text
            lp_payload = server_utils.to_openai_logprobs(getattr(item, "logprobs", None), llm, want_top=(request.logprobs or 0) > 0)
            finish_reason = getattr(item, "finish_reason", "stop")
            text_out, lp_payload, stop_reason = server_utils.apply_stops_to_text_and_logprobs(text_out, lp_payload, request.stop)
            if stop_reason is not None:
                finish_reason = stop_reason

            prompt_tokens_total += item.prompt_token_count
            completion_tokens_total += item.completion_token_count

            choices.append(CompletionChoice(
                text=text_out,
                index=idx,
                logprobs=lp_payload,
                finish_reason=finish_reason,
            ))

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

def main_loop():
    """The main processing loop with dynamic batching for all JAX processes."""
    while True:
        # Step 1: Collect a batch of requests (rank 0 only)
        batched_items_rank_0 = []
        if jax.process_index() == 0:
            start_time = time.time()
            while len(batched_items_rank_0) < llm.batch_size and (time.time() - start_time) < BATCH_TIMEOUT_S:
                try:
                    item = request_queue.get(timeout=0.01)
                    batched_items_rank_0.append(item)
                except queue.Empty:
                    if batched_items_rank_0:
                        break  # Process what we have if timeout is reached

        # Step 2: Prepare batch for broadcast (rank 0 only)
        payload_bytes = b''
        payload_len = 0
        request_info_map = []  # To map completions back to requests
        if jax.process_index() == 0 and batched_items_rank_0:
            first_request_id, first_request = batched_items_rank_0[0]

            logprobs_param = None
            if isinstance(first_request, ChatCompletionRequest):
                if first_request.logprobs:
                    # If logprobs are enabled, we need to pass an integer to the generator.
                    # We'll use top_logprobs if provided, otherwise default to 1 to just get the sampled token's logprob.
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
            for req_id, req in batched_items_rank_0:
                is_chat = isinstance(req, ChatCompletionRequest)
                if is_chat:
                    messages = [m.model_dump() for m in req.messages]
                    formatted_prompt = llm.tokenizer.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts_for_req = [formatted_prompt]
                else:
                    prompts_for_req = server_utils.normalize_prompts(req.prompt, llm)

                all_prompts.extend(prompts_for_req)
                request_info_map.append((req_id, req, is_chat, len(prompts_for_req)))

            broadcast_payload = {"prompts": all_prompts, "params": params}
            payload_bytes = json.dumps(broadcast_payload).encode("utf-8")
            payload_len = len(payload_bytes)

            if payload_len > MAX_REQUEST_SIZE:
                logger.error(f"Batched request is too large ({payload_len} bytes > {MAX_REQUEST_SIZE})")
                for req_id, _, _, _ in request_info_map:
                    with response_lock:
                        response_dict[req_id] = {"error": "Batched request payload is too large."}
                payload_len = 0 # Signal other ranks to skip
            
        # Step 3: Broadcast the payload to all ranks
        data_to_broadcast = (
            jnp.array([payload_len], dtype=jnp.int32),
            jnp.pad(jnp.frombuffer(payload_bytes, dtype=jnp.uint8), (0, MAX_REQUEST_SIZE - payload_len)),
        )
        received_len_array, received_data_array = multihost_utils.broadcast_one_to_all(data_to_broadcast)

        # Step 4: Process broadcasted data and run generation (all ranks)
        received_len = int(received_len_array[0])
        if received_len == 0:
            time.sleep(0.01)  # sleep briefly to prevent a tight spin loop
            continue

        broadcasted_data = received_data_array[:received_len].tobytes().decode("utf-8")
        payload = json.loads(broadcasted_data)

        try:
            if jax.process_index() == 0:
                logger.info(f"Starting batched generation for {len(payload['prompts'])} prompts with params: {payload['params']}")

            completions = llm.generate_batch(prompts=payload["prompts"], **payload["params"])

            # Step 5: Process results and send responses (rank 0 only)
            if jax.process_index() == 0:
                logger.info(f"Batched generation finished. Processing {len(completions)} completions.")
                completion_idx = 0
                for req_id, req, is_chat, num_prompts in request_info_map:
                    completions_for_req = completions[completion_idx : completion_idx + num_prompts]
                    prompts_for_req = payload["prompts"][completion_idx : completion_idx + num_prompts]
                    completion_idx += num_prompts

                    formatted_prompt = prompts_for_req[0] if is_chat else None
                    response = _create_response(req, completions_for_req, prompts_for_req, is_chat, llm, formatted_prompt)

                    with response_lock:
                        response_dict[req_id] = response

        except Exception as e:
            logger.error(f"Inference failed for batch: {e}", exc_info=True)
            if jax.process_index() == 0:
                for req_id, _, _, _ in request_info_map:
                    with response_lock:
                        response_dict[req_id] = {"error": f"Inference failed: {e}"}


if __name__ == "__main__":
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

        logger.info(f"Starting Uvicorn server in a background thread on coordinator process {jax.process_index()}...")
        server_thread.start()

    try:
        # All processes (coordinator and workers) enter the main processing loop.
        logger.info(f"Process {jax.process_index()} is entering the main processing loop.")
        main_loop()
    except KeyboardInterrupt:
        logger.info(f"Process {jax.process_index()} received KeyboardInterrupt. Shutting down.")
    finally:
        if jax.process_index() == 0 and server is not None and server_thread is not None:
            logger.info("Stopping Uvicorn server...")
            server.should_exit = True
            server_thread.join()
            logger.info("Uvicorn server stopped.")

    logger.info(f"Process {jax.process_index()} has exited.")
