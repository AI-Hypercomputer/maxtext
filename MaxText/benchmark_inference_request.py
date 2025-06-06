# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark inference request TTFT, TPOT, request throughput."""

import dataclasses
import logging
import os
import time
from typing import Any, Sequence

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from jetstream.engine import chunked_prefill
from jetstream.engine import engine_api

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig
from MaxText.inference import piggybacking_decode

_REQUEST_NUM = flags.DEFINE_integer("request_num", 1, "Number of requests to send.")

logger = logging.getLogger(__name__)
if os.environ.get("MAXTEXT_BENCHMARK_INFERENCE_REQUEST_DEBUG_LOG") == "1":
  logger.setLevel(logging.DEBUG)


@dataclasses.dataclass
class _UsedConfig:
  """A class to hold the used config."""

  original_config: pyconfig.HyperParameters
  prompt: str
  prefill_length: int
  use_multimodal: bool
  model_name: str
  image_path: str
  quantization: str
  per_device_batch_size: int
  max_prefill_predict_length: int
  max_target_length: int
  autoregressive_decode_assert: str
  prefill_chunk_size: int
  use_chunked_prefill: bool


def _parse_config(argv: Sequence[str]) -> _UsedConfig:
  """Parses the config from the command line arguments."""
  config = pyconfig.initialize(argv)
  used_config = _UsedConfig(
      original_config=config,
      prompt=config.prompt,
      prefill_length=config.max_prefill_predict_length,
      use_multimodal=config.use_multimodal,
      model_name=config.model_name,
      image_path=config.image_path,
      quantization=config.quantization,
      per_device_batch_size=config.per_device_batch_size,
      max_prefill_predict_length=config.max_prefill_predict_length,
      max_target_length=config.max_target_length,
      autoregressive_decode_assert=config.autoregressive_decode_assert,
      prefill_chunk_size=config.prefill_chunk_size,
      use_chunked_prefill=config.use_chunked_prefill,
  )
  return used_config


def _do_chunked_prefill(
    engine: maxengine.MaxEngine,
    params: maxengine.Params,
    tokenizer: engine_api.Tokenizer,
    tokens: jax.Array | np.ndarray,
    true_length: int,
    existing_prefix: engine_api.ExistingPrefix | None = None,
) -> tuple[engine_api.Prefix, engine_api.ResultTokens]:
  """Do chunked prefill.

  Args:
    engine: The MaxEngine instance to use for processing.
    params: The model parameters.
    tokenizer: The tokenizer to use for chunking.
    tokens: The full input sequence of token IDs.
    true_length: The true length of the input tokens (without padding).
    existing_prefix: An optional existing prefix to prepend to the input
      sequence.

  Returns:
    A tuple containing the final prefill result and the first token of the
    last chunk.
  """
  if not engine.use_chunked_prefill:
    raise ValueError("Chunked prefill is not enabled in the engine.")

  chunk_size = engine.prefill_chunk_size

  # Generate the list of chunked tokens
  chunked_tokens_list = chunked_prefill.gen_chunked_padded_tokens(
      tokens[:true_length],  # Use only the true length portion for chunking
      chunk_size,
      tokenizer,
      existing_prefix_tokens=existing_prefix.common_prefix_tokens if existing_prefix else None,
      jax_padding=True,  # Assuming jax_padding is used in MaxEngine
  )

  if not chunked_tokens_list:
    raise ValueError("No chunked tokens provided.")

  return chunked_prefill.do_chunked_prefill(
      prefill_engine=engine,
      prefill_params=params,
      chunked_tokens_list=chunked_tokens_list,
      existing_prefix=existing_prefix,
  )


def _benchmark(
    config: _UsedConfig,
    engine: maxengine.MaxEngine,
    params: maxengine.Params,
    tokenizer: engine_api.Tokenizer,
    tokens: Any,
    target_length: int,
) -> tuple[float, float, float]:
  """Benchmarks the performance of a single request.

  The TTFT is assuming no prefill request is blocked by the generating.
  TTFT is basically prefill time.
  TPOT is the request latency / num of output tokens.

  Args:
    engine: The MaxEngine instance to benchmark.
    params: The model parameters.
    tokenizer: The tokenizer used for pre-processing and potentially for chunking.
    tokens: The input tokens, pre-tokenized.
    target_length: The target length of the output tokens.

  Returns:
    A tuple containing the TTFT (Time To First Token) and TPOT (Tokens Per
    Output Token), requests throughput.
  """
  # Decode batch size. Decode when slots full or run out of requests.
  decode_slots_num = engine.max_concurrent_decodes
  # Always prefill to the max prefill length for full prefill.
  true_length = len(tokens)
  # Always decode to the max target length.
  output_length = target_length - true_length

  logger.debug("Initializing decode state for benchmark.")
  decode_state = engine.init_decode_state()

  ttft_sum = 0
  tpot_sum = 0
  benchmark_start = time.perf_counter()
  logger.debug("Starting benchmark loop for %s requests.", _REQUEST_NUM.value)
  for remaining_requests in range(_REQUEST_NUM.value, 0, -decode_slots_num):
    request_start = time.perf_counter()
    current_requests = min(remaining_requests, decode_slots_num)
    prefill_result_list = []
    # Prefill
    logger.debug("Starting prefill phase for batch of %s requests.", current_requests)
    for _ in range(current_requests):
      prefill_start = time.perf_counter()
      logger.debug("Prefilling a single request in the current batch.")
      if config.use_chunked_prefill:
        prefill_result, _ = _do_chunked_prefill(
            engine=engine,
            params=params,
            tokenizer=tokenizer,
            tokens=tokens,
            true_length=true_length,
            existing_prefix=None,
        )
      else:
        prefill_result, _ = engine.prefill(
            params=params,
            padded_tokens=tokens,
            true_length=true_length,
        )

      jax.block_until_ready(prefill_result)
      prefill_end = time.perf_counter()
      # Assume the decode token time is not significant.
      ttft_sum += prefill_end - prefill_start

      prefill_result_list.append(prefill_result)

    logger.debug("Prefill phase complete for the batch.")

    # Insert
    logger.debug("Starting insert phase for the batch.")
    for i in range(current_requests):
      decode_state = engine.insert(prefill_result_list[i], decode_state, slot=i)
    logger.debug("Insert phase complete for the batch.")

    # Generate
    logger.debug("Starting generate phase for the batch (output_length: %s).", output_length)
    for i in range(output_length):
      decode_state, _ = engine.generate(params, decode_state)
      if (i + 1) % 32 == 0:
        logger.debug("Generate %d tokens,", i + 1)
    logger.debug("Generate phase complete for the batch.")

    request_end = time.perf_counter()
    tpot_sum += ((request_end - request_start) / output_length) * current_requests

  benchmark_end = time.perf_counter()
  logger.debug("Benchmark loop complete")
  ttft = ttft_sum / _REQUEST_NUM.value
  tpot = tpot_sum / _REQUEST_NUM.value
  request_throughput = _REQUEST_NUM.value / (benchmark_end - benchmark_start)
  return ttft, tpot, request_throughput


def _benchmark_piggybacking_decode(
    config: _UsedConfig,
    engine: maxengine.MaxEngine,
    params: maxengine.Params,
    tokenizer: engine_api.Tokenizer,
    tokens: Any,
    target_length: int,
) -> tuple[float, float, float]:
  """Benchmarks the performance of a single request.

  The TTFT is assuming no prefill request is blocked by the generating.
  TTFT is basically prefill time.
  TPOT is the request latency / num of output tokens.

  Args:
    engine: The MaxEngine instance to benchmark.
    params: The model parameters.
    tokenizer: The tokenizer used for pre-processing and potentially for chunking.
    tokens: The input tokens, pre-tokenized.
    target_length: The target length of the output tokens.

  Returns:
    A tuple containing the TTFT (Time To First Token) and TPOT (Tokens Per
    Output Token), requests throughput.
  """
  # Decode batch size. Decode when slots full or run out of requests.
  decode_slots_num = engine.max_concurrent_decodes
  # Always prefill to the max prefill length for full prefill.
  true_length = len(tokens)
  # Always decode to the max target length.
  output_length = target_length - true_length

  logger.debug("Initializing decode state for benchmark.")
  decode_state = engine.init_decode_state()

  ttft_sum = 0
  tpot_sum = 0
  benchmark_start = time.perf_counter()
  logger.debug("Starting benchmark loop for %s requests.", _REQUEST_NUM.value)
  for remaining_requests in range(_REQUEST_NUM.value, 0, -decode_slots_num):
    request_start = time.perf_counter()
    current_requests = min(remaining_requests, decode_slots_num)
    # Prefill
    logger.debug("Starting prefill phase for batch of %s requests.", current_requests)
    for _ in range(current_requests):
      prefill_start = time.perf_counter()
      logger.debug("Prefilling a single request in the current batch.")
      if config.use_chunked_prefill:
        prefill_result, _ = _do_chunked_prefill(
            engine=engine,
            params=params,
            tokenizer=tokenizer,
            tokens=tokens,
            true_length=true_length,
            existing_prefix=None,
        )
      else:
        # Assume generated tokens are in the padding tokens.
        # Does not need additional prefill.
        piggybacking_decode_params = piggybacking_decode.PiggyBackingDecodeParams(
          prefill_slot = jnp.array([0]),
          generate_slots = jnp.arange(1, decode_slots_num)
        )
        decode_state, _ = engine.prefill_generate(
            params=params,
            padded_tokens=tokens,
            true_length=true_length,
            decode_state=decode_state,
            piggybacking_decode_params=piggybacking_decode_params,
        )

      jax.block_until_ready(decode_state)
      prefill_end = time.perf_counter()
      # Assume the decode token time is not significant.
      ttft_sum += prefill_end - prefill_start

    logger.debug("Prefill phase complete for the batch.")

    # Assume no need insert. Use decode state directly.

    # Assume the generate in piggybacking can happen before the request prefill.
    # Decrease the tokens need to generated by prefill time.
    # Generate
    logger.debug("Starting generate phase for the batch (output_length: %s).", output_length)
    for i in range(max(0, output_length - current_requests)):
      decode_state, _ = engine.generate(params, decode_state)
      if (i + 1) % 32 == 0:
        logger.debug("Generate %d tokens,", i + 1)
    logger.debug("Generate phase complete for the batch.")

    request_end = time.perf_counter()
    tpot_sum += ((request_end - request_start) / output_length) * current_requests

  benchmark_end = time.perf_counter()
  logger.debug("Benchmark loop complete")
  ttft = ttft_sum / _REQUEST_NUM.value
  tpot = tpot_sum / _REQUEST_NUM.value
  request_throughput = _REQUEST_NUM.value / (benchmark_end - benchmark_start)
  return ttft, tpot, request_throughput


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  print(f"request_num: {_REQUEST_NUM.value}")

  used_config = _parse_config(argv)
  max_utils.print_system_information()
  if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Debug logging is active for benchmark_inference_request.")

  engine = maxengine.MaxEngine(used_config.original_config)

  text = used_config.prompt

  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)

  # Always padding to the max prefill length for full prefill.
  tokens, true_length = tokenizer_model.encode(
      text,
      is_bos=True,
      prefill_lengths=[used_config.max_prefill_predict_length],
  )

  assert true_length <= used_config.max_prefill_predict_length, (
      f"Input token length {true_length} is longer than" f" {used_config.max_prefill_predict_length=}"
  )

  params = engine.load_params()

  # Warm up
  print("Start warmup")
  _benchmark(used_config, engine, params, tokenizer_model, tokens, used_config.max_target_length)
  _benchmark_piggybacking_decode(used_config, engine, params, tokenizer_model, tokens, used_config.max_target_length)

  # Benchmark
  print("Start benchmark")
  ttft, tpot, requests_throughput = _benchmark(
      used_config, engine, params, tokenizer_model, tokens, used_config.max_target_length
  )
  print("Normal:")
  print(f"  TTFT: {ttft*1000:.3f} ms, TPOT: {tpot*1000:.3f} ms, Requests/s: {requests_throughput:.3f}")

  ttft, tpot, requests_throughput = _benchmark_piggybacking_decode(
      used_config, engine, params, tokenizer_model, tokens, used_config.max_target_length
  )
  print("Piggybacking decode:")
  print(f"  TTFT: {ttft*1000:.3f} ms, TPOT: {tpot*1000:.3f} ms, Requests/s: {requests_throughput:.3f}")

if __name__ == "__main__":
  app.run(main)
