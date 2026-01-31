# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines the MaxTextGenerator class, which serves as the core
inference engine for the API server. It encapsulates the MaxText model,
tokenizer, and the JAX-based generation logic for both prefill and
autoregressive decoding steps. It handles batching, sampling, and the
low-level details of running inference on TPUs.
"""

from io import StringIO
from typing import Sequence, Optional, List, Union
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp

import numpy as np

from dataclasses import dataclass, field

from MaxText import maxengine, pyconfig
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import max_logging, max_utils

# Set TF log level to avoid verbose startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class LogProbs:
  """
  A dataclass to store detailed log probability information for a sequence of tokens.

  Attributes:
      tokens: A list of token IDs.
      token_logprobs: A list of log probabilities corresponding to each token.
      top_logprobs: An optional list of dictionaries, where each dictionary maps
                    surrounding tokens to their log probabilities at each position.
                    Currently unused and set to None.
      text_offset: A list of character offsets for each token in the generated text.
  """

  tokens: List[int]
  token_logprobs: List[float]
  top_logprobs: Optional[List[None]] = None
  text_offset: List[int] = field(default_factory=list)


@dataclass
class Completion:
  """
  Represents a single completed generation from the model.

  Attributes:
      index: The index of this completion in the batch.
      text: The generated text.
      tokens: The list of token IDs that make up the generated text.
      logprobs: An optional `LogProbs` object containing detailed log probability info.
      finish_reason: The reason the generation finished (e.g., 'stop', 'length').
      prompt_token_count: The number of tokens in the input prompt.
      completion_token_count: The number of tokens in the generated completion.
  """

  index: int
  text: str
  tokens: List[int]
  logprobs: Optional[LogProbs]
  finish_reason: str = "stop"
  prompt_token_count: int = 0
  completion_token_count: int = 0


@dataclass
class GenerationStream:
  """Holds the state for a single generation stream within a batch."""

  # Input state
  tokens: np.ndarray
  true_length: int
  image: Optional[np.ndarray]

  # Output accumulators
  generated_ids: List[int] = field(default_factory=list)
  generated_logprobs: List[float] = field(default_factory=list)

  # For echo=True
  prompt_ids: List[int] = field(default_factory=list)
  prompt_logprobs: List[float] = field(default_factory=list)

  # Status
  finished: bool = False
  finish_reason: str = "length"


class MaxTextGenerator:
  """A reusable class for parallel text generation using MaxText."""

  def __init__(self, argv: Sequence[str]):
    """
    Initializes the MaxText model, tokenizer, and engine.

    Args:
        argv: Command-line arguments for MaxText configuration.
    """
    start_time = time.time()

    argv_list = list(argv)

    # Check for HF_TOKEN env var and inject as a pyconfig argument if not already present.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and not any("hf_access_token" in arg for arg in argv_list):
      max_logging.log("Found HF_TOKEN environment variable. Adding to config.")
      argv_list.append(f"hf_access_token={hf_token}")

    # CRITICAL: Initialize the distributed system and config FIRST.
    # This call to pyconfig.initialize() contains jax.distributed.initialize()
    # and must happen before any other JAX calls.
    self.config = pyconfig.initialize(argv_list)
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    # Now that JAX is initialized, we can set up logging and use JAX functions.
    self.rank = jax.process_index()
    self.logger = logging.getLogger(__name__)
    self.logger.info("Initializing MaxTextGenerator with argv: %s", argv_list)

    self._validate_config(self.config)
    self.logger.info("System information:")
    # Temporarily redirect stdout to capture print output for the log
    old_stdout = sys.stdout
    sys.stdout = captured_stdout = StringIO()
    max_utils.print_system_information()
    sys.stdout = old_stdout
    self.logger.info(captured_stdout.getvalue())

    self.engine = maxengine.MaxEngine(self.config)
    self.rng = jax.random.PRNGKey(1234)

    self.logger.info("Loading model parameters...")
    self.rng, rng_load_params = jax.random.split(self.rng)
    self.params = self.engine.load_params(rng=rng_load_params)
    self.logger.info("Model parameters loaded.")

    self.metadata = self.engine.get_tokenizer()
    self.tokenizer = self.engine.build_tokenizer(self.metadata)
    eos_id = self.tokenizer.eos_id
    if not isinstance(eos_id, list):
      eos_id = [eos_id]
    self.eos_ids = eos_id
    try:
      self.has_chat_template = getattr(self.tokenizer.tokenizer, "chat_template", False)
    except AttributeError:
      self.has_chat_template = False

    self.logger.info("Chat Template available: %s", self.has_chat_template)

    self.batch_size = int(self.config.per_device_batch_size * jax.device_count())

    self.rng, rng_init_decode = jax.random.split(self.rng)
    self.decode_state = self.engine.init_decode_state(rng=rng_init_decode)
    self._jitted_reset_state = jax.jit(lambda state: jax.tree_util.tree_map(jnp.zeros_like, state))

    end_time = time.time()
    self.logger.info(
        f"Initialization complete in {end_time - start_time:.2f} seconds. Max batch size: %d", self.batch_size
    )

  def generate_batch(
      self,
      prompts: List[str],
      image_paths: Optional[List[Optional[str]]] = None,
      max_tokens: int = None,
      logprobs: int = None,
      echo: bool = False,
      stop: Optional[Union[str, List[str]]] = None,
      temperature: Optional[float] = None,
      seed: Optional[int] = None,
      top_k: Optional[int] = None,
      top_p: Optional[float] = None,
  ) -> List[Completion]:
    """
    Generates text for a batch of prompts, handling chunking automatically.

    Args:
        prompts: A list of prompt strings.
        image_paths: An optional list of image paths, one for each prompt.
        max_tokens: The maximum number of tokens to generate.
        logprobs: The number of top log probabilities to return for each token.
        echo: Whether to include the prompt in the generated text.
        stop: An optional list of stop sequences.
        temperature: An optional temperature for sampling.
        seed: An optional seed for deterministic sampling.
        top_k: An optional integer for top-k sampling.
        top_p: An optional float for nucleus sampling.

    Returns:
        A list of generated Completion, corresponding to the input prompts.
    """
    if image_paths is None:
      image_paths = [None] * len(prompts)
    if len(prompts) != len(image_paths):
      raise ValueError("The number of prompts must equal the number of image paths.")

    all_results = []
    num_prompts = len(prompts)
    for i in range(0, num_prompts, self.batch_size):
      prompt_chunk = prompts[i : i + self.batch_size]
      image_chunk = image_paths[i : i + self.batch_size]

      # chunk_count = (i // self.batch_size) + 1
      # total_chunks = (num_prompts + self.batch_size - 1) // self.batch_size

      chunk_results = self._process_chunk(
          prompt_chunk, image_chunk, max_tokens, logprobs, echo, stop, temperature, seed, top_k, top_p
      )
      all_results.extend(chunk_results)

    return all_results

  def _process_chunk(
      self,
      prompts: List[str],
      image_paths: List[Optional[str]],
      max_tokens: int,
      logprobs: int = None,
      echo: bool = False,
      stop: Optional[Union[str, List[str]]] = None,
      temperature: Optional[float] = None,
      seed: Optional[int] = None,
      top_k: Optional[int] = None,
      top_p: Optional[float] = None,
  ) -> List[Completion]:
    """Orchestrates the generation process for a single chunk of prompts."""
    start_time = time.time()

    # for prompt in prompts:
    # self.logger.debug(f"Processing prompt: {prompt}")

    initialize_start_time = time.time()
    # Reset the state to handle the new batch while reusing memory.
    self.decode_state = self._jitted_reset_state(self.decode_state)
    streams, rng = self._initialize_streams_and_state(prompts, image_paths, seed)
    initialize_end_time = time.time()
    self.logger.info(
        "Initialization complete in %.2f seconds. Max batch size: %d",
        initialize_end_time - initialize_start_time,
        self.batch_size,
    )

    if max_tokens is not None and max_tokens <= 0:
      self.logger.warning("max_tokens <= 0, returning empty completions.")
      return [Completion(index=i, text="", tokens=[], logprobs=None) for i in range(len(streams))]

    prefill_start_time = time.time()
    self.decode_state, rng = self._run_prefill_step(
        streams, self.decode_state, rng, logprobs, echo, temperature, top_k, top_p
    )
    prefill_end_time = time.time()
    self.logger.info("Prefill step took %.2fs.", prefill_end_time - prefill_start_time)

    generation_start_time = time.time()
    self.decode_state = self._run_generation_loop(
        streams, self.decode_state, rng, max_tokens, stop, temperature, top_k, top_p
    )
    generation_end_time = time.time()
    self.logger.info("Generation loop took %.2fs.", generation_end_time - generation_start_time)

    completions_start_time = time.time()
    completions = self._build_completions(streams, logprobs, echo)
    completions_end_time = time.time()
    self.logger.info("Completions loop took %.2fs.", completions_end_time - completions_start_time)

    end_time = time.time()
    self.logger.info("Processed %d prompts in %.2fs.", len(prompts), end_time - start_time)
    return completions

  def _initialize_streams_and_state(self, prompts, image_paths, seed):
    """Tokenizes inputs, sets up stream objects, and initializes the decode state."""
    prefill_length = getattr(self.config, "max_prefill_predict_length", 1024)
    streams = []
    for prompt, image_path in zip(prompts, image_paths):
      toks, tlen, imgs = self._preprocess_inputs(prompt, prefill_length, image_path)
      assert tlen <= prefill_length, f"Input token length {tlen} is > {prefill_length}"
      streams.append(GenerationStream(tokens=toks, true_length=tlen, image=imgs))

    if seed is not None:
      rng = jax.random.PRNGKey(seed)
    else:
      self.rng, rng = jax.random.split(self.rng)

    return streams, rng

  def _determine_sampling_algorithm(self, temperature, top_k, top_p):
    """Determines the sampling algorithm based on user-provided parameters."""
    if temperature == 0.0:
      return "greedy"
    if top_k is not None and top_p is not None:
      return "composite"
    if top_k is not None:
      return "topk"
    if top_p is not None:
      return "nucleus"
    if temperature is not None:
      return "weighted"
    # If no specific parameters are provided, return None to let the
    # engine use its default configured `decode_sampling_strategy`.
    return None

  def _run_prefill_step(self, streams, decode_state, rng, logprobs, echo, temperature, top_k, top_p):
    """Runs the prefill step for each stream and inserts results into the decode state."""
    sampling_algorithm = self._determine_sampling_algorithm(temperature, top_k, top_p)
    prefill_results_to_insert = {}

    for i, stream in enumerate(streams):
      rng, rng_prefill = jax.random.split(rng)
      want_prompt_logp = logprobs is not None and echo

      prefill_result, _ = self.engine.prefill(
          params=self.params,
          padded_tokens=stream.tokens,
          true_length=stream.true_length,
          images=stream.image,
          rng=rng_prefill,
          slot=i,
          return_prompt_logp=want_prompt_logp,
          temperature=temperature,
          algorithm=sampling_algorithm,
          topk=top_k,
          nucleus_topp=top_p,
      )
      prefill_results_to_insert[i] = prefill_result

      p_ids = list(map(int, np.array(stream.tokens[: stream.true_length], dtype=np.int32).tolist()))
      stream.prompt_ids.extend(p_ids)
      if prefill_result.get("prompt_logp") is not None:
        p_logp_arr = np.array(prefill_result["prompt_logp"])[0, : stream.true_length]
        stream.prompt_logprobs.extend([float(x) for x in p_logp_arr.tolist()])

      first_token_id = int(np.array(prefill_result["tokens"])[0, 0])
      stream.generated_ids.append(first_token_id)
      if prefill_result.get("token_logp") is not None:
        first_logp = float(np.array(prefill_result["token_logp"])[0, 0])
        stream.generated_logprobs.append(first_logp)

    for slot_idx, result in prefill_results_to_insert.items():
      decode_state = self.engine.insert(prefix=result, decode_state=decode_state, slot=slot_idx)

    return decode_state, rng

  def _run_generation_loop(self, streams, decode_state, rng, max_tokens, stop, temperature, top_k, top_p):
    """Runs the autoregressive generation loop."""
    target_length = getattr(self.config, "max_target_length", 2048)
    prefill_length = getattr(self.config, "max_prefill_predict_length", 1024)
    sampling_algorithm = self._determine_sampling_algorithm(temperature, top_k, top_p)

    stop_sequences = []
    max_stop_seq_len_tokens = 0
    if stop:
      # Ensure stop is a list of non-empty strings
      stop_sequences = [s for s in ([stop] if isinstance(stop, str) else stop) if s]
      if stop_sequences:
        # Calculate the max token length for any stop sequence to define a lookback window.
        for seq in stop_sequences:
          # Use the underlying tokenizer here to avoid potential errors with the wrapper
          # on single-token sequences, as this is a safe, internal calculation.
          token_ids = self.tokenizer.tokenizer.encode(seq, add_special_tokens=False)
          max_stop_seq_len_tokens = max(max_stop_seq_len_tokens, len(token_ids))

    total_steps = target_length - prefill_length
    if max_tokens is not None:
      total_steps = min(total_steps, max_tokens - 1)  # -1 for the token from prefill

    for step in range(total_steps):
      self.logger.debug("Generation step %d/%d", step + 1, total_steps)
      active_streams = [(j, s) for j, s in enumerate(streams) if not s.finished]
      if not active_streams:
        self.logger.info("All streams finished. Breaking generation loop.")
        break

      rng, rng_generate = jax.random.split(rng)
      decode_state, _ = self.engine.generate(
          self.params,
          decode_state,
          rng=rng_generate,
          temperature=temperature,
          algorithm=sampling_algorithm,
          topk=top_k,
          nucleus_topp=top_p,
      )

      state_tokens = np.array(decode_state["tokens"])
      state_logp_np = None
      if (logp := decode_state.get("token_logp")) is not None:
        state_logp_np = np.array(logp)

      for slot_idx, stream in active_streams:
        tok_id = int(state_tokens[slot_idx, 0])
        stream.generated_ids.append(tok_id)
        if state_logp_np is not None:
          stream.generated_logprobs.append(float(state_logp_np[slot_idx, 0]))

        # Check for finish conditions
        current_len = stream.true_length + 1 + step
        is_max_len = current_len >= target_length
        is_eos = tok_id in self.eos_ids
        stop_sequence_found = False

        if stop_sequences:
          # Define a lookback window for decoding that is slightly larger
          # than the longest stop sequence in tokens.
          lookback_window = max_stop_seq_len_tokens + 2
          start_index = max(0, len(stream.generated_ids) - lookback_window)
          trailing_ids = stream.generated_ids[start_index:]

          if trailing_ids:
            # Use the standard jetstream wrapper for decoding as requested.
            trailing_text = self.tokenizer.decode([int(tid) for tid in trailing_ids])
            stop_sequence_found = next(
                (
                    True
                    for stop_seq in stop_sequences
                    # Use 'in' for a more robust check.
                    if stop_seq in trailing_text
                ),
                False,
            )

        if is_max_len or is_eos or stop_sequence_found:
          stream.finished = True
          if is_eos or stop_sequence_found:
            stream.finish_reason = "stop"
          if getattr(self.config, "attention", "") == "paged":
            self.engine.release_pages(slot=slot_idx)

    return decode_state

  def _build_completions(self, streams, logprobs, echo):
    """Builds the final Completion objects from the generated stream states."""
    completions = []
    for i, stream in enumerate(streams):
      gen_ids_for_text = stream.generated_ids[:]
      gen_logps_for_text = stream.generated_logprobs[:]

      if gen_ids_for_text and gen_ids_for_text[-1] in self.eos_ids:
        gen_ids_for_text = gen_ids_for_text[:-1]
        if len(gen_logps_for_text) >= len(stream.generated_ids):
          gen_logps_for_text = gen_logps_for_text[:-1]

      tokens_for_text = stream.prompt_ids + gen_ids_for_text if echo else gen_ids_for_text
      logps_for_text = stream.prompt_logprobs + gen_logps_for_text if echo else gen_logps_for_text

      text = self.tokenizer.decode(tokens_for_text)
      offsets = self._token_offsets(tokens_for_text, 0)

      lp_payload = None
      if logprobs is not None:
        if len(tokens_for_text) != len(logps_for_text):
          self.logger.warning("[warn] Mismatched token/logprob lengths for stream %d. No logprobs returned.", i)
        else:
          lp_payload = LogProbs(
              tokens=tokens_for_text,
              token_logprobs=logps_for_text,
              top_logprobs=None,
              text_offset=offsets,
          )

      completions.append(
          Completion(
              index=i,
              text=text,
              tokens=tokens_for_text,
              logprobs=lp_payload,
              finish_reason=stream.finish_reason,
              prompt_token_count=len(stream.prompt_ids),
              completion_token_count=len(gen_ids_for_text),
          )
      )
    return completions

  def _preprocess_inputs(self, text, prefill_length, image_path):
    """Helper to preprocess a single text and optional image input."""
    processor_output = mm_utils.PreprocessorOutput()
    images = None
    if self.config.use_multimodal and image_path:
      text = mm_processor.reformat_prompt(
          prompt=self.config.prompt,
          image_placeholder=self.config.image_placeholder,
          model_name=self.config.model_name,
          num_images=1,
      )
      processor_output = mm_processor.preprocess_mm_data(self.config)
      prefill_length -= mm_processor.get_image_offsets(self.config.model_name, processor_output=processor_output)
      images = processor_output.pixel_values

    tokens, true_length = self.tokenizer.encode(text, is_bos=not self.has_chat_template, prefill_lengths=[prefill_length])
    if self.config.use_multimodal and image_path:
      tokens = mm_processor.prepare_text_for_image_fusion(
          tokens, model_name=self.config.model_name, processor_output=processor_output
      )
      true_length += mm_processor.get_image_offsets(self.config.model_name, processor_output=processor_output)

    return tokens, true_length, images

  def _validate_config(self, config):
    """Validates configuration."""
    assert (
        config.load_full_state_path == ""
    ), "Decode doesn't operate on full states! Convert to parameter checkpoint first. Using generate_param_only_checkpoint."
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"
    assert config.per_device_batch_size * jax.device_count() >= 1, "Total batch size must be at least 1."

  def _token_offsets(self, token_ids: List[int], start: int = 0) -> List[int]:
    """
    Compute char offsets by decoding cumulatively so context-dependent
    whitespace/normalization is handled correctly (SentencePiece/LLaMA quirk).
    """
    offsets: List[int] = []
    pos = start
    decoded_so_far = ""
    prefix_ids: List[int] = []
    for tid in token_ids:
      offsets.append(pos)
      prefix_ids.append(int(tid))
      new_text = self.tokenizer.decode(prefix_ids)
      piece_len = len(new_text) - len(decoded_so_far)
      # Guard for weird edge cases; shouldn't happen but better safe:
      if piece_len < 0:
        piece_len = len(self.tokenizer.decode([int(tid)]))
      pos += piece_len
      decoded_so_far = new_text
    return offsets


def main():
  def dump_completion(i, comp):
    """
    Dumps the content of a Completion object to the log for debugging.

    Args:
        i: The index of the completion.
        comp: The Completion object to dump.
    """
    max_logging.log(f"\n=== Completion {i} ===")
    max_logging.log(f"index: {comp.index}")
    max_logging.log(f"text: {repr(comp.text)}")

    if comp.logprobs is None:
      max_logging.log("logprobs: None")
      return

    lp = comp.logprobs
    # lengths should match: one logprob/offset per token
    if not len(lp.tokens) == len(lp.token_logprobs) == len(lp.text_offset):
      max_logging.log(
          f"[warn] mismatched lengths: tokens={len(lp.tokens)}, "
          f"logps={len(lp.token_logprobs)}, offsets={len(lp.text_offset)}"
      )

    max_logging.log("logprobs:")
    max_logging.log(f"  tokens (ids): {lp.tokens}")
    max_logging.log(f"  token_logprobs: {[round(x, 6) for x in lp.token_logprobs]}")
    max_logging.log(f"  text_offset: {lp.text_offset}")
    max_logging.log(f"  top_logprobs: {lp.top_logprobs}")

    max_logging.log("  tokens (decoded, id, logprob, offset):")
    for tid, logp, off in zip(lp.tokens, lp.token_logprobs, lp.text_offset):
      piece = llm.tokenizer.decode([int(tid)])
      max_logging.log(f"    {repr(piece):>12s}  id={int(tid):>6d}  logp={logp:>10.6f}  offset={off}")

  # When running standalone, basic logging is automatically configured.
  # For server use, the server configures logging.
  logging.basicConfig(level=logging.INFO)

  # Instantiate first to initialize JAX
  llm = MaxTextGenerator(sys.argv)

  prompts_to_run = [
      "The capital of France is ",
  ]

  max_tokens = 32
  echo = True
  want_logprobs = 5
  temperature = 0.6
  seed = 72
  top_p = 0.95
  top_k = 20

  max_logging.log(
      f"\n--- Starting Batch Generation for {len(prompts_to_run)} Prompts " f"(max_tokens={max_tokens}, echo={echo}) ---"
  )

  completions = llm.generate_batch(
      prompts=prompts_to_run,
      image_paths=None,
      max_tokens=max_tokens,
      logprobs=want_logprobs,
      echo=echo,
      seed=seed,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
  )

  for i, comp in enumerate(completions):
    dump_completion(i, comp)

  start = time.time()

  completions = llm.generate_batch(
      prompts=prompts_to_run,
      image_paths=None,
      max_tokens=max_tokens,
      logprobs=want_logprobs,
      echo=echo,
      seed=seed,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
  )

  max_logging.log("--- Batch Generation Complete ---")

  for i, comp in enumerate(completions):
    dump_completion(i, comp)

  end = time.time()
  max_logging.log(f"total time: {end - start}")


if __name__ == "__main__":
  main()
