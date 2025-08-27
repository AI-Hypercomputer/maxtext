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
Offline Inference Engine

Example usage:
    offline_engine = OfflineEngine(
        config=maxtext_config,
        params=None,
        enable_batch_prefill=True,
    )

    input_data = [
        jax.numpy.arange(80),
        jax.numpy.arange(90),
        jax.numpy.arange(100),
    ]

    results = offline_engine.batch_inference(input_data)

    for completion_output in results:
        text = offline_engine.tokenizer.decode(completion_output.token_ids)
        max_logging.log(f"Output: {text}")
"""

import os
import queue
import signal
import threading
import traceback
import functools
import dataclasses
from enum import Enum
from typing import Any, Callable
from collections.abc import Hashable
from collections import defaultdict
import time

import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.experimental import mesh_utils

from MaxText.maxengine import MaxEngine
from MaxText import max_utils
from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor
from MaxText import max_logging

from pathwaysutils.experimental import reshard as pathways_reshard

DecodeState = Any
Params = Any
MaxTextConfig = Any


@dataclasses.dataclass
class InputData:
  """Container for input data and metadata.

  Attributes:
      id: Unique identifier for this input
      tokens: JAX array containing the tokenized input
      true_length: Actual length of the input before padding
  """

  id: str
  tokens: jax.Array | np.ndarray
  true_length: int


@dataclasses.dataclass
class CompletionOutput:
  """Container for model generation output.

  Attributes:
      index: The index of the output in the request.
      token_ids: The token IDs of the prompt and generated output text.
      logprobs: The log probabilities of the prompt and generated output tokens.
      prompt_length: The number of prompt tokens.
  """

  index: int
  token_ids: np.ndarray
  logprobs: np.ndarray
  prompt_length: int


@dataclasses.dataclass
class TokenOutput:
  """Container for individual token generation result."""

  token: np.ndarray
  log_prob: np.ndarray


@dataclasses.dataclass
class DetokenizationTask:
  """Container for detokenization work to be done on background thread."""

  task_type: str  # "prefill" or "decode"
  # For prefill tasks
  result_tokens: Any = None
  log_prob: Any = None
  prompt_logp: Any = None
  prompt_ids: list = None
  slots: list = None
  # For decode tasks
  tokens_buffer: list = None
  logprob_buffer: list = None


class SafeThread(threading.Thread):
  """Thread class with exception handling to prevent silent failures."""

  def run(self):
    try:
      super().run()
    except Exception as _:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      # Kill the process if a thread encounters an error
      os.kill(os.getpid(), signal.SIGKILL)


class PrefillType(Enum):
  """Enumeration of supported prefill processing methods."""

  DEFAULT = "default"
  BATCH = "batch"


@dataclasses.dataclass
class PrefillResult:
  """Result from prefill processing operation."""

  result_tokens: "jetstream.engine_api.ResultTokens"
  slot: int
  prompt_logp: None | jax.Array


class PrefillHelper:
  """Abstraction layer for different prefill processing strategies.

  Provides a unified interface for both default (single-sequence) and batch
  (packed multi-sequence) prefill processing methods.
  """

  def __init__(
      self,
      prefill_type: PrefillType,
      engine: MaxEngine,
      prefill_lengths: list[int],
      batch_prefill_max_batch_size: int = 16,
      rng=None,
  ):
    """Initialize the PrefillHelper.

    Args:
        type: The type of prefill processor to use ("default" or "batch")
        engine: The MaxEngine instance to use for prefill operations
        prefill_lengths: list of prompt lengths to support
        batch_prefill_max_batch_size: Maximum number of prompts in one packed
            sequence for batch prefill
    """
    self._type = prefill_type
    self.engine = engine
    self.prefill_lengths = sorted(prefill_lengths)
    self.max_prefill_length = self.prefill_lengths[-1]
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    if prefill_type == PrefillType.DEFAULT:
      self._processor = PrefillProcessor(engine)
    elif prefill_type == PrefillType.BATCH:
      self._batch_processor = BatchedPrefillProcessor(
          engine=engine,
          max_batch_size=batch_prefill_max_batch_size,
          auto_layout_supported=False,
      )
      # Keep fallback processor for edge cases
      self._processor = PrefillProcessor(engine)
    else:
      raise ValueError(f"Invalid prefill type: {prefill_type}")

  @functools.partial(jax.jit, static_argnums=(0), donate_argnames=("decode_state",))
  def _jitted_single_prefill(
      self, params, tokens, slot, true_length, decode_state, rng
  ) -> tuple[jax.Array, jax.Array, DecodeState, jax.Array] | tuple[jax.Array, jax.Array, DecodeState]:
    """Prefill a single input."""
    # pylint: disable=protected-access
    first_token, decode_state = self._processor._process(
        params,
        tokens,
        slot,
        true_length,
        decode_state,
        rng,
        return_prompt_logp=True,
    )

    return (
        first_token,
        decode_state,
        decode_state["prompt_logp"],
    )

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: int,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      prefill_done: Callable,
  ) -> None:
    """Process an input through the appropriate prefill processor.

    Args:
        model_params: Model parameters for inference
        decode_state: Current decode state
        decode_slot: The decode slot index to use for this input
        input_id: Unique identifier for this input
        input_tokens_padded: Padded token array for the input
        input_true_length: Actual length of the input before padding
        prefill_done: Callback function called when prefill completes
    """
    padded_length = len(input_tokens_padded)
    # Use default processor if configured or if input is already at max length
    if self._type == PrefillType.DEFAULT or padded_length == self.max_prefill_length:
      first_token, decode_state, prompt_logp = self._jitted_single_prefill(
          model_params,
          input_tokens_padded,
          decode_slot,
          input_true_length,
          decode_state,
          self.rng,
      )
      prefill_done(
          [PrefillResult(first_token, decode_slot, prompt_logp)],
          [input_id],
          decode_state,
      )
    # Use batch processor for inputs that can benefit from prefill packing
    elif self._type == PrefillType.BATCH:
      self._batch_processor.process(
          model_params,
          decode_state,
          decode_slot,
          input_id,
          input_tokens_padded[:input_true_length],
          padded_length,
          self.max_prefill_length,
          prefill_done,
          return_prompt_logp=True,
      )

  def finalize(
      self,
      model_params: Params,
      decode_state: DecodeState,
      prefill_done: Callable,
  ) -> None:
    """Finalize prefill operations, flushing any pending inputs.

    Args:
        model_params: Model parameters for inference
        decode_state: Current decoder state
        prefill_done: Callback function called when prefill completes
    """
    if self._type == PrefillType.DEFAULT:
      # No finalization needed for default processor
      pass
    elif self._type == PrefillType.BATCH:
      # Flush any remaining inputs in the batch processor
      self._batch_processor.flush(model_params, decode_state, prefill_done, return_prompt_logp=True)


class InferenceWorker:
  """
  InferenceWorker runs continuous batching over
  a queue of inputs.

    Continuous batching workflow:
    1. Process inputs one at a time from queue
    2. Prefill input and insert into KV cache
    3. Continue prefilling until enough samples for batch decode
    4. Decode until at least one sequence completes
    5. Refill newly available decode slots with prefill
    6. Repeat until all sequences complete

    Prefill Packing:
        When enable_batch_prefill is True, the prefill processor
        will pack multiple inputs into a single sequence before
        doing the prefill.

        There are multiple buckets for packed sequences, where each bucket
        contains inputs with the same padded length. Only inputs with the same
        padded length can be packed together.

        It is important to sort the inputs by padded length so that the
        buckets fill up quickly.

        When a decode slot frees up, the prefill processor will add the
        sequence to a bucket. If the bucket becomes full, the packed sequence
        will be prefilled.

        E.g.
        Bucket for length 64: [...seq1, ...seq2, ...seq3, ...seq4]
        Bucket for length 128: [...seq1, ...seq2]
        Bucket for length 256: [...seq1]

  """

  def __init__(
      self,
      config: MaxTextConfig,
      params: Params | None,
      min_decode_steps: int,
      enable_batch_prefill: bool,
      devices: list[Any],
      tokenizer: Any,
      eos_ids: list[int],
      prefill_lengths: list[int],
      max_decode_length: int,
      batch_prefill_max_batch_size: int,
      is_pw_reshard: bool = True,
      rng: jax.random.PRNGKey = None,
      mesh: Mesh = None,
      debug: bool = False,
  ):
    """
    Args:
        config: MaxText configuration
        params: Model parameters, if None, the params will be loaded from the config
        min_decode_steps: Minimum number of decode steps to run at once
        enable_batch_prefill: Whether to enable batch prefill
        devices: JAX devices to use for this worker
        tokenizer: Tokenizer to use
        eos_ids: End-of-sequence token IDs
        prefill_lengths: list of supported prefill lengths
        max_decode_length: Maximum tokens to generate per sequence
        batch_prefill_max_batch_size: Maximum batch size for batch prefill
        run_as_a_thread: Whether to run in a separate thread
        rng: Random number generator key
        mesh: JAX mesh for distributed computation
        is_pw_reshard: Whether to use Pathways for resharding
    """
    # Configurations
    self.config = config
    self.params = params
    self.devices = devices
    self.is_pw_reshard = is_pw_reshard
    self.enable_batch_prefill = enable_batch_prefill
    self.prefill_type = PrefillType.BATCH if enable_batch_prefill else PrefillType.DEFAULT
    self.prefill_lengths = prefill_lengths
    self.max_prefill_length = self.prefill_lengths[-1]
    self.max_decode_length = max_decode_length
    self.eos_ids = eos_ids
    self.tokenizer = tokenizer
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.min_decode_steps = min_decode_steps
    self.mesh = mesh
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    self.debug = debug

    # Inference state (initialized later)
    self.running = False
    self.detokenization_queue = queue.Queue()
    self.empty_decode_slots = queue.Queue()
    self.slot_to_id: dict[int, int] = {}
    self.decode_state: DecodeState = None
    self.completion_tokens_by_id: dict[Hashable, list[TokenOutput]] = {}
    self.prompt_logprobs_by_id: dict[Hashable, list[np.ndarray]] = {}
    self.true_lengths: dict[Hashable, int] = {}
    self.completed_sequences: set = set()

    # Model components (initialized later)
    self.engine = None
    self.decode_batch_size = None
    self.prefill_helper = None
    self.generate_fn = None

    start_time = time.time()
    # Initialize MaxEngine(s)
    self.params, self.engine = self._init_engine(self.params)
    self.tokenizer = self._init_tokenizer()
    self.decode_batch_size = self.engine.max_concurrent_decodes
    # Initialize prefill helper
    self.prefill_helper = PrefillHelper(
        self.prefill_type,
        self.engine,
        self.prefill_lengths,
        self.batch_prefill_max_batch_size,
        rng=self.rng,
    )
    # Initialize decode state
    start_time_decode_state = time.time()
    self.generate_fn = self.engine.generate
    self.decode_state = self.engine.init_decode_state(self.rng)

    if self.debug:
      max_logging.log(f"time taken to initialize decode_state: {time.time() - start_time_decode_state} seconds")
    max_logging.log(f"Initialized Inference worker in {time.time() - start_time} seconds")

  def _init_engine(self, params):
    """Initialize the MaxEngine.

    Args:
        params: Model parameters

    Returns:
        tuple of (params, engine)
    """
    start_time = time.time()
    engine = MaxEngine(self.config, self.devices)
    params = engine.load_params(params=params, rng=self.rng)
    max_logging.log(f"Time taken to initialize engine: {time.time() - start_time} seconds")
    return params, engine

  def _init_tokenizer(self):
    """Initialize the tokenizer.

    Returns:
        Initialized tokenizer
    """
    if self.eos_ids is None and self.tokenizer is None:
      tokenizer_params = self.engine.get_tokenizer()
      self.tokenizer = self.engine.build_tokenizer(tokenizer_params)
    if self.eos_ids is None:
      self.eos_ids = [self.tokenizer.eos_id]
    return self.tokenizer

  def update_params(
      self,
      params: Params,
      destination_sharding: jax.sharding.NamedSharding,
      is_pw_reshard: bool,
  ):
    """Update the model parameters"""
    if is_pw_reshard:
      with (
          jax.transfer_guard_device_to_host("disallow_explicit"),
          jax.transfer_guard_host_to_device("disallow_explicit"),
      ):
        self.params = pathways_reshard.reshard(params, destination_sharding, cache_resharding_plans=True)
    else:
      self.params = jax.device_put(params, destination_sharding)

  def reset_state(self):
    """Reset all worker state for a new inference run.

    This allows reusing the same InferenceWorker instance across multiple
    batch_inference calls without recreating the expensive engine components.
    """
    max_logging.log("Resetting InferenceWorker state")

    # Reset inference state
    self.running = False
    self.completion_tokens_by_id = defaultdict(list)
    self.prompt_logprobs_by_id = defaultdict(list)
    self.empty_decode_slots = queue.Queue()
    for i in range(self.decode_batch_size):
      self.empty_decode_slots.put(i)
    self.slot_to_id = {}
    self.true_lengths = {}
    self.detokenization_queue = queue.Queue()
    self.completed_sequences = set()

    max_logging.log("InferenceWorker state reset complete")

  def run_inference(self, data: list[InputData], rng=None):
    """Start the inference process.

    Args:
        data: list of InputData objects containing input sequences
        rng: Random number generator key. If None, the previous key will be used.
    """

    # Reset state for new inference run
    self.reset_state()

    # Reset rng
    if rng is not None:
      self.rng = rng

    # Set up state for this inference run
    self.true_lengths = {input.id: input.true_length for input in data}
    self.running = True

    max_logging.log("Continuous batching started")

    self._run_continous_batching(data)

    return self._build_final_outputs(data)

  def _run_continous_batching(
      self,
      data: list[InputData],
  ):
    """Run inference on a batch of inputs.

    Args:
        data: list of InputData objects containing input sequences
    """

    # Start detokenization thread
    detokenization_thread = SafeThread(
        target=self.background_detokenization,
        name="detokenization",
    )
    detokenization_thread.start()

    # Process each input
    for row in data:
      # 1. Wait for an empty slot
      while self.empty_decode_slots.empty():
        self.decode()
      # 2. Get an available slot
      slot = self.empty_decode_slots.get()
      # 3. Prefill and insert kv cache
      self.prefill_helper.process(
          model_params=self.params,
          decode_state=self.decode_state,
          decode_slot=slot,
          input_id=row.id,
          input_tokens_padded=row.tokens,
          input_true_length=row.true_length,
          prefill_done=self.prefill_done,
      )

    # 4. Flush any pending inputs in batch prefill mode
    self.prefill_helper.finalize(self.params, self.decode_state, self.prefill_done)

    # 5. Continue decoding until all sequences are complete
    while not all(value is None for value in self.slot_to_id.values()):
      self.decode()

    # Wait for detokenization to complete
    self.running = False
    max_logging.log("Inference worker: joining detokenization thread")
    start_time = time.time()

    with jax.profiler.TraceAnnotation("Flushing detokenization thread"):
      detokenization_thread.join()

    max_logging.log(f"Inference worker: detokenization thread joined in {time.time() - start_time} seconds")

  def _build_final_outputs(self, input_data: list[InputData]) -> list[CompletionOutput]:
    """Build the final list of CompletionOutput."""

    with jax.profiler.TraceAnnotation("offline_engine.batch_inference.return_final_output"):
      completion_outputs = []
      for row in input_data:
        input_id = row.id
        prompt_length = row.true_length
        prompt_tokens = row.tokens[: row.true_length].squeeze()
        completion_tokens = np.array(
            [token_output.token for token_output in self.completion_tokens_by_id[input_id]]
        ).flatten()
        logprobs = np.array(
            [token_output.log_prob.flatten() for token_output in self.completion_tokens_by_id[input_id]]
        ).flatten()
        prompt_logprobs = np.array(self.prompt_logprobs_by_id[input_id]).flatten()
        completion_outputs.append(
            CompletionOutput(
                index=input_id,
                prompt_length=prompt_length,
                token_ids=np.concatenate(
                    (
                        prompt_tokens,
                        completion_tokens,
                    )
                ),
                logprobs=np.concatenate(
                    (
                        prompt_logprobs,
                        logprobs,
                    )
                ),
            )
        )
    return completion_outputs

  def prefill_done(self, prefill_result: list[PrefillResult], prompt_ids: list[any], decode_state: DecodeState):
    """Callback function called when prefill completes.
    This function queues the prefill data for background processing.

    Args:
        prefill_result: list of (token, slot) tuples
        prompt_ids: list of prompt IDs
        decode_state: Updated decode state
    """
    # Update decode state
    self.decode_state = decode_state
    # Process each prefill result
    slots = []
    result_tokens_list = []
    prompt_logp_list = []
    for i, result in enumerate(prefill_result):
      input_id = prompt_ids[i]
      slot = result.slot
      self.slot_to_id[slot] = input_id
      slots.append(slot)
      result_tokens_list.append(result.result_tokens)
      prompt_logp_list.append(result.prompt_logp)

    # Queue detokenization task
    task = DetokenizationTask(
        task_type="prefill",
        result_tokens=result_tokens_list,
        prompt_logp=prompt_logp_list,
        prompt_ids=prompt_ids,
        slots=slots,
    )
    self.detokenization_queue.put_nowait(task)

  def decode(self):
    """Run decode steps on current decoder state.

    Performs `self.min_decode_steps` decode operations
    and queues results for background processing.
    """

    for i in range(self.min_decode_steps):
      # Generate next tokens
      self.decode_state, result_tokens, log_prob = self._jitted_generate_fn(self.params, self.decode_state, self.rng)
      if i == self.min_decode_steps - 1:
        # Block on the last token
        jax.block_until_ready(result_tokens)

      # Queue detokenization task
      task = DetokenizationTask(
          task_type="decode",
          tokens_buffer=result_tokens,
          logprob_buffer=log_prob,
      )

      self.detokenization_queue.put_nowait(task)

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def _jitted_generate_fn(self, params, decode_state, rng):
    decode_state, result_tokens = self.engine.generate(params, decode_state, rng=rng)
    return decode_state, result_tokens.data[:, 0], result_tokens.log_prob

  def background_detokenization(self):
    """Background thread that handles all GPU-to-CPU transfers and token emission.

    This thread processes DetokenizationTask objects from the queue,
    performs the numpy conversions, emits tokens, and manages decode slots.
    """
    max_logging.log("Inference worker: starting detokenization thread")

    while True:
      try:
        task = self.detokenization_queue.get(timeout=0.1)
      except queue.Empty:
        if not self.running and self.detokenization_queue.empty():
          break
        continue

      start_time = time.time()
      newly_empty = []

      if task.task_type == "prefill":

        # Process prefill results - convert to numpy and emit
        with jax.profiler.TraceAnnotation("convert_to_numpy_and_emit_prefill"):
          for i, result_tokens in enumerate(task.result_tokens):
            prompt_id = task.prompt_ids[i]
            slot = task.slots[i]

            prompt_logp = task.prompt_logp[i]
            true_length = self.true_lengths[prompt_id]

            # Convert to numpy
            first_token = np.array(result_tokens.data[:, 0])
            log_prob = np.array(result_tokens.log_prob)
            prompt_logp_np = np.array(prompt_logp)[:, :true_length]

            # Emit token directly
            should_terminate = self.emit_token(prompt_id, int(first_token), log_prob, prompt_logp=prompt_logp_np)
            if should_terminate:
              newly_empty.append(slot)

      elif task.task_type == "decode":

        # Check if there are any active sequences before expensive numpy conversion
        active_slots = []
        for slot, id_ in self.slot_to_id.items():
          if id_ is not None and id_ not in self.completed_sequences:
            active_slots.append((slot, id_))

        # Skip processing entirely if no active sequences
        if not active_slots:
          continue

        # Process single decode step - convert to numpy and emit
        with jax.profiler.TraceAnnotation("convert_to_numpy_and_emit_decode_step"):
          result_tokens_step = np.array(task.tokens_buffer)  # Single step tokens
          log_prob_step = np.array(task.logprob_buffer)  # Single step logprobs

          for slot, id_ in active_slots:
            log_prob_at_slot = log_prob_step[slot]
            result_tokens_at_slot = result_tokens_step[slot]
            should_terminate = self.emit_token(id_, int(result_tokens_at_slot), log_prob_at_slot)
            if should_terminate:
              self.completed_sequences.add(id_)
              newly_empty.append(slot)
      # Update decode slots
      for slot in newly_empty:
        self.slot_to_id[slot] = None
        if slot not in self.empty_decode_slots.queue:
          self.empty_decode_slots.put(slot)

      if self.debug:
        max_logging.log(f"Inference worker: detokenization in {time.time() - start_time} seconds")

  def emit_token(
      self,
      prompt_id,
      result_token: int,
      log_prob: float,
      prompt_logp: np.ndarray = None,
  ):
    """Adds the token to the results for the specified prompt ID and
    determines if generation should terminate.

    Args:
        prompt_id: ID of the prompt
        result_token: Token to emit
        log_prob: Log probability of the token
        prompt_logp: Log probabilities for the prompt tokens

    Returns:
        True if this token signals the end of generation, False otherwise
    """
    # Skip if sequence already completed
    if prompt_id in self.completed_sequences:
      return True

    # Return if already reached max decode length
    if len(self.completion_tokens_by_id[prompt_id]) == self.max_decode_length:
      self.completed_sequences.add(prompt_id)
      return True

    # Return if already reached eos
    if (
        len(self.completion_tokens_by_id[prompt_id]) > 0
        and self.completion_tokens_by_id[prompt_id][-1].token in self.eos_ids
    ):
      self.completed_sequences.add(prompt_id)
      return True

    index = len(self.completion_tokens_by_id[prompt_id])
    if prompt_logp is not None:
      self.prompt_logprobs_by_id[prompt_id] = [prompt_logp]

    self.completion_tokens_by_id[prompt_id].append(TokenOutput(np.array(result_token), np.array(log_prob)))

    # Check if this token completes the sequence
    should_terminate = (result_token in self.eos_ids) or (index + 1 == self.max_decode_length)
    if should_terminate:
      self.completed_sequences.add(prompt_id)

    return should_terminate


class OfflineEngine:
  """Class for handling offline inference on batches of inputs."""

  def __init__(
      self,
      config: Any,
      params: None | Params = None,
      enable_batch_prefill: bool = False,
      min_decode_steps: int = 10,
      tokenizer: Any = None,
      eos_ids: list[int] | None = None,
      prefill_lengths: list[int] | str = "auto",
      batch_prefill_max_batch_size: int = 16,
      mesh: Mesh = None,
      rng: jax.random.PRNGKey = None,
      debug: bool = False,
  ):
    """Initialize the OfflineEngine.

    Args:
        config: The MaxText config object which will be used to
          create MaxEngine instance(s).
        params: Model parameters (loaded from engine if None)
        enable_batch_prefill: Whether to use prefill packing.
            config.scan_layers must be False if this is True
        min_decode_steps: Number of decode steps to perform at a time,
            before checking for completion.
        eos_ids: list of EOS token IDs for checking sequence completion.
          If None, the tokenizer's EOS token will be used.
        tokenizer: Tokenizer instance for encoding/decoding text. If None,
          will be created using the config if eos_ids is not provided.
        prefill_lengths: list of expected prefill lengths, or "auto" to
            automatically determine appropriate lengths from the engine
            config. Input sequences will be padded to the nearest length
            in this list.
        batch_prefill_max_batch_size: Maximum number of inputs to pack
          into a single prefill. This is only used when enable_batch_prefill
          is True.
        mesh: JAX Mesh object. Use this
          argument if you want to use only some of the devices for OfflineEngine and
          reserve the rest for other tasks. If None, OfflineEngine will create the mesh
          automatically.
        rng: Random number generator key. If None, a new key will be created.
    """
    max_logging.log("Initializing OfflineEngine")
    # Configurations
    self.config = config
    self.params = params
    self.min_decode_steps = min_decode_steps
    self.enable_batch_prefill = enable_batch_prefill
    self.mesh = mesh
    self.tokenizer = tokenizer
    self.eos_ids = eos_ids
    self.prefill_lengths = prefill_lengths
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.max_prefill_length = self.config.max_prefill_predict_length
    self.max_decode_length = self.config.max_target_length - self.max_prefill_length
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    self.debug = debug
    self._validate_config()

    # Create prefill buckets: [0, 64], (64, 128], (128, 256], ..., [max_length//2, max_length]
    if prefill_lengths == "auto":
      self.prefill_lengths = [2**i for i in range(6, max(6, (self.max_prefill_length - 1).bit_length()) + 1)]
    else:
      self.prefill_lengths = sorted(prefill_lengths)

    # Create meshes
    if not self.mesh:
      self.mesh = OfflineEngine.create_mesh(jax.devices(), self.config)

    self.worker = InferenceWorker(
        config=self.config,
        params=self.params,
        min_decode_steps=self.min_decode_steps,
        enable_batch_prefill=self.enable_batch_prefill,
        mesh=self.mesh,
        devices=self.mesh.devices.flatten(),
        tokenizer=self.tokenizer,
        eos_ids=self.eos_ids,
        prefill_lengths=self.prefill_lengths,
        max_decode_length=self.max_decode_length,
        batch_prefill_max_batch_size=self.batch_prefill_max_batch_size,
        rng=self.rng,
        debug=self.debug,
    )

    self.tokenizer = self.worker.tokenizer

  def update_params(
      self,
      params: Params,
      parition_spec: PartitionSpec,
      is_pw_reshard: bool,
  ):
    """Update model weights."""
    self.worker.update_params(
        params,
        jax.tree_util.tree_map(
            lambda ps: jax.sharding.NamedSharding(self.mesh, ps),
            parition_spec,
        ),
        is_pw_reshard,
    )

  def batch_inference(
      self,
      data: list[InputData] | list[jax.Array] | list[np.ndarray],
      desc: str = "",
      rng=None,
  ) -> list[CompletionOutput]:
    """Run inference on a batch of inputs.

    Args:
        data: list of InputData objects, or JAX or numpy arrays.
            If input is JAX or numpy array, it must not contain padding tokens.
        desc: Description string for logging
        rng: Random number generator key. If None, the previous key will be used.

    Returns:
        list of CompletionOutput objects, one for each input in data
    """
    data = self.prepare_data(data)

    return self.worker.run_inference(data, rng)

  def prepare_data(self, data: list[InputData | jax.Array | np.ndarray]) -> list[InputData]:
    """Pad and if batch prefill is enabled, sort data by length.

    Args:
        data: list of InputData objects, or JAX or numpy arrays

    Returns:
        list of prepared InputData objects
    """
    # Convert JAX arrays to numpy arrays
    if isinstance(data[0], jax.Array):
      data = [np.array(array) for array in data]

    # Convert numpy arrays to InputData objects
    if isinstance(data[0], np.ndarray):
      max_logging.log(
          "When you provide JAX/numpy arrays to Offline Engine, "
          "make sure that the arrays are not padded with padding tokens."
      )
      data = [InputData(id=i, tokens=array, true_length=len(array)) for i, array in enumerate(data)]

    # Make sure all data id is unique
    if len(data) != len({item.id for item in data}):
      raise ValueError("All data ids must be unique")

    data = self.pad_data(data)

    if self.enable_batch_prefill:
      return sorted(data, key=lambda x: x.tokens.shape[0])

    return data

  def pad_data(self, data: list[InputData]) -> list[InputData]:
    """For each input, pad it to the next length in self.prefill_lengths
    that is greater than or equal to its true length.

    Args:
        data: list of InputData objects

    Returns:
        list of padded InputData objects
    """
    padded_data = []

    for item in data:
      # Find the smallest prefill length that can accommodate this input
      target_length = None
      for length in self.prefill_lengths:
        if length >= item.true_length:
          target_length = length
          break

      # If no suitable length found, use the maximum prefill length
      if target_length is None:
        target_length = self.max_prefill_length

      # Pad or truncate as needed
      if len(item.tokens) < target_length:
        # Pad with zeros
        padded_tokens = np.zeros(target_length, dtype=item.tokens.dtype)
        padded_tokens[: item.true_length] = item.tokens[: item.true_length]
      else:
        # Input is too long, truncate to max_prefill_length
        padded_tokens = item.tokens[:target_length]

      # Create new InputData with padded tokens
      padded_data.append(InputData(id=item.id, tokens=padded_tokens, true_length=item.true_length))

    return padded_data

  @staticmethod
  def create_mesh(devices, config):
    """Create data parallelism meshes for each Inference worker."""
    ici_parallelism = max_utils.fill_unspecified_mesh_axes(config.ici_parallelism.copy(), len(devices), "ICI")
    devices_array = mesh_utils.create_device_mesh(
        ici_parallelism,
        devices,
        contiguous_submeshes=False,
        allow_split_physical_axes=config.allow_split_physical_axes or False,
    )
    mesh = Mesh(devices_array.reshape(ici_parallelism), config.mesh_axes)
    return mesh

  def _validate_config(self):
    """Validate configuration parameters and check for incompatible settings."""
    if not self.config.return_log_prob:
      raise ValueError("return_log_prob must be True when using OfflineEngine")
    if self.enable_batch_prefill and self.config.scan_layers:
      raise ValueError("scan_layers must be False if enable_batch_prefill is True")

    if self.max_decode_length <= 0:
      raise ValueError("Make sure max_target_length - max_prefill_predict_length is greater than 0")
    if self.config.scan_layers:
      max_logging.log(
          "WARNING: scan_layers=True will result in slow step time. " "It is recommended for debugging purposes only."
      )
