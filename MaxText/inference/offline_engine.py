"""
MaxText Offline Inference Engine

Features:
- Continuous batching
- Prefill packing
- Single and multihost with Tensor Parallelism (TP)
- Data Parallelism (DP) with Pathways

Example usage:
    offline_engine = OfflineEngine(
        config=maxtext_config,
        params=None,
        enable_batch_prefill=True,
        dp=1,
        dp_meshes=None
    )

    input_data = [
        jax.numpy.arange(80),
        jax.numpy.arange(90),
        jax.numpy.arange(100),
    ]

    results = offline_engine.batch_inference(input_data)

    for tokens in results:
        text = offline_engine.tokenizer.decode(tokens)
        print(f"Output: {text}")

Notes:
    - Prefill packing is only supported with scan_layers=False
    - DO NOT add print statements in the inference loop as it will
      introduce non-deterministic behavior due to the background
      detokenization thread.
"""

import os
import queue
import signal
import logging
import threading
import traceback
import functools
import dataclasses
from enum import Enum
from typing import Any, List, Tuple, Callable, Optional, Dict, Union
from collections import defaultdict

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import pathwaysutils
from jetstream.engine import engine_api
from MaxText.maxengine import MaxEngine
from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor
DecodeState = Any
Params = Any
MaxTextConfig = Any

# Configure logging
log = logging.getLogger(__name__)


@dataclasses.dataclass
class InputData:
    """Class for storing input data and its metadata.

    Attributes:
        id: Unique identifier for this input
        tokens: JAX array containing the tokenized input
        true_length: Actual length of the input before padding
    """

    id: str
    tokens: jax.Array
    true_length: int


@dataclasses.dataclass
class CompletionOutput:
    """Class for returned output.

    Attributes:
        index: The index of the output in the request.
        token_ids: The token IDs of the generated output text.
        logprobs: The log probabilities of the output tokens.
    """

    index: int
    token_ids: jax.Array
    logprobs: jax.Array


@dataclasses.dataclass
class TokenOutput:
    """Class for storing log probabilities.
    """
    token: int
    log_prob: float


@dataclasses.dataclass
class EventCounter:
    """Class for tracking statistics during inference.

    Attributes:
        input: Number of input sequences processed
        prefill: Number of prefill operations completed
        decode: Number of decode operations completed
        detokenize: Number of sequences completely detokenized
    """

    input: int
    prefill: int
    decode: int
    detokenize: int


class SafeThead(threading.Thread):
    """Thread class with exception handling to prevent silent failures."""

    def run(self):
        try:
            super().run()
        except Exception as e:  # pylint: disable=broad-exception-caught
            traceback.print_exc()
            # Kill the process if a thread encounters an error
            os.kill(os.getpid(), signal.SIGKILL)


class PrefillType(Enum):
    DEFAULT = "default"
    BATCH = "batch"


class PrefillHelper:
    """
    This class abstracts the details of two prefill methods (default vs batch)
    and provides a common interface for prefill operations.
    """

    def __init__(
        self,
        type: PrefillType,
        engine: MaxEngine,
        prefill_lengths: List[int],
        batch_prefill_max_batch_size: int = 16,
        auto_layout_supported: bool = False,
    ):
        """Initialize the PrefillHelper.

        Args:
            type: The type of prefill processor to use ("default" or "batch")
            engine: The MaxEngine instance to use for prefill operations
            prefill_lengths: List of prefill lengths to support
            batch_prefill_max_batch_size: Maximum number of inputs in one packed sequence for batch prefill
            auto_layout_supported: Whether to use auto layout.
        """
        self._type = type
        self.engine = engine
        self.prefill_lengths = sorted(prefill_lengths)
        self.max_prefill_length = self.prefill_lengths[-1]
        self.auto_layout_supported = auto_layout_supported

        if type == PrefillType.DEFAULT:
            self._processor = PrefillProcessor(engine)
        elif type == PrefillType.BATCH:
            self._batch_processor = BatchedPrefillProcessor(
                engine=engine, max_batch_size=batch_prefill_max_batch_size
            )
            # Also create a standard processor for fallback cases
            self._processor = PrefillProcessor(engine)
        else:
            raise ValueError(f"Invalid prefill type: {type}")

    def aot_compile(
        self,
        params: Params,
        decode_state: DecodeState,
    ) -> DecodeState:
        """Ahead-of-time compile prefill functions for all supported lengths."""

        if self._type == PrefillType.DEFAULT:
            decode_state = self.aot_compile_default(params, decode_state)
        elif self._type == PrefillType.BATCH:
            decode_state = self.aot_compile_batch(params, decode_state)
        return decode_state

    def aot_compile_default(self, params: Params, decode_state: DecodeState):
        """AOT compile the default prefill processor."""
        for length in self.prefill_lengths:
            print(f"AOT compiling prefill for length: {length}")
            if self.auto_layout_supported:
                self._processor.aot_compile(params, length)
            else:
                tokens = jnp.zeros((length,), dtype=jnp.int32)
                _, decode_state = self._processor._process(
                    params, tokens, 0, length, decode_state
                )
        return decode_state

    def aot_compile_batch(self, params: Params, decode_state: DecodeState):
        """AOT compile the batch prefill processor."""
        max_length = self.max_prefill_length

        for padded_length in self.prefill_lengths:
            for num_prompts in range(1, 2 * max_length // padded_length):
                print(
                    f"AOT compiling batch prefill for length: {padded_length} and num_prompts: {num_prompts}"
                )
                if self.auto_layout_supported:
                    self._batch_processor.aot_compile(
                        params, padded_length, max_length, num_prompts
                    )
                else:
                    tokens = jax.ShapeDtypeStruct((max_length,), jnp.dtype("int32"))
                    slots = jnp.arange(0, self.batch_prefill_max_batch_size, dtype=int)
                    decoder_positions = jnp.arange(0, max_length, dtype=int)
                    decoder_segment_ids = jnp.ones(max_length, dtype=int)
                    start_pos = jnp.arange(
                        0,
                        max_length,
                        max_length // self.batch_prefill_max_batch_size,
                        dtype=int,
                    )
                    true_lengths = jnp.full(
                        self.batch_prefill_max_batch_size, padded_length, dtype=int
                    )
                    _, decode_state = self._batch_processor._process_batch(
                        params,
                        tokens,
                        slots,
                        num_prompts,
                        decoder_positions,
                        decoder_segment_ids,
                        start_pos,
                        padded_length,
                        true_lengths,
                        decode_state,
                    )
        # For fallback
        print(f"AOT compiling prefill for length: {max_length}")
        if self.auto_layout_supported:
            self._processor.aot_compile(params, max_length)
        else:
            tokens = jnp.zeros((max_length,), dtype=jnp.int32)
            _, decode_state = self._processor._process(
                params, tokens, 0, max_length, decode_state
            )
        return decode_state

    def process(
        self,
        model_params: Params,
        decode_state: DecodeState,
        decode_slot: int,
        input_id: int,
        input_tokens_padded: jax.Array,
        input_true_length: int,
        prefill_done: Callable[
            [List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None
        ],
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
        if (
            self._type == PrefillType.DEFAULT
            or padded_length == self.max_prefill_length
        ):
            if self.auto_layout_supported:
                first_token, decode_state = self._processor.process(
                    model_params,
                    decode_state,
                    decode_slot,
                    input_tokens_padded,
                    input_true_length,
                )
            else:
                first_token, decode_state = self._processor._process(
                    model_params,
                    input_tokens_padded,
                    decode_slot,
                    input_true_length,
                    decode_state,
                )
            prefill_done([(first_token, decode_slot)], [input_id], decode_state)
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
            )

    def finalize(
        self,
        model_params: Params,
        decode_state: DecodeState,
        prefill_done: Callable[
            [List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None
        ],
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
            self._batch_processor.flush(model_params, decode_state, prefill_done)


class ReplicaWorker:
    """Worker class that manages inference on a single model replica with
    continuous batching.
    """

    def __init__(
        self,
        config: MaxTextConfig,
        params: Params | None,
        min_decode_steps: int,
        enable_batch_prefill: bool,
        devices: List[Any],
        tokenizer: Any,
        eos_ids: List[int],
        prefill_lengths: List[int],
        batch_prefill_max_batch_size: int,
        worker_id: int = 0,
        auto_layout_supported: bool = False,
        run_as_a_thread=False,
    ):
        """Initialize a ReplicaWorker.

        Args:
            config: MaxText configuration
            params: Model parameters, if None, the params will be loaded from the config
            min_decode_steps: Minimum number of decode steps to run at once
            enable_batch_prefill: Whether to enable batch prefill
            devices: JAX devices to use for this worker
            tokenizer: Tokenizer to use
            eos_ids: End-of-sequence token IDs
            prefill_lengths: List of supported prefill lengths
            batch_prefill_max_batch_size: Maximum batch size for batch prefill
            worker_id: Worker identifier
            auto_layout_supported: Whether auto layout is supported
            run_as_a_thread: Whether to run in a separate thread
        """
        # Configurations
        self.config = config
        self.params = params
        self.min_decode_steps = min_decode_steps
        self.enable_batch_prefill = enable_batch_prefill
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.prefill_lengths = prefill_lengths
        self.devices = devices
        self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
        self.max_prefill_length = self.config.max_prefill_predict_length
        self.max_decode_length = 5 #self.config.max_target_length - self.max_prefill_length
        self.worker_id = worker_id
        self.run_as_a_thread = run_as_a_thread
        self.auto_layout_supported = auto_layout_supported

        # Initialize MaxEngine(s)
        self.params, self.engine = self._init_engine(params)
        self.tokenizer = self._init_tokenizer()
        self.batch_size = self.engine.max_concurrent_decodes

        # Create a Prefill Helper for each maxengine
        if enable_batch_prefill:
            self.prefill_helper = PrefillHelper(
                PrefillType.BATCH,
                self.engine,
                self.prefill_lengths,
                self.batch_prefill_max_batch_size,
                self.auto_layout_supported,
            )
        else:
            self.prefill_helper = PrefillHelper(
                PrefillType.DEFAULT,
                self.engine,
                self.prefill_lengths,
                self.auto_layout_supported,
            )

        # State management
        self.detokenize_backlog = queue.Queue()
        self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
        self.empty_decode_slots: List[int] = []
        self.slot_to_id: Dict[int, int] = {}
        self.decode_state = None
        self.res = None  # Results storage

        # Compiled functions
        if auto_layout_supported:
            self.generate_fn, self.params, init_decode_state_fn = (
                self.engine.aot_compile(self.params, pass_rng_shape=False)
            )
            self.decode_state = init_decode_state_fn(None)
        else:
            self.generate_fn = self.engine.generate
            self.decode_state = self.engine.init_decode_state()

        self.worker_thread = None
        self.warm_up_thread = None
        self.running = False

    def _init_engine(self, params):
        """Initialize the MaxEngine.

        Args:
            params: Model parameters

        Returns:
            Tuple of (params, engine)
        """
        engine = MaxEngine(self.config, self.devices)
        params = engine.load_params(params)
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

    def warm_up_impl(self):
        """Warm up the engine by compiling and running on dummy data."""
        self.decode_state = self.prefill_helper.aot_compile(
            self.params, self.decode_state
        )
        assert self.decode_state is not None

        for _ in range(3):
            print("Warming up generate function")
            self.decode_state, result_tokens = self.generate_fn(
                self.params, self.decode_state, None
            )

    def warm_up(self):
        """Start warm-up process, optionally in a separate thread."""
        if self.run_as_a_thread:
            self.warm_up_thread = SafeThead(
                target=self.warm_up_impl,
                name=f"replica_worker_{self.worker_id}",
            )
            self.warm_up_thread.start()
        else:
            self.warm_up_impl()

    def ensure_warm_up_finished(self):
        """Ensure the warm up thread is finished"""
        if self.warm_up_thread is not None:
            self.warm_up_thread.join()
            self.warm_up_thread = None

    def start_inference(
        self,
        data_queue: queue.Queue,
        results,
        desc: str = "",
    ):
        """Start the inference process.

        Args:
            data_queue: Queue containing input data
            results: Dictionary to store results
            desc: Description for logging
        """
        self.ensure_warm_up_finished()
        self.res = results
        if self.run_as_a_thread:
            self.worker_thread = SafeThead(
                target=self._start_inference,
                args=(data_queue, desc),
                name=f"replica_worker_{self.worker_id}",
            )
            self.worker_thread.start()
        else:
            self._start_inference(data_queue, desc)

    def stop(self):
        """Stop the inference process and wait for completion."""
        if self.worker_thread is not None:
            self.worker_thread.join()
            self.worker_thread = None

    def _start_inference(
        self,
        data_queue: queue.Queue,
        desc: str = "",
    ) -> Dict[str, List[int]]:
        """Run inference on a batch of inputs.

        Args:
            data_queue: Queue of InputData objects containing input sequences
            desc: Description string for logging

        Returns:
            Dictionary mapping input ids to output token sequences
        """
        print("Starting inference")
        # Reset state
        self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
        self.empty_decode_slots = list(range(self.batch_size))
        self.slot_to_id = {}

        # Start detokenization thread
        detokenize_thread = SafeThead(
            target=functools.partial(
                self.detokenize,
            ),
            name="detokenize",
        )
        self.running = True
        detokenize_thread.start()

        # Process each input
        while not data_queue.empty():
            try:
                print("replica worker: ", self.worker_id, " processing input")
                row = data_queue.get_nowait()
            except queue.Empty:
                break

            # 1. Wait for an empty slot
            while not self.empty_decode_slots:
                print("replica worker: ", self.worker_id, " running decode")
                self.decode()

            # 2. Get an available slot
            slot = self.empty_decode_slots.pop()
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
        while self.slot_to_id:
            self.decode()

        # Wait for detokenization to complete
        self.running = False
        print("replica worker: ", self.worker_id, " joining detokenize thread")
        detokenize_thread.join()
        print("replica worker: ", self.worker_id, " detokenize thread joined")

        # Log completion statistics
        log.info(
            "Summary %s: prefills=%d, decodes=%d, detokens=%d completed.",
            desc,
            self.counter.prefill,
            self.counter.decode,
            self.counter.detokenize,
        )
        return self.res

    def emit_token(self, prompt_id, result_token: engine_api.ResultTokens, slot: int):
        """Adds the token to the results for the specified prompt ID and
        determines if generation should terminate.

        Args:
            prompt_id: ID of the prompt
            token: Token to emit

        Returns:
            True if this token signals the end of generation, False otherwise
        """
        token, is_valid, length = result_token.data[slot]
        if not is_valid:
            print("invalid token: ", token, length)
        # log_prob = result_token.log_prob[slot]
        log_prob = 1
        assert length <= self.max_decode_length

        already_reached_eos = (
            len(self.res[prompt_id]) > 0
            and self.res[prompt_id][-1].token == self.tokenizer.eos_id
        )

        if is_valid and not already_reached_eos:
            # print("emitting token: ", token)
            self.res[prompt_id].append(TokenOutput(token, log_prob))

        return (token == self.tokenizer.eos_id) or (length == self.max_decode_length)

    def prefill_done(self, prefill_result, prompt_ids, decode_state):
        """Callback function called when prefill completes.
        This function adds the prefill tokens to the detokenization queue,
        which manages the token emission and decode slot evictions.

        Args:
            prefill_result: List of (token, slot) tuples
            prompt_ids: List of prompt IDs
            decode_state: Updated decode state
        """
        # Update decode state
        self.decode_state = decode_state

        # Process each prefill result
        for i, (first_token, slot) in enumerate(prefill_result):
            self.counter.prefill += 1
            self.slot_to_id[slot] = prompt_ids[i]

            # Add token to detokenization queue
            self.detokenize_backlog.put_nowait(
                (first_token, True, prompt_ids[i], slot)
            )

    def decode(self):
        """Run decode steps on current decoder state.

        Performs `self.min_decode_steps` decode operations
        and puts results in the detokenization queue.
        """
        buffer = []
        for i in range(self.min_decode_steps):
            # Generate next tokens
            self.decode_state, result_tokens = self.generate_fn(
                self.params, self.decode_state, None
            )
            buffer.append(result_tokens)
        
        # Add results to detokenization queue
        # DO not merge this loop with the previous loop
        for result_tokens in buffer:
            self.detokenize_backlog.put_nowait(
                (result_tokens, False, 0, 0)
            )       
            self.counter.decode += 1

    def detokenize(self):
        """Detokenize results and manage decode slots.

        Runs in a background thread to process tokens from
        the detokenization queue, emit tokens, and free up
        decode slots when sequences complete.
        """
        while self.running or not self.detokenize_backlog.empty():
            newly_empty = []

            # Get next item from queue with timeout
            try:
                result_tokens, is_first_token, row_id, slot = self.detokenize_backlog.get_nowait()
                result_tokens = result_tokens.convert_to_numpy()
            except queue.Empty:
                if not self.running:
                    break
                continue

            # Process generated tokens
            if is_first_token:
                should_terminate = self.emit_token(row_id, result_tokens, slot=0)
                if should_terminate:
                    newly_empty.append(slot)
            else:
                for slot, id_ in self.slot_to_id.items():
                    should_terminate = self.emit_token(id_, result_tokens, slot=slot)
                    if should_terminate:
                        newly_empty.append(slot)

            # Update decode slots
            for slot in newly_empty:
                self.counter.detokenize += 1
                del self.slot_to_id[slot]
                self.empty_decode_slots.append(slot)


class OfflineEngine:
    """Class for handling offline inference on batches of inputs.

    The logic is as follows:
    1. Process input one at a time
    2. Pad the input data to the nearest power of 2
    3. Prefill input and insert KV cache
    4. Keep on processing inputs and prefilling until
       there are enough samples to do batch decoding
    5. Decode until at least one of the samples is finished,
       so there is room for new samples
    6. Prefill to fill up the newly emptied decode slots
    7. Repeat steps 5 and 6 until all samples are finished
    8. A background thread is used to detokenize the results
    9. Return the results

    Prefill Packing:
        When enable_batch_prefill is True, the prefill processor
        will pack multiple inputs into a single sequence before
        doing the prefill.

        There are multiple buckets for packed sequences, where each bucket
        contains inputs with the same padded length. Only inputs with the same
        padded length will be packed together.

        It is important to sort the inputs by padded length so that the
        buckets fill up quickly.

        When a decode slot is freed up, the prefill processor will add the
        sequence to the bucket. If the bucket is full, the bucket will be
        prefilled.

        E.g.
        Bucket for length 64: [...seq1, ...seq2, ...seq3, ...seq4]
        Bucket for length 128: [...seq1, ...seq2]
        Bucket for length 256: [...seq1]
    """

    def __init__(
        self,
        config: Any,
        params: Optional[Params] = None,
        enable_batch_prefill: bool = False,
        min_decode_steps: int = 10,
        tokenizer: Any = None,
        eos_ids=None,
        prefill_lengths: Union[List[int], str] = "auto",
        batch_prefill_max_batch_size: int = 16,
        dp=1,
        dp_meshes=None,
        auto_layout_supported=False,
    ):
        """Initialize the OfflineEngine.

        Args:
            config: The MaxText config object which will be used to
              create MaxEngine instance(s) and potentially, the tokenizer.
            params: Model parameters (loaded from engine if None)
            enable_batch_prefill: Whether to use prefill packing.
                config.scan_layers must be False if this is True
            min_decode_steps: Number of decode steps to perform at a time,
                before checking for completion.
            eos_ids: List of EOS token IDs for checking sequence completion.
              If None, the tokenizer's EOS token will be used.
            tokenizer: Tokenizer instance for encoding/decoding text. If None,
              will be created using the config.
            prefill_lengths: List of expected prefill lengths, or "auto" to
                automatically determine appropriate lengths from the engine
                config. Input sequences will be padded to the nearest length
                in this list.
            batch_prefill_max_batch_size: Maximum number of inputs to pack
              into a single prefill. This is only used when enable_batch_prefill
              is True.
            warm_up: Whether to precompile prefill and decode functions for
              each length in the prefill_lengths list during initialization.
              Alternatively compilation will be done during runtime, or by
              directly calling the warm_up() function.
            dp: Data parallelism, number of replicas of the model to run. This
              helps to increase throughput by running multiple inference replicas
              in parallel. When setting dp>1, Pathways must be used. Either provide
              the dp_meshes for each model replica, or let OfflineEngine automatically
              create the meshes which will make use of all devices.
            dp_meshes: List of JAX Mesh objects for each model replica. Use this
              option if you want to use only some of the devices for OfflineEngine and
              reserve the rest for other tasks. If None, OfflineEngine will create the meshes
              automatically.
            auto_layout_supported: Whether auto layout is supported
        """
        # Configurations
        self.config = config
        self.params = params
        self.min_decode_steps = min_decode_steps
        self.enable_batch_prefill = enable_batch_prefill
        self.dp = dp
        self.dp_meshes = dp_meshes
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.prefill_lengths = prefill_lengths
        self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
        self.max_prefill_length = self.config.max_prefill_predict_length
        self.max_decode_length = self.config.max_target_length - self.max_prefill_length
        self.auto_layout_supported = auto_layout_supported
        self.not_warmed_up = True
        self._validate_config()

        # Create prefill buckets: [0, 64], (64, 128], (128, 256], ..., [max_length//2, max_length]
        if prefill_lengths == "auto":
            self.prefill_lengths = [
                2**i
                for i in range(
                    6, max(6, (self.max_prefill_length - 1).bit_length()) + 1
                )
            ]
        else:
            self.prefill_lengths = sorted(prefill_lengths)

        # Create meshes
        devices = jax.devices()

        if not self.dp_meshes:
            self.dp_meshes = []
            num_devices_per_replica = len(devices) // self.dp
            mesh_shape = self.config.ici_parallelism

            for i in range(self.dp):
                mesh_devices = np.array(
                    devices[
                        i * num_devices_per_replica : (i + 1) * num_devices_per_replica
                    ]
                ).reshape(mesh_shape)
                print(f"Mesh devices shape: {mesh_devices.shape}")
                self.dp_meshes.append(Mesh(mesh_devices, self.config.mesh_axes))

        # Initialize ReplicaWorkers
        run_as_a_thread = self.dp > 1  # No need to run worker as a thread if there is only one replica
        self.replica_workers = [
            ReplicaWorker(
                config=self.config,
                params=self.params,
                min_decode_steps=self.min_decode_steps,
                enable_batch_prefill=self.enable_batch_prefill,
                devices=np.squeeze(self.dp_meshes[i].devices),
                tokenizer=self.tokenizer,
                eos_ids=self.eos_ids,
                prefill_lengths=self.prefill_lengths,
                batch_prefill_max_batch_size=self.batch_prefill_max_batch_size,
                worker_id=i,
                auto_layout_supported=auto_layout_supported,
                run_as_a_thread=run_as_a_thread,
            )
            for i in range(self.dp)
        ]
        print(f"Created {self.dp} replica workers")
        self.tokenizer = self.replica_workers[0].tokenizer
        if self.warm_up:
            self.warm_up()

    def warm_up(self):
        if self.not_warmed_up:
            """Warm up all replica workers."""
            for i in range(self.dp):
                self.replica_workers[i].warm_up()
            for i in range(self.dp):
                self.replica_workers[i].ensure_warm_up_finished()
            self.not_warmed_up = False

    def batch_inference(
        self,
        data: Union[List[InputData], List[jax.Array]],
        desc: str = "",
    ) -> Dict[str, List[int]]:
        """Run inference on a batch of inputs.

        Args:
            data: List of InputData objects or JAX arrays containing input tokens (and no padding tokens).
            desc: Description string for logging

        Returns:
            Dictionary mapping input ids to output token sequences or a list of token sequences
        """
        data = self.prepare_data(data)

        # Create thread-safe queue and results container
        data_queue = queue.Queue()
        results = defaultdict(list)

        # Add all data to the queue
        for row in data:
            data_queue.put_nowait(row)

        # Start inference on all replica workers
        for i in range(self.dp):
            self.replica_workers[i].start_inference(data_queue, results, desc)

        # Wait for all workers to complete
        for i in range(self.dp):
            self.replica_workers[i].stop()
            print("replica worker: ", i, " stopped")

        # Return CompletionOutput objects
        completion_outputs = []
        for input_data in data:
            completion_outputs.append(
                CompletionOutput(
                    index=input_data.id,
                    token_ids=[token.token for token in results[input_data.id]],
                    logprobs=[token.log_prob for token in results[input_data.id]],
                )
            )
        return completion_outputs

    def pad_data(self, data: List[InputData]) -> List[InputData]:
        """For each input, pad it to the next length in self.prefill_lengths
        that is greater than or equal to its true length.

        Args:
            data: List of input data objects

        Returns:
            List of padded input data objects
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
                padded_tokens = jax.numpy.zeros(target_length, dtype=item.tokens.dtype)
                padded_tokens = padded_tokens.at[: item.true_length].set(
                    item.tokens[: item.true_length]
                )
            else:
                # Input is too long, truncate to max_prefill_length
                padded_tokens = item.tokens[:target_length]

            # Create new InputData with padded tokens
            padded_data.append(
                InputData(
                    id=item.id, tokens=padded_tokens, true_length=item.true_length
                )
            )

        return padded_data

    def prepare_data(self, data: List[Union[InputData, jax.Array]]) -> List[InputData]:
        """Pad and if batch prefill is enabled, sort data by length.

        Args:
            data: List of InputData objects or JAX arrays

        Returns:
            List of prepared InputData objects
        """
        # Convert JAX arrays to InputData objects
        if isinstance(data[0], jax.Array):
            data = [
                InputData(id=i, tokens=array, true_length=len(array))
                for i, array in enumerate(data)
            ]

        data = self.pad_data(data)

        if self.enable_batch_prefill:
            return sorted(data, key=lambda x: x.tokens.shape[0])

        return data

    def _validate_config(self):
        """Validate configuration parameters and check for incompatible settings."""
        # Check if batch prefill is compatible with scan layers
        if self.enable_batch_prefill and self.config.scan_layers:
            raise ValueError(
                "scan_layers must be False if enable_batch_prefill is True"
            )

        # Ensure there's enough room for decoding
        if self.max_decode_length <= 0:
            raise ValueError(
                "Make sure max_target_length - max_prefill_predict_length is greater than 0"
            )

        # Initialize Pathways if using data parallelism
        if self.dp > 1:
            # Initialize Pathways if not already initialized
            pathwaysutils.initialize()
