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
        max_logging.log(f"Output: {text}")

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
import time

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils


from jetstream.engine import engine_api
from MaxText.maxengine import MaxEngine
from MaxText import max_utils
from MaxText import inference_utils
from jax.sharding import PartitionSpec as P, NamedSharding


from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor
from MaxText import max_logging

DecodeState = Any
Params = Any
MaxTextConfig = Any
DEBUG = os.environ.get("DEBUG", "0") == "1"
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
    """Class for storing log probabilities."""

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
        rng=None,
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
        self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng
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
            max_logging.log(f"AOT compiling prefill for length: {length}")
            if self.auto_layout_supported:
                self._processor.aot_compile(params, length)
            else:
                tokens = jnp.zeros((length,), dtype=jnp.int32)
                _, decode_state = self._processor._process(
                    params, tokens, 0, length, decode_state, rng=self.rng
                )
        return decode_state

    def aot_compile_batch(self, params: Params, decode_state: DecodeState):
        """AOT compile the batch prefill processor."""
        max_length = self.max_prefill_length

        for padded_length in self.prefill_lengths:
            for num_prompts in range(1, 2 * max_length // padded_length):
                max_logging.log(
                    f"AOT compiling batch prefill for length: {padded_length} and num_prompts: {num_prompts}"
                )
                if self.auto_layout_supported:
                    self._batch_processor.aot_compile(
                        params, padded_length, max_length, num_prompts
                    )
                else:
                    tokens = jnp.arange(0, max_length, dtype=int)
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
        if DEBUG:
            max_logging.log(f"AOT compiling prefill for length: {max_length}")
        if self.auto_layout_supported:
            self._processor.aot_compile(params, max_length)
        else:
            tokens = jnp.zeros((max_length,), dtype=jnp.int32)
            _, decode_state = self._processor._process(
                params, tokens, 0, max_length, decode_state
            )
        return decode_state
    
    # TODO: clean up this code logic. 
    @staticmethod
    @functools.partial(jax.jit, static_argnums=(6,7), donate_argnums=(4,))
    def _jitted_process(params, tokens, slot, true_length, decode_state, rng, processor_fn, auto_layout_supported=False):
        if auto_layout_supported:
            first_token, decode_state = processor_fn(
                params,
                decode_state,
                slot,
                tokens,
                true_length,
                rng,
            )
        else:
            first_token, decode_state = processor_fn(
                params,
                tokens,
                slot,
                true_length,
                decode_state,
                rng,
            )
        log_prob = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"][slot], decode_state["tokens"][slot]
        ) 

        # Shapes: (1,), (1,)
        return first_token.data[:, 0], log_prob, decode_state

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
        start_time = time.time()
        padded_length = len(input_tokens_padded)
        # Use default processor if configured or if input is already at max length
        if (
            self._type == PrefillType.DEFAULT
            or padded_length == self.max_prefill_length
        ):
            prefill_fn = self._processor.process if self.auto_layout_supported else self._processor._process
            first_token, log_prob, decode_state = self._jitted_process(
                model_params,
                input_tokens_padded,
                decode_slot,
                input_true_length,
                decode_state,
                self.rng,
                prefill_fn,
                self.auto_layout_supported,
            )
            prefill_done([(first_token, log_prob, decode_slot)], [input_id], decode_state)
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
        end_time = time.time()
        if DEBUG:
            max_logging.log(
                f"Time taken to run prefill: {end_time - start_time} seconds"
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
        max_decode_length: int,
        batch_prefill_max_batch_size: int,
        worker_id: int = 0,
        auto_layout_supported: bool = False,
        run_as_a_thread=False,
        rng=None,
        mesh=None,
    ):
        """Initialize a ReplicaWorker that uses continuous batching over
           a queue of inputs.

        Continuous batching logic:
        1. Process input one at a time
        2. Prefill input and insert KV cache
        3. Keep on processing inputs and prefilling until
           there are enough samples to do batch decoding
        4. Decode until at least one of the samples is finished,
           so there is room for new samples
        5. Prefill to fill up the newly emptied decode slots
        6. Repeat steps 5 and 6 until all samples are finished
        7. A background thread is used to detokenize the results
        8. Return the results

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
        self.worker_id = worker_id
        self.config = config
        self.params = params
        self.devices = devices
        self.enable_batch_prefill = enable_batch_prefill
        self.prefill_lengths = prefill_lengths
        self.max_prefill_length = self.prefill_lengths[-1]
        self.max_decode_length = max_decode_length
        self.eos_ids = eos_ids
        self.tokenizer = tokenizer
        self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
        self.min_decode_steps = min_decode_steps
        self.run_as_a_thread = run_as_a_thread
        self.auto_layout_supported = auto_layout_supported
        self.mesh = mesh
        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng
        # Thread management
        self.worker_thread = None
        self.warm_up_thread = None
        self.init_thread = None
        self.running = False

        # State management
        self.detokenize_backlog = queue.Queue()
        self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
        self.empty_decode_slots: List[int] = []
        self.slot_to_id: Dict[int, int] = {}
        self.decode_state = None
        self.res = None

        # Precompiled functions
        self.generate_fn = None

        # These will be populated in the init thread
        self.engine = None
        self.decode_batch_size = None
        self.prefill_helper = None

        self._init()

    def _init(self):
        if self.run_as_a_thread:
            self.init_thread = SafeThead(
                target=self._init_impl,
                name=f"replica_worker_{self.worker_id}",
            )
            self.init_thread.start()
        else:
            self._init_impl()

    def ensure_init_finished(self):
        if self.init_thread is not None:
            self.init_thread.join()
            self.init_thread = None

    def _init_impl(self):
        # Initialize MaxEngine(s)
        self.params, self.engine = self._init_engine(self.params)
        self.tokenizer = self._init_tokenizer()
        self.decode_batch_size = self.engine.max_concurrent_decodes

        # Create a Prefill Helper
        if self.enable_batch_prefill:
            self.prefill_helper = PrefillHelper(
                PrefillType.BATCH,
                self.engine,
                self.prefill_lengths,
                self.batch_prefill_max_batch_size,
                self.auto_layout_supported,
                rng=self.rng,
            )
        else:
            self.prefill_helper = PrefillHelper(
                PrefillType.DEFAULT,
                self.engine,
                self.prefill_lengths,
                self.auto_layout_supported,
                rng=self.rng,
            )

        # Initialize decode state and generate function
        if self.auto_layout_supported:
            start_time = time.time()
            self.generate_fn, self.params, init_decode_state_fn = (
                self.engine.aot_compile(self.params, pass_rng_shape=True)
            )
            end_time = time.time()
            max_logging.log(
                f"time taken to compile generate_fn: {end_time - start_time} seconds"
            )
            self.decode_state = init_decode_state_fn(self.rng)
        else:
            self.generate_fn = self.engine.generate
            start_time = time.time()
            self.decode_state = self.engine.init_decode_state(self.rng)
            end_time = time.time()
            max_logging.log(
                f"time taken to initialize decode_state: {end_time - start_time} seconds"
            )

        self.generate_fn = self.engine.generate

    def _init_engine(self, params):
        """Initialize the MaxEngine.

        Args:
            params: Model parameters

        Returns:
            Tuple of (params, engine)
        """
        start_time = time.time()
        engine = MaxEngine(self.config, self.devices)
        params = engine.load_params(params=params, rng=self.rng)
        end_time = time.time()
        max_logging.log(
            f"time taken to initialize engine: {end_time - start_time} seconds"
        )
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

        for i in range(3):
            if DEBUG:
                max_logging.log(f"Warm up generate function ({i + 1}/3)")
            self.decode_state, result_tokens = self.generate_fn(
                self.params, self.decode_state, self.rng
            )

    def warm_up(self):
        """Start warm-up process, optionally in a separate thread."""
        self.ensure_init_finished()
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

    def update_params(self, params: Params):
        self.params = self.engine.load_params(params=params, rng=self.rng)

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
        self.ensure_init_finished()
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

    def ensure_inference_finished(self):
        """Stop the inference process and wait for completion."""
        if self.worker_thread is not None:
            self.worker_thread.join()
            self.worker_thread = None

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(5,))
    def _jitted_prefill(
        engine,
        params: Params,
        tokens: jax.Array,
        slot: int,
        true_length: int,
        decode_state: DecodeState,
        rng: Any,
    ) -> Tuple[engine_api.ResultTokens, DecodeState]:
        """Prefill and insert a request."""

        prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length, rng=rng)
        decode_state = engine.insert(prefill_result, decode_state, slot)
        log_prob = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"][slot], decode_state["tokens"][slot]
        ) 
        return first_token.data[:, 0], log_prob, decode_state

    def simple_prefill(self, model_params, decode_state, decode_slot, input_id, input_tokens_padded, input_true_length, prefill_done):
        first_token, log_prob, decode_state = self._jitted_prefill(self.engine, model_params, input_tokens_padded, decode_slot, input_true_length, decode_state, self.rng)        
        prefill_done([(first_token, log_prob, decode_slot)], [input_id], decode_state)

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
        if DEBUG:
            max_logging.log(f"Replica {self.worker_id}. Starting inference")
        # Reset state
        self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
        self.empty_decode_slots = list(range(self.decode_batch_size))
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
                row = data_queue.get(timeout=0.01)
            except queue.Empty:
                break

            # 1. Wait for an empty slot
            while not self.empty_decode_slots:
                self.decode()

            # 2. Get an available slot
            slot = self.empty_decode_slots.pop()
            # 3. Prefill and insert kv cache
            # TODO: This function is recompiling 
            # self.prefill_helper.process(
            #     model_params=self.params,
            #     decode_state=self.decode_state,
            #     decode_slot=slot,
            #     input_id=row.id,
            #     input_tokens_padded=row.tokens,
            #     input_true_length=row.true_length,
            #     prefill_done=self.prefill_done,
            # )

            self.simple_prefill( 
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
        if DEBUG:
            max_logging.log(
                f"replica worker {self.worker_id}: joining detokenize thread"
            )
        detokenize_thread.join()
        if DEBUG:
            max_logging.log(
                f"replica worker {self.worker_id}: detokenize thread joined"
            )

        # Log completion statistics
        if DEBUG:
            max_logging.log(
                f"Summary {desc}: prefills={self.counter.prefill}, decodes={self.counter.decode}, detokens={self.counter.detokenize} completed."
            )
        return self.res

        
    def prefill_done(self, prefill_result, prompt_ids, decode_state):
        """Callback function called when prefill completes.
        This function adds the prefill tokens to the detokenization queue,
        which manages the token emission and decode slot evictions.

        Args:
            prefill_result: List of (token, slot) tuples
            prompt_ids: List of prompt IDs
            decode_state: Updated decode state
        """
        if DEBUG:
            max_logging.log("Replica {}. Prefill done".format(self.worker_id))
        # Update decode state
        self.decode_state = decode_state
        # Process each prefill result
        for i, (first_token, log_prob, slot) in enumerate(prefill_result):
            self.counter.prefill += 1
            self.slot_to_id[slot] = prompt_ids[i]
            # Add token to detokenization queue
            self.detokenize_backlog.put_nowait(
                (first_token, log_prob, True, prompt_ids[i], slot)
            )

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
    def _jitted_generate_fn(self, params, decode_state, rng):
        decode_state, result_tokens = self.engine.generate(
            params, decode_state, rng=rng
        )
        logps = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"], decode_state["tokens"]
        )

        tokens_gathered = jax.device_get(result_tokens.data[:, 0])
        logps_gathered = jax.device_get(logps) 

        return decode_state, tokens_gathered, logps_gathered

    def _scan_generate_step(self, carry, _):
        rng, decode_state = carry
        rng, rng_generate = jax.random.split(rng)
        decode_state, result_tokens = self.engine.generate(
            self.params, decode_state, rng=rng_generate
        )
        logps = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"], decode_state["tokens"]
        )
        return (rng, decode_state), (result_tokens.data[:, 0], logps)

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _jitted_scan_generate_step(self, decode_state):
        (_, decode_state), (all_tokens, all_logps) = jax.lax.scan(
            self._scan_generate_step,
            init=(self.rng, decode_state),
            xs=None,
            length=self.min_decode_steps,
        )
        jax.lax.with_sharding_constraint(all_tokens, NamedSharding(self.mesh, P(None)))
        jax.lax.with_sharding_constraint(all_logps, NamedSharding(self.mesh, P(None)))
        return decode_state, all_tokens, all_logps
    
    def decode(self):
        """Run decode steps on current decoder state.

        Performs `self.min_decode_steps` decode operations
        and puts results in the detokenization queue.
        """

        # DO NOT SUBMIT. Scan version
        # self.decode_state, all_tokens, all_logps = self._jitted_scan_generate_step(self.decode_state)
        # all_logps.block_until_ready()
        # self.detokenize_backlog.put_nowait((all_tokens, all_logps, False, 0, 0))
        # self.counter.decode += 1
        

        ### DO NOT SUBMIT. Non-scan version ###
        buffer = []
        for i in range(self.min_decode_steps):
            # Generate next tokens
            start_time = time.time()
            self.decode_state, result_tokens, log_prob = self._jitted_generate_fn(
                self.params, self.decode_state, self.rng
            )
            with jax.profiler.TraceAnnotation("log_prob_block_until_ready"):
                log_prob.block_until_ready()
            end_time = time.time()
            if DEBUG:
                max_logging.log("Replica {}. Time taken to run generate_fn: {} seconds".format(self.worker_id, end_time - start_time))
            buffer.append((result_tokens, log_prob))

        # Add results to detokenization queue
        self.detokenize_backlog.put_nowait(([result_token for result_token, _ in buffer], [log_prob for _, log_prob in buffer], False, 0, 0))
        self.counter.decode += 1
        ### DO NOT SUBMIT. Non-scan version ends ###

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
                result_tokens, log_prob, is_first_token, row_id, slot = (
                    self.detokenize_backlog.get(timeout=1)
                )

            except queue.Empty:
                if not self.running:
                    break
                continue

            with jax.profiler.TraceAnnotation("convert_to_numpy"):
                result_tokens = np.array(result_tokens)
                log_prob = np.array(log_prob)

            # Process generated tokens
            if is_first_token:
                should_terminate = self.emit_token(
                    row_id, int(result_tokens), int(log_prob)
                )
                if should_terminate:
                    newly_empty.append(slot)
            else:

                for decode_step in range(self.min_decode_steps):

                    for slot, id_ in self.slot_to_id.items():
                        if id_ is None:
                            continue
                        with jax.profiler.TraceAnnotation("log_prob_slot"):
                            log_prob_slot = log_prob[decode_step][slot]
                            result_tokens_slot = result_tokens[decode_step][slot]
                        should_terminate = self.emit_token(
                            id_, int(result_tokens_slot), int(log_prob_slot)
                        )
                        if should_terminate:
                            newly_empty.append(slot)

            # Update decode slots
            for slot in newly_empty:
                self.counter.detokenize += 1
                self.slot_to_id[slot] = None
                self.empty_decode_slots.append(slot)
    
    def emit_token(
        self, prompt_id, result_token: int, log_prob, slot: int = None
    ):
        """Adds the token to the results for the specified prompt ID and
        determines if generation should terminate.

        Args:
            prompt_id: ID of the prompt
            token: Token to emit

        Returns:
            True if this token signals the end of generation, False otherwise
        """
        # Return if already reached max decode length
        if len(self.res[prompt_id]) == self.max_decode_length:
            return True

        # Return if already reached eos
        if (
            len(self.res[prompt_id]) > 0
            and self.res[prompt_id][-1].token in self.eos_ids
        ):
            return True

        index = len(self.res[prompt_id])

        self.res[prompt_id].append(TokenOutput(result_token, log_prob))

        return (result_token in self.eos_ids) or (index + 1 == self.max_decode_length)


class OfflineEngine:
    """Class for handling offline inference on batches of inputs."""

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
        warm_up=False,
        rng=None,
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
            eos_ids: List of EOS token IDs for checking sequence completion.
              If None, the tokenizer's EOS token will be used.
            tokenizer: Tokenizer instance for encoding/decoding text. If None,
              will be created using the config if eos_ids is not provided.
            prefill_lengths: List of expected prefill lengths, or "auto" to
                automatically determine appropriate lengths from the engine
                config. Input sequences will be padded to the nearest length
                in this list.
            batch_prefill_max_batch_size: Maximum number of inputs to pack
              into a single prefill. This is only used when enable_batch_prefill
              is True.
            warm_up: Whether to precompile prefill and decode functions for
              each length in the prefill_lengths list during initialization.
              Alternatively compilation will be done during runtime, or through
              directly calling the warm_up() function.
            dp: Data parallelism, number of replicas of the model to run. This
              helps to increase throughput by running multiple inference replicas
              in parallel. When setting dp>1, Pathways must be used. Either provide
              the dp_meshes for each model replica, or let OfflineEngine automatically
              create the meshes which will make use of all visible devices.
            dp_meshes: List of JAX Mesh objects for each model replica. Use this
              option if you want to use only some of the devices for OfflineEngine and
              reserve the rest for other tasks. If None, OfflineEngine will create the meshes
              automatically.
            auto_layout_supported: Whether auto layout is supported. Auto layout introduces
                some start up overhead but can result in faster step times. TODO: Pathways
                has some bugs with auto layout, so it is recommended to set this to False
                for now when using Pathways.
            rng: Random number generator key. If None, a new key will be created.
        """
        max_logging.log("Initializing OfflineEngine")
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
        self.should_warm_up = warm_up
        self._not_warmed_up = True
        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng
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
            ici_parallelism = max_utils.fill_unspecified_mesh_axes(
                config.ici_parallelism.copy(), len(devices), "ICI"
            )
            devices_array = mesh_utils.create_device_mesh(
                ici_parallelism,
                devices,
                contiguous_submeshes=False,
                allow_split_physical_axes=config.allow_split_physical_axes or False,
            )
            flat_devices = devices_array.flatten()
            inference_devices = flat_devices.reshape((self.dp, len(devices) // self.dp))
            self.dp_meshes = [
                Mesh(devices.reshape(config.ici_parallelism), config.mesh_axes)
                for devices in inference_devices
            ]

        # Initialize ReplicaWorkers
        run_as_a_thread = (
            self.dp > 1
        )  # No need to run worker as a thread if there is only one replica
        replica_rngs = jax.random.split(self.rng, self.dp)
        assert replica_rngs[0].shape == self.rng.shape
        self.replica_workers = []
        for i in range(self.dp):
            worker = ReplicaWorker(
                config=self.config,
                params=self.params,
                min_decode_steps=self.min_decode_steps,
                enable_batch_prefill=self.enable_batch_prefill,
                mesh = self.dp_meshes[i],
                devices=np.squeeze(self.dp_meshes[i].devices),
                tokenizer=self.tokenizer,
                eos_ids=self.eos_ids,
                prefill_lengths=self.prefill_lengths,
                max_decode_length=self.max_decode_length,
                batch_prefill_max_batch_size=self.batch_prefill_max_batch_size,
                worker_id=i,
                auto_layout_supported=auto_layout_supported,
                run_as_a_thread=run_as_a_thread,
                rng=replica_rngs[i],
            )
            if i == 0:
                worker.ensure_init_finished()
            self.replica_workers.append(worker)
            

        for replica in self.replica_workers:
            replica.ensure_init_finished()

        max_logging.log(f"Created {self.dp} replica workers")

        self.tokenizer = self.replica_workers[0].tokenizer
        if self.should_warm_up:
            self.warm_up()

    def warm_up(self):
        if self._not_warmed_up:
            """Warm up all replica workers."""
            for i in range(self.dp):
                self.replica_workers[i].warm_up()
            for i in range(self.dp):
                self.replica_workers[i].ensure_warm_up_finished()
            self._not_warmed_up = False

    def update_params(self, params: Params):
        for i in range(self.dp):
            self.replica_workers[i].update_params(params)

    def batch_inference(
        self,
        data: Union[List[InputData], List[jax.Array], List[np.ndarray]],
        desc: str = "",
    ) -> List[CompletionOutput]:
        """Run inference on a batch of inputs.

        Args:
            data: List of InputData objects, or JAX or numpy arrays.
                If input is JAX or numpy array, it must not contain padding tokens.
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
            self.replica_workers[i].ensure_inference_finished()
            max_logging.log(f"replica worker {i} stopped")

        # Return CompletionOutput objects
        completion_outputs = []
        with jax.profiler.TraceAnnotation("return final output"):
            for input_data in data:
                completion_outputs.append(
                    CompletionOutput(
                        index=input_data.id,
                        token_ids=np.array(
                            [token_output.token for token_output in results[input_data.id]]
                        ),
                        logprobs=np.array(
                            [token_output.log_prob for token_output in results[input_data.id]]
                        ),
                    )
                )
        return completion_outputs

    def pad_data(self, data: List[InputData]) -> List[InputData]:
        """For each input, pad it to the next length in self.prefill_lengths
        that is greater than or equal to its true length.

        Args:
            data: List of InputData objects

        Returns:
            List of padded InputData objects
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
                if isinstance(item.tokens, jax.Array):
                    padded_tokens = jax.numpy.zeros(
                        target_length, dtype=item.tokens.dtype
                    )
                    padded_tokens = padded_tokens.at[: item.true_length].set(
                        item.tokens[: item.true_length]
                    )
                else:
                    padded_tokens = np.zeros(target_length, dtype=item.tokens.dtype)
                    padded_tokens[: item.true_length] = item.tokens[: item.true_length]
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
        if isinstance(data[0], jax.Array) or isinstance(data[0], np.ndarray):
            data = [
                InputData(id=i, tokens=array, true_length=len(array))
                for i, array in enumerate(data)
            ]

        # Make sure all data id is unique
        if len(data) != len(set([item.id for item in data])):
            raise ValueError("All data ids must be unique")

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
            import pathwaysutils

            pathwaysutils.initialize()

        if self.enable_batch_prefill and not self.auto_layout_supported:
            raise ValueError(
                "auto_layout_supported must be True if enable_batch_prefill is True"
            )

        if self.config.scan_layers:
            max_logging.log(
                "WARNING: scan_layers=True will result in slow step time. It is recommended for debugging purposes only."
            )
