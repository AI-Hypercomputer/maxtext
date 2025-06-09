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
    - auto_layout_supported is not supported on Pathways.
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
from jax.sharding import PartitionSpec
from jax.experimental import mesh_utils


from jetstream.engine import engine_api
from MaxText.maxengine import MaxEngine
from MaxText import max_utils
from MaxText import inference_utils
from jax.sharding import PartitionSpec as P, NamedSharding

from MaxText.experimental.rl import pathwaysutils_reshard


from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor
from MaxText import max_logging

DecodeState = Any
Params = Any
MaxTextConfig = Any
DEBUG = os.environ.get("DEBUG", "0") == "1"
# Configure logging
log = logging.getLogger(__name__)

# logging.getLogger("jax").setLevel(logging.DEBUG)


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

@dataclasses.dataclass
class PrefillResult:
    """Class for storing log probabilities."""

    token: jax.Array
    log_prob: jax.Array
    slot: int
    prompt_logp: jax.Array


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
                engine=engine, max_batch_size=batch_prefill_max_batch_size, auto_layout_supported=auto_layout_supported
            )
            # Also create a standard processor for fallback cases
            self._processor = PrefillProcessor(engine)
        else:
            raise ValueError(f"Invalid prefill type: {type}")

    ## AOT compile logic is yet to be tested.###
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
    @functools.partial(jax.jit, static_argnums=(6, 7), donate_argnums=(4,))
    def _jitted_process(
        params,
        tokens,
        slot,
        true_length,
        decode_state,
        rng,
        processor_fn,
        auto_layout_supported=False,
    ):
        if auto_layout_supported:
            first_token, decode_state = processor_fn(
                params,
                decode_state,
                slot,
                tokens,
                true_length,
                rng,
                return_prompt_logp=True,
            )
        else:
            first_token, decode_state = processor_fn(
                params,
                tokens,
                slot,
                true_length,
                decode_state,
                rng,
                return_prompt_logp=True,
            )
        log_prob = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"][slot], decode_state["tokens"][slot]
        )

        # Shapes: (1,), (1,)
        return first_token.data[:, 0], log_prob, decode_state, decode_state["prompt_logp"]

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
            prefill_fn = (
                self._processor.process
                if self.auto_layout_supported
                else self._processor._process
            )
            first_token, log_prob, decode_state, prompt_logp = self._jitted_process(
                model_params,
                input_tokens_padded,
                decode_slot,
                input_true_length,
                decode_state,
                self.rng,
                prefill_fn,
                self.auto_layout_supported,
            )
            prefill_done(
                [PrefillResult(first_token, log_prob, decode_slot, prompt_logp)], [input_id], decode_state
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
            self._batch_processor.flush(model_params, decode_state, prefill_done, return_prompt_logp=True)


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
        is_pw_reshard: bool = True,
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
        self.is_pw_reshard = is_pw_reshard
        self.enable_batch_prefill = enable_batch_prefill
        if self.enable_batch_prefill:
            self.prefill_type = PrefillType.BATCH
        else:
            self.prefill_type = PrefillType.DEFAULT
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
        self.init_thread = None
        self.running = False

        # State management
        self.detokenize_backlog = queue.Queue()
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

        start_time = time.time()
        # Initialize MaxEngine(s)
        self.params, self.engine = self._init_engine(self.params)
        self.tokenizer = self._init_tokenizer()
        self.decode_batch_size = self.engine.max_concurrent_decodes

        # Create a Prefill Helper
        self.prefill_helper = PrefillHelper(
            self.prefill_type,
            self.engine,
            self.prefill_lengths,
            self.batch_prefill_max_batch_size,
            self.auto_layout_supported,
            rng=self.rng,
        )

        # Initialize decode state and generate function
        start_time_decode_state = time.time()
        if self.auto_layout_supported:
            self.generate_fn, self.params, init_decode_state_fn = (
                self.engine.aot_compile(self.params, pass_rng_shape=True)
            )
            self.decode_state = init_decode_state_fn(self.rng)
        else:
            self.generate_fn = self.engine.generate
            self.decode_state = self.engine.init_decode_state(self.rng)
            
        max_logging.log(
            f"time taken to initialize decode_state: {time.time() - start_time_decode_state} seconds"
        )
        max_logging.log(f"Initialized replica worker {self.worker_id} in {time.time() - start_time} seconds")

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
            f"Time taken to initialize engine: {end_time - start_time} seconds"
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

    def update_params(
        self,
        params: Params,
        destination_sharding: jax.sharding.NamedSharding,
        is_pw_reshard: bool = True,
    ):
        self.ensure_init_finished()
        if self.run_as_a_thread:
            self.update_thread = SafeThead(
                target=self.update_impl,
                args=(params, destination_sharding, is_pw_reshard),
                name=f"replica_worker_{self.worker_id}",
            )
            self.update_thread.start()
        else:
            self.update_impl(params, destination_sharding, is_pw_reshard)

    def ensure_update_finished(self):
        if self.update_thread is not None:
            self.update_thread.join()
            self.update_thread = None

    def update_impl(
        self,
        params: Params,
        destination_sharding: jax.sharding.NamedSharding,
        is_pw_reshard: bool,
    ):
        if is_pw_reshard:
            with (
                jax.transfer_guard_device_to_host("disallow_explicit"),
                jax.transfer_guard_host_to_device("disallow_explicit"),
            ):
                self.params = pathwaysutils_reshard.reshard(
                    params, destination_sharding, cache_resharding_plans=True
                )
        else:
            self.params = jax.device_put(params, destination_sharding)

    def start_inference(
        self,
        data_queue: queue.Queue,
        desc: str = "",
    ):
        """Start the inference process.

        Args:
            data_queue: Queue containing input data
            results: Dictionary to store results
            desc: Description for logging
        """
        self.ensure_init_finished()
        # self.ensure_update_finished() TODO(mohitkhatwani) might not need it
        self.res = defaultdict(list)
        self.res_prompt_logp = defaultdict(list)
        self.res_prompt_tokens = defaultdict(list)
        self.completion_outputs = []

        if self.run_as_a_thread:
            self.worker_thread = SafeThead(
                target=self._start_inference,
                args=(data_queue, desc),
                name=f"replica_worker_{self.worker_id}",
            )
            self.worker_thread.start()
        else:
            self._start_inference(data_queue, desc)

    def ensure_inference_finished_and_return_results(self):
        """Stop the inference process and wait for completion."""
        if self.worker_thread is not None:
            self.worker_thread.join()
            self.worker_thread = None
        return self.completion_outputs
    
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
        max_logging.log(f"Replica {self.worker_id}. Starting inference")
        # Reset state
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
                row = data_queue.get()
            except queue.Empty:
                break

            # 1. Wait for an empty slot
            while not self.empty_decode_slots:
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
            self.res_prompt_tokens[row.id] = row.tokens[:row.true_length]

        # 4. Flush any pending inputs in batch prefill mode
        self.prefill_helper.finalize(self.params, self.decode_state, self.prefill_done)

        # 5. Continue decoding until all sequences are complete
        while not all(value is None for value in self.slot_to_id.values()):
            self.decode()

        # Wait for detokenization to complete
        self.running = False
        max_logging.log(
            f"Replica worker {self.worker_id}: joining detokenize thread. There are {self.detokenize_backlog.qsize()} elements in the backlog"
        )
        start_time = time.time()
        with jax.profiler.TraceAnnotation("Flushing detokenization thread"):
            detokenize_thread.join()
        end_time = time.time()
        max_logging.log(f"Replica worker {self.worker_id}: detokenize thread joined in {end_time - start_time} seconds")
        
        with jax.profiler.TraceAnnotation("offline_engine.batch_inference.return_final_output"):
            for input_id in self.res.keys():
                self.completion_outputs.append(
                    CompletionOutput(
                        index=input_id,
                        token_ids=np.concatenate((self.res_prompt_tokens[input_id].squeeze(), 
                            np.array([
                                token_output.token
                                for token_output in self.res[input_id]
                            ]).squeeze()
                        )),
                        logprobs=np.concatenate((
                            self.res_prompt_logp[input_id].squeeze(),
                            np.array([
                                token_output.log_prob
                                for token_output in self.res[input_id]
                            ]).squeeze()
                        ))
                    )
                )

    @staticmethod
    @jax.jit
    def _jitted_log_prob_and_slice_token(token, decode_state, slot):
        
        log_prob = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"][slot], decode_state["tokens"][slot]
        )
        return token.data[:, 0], log_prob

    def prefill_done(self, prefill_result, prompt_ids, decode_state):
        """Callback function called when prefill completes.
        This function adds the prefill tokens to the detokenization queue,
        which manages the token emission and decode slot evictions.

        Args:
            prefill_result: List of (token, slot) tuples
            prompt_ids: List of prompt IDs
            decode_state: Updated decode state
        """
        max_logging.log("Replica {}. Prefill done".format(self.worker_id))
        # Update decode state
        self.decode_state = decode_state

        # Process each prefill result
        for i, prefill_result in enumerate(prefill_result):
            
            first_token = prefill_result.token
            log_prob = prefill_result.log_prob
            slot = prefill_result.slot
            prompt_logp = prefill_result.prompt_logp
            
            #TODO: clean up this code logic.
            if log_prob is None:
                first_token, log_prob = self._jitted_log_prob_and_slice_token(first_token, self.decode_state, slot)
            
            self.slot_to_id[slot] = prompt_ids[i]
            
            # Add token to detokenization queue
            start_time = time.time()
            with jax.profiler.TraceAnnotation("convert_to_numpy"):
                first_token = np.array(first_token)
                log_prob = np.array(log_prob)
                prompt_logp = np.array(prompt_logp)
            end_time = time.time()
            max_logging.log(f"Replica worker {self.worker_id}: convert to numpy in Prefill in {end_time - start_time} seconds")
            self.detokenize_backlog.put_nowait(
                (first_token, log_prob, True, prompt_ids[i], slot, prompt_logp)
            )

    def decode(self):
        """Run decode steps on current decoder state.

        Performs `self.min_decode_steps` decode operations
        and puts results in the detokenization queue.
        """

        buffer = []
        for i in range(self.min_decode_steps):
            # Generate next tokens
            self.decode_state, result_tokens, log_prob = self._jitted_generate_fn(
                self.params, self.decode_state, self.rng
            )
            # Add token to detokenization queue
            start_time = time.time()
            with jax.profiler.TraceAnnotation("convert_to_numpy"):
                result_tokens = np.array(result_tokens)
                log_prob = np.array(log_prob)
            end_time = time.time()
            max_logging.log(f"Replica worker {self.worker_id}: convert to numpy in Decode in {end_time - start_time} seconds")

            buffer.append((result_tokens, log_prob))

        # Add results to detokenization queue
        self.detokenize_backlog.put_nowait(
            (
                [result_token for result_token, _ in buffer],
                [log_prob for _, log_prob in buffer],
                False,
                0,
                0,
                None,
            )
        )

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
    def _jitted_generate_fn(self, params, decode_state, rng):
        decode_state, result_tokens = self.engine.generate(
            params, decode_state, rng=rng
        )
        logps = inference_utils.log_prob_of_chosen_token(
            decode_state["logits"], decode_state["tokens"]
        )
        return decode_state, result_tokens.data[:, 0], logps

    def detokenize(self):
        """Detokenize results and manage decode slots.

        Runs in a background thread to process tokens from
        the detokenization queue, emit tokens, and free up
        decode slots when sequences complete.
        """
        max_logging.log(f"Replica worker {self.worker_id}: starting detokenization thread")
        while self.running or not self.detokenize_backlog.empty():
            newly_empty = []

            # Get next item from queue with timeout
            try:
                result_tokens, log_prob, is_first_token, row_id, slot, prompt_logp = (
                    self.detokenize_backlog.get(timeout=0.01)
                )
            except queue.Empty:
                if not self.running:
                    break
                continue
            
            # Process generated tokens
            start_time = time.time()
            if is_first_token:
                should_terminate = self.emit_token(
                    row_id, int(result_tokens), log_prob, prompt_logp=prompt_logp
                )
                if should_terminate:
                    newly_empty.append(slot)
            else:
                for decode_step in range(self.min_decode_steps):
                    for slot, id_ in self.slot_to_id.items():
                        if id_ is None:
                            continue
                        log_prob_at_slot = log_prob[decode_step][slot]
                        result_tokens_at_slot = result_tokens[decode_step][slot]
                        should_terminate = self.emit_token(
                            id_, int(result_tokens_at_slot), log_prob_at_slot
                        )
                        if should_terminate:
                            newly_empty.append(slot)

            # Update decode slots
            for slot in newly_empty:
                self.slot_to_id[slot] = None
                self.empty_decode_slots.append(slot)
            end_time = time.time()
            max_logging.log(f"replica worker {self.worker_id}: detokenize in {end_time - start_time} seconds")

    def emit_token(self, prompt_id, result_token: int, log_prob, prompt_logp = None, ):
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
        if prompt_logp is not None:
            self.res_prompt_logp[prompt_id] = prompt_logp
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
            self.dp_meshes = OfflineEngine.create_dp_meshes(devices, self.dp, self.config)

        # Initialize ReplicaWorkers
        run_as_a_thread = (
            self.dp > 1
        )  # No need to run worker as a thread if there is only one replica
        replica_rngs = jax.random.split(self.rng, self.dp)
        self.replica_workers: List[ReplicaWorker] = []
        for i in range(self.dp):
            worker = ReplicaWorker(
                config=self.config,
                params=self.params,
                min_decode_steps=self.min_decode_steps,
                enable_batch_prefill=self.enable_batch_prefill,
                mesh = self.dp_meshes[i],
                devices=self.dp_meshes[i].devices.flatten(),
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
            self.replica_workers.append(worker)

        # Skipping rebuilding the tokenizer
        self.tokenizer = self.replica_workers[0].tokenizer

    def update_params(
        self, params: Params, parition_spec: PartitionSpec, is_pw_reshard
    ):
        for i in range(self.dp):
            self.replica_workers[i].update_params(
                params,
                jax.tree_util.tree_map(
                    lambda ps: jax.sharding.NamedSharding(self.dp_meshes[i], ps),
                    parition_spec,
                ),
                is_pw_reshard,
            )

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

        # Add all data to the queue
        for row in data:
            data_queue.put_nowait(row)

        # Start inference on all replica workers
        completion_outputs = {}
        for i in range(self.dp):
            self.replica_workers[i].start_inference(data_queue, desc)
        
        # Wait for all workers to complete
        for i in range(self.dp):
            replica_completion_outputs = self.replica_workers[i].ensure_inference_finished_and_return_results()
            completion_outputs.update({replica_completion_outputs[i].index: replica_completion_outputs[i] for i in range(len(replica_completion_outputs))})
            max_logging.log(f"replica worker {i} stopped")
        
        sorted_completion_outputs = []
        for input_data in data:
            sorted_completion_outputs.append(completion_outputs[input_data.id])
        return sorted_completion_outputs

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

    @staticmethod
    def create_dp_meshes(devices, dp, config):
        """Create data parallelism meshes for each replica worker."""
        ici_parallelism = max_utils.fill_unspecified_mesh_axes(
            config.ici_parallelism.copy(), len(devices), "ICI"
        )
        devices_array = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=config.allow_split_physical_axes or False,
        )
        inference_devices = devices_array.reshape(
            (dp, -1)
        )
        dp_meshes = [
            Mesh(devices.reshape(ici_parallelism), config.mesh_axes)
            for devices in inference_devices
        ]
        return dp_meshes


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

        if self.config.scan_layers:
            max_logging.log(
                "WARNING: scan_layers=True will result in slow step time. It is recommended for debugging purposes only."
            )