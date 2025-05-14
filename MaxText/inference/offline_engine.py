"""
MaxText Offline Inference Engine

Features
- Continous batching
- Prefill packing
- Single and multihost with TP
- DP with Pathways

Example usage:

    offline_engine = OfflineEngine(
        config=maxtext_config, params=None, enable_batch_prefill=True, dp = 1, dp_meshes = None
    )

    input_data = [jax.numpy.arange(80),
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

import pathwaysutils
from typing import Any, List, Tuple, Callable, Optional, Dict, Union
from collections import defaultdict
import dataclasses
import functools
import logging
import os
import queue
import signal
import threading
import traceback
import jax
from jetstream.engine import engine_api
from enum import Enum

# pylint: disable=no-name-in-module
from MaxText.maxengine import MaxEngine
from MaxText.prefill_packing import PrefillProcessor, BatchedPrefillProcessor
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import numpy as np
import jax.numpy as jnp

DecodeState = Any
Params = Any

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
    """Class for returned output

    Attributes:
    index: The index of the output in the request.
    token_ids: The token IDs of the generated output text.
    logprobs: The log probabilities of the output tokens
    """

    index: int
    token_ids: jax.Array  # globally replicated or not?
    logprobs: jax.Array


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


class JetThread(threading.Thread):
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
        """
        self._type = type
        self.engine = engine
        self.prefill_lengths = prefill_lengths
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
            raise ValueError(f"Invalid type: {type}")

    def aot_compile(
        self,
        params: Params,
        decode_state: DecodeState,
    ) -> None:
        max_length = self.prefill_lengths[-1]
        if self._type == PrefillType.DEFAULT:
            for length in self.prefill_lengths:
                print(f"AOT compiling prefill for length: {length}")
                if self.auto_layout_supported:
                    self._processor.aot_compile(params, length)
                else:
                    tokens = jnp.zeros((length,), dtype=jnp.int32)
                    first_token, decode_state = self._processor._process(
                        params, tokens, 0, length, decode_state
                    )
        elif self._type == PrefillType.BATCH:
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
                        slots = jnp.arange(0, self.max_batch_size, dtype=int)
                        decoder_positions = jnp.arange(0, max_length, dtype=int)
                        decoder_segment_ids = jnp.ones(max_length, dtype=int)
                        start_pos = jnp.arange(
                            0, max_length, max_length // self.max_batch_size, dtype=int
                        )
                        true_lengths = jnp.full(
                            self.max_batch_size, padded_length, dtype=int
                        )

                        first_tokens, decode_state = (
                            self._batch_processor._process_batch(
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
                        )
            # for fallback
            print(f"AOT compiling prefill for length: {max_length}")
            if self.auto_layout_supported:
                self._processor.aot_compile(params, max_length)
            else:
                tokens = jnp.zeros((max_length,), dtype=jnp.int32)
                first_token, decode_state = self._processor._process(
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
        max_length: int,
        prefill_done: Callable[
            [List[Tuple[engine_api.ResultTokens, int]], List[int], DecodeState], None
        ],
    ) -> None:
        """Process an input through the appropriate prefill processor.

        This method routes the input to either the default processor or the
        batch processor based on configuration and input characteristics.

        Args:
            model_params: Model parameters for inference
            decode_state: Current decoder state
            decode_slot: The decode slot index to use for this input
            input_id: Unique identifier for this input
            input_tokens_padded: Padded token array for the input
            input_true_length: Actual length of the input before padding
            max_length: Maximum prefill length allowed
            prefill_done: Callback function called when prefill completes
        """
        padded_length = len(input_tokens_padded)
        # Use default processor if configured or if input is already at max length
        if self._type == PrefillType.DEFAULT or padded_length == max_length:
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
                max_length,
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
    def __init__(
        self,
        config,
        params,
        min_decode_steps,
        enable_batch_prefill,
        devices,
        tokenizer,
        eos_ids,
        prefill_lengths,
        batch_prefill_max_batch_size,
        worker_id=0,
        auto_layout_supported=False,
        run_as_a_thread=False,
    ):
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
        self.max_decode_length = self.config.max_target_length - self.max_prefill_length
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
        self.detokenize_backlog = queue.Queue(maxsize=10)
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
        self.running = False

    def _init_engine(self, params):
        """Initialize the MaxEngine."""
        engine = MaxEngine(self.config, self.devices)
        params = engine.load_params(params)
        return params, engine

    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        if self.eos_ids is None and self.tokenizer is None:
            tokenizer_params = self.engine.get_tokenizer()
            self.tokenizer = self.engine.build_tokenizer(tokenizer_params)
        if self.eos_ids is None:
            self.eos_ids = [self.tokenizer.eos_id]
        return self.tokenizer

    def warm_up_impl(self):
        self.decode_state = self.prefill_helper.aot_compile(
            self.params, self.decode_state
        )
        assert self.decode_state is not None

        for _ in range(3):
            print("warm up generate fn")
            self.decode_state, result_tokens = self.generate_fn(
                self.params, self.decode_state, None
            )

    def warm_up(
        self,
    ):
        if self.run_as_a_thread:
            self.warm_up_thread = JetThread(
                target=self.warm_up_impl,
                name=f"replica_worker_{self.worker_id}",
            )
            self.warm_up_thread.start()
        else:
            self.warm_up_impl()

    def finish_warm_up(self):
        if self.run_as_a_thread:
            self.warm_up_thread.join()

    def start_inference(
        self,
        data_queue: queue.Queue,
        results,
        desc: str = "",
    ):
        self.res = results
        if self.run_as_a_thread:
            self.worker_thread = JetThread(
                target=self._start_inference,
                args=(data_queue, desc),
                name=f"replica_worker_{self.worker_id}",
            )
            self.worker_thread.start()
        else:
            self._start_inference(data_queue, desc)

    def stop(self):
        if self.run_as_a_thread:
            self.worker_thread.join()

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
        print("start inference")
        # Reset state
        self.counter = EventCounter(input=0, prefill=0, decode=0, detokenize=0)
        self.empty_decode_slots = list(range(self.batch_size))
        self.slot_to_id = {}

        # Start detokenization thread
        detokenize_thread = JetThread(
            target=functools.partial(
                self.detokenize,
            ),
            name="detokenize",
        )
        self.running = True
        detokenize_thread.start()

        # Process each input
        while not data_queue.empty():
            row = data_queue.get_nowait()

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
                max_length=self.max_prefill_length,
                prefill_done=self.prefill_done,
            )

        # 4. Flush any pending inputs in batch prefill mode
        self.prefill_helper.finalize(self.params, self.decode_state, self.prefill_done)

        # 5. Continue decoding until all sequences are complete
        while self.slot_to_id:
            self.decode()

        # Wait for detokenization to complete
        self.running = False
        detokenize_thread.join()

        # Log completion statistics
        log.info(
            "summary-%s-prefills-%d-decodes-%d-detokens-%d completed.",
            desc,
            self.counter.prefill,
            self.counter.decode,
            self.counter.detokenize,
        )
        return self.res

    def emit_token(self, prompt_id, token):
        """Adds the token to the results for the specified prompt ID and
        determines if generation should terminate.

        Args:
            prompt_id: ID of the prompt
            token: Token to emit

        Returns:
            True if this token signals the end of generation, False otherwise
        """
        # print("emit_token", prompt_id, token)
        already_reached_eos = (
            len(self.res[prompt_id]) > 0
            and self.res[prompt_id][-1] == self.tokenizer.eos_id
        )

        if not already_reached_eos:
            self.res[prompt_id].append(token)

        return token == self.tokenizer.eos_id

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
            self.detokenize_backlog.put(
                (first_token, True, prompt_ids[i], slot), block=True
            )

    def decode(self):
        """Run decode steps on current decoder state.

        Performs `self.min_decode_steps` decode operations
        and puts results in the detokenization queue.
        """
        for i in range(self.min_decode_steps):
            # Generate next tokens
            self.decode_state, result_tokens = self.generate_fn(
                self.params, self.decode_state, None
            )
            # print("putting result_tokens to detokenize_backlog")
            # Add results to detokenization queue
            self.detokenize_backlog.put(
                (result_tokens.convert_to_numpy(), False, 0, 0), block=True
            )

        self.counter.decode += 1

    def detokenize(self):
        """Detokenize results and manage decode slots.

        Runs in a background thread to process tokens from
        the detokenization queue, emit tokens, and free up
        decode slots when sequences complete.
        """
        print("detokenize start")
        # while self.counter.detokenize < self.counter.input:
        while self.running or not self.detokenize_backlog.empty():
            newly_empty = []

            # Get next item from queue with timeout
            result_tokens, is_first_token, row_id, slot = self.detokenize_backlog.get()
            # print("got result_tokens from detokenize_backlog")

            # Process generated tokens
            if is_first_token:
                first_token, is_valid, length = result_tokens.data[0]
                should_terminate = self.emit_token(row_id, first_token)

                # Free up slot if terminated
                if should_terminate:
                    newly_empty.append(slot)
            else:
                for slot, id_ in self.slot_to_id.items():
                    token, is_valid, length = result_tokens.data[slot]
                    should_terminate = False
                    if is_valid:
                        should_terminate = self.emit_token(id_, token.item())

                    # Free up slot if terminated or max length reached
                    if should_terminate or length >= self.max_decode_length:
                        newly_empty.append(slot)

            # Update decode slots
            for slot in newly_empty:
                self.counter.detokenize += 1
                del self.slot_to_id[slot]
                self.empty_decode_slots.append(slot)


class OfflineEngine:
    """Class for handling offline inference on batches of inputs.

    The logic is the following:
    1. Process input one at a time
    2. Pad the input data to the nearest power of 2
    3. Prefill input and insert kv cache
    4. Keep on processing inputs and prefilling until
    there are enough samples to do batch decoding
    5. Decode until at least one of the samples is finished,
    so there is room for new samples
    6. Prefill to fill up the newly emptied decode slots
    7. Repeat step 5 and 6 until all samples are finished
    8. A background thread is used to detokenize the results
    9. Return the results

    Prefill Packing:
        When enable_batch_prefill is True, the prefill processor
        will pack multiple inputs into a single sequence before
        doing the prefill.

        There are multiple buckets for packed sequences, where each bucket
        contains inputs with same padded length. Only inputs with same
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
        """
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
        # self.warm_up = warm_up
        self.max_prefill_length = self.config.max_prefill_predict_length
        self.max_decode_length = self.config.max_target_length - self.max_prefill_length
        self.auto_layout_supported = auto_layout_supported
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
            # print(f"mesh_shape: {mesh_shape}")
            # TODO: find the optimal devices for this mesh
            for i in range(self.dp):
                mesh_devices = np.array(
                    devices[
                        i * num_devices_per_replica : (i + 1) * num_devices_per_replica
                    ]
                ).reshape(mesh_shape)
                print(f"mesh_devices: {mesh_devices.shape}")
                # optimized_mesh_devices = mesh_utils.create_device_mesh(mesh_shape, mesh_devices)
                self.dp_meshes.append(Mesh(mesh_devices, self.config.mesh_axes))

        # Initialize ReplicaWorkers
        run_as_a_thread = self.dp > 1  # No need to run as a thread if dp=1
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
        print("created replica workers")
        self.tokenizer = self.replica_workers[0].tokenizer

    def warm_up(self):
        for i in range(self.dp):
            self.replica_workers[i].warm_up()
        for i in range(self.dp):
            self.replica_workers[i].finish_warm_up()

    def batch_inference(
        self,
        data: Union[List[InputData], List[jax.Array]],
        data_is_padded: bool = False,
        desc: str = "",
    ) -> Dict[str, List[int]]:
        """Run inference on a batch of inputs.

        Args:
            data: List of InputData objects containing input sequences
            desc: Description string for logging

        Returns:
            Dictionary mapping input ids to output token sequences
        """
        # Prepare input data (pad, and sort if using batch prefill)
        data_is_jax_array = isinstance(data[0], jax.Array)
        print("batch inference start")
        data = self.prepare_data(data, data_is_padded)
        print("prepared data")

        # Thread-safe
        data_queue = queue.Queue()
        results = defaultdict(list)

        for row in data:
            data_queue.put(row)
        for i in range(self.dp):
            self.replica_workers[i].start_inference(data_queue, results)
        for i in range(self.dp):
            self.replica_workers[i].stop()

        if data_is_jax_array:
            res = [results[input_data.id] for input_data in data]
            return res

        return results

    def sort_data(self, data: List[InputData]) -> List[InputData]:
        """Sort input data by padded length. This helps with batch prefilling by
        filling up the buckets quickly.
        """
        data_dict = defaultdict(list)
        padded_lengths = []

        # Group inputs by padded length
        for row in data:
            padded_lengths.append(row.tokens.shape[0])
            data_dict[row.tokens.shape[0]].append(row)

        data = []
        for padded_len in sorted(set(padded_lengths)):
            log.info("padded len: %d, num: %d", padded_len, len(data_dict[padded_len]))
            data += data_dict[padded_len]

        return data

    def pad_data(self, data: List[InputData]) -> List[InputData]:
        """For each input, pad it to the next length in self.prefill_lengths
        that is greater than or equal to its true length.
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

    def prepare_data(
        self, data: List[InputData], data_is_padded: bool = True
    ) -> List[InputData]:
        """Prepare input data for inference by padding and optionally sorting."""
        if isinstance(data[0], jax.Array):
            data = [
                InputData(id=i, tokens=array, true_length=len(array))
                for i, array in enumerate(data)
            ]

        if not data_is_padded:
            data = self.pad_data(data)

        # Sort data by length if using batch prefill
        if self.enable_batch_prefill:
            # Sort data by length when doing batch prefilling so
            # buckets fill up quickly
            return self.sort_data(data)

        return data

    def _validate_config(self):
        if self.enable_batch_prefill and self.engine.config.scan_layers:
            raise ValueError(
                "scan_layers must be False if enable_batch_prefill is True"
            )
        if self.max_decode_length <= 0:
            raise ValueError(
                "Make sure max_target_length - max_prefill_predict_length is greater than 0"
            )
        if self.dp > 1:
            # Initialize Pathways if not already initialized
            pathwaysutils.initialize()
