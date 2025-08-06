import os
import time
import uuid
from typing import Sequence, Optional, List

import jax
import jax.numpy as jnp
from absl import app

from MaxText import max_utils, maxengine, pyconfig, multimodal_utils

# Set TF log level to avoid verbose startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


class MaxTextGenerator:
    """A reusable class for parallel text generation using MaxText."""

    def __init__(self, argv: Sequence[str]):
        """
        Initializes the MaxText model, tokenizer, and engine.

        Args:
            argv: Command-line arguments for MaxText configuration.
        """
        print("Initializing MaxTextGenerator...")
        start_time = time.time()
        jax.config.update("jax_default_prng_impl", "unsafe_rbg")
        self.config = pyconfig.initialize(argv)
        self._validate_config(self.config)
        max_utils.print_system_information()

        self.engine = maxengine.MaxEngine(self.config)
        self.rng = jax.random.PRNGKey(1234)

        # Load parameters
        self.rng, rng_load_params = jax.random.split(self.rng)
        self.params = self.engine.load_params(rng=rng_load_params)

        # Build tokenizer
        self.metadata = self.engine.get_tokenizer()
        self.tokenizer = self.engine.build_tokenizer(self.metadata)
        self.eos_id = self.tokenizer.eos_id
        try:
            self.has_chat_template = getattr(self.tokenizer.tokenizer, "chat_template", False)
        except AttributeError:
            self.has_chat_template = False

        self.batch_size = int(self.config.per_device_batch_size * jax.device_count())
        end_time = time.time()
        print(f"Initialization complete in {end_time - start_time:.2f} seconds. Max batch size: {self.batch_size}")

    def generate_batch(
        self,
        prompts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        max_target_length: Optional[int] = None,
        max_prefill_length: Optional[int] = None,
    ) -> List[str]:
        """
        Generates text for a batch of prompts, handling chunking automatically.

        Args:
            prompts: A list of prompt strings.
            image_paths: An optional list of image paths, one for each prompt.
            max_target_length: The maximum length of the generated sequence.
            max_prefill_length: The maximum length of the prefill.

        Returns:
            A list of generated text strings, corresponding to the input prompts.
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

            chunk_count = (i // self.batch_size) + 1
            total_chunks = (num_prompts + self.batch_size - 1) // self.batch_size
            print(f"\nProcessing chunk {chunk_count}/{total_chunks} with {len(prompt_chunk)} prompts...")

            chunk_results = self._process_chunk(prompt_chunk, image_chunk, max_target_length, max_prefill_length)
            all_results.extend(chunk_results)

        return all_results

    def _process_chunk(
        self,
        prompts: List[str],
        image_paths: List[Optional[str]],
        max_target_length: Optional[int],
        max_prefill_length: Optional[int],
    ) -> List[str]:
        """Processes a single chunk of prompts that fits within the batch size."""
        start_time = time.time()
        _NUM_STREAMS = len(prompts)
        target_length = max_target_length if max_target_length is not None else self.config.max_target_length
        prefill_length = max_prefill_length if max_prefill_length is not None else self.config.max_prefill_predict_length

        # 1. Preprocess and tokenize all inputs for the chunk
        stream_tokens, stream_true_lengths, stream_images = [], [], []
        for prompt, image_path in zip(prompts, image_paths):
            tokens, true_length, images = self._preprocess_inputs(prompt, prefill_length, image_path)
            assert true_length <= prefill_length, f"Input token length {true_length} is > {prefill_length}"
            stream_tokens.append(tokens)
            stream_true_lengths.append(true_length)
            stream_images.append(images)

        # 2. Initialize state for the new chunk
        self.rng, rng_init_decode = jax.random.split(self.rng)
        decode_state = self.engine.init_decode_state(rng=rng_init_decode)
        streams_results: dict[int, List[int]] = {i: [] for i in range(_NUM_STREAMS)}
        streams_finished: List[bool] = [False] * _NUM_STREAMS

        # 3. Prefill and Insert all streams in the chunk
        prefill_results_to_insert = {}
        for i in range(_NUM_STREAMS):
            self.rng, rng_prefill = jax.random.split(self.rng)
            prefill_result, first_token = self.engine.prefill(
                params=self.params,
                padded_tokens=stream_tokens[i],
                true_length=stream_true_lengths[i],
                images=stream_images[i],
                rng=rng_prefill,
                slot=i,
            )
            prefill_results_to_insert[i] = prefill_result
            streams_results[i].append(first_token.get_result_at_slot(0).tokens.item())

        for slot_idx, prefill_result in prefill_results_to_insert.items():
            decode_state = self.engine.insert(
                prefix=prefill_result, decode_state=decode_state, slot=slot_idx
            )

        # 4. Interleaved generation loop
        total_steps = target_length - prefill_length
        for step in range(total_steps):
            active_stream_indices = [i for i, finished in enumerate(streams_finished) if not finished]
            if not active_stream_indices:
                break  # All streams in the chunk are finished

            self.rng, rng_generate = jax.random.split(self.rng)
            decode_state, sampled_tokens = self.engine.generate(self.params, decode_state, rng=rng_generate)

            for slot_idx in active_stream_indices:
                token_for_slot = sampled_tokens.get_result_at_slot(slot_idx).tokens.item()
                streams_results[slot_idx].append(token_for_slot)

                current_len = stream_true_lengths[slot_idx] + step + 1
                is_max_len = current_len >= target_length
                is_eos = token_for_slot == self.eos_id

                if is_max_len or is_eos:
                    streams_finished[slot_idx] = True
                    self.engine.release_pages(slot=slot_idx)

        # 5. Decode results and return
        final_outputs = []
        for i in range(_NUM_STREAMS):
            output_tokens = streams_results[i]
            if output_tokens and output_tokens[-1] == self.eos_id:
                output_tokens = output_tokens[:-1]
            output_string = self.tokenizer.decode(output_tokens)
            final_outputs.append(output_string.strip())
        
        end_time = time.time()
        print(f"Chunk processing time: {end_time - start_time:.2f}s")
        return final_outputs


    def _preprocess_inputs(self, text, prefill_length, image_path):
        """Helper to preprocess a single text and optional image input."""
        processor_output = multimodal_utils.PreprocessorOutput()
        images = None
        if self.config.use_multimodal and image_path:
            text = multimodal_utils.reformat_prompt(
                text, image_placeholder=self.config.image_placeholder, model_name=self.config.model_name
            )
            loaded_images = multimodal_utils.load_image_from_path(image_path)
            processor_output = multimodal_utils.pre_process_image(loaded_images, model_name=self.config.model_name)
            prefill_length -= multimodal_utils.get_image_offsets(
                self.config.model_name, processor_output=processor_output
            )
            images = processor_output.pixel_values

        tokens, true_length = self.tokenizer.encode(text, is_bos=not self.has_chat_template, prefill_lengths=[prefill_length])
        if self.config.use_multimodal and image_path:
            tokens = multimodal_utils.prepare_text_for_image_fusion(tokens, model_name=self.config.model_name, processor_output=processor_output)
            true_length += multimodal_utils.get_image_offsets(self.config.model_name, processor_output=processor_output)

        return tokens, true_length, images

    def _validate_config(self, config):
        """Validates configuration."""
        assert config.load_full_state_path == "", "Decode doesn't operate on full states!"
        assert config.quantization not in ["fp8", "nanoo_fp8"], "FP8 quantization is not supported in this script."
        assert config.per_device_batch_size * jax.device_count() >= 1, "Total batch size must be at least 1."


if __name__ == "__main__":
    import sys
    llm = MaxTextGenerator(sys.argv)

    # Example with a list of prompts that is larger than the batch size
    # This will trigger the chunking logic.
    prompts_to_run = [
        "The best thing about Seattle, Washington on a rainy afternoon is",
        "What is the theory of special relativity in simple terms?",
        "Write a short, four-line poem about a computer dreaming.",
        "The capital of France is",
        "Introduce the concept of a Large Language Model.",
        "What are the main differences between a CPU and a GPU?",
        "Explain how a sourdough starter works.",
        "The amount of access cabinet secretaries have to the president is most likely to be controlled by the\nA. vice president\nB. president's chief of staff\nC. national security advisor\nD. chair of the Federal Reserve Board\nAnswer:",
        "Which principle was established by the Supreme Court's decision in Marbury v. Madison?\nA. One man, one vote\nB. Separate but equal\nC. Judicial review\nD. Right to privacy\nAnswer:",
        "Introduce Elon Musk in 50 words.",   # 500 words
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, gravity, and the universe. Discuss the two main components of this theory, special relativity and general relativity. Explain the key postulates and concepts of each, such as time dilation, length contraction, and the equivalence principle. Furthermore, elaborate on the experimental evidence that supports these theories and their profound implications for modern physics, including the development of GPS technology and our understanding of black holes and the expansion of the universe.",   # 700 words
        "The exclusionary rule was established to\nA. create 'separate but equal' facilities to facilitate racial segregation\nB. allow private organizations to restrict their memberships\nC. limit the government's ability to use illegally obtained evidence\nD. deny control of interstate commerce to the states\nAnswer:",
        "Q: What is the capital of Germany?\nA: Berlin\n\nQ: What is the capital of Spain?\nA: Madrid\n\nQ: What is the capital of France?\nA:",
        "The best thing about Seattle, Washington on a rainy afternoon is",
        "What is the theory of special relativity in simple terms?",
        "Write a short, four-line poem about a computer dreaming.",
        "The capital of France is",
        "Introduce the concept of a Large Language Model.",
        "What are the main differences between a CPU and a GPU?",
        "Explain how a sourdough starter works.",
        "The amount of access cabinet secretaries have to the president is most likely to be controlled by the\nA. vice president\nB. president's chief of staff\nC. national security advisor\nD. chair of the Federal Reserve Board\nAnswer:",
        "Which principle was established by the Supreme Court's decision in Marbury v. Madison?\nA. One man, one vote\nB. Separate but equal\nC. Judicial review\nD. Right to privacy\nAnswer:",
        "Introduce Elon Musk in 50 words.",   # 500 words
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, gravity, and the universe. Discuss the two main components of this theory, special relativity and general relativity. Explain the key postulates and concepts of each, such as time dilation, length contraction, and the equivalence principle. Furthermore, elaborate on the experimental evidence that supports these theories and their profound implications for modern physics, including the development of GPS technology and our understanding of black holes and the expansion of the universe.",   # 700 words
        "The exclusionary rule was established to\nA. create 'separate but equal' facilities to facilitate racial segregation\nB. allow private organizations to restrict their memberships\nC. limit the government's ability to use illegally obtained evidence\nD. deny control of interstate commerce to the states\nAnswer:",
        "Answer the following three questions.\nQ: What is the capital of Germany?\nA: Berlin\n\nQ: What is the capital of Spain?\nA: Madrid\n\nQ: What is the capital of France?\nA:"
    ]
    
    print(f"\n--- Starting Batch Generation for {len(prompts_to_run)} Prompts ---")
    generated_results = llm.generate_batch(prompts_to_run)
    print("\n--- Batch Generation Complete ---")

    for i, (prompt, result) in enumerate(zip(prompts_to_run, generated_results)):
        print(f"\n[Prompt {i+1}] {prompt}")
        print(f"[Result {i+1}] {result}")