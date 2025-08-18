import os
import time
import uuid
from typing import Sequence, Optional, List, Union

import jax
import jax.numpy as jnp
from absl import app
import numpy as np

from MaxText import max_utils, maxengine, pyconfig, multimodal_utils

from dataclasses import dataclass, field

# Set TF log level to avoid verbose startup messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@dataclass
class LogProbs:
    tokens: List[int]
    token_logprobs: List[float]
    top_logprobs: Optional[List[None]] = None
    text_offset: List[int] = field(default_factory=list)

@dataclass
class Completion:
    index: int
    text: str
    logprobs: Optional[LogProbs]
    finish_reason: str = "stop"


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

        print("Chat Template: \n", self.has_chat_template)

        self.batch_size = int(self.config.per_device_batch_size * jax.device_count())
        end_time = time.time()
        print(f"Initialization complete in {end_time - start_time:.2f} seconds. Max batch size: {self.batch_size}")

    def generate_batch(
        self,
        prompts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        max_tokens: int = None,
        logprobs: int = None,
        echo: bool = False,
    ) -> List[Completion]:
        """
        Generates text for a batch of prompts, handling chunking automatically.

        Args:
            prompts: A list of prompt strings.
            image_paths: An optional list of image paths, one for each prompt.

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

            chunk_count = (i // self.batch_size) + 1
            total_chunks = (num_prompts + self.batch_size - 1) // self.batch_size
            print(f"\nProcessing chunk {chunk_count}/{total_chunks} with {len(prompt_chunk)} prompts...")

            chunk_results = self._process_chunk(prompt_chunk, image_chunk, max_tokens, logprobs, echo)
            all_results.extend(chunk_results)

        return all_results

    def _process_chunk(
        self,
        prompts: List[str],
        image_paths: List[Optional[str]],
        max_tokens: int,
        logprobs: int = None,   # top_logprobs count; ignored for now
        echo: bool = False
    ) -> List[Completion]:
        start_time = time.time()
        _NUM_STREAMS = len(prompts)
        target_length = getattr(self.config, "max_target_length", 2048)
        prefill_length = getattr(self.config, "max_prefill_predict_length", 1024)

        # Guard: max_tokens counts ALL generated tokens (incl. prefill one)
        if max_tokens is not None and max_tokens <= 0:
            return [Completion(index=i, text="", logprobs=None) for i in range(_NUM_STREAMS)]

        # 1) tokenize inputs
        stream_tokens, stream_true_lengths, stream_images = [], [], []
        for prompt, image_path in zip(prompts, image_paths):
            toks, tlen, imgs = self._preprocess_inputs(prompt, prefill_length, image_path)
            assert tlen <= prefill_length, f"Input token length {tlen} is > {prefill_length}"
            stream_tokens.append(toks)
            stream_true_lengths.append(tlen)
            stream_images.append(imgs)

        # 2) init decode state
        self.rng, rng_init_decode = jax.random.split(self.rng)
        decode_state = self.engine.init_decode_state(rng=rng_init_decode)

        # Per-stream accumulators
        streams_finished = [False] * _NUM_STREAMS
        finish_reasons: dict[int, str] = {i: "length" for i in range(_NUM_STREAMS)}
        # prompt
        prompt_ids: dict[int, List[int]] = {i: [] for i in range(_NUM_STREAMS)}
        prompt_logps: dict[int, List[float]] = {i: [] for i in range(_NUM_STREAMS)}
        # generated (includes prefill-first token)
        gen_ids: dict[int, List[int]] = {i: [] for i in range(_NUM_STREAMS)}
        gen_logps: dict[int, List[float]] = {i: [] for i in range(_NUM_STREAMS)}

        # 3) Prefill each stream (one at a time), then insert
        prefill_results_to_insert = {}
        for i in range(_NUM_STREAMS):
            self.rng, rng_prefill = jax.random.split(self.rng)
            prefill_result, _first_token_obj = self.engine.prefill(
                params=self.params,
                padded_tokens=stream_tokens[i],
                true_length=stream_true_lengths[i],
                images=stream_images[i],
                rng=rng_prefill,
                slot=i,
                return_prompt_logp=True,
            )
            prefill_results_to_insert[i] = prefill_result

            # prompt ids/logps
            p_ids = list(map(int, np.array(stream_tokens[i][:stream_true_lengths[i]], dtype=np.int32).tolist()))
            prompt_ids[i].extend(p_ids)
            if prefill_result.get("prompt_logp", None) is not None:
                p_logp_arr = np.array(prefill_result["prompt_logp"][0, :stream_true_lengths[i]])
                prompt_logps[i].extend([float(x) for x in p_logp_arr.tolist()])

            # first generated token from prefill_result dict
            # shapes expected: tokens [1,1], token_logp [1,1] (if enabled)
            first_token_arr = np.array(prefill_result["tokens"])
            first_id = int(first_token_arr[0, 0])
            gen_ids[i].append(first_id)

            first_lp = prefill_result.get("token_logp", None)
            if first_lp is not None:
                gen_logps[i].append(float(np.array(first_lp)[0, 0]))

        # Insert all prefixes into decode_state
        for slot_idx, prefill_result in prefill_results_to_insert.items():
            decode_state = self.engine.insert(prefix=prefill_result, decode_state=decode_state, slot=slot_idx)

        # 4) Generate loop
        total_steps = target_length - prefill_length
        if max_tokens is not None:
            total_steps = min(total_steps, max_tokens - 1)  # -1 since we already sampled one in prefill

        for step in range(total_steps):
            active_stream_indices = [i for i, done in enumerate(streams_finished) if not done]
            if not active_stream_indices:
                break

            self.rng, rng_generate = jax.random.split(self.rng)
            decode_state, _sampled_tokens = self.engine.generate(self.params, decode_state, rng=rng_generate)

            # Pull ids/logps directly from decode_state
            # tokens shape [batch,1]; token_logp shape [batch,1] (if enabled)
            state_tokens = np.array(decode_state["tokens"])
            state_token_logp = decode_state.get("token_logp", None)
            state_token_logp_np = None if state_token_logp is None else np.array(state_token_logp)

            for slot_idx in active_stream_indices:
                tok_id = int(state_tokens[slot_idx, 0])
                gen_ids[slot_idx].append(tok_id)

                if state_token_logp_np is not None:
                    gen_logps[slot_idx].append(float(state_token_logp_np[slot_idx, 0]))

                # EOS / max len
                current_len = stream_true_lengths[slot_idx] + 1 + step  # +1 for prefill token already generated
                is_max_len = current_len >= target_length
                is_eos = tok_id == self.eos_id
                if is_max_len or is_eos:
                    streams_finished[slot_idx] = True
                    if is_eos:
                        finish_reasons[slot_idx] = "stop"
                    self.engine.release_pages(slot=slot_idx)

        # 5) Build Completions
        completions: List[Completion] = []
        for i in range(_NUM_STREAMS):
            # Align tokens/logprobs with the TEXT we will return
            gen_ids_for_text = gen_ids[i][:]
            gen_logps_for_text = gen_logps[i][:]

            # If you drop EOS from text, drop it here too
            if gen_ids_for_text and gen_ids_for_text[-1] == self.eos_id:
                gen_ids_for_text = gen_ids_for_text[:-1]
                if len(gen_logps_for_text) >= len(gen_ids[i]):  # safety for any edge cases
                    gen_logps_for_text = gen_logps_for_text[:-1]

            if echo:
                tokens_for_text = prompt_ids[i] + gen_ids_for_text
                logps_for_text  = prompt_logps[i] + gen_logps_for_text
            else:
                tokens_for_text = gen_ids_for_text
                logps_for_text  = gen_logps_for_text

            # IMPORTANT: do NOT .strip() — that would break offsets
            text = self.tokenizer.decode(tokens_for_text)

            # print("Output generation: ", text)

            # Offsets must be computed over the *same* tokens that build `text`
            offsets = self._token_offsets(tokens_for_text, 0)

            if logprobs is not None:
                # Lengths must match: one offset/logprob per token in `text`
                if len(tokens_for_text) != len(logps_for_text):
                    print(f"[warn] token/logprob length mismatch for stream {i}: "
                        f"{len(tokens_for_text)} vs {len(logps_for_text)}. Returning no logprobs.")
                    lp = None
                else:
                    lp = LogProbs(
                        tokens=tokens_for_text,
                        token_logprobs=logps_for_text,
                        top_logprobs=None,
                        text_offset=offsets,
                    )
            else:
                lp = None

            completions.append(Completion(
                index=i,
                text=text,
                logprobs=lp,
                finish_reason=finish_reasons[i]
            ))

        end_time = time.time()
        print(f"Chunk processing time: {end_time - start_time:.2f}s")
        return completions


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




if __name__ == "__main__":
    import sys

    # Instantiate
    llm = MaxTextGenerator(sys.argv)

    # Prompts to run
    prompts_to_run = [
        # "The best thing about Seattle, Washington is",
        # "The capital of Germany",
        # "<s>[INST] What is the capital of France? [/INST]",
        "<s> [INST] Problem: Find the sum of all integer bases $b > 9$ for which $17_b$ is a divisor of $97_b$.\nMark your solution with \boxed\nAnswer: [/INST]"
        ]

    # Generation config
    max_tokens = 1024          # counts ALL generated tokens (includes prefill's 1st token)
    echo = True             # include prompt tokens in text/logprobs if True
    want_logprobs = 1        # non-None triggers logprob collection; top_logprobs left None for now

    print(f"\n--- Starting Batch Generation for {len(prompts_to_run)} Prompts "
          f"(max_tokens={max_tokens}, echo={echo}) ---")

    completions = llm.generate_batch(
        prompts=prompts_to_run,
        image_paths=None,
        max_tokens=max_tokens,
        logprobs=want_logprobs,
        echo=echo,
    )

    print("--- Batch Generation Complete ---")

    # Pretty printer
    def dump_completion(i, comp):
        print(f"\n=== Completion {i} ===")
        print(f"index: {comp.index}")
        print(f"text: {repr(comp.text)}")


        """
        if comp.logprobs is None:
            print("logprobs: None")
            return

        lp = comp.logprobs
        # lengths should match: one logprob/offset per token
        if not (len(lp.tokens) == len(lp.token_logprobs) == len(lp.text_offset)):
            print(f"[warn] mismatched lengths: tokens={len(lp.tokens)}, "
                  f"logps={len(lp.token_logprobs)}, offsets={len(lp.text_offset)}")

        print("logprobs:")
        print(f"  tokens (ids): {lp.tokens}")
        print(f"  token_logprobs: {[round(x, 6) for x in lp.token_logprobs]}")
        print(f"  text_offset: {lp.text_offset}")
        print(f"  top_logprobs: {lp.top_logprobs}")  # None for now

        print("  tokens (decoded, id, logprob, offset):")
        for tid, logp, off in zip(lp.tokens, lp.token_logprobs, lp.text_offset):
            piece = llm.tokenizer.decode([int(tid)])
            print(f"    {repr(piece):>12s}  id={int(tid):>6d}  logp={logp:>10.6f}  offset={off}")
        """

    # Print & validate each completion
    for i, comp in enumerate(completions):
        dump_completion(i, comp)