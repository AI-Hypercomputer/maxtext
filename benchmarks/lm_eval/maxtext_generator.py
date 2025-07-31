
import os
from typing import Sequence, Optional

import jax
import jax.numpy as jnp

from absl import app

from jetstream.engine import engine_api

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig
from MaxText import profiler
from MaxText import multimodal_utils

import sys
print(f"DEBUG: sys.argv received by Python is: {sys.argv}")


class MaxTextGenerator:
    def __init__(self, argv: Sequence[str]):
        print("Initializing MaxTextGenerator...")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" 
        jax.config.update("jax_default_prng_impl", "unsafe_rbg")
        self.config = pyconfig.initialize(argv)
        self._validate_config(self.config)
        jax.config.update("jax_use_shardy_partitioner", self.config.shardy)
        max_utils.print_system_information()

        self.engine = maxengine.MaxEngine(self.config)
        self.rng = jax.random.PRNGKey(1234)
        self.rng, self.rng_load_params = jax.random.split(self.rng)
        self.params = self.engine.load_params(self.rng_load_params)

        self.metadata = self.engine.get_tokenizer()
        self.tokenizer = self.engine.build_tokenizer(self.metadata)
        try:
            self.has_chat_template = getattr(self.tokenizer.tokenizer, "chat_template", False)
        except AttributeError:
            self.has_chat_template = False
    
        self.batch_size = int(self.config.per_device_batch_size * jax.device_count())
        print("Initialization complete.")

    def generate(self, prompt: str, image_path: Optional[str] = None, 
                max_target_length: Optional[int] = None, 
                max_prefill_length: Optional[int] = None):
        target_length = max_target_length if max_target_length is not None else self.config.max_target_length
        prefill_length = max_prefill_length if max_prefill_length is not None else self.config.max_prefill_predict_length
        tokens, true_length, images = self._preprocess_inputs(prompt, prefill_length, image_path)
        assert true_length <= prefill_length, f"Input token length {true_length} is > {prefill_length}"

        rng_generate, self.rng = jax.random.split(self.rng)
        rng_prefill, rng_init_decode, rng_generate = jax.random.split(rng_generate, 3)

        prefill_result, first_token = self.engine.prefill(
            params=self.params,
            padded_tokens=tokens,
            images=images,
            true_length=true_length,
            rng=rng_prefill,
            slot=0
        )

        decode_state = self.engine.init_decode_state(rng_init_decode)
        decode_state = self.engine.insert(prefill_result, decode_state, slot=0)

        sampled_tokens_list = [first_token]
        steps = range(prefill_length, target_length)
        for _ in steps:
            rng_step, rng_generate = jax.random.split(rng_generate)
            decode_state, sampled_tokens = self.engine.generate(self.params, decode_state, rng=rng_step)
            sampled_tokens_list.append(sampled_tokens)
    
        result_tokens = [t.get_result_at_slot(0).tokens.item() for t in sampled_tokens_list]
        output_string = self.tokenizer.decode(result_tokens)

        return output_string

    def _preprocess_inputs(self, text, prefill_length, image_path):
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
        assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
        assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD GPUs is not supported yet"
        assert self.config.per_device_batch_size * jax.device_count() >= 1, "Total batch size must be at least 1."



if __name__ == '__main__':
    import sys
    args = sys.argv

    llm = MaxTextGenerator(args)

    prompt1 = "What is the best thing about Large Language Models?"
    print(f"Prompt: {prompt1}")
    output1 = llm.generate(prompt1, )
    print(f"Generated: {output1}\n")

    prompt2 = "Write a short poem about coding in JAX."
    print(f"Prompt: {prompt2}")
    output2 = llm.generate(prompt2)
    print(f"Generated: {output2}\n")

    prompt2 = "Who is Elon Musk?"
    print(f"Prompt: {prompt2}")
    output2 = llm.generate(prompt2)
    print(f"Generated: {output2}\n")