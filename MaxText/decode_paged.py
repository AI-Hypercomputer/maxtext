# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utility for running inference on a single stream with PagedAttention."""

import jax
import max_utils
import maxengine
import os
import pyconfig
from typing import Sequence
from absl import app

def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    pyconfig.initialize(argv)
    config = pyconfig.config
    validate_config(config)
    max_utils.print_system_information()

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)

    params = engine.load_params(rng_load_params)

    text = config.prompt
    metadata = engine.get_tokenizer()
    tokenizer_model = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
    assert true_length <= config.max_prefill_predict_length, "Prompt is too long for max_prefill_predict_length"
    print(f"padded_tokens: {tokens}")
    print(f"true_length: {true_length}")
    print(f"tokens.dtype: {tokens.dtype}") # Check the data type of tokens

    if isinstance(params, dict):
        print(f"params keys: {params.keys()}")
    else:
        print(f"params is of type: {type(params)}")

    print(f"\n{'='*60}")
    print(f"Input: `{text}`")
    print(f"True length: {true_length}")
    print(f"Token sequence: {tokens[:true_length]}")
    print(f"{'='*60}\n")

    # 1. Initialize decode_state
    rng, rng_init_decode = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init_decode)

    # 2. Prefill
    slot = 0  # Use slot 0 for this single stream
    rng, rng_prefill = jax.random.split(rng)
    prefill_result, first_token, page_state = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        slot=slot,
        rng=rng_prefill,
    )
    
    # Update decode_state with the prefill results
    decode_state = {
        "logits": prefill_result["logits"],
        "cache": prefill_result["cache"],
        "next_pos": prefill_result["next_pos"],
        "generated_tokens": prefill_result["generated_tokens"],
        "tokens": prefill_result["tokens"],
    }

    # Print the first token
    first_token_id = first_token.data[0, 0]
    first_token_text = tokenizer_model.decode(jax.numpy.array([first_token_id]))
    print(f"First generated token: {first_token_id} -> '{first_token_text}'")

    # 3. Generation Loop
    generated_tokens = [first_token_id]
    print(f"About to start generate loop using {config.max_target_length} - {true_length}")
    
    # for i in range(config.max_target_length - true_length):
    #     rng, rng_generate = jax.random.split(rng)
    #     decode_state, next_token_result = engine.generate(
    #         params=params,
    #         decode_state=decode_state,
    #         slot=slot,
    #         rng=rng_generate,
    #     )
        
    #     # Get the token and add to our list
    #     next_token_id = next_token_result.data[0, 0]
    #     generated_tokens.append(next_token_id)
        
    #     # Print incremental progress
    #     if i % 2 == 0:
    #         # Every 2 tokens, show the in-progress generation
    #         current_output = tokenizer_model.decode(jax.numpy.array(generated_tokens))
    #         print(f"Progress [{i+1}/{config.max_target_length - true_length}]: '{current_output}'")

    # # Decode the generated tokens
    # output = tokenizer_model.decode(jax.numpy.array(generated_tokens))
    # print(f"\n{'='*60}")
    # print(f"Input: `{text}`")
    # print(f"Output: `{output}`")
    # print(f"{'='*60}\n")

    # # Verify against expected output if specified
    # if hasattr(config, 'autoregressive_decode_assert') and config.autoregressive_decode_assert:
    #     assert output.startswith(config.autoregressive_decode_assert), (
    #         f"Generated text mismatch.  Expected start: '{config.autoregressive_decode_assert}', "
    #         f"Got: '{output}'"
    #     )


def validate_config(config):
    assert config.load_full_state_path == "", (
        "Decode doesn't operate on full states! Convert to parameter checkpoint first "
        "using generate_param_only_checkpoint."
    )
    assert config.attention == "paged", "This script is designed for PagedAttention. Set config.attention = 'paged'."

if __name__ == "__main__":
    app.run(main)