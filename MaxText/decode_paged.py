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

    # 1.  Initialize decode_state (using init_decode_state)
    rng, rng_init_decode = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init_decode)

    # 2.  Prefill (with slot and true_length)
    slot = 0  # Use slot 0 for this single stream
    rng, rng_prefill = jax.random.split(rng)
    prefill_result, first_token, page_state = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        slot=slot,
        rng=rng_prefill,
    )
    # Update decode_state with the prefill results and page_state
    decode_state = {
        "logits": prefill_result["logits"],
        "cache": prefill_result["cache"], # Includes page_manager
        "next_pos": prefill_result["next_pos"],
        "generated_tokens": prefill_result["generated_tokens"],
        "tokens": prefill_result["tokens"],
    }

    # 3.  Generation Loop (using updated decode_state)
    generated_tokens = [first_token.data[0, 0]]  # Store generated tokens
    for i in range(config.max_target_length - true_length):
        rng, rng_generate = jax.random.split(rng)
        decode_state, next_token_result = engine.generate(
            params=params,
            decode_state=decode_state,
            slot=slot,
            rng=rng_generate,
        )
        generated_tokens.append(next_token_result.data[0, 0])

    # Decode the generated tokens
    output = tokenizer_model.decode(jax.numpy.array(generated_tokens))
    print(f"Input: `{text}`")
    print(f"Output: `{output}`")

    assert output.startswith(config.autoregressive_decode_assert), (
        f"Generated text mismatch.  Expected start: '{config.autoregressive_decode_assert}', "
        f"Got: '{output}'"
    )


def validate_config(config):
    assert config.load_full_state_path == "", (
        "Decode doesn't operate on full states! Convert to parameter checkpoint first "
        "using generate_param_only_checkpoint."
    )
    assert config.attention == "paged", "This script is designed for PagedAttention. Set config.attention = 'paged'."

if __name__ == "__main__":
    app.run(main)