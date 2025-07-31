import os
import uuid
from typing import Sequence, List

from absl import app
import jax
import numpy as np

import time

from MaxText import max_utils, maxengine, pyconfig

def _validate_config(config, num_streams, batch_size):
    assert config.load_full_state_path == "", (
        "Decode doesn't operate on full states! Convert to parameter checkpoint first."
    )
    assert 0 < num_streams <= batch_size, (
        f"_NUM_STREAMS ({num_streams}) must be > 0 and <= batch size ({batch_size})"
    )

def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    config = pyconfig.initialize(argv)
    max_utils.print_system_information()

    # Matching the number of streams from your error log
    _NUM_STREAMS = 16
    batch_size = int(config.per_device_batch_size * jax.device_count())
    _validate_config(config, _NUM_STREAMS, batch_size)

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params, rng_init_decode, fixed_rng = jax.random.split(rng, 4)

    params = engine.load_params(rng=rng_load_params)

    prompts = [
        "The best thing about Kirkland, Washington on a sunny afternoon is",
    ]
    if _NUM_STREAMS > len(prompts):
        prompts = [prompts[i % len(prompts)] for i in range(_NUM_STREAMS)]
    else:
        prompts = prompts[:_NUM_STREAMS]

    metadata = engine.get_tokenizer()
    tokenizer_model = engine.build_tokenizer(metadata)

    # =========== Final Corrected Logic for Tokenization and Batching ===========

    # 1. Create the final, uniformly-shaped container FIRST.
    #    Its length is determined by the maximum possible prefill length.
    pad_id = tokenizer_model.pad_id if hasattr(tokenizer_model, 'pad_id') else 0
    final_padded_tokens = np.full(
        (batch_size, config.max_prefill_predict_length), pad_id, dtype=np.int32
    )
    true_lengths = []

    print(f"Tokenizing all {_NUM_STREAMS} prompts...")
    for i, prompt in enumerate(prompts):
        # 2. Call encode to get the bucket-padded tokens (e.g., length 128 or 256).
        bucket_padded_tokens, true_length = tokenizer_model.encode(
            prompt,
            is_bos=True,
            max_prefill_length=config.max_prefill_predict_length
        )
        assert true_length <= config.max_prefill_predict_length, f"Prompt '{prompt[:30]}...' is too long!"

        # 3. Copy the bucketed result into our final, max-length container.
        #    This ensures every row in `final_padded_tokens` has the same length.
        final_padded_tokens[i, :bucket_padded_tokens.shape[0]] = bucket_padded_tokens
        true_lengths.append(true_length)

    # 4. Move the final, uniform batch to the device.
    final_padded_tokens = jax.device_put(final_padded_tokens)
    print("Finished tokenizing and batching.")

    start = time.time()

    # ========================================================================
    print("Jax Device Count:", jax.device_count())
    print(f"Batch size: {batch_size}, _NUM_STREAMS: {_NUM_STREAMS}")

    decode_state = engine.init_decode_state(rng=rng_init_decode)
    print("Initial decode state initialized.")

    # Use the same fixed RNG for all prefill and generate steps
    print(f"Prefilling and inserting all {_NUM_STREAMS} streams...")
    prefill_results = []
    first_tokens = []
    for slot_idx in range(_NUM_STREAMS):
        request_id = uuid.uuid4()
        prefill_result, first_token = engine.prefill(
            params=params,
            padded_tokens=final_padded_tokens[slot_idx],
            true_length=true_lengths[slot_idx],
            rng=fixed_rng,
            slot=slot_idx,
            request_id=request_id,
        )
        decode_state = engine.insert(
            prefix=prefill_result,
            decode_state=decode_state,
            slot=slot_idx,
            request_id=uuid.uuid4(),
        )
        prefill_results.append(prefill_result)
        first_tokens.append(first_token)

    print("All streams prefilled and inserted. Starting generation loop...")

    streams_results: List[List[int]] = [[] for _ in range(_NUM_STREAMS)]
    for slot_idx in range(_NUM_STREAMS):
        streams_results[slot_idx].append(first_tokens[slot_idx].get_result_at_slot(0).tokens.item())

    steps = range(config.max_prefill_predict_length, config.max_target_length)
    for step in steps:
        decode_state, sampled_tokens = engine.generate(params, decode_state, rng=fixed_rng)
        for slot_idx in range(_NUM_STREAMS):
            if slot_idx < sampled_tokens.data.shape[0]:
                token_for_slot = sampled_tokens.get_result_at_slot(slot_idx).tokens.item()
                streams_results[slot_idx].append(token_for_slot)

    end = time.time()



    print("\n--- Final Results ---")
    for i in range(_NUM_STREAMS):
        if streams_results[i]:
            output = tokenizer_model.decode(streams_results[i])
            print(f"Stream {i}: Input=`{prompts[i]}` -> Output=`{output}`")
            if i == 0 and hasattr(config, "autoregressive_decode_assert"):
                assert output.startswith(
                    config.autoregressive_decode_assert
                ), f"Stream {i} generated text mismatch: `{output}` vs expected start `{config.autoregressive_decode_assert}`"
        else:
            print(f"Stream {i}: Was not activated.")

    
    print("Decode time used: ", end - start)

if __name__ == "__main__":
    app.run(main)