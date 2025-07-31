"""CLI utility for running inference with interleaved prefill and generate."""

import os
import uuid
from typing import Sequence, List

from absl import app

import jax

from MaxText import max_utils, maxengine, pyconfig

_NUM_STREAMS = 5
# How many streams to prefill initially before starting generation.
_INITIAL_PREFILL_STREAMS = 5  # Example: Start generating after 2 streams are ready


def _validate_config(config):
    """Validate configuration settings."""
    assert config.load_full_state_path == "", (
        "Decode doesn't operate on full states! Convert to parameter checkpoint first."
        "Using generate_param_only_checkpoint."
    )
    assert (
        0 < _INITIAL_PREFILL_STREAMS <= _NUM_STREAMS
    ), f"_INITIAL_PREFILL_STREAMS ({_INITIAL_PREFILL_STREAMS}) must be > 0 and <= _NUM_STREAMS ({_NUM_STREAMS})"


def main(argv: Sequence[str]) -> None:
    """Main function to run interleaved inference."""
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    config = pyconfig.initialize(argv)
    # For demo: supply different prompts per stream here (can load from file or args)
    prompts = [
        "The amount of access cabinet secretaries have to the president is most likely to be controlled by the\nA. vice president\nB. president's chief of staff\nC. national security advisor\nD. chair of the Federal Reserve Board\nAnswer:",
        "Which principle was established by the Supreme Court's decision in Marbury v. Madison?\nA. One man, one vote\nB. Separate but equal\nC. Judicial review\nD. Right to privacy\nAnswer:",
        "Introduce Elon Musk in 50 words.",   # 500 words
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, gravity, and the universe. Discuss the two main components of this theory, special relativity and general relativity. Explain the key postulates and concepts of each, such as time dilation, length contraction, and the equivalence principle. Furthermore, elaborate on the experimental evidence that supports these theories and their profound implications for modern physics, including the development of GPS technology and our understanding of black holes and the expansion of the universe.",   # 700 words
        "The exclusionary rule was established to\nA. create 'separate but equal' facilities to facilitate racial segregation\nB. allow private organizations to restrict their memberships\nC. limit the government's ability to use illegally obtained evidence\nD. deny control of interstate commerce to the states\nAnswer:",  # 1000 words
    ]
    _validate_config(config)
    max_utils.print_system_information()

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng=rng_load_params)

    metadata = engine.get_tokenizer()
    tokenizer_model = engine.build_tokenizer(metadata)
    # We need the EOS token ID to check for it during generation.
    # The exact attribute may vary, but '.eos_id' is common.
    eos_id = tokenizer_model.eos_id

    # Store per-stream prompt tokens and true_lengths
    stream_tokens = []
    stream_true_lengths = []
    for i, text in enumerate(prompts):
        tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
        assert true_length <= config.max_prefill_predict_length, f"Prompt {i} too long for prefill length"
        stream_tokens.append(tokens)
        stream_true_lengths.append(true_length)

    batch_size = int(config.per_device_batch_size * jax.device_count())
    assert 0 < _NUM_STREAMS <= batch_size, f"The number of streams {_NUM_STREAMS} must be > 0 and <= batch size {batch_size}"

    rng, rng_init_decode = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng=rng_init_decode)
    print("Initial decode state initialized.")

    streams_results: dict[int, List[int]] = {i: [] for i in range(_NUM_STREAMS)}
    streams_active: List[bool] = [False] * _NUM_STREAMS
    streams_finished: List[bool] = [False] * _NUM_STREAMS
    streams_prefilled_count = 0
    streams_inserted_count = 0

    print(f"Starting initial prefill for {_INITIAL_PREFILL_STREAMS} streams...")
    prefill_results_to_insert = {}
    for i in range(_INITIAL_PREFILL_STREAMS):
        slot_idx = i
        print(f"  Prefilling stream for slot {slot_idx}...")
        rng, rng_prefill = jax.random.split(rng)
        request_id = uuid.uuid4()
        prefill_result, first_token = engine.prefill(
            params=params,
            padded_tokens=stream_tokens[slot_idx],
            true_length=stream_true_lengths[slot_idx],
            rng=rng_prefill,
            slot=slot_idx,
            request_id=request_id,
        )
        prefill_results_to_insert[slot_idx] = prefill_result
        streams_results[slot_idx].append(first_token.get_result_at_slot(0).tokens.item())
        streams_prefilled_count += 1
        print(f"After prefill stream {slot_idx}")

    print("Inserting initial prefill results...")
    for slot_idx, prefill_result in prefill_results_to_insert.items():
        request_id = uuid.uuid4()
        decode_state = engine.insert(
            prefix=prefill_result,
            decode_state=decode_state,
            slot=slot_idx,
            request_id=request_id,
        )
        streams_active[slot_idx] = True
        streams_inserted_count += 1
        print(f"  Inserted prefill for slot {slot_idx}")

    print("Starting interleaved generation loop...")
    total_steps = config.max_target_length - config.max_prefill_predict_length
    for step in range(total_steps):
        print(f"\n--- Step {step + 1} / {total_steps} ---")
        active_stream_indices = [i for i, active in enumerate(streams_active) if active and not streams_finished[i]]
        if active_stream_indices:
            print(f"  Generating for active slots: {active_stream_indices}")
            rng, rng_generate = jax.random.split(rng)
            decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)

            ## MODIFICATION START ##
            # This logic is updated to handle both EOS and max length termination.
            for slot_idx in active_stream_indices:
                # First, get and store the newly generated token for this stream
                token_for_slot = sampled_tokens.get_result_at_slot(slot_idx).tokens.item()
                streams_results[slot_idx].append(token_for_slot)

                # Next, check for termination conditions
                current_len = stream_true_lengths[slot_idx] + step + 1
                is_max_len = current_len >= config.max_target_length
                is_eos = token_for_slot == eos_id

                if is_max_len or is_eos:
                    # If either condition is met, mark the stream as finished.
                    streams_finished[slot_idx] = True
                    streams_active[slot_idx] = False

                    if is_max_len:
                        print(f"    Stream in slot {slot_idx} reached max target length.")
                    if is_eos:
                        print(f"    Stream in slot {slot_idx} generated EOS token.")

                    # Release the memory pages associated with the finished stream
                    print(f"    Calling engine to release pages for finished slot {slot_idx}...")
                    engine.release_pages(slot=slot_idx)
            ## MODIFICATION END ##
        else:
            print("  No active streams to generate for.")

        if all(streams_finished):
            print("\nAll streams finished generation.")
            break

        num_active_not_finished = sum(1 for i in range(_NUM_STREAMS) if streams_active[i] and not streams_finished[i])
        available_slots = batch_size - num_active_not_finished
        can_prefill_more = streams_prefilled_count < _NUM_STREAMS

        if can_prefill_more and available_slots > 0:
            try:
                next_available_slot = streams_active.index(False)
                print(f"  Prefilling new stream for slot {next_available_slot}...")
                rng, rng_prefill = jax.random.split(rng)
                request_id = uuid.uuid4()
                prefill_result, first_token = engine.prefill(
                    params=params,
                    padded_tokens=stream_tokens[next_available_slot],
                    true_length=stream_true_lengths[next_available_slot],
                    rng=rng_prefill,
                    slot=next_available_slot,
                    request_id=request_id,
                )
                streams_prefilled_count += 1

                print(f"  Inserting new stream into slot {next_available_slot}...")
                request_id_insert = uuid.uuid4()
                decode_state = engine.insert(
                    prefix=prefill_result,
                    decode_state=decode_state,
                    slot=next_available_slot,
                    request_id=request_id_insert,
                )
                streams_active[next_available_slot] = True
                streams_inserted_count += 1
                streams_results[next_available_slot].append(first_token.get_result_at_slot(0).tokens.item())

            except ValueError:
                print("  Warning: Available slots detected but couldn't find an inactive one.")
        elif can_prefill_more:
            print("  Generate step finished, but no available slots to prefill new stream.")
        else:
            print("  Generate step finished, all streams already prefilled.")

    print("\n--- Final Results ---")
    for i in range(_NUM_STREAMS):
        if streams_results[i]:
            # The last token might be EOS, which we often don't want in the final output string.
            output_tokens = streams_results[i]
            if output_tokens and output_tokens[-1] == eos_id:
                output_tokens = output_tokens[:-1]

            output = tokenizer_model.decode(output_tokens)
            print(f"Stream {i}: Input=`{prompts[i]}` -> Output=`{output}`")
            # Optional: perform asserts/checks as needed
        else:
            print(f"Stream {i}: Was not activated.")

if __name__ == "__main__":
    app.run(main)