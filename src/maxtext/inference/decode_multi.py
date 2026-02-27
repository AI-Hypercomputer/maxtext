# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utility for running inference with interleaved prefill and generate."""

import os
import uuid
from typing import Sequence

from absl import app

import jax

from MaxText import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils

_NUM_STREAMS = 5
# How many streams to prefill initially before starting generation.
_INITIAL_PREFILL_STREAMS = 2  # Example: Start generating after 2 streams are ready


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
  _validate_config(config)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng=rng_load_params)

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert true_length <= config.max_prefill_predict_length, "Prompt too long for prefill length"

  batch_size = int(config.per_device_batch_size * jax.device_count())
  assert (
      0 < _NUM_STREAMS <= batch_size
  ), f"The number of streams {_NUM_STREAMS} must be > 0 and <= batch size {batch_size}"

  # Initialize decode state
  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng=rng_init_decode)
  print("Initial decode state initialized.")

  # Keep track of results per stream (slot)
  streams_results: dict[int, list[int]] = {i: [] for i in range(_NUM_STREAMS)}
  streams_active: list[bool] = [False] * _NUM_STREAMS  # Track which slots are active
  streams_finished: list[bool] = [False] * _NUM_STREAMS  # Track finished streams
  streams_prefilled_count = 0
  streams_inserted_count = 0

  # --- Initial Prefill Phase ---
  print(f"Starting initial prefill for {_INITIAL_PREFILL_STREAMS} streams...")
  prefill_results_to_insert = {}  # Store prefill results before inserting
  for i in range(_INITIAL_PREFILL_STREAMS):
    slot_idx = i
    print(f"  Prefilling stream for slot {slot_idx}...")
    rng, rng_prefill = jax.random.split(rng)
    request_id = uuid.uuid4()
    prefill_result, first_token = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        rng=rng_prefill,
        slot=slot_idx,
        request_id=request_id,
    )
    prefill_results_to_insert[slot_idx] = prefill_result
    streams_results[slot_idx].append(first_token.get_result_at_slot(0).tokens.item())
    streams_prefilled_count += 1
    print(f"After prefill stream {slot_idx}")

  # --- Insert Initial Prefills ---
  print("Inserting initial prefill results...")
  for slot_idx, prefill_result in prefill_results_to_insert.items():
    request_id = uuid.uuid4()
    decode_state = engine.insert(
        prefix=prefill_result,
        decode_state=decode_state,
        slot=slot_idx,
        request_id=request_id,  # Pass request_id
    )
    streams_active[slot_idx] = True  # Mark stream as active
    streams_inserted_count += 1
    print(f"  Inserted prefill for slot {slot_idx}")

  print("Starting interleaved generation loop...")
  total_steps = config.max_target_length - config.max_prefill_predict_length
  for step in range(total_steps):
    print(f"\n--- Step {step + 1} / {total_steps} ---")

    # Generate step for all active streams
    active_stream_indices = [i for i, active in enumerate(streams_active) if active and not streams_finished[i]]
    if active_stream_indices:
      print(f"  Generating for active slots: {active_stream_indices}")
      rng, rng_generate = jax.random.split(rng)
      decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)

      # Store the generated token and check for finished streams
      for slot_idx in active_stream_indices:
        # Check if the stream finished this step
        current_len = config.max_prefill_predict_length + step + 1  # Includes prefill + current step
        finished_this_step = False
        if current_len >= config.max_target_length:
          print(f"    Stream in slot {slot_idx} reached max target length.")
          streams_finished[slot_idx] = True
          streams_active[slot_idx] = False
          finished_this_step = True

        # Store token if it wasn't already finished before this step or if it finished on this step
        if not streams_finished[slot_idx] or finished_this_step:
          # Ensure we don't try to access results for a slot that might not exist
          if slot_idx < sampled_tokens.data.shape[0]:
            token_for_slot = sampled_tokens.get_result_at_slot(slot_idx).tokens.item()
            streams_results[slot_idx].append(token_for_slot)
          else:
            print(f"Warning: Tried to get token for slot {slot_idx}, but batch size seems smaller.")

        # Call release_pages if finished this step
        if finished_this_step:
          print(f"    Calling engine to release pages for finished slot {slot_idx}...")
          engine.release_pages(slot=slot_idx)

    else:
      print("  No active streams to generate for.")

    # 2. Check if all streams are finished (can exit loop early)
    if all(streams_finished):
      print("\nAll streams finished generation.")
      break

    # 3. Prefill and Insert new streams if capacity allows
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
            padded_tokens=tokens,
            true_length=true_length,
            rng=rng_prefill,
            slot=next_available_slot,
            request_id=request_id,
        )
        streams_prefilled_count += 1

        # Insert the new prefill
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
      output = tokenizer_model.decode(streams_results[i])
      print(f"Stream {i}: Input=`{text}` -> Output=`{output}`")

      if i == 0:  # Check first stream as an example
        assert output.startswith(
            config.autoregressive_decode_assert
        ), f"Stream {i} generated text mismatch: `{output}` vs expected start `{config.autoregressive_decode_assert}`"
    else:
      print(f"Stream {i}: Was not activated.")


if __name__ == "__main__":
  app.run(main)
