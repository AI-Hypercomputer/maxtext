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

"""CLI utility for testing paged attention with layer-specific page state handling"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import max_utils
import maxengine
from inference.page_manager import PageManager, PageState

import os
import pyconfig
import time
import sys
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from typing import Sequence
from absl import app, flags

# Define flags for debugging and visualization
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_tokens_to_generate", 10, "Number of tokens to generate")
flags.DEFINE_boolean("debug", True, "Enable debug output")
flags.DEFINE_boolean("visualize", True, "Create visualizations of page allocations")


# Debug visualization functions
def visualize_page_allocation(page_state, num_layers, num_pages, step, output_dir="./debug_output"):
  """Create a visualization of page allocations across layers."""
  os.makedirs(output_dir, exist_ok=True)

  # Create a matrix showing page allocations across layers
  page_status = page_state.page_status  # [num_layers, num_pages]

  # Create a heatmap
  plt.figure(figsize=(15, 8))
  plt.imshow(page_status, aspect="auto", cmap="Blues")
  plt.colorbar(label="Status (0=free, 1=allocated)")
  plt.xlabel("Page Index")
  plt.ylabel("Layer Index")
  plt.title(f"Page Allocation Status by Layer - Step {step}")

  # Add annotations for page maps for slot 0
  slot = 0
  markers = ["ro", "go", "bo", "mo"]  # Different colors for each slot if needed

  # Plot current page position for each layer
  for layer in range(num_layers):
    current_page = page_state.current_page[layer, slot]
    if current_page >= 0:  # Only show valid pages
      current_pos = page_state.current_page_position[layer, slot]
      plt.plot(current_page, layer, "y*", markersize=10)
      plt.annotate(f"pos: {current_pos}", (current_page, layer), textcoords="offset points", xytext=(0, 10), ha="center")

  # Add all allocated pages for slot 0
  for layer in range(num_layers):
    page_indices = page_state.page_map[layer, slot]
    valid_pages = [p for p in page_indices if p >= 0]
    if valid_pages:
      plt.plot(
          valid_pages,
          [layer] * len(valid_pages),
          markers[slot % len(markers)],
          markersize=4,
          label=f"Slot {slot}" if layer == 0 else "",
      )

  # Add some stats as text
  stats_text = f"Pages allocated: {np.sum(page_status)}/{num_layers * num_pages}\n"
  for layer in range(min(5, num_layers)):  # Show stats for first few layers
    pages_used = np.sum(page_state.num_pages_used[layer])
    stats_text += f"Layer {layer}: {pages_used} pages\n"

  plt.figtext(0.02, 0.02, stats_text, bbox=dict(facecolor="white", alpha=0.8))
  plt.legend()
  plt.tight_layout()
  plt.savefig(f"{output_dir}/page_allocation_step_{step}.png")
  plt.close()


def check_page_state_consistency(page_state, num_layers, step):
  """Verify the consistency of page state across layers."""
  issues_found = False

  # Check for each layer
  for layer_idx in range(num_layers):
    # Check that page_status matches page_map
    allocated_pages = []
    for slot in range(page_state.page_map.shape[1]):
      pages = [p for p in page_state.page_map[layer_idx, slot] if p >= 0]
      allocated_pages.extend(pages)

    # Check for duplicate allocations
    duplicates = set([p for p in allocated_pages if allocated_pages.count(p) > 1])
    if duplicates:
      print(f"ERROR at step {step}: Layer {layer_idx} has duplicate page allocations: {duplicates}")
      issues_found = True

    # Check page_status consistency
    for page in allocated_pages:
      if page_state.page_status[layer_idx, page] != 1:
        print(f"ERROR at step {step}: Layer {layer_idx} has inconsistent state for page {page}")
        print(f"  Page is in page_map but page_status is {page_state.page_status[layer_idx, page]}")
        issues_found = True

    # Check current page is valid
    for slot in range(page_state.current_page.shape[1]):
      current_page = page_state.current_page[layer_idx, slot]
      if current_page >= 0:
        if page_state.page_status[layer_idx, current_page] != 1:
          print(f"ERROR at step {step}: Layer {layer_idx} slot {slot} current_page {current_page} not allocated")
          issues_found = True

  if not issues_found:
    print(f"✓ Page state consistency check passed at step {step}")
  return not issues_found


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )

  # Additional validation for paged attention
  if config.attention == "paged":
    assert hasattr(config, "pagedattn_num_pages"), "Config must include pagedattn_num_pages for paged attention"
    assert hasattr(config, "pagedattn_tokens_per_page"), "Config must include pagedattn_tokens_per_page for paged attention"
    assert hasattr(
        config, "pagedattn_max_pages_per_group"
    ), "Config must include pagedattn_max_pages_per_group for paged attention"
    assert hasattr(config, "pagedattn_max_page_groups"), "Config must include pagedattn_max_page_groups for paged attention"


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Initialize with argv
  config = pyconfig.initialize(argv)

  # Print attention type from config
  is_using_paged = config.attention == "paged"
  print(f"⚙️ Using {'PAGED' if is_using_paged else 'standard dot_product'} attention")

  # Validate the config and print system info
  validate_config(config)
  max_utils.print_system_information()

  # Create engine - it internally initializes page_state if using paged attention
  print("🔧 Initializing engine...")
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)

  # Load model parameters
  print("📥 Loading parameters...")
  params = engine.load_params(rng=rng_load_params)

  # Get tokenizer and encode input text
  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"

  print(f"🔤 Input text: '{text}'")
  print(f"📊 Input tokens: {tokens[:true_length]}")
  print(f"📏 True length: {true_length}")

  slot = 0  # Always use decode batch slot 0.

  # Debug: Check initial page state
  if is_using_paged and FLAGS.debug:
    print("\n📝 INITIAL PAGE STATE:")
    if FLAGS.visualize:
      visualize_page_allocation(engine.page_state, config.num_decoder_layers, config.pagedattn_num_pages, step="initial")
    print(f"  Initial Page Status sum: {np.sum(engine.page_state.page_status)}")
    print(f"  Initial Pages Used (Layer 0): {engine.page_state.num_pages_used[0, 0]}")
    check_page_state_consistency(engine.page_state, config.num_decoder_layers, "initial")

  # Prefill - page state update happens inside the prefill method
  print(f"\n💫 Running prefill...")
  prefill_start = time.time()
  rng, rng_prefill = jax.random.split(rng)
  prefill_result, first_token = engine.prefill(
      params=params,
      padded_tokens=tokens,
      true_length=true_length,
      rng=rng_prefill,
      slot=slot,  # Pass slot for page state management
  )
  prefill_time = time.time() - prefill_start
  print(f"  Prefill time: {prefill_time:.4f}s")

  # Debug: Check page state after prefill
  if is_using_paged and FLAGS.debug:
    print("\n📝 PAGE STATE AFTER PREFILL:")
    if FLAGS.visualize:
      visualize_page_allocation(
          engine.page_state, config.num_decoder_layers, config.pagedattn_num_pages, step="after_prefill"
      )
    # Show allocations for first few layers
    for layer_idx in range(min(3, config.num_decoder_layers)):
      pages_used = engine.page_state.num_pages_used[layer_idx, slot]
      print(f"  Layer {layer_idx}: {pages_used} pages allocated")
      if pages_used > 0:
        page_indices = engine.page_state.page_map[layer_idx, slot]
        valid_pages = [p for p in page_indices[:pages_used] if p >= 0]
        print(f"    Allocated pages: {valid_pages}")

    check_page_state_consistency(engine.page_state, config.num_decoder_layers, "after_prefill")

  # Insert prefill result into decode state
  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode, page_state=engine.page_state)  # Pass page_state
  decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  # Generate tokens - page state updates happen inside the generate method
  num_tokens = min(FLAGS.num_tokens_to_generate, config.max_target_length - config.max_prefill_predict_length)
  print(f"\n🚀 Generating {num_tokens} tokens...")
  sampled_tokens_list = []
  sampled_tokens_list.append(first_token)
  generation_times = []

  for i in range(num_tokens):
    rng, rng_generate = jax.random.split(rng)

    # Generate token - page state update happens inside the method
    generate_start = time.time()
    decode_state, sampled_tokens = engine.generate(
        params, decode_state, rng=rng_generate, slot=slot  # Pass slot for page state management
    )
    generation_time = time.time() - generate_start
    generation_times.append(generation_time)

    sampled_tokens_list.append(sampled_tokens)

    # Token info
    token_id = sampled_tokens.get_result_at_slot(slot).tokens.item()
    token_text = tokenizer_model.decode([token_id])
    print(f"  Token {i+1}: '{token_text}' (id={token_id}) - {generation_time:.4f}s")

    # Debug: Check page state after token generation
    if is_using_paged and FLAGS.debug and (i == 0 or i == num_tokens - 1 or i % 3 == 0):
      print(f"\n📝 PAGE STATE AFTER GENERATING TOKEN {i+1}:")

      if FLAGS.visualize:
        visualize_page_allocation(
            engine.page_state, config.num_decoder_layers, config.pagedattn_num_pages, step=f"token_{i+1}"
        )

      # Show current page and position for first few layers
      for layer_idx in range(min(3, config.num_decoder_layers)):
        current_page = engine.page_state.current_page[layer_idx, slot]
        current_pos = engine.page_state.current_page_position[layer_idx, slot]
        print(f"  Layer {layer_idx}: current_page={current_page}, position={current_pos}")

      check_page_state_consistency(engine.page_state, config.num_decoder_layers, f"token_{i+1}")

  # Process final results
  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer_model.decode(results)

  # Print results
  print(f"\n✨ GENERATION COMPLETE")
  print(f"Input: '{text}'")
  print(f"Output: '{output}'")
  print(f"Average generation time: {sum(generation_times)/len(generation_times):.4f}s per token")


if __name__ == "__main__":
  app.run(main)
