"""Test script for paged attention."""

import os
from typing import Sequence
import jax
import jax.numpy as jnp
from maxengine import MaxEngine
import max_utils
from max_logging import log
import pyconfig
from absl import app

# Local defaults for paged attention testing


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


def print_state_info(state, prefix=""):
  """Helper to print decode state info."""
  print(f"\n{prefix} Decode State Info:")
  print(f"  Logits shape: {state['logits'].shape}")
  if isinstance(state["cache"], dict):
    print("  Cache structure:")
    if "page_manager" in state["cache"]:
      # Print some page_manager state info
      page_state = state["cache"]["page_manager"]
      print(f"    - Page Status shape: {page_state.page_status.shape}")
      print(f"    - Page Map shape: {page_state.page_map.shape}")
      print(f"    - Current Page: {page_state.current_page}")
    for key in state["cache"]["decoder"].keys():
      print(f"    - {key}:")
      layer_dict = state["cache"]["decoder"][key]
      for k, v in layer_dict.items():
        print(f"      {k}: shape {v.shape}")
  print(f"  Next position: {state['next_pos']}")
  print(f"  Generated tokens: {state['generated_tokens']}")
  print(f"  Latest token: {state['tokens']}")


def test_paged_attention(config):
  """Main test function."""
  print("\n=== Starting Paged Attention Test ===")
  engine = MaxEngine(config)

  print("\nInitializing parameters...")
  rng = jax.random.PRNGKey(0)
  params = engine.load_params(rng=rng)
  max_utils.print_mem_stats("After loading params")

  print("\nCreating decode state...")
  decode_state = engine.create_decode_state(rng=rng)
  max_utils.print_mem_stats("After creating decode state")

  # Test prefill
  print("\nTesting prefill...")
  input_tokens = jnp.ones((32,), dtype=jnp.int32)  # Small test input
  true_length = 16  # Use partial length to test padding handling
  slot = 0  # Use first slot

  decode_state, first_token, page_state = engine.prefill(
      params=params, padded_tokens=input_tokens, true_length=true_length, rng=rng, slot=slot
  )
  print_state_info(decode_state, "After prefill:")

  # Test generate
  print("\nTesting generate...")
  for i in range(3):  # Generate a few tokens
    print(f"\nGeneration step {i}")
    decode_state, result = engine.generate(params=params, decode_state=decode_state, rng=rng, slot=slot)
    print_state_info(decode_state, f"After generate step {i}:")
    decode_state = decode_state  # Update for next iteration

  print("\n=== Test Complete ===")


def main(argv: Sequence[str]) -> None:
  """Main function."""
  # Initialize environment
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Load base config
  pyconfig.initialize(argv)
  config = pyconfig.config

  # Validate final config
  validate_config(config)
  max_utils.print_system_information()

  # Log key config values
  log("Testing with configuration:")
  log(f"  attention: {config.attention}")
  log(f"  num_pages: {config.num_pages}")
  log(f"  tokens_per_page: {config.tokens_per_page}")
  log(f"  max_target_length: {config.max_target_length}")
  log(f"  max_prefill_predict_length: {config.max_prefill_predict_length}")
  log(f"  per_device_batch_size: {config.per_device_batch_size}")
  log(f"  mesh_axes: {config.mesh_axes}")
  log(f"  model_name: {config.model_name}")

  # Run test
  test_paged_attention(config)


if __name__ == "__main__":
  app.run(main)
