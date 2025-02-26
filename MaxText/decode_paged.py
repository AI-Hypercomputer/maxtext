"""JAX-compatible CLI utility for running inference with paged attention."""

import jax
import jax.numpy as jnp
import max_utils
import maxengine
import os
import pyconfig
from typing import Sequence
from absl import app


def print_state_info(state, prefix=""):
  """Helper to print decode state info."""
  print(f"\n{prefix} Decode State Info:")
  print(f"  Logits shape: {state['logits'].shape}")
  if isinstance(state["cache"], dict):
    print("  Cache structure:")
    if "page_state" in state["cache"]:
      # Print some page_state info
      page_state = state["cache"]["page_state"]
      print(f"    - Page Status shape: {page_state.page_status.shape}")
      print(f"    - Page Map shape: {page_state.page_map.shape}")
      print(f"    - Current Page: {page_state.current_page}")
      print(f"    - Sequence Lengths: {page_state.sequence_lengths}")
  print(f"  Next position: {state['next_pos']}")
  print(f"  Generated tokens: {state['generated_tokens']}")
  print(f"  Latest token: {state['tokens']}")


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


def test_decode_paged_attention(config):
  """Test decode with JAX-compatible paged attention."""
  print("\n=== Starting Decode Paged Attention Test ===")
  engine = maxengine.MaxEngine(config)

  print("\nInitializing parameters...")
  rng = jax.random.PRNGKey(0)
  params = engine.load_params(rng=rng)
  max_utils.print_mem_stats("After loading params")

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert true_length <= config.max_prefill_predict_length, "Too many tokens for prefill"

  print("\nCreating decode state...")
  decode_state = engine.create_decode_state(rng=rng)
  max_utils.print_mem_stats("After creating decode state")

  # Test prefill
  print("\nTesting prefill...")
  slot = 0  # Use first slot

  # Run prefill - now all page state management happens inside the function
  # Important: Capture the returned page state
  decode_state, first_token, updated_page_state = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length, rng=rng, slot=slot
  )

  # Store updated page state in decode_state to maintain pure functional approach
  if updated_page_state is not None:
    decode_state["cache"]["page_state"] = updated_page_state

  print_state_info(decode_state, "After prefill:")
  sampled_tokens_list = []
  sampled_tokens_list.append(first_token)

  # Test generate
  print("\nTesting generate...")
  for i in range(config.max_target_length - true_length):  # Generate remaining tokens
    print(f"\nGeneration step {i}")

    # Run generation step - page state management is handled in pure functions
    # The updated page state is returned as part of decode_state
    decode_state, sampled_tokens = engine.generate(params=params, decode_state=decode_state, rng=rng, slot=slot)

    print_state_info(decode_state, f"After generate step {i}:")
    sampled_tokens_list.append(sampled_tokens)

  # Process results
  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer_model.decode(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert (
        output == config.autoregressive_decode_assert
    ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"

  # Clean up by releasing pages
  print("\nCleaning up resources...")
  if "page_state" in decode_state["cache"]:
    # Release the page group to free resources
    updated_page_state = engine.page_manager.release_page_group(
        state=decode_state["cache"]["page_state"], page_group_id=slot
    )
    # Store the updated state (in a real application, we might want to return this)
    decode_state["cache"]["page_state"] = updated_page_state

  print("\n=== Test Complete ===")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_config(config)
  max_utils.print_system_information()
  test_decode_paged_attention(config)


if __name__ == "__main__":
  app.run(main)
