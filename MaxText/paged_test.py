import jax
import jax.numpy as jnp
import max_utils
import maxengine
import os
import pyconfig
from typing import Sequence
from absl import app
import common_types
from page_manager import PageState
from layers.attentions import PagedAttentionOp


def verify_cache_contents(prefill_result):
    """Verify that the KV cache contains non-zero elements after prefill."""
    print("\nVerifying KV cache contents:")
    print(f"Prefill result keys: {prefill_result.keys()}")
    
    if "cache" not in prefill_result:
        print("ERROR: No cache found in prefill result")
        return False
    
    cache = prefill_result["cache"]
    print(f"Cache keys: {cache.keys()}")
    
    if "decoder" not in cache:
        print("ERROR: No decoder found in cache")
        return False
    
    decoder_cache = cache["decoder"]
    print(f"Decoder cache keys: {decoder_cache.keys()}")
    
    has_data = False
    
    for layer_id in range(3):  # Check first 3 layers
        layer_key = f"layers_{layer_id}"
        if layer_key in decoder_cache:
            layer_cache = decoder_cache[layer_key]
            print(f"Layer {layer_id} cache keys: {layer_cache.keys()}")
            
            # Navigate through the nested structure
            if "self_attention" in layer_cache:
                attn_cache = layer_cache["self_attention"]
                print(f"  self_attention keys: {attn_cache.keys()}")
                
                if "attention_op" in attn_cache:
                    op_cache = attn_cache["attention_op"]
                    print(f"    attention_op keys: {op_cache.keys()}")
                    
                    # Check for key_pages and value_pages
                    if "key_pages" in op_cache:
                        key_pages = op_cache["key_pages"]
                        print(f"      key_pages shape: {key_pages.shape}")
                        print(f"      key_pages dtype: {key_pages.dtype}")
                        
                        # Count non-zero elements
                        num_nonzero = jnp.sum(jnp.any(key_pages != 0, axis=(2, 3)))
                        print(f"      Layer {layer_id}: {num_nonzero} pages contain non-zero key data")
                        has_data = has_data or (num_nonzero > 0)
                        
                        # Also print a sample value
                        print(f"      Sample value: {key_pages[0, 0, 0, 0]}")
                    else:
                        print(f"      ERROR: No key_pages found in attention_op")
                else:
                    print(f"    ERROR: No attention_op found in self_attention")
            else:
                print(f"  ERROR: No self_attention found in layer {layer_id}")
    
    if not has_data:
        print("ERROR: No key/value data found in any pages after prefill!")
    else:
        print("SUCCESS: Found non-zero key/value data in cache")
    
    return has_data


def verify_page_content_bounds(key_pages, page_idx, expected_tokens):
  """Verify only expected positions contain non-zero values"""
  nonzero_positions = jnp.sum(jnp.any(key_pages[page_idx] != 0, axis=(1, 2)))
  print(f"Page {page_idx}: {nonzero_positions} positions contain data")


def verify_page_memory_access(prefill_result, page_state, slot, true_length, config):
  """Verify memory access patterns and content for paged attention.

  Args:
      prefill_result: Dict containing cache with decoder key/value pages
      page_state: PageState with allocation info
      slot: Current page group being checked
      true_length: Actual sequence length
      config: Config object with attention params
  """
  print("\n=== Memory Access Verification ===")

  # Verify basic parameters are valid
  num_pages_needed = (true_length + config.tokens_per_page - 1) // config.tokens_per_page
  print(f"\nSequence requires {num_pages_needed} pages:")
  print(f"- True length: {true_length}")
  print(f"- Tokens per page: {config.tokens_per_page}")

  # Access decoder cache and verify structure
  cache = prefill_result["cache"]["decoder"]
  if not isinstance(cache, dict) or not any(k.startswith("layers_") for k in cache.keys()):
    raise ValueError(f"Unexpected cache structure: {cache.keys()}")

  # Print key shape info for validation
  print("\nShape verification:")
  print(f"- page_status: {page_state.page_status.shape}")
  print(f"- page_map: {page_state.page_map.shape}")
  print(f"- key_pages (layer 0): {cache['layers_0']['key_pages'].shape}")
  print(f"- value_pages (layer 0): {cache['layers_0']['value_pages'].shape}")

  # Identify allocated pages for this slot
  mask = page_state.page_map[slot] >= 0
  used_pages = page_state.page_map[slot][mask]
  print(f"\nAllocated pages for slot {slot}: {used_pages}")

  # Verify each layer's memory content
  for layer_id in range(config.num_decoder_layers):
    key_pages = cache[f"layers_{layer_id}"]["key_pages"]
    value_pages = cache[f"layers_{layer_id}"]["value_pages"]

    print(f"\nLayer {layer_id} verification:")
    # Check allocated pages have non-zero content
    for page_idx in used_pages:
      key_nonzero = jnp.any(key_pages[page_idx] != 0)
      value_nonzero = jnp.any(value_pages[page_idx] != 0)
      print(f"  Page {page_idx}:")
      print(f"    - Has key content: {key_nonzero}")
      print(f"    - Has value content: {value_nonzero}")
      expected_tokens = min(config.tokens_per_page, true_length - page_idx * config.tokens_per_page)
      verify_page_content_bounds(key_pages, page_idx, expected_tokens)

    # Sample some content for visual inspection
    if len(used_pages) > 0:
      first_page = used_pages[0]
      print(f"\n  First allocated page ({first_page}) content sample:")
      print(f"    Keys (first row): {key_pages[first_page][0][0]}")
      print(f"    Values (first row): {value_pages[first_page][0][0]}")


def verify_paged_attention_correctness():
    """Test to verify that paged attention produces the same results as standard attention."""
    print("\n=== Testing Paged Attention Correctness ===")
    
    # Setup test parameters
    batch_size = 1
    seq_len = 16
    num_query_heads = 8
    num_kv_heads = 4
    head_dim = 32
    num_pages = 4
    tokens_per_page = 4
    
    # Create random test data with a fixed seed for reproducibility
    rng = jax.random.PRNGKey(42)
    rng, q_rng, k_rng, v_rng = jax.random.split(rng, 4)
    
    query = jax.random.normal(q_rng, (batch_size, seq_len, num_query_heads, head_dim))
    key = jax.random.normal(k_rng, (batch_size, seq_len, num_kv_heads, head_dim))
    value = jax.random.normal(v_rng, (batch_size, seq_len, num_kv_heads, head_dim))
    
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.dtype = jnp.float32
            self.num_pages = num_pages
            self.tokens_per_page = tokens_per_page
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.num_query_heads = num_query_heads
            self.num_decoder_layers = 1
            self.max_pages_per_group = 10
            self.attention = "paged"
            self.max_prefill_predict_length = 128
            self.max_target_length = 256
    
    config = MockConfig()
    
    # Create page state for a batch_size=1 scenario with seq_len tokens
    # Configure a simple linear page map (page 0->0, 1->1, etc.)
    num_logical_pages = (seq_len + tokens_per_page - 1) // tokens_per_page
    
    # Create page state
    page_status = jnp.zeros((1, num_pages), dtype=jnp.int32)
    page_map = jnp.zeros((1, 2, 10), dtype=jnp.int32) - 1  # Initialize with -1
    
    # Set up page mapping (physical pages for the sequence)
    for i in range(min(num_logical_pages, 10)):
        page_map = page_map.at[0, 0, i].set(i % num_pages)
    
    sequence_lengths = jnp.array([[seq_len, 0]], dtype=jnp.int32)
    num_pages_used = jnp.array([[num_logical_pages, 0]], dtype=jnp.int32)
    current_page = jnp.array([[num_logical_pages - 1, -1]], dtype=jnp.int32)
    current_page_position = jnp.array([[(seq_len - 1) % tokens_per_page, 0]], dtype=jnp.int32)
    
    page_state = PageState(
        page_status=page_status,
        page_map=page_map,
        sequence_lengths=sequence_lengths,
        num_pages_used=num_pages_used,
        current_page=current_page,
        current_page_position=current_page_position,
    )
    
    print(f"Page State configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tokens per page: {tokens_per_page}")
    print(f"  Logical pages needed: {num_logical_pages}")
    print(f"  Page map for first few pages: {page_map[0, 0, :num_logical_pages]}")
    
    # Initialize paged attention op
    paged_op = PagedAttentionOp(
        mesh=None,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        kv_head_dim_size=head_dim,
        config=config,
        output_dim=num_query_heads * head_dim,
        dtype=jnp.float32,
    )
    
    # Create cache with pre-populated key/value pages
    key_pages = jnp.zeros((num_pages, tokens_per_page, num_kv_heads, head_dim))
    value_pages = jnp.zeros((num_pages, tokens_per_page, num_kv_heads, head_dim))
    
    # Fill pages with key/value data
    for page_idx in range(min(num_pages, num_logical_pages)):
        start_idx = page_idx * tokens_per_page
        end_idx = min(start_idx + tokens_per_page, seq_len)
        tokens_in_page = end_idx - start_idx
        
        if tokens_in_page > 0:
            key_pages = key_pages.at[page_idx, :tokens_in_page].set(key[0, start_idx:end_idx])
            value_pages = value_pages.at[page_idx, :tokens_in_page].set(value[0, start_idx:end_idx])
    
    print(f"Initialized {num_pages} key/value pages")
    for p in range(num_pages):
        key_nonzero = jnp.any(key_pages[p] != 0)
        print(f"  Page {p}: contains data = {key_nonzero}")
    
    # Define variables for PagedAttentionOp
    variables = {
        "cache": {
            "key_pages": key_pages,
            "value_pages": value_pages,
        },
        "params": {}
    }
    
    # Create reference attention function (standard MHA)
    def compute_reference_attention(q, k, v):
        # q: [batch, seq, heads, dim]
        # k, v: [batch, seq, kv_heads, dim]
        
        # Handle grouped-query attention
        if num_query_heads > num_kv_heads:
            heads_ratio = num_query_heads // num_kv_heads
            # Repeat k and v for each query group
            k = jnp.repeat(k, heads_ratio, axis=2)
            v = jnp.repeat(v, heads_ratio, axis=2)
        
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(head_dim)
        
        # Reshape for batch matrix multiply
        batch, seq, heads, dim = q.shape
        q_reshaped = q.reshape(batch * heads, seq, dim)
        k_reshaped = k.reshape(batch * heads, seq, dim)
        v_reshaped = v.reshape(batch * heads, seq, dim)
        
        # Compute attention scores
        scores = jnp.matmul(q_reshaped, jnp.transpose(k_reshaped, (0, 2, 1))) * scale
        
        # Apply causal mask
        mask = jnp.triu(jnp.ones((seq, seq)) * (-1e10), k=1)
        scores = scores + mask
        
        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention weights
        attn_output = jnp.matmul(attn_weights, v_reshaped)
        
        # Reshape back
        return attn_output.reshape(batch, seq, heads, dim)
    
    # Compute reference output using standard attention
    reference_output = compute_reference_attention(query, key, value)
    
    print("\nComputing paged attention output...")
    
    # Run paged attention with the initialized pages
    paged_output, updated_vars = paged_op.apply(
        variables,
        query,
        key,
        value,
        None,  # decoder_segment_ids
        common_types.MODEL_MODE_PREFILL,
        page_state=page_state,
        page_group_id=0,
        true_length=seq_len,
        layer_id=0,
        mutable=["cache"]
    )
    
    # Check that output shape matches
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Paged output shape: {paged_output.shape}")
    
    if reference_output.shape != paged_output.shape:
        print("ERROR: Output shapes don't match!")
        return False
    
    # Compare outputs
    max_abs_diff = jnp.max(jnp.abs(reference_output - paged_output))
    mean_abs_diff = jnp.mean(jnp.abs(reference_output - paged_output))
    print(f"Maximum absolute difference: {max_abs_diff}")
    print(f"Mean absolute difference: {mean_abs_diff}")
    
    # Check if outputs are close enough (allow some floating point precision differences)
    is_close = max_abs_diff < 1e-3
    print(f"Outputs match: {is_close}")
    
    # Now test autoregressive generation
    print("\nTesting autoregressive generation...")
    
    # Get a single token query for the next position
    ar_query = jnp.zeros((batch_size, 1, num_query_heads, head_dim))
    ar_key = jnp.zeros((batch_size, 1, num_kv_heads, head_dim))
    ar_value = jnp.zeros((batch_size, 1, num_kv_heads, head_dim))
    
    # Fill with random data
    ar_query = ar_query.at[0, 0].set(jax.random.normal(jax.random.PRNGKey(100), (num_query_heads, head_dim)))
    ar_key = ar_key.at[0, 0].set(jax.random.normal(jax.random.PRNGKey(101), (num_kv_heads, head_dim)))
    ar_value = ar_value.at[0, 0].set(jax.random.normal(jax.random.PRNGKey(102), (num_kv_heads, head_dim)))
    
    # Update page state for autoregressive step
    ar_page_state = PageState(
        page_status=page_state.page_status,
        page_map=page_state.page_map,
        sequence_lengths=page_state.sequence_lengths,
        num_pages_used=page_state.num_pages_used,
        current_page=page_state.current_page,
        current_page_position=page_state.current_page_position,
    )
    
    # Run autoregressive step
    ar_paged_output, ar_updated_vars = paged_op.apply(
        variables,
        ar_query,
        ar_key,
        ar_value,
        None,  # decoder_segment_ids
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        page_state=ar_page_state,
        page_group_id=0,
        layer_id=0,
        mutable=["cache"]
    )
    
    # Check that the page state was updated
    if "updated_page_state" in ar_updated_vars["cache"]:
        updated_page_state = ar_updated_vars["cache"]["updated_page_state"]
        new_seq_len = updated_page_state.sequence_lengths[0, 0]
        print(f"Sequence length after autoregressive step: {new_seq_len}")
        seq_len_increased = new_seq_len > seq_len
        print(f"Sequence length increased: {seq_len_increased}")
    else:
        print("WARNING: No updated page state found in autoregressive output")
        seq_len_increased = False
    
    # Check that key/value pages were updated
    if "key_pages" in ar_updated_vars["cache"]:
        new_key_pages = ar_updated_vars["cache"]["key_pages"]
        
        # Get the expected physical page and position
        expected_page = (seq_len // tokens_per_page) % num_pages
        expected_pos = seq_len % tokens_per_page
        
        # Check if the new token was stored
        token_stored = jnp.any(new_key_pages[expected_page, expected_pos] != 0)
        print(f"New token stored in key_pages at page {expected_page}, position {expected_pos}: {token_stored}")
    else:
        print("WARNING: No updated key_pages found in autoregressive output")
        token_stored = False
    
    # Overall test success
    test_success = is_close and seq_len_increased and token_stored
    if test_success:
        print("\nPaged attention implementation is working correctly!")
    else:
        print("\nPaged attention implementation has issues that need to be fixed:")
        if not is_close:
            print("- Attention outputs don't match reference implementation")
        if not seq_len_increased:
            print("- Sequence length not properly updated in autoregressive mode")
        if not token_stored:
            print("- New token not properly stored in key/value pages")
    
    return test_success


def verify_paged_attention(query_shape, key_pages_shape, page_state, config):
  """Verify paged attention shapes and parameters."""
  print("\n=== Paged Attention Verification ===")
  print("Expected shapes:")
  print(f"query: {query_shape}")
  print(f"key_pages: {key_pages_shape}")
  print(f"sequence_lengths: {page_state.sequence_lengths.shape}")
  print(f"page_map: {page_state.page_map.shape}")

  print("\nConfiguration:")
  print(f"num_kv_heads: {config.num_kv_heads}")
  print(f"tokens_per_page: {config.tokens_per_page}")

  # Verify shape compatibility
  batch_size = page_state.sequence_lengths.shape[0]
  assert query_shape[0] == batch_size, f"Query batch size {query_shape[0]} doesn't match page state batch size {batch_size}"

  num_pages, tokens_per_page, num_kv_heads, head_dim = key_pages_shape
  assert (
      tokens_per_page == config.tokens_per_page
  ), f"Key pages tokens_per_page {tokens_per_page} doesn't match config {config.tokens_per_page}"
  assert (
      num_kv_heads == config.num_kv_heads
  ), f"Key pages num_kv_heads {num_kv_heads} doesn't match config {config.num_kv_heads}"

  print("\nVerification passed!")


def print_paged_attention_info(page_state, cache):
  """Print info relevant to paged attention operation"""
  print("\n=== Paged Attention Info ===")
  print("Cache Data:")
  print(f"- Key pages shape: {cache['decoder']['layers_0']['key_pages'].shape}")
  print(f"- Value pages shape: {cache['decoder']['layers_0']['value_pages'].shape}")

  print("\nPage State:")
  print(f"- Sequence lengths: {page_state.sequence_lengths.shape}")
  print(f"- Page map: {page_state.page_map.shape}")

  print("\nExpected by PagedAttentionOp:")
  print("- query: [batch, seq_len, num_heads, head_dim]")
  print("- key_pages: [num_pages, tokens_per_page, num_kv_heads, head_dim]")
  print("- sequence_lengths: [batch]")
  print("- page_map: [batch, max_pages_per_group]")

  # Add explicit verification
  print("\nShape Verification:")
  num_pages, tokens_per_page, num_kv_heads, head_dim = cache["decoder"]["layers_0"]["key_pages"].shape
  batch = page_state.sequence_lengths.shape[0]
  print(f"From cache:")
  print(f"- num_pages: {num_pages}")
  print(f"- tokens_per_page: {tokens_per_page}")
  print(f"- num_kv_heads: {num_kv_heads}")
  print(f"- head_dim: {head_dim}")
  print(f"From page_state:")
  print(f"- batch size: {batch}")
  print(f"- pages per group: {page_state.page_map.shape[1]}")


def test_dynamic_prefill():
    """Test the dynamic version of prefill KV cache population."""
    print("\n=== Testing Dynamic Prefill KV Cache Population ===")
    
    config = pyconfig.config
    engine = maxengine.MaxEngine(config)
    
    # Load parameters
    params = engine.load_params()
    
    # Create test tokens
    text = "This is a test sequence for paged attention."
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, _ = tokenizer.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
    
    # Run prefill with dynamic_kv_cache=True
    true_length = 20
    slot = 0
    
    prefill_result, _, page_state = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        slot=slot,
        rng=jax.random.PRNGKey(0),
        dynamic_kv_cache=True,  # Use dynamic implementation
    )
    
    # Verify results
    print(f"Prefill completed successfully with dynamic KV cache")
    
    # Check for non-zero values in key/value pages
    for layer_id in range(config.num_decoder_layers):
        layer_key = f"layers_{layer_id}"
        if layer_key in prefill_result["cache"]["decoder"]:
            key_pages = prefill_result["cache"]["decoder"][layer_key]["key_pages"]
            value_pages = prefill_result["cache"]["decoder"][layer_key]["value_pages"]
            
            key_nonzero = jnp.any(key_pages != 0)
            value_nonzero = jnp.any(value_pages != 0)
            
            print(f"Layer {layer_id}: Key pages non-zero: {key_nonzero}, Value pages non-zero: {value_nonzero}")
    
    return prefill_result, page_state
  
def test_paged_attention_end_to_end():
    """Test for end-to-end paged attention with prefill and generation."""
    print("\n=== Testing Paged Attention End-to-End ===")
    
    config = pyconfig.config
    engine = maxengine.MaxEngine(config)
    
    # Load parameters
    params = engine.load_params()
    
    # Create test tokens
    text = "I love to"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
    
    # Run prefill
    slot = 0
    
    print("\n1. Running Prefill")
    prefill_result, first_token, page_state = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        slot=slot,
        rng=jax.random.PRNGKey(0),
    )
    
    print(f"First token: {first_token.data[0, 0]}")
    if page_state is not None:
        print(f"Sequence length after prefill: {page_state.sequence_lengths[0, slot]}")
    
    # Verify cache contents outside of JIT-compiled function
    has_kv_data = verify_cache_contents(prefill_result)
    
    if not has_kv_data:
        print("No KV data found in cache - aborting test")
        return None
    
    # Run a few generate steps
    print("\n2. Running Generate Steps")
    decode_state = prefill_result
    generated_tokens = [first_token.data[0, 0]]
    
    for i in range(3):  # Generate 3 more tokens
        print(f"\nGenerate step {i+1}")
        decode_state, result = engine.generate(
            params=params,
            decode_state=decode_state,
            slot=slot,
            rng=jax.random.PRNGKey(i+1),
        )
        
        # Get the token and append to list
        token = result.data[0, 0]
        generated_tokens.append(token)
        print(f"Generated token: {token}")
        
        # Check updated page state
        if "page_manager" in decode_state["cache"]:
            updated_page_state = decode_state["cache"]["page_manager"]
            current_seq_len = updated_page_state.sequence_lengths[0, slot]
            print(f"Sequence length after generation: {current_seq_len}")
            
            # Verify sequence length is increasing
            expected_length = true_length + i + 1
            if current_seq_len != expected_length:
                print(f"WARNING: Sequence length {current_seq_len} doesn't match expected {expected_length}")
        else:
            print("WARNING: No page_manager found in decode_state")
    
    # Try to decode the generated text
    print("\n3. Generated Text:")
    try:
        text = tokenizer.decode(jnp.array(generated_tokens))
        print(f"Decoded text: {text}")
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        print(f"Raw tokens: {generated_tokens}")
    
    # Verify that the sequence of tokens makes sense
    print("\n4. Verifying token sequence:")
    are_all_tokens_reasonable = all(0 < token < config.vocab_size for token in generated_tokens)
    print(f"All tokens within vocabulary range: {are_all_tokens_reasonable}")
    
    print("\n5. End-to-End Summary:")
    success = has_kv_data and are_all_tokens_reasonable
    if success:
        print("✅ Paged attention end-to-end test PASSED")
    else:
        print("❌ Paged attention end-to-end test FAILED")
        
    return generated_tokens, decode_state


def print_page_allocation_info(slot, true_length, page_state):
  """Print page allocation info after JIT compilation is complete"""
  print("\n=== Page Allocation Info (Post-JIT) ===")
  print(f"Requested allocation:")
  print(f"- Slot (page group): {slot}")
  print(f"- True length: {true_length}")
  print("\nPage status:")
  print(f"- Pages used for this slot: {page_state.sequence_lengths}")
  print(f"- Allocated page count: {page_state.num_pages_used}")
  print(f"- Current active page: {page_state.current_page}")
  print(f"- Position in current page: {page_state.current_page_position}")
  print("\nPage map details:")
  print(f"- Page map shape: {page_state.page_map.shape}")
  print(f"- Pages for this slot: {page_state.page_map[slot]}")
  print(f"- Free pages count: {jax.numpy.sum(page_state.page_status == 0)}")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  pyconfig.initialize(argv)
  config = pyconfig.config
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)

  # Print initial configuration
  print("\n=== Configuration ===")
  print(f"Attention type: {config.attention}")
  if config.attention == "paged":
    print(f"Number of pages: {config.num_pages}")
    print(f"Tokens per page: {config.tokens_per_page}")
    print(f"Pages per compute block: {config.pages_per_compute_block}")
    print(f"Max prefill length: {config.max_prefill_predict_length}")
    print(f"Max target length: {config.max_target_length}")

  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)

  # Load parameters
  print("\n=== Loading Parameters ===")
  params = engine.load_params(rng_load_params)
  print(f"Parameter structure: {jax.tree_util.tree_structure(params)}")

  # Create decode state
  rng, rng_decode = jax.random.split(rng)
  decode_state = engine.create_decode_state(rng_decode)

  # Tokenize input
  text = "Short test"
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])

  print("\n=== Tokenization Info ===")
  print(f"Input text: '{text}'")
  print(f"Token shape: {tokens.shape}")
  print(f"True length: {true_length}")
  print(f"Tokens: {tokens[:true_length]}")

  slot = 0
  true_length = 72

  # Prefill
  rng, rng_prefill = jax.random.split(rng)
  # print("\n=== Running Prefill ===")
  # prefill_result, first_token, page_state = engine.prefill(
  #     params=params,
  #     padded_tokens=tokens,
  #     true_length=true_length,
  #     rng=rng_prefill,
  #     slot=slot,
  # )

  # print("\n=== Prefill Results ===")
  # if config.attention == "paged":
  #   # page_state = engine.page_manager(
  #   #     model_mode=None  # Just get current state
  #   # )
  #   print(f"Prefill Result Cache Structure: {jax.tree_util.tree_structure(prefill_result['cache'])}")
  #   print_page_allocation_info(slot, true_length, page_state)
  #   print_cache_info(prefill_result)
  #   print_paged_attention_info(page_state, prefill_result["cache"])
  #   batch_size = config.per_device_batch_size * jax.device_count()
  #   query_shape = (batch_size, 1, config.num_query_heads, config.head_dim)  # is this right?
  #   key_pages_shape = (config.num_pages, config.tokens_per_page, config.num_kv_heads, config.head_dim)

  #   verify_paged_attention(query_shape, key_pages_shape, page_state, config)

  #   print(f"First key page (layer 0): {prefill_result['cache']['decoder']['layers_0']['key_pages'][0, 0]}")
  #   print(f"First value page (layer 0): {prefill_result['cache']['decoder']['layers_0']['value_pages'][0, 0]}")
  #   print(f"First key page (layer 0): {prefill_result['cache']['decoder']['layers_0']['key_pages'][0, 1]}")
  #   print(f"First value page (layer 0): {prefill_result['cache']['decoder']['layers_0']['value_pages'][0, 1]}")
  #   print("\n=== Verification: Step 3 (Prefill - All Tokens, First Page) ===")
  #   print(f"True Length: {true_length}")

  # test_dynamic_prefill()
  # verify_paged_attention_correctness()
  test_paged_attention_end_to_end()

    # num_layers = config.num_decoder_layers
    # for layer_id in range(num_layers):
    #     print(f"\nLayer {layer_id}:")
    #     key_pages = prefill_result['cache']['decoder'][f'layers_{layer_id}']['key_pages']
    #     value_pages = prefill_result['cache']['decoder'][f'layers_{layer_id}']['value_pages']

    #     print(f"  Key Pages (Page 0, first {true_length+1} tokens):")
    #     print(key_pages[0, :true_length+1, :, :])

    #     print(f"  Value Pages (Page 0, first {true_length+1} tokens):")
    #     print(value_pages[0, :true_length+1, :, :])
    # verify_page_memory_access(prefill_result, page_state, slot, true_length, config)


if __name__ == "__main__":
  app.run(main)
