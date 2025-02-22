import jax
import max_utils
import maxengine
import os
import pyconfig
from typing import Sequence
from absl import app

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
    assert tokens_per_page == config.tokens_per_page, \
        f"Key pages tokens_per_page {tokens_per_page} doesn't match config {config.tokens_per_page}"
    assert num_kv_heads == config.num_kv_heads, \
        f"Key pages num_kv_heads {num_kv_heads} doesn't match config {config.num_kv_heads}"
    
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
    num_pages, tokens_per_page, num_kv_heads, head_dim = cache['decoder']['layers_0']['key_pages'].shape
    batch = page_state.sequence_lengths.shape[0]
    print(f"From cache:")
    print(f"- num_pages: {num_pages}")
    print(f"- tokens_per_page: {tokens_per_page}")
    print(f"- num_kv_heads: {num_kv_heads}")
    print(f"- head_dim: {head_dim}")
    print(f"From page_state:")
    print(f"- batch size: {batch}")
    print(f"- pages per group: {page_state.page_map.shape[1]}")

def print_cache_info(prefill_result):
    """Print info about the cache structure"""
    print("\n=== Cache Structure Info ===")
    if "cache" in prefill_result:
        cache = prefill_result["cache"]
        print(f"Cache keys: {cache.keys()}")
        
        if "decoder" in cache:
            decoder = cache["decoder"]
            print("\nDecoder layers:")
            for layer_name, layer_data in decoder.items():
                print(f"\n  {layer_name}:")
                for k, v in layer_data.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        if "page_manager" in cache:
            print("\nPage Manager state present in cache")
    else:
        print("No cache found in prefill result")

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
    tokens, true_length = tokenizer_model.encode(
        text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length]
    )
    
    print("\n=== Tokenization Info ===")
    print(f"Input text: '{text}'")
    print(f"Token shape: {tokens.shape}")
    print(f"True length: {true_length}")
    print(f"Tokens: {tokens[:true_length]}")

    slot = 0

    # Prefill
    rng, rng_prefill = jax.random.split(rng)
    print("\n=== Running Prefill ===")
    prefill_result, first_token, page_state = engine.prefill(
        params=params,
        padded_tokens=tokens,
        true_length=true_length,
        rng=rng_prefill,
        slot=slot,
    )

    print("\n=== Prefill Results ===")
    if config.attention == "paged":
        # page_state = engine.page_manager(
        #     model_mode=None  # Just get current state
        # )
        print_page_allocation_info(slot, true_length, page_state)
        print_cache_info(prefill_result)
        print_paged_attention_info(page_state, prefill_result["cache"])
        batch_size = config.per_device_batch_size * jax.device_count()
        query_shape = (batch_size, 1, config.num_query_heads, config.head_dim) # is this right?
        key_pages_shape = (
            config.num_pages,
            config.tokens_per_page,
            config.num_kv_heads,
            config.head_dim
        )
        
        verify_paged_attention(
            query_shape,
            key_pages_shape,
            page_state,
            config
        )

        

    # print(f"First token shape: {first_token.data.shape}")
    # print(f"First token value: {first_token.data[0,0]}")
    
    # # Try decoding the first token
    # if hasattr(tokenizer_model, 'decode'):
    #     try:
    #         decoded = tokenizer_model.decode(first_token.data[0,0:1])
    #         print(f"Decoded first token: '{decoded}'")
    #     except Exception as e:
    #         print(f"Could not decode first token: {e}")

if __name__ == "__main__":
    app.run(main)