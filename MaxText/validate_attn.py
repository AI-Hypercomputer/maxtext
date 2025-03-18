import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

def fixed_paged_attention(
    query: jnp.ndarray,
    key_pages: jnp.ndarray,
    value_pages: jnp.ndarray,
    page_indices: jnp.ndarray,
    page_token_counts: jnp.ndarray,
    pages_used: jnp.ndarray,
    lengths: jnp.ndarray,
    mask_value: float = -1e7,
    attn_logits_soft_cap: Optional[float] = None,
) -> jnp.ndarray:
    """Corrected paged attention implementation with empty sequence handling."""
    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = key_pages.shape[0]
    tokens_per_page = key_pages.shape[2]

    queries_per_kv = num_heads // num_kv_heads
    scale_factor = 1.0 / jnp.sqrt(head_dim)
    query_reshaped = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
    output = jnp.zeros((batch_size, num_heads, head_dim))

    for b_idx in range(batch_size):
        for kv_idx in range(num_kv_heads):
            q = query_reshaped[b_idx, kv_idx]  # [queries_per_kv, head_dim]

            # Accumulate keys and values
            all_k = []
            all_v = []

            for p_idx in range(pages_used[b_idx]):
                page_index = page_indices[b_idx, p_idx]
                k = key_pages[kv_idx, page_index]  # [tokens_per_page, head_dim]
                v = value_pages[kv_idx, page_index]  # [tokens_per_page, head_dim]

                tokens_in_page = page_token_counts[b_idx, p_idx]
                if tokens_in_page > 0:  # Only add if there are tokens
                    # Slice up to tokens_in_page
                    all_k.append(k[:tokens_in_page])
                    all_v.append(v[:tokens_in_page])

            # Skip if no tokens or length is 0
            if not all_k or lengths[b_idx] == 0:
                # Set zero result for this head
                start_idx = kv_idx * queries_per_kv
                output = output.at[b_idx, start_idx:start_idx + queries_per_kv].set(
                    jnp.zeros((queries_per_kv, head_dim)))
                continue

            # Concatenate along the sequence length dimension (axis 0)
            k_combined = jnp.concatenate(all_k, axis=0)
            v_combined = jnp.concatenate(all_v, axis=0)

            # Perform attention
            scores = jnp.einsum('qd,td->qt', q, k_combined) * scale_factor
            if attn_logits_soft_cap is not None:
                scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)

            exp_scores = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
            head_result = jnp.einsum('qt,td->qd', attention_weights, v_combined) # [queries_per_kv, head_dim]

            start_idx = kv_idx * queries_per_kv
            # Corrected indexing:  head_result has shape [queries_per_kv, head_dim]
            output = output.at[b_idx, start_idx : start_idx + queries_per_kv].set(head_result)

    return output

def naive_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    lengths: jnp.ndarray,
    mask_value: float = -1e7,
    attn_logits_soft_cap: Optional[float] = None,
) -> jnp.ndarray:
    """Naive attention without paging - fixed to match paged attention."""
    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]  # key shape: [batch_size, num_kv_heads, max_len, head_dim]

    queries_per_kv = num_heads // num_kv_heads
    scale_factor = 1.0 / jnp.sqrt(head_dim)
    query_reshaped = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
    output = jnp.zeros((batch_size, num_heads, head_dim))

    for b_idx in range(batch_size):
        for kv_idx in range(num_kv_heads):
            q = query_reshaped[b_idx, kv_idx]  # [queries_per_kv, head_dim]
            
            # Skip if no tokens or length is 0
            if lengths[b_idx] == 0:
                start_idx = kv_idx * queries_per_kv
                output = output.at[b_idx, start_idx:start_idx + queries_per_kv].set(
                    jnp.zeros((queries_per_kv, head_dim)))
                continue
                
            k = key[b_idx, kv_idx, :lengths[b_idx]]  # [length, head_dim]
            v = value[b_idx, kv_idx, :lengths[b_idx]]  # [length, head_dim]

            scores = jnp.einsum('qd,td->qt', q, k) * scale_factor
            if attn_logits_soft_cap is not None:
                scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)

            exp_scores = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
            head_result = jnp.einsum('qt,td->qd', attention_weights, v)  # [queries_per_kv, head_dim]

            start_idx = kv_idx * queries_per_kv
            output = output.at[b_idx, start_idx:start_idx + queries_per_kv].set(head_result)

    return output

def flatten_paged_data(key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths):
    """Flattens paged data with proper batch dimension."""
    num_kv_heads, max_pages, tokens_per_page, head_dim = key_pages.shape
    batch_size = lengths.shape[0]
    max_len = jnp.max(lengths)

    # Create arrays with separate dimensions for batch and sequence
    key_flat = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))
    value_flat = jnp.zeros((batch_size, num_kv_heads, max_len, head_dim))

    for b_idx in range(batch_size):
        token_idx = 0  # Reset for each batch
        
        # Process each page for this batch
        for p_idx in range(pages_used[b_idx]):
            page_index = page_indices[b_idx, p_idx]
            tokens_in_page = page_token_counts[b_idx, p_idx]
            
            # Skip empty pages
            if tokens_in_page == 0:
                continue
                
            # Add tokens from this page to the flattened arrays
            for kv_idx in range(num_kv_heads):
                k_page = key_pages[kv_idx, page_index]
                v_page = value_pages[kv_idx, page_index]
                
                key_flat = key_flat.at[b_idx, kv_idx, token_idx:token_idx + tokens_in_page].set(
                    k_page[:tokens_in_page])
                value_flat = value_flat.at[b_idx, kv_idx, token_idx:token_idx + tokens_in_page].set(
                    v_page[:tokens_in_page])
            
            token_idx += tokens_in_page
            
            # Ensure we don't exceed the actual sequence length
            if token_idx >= lengths[b_idx]:
                break
    
    return key_flat, value_flat

def generate_test_data(key, batch_size, num_heads, num_kv_heads, head_dim, tokens_per_page, max_pages, max_seq_len):
    """Generates random test data."""
    query_key, key_key, value_key, page_key, lengths_key = jax.random.split(key, 5)
    query = jax.random.normal(query_key, (batch_size, num_heads, head_dim))
    key_pages = jax.random.normal(key_key, (num_kv_heads, max_pages, tokens_per_page, head_dim))
    value_pages = jax.random.normal(value_key, (num_kv_heads, max_pages, tokens_per_page, head_dim))
    page_indices = jax.random.randint(page_key, (batch_size, max_pages), 0, max_pages)
    lengths = jax.random.randint(lengths_key, (batch_size,), 1, max_seq_len + 1)

    page_token_counts = jnp.zeros((batch_size, max_pages), dtype=jnp.int32)
    pages_used = jnp.zeros(batch_size, dtype=jnp.int32)
    for b in range(batch_size):
        remaining_tokens = lengths[b]
        p_idx = 0
        while remaining_tokens > 0 and p_idx < max_pages:
            tokens_in_page = min(remaining_tokens, tokens_per_page)
            page_token_counts = page_token_counts.at[b, p_idx].set(tokens_in_page)
            remaining_tokens -= tokens_in_page
            p_idx += 1
        pages_used = pages_used.at[b].set(p_idx)

    return query, key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths

def run_test(key, batch_size, num_heads, num_kv_heads, head_dim, tokens_per_page, max_pages, max_seq_len, attn_logits_soft_cap=None):
    """Runs a single test."""
    print("Running test with:")
    print(f"  batch_size={batch_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  tokens_per_page={tokens_per_page}, max_pages={max_pages}, max_seq_len={max_seq_len}")
    print(f"  attn_logits_soft_cap={attn_logits_soft_cap}")

    (query, key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths) = generate_test_data(
        key, batch_size, num_heads, num_kv_heads, head_dim, tokens_per_page, max_pages, max_seq_len)

    paged_output = fixed_paged_attention(
        query, key_pages, value_pages, page_indices, page_token_counts,
        pages_used, lengths, attn_logits_soft_cap=attn_logits_soft_cap
    )

    key_flat, value_flat = flatten_paged_data(
        key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths
    )

    naive_output = naive_attention(
        query, key_flat, value_flat, lengths, attn_logits_soft_cap=attn_logits_soft_cap
    )

    try:
        np.testing.assert_allclose(naive_output, paged_output, rtol=1e-5, atol=1e-6)
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed:\n{e}")
        print("Naive Output shape:", naive_output.shape)
        print("Paged Output shape:", paged_output.shape)
        print("Different values:", np.sum(np.abs(naive_output - paged_output) > 1e-5))
        raise

def main():
    key = jax.random.PRNGKey(0)

    # Test 1: Basic test
    run_test(key, batch_size=2, num_heads=4, num_kv_heads=2, head_dim=8, tokens_per_page=4, max_pages=3, max_seq_len=10)

    # Test 2: Varying sequence lengths
    run_test(key, batch_size=2, num_heads=2, num_kv_heads=1, head_dim=4, tokens_per_page=4, max_pages=2, max_seq_len=7)

    # Test 3: Single-element batch
    run_test(key, batch_size=1, num_heads=2, num_kv_heads=1, head_dim=4, tokens_per_page=4, max_pages=2, max_seq_len=7)

    # Test 4: Single page
    run_test(key, batch_size=2, num_heads=4, num_kv_heads=2, head_dim=8, tokens_per_page=8, max_pages=1, max_seq_len=8)

    # Test 5: Soft cap
    run_test(key, batch_size=2, num_heads=4, num_kv_heads=2, head_dim=8, tokens_per_page=4, max_pages=3, max_seq_len=10, attn_logits_soft_cap=2.0)

    # Test 6: Empty sequences
    key, subkey = jax.random.split(key)
    query = jax.random.normal(subkey, (2, 4, 8))
    key_pages = jax.random.normal(subkey, (2, 3, 4, 8))
    value_pages = jax.random.normal(subkey, (2, 3, 4, 8))
    page_indices = jnp.array([[0, 1, 2], [0, 1, 2]])
    page_token_counts = jnp.array([[0, 0, 0], [0, 0, 0]])
    pages_used = jnp.array([3, 3])
    lengths = jnp.array([0, 0])

    paged_output = fixed_paged_attention(query, key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths)
    key_flat, value_flat = flatten_paged_data(key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths)
    naive_output = naive_attention(query, key_flat, value_flat, lengths)
    
    try:
        np.testing.assert_allclose(naive_output, paged_output, rtol=1e-5, atol=1e-6)
        print("Edge case test (empty sequences) passed!")
    except AssertionError as e:
        print(f"Edge case test failed: {e}")
        raise

    # Test 7: Single head and kv head
    run_test(key, batch_size=2, num_heads=1, num_kv_heads=1, head_dim=8, tokens_per_page=8, max_pages=3, max_seq_len=15)

    # Test 8: Larger max_seq_len and more pages
    run_test(key, batch_size=4, num_heads=8, num_kv_heads=4, head_dim=32, tokens_per_page=16, max_pages=10, max_seq_len=128)

    # Test 9: Zero tokens on a used page
    key, subkey = jax.random.split(key)
    query = jax.random.normal(subkey, (1, 2, 4))
    key_pages = jax.random.normal(subkey, (1, 2, 4, 4))
    value_pages = jax.random.normal(subkey, (1, 2, 4, 4))
    page_indices = jnp.array([[0, 1]])
    page_token_counts = jnp.array([[3, 0]])  # Page 0 has tokens, page 1 has none
    pages_used = jnp.array([2])
    lengths = jnp.array([3])
    
    paged_output = fixed_paged_attention(query, key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths)
    key_flat, value_flat = flatten_paged_data(key_pages, value_pages, page_indices, page_token_counts, pages_used, lengths)
    naive_output = naive_attention(query, key_flat, value_flat, lengths)
    
    try:
        np.testing.assert_allclose(naive_output, paged_output, rtol=1e-5, atol=1e-6)
        print("Test 9: Zero tokens in a page passed!")
    except AssertionError as e:
        print(f"Test 9 failed: {e}")
        raise

    print("All tests completed successfully!")

if __name__ == "__main__":
    main()