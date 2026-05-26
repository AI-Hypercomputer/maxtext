import re

with open('src/maxtext/layers/attention_mla.py', 'r') as f:
    content = f.read()

old_query_proc = """    # Query Processing: Project from Latent low_rank_q
    q = self.wq_b(low_rank_q)  # [b, t, q_lora_rank] -> [b, t, h * d]
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)  # [b, t, h, d]
    q = self.apply_partial_rope(q, inputs_positions=inputs_positions)"""

new_query_proc = """    # Query Processing: Project from Latent low_rank_q
    q = self.prepare_query(low_rank_q, inputs_positions, self.apply_partial_rope)"""

content = content.replace(old_query_proc, new_query_proc)


old_compute_scores = """    # Compute Index Scores
    # QK product: relu(q @ k.T), [b, t, s, h]
    # Similar to MQA, each key is shared by h query head
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    # Compute head weights: project from input, [b, t, embed_dim] -> [b, t, h]
    weights = self.weights_proj(inputs_q)
    # Weights scaling affect indexer_score, but does not affect topk_indices. Keep scaling for numerical stability.
    # https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L478-L480
    weights = weights * (self.n_heads**-0.5) * self.softmax_scale
    # Aggregate head-wise logits: logits @ weights
    indexer_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision)  # [b, t, s]

    internal_padding_mask = None
    if cached_s is not None:
      # cached_s marks valid tokens from the original prefill step and all subsequent AR steps
      internal_padding_mask = jnp.where(cached_s > 0, 0.0, DEFAULT_MASK_VALUE)
      indexer_score += internal_padding_mask[:, None, :]

    # Apply attention mask before TopK
    if attention_mask is not None:
      indexer_score += attention_mask

    # TopK selection based on index score
    _, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)  # topk_indices [b, t, k]"""

new_compute_scores = """    # Compute Index Scores
    internal_padding_mask = None
    combined_mask = None
    if cached_s is not None:
      # cached_s marks valid tokens from the original prefill step and all subsequent AR steps
      internal_padding_mask = jnp.where(cached_s > 0, 0.0, DEFAULT_MASK_VALUE)
      combined_mask = internal_padding_mask[:, None, :]

    if attention_mask is not None:
      if combined_mask is None:
        combined_mask = attention_mask
      else:
        combined_mask += attention_mask

    topk_indices, indexer_score = self.compute_topk(q, k, inputs_q, combined_mask)"""

content = content.replace(old_compute_scores, new_compute_scores)

with open('src/maxtext/layers/attention_mla.py', 'w') as f:
    f.write(content)

