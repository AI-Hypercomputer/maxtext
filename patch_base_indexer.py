import re

with open('src/maxtext/layers/attention_mla.py', 'r') as f:
    content = f.read()

base_indexer_code = """
class BaseIndexer(nnx.Module):
  \"\"\"Base Indexer for Sparse Attention.\"\"\"

  def __init__(
      self,
      config: Any,
      kernel_init: NdInitializer,
      quant: Optional[Quant] = None,
      weights_proj_in_features: Optional[int] = None,
      weights_proj_dtype: Optional[DType] = None,
      weights_proj_quant: Optional[Quant] = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.indexer_topk = config.indexer_topk
    self.q_lora_rank = config.q_lora_rank
    self.softmax_scale = self.head_dim**-0.5
    self.weights_scaling = self.n_heads**-0.5  # Optional, used mostly by V4

    # Query Projection: Latent Query -> Indexer Query
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    wp_in_features = weights_proj_in_features if weights_proj_in_features is not None else config.emb_dim
    wp_dtype = weights_proj_dtype if weights_proj_dtype is not None else self.dtype

    # Projection: Input -> Importance Weights for Heads
    self.weights_proj = DenseGeneral(
        in_features_shape=wp_in_features,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "q_heads"),
        dtype=wp_dtype,
        weight_dtype=wp_dtype,
        quant=weights_proj_quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

  def prepare_query(self, low_rank_q: jnp.ndarray, inputs_positions: jnp.ndarray, apply_partial_rope) -> jnp.ndarray:
    bsz, seqlen, _ = low_rank_q.shape
    q = self.wq_b(low_rank_q)
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)
    q = apply_partial_rope(q, inputs_positions=inputs_positions)
    return q

  def compute_topk(self, q: jnp.ndarray, k: jnp.ndarray, inputs_q: jnp.ndarray, mask: Optional[jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    weights = self.weights_proj(inputs_q)
    weights = weights * self.weights_scaling * self.softmax_scale
    indexer_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision)

    if mask is not None:
      indexer_score += mask

    _, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)
    return topk_indices, indexer_score
"""

content = content.replace("class Indexer(nnx.Module):", base_indexer_code + "\n\nclass Indexer(BaseIndexer):")

with open('src/maxtext/layers/attention_mla.py', 'w') as f:
    f.write(content)
