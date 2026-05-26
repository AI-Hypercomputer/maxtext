import re

with open('src/maxtext/layers/attention_mla.py', 'r') as f:
    content = f.read()

indexer_init_old = """  def __init__(
      self,
      config: Any,
      rotary_embedding,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.rotary_embedding = rotary_embedding
    self.quant = quant
    self.kernel_init = kernel_init
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.max_target_length = config.max_target_length

    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.indexer_topk = config.indexer_topk
    self.emb_dim = config.emb_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.q_lora_rank = config.q_lora_rank
    # scale head weights for numerical stability
    self.softmax_scale = self.head_dim**-0.5

    # Query Projection: Latent Query -> Indexer Query
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Projection: Input -> Shared Indexer Key
    self.wk = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Normalization with Bias
    self.k_norm = nnx.LayerNorm(num_features=self.head_dim, use_bias=True, dtype=self.weight_dtype, rngs=rngs)

    # Projection: Input -> Importance Weights for Heads
    # deepseek3.2 enforces FP32 and does not quantize, for precision and stability.
    self.weights_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "q_heads"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        quant=None,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )"""

indexer_init_new = """  def __init__(
      self,
      config: Any,
      rotary_embedding,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    super().__init__(
        config=config,
        kernel_init=kernel_init,
        quant=quant,
        weights_proj_in_features=config.emb_dim,
        weights_proj_dtype=jnp.float32,
        weights_proj_quant=None,
        rngs=rngs,
    )
    self.rotary_embedding = rotary_embedding
    self.model_mode = model_mode
    self.max_target_length = config.max_target_length
    self.emb_dim = config.emb_dim
    self.rope_head_dim = config.qk_rope_head_dim

    # Key Projection: Input -> Shared Indexer Key
    self.wk = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=rngs,
    )

    # Key Normalization with Bias
    self.k_norm = nnx.LayerNorm(num_features=self.head_dim, use_bias=True, dtype=self.weight_dtype, rngs=rngs)"""

content = content.replace(indexer_init_old, indexer_init_new)

with open('src/maxtext/layers/attention_mla.py', 'w') as f:
    f.write(content)
