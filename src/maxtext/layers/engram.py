# Copyright 2023–2026 Google LLC
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

"""
DeepSeek-AI, `Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
  <https://arxiv.org/pdf/2601.07372>`_, 2026
  
Reference implementation: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
"""

from typing import List, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, MODEL_MODE_TRAIN
from maxtext.input_pipeline.tokenizer import HFTokenizer
from maxtext.layers.embeddings import Embed
from maxtext.layers.initializers import NdInitializer, nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
import numpy as np
import sympy
import tokenizers
from tokenizers import normalizers


class CompressedTokenizer:
  """
  A canonicalizing wrapper that reduces vocabulary sparsity for n-gram lookup.

  This class maps semantically equivalent tokens (e.g., "Apple", " apple", "APPLE")
  to a single unified ID. This many-to-one mapping significantly reduces the
  combinatorial size of the n-gram space.

  Attributes:
    lookup_table: Array mapping `original_id` -> `compressed_id`.
    num_new_token: Size of the compressed vocabulary.
  """

  def __init__(self, tokenizer: HFTokenizer):
    normalizer = self._build_normalizer()
    self.lookup_table_np, self.num_new_token = self._build_lookup_table(tokenizer, normalizer)
    self.lookup_table = jnp.array(self.lookup_table_np, dtype=jnp.int64)

  def __len__(self) -> int:
    return self.num_new_token

  def _build_normalizer(self) -> normalizers.Sequence:
    """
    Builds the normalization pipeline for text processing.
    """
    # Private use Unicode character to protect single spaces during stripping
    SENTINEL = "\uE000"

    # Normalization pipeline: ensures variations like "Café" and "cafe" map to the same ID
    normalizer = normalizers.Sequence(
        [
            # Compatibility decomposition (e.g., ½ -> 1/2)
            normalizers.NFKC(),
            # Canonical decomposition (e.g., é -> e + ´)
            normalizers.NFD(),
            # Strip diacritics (e.g., e + ´ -> e)
            normalizers.StripAccents(),
            # Lowercase conversion ("The" -> "the")
            normalizers.Lowercase(),
            # Collapse all whitespace variations to a single space
            normalizers.Replace(tokenizers.Regex(r"[ \t\r\n]+"), " "),
            # Protect standalone spaces from subsequent stripping
            normalizers.Replace(tokenizers.Regex(r"^ $"), SENTINEL),
            # Remove leading/trailing whitespace
            normalizers.Strip(),
            # Restore protected spaces
            normalizers.Replace(SENTINEL, " "),
        ]
    )
    return normalizer

  def _build_lookup_table(self, tokenizer: HFTokenizer, normalizer: normalizers.Sequence) -> tuple[np.ndarray, int]:
    """
    Builds the mapping from the original vocabulary to the compressed vocabulary.
    """
    vocab_size = len(tokenizer)
    # Mapping: original_tid -> compressed_nid (Many-to-One)
    old2new = np.empty(vocab_size, dtype=np.int64)
    # Mapping: normalized_string -> compressed_nid (One-to-One)
    key2new = {}

    # Batch decode token to raw text
    texts = tokenizer.batch_decode([[tid] for tid in range(vocab_size)], skip_special_tokens=False)

    for tid, text in zip(range(vocab_size), texts):
      if "\ufffd" in text:
        # Handle invalid UTF-8 (replacement char �). Use raw token instead.
        key = tokenizer.convert_ids_to_tokens(tid)
      else:
        # Normalize text (e.g., "  APPLE" -> "apple")
        normalized_text = normalizer.normalize_str(text)
        # Fallback to raw text if normalization creates an empty string
        key = normalized_text if normalized_text else text

      # Assign compressed ID
      nid = key2new.get(key)
      if nid is None:
        nid = len(key2new)
        key2new[key] = nid

      old2new[tid] = nid

    return old2new, len(key2new)

  def __call__(self, input_ids) -> Array:
    """
    Maps original token IDs to compressed IDs.
    """
    input_ids = jnp.asarray(input_ids, dtype=jnp.int64)

    # Map negative IDs to 0 for lookup, then mask output back.
    safe_ids = jnp.where(input_ids < 0, 0, input_ids)
    mapped_ids = self.lookup_table[safe_ids]

    # Restore negative IDs (padding)
    output_ids = jnp.where(input_ids < 0, input_ids, mapped_ids)
    return output_ids


class NgramHashMapping:
  """
  Deterministically maps token indices to n-gram hash indices for embedding lookups.

  This class implements Multi-Head Hashing to bypass the combinatorial memory requirements
  of explicit n-gram vocabularies. Specifically, it applies multiplicative-XOR hashing
  to each n-gram window.

  Key Mechanisms for Collision Mitigation:
  - Multi-Head Factorization: Uses K distinct hash heads per n-gram order to increase
    effective capacity within fixed memory constraints.
  - Unique Prime Moduli: Assigns a unique prime vocabulary size to each head to
    minimize simultaneous collisions.
  """

  def __init__(
      self,
      engram_vocab_bases: List[int],
      max_ngram_size: int,
      engram_num_heads: int,
      layer_ids: List[int],
      tokenizer: HFTokenizer,
      pad_id: int,
      seed: int,
  ):
    """
    Args:
      engram_vocab_bases: List of minimum head vocab sizes for each n-gram order.
      max_ngram_size: Max n-gram size to track (e.g., 3 tracks 2-grams and 3-grams).
      engram_num_heads: Number of parallel heads per n-gram order.
      layer_ids: List of layer indices using Engram.
      tokenizer: Base Hugging Face tokenizer.
      pad_id: Padding token ID.
      seed: Random seed for hash multiplier generation.
    """
    self.min_head_vocab_size_per_ngram = engram_vocab_bases
    self.max_ngram_size = max_ngram_size
    self.n_head_per_ngram = engram_num_heads
    self.layer_ids = layer_ids

    # Initialize compressed tokenizer
    self.compressed_tokenizer = CompressedTokenizer(tokenizer)
    self.tokenizer_vocab_size = len(self.compressed_tokenizer)
    if pad_id is None:
      raise ValueError("The `pad_id` must be provided and cannot be None.")
    # Pre-calculate pad_id on CPU using numpy array to avoid ConcretizationTypeError
    self.pad_id = int(self.compressed_tokenizer.lookup_table_np[pad_id])

    # Pre-calculate odd multipliers for hashing: {layer_id: multipliers}
    # Store as JAX arrays
    self.layer_multipliers = {
        k: jnp.array(v, dtype=jnp.int64) for k, v in self._calculate_multipliers_across_layers(seed).items()
    }

    # Pre-calculate unique prime vocab sizes for every head
    # Structure: {layer_id: [[2gram_head1, ..., 2gram_headH], ..., [Ngram_head1, ..., Ngram_headH]]}
    self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

  def _calculate_multipliers_across_layers(self, seed: int) -> dict[int, np.ndarray]:
    """
    Pre-calculates random odd multipliers for each layer and n-gram position.

    Returns:
      A dictionary mapping layer_id to a list of `max_ngram_size` multipliers.
    """
    # Pre-calculate bounds for random generation
    max_long = np.iinfo(np.int64).max
    m_max = int(max_long // self.tokenizer_vocab_size)
    half_bound = max(1, m_max // 2)
    # Hard-code prime number to align with reference
    LAYER_PRIME_OFFSET = 10007

    layer_multipliers = {}
    for layer_id in self.layer_ids:
      # Offset seed to decorrelate layers
      layer_seed = int(seed + LAYER_PRIME_OFFSET * int(layer_id))
      np_rng = np.random.default_rng(layer_seed)
      # Generate random odd integers
      random_value = np_rng.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
      multipliers = random_value * 2 + 1
      layer_multipliers[layer_id] = multipliers
    return layer_multipliers

  def _calculate_vocab_size_across_layers(self) -> dict[int, List[List[int]]]:
    """
    Calculates unique prime vocabulary sizes for every head in every layer.
    Using unique primes minimizes the probability of simultaneous collisions across heads.
    """

    def find_next_unseen_prime(start: int, seen_primes: set) -> int:
      candidate = start + 1
      while candidate in seen_primes or not sympy.isprime(candidate):
        candidate += 1
      return candidate

    seen_primes = set()
    vocab_size_across_layers = {}

    for layer_id in self.layer_ids:
      all_ngram_vocab_sizes = []

      for n in range(2, self.max_ngram_size + 1):
        current_ngram_heads_sizes = []
        # Start search from the configured minimum size
        vocab_size = self.min_head_vocab_size_per_ngram[n - 2]
        current_prime_search_start = vocab_size - 1
        # Find unique primes for each head
        num_heads = self.n_head_per_ngram
        for _ in range(num_heads):
          found_prime = find_next_unseen_prime(current_prime_search_start, seen_primes)
          seen_primes.add(found_prime)
          current_ngram_heads_sizes.append(found_prime)
          current_prime_search_start = found_prime
        all_ngram_vocab_sizes.append(current_ngram_heads_sizes)

      vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

    return vocab_size_across_layers

  def get_vocab_sizes(self, layer_id: int) -> List[int]:
    """
    Returns a flattened list of prime vocabulary sizes for a specific layer.
    """
    return [head_size for ngram_size in self.vocab_size_across_layers[layer_id] for head_size in ngram_size]

  def _get_ngram_hashes(self, compressed_ids: Array, layer_id: int) -> Array:
    """
    Computes hash indices for all n-grams in the input batch.

    Args:
      compressed_ids: [B, S] input token IDs.
      layer_id: engram layer id.

    Returns:
      hash_ids: [B, S, H_total] where H_total = H * num_ngram_orders
    """
    x = jnp.asarray(compressed_ids, dtype=jnp.int64)
    B, _ = x.shape

    # 1. Create Sliding Windows via Shifting
    shifted_inputs = []
    for k in range(self.max_ngram_size):
      if k == 0:
        shifted_inputs.append(x)
      else:
        # Pre-allocate full array with PAD_ID
        padding = jnp.full((B, k), self.pad_id, dtype=jnp.int64)
        # Fast memory copy, slicing and assignment
        # e.g., k=1, [PAD, The, cat]
        #       k=2, [PAD, PAD, The]
        shifted_x = jnp.concatenate([padding, x[:, :-k]], axis=1)
        shifted_inputs.append(shifted_x)

    # 2. Retrieve layer-specific hash multipliers
    multipliers = self.layer_multipliers[layer_id]

    # 3. Compute Hashes: multiplicative bitwise XOR
    # Implements hash: H_n = (shift_0 * m_0) ^ ... ^ (shift_k * m_k)
    # e.g., (The * m_0) ^ (PAD * m_1) ^ (PAD * m_2)
    #       (cat * m_0) ^ (The * m_1) ^ (PAD * m_2)
    #       (sat * m_0) ^ (cat * m_1) ^ (The * m_2)
    all_hashes = []
    # Initialize with unigrams, shape: [B, S]
    ngram_hash = shifted_inputs[0] * multipliers[0]
    # Pre-fetch vocab sizes for modulo
    vocab_sizes = self.vocab_size_across_layers[layer_id]

    for n in range(2, self.max_ngram_size + 1):
      # Update hash with next history token
      ngram_hash = jnp.bitwise_xor(ngram_hash, shifted_inputs[n - 1] * multipliers[n - 1])

      # Retrieve prime vocab sizes for all heads of this n-gram order
      vocab_sizes_for_this_gram = vocab_sizes[n - 2]
      mods = jnp.array(vocab_sizes_for_this_gram, dtype=jnp.int64)

      # Broadcast Modulo: Map hash to valid table indices
      # [B, S, 1] % [H] -> [B, S, H]
      head_hashes = ngram_hash[..., None] % mods
      all_hashes.append(head_hashes)

    # Concatenate all heads: [B, S, H_total] where H_total = H * num_ngram_orders
    return jnp.concatenate(all_hashes, axis=2)

  def __call__(self, input_ids) -> dict[int, Array]:
    # input_ids from standard tokenizer
    compressed_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}
    for layer_id in self.layer_ids:
      hash_ids = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
      hash_ids_for_all_layers[layer_id] = hash_ids
    return hash_ids_for_all_layers


class StaticWrapper:
  """Wrapper to prevent nnx from treating the value as a variable."""

  def __init__(self, val):
    self.val = val


class MultiHeadEmbedding(nnx.Module):
  """
  A flattened table representation for multi-head embedding spaces across n-gram orders.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      vocab_sizes: List[int],
      head_dim: int,
      rngs: nnx.Rngs = None,
  ):
    """
    Args:
      config: The model configuration.
      mesh: Device mesh for partitioning.
      vocab_sizes: Flattened list of prime vocabulary sizes for all heads across all n-gram orders.
        Example: [2gram_Head1, 2gram_Head2, 3gram_Head1, ...]
      head_dim: Embedding dimension for a single head.
      rngs: Random number generators for initialization.
    """
    self.num_heads = len(vocab_sizes)

    # Compute starting index for each head's segment in the flattened table.
    # Offsets serve as the "base address" for each head.
    offsets = np.cumsum([0] + vocab_sizes[:-1])  # prefix sum
    self.offsets = StaticWrapper(np.array(offsets, dtype=np.int64))

    # The total embedding size is the sum of all individual head vocabularies.
    self.embedding = Embed(num_embeddings=sum(vocab_sizes), num_features=head_dim, config=config, mesh=mesh, rngs=rngs)

  def __call__(self, input_ids: Array, model_mode: str = MODEL_MODE_TRAIN) -> Array:
    """
    Retrieves embeddings for multi-head indices.

    Args:
      input_ids: Hashed indices. Shape [B, S, H_total], where H_total is the total number of heads.
      model_mode: The model's operational mode (e.g., 'train', 'prefill').

    Returns:
      embeddings: Shape [B, S, H_total, D_head].
    """
    # Broadcasting Add: [B, S, H] + [H] -> [B, S, H]
    # Shifts local indices (0..Prime-1) to global table positions.
    shifted_ids = input_ids + self.offsets.val

    # Embedding lookup: [B, S, H_total] -> [B, S, H_total, D_head]
    return self.embedding(shifted_ids, model_mode=model_mode)


class ShortConv(nnx.Module):
  """
  Depthwise causal 1D convolution, with multi-branch integration.

  Applies local temporal smoothing
  - Independent RMSNorms to each branch
  - Convolution to mix time steps [t-k, t]
  """

  def __init__(
      self,
      config: Config,
      hidden_size: int,
      kernel_size: int,
      dilation: int,
      mhc_expansion_rate: int,
      rngs: nnx.Rngs = None,
  ):
    """
    Args:
      config: The model configuration.
      hidden_size (D): Dimension of a single branch.
      kernel_size: Temporal window size.
      dilation: Dilation rate for the convolution.
      mhc_expansion_rate (G): Number of branches.
      rngs: RNG state for initialization.
    """
    self.mhc_expansion_rate = mhc_expansion_rate

    # Norms
    # Vectorized Init: Independent weights per branch
    #   rngs: [G, 2] split RNGs, vectorize over G, `in_axes=0`
    #   Stack weights at axis 0 to get [G, D], `out_axes=0`
    @nnx.split_rngs(splits=mhc_expansion_rate)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(rngs):
      return RMSNorm(
          num_features=hidden_size,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=rngs,
      )

    self.norm = create_norms(rngs)

    # Convolution (Batch over branch)
    # Depthwise: feature_group_count == in_features ensures no mixing across channels and branches
    # Causal: Ensures output at t only depends on inputs <= t.
    # Weights: {"kernel": shape [kernel_size, in_features//feature_group_count, total_channels]}
    total_channels = mhc_expansion_rate * hidden_size  # G * D
    self.conv = nnx.Conv(
        in_features=total_channels,
        out_features=total_channels,
        kernel_size=(kernel_size,),
        feature_group_count=total_channels,
        kernel_dilation=(dilation,),
        padding="CAUSAL",
        use_bias=False,
        # convolution parameters are initialized to zero
        # to strictly preserve the identity mapping at the start of training
        kernel_init=nnx.initializers.zeros,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, x: Array) -> Array:
    """
    Compute y^i = SiLU(Conv1D(RMSNorm^i(x^i))) for each branch i.

    Args:
      x: Input tensor of shape [B, S, G, D]
    Returns:
      Output tensor of shape [B, S, G, D]

    Shape annotation:
      B: Batch size
      S: Sequence length (temporal dimension)
      G: Number of branches (mhc_expansion_rate)
      D: Hidden size (base_emb_dim)
    """
    B, S, G, D = x.shape

    # Vectorized Apply
    #   norms: [G, D], vectorize over G, `in_axes=0`
    #   x: [B, S, G, D], vectorize over G, `in_axes=2`
    #   Stack results at axis 2 to get [B, S, G, D], `out_axes=2`
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    # [B, S, G, D] shape stays
    x = apply_norms(self.norm, x)

    # Flatten branches into channel: [B, S, G, D] -> [B, S, G * D]
    x_flat = x.reshape(B, S, G * D)
    # Depthwise Convolution to mix temporal dimension S only. [B, S, G * D] shape stays
    y = self.conv(x_flat)
    y = jax.nn.silu(y)
    # Restore branch: [B, S, G * D] -> [B, S, G, D]
    return y.reshape(B, S, G, D)


class Engram(nnx.Module):
  """
  Engram Memory Layer with n-gram embedding, with multi-branch integration.

  Main components:
  - Context-independent Retrieval: Fetch static n-gram embeddings via Multi-Head Hashing.
  - Context-aware Gating: Compute similarity between memory (Key) and context (Query) to determine relevance.
  - Mix: Apply local temporal smoothing via convolution.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      vocab_sizes: List[int],
      engram_num_heads: int,
      engram_head_dim: int,
      engram_max_ngram_size: int,
      engram_kernel_size: int,
      mhc_expansion_rate: int,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    """
    Args:
      config: The model configuration.
      mesh: Partitioning mesh.
      vocab_sizes: Flattened list of prime vocabulary sizes for all heads across all n-gram orders.
        Example: [2gram_Head1, 2gram_Head2, 3gram_Head1, ...]
      engram_num_heads (H): Heads per n-gram order.
      engram_head_dim (D_head): Dimension per head.
      engram_max_ngram_size: Max n-gram order (e.g., 3 covers 2-grams and 3-grams).
      engram_kernel_size: convolution kernel size.
      mhc_expansion_rate (G): Number of branches.
      kernel_init: Weight initializer.
      quant: Quantization config.
      rngs: RNG state for initialization
    """
    self.config = config
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.dtype
    self.kernel_init = kernel_init
    self.quant = quant
    self.rngs = rngs
    self.mhc_expansion_rate = mhc_expansion_rate

    # Hierarchy: Engram -> n-gram Order -> h-th Head
    self.max_ngram_size = engram_max_ngram_size
    self.conv_kernel_size = engram_kernel_size
    num_ngram_orders = self.max_ngram_size - 1
    # D_en: Final concatenated size of the retrieved memory
    self.engram_dim = engram_head_dim * engram_num_heads * num_ngram_orders

    # Embedding: one flattened table to store all n-gram heads across orders
    self.multi_head_embedding = MultiHeadEmbedding(
        config=config, mesh=mesh, vocab_sizes=vocab_sizes, head_dim=engram_head_dim, rngs=rngs
    )

    # Key Projection (Batch over branch)
    # retrieved n-gram memory -> Key, from D_en to [G, D]
    self.key_proj = DenseGeneral(
        in_features_shape=self.engram_dim,
        out_features_shape=(mhc_expansion_rate, config.base_emb_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("engram_dim", "mhc", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        use_bias=True,
        rngs=rngs,
    )

    # Norms
    # Vectorized Init: Independent weights per branch
    #   rngs: [G, 2] split RNGs, vectorize over G, `in_axes=0`
    #   Stack weights at axis 0 to get [G, D], `out_axes=0`
    @nnx.split_rngs(splits=mhc_expansion_rate)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(rngs):
      return RMSNorm(
          num_features=config.base_emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=rngs,
      )

    # Key Normalization
    self.k_norm = create_norms(rngs)
    # Query Normalization
    self.q_norm = create_norms(rngs)

    # Value Projection (Shared): Retrieved memory -> Value
    self.value_proj = DenseGeneral(
        in_features_shape=self.engram_dim,
        out_features_shape=config.base_emb_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("engram_dim", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        use_bias=True,
        rngs=self.rngs,
    )

    # Short Convolution (Vectorized Internally)
    # Applies depthwise causal convolution to smooth the retrieved memory over time.
    self.short_conv = ShortConv(
        config=config,
        hidden_size=config.base_emb_dim,
        kernel_size=self.conv_kernel_size,
        dilation=self.max_ngram_size,
        mhc_expansion_rate=mhc_expansion_rate,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, hash_input_ids: Array) -> Array:
    """
    Computes the Engram output by retrieving, gating, and smoothing n-gram memory.

    Args:
      hidden_states: current transformer state. Shape: [B, S, G, D].
      hash_input_ids: Hashed token IDs. Shape: [B, S, H_total].
        Produced by `hash_mapping.hash(input_ids)[layer_id]`.

    Returns:
      Shape: [B, S, G, D]

    Shape annotation:
      B: Batch Size
      S: Sequence Length
      G: mhc_expansion_rate, Number of Branches
      H_total: Total number of heads across n-grams. num_head * num_ngrams
      D: base_emb_dim
      D_head: Dimension of a single head embedding
      D_en: Dimension of flattened embedding across heads and n-grams
    """
    B, S, _, D = hidden_states.shape

    # 1. Retrieve Memory from Embedding
    # [B, S, H_total] -> [B, S, H_total, D_head]
    embeddings = self.multi_head_embedding(hash_input_ids)
    # [B, S, H_total, D_head] -> [B, S, D_en]
    embeddings = embeddings.reshape(B, S, -1)

    # 2. Static Memory as Key
    # [B, S, D_en] -> [B, S, G, D]
    key = self.key_proj(embeddings)

    # 3. Compute Norms
    # Vectorized Apply
    #   norms: [G, D], vectorize over G, `in_axes=0`
    #   x: [B, S, G, D], vectorize over G, `in_axes=2`
    #   Stack results at axis 2 to get [B, S, G, D], `out_axes=2`
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    # [B, S, G, D] shape stays
    key = apply_norms(self.k_norm, key)

    # 4. Dynamic Context as Query
    # [B, S, G, D] shape stays
    query = apply_norms(self.q_norm, hidden_states)

    # 5. QK product as Gates
    # Compute similarity of memory (Key) and current state (Query)
    qk_product = jnp.einsum("bsgd,bsgd->bsg", query, key, precision=self.config.matmul_precision)
    gate = qk_product / jnp.sqrt(D)
    # Range Compression: Apply signed square-root to prevent sigmoid saturation
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
    # Sigmoid activation to get gating probability [0, 1]
    gate = jax.nn.sigmoid(gate)  # [B, S, G]

    # 6. Static Memory as Value
    # [B, S, D_en] -> [B, S, D]
    value = self.value_proj(embeddings)

    # 7. Apply Gates to Value
    # [B, S, G, 1] * [B, S, 1, D] -> [B, S, G, D]
    gated_value = gate[:, :, :, None] * value[:, :, None, :]

    # 8. ShortConv as Temporal Smoothing
    # [B, S, G, D] shape stays
    # Apply depthwise conv to mix S
    conv_output = self.short_conv(gated_value)
    # residual connection for conv component
    output = gated_value + conv_output

    # Note: residual connection for hidden_states will be added by the caller
    return output
