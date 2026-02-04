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


from typing import List, Optional
import numpy as np
from sympy import isprime
from tokenizers import normalizers, Regex

import jax
import jax.numpy as jnp
from flax import nnx

from MaxText.common_types import MODEL_MODE_TRAIN, Array, Config
from MaxText.layers.embeddings import Embed
from MaxText.layers.initializers import nd_dense_init, NdInitializer
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant

"""
DeepSeek-AI, `Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
  <https://arxiv.org/pdf/2601.07372>`_, 2026
  
Reference implementation: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
"""


class CompressedTokenizer:
  """
  A canonicalizing wrapper that reduces vocabulary sparsity for n-gram lookup.

  This class maps semantically equivalent tokens (e.g., "Apple", " apple", "APPLE")
  to a single unified ID. This many-to-one mapping significantly reduces the
  combinatorial size of the n-gram space.

  Attributes:
    tokenizer: Base Hugging Face tokenizer.
    normalizer: Pipeline of text normalization rules.
    lookup_table: Array mapping `original_id` -> `compressed_id`.
    num_new_token: Size of the compressed vocabulary.
  """

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.normalizer = self._build_normalizer()
    self.lookup_table, self.num_new_token = self._build_lookup_table()

  def __len__(self):
    return self.num_new_token

  def _build_normalizer(self):
    # Private use Unicode character to protect single spaces during stripping
    SENTINEL = "\uE000"

    # Normalization pipeline: ensures variations like "Café" and "cafe" map to the same ID
    normalizer = normalizers.Sequence(
        [
            # 1. Compatibility decomposition (e.g., ½ -> 1/2)
            normalizers.NFKC(),
            # 2. Canonical decomposition (e.g., é -> e + ´)
            normalizers.NFD(),
            # 3. Strip diacritics (e.g., e + ´ -> e)
            normalizers.StripAccents(),
            # 4. Lowercase conversion ("The" -> "the")
            normalizers.Lowercase(),
            # 5. Collapse all whitespace variations to a single space
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            # 6. Protect standalone spaces from subsequent stripping
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            # 7. Remove leading/trailing whitespace
            normalizers.Strip(),
            # 8. Restore protected spaces
            normalizers.Replace(SENTINEL, " "),
        ]
    )
    return normalizer

  def _build_lookup_table(self):
    """
    Builds the mapping from the original vocabulary to the compressed vocabulary.
    """
    vocab_size = len(self.tokenizer)
    # Mapping: original_tid -> compressed_nid (Many-to-One)
    old2new = np.empty(vocab_size, dtype=np.int64)
    # Mapping: normalized_string -> compressed_nid (One-to-One)
    key2new = {}

    for tid in range(vocab_size):
      # Decode token to raw text
      text = self.tokenizer.decode([tid], skip_special_tokens=False)

      if "\ufffd" in text:
        # Handle invalid UTF-8 (replacement char �). Use raw token instead.
        key = self.tokenizer.convert_ids_to_tokens(tid)
      else:
        # Normalize text (e.g., "  APPLE" -> "apple")
        normalized_text = self.normalizer.normalize_str(text)
        # Fallback to raw text if normalization creates an empty string
        key = normalized_text if normalized_text else text

      # Assign compressed ID
      nid = key2new.get(key)
      if nid is None:
        nid = len(key2new)
        key2new[key] = nid

      old2new[tid] = nid

    return old2new, len(key2new)

  def __call__(self, input_ids):
    """
    Maps original token IDs to compressed IDs.
    """
    input_ids = np.asarray(input_ids, dtype=np.int64)

    # Identify valid tokens (ignore padding/masks usually marked with negative IDs)
    valid_mask = input_ids >= 0
    valid_ids = input_ids[valid_mask]

    # Vectorized lookup: O(1) per token
    output_ids = input_ids.copy()
    output_ids[valid_mask] = self.lookup_table[valid_ids]
    return output_ids


class NgramHashMapping:
  """
  Maps n-gram sequences to hash-based indices for memory lookup.

  This class implements the Engram hashing mechanism. It converts variable-length
  n-grams into fixed integer IDs. To handle the large combinatorial space, it uses:
  1.  **Unique Prime Vocabularies:** Per-head prime moduli to minimize collision overlap.
  2.  **Sliding Window:** Efficient shifting to generate n-gram views.
  3.  **Lightweight Hashing:** A multiplicative-XOR function (Rabin-Karp variant).
  """

  def __init__(
      self,
      engram_vocab_size,
      max_ngram_size,
      engram_num_heads,
      layer_ids,
      tokenizer,
      pad_id,
      seed,
  ):
    """
    Args:
      engram_vocab_size: List of minimum vocab sizes for each n-gram order.
      max_ngram_size: Max n-gram size to track (e.g., 3 tracks 2-grams and 3-grams).
      engram_num_heads: Number of parallel heads per n-gram order.
      layer_ids: List of layer indices using Engram.
      tokenizer: Base Hugging Face tokenizer.
      pad_id: Padding token ID.
      seed: Random seed for hash multiplier generation.
    """
    self.min_head_vocab_size_per_ngram = engram_vocab_size
    self.max_ngram_size = max_ngram_size
    self.n_head_per_ngram = engram_num_heads
    self.layer_ids = layer_ids

    # Initialize compressed tokenizer
    self.compressed_tokenizer = CompressedTokenizer(tokenizer)
    self.tokenizer_vocab_size = len(self.compressed_tokenizer)
    # TODO(shuningjin): why not just use pad_id = tokenizer.pad_id
    if pad_id is not None:
      self.pad_id = int(self.compressed_tokenizer.lookup_table[pad_id])

    # Pre-calculate odd multipliers for hashing: {layer_id: multipliers}
    self.layer_multipliers = self._calculate_multipliers_across_layers(seed)

    # Pre-calculate unique prime vocab sizes for every head
    # Structure: {layer_id: [[2gram_head1, ..., 2gram_headH], ..., [Ngram_head1, ..., Ngram_headH]]}
    self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

  def _calculate_multipliers_across_layers(self, seed: int):
    """
    Pre-calculates random odd multipliers for each layer and n-gram position.

    Returns:
      A dictionary mapping layer_id to a list of `max_ngram_size` multipliers.
    """
    # Pre-calculate bounds for random generation
    max_long = np.iinfo(np.int64).max
    m_max = int(max_long // self.tokenizer_vocab_size)
    half_bound = max(1, m_max // 2)
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

  def _calculate_vocab_size_across_layers(self):
    """
    Calculates unique prime vocabulary sizes for every head in every layer.
    Using unique primes minimizes the probability of simultaneous collisions across heads.
    """

    def find_next_unseen_prime(start, seen_primes):
      candidate = start + 1
      while candidate in seen_primes or not isprime(candidate):
        candidate += 1
      return candidate

    seen_primes = set()
    vocab_size_across_layers = {}

    for layer_id in self.layer_ids:
      all_ngram_vocab_sizes = []
      for ngram in range(2, self.max_ngram_size + 1):
        current_ngram_heads_sizes = []

        # Start search from the configured minimum size
        n_gram_index = ngram - 2
        vocab_size = self.min_head_vocab_size_per_ngram[n_gram_index]
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

  def get_vocab_sizes(self, layer_id: int):
    """
    Returns a flattened list of prime vocabulary sizes for a specific layer.
    """
    return [head for ngram in self.vocab_size_across_layers[layer_id] for head in ngram]

  def _get_ngram_hashes(
      self,
      compressed_ids: np.ndarray,
      layer_id: int,
  ) -> np.ndarray:
    """
    Computes hash indices for all n-grams in the input batch.

    Args:
      compressed_ids: (B, S) input token IDs.
      layer_id: engram layer id.

    Returns:
      hash_ids: (B, S, H_total) where H_total = H * num_ngram_orders
    """
    x = np.asarray(compressed_ids, dtype=np.int64)
    B, T = x.shape

    # 1. Create Sliding Windows via Shifting
    base_shifts = []
    for k in range(self.max_ngram_size):
      if k == 0:
        # e.g., [The, cat, sat]
        base_shifts.append(x)
      else:
        # Pre-allocate full array with PAD_ID
        shifted = np.full((B, T), self.pad_id, dtype=np.int64)
        # Fast memory copy, slicing and assignment
        # e.g., k=1, [PAD, The, cat]
        shifted[:, k:] = x[:, :-k]
        base_shifts.append(shifted)

    # 2. Retrieve layer-specific hash multipliers
    multipliers = self.layer_multipliers[layer_id]

    # 3. Compute Hashes: multiplicative bitwise XOR
    # Implements rolling hash: H_n = (Token_0 * m_0) ^ ... ^ (Token_k * m_k)
    all_hashes = []

    # Initialize rolling hash with 1-gram
    rolling_hash = base_shifts[0] * multipliers[0]

    # Pre-fetch vocab sizes for modulo
    vocab_sizes = self.vocab_size_across_layers[layer_id]

    for n in range(2, self.max_ngram_size + 1):
      # Update rolling hash with next token position
      k = n - 1
      rolling_hash = np.bitwise_xor(rolling_hash, base_shifts[k] * multipliers[k])

      # Retrieve prime vocab sizes for all heads of this n-gram order
      n_gram_index = n - 2
      vocab_sizes_for_this_gram = vocab_sizes[n_gram_index]
      mods = np.array(vocab_sizes_for_this_gram, dtype=np.int64)

      # Broadcast Modulo: Map hash to valid table indices
      # (B, S, 1) % (H,) -> (B, S, H)
      head_hashes = rolling_hash[..., None] % mods
      all_hashes.append(head_hashes)

    # Concatenate all heads: (B, S, H_total) where H_total = H * num_ngram_orders
    return np.concatenate(all_hashes, axis=2)

  def __call__(self, input_ids):
    compressed_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}
    for layer_id in self.layer_ids:
      hash_ids = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
      hash_ids_for_all_layers[layer_id] = hash_ids
    return hash_ids_for_all_layers


class MultiHeadEmbedding(nnx.Module):
  """
  A flattened table representation for multi-head embedding spaces across n-gram orders.

  """

  def __init__(self, vocab_sizes: List[int], head_dim: int, config: Config, mesh, rngs: nnx.Rngs):
    """
    Args:
      vocab_sizes: Flattened list of prime vocabulary sizes for all heads across all n-gram orders.
        Example: [Size_2gram_Head1, Size_2gram_Head2, Size_3gram_Head1, ...].
      head_dim: Embedding dimension for a single head.
      config: The model configuration.
      mesh: Device mesh for partitioning.
      rngs: Random number generators for initialization.
    """
    self.num_heads = len(vocab_sizes)

    # Compute starting index for each head's segment in the flattened table.
    # Offsets serve as the "base address" for each head.
    offsets = np.cumsum([0] + vocab_sizes[:-1])  # prefix sum
    self.offsets = jnp.array(offsets, dtype=jnp.int32)

    # The total embedding size is the sum of all individual head vocabularies.
    self.embedding = Embed(num_embeddings=sum(vocab_sizes), num_features=head_dim, config=config, mesh=mesh, rngs=rngs)

  def __call__(self, input_ids: Array, model_mode: str = MODEL_MODE_TRAIN) -> Array:
    """
    Retrieves embeddings for multi-head indices.

    Args:
      input_ids: Hashed indices. Shape (B, S, H_total), where H_total is the total number of heads.
      model_mode: The model's operational mode (e.g., 'train', 'prefill').

    Returns:
      embeddings: Shape (B, S, H_total, D_head).
    """
    # Broadcasting Add: (B, S, H) + (H,) -> (B, S, H)
    # Shifts local indices (0..Prime-1) to global table positions.
    shifted_ids = input_ids + self.offsets

    # Embedding lookup: (B, S, H_total) -> (B, S, H_total, D_head)
    return self.embedding(shifted_ids, model_mode=model_mode)


class ShortConv(nnx.Module):
  """
  Depthwise causal 1D convolution, with multi-branch integration.

  Applies local temporal smoothing
  - Independent RMSNorms to each branch
  - Shared convolution to mix time steps [t-k, t]
  """

  def __init__(
      self,
      config: Config,
      hidden_size: int,
      kernel_size: int = 4,
      dilation: int = 1,
      hc_mult: int = 4,
      rngs: nnx.Rngs = None,
  ):
    """
    Args:
      config: The model configuration.
      hidden_size (D): Dimension of a single branch.
      kernel_size: Temporal window size.
      dilation: Dilation rate for the convolution.
      hc_mult (G): Number of branches.
      rngs: RNG state for initialization.
    """
    self.hc_mult = hc_mult
    # Total channels = G * D
    total_channels = hidden_size * hc_mult

    # Norm (Vectorized)
    # independent weight per branch, branched input
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(r):
      return RMSNorm(
          num_features=hidden_size,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=r,
      )

    # Weights: {"scale": (G, D)}
    self.norms = create_norms(rngs)

    # Convolution (Shared)
    # Depthwise: feature_group_count == in_features ensures no mixing across channels.
    # Causal: Ensures output at t only depends on inputs <= t.
    # Weights: {"kernel": (kernel_size, in_features//feature_group_count, total_channels)}
    self.conv = nnx.Conv(
        in_features=total_channels,
        out_features=total_channels,
        kernel_size=(kernel_size,),
        feature_group_count=total_channels,
        kernel_dilation=(dilation,),
        padding="CAUSAL",
        use_bias=False,
        rngs=rngs,
    )

  def __call__(self, x: Array) -> Array:
    """
    Compute y^i = SiLU(Conv1D(RMSNorm^i(x^i))) for each branch i.

    Args:
      x: Input tensor of shape (B, S, G, D)
    Returns:
      Output tensor of shape (B, S, G, D)

    Shape annotation:
      B: Batch size
      S: Sequence length (temporal dimension)
      G: Number of branches (hc_mult)
      D: Hidden size (base_emb_dim)
    """
    B, S, G, D = x.shape

    # Apply Norms (Vectorized over Group dim)
    # `in_axes=(0, 2)`: norms is axis 0, x is axis 2
    # `out_axes=2`: put the group dim back at axis 2
    # shape stays (B, S, G, D)
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    x = apply_norms(self.norms, x)

    # Flatten branches into channel: (B, S, G, D) -> (B, S, G * D)
    x_flat = x.reshape(B, S, G * D)
    # Depthwise Convolution to mix temporal dimension S only. Shape stays (B, S, G * D)
    y = self.conv(x_flat)
    y = jax.nn.silu(y)
    # Restore branch: (B, S, G * D) -> (B, S, G, D)
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
      rngs: nnx.Rngs,
      config: Config,
      mesh,
      quant: Optional[Quant] = None,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      *,
      hc_mult: int = 4,
      vocab_sizes,
      engram_num_heads: int,
      engram_head_dim: int,
      engram_max_ngram_size: int,
      engram_kernel_size: int,
  ):
    """
    Args:
      rngs: RNG state for initialization
      config: The model configuration.
      mesh: Partitioning mesh.
      quant: Quantization config.
      kernel_init: Weight initializer.
      hc_mult (G): Number of branches.
      vocab_sizes: List of prime vocabulary sizes for the embedding table.
      engram_num_heads (H): Heads per n-gram order.
      engram_head_dim (D_head): Dimension per head.
      engram_max_ngram_size: Max n-gram order (e.g., 3 covers 2-grams and 3-grams).
      engram_kernel_size: convolution kernel size.
    """
    self.config = config
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.dtype
    self.kernel_init = kernel_init
    self.quant = quant
    self.rngs = rngs
    self.hc_mult = hc_mult

    # Hierarchy: Engram -> n-gram Order -> h-th Head
    self.max_ngram_size = engram_max_ngram_size
    self.conv_kernel_size = engram_kernel_size
    num_ngram_orders = self.max_ngram_size - 1
    # D_en: Final concatenated size of the retrieved memory
    self.engram_dim = engram_head_dim * engram_num_heads * num_ngram_orders

    # Embedding: one flattened table to store all n-gram heads across orders
    self.multi_head_embedding = MultiHeadEmbedding(
        vocab_sizes=vocab_sizes, head_dim=engram_head_dim, config=config, mesh=mesh, rngs=rngs
    )

    # Key Projection (vectorized): retrieved n-gram memory -> Key
    # Independent weights per branch, Shared input
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_key_projs(r):
      return DenseGeneral(
          in_features_shape=self.engram_dim,
          out_features_shape=config.base_emb_dim,
          axis=-1,
          kernel_init=self.kernel_init,
          # TODO(shuningjin): this needs to be actual logical axis? @reviewer
          kernel_axes=("engram_dim", "embed"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=r,
          use_bias=True,
      )

    self.key_projs = create_key_projs(rngs)

    # Norms (vectorized)
    # Independent weights per branch, Branched input
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(r):
      return RMSNorm(
          num_features=config.base_emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          epsilon=config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=r,
      )

    # Key Normalization
    self.k_norms = create_norms(rngs)
    # Query Normalization
    self.q_norms = create_norms(rngs)

    # Value Projection (shared): Retrieved memory -> Value
    self.value_proj = DenseGeneral(
        in_features_shape=self.engram_dim,
        out_features_shape=config.base_emb_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        # TODO(shuningjin): this needs to be actual logical axis? @reviewer
        kernel_axes=("engram_dim", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
        use_bias=True,
    )

    # Short Convolution (vectorized internally)
    # Applies depthwise causal convolution to smooth the retrieved memory over time.
    self.short_conv = ShortConv(
        config=config,
        hidden_size=config.base_emb_dim,
        kernel_size=self.conv_kernel_size,
        dilation=self.max_ngram_size,
        hc_mult=hc_mult,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, hash_input_ids: Array) -> Array:
    """
    Computes the Engram output by retrieving, gating, and smoothing n-gram memory.

    Args:
      hidden_states: Current transformer state (Query). Shape: (B, S, G, D).
      hash_input_ids: Hashed token IDs. Shape: (B, S, H_total).
        Produced by `hash_mapping.hash(input_ids)[layer_id]`.

    Returns:
      Shape: (B, S, G, D)


    Shape annotation:
      B: Batch Size
      S: Sequence Length
      G: hc_mult, Number of Branches
      H_total: Total number of heads across n-grams. num_head * num_ngrams
      D: base_emb_dim
      D_head: Dimension of a single head embedding
      D_en: Dimension of flattened embedding across heads and n-grams
    """
    B, S, _, D = hidden_states.shape

    # 1. Retrieve Memory from Embedding
    # (B, S, H_total) -> (B, S, H_total, D_head)
    embeddings = self.multi_head_embedding(hash_input_ids)
    # (B, S, H_total, D_head) -> (B, S, D_en)
    embeddings = embeddings.reshape(B, S, -1)

    # 2. Static Memory as Key
    # Vectorized broadcast: apply each of the G key_projs to the SAME embeddings.
    # in_axes: (0, None) -> 0 splits the Dense layers, None broadcasts embeddings
    # out_axes: 2        -> Stack the results at axis 2 to get (B, S, G, D)
    @nnx.vmap(in_axes=(0, None), out_axes=2)
    def apply_projs(projs, x):
      return projs(x)

    # (B, S, D_en) ->  (B, S, G, D)
    key = apply_projs(self.key_projs, embeddings)

    # 3. Compute Norms
    # Vectorized Map: Map over the G dimension (Axis 2) for both weights and input
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    # (B, S, G, D) shape stays
    key = apply_norms(self.k_norms, key)

    # 4. Dynamic Context as Query
    # (B, S, G, D) shape stays
    query = apply_norms(self.q_norms, hidden_states)

    # 5. QK product as Gates
    # Compute similarity of memory (Key) and current state (Query)
    qk_product = jnp.einsum("bsgc,bsgc->bsg", query, key, precision=self.config.matmul_precision)
    gate = qk_product / jnp.sqrt(D)
    # Range Compression: Apply signed square-root to prevent sigmoid saturation
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
    # Sigmoid activation to get gating probability [0, 1]
    gate = jax.nn.sigmoid(gate)  # (B, S, G)

    # 6. Static Memory as Value
    # (B, S, D_en) -> (B, S, D)
    value = self.value_proj(embeddings)

    # 7. Apply Gates to Value
    # (B, S, G, 1) * (B, S, 1, D) -> (B, S, G, D)
    gated_value = gate[:, :, :, None] * value[:, :, None, :]

    # 8. ShortConv as Temporal Smoothing
    # Shape remains, (B, S, G, D)
    # Apply depthwise conv to mix S
    conv_output = self.short_conv(gated_value)
    # residual for conv component
    output = gated_value + conv_output

    # Note: residual connection for hidden_states will be added by the caller
    return output
