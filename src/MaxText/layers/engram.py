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


from typing import List, Optional, Callable
from dataclasses import dataclass, field
import math
import numpy as np
from sympy import isprime
from tokenizers import normalizers, Regex

import jax
import jax.numpy as jnp
from flax import nnx

from MaxText.common_types import MODEL_MODE_TRAIN, ShardMode, MODEL_MODE_PREFILL, Array, Config, DType
from MaxText.layers.embeddings import Embed
from MaxText.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned
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
  A lossy, canonicalizing wrapper for a standard tokenizer to optimize n-gram lookup.

  By applying normalization (lowercasing, accent stripping, and whitespace
  collapsing), this class maps semantically equivalent tokens (e.g., 'Apple', ' apple',
  'APPLE') to a single unified ID. This many-to-one mapping reduces the combinatorial n-gram space.

  Attributes:
    tokenizer: The base Hugging Face tokenizer.
    normalizer: A tokenizers.Normalizer sequence defining the canonicalization rules.
    lookup_table: A NumPy array where `lookup_table[original_id] = compressed_id`.
    num_new_token: The size of the resulting compressed vocabulary.
  """

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.normalizer = self._build_normalizer()
    self.lookup_table, self.num_new_token = self._build_lookup_table()

  def __len__(self):
    return self.num_new_token

  def _build_normalizer(self):
    # Private use Unicode character used to protect single spaces during stripping
    SENTINEL = "\uE000"
    # Text normalization pipeline: ensures "Café" and "cafe" produce the same ID
    normalizer = normalizers.Sequence(
        [
            # Compatibility decomposition (e.g., ½ -> 1/2)
            normalizers.NFKC(),
            # Canonical decomposition (e.g., é -> e + ´)
            normalizers.NFD(),
            # Removes diacritics (e.g., e + ´ -> e)
            normalizers.StripAccents(),
            # "The" -> "the"
            normalizers.Lowercase(),
            # Collapses all variations of whitespace into a single space
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            # Protects standalone spaces from being deleted by Strip()
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            # Removes leading/trailing whitespace
            normalizers.Strip(),
            # Restores protected spaces
            normalizers.Replace(SENTINEL, " "),
        ]
    )
    return normalizer

  def _build_lookup_table(self):
    """
    Build the mapping from original vocabulary to the compressed vocabulary
    """
    vocab_size = len(self.tokenizer)
    # Lookup table: Maps original_tid -> compressed_nid, many-to-one
    old2new = np.empty(vocab_size, dtype=np.int64)
    # Maps normalized_string -> compressed_nid, one-to-one
    key2new = {}

    for tid in range(vocab_size):
      # Decode the token back to raw text
      text = self.tokenizer.decode([tid], skip_special_tokens=False)
      if "\ufffd" in text:
        # Case 1: Handle invalid/broken byte sequences
        # If decode produces replacement character � (unicode \ufffd)
        # use the raw token string instead
        key = self.tokenizer.convert_ids_to_tokens(tid)
      else:
        # Normalize the text (e.g., "  APPLE" -> "apple")
        normalized_text = self.normalizer.normalize_str(text)
        # Case 2: Use normalized text
        # Case 3: Fall back to raw text if normalization results in empty string
        key = normalized_text if normalized_text else text

      # update key2new
      nid = key2new.get(key)
      if nid is None:
        nid = len(key2new)
        key2new[key] = nid
      # update old2new
      old2new[tid] = nid

    # lookup_table, num_new_token
    return old2new, len(key2new)

  def __call__(self, input_ids):
    """
    Maps original token IDs to canonical IDs using the pre-computed table.
    """
    input_ids = np.asarray(input_ids, dtype=np.int64)
    # Ignore negative IDs (often used for padding/masks)
    valid_mask = input_ids >= 0
    valid_ids = input_ids[valid_mask]
    # Vectorized replacement: O(1) lookup per token
    output_ids = input_ids.copy()
    output_ids[valid_mask] = self.lookup_table[valid_ids]
    return output_ids


class NgramHashMapping:
  """
  Use hash-based mapping for n-gram, as the combinatorial space is intractable for one-to-one mapping.
  - To alleviate collision, initialize with unique prime vocab sizes across heads.
  - When lookup, create n-gram window via shift. Hash via lightweight multiplicative-XOR inside window.
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
    engram_vocab_size: each n-gram has total size > engram_vocab_size[n-2] * n_head_per_ngram
    engram_num_heads: n_head_per_ngram
    """
    self.min_head_vocab_size_per_ngram = engram_vocab_size
    self.max_ngram_size = max_ngram_size
    self.n_head_per_ngram = engram_num_heads
    self.layer_ids = layer_ids

    # initialize compreseed tokenizer
    self.compressed_tokenizer = CompressedTokenizer(tokenizer)
    self.tokenizer_vocab_size = len(self.compressed_tokenizer)
    # TODO: why not just use pad_id = tokenizer.pad_id
    if pad_id is not None:
      self.pad_id = int(self.compressed_tokenizer.lookup_table[pad_id])

    # calculate layer multipliers {layer_id: multiplier}
    self.layer_multipliers = self._calculate_multipliers_across_layers(seed)

    # Each head k maps to an embedding table of prime size
    # {layer_id: [[2gram_head1,...,2gram_headH], ..., [Ngram_head1,...,Ngram_headH]}
    self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

  def _calculate_multipliers_across_layers(self, seed: int):
    """
    return dict {layer_id: a list of max_ngram_size multipliers}
    """
    # Pre-calculate bounds
    max_long = np.iinfo(np.int64).max
    m_max = int(max_long // self.tokenizer_vocab_size)
    half_bound = max(1, m_max // 2)
    # Large prime for seed decorrelation
    LAYER_PRIME_OFFSET = 10007

    layer_multipliers = {}
    for layer_id in self.layer_ids:
      # Generate a layer-specific seed
      layer_seed = int(seed + LAYER_PRIME_OFFSET * int(layer_id))
      np_rng = np.random.default_rng(layer_seed)
      # generate max_ngram_size random integers and transform to odd
      random_value = np_rng.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
      multipliers = random_value * 2 + 1
      layer_multipliers[layer_id] = multipliers
    return layer_multipliers

  def _calculate_vocab_size_across_layers(self):
    """
    Vocabulary Size Calculation (Global Prime Sequence): Engram uses unique prime numbers for the vocabulary size
    of every head in every layer to maximize hash collision independence.

    Example:
      Layers: [0], N-gram sizes: 2-gram and 3-gram (max_ngram_size = 3),
      Heads per N-gram: 2 (n_head_per_ngram = 2), Min Vocab Sizes: [10, 12] (for 2-grams and 3-grams)

      layer 0 - 2-gram - h1 - start (10-1) -> first unseen prime 11
      layer 0 - 2-gram - h2 - start 11 -> first unseen prime 13
      layer 0 - 3-gram - h1 - start (12-1) -> first unseen prime 17
      layer 0 - 3-gram - h2 - start 17 -> first unseen prime 19

      Output: {0: [[11, 13], [17, 19]]}
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

        # use predefined size as start
        n_gram_index = ngram - 2
        vocab_size = self.min_head_vocab_size_per_ngram[n_gram_index]
        current_prime_search_start = vocab_size - 1

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
    Extract the specific list of primes for THIS layer
    The structure is [[ngram2_head1, ngram2_head2...], [ngram3_head1...]]
    We flatten it into a single list of ints: [N1, N2, N3, ...]
    """
    return [x for y in self.vocab_size_across_layers[layer_id] for x in y]

  def _get_ngram_hashes(
      self,
      compressed_ids: np.ndarray,
      layer_id: int,
  ) -> np.ndarray:
    """
    Args:
      compressed_ids: (B, S)

    Returns:
      hash_ids for this layer, (B, S, H_total) where H_total = H * num_ngram_orders

    Example:
      Tokens: The cat sat -> Compressed IDs: [10, 25, 32]
      max ngram = 3

      3 multiplier: m0=3, m1=5, m3=7 (specific to l-th layer)

      sliding window via shifting:
        shift0: [The, cat, sat] -> [10, 25, 32]
        shift1: [PAD, The, cat] -> [0, 10, 25]
        shift2: [PAD, PAD, The] -> [0, 0, 10]

      2-gram: (shift0 * m0) XOR (shift1 * m1)
        [sat, cat] -> (32 * 3) ^ (25 * 5)

      3-gram: (shift0 * m0) XOR (shift1 * m1) XOR (shift2 * m2)
        [sat, cat, The] -> (32 * 3) ^ (25 * 5) ^ (10 * 7)

      mod by vocab size m (specific to l-th layer, n-gram, h-th head)
    """

    # 1 sliding window via shifting
    def shift_k(x, k: int) -> np.ndarray:
      if k == 0:
        return x
      T = x.shape[1]
      shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[:, :T]
      return shifted

    x = np.asarray(compressed_ids, dtype=np.int64)
    base_shifts = [shift_k(x, k) for k in range(self.max_ngram_size)]

    # 2 multiplier for each shift
    multipliers = self.layer_multipliers[layer_id]

    # 3 calculate hashes across n-grams
    all_hashes = []
    mix = base_shifts[0] * multipliers[0]  # init hash
    vocab_sizes = self.vocab_size_across_layers[layer_id]  # init vocab as mod
    for n in range(2, self.max_ngram_size + 1):
      # multiplicative bitwise xor as hash function
      k = n - 1
      mix = np.bitwise_xor(mix, base_shifts[k] * multipliers[k])
      # for H heads at this n-gram level, get primes vocab sizes
      n_gram_index = n - 2
      vocab_sizes_for_this_gram = vocab_sizes[n_gram_index]
      mods = np.array(vocab_sizes_for_this_gram, dtype=np.int64)
      # Broadcast Modulo: (B, S, 1) % (H,) -> (B, S, H)
      head_hashes = mix[..., None] % mods
      all_hashes.append(head_hashes)

    # (B, S, H_total), H_total = H * num_ngram_orders
    all_hashes = np.concatenate(all_hashes, axis=2)
    return all_hashes

  def __call__(self, input_ids):
    compressed_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}
    for layer_id in self.layer_ids:
      hash_ids = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
      hash_ids_for_all_layers[layer_id] = hash_ids
    return hash_ids_for_all_layers


class MultiHeadEmbedding(nnx.Module):

  def __init__(self, vocab_sizes: List[int], head_dim: int, config, mesh, rngs: nnx.Rngs):
    """
    Args:
      vocab_sizes: A list of prime-based vocab sizes, each element is h-th head for n-gram
        for example, 2-gram and 3-gram, each with 2 heads. For instance, flattened list can be
        m[n=2,h=0]=2, m[n=2,h=1]=3, m[n=3,h=0]=5, m[n=3,h=1]=7
      head_dim: The embedding dimension for a single head.
    """
    self.num_heads = len(vocab_sizes)

    # The embedding for heads across n-grams are stored in a single flattened table.
    # Offsets act as the boundaries. Prefix sum to get the start position.
    # If vocab_sizes is [100, 200, 150], offsets will be [0, 100, 300].
    offsets = np.cumsum([0] + vocab_sizes[:-1])
    self.offsets = jnp.array(offsets, dtype=jnp.int32)

    # Total size is the sum of vocabularies acorss all heads all n-grams
    self.embedding = Embed(num_embeddings=sum(vocab_sizes), num_features=head_dim, config=config, mesh=mesh, rngs=rngs)

  def __call__(self, input_ids: jax.Array, model_mode: str = MODEL_MODE_TRAIN) -> jax.Array:
    """
    Args:
      input_ids: Indices from MultiHeadHashing, shape (B, S, H_total)

    Returns:
      embedding: (B, S, H_total, D_head)
    """
    # Shift to indices in flattened table
    shifted_ids = input_ids + self.offsets

    # (batch, length, num_heads_total, head_dim)
    return self.embedding(shifted_ids, model_mode=model_mode)


class ShortConv(nnx.Module):
  """
  Implements a Grouped Depthwise Causal Convolution block.

  Applies local temporal mixing (smoothing) to the retrieved embeddings.
  - Independent RMSNorms for each branch
  - 1D convolution. Note it is depth-wise:
    mixes information across time steps [t-k, t] without mixing across channels.
  """

  def __init__(
      self,
      config,
      hidden_size: int,  # base_emb_dim
      kernel_size: int = 4,  # Temporal Window Size
      dilation: int = 1,
      hc_mult: int = 4,
      rngs: nnx.Rngs = None,
  ):
    self.hc_mult = hc_mult
    # (G * D) as the total number of input channels
    total_channels = hidden_size * hc_mult

    # Norm (vectorized)
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

    # Weights: {"scale": (hc_mult, hidden_size)}
    self.norms = create_norms(rngs)

    # Convolution (shared)
    # Depthwise: feature_group_count=in_features, channels don't mix. mixes temporal dimension only
    # Padding: "CAUSAL" ensures output[t] only depends on input[t-k : t]
    # Weights: {"kernel": (kernel_size, in_features//feature_group_count=1, total_channels)}
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

  def __call__(self, x: jax.Array) -> jax.Array:
    """
    Compute y^i = SiLU(Conv1D(RMSNorm^i(x^i))), i is index for branch dim.

    Args:
      x: Input tensor of shape (B, S, G, D)
    Returns:
      Tensor of shape (B, S, G, D)

    Shape annotation:
      B: Batch Size
      S: Sequence Length
      G: Number of Branches (hc_mult)
      D: base_emb_dim
    """
    B, S, G, D = x.shape

    # Apply Norms (Vectorized over Group dim)
    # in_axes=(0, 2): norms is axis 0, x is axis 2, out_axes=2: put the group dim back at axis 2
    # shape stays (B, S, G, D)
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    x = apply_norms(self.norms, x)

    # Flatten branch: (B, S, G, D) -> (B, S, G * D)
    x_flat = x.reshape(B, S, G * D)
    # Depthwise Convolution to mixes temporal dimension S only. Shape stays (B, S, G * D)
    y = self.conv(x_flat)
    # Activation
    y = jax.nn.silu(y)
    # Restore branch: (B, S, G * D) -> (B, S, G, D)
    return y.reshape(B, S, G, D)


class Engram(nnx.Module):
  """
  Implements the Engram Memory Layer with n-gram statistics.

  It follows a Retrieve-and-Gate paradigm:
  - Context-independent Retrieval: Fetch n-gram embeddings using Multi-Head Hashing as static memory.
  - Context-aware Gating: Decide how much of this retrieved memory to use based on the current dynamic state.
  - Mix: Apply local temporal smoothing via ShortConv.

  Note: vocab_sizes = hash_mapping.get_vocab_sizes(layer_id)
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      config,
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
    hc_mult (G): > 1 - use MHC, 1 - not use MHC
    engram_max_ngram_size: track 2...N-grams
    engram_kernel_size: 1d convolution kernel size
    engram_num_heads (H): Number of heads per n-gram order
    engram_head_dim (D_head): Dimension of a single head. The actual size stored in the table.
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
    # D_en: Final concatenated size
    self.engram_dim = engram_head_dim * engram_num_heads * num_ngram_orders

    # Embedding: all n-gram heads in one flattened table
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

  def __call__(self, hidden_states: jax.Array, hash_input_ids: jax.Array) -> jax.Array:
    """
    Args:
      hidden_states: Current transformer state (Query). Shape: (B, S, G, D)
      hash_input_ids: Hashed token IDs. Shape: (B, S, H_total).
        Produced by `hash_mapping.hash(input_ids)[layer_id]`.
    Returns:
      Shape: (B, S, G, D)

    - retireve memory E
    - compute similarity between memory and context
        K^i = RMSNormK^i(Wk^i @ E)
        Q^i = RMSNormQ^i(hidden)
        gate^i = sigmoid(K^i @ Q^i)
    - updated memory
        V = Wv @ E
        Vnew^i = gate^i * V
    - temporal smooth
        out = ShortConv(Vnew) + Vnew

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
    # compress range for gate: sign(x) * sqrt(max(|x|, 1e-6))
    # stabilize input to sigmoid
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
