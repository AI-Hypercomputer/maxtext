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
  Implements a lossy, canonicalizing wrapper for a standard tokenizer to optimize
  n-gram pattern matching.

  By applying aggressive normalization (lowercasing, accent stripping, and whitespace
  collapsing), this class maps multiple distinct tokens (e.g., 'Apple', ' apple',
  'APPLE') to a single unified ID. This many-to-one mapping effectively reduces
  the combinatorial n-gram space.

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
    # num_new_token equals to len(key2new)
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

  def __init__(
      self,
      engram_vocab_size,
      max_ngram_size,
      n_head_per_ngram,
      layer_ids,
      tokenizer,
      pad_id,
      seed,
  ):
    """
    engram_vocab_size: each n-gram has total size > engram_vocab_size[n-2] * n_head_per_ngram
    """
    self.min_head_vocab_size_per_ngram = engram_vocab_size
    self.max_ngram_size = max_ngram_size
    self.n_head_per_ngram = n_head_per_ngram
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

    Example: Layers: [0], N-gram sizes: 2-gram and 3-gram (max_ngram_size = 3),
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

        num_head = self.n_head_per_ngram
        for _ in range(num_head):
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

    # (B, S, H * (max_ngram_size-1))
    return np.concatenate(all_hashes, axis=2)

  def __call__(self, input_ids):
    compressed_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}
    for layer_id in self.layer_ids:
      hash_ids = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
      hash_ids_for_all_layers[layer_id] = hash_ids
    return hash_ids_for_all_layers


class MultiHeadEmbedding(nnx.Module):

  def __init__(self, vocab_sizes: List[int], dim_per_head: int, config, mesh, rngs: nnx.Rngs):
    """
    Args:
      vocab_sizes: A list of prime-based vocab sizes, each element is h-th head for n-gram
          for example, 2-gram and 3-gram, each with 2 heads. For instance, flattened list can be
          m[n=2,h=0]=2, m[n=2,h=1]=3, m[n=3,h=0]=5, m[n=3,h=1]=7
      dim_per_head: The embedding dimension for a single head.
    """
    self.num_heads = len(vocab_sizes)

    # The embedding for heads across n-grams are stored in a single flattened table.
    # Offsets act as the boundaries. Prefix sum to get the start position.
    # If vocab_sizes is [100, 200, 150], offsets will be [0, 100, 300].
    offsets = np.cumsum([0] + vocab_sizes[:-1])
    self.offsets = jnp.array(offsets, dtype=jnp.int32)

    # Total size is the sum of vocabularies acorss all heads all n-grams
    self.embedding = Embed(
        num_embeddings=sum(vocab_sizes), num_features=dim_per_head, config=config, mesh=mesh, rngs=rngs
    )

  def __call__(self, input_ids: jax.Array, model_mode: str = MODEL_MODE_TRAIN) -> jax.Array:
    """
    Args:
      input_ids: Indices from MultiHeadHashing, shape (Batch, Length, Num_Heads)
    """
    # Shift to indices in flattened table
    shifted_ids = input_ids + self.offsets

    # (batch, length, num_heads, dim)
    return self.embedding(shifted_ids, model_mode=model_mode)


class ShortConv(nnx.Module):
  """
  Implements a Grouped Depthwise Causal Convolution block.

  Applies local temporal mixing (smoothing) to the retrieved embeddings.
  - It uses independent RMSNorms for each branch
  - followed by a 1D convolution. Note it is depth-wise:
    mixes information across time steps [t-k, t] without mixing across channels.

  Shape Legend:
      B: Batch Size
      S: Sequence Length
      G: Number of Branches (hc_mult) - logical grouping of heads
      C: Embedding Dimension per Branch (hidden_size)

  Note on Convolution:
      Conv1D - (G * C) as the total number of input channels.
      Depthwise - It applies a separate filter to every single dimension in C, for every group G.
  """

  def __init__(
      self,
      config,
      hidden_size: int,
      kernel_size: int = 4,  # Temporal Window Size
      dilation: int = 1,
      hc_mult: int = 4,
      activation: bool = True,
      rngs: nnx.Rngs = None,
  ):
    self.hc_mult = hc_mult
    total_channels = hidden_size * hc_mult

    # B: Vectorized Norms (One unique module per group)
    # TODO(shuningjin): eps, epsilon=config.normalization_layer_epsilon,
    #                   epsilon=1e-5,  # Match PyTorch default
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(r):
      return RMSNorm(
          num_features=hidden_size,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=1e-5,
          rngs=r,
      )

    # Weights: {"scale": (hc_mult, hidden_size)}
    self.norms = create_norms(rngs)

    # Single Shared Convolution
    #    Depthwise: channels don't mix, feature_group_count=in_features
    #    Padding: "CAUSAL" ensures output[t] only depends on input[t-k : t].
    # Weights: {"kernel": (kernel_size, in_features//feature_group_count=1, total_channels)}
    self.conv = nnx.Conv(
        in_features=total_channels,
        out_features=total_channels,
        kernel_size=(kernel_size,),
        feature_group_count=total_channels,
        kernel_dilation=(dilation,),
        padding="CAUSAL",  # To match the slice [..., :T] logic, TODO(shuningjin)
        use_bias=False,
        rngs=rngs,
    )

    self.act_fn = jax.nn.silu if activation else lambda x: x

  def __call__(self, x: jax.Array) -> jax.Array:
    """
    y = SiLU(Conv1D(RMSNorm(x)))

    Args:
        x: Input tensor of shape (B, S, G, C)
    Returns:
        Tensor of shape (B, S, G, C)

    Note: G = hc_mult
    """
    B, S, G, C = x.shape

    # Apply Norms (Vectorized over Group dim)
    # in_axes=(0, 2): norms is axis 0, x is axis 2, out_axes=2: put the group dim back at axis 2
    # shape stays (B, S, G, C)
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    x = apply_norms(self.norms, x)

    # Flatten branch: (B, S, G, C) -> (B, S, G * C)
    x_flat = x.reshape(B, S, G * C)
    # Apply depthwise conv to mixes temporal dimension only
    # Shape stays (B, S, G * C)
    y = self.conv(x_flat)
    y = self.act_fn(y)

    # Restore branch: (B, S, G * C) -> (B, S, G, C)
    return y.reshape(B, S, G, C)


class Engram(nnx.Module):
  """
  Implements the Engram Memory Layer.

  This layer augments the standard transformer hidden states with long-range
  n-gram statistics. It follows a Retrieve-and-Gate architecture:
  1. Retrieve: Fetch embeddings for current n-gram contexts using Multi-Head Hashing.
  2. Gate: Decide how much of this retrieved memory to merge based on the current state.
  3. Mix: Apply local temporal smoothing via ShortConv.

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
      vocab_sizes,
      hc_mult,
      engram_heads_per_ngram,
      engram_embed_dim_per_ngram,
      engram_max_ngram_size,
      engram_kernel_size,
  ):
    self.config = config
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.dtype
    self.kernel_init = kernel_init
    self.quant = quant
    self.rngs = rngs
    self.hc_mult = hc_mult

    # Hierarchy: Engram -> n-gram Order -> h-th Head
    # Raw Inputs
    self.max_ngram_size = engram_max_ngram_size  # e.g., 4 (tracks 2,3,4-grams)
    self.conv_kernel_size = engram_kernel_size
    # H: Number of heads per n-gram order
    self.num_heads = engram_heads_per_ngram
    # D_total: Total embedding dimension for ONE n-gram order (sum of all K heads)
    self.dim_per_ngram = engram_embed_dim_per_ngram
    # D_head: Dimension of a single head (The actual size stored in the table)
    # Logic: We split the total dimension D evenly across K heads.
    self.dim_per_head = self.dim_per_ngram // self.num_heads
    # How many n-gram orders are we tracking? (e.g. 2, 3, 4 -> 3 orders)
    self.num_orders = self.max_ngram_size - 1
    # Final concatenated size: (num orders) * (dim per rrder)
    self.engram_dim = self.num_orders * self.dim_per_ngram

    # Embedding for all n-gram heads in one flattened table.
    self.mhe = MultiHeadEmbedding(
        vocab_sizes=vocab_sizes, dim_per_head=self.dim_per_head, config=config, mesh=mesh, rngs=rngs
    )

    # --- 2. Short Convolution (Already Vectorized internally) ---
    # Applies depthwise causal convolution to smooth the retrieved memory over time.
    self.short_conv = ShortConv(
        config=config,
        hidden_size=config.base_emb_dim,
        kernel_size=self.conv_kernel_size,
        dilation=self.max_ngram_size,
        hc_mult=hc_mult,
        rngs=rngs,
    )

    # --- 3. Vectorized Gating Layers ---

    # Project retrieved memory into Value space
    # A. Value Projection (Shared input, Shared weights -> Standard Layer)
    self.value_proj = DenseGeneral(
        in_features_shape=self.engram_dim,
        out_features_shape=config.base_emb_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("a", "b"),  # TODO(shuningjin)
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
        use_bias=True,
    )

    # Project retrieved memory into Key space (one per branch)
    # B. Key Projections (Shared input, Independent weights per group)
    # We create G copies of DenseGeneral
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_key_projs(r):
      return DenseGeneral(
          in_features_shape=self.engram_dim,
          out_features_shape=config.base_emb_dim,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("a", "b"),  # TODO(shuningjin)
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=r,
          use_bias=True,
      )

    self.key_projs = create_key_projs(rngs)

    # C. Norms (Independent weights per group)
    # TODO(shuningjin): epsilon=config.normalization_layer_epsilon,
    # epsilon=1e-6,  # Match PyTorch default
    @nnx.split_rngs(splits=hc_mult)
    @nnx.vmap(in_axes=0, out_axes=0)
    def create_norms(r):
      return RMSNorm(
          num_features=config.base_emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=1e-6,
          rngs=r,
      )

    # Normalization before Gating (Key)
    self.k_norms = create_norms(rngs)
    # Normalization before Gating (Query)
    # Note: Creates separate parameters for Q norms
    self.q_norms = create_norms(rngs)

  def __call__(self, hidden_states: jax.Array, hash_input_ids: jax.Array) -> jax.Array:
    """
    Args:
        hidden_states: Current transformer state (Query). Shape: (B, S, G, C)
        input_ids: Raw token IDs. Shape: (B, S)
    Returns:
        output: Engram-augmented residuals. Shape: (B, S, G, C)

    Note: G = hc_mult
    Note: hash_input_ids = hash_mapping.hash(input_ids)[layer_id]
    """
    B, S, G, C = hidden_states.shape

    # 1. Retrieve Memory
    # 1. Generate Hash Indices
    # Map raw text -> n-gram contexts -> hash indices z_{t,n,k}
    # (B, S) -> (B, S, H_en), where H_en is the total count of heads across all n-gram orders.
    # hash_input_ids = jnp.array(self.hash_mapping.hash(input_ids)[self.layer_id])

    # 2. Retrieve Memory
    # Fetch e_{t,n,k} from the embedding table.
    # Flatten all n-gram heads into one vector e_t
    # (B, S, H_en) -> (B, S, H_en, D_head) -> (B, S, D_en)
    embeddings = self.mhe(hash_input_ids).reshape(B, S, -1)

    # 3. Gating Mechanism (Scaled Dot-Product)
    # Decide relevance of memory (Key) to current state (Query)

    # 2. Compute Keys (Vectorized Broadcast)
    # We want to apply each of the G key_projs to the SAME embeddings.
    # in_axes: (0, None) -> 0 splits the Dense layers, None broadcasts embeddings
    # out_axes: 2        -> Stack the results at axis 2 to get (B, S, G, C)
    @nnx.vmap(in_axes=(0, None), out_axes=2)
    def apply_projs(projs, x):
      return projs(x)

    # Key: Projection of retrieved n-gram memory
    #  (B, S, D_en) ->  (B, S, G, C)
    keys_unnorm = apply_projs(self.key_projs, embeddings)

    # 3. Compute Norms (Vectorized Map)
    # Map over the G dimension (Axis 2) for both weights and input
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    # (B, S, G, C) shape stays
    k = apply_norms(self.k_norms, keys_unnorm)

    # Query: Current hidden state from the transformer
    # (B, S, G, C) shape stays
    q = apply_norms(self.q_norms, hidden_states)

    # 4. Gating (Vectorized)
    # Gate Score (Scalar per token)

    # Dot product: (B, S, G)
    qk_product = jnp.einsum("bsgc,bsgc->bsg", q, k, precision=self.config.matmul_precision)
    # Scale
    gate_logits = qk_product / jnp.sqrt(C)
    # Stabilize
    # TODO(shuningjin): why, Stabilization trick: sign(x) * sqrt(|x|)
    gate_logits = jnp.sqrt(jnp.maximum(jnp.abs(gate_logits), 1e-6)) * jnp.sign(gate_logits)
    # Sigmoid activation to get gating probability [0, 1]
    gates = jax.nn.sigmoid(gate_logits)  # (B, S, G)

    # 5. Value Projection & Gating: v = Gate * Value_Proj(Memory)

    # Project Engram Memory to Value: (B, S, D_en) -> (B, S, C)
    value = self.value_proj(embeddings)

    # Apply gate to value: broadcast and multiply
    # (B, S, G, 1) * (B, S, 1, C) -> (B, S, G, C)
    v = gates[:, :, :, None] * value[:, :, None, :]

    # 6. Temporal Smoothing (ShortConv)
    # Apply Depthwise Conv (mixes L, keeps G and C independent)
    # Shape remains, (B, S, G, C)
    conv_out = self.short_conv(v)

    # residual for conv component
    out = v + conv_out

    # Note: hidden_states will be added by the caller
    return out
