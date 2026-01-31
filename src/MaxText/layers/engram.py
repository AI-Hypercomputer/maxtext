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

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

import jax
import jax.numpy as jnp
from flax import nnx

from MaxText.common_types import ShardMode, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, Array, Config, DType
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
    # Maps original_tid -> compressed_nid
    old2new = {}
    # Maps normalized_string -> compressed_nid
    key2new = {}
    # List of unique canonical strings
    new_tokens = []

    vocab_size = len(self.tokenizer)
    for tid in range(vocab_size):
      # 1. Decode the token back to raw text
      text = self.tokenizer.decode([tid], skip_special_tokens=False)

      # 2. Handle invalid/broken byte sequences (e.g., partial UTF-8 tokens)
      if "�" in text:
        # If decode fails, use the raw token string (e.g., 'Ġ', '<0x0A>')
        key = self.tokenizer.convert_ids_to_tokens(tid)
      else:
        # 3. Normalize the text (e.g., "  APPLE" -> "apple")
        norm = self.normalizer.normalize_str(text)
        # Fallback to original if norm results in empty
        key = norm if norm else text

      # 4. Deduplicate: if "Apple" and "apple" both become "apple", they get the same ID
      nid = key2new.get(key)
      if nid is None:
        nid = len(new_tokens)
        key2new[key] = nid
        new_tokens.append(key)
      old2new[tid] = nid

    # 5. Create a high-speed NumPy lookup array for the forward pass
    lookup = np.empty(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
      lookup[tid] = old2new[tid]

    return lookup, len(new_tokens)

  def _compress(self, input_ids):
    """
    Maps original token IDs to canonical IDs using the pre-computed table.
    """
    input_ids = np.asarray(input_ids, dtype=np.int64)
    # Ignore negative IDs (often used for padding/masks)
    valid_mask = input_ids >= 0
    valid_ids = input_ids[valid_mask]
    output_ids = input_ids.copy()
    # Vectorized replacement: O(1) lookup per token
    output_ids[valid_mask] = self.lookup_table[valid_ids]
    return output_ids

  def __call__(self, input_ids):
    return self._compress(input_ids)


class NgramHashMapping:
  """
  non-parametric
  """

  def __init__(
      self,
      engram_vocab_size,
      max_ngram_size,
      n_embed_per_ngram,
      n_head_per_ngram,
      layer_ids,
      tokenizer,
      pad_id,
      seed,
  ):
    self.vocab_size_per_ngram = engram_vocab_size
    self.max_ngram_size = max_ngram_size
    self.n_embed_per_ngram = n_embed_per_ngram
    self.n_head_per_ngram = n_head_per_ngram
    self.pad_id = pad_id
    self.layer_ids = layer_ids

    self.compressed_tokenizer = CompressedTokenizer(tokenizer)
    self.tokenizer_vocab_size = len(self.compressed_tokenizer)
    if self.pad_id is not None:
      self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

    max_long = np.iinfo(np.int64).max
    M_max = int(max_long // self.tokenizer_vocab_size)
    half_bound = max(1, M_max // 2)
    PRIME_1 = 10007

    self.layer_multipliers = {}

    for layer_id in self.layer_ids:
      base_seed = int(seed + PRIME_1 * int(layer_id))
      g = np.random.default_rng(base_seed)
      r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
      multipliers = r * 2 + 1
      self.layer_multipliers[layer_id] = multipliers

    # Each head k maps to an embedding table of prime size
    self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

  def find_next_prime(self, start, seen_primes):
    candidate = start + 1
    while candidate in seen_primes or not isprime(candidate):
      candidate += 1
    return candidate

  def calculate_vocab_size_across_layers(self):
    seen_primes = set()
    vocab_size_across_layers = {}

    for layer_id in self.layer_ids:
      all_ngram_vocab_sizes = []
      for ngram in range(2, self.max_ngram_size + 1):
        current_ngram_heads_sizes = []
        # Maps n-gram order to the list index
        # If ngram=2 (Bigram), index is 0.
        # If ngram=3 (Trigram), index is 1.
        vocab_size = self.vocab_size_per_ngram[ngram - 2]
        current_prime_search_start = vocab_size - 1

        num_head = self.n_head_per_ngram
        for _ in range(num_head):
          found_prime = self.find_next_prime(current_prime_search_start, seen_primes)
          seen_primes.add(found_prime)
          current_ngram_heads_sizes.append(found_prime)
          current_prime_search_start = found_prime

        all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
      vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

    return vocab_size_across_layers

  def _get_ngram_hashes(
      self,
      input_ids: np.ndarray,
      layer_id: int,
  ) -> np.ndarray:
    x = np.asarray(input_ids, dtype=np.int64)
    B, T = x.shape

    multipliers = self.layer_multipliers[layer_id]

    def shift_k(k: int) -> np.ndarray:
      if k == 0:
        return x
      shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[:, :T]
      return shifted

    base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

    all_hashes = []

    for n in range(2, self.max_ngram_size + 1):
      n_gram_index = n - 2
      tokens = base_shifts[:n]
      # Lightweight multiplicative-XOR hash
      mix = tokens[0] * multipliers[0]
      for k in range(1, n):
        mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
      num_heads_for_this_ngram = self.n_head_per_ngram
      head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

      # For each n-gram order n, we employ K distinct hash heads
      for j in range(num_heads_for_this_ngram):
        mod = int(head_vocab_sizes[j])
        head_hash = mix % mod
        all_hashes.append(head_hash.astype(np.int64, copy=False))

    return np.stack(all_hashes, axis=2)

  def hash(self, input_ids):
    input_ids = self.compressed_tokenizer(input_ids)
    hash_ids_for_all_layers = {}
    for layer_id in self.layer_ids:
      hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
    return hash_ids_for_all_layers

  def get_vocab_sizes(self, layer_id: int):
    # 2. Extract the specific list of primes for THIS layer
    # The structure is [[ngram2_head1, ngram2_head2...], [ngram3_head1...]]
    # We flatten it into a single list of ints: [N1, N2, N3, ...]
    return [x for y in self.vocab_size_across_layers[layer_id] for x in y]


class MultiHeadEmbedding(nnx.Module):

  def __init__(self, list_of_N: List[int], D: int, config, mesh, rngs: nnx.Rngs):
    """
    Args:
      list_of_N: A list of prime-based vocab sizes, each element is k-th head for n-gram
          for example, 2-gram and 3-gram, each with 2 heads, the flattened list is
          e.g., m[n=2,k=0]=2, m[n=2,k=1]=3, m[n=3,k=0]=5, m[n=3,k=1]=7
      D: The embedding dimension (for a single head).
    """
    self.num_heads = len(list_of_N)

    # To implement the concatenation (||) efficiently, we store all E[n,k] for all k
    # in a single flattened table. Offsets act as the boundaries between
    # E[n,k] and E[n,k+1]

    # 1. Calculate the starting position for each head's memory block.
    # If list_of_N is [100, 200, 150], offsets will be [0, 100, 300].
    # This prevents different heads from colliding in the single large table.

    # prefix sum
    offsets = np.cumsum([0] + list_of_N[:-1])
    self.offsets = jnp.array(offsets, dtype=jnp.int32)

    # 2. Instantiate a single large embedding table.
    # Total size is the sum of all individual head vocabularies.
    self.embedding = Embed(num_embeddings=sum(list_of_N), num_features=D, config=config, mesh=mesh, rngs=rngs)

  def __call__(self, input_ids: jax.Array, model_mode: str = MODEL_MODE_TRAIN) -> jax.Array:
    """
    Args:
      input_ids: Indices from MultiHeadHashing, shape (Batch, Length, Num_Heads)
    """
    # 3. Shift local head indices to global table indices.
    # Example: If Head 1 wants index 5, it stays 5. If Head 2 wants index 5,
    # it becomes 5 + 100 = 105.
    shifted_ids = input_ids + self.offsets

    # 4. Perform a vectorized gather from the large embedding table.
    # Result shape: (Batch, Length, Num_Heads, D)
    return self.embedding(shifted_ids, model_mode=model_mode)


class ShortConv(nnx.Module):
  """
  Implements a Grouped Depthwise Causal Convolution block.

  This module applies local temporal mixing (smoothing) to the retrieved embeddings.
  - It uses independent RMSNorms for each branch
  - followed by a 1D convolution. Note it is depth-wise:
    mixes information across time steps [t-k, t] without mixing across channels.

  Shape Legend:
      B: Batch Size
      L: Sequence Length
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

    # A: Single Shared Convolution
    # Note: feature_group_count=in_features makes this Depthwise (channels don't mix)
    # Padding="CAUSAL" ensures output[t] only depends on input[t-k : t].
    # weights: {"kernel": (kernel_size, in_features//feature_group_count=1, total_channels)}
    self.conv = nnx.Conv(
        in_features=total_channels,
        out_features=total_channels,
        kernel_size=(kernel_size,),
        feature_group_count=total_channels,  # Depthwise
        kernel_dilation=(dilation,),
        padding="CAUSAL",  # To match the slice [..., :T] logic
        use_bias=False,
        rngs=rngs,
    )

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

    # weights: {"scale": (hc_mult, hidden_size)}
    self.norms = create_norms(rngs)

    self.act_fn = jax.nn.silu if activation else lambda x: x

  def __call__(self, x: jax.Array) -> jax.Array:
    """
    y = SiLU(Conv1D(RMSNorm(x)))

    Args:
        x: Input tensor of shape (B, L, G, C)
    Returns:
        Tensor of shape (B, L, G, C)

    Note: G = hc_mult
    """
    B, L, G, C = x.shape

    # 1. Apply Norms (Vectorized over Group dim)
    # in_axes=(0, 2):  norms is axis 0, x is axis 2
    # out_axes=2:      put the group dim back at axis 2
    # shape stays: (B, L, G, C)
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    x = apply_norms(self.norms, x)

    # 2. Flatten Groups for Conv
    # (B, L, G, C) -> (B, L, G * C)
    x_flat = x.reshape(B, L, G * C)

    # 3. Apply Single Conv
    # Causal; Depthwise (Mixes temporal dimension L. Channels remain independent)
    # Shape stays: (B, L, G * C)
    y = self.conv(x_flat)
    y = self.act_fn(y)

    # 4. Reshape back to Branched Layout
    # (B, L, G * C) -> (B, L, G, C)
    return y.reshape(B, L, G, C)

  # -----------------------------------------------------------------------
  # Vocabulary Size Calculation (Global Prime Sequence)
  # -----------------------------------------------------------------------
  # Engram uses unique prime numbers for the vocabulary size of every head
  # in every layer to maximize hash collision independence.

  # We instantiate NgramHashMapping here to replicate this deterministic
  # sequence. Ideally, this mapping should be created once globally and
  # the resulting vocab_sizes passed into this layer to improve startup time.

  # # --- Hash Mapping ---
  # self.hash_mapping = NgramHashMapping(
  #     engram_vocab_size=engram_vocab_size,
  #     max_ngram_size=engram_max_ngram_size,
  #     n_embed_per_ngram=engram_embed_dim_per_ngram,
  #     n_head_per_ngram=engram_heads_per_ngram,
  #     # IMPORTANT: We must pass the FULL list of layer_ids, not just self.layer_id.
  #     # The mapping finds primes sequentially across all layers; passing a partial
  #     # list would reset the prime search and break alignment with the reference model.
  #     layer_ids=layer_ids,
  #     # Inject the pre-loaded tokenizer to avoid redundant disk I/O per layer.
  #     tokenizer=tokenizer,
  #     pad_id=pad_id,
  #     seed=seed,
  # )

  # # Extract the specific list of primes [M_{n,k}] for THIS layer only.
  # # The structure is [[ngram2_head1, ngram2_head2...], [ngram3_head1...]]
  # # We flatten it into a single list of ints: [N1, N2, N3, ...]
  # vocab_sizes = self.hash_mapping.get_vocab_sizes(self.layer_id)
  # print(f"DEBUG JAX Layer, Vocab Sizes: {vocab_sizes}")


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

    # --- Dimensions ---
    # -----------------------------------------------------------------------
    # 2. Engram Dimensions & Hyperparameters
    # -----------------------------------------------------------------------

    # Hierarchy: Engram -> n-gram order (n) -> k-th head (k)
    # Raw Inputs
    self.max_ngram_size = engram_max_ngram_size  # e.g., 4 (tracks 2,3,4-grams)
    # self.vocab_size = engram_vocab_size
    self.conv_kernel_size = engram_kernel_size
    # The Hierarchy (Paper Notation)
    # K: Number of heads per n-gram order
    self.num_heads = engram_heads_per_ngram
    # D_total: Total embedding dimension for ONE n-gram order (sum of all K heads)
    self.dim_per_ngram = engram_embed_dim_per_ngram
    # D_head: Dimension of a single head (The actual size stored in the table)
    # Logic: We split the total dimension D evenly across K heads.
    self.dim_per_head = self.dim_per_ngram // self.num_heads
    # Flattened Tensor Sizes
    # How many n-gram orders are we tracking? (e.g. 2, 3, 4 -> 3 orders)
    self.num_orders = self.max_ngram_size - 1
    # Final concatenated size: (Num Orders) * (Dim per Order)
    self.engram_dim = self.num_orders * self.dim_per_ngram

    # --- 1. Multi-Head Embedding ---
    # Stores the learnable vectors E_{n,k} for all n-gram heads in one flattened table.
    self.mhe = MultiHeadEmbedding(list_of_N=vocab_sizes, D=self.dim_per_head, config=config, mesh=mesh, rngs=rngs)

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
        hidden_states: Current transformer state (Query). Shape: (B, L, G, C)
        input_ids: Raw token IDs. Shape: (B, L)
    Returns:
        output: Engram-augmented residuals. Shape: (B, L, G, C)

    Note: G = hc_mult
    Note: hash_input_ids = hash_mapping.hash(input_ids)[layer_id]
    """
    B, L, G, C = hidden_states.shape

    # 1. Retrieve Memory
    # 1. Generate Hash Indices
    # Map raw text -> n-gram contexts -> hash indices z_{t,n,k}
    # (B, L) -> (B, L, H_en), where H_en is the total count of heads across all n-gram orders.
    # hash_input_ids = jnp.array(self.hash_mapping.hash(input_ids)[self.layer_id])

    # 2. Retrieve Memory
    # Fetch e_{t,n,k} from the embedding table.
    # Flatten all n-gram heads into one vector e_t
    # (B, L, H_en) -> (B, L, H_en, D_head) -> (B, L, D_en)
    embeddings = self.mhe(hash_input_ids).reshape(B, L, -1)

    # 3. Gating Mechanism (Scaled Dot-Product)
    # Decide relevance of memory (Key) to current state (Query)

    # 2. Compute Keys (Vectorized Broadcast)
    # We want to apply each of the G key_projs to the SAME embeddings.
    # in_axes: (0, None) -> 0 splits the Dense layers, None broadcasts embeddings
    # out_axes: 2        -> Stack the results at axis 2 to get (B, L, G, C)
    @nnx.vmap(in_axes=(0, None), out_axes=2)
    def apply_projs(projs, x):
      return projs(x)

    # Key: Projection of retrieved n-gram memory
    #  (B, L, D_en) ->  (B, L, G, C)
    keys_unnorm = apply_projs(self.key_projs, embeddings)

    # 3. Compute Norms (Vectorized Map)
    # Map over the G dimension (Axis 2) for both weights and input
    @nnx.vmap(in_axes=(0, 2), out_axes=2)
    def apply_norms(norms, x):
      return norms(x)

    # K: (B, L, G, C) shape stays
    k = apply_norms(self.k_norms, keys_unnorm)

    # Query: Current hidden state from the transformer
    # Q: (B, L, G, C) shape stays
    q = apply_norms(self.q_norms, hidden_states)

    # 4. Gating (Vectorized)
    # Gate Score (Scalar per token)

    # Dot product: (B, L, G)
    qk_product = jnp.einsum("blgc,blgc->blg", q, k, precision=self.config.matmul_precision)
    # Scale
    gate_logits = qk_product / jnp.sqrt(C)
    # Stabilize
    # TODO(shuningjin): why, Stabilization trick: sign(x) * sqrt(|x|)
    gate_logits = jnp.sqrt(jnp.maximum(jnp.abs(gate_logits), 1e-6)) * jnp.sign(gate_logits)
    # Sigmoid activation to get gating probability [0, 1]
    gates = jax.nn.sigmoid(gate_logits)  # (B, L, G)

    # 5. Value Projection & Gating: v = Gate * Value_Proj(Memory)

    # Project Engram Memory to Value: (B, L, D_en) -> (B, L, C)
    value = self.value_proj(embeddings)

    # Apply gate to value: broadcast and multiply
    # (B, L, G, 1) * (B, L, 1, C) -> (B, L, G, C)
    v = gates[:, :, :, None] * value[:, :, None, :]

    # 6. Temporal Smoothing (ShortConv)
    # Apply Depthwise Conv (mixes L, keeps G and C independent)
    # Shape remains, (B, L, G, C)
    conv_out = self.short_conv(v)

    # residual for conv component
    out = v + conv_out

    # Note: The 'hidden_states' (residual x) is usually added by the caller/block wrapper
    return out
