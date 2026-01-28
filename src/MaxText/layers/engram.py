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
from dataclasses import dataclass, field
import math
from typing import List, Callable

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
  
Implementation: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
"""


class CompressedTokenizer:
  """
  Implements a lossy, canonicalizing wrapper for a standard tokenizer to optimize
  n-gram pattern matching.

  By applying aggressive normalization (lowercasing, accent stripping, and whitespace
  collapsing), this class maps multiple distinct tokens (e.g., 'Apple', ' apple',
  'APPLE') to a single unified ID. This many-to-one mapping effectively reduces
  the sparsity of the combinatorial n-gram space, increasing the hit rate for
  the Scalable Lookup/Engram memory mechanism.

  Attributes:
      tokenizer: The base Hugging Face tokenizer.
      normalizer: A tokenizers.Normalizer sequence defining the canonicalization rules.
      lookup_table: A NumPy array where `lookup_table[original_id] = compressed_id`.
      num_new_token: The size of the resulting collapsed vocabulary.
  """

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

    # Private use Unicode character used to protect single spaces during stripping
    SENTINEL = "\uE000"
    # Text normalization pipeline: ensures "Café" and "cafe" produce the same ID
    self.normalizer = normalizers.Sequence(
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

    self.lookup_table, self.num_new_token = self._build_lookup_table()

  def __len__(self):
    return self.num_new_token

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
    Replaces original token IDs with canonical IDs using the pre-computed table.
    """
    arr = np.asarray(input_ids, dtype=np.int64)
    # Ignore negative IDs (often used for padding/masks)
    pos_mask = arr >= 0
    out = arr.copy()
    valid_ids = arr[pos_mask]

    # Vectorized replacement: O(1) lookup per token
    out[pos_mask] = self.lookup_table[valid_ids]
    return out

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
      tokenizer,  # pass the global tokenizer to avoid re-loading
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
    while True:
      if isprime(candidate) and candidate not in seen_primes:
        return candidate
      candidate += 1

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
        num_head = self.n_head_per_ngram
        current_prime_search_start = vocab_size - 1

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

  def __init__(self, list_of_N: List[int], D: int, cfg, mesh, rngs: nnx.Rngs):
    """
    Args:
      list_of_N: A list of prime-based vocab sizes,
          each element is k-th head for n-gram
          for example, 2-gram and 3-gram, each with 2 heads, the flattened list is
          e.g., m[n=2,k=0]=2, m[n=2,k=1]=3, m[n=3,k=0]=5, m[n=3,k=1]=7
      D: The embedding dimension for a single head.
    """
    self.num_heads = len(list_of_N)
    """       
    
    """

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
    self.embedding = Embed(num_embeddings=sum(list_of_N), num_features=D, config=cfg, mesh=mesh, rngs=rngs)

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

  def __init__(
      self,
      cfg,
      hidden_size: int,
      kernel_size: int = 4,
      dilation: int = 1,
      hc_mult: int = 4,
      activation: bool = True,
      rngs: nnx.Rngs = None,
  ):
    self.hc_mult = hc_mult
    self.activation = activation
    total_channels = hidden_size * hc_mult
    self.rngs = rngs

    # In NNX, we define the dimension layout.
    # MaxText typically expects (Batch, Length, Channels)
    self.conv = nnx.Conv(
        in_features=total_channels,
        out_features=total_channels,
        kernel_size=(kernel_size,),
        feature_group_count=total_channels,
        kernel_dilation=(dilation,),
        padding="CAUSAL",  # To match the slice [..., :T] logic
        use_bias=False,
        rngs=rngs,
    )

    # Vectorized RMSNorms for the groups
    # TODO(shuningjin): eps
    self.norms = nnx.List(
        [
            RMSNorm(
                num_features=hidden_size,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                kernel_axes=("norm",),
                # epsilon=cfg.normalization_layer_epsilon,
                epsilon=1e-5,  # Match PyTorch default
                rngs=self.rngs,
            )
            for _ in range(hc_mult)
        ]
    )

    self.act_fn = jax.nn.silu if activation else lambda x: x

  def __call__(self, x: jax.Array) -> jax.Array:
    # Input: (B, L, G, C)
    B, L, G, C = x.shape

    # Apply group-wise norms
    normed_chunks = []
    for i in range(G):
      normed_chunks.append(self.norms[i](x[:, :, i, :]))

    # Concatenate and apply Depthwise Conv
    x_flat = jnp.concatenate(normed_chunks, axis=-1)  # (B, L, G*C)
    y = self.conv(x_flat)
    y = self.act_fn(y)

    return y.reshape(B, L, G, C)


class Engram(nnx.Module):

  def __init__(
      self,
      layer_id: int,
      rngs: nnx.Rngs,
      config,
      mesh,
      tokenizer,  # pass the tokenizer to avoid re-loading
      quant: Optional[Quant] = None,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      *,
      hc_mult,
      engram_heads_per_ngram,
      engram_embed_dim_per_ngram,
      engram_max_ngram_size,
      engram_kernel_size,
      engram_vocab_size,
      layer_ids,
      pad_id,
      seed,
  ):
    self.config = config
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.dtype
    self.kernel_init = kernel_init
    self.quant = quant
    self.rngs = rngs
    self.layer_id = layer_id

    self.engram_heads_per_ngram = engram_heads_per_ngram
    self.engram_embed_dim_per_ngram = engram_embed_dim_per_ngram
    self.engram_max_ngram_size = engram_max_ngram_size
    self.engram_kernel_size = engram_kernel_size
    self.engram_vocab_size = engram_vocab_size
    self.layer_ids = layer_ids
    self.pad_id = pad_id
    self.seed = seed

    # -----------------------------------------------------------------------
    # Vocabulary Size Calculation (Global Prime Sequence)
    # -----------------------------------------------------------------------
    # Engram uses unique prime numbers for the vocabulary size of every head
    # in every layer to maximize hash collision independence.

    # We instantiate NgramHashMapping here to replicate this deterministic
    # sequence. Ideally, this mapping should be created once globally and
    # the resulting vocab_sizes passed into this layer to improve startup time.

    self.hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_vocab_size,
        max_ngram_size=engram_max_ngram_size,
        n_embed_per_ngram=engram_embed_dim_per_ngram,
        n_head_per_ngram=engram_heads_per_ngram,
        # IMPORTANT: We must pass the FULL list of layer_ids, not just self.layer_id.
        # The mapping finds primes sequentially across all layers; passing a partial
        # list would reset the prime search and break alignment with the reference model.
        layer_ids=layer_ids,
        # Inject the pre-loaded tokenizer to avoid redundant disk I/O per layer.
        tokenizer=tokenizer,
        pad_id=pad_id,
        seed=seed,
    )
    # Extract the specific list of primes for THIS layer
    # The structure is [[ngram2_head1, ngram2_head2...], [ngram3_head1...]]
    # We flatten it into a single list of ints: [N1, N2, N3, ...]
    vocab_sizes = self.hash_mapping.get_vocab_sizes(self.layer_id)

    # DEBUG PRINT (Uncomment if tests fail)
    # print(f"DEBUG JAX Layer {self.layer_id} Vocab Sizes: {vocab_sizes}")
    print(f"DEBUG JAX Layer, Vocab Sizes: {vocab_sizes}")

    # -----------------------------------------------------------------------

    # init module
    self.mhe = MultiHeadEmbedding(
        vocab_sizes, self.engram_embed_dim_per_ngram // self.engram_heads_per_ngram, config, mesh, rngs
    )
    self.short_conv = ShortConv(
        config, config.base_emb_dim, self.engram_kernel_size, self.engram_max_ngram_size, hc_mult, rngs=rngs
    )

    engram_hidden_size = engram_embed_dim_per_ngram * (self.engram_max_ngram_size - 1)

    self.value_proj = DenseGeneral(
        in_features_shape=engram_hidden_size,
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

    self.key_projs = nnx.List(
        [
            DenseGeneral(
                in_features_shape=engram_hidden_size,
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
            for _ in range(hc_mult)
        ]
    )

    self.norm1 = nnx.List(
        [
            RMSNorm(
                num_features=config.base_emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                # epsilon=config.normalization_layer_epsilon,
                epsilon=1e-6,  # Match PyTorch default
                rngs=self.rngs,
            )
            for _ in range(hc_mult)
        ]
    )
    self.norm2 = nnx.List(
        [
            RMSNorm(
                num_features=config.base_emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                # epsilon=config.normalization_layer_epsilon,
                epsilon=1e-6,  # Match PyTorch default
                rngs=self.rngs,
            )
            for _ in range(hc_mult)
        ]
    )

  def __call__(self, hidden_states: jax.Array, input_ids: jax.Array) -> jax.Array:
    # hash_input_ids is calculated via NgramHashMapping

    hash_input_ids = jnp.array(self.hash_mapping.hash(input_ids)[self.layer_id])
    print("jax1", hash_input_ids)

    embeddings = self.mhe(hash_input_ids)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)  # Flatten heads
    print("jax2", embeddings)

    gates = []
    for i in range(len(self.key_projs)):
      k = self.norm1[i](self.key_projs[i](embeddings))
      print("jax3", k)
      q = self.norm2[i](hidden_states[:, :, i, :])
      print("jax4", q)
      # Scaled Dot Product Gating
      gate = jnp.sum(k * q, axis=-1) / jnp.sqrt(k.shape[-1])
      print("jax5", gate)
      gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
      print("jax6", gate)
      gate = jax.nn.sigmoid(gate)[..., None]
      print("jax7", gate)
      gates.append(gate)

    gates_stack = jnp.stack(gates, axis=2)
    v = gates_stack * self.value_proj(embeddings)[:, :, None, :]

    output = v + self.short_conv(v)
    return output
