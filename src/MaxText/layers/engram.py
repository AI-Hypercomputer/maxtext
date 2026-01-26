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

from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from typing import List, Callable


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
  create a canonical version of the text
  """

  def __init__(
      self,
      tokenizer,  # hf tokenizer
      # tokenizer_name_or_path,
  ):
    # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    self.tokenizer = tokenizer

    SENTINEL = "\uE000"
    self.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ]
    )

    self.lookup_table, self.num_new_token = self._build_lookup_table()

  def __len__(self):
    return self.num_new_token

  def _build_lookup_table(self):
    old2new = {}
    key2new = {}
    new_tokens = []

    vocab_size = len(self.tokenizer)
    for tid in range(vocab_size):
      text = self.tokenizer.decode([tid], skip_special_tokens=False)

      if "�" in text:
        key = self.tokenizer.convert_ids_to_tokens(tid)
      else:
        norm = self.normalizer.normalize_str(text)
        key = norm if norm else text

      nid = key2new.get(key)
      if nid is None:
        nid = len(new_tokens)
        key2new[key] = nid
        new_tokens.append(key)
      old2new[tid] = nid

    lookup = np.empty(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
      lookup[tid] = old2new[tid]

    return lookup, len(new_tokens)

  def _compress(self, input_ids):
    arr = np.asarray(input_ids, dtype=np.int64)
    pos_mask = arr >= 0
    out = arr.copy()
    valid_ids = arr[pos_mask]
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
      mix = tokens[0] * multipliers[0]
      for k in range(1, n):
        mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
      num_heads_for_this_ngram = self.n_head_per_ngram
      head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

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


class MultiHeadEmbedding(nnx.Module):

  def __init__(self, list_of_N: List[int], D: int, cfg, mesh, rngs: nnx.Rngs):
    self.num_heads = len(list_of_N)

    # Static offsets for hashing heads
    offsets = np.cumsum([0] + list_of_N[:-1])
    # self.offsets = jnp.array(offsets, dtype=jnp.int32)
    self.offsets = nnx.Variable(jnp.array(offsets, dtype=jnp.int32))

    # Reuse MaxText's Embed for the actual heavy lifting
    self.embedding = Embed(num_embeddings=sum(list_of_N), num_features=D, config=cfg, mesh=mesh, rngs=rngs)

  def __call__(self, input_ids: jax.Array, model_mode: str = MODEL_MODE_TRAIN) -> jax.Array:
    # input_ids: (Batch, Length, Num_Heads)
    # Apply offsets to make indices unique across the concatenated table
    shifted_ids = input_ids + self.offsets.value

    # Let MaxText handle the sharding and retrieval
    return self.embedding(shifted_ids, model_mode=model_mode)


class Engram(nnx.Module):

  def __init__(
      self,
      layer_id: int,
      config,
      backbone_cfg,
      rngs: nnx.Rngs,
      cfg,
      mesh,
      tokenizer,
      quant: Optional[Quant] = None,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
  ):
    self.cfg = cfg
    self.mesh = mesh
    self.dtype = self.cfg.dtype
    self.weight_dtype = self.cfg.dtype
    self.kernel_init = kernel_init
    self.quant = quant
    self.rngs = rngs
    self.layer_id = layer_id

    # Note: NgramHashMapping remains a CPU/NumPy helper for pre-processing
    # or it can be ported to JAX if needed inside the graph.

    # Placeholder for the same vocab logic from your PyTorch code
    # vocab_sizes = [129280 * 5] * (config.max_ngram_size - 1) * config.n_head_per_ngram

    # -----------------------------------------------------------------------
    # CRITICAL FIX: Replicate NgramHashMapping Logic
    # -----------------------------------------------------------------------
    # We must instantiate NgramHashMapping to get the EXACT prime numbers
    # used by the PyTorch model for this specific layer.

    # Note: In a real training run, you might want to calculate this once
    # globally and pass it in, rather than recalculating per layer.
    # For now, this ensures correctness.

    # 1. We need a tokenizer to initialize the mapper
    # (In production, pass the tokenizer or the pre-calced list to avoid re-loading)
    # tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, trust_remote_code=True)

    hash_mapping = NgramHashMapping(
        engram_vocab_size=config.engram_vocab_size,
        max_ngram_size=config.max_ngram_size,
        n_embed_per_ngram=config.n_embed_per_ngram,
        n_head_per_ngram=config.n_head_per_ngram,
        layer_ids=config.layer_ids,  # Pass all layer IDs so the sequence of primes is correct
        # tokenizer_name_or_path=config.tokenizer_name_or_path, # Adjusted to match your NgramHashMapping signature
        tokenizer=tokenizer,
        pad_id=config.pad_id,
        seed=config.seed,
    )
    self.hash_mapping = hash_mapping

    # 2. Extract the specific list of primes for THIS layer
    # The structure is [[ngram2_head1, ngram2_head2...], [ngram3_head1...]]
    # We flatten it into a single list of ints: [N1, N2, N3, ...]
    vocab_sizes = [x for y in hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]

    # DEBUG PRINT (Uncomment if tests fail)
    # print(f"DEBUG JAX Layer {self.layer_id} Vocab Sizes: {vocab_sizes}")
    print(f"DEBUG JAX Layer, Vocab Sizes: {vocab_sizes}")

    # -----------------------------------------------------------------------

    # init module
    self.mhe = MultiHeadEmbedding(vocab_sizes, config.n_embed_per_ngram // config.n_head_per_ngram, cfg, mesh, rngs)
    self.short_conv = ShortConv(
        cfg, backbone_cfg.hidden_size, config.kernel_size, config.max_ngram_size, backbone_cfg.hc_mult, rngs=rngs
    )

    engram_hidden_size = config.n_embed_per_ngram * (config.max_ngram_size - 1)

    # self.value_proj = nnx.Linear(engram_hidden_size, backbone_cfg.hidden_size, use_bias=False, rngs=rngs)

    self.value_proj = DenseGeneral(
        in_features_shape=engram_hidden_size,
        out_features_shape=backbone_cfg.hidden_size,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("a", "b"),  # TODO(shuningjin)
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.cfg.matmul_precision,
        shard_mode=self.cfg.shard_mode,
        rngs=self.rngs,
        use_bias=True,
    )

    self.key_projs = nnx.List(
        [
            DenseGeneral(
                in_features_shape=engram_hidden_size,
                out_features_shape=backbone_cfg.hidden_size,
                axis=-1,
                kernel_init=self.kernel_init,
                kernel_axes=("a", "b"),  # TODO(shuningjin)
                dtype=self.dtype,
                weight_dtype=self.weight_dtype,
                quant=self.quant,
                matmul_precision=self.cfg.matmul_precision,
                shard_mode=self.cfg.shard_mode,
                rngs=self.rngs,
                use_bias=True,
            )
            for _ in range(backbone_cfg.hc_mult)
        ]
    )

    # torch.finfo(torch.float32).eps
    self.norm1 = nnx.List(
        [
            RMSNorm(
                num_features=backbone_cfg.hidden_size,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                kernel_axes=("norm",),
                # epsilon=cfg.normalization_layer_epsilon,
                epsilon=1e-6,  # Match PyTorch default
                rngs=self.rngs,
            )
            for _ in range(backbone_cfg.hc_mult)
        ]
    )
    self.norm2 = nnx.List(
        [
            RMSNorm(
                num_features=backbone_cfg.hidden_size,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                kernel_axes=("norm",),
                # epsilon=cfg.normalization_layer_epsilon,
                epsilon=1e-6,  # Match PyTorch default
                rngs=self.rngs,
            )
            for _ in range(backbone_cfg.hc_mult)
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
