#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""DeepSeek-V4.1 engram: n-gram hash lookups written into the hyper-connection residual stream.

Each position is hashed together with the `max_ngram_size - 1` tokens before it (over a compressed token id space
in which tokens that normalize alike collapse): the multiplied ids are XOR-ed one lookback at a time, so the
running value after lookback i hashes the (i+1)-gram, and each (n-gram size, head) pair maps it into its own
prime-sized bucket range of one shared table. The fetched rows go through `wkv` into one key per hc copy plus a
shared value; the value is added to each copy, gated by a normalized dot product of that copy against its key.

The hashes use int64 arithmetic upstream. To stay exact on TPU without enabling x64, the per-(layer, lookback)
products `token * multiplier` are precomputed on the host for the whole compressed vocabulary as (hi, lo) uint32
pairs; XOR works limb-wise and the final modulo is computed limb-wise with 24-bit-safe modular arithmetic.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py and `Engram` in
inference/model.py.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy
from flax import nnx

from maxtext.common.common_types import Array, Config
from maxtext.layers import deepseek41_quant
from maxtext.layers.initializers import nd_dense_init, NdInitializer
from maxtext.layers.linears import DenseGeneral


def build_compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
  """Maps every token id onto a smaller id space where tokens that normalize alike collapse together.

  Returns the lookup ([vocab] int32) and the compressed vocab size (every hash multiplier derives from it).
  """
  from tokenizers import Regex, normalizers  # pylint: disable=import-outside-toplevel

  # A private-use char, so a token that is exactly one space survives Strip() instead of collapsing to "".
  sentinel = "\ue000"
  normalizer = normalizers.Sequence(
      [
          normalizers.NFKC(),
          normalizers.NFD(),
          normalizers.StripAccents(),
          normalizers.Lowercase(),
          normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
          normalizers.Replace(Regex(r"^ $"), sentinel),
          normalizers.Strip(),
          normalizers.Replace(sentinel, " "),
      ]
  )
  backend = tokenizer.backend_tokenizer
  key_to_new: dict[str, int] = {}
  lookup = np.zeros(len(tokenizer), dtype=np.int32)
  for token_id in range(len(tokenizer)):
    text = backend.decode([token_id], skip_special_tokens=False)
    if "\ufffd" in text:
      # A partial UTF-8 byte token: nothing to normalize, so key it by its raw form.
      key = backend.id_to_token(token_id)
    else:
      normalized = normalizer.normalize_str(text)
      key = normalized if normalized else text
    new_id = key_to_new.get(key)
    if new_id is None:
      new_id = len(key_to_new)
      key_to_new[key] = new_id
    lookup[token_id] = new_id
  return lookup, len(key_to_new)


def engram_primes(layer_ids: Sequence[int], max_ngram_size: int, n_heads: int, vocab_size: int) -> list:
  """[layer][n-gram size - 2][head] prime bucket sizes: drawn in order above `vocab_size - 1`, never reused."""
  primes, seen = [], set()
  for _ in layer_ids:
    per_ngram = []
    for _ in range(max_ngram_size - 1):
      sizes, current = [], vocab_size - 1
      for _ in range(n_heads):
        candidate = current + 1
        while not sympy.isprime(candidate) or candidate in seen:
          candidate += 1
        current = candidate
        seen.add(current)
        sizes.append(current)
      per_ngram.append(sizes)
    primes.append(per_ngram)
  return primes


def compute_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int) -> np.ndarray:
  """One odd multiplier per (layer, lookback), bounded so `token_id * multiplier` fits in int64. [L, N] int64."""
  max_long = np.iinfo(np.int64).max
  multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
  rows = []
  for layer_id in layer_ids:
    generator = np.random.default_rng(10007 * layer_id)
    values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
    rows.append(values * 2 + 1)
  return np.stack(rows)


def _mulmod_u32(a: Array, b: Array, p: Array) -> Array:
  """(a * b) % p for uint32 a, b < p < 2**24 without overflowing uint32 (b split into 8-bit chunks)."""
  b2, b1, b0 = b >> 16, (b >> 8) & 0xFF, b & 0xFF
  acc = (a * b2) % p
  acc = ((acc << 8) % p + (a * b1) % p) % p
  acc = ((acc << 8) % p + (a * b0) % p) % p
  return acc


class DeepSeek41NgramHasher:
  """Host-side tables + on-device n-gram hashing (mirrors `NgramHashState`, with `image_mask` `DEAD` barriers)."""

  def __init__(
      self,
      token_map: np.ndarray,
      compressed_vocab_size: int,
      pad_id: int,
      layer_ids: Sequence[int],
      layer_id: int,
      max_ngram_size: int,
      n_heads: int,
      vocab_size: int,
  ):
    layer_index = list(layer_ids).index(layer_id)
    primes = engram_primes(layer_ids, max_ngram_size, n_heads, vocab_size)[layer_index]
    flat_primes = np.array([p for per_ngram in primes for p in per_ngram], dtype=np.int64)
    if flat_primes.max() >= 2**24:
      raise ValueError(f"engram bucket sizes must be < 2**24 for the uint32 hash, got {flat_primes.max()}")
    multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size)[layer_index]
    ids = np.arange(compressed_vocab_size, dtype=np.uint64)
    products = ids[None, :] * multipliers.astype(np.uint64)[:, None]  # [N, V], each < 2**63
    self.max_ngram_size = max_ngram_size
    self.n_heads = n_heads
    self.num_embeddings = int(flat_primes.sum())
    self.token_map = token_map.astype(np.int32)
    self.pad_id = int(token_map[pad_id])
    self.prod_hi = (products >> np.uint64(32)).astype(np.uint32)
    self.prod_lo = (products & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    self.primes = flat_primes.astype(np.uint32)  # [(N - 1) * H]
    self.pow32_mod_p = (np.int64(2**32) % flat_primes).astype(np.uint32)
    self.offsets = np.concatenate([[0], np.cumsum(flat_primes)[:-1]]).astype(np.int32)

  def _gather_ngram_tokens(
      self,
      input_ids: Array,
      decoder_segment_ids: Array | None = None,
      image_mask: Array | None = None,
  ) -> list[Array]:
    """Returns per-shift compressed token IDs `[B, S]` with segment and image DEAD barriers applied."""
    compressed = jnp.asarray(self.token_map)[input_ids]
    b, s = compressed.shape
    if decoder_segment_ids is not None:
      compressed = jnp.where(decoder_segment_ids != 0, compressed, self.pad_id)
    blocked = None
    img_mask = None
    if image_mask is not None:
      if image_mask.shape != (b, s):
        raise ValueError(f"DeepSeek41NgramHasher image_mask shape {image_mask.shape} != expected {(b, s)}.")
      img_mask = jnp.asarray(image_mask, dtype=jnp.bool_)
      blocked = img_mask
    per_shift_tokens: list[Array] = []
    for shift in range(self.max_ngram_size):
      if shift == 0:
        tokens = compressed
      else:
        pad = jnp.full((b, min(shift, s)), self.pad_id, dtype=compressed.dtype)
        tokens = jnp.concatenate([pad, compressed[:, : max(s - shift, 0)]], axis=1)[:, :s]
        if decoder_segment_ids is not None:
          seg_pad = jnp.zeros((b, min(shift, s)), dtype=decoder_segment_ids.dtype)
          prev_seg = jnp.concatenate([seg_pad, decoder_segment_ids[:, : max(s - shift, 0)]], axis=1)[:, :s]
          tokens = jnp.where((prev_seg == decoder_segment_ids) & (decoder_segment_ids != 0), tokens, self.pad_id)
      if blocked is not None:
        if shift > 0:
          img_pad = jnp.zeros((b, min(shift, s)), dtype=jnp.bool_)
          shifted_img = jnp.concatenate([img_pad, img_mask[:, : max(s - shift, 0)]], axis=1)[:, :s]
          blocked = blocked | shifted_img
        tokens = jnp.where(blocked, self.pad_id, tokens)
      per_shift_tokens.append(tokens)
    return per_shift_tokens

  def __call__(
      self,
      input_ids: Array,
      decoder_segment_ids: Array | None = None,
      image_mask: Array | None = None,
  ) -> Array:
    """[B, S] token ids (positions contiguous from 0) -> [B, S, (max_ngram_size - 1) * n_heads] row ids."""
    per_shift_tokens = self._gather_ngram_tokens(
        input_ids, decoder_segment_ids=decoder_segment_ids, image_mask=image_mask
    )
    prod_hi, prod_lo = jnp.asarray(self.prod_hi), jnp.asarray(self.prod_lo)
    rolling_hi = rolling_lo = None
    hashes = []
    for shift, tokens in enumerate(per_shift_tokens):
      hi, lo = prod_hi[shift][tokens], prod_lo[shift][tokens]
      if shift == 0:
        rolling_hi, rolling_lo = hi, lo
        continue
      rolling_hi = jnp.bitwise_xor(rolling_hi, hi)
      rolling_lo = jnp.bitwise_xor(rolling_lo, lo)
      cols = slice((shift - 1) * self.n_heads, shift * self.n_heads)
      p = jnp.asarray(self.primes[cols])
      r_hi = rolling_hi[..., None] % p
      r_lo = rolling_lo[..., None] % p
      r = (_mulmod_u32(r_hi, jnp.asarray(self.pow32_mod_p[cols]), p) + r_lo) % p
      hashes.append(r)
    return jnp.concatenate(hashes, axis=-1).astype(jnp.int32) + jnp.asarray(self.offsets)


# The engram tables are sharded by rows over every chip; their (prime-sum) row counts are padded to a multiple of
# this so any power-of-two chip count up to it divides them. Hash ids never reach the padding rows.
ENGRAM_ROW_PADDING = 1024


def padded_engram_rows(num_embeddings: int) -> int:
  return -(-num_embeddings // ENGRAM_ROW_PADDING) * ENGRAM_ROW_PADDING


class DeepSeek41Engram(nnx.Module):
  """Engram layer: gated n-gram memory added into each hyper-connection copy."""

  def __init__(
      self,
      config: Config,
      num_embeddings: int,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.dim = config.emb_dim
    self.hc_mult = config.mhc_expansion_rate
    self.head_dim = config.ds41_engram_head_dim
    self.n_hash_cols = (config.ds41_engram_max_ngram_size - 1) * config.ds41_engram_n_heads
    self.eps = config.normalization_layer_epsilon
    # Zero-initialized: the full model's tables have ~384M rows, and generating random values for them is not
    # partitionable with every PRNG implementation (e.g. `unsafe_rbg`), which would materialize a whole table on
    # each chip. Zero rows just make the engram contribute nothing until trained / loaded.
    self.embed = nnx.Param(
        jnp.zeros((padded_engram_rows(num_embeddings), self.head_dim), dtype=config.weight_dtype),
        # Rows over every chip (as upstream's ParallelEngramEmbedding): the lookup is a masked local gather plus a
        # reduction, and loading never moves the ~100 GB tables across chips.
        out_sharding=("embed", None),
    )
    self.wkv = DenseGeneral(
        in_features_shape=self.n_hash_cols * self.head_dim,
        out_features_shape=self.dim * (self.hc_mult + 1),
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("engram_dim", "embed"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.q_weight = nnx.Param(jnp.ones((self.hc_mult, self.dim), dtype=config.weight_dtype), out_sharding=(None, "embed"))
    self.k_weight = nnx.Param(jnp.ones((self.hc_mult, self.dim), dtype=config.weight_dtype), out_sharding=(None, "embed"))

  def __call__(
      self,
      h: Array,
      hash_ids: Array,
      decoder_segment_ids: Array | None = None,
      image_mask: Array | None = None,
  ) -> Array:
    """h: [B, S, hc, D] residual stream; hash_ids: [B, S, n_hash_cols] -> [B, S, hc, D]."""
    b, s = hash_ids.shape[:2]
    rows = jnp.take(self.embed[...], hash_ids, axis=0).astype(self.config.dtype)  # [B, S, cols, hd]
    rows = rows.reshape(b, s, self.n_hash_cols * self.head_dim)
    if deepseek41_quant.enabled(self.config):  # the reference's FP8 `wkv` quantizes its input
      rows = deepseek41_quant.fp8_qdq(rows)
    kv = self.wkv(rows)
    work_dtype = jnp.promote_types(h.dtype, jnp.float32)
    key = kv[..., : self.hc_mult * self.dim].astype(work_dtype).reshape(b, s, self.hc_mult, self.dim)
    value = kv[..., self.hc_mult * self.dim :].astype(work_dtype)
    weight = self.q_weight[...].astype(work_dtype) * self.k_weight[...].astype(work_dtype)
    hf = h.astype(work_dtype)
    # Normalized per (token, hc copy) over `dim`, not jointly over the copies.
    rstd = jax.lax.rsqrt(jnp.mean(hf**2, axis=-1) + self.eps) * jax.lax.rsqrt(jnp.mean(key**2, axis=-1) + self.eps)
    dot = jnp.sum(hf * weight * key, axis=-1) * rstd * self.dim**-0.5
    # Signed square root before the sigmoid.
    root = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -root, root))
    if decoder_segment_ids is not None:
      gate = jnp.where((decoder_segment_ids != 0)[..., None], gate, 0.0)
    if image_mask is not None:
      if image_mask.shape != (b, s):
        raise ValueError(f"DeepSeek41Engram image_mask shape {image_mask.shape} != expected {(b, s)}.")
      gate = jnp.where(jnp.asarray(image_mask, dtype=jnp.bool_)[..., None], 0.0, gate)
    return (hf + gate[..., None] * value[:, :, None, :]).astype(h.dtype)


def load_tokenizer_token_map(config: Config) -> tuple[np.ndarray, int]:
  """Loads the HF tokenizer at `config.tokenizer_path` and builds its compressed token map."""
  import transformers  # pylint: disable=import-outside-toplevel

  tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
  token_map, vocab_size = build_compressed_token_map(tokenizer)
  expected = config.ds41_engram_compressed_vocab_size
  if expected and vocab_size != expected:
    raise ValueError(f"compressed tokenizer vocab size {vocab_size} != ds41_engram_compressed_vocab_size {expected}")
  return token_map, vocab_size


def make_hasher(config: Config, layer_id: int, token_map: np.ndarray, compressed_vocab_size: int) -> Any:
  return DeepSeek41NgramHasher(
      token_map=token_map,
      compressed_vocab_size=compressed_vocab_size,
      pad_id=config.ds41_engram_pad_id,
      layer_ids=config.ds41_engram_layers,
      layer_id=layer_id,
      max_ngram_size=config.ds41_engram_max_ngram_size,
      n_heads=config.ds41_engram_n_heads,
      vocab_size=config.ds41_engram_vocab_size,
  )
