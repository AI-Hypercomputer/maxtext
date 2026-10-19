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

import math
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P

from maxtext.common.common_types import Array, Config
from maxtext.layers.initializers import nd_dense_init, NdInitializer
from maxtext.layers.linears import DenseGeneral
from maxtext.utils import sharding as sharding_utils


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
  """Host-side tables + on-device hashing for one engram layer (mirrors upstream `NgramHashState`, text only)."""

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

  def __call__(self, input_ids: Array) -> Array:
    """[B, S] token ids (positions contiguous from 0) -> [B, S, (max_ngram_size - 1) * n_heads] row ids."""
    compressed = jnp.asarray(self.token_map)[input_ids]
    b, s = compressed.shape
    prod_hi, prod_lo = jnp.asarray(self.prod_hi), jnp.asarray(self.prod_lo)
    rolling_hi = rolling_lo = None
    hashes = []
    for shift in range(self.max_ngram_size):
      if shift == 0:
        tokens = compressed
      else:
        pad = jnp.full((b, min(shift, s)), self.pad_id, dtype=compressed.dtype)
        tokens = jnp.concatenate([pad, compressed[:, : s - shift]], axis=1)[:, :s]
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


def sharded_row_lookup(
    table: Array,
    ids: Array,
    mesh: Optional[Mesh],
    rules: Any,
    out_dtype: Any,
    method: str = "all_to_all",
    capacity_factor: float = 2.0,
    seq_chunk: int = 512,
    num_rows: Optional[int] = None,
) -> Array:
  """`jnp.take(table, ids, axis=0)` for a table whose rows are sharded over many chips (training-time Engram).

  This is the training layout of the Engram paper (arXiv 2601.07372, Sec. 2.5): the tables are sharded over the
  accelerators and "an All-to-All communication primitive is used to gather active rows in the forward pass and
  dispatch gradients in the backward pass". (Host-memory offload with prefetching is the paper's *inference*
  layout: the tables are frozen there, while training needs gradients and optimizer state for them every step.)

  methods (inside a `shard_map` over the mesh):
    * "all_to_all": each chip de-duplicates its ids, sends each owner chip the (row-local) ids it holds in a
      [group, capacity] buffer, owners gather the rows, and a second all-to-all returns them. Traffic per chip is
      ~ its own tokens' rows (tens of MB per lookup). The capacity is `capacity_factor` x the uniform share over the
      chips that hold real (non-padding) rows, capped at the number of ids (where it is exact by construction).
      Requests beyond it would read zeros; hashed rows are spread uniformly and de-duplication removes repeats, so
      on the real tables (e.g. mean ~384 distinct ids per owner on 256 chips, capacity 768) this does not happen.
    * "psum_scatter": every chip gathers the ids of its whole row group, looks up the rows it owns (zeros elsewhere)
      and a psum_scatter returns each chip its rows: exact for any ids, but moves [group batch, seq, cols, head_dim]
      per chip (~12 GB per lookup per chip on 256 chips), processed in `seq_chunk` pieces.
    * "gather": plain `jnp.take`, partitioned by XLA.
  Exactly one chip contributes each row, so the bf16 exchange is exact. The backward pass is the transpose of the
  same collectives plus a local scatter-add into each chip's slice of the table.

  Args:
    table: [rows, head_dim], rows sharded over the mesh axes of the logical axis "embed".
    ids: [batch, seq, cols] int32 row ids, batch sharded like activations ("activation_batch").
    mesh: device mesh (None or a mesh where the rows are not sharded: plain local gather).
    rules: logical axis rules used to resolve the shardings.
    out_dtype: dtype of the returned rows (the exchange runs in it).
    method: "all_to_all" | "psum_scatter" | "gather".
    capacity_factor: all_to_all buffer size per (chip, owner) pair, as a multiple of the uniform share.
    seq_chunk: sequence positions per psum_scatter step.
    num_rows: number of real rows (the rest of `table` is padding no id reaches); defaults to all rows.

  Returns:
    [batch, seq, cols, head_dim] rows.
  """
  row_axes = ()
  if mesh is not None and method != "gather":
    row_spec = sharding_utils.logical_to_mesh_axes(("embed", None), mesh, rules=rules)
    row_axes = sharding_utils.mesh_axes_for_dim(row_spec[0]) if row_spec is not None and len(row_spec) else ()
  if not row_axes:
    return jnp.take(table, ids, axis=0).astype(out_dtype)
  if method not in ("all_to_all", "psum_scatter"):
    raise ValueError(f"unknown engram lookup method {method!r}")

  batch_spec = sharding_utils.logical_to_mesh_axes(("activation_batch", None, None), mesh, rules=rules)
  batch_axes = batch_spec[0] if batch_spec is not None and len(batch_spec) else None
  batch_shards = int(np.prod([mesh.shape[a] for a in sharding_utils.mesh_axes_for_dim(batch_axes)]))
  if ids.shape[0] % batch_shards:
    # A batch the mesh cannot split (e.g. a single prompt): with replicated ids every chip of the row group looks up
    # the same tokens, which is still exact.
    batch_axes = None
  table_pspec = P(row_axes if len(row_axes) > 1 else row_axes[0], None)
  ids_pspec = P(batch_axes, None, None)
  out_pspec = P(batch_axes, None, None, None)
  axes = row_axes if len(row_axes) > 1 else row_axes[0]
  group = int(np.prod([mesh.shape[a] for a in row_axes]))
  # Chips whose table slice holds real rows: the ids are spread over those only.
  real_rows = table.shape[0] if num_rows is None else num_rows
  owners = max(1, min(group, -(-real_rows // max(1, table.shape[0] // group))))

  def local_all_to_all(table_shard, local_ids):
    shard_rows, head_dim = table_shard.shape
    flat = local_ids.reshape(-1)
    t = flat.shape[0]
    cap = min(t, max(1, int(math.ceil(capacity_factor * t / owners))))
    # De-duplicate: sorted ids, one slot per distinct id (repeated n-grams share a row and its gradient).
    order = jnp.argsort(flat)
    sorted_ids = flat[order]
    first = jnp.concatenate([jnp.ones((1,), jnp.bool_), sorted_ids[1:] != sorted_ids[:-1]])
    slot = jnp.cumsum(first.astype(jnp.int32)) - 1  # sorted position -> distinct-id slot
    uniq = jnp.zeros((t,), flat.dtype).at[slot].set(sorted_ids)  # ascending distinct ids, then filler
    is_uniq = jnp.arange(t) < slot[-1] + 1
    owner = jnp.where(is_uniq, uniq // shard_rows, group)  # `group` = no owner (filler)
    counts = jnp.zeros((group + 1,), jnp.int32).at[owner].add(1)
    rank = jnp.arange(t) - (jnp.cumsum(counts) - counts)[owner]  # position among the ids sent to that owner
    fits = is_uniq & (rank < cap)
    dst, pos = jnp.where(fits, owner, group), jnp.where(fits, rank, 0)
    send = (
        jnp.full((group, cap), -1, jnp.int32).at[dst, pos].set((uniq - owner * shard_rows).astype(jnp.int32), mode="drop")
    )
    requested = jax.lax.all_to_all(send, axes, 0, 0, tiled=True)  # [group, cap] local rows asked for by each chip
    rows = jnp.take(table_shard, jnp.maximum(requested, 0), axis=0)
    rows = jnp.where((requested >= 0)[..., None], rows, 0).astype(out_dtype)
    returned = jax.lax.all_to_all(rows, axes, 0, 0, tiled=True)  # [group, cap, head_dim] rows of my requests
    vals = jnp.where(fits[:, None], returned[jnp.where(fits, owner, 0), pos], 0)  # per distinct id
    slot_of = jnp.zeros((t,), jnp.int32).at[order].set(slot)  # original position -> distinct-id slot
    return vals[slot_of].reshape(local_ids.shape + (head_dim,))

  def local_psum_scatter(table_shard, local_ids):
    shard_rows = table_shard.shape[0]
    start = jax.lax.axis_index(axes) * shard_rows
    seq = local_ids.shape[1]
    pieces = []
    for lo in range(0, seq, seq_chunk):
      chunk = local_ids[:, lo : lo + seq_chunk]
      group_ids = jax.lax.all_gather(chunk, axes, axis=0, tiled=True) - start  # [batch of the row group, c, cols]
      owned = (group_ids >= 0) & (group_ids < shard_rows)
      rows = jnp.take(table_shard, jnp.where(owned, group_ids, 0), axis=0)
      rows = jnp.where(owned[..., None], rows, 0).astype(out_dtype)
      pieces.append(jax.lax.psum_scatter(rows, axes, scatter_dimension=0, tiled=True))
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=1)

  local = local_all_to_all if method == "all_to_all" else local_psum_scatter
  return jax.shard_map(local, mesh=mesh, in_specs=(table_pspec, ids_pspec), out_specs=out_pspec, check_vma=False)(
      table, ids
  )


class DeepSeek41Engram(nnx.Module):
  """Engram layer: gated n-gram memory added into each hyper-connection copy."""

  def __init__(
      self,
      config: Config,
      num_embeddings: int,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.num_rows = num_embeddings
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
        # Rows over every chip (as in the Engram paper's training setup): lookups exchange only the active rows (see
        # `sharded_row_lookup`), and loading never moves the ~100 GB tables across chips.
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

  def __call__(self, h: Array, hash_ids: Array) -> Array:
    """h: [B, S, hc, D] residual stream; hash_ids: [B, S, n_hash_cols] -> [B, S, hc, D]."""
    b, s = hash_ids.shape[:2]
    rows = sharded_row_lookup(
        self.embed[...],
        hash_ids,
        self.mesh,
        self.config.logical_axis_rules,
        self.config.dtype,
        method=self.config.ds41_engram_lookup,
        capacity_factor=self.config.ds41_engram_a2a_capacity_factor,
        num_rows=self.num_rows,
    )  # [B, S, cols, hd]
    kv = self.wkv(rows.reshape(b, s, self.n_hash_cols * self.head_dim))
    key = kv[..., : self.hc_mult * self.dim].astype(jnp.float32).reshape(b, s, self.hc_mult, self.dim)
    value = kv[..., self.hc_mult * self.dim :].astype(jnp.float32)
    weight = self.q_weight[...].astype(jnp.float32) * self.k_weight[...].astype(jnp.float32)
    hf = h.astype(jnp.float32)
    # Normalized per (token, hc copy) over `dim`, not jointly over the copies.
    rstd = jax.lax.rsqrt(jnp.mean(hf**2, axis=-1) + self.eps) * jax.lax.rsqrt(jnp.mean(key**2, axis=-1) + self.eps)
    dot = jnp.sum(hf * weight * key, axis=-1) * rstd * self.dim**-0.5
    # Signed square root before the sigmoid.
    root = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -root, root))
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
