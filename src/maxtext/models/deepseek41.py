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

"""DeepSeek-V4.1 decoder.

Architecture (per layer, residual stream carried as `mhc_expansion_rate` hyper-connection copies):

  [engram]  -> hc_pre(pre_mix) -> attn_norm -> attention -> hc_post
            -> hc_pre(attn_pre) -> ffn_norm -> MoE (routed + 1 shared expert) -> hc_post

Each sub-block's `hc_mixes` produces the collapse weights (`pre`) used by the *next* sub-block, so a layer consumes
the `pre_mix` of the previous layer's FFN and hands its own FFN's `pre` to the next layer; the very first layer
uses a one-hot mix and the output head collapses the stream with the last layer's FFN `pre`.

Attention layers share compressed KV / index keys / top-k selections / candidate blocks with later layers (see
`attention_deepseek41`). The decoder state threaded through the layers is therefore a dict:
`{"h": [B, S, hc, D], "pre_mix": [B, S, hc], <shared attention state>}`.

Layers are grouped so every group starts with a sharing source (or is a single sliding-window layer); consecutive
groups with identical structure form a segment, scanned with `jax.lax.scan` when `scan_layers=True`. For
DeepSeek-V4.1-Flash: [0] [1+engram] [2-7, 8-13] [14-19 (+engram)] [20-23] [24-27, 28-31, 32-35, 36-39].
"""

import dataclasses
from typing import Any, Optional

import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.layers import attention_deepseek41
from maxtext.layers import engram_deepseek41
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.engram import StaticWrapper
from maxtext.layers.normalizations import RMSNorm

SHARED_STATE_KEYS = ("compress_kv", "index_k", "topk_mask", "candidates")


@dataclasses.dataclass(frozen=True)
class LayerSpec:
  """Static structure of one DeepSeek-V4.1 layer."""

  layer_id: int
  compress_ratio: int
  is_kv_source: bool
  is_index_source: bool
  is_candidate_source: bool
  uses_candidates: bool
  has_engram: bool

  def signature(self) -> tuple:
    """Structure that must match for two layers to share a scanned stack (engram layers carry per-layer hash
    constants and table sizes, so they never do)."""
    return (
        self.compress_ratio,
        self.is_kv_source,
        self.is_index_source,
        self.is_candidate_source,
        self.uses_candidates,
        self.layer_id if self.has_engram else None,
    )


def build_layer_specs(config: Config) -> list[LayerSpec]:
  specs = []
  for i in range(config.num_decoder_layers):
    ratio = config.compress_ratios[i]
    specs.append(
        LayerSpec(
            layer_id=i,
            compress_ratio=ratio,
            is_kv_source=ratio > 0 and i in config.kv_source_layers,
            is_index_source=ratio > 0 and i in config.index_source_layers,
            is_candidate_source=ratio > 0 and i == config.candidate_source_layer,
            uses_candidates=0 <= config.candidate_source_layer < i,
            has_engram=i in config.ds41_engram_layers,
        )
    )
  return specs


@dataclasses.dataclass(frozen=True)
class Segment:
  """`len(groups)` consecutive, structurally identical groups of layers (scanned together)."""

  groups: tuple[tuple[int, ...], ...]

  @property
  def repeats(self) -> int:
    return len(self.groups)

  @property
  def block_size(self) -> int:
    return len(self.groups[0])


def build_segments(config: Config) -> list[Segment]:
  """Splits the layers into groups (each starting at a KV / index source, or a lone sliding-window layer) and
  merges consecutive structurally identical groups into segments."""
  specs = build_layer_specs(config)
  groups: list[list[int]] = []
  for spec in specs:
    prev = specs[spec.layer_id - 1] if spec.layer_id > 0 else None
    starts_group = (
        prev is None
        or spec.compress_ratio == 0
        or spec.compress_ratio != prev.compress_ratio
        or spec.is_kv_source
        or spec.is_index_source
    )
    if starts_group:
      groups.append([spec.layer_id])
    else:
      groups[-1].append(spec.layer_id)

  def group_signature(group):
    return tuple(specs[i].signature() for i in group)

  segments: list[list[list[int]]] = []
  for group in groups:
    if segments and group_signature(segments[-1][-1]) == group_signature(group):
      segments[-1].append(group)
    else:
      segments.append([group])
  return [Segment(tuple(tuple(g) for g in seg)) for seg in segments]


_TOKEN_MAP_CACHE: dict = {}


def compressed_token_map(config: Config):
  """The engram's compressed token map for `config.tokenizer_path` (built once per process)."""
  key = (config.tokenizer_path, config.ds41_engram_compressed_vocab_size)
  if key not in _TOKEN_MAP_CACHE:
    _TOKEN_MAP_CACHE[key] = engram_deepseek41.load_tokenizer_token_map(config)
  return _TOKEN_MAP_CACHE[key]


def _sinkhorn(t: jax.Array, iters: int, eps: float) -> jax.Array:
  """Row softmax, then alternating column / row normalization (column first), `iters` column passes in total."""
  t = jax.nn.softmax(t, axis=-1) + eps
  t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)
  for _ in range(iters - 1):
    t = t / (jnp.sum(t, axis=-1, keepdims=True) + eps)
    t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)
  return t


class DeepSeek41DecoderLayer(nnx.Module):
  """One DeepSeek-V4.1 layer: optional engram, hyper-connected attention and MoE sub-blocks."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      spec: LayerSpec,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.spec = spec
    self.hc_mult = config.mhc_expansion_rate
    self.norm_eps = config.normalization_layer_epsilon
    kernel_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    norm_kwargs = dict(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.pre_self_attention_layer_norm = RMSNorm(num_features=config.emb_dim, **norm_kwargs)
    self.post_self_attention_layer_norm = RMSNorm(num_features=config.emb_dim, **norm_kwargs)
    self.self_attention = attention_deepseek41.DeepSeek41Attention(
        config=config,
        compress_ratio=spec.compress_ratio,
        is_kv_source=spec.is_kv_source,
        is_index_source=spec.is_index_source,
        is_candidate_source=spec.is_candidate_source,
        uses_candidates=spec.uses_candidates,
        quant=quant,
        kernel_init=kernel_init,
        rngs=rngs,
    )
    self.mlp = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=kernel_init,
        kernel_axes=("embed", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        is_hash_routing=False,
        rngs=rngs,
    )
    mix_hc = (2 + self.hc_mult) * self.hc_mult
    hc_dim = self.hc_mult * config.emb_dim
    init = jax.nn.initializers.normal(stddev=0.02)
    for name in ("hc_attn", "hc_ffn"):
      setattr(
          self, f"{name}_fn", nnx.Param(init(rngs.params(), (hc_dim, mix_hc), jnp.float32), out_sharding=("embed", None))
      )
      setattr(self, f"{name}_base", nnx.Param(jnp.zeros((mix_hc,), jnp.float32), out_sharding=(None,)))
      setattr(self, f"{name}_scale", nnx.Param(jnp.ones((3,), jnp.float32), out_sharding=(None,)))

    if spec.has_engram:
      token_map, compressed_vocab_size = compressed_token_map(config)
      hasher = engram_deepseek41.make_hasher(config, spec.layer_id, token_map, compressed_vocab_size)
      self.engram_hasher = StaticWrapper(hasher)
      self.engram = engram_deepseek41.DeepSeek41Engram(config, hasher.num_embeddings, kernel_init, rngs=rngs)

  def hc_mixes(self, h: jax.Array, prefix: str):
    """h: [B, S, hc, D] -> pre [B, S, hc], post [B, S, hc], comb [B, S, hc, hc] (all float32)."""
    hc = self.hc_mult
    fn, scale, base = (getattr(self, f"{prefix}_{n}")[...] for n in ("fn", "scale", "base"))
    x = h.reshape(h.shape[:2] + (-1,)).astype(jnp.float32)
    rsqrt = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.norm_eps)
    mixes = jnp.einsum("bsk,km->bsm", x, fn, precision=jax.lax.Precision(self.config.matmul_precision)) * rsqrt
    eps = self.config.ds41_hc_eps
    pre = jax.nn.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * jax.nn.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(mixes.shape[:2] + (hc, hc))
    comb = _sinkhorn(comb, self.config.ds41_hc_sinkhorn_iters, eps)
    return pre, post, comb

  @staticmethod
  def hc_pre(h: jax.Array, mix: jax.Array) -> jax.Array:
    """Collapses the hc copies: [B, S, hc, D] x [B, S, hc] -> [B, S, D]."""
    return jnp.sum(mix[..., None] * h.astype(jnp.float32), axis=2).astype(h.dtype)

  @staticmethod
  def hc_post(x: jax.Array, residual: jax.Array, post: jax.Array, comb: jax.Array) -> jax.Array:
    """Expands a sub-block output back to hc copies, mixing the residual in through `comb`."""
    mixed = jnp.einsum(
        "bsjk,bsjd->bskd",
        comb,
        residual.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    return (post[..., None] * x.astype(jnp.float32)[:, :, None, :] + mixed).astype(x.dtype)

  def __call__(
      self,
      carry: dict,
      decoder_positions: jax.Array,
      decoder_input_tokens: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> dict:
    h, pre_mix = carry["h"], carry["pre_mix"]
    shared = {k: v for k, v in carry.items() if k in SHARED_STATE_KEYS}
    if self.spec.has_engram:
      hash_ids = self.engram_hasher.val(decoder_input_tokens, decoder_segment_ids=decoder_segment_ids)
      h = self.engram(h, hash_ids, decoder_segment_ids=decoder_segment_ids)

    residual = h
    attn_pre, attn_post, attn_comb = self.hc_mixes(h, "hc_attn")
    x = self.pre_self_attention_layer_norm(self.hc_pre(h, pre_mix))
    x, updates = self.self_attention(x, decoder_positions, shared, decoder_segment_ids=decoder_segment_ids)
    h = self.hc_post(x, residual, attn_post, attn_comb)

    residual = h
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(h, "hc_ffn")
    x = self.post_self_attention_layer_norm(self.hc_pre(h, attn_pre))
    x, load_balance_loss, moe_bias_updates = self.mlp(x)
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)
    h = self.hc_post(x, residual, ffn_post, ffn_comb)
    return {**carry, **updates, "h": h, "pre_mix": ffn_pre}


def _run_deepseek41_layer_remat(
    layer: nnx.Module,
    carry: dict,
    *,
    decoder_positions: jax.Array,
    decoder_input_tokens: jax.Array,
    decoder_segment_ids: jax.Array | None = None,
    config: Config,
    policy: Any = None,
    prevent_cse: Optional[bool] = None,
) -> dict:
  """Runs one DeepSeek41DecoderLayer under jax.checkpoint inside a length-1 scan."""
  graphdef, params, other = nnx.split(layer, nnx.Param, ...)
  params_xs = jax.tree.map(lambda x: x[None], params)
  other_xs = jax.tree.map(lambda x: x[None], other)
  carry_xs = jax.tree.map(lambda x: x[None], carry)

  def run_single_layer(dummy_carry, xs_slice):
    params_slice, other_slice, carry_slice = xs_slice
    merged_layer = nnx.merge(graphdef, params_slice, other_slice)
    out_carry = merged_layer(
        carry_slice,
        decoder_positions,
        decoder_input_tokens,
        decoder_segment_ids=decoder_segment_ids,
    )
    _, _, new_other = nnx.split(merged_layer, nnx.Param, ...)
    return dummy_carry, (out_carry, new_other)

  if prevent_cse is None:
    prevent_cse = not config.scan_layers and not (
        config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess")
    )
  run_single_layer = jax.checkpoint(run_single_layer, policy=policy, prevent_cse=prevent_cse)
  with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
    _, (stacked_out_carry, stacked_other) = jax.lax.scan(run_single_layer, (), (params_xs, other_xs, carry_xs), length=1)
  nnx.update(layer, jax.tree.map(lambda x: x[0], stacked_other))
  return jax.tree.map(lambda x: x[0], stacked_out_carry)


class DeepSeek41Block(nnx.Module):
  """One group of layers (`layers_0`, `layers_1`, ...); the unit scanned over a segment's repeats."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      specs: tuple[LayerSpec, ...],
      quant: Optional[quantizations.AqtQuantization] = None,
      remat_policy_fn: Any = None,
      prevent_cse_in_remat: Optional[bool] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.model_mode = model_mode
    self.remat_policy_fn = remat_policy_fn
    self.prevent_cse_in_remat = prevent_cse_in_remat
    self.num_layers = len(specs)
    for j, spec in enumerate(specs):
      setattr(
          self,
          f"layers_{j}",
          DeepSeek41DecoderLayer(config=config, mesh=mesh, model_mode=model_mode, spec=spec, quant=quant, rngs=rngs),
      )

  def __call__(
      self,
      carry: dict,
      decoder_positions: jax.Array,
      decoder_input_tokens: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ):
    use_remat = self.model_mode == "train" and bool(self.config.remat_policy) and self.config.remat_policy != "none"
    out = carry
    for j in range(self.num_layers):
      layer = getattr(self, f"layers_{j}")
      if use_remat:
        out = _run_deepseek41_layer_remat(
            layer,
            out,
            decoder_positions=decoder_positions,
            decoder_input_tokens=decoder_input_tokens,
            decoder_segment_ids=decoder_segment_ids,
            config=self.config,
            policy=self.remat_policy_fn,
            prevent_cse=self.prevent_cse_in_remat,
        )
      else:
        out = layer(out, decoder_positions, decoder_input_tokens, decoder_segment_ids=decoder_segment_ids)
    # A scan carry keeps its structure: drop shared state that only lives inside this group. Every group starts
    # with its own sharing source, so nothing produced here is needed by a later repeat.
    return {k: out[k] for k in carry}, None


def initial_carry(h: jax.Array, hc_mult: int) -> dict:
  """Expands the embeddings to hc copies with the one-hot initial collapse mix."""
  h = jnp.repeat(h[:, :, None, :], hc_mult, axis=2)
  pre_mix = jnp.zeros(h.shape[:3], jnp.float32).at[..., 0].set(1.0)
  return {"h": h, "pre_mix": pre_mix}


def _group_reads_and_writes(specs: list[LayerSpec], group: tuple[int, ...]) -> tuple[set, set]:
  """Shared-state keys a group reads before writing them, and the keys it writes."""
  reads, written = set(), set()
  for i in group:
    s = specs[i]
    if s.compress_ratio == 0:
      continue
    needs = set()
    if not s.is_kv_source:
      needs.add("compress_kv")
      if s.is_index_source:
        needs.add("index_k")
    if not s.is_index_source:
      needs.add("topk_mask")
    if s.is_index_source and s.uses_candidates and not s.is_candidate_source:
      needs.add("candidates")
    reads |= needs - written
    if s.is_kv_source:
      written |= {"compress_kv"} | ({"index_k"} if s.is_index_source else set())
    if s.is_index_source:
      written.add("topk_mask")
    if s.is_candidate_source:
      written.add("candidates")
  return reads, written


@dataclasses.dataclass(frozen=True)
class SegmentCarryPlan:
  """How the decoder state is threaded through a scanned segment.

  add_before: shared-state keys this segment produces for later segments; zero placeholders of the right shape are
    added before the segment so its scan carry has a fixed structure.
  keep_after: shared-state keys still needed by later segments.
  compress_ratio: compression ratio of the segment (sizes the placeholders).
  """

  add_before: tuple[str, ...]
  keep_after: tuple[str, ...]
  compress_ratio: int


def build_segment_carry_plans(config: Config, segments: list[Segment]) -> list[SegmentCarryPlan]:
  specs = build_layer_specs(config)
  rw = [_group_reads_and_writes(specs, seg.groups[0]) for seg in segments]
  add_before = [set() for _ in segments]
  keep_after = [set() for _ in segments]
  for j, (reads, _) in enumerate(rw):
    for key in reads:
      writers = [i for i in range(j) if key in rw[i][1]]
      if not writers:
        raise ValueError(f"DeepSeek-V4.1 layers {segments[j].groups[0]} read '{key}' that no earlier layer writes.")
      m = writers[-1]
      add_before[m].add(key)
      for i in range(m, j):
        keep_after[i].add(key)
  plans = []
  for k, seg in enumerate(segments):
    ratio = max(specs[i].compress_ratio for i in seg.groups[0])
    plans.append(
        SegmentCarryPlan(
            add_before=tuple(key for key in SHARED_STATE_KEYS if key in add_before[k]),
            keep_after=tuple(key for key in SHARED_STATE_KEYS if key in keep_after[k]),
            compress_ratio=ratio,
        )
    )
  return plans


def add_placeholders(carry: dict, keys: tuple[str, ...], compress_ratio: int, config: Config) -> dict:
  """Adds zero-valued shared-state entries (with the shapes / dtypes the attention layers produce)."""
  if not keys:
    return carry
  b, s = carry["h"].shape[:2]
  c = s // compress_ratio
  shapes = {
      "compress_kv": ((b, c, config.head_dim), config.dtype),
      "index_k": ((b, c, config.indexer_head_dim), config.dtype),
      "topk_mask": ((b, s, c), jnp.bool_),
      "candidates": ((b, s, c), jnp.bool_),
  }
  out = dict(carry)
  for key in keys:
    if key not in out:
      shape, dtype = shapes[key]
      out[key] = jnp.zeros(shape, dtype)
  return out
