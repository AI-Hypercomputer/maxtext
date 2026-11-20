# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 (Flash 284B / Pro 1600B) attention and hyper-connection layers for m3.

Pure Flax NNX port of the Hugging Face `transformers` DeepSeek-V4 reference (`modeling_deepseek_v4.py`):

* Shared-KV multi-query attention (one KV head, K == V, `head_dim=512`) with low-rank queries, a per-head attention
  sink, partial interleaved RoPE on the trailing `qk_rope_head_dim` channels, reverse RoPE on the output, and a grouped
  low-rank output projection.
* Three layer types: sliding-window attention, heavily compressed attention (HCA, one entry per `compress_rate_hca`
  tokens), and compressed sparse attention (CSA, overlapping windows of `2 * compress_rate_csa` tokens with a Lightning
  Indexer that keeps the top `index_topk` entries per query).
* Manifold-constrained hyper-connections (4 residual streams, Sinkhorn-projected stream mixer) and the final
  hyper-head collapse.

Precision follows the reference: RMSNorm, softmax, Sinkhorn, compressor gates and indexer scores run in float32;
hyper-connection parameters, attention sinks and compressor position biases are stored in float32; RoPE inverse
frequencies are computed on the host with the reference's float32 rounding. Packed sequences are supported through
`segment_ids`: every document gets its own positions and its own complete compression windows, so packed and
standalone documents give the same outputs.

The decoder layer, MoE, model and registry wiring that call these layers arrive with the MoE follow-up.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
import numpy as np

from maxtext.m3.core import sharding as m3_sharding

SLIDING_ATTENTION = "sliding_attention"
COMPRESSED_SPARSE_ATTENTION = "compressed_sparse_attention"
HEAVILY_COMPRESSED_ATTENTION = "heavily_compressed_attention"
LAYER_TYPES = (SLIDING_ATTENTION, COMPRESSED_SPARSE_ATTENTION, HEAVILY_COMPRESSED_ATTENTION)
# Per-layer `compress_ratios` values of the released checkpoints; the reference uses the same fixed mapping.
_COMPRESS_RATIO_TO_LAYER_TYPE = {0: SLIDING_ATTENTION, 4: COMPRESSED_SPARSE_ATTENTION, 128: HEAVILY_COMPRESSED_ATTENTION}

_HIGHEST = jax.lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeepseekV4Config:
  """Static DeepSeek-V4 attention and hyper-connection configuration.

  Attributes:
    hidden_size: Model width D.
    num_hidden_layers: Decoder depth; must equal `len(layer_types)`.
    num_attention_heads: Query heads H (one shared KV head).
    head_dim: Per-head width of queries, keys and values.
    qk_rope_head_dim: Trailing channels per head that receive RoPE.
    q_lora_rank: Width of the low-rank query bottleneck.
    o_groups: Head groups of the grouped output projection.
    o_lora_rank: Per-group width of the grouped output projection.
    sliding_window: Local attention window (query attends to keys with 0 <= q - k < window).
    layer_types: Attention type per layer, from `LAYER_TYPES`.
    compress_rate_csa: Window stride m of compressed sparse attention.
    compress_rate_hca: Window size m' of heavily compressed attention.
    index_n_heads: Lightning Indexer query heads.
    index_head_dim: Lightning Indexer head width.
    index_topk: Compressed entries kept per query by the indexer.
    hc_mult: Number of hyper-connection residual streams.
    hc_sinkhorn_iters: Sinkhorn-Knopp iterations of the stream mixer.
    hc_eps: Hyper-connection numerical floor.
    rms_norm_eps: RMSNorm epsilon.
    rope_theta: RoPE base of sliding-window layers ("main").
    compress_rope_theta: RoPE base of compressed layers and the indexer ("compress").
    yarn_factor: YaRN frequency interpolation factor of the "compress" RoPE.
    yarn_beta_fast: YaRN fast boundary.
    yarn_beta_slow: YaRN slow boundary.
    yarn_original_max_position_embeddings: YaRN original context length.
    initializer_range: Standard deviation of the random weight initializer.
    dtype: Activation dtype name.
    weight_dtype: Storage dtype name of linear and norm weights (precision-sensitive parameters stay float32).
    indexer_loss_scaling_factor: Weight of the optional indexer KL loss; 0 keeps the indexer frozen, as in the
      reference.
    attn_query_chunk_size: Query-sequence tile size used when chunking sliding and compressed attention.
    index_query_chunk_size: Query-sequence tile size used when chunking Lightning Indexer scoring and selection.
    index_window_chunk_size: Compressed-window tile size used when chunking Lightning Indexer scoring and selection.
    index_head_chunk_size: Indexer-head tile size used when chunking Lightning Indexer scoring and selection.
    megablox: Grouped-matmul selection forwarded to the routed-expert executor (consumed by the MoE layers).
    hardware: Target hardware forwarded to the routed-expert executor (consumed by the MoE layers).
  """

  hidden_size: int
  num_hidden_layers: int
  num_attention_heads: int
  head_dim: int
  qk_rope_head_dim: int
  q_lora_rank: int
  o_groups: int
  o_lora_rank: int
  sliding_window: int
  layer_types: tuple[str, ...]
  compress_rate_csa: int
  compress_rate_hca: int
  index_n_heads: int
  index_head_dim: int
  index_topk: int
  hc_mult: int
  hc_sinkhorn_iters: int
  hc_eps: float
  rms_norm_eps: float
  rope_theta: float
  compress_rope_theta: float
  yarn_factor: float
  yarn_beta_fast: float
  yarn_beta_slow: float
  yarn_original_max_position_embeddings: int
  initializer_range: float
  dtype: str
  weight_dtype: str
  indexer_loss_scaling_factor: float = 0.0
  attn_query_chunk_size: int = 128
  index_query_chunk_size: int = 128
  index_window_chunk_size: int = 512
  index_head_chunk_size: int = 16
  megablox: bool = True
  hardware: str = "tpu"

  def __post_init__(self):
    if len(self.layer_types) != self.num_hidden_layers:
      raise ValueError(f"len(layer_types)={len(self.layer_types)} != num_hidden_layers={self.num_hidden_layers}")
    unknown = sorted(set(self.layer_types) - set(LAYER_TYPES))
    if unknown:
      raise ValueError(f"Unknown layer types {unknown}; expected values from {LAYER_TYPES}")
    max_rope_dim = min(self.head_dim, self.index_head_dim)
    if self.qk_rope_head_dim % 2 or not 0 < self.qk_rope_head_dim <= max_rope_dim:
      raise ValueError(
          f"qk_rope_head_dim={self.qk_rope_head_dim} must be even and in "
          f"(0, min(head_dim={self.head_dim}, index_head_dim={self.index_head_dim})]"
      )
    if (self.num_attention_heads * self.head_dim) % self.o_groups:
      raise ValueError(f"num_attention_heads * head_dim must divide by o_groups={self.o_groups}")
    if self.hc_mult < 1 or self.hc_sinkhorn_iters < 1:
      raise ValueError(f"hc_mult={self.hc_mult} and hc_sinkhorn_iters={self.hc_sinkhorn_iters} must be >= 1")
    if self.sliding_window < 1:
      raise ValueError(f"sliding_window={self.sliding_window} must be >= 1")
    if self.indexer_loss_scaling_factor < 0.0:
      raise ValueError(f"indexer_loss_scaling_factor={self.indexer_loss_scaling_factor} must be >= 0")
    chunk_sizes = (
        self.attn_query_chunk_size,
        self.index_query_chunk_size,
        self.index_window_chunk_size,
        self.index_head_chunk_size,
    )
    if any(c < 1 for c in chunk_sizes):
      raise ValueError(f"Attention and indexer chunk sizes {chunk_sizes} must all be >= 1")

  @property
  def compute_dtype(self) -> DTypeLike:
    return jnp.dtype(self.dtype)

  @property
  def param_dtype(self) -> DTypeLike:
    return jnp.dtype(self.weight_dtype)

  def compress_rate(self, layer_type: str) -> int:
    """Returns the compression window of a compressed layer type."""
    if layer_type == COMPRESSED_SPARSE_ATTENTION:
      return self.compress_rate_csa
    if layer_type == HEAVILY_COMPRESSED_ATTENTION:
      return self.compress_rate_hca
    raise ValueError(f"Layer type {layer_type!r} has no compressor")

  @classmethod
  def from_hf_config(
      cls,
      hf: Mapping[str, Any],
      *,
      dtype: str,
      weight_dtype: str,
      indexer_loss_scaling_factor: float = 0.0,
      attn_query_chunk_size: int = 128,
      index_query_chunk_size: int = 128,
      index_window_chunk_size: int = 512,
      index_head_chunk_size: int = 16,
      megablox: bool = True,
      hardware: str = "tpu",
  ) -> "DeepseekV4Config":
    """Builds the config from a Hugging Face `config.json` mapping.

    The per-layer schedule is always taken explicitly from `layer_types` or `compress_ratios[:num_hidden_layers]`
    (release checkpoints append entries for draft layers). The reference's fallback schedule for configs without
    either key is never used, because it inverts layers 2..N-1 of DeepSeek-V4-Pro.

    Args:
      hf: Parsed Hugging Face config.
      dtype: Activation dtype name.
      weight_dtype: Weight storage dtype name.
      indexer_loss_scaling_factor: Weight of the optional indexer KL loss.
      index_query_chunk_size: Query-sequence tile size for the Lightning Indexer.
      index_window_chunk_size: Compressed-window tile size for the Lightning Indexer.
      index_head_chunk_size: Indexer-head tile size for the Lightning Indexer.
      megablox: Grouped-matmul selection for the routed-expert executor.
      hardware: Target hardware for the routed-expert executor.

    Returns:
      The matching DeepseekV4Config.

    Raises:
      ValueError: If the config is missing a schedule, uses more than one KV head, or has an unsupported RoPE setup.
    """
    n = hf["num_hidden_layers"]
    if hf["num_key_value_heads"] != 1:
      raise ValueError(f"DeepSeek-V4 uses one shared KV head; got num_key_value_heads={hf['num_key_value_heads']}")
    if "compress_rates" in hf:
      rate_csa = hf["compress_rates"][COMPRESSED_SPARSE_ATTENTION]
      rate_hca = hf["compress_rates"][HEAVILY_COMPRESSED_ATTENTION]
    else:
      rate_csa, rate_hca = 4, 128
    if hf.get("layer_types") is not None:
      schedule = list(hf["layer_types"])
    elif "compress_ratios" in hf:
      ratio_to_type = {
          0: SLIDING_ATTENTION,
          rate_csa: COMPRESSED_SPARSE_ATTENTION,
          rate_hca: HEAVILY_COMPRESSED_ATTENTION,
      }
      unknown = sorted(set(hf["compress_ratios"][:n]) - set(ratio_to_type))
      if unknown:
        raise ValueError(f"compress_ratios has values {unknown} outside {sorted(ratio_to_type)}")
      schedule = [ratio_to_type[r] for r in hf["compress_ratios"][:n]]
    else:
      raise ValueError("Config has neither layer_types nor compress_ratios; refusing to guess the layer schedule")
    if len(schedule) < n:
      raise ValueError(f"Layer schedule has {len(schedule)} entries for num_hidden_layers={n}")

    head_dim = hf["head_dim"]
    rp = hf.get("rope_parameters") or {}
    rs = hf.get("rope_scaling")
    if isinstance(rp.get("main"), Mapping) and isinstance(rp.get("compress"), Mapping):
      main_rp: dict[str, Any] = dict(rp["main"])
      yarn: dict[str, Any] = dict(rp["compress"])
    elif rs is not None:
      main_rp = {}
      yarn = dict(rs)
    elif rp:
      main_rp = {}
      yarn = {k: v for k, v in rp.items() if k != "rope_theta"}
    else:
      raise ValueError("Config has no YaRN parameters for the compressed RoPE")

    if hf.get("qk_rope_head_dim") is not None:
      rope_dim = int(hf["qk_rope_head_dim"])
    elif hf.get("partial_rotary_factor") is not None:
      rope_dim = int(head_dim * hf["partial_rotary_factor"])
    elif main_rp.get("partial_rotary_factor") is not None:
      rope_dim = int(head_dim * main_rp["partial_rotary_factor"])
    else:
      raise ValueError("Config has neither qk_rope_head_dim nor partial_rotary_factor")

    rope_theta = hf.get("rope_theta")
    if rope_theta is None:
      rope_theta = main_rp.get("rope_theta", rp.get("rope_theta"))
    if rope_theta is None:
      raise ValueError("Config has no rope_theta in top-level fields or rope_parameters")
    compress_rope_theta = hf.get("compress_rope_theta")
    if compress_rope_theta is None and isinstance(rp.get("compress"), Mapping):
      compress_rope_theta = rp["compress"].get("rope_theta")
    if compress_rope_theta is None:
      raise ValueError("Config has no compress_rope_theta in top-level fields or rope_parameters['compress']")

    yarn_type = yarn["type"] if "type" in yarn else yarn["rope_type"]
    if yarn_type != "yarn":
      raise ValueError(f"Compressed RoPE must be YaRN; got {yarn_type!r}")
    if "attention_factor" in yarn and yarn["attention_factor"] != 1.0:
      raise ValueError(f"DeepSeek-V4 does not scale cos/sin; got attention_factor={yarn['attention_factor']}")
    if "mscale" in yarn or "mscale_all_dim" in yarn or yarn.get("truncate", True) is not True:
      raise ValueError(f"Unsupported YaRN options {sorted(yarn)}")

    return cls(
        hidden_size=hf["hidden_size"],
        num_hidden_layers=n,
        num_attention_heads=hf["num_attention_heads"],
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        q_lora_rank=hf["q_lora_rank"],
        o_groups=hf["o_groups"],
        o_lora_rank=hf["o_lora_rank"],
        sliding_window=hf["sliding_window"],
        layer_types=tuple(schedule[:n]),
        compress_rate_csa=rate_csa,
        compress_rate_hca=rate_hca,
        index_n_heads=hf["index_n_heads"],
        index_head_dim=hf["index_head_dim"],
        index_topk=hf["index_topk"],
        hc_mult=hf["hc_mult"],
        hc_sinkhorn_iters=hf["hc_sinkhorn_iters"],
        hc_eps=hf["hc_eps"],
        rms_norm_eps=hf["rms_norm_eps"],
        rope_theta=float(rope_theta),
        compress_rope_theta=float(compress_rope_theta),
        yarn_factor=yarn["factor"],
        yarn_beta_fast=yarn["beta_fast"],
        yarn_beta_slow=yarn["beta_slow"],
        yarn_original_max_position_embeddings=yarn["original_max_position_embeddings"],
        initializer_range=hf["initializer_range"],
        dtype=dtype,
        weight_dtype=weight_dtype,
        indexer_loss_scaling_factor=indexer_loss_scaling_factor,
        attn_query_chunk_size=attn_query_chunk_size,
        index_query_chunk_size=index_query_chunk_size,
        index_window_chunk_size=index_window_chunk_size,
        index_head_chunk_size=index_head_chunk_size,
        megablox=megablox,
        hardware=hardware,
    )


def _is_explicit_mesh() -> bool:
  """Returns whether an explicit-sharding mesh is currently active."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    return False
  axis_types = tuple(mesh.axis_types)
  explicit = sum(1 for t in axis_types if t == jax.sharding.AxisType.Explicit)
  if 0 < explicit < len(axis_types):
    raise ValueError(f"Mesh mixes explicit and non-explicit axis types ({axis_types})")
  return explicit == len(axis_types) and explicit > 0


@dataclasses.dataclass(frozen=True)
class DeepseekV4Sharding(m3_sharding.DefaultSharding):
  """DefaultSharding plus DeepSeek-V4 axes, with mesh axes trimmed to the ones that divide each dimension.

  The single shared KV head cannot be split over `tensor`, so KV and compressor projections shard their input
  (`embed`) dimension over `fsdp` x `expert` instead, and the grouped output projection shards its `o_groups`
  dimension over `tensor`. Small dimensions (indexer position bias, 7168-wide DeepSeek-V4-Pro projections on large
  `fsdp` x `expert` meshes) keep only the leading mesh axes whose product divides them.
  """

  def map_axis(self, axis: str, tensor_name: str, tensor_type: m3_sharding.TensorType) -> m3_sharding.MeshAxes:
    match axis, tensor_type:
      case ("q_lora" | "o_in" | "compress_feature"), m3_sharding.TensorType.WEIGHT:
        return ("fsdp", "expert")
      case ("q_lora" | "o_in" | "compress_feature"), m3_sharding.TensorType.ACTIVATION:
        return None
      case ("o_groups" | "o_mix" | "index_heads"), _:
        return "tensor"
      case _:
        return super().map_axis(axis, tensor_name, tensor_type)

  def _trim_spec(self, tensor_name: str, spec: P, shape: Sequence[int]) -> P:
    """Drops trailing mesh axes on any dimension whose size is not divisible by their product."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
      raise ValueError(f"Tensor {tensor_name!r} is being sharded with no mesh in scope")
    sizes = dict(mesh.shape)
    trimmed = []
    for dim, entry in zip(shape, spec):
      names = (entry,) if isinstance(entry, str) else tuple(entry or ())
      kept, product = [], 1
      for name in names:
        if dim % (product * sizes[name]):
          break
        kept.append(name)
        product *= sizes[name]
      trimmed.append(None if not kept else kept[0] if len(kept) == 1 else tuple(kept))
    return P(*trimmed)

  def param_spec(self, tensor_name: str, axes: Sequence[str | None], shape: Sequence[int]) -> P:
    """Resolves `axes` for a parameter of `shape`, dropping mesh axes that do not divide their dimension."""
    if len(axes) != len(shape):
      raise ValueError(f"Tensor {tensor_name!r}: {len(axes)} logical axes for shape {tuple(shape)}")
    return self._trim_spec(tensor_name, self(tensor_name, axes), shape)

  def activation_spec(self, tensor_name: str, axes: Sequence[str | None], shape: Sequence[int]) -> P:
    """Resolves `axes` for an activation of `shape`, dropping mesh axes that do not divide their dimension."""
    if len(axes) != len(shape):
      raise ValueError(f"Tensor {tensor_name!r}: {len(axes)} logical axes for shape {tuple(shape)}")
    return self._trim_spec(tensor_name, self(tensor_name, axes, m3_sharding.TensorType.ACTIVATION), shape)

  def constrain(self, x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array:
    """Applies a divisibility-trimmed activation sharding constraint to `x`."""
    if len(axes) != x.ndim:
      raise ValueError(f"Tensor {tensor_name!r}: {len(axes)} logical axes for shape {x.shape}")
    if jax.sharding.get_abstract_mesh().empty:
      return x
    spec = self.activation_spec(tensor_name, axes, x.shape)
    if _is_explicit_mesh():
      return jax.sharding.reshard(x, spec)
    return jax.lax.with_sharding_constraint(x, spec)


def _param_metadata(
    sharding: DeepseekV4Sharding | None, name: str, axes: Sequence[str | None], shape: Sequence[int]
) -> dict[str, Any]:
  """Placement metadata for one parameter (see `m3_sharding.param_metadata`); empty when unsharded."""
  if sharding is None:
    return {}
  return {"out_sharding": sharding.param_spec(name, axes, shape), "eager_sharding": True}


def _constrain(sharding: DeepseekV4Sharding | None, x: jax.Array, name: str, axes: Sequence[str | None]) -> jax.Array:
  return x if sharding is None else sharding.constrain(x, name, axes)


def _replicate_axis_if_explicit(x: jax.Array, axis: int) -> jax.Array:
  """Replicates `axis` before chunking when running under an explicit mesh."""
  if not _is_explicit_mesh():
    return x
  cur_spec = tuple(jax.typeof(x).sharding.spec)
  padded_spec = list(cur_spec) + [None] * (x.ndim - len(cur_spec))
  if padded_spec[axis] is None:
    return x
  padded_spec[axis] = None
  return jax.sharding.reshard(x, P(*padded_spec))


def _batch_aligned_full(ref: jax.Array, shape: tuple[int, ...], fill_value: Any, dtype: DTypeLike) -> jax.Array:
  """Creates a constant array matching `ref`'s leading batch sharding under explicit meshes."""
  out = jnp.full(shape, fill_value, dtype=dtype)
  if not _is_explicit_mesh():
    return out
  ref_spec = tuple(jax.typeof(ref).sharding.spec)
  batch_axis = ref_spec[0] if ref_spec else None
  return jax.sharding.reshard(out, P(batch_axis, *([None] * (len(shape) - 1))))


def _match_bh_sharding(ref_bh: jax.Array, x: jax.Array) -> jax.Array:
  """Aligns leading (batch, heads) sharding of `x` to `ref_bh` under explicit meshes."""
  if not _is_explicit_mesh():
    return x
  spec = tuple(jax.typeof(ref_bh).sharding.spec)
  b_ax = spec[0] if spec else None
  h_ax = spec[1] if len(spec) > 1 else None
  return jax.sharding.reshard(x, P(b_ax, h_ax, *([None] * (x.ndim - 2))))


def _masked_logits(logits: jax.Array, mask_bcw: jax.Array, fill: float = -jnp.inf) -> jax.Array:
  """Masks [B, H, c_q, W] `logits` with [B, c_q, W] `mask_bcw`, aligning sharding on explicit meshes."""
  mask = mask_bcw[:, None]
  if _is_explicit_mesh():
    mask = _match_bh_sharding(logits, jnp.broadcast_to(mask, logits.shape))
  return jnp.where(mask, logits, fill)


def _linear(
    cfg: DeepseekV4Config,
    sharding: DeepseekV4Sharding | None,
    rngs: nnx.Rngs,
    name: str,
    in_features: int,
    out_features: int,
    axes: tuple[str | None, str | None],
) -> nnx.Linear:
  """Bias-free linear layer with HIGHEST matmul precision (kernel layout [in, out])."""
  dot_prec = jax.lax.Precision.DEFAULT if cfg.compute_dtype == jnp.bfloat16 else _HIGHEST
  return nnx.Linear(
      in_features,
      out_features,
      use_bias=False,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
      precision=dot_prec,
      kernel_init=nnx.initializers.normal(cfg.initializer_range),
      kernel_metadata=_param_metadata(sharding, name, axes, (in_features, out_features)),
      rngs=rngs,
  )


class DeepseekV4RMSNorm(nnx.Module):
  """Weighted RMSNorm: at-least-float32 normalization and scale, rounded once to the input dtype.

  The scale is stored in `weight_dtype` (the released checkpoints ship it in bfloat16) and upcast inside the call,
  which equals the reference's float32-upcast weight bit for bit while preserving float64 in T14 parity checks.
  """

  def __init__(self, dim: int, *, eps: float, param_dtype: DTypeLike, sharding: DeepseekV4Sharding | None, name: str):
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((dim,), dtype=param_dtype), **_param_metadata(sharding, name, ("norm",), (dim,)))

  def __call__(self, x: jax.Array) -> jax.Array:
    acc = jnp.promote_types(x.dtype, jnp.float32)
    xf = x.astype(acc)
    normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
    return (self.weight[...].astype(acc) * normed).astype(x.dtype)


def unweighted_rms_norm(x: jax.Array, eps: float) -> jax.Array:
  """Parameter-free RMSNorm: x * rsqrt(mean(x^2) + eps), with the statistic in at least float32."""
  acc = jnp.promote_types(x.dtype, jnp.float32)
  xf = x.astype(acc)
  return x * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps).astype(x.dtype)


def rope_inv_freq(cfg: DeepseekV4Config, rope_type: str, *, dtype: DTypeLike = np.float32) -> np.ndarray:
  """Host RoPE inverse frequencies matching the reference's float32 rounding (or exact float64 for T14).

  `base ** (2i / dim)` is evaluated in float64 and rounded once to `out_dtype` (matching the reference's float32 power
  when `dtype=np.float32`, or staying in float64 when `dtype=np.float64`), then the reciprocal and YaRN blend run in
  `out_dtype`. Computing the power on device instead adds 1-2 ULP errors that grow linearly with position.

  Args:
    cfg: Model config.
    rope_type: "main" (sliding-window layers) or "compress" (compressed layers and indexer, YaRN).
    dtype: Output precision (`np.float32` by default; `np.float64` when running float64 T14 checks).

  Returns:
    Array of shape [qk_rope_head_dim // 2] in `out_dtype`.
  """
  out_dtype = np.float64 if np.dtype(dtype) == np.float64 else np.float32
  dim = cfg.qk_rope_head_dim
  # Integer / dim is exact in float64; rounding to out_dtype matches the reference division when out_dtype is float32.
  exponents = (np.arange(0, dim, 2, dtype=np.float64) / dim).astype(out_dtype).astype(np.float64)
  if rope_type == "main":
    pos_freqs = (np.float64(cfg.rope_theta) ** exponents).astype(out_dtype)
    return (out_dtype(1.0) / pos_freqs).astype(out_dtype)
  if rope_type != "compress":
    raise ValueError(f"Unknown rope_type {rope_type!r}")
  base = cfg.compress_rope_theta
  pos_freqs = (np.float64(base) ** exponents).astype(out_dtype)
  inv_extrapolation = out_dtype(1.0) / pos_freqs
  inv_interpolation = out_dtype(1.0) / (out_dtype(cfg.yarn_factor) * pos_freqs)

  def correction_dim(num_rotations: float) -> float:
    return (dim * math.log(cfg.yarn_original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

  low = max(math.floor(correction_dim(cfg.yarn_beta_fast)), 0)
  high = min(math.ceil(correction_dim(cfg.yarn_beta_slow)), dim - 1)
  if low == high:
    high += 0.001
  ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / np.float64(out_dtype(high - low)), 0, 1)
  extrapolation_factor = out_dtype(1.0) - ramp.astype(out_dtype)
  inv_freq = inv_interpolation * (out_dtype(1.0) - extrapolation_factor) + inv_extrapolation * extrapolation_factor
  return inv_freq.astype(out_dtype)


def rope_cos_sin(inv_freq: Sequence[float], positions: jax.Array, dtype: DTypeLike) -> tuple[jax.Array, jax.Array]:
  """Returns cos/sin of shape [..., rope_dim // 2] for integer `positions`, cast to `dtype` like the reference."""
  acc = jnp.promote_types(dtype, jnp.float32)
  freqs = positions.astype(acc)[..., None] * jnp.asarray(inv_freq, dtype=acc)
  return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def _rotate_interleaved(x: jax.Array) -> jax.Array:
  """Maps interleaved pairs (x0, x1) -> (-x1, x0)."""
  return jnp.stack((-x[..., 1::2], x[..., 0::2]), axis=-1).reshape(x.shape)


def apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
  """Interleaved RoPE on the trailing `2 * cos.shape[-1]` channels of `x`, computed in at least float32.

  Args:
    x: [B, S, D] or [B, S, H, D].
    cos: [B, S, rope_dim // 2].
    sin: [B, S, rope_dim // 2]; pass `-sin` for the inverse rotation.

  Returns:
    Array shaped like `x`; the leading `D - rope_dim` channels are unchanged.
  """
  acc = jnp.promote_types(x.dtype, jnp.float32)
  cos = jnp.repeat(cos, 2, axis=-1).astype(acc)
  sin = jnp.repeat(sin, 2, axis=-1).astype(acc)
  if x.ndim == 4:
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = (rope.astype(acc) * cos + _rotate_interleaved(rope).astype(acc) * sin).astype(x.dtype)
  return jnp.concatenate([nope, rotated], axis=-1)


class DeepseekV4GroupedLinear(nnx.Module):
  """Block-diagonal output projection: each of `n_groups` head groups maps to its own `out_per_group` features."""

  def __init__(
      self,
      cfg: DeepseekV4Config,
      in_per_group: int,
      out_per_group: int,
      n_groups: int,
      *,
      sharding: DeepseekV4Sharding | None,
      rngs: nnx.Rngs,
      name: str,
  ):
    shape = (n_groups, in_per_group, out_per_group)
    self.dtype = cfg.compute_dtype
    self.weight = nnx.Param(
        nnx.initializers.normal(cfg.initializer_range)(rngs.params(), shape, cfg.param_dtype),
        **_param_metadata(sharding, name, ("o_groups", "o_in", None), shape),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Maps [..., n_groups, in_per_group] to [..., n_groups, out_per_group]."""
    dot_prec = jax.lax.Precision.DEFAULT if self.dtype == jnp.bfloat16 else _HIGHEST
    return jnp.einsum("...gi,gio->...go", x.astype(self.dtype), self.weight[...].astype(self.dtype), precision=dot_prec)


class DeepseekV4HyperConnection(nnx.Module):
  """Manifold-constrained hyper-connection mapping (at least float32), returning (post, comb, collapsed).

  `pre` collapses the `hc_mult` streams into the sublayer input, `post` places the sublayer output back on each
  stream, and `comb` is the Sinkhorn-projected doubly-stochastic stream mixer applied as comb^T @ streams.
  """

  def __init__(self, cfg: DeepseekV4Config, *, sharding: DeepseekV4Sharding | None, rngs: nnx.Rngs, name: str):
    self.hc_mult = cfg.hc_mult
    self.iters = cfg.hc_sinkhorn_iters
    self.eps = cfg.hc_eps
    self.norm_eps = cfg.rms_norm_eps
    mix = (2 + cfg.hc_mult) * cfg.hc_mult
    fn_shape = (cfg.hc_mult * cfg.hidden_size, mix)
    p_dtype = jnp.promote_types(cfg.param_dtype, jnp.float32)
    self.fn = nnx.Param(
        nnx.initializers.normal(cfg.initializer_range)(rngs.params(), fn_shape, p_dtype),
        **_param_metadata(sharding, f"{name}.fn", ("embed", None), fn_shape),
    )
    self.base = nnx.Param(jnp.zeros((mix,), p_dtype), **_param_metadata(sharding, f"{name}.base", (None,), (mix,)))
    self.scale = nnx.Param(jnp.ones((3,), p_dtype), **_param_metadata(sharding, f"{name}.scale", (None,), (3,)))

  def __call__(self, streams: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Maps [B, S, hc, D] streams to post [B, S, hc], comb [B, S, hc, hc], and collapsed [B, S, D]."""
    hc, eps = self.hc_mult, self.eps
    b, s = streams.shape[:2]
    acc = jnp.promote_types(streams.dtype, jnp.float32)
    flat = unweighted_rms_norm(streams.reshape(b, s, -1).astype(acc), self.norm_eps)
    mixes = jnp.matmul(flat, self.fn[...].astype(acc), precision=_HIGHEST)
    pre_w, post_w, comb_w = jnp.split(mixes, [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = jnp.split(self.base[...].astype(acc), [hc, 2 * hc])
    pre_s, post_s, comb_s = self.scale[...].astype(acc)
    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)
    comb = jax.nn.softmax(comb_w.reshape(b, s, hc, hc) * comb_s + comb_b.reshape(hc, hc), axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(self.iters - 1):
      comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps)
      comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    collapsed = jnp.sum(pre[..., None] * streams.astype(acc), axis=2).astype(streams.dtype)
    return post, comb, collapsed


class DeepseekV4HyperHead(nnx.Module):
  """Final sigmoid-weighted collapse of the `hc_mult` streams (at least float32 mapping)."""

  def __init__(self, cfg: DeepseekV4Config, *, sharding: DeepseekV4Sharding | None, rngs: nnx.Rngs, name: str):
    self.eps = cfg.hc_eps
    self.norm_eps = cfg.rms_norm_eps
    fn_shape = (cfg.hc_mult * cfg.hidden_size, cfg.hc_mult)
    p_dtype = jnp.promote_types(cfg.param_dtype, jnp.float32)
    self.hc_fn = nnx.Param(
        nnx.initializers.normal(cfg.initializer_range)(rngs.params(), fn_shape, p_dtype),
        **_param_metadata(sharding, f"{name}.hc_fn", ("embed", None), fn_shape),
    )
    self.hc_base = nnx.Param(
        jnp.zeros((cfg.hc_mult,), p_dtype), **_param_metadata(sharding, f"{name}.hc_base", (None,), (cfg.hc_mult,))
    )
    self.hc_scale = nnx.Param(jnp.ones((1,), p_dtype), **_param_metadata(sharding, f"{name}.hc_scale", (None,), (1,)))

  def __call__(self, streams: jax.Array) -> jax.Array:
    b, s = streams.shape[:2]
    acc = jnp.promote_types(streams.dtype, jnp.float32)
    flat = unweighted_rms_norm(streams.reshape(b, s, -1).astype(acc), self.norm_eps)
    mixes = jnp.matmul(flat, self.hc_fn[...].astype(acc), precision=_HIGHEST)
    pre = jax.nn.sigmoid(mixes * self.hc_scale[...].astype(acc) + self.hc_base[...].astype(acc)) + self.eps
    return jnp.sum(pre[..., None] * streams.astype(acc), axis=2).astype(streams.dtype)


@dataclasses.dataclass(frozen=True)
class _Documents:
  """Per-token document bookkeeping; padding tokens carry segment id 0."""

  segment_ids: jax.Array  # [B, S] int32
  positions: jax.Array  # [B, S] position inside the document
  doc_start: jax.Array  # [B, S] absolute index of the document's first token
  single_document: bool  # True when no segment ids were given (one unpadded document per row)


def _documents(segment_ids: jax.Array | None, batch: int, seq: int) -> _Documents:
  idx = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq))
  if segment_ids is None:
    return _Documents(jnp.ones((batch, seq), jnp.int32), idx, jnp.zeros((batch, seq), jnp.int32), True)
  seg = _replicate_axis_if_explicit(segment_ids.astype(jnp.int32), 1)
  first = _batch_aligned_full(seg, (batch, 1), True, bool)
  is_start = jnp.concatenate([first, seg[:, 1:] != seg[:, :-1]], axis=1)
  doc_start = jax.lax.cummax(jnp.where(is_start, idx, 0), axis=1)
  return _Documents(seg, idx - doc_start, doc_start, False)


@dataclasses.dataclass(frozen=True)
class _Windows:
  """Complete compression windows, one slot per possible window (`seq // rate` slots)."""

  starts: jax.Array  # [B, n] absolute index of each window's first token
  valid: jax.Array  # [B, n] slot holds a real window
  index_in_doc: jax.Array  # [B, n] window number inside its document
  segment_ids: jax.Array  # [B, n]
  doc_start: jax.Array  # [B, n]
  rate: int
  single_document: bool


def _windows(docs: _Documents, rate: int) -> _Windows:
  """Finds every complete `rate`-token window that lies inside one document and starts at a multiple of `rate`."""
  batch, seq = docs.segment_ids.shape
  n = seq // rate
  if docs.single_document:
    slots = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (batch, n))
    return _Windows(
        slots * rate,
        jnp.ones((batch, n), bool),
        slots,
        jnp.ones((batch, n), jnp.int32),
        jnp.zeros((batch, n), jnp.int32),
        rate,
        True,
    )
  idx = jnp.arange(seq)
  last = jnp.clip(idx + rate - 1, 0, seq - 1)
  is_start = (
      (docs.segment_ids != 0)
      & (docs.positions % rate == 0)
      & (idx + rate - 1 < seq)[None, :]
      & (docs.segment_ids[:, last] == docs.segment_ids)
      & (docs.doc_start[:, last] == docs.doc_start)
  )
  starts = jax.vmap(lambda row: jnp.nonzero(row, size=n, fill_value=0)[0])(is_start).astype(jnp.int32)
  valid = jnp.arange(n)[None, :] < jnp.sum(is_start, axis=1, keepdims=True)

  def take(arr: jax.Array) -> jax.Array:
    return jnp.take_along_axis(arr, starts, axis=1)

  return _Windows(starts, valid, take(docs.positions) // rate, take(docs.segment_ids), take(docs.doc_start), rate, False)


def _gather_windows(x: jax.Array, win: _Windows) -> jax.Array:
  """Gathers [B, S, F] into [B, n, rate, F] window chunks."""
  batch, _, features = x.shape
  n = win.starts.shape[1]
  if win.single_document:
    return x[:, : n * win.rate].reshape(batch, n, win.rate, features)
  idx = (win.starts[:, :, None] + jnp.arange(win.rate)[None, None, :]).reshape(batch, n * win.rate)
  return jnp.take_along_axis(x, idx[:, :, None], axis=1).reshape(batch, n, win.rate, features)


def _visible_windows(docs: _Documents, win: _Windows) -> jax.Array:
  """[B, S, n] mask: query t sees window w iff w is a complete window of t's document ending at or before t."""
  causal = win.index_in_doc[:, None, :] < ((docs.positions + 1) // win.rate)[:, :, None]
  if win.single_document:
    return causal
  same_doc = (win.segment_ids[:, None, :] == docs.segment_ids[:, :, None]) & (
      win.doc_start[:, None, :] == docs.doc_start[:, :, None]
  )
  return causal & same_doc & win.valid[:, None, :] & (docs.segment_ids != 0)[:, :, None]


def _pool_windows(
    kv: jax.Array,
    gate: jax.Array,
    position_bias: jax.Array,
    win: _Windows,
    *,
    head_dim: int,
    overlap: bool,
) -> jax.Array:
  """Softmax-gated pooling of each window (float32 softmax), returning [B, n, head_dim] before norm and RoPE.

  With `overlap`, projections carry two series per token (`[..., :head_dim]` = Ca, `[..., head_dim:]` = Cb) and
  window w pools the previous window's Ca slots together with its own Cb slots; the first window of a document has
  no previous slots (zero values, -inf gates).
  """
  chunk_kv = _gather_windows(kv, win)
  bias = _replicate_axis_if_explicit(position_bias, 1).astype(gate.dtype)
  chunk_gate = _gather_windows(gate, win) + bias
  if overlap:
    has_prev = (win.index_in_doc > 0)[:, :, None, None]
    prev_kv = jnp.roll(chunk_kv[..., :head_dim], 1, axis=1)
    prev_gate = jnp.roll(chunk_gate[..., :head_dim], 1, axis=1)
    chunk_kv = jnp.concatenate([jnp.where(has_prev, prev_kv, 0), chunk_kv[..., head_dim:]], axis=2)
    chunk_gate = jnp.concatenate([jnp.where(has_prev, prev_gate, -jnp.inf), chunk_gate[..., head_dim:]], axis=2)
  acc = jnp.promote_types(chunk_kv.dtype, jnp.float32)
  weights = jax.nn.softmax(chunk_gate.astype(acc), axis=2).astype(chunk_kv.dtype)
  return jnp.sum(chunk_kv * weights, axis=2)


class _WindowCompressor(nnx.Module):
  """Shared parameters and pooling of the HCA compressor, the CSA compressor and the indexer's compressor."""

  def __init__(
      self,
      cfg: DeepseekV4Config,
      *,
      head_dim: int,
      rate: int,
      overlap: bool,
      sharding: DeepseekV4Sharding | None,
      rngs: nnx.Rngs,
      name: str,
  ):
    self.head_dim, self.rate, self.overlap = head_dim, rate, overlap
    self.dtype = cfg.compute_dtype
    rope_dtype = np.float64 if cfg.compute_dtype == jnp.float64 else np.float32
    self.inv_freq = tuple(float(v) for v in rope_inv_freq(cfg, "compress", dtype=rope_dtype))
    width = 2 * head_dim if overlap else head_dim
    self.kv_proj = _linear(cfg, sharding, rngs, f"{name}.kv_proj", cfg.hidden_size, width, ("embed", None))
    self.gate_proj = _linear(cfg, sharding, rngs, f"{name}.gate_proj", cfg.hidden_size, width, ("embed", None))
    p_dtype = jnp.promote_types(cfg.param_dtype, jnp.float32)
    self.position_bias = nnx.Param(
        jnp.zeros((rate, width), p_dtype),
        **_param_metadata(sharding, f"{name}.position_bias", (None, "compress_feature"), (rate, width)),
    )
    self.kv_norm = DeepseekV4RMSNorm(
        head_dim, eps=cfg.rms_norm_eps, param_dtype=cfg.param_dtype, sharding=sharding, name=f"{name}.kv_norm"
    )

  def compress(self, hidden: jax.Array, win: _Windows) -> jax.Array:
    """Returns the RoPE-rotated compressed entries [B, n, head_dim] of every window slot."""
    pooled = _pool_windows(
        self.kv_proj(hidden),
        self.gate_proj(hidden),
        self.position_bias[...],
        win,
        head_dim=self.head_dim,
        overlap=self.overlap,
    )
    compressed = self.kv_norm(pooled)
    cos, sin = rope_cos_sin(self.inv_freq, win.index_in_doc * self.rate, compressed.dtype)
    return apply_rotary(compressed, cos, sin)


def _pad_axis(x: jax.Array, axis: int, target: int, pad_value: Any = 0) -> jax.Array:
  """Pads `x` along `axis` on the right to `target`."""
  pad = target - x.shape[axis]
  if pad == 0:
    return x
  widths = [(0, 0)] * x.ndim
  widths[axis] = (0, pad)
  return jnp.pad(x, widths, constant_values=pad_value)


def _score_tile(
    q_c: jax.Array,
    w_c: jax.Array,
    k_w: jax.Array,
    *,
    head_scale: float,
    head_chunk_size: int,
) -> jax.Array:
  """Computes [B, c_q, c_w] indexer scores without materializing more than `head_chunk_size` heads at once."""
  acc = w_c.dtype
  batch, c_q, num_heads, head_dim = q_c.shape
  c_w = k_w.shape[1]
  c_h = min(num_heads, head_chunk_size)
  if num_heads <= c_h:
    logits = jnp.einsum("bchd,bwd->bchw", q_c.astype(acc), k_w.astype(acc), precision=_HIGHEST)
    logits = jax.nn.relu(logits) * head_scale
    return jnp.sum(logits * w_c[..., None], axis=2)

  q_rep = _replicate_axis_if_explicit(q_c, 2)
  w_rep = _replicate_axis_if_explicit(w_c, 2)
  n_h = (num_heads + c_h - 1) // c_h
  padded_h = n_h * c_h
  q_chunks = jnp.moveaxis(_pad_axis(q_rep, 2, padded_h).reshape(batch, c_q, n_h, c_h, head_dim), 2, 0)
  w_chunks = jnp.moveaxis(_pad_axis(w_rep, 2, padded_h).reshape(batch, c_q, n_h, c_h), 2, 0)
  k_acc = k_w.astype(acc)

  @jax.checkpoint
  def head_step(total: jax.Array, hw: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
    q_h, w_h = hw
    logits = jnp.einsum("bchd,bwd->bchw", q_h.astype(acc), k_acc, precision=_HIGHEST)
    logits = jax.nn.relu(logits) * head_scale
    return total + jnp.sum(logits * w_h[..., None], axis=2), None

  init = _batch_aligned_full(q_rep, (batch, c_q, c_w), 0, acc)
  total, _ = jax.lax.scan(head_step, init, (q_chunks, w_chunks))
  return total


def _topk_to_mask(topk: jax.Array, n: int) -> jax.Array:
  """Converts `topk` indices [B, S, k] (-1 for unusable picks) to a [B, S, n] boolean mask without 4D one_hot."""
  batch, seq, _ = topk.shape
  if n == 0 or seq == 0:
    return jnp.zeros((batch, seq, n), dtype=bool)

  def row_mask(idx_row: jax.Array) -> jax.Array:
    buf = jnp.zeros((n + 1,), dtype=bool)
    return buf.at[idx_row + 1].set(idx_row >= 0)[1:]

  return jax.vmap(jax.vmap(row_mask))(topk)


class DeepseekV4Indexer(nnx.Module):
  """Lightning Indexer: scores compressed windows with sum_h w_h * ReLU(q_h . k) and keeps the top `index_topk`.

  Scoring and top-k selection run over bounded query, window, and head tiles via checkpointed `jax.lax.scan` loops so
  the dense 4D `[B, S, H, n]` tensor is never materialized in HBM (and `[B, S, n]` scores are also avoided when the
  optional indexer KL loss is disabled). Inputs are stop-gradient so that the optional indexer KL loss trains only the
  indexer's own parameters.
  """

  def __init__(self, cfg: DeepseekV4Config, *, sharding: DeepseekV4Sharding | None, rngs: nnx.Rngs, name: str):
    self.num_heads, self.head_dim, self.topk = cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk
    self.query_chunk_size = cfg.index_query_chunk_size
    self.window_chunk_size = cfg.index_window_chunk_size
    self.head_chunk_size = cfg.index_head_chunk_size
    rope_dtype = np.float64 if cfg.compute_dtype == jnp.float64 else np.float32
    self.inv_freq = tuple(float(v) for v in rope_inv_freq(cfg, "compress", dtype=rope_dtype))
    self.compressor = _WindowCompressor(
        cfg,
        head_dim=cfg.index_head_dim,
        rate=cfg.compress_rate_csa,
        overlap=True,
        sharding=sharding,
        rngs=rngs,
        name=name,
    )
    self.q_b_proj = _linear(
        cfg,
        sharding,
        rngs,
        f"{name}.q_b_proj",
        cfg.q_lora_rank,
        cfg.index_n_heads * cfg.index_head_dim,
        ("q_lora", "index_heads"),
    )
    self.weights_proj = _linear(
        cfg, sharding, rngs, f"{name}.weights_proj", cfg.hidden_size, cfg.index_n_heads, ("embed", "index_heads")
    )

  def _project_inputs(
      self, hidden: jax.Array, q_residual: jax.Array, docs: _Documents, win: _Windows
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projects stop-gradient inputs to rotary queries [B, S, H, D], compressed keys [B, n, D], and head weights."""
    hidden = jax.lax.stop_gradient(hidden)
    q_residual = jax.lax.stop_gradient(q_residual)
    keys = self.compressor.compress(hidden, win)
    batch, seq, _ = hidden.shape
    q = self.q_b_proj(q_residual).reshape(batch, seq, self.num_heads, self.head_dim)
    cos, sin = rope_cos_sin(self.inv_freq, docs.positions, q.dtype)
    q = apply_rotary(q, cos, sin)
    acc = jnp.promote_types(hidden.dtype, jnp.float32)
    head_weights = self.weights_proj(hidden).astype(acc) * (self.num_heads**-0.5)
    return q, keys, head_weights

  def _score_query_chunk(self, q_c: jax.Array, w_c: jax.Array, keys: jax.Array) -> jax.Array:
    """Scores one query chunk [B, c_q, ...] against all `n` windows in bounded window/head tiles."""
    batch = q_c.shape[0]
    n, head_dim = keys.shape[1], keys.shape[2]
    c_w = min(n, self.window_chunk_size)
    head_scale = self.head_dim**-0.5
    if n <= c_w:
      return _score_tile(q_c, w_c, keys, head_scale=head_scale, head_chunk_size=self.head_chunk_size)

    n_w = (n + c_w - 1) // c_w
    k_chunks = jnp.moveaxis(_pad_axis(keys, 1, n_w * c_w).reshape(batch, n_w, c_w, head_dim), 1, 0)

    @jax.checkpoint
    def win_step(_: None, k_w: jax.Array) -> tuple[None, jax.Array]:
      return None, _score_tile(q_c, w_c, k_w, head_scale=head_scale, head_chunk_size=self.head_chunk_size)

    _, tiles = jax.lax.scan(win_step, None, k_chunks)
    return jnp.moveaxis(tiles, 0, 2).reshape(batch, q_c.shape[1], n_w * c_w)[:, :, :n]

  def scores(self, hidden: jax.Array, q_residual: jax.Array, docs: _Documents, win: _Windows) -> jax.Array:
    """Returns at-least-float32 index scores [B, S, n] before masking without materializing [B, S, H, n]."""
    batch, seq, _ = hidden.shape
    n = win.starts.shape[1]
    acc = jnp.promote_types(hidden.dtype, jnp.float32)
    if seq == 0 or n == 0:
      return jnp.zeros((batch, seq, n), dtype=acc)
    q, keys, head_weights = self._project_inputs(hidden, q_residual, docs, win)
    c_q = min(seq, self.query_chunk_size)
    if seq <= c_q:
      return self._score_query_chunk(q, head_weights, keys)

    n_q = (seq + c_q - 1) // c_q
    padded_seq = n_q * c_q
    q_chunks = jnp.moveaxis(_pad_axis(q, 1, padded_seq).reshape(batch, n_q, c_q, self.num_heads, self.head_dim), 1, 0)
    w_chunks = jnp.moveaxis(_pad_axis(head_weights, 1, padded_seq).reshape(batch, n_q, c_q, self.num_heads), 1, 0)

    @jax.checkpoint
    def query_step(_: None, qw: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
      q_c, w_c = qw
      return None, self._score_query_chunk(q_c, w_c, keys)

    _, chunk_scores = jax.lax.scan(query_step, None, (q_chunks, w_chunks))
    return jnp.moveaxis(chunk_scores, 0, 1).reshape(batch, padded_seq, n)[:, :seq, :]

  def select_topk(self, hidden: jax.Array, q_residual: jax.Array, docs: _Documents, win: _Windows) -> jax.Array:
    """Streams top-k window selection [B, S, k] (-1 for unusable picks) without materializing [B, S, H, n] or [B, S, n]."""
    batch, seq, _ = hidden.shape
    n = win.starts.shape[1]
    k = min(self.topk, n)
    if seq == 0 or k == 0:
      return jnp.zeros((batch, seq, k), dtype=jnp.int32)
    q, keys, head_weights = jax.lax.stop_gradient(self._project_inputs(hidden, q_residual, docs, win))
    acc = head_weights.dtype
    head_scale = self.head_dim**-0.5
    c_q = min(seq, self.query_chunk_size)
    c_w = min(n, self.window_chunk_size)
    n_q = (seq + c_q - 1) // c_q
    n_w = (n + c_w - 1) // c_w
    padded_seq = n_q * c_q
    padded_n = n_w * c_w

    def chunk_seq(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
      padded = _pad_axis(arr, 1, padded_seq, pad_val)
      return jnp.moveaxis(padded.reshape(batch, n_q, c_q, *arr.shape[2:]), 1, 0)

    def chunk_win(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
      padded = _pad_axis(arr, 1, padded_n, pad_val)
      return jnp.moveaxis(padded.reshape(batch, n_w, c_w, *arr.shape[2:]), 1, 0)

    # Padded queries get position -1 and segment 0; padded windows get valid=False and index_in_doc=seq+1 so both
    # single-document and packed-document visibility masks evaluate strictly to False on all padding slots.
    q_xs = (
        chunk_seq(q),
        chunk_seq(head_weights),
        chunk_seq(docs.segment_ids, 0),
        chunk_seq(docs.positions, -1),
        chunk_seq(docs.doc_start, 0),
    )
    w_xs = (
        chunk_win(keys),
        jnp.arange(padded_n, dtype=jnp.int32).reshape(n_w, c_w),
        chunk_win(win.starts, 0),
        chunk_win(win.valid, False),
        chunk_win(win.index_in_doc, seq + 1),
        chunk_win(win.segment_ids, 0),
        chunk_win(win.doc_start, 0),
    )

    @jax.checkpoint
    def select_query_chunk(
        q_c: jax.Array,
        w_c: jax.Array,
        seg_q: jax.Array,
        pos_q: jax.Array,
        start_q: jax.Array,
    ) -> jax.Array:
      docs_c = _Documents(seg_q, pos_q, start_q, docs.single_document)

      @jax.checkpoint
      def win_step(
          carry: tuple[jax.Array, jax.Array],
          w_item: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
      ) -> tuple[tuple[jax.Array, jax.Array], None]:
        run_vals, run_idx = carry
        k_w, w_ids, w_starts, w_valid, w_idx_doc, w_seg, w_doc_start = w_item
        win_w = _Windows(w_starts, w_valid, w_idx_doc, w_seg, w_doc_start, win.rate, win.single_document)
        scores_cw = _score_tile(q_c, w_c, k_w, head_scale=head_scale, head_chunk_size=self.head_chunk_size)
        vis_cw = _visible_windows(docs_c, win_w)
        chunk_vals = jnp.where(vis_cw, scores_cw, -jnp.inf)
        chunk_idx = _batch_aligned_full(q_c, (batch, c_q, c_w), 0, jnp.int32) + w_ids[None, None, :]
        # Concatenating `run_vals` ahead of `chunk_vals` preserves exact lowest-window-index tie-breaking across tiles.
        cand_vals = jnp.concatenate([run_vals, chunk_vals], axis=-1)
        cand_idx = jnp.concatenate([run_idx, chunk_idx], axis=-1)
        merged_vals, pos = jax.lax.top_k(cand_vals, k)
        merged_idx = jnp.take_along_axis(cand_idx, pos, axis=-1)
        merged_idx = jnp.where(merged_vals > -jnp.inf, merged_idx, -1)
        return (merged_vals, merged_idx), None

      if n_w == 1:
        k_w, w_ids, w_starts, w_valid, w_idx_doc, w_seg, w_doc_start = (arr[0] for arr in w_xs)
        win_w = _Windows(w_starts, w_valid, w_idx_doc, w_seg, w_doc_start, win.rate, win.single_document)
        scores_cw = _score_tile(q_c, w_c, k_w, head_scale=head_scale, head_chunk_size=self.head_chunk_size)
        vis_cw = _visible_windows(docs_c, win_w)
        top_vals, top_pos = jax.lax.top_k(jnp.where(vis_cw, scores_cw, -jnp.inf), k)
        top_idx = jnp.take_along_axis(
            _batch_aligned_full(q_c, (batch, c_q, c_w), 0, jnp.int32) + w_ids[None, None, :], top_pos, axis=-1
        )
        return jnp.where(top_vals > -jnp.inf, top_idx, -1)

      init_vals = _batch_aligned_full(q_c, (batch, c_q, k), -jnp.inf, acc)
      init_idx = _batch_aligned_full(q_c, (batch, c_q, k), -1, jnp.int32)
      (_, final_idx), _ = jax.lax.scan(win_step, (init_vals, init_idx), w_xs)
      return final_idx

    if n_q == 1:
      return select_query_chunk(*(arr[0] for arr in q_xs))[:, :seq, :]

    def query_step(
        _: None, q_item: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[None, jax.Array]:
      return None, select_query_chunk(*q_item)

    _, chunk_topk = jax.lax.scan(query_step, None, q_xs)
    return jnp.moveaxis(chunk_topk, 0, 1).reshape(batch, padded_seq, k)[:, :seq, :]

  def select(self, scores: jax.Array, visible: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns (indices [B, S, k] with -1 for unusable picks, selected-window mask [B, S, n])."""
    batch, seq, n = scores.shape
    k = min(self.topk, n)
    if k == 0 or seq == 0:
      return jnp.zeros((batch, seq, k), dtype=jnp.int32), jnp.zeros((batch, seq, n), dtype=bool)
    top_vals, top_idx = jax.lax.top_k(jnp.where(visible, scores, -jnp.inf), k)
    topk = jnp.where(top_vals > -jnp.inf, top_idx, -1)
    return topk, _topk_to_mask(topk, n)

  def kl_loss(
      self,
      hidden: jax.Array,
      q_residual: jax.Array,
      query: jax.Array,
      compressed_kv: jax.Array,
      docs: _Documents,
      win: _Windows,
      *,
      scale: float,
  ) -> jax.Array:
    """Streams KL(attention target || indexer) over bounded query/window tiles without [B, H, S, n] or [B, S, n]."""
    batch, seq, _ = hidden.shape
    n = win.starts.shape[1]
    if seq == 0 or n == 0:
      return jnp.zeros((), dtype=jnp.float32)
    dot_prec = jax.lax.Precision.DEFAULT if query.dtype == jnp.bfloat16 else _HIGHEST
    q_idx, idx_keys, head_weights = self._project_inputs(hidden, q_residual, docs, win)
    q_attn = _replicate_axis_if_explicit(_replicate_axis_if_explicit(jax.lax.stop_gradient(query), 1), 2).astype(
        jnp.float32
    )
    comp_kv = _replicate_axis_if_explicit(jax.lax.stop_gradient(compressed_kv), 1).astype(jnp.float32)
    q_idx_rep = _replicate_axis_if_explicit(q_idx, 1)
    w_idx_rep = _replicate_axis_if_explicit(head_weights, 1)
    idx_keys_rep = _replicate_axis_if_explicit(idx_keys, 1)
    seg_rep = _batch_aligned_full(q_attn, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.segment_ids, 1)
    pos_rep = _batch_aligned_full(q_attn, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.positions, 1)
    dstart_rep = _batch_aligned_full(q_attn, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.doc_start, 1)

    c_q = min(seq, self.query_chunk_size)
    c_w = min(n, self.window_chunk_size)
    n_q = (seq + c_q - 1) // c_q
    n_w = (n + c_w - 1) // c_w
    padded_seq = n_q * c_q
    padded_n = n_w * c_w
    head_scale = self.head_dim**-0.5

    def chunk_q(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
      padded = _pad_axis(arr, 1, padded_seq, pad_val)
      return jnp.moveaxis(padded.reshape(batch, n_q, c_q, *arr.shape[2:]), 1, 0)

    def chunk_w(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
      padded = _pad_axis(arr, 1, padded_n, pad_val)
      return jnp.moveaxis(padded.reshape(batch, n_w, c_w, *arr.shape[2:]), 1, 0)

    w_starts_b = _batch_aligned_full(q_attn, (batch, n), 0, jnp.int32) + _replicate_axis_if_explicit(win.starts, 1)
    w_valid_b = _batch_aligned_full(q_attn, (batch, n), False, bool) | _replicate_axis_if_explicit(win.valid, 1)
    w_idx_doc_b = _batch_aligned_full(q_attn, (batch, n), 0, jnp.int32) + _replicate_axis_if_explicit(win.index_in_doc, 1)
    w_seg_b = _batch_aligned_full(q_attn, (batch, n), 0, jnp.int32) + _replicate_axis_if_explicit(win.segment_ids, 1)
    w_dstart_b = _batch_aligned_full(q_attn, (batch, n), 0, jnp.int32) + _replicate_axis_if_explicit(win.doc_start, 1)

    q_xs = (
        chunk_q(q_idx_rep),
        chunk_q(w_idx_rep),
        chunk_q(q_attn),
        chunk_q(seg_rep, 0),
        chunk_q(pos_rep, -1),
        chunk_q(dstart_rep, 0),
    )
    w_xs = (
        chunk_w(idx_keys_rep),
        chunk_w(comp_kv),
        chunk_w(w_starts_b, 0),
        chunk_w(w_valid_b, False),
        chunk_w(w_idx_doc_b, seq + 1),
        chunk_w(w_seg_b, 0),
        chunk_w(w_dstart_b, 0),
    )

    @jax.checkpoint
    def kl_for_query_chunk(
        q_ic: jax.Array,
        w_ic: jax.Array,
        q_ac: jax.Array,
        seg_q: jax.Array,
        pos_q: jax.Array,
        start_q: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      docs_c = _Documents(seg_q, pos_q, start_q, docs.single_document)
      if n <= c_w:
        win_all = _Windows(w_starts_b, w_valid_b, w_idx_doc_b, w_seg_b, w_dstart_b, win.rate, win.single_document)
        vis_c = _visible_windows(docs_c, win_all)
        scores_c = _score_tile(
            q_ic, w_ic, idx_keys_rep, head_scale=head_scale, head_chunk_size=self.head_chunk_size
        ).astype(jnp.float32)
        return _kl_chunk_dense_windows(scores_c, q_ac, comp_kv, vis_c, scale=scale, dot_prec=dot_prec)

      def eval_w_tile(w_item: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
        idx_kw, attn_kw, ws, wv, wid, wsg, wds = w_item
        win_w = _Windows(ws, wv, wid, wsg, wds, win.rate, win.single_document)
        vis_w = _visible_windows(docs_c, win_w)
        scores_w = _score_tile(q_ic, w_ic, idx_kw, head_scale=head_scale, head_chunk_size=self.head_chunk_size).astype(
            jnp.float32
        )
        return scores_w, attn_kw, vis_w

      return _kl_chunk_tiled_windows(eval_w_tile, w_xs, q_ac, scale=scale, dot_prec=dot_prec)

    if n_q == 1:
      kl_sum, cnt_sum = kl_for_query_chunk(*(arr[0] for arr in q_xs))
    else:

      def q_step(
          carry: tuple[jax.Array, jax.Array], q_item: tuple[jax.Array, ...]
      ) -> tuple[tuple[jax.Array, jax.Array], None]:
        kl_acc, cnt_acc = carry
        dk, dc = kl_for_query_chunk(*q_item)
        return (kl_acc + dk, cnt_acc + dc), None

      z0 = jnp.zeros((), dtype=jnp.float32)
      (kl_sum, cnt_sum), _ = jax.lax.scan(q_step, (z0, z0), q_xs)
    return kl_sum / jnp.maximum(cnt_sum, 1.0)


def _kl_chunk_dense_windows(
    scores_c: jax.Array,
    q_c: jax.Array,
    compressed_kv: jax.Array,
    vis_c: jax.Array,
    *,
    scale: float,
    dot_prec: Any,
) -> tuple[jax.Array, jax.Array]:
  """Computes `(kl_sum, valid_query_count)` for one query tile `[B, c_q, ...]` when `n <= window_chunk_size`."""
  has_target = jnp.any(vis_c, axis=-1)
  logits = jnp.einsum("bchd,btd->bhct", q_c, compressed_kv, precision=dot_prec) * scale
  logits = _masked_logits(logits, vis_c & has_target[:, :, None], -1e30)
  target = jnp.sum(jax.nn.softmax(logits, axis=-1), axis=1)
  target = jnp.where(vis_c, target, 0.0)
  target = target / jnp.maximum(jnp.sum(target, axis=-1, keepdims=True), 1e-30)
  log_student = jax.nn.log_softmax(jnp.where(vis_c, scores_c, -1e30), axis=-1)
  kl = jnp.where(target > 0, target * (jnp.log(jnp.where(target > 0, target, 1.0)) - log_student), 0.0)
  per_query = jnp.where(has_target, jnp.sum(kl, axis=-1), 0.0)
  return jnp.sum(per_query), jnp.sum(has_target.astype(jnp.float32))


def _kl_chunk_tiled_windows(
    eval_w_tile: Any,
    w_xs: tuple[jax.Array, ...],
    q_c: jax.Array,
    *,
    scale: float,
    dot_prec: Any,
) -> tuple[jax.Array, jax.Array]:
  """Two-pass online-softmax KL over window tiles so neither `[B, H, c_q, n]` nor `[B, c_q, n]` is materialized."""
  batch, c_q, num_heads, _ = q_c.shape

  @jax.checkpoint
  def pass1_step(
      carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
      w_item: tuple[jax.Array, ...],
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
    m_t, l_t, m_s, l_s, any_vis = carry
    scores_w, attn_kw, vis_w = eval_w_tile(w_item)
    any_vis_new = any_vis | jnp.any(vis_w, axis=-1)

    logits_w = jax.lax.stop_gradient(jnp.einsum("bchd,bwd->bhcw", q_c, attn_kw, precision=dot_prec) * scale)
    logits_w = _masked_logits(logits_w, vis_w, -1e30)
    m_tw = jnp.max(logits_w, axis=-1)
    m_t_new = jnp.maximum(m_t, m_tw)
    l_t_new = l_t * jnp.exp(m_t - m_t_new) + jnp.sum(jnp.exp(logits_w - m_t_new[..., None]), axis=-1)

    s_masked = jnp.where(vis_w, scores_w, -1e30)
    m_sw = jnp.max(s_masked, axis=-1)
    m_s_new = jnp.maximum(m_s, m_sw)
    l_s_new = l_s * jnp.exp(m_s - m_s_new) + jnp.sum(jnp.exp(s_masked - m_s_new[..., None]), axis=-1)
    return (
        jax.lax.stop_gradient(m_t_new),
        jax.lax.stop_gradient(l_t_new),
        m_s_new,
        l_s_new,
        jax.lax.stop_gradient(any_vis_new),
    ), None

  m_t0 = _batch_aligned_full(q_c, (batch, num_heads, c_q), -1e30, jnp.float32)
  l_t0 = _batch_aligned_full(q_c, (batch, num_heads, c_q), 0.0, jnp.float32)
  m_s0 = _batch_aligned_full(q_c, (batch, c_q), -1e30, jnp.float32)
  l_s0 = _batch_aligned_full(q_c, (batch, c_q), 0.0, jnp.float32)
  vis0 = _batch_aligned_full(q_c, (batch, c_q), False, bool)
  (m_t, l_t, m_s, l_s, has_target), _ = jax.lax.scan(pass1_step, (m_t0, l_t0, m_s0, l_s0, vis0), w_xs)
  lse_s = m_s + jnp.log(jnp.maximum(l_s, 1e-30))

  @jax.checkpoint
  def pass2_step(
      carry: tuple[jax.Array, jax.Array, jax.Array],
      w_item: tuple[jax.Array, ...],
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
    z_acc, u_logu_acc, u_s_acc = carry
    scores_w, attn_kw, vis_w = eval_w_tile(w_item)
    active_w = vis_w & has_target[:, :, None]
    logits_w = jax.lax.stop_gradient(jnp.einsum("bchd,bwd->bhcw", q_c, attn_kw, precision=dot_prec) * scale)
    logits_w = _masked_logits(logits_w, active_w, -1e30)
    probs_w = jnp.exp(logits_w - m_t[..., None]) / jnp.maximum(l_t[..., None], 1e-30)
    u_w = jax.lax.stop_gradient(jnp.where(active_w, jnp.sum(probs_w, axis=1), 0.0))
    u_logu = jnp.where(u_w > 0, u_w * jnp.log(jnp.where(u_w > 0, u_w, 1.0)), 0.0)
    u_s = u_w * jnp.where(active_w, scores_w, 0.0)
    return (z_acc + jnp.sum(u_w, axis=-1), u_logu_acc + jnp.sum(u_logu, axis=-1), u_s_acc + jnp.sum(u_s, axis=-1)), None

  acc0 = _batch_aligned_full(q_c, (batch, c_q), 0.0, jnp.float32)
  (z_sum, u_logu_sum, u_s_sum), _ = jax.lax.scan(pass2_step, (acc0, acc0, acc0), w_xs)
  z_safe = jnp.maximum(z_sum, 1e-30)
  kl_q = (u_logu_sum - u_s_sum) / z_safe - jnp.log(z_safe) + (z_sum / z_safe) * lse_s
  per_query = jnp.where(has_target, kl_q, 0.0)
  return jnp.sum(per_query), jnp.sum(has_target.astype(jnp.float32))


def csa_indexer_kl_loss(
    index_scores: jax.Array,
    query: jax.Array,
    compressed_kv: jax.Array,
    visible: jax.Array,
    *,
    scale: float,
    query_chunk_size: int = 128,
    window_chunk_size: int = 512,
) -> jax.Array:
  """KL(attention target || indexer) over compressed windows, averaged over queries that see at least one window.

  Tiled over bounded query and window chunks with checkpointed `jax.lax.scan` loops so neither the 4D `[B, H, S, n]`
  teacher logits nor `[B, S, n]` target/student probability matrices are materialized in HBM.

  Args:
    index_scores: [B, S, n] float32 indexer scores (differentiable w.r.t. the indexer parameters).
    query: [B, S, H, D] attention queries after RoPE.
    compressed_kv: [B, n, D] compressed entries.
    visible: [B, S, n] causal and same-document visibility.
    scale: Attention logit scale.
    query_chunk_size: Query-sequence tile size (`>= 1`).
    window_chunk_size: Compressed-window tile size (`>= 1`).

  Returns:
    Scalar float32 loss.
  """
  batch, seq, n = index_scores.shape
  if seq == 0 or n == 0:
    return jnp.zeros((), dtype=jnp.float32)
  dot_prec = jax.lax.Precision.DEFAULT if query.dtype == jnp.bfloat16 else _HIGHEST
  q_f32 = _replicate_axis_if_explicit(_replicate_axis_if_explicit(jax.lax.stop_gradient(query), 1), 2).astype(jnp.float32)
  kv_f32 = _replicate_axis_if_explicit(jax.lax.stop_gradient(compressed_kv), 1).astype(jnp.float32)
  scores_f32 = _replicate_axis_if_explicit(_replicate_axis_if_explicit(index_scores, 1), 2).astype(jnp.float32)
  vis_aligned = _batch_aligned_full(q_f32, (batch, seq, n), False, bool) | _replicate_axis_if_explicit(
      _replicate_axis_if_explicit(visible, 1), 2
  )

  c_q = min(seq, query_chunk_size)
  c_w = min(n, window_chunk_size)
  n_q = (seq + c_q - 1) // c_q
  n_w = (n + c_w - 1) // c_w
  padded_seq = n_q * c_q
  padded_n = n_w * c_w

  @jax.checkpoint
  def eval_q_chunk(s_c: jax.Array, q_c: jax.Array, v_c: jax.Array) -> tuple[jax.Array, jax.Array]:
    if n <= c_w:
      return _kl_chunk_dense_windows(s_c, q_c, kv_f32, v_c, scale=scale, dot_prec=dot_prec)
    s_w_xs = jnp.moveaxis(_pad_axis(s_c, 2, padded_n, -1e30).reshape(batch, c_q, n_w, c_w), 2, 0)
    v_w_xs = jnp.moveaxis(_pad_axis(v_c, 2, padded_n, False).reshape(batch, c_q, n_w, c_w), 2, 0)
    k_w_xs = jnp.moveaxis(_pad_axis(kv_f32, 1, padded_n, 0.0).reshape(batch, n_w, c_w, kv_f32.shape[-1]), 1, 0)
    return _kl_chunk_tiled_windows(lambda item: item, (s_w_xs, k_w_xs, v_w_xs), q_c, scale=scale, dot_prec=dot_prec)

  if n_q == 1:
    kl_sum, cnt_sum = eval_q_chunk(scores_f32, q_f32, vis_aligned)
  else:
    s_chunks = jnp.moveaxis(_pad_axis(scores_f32, 1, padded_seq, -1e30).reshape(batch, n_q, c_q, n), 1, 0)
    q_chunks = jnp.moveaxis(_pad_axis(q_f32, 1, padded_seq, 0.0).reshape(batch, n_q, c_q, *q_f32.shape[2:]), 1, 0)
    v_chunks = jnp.moveaxis(_pad_axis(vis_aligned, 1, padded_seq, False).reshape(batch, n_q, c_q, n), 1, 0)

    def q_step(
        carry: tuple[jax.Array, jax.Array], item: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
      kl_acc, cnt_acc = carry
      dk, dc = eval_q_chunk(*item)
      return (kl_acc + dk, cnt_acc + dc), None

    z0 = jnp.zeros((), dtype=jnp.float32)
    (kl_sum, cnt_sum), _ = jax.lax.scan(q_step, (z0, z0), (s_chunks, q_chunks, v_chunks))
  return kl_sum / jnp.maximum(cnt_sum, 1.0)


class DeepseekV4Compressor(nnx.Module):
  """HCA or CSA compressor: compressed entries [B, n, head_dim] plus their per-query visibility [B, S, n]."""

  def __init__(
      self, cfg: DeepseekV4Config, layer_type: str, *, sharding: DeepseekV4Sharding | None, rngs: nnx.Rngs, name: str
  ):
    self.rate = cfg.compress_rate(layer_type)
    self.sparse = layer_type == COMPRESSED_SPARSE_ATTENTION
    self.retain_scores = cfg.indexer_loss_scaling_factor > 0.0
    self.pool = _WindowCompressor(
        cfg,
        head_dim=cfg.head_dim,
        rate=self.rate,
        overlap=self.sparse,
        sharding=sharding,
        rngs=rngs,
        name=name,
    )
    self.indexer = DeepseekV4Indexer(cfg, sharding=sharding, rngs=rngs, name=f"{name}.indexer") if self.sparse else None

  def __call__(
      self, hidden: jax.Array, q_residual: jax.Array, docs: _Documents, *, return_mask: bool = True
  ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Returns (compressed entries, attendable-entry mask, diagnostics with indexer outputs for CSA)."""
    win = _windows(docs, self.rate)
    compressed = self.pool.compress(hidden, win)
    if self.indexer is None:
      visible = _visible_windows(docs, win) if return_mask else jnp.zeros((*hidden.shape[:2], 0), dtype=bool)
      return compressed, visible, {}
    if not self.retain_scores:
      topk = self.indexer.select_topk(hidden, q_residual, docs, win)
      selected = (
          _topk_to_mask(topk, win.starts.shape[1]) if return_mask else jnp.zeros((*hidden.shape[:2], 0), dtype=bool)
      )
      return compressed, selected, {"topk_indices": topk}
    visible = _visible_windows(docs, win)
    scores = self.indexer.scores(hidden, q_residual, docs, win)
    topk, selected = self.indexer.select(scores, visible)
    if not return_mask:
      selected = jnp.zeros((*hidden.shape[:2], 0), dtype=bool)
    return compressed, selected, {"index_scores": scores, "topk_indices": topk, "visible": visible}


def _attend_chunks(
    q: jax.Array,
    kv: jax.Array,
    sinks: jax.Array,
    docs: _Documents,
    *,
    sliding_window: int,
    scale: float,
    acc: DTypeLike,
    dot_prec: Any,
    query_chunk_size: int,
    window_chunk_size: int,
    compressed: jax.Array | None = None,
    topk_indices: jax.Array | None = None,
    hca_win: _Windows | None = None,
) -> jax.Array:
  """Computes sliding + optional CSA/HCA attention in bounded query/window tiles without `[B, H, S, S]` or `[B, S, S]`."""
  batch, seq, h, hd = q.shape
  if seq == 0:
    return _batch_aligned_full(q, (batch, 0, h, hd), 0, acc)

  q_rep = _replicate_axis_if_explicit(q, 1)
  kv_rep = _replicate_axis_if_explicit(kv, 1)
  seg_rep = _batch_aligned_full(q_rep, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.segment_ids, 1)
  pos_rep = _batch_aligned_full(q_rep, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.positions, 1)
  dstart_rep = _batch_aligned_full(q_rep, (batch, seq), 0, jnp.int32) + _replicate_axis_if_explicit(docs.doc_start, 1)
  comp_rep = _replicate_axis_if_explicit(compressed, 1) if compressed is not None else None
  topk_rep = (
      _batch_aligned_full(q_rep, topk_indices.shape, 0, jnp.int32) + _replicate_axis_if_explicit(topk_indices, 1)
      if topk_indices is not None
      else None
  )

  c_q = min(seq, query_chunk_size)
  n_q = (seq + c_q - 1) // c_q
  padded_seq = n_q * c_q
  win_width = min(seq, c_q + sliding_window - 1)
  sinks_acc = sinks.astype(acc)

  def chunk_q(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
    padded = _pad_axis(arr, 1, padded_seq, pad_val)
    return jnp.moveaxis(padded.reshape(batch, n_q, c_q, *arr.shape[2:]), 1, 0)

  hca_aligned: _Windows | None = None
  if hca_win is not None:
    n_hca = hca_win.starts.shape[1]
    hca_aligned = _Windows(
        _batch_aligned_full(q_rep, (batch, n_hca), 0, jnp.int32) + _replicate_axis_if_explicit(hca_win.starts, 1),
        _batch_aligned_full(q_rep, (batch, n_hca), False, bool) | _replicate_axis_if_explicit(hca_win.valid, 1),
        _batch_aligned_full(q_rep, (batch, n_hca), 0, jnp.int32) + _replicate_axis_if_explicit(hca_win.index_in_doc, 1),
        _batch_aligned_full(q_rep, (batch, n_hca), 0, jnp.int32) + _replicate_axis_if_explicit(hca_win.segment_ids, 1),
        _batch_aligned_full(q_rep, (batch, n_hca), 0, jnp.int32) + _replicate_axis_if_explicit(hca_win.doc_start, 1),
        hca_win.rate,
        hca_win.single_document,
    )

  @jax.checkpoint
  def attend_query_chunk(
      q_start: jax.Array,
      q_c: jax.Array,
      seg_c: jax.Array,
      pos_c: jax.Array,
      dstart_c: jax.Array,
      topk_c: jax.Array | None,
  ) -> jax.Array:
    q_c_acc = q_c.astype(acc)
    k_start = jnp.clip(q_start - (sliding_window - 1), 0, seq - win_width)
    kv_win = jax.lax.dynamic_slice_in_dim(kv_rep, k_start, win_width, axis=1)
    seg_win = jax.lax.dynamic_slice_in_dim(seg_rep, k_start, win_width, axis=1)
    dstart_win = jax.lax.dynamic_slice_in_dim(dstart_rep, k_start, win_width, axis=1)

    q_idx = (q_start + jnp.arange(c_q, dtype=jnp.int32))[:, None]
    k_idx = (k_start + jnp.arange(win_width, dtype=jnp.int32))[None, :]
    in_win = (k_idx <= q_idx) & (q_idx - k_idx < sliding_window) & (q_idx < seq)
    local_c = (
        in_win[None]
        & (seg_c[:, :, None] == seg_win[:, None, :])
        & (dstart_c[:, :, None] == dstart_win[:, None, :])
        & (seg_c != 0)[:, :, None]
    )
    local_logits = jnp.einsum("bchd,bwd->bhcw", q_c_acc, kv_win.astype(acc), precision=dot_prec) * scale
    local_logits = _masked_logits(local_logits, local_c)
    sinks_c = _match_bh_sharding(local_logits, jnp.broadcast_to(sinks_acc[None, :, None, None], (batch, h, c_q, 1)))

    # 1. Sliding-only (or zero compressed windows).
    if comp_rep is None:
      probs = jax.nn.softmax(jnp.concatenate([local_logits, sinks_c], axis=-1), axis=-1)
      p_local = probs[..., :win_width].astype(kv.dtype).astype(acc)
      return jnp.einsum("bhcw,bwd->bchd", p_local, kv_win.astype(acc), precision=dot_prec)

    # 2. Compressed Sparse Attention: gather only the `k` selected compressed entries per query.
    if topk_c is not None:
      k = topk_c.shape[-1]
      if k == 0:
        probs = jax.nn.softmax(jnp.concatenate([local_logits, sinks_c], axis=-1), axis=-1)
        p_local = probs[..., :win_width].astype(kv.dtype).astype(acc)
        return jnp.einsum("bhcw,bwd->bchd", p_local, kv_win.astype(acc), precision=dot_prec)
      valid_k = (topk_c >= 0) & (seg_c != 0)[:, :, None]
      safe_idx = jnp.where(valid_k, topk_c, 0)
      comp_k = jnp.take_along_axis(comp_rep, safe_idx.reshape(batch, c_q * k, 1), axis=1).reshape(batch, c_q, k, hd)
      comp_logits = jnp.einsum("bchd,bckd->bhck", q_c_acc, comp_k.astype(acc), precision=dot_prec) * scale
      comp_logits = _masked_logits(comp_logits, valid_k)
      probs = jax.nn.softmax(jnp.concatenate([local_logits, comp_logits, sinks_c], axis=-1), axis=-1)
      p_local = probs[..., :win_width].astype(kv.dtype).astype(acc)
      p_comp = probs[..., win_width : win_width + k].astype(comp_rep.dtype).astype(acc)
      return jnp.einsum("bhcw,bwd->bchd", p_local, kv_win.astype(acc), precision=dot_prec) + jnp.einsum(
          "bhck,bckd->bchd", p_comp, comp_k.astype(acc), precision=dot_prec
      )

    # 3. Heavily Compressed Attention: procedural per-chunk visibility, with 2-pass exact-dtype window tiling if needed.
    assert hca_aligned is not None
    docs_c = _Documents(seg_c, pos_c, dstart_c, docs.single_document)
    n_hca = comp_rep.shape[1]
    c_w = min(n_hca, window_chunk_size)
    if n_hca <= c_w:
      vis_hca = _visible_windows(docs_c, hca_aligned) & (seg_c != 0)[:, :, None]
      comp_logits = jnp.einsum("bchd,btd->bhct", q_c_acc, comp_rep.astype(acc), precision=dot_prec) * scale
      comp_logits = _masked_logits(comp_logits, vis_hca)
      probs = jax.nn.softmax(jnp.concatenate([local_logits, comp_logits, sinks_c], axis=-1), axis=-1)
      p_local = probs[..., :win_width].astype(kv.dtype).astype(acc)
      p_comp = probs[..., win_width : win_width + n_hca].astype(comp_rep.dtype).astype(acc)
      return jnp.einsum("bhcw,bwd->bchd", p_local, kv_win.astype(acc), precision=dot_prec) + jnp.einsum(
          "bhct,btd->bchd", p_comp, comp_rep.astype(acc), precision=dot_prec
      )

    n_w = (n_hca + c_w - 1) // c_w
    padded_n = n_w * c_w

    def chunk_hca_w(arr: jax.Array, pad_val: Any = 0) -> jax.Array:
      padded = _pad_axis(arr, 1, padded_n, pad_val)
      return jnp.moveaxis(padded.reshape(batch, n_w, c_w, *arr.shape[2:]), 1, 0)

    hca_w_xs = (
        chunk_hca_w(comp_rep),
        chunk_hca_w(hca_aligned.starts, 0),
        chunk_hca_w(hca_aligned.valid, False),
        chunk_hca_w(hca_aligned.index_in_doc, seq + 1),
        chunk_hca_w(hca_aligned.segment_ids, 0),
        chunk_hca_w(hca_aligned.doc_start, 0),
    )
    base_logits = jnp.concatenate([local_logits, sinks_c], axis=-1)
    m0 = jnp.max(base_logits, axis=-1)
    l0 = jnp.sum(jnp.exp(base_logits - m0[..., None]), axis=-1)

    @jax.checkpoint
    def hca_pass1(
        carry: tuple[jax.Array, jax.Array], w_item: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
      m_cur, l_cur = carry
      comp_w, ws, wv, wid, wsg, wds = w_item
      win_w = _Windows(ws, wv, wid, wsg, wds, hca_aligned.rate, hca_aligned.single_document)
      vis_w = _visible_windows(docs_c, win_w) & (seg_c != 0)[:, :, None]
      logits_w = _masked_logits(
          jnp.einsum("bchd,bwd->bhcw", q_c_acc, comp_w.astype(acc), precision=dot_prec) * scale, vis_w
      )
      m_new = jnp.maximum(m_cur, jnp.max(logits_w, axis=-1))
      l_new = l_cur * jnp.exp(m_cur - m_new) + jnp.sum(jnp.exp(logits_w - m_new[..., None]), axis=-1)
      return (m_new, l_new), None

    (m_tot, l_tot), _ = jax.lax.scan(hca_pass1, (m0, l0), hca_w_xs)
    p_local = (jnp.exp(local_logits - m_tot[..., None]) / l_tot[..., None]).astype(kv.dtype).astype(acc)
    out_init = jnp.einsum("bhcw,bwd->bchd", p_local, kv_win.astype(acc), precision=dot_prec)

    @jax.checkpoint
    def hca_pass2(out_cur: jax.Array, w_item: tuple[jax.Array, ...]) -> tuple[jax.Array, None]:
      comp_w, ws, wv, wid, wsg, wds = w_item
      win_w = _Windows(ws, wv, wid, wsg, wds, hca_aligned.rate, hca_aligned.single_document)
      vis_w = _visible_windows(docs_c, win_w) & (seg_c != 0)[:, :, None]
      logits_w = _masked_logits(
          jnp.einsum("bchd,bwd->bhcw", q_c_acc, comp_w.astype(acc), precision=dot_prec) * scale, vis_w
      )
      p_w = (jnp.exp(logits_w - m_tot[..., None]) / l_tot[..., None]).astype(comp_rep.dtype).astype(acc)
      return out_cur + jnp.einsum("bhcw,bwd->bchd", p_w, comp_w.astype(acc), precision=dot_prec), None

    out_c, _ = jax.lax.scan(hca_pass2, out_init, hca_w_xs)
    return out_c

  if n_q == 1:
    return attend_query_chunk(jnp.int32(0), q_rep, seg_rep, pos_rep, dstart_rep, topk_rep)

  q_starts = jnp.arange(n_q, dtype=jnp.int32) * c_q
  q_chunks = chunk_q(q_rep)
  seg_chunks = chunk_q(seg_rep, 0)
  pos_chunks = chunk_q(pos_rep, -1)
  dstart_chunks = chunk_q(dstart_rep, 0)
  if topk_rep is not None:
    topk_chunks = chunk_q(topk_rep, -1)

    def step_csa(_: None, item: tuple[jax.Array, ...]) -> tuple[None, jax.Array]:
      qs, qc, sc, pc, dc, tkc = item
      return None, attend_query_chunk(qs, qc, sc, pc, dc, tkc)

    _, out_chunks = jax.lax.scan(step_csa, None, (q_starts, q_chunks, seg_chunks, pos_chunks, dstart_chunks, topk_chunks))
  else:

    def step_dense(_: None, item: tuple[jax.Array, ...]) -> tuple[None, jax.Array]:
      qs, qc, sc, pc, dc = item
      return None, attend_query_chunk(qs, qc, sc, pc, dc, None)

    _, out_chunks = jax.lax.scan(step_dense, None, (q_starts, q_chunks, seg_chunks, pos_chunks, dstart_chunks))

  return jnp.moveaxis(out_chunks, 0, 1).reshape(batch, padded_seq, h, hd)[:, :seq]


class DeepseekV4Attention(nnx.Module):
  """Shared-KV multi-query attention with sinks, optional compressed entries, and grouped output projection."""

  def __init__(
      self,
      cfg: DeepseekV4Config,
      layer_idx: int,
      *,
      sharding: DeepseekV4Sharding | None,
      rngs: nnx.Rngs,
      name: str = "self_attn",
  ):
    if rngs is None:
      raise ValueError("rngs must be provided to DeepseekV4Attention")
    self.cfg = cfg
    self.sharding = sharding
    self.layer_type = cfg.layer_types[layer_idx]
    rope_dtype = np.float64 if cfg.compute_dtype == jnp.float64 else np.float32
    self.inv_freq = tuple(
        float(v)
        for v in rope_inv_freq(cfg, "main" if self.layer_type == SLIDING_ATTENTION else "compress", dtype=rope_dtype)
    )
    h, hd = cfg.num_attention_heads, cfg.head_dim
    self.scale = hd**-0.5
    pd = cfg.param_dtype
    self.q_a_proj = _linear(cfg, sharding, rngs, f"{name}.q_a_proj", cfg.hidden_size, cfg.q_lora_rank, ("embed", None))
    self.q_a_norm = DeepseekV4RMSNorm(
        cfg.q_lora_rank, eps=cfg.rms_norm_eps, param_dtype=pd, sharding=sharding, name=f"{name}.q_a_norm"
    )
    self.q_b_proj = _linear(cfg, sharding, rngs, f"{name}.q_b_proj", cfg.q_lora_rank, h * hd, ("q_lora", "heads"))
    self.kv_proj = _linear(cfg, sharding, rngs, f"{name}.kv_proj", cfg.hidden_size, hd, ("embed", None))
    self.kv_norm = DeepseekV4RMSNorm(hd, eps=cfg.rms_norm_eps, param_dtype=pd, sharding=sharding, name=f"{name}.kv_norm")
    self.o_a_proj = DeepseekV4GroupedLinear(
        cfg, h * hd // cfg.o_groups, cfg.o_lora_rank, cfg.o_groups, sharding=sharding, rngs=rngs, name=f"{name}.o_a_proj"
    )
    self.o_b_proj = _linear(
        cfg, sharding, rngs, f"{name}.o_b_proj", cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, ("o_mix", "embed")
    )
    sink_dtype = jnp.promote_types(pd, jnp.float32)
    self.sinks = nnx.Param(jnp.zeros((h,), sink_dtype), **_param_metadata(sharding, f"{name}.sinks", ("heads",), (h,)))
    self.compressor = (
        None
        if self.layer_type == SLIDING_ATTENTION
        else DeepseekV4Compressor(cfg, self.layer_type, sharding=sharding, rngs=rngs, name=f"{name}.compressor")
    )

  def __call__(self, hidden: jax.Array, segment_ids: jax.Array | None = None) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Attends over the local window and, for compressed layers, the selected compressed entries.

    Args:
      hidden: [B, S, D] sublayer input.
      segment_ids: Optional [B, S] document ids for packed sequences (0 marks padding). Without them each row is one
        unpadded document.

    Returns:
      (output [B, S, D], auxiliary losses). The only auxiliary loss is "csa_indexer_loss", returned by compressed
      sparse layers when `indexer_loss_scaling_factor > 0` (already multiplied by the factor).
    """
    cfg = self.cfg
    hidden = _constrain(self.sharding, hidden, "attn_in", ("batch", "length", "embed"))
    if segment_ids is not None:
      segment_ids = _constrain(self.sharding, segment_ids, "segment_ids", ("batch", "length"))
    batch, seq, _ = hidden.shape
    h, hd = cfg.num_attention_heads, cfg.head_dim
    acc = jnp.promote_types(hidden.dtype, jnp.float32)
    docs = _documents(segment_ids, batch, seq)
    cos, sin = rope_cos_sin(self.inv_freq, docs.positions, hidden.dtype)

    q_residual = self.q_a_norm(self.q_a_proj(hidden))
    q = unweighted_rms_norm(self.q_b_proj(q_residual).reshape(batch, seq, h, hd), cfg.rms_norm_eps)
    q = checkpoint_name(
        _constrain(self.sharding, apply_rotary(q, cos, sin), "attn_q", ("batch", "length", "heads", "head_dim")),
        "query_proj",
    )
    kv = checkpoint_name(
        _constrain(
            self.sharding,
            apply_rotary(self.kv_norm(self.kv_proj(hidden)), cos, sin),
            "attn_kv",
            ("batch", "length", "head_dim"),
        ),
        "kv_proj",
    )

    aux: dict[str, jax.Array] = {}
    if self.compressor is not None and self.compressor.sparse and cfg.indexer_loss_scaling_factor > 0.0:
      aux["csa_indexer_loss"] = jnp.zeros((), dtype=jnp.float32)

    compressed: jax.Array | None = None
    topk_indices: jax.Array | None = None
    hca_win: _Windows | None = None
    if self.compressor is not None and seq // self.compressor.rate > 0:
      win = _windows(docs, self.compressor.rate)
      compressed = self.compressor.pool.compress(hidden, win)
      if self.compressor.indexer is not None:
        topk_indices = self.compressor.indexer.select_topk(hidden, q_residual, docs, win)
        if cfg.indexer_loss_scaling_factor > 0.0:
          aux["csa_indexer_loss"] = cfg.indexer_loss_scaling_factor * self.compressor.indexer.kl_loss(
              hidden, q_residual, q, compressed, docs, win, scale=self.scale
          )
      else:
        hca_win = win

    dot_prec = jax.lax.Precision.DEFAULT if hidden.dtype == jnp.bfloat16 else _HIGHEST
    out = _attend_chunks(
        q,
        kv,
        self.sinks[...],
        docs,
        sliding_window=cfg.sliding_window,
        scale=self.scale,
        acc=acc,
        dot_prec=dot_prec,
        query_chunk_size=cfg.attn_query_chunk_size,
        window_chunk_size=cfg.index_window_chunk_size,
        compressed=compressed,
        topk_indices=topk_indices,
        hca_win=hca_win,
    )
    out = checkpoint_name(apply_rotary(out.astype(hidden.dtype), cos, -sin), "context")
    g_shape = (batch, seq, cfg.o_groups, h * hd // cfg.o_groups)
    if self.sharding is not None and _is_explicit_mesh():
      g_spec = self.sharding.activation_spec("attn_grouped_in", ("batch", "length", "o_groups", None), g_shape)
      grouped = self.o_a_proj(jax.lax.reshape(out, g_shape, out_sharding=g_spec))
      out_spec = self.sharding.activation_spec("attn_out", ("batch", "length", "embed"), (batch, seq, cfg.hidden_size))
      raw_out = self.o_b_proj(grouped.reshape(batch, seq, cfg.o_groups * cfg.o_lora_rank), out_sharding=out_spec)
    else:
      grouped = self.o_a_proj(out.reshape(*g_shape))
      raw_out = self.o_b_proj(grouped.reshape(batch, seq, cfg.o_groups * cfg.o_lora_rank))
    output = checkpoint_name(raw_out, "out_proj")
    return _constrain(self.sharding, output, "attn_out", ("batch", "length", "embed")), aux
