# Copyright 2026 Google LLC
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

"""GraniteMoe (Granite 3.0 MoE) pure Flax NNX implementation for MaxText m3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import json
import os
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.m3.core import rope as m3_rope
from maxtext.m3.core import sharding as m3_sharding
import numpy as np

# ==============================================================================
# Section 1: Configuration & 4-Axis Logical Sharding
# ==============================================================================

_GRANITEMOE_ONLY_DEFAULTS: dict[str, Any] = {
    "embedding_multiplier": 12.0,
    "attention_multiplier": 0.015625,
    "residual_multiplier": 0.22,
    "logits_scaling": 6.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "matmul_precision": "default",
}


def _resolve_jnp_dtype(dt: Any) -> jnp.dtype:
  """Resolves a string, NumPy dtype, or JAX dtype into a `jnp.dtype`."""
  if isinstance(dt, jnp.dtype):
    return dt
  if isinstance(dt, np.dtype):
    return jnp.dtype(dt)
  if isinstance(dt, str):
    if hasattr(jnp, dt):
      return jnp.dtype(getattr(jnp, dt))
    return jnp.dtype(dt)
  return jnp.dtype(dt)


def _resolve_precision(precision_name: str) -> jax.lax.Precision:
  """Maps config precision string to `jax.lax.Precision`."""
  match precision_name:
    case "highest" | "float32":
      return jax.lax.Precision.HIGHEST
    case "high":
      return jax.lax.Precision.HIGH
    case "default" | "bfloat16" | "":
      return jax.lax.Precision.DEFAULT
    case _:
      raise ValueError(f"Unsupported matmul_precision {precision_name!r}.")


@dataclasses.dataclass(frozen=True)
class GraniteMoeConfig:
  """Configuration for GraniteMoe (`ibm-granite/granite-3.0-1b-a400m-instruct` defaults)."""

  vocab_size: int = 49155
  hidden_size: int = 1024
  intermediate_size: int = 512
  num_hidden_layers: int = 24
  num_attention_heads: int = 16
  num_key_value_heads: int = 8
  num_local_experts: int = 32
  num_experts_per_tok: int = 8
  rms_norm_eps: float = 1e-6
  rope_theta: float = 10000.0
  attention_bias: bool = False
  attention_dropout: float = 0.0
  hidden_act: str = "silu"
  tie_word_embeddings: bool = True
  embedding_multiplier: float = 12.0
  attention_multiplier: float = 0.015625
  residual_multiplier: float = 0.22
  logits_scaling: float = 6.0
  router_aux_loss_coef: float = 0.0
  dtype: str = "float32"
  weight_dtype: str = "float32"
  scan_layers: bool = True
  remat_policy: str = "none"
  matmul_precision: str = "default"

  def __post_init__(self) -> None:
    if self.hidden_size <= 0 or self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          f"hidden_size ({self.hidden_size}) must be positive and divisible by "
          f"num_attention_heads ({self.num_attention_heads})."
      )
    if (self.hidden_size // self.num_attention_heads) % 2 != 0:
      raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE.")
    if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if not 1 <= self.num_experts_per_tok <= self.num_local_experts:
      raise ValueError(f"num_experts_per_tok ({self.num_experts_per_tok}) must be in [1, {self.num_local_experts}].")
    if self.logits_scaling == 0.0:
      raise ValueError("logits_scaling must be non-zero.")
    if not 0.0 <= self.attention_dropout < 1.0:
      raise ValueError(f"attention_dropout ({self.attention_dropout}) must be in [0.0, 1.0).")
    if self.hidden_act != "silu":
      raise ValueError(f"GraniteMoe supports hidden_act='silu', got {self.hidden_act!r}.")
    if self.remat_policy not in ("none", "full", "minimal"):
      raise ValueError(f"Unsupported remat_policy {self.remat_policy!r}; expected 'none', 'full', or 'minimal'.")

  @property
  def head_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads

  @classmethod
  def from_config(cls, config: Any, **overrides: Any) -> GraniteMoeConfig:
    """Builds `GraniteMoeConfig` from `GraniteMoeConfig`, M3 `BaseConfig`, `MaxTextConfig`, HF config, or dict."""
    if isinstance(config, cls):
      return dataclasses.replace(config, **overrides) if overrides else config

    if hasattr(config, "use_m3_model") and not getattr(config, "use_m3_model"):
      raise ValueError("GraniteMoe requires use_m3_model=True.")

    def _get(primary: str, *fallbacks: str, default: Any = dataclasses.MISSING) -> Any:
      for key in (primary, *fallbacks):
        if isinstance(config, Mapping) and key in config and config[key] is not None:
          return config[key]
        if not isinstance(config, Mapping) and hasattr(config, key):
          val = getattr(config, key)
          if val is not None:
            return val
      if default is not dataclasses.MISSING:
        return default
      raise KeyError(f"Missing required GraniteMoe config field {primary!r} (checked {(primary, *fallbacks)}).")

    dtype_val = _get("dtype", "activations_dtype", default="float32")
    weight_dtype_val = _get("weight_dtype", "weights_dtype", default="float32")
    remat_val = _get("remat_policy", default="none")
    if remat_val is None or remat_val is False:
      remat_val = "none"
    elif remat_val is True:
      remat_val = "full"
    else:
      remat_val = str(getattr(remat_val, "value", remat_val)).lower()
      if remat_val not in ("none", "full", "minimal"):
        remat_val = "minimal"

    mp_val = _get("matmul_precision", default=_GRANITEMOE_ONLY_DEFAULTS["matmul_precision"])
    mp_str = str(getattr(mp_val, "value", mp_val)).lower()

    kwargs: dict[str, Any] = {
        "vocab_size": int(_get("vocab_size", default=49155)),
        "hidden_size": int(_get("hidden_size", "emb_dim", "base_emb_dim", default=1024)),
        "intermediate_size": int(
            _get("intermediate_size", "moe_mlp_dim", "base_moe_mlp_dim", "mlp_dim", "base_mlp_dim", default=512)
        ),
        "num_hidden_layers": int(_get("num_hidden_layers", "num_decoder_layers", "base_num_decoder_layers", default=24)),
        "num_attention_heads": int(_get("num_attention_heads", "num_query_heads", "base_num_query_heads", default=16)),
        "num_key_value_heads": int(_get("num_key_value_heads", "num_kv_heads", "base_num_kv_heads", default=8)),
        "num_local_experts": int(_get("num_local_experts", "num_experts", default=32)),
        "num_experts_per_tok": int(_get("num_experts_per_tok", default=8)),
        "rms_norm_eps": float(_get("rms_norm_eps", "normalization_layer_epsilon", default=1e-6)),
        "rope_theta": float(_get("rope_theta", "rope_max_timescale", default=10000.0)),
        "attention_bias": bool(_get("attention_bias", default=_GRANITEMOE_ONLY_DEFAULTS["attention_bias"])),
        "attention_dropout": float(_get("attention_dropout", default=_GRANITEMOE_ONLY_DEFAULTS["attention_dropout"])),
        "hidden_act": str(_get("hidden_act", default=_GRANITEMOE_ONLY_DEFAULTS["hidden_act"])),
        "tie_word_embeddings": bool(_get("tie_word_embeddings", "logits_via_embedding", default=True)),
        "embedding_multiplier": float(
            _get("embedding_multiplier", default=_GRANITEMOE_ONLY_DEFAULTS["embedding_multiplier"])
        ),
        "attention_multiplier": float(
            _get("attention_multiplier", default=_GRANITEMOE_ONLY_DEFAULTS["attention_multiplier"])
        ),
        "residual_multiplier": float(
            _get("residual_multiplier", default=_GRANITEMOE_ONLY_DEFAULTS["residual_multiplier"])
        ),
        "logits_scaling": float(_get("logits_scaling", default=_GRANITEMOE_ONLY_DEFAULTS["logits_scaling"])),
        "router_aux_loss_coef": float(_get("router_aux_loss_coef", "load_balance_loss_weight", default=0.0)),
        "dtype": _resolve_jnp_dtype(dtype_val).name,
        "weight_dtype": _resolve_jnp_dtype(weight_dtype_val).name,
        "scan_layers": bool(_get("scan_layers", default=True)),
        "remat_policy": str(remat_val),
        "matmul_precision": mp_str,
    }
    kwargs.update(overrides)
    return cls(**kwargs)

  from_maxtext_config = from_config

  @classmethod
  def from_hf_config_json(
      cls,
      path_or_dict: str | os.PathLike[str] | Mapping[str, Any],
      **overrides: Any,
  ) -> GraniteMoeConfig:
    """Loads `GraniteMoeConfig` from an HF `config.json` file, directory, or parsed dict."""
    if isinstance(path_or_dict, Mapping):
      raw = dict(path_or_dict)
    else:
      cfg_path = os.fspath(path_or_dict)
      if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
      with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    allowed = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in allowed and v is not None}
    filtered.update(overrides)
    return cls(**filtered)


@dataclasses.dataclass(frozen=True)
class GraniteMoeSharding(m3_sharding.Sharding):
  """4-axis `('data', 'fsdp', 'tensor', 'expert')` logical sharding strategy for GraniteMoe."""

  vocab_size: int = 49155
  num_experts: int = 32
  dp_size: int = 1
  fsdp_size: int = 1
  tp_size: int = 1
  ep_size: int = 1

  @property
  def vocab_tp(self) -> bool:
    return self.tp_size > 1 and self.vocab_size > 0 and (self.vocab_size % self.tp_size == 0)

  @property
  def compound_ep(self) -> bool:
    return self.num_experts > 0 and (self.num_experts % (self.fsdp_size * self.ep_size) == 0)

  @classmethod
  def from_mesh(
      cls,
      mesh: jax.sharding.Mesh,
      config: GraniteMoeConfig | None = None,
  ) -> GraniteMoeSharding:
    """Builds `GraniteMoeSharding` from a driver `Mesh` and optional `GraniteMoeConfig`."""
    required = set(m3_sharding.MESH_AXIS_NAMES)
    axis_names = set(mesh.axis_names)
    if not required.issubset(axis_names):
      raise ValueError(
          f"Unsupported mesh axis_names {mesh.axis_names}; expected {m3_sharding.MESH_AXIS_NAMES}, "
          f"missing {required - axis_names}."
      )
    for ax in mesh.axis_names:
      if ax not in required and mesh.shape[ax] > 1:
        raise NotImplementedError(f"Unsupported active mesh axis {ax!r} of size {mesh.shape[ax]}.")

    dp_size = int(mesh.shape["data"])
    fsdp_size = int(mesh.shape["fsdp"])
    tp_size = int(mesh.shape["tensor"])
    ep_size = int(mesh.shape["expert"])
    vocab_size = config.vocab_size if config is not None else 49155
    num_experts = config.num_local_experts if config is not None else 32

    if config is not None:
      if tp_size > 1:
        if config.num_attention_heads % tp_size != 0:
          raise ValueError(f"num_attention_heads ({config.num_attention_heads}) not divisible by tensor={tp_size}.")
        if config.num_key_value_heads % tp_size != 0:
          raise ValueError(f"num_key_value_heads ({config.num_key_value_heads}) not divisible by tensor={tp_size}.")
        if config.intermediate_size % tp_size != 0:
          raise ValueError(f"intermediate_size ({config.intermediate_size}) not divisible by tensor={tp_size}.")
      if config.hidden_size % (fsdp_size * ep_size) != 0:
        raise ValueError(f"hidden_size ({config.hidden_size}) not divisible by fsdp*expert={fsdp_size * ep_size}.")
      if num_experts % (fsdp_size * ep_size) != 0 and num_experts % ep_size != 0:
        raise ValueError(f"num_local_experts ({num_experts}) not divisible by expert={ep_size}.")

    return cls(
        vocab_size=vocab_size,
        num_experts=num_experts,
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        tp_size=tp_size,
        ep_size=ep_size,
    )

  def map_axis(
      self,
      axis: str,
      tensor_name: str,
      tensor_type: m3_sharding.TensorType,
  ) -> m3_sharding.MeshAxes:
    """Translates logical axis names onto `('data', 'fsdp', 'tensor', 'expert')`."""
    match axis, tensor_type:
      case "batch", m3_sharding.TensorType.ACTIVATION:
        return ("data", "fsdp", "expert")
      case "embed", m3_sharding.TensorType.WEIGHT:
        return ("fsdp", "expert")
      case "embed", m3_sharding.TensorType.ACTIVATION:
        return None
      case "experts", m3_sharding.TensorType.WEIGHT:
        return ("fsdp", "expert") if self.compound_ep else "expert"
      case "embed_exp", m3_sharding.TensorType.WEIGHT:
        return None if self.compound_ep else "fsdp"
      case ("heads" | "kv_heads" | "mlp"), _:
        return "tensor" if self.tp_size > 1 else "tensor"
      case "vocab", _:
        return "tensor" if self.vocab_tp else None
      case ("length" | "head_dim" | "norm"), _:
        return None
      case _:
        raise KeyError(f"No GraniteMoeSharding rule for logical axis {axis!r} ({tensor_type.value}) on {tensor_name!r}.")


def _constrain_activation(
    sharding: GraniteMoeSharding | m3_sharding.Sharding | None,
    x: jax.Array,
    name: str,
    axes: tuple[str | None, ...],
) -> jax.Array:
  """Applies activation sharding constraint, relaxing `'batch'` when batch size < data*fsdp*expert."""
  if sharding is None:
    return x
  resolved_axes = list(axes)
  if isinstance(sharding, GraniteMoeSharding) and "batch" in resolved_axes:
    b_idx = resolved_axes.index("batch")
    batch_shards = sharding.dp_size * sharding.fsdp_size * sharding.ep_size
    if batch_shards > 1 and x.shape[b_idx] % batch_shards != 0:
      resolved_axes[b_idx] = None
  return sharding.constrain(x, name, tuple(resolved_axes))


# ==============================================================================
# Section 2: External KV-Cache & Forward Output Types
# ==============================================================================


class GraniteMoeKVCache(NamedTuple):
  """External immutable KV-cache PyTree of shape `(num_layers, batch, max_seq_len, num_kv_heads, head_dim)`."""

  k: jax.Array
  v: jax.Array


def init_kv_cache(
    config: GraniteMoeConfig,
    batch_size: int,
    max_seq_len: int,
    *,
    dtype: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    sharding: GraniteMoeSharding | None = None,
) -> GraniteMoeKVCache:
  """Allocates an empty zero-initialized external `GraniteMoeKVCache` without mutating model state."""
  dt = _resolve_jnp_dtype(config.dtype if dtype is None else dtype)
  shape = (config.num_hidden_layers, batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
  k = jnp.zeros(shape, dtype=dt)
  v = jnp.zeros(shape, dtype=dt)
  if mesh is not None and sharding is not None:
    batch_axis: Any = ("data", "fsdp", "expert")
    if batch_size % (sharding.dp_size * sharding.fsdp_size * sharding.ep_size) != 0:
      batch_axis = None
    kv_axis = sharding.map_axis("kv_heads", "kv_cache", m3_sharding.TensorType.ACTIVATION)
    spec = jax.sharding.NamedSharding(mesh, P(None, batch_axis, None, kv_axis, None))
    k = jax.device_put(k, spec)
    v = jax.device_put(v, spec)
  return GraniteMoeKVCache(k=k, v=v)


class GraniteMoeForwardOutput(NamedTuple):
  """Detailed output returned by `GraniteMoeModel.forward_with_details`."""

  logits: jax.Array
  aux: dict[str, jax.Array]
  router_logits: jax.Array
  top_k_indices: jax.Array
  kv_cache: GraniteMoeKVCache | None
  hidden_states: jax.Array


# ==============================================================================
# Section 3: RMSNorm & Grouped-Query Attention (with `attention_multiplier`)
# ==============================================================================


class GraniteMoeRMSNorm(nnx.Module):
  """T5-style RMSNorm matching `transformers` `GraniteMoeRMSNorm` (`fp32` variance, scaled by `weight`)."""

  def __init__(
      self,
      hidden_size: int,
      eps: float = 1e-6,
      *,
      param_dtype: Any = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.hidden_size = hidden_size
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding is not None else None
    self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype), out_sharding=spec)

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    input_dtype = hidden_states.dtype
    h_f32 = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(h_f32), axis=-1, keepdims=True)
    normed = (h_f32 * jax.lax.rsqrt(variance + self.eps)).astype(input_dtype)
    return self.weight[...].astype(input_dtype) * normed


class GraniteMoeAttention(nnx.Module):
  """GraniteMoe GQA self-attention using `attention_multiplier` (replaces `1/sqrt(head_dim)`)."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeAttention requires a non-None nnx.Rngs instance.")
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.num_kv_groups = self.num_heads // self.num_kv_heads
    self.head_dim = config.head_dim
    self.rope_theta = config.rope_theta
    self.attention_multiplier = config.attention_multiplier
    self.attention_dropout = float(config.attention_dropout)
    self.precision = _resolve_precision(config.matmul_precision)
    self.sharding = sharding

    act_dt = _resolve_jnp_dtype(config.dtype)
    w_dt = _resolve_jnp_dtype(config.weight_dtype)
    use_bias = config.attention_bias

    q_spec = sharding("q_proj", ("embed", "heads")) if sharding else None
    kv_spec = sharding("kv_proj", ("embed", "kv_heads")) if sharding else None
    o_spec = sharding("o_proj", ("heads", "embed")) if sharding else None
    qb_spec = sharding("q_bias", ("heads",)) if (sharding and use_bias) else None
    kvb_spec = sharding("kv_bias", ("kv_heads",)) if (sharding and use_bias) else None
    ob_spec = sharding("o_bias", ("norm",)) if (sharding and use_bias) else None

    self.q_proj = nnx.Linear(
        self.hidden_size,
        self.num_heads * self.head_dim,
        use_bias=use_bias,
        dtype=act_dt,
        param_dtype=w_dt,
        precision=self.precision,
        kernel_metadata={"out_sharding": q_spec} if q_spec is not None else {},
        bias_metadata={"out_sharding": qb_spec} if qb_spec is not None else {},
        rngs=rngs,
    )
    self.k_proj = nnx.Linear(
        self.hidden_size,
        self.num_kv_heads * self.head_dim,
        use_bias=use_bias,
        dtype=act_dt,
        param_dtype=w_dt,
        precision=self.precision,
        kernel_metadata={"out_sharding": kv_spec} if kv_spec is not None else {},
        bias_metadata={"out_sharding": kvb_spec} if kvb_spec is not None else {},
        rngs=rngs,
    )
    self.v_proj = nnx.Linear(
        self.hidden_size,
        self.num_kv_heads * self.head_dim,
        use_bias=use_bias,
        dtype=act_dt,
        param_dtype=w_dt,
        precision=self.precision,
        kernel_metadata={"out_sharding": kv_spec} if kv_spec is not None else {},
        bias_metadata={"out_sharding": kvb_spec} if kvb_spec is not None else {},
        rngs=rngs,
    )
    self.o_proj = nnx.Linear(
        self.num_heads * self.head_dim,
        self.hidden_size,
        use_bias=use_bias,
        dtype=act_dt,
        param_dtype=w_dt,
        precision=self.precision,
        kernel_metadata={"out_sharding": o_spec} if o_spec is not None else {},
        bias_metadata={"out_sharding": ob_spec} if ob_spec is not None else {},
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      positions: jax.Array,
      *,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      layer_kv_cache: tuple[jax.Array, jax.Array] | None = None,
      cache_position: jax.Array | int | None = None,
      attention_multiplier: float | jax.Array | None = None,
      deterministic: bool = True,
      dropout_rng: jax.Array | None = None,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array] | None]:
    bsz, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_kv_heads, self.head_dim)

    q = _constrain_activation(self.sharding, q, "attn_q", ("batch", "length", "heads", "head_dim"))
    k = _constrain_activation(self.sharding, k, "attn_k", ("batch", "length", "kv_heads", "head_dim"))
    v = _constrain_activation(self.sharding, v, "attn_v", ("batch", "length", "kv_heads", "head_dim"))

    q = m3_rope.apply_rope(q, positions, max_timescale=self.rope_theta)
    k = m3_rope.apply_rope(k, positions, max_timescale=self.rope_theta)

    updated_kv: tuple[jax.Array, jax.Array] | None = None
    if layer_kv_cache is not None:
      k_cache, v_cache = layer_kv_cache
      start_idx = jnp.asarray(0 if cache_position is None else cache_position, dtype=jnp.int32)
      k_cache = jax.lax.dynamic_update_slice(k_cache, k.astype(k_cache.dtype), (0, start_idx, 0, 0))
      v_cache = jax.lax.dynamic_update_slice(v_cache, v.astype(v_cache.dtype), (0, start_idx, 0, 0))
      updated_kv = (k_cache, v_cache)
      k_full, v_full = k_cache, v_cache
      kv_len = k_full.shape[1]
      k_pos = jnp.arange(kv_len, dtype=jnp.int32)[None, :]
      causal_mask = positions[:, :, None] >= k_pos[:, None, :]
      if segment_ids is not None:
        if isinstance(cache_position, int) and cache_position + q_len > segment_ids.shape[1]:
          raise ValueError(
              f"KV-cache segment_ids length ({segment_ids.shape[1]}) must cover cache_position + q_len "
              f"({cache_position + q_len})."
          )
        k_seg = segment_ids[:, :kv_len]
        if k_seg.shape[1] < kv_len:
          k_seg = jnp.pad(k_seg, ((0, 0), (0, kv_len - k_seg.shape[1])), constant_values=0)
        q_seg = jax.lax.dynamic_slice_in_dim(k_seg, start_idx, q_len, axis=1)
        causal_mask = causal_mask & (q_seg[:, :, None] == k_seg[:, None, :]) & (k_seg[:, None, :] != 0)
    else:
      k_full, v_full = k, v
      kv_len = q_len
      q_idx = jnp.arange(q_len, dtype=jnp.int32)[None, :, None]
      k_idx = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :]
      causal_mask = q_idx >= k_idx
      if segment_ids is not None:
        causal_mask = causal_mask & (segment_ids[:, :, None] == segment_ids[:, None, :]) & (segment_ids[:, None, :] != 0)

    if attention_mask is not None:
      valid_kv = attention_mask[:, :kv_len] != 0
      if valid_kv.shape[1] < kv_len:
        valid_kv = jnp.pad(valid_kv, ((0, 0), (0, kv_len - valid_kv.shape[1])), constant_values=False)
      causal_mask = causal_mask & valid_kv[:, None, :]

    # Group queries as (B, H_kv, n_rep, q_len, D) so each TP shard stays local to its KV head shard.
    q_grouped = q.reshape(bsz, q_len, self.num_kv_heads, self.num_kv_groups, self.head_dim).transpose(0, 2, 3, 1, 4)
    k_grouped = k_full[:, :, :, None, :].transpose(0, 2, 3, 1, 4)
    v_grouped = v_full[:, :, :, None, :].transpose(0, 2, 3, 1, 4)

    scale = self.attention_multiplier if attention_multiplier is None else attention_multiplier
    scores = jnp.einsum("bhgsd,bhgkd->bhgsk", q_grouped, k_grouped, precision=self.precision) * jnp.asarray(
        scale, dtype=q.dtype
    )
    mask_5d = causal_mask[:, None, None, :, :]
    scores = jnp.where(mask_5d, scores, jnp.finfo(scores.dtype).min)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    probs = jnp.where(jnp.any(mask_5d, axis=-1, keepdims=True), probs, jnp.zeros_like(probs))
    if not deterministic and self.attention_dropout > 0.0:
      if dropout_rng is None:
        raise ValueError("dropout_rng must be provided when deterministic=False and attention_dropout > 0.")
      keep_prob = 1.0 - self.attention_dropout
      keep_mask = jax.random.bernoulli(dropout_rng, p=keep_prob, shape=probs.shape)
      probs = jnp.where(keep_mask, probs / jnp.asarray(keep_prob, dtype=probs.dtype), jnp.zeros_like(probs))

    attn_out = jnp.einsum("bhgsk,bhgkd->bhgsd", probs, v_grouped, precision=self.precision)
    attn_out = attn_out.transpose(0, 3, 1, 2, 4).reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_out = _constrain_activation(self.sharding, attn_out, "attn_out", ("batch", "length", "heads"))
    out = self.o_proj(attn_out)
    out = _constrain_activation(self.sharding, out, "attn_proj_out", ("batch", "length", "embed"))
    return out, updated_kv


# ==============================================================================
# Section 4: Top-K Router, Sharded `ragged_dot` MoE & Switch Load-Balancing Loss
# ==============================================================================


def _local_ragged_swiglu(
    x_tokens: jax.Array,
    top_k_indices: jax.Array,
    top_k_weights: jax.Array,
    gu_w: jax.Array,
    down_w: jax.Array,
    *,
    expert_offset: jax.Array | int,
    num_local_experts: int,
    top_k: int,
    precision: jax.lax.Precision,
) -> jax.Array:
  """Executes SwiGLU `jax.lax.ragged_dot` for experts `[expert_offset, expert_offset + num_local_experts)`."""
  dt = x_tokens.dtype
  num_tokens, hidden_size = x_tokens.shape
  total_slots = num_tokens * top_k

  flat_idx = top_k_indices.reshape(total_slots)
  flat_w = top_k_weights.astype(dt).reshape(total_slots)

  local_idx = flat_idx - expert_offset
  is_local = (local_idx >= 0) & (local_idx < num_local_experts)
  bucket = jnp.where(is_local, local_idx, num_local_experts)
  masked_w = jnp.where(is_local, flat_w, jnp.zeros_like(flat_w))

  perm = jnp.argsort(bucket, stable=True)
  inv_perm = jnp.argsort(perm)
  sorted_valid = is_local[perm][:, None]
  token_positions = perm // top_k
  # Explicitly zero non-local tail rows [sum(group_sizes):total_slots] on both inputs and outputs of
  # `jax.lax.ragged_dot`: on TPU XLA, `ragged-dot` lhs VJP does not zero rows past `sum(group_sizes)`.
  zero_scalar = jnp.zeros((), dtype=dt)
  sorted_x = jnp.where(sorted_valid, x_tokens[token_positions], zero_scalar)
  group_sizes = jnp.bincount(bucket, length=num_local_experts + 1)[:num_local_experts]

  gu = gu_w.astype(dt)
  gate = jax.lax.ragged_dot(
      sorted_x,
      gu[:, :, 0, :],
      group_sizes=group_sizes,
      preferred_element_type=dt,
      precision=precision,
  )
  up = jax.lax.ragged_dot(
      sorted_x,
      gu[:, :, 1, :],
      group_sizes=group_sizes,
      preferred_element_type=dt,
      precision=precision,
  )
  act = jnp.where(sorted_valid, jax.nn.silu(gate) * up, zero_scalar)
  sorted_out = jax.lax.ragged_dot(
      act,
      down_w.astype(dt),
      group_sizes=group_sizes,
      preferred_element_type=dt,
      precision=precision,
  )
  sorted_out = jnp.where(sorted_valid, sorted_out, zero_scalar)
  unsorted_out = sorted_out[inv_perm] * masked_w[:, None]
  return jnp.sum(unsorted_out.reshape(num_tokens, top_k, hidden_size), axis=1)


class GraniteMoeTopKRouter(nnx.Module):
  """FP32 top-k router matching `GraniteMoeTopKGating` (supports optional `frozen_topk_indices`)."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeTopKRouter requires a non-None nnx.Rngs instance.")
    self.hidden_size = config.hidden_size
    self.num_experts = config.num_local_experts
    self.top_k = config.num_experts_per_tok
    w_dt = _resolve_jnp_dtype(config.weight_dtype)
    spec = sharding("router", ("embed", None)) if sharding is not None else None
    init_val = nnx.initializers.normal(stddev=0.02)(rngs.params(), (self.hidden_size, self.num_experts), w_dt)
    self.weight = nnx.Param(init_val, out_sharding=spec)

  def __call__(
      self,
      x_flat: jax.Array,
      *,
      frozen_topk_indices: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    router_logits = jnp.matmul(
        x_flat.astype(jnp.float32),
        self.weight[...].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    if frozen_topk_indices is not None:
      top_k_indices = frozen_topk_indices.astype(jnp.int32)
      top_k_logits = jnp.take_along_axis(router_logits, top_k_indices, axis=-1)
    else:
      top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.top_k)
    top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)
    return top_k_weights, top_k_indices, router_logits


class GraniteMoeExperts(nnx.Module):
  """3D/4D stacked SwiGLU experts with zero expert-weight `all-gather` across FSDP, TP, and EP."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeExperts requires a non-None nnx.Rngs instance.")
    self.num_experts = config.num_local_experts
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.top_k = config.num_experts_per_tok
    self.precision = _resolve_precision(config.matmul_precision)
    self.sharding = sharding

    w_dt = _resolve_jnp_dtype(config.weight_dtype)
    gu_spec = sharding("gate_up_proj", ("experts", "embed_exp", None, "mlp")) if sharding else None
    down_spec = sharding("down_proj", ("experts", "mlp", "embed_exp")) if sharding else None

    k1, k2 = jax.random.split(rngs.params())
    init_fn = nnx.initializers.normal(stddev=0.02)
    self.gate_up_proj = nnx.Param(
        init_fn(k1, (self.num_experts, self.hidden_size, 2, self.intermediate_size), w_dt),
        out_sharding=gu_spec,
    )
    self.down_proj = nnx.Param(
        init_fn(k2, (self.num_experts, self.intermediate_size, self.hidden_size), w_dt),
        out_sharding=down_spec,
    )

  def __call__(
      self,
      x_3d: jax.Array,
      top_k_indices_3d: jax.Array,
      top_k_weights_3d: jax.Array,
  ) -> jax.Array:
    bsz, seq_len, hidden_size = x_3d.shape
    num_tokens = bsz * seq_len
    gu_w = self.gate_up_proj[...]
    down_w = self.down_proj[...]

    sh = self.sharding
    abs_mesh = jax.sharding.get_abstract_mesh()
    use_compound_shard_map = (
        isinstance(sh, GraniteMoeSharding)
        and sh.compound_ep
        and not abs_mesh.empty
        and (sh.fsdp_size * sh.ep_size > 1 or sh.tp_size > 1)
    )

    if not use_compound_shard_map:
      if isinstance(sh, GraniteMoeSharding) and not sh.compound_ep and sh.ep_size > 1:
        raise NotImplementedError(
            f"Non-compound EP (num_experts={self.num_experts}, fsdp={sh.fsdp_size}, expert={sh.ep_size}) "
            "requires num_experts divisible by fsdp * expert."
        )
      out_2d = _local_ragged_swiglu(
          x_3d.reshape(num_tokens, hidden_size),
          top_k_indices_3d.reshape(num_tokens, self.top_k),
          top_k_weights_3d.reshape(num_tokens, self.top_k),
          gu_w,
          down_w,
          expert_offset=0,
          num_local_experts=self.num_experts,
          top_k=self.top_k,
          precision=self.precision,
      )
      return out_2d.reshape(bsz, seq_len, hidden_size)

    assert isinstance(sh, GraniteMoeSharding)
    fe_shards = sh.fsdp_size * sh.ep_size
    local_experts = self.num_experts // fe_shards
    top_k = self.top_k
    precision = self.precision
    dp_active = sh.dp_size > 1
    tp_active = sh.tp_size > 1
    fe_active = fe_shards > 1

    # Batch tiling check: shard batch across ('data', 'fsdp', 'expert') when evenly divisible;
    # otherwise (e.g. B=1 autoregressive decode on multi-chip mesh) keep activations replicated on ('fsdp', 'expert').
    tile_batch_on_fe = fe_active and (bsz % (sh.dp_size * fe_shards) == 0)
    tile_batch_on_dp = dp_active and (bsz % sh.dp_size == 0)
    if tile_batch_on_fe:
      act_batch_axis: Any = ("data", "fsdp", "expert")
    elif tile_batch_on_dp:
      act_batch_axis = "data"
    else:
      act_batch_axis = None

    act_spec = P(act_batch_axis, None, None)
    gu_spec = P(("fsdp", "expert"), None, None, "tensor")
    down_spec = P(("fsdp", "expert"), "tensor", None)

    def _ep_tp_kernel(
        x_local: jax.Array,
        idx_local: jax.Array,
        w_local: jax.Array,
        gu_local: jax.Array,
        down_local: jax.Array,
    ) -> jax.Array:
      if tile_batch_on_fe:
        x_full = jax.lax.all_gather(x_local, ("fsdp", "expert"), axis=0, tiled=True)
        idx_full = jax.lax.all_gather(idx_local, ("fsdp", "expert"), axis=0, tiled=True)
        w_full = jax.lax.all_gather(w_local, ("fsdp", "expert"), axis=0, tiled=True)
      else:
        x_full, idx_full, w_full = x_local, idx_local, w_local

      b_slice, s_slice, h_dim = x_full.shape
      t_slice = b_slice * s_slice
      ef_rank = jax.lax.axis_index(("fsdp", "expert")) if fe_active else 0
      exp_offset = ef_rank * local_experts

      y_partial = _local_ragged_swiglu(
          x_full.reshape(t_slice, h_dim),
          idx_full.reshape(t_slice, top_k),
          w_full.reshape(t_slice, top_k),
          gu_local,
          down_local,
          expert_offset=exp_offset,
          num_local_experts=local_experts,
          top_k=top_k,
          precision=precision,
      ).reshape(b_slice, s_slice, h_dim)

      if tp_active:
        y_partial = jax.lax.psum(y_partial, "tensor")
      if tile_batch_on_fe:
        return jax.lax.psum_scatter(y_partial, ("fsdp", "expert"), scatter_dimension=0, tiled=True)
      if fe_active:
        return jax.lax.psum(y_partial, ("fsdp", "expert"))
      return y_partial

    return jax.shard_map(
        _ep_tp_kernel,
        in_specs=(act_spec, act_spec, act_spec, gu_spec, down_spec),
        out_specs=act_spec,
        check_vma=False,
    )(x_3d, top_k_indices_3d, top_k_weights_3d, gu_w, down_w)


class GraniteMoeMoE(nnx.Module):
  """GraniteMoe sparse MoE block (`router` + `experts`)."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeMoE requires a non-None nnx.Rngs instance.")
    self.num_experts = config.num_local_experts
    self.top_k = config.num_experts_per_tok
    self.sharding = sharding
    self.router = GraniteMoeTopKRouter(config, sharding=sharding, rngs=rngs)
    self.experts = GraniteMoeExperts(config, sharding=sharding, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      *,
      frozen_topk_indices: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    bsz, seq_len, hidden_size = hidden_states.shape
    h_flat = hidden_states.reshape(bsz * seq_len, hidden_size)
    frozen_flat = frozen_topk_indices.reshape(bsz * seq_len, self.top_k) if frozen_topk_indices is not None else None
    top_k_weights, top_k_indices, router_logits = self.router(h_flat, frozen_topk_indices=frozen_flat)

    out = self.experts(
        hidden_states,
        top_k_indices.reshape(bsz, seq_len, self.top_k),
        top_k_weights.reshape(bsz, seq_len, self.top_k),
    )
    out = _constrain_activation(self.sharding, out, "moe_out", ("batch", "length", "embed"))
    return (
        out,
        router_logits.reshape(bsz, seq_len, self.num_experts),
        top_k_indices.reshape(bsz, seq_len, self.top_k),
    )


def compute_granitemoe_lb_loss(
    router_logits: jax.Array,
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
  """Switch Transformer load-balancing loss matching HF `modeling_granitemoe.load_balancing_loss_func`.

  Note: `tokens_per_expert` divides by `num_valid_tokens` (NOT `num_valid_tokens * top_k`), so
  `sum_e f_e = top_k` and uniform routing over `num_experts` yields `aux_loss = top_k`.

  Args:
    router_logits: Stacked router logits of shape `(num_layers, batch, seq_len, num_experts)` or `(L, T, E)`.
    num_experts: Number of routed experts (`E`).
    top_k: Number of experts selected per token (`k`).
    attention_mask: Optional mask of shape `(batch, seq_len)` where non-zero marks valid tokens.

  Returns:
    Scalar `float32` unscaled load-balancing auxiliary loss.
  """
  num_layers = router_logits.shape[0]
  logits_3d = router_logits.reshape(num_layers, -1, num_experts).astype(jnp.float32)
  num_tokens = logits_3d.shape[1]

  routing_weights = jax.nn.softmax(logits_3d, axis=-1)
  _, selected_experts = jax.lax.top_k(routing_weights, top_k)
  expert_mask = jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32)  # (L, T, k, E)

  if attention_mask is None:
    denom = jnp.asarray(max(num_layers * num_tokens, 1), dtype=jnp.float32)
    tokens_per_expert = jnp.sum(expert_mask, axis=(0, 1, 2)) / denom
    router_prob_per_expert = jnp.sum(routing_weights, axis=(0, 1)) / denom
  else:
    valid_2d = (attention_mask.reshape(1, num_tokens) != 0).astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(valid_2d) * float(num_layers), 1.0)
    tokens_per_expert = jnp.sum(expert_mask * valid_2d[:, :, None, None], axis=(0, 1, 2)) / denom
    router_prob_per_expert = jnp.sum(routing_weights * valid_2d[:, :, None], axis=(0, 1)) / denom

  return (jnp.sum(tokens_per_expert * router_prob_per_expert) * float(num_experts)).astype(jnp.float32)


# ==============================================================================
# Section 5: Decoder Block, Stacked `vmap`/`scan`/`remat` & Full `GraniteMoeModel`
# ==============================================================================


class GraniteMoeDecoderLayer(nnx.Module):
  """Single GraniteMoe decoder block (never stores Python `int` `layer_idx` so `vmap`/`scan` graphdef stays uniform)."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeDecoderLayer requires a non-None nnx.Rngs instance.")
    w_dt = _resolve_jnp_dtype(config.weight_dtype)
    self.residual_multiplier = config.residual_multiplier
    self.sharding = sharding
    self.input_layernorm = GraniteMoeRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, param_dtype=w_dt, sharding=sharding
    )
    self.self_attn = GraniteMoeAttention(config, sharding=sharding, rngs=rngs)
    self.post_attention_layernorm = GraniteMoeRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, param_dtype=w_dt, sharding=sharding
    )
    self.block_sparse_moe = GraniteMoeMoE(config, sharding=sharding, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      positions: jax.Array,
      *,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      layer_kv_cache: tuple[jax.Array, jax.Array] | None = None,
      cache_position: jax.Array | int | None = None,
      frozen_topk_indices: jax.Array | None = None,
      attention_multiplier: float | jax.Array | None = None,
      residual_multiplier: float | jax.Array | None = None,
      deterministic: bool = True,
      dropout_rng: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None]:
    res_scale = jnp.asarray(
        self.residual_multiplier if residual_multiplier is None else residual_multiplier,
        dtype=hidden_states.dtype,
    )

    attn_out, updated_kv = self.self_attn(
        self.input_layernorm(hidden_states),
        positions,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        layer_kv_cache=layer_kv_cache,
        cache_position=cache_position,
        attention_multiplier=attention_multiplier,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
    )
    hidden_states = hidden_states + attn_out * res_scale
    hidden_states = _constrain_activation(self.sharding, hidden_states, "post_attn_res", ("batch", "length", "embed"))

    moe_out, router_logits, top_k_indices = self.block_sparse_moe(
        self.post_attention_layernorm(hidden_states),
        frozen_topk_indices=frozen_topk_indices,
    )
    hidden_states = hidden_states + moe_out * res_scale
    hidden_states = _constrain_activation(self.sharding, hidden_states, "post_moe_res", ("batch", "length", "embed"))
    return hidden_states, router_logits, top_k_indices, updated_kv


def _create_stacked_decoder_layers(
    config: GraniteMoeConfig,
    *,
    sharding: m3_sharding.Sharding | None,
    mesh: jax.sharding.Mesh | None,
    rngs: nnx.Rngs,
) -> GraniteMoeDecoderLayer:
  """Constructs `num_hidden_layers` stacked `GraniteMoeDecoderLayer` placed onto `NamedSharding(mesh, P(None, ...))`."""

  def _init_single(r: nnx.Rngs) -> GraniteMoeDecoderLayer:
    return GraniteMoeDecoderLayer(config, sharding=sharding, rngs=r)

  forked_rngs = rngs.split(config.num_hidden_layers)
  stacked = nnx.vmap(_init_single, in_axes=0, out_axes=0)(forked_rngs)
  graphdef, state = nnx.split(stacked)

  def _place_var(path: tuple[Any, ...], var: Any) -> Any:
    if not isinstance(var, nnx.Variable):
      return var
    orig_sh = var.get_metadata("out_sharding", None)
    if orig_sh is None:
      if sharding is not None:
        raise ValueError(f"Missing out_sharding metadata on stacked parameter at {path}.")
      return var
    orig_spec = orig_sh.spec if isinstance(orig_sh, jax.sharding.NamedSharding) else orig_sh
    new_spec = P(None, *orig_spec)
    raw_val = var.get_raw_value()
    if mesh is not None and not mesh.empty and not isinstance(raw_val, jax.core.Tracer):
      placed_val = jax.device_put(raw_val, jax.sharding.NamedSharding(mesh, new_spec))
      return var.replace(placed_val, out_sharding=new_spec)
    return var.replace(raw_val, out_sharding=new_spec)

  updated_state = jax.tree_util.tree_map_with_path(
      _place_var,
      state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  return nnx.merge(graphdef, updated_state)


def _slice_stacked_state(state: Any, layer_idx: int) -> Any:
  """Slices index `layer_idx` along axis 0 of a stacked NNX state tree, stripping the leading `PartitionSpec` entry."""

  def _slice_leaf(v: Any) -> Any:
    if isinstance(v, nnx.Variable):
      orig_sh = v.get_metadata("out_sharding", None)
      sliced_val = v.get_raw_value()[layer_idx]
      if orig_sh is not None:
        orig_spec = orig_sh.spec if isinstance(orig_sh, jax.sharding.NamedSharding) else orig_sh
        return v.replace(sliced_val, out_sharding=P(*orig_spec[1:]))
      return v.replace(sliced_val)
    return v[layer_idx]

  return jax.tree_util.tree_map(_slice_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def _resolve_remat_callable(fn: Callable[..., Any], remat_policy: str) -> Callable[..., Any]:
  """Wraps `fn` with `jax.checkpoint` according to `remat_policy` without altering NNX parameter paths."""
  if remat_policy == "none":
    return fn
  if remat_policy == "full":
    return jax.checkpoint(fn)
  if remat_policy == "minimal":
    return jax.checkpoint(fn, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
  raise ValueError(f"Unsupported remat_policy {remat_policy!r}.")


class GraniteMoeModel(nnx.Module):
  """Self-contained GraniteMoe causal language model in pure Flax NNX (`m3`)."""

  def __init__(
      self,
      config: GraniteMoeConfig,
      *,
      mesh: jax.sharding.Mesh | None = None,
      sharding: GraniteMoeSharding | m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if rngs is None:
      raise ValueError("GraniteMoeModel requires a non-None nnx.Rngs instance.")
    self.config = config
    self.mesh = mesh
    if sharding is None and mesh is not None and not mesh.empty:
      sharding = GraniteMoeSharding.from_mesh(mesh, config)
    self.sharding = sharding
    self.precision = _resolve_precision(config.matmul_precision)

    act_dt = _resolve_jnp_dtype(config.dtype)
    w_dt = _resolve_jnp_dtype(config.weight_dtype)

    emb_spec = sharding("embed_tokens", ("vocab", "embed")) if sharding else None
    self.embed_tokens = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        dtype=act_dt,
        param_dtype=w_dt,
        embedding_metadata={"out_sharding": emb_spec} if emb_spec is not None else {},
        rngs=rngs,
    )
    self.layers = _create_stacked_decoder_layers(config, sharding=sharding, mesh=mesh, rngs=rngs)
    self.norm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, param_dtype=w_dt, sharding=sharding)

    if config.tie_word_embeddings:
      self.lm_head = None
    else:
      lm_spec = sharding("lm_head", ("embed", "vocab")) if sharding else None
      self.lm_head = nnx.Linear(
          config.hidden_size,
          config.vocab_size,
          use_bias=False,
          dtype=act_dt,
          param_dtype=w_dt,
          precision=self.precision,
          kernel_metadata={"out_sharding": lm_spec} if lm_spec is not None else {},
          rngs=rngs,
      )

  def forward_with_details(
      self,
      input_ids: jax.Array,
      positions: jax.Array | None = None,
      *,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      kv_cache: GraniteMoeKVCache | None = None,
      cache_position: jax.Array | int | None = None,
      frozen_topk_indices: jax.Array | None = None,
      deterministic: bool = True,
      dropout_rng: jax.Array | None = None,
      rngs: nnx.Rngs | None = None,
  ) -> GraniteMoeForwardOutput:
    """Runs GraniteMoe forward pass returning logits, aux losses, per-layer routing info, and updated KV cache."""
    use_dropout = (not deterministic) and (self.config.attention_dropout > 0.0)
    if use_dropout and dropout_rng is None and rngs is not None:
      dropout_rng = rngs.dropout()
    if use_dropout and dropout_rng is None:
      raise ValueError("Pass `dropout_rng` (or `rngs`) when `deterministic=False` and `attention_dropout > 0`.")
    layer_dropout_rngs = (
        jax.random.split(dropout_rng, self.config.num_hidden_layers) if use_dropout and dropout_rng is not None else None
    )

    bsz, seq_len = input_ids.shape
    if positions is None:
      start = jnp.asarray(0 if cache_position is None else cache_position, dtype=jnp.int32)
      positions = start + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      positions = jnp.broadcast_to(positions, (bsz, seq_len))

    act_dt = _resolve_jnp_dtype(self.config.dtype)
    hidden_states = self.embed_tokens(input_ids).astype(act_dt) * jnp.asarray(
        self.config.embedding_multiplier, dtype=act_dt
    )
    hidden_states = _constrain_activation(self.sharding, hidden_states, "embed_out", ("batch", "length", "embed"))

    graphdef, params_state, non_params_state = nnx.split(self.layers, nnx.Param, ...)
    attn_mult = self.config.attention_multiplier
    res_mult = self.config.residual_multiplier
    use_cache = kv_cache is not None
    use_frozen = frozen_topk_indices is not None

    # pylint: disable=too-many-positional-arguments
    def _run_one_layer(
        p_slice: Any,
        np_slice: Any,
        h_in: jax.Array,
        k_layer: jax.Array | None,
        v_layer: jax.Array | None,
        frozen_layer: jax.Array | None,
        drop_rng: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
      layer_mod = nnx.merge(graphdef, p_slice, np_slice)
      kv_in = (k_layer, v_layer) if use_cache else None
      h_out, r_logits, topk_idx, kv_out = layer_mod(
          h_in,
          positions,
          attention_mask=attention_mask,
          segment_ids=segment_ids,
          layer_kv_cache=kv_in,
          cache_position=cache_position,
          frozen_topk_indices=frozen_layer,
          attention_multiplier=attn_mult,
          residual_multiplier=res_mult,
          deterministic=deterministic,
          dropout_rng=drop_rng,
      )
      if use_cache and kv_out is not None:
        return h_out, r_logits, topk_idx, kv_out[0], kv_out[1]
      return h_out, r_logits, topk_idx, None, None

    layer_fn = _resolve_remat_callable(_run_one_layer, self.config.remat_policy)

    if self.config.scan_layers:
      scan_xs = (
          params_state,
          non_params_state,
          kv_cache.k if use_cache else None,
          kv_cache.v if use_cache else None,
          frozen_topk_indices if use_frozen else None,
          layer_dropout_rngs if use_dropout else None,
      )

      def _scan_body(carry: jax.Array, xs: Any) -> tuple[jax.Array, tuple[Any, ...]]:
        p_s, np_s, k_s, v_s, f_s, d_s = xs
        h_next, r_log, tk_idx, k_next, v_next = layer_fn(p_s, np_s, carry, k_s, v_s, f_s, d_s)
        return h_next, (r_log, tk_idx, k_next, v_next)

      hidden_states, (all_router_logits, all_topk_indices, new_k_all, new_v_all) = jax.lax.scan(
          _scan_body, hidden_states, scan_xs
      )
      updated_cache = GraniteMoeKVCache(k=new_k_all, v=new_v_all) if use_cache else None
    else:
      r_logits_list: list[jax.Array] = []
      topk_list: list[jax.Array] = []
      k_list: list[jax.Array] = []
      v_list: list[jax.Array] = []
      for idx in range(self.config.num_hidden_layers):
        p_i = _slice_stacked_state(params_state, idx)
        np_i = _slice_stacked_state(non_params_state, idx)
        k_i = kv_cache.k[idx] if use_cache else None
        v_i = kv_cache.v[idx] if use_cache else None
        f_i = frozen_topk_indices[idx] if use_frozen else None
        d_i = layer_dropout_rngs[idx] if (use_dropout and layer_dropout_rngs is not None) else None
        hidden_states, r_i, tk_i, k_out_i, v_out_i = layer_fn(p_i, np_i, hidden_states, k_i, v_i, f_i, d_i)
        r_logits_list.append(r_i)
        topk_list.append(tk_i)
        if use_cache and k_out_i is not None and v_out_i is not None:
          k_list.append(k_out_i)
          v_list.append(v_out_i)
      all_router_logits = jnp.stack(r_logits_list, axis=0)
      all_topk_indices = jnp.stack(topk_list, axis=0)
      updated_cache = GraniteMoeKVCache(k=jnp.stack(k_list, axis=0), v=jnp.stack(v_list, axis=0)) if use_cache else None

    normed = self.norm(hidden_states)
    normed = _constrain_activation(self.sharding, normed, "final_norm_out", ("batch", "length", "embed"))

    if self.config.tie_word_embeddings:
      # CRITICAL: Tied lm_head uses RAW unscaled embedding weights, never scaled by embedding_multiplier.
      emb_w = self.embed_tokens.embedding[...].astype(normed.dtype)
      raw_logits = jnp.einsum("bsh,vh->bsv", normed, emb_w, precision=self.precision)
    else:
      assert self.lm_head is not None
      raw_logits = self.lm_head(normed)

    raw_logits = _constrain_activation(self.sharding, raw_logits, "logits", ("batch", "length", "vocab"))
    logits = raw_logits / jnp.asarray(self.config.logits_scaling, dtype=raw_logits.dtype)

    # Mask padded query tokens in Switch load-balancing loss when segment_ids and/or non-cached attention_mask exist.
    lb_mask: jax.Array | None = None
    if segment_ids is not None and segment_ids.shape[-1] == seq_len:
      lb_mask = segment_ids != 0
    if attention_mask is not None and attention_mask.shape[-1] == seq_len:
      valid_attn = attention_mask != 0
      lb_mask = valid_attn if lb_mask is None else (lb_mask & valid_attn)

    moe_lb_loss = compute_granitemoe_lb_loss(
        all_router_logits,
        num_experts=self.config.num_local_experts,
        top_k=self.config.num_experts_per_tok,
        attention_mask=lb_mask,
    )
    return GraniteMoeForwardOutput(
        logits=logits,
        aux={"moe_lb_loss": moe_lb_loss},
        router_logits=all_router_logits,
        top_k_indices=all_topk_indices,
        kv_cache=updated_cache,
        hidden_states=normed,
    )

  def __call__(
      self,
      input_ids: jax.Array | None = None,
      positions: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      *,
      decoder_input_tokens: jax.Array | None = None,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      kv_cache: GraniteMoeKVCache | None = None,
      cache_position: jax.Array | int | None = None,
      frozen_topk_indices: jax.Array | None = None,
      deterministic: bool = True,
      enable_dropout: bool | None = None,
      dropout_rng: jax.Array | None = None,
      rngs: nnx.Rngs | None = None,
      return_aux: bool = False,
      **unused_kwargs: Any,
  ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    """Standard M3 forward entrypoint returning `logits` by default (`return_aux=False`) or `(logits, aux)`."""
    del unused_kwargs
    toks = decoder_input_tokens if input_ids is None else input_ids
    if toks is None:
      raise ValueError("Either `input_ids` or `decoder_input_tokens` must be provided.")
    pos = decoder_positions if positions is None else positions
    seg = decoder_segment_ids if segment_ids is None else segment_ids
    det = (not enable_dropout) if enable_dropout is not None else deterministic
    out = self.forward_with_details(
        toks,
        positions=pos,
        attention_mask=attention_mask,
        segment_ids=seg,
        kv_cache=kv_cache,
        cache_position=cache_position,
        frozen_topk_indices=frozen_topk_indices,
        deterministic=det,
        dropout_rng=dropout_rng,
        rngs=rngs,
    )
    if return_aux:
      return out.logits, out.aux
    return out.logits


# ==============================================================================
# Section 6: Masked Cross-Entropy + Router `aux_loss` & Factory Entrypoint
# ==============================================================================


def compute_granitemoe_cross_entropy_loss(
    logits: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes `fp32` masked next-token cross-entropy `(ce_loss, xent_sum, total_weights)`."""
  logits_f32 = logits.astype(jnp.float32)
  log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
  token_nll = -jnp.take_along_axis(log_probs, decoder_target_tokens[..., None].astype(jnp.int32), axis=-1).squeeze(-1)

  if decoder_target_mask is None:
    total_weights = jnp.asarray(token_nll.size, dtype=jnp.float32)
    xent_sum = jnp.sum(token_nll)
  else:
    mask_f32 = (decoder_target_mask != 0).astype(jnp.float32)
    total_weights = jnp.sum(mask_f32)
    xent_sum = jnp.sum(token_nll * mask_f32)

  safe_denom = jnp.maximum(total_weights, 1.0)
  ce_loss = (xent_sum / safe_denom).astype(jnp.float32)
  return ce_loss, xent_sum.astype(jnp.float32), total_weights.astype(jnp.float32)


def compute_granitemoe_loss(
    model: GraniteMoeModel,
    *,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array | None = None,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    router_aux_loss_coef: float | None = None,
    is_train: bool = True,
    dropout_rng: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes `total_loss = ce_loss + router_aux_loss_coef * moe_lb_loss` and auxiliary metrics.

  Note: `decoder_target_mask` (`targets_segmentation`) masks ONLY `ce_loss` (never `attention_mask` or `moe_lb_loss`),
  so SFT prompt masking (`0` on prompt, `1` on response) never blinds response tokens to prompt context.
  """
  logits, aux = model(
      tokens,
      positions=decoder_positions,
      attention_mask=attention_mask,
      segment_ids=decoder_segment_ids,
      deterministic=not is_train,
      dropout_rng=dropout_rng,
      return_aux=True,
  )
  ce_loss, xent_sum, total_weights = compute_granitemoe_cross_entropy_loss(
      logits, decoder_target_tokens, decoder_target_mask
  )
  moe_lb_loss = aux["moe_lb_loss"].astype(jnp.float32)
  coef = model.config.router_aux_loss_coef if router_aux_loss_coef is None else router_aux_loss_coef
  total_loss = (ce_loss + jnp.float32(coef) * moe_lb_loss).astype(jnp.float32)
  metrics = {
      "ce_loss": ce_loss,
      "moe_lb_loss": moe_lb_loss,
      "xent_sum": xent_sum,
      "total_weights": total_weights,
  }
  return total_loss, metrics


def granitemoe_loss_fn(
    model: GraniteMoeModel,
    *,
    inputs: jax.Array,
    targets: jax.Array,
    targets_segmentation: jax.Array | None = None,
    is_train: bool = True,
    inputs_segmentation: jax.Array | None = None,
    inputs_position: jax.Array | None = None,
    targets_position: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    dropout_rng: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """MaxText `loss_fn` signature adapter folding `moe_lb_loss` into primary scalar loss."""
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be bool, got {type(is_train).__name__}.")
  del targets_position
  return compute_granitemoe_loss(
      model=model,
      tokens=inputs,
      decoder_target_tokens=targets,
      decoder_target_mask=targets_segmentation,
      decoder_segment_ids=inputs_segmentation,
      decoder_positions=inputs_position,
      attention_mask=attention_mask,
      is_train=is_train,
      dropout_rng=dropout_rng,
  )


def create_granitemoe_model(
    config: Any,
    mesh: jax.sharding.Mesh | None = None,
    *,
    rngs: nnx.Rngs,
    quant_mode_str: str | None = None,
    sharding: GraniteMoeSharding | m3_sharding.Sharding | None = None,
    **overrides: Any,
) -> GraniteMoeModel:
  """Constructs `GraniteMoeModel` eagerly sharded on `mesh` (`model.mesh is mesh`), supporting `nnx.eval_shape`."""
  if rngs is None or not isinstance(rngs, nnx.Rngs):
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  if quant_mode_str not in (None, "", "train", "serve", "convert"):
    raise ValueError(f"Unsupported quant_mode_str for GraniteMoe: {quant_mode_str!r}.")
  if not isinstance(config, GraniteMoeConfig) and getattr(config, "quantization", None) not in (None, ""):
    raise ValueError(f"Quantization is unsupported for GraniteMoe: {getattr(config, 'quantization', None)!r}.")

  granite_cfg = GraniteMoeConfig.from_config(config, **overrides)
  if mesh is not None and not mesh.empty:
    resolved_sharding = sharding if sharding is not None else GraniteMoeSharding.from_mesh(mesh, granite_cfg)
    is_tracing = isinstance(jnp.zeros(()), jax.core.Tracer)
    if is_tracing:
      with nnx.use_eager_sharding(False):
        return GraniteMoeModel(granite_cfg, mesh=mesh, sharding=resolved_sharding, rngs=rngs)
    with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
      return GraniteMoeModel(granite_cfg, mesh=mesh, sharding=resolved_sharding, rngs=rngs)
  return GraniteMoeModel(granite_cfg, mesh=mesh, sharding=sharding, rngs=rngs)
