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

"""Streaming MaxText-to-vLLM (`vllm-torchtpu`) weight converter for Raiden sync.

Raiden (`tpu_sync`) transfers rectangular tensor slices without transposing,
casting, padding, or fusing on the wire. `VllmTorchWeightConverter` converts
MaxText Qwen3.5 trainer state into global JAX arrays matching `vllm-torchtpu`'s
rank-major post-load storage layout inside cached per-group `jax.jit` programs,
popping consumed flat-dict entries layer by layer so intermediate slices do not
accumulate alongside the trainer's live state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
import re
from typing import Any, Callable, Mapping, Sequence

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.integration.vllm.convert_utils import pad_to_tpu_lanes

_LAYERS_N_RE = re.compile(r"^layers_(\d+)$")
_LAYER_SLOT_RE = re.compile(r"^layer_(\d+)$")

_DTYPE_MAP: dict[str, Any] = {
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float32": jnp.float32,
    "fp32": jnp.float32,
    "float16": jnp.float16,
    "fp16": jnp.float16,
}
_SUPPORTED_FLOAT_DTYPES = frozenset(
    {
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float16),
        jnp.dtype(jnp.float32),
    }
)
_ROUTED_EXPERT_SUFFIXES = ("experts.w13_weight", "experts.w2_weight")
_SUPPORTED_DECODER_BLOCKS = frozenset({"qwen3_5", "qwen3_next"})
_SKIPPED_ROOTS = ("vision_encoder", "visual", "audio_encoder", "mtp", "mtp_block")
_TOP_LEVEL_SRC_KEYS = (
    "token_embedder.embedding",
    "decoder.decoder_norm.scale",
    "decoder.logits_dense.kernel",
)
# Normalize legacy vs. current MaxText layer subpath prefixes once when building
# layer plans so conversion and default-layout synthesis share one key vocabulary.
_SUBPATH_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("attention.attention.", "self_attn."),
    ("self_attention.", "self_attn."),
    ("attention.norm.rms_norm.", "gdn.norm."),
    ("linear_attention.norm.rms_norm.", "gdn.norm."),
    ("attention.", "gdn."),
    ("linear_attention.", "gdn."),
    ("mlp.routed_experts.", "moe."),
    ("moe_block.shared_experts.", "shared_expert."),
    ("moe_block.shared_expert.", "shared_expert."),
    ("mlp.shared_expert.", "shared_expert."),
    ("moe_block.shared_expert_gate.", "shared_expert_gate."),
    ("mlp.shared_expert_gate.", "shared_expert_gate."),
    ("moe_block.", "moe."),
)


def _normalize_layer_subpath(subpath: str) -> str:
  for prefix, canonical in _SUBPATH_PREFIX_ALIASES:
    if subpath.startswith(prefix):
      return canonical + subpath[len(prefix) :]
  return subpath


def _resolve_jnp_dtype(dtype_spec: Any, fallback: Any = jnp.bfloat16) -> Any:
  """Resolves `dtype_spec` into a validated `{bf16, fp16, fp32}` `jnp.dtype`."""
  if dtype_spec is None:
    resolved = jnp.dtype(fallback)
  elif isinstance(dtype_spec, str):
    clean = dtype_spec.removeprefix("torch.").removeprefix("jnp.")
    try:
      resolved = jnp.dtype(_DTYPE_MAP.get(clean, clean))
    except TypeError as exc:
      raise ValueError(f"Unsupported dtype {dtype_spec!r} for vllm_torch conversion.") from exc
  else:
    resolved = jnp.dtype(dtype_spec)
  if resolved not in _SUPPORTED_FLOAT_DTYPES:
    raise ValueError(
        f"Unsupported dtype {dtype_spec!r} ({resolved}) for vllm_torch conversion; "
        "quantized/FP8 weight sync is not supported."
    )
  return resolved


@dataclass(frozen=True)
class TargetParamLayout:
  """Validated target layout descriptor for one canonical `vllm-torchtpu` tensor.

  Frozen and hashable so per-layer layout dicts can serve directly in the
  `jax.jit` compilation cache key. `flipped` and `conv_transposed` are tri-state
  (`None` when omitted by a caller-supplied mapping) so packers can preserve
  default polarity without aliasing `None` and `False` in the JIT cache.
  """

  local_shape: tuple[int, ...]
  global_shape: tuple[int, ...]
  dtype: Any
  shard_dim: int | None = None
  num_shards: int = 1
  shard_domain: str = "replicated"
  flipped: bool | None = None
  conv_transposed: bool | None = None
  replication_factors: tuple[int, ...] | None = None
  layer_idx: int | None = field(default=None, compare=False)

  @classmethod
  def create(
      cls,
      local_shape: Sequence[int],
      *,
      dtype: Any,
      shard_dim: int | None = None,
      num_shards: int = 1,
      shard_domain: str = "replicated",
      flipped: bool = False,
      conv_transposed: bool = False,
      replication_factors: Sequence[int] | None = None,
  ) -> TargetParamLayout:
    """Constructs a canonical layout descriptor, deriving `global_shape`."""
    local_tuple = tuple(int(dim) for dim in local_shape)
    effective_dim = shard_dim if num_shards > 1 else None
    effective_domain = shard_domain if num_shards > 1 else "replicated"
    global_dims = list(local_tuple)
    if effective_dim is not None:
      global_dims[effective_dim] *= int(num_shards)
    reps = tuple(int(r) for r in replication_factors) if replication_factors is not None else None
    return cls(
        local_shape=local_tuple,
        global_shape=tuple(global_dims),
        dtype=_resolve_jnp_dtype(dtype),
        shard_dim=effective_dim,
        num_shards=int(num_shards),
        shard_domain=effective_domain,
        flipped=flipped,
        conv_transposed=conv_transposed,
        replication_factors=reps,
    )

  @classmethod
  def from_mapping(
      cls,
      param_name: str,
      spec: Mapping[str, Any],
      default_dtype: Any,
  ) -> TargetParamLayout:
    """Parses and validates a caller-supplied layout dict once at the boundary."""
    if not isinstance(spec, Mapping) or "global_shape" not in spec:
      raise ValueError(
          f"Invalid target_layout entry for {param_name!r}: expected a flat layout "
          "mapping containing 'global_shape' (not a nested state pytree)."
      )
    try:
      dtype = _resolve_jnp_dtype(spec.get("dtype"), default_dtype)
    except ValueError as exc:
      raise ValueError(f"Unsupported target_layout dtype on {param_name}: {exc}") from exc
    num_shards = int(spec.get("num_shards", 1))
    if num_shards < 1:
      raise ValueError(f"Expected num_shards >= 1 on {param_name}, got {num_shards}.")
    shard_dim = spec.get("shard_dim")
    if shard_dim is None and num_shards > 1:
      raise ValueError(f"Inconsistent target_layout on {param_name}: shard_dim=None with num_shards={num_shards}.")
    local_shape = tuple(int(dim) for dim in spec.get("local_shape", ()))
    global_shape = tuple(int(dim) for dim in spec["global_shape"])
    raw_reps = spec.get("replication_factors")
    reps = tuple(int(r) for r in raw_reps) if raw_reps is not None else None
    raw_layer_idx = spec.get("layer_idx")
    return cls(
        local_shape=local_shape,
        global_shape=global_shape,
        dtype=dtype,
        shard_dim=shard_dim,
        num_shards=num_shards,
        shard_domain=str(spec.get("shard_domain", "replicated")),
        flipped=None if "flipped" not in spec else bool(spec["flipped"]),
        conv_transposed=None if "conv_transposed" not in spec else bool(spec["conv_transposed"]),
        replication_factors=reps,
        layer_idx=None if raw_layer_idx is None else int(raw_layer_idx),
    )


@dataclass(frozen=True)
class _LayerPlan:
  """Source-tree lookup plan for one logical decoder layer."""

  layer_idx: int
  group_tag: str
  scan_indices: tuple[int, ...]
  key_map: dict[str, str]


def _to_pure_flat(src_pytree: Any) -> dict[str, jax.Array]:
  """Flattens an NNX/FrozenDict/dict parameter tree to stripped dotted keys."""
  if isinstance(src_pytree, nnx.State):
    src_pytree = src_pytree.filter(nnx.Param)
  if hasattr(src_pytree, "to_pure_dict"):
    src_pytree = src_pytree.to_pure_dict()
  elif hasattr(src_pytree, "unfreeze"):
    src_pytree = src_pytree.unfreeze()
  out: dict[str, jax.Array] = {}
  for key_tuple, val in traverse_util.flatten_dict(src_pytree).items():
    if isinstance(val, nnx.Variable) and not isinstance(val, nnx.Param):
      continue
    parts = [str(k) for k in key_tuple]
    while parts and parts[0] in ("base", "params", "model"):
      parts.pop(0)
    if parts and parts[-1] == "value":
      parts.pop()
    if not parts or parts[0] in _SKIPPED_ROOTS:
      continue
    out[".".join(parts)] = val.get_value() if isinstance(val, nnx.Variable) else val
  return out


def _infer_mesh(flat_arrays: Mapping[str, Any]) -> Mesh | None:
  for arr in flat_arrays.values():
    mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
    if isinstance(mesh, Mesh):
      return mesh
  return None


def _multi_axis_out_sharding(
    mesh: Mesh | None,
    shape: tuple[int, ...],
    param_name: str = "",
) -> NamedSharding | None:
  """Assigns distinct trainer mesh axes across divisible tensor dimensions."""
  if mesh is None:
    return None
  priority = ("fsdp", "fsdp_transpose", "data", "stage", "tensor", "model", "expert")
  active = [axis for axis in mesh.axis_names if mesh.shape[axis] > 1]
  available = [axis for axis in priority if axis in active] + [axis for axis in active if axis not in priority]
  if not available or not shape:
    return NamedSharding(mesh, P())

  spec: list[str | None] = [None] * len(shape)
  used: set[str] = set()
  # Bind routed expert count (axis 0) to `"expert"` first even when `E < H`.
  if param_name.endswith(_ROUTED_EXPERT_SUFFIXES) and "expert" in available:
    expert_size = int(mesh.shape["expert"])
    if shape[0] >= expert_size and shape[0] % expert_size == 0:
      spec[0] = "expert"
      used.add("expert")

  for dim in sorted(range(len(shape)), key=lambda d: shape[d], reverse=True):
    if spec[dim] is not None:
      continue
    for axis in available:
      axis_size = int(mesh.shape[axis])
      if axis not in used and shape[dim] >= axis_size and shape[dim] % axis_size == 0:
        spec[dim] = axis
        used.add(axis)
        break
  return NamedSharding(mesh, P(*spec))


def _interleave_rank_major(
    segments: list[jax.Array],
    num_shards: int,
    rep_factors: Sequence[int],
    label: str,
) -> jax.Array:
  """Interleaves N-D segments along axis 0 into rank-major order across shards."""
  if num_shards < 1:
    raise ValueError(f"Expected num_shards >= 1 for {label}, got {num_shards}.")
  if len(rep_factors) != len(segments):
    raise ValueError(f"Expected {len(segments)} replication_factors for {label}, got {list(rep_factors)}.")
  reps = [int(r) for r in rep_factors]
  for rep_i in reps:
    if rep_i < 1 or num_shards % rep_i != 0:
      raise ValueError(f"Invalid replication factor {rep_i} for num_shards={num_shards} ({label}).")

  if num_shards == 1:
    return jnp.concatenate(segments, axis=0)

  tail_shape = segments[0].shape[1:]
  by_rank: list[jax.Array] = []
  for seg, rep_i in zip(segments, reps):
    unique_groups = num_shards // rep_i
    if seg.shape[0] % unique_groups != 0:
      raise ValueError(
          f"{label} dim {seg.shape[0]} is not divisible by unique_groups={unique_groups} "
          f"(num_shards={num_shards}, rep={rep_i})."
      )
    grouped = seg.reshape(unique_groups, seg.shape[0] // unique_groups, *tail_shape)
    by_rank.append(jnp.repeat(grouped, rep_i, axis=0) if rep_i > 1 else grouped)
  return jnp.concatenate(by_rank, axis=1).reshape(-1, *tail_shape)


def _pack_column_segments(
    segments_out_in: list[jax.Array],
    layout: TargetParamLayout,
    default_reps: Sequence[int] | None = None,
) -> jax.Array:
  """Packs `[out_seg, in]` segments into rank-major global column-linear layout."""
  rep_factors = layout.replication_factors or default_reps or (1,) * len(segments_out_in)
  fused = _interleave_rank_major(
      segments_out_in,
      num_shards=layout.num_shards,
      rep_factors=rep_factors,
      label="column segments",
  )
  out = fused.T if bool(layout.flipped) else fused
  return out.astype(layout.dtype)


def _pack_linear(
    weight_out_in: jax.Array,
    layout: TargetParamLayout,
) -> jax.Array:
  out = weight_out_in.T if bool(layout.flipped) else weight_out_in
  return out.astype(layout.dtype)


def _pack_conv1d(
    segments_c_1_k: list[jax.Array],
    layout: TargetParamLayout,
    default_reps: Sequence[int] | None = None,
) -> jax.Array:
  """Packs `[C_seg, 1, K]` depthwise conv segments into rank-major layout."""
  rep_factors = layout.replication_factors or default_reps or (1,) * len(segments_c_1_k)
  fused = _interleave_rank_major(
      segments_c_1_k,
      num_shards=layout.num_shards,
      rep_factors=rep_factors,
      label="conv1d segments",
  )
  out = jnp.transpose(fused, (2, 1, 0)) if bool(layout.conv_transposed) else fused
  return out.astype(layout.dtype)


def _pack_routed_experts_w13(
    gate_e_h_i: jax.Array,
    up_e_h_i: jax.Array,
    layout: TargetParamLayout,
) -> jax.Array:
  """Packs routed expert gate/up `[E, H, I]` into EP (`tp_shards=1`) or TP-MoE w13 layout."""
  flipped = True if layout.flipped is None else layout.flipped
  if len(layout.local_shape) != 3:
    raise ValueError(f"Invalid w13_weight local_shape={layout.local_shape}; expected 3D.")
  w13_dim = 2 if flipped else 1
  if layout.local_shape[w13_dim] <= 0 or layout.local_shape[w13_dim] % 2 != 0:
    raise ValueError(f"Expected positive even w13_weight dim at axis {w13_dim}, got {layout.local_shape}.")

  num_experts, hidden_dim, inter_dim = gate_e_h_i.shape
  tp_shards = layout.num_shards if (layout.shard_domain == "tp" and layout.num_shards > 1) else 1
  if inter_dim % tp_shards != 0:
    raise ValueError(f"Expert inter_dim={inter_dim} is not divisible by tp_shards={tp_shards}.")
  chunk = inter_dim // tp_shards
  aligned_half = layout.local_shape[w13_dim] // 2
  if aligned_half < chunk:
    raise ValueError(f"w13_weight half-dim {aligned_half} < chunk {chunk} (local_shape={layout.local_shape}).")

  gate = gate_e_h_i.reshape(num_experts, hidden_dim, tp_shards, chunk)
  up = up_e_h_i.reshape(num_experts, hidden_dim, tp_shards, chunk)
  if aligned_half > chunk:
    pad_cfg = ((0, 0), (0, 0), (0, 0), (0, aligned_half - chunk))
    gate = jnp.pad(gate, pad_cfg)
    up = jnp.pad(up, pad_cfg)
  fused = jnp.concatenate([gate, up], axis=3)
  total_w13 = tp_shards * 2 * aligned_half
  if flipped:
    return fused.reshape(num_experts, hidden_dim, total_w13).astype(layout.dtype)
  return jnp.transpose(fused, (0, 2, 3, 1)).reshape(num_experts, total_w13, hidden_dim).astype(layout.dtype)


def _pack_routed_experts_w2(
    down_e_i_h: jax.Array,
    layout: TargetParamLayout,
) -> jax.Array:
  """Packs routed expert down `[E, I, H]` into EP (`tp_shards=1`) or TP-MoE w2 layout."""
  flipped = True if layout.flipped is None else layout.flipped
  if len(layout.local_shape) != 3:
    raise ValueError(f"Invalid w2_weight local_shape={layout.local_shape}; expected 3D.")

  num_experts, inter_dim, hidden_dim = down_e_i_h.shape
  tp_shards = layout.num_shards if (layout.shard_domain == "tp" and layout.num_shards > 1) else 1
  if inter_dim % tp_shards != 0:
    raise ValueError(f"Expert inter_dim={inter_dim} is not divisible by tp_shards={tp_shards}.")
  chunk = inter_dim // tp_shards
  aligned_i = layout.local_shape[1 if flipped else 2]
  if aligned_i < chunk:
    raise ValueError(f"w2_weight dim {aligned_i} < chunk {chunk} (local_shape={layout.local_shape}).")

  down = down_e_i_h.reshape(num_experts, tp_shards, chunk, hidden_dim)
  if aligned_i > chunk:
    down = jnp.pad(down, ((0, 0), (0, 0), (0, aligned_i - chunk), (0, 0)))
  total_w2 = tp_shards * aligned_i
  if flipped:
    return down.reshape(num_experts, total_w2, hidden_dim).astype(layout.dtype)
  return jnp.transpose(down, (0, 3, 1, 2)).reshape(num_experts, hidden_dim, total_w2).astype(layout.dtype)


def _unscanned_shape(
    shape: Sequence[int],
    scan_axis: int,
    num_scan_dims: int,
) -> tuple[int, ...]:
  dims = [int(d) for d in shape]
  for _ in range(num_scan_dims):
    dims.pop(scan_axis)
  return tuple(dims)


class VllmTorchWeightConverter:
  """Streams MaxText Qwen3.5 trainer state into `vllm-torchtpu` `target_layout`."""

  def __init__(
      self,
      config: Any,
      *,
      tp: int = 1,
      dp: int | None = None,
      kv_tp_size: int | None = None,
      moe_mlp_tp_size: int | None = None,
      fp8_moe: bool | None = None,
      target_dtype: Any = None,
      debug: bool = False,
  ) -> None:
    raw_block = config.decoder_block.value if hasattr(config.decoder_block, "value") else str(config.decoder_block)
    if raw_block not in _SUPPORTED_DECODER_BLOCKS:
      raise ValueError(
          f"VllmTorchWeightConverter requires decoder_block in {sorted(_SUPPORTED_DECODER_BLOCKS)}, "
          f"got decoder_block={raw_block!r}."
      )
    effective_fp8_moe = bool(config.fp8_moe or config.rollout_fp8_moe) if fp8_moe is None else bool(fp8_moe)
    if effective_fp8_moe:
      raise ValueError("VllmTorchWeightConverter does not support fp8_moe / rollout_fp8_moe.")
    for unsupported_flag in ("moe_flat_fsdp_weights", "moe_use_fused_gmm3d", "fused_qkv"):
      if bool(getattr(config, unsupported_flag, False)):
        raise ValueError(f"VllmTorchWeightConverter does not support {unsupported_flag}=True.")

    self.config = config
    self.default_tp = max(1, int(tp))
    cfg_dp = getattr(config, "rollout_data_parallelism", 1)
    resolved_dp = dp if dp is not None else (cfg_dp if cfg_dp is not None and int(cfg_dp) > 0 else 1)
    if int(resolved_dp) < 1:
      raise ValueError(f"Expected dp >= 1, got dp={resolved_dp}.")
    self.default_dp = int(resolved_dp)
    self.kv_tp_size = int(kv_tp_size if kv_tp_size is not None else (config.kv_tp_size or 1))
    self.moe_mlp_tp_size = int(moe_mlp_tp_size if moe_mlp_tp_size is not None else (config.moe_mlp_tp_size or 1))
    self.enable_dp_attention = bool(config.enable_dp_attention)
    self.default_dtype = _resolve_jnp_dtype(
        target_dtype or (config.rollout_weight_dtype if config.rollout_weight_dtype else config.weight_dtype),
        jnp.bfloat16,
    )
    self.debug = bool(debug)
    self.num_layers = int(config.num_decoder_layers)
    self.cycle = int(config.inhomogeneous_layer_cycle_interval or 1)
    self.scan_axis = int(config.param_scan_axis)
    self.scan_layers = bool(config.scan_layers)
    self.emb_dim = int(config.emb_dim)
    self.head_dim = int(config.head_dim)
    self.base_num_kv_heads = int(config.num_kv_heads)
    scaled_moe_dim = config.moe_mlp_dim
    raw_moe_dim = scaled_moe_dim if (scaled_moe_dim is not None and int(scaled_moe_dim) > 0) else config.base_moe_mlp_dim
    self.moe_mlp_dim = int(raw_moe_dim) if (raw_moe_dim is not None and int(raw_moe_dim) > 0) else 0
    self.gdn_key_heads = int(config.gdn_num_key_heads)
    self.gdn_value_heads = int(config.gdn_num_value_heads)
    self.gdn_key_head_dim = int(config.gdn_key_head_dim)
    self.gdn_value_head_dim = int(config.gdn_value_head_dim)
    self._jit_cache: dict[tuple[Any, ...], Callable[..., dict[str, jax.Array]]] = {}

  def _run_jitted(
      self,
      cache_key: tuple[Any, ...],
      mesh: Mesh | None,
      fn: Callable[..., dict[str, jax.Array]],
      *args: jax.Array,
  ) -> dict[str, jax.Array]:
    """Compiles (once per signature + mesh) and executes `fn`."""
    full_key = (cache_key, mesh)
    jitted = self._jit_cache.get(full_key)
    if jitted is None:
      abstract_out = jax.eval_shape(fn, *args)
      out_shardings = (
          {k: _multi_axis_out_sharding(mesh, tuple(v.shape), param_name=k) for k, v in abstract_out.items()}
          if mesh is not None
          else None
      )
      jitted = jax.jit(fn, out_shardings=out_shardings) if out_shardings is not None else jax.jit(fn)
      self._jit_cache[full_key] = jitted
    outs = jitted(*args)
    if self.debug:
      jax.block_until_ready(list(outs.values()))
    return outs

  def _convert_full_attn(
      self,
      p: Mapping[str, jax.Array],
      layer_layouts: Mapping[str, TargetParamLayout],
      out: dict[str, jax.Array],
  ) -> None:
    """Converts GQA query/key/value/out projections and norms for one layer."""
    qk = p["self_attn.query.kernel"]
    kk = p["self_attn.key.kernel"]
    vk = p["self_attn.value.kernel"]
    if kk.shape[1] > self.base_num_kv_heads and kk.shape[1] % self.base_num_kv_heads == 0:
      step = kk.shape[1] // self.base_num_kv_heads
      kk = kk[:, ::step, :]
      vk = vk[:, ::step, :]

    qkv_layout = layer_layouts["self_attn.qkv_proj.weight"]
    kv_rep = max(1, qkv_layout.num_shards // self.base_num_kv_heads)
    out["self_attn.qkv_proj.weight"] = _pack_column_segments(
        [
            qk.reshape(self.emb_dim, -1).T,
            kk.reshape(self.emb_dim, -1).T,
            vk.reshape(self.emb_dim, -1).T,
        ],
        qkv_layout,
        default_reps=(1, kv_rep, kv_rep),
    )
    ok = p["self_attn.out.kernel"]
    o_in_out = ok if ok.ndim == 2 else ok.reshape(-1, self.emb_dim)
    out["self_attn.o_proj.weight"] = _pack_linear(
        o_in_out.T,
        layer_layouts["self_attn.o_proj.weight"],
    )
    if "self_attn.q_norm.weight" in layer_layouts:
      out["self_attn.q_norm.weight"] = p["self_attn.query_norm.scale"].astype(
          layer_layouts["self_attn.q_norm.weight"].dtype
      )
    if "self_attn.k_norm.weight" in layer_layouts:
      out["self_attn.k_norm.weight"] = p["self_attn.key_norm.scale"].astype(
          layer_layouts["self_attn.k_norm.weight"].dtype
      )

  def _convert_gdn_attn(
      self,
      p: Mapping[str, jax.Array],
      layer_layouts: Mapping[str, TargetParamLayout],
      out: dict[str, jax.Array],
  ) -> None:
    """Converts GDN linear-attention projections, conv1d, A_log, and norms."""
    hk = self.gdn_key_heads
    hv = self.gdn_value_heads
    dk = self.gdn_key_head_dim
    dv = self.gdn_value_head_dim
    vpk = hv // hk
    key_dim = hk * dk
    val_dim = hv * dv

    w_qkvz = p["gdn.in_proj_qkvz.kernel"].reshape(self.emb_dim, hk, 2 * dk + 2 * vpk * dv)
    qkvz_segs = [
        w_qkvz[:, :, :dk].reshape(self.emb_dim, key_dim).T,
        w_qkvz[:, :, dk : 2 * dk].reshape(self.emb_dim, key_dim).T,
        w_qkvz[:, :, 2 * dk : 2 * dk + vpk * dv].reshape(self.emb_dim, val_dim).T,
        w_qkvz[:, :, 2 * dk + vpk * dv :].reshape(self.emb_dim, val_dim).T,
    ]
    qkvz_layout = layer_layouts["linear_attn.in_proj_qkvz.weight"]
    kq_rep = max(1, qkvz_layout.num_shards // hk)
    out["linear_attn.in_proj_qkvz.weight"] = _pack_column_segments(
        qkvz_segs,
        qkvz_layout,
        default_reps=(kq_rep, kq_rep, 1, 1),
    )

    w_ba = p["gdn.in_proj_ba.kernel"].reshape(self.emb_dim, hk, 2 * vpk)
    out["linear_attn.in_proj_ba.weight"] = _pack_column_segments(
        [
            w_ba[:, :, :vpk].reshape(self.emb_dim, hv).T,
            w_ba[:, :, vpk:].reshape(self.emb_dim, hv).T,
        ],
        layer_layouts["linear_attn.in_proj_ba.weight"],
    )

    cw = jnp.transpose(p["gdn.conv1d.kernel"], (2, 1, 0))
    conv_layout = layer_layouts["linear_attn.conv1d.weight"]
    conv_rep = max(1, conv_layout.num_shards // hk)
    out["linear_attn.conv1d.weight"] = _pack_conv1d(
        [
            cw[:key_dim],
            cw[key_dim : 2 * key_dim],
            cw[2 * key_dim :],
        ],
        conv_layout,
        default_reps=(conv_rep, conv_rep, 1),
    )

    out["linear_attn.A_log"] = p["gdn.A_log"].astype(layer_layouts["linear_attn.A_log"].dtype)
    out["linear_attn.dt_bias"] = p["gdn.dt_bias"].astype(layer_layouts["linear_attn.dt_bias"].dtype)
    out["linear_attn.norm.weight"] = p["gdn.norm.scale"].astype(layer_layouts["linear_attn.norm.weight"].dtype)
    out_proj_in_out = p["gdn.out_proj.kernel"].reshape(-1, self.emb_dim)
    out["linear_attn.out_proj.weight"] = _pack_linear(
        out_proj_in_out.T,
        layer_layouts["linear_attn.out_proj.weight"],
    )

  def _convert_moe_mlp(
      self,
      p: Mapping[str, jax.Array],
      layer_layouts: Mapping[str, TargetParamLayout],
      out: dict[str, jax.Array],
  ) -> None:
    """Converts routed experts, router gate, and optional shared-expert weights."""
    out["mlp.gate.weight"] = _pack_linear(
        p["moe.gate.kernel"].T,
        layer_layouts["mlp.gate.weight"],
    )
    inter_i = self.moe_mlp_dim
    if inter_i <= 0:
      raise ValueError(f"moe_mlp_dim must be positive for routed MoE conversion, got {inter_i}.")
    if "moe.wi" in p:
      wi = p["moe.wi"]
      if wi.ndim != 3:
        raise ValueError(f"Expected 3-D routed MoE wi tensor [E, H, 2*I], got shape {wi.shape}.")
      half_pad = wi.shape[-1] // 2
      if half_pad < inter_i:
        raise ValueError(f"moe_mlp_dim={inter_i} exceeds prefused wi half-dim {half_pad} (shape={wi.shape}).")
      gate_e_h_i = wi[:, :, :inter_i]
      up_e_h_i = wi[:, :, half_pad : half_pad + inter_i]
    else:
      wi_0 = p["moe.wi_0"]
      wi_1 = p["moe.wi_1"]
      if wi_0.ndim != 3 or wi_1.ndim != 3:
        raise ValueError(f"Expected 3-D routed MoE wi_0/wi_1 [E, H, I], got {wi_0.shape} and {wi_1.shape}.")
      if wi_0.shape[-1] < inter_i or wi_1.shape[-1] < inter_i:
        raise ValueError(f"moe_mlp_dim={inter_i} exceeds wi_0/wi_1 last dim ({wi_0.shape}, {wi_1.shape}).")
      gate_e_h_i = wi_0[:, :, :inter_i]
      up_e_h_i = wi_1[:, :, :inter_i]

    wo = p["moe.wo"]
    if wo.ndim != 3 or wo.shape[1] < inter_i:
      raise ValueError(f"Expected 3-D routed MoE wo [E, >= {inter_i}, H], got shape {wo.shape}.")
    out["mlp.experts.w13_weight"] = _pack_routed_experts_w13(
        gate_e_h_i,
        up_e_h_i,
        layer_layouts["mlp.experts.w13_weight"],
    )
    out["mlp.experts.w2_weight"] = _pack_routed_experts_w2(
        wo[:, :inter_i, :],
        layer_layouts["mlp.experts.w2_weight"],
    )

    if "mlp.shared_expert.gate_up_proj.weight" in layer_layouts:
      out["mlp.shared_expert.gate_up_proj.weight"] = _pack_column_segments(
          [
              p["shared_expert.wi_0.kernel"].T,
              p["shared_expert.wi_1.kernel"].T,
          ],
          layer_layouts["mlp.shared_expert.gate_up_proj.weight"],
      )
      out["mlp.shared_expert.down_proj.weight"] = _pack_linear(
          p["shared_expert.wo.kernel"].T,
          layer_layouts["mlp.shared_expert.down_proj.weight"],
      )
    if "mlp.shared_expert_gate.weight" in layer_layouts:
      out["mlp.shared_expert_gate.weight"] = _pack_linear(
          p["shared_expert_gate.kernel"].T,
          layer_layouts["mlp.shared_expert_gate.weight"],
      )

  def _convert_layer_params(
      self,
      p: Mapping[str, jax.Array],
      layer_layouts: Mapping[str, TargetParamLayout],
  ) -> dict[str, jax.Array]:
    """Converts all parameters of a single decoder layer per `layer_layouts`."""
    out: dict[str, jax.Array] = {
        "input_layernorm.weight": p["input_layernorm.scale"].astype(layer_layouts["input_layernorm.weight"].dtype),
        "post_attention_layernorm.weight": p["post_attention_layernorm.scale"].astype(
            layer_layouts["post_attention_layernorm.weight"].dtype
        ),
    }
    if "self_attn.query.kernel" in p:
      self._convert_full_attn(p, layer_layouts, out)
    else:
      self._convert_gdn_attn(p, layer_layouts, out)

    if "moe.gate.kernel" in p:
      self._convert_moe_mlp(p, layer_layouts, out)
    else:
      out["mlp.gate_up_proj.weight"] = _pack_column_segments(
          [
              p["mlp.wi_0.kernel"].T,
              p["mlp.wi_1.kernel"].T,
          ],
          layer_layouts["mlp.gate_up_proj.weight"],
      )
      out["mlp.down_proj.weight"] = _pack_linear(
          p["mlp.wo.kernel"].T,
          layer_layouts["mlp.down_proj.weight"],
      )

    missing = set(layer_layouts) - set(out)
    if missing:
      raise ValueError(f"vllm_torch layer conversion missed target parameters: {sorted(missing)}")
    return out

  def _plan_layer_sources(self, flat_src: Mapping[str, jax.Array]) -> list[_LayerPlan]:
    """Builds canonical-subpath source lookup plans for all decoder layers."""
    by_container: dict[str, dict[str, str]] = {}
    unscanned: dict[int, dict[str, str]] = {}
    for full_key in flat_src:
      if not full_key.startswith("decoder."):
        continue
      head, sep, tail = full_key[len("decoder.") :].partition(".")
      if not sep:
        continue
      m_unscanned = _LAYERS_N_RE.match(head)
      if m_unscanned:
        unscanned.setdefault(int(m_unscanned.group(1)), {})[_normalize_layer_subpath(tail)] = full_key
      elif head == "layers":
        sub_head, sub_sep, sub_tail = tail.partition(".")
        if (sub_head in ("local_layers", "global_layer") or _LAYER_SLOT_RE.match(sub_head)) and sub_sep:
          by_container.setdefault(f"layers.{sub_head}", {})[_normalize_layer_subpath(sub_tail)] = full_key
        else:
          by_container.setdefault("layers", {})[_normalize_layer_subpath(tail)] = full_key
      elif head == "layers_remainder":
        sub_head, sub_sep, sub_tail = tail.partition(".")
        if sub_head in ("local_layers", "global_layer") and sub_sep:
          by_container.setdefault(f"layers_remainder.{sub_head}", {})[_normalize_layer_subpath(sub_tail)] = full_key

    if unscanned:
      return [
          _LayerPlan(
              layer_idx=idx,
              group_tag=f"unscanned_attn{int('self_attn.query.kernel' in unscanned[idx])}",
              scan_indices=(),
              key_map=unscanned[idx],
          )
          for idx in sorted(unscanned)
      ]

    num_full_blocks = self.num_layers // self.cycle if self.scan_layers else 0
    full_layers = num_full_blocks * self.cycle
    nested_keys = (
        "layers.local_layers",
        "layers.global_layer",
        "layers_remainder.local_layers",
        "layers_remainder.global_layer",
    )
    if any(k in by_container for k in nested_keys):
      plans: list[_LayerPlan] = []
      for idx in range(full_layers):
        block, slot = divmod(idx, self.cycle)
        if slot == self.cycle - 1:
          plans.append(_LayerPlan(idx, "nested_global", (block,), by_container["layers.global_layer"]))
        else:
          plans.append(_LayerPlan(idx, "nested_local", (block, slot), by_container["layers.local_layers"]))
      for idx in range(full_layers, self.num_layers):
        rem_slot = idx - full_layers
        if rem_slot == self.cycle - 1 and "layers_remainder.global_layer" in by_container:
          plans.append(_LayerPlan(idx, "rem_global", (), by_container["layers_remainder.global_layer"]))
        else:
          plans.append(_LayerPlan(idx, "rem_local", (rem_slot,), by_container["layers_remainder.local_layers"]))
      return plans

    if any(k.startswith("layers.layer_") for k in by_container):
      return [
          _LayerPlan(
              layer_idx=idx,
              group_tag=f"slot_{idx % self.cycle}",
              scan_indices=(idx // self.cycle,) if self.scan_layers else (),
              key_map=by_container[f"layers.layer_{idx % self.cycle}"],
          )
          for idx in range(self.num_layers)
      ]
    if "layers" in by_container:
      return [
          _LayerPlan(
              layer_idx=idx,
              group_tag="homogeneous_layers",
              scan_indices=(idx,) if self.scan_layers else (),
              key_map=by_container["layers"],
          )
          for idx in range(self.num_layers)
      ]
    raise ValueError(f"Could not locate decoder layers in MaxText state; sample keys: {sorted(flat_src)[:10]}")

  def _synth_full_attn_layout(
      self,
      layer_idx: int,
      pfx: str,
      key_map: Mapping[str, str],
      shape_of: Callable[[str], tuple[int, ...]],
      layouts: dict[str, TargetParamLayout],
  ) -> None:
    """Synthesizes default `TargetParamLayout` entries for a GQA attention block."""
    tp = self.default_tp
    h = self.emb_dim
    dtype = self.default_dtype
    q_shape = shape_of("self_attn.query.kernel")
    num_q_heads = q_shape[1]
    q_total = num_q_heads * q_shape[2]
    nkv = self.base_num_kv_heads
    if num_q_heads % tp != 0 or (nkv >= tp and nkv % tp != 0) or (nkv < tp and tp % nkv != 0):
      raise ValueError(f"Layer {layer_idx} GQA heads (num_q_heads={num_q_heads}, nkv={nkv}) incompatible with tp={tp}.")
    kv_rep = max(1, tp // nkv)
    kv_local = max(1, nkv // tp) * self.head_dim
    layouts[pfx + "self_attn.qkv_proj.weight"] = TargetParamLayout.create(
        [h, q_total // tp + 2 * kv_local],
        dtype=dtype,
        shard_dim=1,
        num_shards=tp,
        shard_domain="tp",
        flipped=True,
        replication_factors=(1, kv_rep, kv_rep),
    )
    o_in = int(math.prod(shape_of("self_attn.out.kernel")[:-1]))
    if o_in % tp != 0:
      raise ValueError(f"Layer {layer_idx} o_proj input dim {o_in} is not divisible by tp={tp}.")
    layouts[pfx + "self_attn.o_proj.weight"] = TargetParamLayout.create(
        [o_in // tp, h],
        dtype=dtype,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
        flipped=True,
    )
    if "self_attn.query_norm.scale" in key_map:
      layouts[pfx + "self_attn.q_norm.weight"] = TargetParamLayout.create([self.head_dim], dtype=dtype)
    if "self_attn.key_norm.scale" in key_map:
      layouts[pfx + "self_attn.k_norm.weight"] = TargetParamLayout.create([self.head_dim], dtype=dtype)

  def _synth_gdn_layout(
      self,
      layer_idx: int,
      pfx: str,
      shape_of: Callable[[str], tuple[int, ...]],
      layouts: dict[str, TargetParamLayout],
  ) -> None:
    """Synthesizes default `TargetParamLayout` entries for a GDN linear-attention block."""
    tp = self.default_tp
    h = self.emb_dim
    dtype = self.default_dtype
    hk = self.gdn_key_heads
    hv = self.gdn_value_heads
    dk = self.gdn_key_head_dim
    dv = self.gdn_value_head_dim
    if hv % tp != 0 or (hk >= tp and hk % tp != 0) or (hk < tp and tp % hk != 0):
      raise ValueError(f"Layer {layer_idx} GDN heads (hk={hk}, hv={hv}) incompatible with tp={tp}.")
    k_local = max(1, hk // tp)
    v_local = hv // tp
    kq_rep = max(1, tp // hk)
    conv_k = shape_of("gdn.conv1d.kernel")[0]
    layouts[pfx + "linear_attn.in_proj_qkvz.weight"] = TargetParamLayout.create(
        [2 * k_local * dk + 2 * v_local * dv, h],
        dtype=dtype,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
        flipped=False,
        replication_factors=(kq_rep, kq_rep, 1, 1),
    )
    layouts[pfx + "linear_attn.in_proj_ba.weight"] = TargetParamLayout.create(
        [h, 2 * v_local],
        dtype=dtype,
        shard_dim=1,
        num_shards=tp,
        shard_domain="tp",
        flipped=True,
        replication_factors=(1, 1),
    )
    layouts[pfx + "linear_attn.conv1d.weight"] = TargetParamLayout.create(
        [2 * k_local * dk + v_local * dv, 1, conv_k],
        dtype=dtype,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
        replication_factors=(kq_rep, kq_rep, 1),
    )
    layouts[pfx + "linear_attn.A_log"] = TargetParamLayout.create(
        [v_local],
        dtype=jnp.float32,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
    )
    layouts[pfx + "linear_attn.dt_bias"] = TargetParamLayout.create(
        [v_local],
        dtype=dtype,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
    )
    layouts[pfx + "linear_attn.norm.weight"] = TargetParamLayout.create([dv], dtype=dtype)
    layouts[pfx + "linear_attn.out_proj.weight"] = TargetParamLayout.create(
        [v_local * dv, h],
        dtype=dtype,
        shard_dim=0,
        num_shards=tp,
        shard_domain="tp",
        flipped=True,
    )

  def _synth_mlp_layout(
      self,
      layer_idx: int,
      pfx: str,
      key_map: Mapping[str, str],
      shape_of: Callable[[str], tuple[int, ...]],
      layouts: dict[str, TargetParamLayout],
  ) -> None:
    """Synthesizes default `TargetParamLayout` entries for an MoE or dense MLP block."""
    tp = self.default_tp
    h = self.emb_dim
    dtype = self.default_dtype
    if "moe.gate.kernel" not in key_map:
      dense_i = shape_of("mlp.wi_0.kernel")[1]
      if dense_i % tp != 0:
        raise ValueError(f"Layer {layer_idx} dense MLP dim {dense_i} not divisible by tp={tp}.")
      layouts[pfx + "mlp.gate_up_proj.weight"] = TargetParamLayout.create(
          [h, 2 * (dense_i // tp)],
          dtype=dtype,
          shard_dim=1,
          num_shards=tp,
          shard_domain="tp",
          flipped=True,
          replication_factors=(1, 1),
      )
      layouts[pfx + "mlp.down_proj.weight"] = TargetParamLayout.create(
          [dense_i // tp, h],
          dtype=dtype,
          shard_dim=0,
          num_shards=tp,
          shard_domain="tp",
          flipped=True,
      )
      return

    if self.moe_mlp_dim <= 0:
      raise ValueError(f"moe_mlp_dim must be positive for routed MoE layer {layer_idx}, got {self.moe_mlp_dim}.")
    if self.moe_mlp_tp_size not in (1, tp):
      raise ValueError(
          f"Cannot synthesize default layout for layer {layer_idx} when moe_mlp_tp_size={self.moe_mlp_tp_size} "
          f"is not in {{1, tp={tp}}}."
      )
    num_experts = shape_of("moe.gate.kernel")[1]
    layouts[pfx + "mlp.gate.weight"] = TargetParamLayout.create([num_experts, h], dtype=dtype)
    if self.moe_mlp_tp_size == 1:
      ep_size = self.default_dp * tp
      if num_experts % ep_size != 0:
        raise ValueError(f"Layer {layer_idx} num_experts={num_experts} not divisible by ep_size={ep_size}.")
      aligned_i = pad_to_tpu_lanes(self.moe_mlp_dim)
      layouts[pfx + "mlp.experts.w13_weight"] = TargetParamLayout.create(
          [num_experts // ep_size, h, 2 * aligned_i],
          dtype=dtype,
          shard_dim=0,
          num_shards=ep_size,
          shard_domain="ep",
          flipped=True,
      )
      layouts[pfx + "mlp.experts.w2_weight"] = TargetParamLayout.create(
          [num_experts // ep_size, aligned_i, h],
          dtype=dtype,
          shard_dim=0,
          num_shards=ep_size,
          shard_domain="ep",
          flipped=True,
      )
    else:
      moe_tp = self.moe_mlp_tp_size
      if self.moe_mlp_dim % moe_tp != 0:
        raise ValueError(f"Layer {layer_idx} moe_mlp_dim={self.moe_mlp_dim} not divisible by moe_mlp_tp_size={moe_tp}.")
      aligned_half = pad_to_tpu_lanes(self.moe_mlp_dim // moe_tp)
      layouts[pfx + "mlp.experts.w13_weight"] = TargetParamLayout.create(
          [num_experts, h, 2 * aligned_half],
          dtype=dtype,
          shard_dim=2,
          num_shards=moe_tp,
          shard_domain="tp",
          flipped=True,
      )
      layouts[pfx + "mlp.experts.w2_weight"] = TargetParamLayout.create(
          [num_experts, aligned_half, h],
          dtype=dtype,
          shard_dim=1,
          num_shards=moe_tp,
          shard_domain="tp",
          flipped=True,
      )

    if "shared_expert.wi_0.kernel" in key_map:
      shared_i = shape_of("shared_expert.wi_0.kernel")[1]
      if shared_i % tp != 0:
        raise ValueError(f"Layer {layer_idx} shared_expert dim {shared_i} not divisible by tp={tp}.")
      layouts[pfx + "mlp.shared_expert.gate_up_proj.weight"] = TargetParamLayout.create(
          [h, 2 * (shared_i // tp)],
          dtype=dtype,
          shard_dim=1,
          num_shards=tp,
          shard_domain="tp",
          flipped=True,
          replication_factors=(1, 1),
      )
      layouts[pfx + "mlp.shared_expert.down_proj.weight"] = TargetParamLayout.create(
          [shared_i // tp, h],
          dtype=dtype,
          shard_dim=0,
          num_shards=tp,
          shard_domain="tp",
          flipped=True,
      )
    if "shared_expert_gate.kernel" in key_map:
      layouts[pfx + "mlp.shared_expert_gate.weight"] = TargetParamLayout.create([1, h], dtype=dtype)

  def _synthesize_default_target_layout(
      self,
      flat_src: Mapping[str, jax.Array],
      layer_plans: Sequence[_LayerPlan],
  ) -> dict[str, TargetParamLayout]:
    """Synthesizes canonical `vllm-torchtpu` Qwen3.5 `TargetParamLayout`s."""
    tp = self.default_tp
    h = self.emb_dim
    dtype = self.default_dtype
    if self.enable_dp_attention and tp > 1:
      raise ValueError(f"Cannot synthesize default layout when enable_dp_attention=True and tp={tp} > 1.")
    if self.kv_tp_size not in (1, tp):
      raise ValueError(f"Cannot synthesize default layout when kv_tp_size={self.kv_tp_size} not in {{1, tp={tp}}}.")

    layouts: dict[str, TargetParamLayout] = {}
    if "token_embedder.embedding" in flat_src:
      vocab_size = int(flat_src["token_embedder.embedding"].shape[0])
      if vocab_size % tp != 0:
        raise ValueError(f"vocab_size={vocab_size} is not divisible by rollout tp={tp}.")
      layouts["embed_tokens.weight"] = TargetParamLayout.create(
          [vocab_size // tp, h],
          dtype=dtype,
          shard_dim=0,
          num_shards=tp,
          shard_domain="tp",
      )
    if "decoder.decoder_norm.scale" in flat_src:
      layouts["norm.weight"] = TargetParamLayout.create([h], dtype=dtype)
    if "decoder.logits_dense.kernel" in flat_src:
      lm_vocab = int(flat_src["decoder.logits_dense.kernel"].shape[1])
      if lm_vocab % tp != 0:
        raise ValueError(f"lm_head vocab_size={lm_vocab} is not divisible by rollout tp={tp}.")
      layouts["lm_head.weight"] = TargetParamLayout.create(
          [lm_vocab // tp, h],
          dtype=dtype,
          shard_dim=0,
          num_shards=tp,
          shard_domain="tp",
      )

    for plan in layer_plans:
      pfx = f"layers.{plan.layer_idx}."
      num_scan = len(plan.scan_indices)

      def _shape_of(
          sub_k: str,
          _km: Mapping[str, str] = plan.key_map,
          _ns: int = num_scan,
      ) -> tuple[int, ...]:
        return _unscanned_shape(flat_src[_km[sub_k]].shape, self.scan_axis, _ns)

      layouts[pfx + "input_layernorm.weight"] = TargetParamLayout.create([h], dtype=dtype)
      layouts[pfx + "post_attention_layernorm.weight"] = TargetParamLayout.create([h], dtype=dtype)
      if "self_attn.query.kernel" in plan.key_map:
        self._synth_full_attn_layout(plan.layer_idx, pfx, plan.key_map, _shape_of, layouts)
      else:
        self._synth_gdn_layout(plan.layer_idx, pfx, _shape_of, layouts)
      self._synth_mlp_layout(plan.layer_idx, pfx, plan.key_map, _shape_of, layouts)
    return layouts

  def _resolve_target_layouts(
      self,
      flat_src: Mapping[str, jax.Array],
      layer_plans: Sequence[_LayerPlan],
      target_layout: Mapping[str, Mapping[str, Any]] | None,
  ) -> dict[str, TargetParamLayout]:
    """Resolves typed `TargetParamLayout`s and checks source parameter dtypes."""
    if target_layout:
      layouts = {
          name: TargetParamLayout.from_mapping(name, spec, self.default_dtype) for name, spec in target_layout.items()
      }
    else:
      layouts = self._synthesize_default_target_layout(flat_src, layer_plans)

    checked_keys = [k for k in _TOP_LEVEL_SRC_KEYS if k in flat_src]
    for plan in layer_plans:
      checked_keys.extend(plan.key_map.values())
    for src_key in checked_keys:
      if jnp.dtype(flat_src[src_key].dtype) not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(f"Unsupported source parameter dtype {flat_src[src_key].dtype} on {src_key}.")
    return layouts

  def _convert_top_level(
      self,
      flat_src: dict[str, jax.Array],
      top_layouts: Mapping[str, TargetParamLayout],
      mesh: Mesh | None,
      result_flat: dict[tuple[str, ...], Any],
  ) -> None:
    """Converts embeddings, final norm, and lm_head and pops their source refs."""
    top_src_keys = [k for k in _TOP_LEVEL_SRC_KEYS if k in flat_src]

    def _top_fn(*arrs: jax.Array) -> dict[str, jax.Array]:
      p = dict(zip(top_src_keys, arrs))
      out: dict[str, jax.Array] = {}
      if "embed_tokens.weight" in top_layouts:
        out["embed_tokens.weight"] = p["token_embedder.embedding"].astype(top_layouts["embed_tokens.weight"].dtype)
      if "norm.weight" in top_layouts:
        out["norm.weight"] = p["decoder.decoder_norm.scale"].astype(top_layouts["norm.weight"].dtype)
      if "lm_head.weight" in top_layouts:
        lm_src = (
            p["decoder.logits_dense.kernel"].T if "decoder.logits_dense.kernel" in p else p["token_embedder.embedding"]
        )
        out["lm_head.weight"] = lm_src.astype(top_layouts["lm_head.weight"].dtype)
      return out

    layout_sig = tuple((k, top_layouts[k]) for k in sorted(top_layouts))
    top_outs = self._run_jitted(
        ("top", tuple(top_src_keys), layout_sig),
        mesh,
        _top_fn,
        *[flat_src[k] for k in top_src_keys],
    )
    for k in top_src_keys:
      flat_src.pop(k, None)
    for out_name, arr in top_outs.items():
      expected = top_layouts[out_name].global_shape
      if tuple(arr.shape) != expected:
        raise ValueError(f"Shape mismatch on {out_name}: produced {tuple(arr.shape)}, expected {expected}.")
      result_flat[tuple(out_name.split("."))] = nnx.Param(arr)

  def _convert_layers(
      self,
      flat_src: dict[str, jax.Array],
      layer_plans: Sequence[_LayerPlan],
      layouts: Mapping[str, TargetParamLayout],
      mesh: Mesh | None,
      result_flat: dict[tuple[str, ...], Any],
  ) -> None:
    """Streams decoder layers one by one inside cached per-group `jax.jit`s."""
    last_step_for_src = {src_k: step for step, plan in enumerate(layer_plans) for src_k in plan.key_map.values()}
    for step_idx, plan in enumerate(layer_plans):
      prefix = f"layers.{plan.layer_idx}."
      layer_layouts = {k[len(prefix) :]: v for k, v in layouts.items() if k.startswith(prefix)}
      if not layer_layouts:
        raise ValueError(f"No target_layout entries found for layer {plan.layer_idx} (prefix {prefix!r}).")
      sub_names = sorted(plan.key_map)

      def _layer_fn(
          *all_args: jax.Array,
          _sub_names: list[str] = sub_names,
          _layouts: Mapping[str, TargetParamLayout] = layer_layouts,
      ) -> dict[str, jax.Array]:
        arrs = all_args[: len(_sub_names)]
        idx_scalars = all_args[len(_sub_names) :]
        sliced: dict[str, jax.Array] = {}
        for sub_k, val in zip(_sub_names, arrs):
          cur = val
          for idx_val in idx_scalars:
            cur = jax.lax.dynamic_index_in_dim(cur, idx_val, axis=self.scan_axis, keepdims=False)
          sliced[sub_k] = cur
        return self._convert_layer_params(sliced, _layouts)

      call_args = [flat_src[plan.key_map[s]] for s in sub_names] + [jnp.int32(idx_val) for idx_val in plan.scan_indices]
      layout_sig = tuple((k, layer_layouts[k]) for k in sorted(layer_layouts))
      cache_key = (plan.group_tag, tuple(sub_names), len(plan.scan_indices), layout_sig)
      layer_outs = self._run_jitted(cache_key, mesh, _layer_fn, *call_args)

      for src_key in plan.key_map.values():
        if last_step_for_src.get(src_key) == step_idx:
          flat_src.pop(src_key, None)
      for sub_out_name, arr in layer_outs.items():
        full_out_name = prefix + sub_out_name
        expected = layer_layouts[sub_out_name].global_shape
        if tuple(arr.shape) != expected:
          raise ValueError(f"Shape mismatch on {full_out_name}: produced {tuple(arr.shape)}, expected {expected}.")
        result_flat[tuple(full_out_name.split("."))] = nnx.Param(arr)

  @staticmethod
  def _verify_layer_indices(
      unflattened: dict[str, Any],
      layouts: Mapping[str, TargetParamLayout],
  ) -> None:
    """Verifies JAX pytree traversal order matches rollout `layer_idx` metadata."""
    if not any(spec.layer_idx is not None for spec in layouts.values()):
      return
    leaves = jax.tree_util.tree_leaves_with_path(unflattened, is_leaf=lambda x: isinstance(x, nnx.Variable))
    for actual_idx, (key_path, _) in enumerate(leaves):
      out_name = ".".join(str(getattr(k, "key", k)) for k in key_path)
      expected_idx = layouts[out_name].layer_idx
      if expected_idx is not None and expected_idx != actual_idx:
        raise ValueError(f"Parameter layer_idx mismatch on {out_name}: traversal index {actual_idx} != {expected_idx}.")

  def convert(
      self,
      src_pytree: Any,
      *,
      target_layout: Mapping[str, Mapping[str, Any]] | None = None,
  ) -> dict[str, Any]:
    """Converts MaxText state into a canonical nested dict of `nnx.Param`s."""
    flat_src = _to_pure_flat(src_pytree)
    mesh = _infer_mesh(flat_src)
    layer_plans = self._plan_layer_sources(flat_src)
    layouts = self._resolve_target_layouts(flat_src, layer_plans, target_layout)

    result_flat: dict[tuple[str, ...], Any] = {}
    top_layouts = {k: v for k, v in layouts.items() if not k.startswith("layers.")}
    self._convert_top_level(flat_src, top_layouts, mesh, result_flat)
    self._convert_layers(flat_src, layer_plans, layouts, mesh, result_flat)

    flat_src.clear()
    gc.collect()

    missing_total = set(layouts) - {".".join(k) for k in result_flat}
    if missing_total:
      raise ValueError(
          f"vllm_torch conversion missed {len(missing_total)} target parameters: {sorted(missing_total)[:5]}"
      )
    unflattened = traverse_util.unflatten_dict(result_flat)
    self._verify_layer_indices(unflattened, layouts)
    return unflattened
