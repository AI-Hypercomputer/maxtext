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

"""Shared attention kernel dispatcher for m3 models.

Models resolve a frozen `AttentionKernel` descriptor from `(config, sharding)`
once at construction and invoke it during `__call__`:

    self.attention_kernel = kernels.attention_from_config(config, self.sharding)
    ...
    out = self.attention_kernel(query, key, value, segment_ids=..., positions=...)

Resolving `PartitionSpec` axes from `Sharding` at construction avoids reading
`jax.typeof(x).sharding` at trace time, which reports replicated `P(None, ...)`
on `Auto` mesh axes inside `jax.jit` and `ScannedLayers`. Because
`AttentionKernel` and `SplashConfig` are frozen dataclasses, Flax NNX treats
them as static module attributes across `nnx.split`, `nnx.merge`, `jax.jit`,
and layer scan transforms.
"""

import dataclasses
import functools
import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as mtk_splash
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as mtk_mask
from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType

AttentionBackend = Literal["xla", "splash"]

_SPLASH_TILE_ALIGNMENT = 128
_VALID_ATTENTION_MODES = ("dot_product", "flash", "autoselected")
_VALID_ATTENTION_TYPES = ("global", "local_sliding", "mla")
_VALID_QKV_LAYOUTS = ("HEAD_DIM_MINOR", "SEQ_MINOR")
_KNOWN_MESH_AXIS_NAMES = frozenset(
    {
        "data",
        "fsdp",
        "fsdp_transpose",
        "expert",
        "context",
        "stage",
        "tensor",
    }
)


def mesh_axis_names(axes: MeshAxes) -> tuple[str, ...]:
  """Normalizes one `PartitionSpec` entry into a tuple of mesh axis names."""
  if axes is None:
    return ()
  return (axes,) if isinstance(axes, str) else tuple(axes)


def resolve_mesh_axes(mesh: Any, axes: MeshAxes) -> MeshAxes:
  """Filters standard m3 mesh axes to `mesh.axis_names`, failing closed on unknown axes."""
  present = set(getattr(mesh, "axis_names", ()))
  kept: list[str] = []
  for name in mesh_axis_names(axes):
    if name in present:
      kept.append(name)
    elif name not in _KNOWN_MESH_AXIS_NAMES:
      raise ValueError(f"Unknown mesh axis {name!r}; active mesh axes are {tuple(getattr(mesh, 'axis_names', ()))}.")
  if not kept:
    return None
  return kept[0] if len(kept) == 1 else tuple(kept)


def mesh_axis_size(mesh: Any, axes: MeshAxes) -> int:
  """Returns the product of shard counts across `axes` on `mesh`."""
  resolved = resolve_mesh_axes(mesh, axes)
  return math.prod(int(mesh.shape[name]) for name in mesh_axis_names(resolved))


def mesh_in_scope() -> Any:
  """Returns the active abstract or concrete JAX mesh, raising if none is set."""
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty:
    return abstract_mesh
  try:
    mesh = jax.sharding.get_mesh()
  except ValueError:
    mesh = None
  if mesh is not None and not mesh.empty:
    return mesh
  raise ValueError(
      "Fused kernels run under jax.shard_map and require an active mesh; "
      "call inside `with jax.set_mesh(mesh):` or `with mesh:`."
  )


def is_tpu_mesh(mesh: Any) -> bool:
  """Returns True when `mesh` targets real or virtual AOT TPU devices."""
  abstract_device = getattr(mesh, "abstract_device", None)
  if abstract_device is not None:
    return getattr(abstract_device, "platform", None) == "tpu"
  try:
    devices = getattr(mesh, "devices", None)
  except ValueError:
    devices = None
  if devices is not None and getattr(devices, "size", 0) > 0:
    return getattr(devices.flat[0], "platform", None) == "tpu"
  return jax.default_backend() == "tpu"


def _normalize_layout_name(layout: Any, field_name: str) -> str:
  """Validates a Splash QKV layout setting and returns its canonical uppercase name."""
  if isinstance(layout, mtk_splash.QKVLayout):
    return layout.name
  name = str(getattr(layout, "name", getattr(layout, "value", layout))).upper()
  if name not in _VALID_QKV_LAYOUTS:
    raise ValueError(f"{field_name} must be one of {_VALID_QKV_LAYOUTS}; got {layout!r}.")
  return name


def _parse_qkv_layout(layout: str, field_name: str) -> mtk_splash.QKVLayout:
  """Converts a validated layout string into `mtk_splash.QKVLayout`."""
  return mtk_splash.QKVLayout[_normalize_layout_name(layout, field_name)]


def _clamp_splash_block(block: int, seq_len: int, field_name: str) -> int:
  """Clamps a Splash tile size to `seq_len` and validates 128-token alignment."""
  resolved = min(int(block), int(seq_len))
  if resolved < _SPLASH_TILE_ALIGNMENT or resolved % _SPLASH_TILE_ALIGNMENT != 0:
    raise ValueError(
        f"Splash {field_name} resolved to {resolved} (configured={block}, seq_len={seq_len}), "
        f"which must be a positive multiple of {_SPLASH_TILE_ALIGNMENT}."
    )
  return resolved


def _configured_sequence_length(config: Any) -> int:
  """Reads the target sequence length from legacy or m3 config fields."""
  seq_len = getattr(config, "max_target_length", None)
  if seq_len is None:
    seq_len = getattr(config, "max_sequence_length", 0)
  return int(seq_len or 0)


@dataclasses.dataclass(frozen=True)
class SplashConfig:
  """Static Splash attention tile, layout, and numerical settings."""

  block_q: int = 512
  block_kv: int = 512
  block_kv_compute: int = 512
  block_q_dkv: int = 512
  block_kv_dkv: int = 512
  block_kv_dkv_compute: int = 512
  q_layout: str = "HEAD_DIM_MINOR"
  k_layout: str = "HEAD_DIM_MINOR"
  v_layout: str = "HEAD_DIM_MINOR"
  fuse_reciprocal: bool = True
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  use_experimental_scheduler: bool = False
  bwd_dkv_megacore: bool = False
  residual_checkpoint_name: str | None = "context"

  def __post_init__(self) -> None:
    for field_name in (
        "block_q",
        "block_kv",
        "block_kv_compute",
        "block_q_dkv",
        "block_kv_dkv",
        "block_kv_dkv_compute",
    ):
      val = getattr(self, field_name)
      if int(val) < _SPLASH_TILE_ALIGNMENT or int(val) % _SPLASH_TILE_ALIGNMENT != 0:
        raise ValueError(
            f"SplashConfig.{field_name} must be a positive multiple of {_SPLASH_TILE_ALIGNMENT}; got {val!r}."
        )
    object.__setattr__(self, "q_layout", _normalize_layout_name(self.q_layout, "q_layout"))
    object.__setattr__(self, "k_layout", _normalize_layout_name(self.k_layout, "k_layout"))
    object.__setattr__(self, "v_layout", _normalize_layout_name(self.v_layout, "v_layout"))
    if self.max_logit_const is not None and float(self.max_logit_const) <= 0.0:
      raise ValueError(f"SplashConfig.max_logit_const must be > 0 when set; got {self.max_logit_const!r}.")

  @classmethod
  def from_config(cls, config: Any, *, local: bool = False) -> "SplashConfig":
    """Extracts global or local-window Splash settings from `config`."""

    def _pick(local_attr: str, global_attr: str, default: Any) -> Any:
      if local:
        local_val = getattr(config, local_attr, None)
        if local_val is not None:
          return local_val
      val = getattr(config, global_attr, default)
      return default if val is None else val

    max_logit_est = float(getattr(config, "use_max_logit_estimate", -1.0) or -1.0)
    max_logit_const = max_logit_est if max_logit_est > 0.0 else None

    return cls(
        block_q=int(_pick("local_sa_block_q", "sa_block_q", 512)),
        block_kv=int(_pick("local_sa_block_kv", "sa_block_kv", 512)),
        block_kv_compute=int(_pick("local_sa_block_kv_compute", "sa_block_kv_compute", 512)),
        block_q_dkv=int(_pick("local_sa_block_q_dkv", "sa_block_q_dkv", 512)),
        block_kv_dkv=int(_pick("local_sa_block_kv_dkv", "sa_block_kv_dkv", 512)),
        block_kv_dkv_compute=int(_pick("local_sa_block_kv_dkv_compute", "sa_block_kv_dkv_compute", 512)),
        q_layout=_pick("local_sa_q_layout", "sa_q_layout", "HEAD_DIM_MINOR"),
        k_layout=_pick("local_sa_k_layout", "sa_k_layout", "HEAD_DIM_MINOR"),
        v_layout=_pick("local_sa_v_layout", "sa_v_layout", "HEAD_DIM_MINOR"),
        fuse_reciprocal=bool(_pick("local_sa_fuse_reciprocal", "sa_fuse_reciprocal", True)),
        use_base2_exp=bool(_pick("local_sa_use_base2_exp", "sa_use_base2_exp", True)),
        max_logit_const=max_logit_const,
        use_experimental_scheduler=bool(_pick("local_use_splash_scheduler", "use_splash_scheduler", False)),
        bwd_dkv_megacore=bool(getattr(config, "sa_bwd_dkv_megacore", False)),
    )

  def resolve(
      self,
      q_length: int,
      kv_length: int,
      *,
      attn_logits_soft_cap: float | None,
      interpret: bool,
  ) -> mtk_splash.SplashConfig:
    """Builds the Pallas `mtk_splash.SplashConfig` clamped to `(q_length, kv_length)`."""
    block_q = _clamp_splash_block(self.block_q, q_length, "block_q")
    block_kv = _clamp_splash_block(self.block_kv, kv_length, "block_kv")
    block_kv_compute = _clamp_splash_block(min(self.block_kv_compute, block_kv), kv_length, "block_kv_compute")
    block_q_dkv = _clamp_splash_block(self.block_q_dkv, q_length, "block_q_dkv")
    block_kv_dkv = _clamp_splash_block(self.block_kv_dkv, kv_length, "block_kv_dkv")
    block_kv_dkv_compute = _clamp_splash_block(
        min(self.block_kv_dkv_compute, block_kv_dkv), kv_length, "block_kv_dkv_compute"
    )
    return mtk_splash.SplashConfig(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        use_fused_bwd_kernel=True,
        q_layout=_parse_qkv_layout(self.q_layout, "q_layout"),
        k_layout=_parse_qkv_layout(self.k_layout, "k_layout"),
        v_layout=_parse_qkv_layout(self.v_layout, "v_layout"),
        residual_checkpoint_name=self.residual_checkpoint_name,
        attn_logits_soft_cap=attn_logits_soft_cap,
        fuse_reciprocal=self.fuse_reciprocal,
        use_base2_exp=self.use_base2_exp,
        max_logit_const=self.max_logit_const,
        interpret=interpret,
        use_experimental_scheduler=self.use_experimental_scheduler,
        bwd_dkv_megacore=self.bwd_dkv_megacore,
    )


@dataclasses.dataclass(frozen=True)
class AttentionKernel:
  """Static attention kernel descriptor resolved from `(config, sharding)` at init."""

  backend: AttentionBackend = "xla"
  batch: MeshAxes = None
  heads: MeshAxes = None
  kv_heads: MeshAxes = None
  length: MeshAxes = None
  kv_length: MeshAxes = None
  causal: bool = True
  sliding_window: int | None = None
  attn_logits_soft_cap: float | None = None
  float32_logits: bool = True
  gqa_head_stacking: bool = False
  splash: SplashConfig = SplashConfig()

  def __post_init__(self) -> None:
    if self.backend not in ("xla", "splash"):
      raise ValueError(f"AttentionKernel.backend must be 'xla' or 'splash'; got {self.backend!r}.")
    if self.sliding_window is not None and int(self.sliding_window) <= 0:
      raise ValueError(f"AttentionKernel.sliding_window must be > 0 when set; got {self.sliding_window!r}.")
    if self.attn_logits_soft_cap is not None and float(self.attn_logits_soft_cap) <= 0.0:
      raise ValueError(f"AttentionKernel.attn_logits_soft_cap must be > 0 when set; got {self.attn_logits_soft_cap!r}.")
    if self.gqa_head_stacking and self.backend != "splash":
      raise ValueError("gqa_head_stacking=True requires backend='splash'.")

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      *,
      segment_ids: jax.Array | None = None,
      positions: jax.Array | None = None,
      sinks: jax.Array | None = None,
      max_logit_value: jax.Array | float | None = None,
  ) -> jax.Array:
    """Executes attention with the statically resolved backend and sharding axes."""
    return dot_product_attention(
        query,
        key,
        value,
        kernel=self,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )


XLA_ATTENTION = AttentionKernel("xla")


def validate_attention_config(config: Any) -> None:
  """Validates m3 attention kernel settings eagerly before model construction."""
  attention_mode = getattr(config, "attention", "autoselected")
  if attention_mode not in _VALID_ATTENTION_MODES:
    raise ValueError(f"m3 supports attention in {_VALID_ATTENTION_MODES}; got {attention_mode!r}.")

  attention_type = getattr(config, "attention_type", "global")
  attention_type = getattr(attention_type, "value", attention_type)
  if attention_type not in _VALID_ATTENTION_TYPES:
    raise ValueError(f"m3 supports attention_type in {_VALID_ATTENTION_TYPES}; got {attention_type!r}.")

  if attention_type == "local_sliding":
    window_size = getattr(config, "sliding_window_size", 0)
    if window_size is None or int(window_size) <= 0:
      raise ValueError(f"attention_type='local_sliding' requires sliding_window_size > 0; got {window_size!r}.")

  for field_name in ("sa_q_layout", "sa_k_layout", "sa_v_layout"):
    val = getattr(config, field_name, "HEAD_DIM_MINOR")
    if val is not None:
      _normalize_layout_name(val, field_name)
  for field_name in ("local_sa_q_layout", "local_sa_k_layout", "local_sa_v_layout"):
    val = getattr(config, field_name, None)
    if val is not None:
      _normalize_layout_name(val, field_name)

  gqa_stacking = bool(getattr(config, "sa_gqa_head_stacking", False))
  if gqa_stacking:
    if attention_mode == "dot_product":
      raise ValueError("sa_gqa_head_stacking=True requires Splash attention (attention='flash' or 'autoselected').")
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None and int(head_dim) != 64:
      raise ValueError(f"sa_gqa_head_stacking=True requires head_dim == 64; got head_dim={head_dim!r}.")
    n_q = getattr(config, "num_query_heads", None)
    n_kv = getattr(config, "num_kv_heads", None)
    if n_q is not None and n_kv is not None:
      if int(n_kv) <= 0 or int(n_q) % int(n_kv) != 0:
        raise ValueError(f"num_query_heads ({n_q}) must be a positive multiple of num_kv_heads ({n_kv}).")
      group_size = int(n_q) // int(n_kv)
      if group_size < 2 or group_size % 2 != 0:
        raise ValueError(
            f"sa_gqa_head_stacking=True requires an even GQA group size >= 2; got {n_q} // {n_kv} = {group_size}."
        )

  if attention_mode == "flash":
    seq_len = _configured_sequence_length(config)
    if seq_len > 0 and (seq_len < _SPLASH_TILE_ALIGNMENT or seq_len % _SPLASH_TILE_ALIGNMENT != 0):
      raise ValueError(
          f"attention='flash' requires sequence length to be a multiple of {_SPLASH_TILE_ALIGNMENT}; got {seq_len}."
      )
    SplashConfig.from_config(config, local=attention_type == "local_sliding")


def attention_from_config(
    config: Any,
    sharding: Sharding | None = None,
    *,
    causal: bool = True,
    sliding_window: int | None = None,
    attn_logits_soft_cap: float | None = None,
) -> AttentionKernel:
  """Resolves an `AttentionKernel` from `config` and `sharding` without touching traced inputs."""
  validate_attention_config(config)
  resolved_sharding = sharding or DefaultSharding()

  attention_type = getattr(config, "attention_type", "global")
  attention_type = getattr(attention_type, "value", attention_type)
  if sliding_window is None and attention_type == "local_sliding":
    sliding_window = int(getattr(config, "sliding_window_size"))

  if attn_logits_soft_cap is None:
    cfg_cap = getattr(config, "attn_logits_soft_cap", None)
    if cfg_cap is not None and float(cfg_cap) > 0.0:
      attn_logits_soft_cap = float(cfg_cap)

  attention_mode = getattr(config, "attention", "autoselected")
  gqa_stacking = bool(getattr(config, "sa_gqa_head_stacking", False))
  if attention_mode == "dot_product":
    backend: AttentionBackend = "xla"
  elif attention_mode == "flash":
    backend = "splash"
  else:
    seq_len = _configured_sequence_length(config)
    targets_tpu = bool(getattr(config, "compile_topology", "")) or jax.default_backend() == "tpu"
    seq_aligned = seq_len >= _SPLASH_TILE_ALIGNMENT and seq_len % _SPLASH_TILE_ALIGNMENT == 0
    if gqa_stacking and not seq_aligned:
      raise ValueError(
          f"sa_gqa_head_stacking=True requires sequence length divisible by {_SPLASH_TILE_ALIGNMENT}; got {seq_len}."
      )
    backend = "splash" if (targets_tpu and seq_aligned) or gqa_stacking else "xla"

  q_spec = resolved_sharding("attention_kernel_q", ("batch", "heads", "length", "head_dim"), TensorType.ACTIVATION)
  kv_spec = resolved_sharding("attention_kernel_kv", ("batch", "kv_heads", "length", "head_dim"), TensorType.ACTIVATION)
  float32_logits = bool(getattr(config, "float32_logits", True))
  splash_cfg = (
      SplashConfig.from_config(config, local=sliding_window is not None) if backend == "splash" else SplashConfig()
  )
  return AttentionKernel(
      backend=backend,
      batch=q_spec[0],
      heads=q_spec[1],
      kv_heads=kv_spec[1],
      length=q_spec[2],
      kv_length=kv_spec[2],
      causal=causal,
      sliding_window=sliding_window,
      attn_logits_soft_cap=attn_logits_soft_cap,
      float32_logits=float32_logits,
      gqa_head_stacking=gqa_stacking if backend == "splash" else False,
      splash=splash_cfg,
  )


class _StackedGQAMask(mtk_mask._ComputableMask):  # pylint: disable=protected-access
  """Computable mask for 2x query-head sequence folding (`q_idx // group_size` token mapping)."""

  group_size: int
  is_causal: bool
  sliding_window: int | None

  def __init__(
      self,
      shape: tuple[int, int],
      *,
      group_size: int = 2,
      is_causal: bool = True,
      sliding_window: int | None = None,
      shard_count: int = 1,
  ):
    self.group_size = int(group_size)
    self.is_causal = bool(is_causal)
    self.sliding_window = None if sliding_window is None else int(sliding_window)
    self._shard_count_val = int(shard_count)

    def mask_fn(q_ids, kv_ids):
      orig_q = q_ids // self.group_size
      allowed = (orig_q >= kv_ids) if self.is_causal else ((orig_q >= 0) & (kv_ids >= 0))
      if self.sliding_window is not None:
        left = self.sliding_window - 1
        right = 0 if self.is_causal else (self.sliding_window - 1)
        allowed = allowed & ((orig_q - left) <= kv_ids) & (kv_ids <= (orig_q + right))
      return allowed

    super().__init__(shape=shape, mask_function=mask_fn, shard_count=shard_count)

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, _StackedGQAMask)
        and self.shape == other.shape
        and self.group_size == other.group_size
        and self.is_causal == other.is_causal
        and self.sliding_window == other.sliding_window
        and self._shard_count_val == other._shard_count_val
    )

  def __bool__(self) -> bool:
    return True

  def __hash__(self) -> int:
    return hash(
        (
            type(self),
            self.shape,
            self.group_size,
            self.is_causal,
            self.sliding_window,
            self._shard_count_val,
        )
    )


def _build_single_head_mask(
    q_len: int,
    kv_len: int,
    *,
    causal: bool,
    sliding_window: int | None,
    gqa_head_stacking: bool,
    q_seq_shards: int,
) -> mtk_mask.Mask:
  """Constructs a single-head Splash `Mask` for causal, full, SWA, or 2x stacked GQA."""
  if gqa_head_stacking:
    return _StackedGQAMask(
        shape=(2 * q_len, kv_len),
        group_size=2,
        is_causal=causal,
        sliding_window=sliding_window,
        shard_count=q_seq_shards,
    )
  mask_shape = (q_len, kv_len)
  mask: mtk_mask.Mask = (
      mtk_mask.CausalMask(shape=mask_shape, shard_count=q_seq_shards) if causal else mtk_mask.FullMask(mask_shape)
  )
  if sliding_window is not None:
    window_bounds = (sliding_window - 1, 0) if causal else (sliding_window - 1, sliding_window - 1)
    local_mask = mtk_mask.LocalMask(shape=mask_shape, window_size=window_bounds, offset=0, shard_count=q_seq_shards)
    mask = mask & local_mask
  return mask


@functools.partial(jax.jit, static_argnames=("mask", "q_seq_shards", "splash_config"))
def _build_splash_kernel(
    mask: mtk_mask.Mask,
    q_seq_shards: int,
    splash_config: mtk_splash.SplashConfig,
) -> mtk_splash.SplashAttentionKernel:
  """Compiles and caches Splash block-mask tables per static `(mask, q_seq_shards, splash_config)`."""
  return mtk_splash.make_splash_mha(
      mask=mask,
      config=splash_config,
      q_seq_shards=q_seq_shards,
  )


def _maybe_constrain_xla_qkv(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    kernel: AttentionKernel,
) -> tuple[jax.Array, jax.Array, jax.Array, P | None]:
  """Applies static head/batch sharding constraints on Auto meshes when dimensions divide."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or any(t != jax.sharding.AxisType.Auto for t in mesh.axis_types):
    return query, key, value, None
  batch_axis = resolve_mesh_axes(mesh, kernel.batch)
  heads_axis = resolve_mesh_axes(mesh, kernel.heads)
  kv_heads_axis = resolve_mesh_axes(mesh, kernel.kv_heads)
  length_axis = resolve_mesh_axes(mesh, kernel.length)
  kv_length_axis = resolve_mesh_axes(mesh, kernel.kv_length)
  if query.shape[0] % mesh_axis_size(mesh, batch_axis) != 0:
    batch_axis = None
  if query.shape[1] % mesh_axis_size(mesh, length_axis) != 0:
    length_axis = None
  if key.shape[1] % mesh_axis_size(mesh, kv_length_axis) != 0:
    kv_length_axis = None
  head_shards = mesh_axis_size(mesh, heads_axis)
  kv_head_shards = mesh_axis_size(mesh, kv_heads_axis)
  if query.shape[2] % head_shards != 0:
    heads_axis = None
    kv_heads_axis = None
  elif key.shape[2] % kv_head_shards != 0 or kv_head_shards not in (1, head_shards):
    kv_heads_axis = None
  q_spec = P(batch_axis, length_axis, heads_axis, None)
  kv_spec = P(batch_axis, kv_length_axis, kv_heads_axis, None)
  query = jax.lax.with_sharding_constraint(query, q_spec)
  key = jax.lax.with_sharding_constraint(key, kv_spec)
  value = jax.lax.with_sharding_constraint(value, kv_spec)
  return query, key, value, q_spec


def _validate_max_logit_value(
    max_logit_value: jax.Array | float | None,
    kernel: AttentionKernel,
    num_q_heads: int,
) -> jax.Array | None:
  """Validates runtime `max_logit_value` online-softmax bound across backends."""
  if max_logit_value is None:
    return None
  if kernel.splash.max_logit_const is not None:
    raise ValueError(
        "Cannot pass runtime max_logit_value when SplashConfig.max_logit_const "
        f"({kernel.splash.max_logit_const}) is already set."
    )
  if isinstance(max_logit_value, (int, float)) and float(max_logit_value) <= 0.0:
    raise ValueError(f"max_logit_value must be > 0 when set; got {max_logit_value!r}.")
  max_lv_arr = jnp.asarray(max_logit_value, dtype=jnp.float32)
  if max_lv_arr.ndim == 0:
    return max_lv_arr
  if max_lv_arr.shape == (num_q_heads,):
    if kernel.gqa_head_stacking:
      raise ValueError("Per-head 1D max_logit_value is not supported with gqa_head_stacking=True; pass a scalar bound.")
    return max_lv_arr
  raise ValueError(f"max_logit_value must be a scalar or have shape ({num_q_heads},); got {max_lv_arr.shape}.")


def _xla_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    kernel: AttentionKernel,
    segment_ids: jax.Array | None,
    positions: jax.Array | None,
    sinks: jax.Array | None,
    max_logit_value: jax.Array | float | None = None,
) -> jax.Array:
  """Portable XLA scaled dot-product attention via `jax.nn.dot_product_attention`.

  Note: `max_logit_value` / `SplashConfig.max_logit_const` is a single-pass
  Splash online-softmax running-max lower bound ($m_0$). XLA computes the exact
  two-pass row maximum and validates `max_logit_value` for cross-backend parity.
  """
  batch, q_len, num_q_heads, head_dim_qk = query.shape
  kv_len = key.shape[1]
  num_kv_heads = key.shape[2]
  head_dim_v = value.shape[3]

  if segment_ids is not None and (segment_ids.shape != (batch, q_len) or q_len != kv_len):
    raise ValueError(
        f"segment_ids must have shape ({batch}, {q_len}) with q_len == kv_len; got {segment_ids.shape} "
        f"for q_len={q_len}, kv_len={kv_len}."
    )
  if sinks is not None and sinks.shape != (num_q_heads,):
    raise ValueError(f"sinks must have shape ({num_q_heads},); got {sinks.shape}.")
  _validate_max_logit_value(max_logit_value, kernel, num_q_heads)

  query, key, value, out_spec = _maybe_constrain_xla_qkv(query, key, value, kernel)

  # Fast native XLA dot_product_attention path when soft-cap and sinks are unused.
  # Keep Q/K/V in value.dtype so XLA uses BF16_BF16_F32 for QK^T, fp32 for softmax,
  # and bfloat16 for probs @ V rather than promoting both GEMMs to float32.
  if sinks is None and kernel.attn_logits_soft_cap is None:
    q_in = query.astype(value.dtype)
    k_in = key.astype(value.dtype)
    v_in = value
    if positions is None and segment_ids is None:
      window_size = None
      if kernel.sliding_window is not None:
        left = kernel.sliding_window - 1
        right = 0 if kernel.causal else (kernel.sliding_window - 1)
        window_size = (left, right)
      out = jax.nn.dot_product_attention(
          q_in,
          k_in,
          v_in,
          scale=1.0,
          is_causal=kernel.causal,
          local_window_size=window_size,
          implementation="xla",
      ).astype(value.dtype)
    else:
      q_pos = jnp.arange(q_len, dtype=jnp.int32)[None, :] if positions is None else positions
      k_pos = jnp.arange(kv_len, dtype=jnp.int32)[None, :] if (positions is None or kv_len != q_len) else positions
      mask = (
          (k_pos[..., None, :] <= q_pos[..., :, None])
          if kernel.causal
          else jnp.ones(q_pos.shape + (kv_len,), dtype=jnp.bool_)
      )
      if kernel.sliding_window is not None:
        left = kernel.sliding_window - 1
        right = 0 if kernel.causal else (kernel.sliding_window - 1)
        mask = (
            mask
            & ((q_pos[..., :, None] - left) <= k_pos[..., None, :])
            & (k_pos[..., None, :] <= (q_pos[..., :, None] + right))
        )
      if segment_ids is not None:
        valid_seg = (
            (segment_ids[..., :, None] == segment_ids[..., None, :])
            & (segment_ids[..., :, None] != 0)
            & (segment_ids[..., None, :] != 0)
        )
        mask = mask & valid_seg
      any_valid = jnp.any(mask, axis=-1, keepdims=True)
      fallback_row = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :] == 0
      safe_mask = jnp.where(any_valid, mask, fallback_row)
      out = jax.nn.dot_product_attention(
          q_in,
          k_in,
          v_in,
          mask=safe_mask[:, None, :, :],
          scale=1.0,
          implementation="xla",
      )
      out = jnp.where(any_valid[..., None], out, 0).astype(value.dtype)
      if segment_ids is not None:
        out = jnp.where((segment_ids != 0)[..., None, None], out, 0)
    if out_spec is not None:
      out = jax.lax.with_sharding_constraint(out, out_spec)
    return out

  groups = num_q_heads // num_kv_heads
  query_g = jnp.reshape(query, (batch, q_len, num_kv_heads, groups, head_dim_qk))
  einsum_dtype = jnp.float32 if kernel.float32_logits else None
  scores = jnp.einsum("btkgd,bskd->bkgts", query_g, key, preferred_element_type=einsum_dtype)
  if kernel.attn_logits_soft_cap is not None:
    cap = jnp.asarray(kernel.attn_logits_soft_cap, dtype=scores.dtype)
    scores = jnp.tanh(scores / cap) * cap

  q_pos = jnp.arange(q_len, dtype=jnp.int32)[None, :] if positions is None else positions
  k_pos = jnp.arange(kv_len, dtype=jnp.int32)[None, :] if (positions is None or kv_len != q_len) else positions
  mask = (
      (k_pos[..., None, :] <= q_pos[..., :, None])
      if kernel.causal
      else jnp.ones(q_pos.shape + (kv_len,), dtype=jnp.bool_)
  )
  if kernel.sliding_window is not None:
    left = kernel.sliding_window - 1
    right = 0 if kernel.causal else (kernel.sliding_window - 1)
    mask = (
        mask
        & ((q_pos[..., :, None] - left) <= k_pos[..., None, :])
        & (k_pos[..., None, :] <= (q_pos[..., :, None] + right))
    )
  if segment_ids is not None:
    valid_seg = (
        (segment_ids[..., :, None] == segment_ids[..., None, :])
        & (segment_ids[..., :, None] != 0)
        & (segment_ids[..., None, :] != 0)
    )
    mask = mask & valid_seg

  any_valid = jnp.any(mask, axis=-1, keepdims=True)
  if sinks is not None:
    mask_val = -1e4 if (not kernel.float32_logits and scores.dtype == jnp.float16) else -1e10
    scores = jnp.where(mask[:, None, None, :, :], scores, mask_val)
    if kernel.float32_logits:
      scores = scores.astype(jnp.float32)
    logits = jnp.reshape(scores, (batch, num_q_heads, q_len, kv_len))
    sink_col = jnp.broadcast_to(
        sinks.astype(logits.dtype).reshape((1, num_q_heads, 1, 1)),
        (batch, num_q_heads, q_len, 1),
    )
    logits_all = jnp.concatenate([logits, sink_col], axis=-1)
    local_max = jnp.max(logits_all, axis=-1, keepdims=True)
    local_exp_all = jnp.exp(logits_all - local_max)
    local_sum = jnp.sum(local_exp_all, axis=-1, keepdims=True)
    local_exp = local_exp_all[..., :kv_len]
  else:
    fallback_row = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :] == 0
    safe_mask = jnp.where(any_valid, mask, fallback_row)
    mask_val = -1e4 if (not kernel.float32_logits and scores.dtype == jnp.float16) else -1e10
    scores = jnp.where(safe_mask[:, None, None, :, :], scores, mask_val)
    if kernel.float32_logits:
      scores = scores.astype(jnp.float32)
    logits = jnp.reshape(scores, (batch, num_q_heads, q_len, kv_len))
    local_max = jnp.max(logits, axis=-1, keepdims=True)
    local_exp = jnp.exp(logits - local_max)
    local_sum = jnp.sum(local_exp, axis=-1, keepdims=True)

  local_exp_grouped = jnp.reshape(local_exp.astype(value.dtype), (batch, num_kv_heads, groups, q_len, kv_len))
  out = jnp.einsum("bkgts,bskd->btkgd", local_exp_grouped, value)
  out = jnp.reshape(out, (batch, q_len, num_q_heads, head_dim_v))
  denom = jnp.transpose(jnp.where(local_sum > 0, local_sum, 1.0), (0, 2, 1, 3))
  out = (out / denom).astype(value.dtype)
  if sinks is None:
    out = jnp.where(any_valid[..., None], out, 0)
  if segment_ids is not None:
    out = jnp.where((segment_ids != 0)[..., None, None], out, 0)
  if out_spec is not None:
    out = jax.lax.with_sharding_constraint(out, out_spec)
  return out


def _remap_padding_segment_ids(segment_ids: jax.Array, offset: int = 0) -> jax.Array:
  """Maps `segment_ids == 0` padding slots to distinct negative IDs so Splash never sees 0/0 NaNs."""
  seq_len = segment_ids.shape[-1]
  neg_ids = -(jnp.arange(seq_len, dtype=jnp.int32) + 1 + offset)
  return jnp.where(segment_ids == 0, neg_ids, segment_ids.astype(jnp.int32))


def _splash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    kernel: AttentionKernel,
    segment_ids: jax.Array | None,
    sinks: jax.Array | None,
    max_logit_value: jax.Array | float | None,
) -> jax.Array:
  """Sharded Pallas Splash attention supporting SEQ_MINOR, SWA, packing, and 2x GQA head stacking."""
  batch, q_len, num_q_heads, head_dim_qk = query.shape
  kv_len = key.shape[1]
  num_kv_heads = key.shape[2]
  head_dim_v = value.shape[3]

  if q_len < _SPLASH_TILE_ALIGNMENT or q_len % _SPLASH_TILE_ALIGNMENT != 0:
    raise ValueError(f"Splash attention requires q_len to be a multiple of {_SPLASH_TILE_ALIGNMENT}; got q_len={q_len}.")
  if kv_len < _SPLASH_TILE_ALIGNMENT or kv_len % _SPLASH_TILE_ALIGNMENT != 0:
    raise ValueError(
        f"Splash attention requires kv_len to be a multiple of {_SPLASH_TILE_ALIGNMENT}; got kv_len={kv_len}."
    )
  if segment_ids is not None and (segment_ids.shape != (batch, q_len) or q_len != kv_len):
    raise ValueError(
        f"Splash segment_ids must have shape ({batch}, {q_len}) with q_len == kv_len; got {segment_ids.shape} "
        f"for q_len={q_len}, kv_len={kv_len}."
    )
  if sinks is not None and sinks.shape != (num_q_heads,):
    raise ValueError(f"sinks must have shape ({num_q_heads},); got {sinks.shape}.")

  mesh = mesh_in_scope()
  batch_axis = resolve_mesh_axes(mesh, kernel.batch)
  heads_axis = resolve_mesh_axes(mesh, kernel.heads)
  kv_heads_axis = resolve_mesh_axes(mesh, kernel.kv_heads)
  length_axis = resolve_mesh_axes(mesh, kernel.length)
  kv_length_axis = resolve_mesh_axes(mesh, kernel.kv_length)

  batch_shards = mesh_axis_size(mesh, batch_axis)
  head_shards = mesh_axis_size(mesh, heads_axis)
  kv_head_shards = mesh_axis_size(mesh, kv_heads_axis)
  q_seq_shards = mesh_axis_size(mesh, length_axis)
  kv_seq_shards = mesh_axis_size(mesh, kv_length_axis)

  if kv_seq_shards > 1:
    raise NotImplementedError(
        f"Splash attention with sharded KV sequence (kv_length={kernel.kv_length!r}, shards={kv_seq_shards}) "
        "requires ring attention, which is not enabled."
    )
  if batch % batch_shards != 0:
    raise ValueError(f"Batch size ({batch}) is not divisible by batch shards ({batch_shards}) on axis {batch_axis!r}.")
  if num_q_heads % head_shards != 0:
    raise ValueError(
        f"num_query_heads ({num_q_heads}) is not divisible by head shards ({head_shards}) on axis {heads_axis!r}."
    )
  if num_kv_heads % kv_head_shards != 0:
    raise ValueError(
        f"num_kv_heads ({num_kv_heads}) is not divisible by kv_head shards ({kv_head_shards}) "
        f"on axis {kv_heads_axis!r}."
    )
  if kv_head_shards not in (1, head_shards):
    raise ValueError(
        f"Splash attention requires kv_head_shards ({kv_head_shards}) to equal 1 or head_shards ({head_shards})."
    )
  if head_shards > 1 and kv_head_shards == 1 and mesh_axis_names(heads_axis) == mesh_axis_names(kv_heads_axis):
    raise ValueError("Inconsistent head and kv_head shard counts.")

  local_q_heads = num_q_heads // head_shards
  local_kv_heads = num_kv_heads // kv_head_shards
  if local_q_heads % local_kv_heads != 0:
    raise ValueError(
        f"Per-shard query heads ({local_q_heads}) must be divisible by per-shard kv heads ({local_kv_heads})."
    )

  if kernel.gqa_head_stacking:
    if q_len != kv_len:
      raise ValueError(f"gqa_head_stacking=True requires q_len == kv_len; got {q_len} vs {kv_len}.")
    if head_dim_qk != 64 or head_dim_v != 64:
      raise ValueError(
          f"gqa_head_stacking=True requires head_dim == 64 for Q/K/V; got D_qk={head_dim_qk}, D_v={head_dim_v}."
      )
    if sinks is not None:
      raise ValueError("gqa_head_stacking=True does not support attention sinks.")
    if q_seq_shards > 1:
      raise NotImplementedError("gqa_head_stacking=True does not support sharded query sequence.")
    local_group_size = local_q_heads // local_kv_heads
    if local_group_size < 2 or local_group_size % 2 != 0:
      raise ValueError(
          f"gqa_head_stacking=True requires per-shard GQA group size ({local_group_size}) to be an even integer >= 2."
      )

  effective_q_len = 2 * q_len if kernel.gqa_head_stacking else q_len
  if effective_q_len % (q_seq_shards * _SPLASH_TILE_ALIGNMENT) != 0:
    raise ValueError(
        f"Effective query length ({effective_q_len}) must be divisible by "
        f"q_seq_shards * {_SPLASH_TILE_ALIGNMENT} ({q_seq_shards * _SPLASH_TILE_ALIGNMENT})."
    )

  splash_cfg = kernel.splash.resolve(
      effective_q_len // q_seq_shards,
      kv_len,
      attn_logits_soft_cap=kernel.attn_logits_soft_cap,
      interpret=not is_tpu_mesh(mesh),
  )
  mask = _build_single_head_mask(
      q_len,
      kv_len,
      causal=kernel.causal,
      sliding_window=kernel.sliding_window,
      gqa_head_stacking=kernel.gqa_head_stacking,
      q_seq_shards=q_seq_shards,
  )
  splash_kern = _build_splash_kernel(mask, q_seq_shards, splash_cfg)
  splash_kern_spec = splash_kern.manual_sharding_spec(NamedSharding(mesh, P(length_axis)))

  max_lv_arr = _validate_max_logit_value(max_logit_value, kernel, num_q_heads)
  max_lv_spec: P | None = None
  if max_lv_arr is not None:
    max_lv_spec = P() if max_lv_arr.ndim == 0 else P(heads_axis)

  q_spec = P(batch_axis, length_axis, heads_axis, None)
  kv_spec = P(batch_axis, kv_length_axis, kv_heads_axis, None)
  seg_q_spec = None if segment_ids is None else P(batch_axis, length_axis)
  seg_kv_spec = None if segment_ids is None else P(batch_axis, None)
  sinks_spec = None if sinks is None else P(heads_axis)

  seg_clean = None if segment_ids is None else _remap_padding_segment_ids(segment_ids)
  pad_qk = (-head_dim_qk) % _SPLASH_TILE_ALIGNMENT
  pad_v = (-head_dim_v) % _SPLASH_TILE_ALIGNMENT

  def _per_shard(ql, kl, vl, seg_ql, seg_kvl, sinks_l, mlv_l, kern_l):  # pylint: disable=too-many-positional-arguments
    b_l, sq_l, hq_l, _ = ql.shape
    hkv_l = kl.shape[2]
    if pad_qk:
      ql = jnp.pad(ql, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
      kl = jnp.pad(kl, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
    if pad_v:
      vl = jnp.pad(vl, ((0, 0), (0, 0), (0, 0), (0, pad_v)))
    padded_qk = ql.shape[-1]
    padded_v = vl.shape[-1]
    if padded_qk != padded_v:
      target_d = max(padded_qk, padded_v)
      if padded_qk < target_d:
        ql = jnp.pad(ql, ((0, 0), (0, 0), (0, 0), (0, target_d - padded_qk)))
        kl = jnp.pad(kl, ((0, 0), (0, 0), (0, 0), (0, target_d - padded_qk)))
      if padded_v < target_d:
        vl = jnp.pad(vl, ((0, 0), (0, 0), (0, 0), (0, target_d - padded_v)))
    target_d = ql.shape[-1]

    if kernel.gqa_head_stacking:
      g_l = hq_l // hkv_l
      q_swapped = (
          ql.reshape((b_l, sq_l, hkv_l, g_l // 2, 2, target_d))
          .transpose((0, 2, 3, 1, 4, 5))
          .reshape((b_l, hq_l // 2, 2 * sq_l, target_d))
      )
      eff_hq = hq_l // 2
      seg_q_eff = None if seg_ql is None else jnp.repeat(seg_ql, 2, axis=-1)
    else:
      q_swapped = jnp.swapaxes(ql, 1, 2)
      eff_hq = hq_l
      seg_q_eff = seg_ql

    k_swapped = jnp.swapaxes(kl, 1, 2)
    v_swapped = jnp.swapaxes(vl, 1, 2)
    mlv_vec = None if mlv_l is None else jnp.broadcast_to(mlv_l, (eff_hq,))
    sinks_vec = None if sinks_l is None else sinks_l.astype(jnp.float32)

    if seg_q_eff is None:
      out_swapped = jax.vmap(lambda qi, ki, vi: kern_l(qi, ki, vi, None, sinks_vec, max_logit_value=mlv_vec))(
          q_swapped, k_swapped, v_swapped
      )
    else:
      out_swapped = jax.vmap(
          lambda qi, ki, vi, sqi, ski: kern_l(
              qi,
              ki,
              vi,
              mtk_splash.SegmentIds(sqi, ski),
              sinks_vec,
              max_logit_value=mlv_vec,
          )
      )(q_swapped, k_swapped, v_swapped, seg_q_eff, seg_kvl)

    out_sliced = out_swapped[..., :head_dim_v]
    if kernel.gqa_head_stacking:
      out_local = (
          out_sliced.reshape((b_l, hq_l // 2, sq_l, 2, head_dim_v))
          .transpose((0, 2, 1, 3, 4))
          .reshape((b_l, sq_l, hq_l, head_dim_v))
      )
    else:
      out_local = jnp.swapaxes(out_sliced, 1, 2)
    return out_local.astype(value.dtype)

  sharded_fn = jax.shard_map(
      _per_shard,
      mesh=mesh,
      in_specs=(
          q_spec,
          kv_spec,
          kv_spec,
          seg_q_spec,
          seg_kv_spec,
          sinks_spec,
          max_lv_spec,
          splash_kern_spec,
      ),
      out_specs=q_spec,
      check_vma=False,
  )
  out = sharded_fn(query, key, value, seg_clean, seg_clean, sinks, max_lv_arr, splash_kern)
  if segment_ids is not None:
    out = jnp.where((segment_ids != 0)[..., None, None], out, 0)
  return out


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    kernel: AttentionKernel = XLA_ATTENTION,
    segment_ids: jax.Array | None = None,
    positions: jax.Array | None = None,
    sinks: jax.Array | None = None,
    max_logit_value: jax.Array | float | None = None,
) -> jax.Array:
  """Dispatches scaled dot-product attention (`query` pre-scaled) to `kernel.backend`."""
  if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
    raise ValueError(
        f"query, key, value must be rank-4 (batch, length, heads, head_dim); "
        f"got shapes {query.shape}, {key.shape}, {value.shape}."
    )
  if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
    raise ValueError(
        f"Batch size mismatch across query ({query.shape[0]}), key ({key.shape[0]}), value ({value.shape[0]})."
    )
  if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
    raise ValueError(f"Key ({key.shape}) and value ({value.shape}) must match in sequence length and kv_heads.")
  if query.shape[3] != key.shape[3]:
    raise ValueError(f"Query head_dim ({query.shape[3]}) must match key head_dim ({key.shape[3]}).")
  if key.shape[2] <= 0 or query.shape[2] % key.shape[2] != 0:
    raise ValueError(f"Query heads ({query.shape[2]}) must be a positive multiple of KV heads ({key.shape[2]}).")

  if kernel.backend == "xla":
    return _xla_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )
  if kernel.backend == "splash":
    return _splash_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )
  raise ValueError(f"Unsupported attention backend: {kernel.backend!r}.")
