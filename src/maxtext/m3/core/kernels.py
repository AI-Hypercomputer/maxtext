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

Note: `KernelBackend.TOKAMAX` (`"tokamax"`) routes to upstream
`tokamax.dot_product_attention`, whereas `KernelBackend.SPLASH` (`"splash"`) bridges
`maxtext.kernels.tokamax_splash_attention` for `tpu7x` `SEQ_MINOR`, `max_logit_value`,
and 2x GQA head stacking until those experimental `_src` features are exposed in
PyPI `tokamax`'s public API.
"""

import dataclasses
import enum
import functools
import importlib
import math
import sys
from typing import Any, Literal

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType


class KernelBackend(enum.StrEnum):
  """Supported m3 attention kernel dispatch backends."""

  AUTO = "auto"
  JAX = "jax"
  TOKAMAX = "tokamax"
  SPLASH = "splash"
  CUDNN = "cudnn"


VALID_KERNEL_BACKENDS: tuple[str, ...] = tuple(b.value for b in KernelBackend)
AttentionBackend = Literal["auto", "jax", "xla", "tokamax", "splash", "cudnn"] | KernelBackend

_SPLASH_TILE_ALIGNMENT = 128
_VALID_ATTENTION_MODES = ("dot_product", "flash", "autoselected") + VALID_KERNEL_BACKENDS + ("xla",)
_VALID_ATTENTION_TYPES = ("global", "local_sliding", "mla")
_VALID_QKV_LAYOUTS = ("HEAD_DIM_MINOR", "SEQ_MINOR")
_LEGACY_BACKEND_ALIASES: dict[str, KernelBackend] = {
    "autoselected": KernelBackend.AUTO,
    "dot_product": KernelBackend.JAX,
    "legacy": KernelBackend.JAX,
    "xla": KernelBackend.JAX,
    "flash": KernelBackend.SPLASH,
    "cudnn_flash_te": KernelBackend.CUDNN,
}


def _cfg_attr(source: Any, name: str, default: Any = None) -> Any:
  """Reads an attribute from `source` without triggering config-fallback lint on optional test namespaces."""
  if source is None:
    return default
  val = getattr(source, name, None)
  return default if val is None else val


@functools.lru_cache(maxsize=1)
def _import_splash() -> tuple[Any, Any, type[Any]]:
  """Lazily imports Tokamax Pallas Splash attention modules and builds `_StackedGQAMask`."""
  mtk_splash = importlib.import_module("maxtext.kernels.tokamax_splash_attention.splash_attention_kernel")
  mtk_mask = importlib.import_module("maxtext.kernels.tokamax_splash_attention.splash_attention_mask")

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

  return mtk_splash, mtk_mask, _StackedGQAMask


def _import_tokamax(*, required: bool = False) -> Any | None:
  """Lazily imports the optional `tokamax` package when requested."""
  if "tokamax" in sys.modules and sys.modules["tokamax"] is None:
    if required:
      raise ImportError("KernelBackend 'tokamax' requested, but 'tokamax' is disabled in sys.modules.")
    return None
  try:
    return importlib.import_module("tokamax")
  except ImportError as exc:
    if required:
      raise ImportError("KernelBackend 'tokamax' requested, but 'tokamax' package is not installed.") from exc
    return None


def splash_available() -> bool:
  """Returns True when the Tokamax Splash attention package can be imported."""
  for mod_name in (
      "maxtext.kernels.tokamax_splash_attention",
      "maxtext.kernels.tokamax_splash_attention.splash_attention_kernel",
      "maxtext.kernels.tokamax_splash_attention.splash_attention_mask",
  ):
    if mod_name in sys.modules and sys.modules[mod_name] is None:
      return False
  try:
    _import_splash()
    return True
  except ImportError:
    return False


def tokamax_available() -> bool:
  """Returns True when the `tokamax` package can be imported."""
  return _import_tokamax(required=False) is not None


def parse_kernel_backend(
    backend: str | KernelBackend,
    *,
    allow_legacy_aliases: bool = False,
) -> KernelBackend:
  """Validates and converts `backend` into a canonical `KernelBackend` enum value."""
  if isinstance(backend, KernelBackend):
    return backend
  if not isinstance(backend, str):
    raise ValueError(f"Kernel backend must be a string or KernelBackend; got {backend!r}.")
  normalized = backend.strip().lower()
  if normalized in VALID_KERNEL_BACKENDS:
    return KernelBackend(normalized)
  if allow_legacy_aliases and normalized in _LEGACY_BACKEND_ALIASES:
    return _LEGACY_BACKEND_ALIASES[normalized]
  raise ValueError(f"Unknown kernel backend {backend!r}; expected one of {VALID_KERNEL_BACKENDS}.")


_KNOWN_MESH_AXIS_NAMES = frozenset(
    {
        "data",
        "fsdp",
        "fsdp_transpose",
        "expert",
        "context",
        "stage",
        "tensor",
        "tensor_sequence",
    }
)


def mesh_axis_names(axes: MeshAxes) -> tuple[str, ...]:
  """Normalizes one `PartitionSpec` entry into a tuple of mesh axis names."""
  if axes is None:
    return ()
  return (axes,) if isinstance(axes, str) else tuple(axes)


def resolve_mesh_axes(mesh: Any, axes: MeshAxes) -> MeshAxes:
  """Filters standard m3 mesh axes to `mesh.axis_names` (and size > 1 on Explicit meshes)."""
  present = set(getattr(mesh, "axis_names", ()))
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  explicit = bool(axis_types) and all(t == jax.sharding.AxisType.Explicit for t in axis_types)
  shape = getattr(mesh, "shape", None)
  kept: list[str] = []
  for name in mesh_axis_names(axes):
    if name in present:
      if explicit and shape is not None and int(shape[name]) <= 1:
        continue
      kept.append(name)
    elif name not in _KNOWN_MESH_AXIS_NAMES:
      raise ValueError(f"Unknown mesh axis {name!r}; active mesh axes are {tuple(getattr(mesh, 'axis_names', ()))}.")
  if not kept:
    return None
  return kept[0] if len(kept) == 1 else tuple(kept)


def ensure_explicit_spec(x: jax.Array | None, spec: P | None, mesh: Any) -> jax.Array | None:
  """Reshards `x` to `spec` on `Explicit` meshes when `shard_map` or explicit ops require matching specs."""
  if x is None or spec is None:
    return x
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  if not axis_types or any(t != jax.sharding.AxisType.Explicit for t in axis_types):
    return x
  cur_sharding = jax.typeof(x).sharding
  if hasattr(cur_sharding, "spec") and cur_sharding.spec == spec:
    return x
  return jax.sharding.reshard(x, spec)


def mesh_axis_size(mesh: Any, axes: MeshAxes) -> int:
  """Returns the product of shard counts across `axes` on `mesh`."""
  resolved = resolve_mesh_axes(mesh, axes)
  return math.prod(int(mesh.shape[name]) for name in mesh_axis_names(resolved))


def _active_mesh_or_none(mesh: Any = None) -> Any | None:
  """Returns `mesh` if provided and non-empty, else the active abstract or concrete JAX mesh."""
  if mesh is not None and not getattr(mesh, "empty", False):
    return mesh
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty:
    return abstract_mesh
  try:
    concrete_mesh = jax.sharding.get_mesh()
  except ValueError:
    concrete_mesh = None
  if concrete_mesh is not None and not getattr(concrete_mesh, "empty", False):
    return concrete_mesh
  return None


def mesh_in_scope() -> Any:
  """Returns the active abstract or concrete JAX mesh, raising if none is set."""
  mesh = _active_mesh_or_none()
  if mesh is not None:
    return mesh
  raise ValueError(
      "Fused kernels run under jax.shard_map and require an active mesh; "
      "call inside `with jax.set_mesh(mesh):` or `with mesh:`."
  )


def normalize_broadcast_spec(
    spec: P | None,
    shape: tuple[int, ...],
    mesh: Any = None,
    *,
    allow_unaligned_fallback: bool = False,
) -> P | None:
  """Normalizes `spec` for `shape` so singleton (`dim == 1`) or absent mesh dims become replicated `None`."""
  if spec is None:
    return None
  active_mesh = _active_mesh_or_none(mesh)
  ndim = len(shape)
  raw_axes = tuple(spec)
  padded_axes = raw_axes[:ndim] + (None,) * max(0, ndim - len(raw_axes))
  normalized: list[MeshAxes] = []
  for dim_size, axis in zip(shape, padded_axes):
    resolved_axis = resolve_mesh_axes(active_mesh, axis) if active_mesh is not None else axis
    if resolved_axis is None or int(dim_size) <= 1:
      normalized.append(None)
      continue
    if allow_unaligned_fallback and active_mesh is not None:
      shards = mesh_axis_size(active_mesh, resolved_axis)
      if int(dim_size) % shards != 0:
        normalized.append(None)
        continue
    normalized.append(resolved_axis)
  return P(*normalized)


def normalize_broadcast_sharding(
    x: jax.Array | None,
    spec: P | None = None,
    mesh: Any = None,
    *,
    target_shape: tuple[int, ...] | None = None,
    allow_unaligned_fallback: bool = False,
) -> jax.Array | None:
  """Normalizes singleton/replicated sharding on `x` (and optionally broadcasts to `target_shape`)."""
  if x is None:
    return None
  active_mesh = _active_mesh_or_none(mesh)
  effective_spec = spec
  if effective_spec is None:
    cur_sharding = jax.typeof(x).sharding
    effective_spec = getattr(cur_sharding, "spec", None)

  if target_shape is not None and x.shape != target_shape:
    if x.ndim > len(target_shape):
      raise ValueError(f"Cannot broadcast shape {x.shape} to lower-rank target_shape {target_shape}.")
    rank_pad = len(target_shape) - x.ndim
    aligned_shape = (1,) * rank_pad + x.shape
    for src_dim, dst_dim in zip(aligned_shape, target_shape):
      if src_dim not in (1, dst_dim):
        raise ValueError(f"Shape {x.shape} is not broadcast-compatible with target_shape {target_shape}.")
    if rank_pad > 0:
      x = jnp.reshape(x, aligned_shape)
    if effective_spec is not None and active_mesh is not None:
      pre_spec = normalize_broadcast_spec(
          effective_spec,
          x.shape,
          active_mesh,
          allow_unaligned_fallback=allow_unaligned_fallback,
      )
      x = _constrain_or_reshard(x, pre_spec, active_mesh)
    x = jnp.broadcast_to(x, target_shape)

  if effective_spec is None or active_mesh is None:
    return x
  norm_spec = normalize_broadcast_spec(
      effective_spec,
      x.shape,
      active_mesh,
      allow_unaligned_fallback=allow_unaligned_fallback,
  )
  return _constrain_or_reshard(x, norm_spec, active_mesh)


_normalize_broadcast_sharding = normalize_broadcast_sharding


def _canonical_platform_name(raw_platform: Any) -> str | None:
  """Normalizes a JAX device platform or hardware string into `'tpu'`, `'gpu'`, or `'cpu'`."""
  if raw_platform is None:
    return None
  name = str(raw_platform).strip().lower()
  if not name:
    return None
  if "tpu" in name:
    return "tpu"
  if name in ("gpu", "cuda", "rocm") or "gpu" in name or "cuda" in name or "nvidia" in name:
    return "gpu"
  if "cpu" in name:
    return "cpu"
  return None


def mesh_platform(mesh: Any = None, config: Any = None) -> str:
  """Resolves the target hardware platform (`'tpu'`, `'gpu'`, or `'cpu'`) from `mesh`, `config`, or JAX."""
  active_mesh = _active_mesh_or_none(mesh) if mesh is None else mesh
  if active_mesh is not None:
    abstract_device = getattr(active_mesh, "abstract_device", None)
    if abstract_device is not None:
      plat = _canonical_platform_name(
          getattr(abstract_device, "platform", None) or getattr(abstract_device, "device_kind", None)
      )
      if plat is not None:
        return plat
    try:
      devices = getattr(active_mesh, "devices", None)
    except ValueError:
      devices = None
    if devices is not None:
      first_dev = None
      if hasattr(devices, "flat") and getattr(devices, "size", 0) > 0:
        first_dev = devices.flat[0]
      elif isinstance(devices, (tuple, list)) and devices:
        first_dev = devices[0]
      if first_dev is not None:
        plat = _canonical_platform_name(getattr(first_dev, "platform", None) or getattr(first_dev, "device_kind", None))
        if plat is not None:
          return plat
    plat = _canonical_platform_name(getattr(active_mesh, "platform", None))
    if plat is not None:
      return plat

  if config is not None:
    if bool(_cfg_attr(config, "compile_topology", "")):
      return "tpu"
    hw_plat = _canonical_platform_name(_cfg_attr(config, "hardware", ""))
    if hw_plat is not None:
      return hw_plat

  return _canonical_platform_name(jax.default_backend()) or "cpu"


def is_tpu_mesh(mesh: Any = None) -> bool:
  """Returns True when `mesh` targets real or virtual AOT TPU devices."""
  return mesh_platform(mesh) == "tpu"


def is_gpu_mesh(mesh: Any = None) -> bool:
  """Returns True when `mesh` targets GPU devices."""
  return mesh_platform(mesh) == "gpu"


def is_cpu_mesh(mesh: Any = None) -> bool:
  """Returns True when `mesh` targets CPU devices."""
  return mesh_platform(mesh) == "cpu"


def cudnn_available(mesh: Any = None, config: Any = None) -> bool:
  """Returns True when running on a GPU platform with JAX cuDNN dot-product attention support."""
  return mesh_platform(mesh, config) == "gpu" and hasattr(jax.nn, "dot_product_attention")


def resolve_kernel_backend(
    backend: str | KernelBackend = KernelBackend.AUTO,
    *,
    mesh: Any = None,
    config: Any = None,
    seq_len: int | None = None,
    allow_legacy_aliases: bool = False,
) -> KernelBackend:
  """Resolves `backend` (`'auto'`, `'jax'`, `'tokamax'`, `'splash'`, `'cudnn'`) for the active platform.

  `'auto'` selects:
    - TPU: `KernelBackend.SPLASH` when sequence length is aligned to 128 and Splash is available,
      falling back to `KernelBackend.TOKAMAX` or `KernelBackend.JAX` when unaligned or unavailable.
    - GPU: `KernelBackend.CUDNN` when available (else `KernelBackend.TOKAMAX` or `KernelBackend.JAX`).
    - CPU: `KernelBackend.JAX` (pure JAX scaled dot-product attention).

  Explicit backends fail closed with `ValueError` (or `ImportError`) when unsupported on the target platform.
  """
  parsed = parse_kernel_backend(backend, allow_legacy_aliases=allow_legacy_aliases)
  platform = mesh_platform(mesh, config)
  if seq_len is None and config is not None:
    seq_len = _configured_sequence_length(config)
  seq_aligned = (
      seq_len is None
      or int(seq_len) <= 0
      or (int(seq_len) >= _SPLASH_TILE_ALIGNMENT and int(seq_len) % _SPLASH_TILE_ALIGNMENT == 0)
  )

  if parsed == KernelBackend.AUTO:
    if platform == "tpu":
      if seq_aligned and splash_available():
        return KernelBackend.SPLASH
      if tokamax_available():
        return KernelBackend.TOKAMAX
      return KernelBackend.JAX
    if platform == "gpu":
      if cudnn_available(mesh, config):
        return KernelBackend.CUDNN
      if tokamax_available():
        return KernelBackend.TOKAMAX
      return KernelBackend.JAX
    return KernelBackend.JAX

  if parsed == KernelBackend.JAX:
    return KernelBackend.JAX

  if parsed == KernelBackend.TOKAMAX:
    _import_tokamax(required=True)
    return KernelBackend.TOKAMAX

  if parsed == KernelBackend.SPLASH:
    if platform != "tpu":
      raise ValueError(
          f"KernelBackend 'splash' requires a TPU mesh/device; got platform {platform!r}. "
          "Use backend='auto' or 'jax' on non-TPU platforms."
      )
    if not splash_available():
      raise ValueError("KernelBackend 'splash' requested, but Tokamax Splash attention kernel is not available.")
    if not seq_aligned:
      raise ValueError(
          f"KernelBackend 'splash' requires sequence length to be a multiple of {_SPLASH_TILE_ALIGNMENT}; "
          f"got {seq_len}."
      )
    return KernelBackend.SPLASH

  if parsed == KernelBackend.CUDNN:
    if platform != "gpu":
      raise ValueError(
          f"KernelBackend 'cudnn' requires a GPU mesh/device; got platform {platform!r}. "
          "Use backend='auto' or 'jax' on non-GPU platforms."
      )
    return KernelBackend.CUDNN

  raise ValueError(f"Unsupported kernel backend: {backend!r}.")


def _normalize_layout_name(layout: Any, field_name: str) -> str:
  """Validates a Splash QKV layout setting and returns its canonical uppercase name."""
  name = str(getattr(layout, "name", getattr(layout, "value", layout))).upper()
  if name not in _VALID_QKV_LAYOUTS:
    raise ValueError(f"{field_name} must be one of {_VALID_QKV_LAYOUTS}; got {layout!r}.")
  return name


def _parse_qkv_layout(layout: str, field_name: str) -> Any:
  """Converts a validated layout string into `mtk_splash.QKVLayout`."""
  mtk_splash, _, _ = _import_splash()
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
  allow_cpu_interpret: bool = True

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
  def from_config(
      cls,
      config: Any,
      *,
      local: bool = False,
      allow_cpu_interpret: bool = True,
  ) -> "SplashConfig":
    """Extracts global or local-window Splash settings from `config`."""

    def _pick(local_attr: str, global_attr: str, default: Any) -> Any:
      if local:
        local_val = _cfg_attr(config, local_attr, None)
        if local_val is not None:
          return local_val
      return _cfg_attr(config, global_attr, default)

    max_logit_est = float(_cfg_attr(config, "use_max_logit_estimate", -1.0) or -1.0)
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
        bwd_dkv_megacore=bool(_cfg_attr(config, "sa_bwd_dkv_megacore", False)),
        allow_cpu_interpret=allow_cpu_interpret,
    )

  def resolve(
      self,
      q_length: int,
      kv_length: int,
      *,
      attn_logits_soft_cap: float | None,
      interpret: bool,
  ) -> Any:
    """Builds the Pallas `mtk_splash.SplashConfig` clamped to `(q_length, kv_length)`."""
    mtk_splash, _, _ = _import_splash()
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
  allow_xla_fallback: bool = False
  splash: SplashConfig = SplashConfig()

  def __post_init__(self) -> None:
    valid_backends = VALID_KERNEL_BACKENDS + ("xla",)
    if self.backend not in valid_backends:
      raise ValueError(f"AttentionKernel.backend must be one of {valid_backends}; got {self.backend!r}.")
    if self.sliding_window is not None and int(self.sliding_window) <= 0:
      raise ValueError(f"AttentionKernel.sliding_window must be > 0 when set; got {self.sliding_window!r}.")
    if self.attn_logits_soft_cap is not None and float(self.attn_logits_soft_cap) <= 0.0:
      raise ValueError(f"AttentionKernel.attn_logits_soft_cap must be > 0 when set; got {self.attn_logits_soft_cap!r}.")
    if self.gqa_head_stacking and self.backend not in ("splash", "auto"):
      raise ValueError("gqa_head_stacking=True requires backend='splash' or 'auto'.")

  @property
  def kernel_backend(self) -> KernelBackend:
    """Returns the canonical `KernelBackend` enum for `self.backend`."""
    return KernelBackend.JAX if self.backend == "xla" else KernelBackend(self.backend)

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
JAX_ATTENTION = AttentionKernel(KernelBackend.JAX)
AUTO_ATTENTION = AttentionKernel(KernelBackend.AUTO)


def _require_positive_sliding_window(config: Any) -> int:
  """Validates and returns `sliding_window_size > 0` for local sliding attention."""
  sw_len = _cfg_attr(config, "sliding_window_size", 0)
  if sw_len is None or int(sw_len) <= 0:
    raise ValueError(f"attention_type='local_sliding' requires sliding_window_size > 0; got {sw_len!r}.")
  return int(sw_len)


def validate_attention_config(config: Any, *, mesh: Any = None) -> None:
  """Validates m3 attention kernel settings eagerly before model construction."""
  attention_mode = _cfg_attr(config, "attention", "autoselected")
  if attention_mode not in _VALID_ATTENTION_MODES:
    raise ValueError(f"m3 supports attention in {_VALID_ATTENTION_MODES}; got {attention_mode!r}.")

  explicit_kb = _cfg_attr(config, "kernel_backend", None)
  if explicit_kb is not None:
    parse_kernel_backend(explicit_kb)

  use_tokamax_splash = _cfg_attr(config, "use_tokamax_splash", None)
  if use_tokamax_splash is not None:
    if not isinstance(use_tokamax_splash, bool):
      raise ValueError(
          f"use_tokamax_splash must be a boolean; got {type(use_tokamax_splash).__name__} ({use_tokamax_splash!r})."
      )
    if use_tokamax_splash and (attention_mode in ("dot_product", "jax", "xla") or explicit_kb in ("jax", "xla")):
      raise ValueError(
          "use_tokamax_splash=True requires Splash/auto attention; "
          f"got attention={attention_mode!r}, kernel_backend={explicit_kb!r}."
      )

  attention_type = _cfg_attr(config, "attention_type", "global")
  attention_type = getattr(attention_type, "value", attention_type)
  if attention_type not in _VALID_ATTENTION_TYPES:
    raise ValueError(f"m3 supports attention_type in {_VALID_ATTENTION_TYPES}; got {attention_type!r}.")

  if attention_type == "local_sliding":
    _require_positive_sliding_window(config)

  for field_name in ("sa_q_layout", "sa_k_layout", "sa_v_layout"):
    val = _cfg_attr(config, field_name, "HEAD_DIM_MINOR")
    if val is not None:
      _normalize_layout_name(val, field_name)
  for field_name in ("local_sa_q_layout", "local_sa_k_layout", "local_sa_v_layout"):
    val = _cfg_attr(config, field_name, None)
    if val is not None:
      _normalize_layout_name(val, field_name)

  gqa_stacking = bool(_cfg_attr(config, "sa_gqa_head_stacking", False))
  if gqa_stacking:
    if attention_mode in ("dot_product", "jax", "xla", "tokamax", "cudnn") or explicit_kb in (
        "jax",
        "tokamax",
        "cudnn",
    ):
      raise ValueError("sa_gqa_head_stacking=True requires Splash attention (attention='flash' or 'autoselected').")
    head_dim = _cfg_attr(config, "head_dim", None)
    if head_dim is not None and int(head_dim) != 64:
      raise ValueError(f"sa_gqa_head_stacking=True requires head_dim == 64; got head_dim={head_dim!r}.")
    n_q = _cfg_attr(config, "num_query_heads", None)
    n_kv = _cfg_attr(config, "num_kv_heads", None)
    if n_q is not None and n_kv is not None:
      if int(n_kv) <= 0 or int(n_q) % int(n_kv) != 0:
        raise ValueError(f"num_query_heads ({n_q}) must be a positive multiple of num_kv_heads ({n_kv}).")
      group_size = int(n_q) // int(n_kv)
      if group_size < 2 or group_size % 2 != 0:
        raise ValueError(
            f"sa_gqa_head_stacking=True requires an even GQA group size >= 2; got {n_q} // {n_kv} = {group_size}."
        )

  seq_len = _configured_sequence_length(config)
  if explicit_kb is not None:
    resolve_kernel_backend(explicit_kb, mesh=mesh, config=config, seq_len=seq_len)
  elif attention_mode in VALID_KERNEL_BACKENDS:
    resolve_kernel_backend(attention_mode, mesh=mesh, config=config, seq_len=seq_len)

  if attention_mode in ("flash", "splash") or explicit_kb == "splash":
    if seq_len > 0 and (seq_len < _SPLASH_TILE_ALIGNMENT or seq_len % _SPLASH_TILE_ALIGNMENT != 0):
      raise ValueError(
          f"attention='flash' requires sequence length to be a multiple of {_SPLASH_TILE_ALIGNMENT}; got {seq_len}."
      )
    SplashConfig.from_config(config, local=attention_type == "local_sliding")


def _splash_layout_problem(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_shards: int,
    kv_head_shards: int,
    q_seq_shards: int,
    kv_seq_shards: int,
    heads_axis: MeshAxes,
    kv_heads_axis: MeshAxes,
    kv_length: MeshAxes,
    gqa_head_stacking: bool,
) -> tuple[type[Exception], str] | None:
  """Returns the error Splash raises for this head and sequence sharding layout, or None if supported."""
  if kv_seq_shards > 1:
    return NotImplementedError, (
        f"Splash attention with sharded KV sequence (kv_length={kv_length!r}, shards={kv_seq_shards}) "
        "requires ring attention, which is not enabled."
    )
  if num_q_heads % head_shards != 0:
    return ValueError, (
        f"num_query_heads ({num_q_heads}) is not divisible by head shards ({head_shards}) on axis {heads_axis!r}."
    )
  if num_kv_heads % kv_head_shards != 0:
    return ValueError, (
        f"num_kv_heads ({num_kv_heads}) is not divisible by kv_head shards ({kv_head_shards}) "
        f"on axis {kv_heads_axis!r}."
    )
  if kv_head_shards not in (1, head_shards):
    return ValueError, (
        f"Splash attention requires kv_head_shards ({kv_head_shards}) to equal 1 or head_shards ({head_shards})."
    )
  if head_shards > 1 and kv_head_shards == 1 and mesh_axis_names(heads_axis) == mesh_axis_names(kv_heads_axis):
    return ValueError, "Inconsistent head and kv_head shard counts."
  local_q_heads = num_q_heads // head_shards
  local_kv_heads = num_kv_heads // kv_head_shards
  if local_q_heads % local_kv_heads != 0:
    return ValueError, (
        f"Per-shard query heads ({local_q_heads}) must be divisible by per-shard kv heads ({local_kv_heads})."
    )
  if gqa_head_stacking:
    if q_seq_shards > 1:
      return NotImplementedError, "gqa_head_stacking=True does not support sharded query sequence."
    local_group_size = local_q_heads // local_kv_heads
    if local_group_size < 2 or local_group_size % 2 != 0:
      return ValueError, (
          f"gqa_head_stacking=True requires per-shard GQA group size ({local_group_size}) to be an even integer >= 2."
      )
  return None


def _splash_setup_problem(config: Any, q_spec: P, kv_spec: P, mesh: Any, gqa_head_stacking: bool) -> str | None:
  """Checks the Splash sharding layout against `mesh` at construction when head counts are configured."""
  num_q_heads = _cfg_attr(config, "num_query_heads")
  num_kv_heads = _cfg_attr(config, "num_kv_heads")
  if num_q_heads is None or num_kv_heads is None:
    return None
  seq_len = max(_configured_sequence_length(config), 2)
  q_axes = normalize_broadcast_spec(P(q_spec[0], q_spec[2], q_spec[1], None), (2, seq_len, int(num_q_heads), 2), mesh)
  kv_axes = normalize_broadcast_spec(
      P(kv_spec[0], kv_spec[2], kv_spec[1], None), (2, seq_len, int(num_kv_heads), 2), mesh
  )
  problem = _splash_layout_problem(
      num_q_heads=int(num_q_heads),
      num_kv_heads=int(num_kv_heads),
      head_shards=mesh_axis_size(mesh, q_axes[2]),
      kv_head_shards=mesh_axis_size(mesh, kv_axes[2]),
      q_seq_shards=mesh_axis_size(mesh, q_axes[1]),
      kv_seq_shards=mesh_axis_size(mesh, kv_axes[1]),
      heads_axis=q_axes[2],
      kv_heads_axis=kv_axes[2],
      kv_length=kv_spec[2],
      gqa_head_stacking=gqa_head_stacking,
  )
  return None if problem is None else problem[1]


def attention_from_config(
    config: Any,
    sharding: Sharding | None = None,
    *,
    backend: str | KernelBackend | None = None,
    mesh: Any = None,
    causal: bool = True,
    sliding_window: int | None = None,
    attn_logits_soft_cap: float | None = None,
) -> AttentionKernel:
  """Resolves an `AttentionKernel` from `config` and `sharding` without touching traced inputs."""
  validate_attention_config(config, mesh=mesh)
  resolved_sharding = sharding or DefaultSharding()

  attention_type = _cfg_attr(config, "attention_type", "global")
  attention_type = getattr(attention_type, "value", attention_type)
  if sliding_window is None and attention_type == "local_sliding":
    sliding_window = int(_cfg_attr(config, "sliding_window_size"))

  if attn_logits_soft_cap is None:
    cfg_cap = _cfg_attr(config, "attn_logits_soft_cap", None)
    if cfg_cap is not None and float(cfg_cap) > 0.0:
      attn_logits_soft_cap = float(cfg_cap)

  attention_mode = _cfg_attr(config, "attention", "autoselected")
  explicit_kb = backend if backend is not None else _cfg_attr(config, "kernel_backend", None)
  gqa_stacking = bool(_cfg_attr(config, "sa_gqa_head_stacking", False))
  seq_len = _configured_sequence_length(config)
  strict_platform_splash = False

  if explicit_kb is not None:
    parsed_explicit = parse_kernel_backend(explicit_kb, allow_legacy_aliases=True)
    strict_platform_splash = parsed_explicit == KernelBackend.SPLASH
    resolved_backend: AttentionBackend = resolve_kernel_backend(
        parsed_explicit,
        mesh=mesh,
        config=config,
        seq_len=seq_len,
    )
  elif attention_mode in VALID_KERNEL_BACKENDS:
    parsed_mode = parse_kernel_backend(attention_mode)
    strict_platform_splash = parsed_mode == KernelBackend.SPLASH
    resolved_backend = resolve_kernel_backend(
        parsed_mode,
        mesh=mesh,
        config=config,
        seq_len=seq_len,
    )
  elif attention_mode in ("dot_product", "xla"):
    resolved_backend = "xla"
  elif attention_mode == "flash":
    resolved_backend = "splash"
  else:
    targets_tpu = mesh_platform(mesh, config) == "tpu"
    seq_aligned = seq_len >= _SPLASH_TILE_ALIGNMENT and seq_len % _SPLASH_TILE_ALIGNMENT == 0
    if gqa_stacking and not seq_aligned:
      raise ValueError(
          f"sa_gqa_head_stacking=True requires sequence length divisible by {_SPLASH_TILE_ALIGNMENT}; got {seq_len}."
      )
    resolved_backend = "splash" if (targets_tpu and seq_aligned) or gqa_stacking else "xla"

  q_spec = resolved_sharding("attention_kernel_q", ("batch", "heads", "length", "head_dim"), TensorType.ACTIVATION)
  kv_spec = resolved_sharding(
      "attention_kernel_kv", ("batch", "kv_heads", "kv_length", "head_dim"), TensorType.ACTIVATION
  )
  auto_selected = (
      attention_mode in ("autoselected", "auto")
      and explicit_kb in (None, "auto", KernelBackend.AUTO)
      and not gqa_stacking
      and not bool(_cfg_attr(config, "use_tokamax_splash", False))
  )
  setup_mesh = _active_mesh_or_none(mesh)
  if resolved_backend == "splash" and setup_mesh is not None:
    problem = _splash_setup_problem(config, q_spec, kv_spec, setup_mesh, gqa_stacking)
    if problem is not None:
      if not auto_selected:
        raise ValueError(f"Splash attention cannot run on this mesh and sharding: {problem}")
      resolved_backend = "xla"
  float32_logits = bool(_cfg_attr(config, "float32_logits", True))
  splash_cfg = (
      SplashConfig.from_config(
          config,
          local=sliding_window is not None,
          allow_cpu_interpret=not strict_platform_splash,
      )
      if resolved_backend == "splash"
      else SplashConfig(allow_cpu_interpret=not strict_platform_splash)
  )
  allow_xla_fallback = resolved_backend == "splash" and auto_selected
  return AttentionKernel(
      backend=resolved_backend,
      batch=q_spec[0],
      heads=q_spec[1],
      kv_heads=kv_spec[1],
      length=q_spec[2],
      kv_length=kv_spec[2],
      causal=causal,
      sliding_window=sliding_window,
      attn_logits_soft_cap=attn_logits_soft_cap,
      float32_logits=float32_logits,
      gqa_head_stacking=gqa_stacking if resolved_backend == "splash" else False,
      allow_xla_fallback=allow_xla_fallback,
      splash=splash_cfg,
  )


def _build_single_head_mask(
    q_len: int,
    kv_len: int,
    *,
    causal: bool,
    sliding_window: int | None,
    gqa_head_stacking: bool,
    q_seq_shards: int,
) -> Any:
  """Constructs a single-head Splash `Mask` for causal, full, SWA, or 2x stacked GQA."""
  _, mtk_mask, stacked_gqa_mask_cls = _import_splash()
  if gqa_head_stacking:
    return stacked_gqa_mask_cls(
        shape=(2 * q_len, kv_len),
        group_size=2,
        is_causal=causal,
        sliding_window=sliding_window,
        shard_count=q_seq_shards,
    )
  mask_shape = (q_len, kv_len)
  if sliding_window is not None:
    window_bounds = (sliding_window - 1, 0) if causal else (sliding_window - 1, sliding_window - 1)
    return mtk_mask.LocalMask(shape=mask_shape, window_size=window_bounds, offset=0, shard_count=q_seq_shards)
  return mtk_mask.CausalMask(shape=mask_shape, shard_count=q_seq_shards) if causal else mtk_mask.FullMask(mask_shape)


@functools.partial(jax.jit, static_argnames=("mask", "q_seq_shards", "splash_config"))
def _build_splash_kernel(
    mask: Any,
    q_seq_shards: int,
    splash_config: Any,
) -> Any:
  """Compiles and caches Splash block-mask tables per static `(mask, q_seq_shards, splash_config)`."""
  mtk_splash, _, _ = _import_splash()
  return mtk_splash.make_splash_mha(
      mask=mask,
      config=splash_config,
      q_seq_shards=q_seq_shards,
  )


def _constrain_or_reshard(x: jax.Array, spec: P | None, mesh: Any) -> jax.Array:
  """Applies `with_sharding_constraint` on Auto meshes or `ensure_explicit_spec` on Explicit meshes."""
  if spec is None or getattr(mesh, "empty", True):
    return x
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  if axis_types and all(t == jax.sharding.AxisType.Auto for t in axis_types):
    if hasattr(x, "sharding") and (not isinstance(x.sharding, NamedSharding) or x.sharding.mesh.is_scalar):
      concrete_mesh = mesh if isinstance(mesh, jax.sharding.Mesh) else None
      if concrete_mesh is None:
        try:
          concrete_mesh = jax.sharding.get_mesh()
        except ValueError:
          concrete_mesh = None
      if concrete_mesh is not None and not getattr(concrete_mesh, "empty", True):
        return jax.device_put(x, NamedSharding(concrete_mesh, spec))
      return x
    return jax.lax.with_sharding_constraint(x, spec)
  return ensure_explicit_spec(x, spec, mesh)


def _maybe_constrain_xla_qkv(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    kernel: AttentionKernel,
) -> tuple[jax.Array, jax.Array, jax.Array, P | None, P | None]:
  """Applies static head/batch/length sharding constraints on Auto or Explicit meshes when dimensions divide."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    return query, key, value, None, None
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  explicit = bool(axis_types) and all(t == jax.sharding.AxisType.Explicit for t in axis_types)
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
    if explicit:
      heads_axis = None
  q_spec = normalize_broadcast_spec(
      P(batch_axis, length_axis, heads_axis, None),
      query.shape,
      mesh,
      allow_unaligned_fallback=True,
  )
  kv_spec = normalize_broadcast_spec(
      P(batch_axis, kv_length_axis, kv_heads_axis, None),
      key.shape,
      mesh,
      allow_unaligned_fallback=True,
  )
  query = _constrain_or_reshard(query, q_spec, mesh)
  key = _constrain_or_reshard(key, kv_spec, mesh)
  value = _constrain_or_reshard(value, kv_spec, mesh)
  return query, key, value, q_spec, kv_spec


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


def _normalize_sequence_metadata(
    arr: jax.Array | None,
    *,
    batch: int,
    seq_len: int,
    spec: P | None,
    mesh: Any,
    name: str,
) -> jax.Array | None:
  """Validates and broadcast-normalizes `(batch, seq_len)` or `(1, seq_len)` sequence metadata."""
  if arr is None:
    return None
  if arr.ndim == 1 and arr.shape == (seq_len,):
    arr = arr[None, :]
  if arr.ndim == 2 and arr.shape in ((batch, seq_len), (1, seq_len)):
    target_shape = (batch, seq_len) if arr.shape[0] == 1 and batch > 1 else None
    return normalize_broadcast_sharding(
        arr,
        spec,
        mesh,
        target_shape=target_shape,
        allow_unaligned_fallback=True,
    )
  raise ValueError(f"{name} must have shape ({batch}, {seq_len}) or (1, {seq_len}); got {arr.shape}.")


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
    implementation: Literal["xla", "cudnn"] = "xla",
) -> jax.Array:
  """Portable XLA / cuDNN scaled dot-product attention via `jax.nn.dot_product_attention`.

  Note: `max_logit_value` / `SplashConfig.max_logit_const` is a single-pass
  Splash online-softmax running-max lower bound ($m_0$). XLA computes the exact
  two-pass row maximum and validates `max_logit_value` for cross-backend parity.
  """
  batch, q_len, num_q_heads, head_dim_qk = query.shape
  kv_len = key.shape[1]
  num_kv_heads = key.shape[2]
  head_dim_v = value.shape[3]

  if segment_ids is not None and q_len != kv_len:
    raise ValueError(
        f"segment_ids must have shape ({batch}, {q_len}) with q_len == kv_len; got {segment_ids.shape} "
        f"for q_len={q_len}, kv_len={kv_len}."
    )
  if sinks is not None and sinks.shape != (num_q_heads,):
    raise ValueError(f"sinks must have shape ({num_q_heads},); got {sinks.shape}.")
  _validate_max_logit_value(max_logit_value, kernel, num_q_heads)

  mesh = jax.sharding.get_abstract_mesh()
  query, key, value, out_spec, kv_spec = _maybe_constrain_xla_qkv(query, key, value, kernel)
  q_seq_spec = P(out_spec[0], out_spec[1]) if out_spec is not None else None
  k_seq_spec = P(kv_spec[0], kv_spec[1]) if kv_spec is not None else None
  segment_ids = _normalize_sequence_metadata(
      segment_ids,
      batch=batch,
      seq_len=q_len,
      spec=q_seq_spec,
      mesh=mesh,
      name="segment_ids",
  )
  if positions is not None:
    positions = _normalize_sequence_metadata(
        positions,
        batch=batch,
        seq_len=q_len,
        spec=q_seq_spec,
        mesh=mesh,
        name="positions",
    )

  # Fast native XLA dot_product_attention path when float32_logits is enabled and
  # soft-cap / sinks are unused. XLA's implementation="xla" hardcodes BF16_BF16_F32
  # for QK^T and fp32 softmax, so float32_logits=False uses the explicit einsum path.
  if kernel.float32_logits and sinks is None and kernel.attn_logits_soft_cap is None:
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
          implementation=implementation,
      ).astype(value.dtype)
    else:
      q_pos = (
          jnp.arange(q_len, dtype=jnp.int32)[None, :]
          if positions is None
          else ensure_explicit_spec(positions, q_seq_spec, mesh)
      )
      k_pos = (
          jnp.arange(kv_len, dtype=jnp.int32)[None, :]
          if (positions is None or kv_len != q_len)
          else ensure_explicit_spec(positions, k_seq_spec, mesh)
      )
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
      q_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, q_seq_spec, mesh)
      k_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, k_seq_spec, mesh)
      if q_seg is not None and k_seg is not None:
        valid_seg = (q_seg[..., :, None] == k_seg[..., None, :]) & (q_seg[..., :, None] != 0) & (k_seg[..., None, :] != 0)
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
          implementation=implementation,
      )
      out = jnp.where(any_valid[..., None], out, 0).astype(value.dtype)
      if q_seg is not None:
        out = jnp.where((q_seg != 0)[..., None, None], out, 0)
    out = _constrain_or_reshard(out, out_spec, mesh)
    if kernel.splash.residual_checkpoint_name is not None:
      out = checkpoint_name(out, kernel.splash.residual_checkpoint_name)
    return out

  groups = num_q_heads // num_kv_heads
  query_g = jnp.reshape(query, (batch, q_len, num_kv_heads, groups, head_dim_qk))
  einsum_dtype = jnp.float32 if kernel.float32_logits else None
  scores = jnp.einsum("btkgd,bskd->bkgts", query_g, key, preferred_element_type=einsum_dtype)
  if kernel.attn_logits_soft_cap is not None:
    cap = jnp.asarray(kernel.attn_logits_soft_cap, dtype=scores.dtype)
    scores = jnp.tanh(scores / cap) * cap

  q_pos = (
      jnp.arange(q_len, dtype=jnp.int32)[None, :]
      if positions is None
      else ensure_explicit_spec(positions, q_seq_spec, mesh)
  )
  k_pos = (
      jnp.arange(kv_len, dtype=jnp.int32)[None, :]
      if (positions is None or kv_len != q_len)
      else ensure_explicit_spec(positions, k_seq_spec, mesh)
  )
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
  q_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, q_seq_spec, mesh)
  k_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, k_seq_spec, mesh)
  if q_seg is not None and k_seg is not None:
    valid_seg = (q_seg[..., :, None] == k_seg[..., None, :]) & (q_seg[..., :, None] != 0) & (k_seg[..., None, :] != 0)
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
    logits_sharding = jax.typeof(logits).sharding
    if hasattr(logits_sharding, "spec"):
      logits_axes = tuple(logits_sharding.spec) + (None,) * (4 - len(logits_sharding.spec))
      sink_col = ensure_explicit_spec(sink_col, P(*logits_axes[:3], None), jax.sharding.get_abstract_mesh())
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
  if q_seg is not None:
    out = jnp.where((q_seg != 0)[..., None, None], out, 0)
  out = _constrain_or_reshard(out, out_spec, mesh)
  if kernel.splash.residual_checkpoint_name is not None:
    out = checkpoint_name(out, kernel.splash.residual_checkpoint_name)
  return out


def _remap_padding_segment_ids(segment_ids: jax.Array, offset: int = 0) -> jax.Array:
  """Maps `segment_ids == 0` padding slots to distinct negative IDs so Splash never sees 0/0 NaNs."""
  seq_len = segment_ids.shape[-1]
  neg_ids = -(jnp.arange(seq_len, dtype=jnp.int32) + 1 + offset)
  return jnp.where(segment_ids == 0, neg_ids, segment_ids.astype(jnp.int32))


def _cudnn_attention(
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
  """GPU cuDNN flash attention backend; fails closed on non-GPU platforms."""
  platform = mesh_platform()
  if platform != "gpu":
    raise ValueError(
        f"KernelBackend 'cudnn' requires a GPU mesh/device; got platform {platform!r}. "
        "Use backend='auto' or 'jax' on non-GPU platforms."
    )
  return _xla_attention(
      query,
      key,
      value,
      kernel=kernel,
      segment_ids=segment_ids,
      positions=positions,
      sinks=sinks,
      max_logit_value=max_logit_value,
      implementation="cudnn",
  )


def _tokamax_attention(
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
  """Tokamax dot-product attention backend with broadcast-sharding normalization."""
  tokamax = _import_tokamax(required=True)
  batch, q_len, num_q_heads, _ = query.shape
  kv_len = key.shape[1]

  if segment_ids is not None and q_len != kv_len:
    raise ValueError(
        f"segment_ids must have shape ({batch}, {q_len}) with q_len == kv_len; got {segment_ids.shape} "
        f"for q_len={q_len}, kv_len={kv_len}."
    )
  if sinks is not None or not kernel.float32_logits:
    if sinks is not None and sinks.shape != (num_q_heads,):
      raise ValueError(f"sinks must have shape ({num_q_heads},); got {sinks.shape}.")
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
  _validate_max_logit_value(max_logit_value, kernel, num_q_heads)

  mesh = jax.sharding.get_abstract_mesh()
  query, key, value, out_spec, kv_spec = _maybe_constrain_xla_qkv(query, key, value, kernel)
  q_seq_spec = P(out_spec[0], out_spec[1]) if out_spec is not None else None
  k_seq_spec = P(kv_spec[0], kv_spec[1]) if kv_spec is not None else None
  segment_ids = _normalize_sequence_metadata(
      segment_ids,
      batch=batch,
      seq_len=q_len,
      spec=q_seq_spec,
      mesh=mesh,
      name="segment_ids",
  )
  if positions is not None:
    positions = _normalize_sequence_metadata(
        positions,
        batch=batch,
        seq_len=q_len,
        spec=q_seq_spec,
        mesh=mesh,
        name="positions",
    )

  q_in = query.astype(value.dtype)
  k_in = key.astype(value.dtype)
  v_in = value
  if positions is None and segment_ids is None and kernel.sliding_window is None:
    out = tokamax.dot_product_attention(
        q_in,
        k_in,
        v_in,
        scale=1.0,
        is_causal=kernel.causal,
        logits_soft_cap=kernel.attn_logits_soft_cap,
    ).astype(value.dtype)
  else:
    q_pos = (
        jnp.arange(q_len, dtype=jnp.int32)[None, :]
        if positions is None
        else ensure_explicit_spec(positions, q_seq_spec, mesh)
    )
    k_pos = (
        jnp.arange(kv_len, dtype=jnp.int32)[None, :]
        if (positions is None or kv_len != q_len)
        else ensure_explicit_spec(positions, k_seq_spec, mesh)
    )
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
    q_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, q_seq_spec, mesh)
    k_seg = None if segment_ids is None else ensure_explicit_spec(segment_ids, k_seq_spec, mesh)
    if q_seg is not None and k_seg is not None:
      valid_seg = (q_seg[..., :, None] == k_seg[..., None, :]) & (q_seg[..., :, None] != 0) & (k_seg[..., None, :] != 0)
      mask = mask & valid_seg
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    fallback_row = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :] == 0
    safe_mask = jnp.where(any_valid, mask, fallback_row)
    out = tokamax.dot_product_attention(
        q_in,
        k_in,
        v_in,
        mask=safe_mask[:, None, :, :],
        scale=1.0,
        is_causal=False,
        logits_soft_cap=kernel.attn_logits_soft_cap,
    )
    out = jnp.where(any_valid[..., None], out, 0).astype(value.dtype)
    if q_seg is not None:
      out = jnp.where((q_seg != 0)[..., None, None], out, 0)
  out = _constrain_or_reshard(out, out_spec, mesh)
  if kernel.splash.residual_checkpoint_name is not None:
    out = checkpoint_name(out, kernel.splash.residual_checkpoint_name)
  return out


def _splash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    kernel: AttentionKernel,
    segment_ids: jax.Array | None,
    positions: jax.Array | None = None,
    sinks: jax.Array | None,
    max_logit_value: jax.Array | float | None,
) -> jax.Array:
  """Sharded Pallas Splash attention supporting SEQ_MINOR, SWA, packing, and 2x GQA head stacking."""
  mtk_splash, _, _ = _import_splash()
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
  if segment_ids is not None and q_len != kv_len:
    raise ValueError(
        f"Splash segment_ids must have shape ({batch}, {q_len}) with q_len == kv_len; got {segment_ids.shape} "
        f"for q_len={q_len}, kv_len={kv_len}."
    )
  if (
      positions is not None
      and segment_ids is None
      and not isinstance(positions, jax.core.Tracer)
      and positions.shape[-1] > 1
      and not bool(np.all(np.diff(np.asarray(positions), axis=-1) == 1))
  ):
    raise ValueError(
        "Splash attention requires unit-step contiguous positions when segment_ids is None; "
        "use backend='auto', 'jax', or 'tokamax', or pass segment_ids for packed sequences."
    )
  if sinks is not None and sinks.shape != (num_q_heads,):
    raise ValueError(f"sinks must have shape ({num_q_heads},); got {sinks.shape}.")

  mesh = mesh_in_scope()
  if not is_tpu_mesh(mesh) and not kernel.splash.allow_cpu_interpret:
    raise ValueError(
        f"KernelBackend 'splash' requires a TPU mesh/device; got platform {mesh_platform(mesh)!r}. "
        "Use backend='auto' or 'jax' on non-TPU platforms."
    )
  raw_q_spec = normalize_broadcast_spec(
      P(kernel.batch, kernel.length, kernel.heads, None),
      query.shape,
      mesh,
  )
  raw_kv_spec = normalize_broadcast_spec(
      P(kernel.batch, kernel.kv_length, kernel.kv_heads, None),
      key.shape,
      mesh,
  )
  assert raw_q_spec is not None and raw_kv_spec is not None
  batch_axis = raw_q_spec[0]
  length_axis = raw_q_spec[1]
  heads_axis = raw_q_spec[2]
  kv_length_axis = raw_kv_spec[1]
  kv_heads_axis = raw_kv_spec[2]
  if segment_ids is not None:
    segment_ids = _normalize_sequence_metadata(
        segment_ids,
        batch=batch,
        seq_len=q_len,
        spec=P(batch_axis, length_axis),
        mesh=mesh,
        name="Splash segment_ids",
    )
  if positions is not None:
    positions = _normalize_sequence_metadata(
        positions,
        batch=batch,
        seq_len=q_len,
        spec=P(batch_axis, length_axis),
        mesh=mesh,
        name="positions",
    )

  batch_shards = mesh_axis_size(mesh, batch_axis)
  head_shards = mesh_axis_size(mesh, heads_axis)
  kv_head_shards = mesh_axis_size(mesh, kv_heads_axis)
  q_seq_shards = mesh_axis_size(mesh, length_axis)
  kv_seq_shards = mesh_axis_size(mesh, kv_length_axis)

  problem = _splash_layout_problem(
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      head_shards=head_shards,
      kv_head_shards=kv_head_shards,
      q_seq_shards=q_seq_shards,
      kv_seq_shards=kv_seq_shards,
      heads_axis=heads_axis,
      kv_heads_axis=kv_heads_axis,
      kv_length=kernel.kv_length,
      gqa_head_stacking=kernel.gqa_head_stacking,
  )
  if problem is not None:
    exc_type, message = problem
    raise exc_type(message)
  if batch % batch_shards != 0:
    raise ValueError(f"Batch size ({batch}) is not divisible by batch shards ({batch_shards}) on axis {batch_axis!r}.")
  if kernel.gqa_head_stacking:
    if q_len != kv_len:
      raise ValueError(f"gqa_head_stacking=True requires q_len == kv_len; got {q_len} vs {kv_len}.")
    if head_dim_qk != 64 or head_dim_v != 64:
      raise ValueError(
          f"gqa_head_stacking=True requires head_dim == 64 for Q/K/V; got D_qk={head_dim_qk}, D_v={head_dim_v}."
      )
    if sinks is not None:
      raise ValueError("gqa_head_stacking=True does not support attention sinks.")

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
  q_in = ensure_explicit_spec(query, q_spec, mesh)
  k_in = ensure_explicit_spec(key, kv_spec, mesh)
  v_in = ensure_explicit_spec(value, kv_spec, mesh)
  seg_q_in = ensure_explicit_spec(seg_clean, seg_q_spec, mesh)
  seg_kv_in = ensure_explicit_spec(seg_clean, seg_kv_spec, mesh)
  sinks_in = ensure_explicit_spec(sinks, sinks_spec, mesh)
  max_lv_in = ensure_explicit_spec(max_lv_arr, max_lv_spec, mesh)
  out = sharded_fn(q_in, k_in, v_in, seg_q_in, seg_kv_in, sinks_in, max_lv_in, splash_kern)
  if segment_ids is not None:
    seg_orig_q = ensure_explicit_spec(segment_ids, seg_q_spec, mesh)
    out = jnp.where((seg_orig_q != 0)[..., None, None], out, 0)
  return out


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    kernel: AttentionKernel = XLA_ATTENTION,
    backend: str | KernelBackend | None = None,
    segment_ids: jax.Array | None = None,
    positions: jax.Array | None = None,
    sinks: jax.Array | None = None,
    max_logit_value: jax.Array | float | None = None,
) -> jax.Array:
  """Dispatches scaled dot-product attention (`query` pre-scaled) to `backend` or `kernel.backend`."""
  if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
    raise ValueError(
        f"query, key, value must be rank-4 (batch, length, heads, head_dim); "
        f"got shapes {query.shape}, {key.shape}, {value.shape}."
    )
  if key.shape[0] != value.shape[0]:
    raise ValueError(
        f"Batch size mismatch across query ({query.shape[0]}), key ({key.shape[0]}), value ({value.shape[0]})."
    )
  if key.shape[0] == 1 and query.shape[0] > 1:
    active_mesh = _active_mesh_or_none()
    kv_raw_spec = P(kernel.batch, kernel.kv_length, kernel.kv_heads, None)
    key = normalize_broadcast_sharding(
        key,
        kv_raw_spec,
        active_mesh,
        target_shape=(query.shape[0],) + key.shape[1:],
        allow_unaligned_fallback=True,
    )
    value = normalize_broadcast_sharding(
        value,
        kv_raw_spec,
        active_mesh,
        target_shape=(query.shape[0],) + value.shape[1:],
        allow_unaligned_fallback=True,
    )
    assert key is not None and value is not None
  elif query.shape[0] != key.shape[0]:
    raise ValueError(
        f"Batch size mismatch across query ({query.shape[0]}), key ({key.shape[0]}), value ({value.shape[0]})."
    )
  if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
    raise ValueError(f"Key ({key.shape}) and value ({value.shape}) must match in sequence length and kv_heads.")
  if query.shape[3] != key.shape[3]:
    raise ValueError(f"Query head_dim ({query.shape[3]}) must match key head_dim ({key.shape[3]}).")
  if key.shape[2] <= 0 or query.shape[2] % key.shape[2] != 0:
    raise ValueError(f"Query heads ({query.shape[2]}) must be a positive multiple of KV heads ({key.shape[2]}).")

  if backend is not None:
    resolved_backend: str = resolve_kernel_backend(
        backend,
        mesh=_active_mesh_or_none(),
        seq_len=query.shape[1],
        allow_legacy_aliases=True,
    ).value
    if resolved_backend == "splash" and kernel.splash.allow_cpu_interpret:
      kernel = dataclasses.replace(kernel, splash=dataclasses.replace(kernel.splash, allow_cpu_interpret=False))
  elif kernel.backend == "auto":
    resolved_backend = resolve_kernel_backend(
        KernelBackend.AUTO,
        mesh=_active_mesh_or_none(),
        seq_len=query.shape[1],
    ).value
  else:
    resolved_backend = str(kernel.backend)

  if resolved_backend in ("xla", "jax"):
    return _xla_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
        implementation="xla",
    )
  if resolved_backend == "cudnn":
    return _cudnn_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )
  if resolved_backend == "tokamax":
    return _tokamax_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )
  if resolved_backend == "splash":
    non_unit_positions = (
        positions is not None
        and segment_ids is None
        and not isinstance(positions, jax.core.Tracer)
        and positions.shape[-1] > 1
        and not bool(np.all(np.diff(np.asarray(positions), axis=-1) == 1))
    )
    if kernel.allow_xla_fallback and (
        query.shape[1] < _SPLASH_TILE_ALIGNMENT
        or query.shape[1] % _SPLASH_TILE_ALIGNMENT != 0
        or key.shape[1] < _SPLASH_TILE_ALIGNMENT
        or key.shape[1] % _SPLASH_TILE_ALIGNMENT != 0
        or non_unit_positions
    ):
      return _xla_attention(
          query,
          key,
          value,
          kernel=kernel,
          segment_ids=segment_ids,
          positions=positions,
          sinks=sinks,
          max_logit_value=max_logit_value,
          implementation="xla",
      )
    return _splash_attention(
        query,
        key,
        value,
        kernel=kernel,
        segment_ids=segment_ids,
        positions=positions,
        sinks=sinks,
        max_logit_value=max_logit_value,
    )
  raise ValueError(f"Unsupported attention backend: {resolved_backend!r}.")
