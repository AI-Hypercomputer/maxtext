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
# ==============================================================================

"""GDN Configuration dataclass defining GDN tiling, dtypes, and kernel configurations."""

import dataclasses
import enum
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_VMEM_LIMIT_FACTOR: float = 0.90

# Optional GDNConfig fields and their defaults.
_TRI_INV_FIELD_DEFAULTS = {
    "tri_inv_algo": "xor_dnc",
    "head_group_size": None,
    "emit_t_inv": True,
}

# GDNConfig fields that change how the kernel lays out a value it computes either way, not what it computes or
# returns. They are left out of the kernel name and metadata, so the lowered program outside the kernel body is the
# same with and without them, and a profile shows the same kernel name for both variants (the run's config tells
# them apart).
_LAYOUT_ONLY_FIELDS = ("conv_dense_token_stack", "is_prefill_only")


class GDNMode(enum.StrEnum):
  """Execution mode for GDN kernel."""

  BATCHED = enum.auto()
  PER_SEQ = enum.auto()
  # PER_SEQ that only carries the conv/recurrent states and t_inv: no output, no chunk states.
  PER_SEQ_STATES = enum.auto()

  def get_seq_tile_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return tile_size
    return 1

  def get_chunk_size(self, tile_size: int) -> int:
    if self == GDNMode.BATCHED:
      return 1
    return tile_size


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
  """Dtypes used across GDN operations."""

  act_in: jnp.dtype
  act_out: jnp.dtype
  compute: jnp.dtype
  recurrent_state: jnp.dtype
  conv_state: jnp.dtype


def get_vmem_limit_bytes(
    vmem_limit_factor: float = DEFAULT_VMEM_LIMIT_FACTOR,
) -> int:
  """Returns the maximum allowable VMEM capacity budget in bytes."""
  tpu_info = pltpu.get_tpu_info()
  return int(vmem_limit_factor * tpu_info.vmem_capacity_bytes)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
  """Configuration dataclass for GDN kernel."""

  mode: GDNMode
  dtypes: Dtypes
  batch_size: int
  dim_size: int
  kernel_size: int
  tile_size: int
  num_kq_heads: int
  num_v_heads: int
  kq_head_dim: int
  v_head_dim: int
  num_buffers: int = 2
  has_seg_ids: bool = False
  use_qk_norm_in_gdn: bool = True
  tri_inv_algo: str = "xor_dnc"
  head_group_size: int | None = None
  emit_t_inv: bool = True
  # Build the Conv1D output from dense per-token views (compute_conv1d.stack_conv1d_tokens; config
  # gdn_conv1d_dense_token_stack). GDNMode.PER_SEQ / PER_SEQ_STATES only.
  conv_dense_token_stack: bool = False
  # Prefill-only tiles are whole chunks: qkv is DMA'd in act_in dtype and upcast in VMEM.
  is_prefill_only: bool = False

  @property
  def chunk_size(self) -> int:
    return self.mode.get_chunk_size(self.tile_size)

  @property
  def seq_tile_size(self) -> int:
    return self.mode.get_seq_tile_size(self.tile_size)

  @property
  def prev_kernel_size(self) -> int:
    return self.kernel_size - 1

  @property
  def v_dim_size(self) -> int:
    return self.num_v_heads * self.v_head_dim

  @property
  def kq_dim_size(self) -> int:
    return self.num_kq_heads * self.kq_head_dim

  @property
  def v_per_kq_head(self) -> int:
    return self.num_v_heads // self.num_kq_heads

  @property
  def aligned_num_v_heads(self) -> int:
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    extra_lanes = 2 if self.has_seg_ids else 0
    return pl.cdiv(self.num_v_heads + extra_lanes, num_lanes) * num_lanes

  def get_kernel_name(self) -> str:
    """Pallas kernel name; non-default triangular-inverse options are appended as suffixes."""
    name = f"fused_conv1d_gdn_{self.mode.value}_b{self.seq_tile_size}" f"_c{self.chunk_size}"
    if self.tri_inv_algo != "xor_dnc":
      name += f"_{self.tri_inv_algo}"
    if self.head_group_size is not None:
      name += f"_hg{self.head_group_size}"
    if not self.emit_t_inv:
      name += "_notinv"
    return name

  def get_metadata(self) -> dict[str, str | int | float]:
    """Flattened config fields for the kernel's metadata (knobs at their defaults and layout-only fields are left out)."""
    cfgs_dict = dataclasses.asdict(self)
    for field_name, default in _TRI_INV_FIELD_DEFAULTS.items():
      if cfgs_dict.get(field_name) == default:
        cfgs_dict.pop(field_name)
    for field_name in _LAYOUT_ONLY_FIELDS:
      cfgs_dict.pop(field_name)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
      key = jax.tree_util.keystr(path, simple=True, separator=".")
      if not isinstance(val, str | int | float):
        val = str(val)
      ret[key] = val
    return ret

  def get_out_shape(self) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        (self.batch_size, self.num_v_heads, self.v_head_dim),
        self.dtypes.act_out,
    )

  def get_scratch_shape_dict(self) -> dict[str, Any]:
    """Returns dictionary of scratch shapes for Pallas pipeline."""
    conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1, self.dim_size)
    recurrent_shape = (
        self.seq_tile_size,
        self.num_v_heads,
        self.kq_head_dim,
        self.v_head_dim,
    )

    carry_conv_scratch = carry_recurrent_scratch = None
    # NOTE: Currently, batched mode only supports case where 1 seq = 1 tile.
    # Therefore, inter tile carry is not needed.
    if self.mode != GDNMode.BATCHED:
      carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
      carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

    ret = {
        "carry_conv_scratch_ref": carry_conv_scratch,
        "carry_recurrent_scratch_ref": carry_recurrent_scratch,
    }
    if self.is_prefill_only:
      ret["qkv_scratch_ref"] = pltpu.VMEM((self.seq_tile_size, self.chunk_size, 1, self.dim_size), jnp.float32)
    return ret
