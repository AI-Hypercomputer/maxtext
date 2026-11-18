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

"""Unified pure-JAX configuration for Qwen MAX (Qwen3.8-2.4T-A95B) & Qwen 3.5 (397B-A17B) on TPU7x."""

from __future__ import annotations

import dataclasses
from typing import Literal

PEAK_TFLOPS_PER_CHIP_BF16_V7 = 2307.0


@dataclasses.dataclass(frozen=True)
class QwenConfig:
  """Architecture, kernel, sharding, and Dual-SparseCore batch-split config.

  Supports both Qwen MAX (`Qwen3.8-2.4T-A95B`: `92L = 23` superblocks, `D=8192`,
  `512` routed `top-10` `E_dim=2048` + `1` shared `2048`) and Qwen 3.5
  (`Qwen3.5-397B-A17B`: `60L = 15` superblocks, `D=4096`, `512` routed `top-10`
  `E_dim=1024` + `1` shared `1024`).

  Core-Minor `[ICI, core]` Invariant (`decouple_ep_d2d=True`):
  - Physical Mesh axes are ordered `('fsdp', 'expert', 'ep_d2d')` (`[yx, z,
  core]`).
  - Multi-axis EP collectives use `ep_axis_names = ('expert', 'ep_d2d')` (`[ICI,
  core]`)
    so adjacent chiplets `[2k, 2k+1]` on each Ghostfish package alternate in the
    ring
    (`~186-200 GB/s` single-SC ReduceScatter, vs `~97 GB/s` under `[core,
    ICI]`).
  - Dense Stage-2 / narrow collectives use
    `dense_stage2_axis_names = ('fsdp', 'ep_d2d')` (`[yx, core]` core-minor,
    `chiplet_scale_factor=1`).
  - Full-mesh dense weights use `all_sharded_axes`:
    - When `dense_1stage_3d_fsdp=True` (default): `('fsdp', 'expert', 'ep_d2d')`
      matching the physical Mesh so SC1 AllGather/ReduceScatter execute in a
      SINGLE 3D core-minor collective across all 6 torus links without a
      Stage-1 -> Stage-2 barrier.
    - When `dense_1stage_3d_fsdp=False`: `('fsdp', 'ep_d2d', 'expert')` for
      2-stage gather (`'expert'` then `('fsdp', 'ep_d2d')`).
  """

  model_variant: Literal["qwen_max", "qwen3_5", "custom"] = "qwen_max"

  # Core architecture (4 layers/superblock: [GDN_0, GDN_1, GDN_2, FullAttn_3])
  num_superblocks: int = 23
  full_num_superblocks: int = (
      23  # Full-model superblock count for proxy MFU extrapolation
  )
  emb_dim: int = 8192
  vocab_size: int = 248320
  norm_eps: float = 1e-6

  # Full Attention (1 per superblock)
  num_query_heads: int = 64
  num_kv_heads: int = 4
  head_dim: int = 256
  partial_rotary_factor: float = 0.25
  rope_max_timescale: float = 10_000_000.0

  # Gated DeltaNet (3 per superblock)
  gdn_num_k_heads: int = 16
  gdn_num_v_heads: int = 128
  gdn_head_k_dim: int = 128
  gdn_head_v_dim: int = 128
  gdn_conv_kernel_size: int = 4
  gdn_chunk_size: int = 64
  gdn_fwd_head_group_size: int = 4
  gdn_bwd_natural_layout: str = "all"

  # Mixture-of-Experts (in every sub-layer)
  num_experts: int = 512
  num_experts_per_tok: int = 10
  moe_mlp_dim: int = 2048
  shared_mlp_dim: int = 2048
  norm_topk_prob: bool = True
  float32_gate_logits: bool = True
  routing_mode: Literal["router", "random"] = "random"
  routing_seed: int = 0
  ragged_buffer_factor: float = 1.05
  moe_routing_index_math: Literal["sort", "count"] = "count"
  ragged_unsort_spread_masked_indices: bool = True

  # Training batch & parallelism (Global Invariant: seq_len = 4096)
  per_device_batch_size: int = 2
  seq_len: int = 4096
  fsdp_size: int = 32
  ep_size: int = 16
  mesh_mode: str = "x1e_yx_core"
  decouple_ep_d2d: bool = True  # Core-minor ('fsdp', 'expert', 'ep_d2d') mesh!
  ep_core_minor: bool = (
      True  # Enforces ('expert', 'ep_d2d') [ICI, core] ring order
  )
  dense_1stage_3d_fsdp: bool = (
      True  # Single-stage 3D ('fsdp', 'expert', 'ep_d2d') SC1 dense AG/RS
  )
  num_vocab_chunks: int = 4
  num_vocab_tiling: int = 2

  # Optimizer (Optax AdamW with fp32/bf16 state or stateless SGD)
  optimizer: Literal["adamw", "sgd"] = "adamw"
  learning_rate: float = 1e-4
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  grad_clip_norm: float = 0.0
  mu_dtype: Literal["bfloat16", "float32"] = "bfloat16"
  master_weights_fp32: bool = False
  embed_mode: Literal[
      "sc_gather", "vocab_all_to_all", "ep_all_to_all_fsdp_gather"
  ] = "ep_all_to_all_fsdp_gather"

  # Mixed-precision & kernel controls (BF16+ mandatory on ALL reductions!)
  use_pallas_kernels: bool = True
  moe_fp8_weights: bool = True  # gather8 on SC1 (`float8_e4m3fn`)
  moe_fp8_gmm_compute: bool = (
      True  # native fp8_e4m3fn x fp8_e4m3fn MXU in gmm_v2/tgmm_v2
  )
  moe_fp8_token_dispatch: bool = (
      True  # fwd EP token all-gather in float8_e4m3fn
  )
  moe_fp8_bwd_dispatch: bool = (
      True  # bwd EP combine transpose all-gather in float8_e5m2/e4m3fn
  )
  moe_fp8_combine_sc0: bool = (
      False  # NEVER reduce in FP8; keep False (BF16+ mandatory on RS/AR)
  )
  moe_use_ragged_silu_3d_fp8: bool = True
  moe_use_forked_3d_gmmv2: bool = True
  moe_act_fp8_scale: float = 0.5
  moe_bwd_dy_fp8_e5m2: bool = True
  moe_local_65key_sort: bool = True
  moe_save_bank_in_superblock: bool = True
  sa_qk_seq_minor: bool = True
  moe_bwd_tie_wgrad_before_gate: bool = True
  f2_light_router_ties: bool = True

  # Optional dense FP8 matmul levers (BF16+ reductions preserved!)
  dense_fp8_matmul: bool = False
  dense_fp8_gdn: bool = False
  dense_fp8_attn: bool = False
  dense_fp8_shared_expert: bool = False
  dense_fp8_lm_head: bool = False

  # Zero-host-offload HBM activation banking (eliminates PCIe arrival skew!)
  save_moe_routing: bool = False
  save_attn_residuals_in_lineage: bool = False
  save_gdn_residuals_sublayers: tuple[int, ...] = ()
  gdn_keep_bf16_states: bool = True
  gdn_recompute_t_inv: bool = False
  save_moe_gate_up_sublayers: tuple[int, ...] = ()
  save_moe_gate_up_recompute_disp: bool = True
  offload_moe_gate_up_sublayers: tuple[
      int, ...
  ] = ()  # Default () = ZERO S(5) host offload!
  save_moe_disp_ep_sublayers: tuple[int, ...] = ()
  offload_moe_disp_ep_sublayers: tuple[int, ...] = ()

  # Forked 3D VMEM-relayout gmmv2 tile sizes
  gate_fwd_tiles: tuple[int, int, int, int] = (512, 2048, 4096, 256)
  lc_fwd_tiles: tuple[int, int, int] = (256, 512, 64)
  lbwd_dgrad_tiles: tuple[int, int, int, int] = (256, 128, 256, 1024)
  lbwd_wgrad_tiles: tuple[int, int, int] = (1024, 256, 2048)
  gbwd_fused_tiles: tuple[int, int, int] = (1024, 256, 1024)

  # SplashAttention block sizes
  sa_block_q: int = 1024
  sa_block_kv: int = 1024
  sa_block_kv_compute: int = 1024
  sa_block_q_dkv: int = 1024
  sa_block_kv_dkv: int = 1024
  sa_block_kv_dkv_compute: int = 1024

  # Megablox / Tokamax gmm_v2 & tgmm_v2 tile sizes
  wi_tile_fwd: tuple[int, int, int] = (768, 4096, 1024)
  wi_tile_fwd_3d_fp8: tuple[int, int, int] = (768, 1024, 4096)
  wi_tile_dlhs: tuple[int, int, int] = (768, 4096, 1024)
  wi_tile_dlhs_fp8: tuple[int, int, int] = (768, 4096, 2048)
  wi_tile_dlhs_3d_fp8: tuple[int, int, int] = (1024, 4096, 1024)
  wi_tile_drhs: tuple[int, int, int] = (768, 1024, 4096)
  wi_tile_drhs_fp8: tuple[int, int, int] = (768, 2048, 2048)
  wi_tile_drhs_3d_fp8: tuple[int, int, int] = (768, 1024, 4096)
  wi_fwd_rhs_buffer_count: int | None = 2
  wo_tile_fwd: tuple[int, int, int] = (768, 2048, 2048)
  wo_tile_fwd_3d_fp8: tuple[int, int, int] = (1024, 2048, 1024)
  wo_tile_dlhs: tuple[int, int, int] = (512, 2048, 2048)
  wo_tile_dlhs_fp8: tuple[int, int, int] = (768, 4096, 2048)
  wo_tile_dlhs_3d_fp8: tuple[int, int, int] = (768, 1024, 2048)
  wo_tile_drhs: tuple[int, int, int] = (768, 1024, 4096)
  wo_tile_drhs_fp8: tuple[int, int, int] = (768, 2048, 2048)
  wo_tile_drhs_3d_fp8: tuple[int, int, int] = (1024, 2048, 1024)

  # 3D zero-relayout TensorCore ragged sort/unsort
  moe_use_tc_ragged: bool = True
  moe_3d_ep_tokens: bool = True
  tc_gather_block_size: int = 1792
  tc_reduce_block_size: int = 896

  # Schedule & Dual-SparseCore (SC0 EP vs SC1 FSDP) knobs
  schedule_mode: Literal["unsplit", "batchsplit"] = "batchsplit"
  ep_sc_pinning: Literal["sc0", "unpinned"] = "sc0"
  fsdp_ag_sc_pinning: Literal["sc1", "sc1_fwd_only", "unpinned"] = "sc1"
  fsdp_rs_sc_pinning: Literal["sc1", "unpinned"] = "sc1"
  fsdp_pipeline_mode: Literal[
      "post_rs_tie", "sublayer_prefetch", "full_prefetch_defer_rs"
  ] = "full_prefetch_defer_rs"
  bwd_tie_mode: Literal[
      "all4", "moe_only", "moe_xmb0_only", "symmetric_4phase", "none"
  ] = "symmetric_4phase"
  dense_sharding_mode: Literal["fsdp_only", "fsdp_expert"] = "fsdp_expert"
  moe_ag_start_tie: Literal["f2_x_mb1", "f2_base_mb0", "none"] = "f2_base_mb0"
  moe_rw_prefetch_in_f3: bool = False
  bwd_sc_balance_mode: Literal[
      "sc1_all", "wi_rs_sc0", "wi_dense_rs_sc0", "dwo_sc0"
  ] = "sc1_all"
  route_before_shared_expert: bool = False
  ep_combine_fwd_unpinned: bool = False
  defer_meta_ag_to_dispatch: bool = True
  pipeline_bwd_remat_dispatch: bool = True
  lineage_defer_mb1_combine: bool = True
  gdn_rms_norm_barrier_stages: bool = True
  routed_w_tiled_layout: bool = True
  ragged_single_sc: bool = True
  tie_f4_before_wo: bool = False
  fuse_sublayer_sc1_collectives: bool = True
  strict_f2_before_f3: bool = True
  defer_mb1_combine_after_f1: bool = True
  precompute_rope_tables: bool = True
  use_max_async_overlap_transform: bool = True
  max_overlap_bundle_quant: bool = True
  max_overlap_bundle_scalar_ar: bool = True
  max_overlap_queues: Literal["all", "sc_only", "sc0_only"] = "sc_only"
  max_overlap_order_mode: Literal["sum", "done_first", "bwd_done_first"] = (
      "bwd_done_first"
  )
  fp8_amax_no_combine: bool = True
  f2_early_mb0_dispatch: bool = True
  attn_qkv_2d_matmul: bool = False
  attn_bf16_out_fused_gate: bool = True
  moe_gmm_no_alias_out: bool = True
  gdn_norm_layout_mode: Literal[
      "token_minor", "bf16_z_token_minor", "natural"
  ] = "natural"
  attn_gate_bf16_align: bool = False
  bwd_tie_b1_before_prev_b4: bool = False
  bwd_early_mb1_dy_pmax: bool = False
  bwd_prefetch_prev_attn3_moe_weights: bool = False
  sc_min_collective_bytes: int = 250_000
  bundle_stage1_dense_collectives: bool = False
  replicate_shared_gate: bool = True
  narrow_dense_stage2_only: bool = True
  narrow_dense_max_cols: int = 1024
  hoist_replicated_grad_ar: bool = True
  sc0_bwd_dy_pmax: bool = False
  remat_bwd_tie_prev_dense_depth: int = 0
  remat_bwd_wi_prefetch_prev_sublayer: bool = False
  use_lineage_pipelined_scan: bool = True

  # HBM activation save & Zero-S(5) host offload policy
  enable_remat: bool = True
  remat_superblock_carry_only: bool = True
  remat_bwd_tie_mode: Literal["dx_mb1", "dense_rs", "none"] = "dx_mb1"
  wrap_pt_across_superblock: bool = True
  save_key_value_proj: bool = True
  save_gdn_residuals: bool = False
  save_decoder_input_hbm: bool = True
  offload_decoder_input: bool = False
  offload_attn_context: bool = False
  offload_query_proj: bool = False
  offload_out_proj: bool = False
  offload_gdn_qkv: bool = False
  save_attn_context_hbm: bool = True
  save_query_proj_hbm: bool = True
  save_out_proj_hbm: bool = True
  hbm_save_offload_targets: bool = False
  vocab_tiling_ag_once: bool = True

  @property
  def num_layers(self) -> int:
    return 4 * self.num_superblocks

  @property
  def full_num_layers(self) -> int:
    return 4 * self.full_num_superblocks

  @property
  def num_devices(self) -> int:
    return self.fsdp_size * self.ep_size

  @property
  def num_chips(self) -> int:
    return max(1, self.num_devices // 2)

  @property
  def ep_ici_size(self) -> int:
    if self.decouple_ep_d2d:
      if self.ep_size % 2 != 0:
        raise ValueError(
            f"decouple_ep_d2d=True requires even ep_size, got {self.ep_size}"
        )
      return self.ep_size // 2
    return self.ep_size

  @property
  def ep_axis_names(self) -> str | tuple[str, str]:
    if not self.decouple_ep_d2d:
      return "expert"
    return ("expert", "ep_d2d") if self.ep_core_minor else ("ep_d2d", "expert")

  @property
  def ep_axis_tuple(self) -> tuple[str, ...]:
    if not self.decouple_ep_d2d:
      return ("expert",)
    return ("expert", "ep_d2d") if self.ep_core_minor else ("ep_d2d", "expert")

  @property
  def dense_stage2_axis_names(self) -> str | tuple[str, str]:
    return ("fsdp", "ep_d2d") if self.decouple_ep_d2d else "fsdp"

  @property
  def dense_stage2_size(self) -> int:
    return self.fsdp_size * 2 if self.decouple_ep_d2d else self.fsdp_size

  @property
  def all_sharded_axes(self) -> tuple[str, ...]:
    # PartitionSpec axis order for full-mesh dense weights & data:
    # - `dense_1stage_3d_fsdp=True`: `('fsdp', 'expert', 'ep_d2d')` matching
    #   physical `Mesh('fsdp', 'expert', 'ep_d2d')` for single-stage 3D
    #   core-minor SC1 AG/RS with zero Stage-1 -> Stage-2 barrier.
    # - `dense_1stage_3d_fsdp=False`: `('fsdp', 'ep_d2d', 'expert')` for 2-stage
    #   gather (`'expert'` then `('fsdp', 'ep_d2d')`).
    if not self.decouple_ep_d2d:
      return ("fsdp", "expert")
    if self.dense_1stage_3d_fsdp:
      return ("fsdp", "expert", "ep_d2d")
    return ("fsdp", "ep_d2d", "expert")

  @property
  def local_num_experts(self) -> int:
    return self.num_experts // self.ep_size

  @property
  def experts_per_device(self) -> int:
    return self.num_experts // self.num_devices

  @property
  def moe_k_shards(self) -> int:
    """Contiguous axis-0 FSDP K-split factor per expert (`1` if E>=devs, `2` at 512c)."""
    if self.num_experts % self.num_devices == 0:
      return 1
    if (
        self.num_experts % self.ep_size == 0
        and self.num_devices % self.num_experts == 0
    ):
      return self.num_devices // self.num_experts
    return 0

  @property
  def gdn_key_dim(self) -> int:
    return self.gdn_num_k_heads * self.gdn_head_k_dim

  @property
  def gdn_value_dim(self) -> int:
    return self.gdn_num_v_heads * self.gdn_head_v_dim

  @property
  def gdn_qkv_dim(self) -> int:
    return 2 * self.gdn_key_dim + self.gdn_value_dim

  @property
  def gdn_qkvz_dim(self) -> int:
    return self.gdn_qkv_dim + self.gdn_value_dim

  @property
  def attn_q_out_dim(self) -> int:
    return self.num_query_heads * self.head_dim

  @property
  def rotary_dim(self) -> int:
    return int(round(self.head_dim * self.partial_rotary_factor))

  def dense_proj_fsdp_expert(self, unsharded_cols: int) -> bool:
    """Returns True if a dense projection of width `unsharded_cols` uses Stage-1+2 sharding."""
    if self.dense_sharding_mode != "fsdp_expert":
      return False
    if (
        self.narrow_dense_stage2_only
        and unsharded_cols <= self.narrow_dense_max_cols
    ):
      return False
    return True

  def ragged_buffer_size(self, ubatch_tokens_per_device: int) -> int:
    """Computes local packed ragged expert buffer size aligned to 3072 (LCM of 1024, 768, 512, 256)."""
    ep_tokens = ubatch_tokens_per_device * self.ep_size
    expected_local_slots = (
        ep_tokens * self.num_experts_per_tok
    ) // self.ep_size
    raw = int(round(expected_local_slots * self.ragged_buffer_factor))
    align = 3072 if self.use_pallas_kernels else 32
    return ((raw + align - 1) // align) * align


# Backward-compatible alias for clean porting from `qwen38_jax`
Qwen38Config = QwenConfig


def qwen_max_config(**overrides) -> QwenConfig:
  """Returns Qwen MAX (`Qwen3.8-2.4T-A95B`, `92L + AdamW`) 256-chip (`gf=4x8x8_untwisted`) config."""
  return QwenConfig(**overrides)


full_config = qwen_max_config


def qwen_max_proxy_128c_config(**overrides) -> QwenConfig:
  """Returns 20L (`5` superblocks) Qwen MAX proxy on `128c` (`gf=4x4x8_untwisted`, `EP_ICI=8`, `EP=16`)."""
  defaults = dict(
      num_superblocks=5,
      full_num_superblocks=23,
      fsdp_size=16,
      ep_size=16,
      decouple_ep_d2d=True,
      ep_core_minor=True,
  )
  defaults.update(overrides)
  return QwenConfig(**defaults)


def qwen_max_proxy_64c_config(**overrides) -> QwenConfig:
  """Returns 20L (`5` superblocks) Qwen MAX proxy on `64c` (`gf=4x4x4`, `EP_ICI=4`, `EP=8`)."""
  defaults = dict(
      num_superblocks=5,
      full_num_superblocks=23,
      fsdp_size=16,
      ep_size=8,
      decouple_ep_d2d=True,
      ep_core_minor=True,
  )
  defaults.update(overrides)
  return QwenConfig(**defaults)


def qwen_max_512c_config(**overrides) -> QwenConfig:
  """Returns 92L (`23` superblocks) Qwen MAX on `512c` (`gf=8x8x8_untwisted`, `FSDP=64` on `(Y=8, X=8)`, `EP_ICI=8` on `Z=8`, `ep_d2d=2`)."""
  defaults = dict(
      num_superblocks=23,
      full_num_superblocks=23,
      fsdp_size=64,
      ep_size=16,
      decouple_ep_d2d=True,
      ep_core_minor=True,
  )
  defaults.update(overrides)
  return QwenConfig(**defaults)


def qwen3_5_config(**overrides) -> QwenConfig:
  """Returns Qwen 3.5 (`Qwen3.5-397B-A17B`, `60L = 15` superblocks) on `256c` (`gf=4x8x8_untwisted`)."""
  defaults = dict(
      model_variant="qwen3_5",
      num_superblocks=15,
      full_num_superblocks=15,
      emb_dim=4096,
      vocab_size=248320,
      norm_eps=1e-6,
      num_query_heads=32,
      num_kv_heads=2,
      head_dim=256,
      partial_rotary_factor=0.25,
      rope_max_timescale=10_000_000.0,
      gdn_num_k_heads=16,
      gdn_num_v_heads=64,
      gdn_head_k_dim=128,
      gdn_head_v_dim=128,
      gdn_conv_kernel_size=4,
      gdn_chunk_size=64,
      gdn_fwd_head_group_size=4,
      num_experts=512,
      num_experts_per_tok=10,
      moe_mlp_dim=1024,
      shared_mlp_dim=1024,
      fsdp_size=32,
      ep_size=16,
      decouple_ep_d2d=True,
      ep_core_minor=True,
      fsdp_pipeline_mode="post_rs_tie",
      remat_superblock_carry_only=False,
      save_attn_residuals_in_lineage=True,
      save_moe_gate_up_sublayers=(0, 1, 2, 3),
      save_query_proj_hbm=True,
      save_out_proj_hbm=True,
      save_gdn_residuals=False,
      save_gdn_residuals_sublayers=(0, 1, 2),
      gdn_keep_bf16_states=True,
      gdn_recompute_t_inv=True,
      ep_combine_fwd_unpinned=True,
      defer_meta_ag_to_dispatch=True,
      pipeline_bwd_remat_dispatch=True,
      lineage_defer_mb1_combine=True,
      gdn_rms_norm_barrier_stages=False,
      attn_bf16_out_fused_gate=True,
      bwd_sc_balance_mode="sc1_all",
      max_overlap_order_mode="bwd_done_first",
      replicate_shared_gate=True,
      narrow_dense_stage2_only=True,
      bwd_early_mb1_dy_pmax=True,
      remat_bwd_tie_prev_dense_depth=2,
      # v21 tuned tiles for emb_dim=4096, moe_mlp_dim=1024:
      gate_fwd_tiles=(1024, 4096, 2048, 256),
      lc_fwd_tiles=(256, 512, 64),
      lbwd_dgrad_tiles=(512, 256, 1024, 4096),
      lbwd_wgrad_tiles=(1024, 256, 4096),
      gbwd_fused_tiles=(1024, 256, 1024),
      tc_gather_block_size=1792,
      tc_reduce_block_size=1792,
      wi_tile_fwd=(768, 2048, 1024),
      wi_tile_fwd_3d_fp8=(768, 1024, 2048),
      wi_tile_dlhs=(768, 2048, 1024),
      wi_tile_dlhs_fp8=(768, 2048, 2048),
      wi_tile_dlhs_3d_fp8=(1024, 2048, 1024),
      wi_tile_drhs=(768, 1024, 2048),
      wi_tile_drhs_fp8=(768, 2048, 2048),
      wi_tile_drhs_3d_fp8=(768, 1024, 2048),
      wo_tile_fwd=(768, 1024, 2048),
      wo_tile_fwd_3d_fp8=(1024, 1024, 1024),
      wo_tile_dlhs=(512, 1024, 2048),
      wo_tile_dlhs_fp8=(768, 2048, 1024),
      wo_tile_dlhs_3d_fp8=(768, 1024, 1024),
      wo_tile_drhs=(768, 1024, 2048),
      wo_tile_drhs_fp8=(768, 1024, 2048),
      wo_tile_drhs_3d_fp8=(1024, 1024, 1024),
  )
  defaults.update(overrides)
  return QwenConfig(**defaults)


def qwen3_5_proxy_128c_config(**overrides) -> QwenConfig:
  """Returns 20L (`5` superblocks) Qwen 3.5 proxy on `128c` (`gf=4x4x8_untwisted`, `EP_ICI=8`, `EP=16`)."""
  defaults = dict(
      num_superblocks=5, full_num_superblocks=15, fsdp_size=16, ep_size=16
  )
  defaults.update(overrides)
  return qwen3_5_config(**defaults)


def qwen3_5_proxy_64c_config(**overrides) -> QwenConfig:
  """Returns 20L (`5` superblocks) Qwen 3.5 proxy on `64c` (`gf=4x4x4`, `EP_ICI=4`, `EP=8`)."""
  defaults = dict(
      num_superblocks=5, full_num_superblocks=15, fsdp_size=16, ep_size=8
  )
  defaults.update(overrides)
  return qwen3_5_config(**defaults)


def tiny_config(**overrides) -> QwenConfig:
  """Returns tiny CPU config for fast deterministic numerical equivalence & xparity testing."""
  defaults = dict(
      model_variant="custom",
      num_superblocks=2,
      full_num_superblocks=2,
      emb_dim=128,
      vocab_size=256,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=32,
      partial_rotary_factor=0.25,
      gdn_num_k_heads=2,
      gdn_num_v_heads=4,
      gdn_head_k_dim=32,
      gdn_head_v_dim=32,
      gdn_conv_kernel_size=4,
      gdn_chunk_size=16,
      gdn_fwd_head_group_size=1,
      num_experts=16,
      num_experts_per_tok=2,
      moe_mlp_dim=64,
      shared_mlp_dim=64,
      routing_mode="router",
      ragged_buffer_factor=3.5,
      per_device_batch_size=2,
      seq_len=32,
      fsdp_size=2,
      ep_size=4,
      decouple_ep_d2d=False,
      ep_core_minor=True,
      num_vocab_chunks=4,
      num_vocab_tiling=2,
      use_pallas_kernels=False,
      moe_fp8_weights=False,
      moe_fp8_gmm_compute=False,
      moe_fp8_token_dispatch=False,
      moe_fp8_bwd_dispatch=False,
      moe_use_ragged_silu_3d_fp8=False,
      moe_use_forked_3d_gmmv2=False,
      moe_bwd_dy_fp8_e5m2=False,
      moe_local_65key_sort=False,
      moe_use_tc_ragged=False,
      moe_3d_ep_tokens=False,
      ep_sc_pinning="unpinned",
      fsdp_ag_sc_pinning="unpinned",
      fsdp_rs_sc_pinning="unpinned",
      fsdp_pipeline_mode="post_rs_tie",
      moe_rw_prefetch_in_f3=False,
      bwd_sc_balance_mode="sc1_all",
      route_before_shared_expert=False,
      ep_combine_fwd_unpinned=True,
      routed_w_tiled_layout=True,
      ragged_single_sc=False,
      enable_remat=True,
      use_max_async_overlap_transform=False,
      fp8_amax_no_combine=False,
      f2_early_mb0_dispatch=False,
      save_decoder_input_hbm=False,
      offload_decoder_input=False,
      offload_attn_context=False,
      offload_query_proj=False,
      offload_out_proj=False,
      save_attn_context_hbm=False,
      save_query_proj_hbm=False,
      save_out_proj_hbm=False,
      sa_qk_seq_minor=False,
      moe_bwd_tie_wgrad_before_gate=False,
      f2_light_router_ties=False,
      save_moe_routing=False,
      save_attn_residuals_in_lineage=False,
      save_gdn_residuals_sublayers=(),
      gdn_keep_bf16_states=False,
      gdn_recompute_t_inv=False,
      save_moe_gate_up_sublayers=(),
      offload_moe_gate_up_sublayers=(),
      save_moe_disp_ep_sublayers=(),
      offload_moe_disp_ep_sublayers=(),
      max_overlap_order_mode="sum",
      attn_qkv_2d_matmul=False,
      attn_bf16_out_fused_gate=False,
      defer_meta_ag_to_dispatch=False,
      pipeline_bwd_remat_dispatch=False,
      lineage_defer_mb1_combine=False,
      gdn_rms_norm_barrier_stages=False,
      moe_gmm_no_alias_out=False,
      gdn_norm_layout_mode="token_minor",
      attn_gate_bf16_align=False,
      bwd_tie_b1_before_prev_b4=False,
      bwd_early_mb1_dy_pmax=False,
      bwd_prefetch_prev_attn3_moe_weights=False,
      replicate_shared_gate=False,
      narrow_dense_stage2_only=False,
      hoist_replicated_grad_ar=True,
      sc0_bwd_dy_pmax=False,
      remat_bwd_tie_prev_dense_depth=0,
      use_lineage_pipelined_scan=False,
  )
  defaults.update(overrides)
  return QwenConfig(**defaults)
