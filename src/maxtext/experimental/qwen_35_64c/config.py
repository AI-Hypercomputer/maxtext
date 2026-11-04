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

"""Pure-JAX configuration for Qwen3.5-397B-A17B on 64-chip TPU v7x (tpu7x:4x4x4)."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal


@dataclasses.dataclass(frozen=True)
class Qwen35Config:
  """Architecture, kernel, sharding, and Dual-SparseCore batch-split config."""

  # Core architecture (60 layers = 15 superblocks x [GDN, GDN, GDN, FullAttn])
  num_superblocks: int = 15
  emb_dim: int = 4096
  vocab_size: int = 248320
  norm_eps: float = 1e-6

  # Full Attention (1 per superblock)
  num_query_heads: int = 32
  num_kv_heads: int = 2
  head_dim: int = 256
  partial_rotary_factor: float = 0.25
  rope_max_timescale: float = 10_000_000.0

  # Gated DeltaNet (3 per superblock)
  gdn_num_k_heads: int = 16
  gdn_num_v_heads: int = 64
  gdn_head_k_dim: int = 128
  gdn_head_v_dim: int = 128
  gdn_conv_kernel_size: int = 4
  gdn_chunk_size: int = 64
  gdn_fwd_head_group_size: int = 4
  gdn_bwd_natural_layout: str = "all"

  # Mixture-of-Experts (in every sub-layer)
  num_experts: int = 512
  num_experts_per_tok: int = 10
  moe_mlp_dim: int = 1024
  shared_mlp_dim: int = 1024
  # No-drop chunked routed MoE: each chunk holds `capacity_factor` x the balanced
  # local-expert slots (see `moe_chunk_capacity`); overflow slots run in extra chunks.
  capacity_factor: float = 1.25
  # Qwen3.5 Switch Transformer router load-balancing auxiliary loss coefficient
  # (official Qwen3.5-397B-A17B `router_aux_loss_coef = 0.001`; 0.0 disables).
  router_aux_loss_coef: float = 0.001

  # Training batch & parallelism
  # Batch is sharded over 'fsdp' and the sequence over the EP group (context
  # parallelism, CP == EP): 32 x 65,536 tokens = 2 sequences x 8,192 tokens/TC.
  global_batch_size: int = 32
  seq_len: int = 65536
  grad_accum_steps: int = 1
  grad_accum_dtype: Literal["bfloat16", "float32"] = "float32"
  # 128 TensorCores on tpu7x:4x4x4: fsdp=16 on (y, x), EP=CP=8 on (core, z).
  fsdp_size: int = 16
  ep_size: int = 8
  mesh_mode: str = "x1e_yx_core"
  decouple_ep_d2d: bool = (
      True  # 3-axis ('fsdp', 'ep_d2d', 'expert') so dense Stage-2 spans (y,x,core)=64 with chiplet_scale_factor=1!
  )
  num_vocab_chunks: int = 4
  num_vocab_tiling: int = 2

  # Optimizer (AdamW in bfloat16, or stateless SGD for schedule studies)
  optimizer: Literal["adamw", "sgd"] = "adamw"
  learning_rate: float = 1e-4
  lr_warmup_steps: int = 0  # linear 0 -> learning_rate warmup, then constant
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  grad_clip_norm: float = 0.0  # 0 disables global norm clip
  mu_dtype: Literal["bfloat16", "float32"] = "bfloat16"
  embed_mode: Literal["sc_gather", "vocab_all_to_all", "ep_all_to_all_fsdp_gather"] = "ep_all_to_all_fsdp_gather"

  # Precision & quantization (3D FP8 + Ragged SiLU GMMs)
  use_pallas_kernels: bool = True
  moe_fp8_weights: bool = True  # gather8 on SC1 (`float8_e4m3fn`)
  moe_fp8_gmm_compute: bool = True  # native fp8xFp8 MXU in all 6 gmm_v2/tgmm_v2 calls
  moe_fp8_token_dispatch: bool = True  # fwd EP token all-gather in float8_e4m3fn
  moe_fp8_bwd_dispatch: bool = True  # bwd EP combine transpose all-gather in float8_e4m3fn
  moe_fp8_combine_sc0: bool = False  # optional SC0-pinned FP8 combine
  moe_use_ragged_silu_3d_fp8: bool = True  # Lineage-style 3D FP8 dispatch + ragged_silu_mul
  moe_use_forked_3d_gmmv2: bool = True  # Forked gmmv2 VMEM-relayout 3D kernels (zero TC sort/silu)
  moe_act_fp8_scale: float = 0.5  # Static e4m3fn scale for gated SwiGLU activations ([-224, 224])
  moe_bwd_dy_fp8_e5m2: bool = True  # Pre-quantize Combine-Bwd dy to fp8_e5m2 before SC0 all-gather
  moe_local_65key_sort: bool = True  # 65-key local router sort (eliminates [327680, 512] one_hot)
  moe_save_bank_in_superblock: bool = True  # Save chunk-0 [capacity//8, 8, 4096] bf16 bank inside Superblock
  sa_qk_seq_minor: bool = False  # HEAD_DIM_MINOR for Tokamax SplashAttention on TPU v7x
  # Unroll B_mb sequences instead of vmap to avoid B_mb x 4 GiB dkv scratch.
  sa_unroll_batch: bool = False
  moe_bwd_tie_wgrad_before_gate: bool = True  # Anchor gate_bwd_fused after linear_bwd_wgrad so dwo RS overlaps B3+B2+B1
  # Stash overflow chunks' grad_gate across loop A -> loop B (disable on large
  # batch to save ~4.7 GiB HBM).
  moe_bwd_stash_overflow: bool = True
  f2_light_router_ties: bool = True  # Tie F2/F3 on local topk_weights instead of SC0 all-gather call-dones
  save_moe_routing: bool = True  # Save discrete MoE routing + routing weights across superblock remat
  moe_router_dense_bwd: bool = (
      True  # Dense broadcasted_iota VJP + checkpointed top_k (eliminates router scatter + bwd top_k)
  )
  moe_dw_scatter_set: bool = True  # Blind .set() instead of RMW .add() on unique routed d(topk_weights) slots
  save_moe_gate_up_sublayers: tuple[int, ...] = (
      1,
      3,
  )  # Sublayers (0..3) saving T(8,128) bank_tiles in HBM
  offload_moe_gate_up_sublayers: tuple[int, ...] = (
      0,
      2,
  )  # Sublayers (0..3) offloading T(8,128) bank_tiles to S(5)
  save_moe_disp_ep_sublayers: tuple[int, ...] = ()  # Sublayers (0..3) saving FP8 disp.codes_ep in HBM
  offload_moe_disp_ep_sublayers: tuple[int, ...] = ()  # Sublayers (0..3) offloading FP8 disp.codes_ep to S(5)
  gate_fwd_tiles: tuple[int, int, int, int] = (256, 4096, 2048, 256)
  lc_fwd_tiles: tuple[int, int, int] = (256, 512, 64)
  lbwd_dgrad_tiles: tuple[int, int, int, int] = (256, 128, 256, 1024)
  lbwd_wgrad_tiles: tuple[int, int, int] = (768, 256, 2048)
  gbwd_fused_tiles: tuple[int, int, int] = (768, 256, 1024)

  # Tokamax SplashAttention + GQA sequence folding (autotuned for 64K on TPU v7x)
  use_tokamax_splash: bool = True
  sa_fold_gqa_into_seq: bool = True
  use_splash_scheduler: bool = True
  sa_fuse_reciprocal: bool = True
  sa_use_base2_exp: bool = True
  sa_dq_reduction_steps: int | None = None
  sa_interpret: bool = False
  sa_block_q: int = 2048
  sa_block_kv: int = 4096
  sa_block_kv_compute: int = 512
  sa_block_q_dkv: int = 1024
  sa_block_kv_dkv: int = 4096
  sa_block_kv_dkv_compute: int = 256

  # 3D zero-relayout TensorCore ragged sort/unsort (offloads SC0 100% to EP collectives)
  moe_use_tc_ragged: bool = True
  moe_3d_ep_tokens: bool = True
  tc_gather_block_size: int = 3072  # 48 MiB FP8 scratch across 4 buffers under 64 MiB TPU v7x scoped VMEM limit
  tc_reduce_block_size: int = 1792  # Widened gather-reduce tile fitting inside 64 MiB TPU v7x kernel VMEM limit

  # Schedule & Dual-SparseCore (SC0 EP vs SC1 FSDP) knobs
  schedule_mode: Literal["unsplit", "batchsplit"] = "batchsplit"
  ep_sc_pinning: Literal["sc0", "unpinned"] = "sc0"
  fsdp_ag_sc_pinning: Literal["sc1", "sc1_fwd_only", "unpinned"] = "sc1"
  fsdp_rs_sc_pinning: Literal["sc1", "unpinned"] = "sc1"
  fsdp_pipeline_mode: Literal["post_rs_tie", "sublayer_prefetch", "full_prefetch_defer_rs"] = "full_prefetch_defer_rs"
  bwd_tie_mode: Literal["all4", "moe_only", "moe_xmb0_only", "symmetric_4phase", "none"] = "symmetric_4phase"
  dense_sharding_mode: Literal["fsdp_only", "fsdp_expert", "auto_sc_min"] = "auto_sc_min"
  replicate_sub_sc_dense: bool = True
  moe_ag_start_tie: Literal["f2_x_mb1", "f2_base_mb0", "none"] = "f2_base_mb0"
  moe_rw_prefetch_in_f3: bool = False
  bwd_sc_balance_mode: Literal["sc1_all", "wi_rs_sc0", "wi_dense_rs_sc0", "dwo_sc0"] = "wi_rs_sc0"
  route_before_shared_expert: bool = True
  ep_combine_fwd_unpinned: bool = False
  ep_routing_weights_bwd_sc: Literal["none", "sc0", "sc1"] = "sc1"
  routed_w_tiled_layout: bool = True
  tie_f4_before_wo: bool = False
  fuse_sublayer_sc1_collectives: bool = True
  strict_f2_before_f3: bool = True
  defer_mb1_combine_after_f1: bool = True
  precompute_rope_tables: bool = True
  use_max_async_overlap_transform: bool = True
  max_overlap_bundle_quant: bool = True
  max_overlap_bundle_scalar_ar: bool = True
  max_overlap_queues: Literal["all", "sc_only", "sc0_only"] = "sc_only"
  max_overlap_order_mode: Literal["sum", "done_first", "bwd_done_first"] = "bwd_done_first"
  fp8_amax_no_combine: bool = True
  f2_early_mb0_dispatch: bool = True
  attn_qkv_2d_matmul: bool = False
  moe_gmm_no_alias_out: bool = True
  gdn_norm_layout_mode: Literal["token_minor", "bf16_z_token_minor", "natural"] = "token_minor"
  attn_gate_bf16_align: bool = False
  bwd_tie_b1_before_prev_b4: bool = False
  bwd_early_mb1_dy_pmax: bool = True
  bwd_prefetch_prev_attn3_moe_weights: bool = False
  sc_min_collective_bytes: int = 2_000_000
  pipeline_lm_head_chunks: bool = True

  # Host activation offload policy (S(5) pinned_host) & device save policy
  enable_remat: bool = True
  save_key_value_proj: bool = True  # key_proj=device value_proj=device
  save_gdn_residuals: bool = False  # gdn=remat
  save_gdn_cp_state: bool = True
  gdn_cp_m_scan: bool = True
  gdn_cp_dh_scan: bool = True
  gdn_cp_matmul_high: bool = True
  save_decoder_input_hbm: bool = True
  offload_decoder_input: bool = False
  offload_attn_context: bool = True
  offload_query_proj: bool = True
  offload_out_proj: bool = True
  offload_gdn_qkv: bool = False
  save_attn_context_hbm: bool = True
  save_query_proj_hbm: bool = True
  save_out_proj_hbm: bool = True
  hbm_save_offload_targets: bool = False
  vocab_tiling_ag_once: bool = False

  def __post_init__(self):
    if not self.capacity_factor > 0:
      raise ValueError(f"capacity_factor must be > 0, got {self.capacity_factor}.")
    if self.router_aux_loss_coef < 0.0:
      raise ValueError("router_aux_loss_coef must be >= 0.0, got" f" {self.router_aux_loss_coef}.")
    # Routed MoE only implements the no-drop chunked forked 3D FP8 path.
    forked_moe_flags = (
        "moe_fp8_weights",
        "moe_fp8_gmm_compute",
        "moe_fp8_token_dispatch",
        "moe_use_ragged_silu_3d_fp8",
        "moe_use_forked_3d_gmmv2",
        "moe_use_tc_ragged",
        "moe_3d_ep_tokens",
    )
    disabled = [name for name in forked_moe_flags if not getattr(self, name)]
    if disabled:
      raise ValueError(
          "Routed MoE only supports the no-drop forked 3D FP8 path; these" f" flags must be True: {disabled}."
      )
    if self.emb_dim % 128:
      raise ValueError("Forked 3D FP8 routed MoE requires emb_dim % 128 == 0, got" f" {self.emb_dim}.")
    if self.sa_dq_reduction_steps not in (None, 1, 3):
      raise ValueError("sa_dq_reduction_steps must be None, 1, or 3, got" f" {self.sa_dq_reduction_steps}.")
    if self.sa_fold_gqa_into_seq and (self.num_query_heads % self.num_kv_heads != 0):
      raise ValueError(
          "GQA sequence folding requires num_query_heads"
          f" ({self.num_query_heads}) divisible by num_kv_heads"
          f" ({self.num_kv_heads})."
      )

  @property
  def num_layers(self) -> int:
    return 4 * self.num_superblocks

  @property
  def num_devices(self) -> int:
    return self.fsdp_size * self.ep_size

  @property
  def ep_ici_size(self) -> int:
    if self.decouple_ep_d2d:
      if self.ep_size % 2 != 0:
        raise ValueError(f"decouple_ep_d2d=True requires even ep_size, got {self.ep_size}")
      return self.ep_size // 2
    return self.ep_size

  @property
  def ep_axis_names(self) -> str | tuple[str, str]:
    return ("ep_d2d", "expert") if self.decouple_ep_d2d else "expert"

  @property
  def ep_axis_tuple(self) -> tuple[str, ...]:
    return ("ep_d2d", "expert") if self.decouple_ep_d2d else ("expert",)

  @property
  def dense_stage2_axis_names(self) -> str | tuple[str, str]:
    return ("fsdp", "ep_d2d") if self.decouple_ep_d2d else "fsdp"

  @property
  def dense_stage2_size(self) -> int:
    return self.fsdp_size * 2 if self.decouple_ep_d2d else self.fsdp_size

  @property
  def all_sharded_axes(self) -> tuple[str, ...]:
    return ("fsdp", "ep_d2d", "expert") if self.decouple_ep_d2d else ("fsdp", "expert")

  @property
  def cp_axis_names(self) -> str | tuple[str, str]:
    """Context-parallel axes in mesh order (as `ppermute` requires): CP rank = core * 4 + z."""
    return ("ep_d2d", "expert") if self.decouple_ep_d2d else "expert"

  @property
  def cp_size(self) -> int:
    return self.ep_size

  @property
  def local_batch_size(self) -> int:
    if self.global_batch_size % self.fsdp_size != 0:
      raise ValueError(f"global_batch_size={self.global_batch_size} must be divisible by" f" fsdp_size={self.fsdp_size}")
    return self.global_batch_size // self.fsdp_size

  @property
  def local_seq_len(self) -> int:
    # Dual-chunk load-balanced attention splits the sequence into 2 * cp_size chunks.
    if self.seq_len % (2 * self.cp_size) != 0:
      raise ValueError(f"seq_len={self.seq_len} must be divisible by 2 *" f" cp_size={2 * self.cp_size}")
    return self.seq_len // self.cp_size

  @property
  def local_num_experts(self) -> int:
    return self.num_experts // self.ep_size

  @property
  def experts_per_device(self) -> int:
    return self.num_experts // self.num_devices

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

  def dense_param_elems(self, name: str) -> int:
    """Returns single-layer global element count for a named dense weight."""
    d = self.emb_dim
    table = {
        "router_gate": d * self.num_experts,
        "shared_wi": d * self.shared_mlp_dim,
        "shared_wo": self.shared_mlp_dim * d,
        "shared_gate": d,
        "in_proj_qkvz": d * self.gdn_qkvz_dim,
        "in_proj_ba": d * 2 * self.gdn_num_v_heads,
        "gdn_out_proj": self.gdn_value_dim * d,
        "q_proj": d * self.num_query_heads * 2 * self.head_dim,
        "kv_proj": d * self.num_kv_heads * self.head_dim,
        "attn_out_proj": self.attn_q_out_dim * d,
    }
    if name not in table:
      raise KeyError(f"Unknown dense parameter name: {name!r}")
    return table[name]

  def dense_param_sharding_kind(self, name: str) -> Literal["replicated", "fsdp_only", "fsdp_expert"]:
    """Selects per-tensor sharding tier above the SC1 offload threshold."""
    global_bytes = 2 * self.dense_param_elems(name)
    if self.replicate_sub_sc_dense and global_bytes < self.sc_min_collective_bytes:
      return "replicated"
    if self.dense_sharding_mode == "fsdp_only":
      return "fsdp_only"
    if self.dense_sharding_mode == "auto_sc_min":
      # Stage-1 ('expert') gather output is `global_bytes // dense_stage2_size`;
      # below `sc_min_collective_bytes`, Stage-1 drops onto synchronous TC ICI.
      stage1_bytes = global_bytes // self.dense_stage2_size
      return "fsdp_only" if stage1_bytes < self.sc_min_collective_bytes else "fsdp_expert"
    return "fsdp_expert"

  def moe_chunk_capacity(self, ubatch_tokens_per_device: int) -> int:
    """Sorted local-expert slots per no-drop routed MoE chunk.

    `capacity_factor` x the balanced local slots (`ubatch_tokens_per_device *
    top_k`), rounded up to 3072 (LCM of the 1024/768/512/256
    gmm_v2/tgmm_v2/ragged
    tiles; 32 for the pure-JAX fallback), but never more than all
    `ep_size * balanced` slots that can be routed to one EP rank.
    """
    balanced_local_slots = ubatch_tokens_per_device * self.num_experts_per_tok
    align = 3072 if self.use_pallas_kernels else 32
    aligned = math.ceil(self.capacity_factor * balanced_local_slots / align) * align
    return min(balanced_local_slots * self.ep_size, aligned)


def full_config(**overrides) -> Qwen35Config:
  """Returns full Qwen3.5-397B-A17B TPU v7x 64-chip (128-TensorCore) config."""
  return Qwen35Config(**overrides)


def tiny_config(**overrides) -> Qwen35Config:
  """Returns tiny CPU config for fast numerical equivalence testing."""
  defaults = dict(
      num_superblocks=2,
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
      capacity_factor=0.5,  # < 1 forces multi-chunk routed MoE on imbalanced tiny batches
      global_batch_size=4,
      seq_len=32,
      fsdp_size=2,
      ep_size=4,
      decouple_ep_d2d=False,
      num_vocab_chunks=4,
      num_vocab_tiling=2,
      use_pallas_kernels=False,
      moe_fp8_bwd_dispatch=False,
      moe_bwd_dy_fp8_e5m2=False,
      ep_sc_pinning="unpinned",
      fsdp_ag_sc_pinning="unpinned",
      fsdp_rs_sc_pinning="unpinned",
      fsdp_pipeline_mode="post_rs_tie",
      dense_sharding_mode="fsdp_expert",
      replicate_sub_sc_dense=False,
      moe_rw_prefetch_in_f3=False,
      bwd_sc_balance_mode="sc1_all",
      route_before_shared_expert=False,
      pipeline_lm_head_chunks=False,
      ep_combine_fwd_unpinned=True,
      ep_routing_weights_bwd_sc="none",
      routed_w_tiled_layout=True,
      enable_remat=True,
      use_max_async_overlap_transform=False,
      fp8_amax_no_combine=False,
      f2_early_mb0_dispatch=False,
      save_decoder_input_hbm=False,
      offload_decoder_input=False,
      save_gdn_cp_state=False,
      gdn_cp_m_scan=False,
      gdn_cp_dh_scan=False,
      gdn_cp_matmul_high=False,
      offload_attn_context=False,
      offload_query_proj=False,
      offload_out_proj=False,
      save_attn_context_hbm=False,
      save_query_proj_hbm=False,
      save_out_proj_hbm=False,
      sa_qk_seq_minor=False,
      use_tokamax_splash=False,
      sa_fold_gqa_into_seq=False,
      moe_bwd_tie_wgrad_before_gate=False,
      f2_light_router_ties=False,
      save_moe_routing=False,
      moe_router_dense_bwd=False,
      moe_dw_scatter_set=False,
      save_moe_gate_up_sublayers=(),
      offload_moe_gate_up_sublayers=(),
      save_moe_disp_ep_sublayers=(),
      offload_moe_disp_ep_sublayers=(),
      max_overlap_order_mode="sum",
      attn_qkv_2d_matmul=False,
      moe_gmm_no_alias_out=False,
      gdn_norm_layout_mode="token_minor",
      attn_gate_bf16_align=False,
      bwd_tie_b1_before_prev_b4=False,
      bwd_early_mb1_dy_pmax=False,
      bwd_prefetch_prev_attn3_moe_weights=False,
  )
  defaults.update(overrides)
  return Qwen35Config(**defaults)
