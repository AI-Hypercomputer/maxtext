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

"""Configuration, validation, parameter counts, and FLOPs for Flint 40B-A2.5B."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax.numpy as jnp

ALL_AXES: tuple[str, str, str] = ("fsdp", "cp", "ep")
_VALID_DTYPES: frozenset[str] = frozenset({"bfloat16", "float32"})
_VALID_REMAT_POLICIES: frozenset[str] = frozenset({"layer", "selective", "global_selective", "none"})
_VALID_MOE_IMPLS: frozenset[str] = frozenset({"xla_ragged_dot", "pallas_gmm_v2"})
_VALID_SC_COLLECTIVE_MODES: frozenset[str] = frozenset({"off", "dual_sc"})
_VALID_FWD_STRIP_SCHEDULES: frozenset[str] = frozenset({"off", "hoisted", "per_strip", "auto"})
_VALID_LR_SCHEDULES: frozenset[str] = frozenset({"warmup_cosine", "constant"})
_VALID_INIT_MODES: frozenset[str] = frozenset({"fan_in_trunc_normal", "normal"})

SELECTIVE_SAVED_CHECKPOINT_NAMES: tuple[str, ...] = (
    "x_norm1",
    "post_attn",
    "x_norm2",
    "attn_qk_pre_norm",
    "attn_q_res",
    "attn_global_kv",
    "attn_local_kv",
    "attn_global_out_lse",
    "attn_local_out_lse",
    "mlp_gate_up",
    "moe_router_pack",
    "moe_router_maps",
    "moe_gate_up",
)

GLOBAL_SELECTIVE_SAVED_CHECKPOINT_NAMES: tuple[str, ...] = (
    "x_norm1",
    "attn_global_kv",
    "attn_global_out_lse",
    "moe_router_maps",
)

LOCAL_SELECTIVE_SAVED_CHECKPOINT_NAMES: tuple[str, ...] = ("moe_router_maps",)

EXTRA_SELECTIVE_SAVED_CHECKPOINT_NAMES: tuple[str, ...] = (
    "moe_sorted_x",
    "moe_down_out",
)

OPTIMIZED_V7X_LIBTPU_ARGS: str = " ".join(
    [
        "--xla_tpu_use_tc_device_shape_on_sc=true",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_enable_offloading_gather_to_sparsecore=false",
        "--xla_tpu_enable_offloading_scatter_to_sparsecore=false",
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
        "--xla_tpu_use_single_sparse_core_for_all_gather_offload=false",
        "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
        "--xla_tpu_enable_concurrent_sparse_core_offloading=true",
        "--xla_tpu_enable_sparse_core_reduce_scatter_padding=true",
        "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
        "--xla_tpu_aggressive_opt_barrier_removal=false",
        "--xla_enable_async_collective_permute=ENABLED",
        "--xla_tpu_enable_all_experimental_scheduler_features=true",
        "--xla_tpu_enable_scheduler_memory_pressure_tracking=ENABLED",
        "--xla_lhs_prioritize_async_depth_over_stall=ENABLED",
        "--xla_tpu_scheduler_percent_shared_memory_limit=98",
        "--xla_latency_hiding_scheduler_rerun=2",
        "--xla_should_allow_loop_variant_parameter_in_chain=ENABLED",
        "--xla_should_add_loop_invariant_op_in_chain=ENABLED",
    ]
)

# Recipe run by the Python entrypoints when no config overrides are given; it
# matches `recipes/flint_64c_256k.sh`.
DEFAULT_RECIPE_NAME: str = "flint-40b-64c-256k"


def optimized_v7x_libtpu_args(
    *,
    global_batch_size: int,
    grad_accum_steps: int,
    fsdp_size: int,
    cp_size: int,
) -> str:
  """Returns TPU v7x compiler flags for a 64-chip recipe shape.

  Args:
    global_batch_size: Total batch size across all chips.
    grad_accum_steps: Number of micro-step gradient accumulation iterations.
    fsdp_size: Size of the FSDP sharding mesh axis.
    cp_size: Size of the context parallelism mesh axis.

  Returns:
    Space-separated compiler flags for `LIBTPU_INIT_ARGS`.
  """
  micro_b = global_batch_size // max(grad_accum_steps, 1)
  if cp_size == 1 and micro_b > fsdp_size:
    return f"{OPTIMIZED_V7X_LIBTPU_ARGS} " "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
  return OPTIMIZED_V7X_LIBTPU_ARGS


def _str_to_jnp_dtype(name: str) -> jnp.dtype:
  if name == "bfloat16":
    return jnp.dtype(jnp.bfloat16)
  if name == "float32":
    return jnp.dtype(jnp.float32)
  raise ValueError(f"Unsupported dtype {name!r}; expected {_VALID_DTYPES}.")


@dataclasses.dataclass(frozen=True)
class FlintModelConfig:
  """Architecture hyperparameters for RecursiveSI/flint."""

  vocab_size: int = 150144
  hidden_size: int = 2048
  intermediate_size: int = 6144
  moe_intermediate_size: int = 512
  shared_expert_intermediate_size: int = 1024
  num_hidden_layers: int = 49
  num_dense_prefix_layers: int = 2
  moe_cycle: int = 4
  num_attention_heads: int = 32
  num_key_value_heads: int = 4
  head_dim: int = 128
  num_experts: int = 256
  num_experts_per_tok: int = 6
  routed_scaling_factor: float = 1.0
  rms_norm_eps: float = 1e-6
  rope_theta: float = 10000.0
  max_position_embeddings: int = 262144
  sliding_window: int = 513
  bos_token_id: int = 150000
  pad_token_id: int = 150001
  scan_layers: bool = True
  init_mode: str = "fan_in_trunc_normal"
  initializer_range: float = 0.02


@dataclasses.dataclass(frozen=True)
class FlintShardingConfig:
  """3D mesh axis sizes ('fsdp', 'cp', 'ep') for Flint."""

  fsdp_size: int = 128
  cp_size: int = 1
  ep_size: int = 1

  ALL_AXES = ALL_AXES

  @property
  def num_devices(self) -> int:
    """Total number of logical TensorCore devices across the 3D mesh."""
    return self.fsdp_size * self.cp_size * self.ep_size

  @property
  def dp_ways(self) -> int:
    """Batch-parallel ways across ('fsdp', 'ep')."""
    return self.fsdp_size * self.ep_size


@dataclasses.dataclass(frozen=True)
class FlintKernelConfig:
  """Kernel, SplashAttention, MoE, Dual-SC, and chunked CE configuration."""

  use_splash: bool
  splash_block_q: int = 512
  splash_block_kv: int = 512
  splash_block_kv_compute: int = 512
  splash_block_q_dkv: int = 512
  splash_block_kv_dkv: int = 512
  splash_block_kv_dkv_compute: int = 512
  splash_dq_reduction_steps: int | None = 3
  splash_use_base2_exp: bool = False
  splash_fuse_reciprocal: bool = True
  capacity_factor: float | None = None
  num_vocab_chunks: int = 17
  vocab_token_chunk_size: int = 4096
  moe_impl: str = "xla_ragged_dot"
  gmm_tile_m: int | None = None
  tgmm_tile_m: int | None = None
  sc_collective_mode: str = "off"
  routed_w_tiled: bool = False
  sc_min_collective_bytes: int = 2_000_000
  moe_bijective_permute: bool = False
  fuse_layer_projections: bool = False
  splash_global_block_q: int | None = None
  splash_global_block_kv: int | None = None
  splash_global_block_kv_compute: int | None = None
  splash_global_block_q_dkv: int | None = None
  splash_global_block_kv_dkv: int | None = None
  splash_global_block_kv_dkv_compute: int | None = None
  splash_qk_diag_skip: bool = False
  splash_sv_diag_skip: bool = False
  replicate_router_gate: bool = False
  shard_routed_on_expert_axis: bool = False
  moe_slot_major_pack: bool = False
  save_extra_selective_residuals: bool = False
  prefetch_routed_weights: bool = False
  fp8_routed_all_gather: bool = False
  transpose_routed_down_ag: bool = False
  transpose_dense_rhs_ag: bool = False
  moe_zero_scatter_router: bool = False
  prefetch_routed_down: bool = False
  precompute_rope_tables: bool = False
  use_fp8: bool = False
  splash_local_dq_reduction_steps: int | None = None
  splash_num_stacked_q_heads: int = 1
  splash_num_stacked_q_heads_dkv: int = 1
  splash_global_num_stacked_q_heads: int | None = None
  splash_global_num_stacked_q_heads_dkv: int | None = None
  splash_use_scheduler: bool = False
  splash_use_scheduler_dkv: bool | None = None
  splash_global_use_scheduler: bool | None = None
  splash_global_use_scheduler_dkv: bool | None = None
  splash_dkv_reuse_scaled_q: bool = False
  splash_global_dkv_reuse_scaled_q: bool | None = None
  splash_use_local_halo_cp: bool = False
  splash_ignore_segment_ids: bool = False
  splash_fwd_strip_schedule: str = "off"
  splash_fwd_strip_hoist_q_scale: bool | None = None
  splash_compact_lse_residual: bool = False
  splash_dkv_head_stitch: bool = False
  splash_global_dkv_head_stitch: bool | None = None
  use_sc_moe_gather_reduce: bool = False
  use_sc_counting_sort: bool = False
  use_pallas_router_topk: bool = False
  compact_router_maps: bool = False
  skip_router_aux_stats_when_zero_coef: bool = False
  use_fused_vocab_vjp_scan: bool = False
  use_lb_cp_layout: bool = False
  early_global_kv_proj: bool = False
  use_pallas_lb_kv_reorder: bool = False
  use_fused_qk_prologue: bool = False


@dataclasses.dataclass(frozen=True)
class FlintTrainConfig:
  """Training, precision, rematerialization, and AdamW configuration."""

  global_batch_size: int = 128
  seq_len: int = 8192
  grad_accum_steps: int = 1
  compute_dtype: str = "bfloat16"
  param_dtype: str = "float32"
  remat_policy: str = "layer"
  learning_rate: float = 3e-5
  min_lr_ratio: float = 0.1
  warmup_steps: int = 6
  total_steps: int = 60
  lr_schedule: str = "warmup_cosine"
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  weight_decay: float = 0.1
  wd_exclude_norms_and_embed: bool = False
  grad_clip_norm: float = 1.0
  router_aux_loss_coef: float = 0.0

  @property
  def compute_jnp_dtype(self) -> jnp.dtype:
    """JAX activation and matmul dtype."""
    return _str_to_jnp_dtype(self.compute_dtype)

  @property
  def param_jnp_dtype(self) -> jnp.dtype:
    """JAX master parameter storage dtype."""
    return _str_to_jnp_dtype(self.param_dtype)


def is_moe_layer(layer_idx: int, model: FlintModelConfig) -> bool:
  """Returns True when `layer_idx` uses the sparse MoE block."""
  if layer_idx < 0 or layer_idx >= model.num_hidden_layers:
    raise ValueError(f"layer_idx={layer_idx} out of range [0, {model.num_hidden_layers}).")
  return layer_idx >= model.num_dense_prefix_layers


def is_global_layer(layer_idx: int, model: FlintModelConfig) -> bool:
  """Returns True when `layer_idx` uses global causal NoPE attention."""
  if layer_idx < 0 or layer_idx >= model.num_hidden_layers:
    raise ValueError(f"layer_idx={layer_idx} out of range [0, {model.num_hidden_layers}).")
  if layer_idx < model.num_dense_prefix_layers:
    return True
  if layer_idx == model.num_hidden_layers - 1:
    return True
  return (layer_idx - model.num_dense_prefix_layers + 1) % model.moe_cycle == 0


def flint_scan_layout(model: FlintModelConfig) -> tuple[int, int, int]:
  """Computes `(prefix_layers, num_superblocks, tail_start)` for scanned MoE.

  Args:
    model: Flint architecture configuration.

  Returns:
    Tuple `(prefix, num_superblocks, tail_start)` where `prefix` dense layers
    precede `num_superblocks` 4-layer MoE superblocks and 3 tail MoE layers.

  Raises:
    ValueError: If layer counts or `moe_cycle` do not fit `prefix + 4*N + 3`.
  """
  prefix = model.num_dense_prefix_layers
  rem = model.num_hidden_layers - prefix
  if model.moe_cycle != 4:
    raise ValueError(f"scan_layers requires moe_cycle == 4, got {model.moe_cycle}.")
  if rem < 3 or (rem - 3) % 4 != 0:
    raise ValueError(
        "scan_layers requires (num_hidden_layers - num_dense_prefix_layers - 3)"
        " to be non-negative and divisible by 4; got num_hidden_layers="
        f"{model.num_hidden_layers}, prefix={prefix} (rem={rem})."
    )
  num_superblocks = (rem - 3) // 4
  tail_start = model.num_hidden_layers - 3
  return prefix, num_superblocks, tail_start


@dataclasses.dataclass(frozen=True)
class FlintConfig:
  """Top-level validated configuration for Flint model, sharding, and train."""

  model: FlintModelConfig
  sharding: FlintShardingConfig
  kernels: FlintKernelConfig
  train: FlintTrainConfig

  def __post_init__(self) -> None:
    self.validate()

  def validate(self) -> None:
    """Validates model, sharding, kernel, and training invariants."""
    self._validate_model()
    self._validate_sharding()
    self._validate_kernels()
    self._validate_train()

  def _validate_model(self) -> None:
    """Validates `FlintModelConfig` dimensions and scanned layer schedule."""
    m = self.model
    positive_ints = (
        ("vocab_size", m.vocab_size),
        ("hidden_size", m.hidden_size),
        ("intermediate_size", m.intermediate_size),
        ("moe_intermediate_size", m.moe_intermediate_size),
        ("shared_expert_intermediate_size", m.shared_expert_intermediate_size),
        ("num_hidden_layers", m.num_hidden_layers),
        ("moe_cycle", m.moe_cycle),
        ("num_attention_heads", m.num_attention_heads),
        ("num_key_value_heads", m.num_key_value_heads),
        ("head_dim", m.head_dim),
        ("num_experts", m.num_experts),
        ("max_position_embeddings", m.max_position_embeddings),
        ("sliding_window", m.sliding_window),
    )
    for field_name, field_val in positive_ints:
      if field_val <= 0:
        raise ValueError(f"model.{field_name} must be > 0, got {field_val}.")
    if not 0 <= m.num_dense_prefix_layers <= m.num_hidden_layers:
      raise ValueError(f"num_dense_prefix_layers={m.num_dense_prefix_layers} must be in" f" [0, {m.num_hidden_layers}].")
    if m.num_attention_heads % m.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads={m.num_attention_heads} must be divisible by"
          f" num_key_value_heads={m.num_key_value_heads}."
      )
    if m.head_dim % 2 != 0:
      raise ValueError(f"head_dim={m.head_dim} must be even for RoPE.")
    if not 1 <= m.num_experts_per_tok <= m.num_experts:
      raise ValueError(f"num_experts_per_tok={m.num_experts_per_tok} must be in" f" [1, {m.num_experts}].")
    if m.rms_norm_eps <= 0.0 or m.rope_theta <= 0.0 or m.initializer_range <= 0:
      raise ValueError("rms_norm_eps, rope_theta, and initializer_range must be > 0.")
    if m.init_mode not in _VALID_INIT_MODES:
      raise ValueError(f"init_mode={m.init_mode!r} must be one of {_VALID_INIT_MODES}.")
    if m.scan_layers:
      prefix, num_sb, tail_start = flint_scan_layout(m)
      for sb_idx in range(num_sb):
        base = prefix + 4 * sb_idx
        pattern = tuple(is_global_layer(base + j, m) for j in range(4))
        if pattern != (False, False, False, True):
          raise ValueError(f"Superblock {sb_idx} has global pattern {pattern}, expected" " (False, False, False, True).")
      tail_pattern = tuple(is_global_layer(tail_start + j, m) for j in range(3))
      if tail_pattern != (False, False, True):
        raise ValueError(f"Tail layers have global pattern {tail_pattern}, expected" " (False, False, True).")

  def _validate_sharding(self) -> None:
    """Validates 3D `('fsdp', 'cp', 'ep')` mesh and batch divisibility."""
    m = self.model
    s = self.sharding
    t = self.train
    if s.fsdp_size < 1 or s.cp_size < 1 or s.ep_size < 1:
      raise ValueError(f"Mesh sizes must be >= 1, got ({s.fsdp_size}, {s.cp_size}," f" {s.ep_size}).")
    if m.hidden_size % s.num_devices != 0:
      raise ValueError(f"hidden_size={m.hidden_size} must be divisible by" f" num_devices={s.num_devices}.")
    if m.num_experts % s.ep_size != 0:
      raise ValueError(f"num_experts={m.num_experts} must be divisible by" f" ep_size={s.ep_size}.")
    if m.hidden_size % (s.fsdp_size * s.cp_size) != 0:
      raise ValueError(
          f"hidden_size={m.hidden_size} must be divisible by" f" fsdp_size * cp_size = {s.fsdp_size * s.cp_size}."
      )
    if t.grad_accum_steps < 1:
      raise ValueError(f"grad_accum_steps={t.grad_accum_steps} must be >= 1.")
    denom = s.dp_ways * t.grad_accum_steps
    if t.global_batch_size < 1 or t.global_batch_size % denom != 0:
      raise ValueError(
          f"global_batch_size={t.global_batch_size} must be >= 1 and divisible"
          f" by dp_ways * grad_accum_steps = {denom}."
      )
    if s.cp_size > 1:
      if t.seq_len % (2 * s.cp_size) != 0:
        raise ValueError(f"seq_len={t.seq_len} must be divisible by 2 * cp_size =" f" {2 * s.cp_size} when cp_size > 1.")
    elif t.seq_len < 1:
      raise ValueError(f"seq_len={t.seq_len} must be >= 1.")

  def _validate_kernels(self) -> None:
    """Validates Splash, Pallas GMM, Dual-SC, fusion, and vocab CE knobs."""
    m = self.model
    s = self.sharding
    k = self.kernels
    t = self.train
    if not isinstance(k.use_splash, bool):
      raise ValueError(f"use_splash must be a bool, got {k.use_splash!r}.")
    if k.num_vocab_chunks < 1 or m.vocab_size % k.num_vocab_chunks != 0:
      raise ValueError(f"vocab_size={m.vocab_size} must be divisible by" f" num_vocab_chunks={k.num_vocab_chunks}.")
    if k.vocab_token_chunk_size < 1:
      raise ValueError(f"vocab_token_chunk_size={k.vocab_token_chunk_size} must be >= 1.")
    if k.capacity_factor is not None and k.capacity_factor <= 0.0:
      raise ValueError(f"capacity_factor must be > 0 when set, got {k.capacity_factor}.")
    if k.moe_impl not in _VALID_MOE_IMPLS:
      raise ValueError(f"moe_impl={k.moe_impl!r} must be one of {_VALID_MOE_IMPLS}.")
    if k.moe_impl == "pallas_gmm_v2":
      if m.hidden_size % 128 != 0 or m.moe_intermediate_size % 128 != 0:
        raise ValueError(
            "moe_impl='pallas_gmm_v2' requires hidden_size and"
            " moe_intermediate_size divisible by 128, got"
            f" hidden_size={m.hidden_size},"
            f" moe_intermediate_size={m.moe_intermediate_size}."
        )
    if k.sc_collective_mode not in _VALID_SC_COLLECTIVE_MODES:
      raise ValueError(f"sc_collective_mode={k.sc_collective_mode!r} must be one of" f" {_VALID_SC_COLLECTIVE_MODES}.")
    bool_fields = (
        ("routed_w_tiled", k.routed_w_tiled),
        ("moe_bijective_permute", k.moe_bijective_permute),
        ("fuse_layer_projections", k.fuse_layer_projections),
        ("splash_qk_diag_skip", k.splash_qk_diag_skip),
        ("splash_sv_diag_skip", k.splash_sv_diag_skip),
        ("replicate_router_gate", k.replicate_router_gate),
        ("shard_routed_on_expert_axis", k.shard_routed_on_expert_axis),
        ("moe_slot_major_pack", k.moe_slot_major_pack),
        (
            "save_extra_selective_residuals",
            k.save_extra_selective_residuals,
        ),
        ("prefetch_routed_weights", k.prefetch_routed_weights),
        ("fp8_routed_all_gather", k.fp8_routed_all_gather),
        ("transpose_routed_down_ag", k.transpose_routed_down_ag),
        ("transpose_dense_rhs_ag", k.transpose_dense_rhs_ag),
        ("moe_zero_scatter_router", k.moe_zero_scatter_router),
        ("prefetch_routed_down", k.prefetch_routed_down),
        ("precompute_rope_tables", k.precompute_rope_tables),
        ("use_fp8", k.use_fp8),
        ("splash_use_scheduler", k.splash_use_scheduler),
        ("splash_dkv_reuse_scaled_q", k.splash_dkv_reuse_scaled_q),
        ("splash_use_local_halo_cp", k.splash_use_local_halo_cp),
        ("splash_ignore_segment_ids", k.splash_ignore_segment_ids),
        ("splash_compact_lse_residual", k.splash_compact_lse_residual),
        ("splash_dkv_head_stitch", k.splash_dkv_head_stitch),
        ("use_sc_moe_gather_reduce", k.use_sc_moe_gather_reduce),
        ("use_sc_counting_sort", k.use_sc_counting_sort),
        ("use_pallas_router_topk", k.use_pallas_router_topk),
        ("compact_router_maps", k.compact_router_maps),
        (
            "skip_router_aux_stats_when_zero_coef",
            k.skip_router_aux_stats_when_zero_coef,
        ),
        ("use_fused_vocab_vjp_scan", k.use_fused_vocab_vjp_scan),
        ("use_lb_cp_layout", k.use_lb_cp_layout),
        ("early_global_kv_proj", k.early_global_kv_proj),
        ("use_pallas_lb_kv_reorder", k.use_pallas_lb_kv_reorder),
        ("use_fused_qk_prologue", k.use_fused_qk_prologue),
    )
    for b_name, b_val in bool_fields:
      if not isinstance(b_val, bool):
        raise ValueError(f"{b_name} must be a bool, got {b_val!r}.")
    opt_bool_fields = (
        ("splash_use_scheduler_dkv", k.splash_use_scheduler_dkv),
        ("splash_global_use_scheduler", k.splash_global_use_scheduler),
        ("splash_global_use_scheduler_dkv", k.splash_global_use_scheduler_dkv),
        (
            "splash_global_dkv_reuse_scaled_q",
            k.splash_global_dkv_reuse_scaled_q,
        ),
        (
            "splash_fwd_strip_hoist_q_scale",
            k.splash_fwd_strip_hoist_q_scale,
        ),
        (
            "splash_global_dkv_head_stitch",
            k.splash_global_dkv_head_stitch,
        ),
    )
    for ob_name, ob_val in opt_bool_fields:
      if ob_val is not None and not isinstance(ob_val, bool):
        raise ValueError(f"{ob_name} must be a bool or None, got {ob_val!r}.")
    if k.splash_fwd_strip_schedule not in _VALID_FWD_STRIP_SCHEDULES:
      raise ValueError(
          f"splash_fwd_strip_schedule={k.splash_fwd_strip_schedule!r} must be" f" one of {_VALID_FWD_STRIP_SCHEDULES}."
      )
    if k.use_fused_vocab_vjp_scan and k.num_vocab_chunks != 1:
      raise ValueError("use_fused_vocab_vjp_scan=True requires num_vocab_chunks == 1, got" f" {k.num_vocab_chunks}.")
    if k.splash_local_dq_reduction_steps is not None and (
        isinstance(k.splash_local_dq_reduction_steps, bool) or k.splash_local_dq_reduction_steps not in (1, 3)
    ):
      raise ValueError(
          "splash_local_dq_reduction_steps must be None, 1, or 3, got" f" {k.splash_local_dq_reduction_steps!r}."
      )
    stacked_fields = (
        ("splash_num_stacked_q_heads", k.splash_num_stacked_q_heads, False),
        (
            "splash_num_stacked_q_heads_dkv",
            k.splash_num_stacked_q_heads_dkv,
            False,
        ),
        (
            "splash_global_num_stacked_q_heads",
            k.splash_global_num_stacked_q_heads,
            True,
        ),
        (
            "splash_global_num_stacked_q_heads_dkv",
            k.splash_global_num_stacked_q_heads_dkv,
            True,
        ),
    )
    for st_name, st_val, allow_none in stacked_fields:
      if allow_none and st_val is None:
        continue
      if isinstance(st_val, bool) or not isinstance(st_val, int) or st_val < 1:
        raise ValueError(f"{st_name} must be an int >= 1, got {st_val!r}.")
    if k.use_fp8 and k.fp8_routed_all_gather:
      raise ValueError(
          "use_fp8=True and fp8_routed_all_gather=True are mutually exclusive;"
          " use use_fp8=True for end-to-end FP8 weights and matmuls."
      )
    if (
        isinstance(k.sc_min_collective_bytes, bool)
        or not isinstance(k.sc_min_collective_bytes, int)
        or k.sc_min_collective_bytes < 0
    ):
      raise ValueError("sc_min_collective_bytes must be an int >= 0, got" f" {k.sc_min_collective_bytes!r}.")
    if k.routed_w_tiled:
      d_routed_loc = m.hidden_size // (s.fsdp_size * s.cp_size)
      if d_routed_loc % 8 != 0 or m.moe_intermediate_size % 8 != 0:
        raise ValueError(
            "routed_w_tiled=True requires (hidden_size // (fsdp_size *"
            f" cp_size))={d_routed_loc} and"
            f" moe_intermediate_size={m.moe_intermediate_size} divisible by 8."
        )
    if k.transpose_routed_down_ag and k.shard_routed_on_expert_axis:
      raise ValueError("transpose_routed_down_ag=True requires" " shard_routed_on_expert_axis=False.")
    if k.shard_routed_on_expert_axis:
      if k.routed_w_tiled:
        raise ValueError("shard_routed_on_expert_axis=True requires routed_w_tiled=False.")
      if s.ep_size != 1:
        raise ValueError("shard_routed_on_expert_axis=True requires ep_size == 1, got" f" ep_size={s.ep_size}.")
      routed_ways = s.fsdp_size * s.cp_size
      if m.num_experts % routed_ways != 0:
        raise ValueError(
            "shard_routed_on_expert_axis=True requires num_experts="
            f"{m.num_experts} divisible by fsdp_size * cp_size = {routed_ways}."
        )
    opt_tile_fields = (
        ("gmm_tile_m", k.gmm_tile_m),
        ("tgmm_tile_m", k.tgmm_tile_m),
        ("splash_global_block_q", k.splash_global_block_q),
        ("splash_global_block_kv", k.splash_global_block_kv),
        ("splash_global_block_kv_compute", k.splash_global_block_kv_compute),
        ("splash_global_block_q_dkv", k.splash_global_block_q_dkv),
        ("splash_global_block_kv_dkv", k.splash_global_block_kv_dkv),
        (
            "splash_global_block_kv_dkv_compute",
            k.splash_global_block_kv_dkv_compute,
        ),
    )
    for opt_name, opt_val in opt_tile_fields:
      if opt_val is not None:
        if isinstance(opt_val, bool) or not isinstance(opt_val, int) or opt_val < 128 or opt_val % 128 != 0:
          raise ValueError(f"{opt_name}={opt_val!r} must be an int >= 128 and divisible by" " 128.")
    if k.use_splash:
      if isinstance(k.splash_dq_reduction_steps, bool) or (k.splash_dq_reduction_steps not in (1, 3)):
        raise ValueError(
            "use_splash=True requires splash_dq_reduction_steps in (1, 3) to"
            " avoid 128 GiB dQ scratch blowup, got"
            f" {k.splash_dq_reduction_steps!r}."
        )
      if s.cp_size > 1 and k.splash_dq_reduction_steps != 3:
        raise ValueError(
            "use_splash=True with cp_size > 1 requires"
            " splash_dq_reduction_steps == 3 for non-contiguous LB CP dkv q"
            f" blocks, got {k.splash_dq_reduction_steps!r}."
        )
      if m.head_dim < 128 or m.head_dim % 128 != 0:
        raise ValueError(f"use_splash=True requires head_dim={m.head_dim} >= 128 and" " divisible by 128.")
      q_chunk = t.seq_len // (2 * s.cp_size) if s.cp_size > 1 else t.seq_len
      if q_chunk < 128 or q_chunk % 128 != 0:
        raise ValueError(f"use_splash=True requires local query chunk length {q_chunk} >=" " 128 and divisible by 128.")
      tile_fields = (
          ("splash_block_q", k.splash_block_q),
          ("splash_block_kv", k.splash_block_kv),
          ("splash_block_kv_compute", k.splash_block_kv_compute),
          ("splash_block_q_dkv", k.splash_block_q_dkv),
          ("splash_block_kv_dkv", k.splash_block_kv_dkv),
          ("splash_block_kv_dkv_compute", k.splash_block_kv_dkv_compute),
      )
      for tile_name, tile_size in tile_fields:
        if tile_size < 128 or tile_size % 128 != 0:
          raise ValueError(f"{tile_name}={tile_size} must be >= 128 and divisible by 128.")
      gbq = k.splash_global_block_q or k.splash_block_q
      gbkv = k.splash_global_block_kv or k.splash_block_kv
      gbkv_c = k.splash_global_block_kv_compute or k.splash_block_kv_compute
      gbq_dkv = k.splash_global_block_q_dkv or k.splash_block_q_dkv
      gbkv_dkv = k.splash_global_block_kv_dkv or k.splash_block_kv_dkv
      gbkv_dkv_c = k.splash_global_block_kv_dkv_compute or k.splash_block_kv_dkv_compute
      if k.splash_qk_diag_skip or k.splash_sv_diag_skip:
        if q_chunk < 256:
          raise ValueError(f"Splash diag_skip requires local query chunk {q_chunk} >= 256.")
      if k.splash_qk_diag_skip:
        if (
            gbq != gbkv
            or gbq < 256
            or gbkv_c < 256
            or gbkv % gbkv_c != 0
            or gbq_dkv != gbkv_dkv
            or gbq_dkv < 256
            or gbkv_dkv_c < 256
            or gbkv_dkv % gbkv_dkv_c != 0
        ):
          raise ValueError(
              "splash_qk_diag_skip=True requires global block_q == block_kv >="
              " 256, block_kv % block_kv_compute == 0 (compute >= 256), and"
              " block_q_dkv == block_kv_dkv >= 256 with block_kv_dkv %"
              " block_kv_dkv_compute == 0 (compute >= 256)."
          )
      if k.splash_sv_diag_skip:
        if gbq != gbkv or gbkv != gbkv_c or gbq < 256:
          raise ValueError(
              "splash_sv_diag_skip=True requires global block_q == block_kv =="
              f" block_kv_compute >= 256, got ({gbq}, {gbkv}, {gbkv_c})."
          )
      q_heads_per_kv = m.num_attention_heads // m.num_key_value_heads
      g_st_q = k.splash_global_num_stacked_q_heads or k.splash_num_stacked_q_heads
      g_st_dkv = k.splash_global_num_stacked_q_heads_dkv or k.splash_num_stacked_q_heads_dkv
      for st_label, st_v in (
          ("splash_num_stacked_q_heads", k.splash_num_stacked_q_heads),
          ("splash_num_stacked_q_heads_dkv", k.splash_num_stacked_q_heads_dkv),
          ("splash_global_num_stacked_q_heads", g_st_q),
          ("splash_global_num_stacked_q_heads_dkv", g_st_dkv),
      ):
        if q_heads_per_kv % st_v != 0:
          raise ValueError(f"{st_label}={st_v} must divide q_heads_per_kv={q_heads_per_kv}.")
      if (k.splash_qk_diag_skip or k.splash_sv_diag_skip) and g_st_q > 1:
        raise ValueError(
            "splash_qk_diag_skip/splash_sv_diag_skip cannot be combined with" f" global num_stacked_q_heads={g_st_q} > 1."
        )
      if k.splash_qk_diag_skip and g_st_dkv > 1:
        raise ValueError(
            "splash_qk_diag_skip=True cannot be combined with global" f" num_stacked_q_heads_dkv={g_st_dkv} > 1."
        )
      g_head_stitch = (
          k.splash_dkv_head_stitch if k.splash_global_dkv_head_stitch is None else k.splash_global_dkv_head_stitch
      )
      if k.splash_qk_diag_skip and g_head_stitch:
        raise ValueError("splash_qk_diag_skip=True cannot be combined with global" " dkv_head_stitch=True.")
      if k.splash_use_local_halo_cp and s.cp_size > 1:
        s_loc = t.seq_len // s.cp_size
        tile_limit = (s_loc // 2) if k.use_lb_cp_layout else s_loc
        if tile_limit < 128:
          raise ValueError("splash_use_local_halo_cp=True requires local stripe length" f" {tile_limit} >= 128.")
        clamped_loc = (tile_limit // 128) * 128
        lbq = min(k.splash_block_q, clamped_loc)
        lbkv = min(k.splash_block_kv, clamped_loc)
        lbq_dkv = min(k.splash_block_q_dkv, clamped_loc)
        lbkv_dkv = min(k.splash_block_kv_dkv, clamped_loc)
        kv_lcm = math.lcm(lbq, lbkv, lbq_dkv, lbkv_dkv)
        halo_len = math.ceil(max(1, m.sliding_window - 1) / kv_lcm) * kv_lcm
        if tile_limit < halo_len or tile_limit % kv_lcm != 0:
          raise ValueError(
              "splash_use_local_halo_cp=True requires local CP stripe length"
              f" {tile_limit} >= halo_len={halo_len} and divisible by"
              f" kv_lcm={kv_lcm}."
          )

  def _validate_train(self) -> None:
    """Validates training dtypes, remat policy, LR schedule, and AdamW args."""
    t = self.train
    if t.compute_dtype not in _VALID_DTYPES:
      raise ValueError(f"compute_dtype={t.compute_dtype!r} must be in {_VALID_DTYPES}.")
    if t.param_dtype not in _VALID_DTYPES:
      raise ValueError(f"param_dtype={t.param_dtype!r} must be in {_VALID_DTYPES}.")
    if t.remat_policy not in _VALID_REMAT_POLICIES:
      raise ValueError(f"remat_policy={t.remat_policy!r} must be in {_VALID_REMAT_POLICIES}.")
    if t.lr_schedule not in _VALID_LR_SCHEDULES:
      raise ValueError(f"lr_schedule={t.lr_schedule!r} must be in {_VALID_LR_SCHEDULES}.")
    if t.warmup_steps < 0 or t.total_steps < 1:
      raise ValueError(f"Invalid steps: warmup_steps={t.warmup_steps}," f" total_steps={t.total_steps}.")
    if t.learning_rate < 0.0 or t.adam_eps <= 0.0 or t.grad_clip_norm < 0.0:
      raise ValueError("learning_rate >= 0, adam_eps > 0, and grad_clip_norm >= 0 required.")


def count_flint_params(model: FlintModelConfig) -> tuple[int, int, int, int]:
  """Computes exact `(total, active, active_matmul, active_vector)` parameters.

  `active_matmul` excludes the `embed` lookup table and replaces `num_experts`
  routed experts with `num_experts_per_tok` active experts per token.

  Args:
    model: Flint architecture configuration.

  Returns:
    Tuple `(total_params, active_params, active_matmul_params,
    active_vector_params)`.
  """
  d = model.hidden_size
  q_dim = model.num_attention_heads * model.head_dim
  kv_dim = model.num_key_value_heads * model.head_dim

  embed_params = model.vocab_size * d
  lm_head_params = d * model.vocab_size
  final_norm_params = d

  attn_matmul_per_layer = 2 * d * q_dim + 2 * d * kv_dim
  vecs_per_layer = 2 * d + 2 * model.head_dim + model.num_attention_heads

  num_dense = model.num_dense_prefix_layers
  num_moe = model.num_hidden_layers - num_dense

  dense_mlp_per_layer = 3 * d * model.intermediate_size
  router_per_layer = d * model.num_experts
  shared_per_layer = 3 * d * model.shared_expert_intermediate_size
  routed_per_expert = 3 * d * model.moe_intermediate_size

  active_vector_params = model.num_hidden_layers * vecs_per_layer + final_norm_params
  active_matmul_params = (
      lm_head_params
      + model.num_hidden_layers * attn_matmul_per_layer
      + num_dense * dense_mlp_per_layer
      + num_moe * (router_per_layer + shared_per_layer + model.num_experts_per_tok * routed_per_expert)
  )
  active_params = active_matmul_params + active_vector_params
  total_params = (
      embed_params
      + active_vector_params
      + lm_head_params
      + model.num_hidden_layers * attn_matmul_per_layer
      + num_dense * dense_mlp_per_layer
      + num_moe * (router_per_layer + shared_per_layer + model.num_experts * routed_per_expert)
  )
  return total_params, active_params, active_matmul_params, active_vector_params


def calculate_tflops_per_device(cfg: FlintConfig) -> tuple[float, float, float]:
  """Calculates `(total_tflops, weight_tflops, attn_tflops)` per device step."""
  m = cfg.model
  pdb = cfg.train.global_batch_size / cfg.sharding.num_devices
  seq_len = cfg.train.seq_len
  _, _, active_matmul_params, _ = count_flint_params(m)

  weight_tflops = 6.0 * pdb * seq_len * active_matmul_params / 1e12
  num_global = sum(1 for idx in range(m.num_hidden_layers) if is_global_layer(idx, m))
  num_sliding = m.num_hidden_layers - num_global
  eff_w = min(m.sliding_window, seq_len)
  sliding_pairs = seq_len * eff_w - 0.5 * eff_w * (eff_w - 1)
  attn_flops_per_seq = (num_global * 6.0 * (seq_len**2) + num_sliding * 12.0 * sliding_pairs) * (
      m.num_attention_heads * m.head_dim
  )
  attn_tflops = pdb * attn_flops_per_seq / 1e12
  return weight_tflops + attn_tflops, weight_tflops, attn_tflops


_MODEL_FIELDS = frozenset(f.name for f in dataclasses.fields(FlintModelConfig))
_SHARD_FIELDS = frozenset(f.name for f in dataclasses.fields(FlintShardingConfig))
_KERNEL_FIELDS = frozenset(f.name for f in dataclasses.fields(FlintKernelConfig))
_TRAIN_FIELDS = frozenset(f.name for f in dataclasses.fields(FlintTrainConfig))


def _apply_overrides(
    model: FlintModelConfig,
    sharding: FlintShardingConfig,
    kernels: FlintKernelConfig,
    train: FlintTrainConfig,
    overrides: dict[str, Any],
) -> FlintConfig:
  """Applies section or flat field overrides and fails fast on unknown keys."""
  model_kw: dict[str, Any] = {}
  shard_kw: dict[str, Any] = {}
  kernel_kw: dict[str, Any] = {}
  train_kw: dict[str, Any] = {}

  for key, val in overrides.items():
    if key == "model":
      if not isinstance(val, FlintModelConfig):
        raise ValueError(f"Expected FlintModelConfig for 'model', got {val!r}.")
      model = val
    elif key == "sharding":
      if not isinstance(val, FlintShardingConfig):
        raise ValueError(f"Expected FlintShardingConfig for 'sharding', got {val!r}.")
      sharding = val
    elif key == "kernels":
      if not isinstance(val, FlintKernelConfig):
        raise ValueError(f"Expected FlintKernelConfig for 'kernels', got {val!r}.")
      kernels = val
    elif key == "train":
      if not isinstance(val, FlintTrainConfig):
        raise ValueError(f"Expected FlintTrainConfig for 'train', got {val!r}.")
      train = val
    elif key in _MODEL_FIELDS:
      model_kw[key] = val
    elif key in _SHARD_FIELDS:
      shard_kw[key] = val
    elif key in _KERNEL_FIELDS:
      kernel_kw[key] = val
    elif key in _TRAIN_FIELDS:
      train_kw[key] = val
    else:
      raise ValueError(f"Unknown FlintConfig override parameter: {key!r}.")

  if model_kw:
    model = dataclasses.replace(model, **model_kw)
  if shard_kw:
    sharding = dataclasses.replace(sharding, **shard_kw)
  if kernel_kw:
    kernels = dataclasses.replace(kernels, **kernel_kw)
  if train_kw:
    train = dataclasses.replace(train, **train_kw)
  return FlintConfig(
      model=model,
      sharding=sharding,
      kernels=kernels,
      train=train,
  )


def flint_40b_config(
    *,
    use_splash: bool = True,
    **section_or_field_overrides: Any,
) -> FlintConfig:
  """Builds a validated `FlintConfig` for RecursiveSI/flint 40B-A2.5B."""
  return _apply_overrides(
      model=FlintModelConfig(),
      sharding=FlintShardingConfig(),
      kernels=FlintKernelConfig(use_splash=use_splash),
      train=FlintTrainConfig(),
      overrides=section_or_field_overrides,
  )


def optimized_64c_recipe_overrides(
    seq_len: int = 8192,
    global_batch_size: int = 1024,
    *,
    grad_accum_steps: int = 1,
    fsdp_size: int = 128,
    cp_size: int = 1,
    ep_size: int = 1,
    remat_policy: str = "auto",
    **extra_overrides: Any,
) -> dict[str, Any]:
  """Builds validated TPU v7x `tpu7x:4x4x4` optimized recipe overrides."""
  micro_tok = ((global_batch_size // grad_accum_steps) * seq_len) // (fsdp_size * cp_size * ep_size)
  if remat_policy == "auto":
    use_sel = micro_tok <= 16384 and fsdp_size >= 64 and ep_size == 1
    resolved_remat = "selective" if use_sel else "global_selective"
  else:
    resolved_remat = remat_policy
  local_q = seq_len // (2 * cp_size) if cp_size > 1 else seq_len
  global_block_q = min(512, local_q)
  global_block_kv = min(2048, seq_len)
  global_block_kv_compute = min(512, global_block_kv)
  global_block_q_dkv = min(256, local_q)
  global_block_kv_dkv = min(4096, seq_len)
  global_block_kv_dkv_compute = min(512, global_block_kv_dkv)
  use_fp8 = resolved_remat in ("selective", "global_selective")
  overrides: dict[str, Any] = {
      "seq_len": seq_len,
      "global_batch_size": global_batch_size,
      "grad_accum_steps": grad_accum_steps,
      "fsdp_size": fsdp_size,
      "cp_size": cp_size,
      "ep_size": ep_size,
      "remat_policy": resolved_remat,
      "save_extra_selective_residuals": resolved_remat == "selective",
      "moe_impl": "pallas_gmm_v2",
      "sc_collective_mode": "dual_sc",
      "routed_w_tiled": (2048 // (fsdp_size * cp_size)) % 8 == 0,
      "moe_bijective_permute": True,
      "fuse_layer_projections": True,
      "num_vocab_chunks": 1,
      "vocab_token_chunk_size": 2048,
      "use_fused_vocab_vjp_scan": True,
      "precompute_rope_tables": True,
      "splash_block_q": 512,
      "splash_block_kv": 512,
      "splash_block_kv_compute": 256,
      "splash_block_q_dkv": 512,
      "splash_block_kv_dkv": 512,
      "splash_block_kv_dkv_compute": 512,
      "splash_local_dq_reduction_steps": 1,
      "splash_num_stacked_q_heads": 8,
      "splash_num_stacked_q_heads_dkv": 8,
      "splash_use_scheduler": True,
      "splash_use_scheduler_dkv": False,
      "splash_dkv_reuse_scaled_q": True,
      "splash_use_local_halo_cp": True,
      "splash_ignore_segment_ids": True,
      "splash_fwd_strip_schedule": "auto",
      "splash_compact_lse_residual": True,
      "splash_dkv_head_stitch": False,
      "splash_global_dkv_head_stitch": True,
      "splash_global_block_q": global_block_q,
      "splash_global_block_kv": global_block_kv,
      "splash_global_block_kv_compute": global_block_kv_compute,
      "splash_global_block_q_dkv": global_block_q_dkv,
      "splash_global_block_kv_dkv": global_block_kv_dkv,
      "splash_global_block_kv_dkv_compute": global_block_kv_dkv_compute,
      "splash_global_num_stacked_q_heads": 8,
      "splash_global_num_stacked_q_heads_dkv": 4,
      "splash_global_use_scheduler": True,
      "splash_global_use_scheduler_dkv": False,
      "splash_global_dkv_reuse_scaled_q": True,
      "splash_qk_diag_skip": False,
      "splash_sv_diag_skip": False,
      "splash_use_base2_exp": True,
      "replicate_router_gate": True,
      "moe_slot_major_pack": False,
      "prefetch_routed_weights": True,
      "transpose_routed_down_ag": True,
      "transpose_dense_rhs_ag": True,
      "moe_zero_scatter_router": True,
      "use_sc_moe_gather_reduce": True,
      "use_sc_counting_sort": True,
      "use_pallas_router_topk": True,
      "compact_router_maps": True,
      "skip_router_aux_stats_when_zero_coef": True,
      "use_lb_cp_layout": True,
      "early_global_kv_proj": True,
      "use_pallas_lb_kv_reorder": True,
      "use_fused_qk_prologue": True,
      "use_fp8": use_fp8,
  }
  overrides.update(extra_overrides)
  flint_40b_config(**overrides)
  return overrides


def default_recipe_overrides() -> dict[str, Any]:
  """Returns overrides for the default optimized 256K recipe.

  S=262,144, GBS=32 (PDB=0.25), GA=1, FSDP=32, CP=4, EP=1 on 64 TPU v7x chips
  (128 TensorCores); identical to `recipes/flint_64c_256k.sh`.
  """
  return optimized_64c_recipe_overrides(
      seq_len=262_144,
      global_batch_size=32,
      grad_accum_steps=1,
      fsdp_size=32,
      cp_size=4,
      ep_size=1,
  )


def mini_flint_config(
    *,
    use_splash: bool = False,
    **overrides: Any,
) -> FlintConfig:
  """Builds a validated 9-layer (`2 + 1x4 + 3`) mini `FlintConfig` for tests."""
  return _apply_overrides(
      model=FlintModelConfig(
          vocab_size=256,
          hidden_size=128,
          intermediate_size=384,
          moe_intermediate_size=32,
          shared_expert_intermediate_size=64,
          num_hidden_layers=9,
          num_dense_prefix_layers=2,
          moe_cycle=4,
          num_attention_heads=8,
          num_key_value_heads=2,
          head_dim=16,
          num_experts=16,
          num_experts_per_tok=6,
          sliding_window=8,
          bos_token_id=0,
          pad_token_id=1,
      ),
      sharding=FlintShardingConfig(fsdp_size=1, cp_size=1, ep_size=1),
      kernels=FlintKernelConfig(
          use_splash=use_splash,
          num_vocab_chunks=2,
          vocab_token_chunk_size=16,
      ),
      train=FlintTrainConfig(
          global_batch_size=2,
          seq_len=16,
          compute_dtype="float32",
          param_dtype="float32",
      ),
      overrides=overrides,
  )
