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

"""Config for the pure-JAX Qwen3.5 text decoder (MaxText `decoder_block: qwen3_5`).

Field names and defaults mirror MaxText's resolved config for
`model_name=qwen3.5-397b-a17b`, so that a MaxText `pyconfig` translates 1:1
(`from_maxtext`).  As in `src/maxtext/experimental/mpmd/jax/deepseek/config.py`, only the knobs the
training forward pass reads are kept, and what this port does not implement is
refused rather than silently ignored.

The port calls a handful of `maxtext.experimental.mpmd.jax.deepseek` functions whose arithmetic is
identical for both models (dense projections, the gated MLP, the plain RMSNorm,
the embedding gather, the routed experts and their dispatch, the attention
kernels, the remat policy table).
`maxtext.experimental.mpmd.jax.deepseek.config.SharedLayerConfig` lists exactly the config fields
those functions read; `Qwen35Config` satisfies it (`qwen35_model_test`
checks both directions), which is why it carries DeepSeek's names for them.

How the model shards is `shard`, a `maxtext.experimental.mpmd.jax.sharding_mode.ShardPolicy`
(the default policy, as DeepSeek's), plus one Qwen knob, `gdn_core`, for where
the Gated DeltaNet's chunked delta rule runs; its default is the program the
port has always traced.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

# The config surface of the shared DeepSeek functions, which Qwen35Config
# satisfies (re-exported: it is the one Protocol both models are held to).
from maxtext.experimental.mpmd.jax.deepseek.config import SHARED_LAYER_FIELDS, SharedLayerConfig  # noqa: F401
from maxtext.experimental.mpmd.jax.deepseek.model import REMAT_POLICIES
from maxtext.experimental.mpmd.jax.sharding_mode import ShardPolicy


ATTENTION_KERNELS = ("splash", "tokamax_splash", "dot_product", "reference", "cudnn")
REMAT_GRANULARITIES = ("block", "layer", "block+layer")
NORM_NUMERICS = ("fp32", "bf16")
GDN_CORES = ("inline", "manual")
# How the GDN's causal depthwise conv runs (`gdn.causal_depthwise_conv`): 'conv', lax.conv_general_dilated as
# MaxText's nnx.Conv (TPU's, bitwise MaxText's), or 'shift', its K shifted multiply-adds in fp32 (XLA:GPU fuses
# them; cuDNN's grouped conv is ~10x slower than the bytes it moves there).
GDN_CONV_IMPLS = ("conv", "shift")
# How the chunked delta rule runs (`gdn.chunk_gated_delta_rule`): 'scan', its inter-chunk recurrence a lax.scan
# (MaxText's), or 'triton', on Triton kernels (`gdn_triton`): at a chunk of 16, 32 or 64 (64 by default) the whole
# rule (gdn_triton.FUSED: the WY factors, the triangular solve, the intra-chunk output and their backward as per-chunk
# kernels around the recurrence's), at another chunk the recurrence alone, one kernel each way, the chunk outputs
# batched over the chunks in XLA.  GPU only, per device -- inside the MPMD pipeline's shard_maps or under
# gdn_core='manual' -- and without packed documents: the kernels restart no state at a document, so the rule refuses
# segment ids, and a caller that passes them (qwen35.train does, all one for unpacked rows) needs gdn_packing=False,
# which keeps them out of the rule.  The MPMD pipeline's GDN takes none.
GDN_SCAN_IMPLS = ("scan", "triton")
# How the chunked delta rule inverts its unit-lower-triangular WY matrix (`gdn.unit_lower_inv`), MaxText's
# `gdn_chunk_inverse`: 'block_doubling', exact sub-block inverses merged by matmuls (MaxText's default), or 'solve',
# `solve_triangular` (MaxText's reference path).
GDN_CHUNK_INVERSES = ("block_doubling", "solve")


@dataclasses.dataclass(frozen=True)
class Qwen35Config:
  """Resolved Qwen3.5 hyperparameters (defaults: qwen3.5-397b-a17b)."""

  # --- vocabulary / shape ---------------------------------------------------
  vocab_size: int = 248320
  emb_dim: int = 4096
  num_decoder_layers: int = 60
  # A block is `cycle` layers: cycle - 1 Gated DeltaNet layers, then one gated
  # full-attention layer (layer i is attention iff (i + 1) % cycle == 0).
  inhomogeneous_layer_cycle_interval: int = 4

  # --- gated full attention -------------------------------------------------
  num_query_heads: int = 32
  num_kv_heads: int = 2
  head_dim: int = 256
  # Which of MaxText's two rotary embeddings the gated attention applies
  # (attention.py): True, the text-only mRoPE of the Qwen3.5 ymls --
  # rotate-half RoPE on the first `rotary_dim` features with fp32 cos/sin; False,
  # PartialRotaryEmbedding, base.yml's default and so Qwen3.8-2.4T's (its yml
  # sets no use_mrope): sin/cos rounded to the activation dtype, the rotation in
  # it.  The same rotation, other roundings.
  use_mrope: bool = True
  # The mRoPE's (t, h, w) sections must sum to rotary_dim // 2 (MaxText checks
  # it); for text positions they do not change the math.  Unread without mRoPE.
  partial_rotary_factor: float = 0.25
  rope_min_timescale: float = 1.0
  rope_max_timescale: float = 10_000_000.0
  mrope_section: tuple[int, ...] = (11, 11, 10)

  # --- Gated DeltaNet (linear attention) -------------------------------------
  gdn_num_key_heads: int = 16
  gdn_num_value_heads: int = 64
  gdn_key_head_dim: int = 128
  gdn_value_head_dim: int = 128
  gdn_conv_kernel_dim: int = 4
  gdn_chunk_size: int = 64
  use_qk_norm_in_gdn: bool = True
  # Where the chunked delta rule runs (`_ops.gdn_core`): 'inline' in the
  # surrounding program (the port's program; typed op by op under explicit),
  # 'manual' per device in a shard_map over the stage's batch axes, in BOTH
  # shard modes -- MaxText's layout, and the same per-device program under
  # auto and explicit.  A strategy pinned in both modes, so it changes the
  # auto program only when set.
  gdn_core: str = "inline"
  # GDN_CONV_IMPLS: 'conv' (MaxText's op) or 'shift' (the GPU default of ds_ep_pipeline).
  gdn_conv_impl: str = "conv"
  # GDN_SCAN_IMPLS: 'scan' (MaxText's) or 'triton'.  Orthogonal to gdn_core, which says where the rule runs
  # ('triton' needs it per device: gdn_core='manual' under auto / explicit, or a caller's shard_map).  'triton' takes
  # no packed documents: with segment ids (the single-controller model always passes them) set gdn_packing=False.
  gdn_scan_impl: str = "scan"
  # How the GDN reads segment ids and inverts its WY matrix: MaxText's options (`from_maxtext` reads them off its
  # config).  The rest is MaxText's GDN as it is (`gdn.gated_delta_net`): the padding's q/k/v projections zeroed
  # before the causal conv as well as after it, g = -exp(A_log) * softplus(a + dt_bias) in fp32, and the WY matrix
  # fp32(k k^T) * (fp32(beta) * exp(dg)), k k^T at the key heads (rounded to `dtype` as it comes out of the matmul),
  # then repeated to the value heads, the query and key heads repeated inside the delta rule.
  #   gdn_packing          -- MaxText's `packing` as its GDN reads it (true by default there): every segment id is a
  #                           document, and the conv taps and the recurrent state stop at its edges (canonical
  #                           documents: a segment id that comes back after another is a new one).  False (MaxText's
  #                           packing=false): segment ids only zero padding (segment 0); the state and the conv window
  #                           run on from one document into the next, as attention does not.  Without segment ids
  #                           (the MPMD pipeline's GDN) neither applies.  gdn_scan_impl 'triton' has no packed path:
  #                           with segment ids it needs False.
  #   gdn_chunk_inverse    -- GDN_CHUNK_INVERSES.
  # HF / FLA reset the state per document (cu_seqlens) as gdn_packing does, but zero padded hidden states before the
  # input projections.
  gdn_packing: bool = True
  gdn_chunk_inverse: str = "block_doubling"

  # --- MoE ------------------------------------------------------------------
  moe_mlp_dim: int = 1024
  num_experts: int = 512
  num_experts_per_tok: int = 10
  # 0 or 1: MaxText builds ONE shared expert of width `moe_mlp_dim` whenever
  # this is positive, with a sigmoid gate in front of it.
  shared_experts: int = 1
  norm_topk_prob: bool = True
  mlp_activations: tuple[str, ...] = ("silu", "linear")
  # `ragged` = MaxText sparse_matmul (sort + grouped matmul), `dense` =
  # dense_matmul (einsum over every expert).  See `src/maxtext/experimental/mpmd/jax/deepseek/moe.py`.
  moe_impl: str = "ragged"
  gmm_fp32_accum: bool = True
  gmm_exact_drhs: bool = False
  moe_use_direct_token_gather: bool = False
  float32_weight_sum: bool = True
  float32_gate_logits: bool = False
  # Not plumbed to the loss (as in the DeepSeek port): > 0 is refused here, at
  # construction, rather than on the first trace.
  load_balance_loss_weight: float = 0.0

  # --- numerics -------------------------------------------------------------
  normalization_layer_epsilon: float = 1.0e-6
  dtype: Any = jnp.bfloat16
  # MaxText's revisions disagree on one leaf: before 2026-09-10 it built
  # `shared_expert_gate` without a weight_dtype, so that kernel stayed fp32
  # whatever this says (`from_maxtext` refuses anything but fp32 for that
  # reason); later revisions type it like every other leaf.
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"
  cast_logits_to_fp32: bool = True
  # How the two RMSNorm kinds round, which differs between MaxText revisions
  # (both compute their statistics and the normalised value in fp32):
  #   'fp32' -- MaxText since 2026-09-10: the zero-centred norms multiply the
  #             fp32 normalised value by `fp32(scale) + 1` and round once; the
  #             GDN gated norm multiplies it by `fp32(scale)` and by
  #             `silu(fp32(z))`, then rounds once.
  #   'bf16' -- MaxText before then (which the port was first written
  #             against): the normalised value is rounded to `dtype` first,
  #             and the scale multiply (`bf16(scale) + 1`, the add itself in
  #             bf16, for the zero-centred norms) runs in `dtype`.
  # HF (transformers 5.13.0, bf16 weights) matches neither: its zero-centred
  # norm rounds as 'fp32' does, its gated norm as 'bf16' does.  With
  # `dtype=float32` the two settings are the same computation.
  norm_numerics: str = "fp32"

  # --- attention kernel (see src/maxtext/experimental/mpmd/jax/deepseek/config.py) --------------------
  attention_kernel: str = "splash"
  sa_block_q: int = 512
  sa_block_kv: int = 512
  sa_block_kv_compute: int = 512
  sa_block_q_dkv: int = 0
  sa_block_kv_dkv: int = 0
  sa_block_kv_dkv_compute: int = 0
  sa_cost_estimate_flops_fwd: int = -1
  sa_cost_estimate_flops_bwd: int = -1
  # tokamax's base-2 softmax, MaxText's: see src/maxtext/experimental/mpmd/jax/deepseek/config.py
  sa_use_base2_exp: bool = True

  # --- training -------------------------------------------------------------
  max_target_length: int = 4096
  scan_layers: bool = True
  # MaxText's policy name for the block's checkpoint, and the layers' unless
  # `layer_remat_policy` names one ('none' and 'full' both mean "save nothing,
  # recompute everything", as in MaxText).
  remat_policy: str = "full"
  # Where the `jax.checkpoint`s go on the scanned path:
  #   'block'       -- one around each 4-layer block body (MaxText before its
  #                    nested Qwen3 scan of 2026-10-21), policy
  #                    `remat_policy`.  Stores one [B, L, E] per block; the
  #                    block's backward holds all four layers' recomputed
  #                    intermediates at once.
  #   'layer'       -- one around each layer, policy `layer_remat_policy`; the
  #                    scan body itself is not checkpointed (MaxText since
  #                    that nested scan: a checkpoint per sub-layer inside
  #                    it, none around the block).  Stores the four layer
  #                    inputs per block, holds one layer's working set.
  #   'block+layer' -- both, nested: the block's checkpoint (`remat_policy`)
  #                    around the layers' (`layer_remat_policy`).  Block remat's
  #                    stored residuals with one layer's working set, for one
  #                    more forward pass of the GDN layers.
  # Measured on v5p (jax 0.12, libtpu 0.0.49), the backward program of a
  # one-block chunk at 397B dims, 32 experts, 4096 tokens: 29.3 / 12.0 / 12.0 GB
  # of temporaries for block / layer / block+layer, storing 34 / 134 / 34 MB of
  # residuals per microbatch (two-block chunk: 37.9 / 21.0 / 21.6 GB); with
  # norm_numerics='bf16', 29.0 / 14.2 / 13.7 GB (38.1 / 21.0 / 21.6 GB).
  # `from_maxtext` does not map it: 'block', whatever the MaxText revision.
  remat_granularity: str = "block"
  # The per-layer checkpoints' policy; '' means `remat_policy`.
  layer_remat_policy: str = ""

  # --- sharding -------------------------------------------------------------
  # How the model shards (`src/maxtext/experimental/mpmd/jax/sharding_mode.py`): the default
  # policy, the program this port has always traced, the only one there is.
  # Frozen, like this class, which stays hashable.
  shard: ShardPolicy = ShardPolicy()

  def __post_init__(self):
    # Hashable: the config is a static argument of jitted helpers
    # (`attention.rope_inv_freq_jit`).
    object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
    object.__setattr__(self, "mlp_activations", tuple(self.mlp_activations))
    cyc = self.inhomogeneous_layer_cycle_interval
    if cyc < 1 or self.num_decoder_layers % cyc:
      raise ValueError(f"num_decoder_layers={self.num_decoder_layers} is not a multiple of " f"the {cyc}-layer block")
    if self.gdn_num_value_heads % self.gdn_num_key_heads:
      raise ValueError(
          f"gdn_num_value_heads={self.gdn_num_value_heads} is not a multiple of "
          f"gdn_num_key_heads={self.gdn_num_key_heads}"
      )
    if self.num_query_heads % self.num_kv_heads:
      raise ValueError(
          f"num_query_heads={self.num_query_heads} is not a multiple of " f"num_kv_heads={self.num_kv_heads}"
      )
    rd = self.rotary_dim
    if rd <= 0 or rd % 2 or rd > self.head_dim:
      raise ValueError(
          f"rotary_dim={rd} (head_dim {self.head_dim} x partial_rotary_factor "
          f"{self.partial_rotary_factor}) must be even and in (0, head_dim]"
      )
    if self.use_mrope and sum(self.mrope_section) != rd // 2:
      raise ValueError(f"mrope_section {tuple(self.mrope_section)} must sum to rotary_dim // 2 " f"= {rd // 2}")
    if not 1 <= self.num_experts_per_tok <= self.num_experts:
      raise ValueError(f"num_experts_per_tok={self.num_experts_per_tok} not in " f"[1, num_experts={self.num_experts}]")
    if self.moe_mlp_dim < 1:
      raise ValueError(f"moe_mlp_dim={self.moe_mlp_dim} must be positive")
    if self.shared_experts not in (0, 1):
      raise ValueError(
          f"shared_experts={self.shared_experts}: Qwen3.5 has at most one shared "
          "expert (MaxText builds one of width moe_mlp_dim for any positive value)"
      )
    if self.moe_impl not in ("ragged", "dense"):
      raise ValueError(f"unknown moe_impl {self.moe_impl!r}")
    if self.attention_kernel not in ATTENTION_KERNELS:
      raise ValueError(f"unknown attention_kernel {self.attention_kernel!r}")
    if self.remat_granularity not in REMAT_GRANULARITIES:
      raise ValueError(f"remat_granularity must be one of {REMAT_GRANULARITIES}, got " f"{self.remat_granularity!r}")
    if self.norm_numerics not in NORM_NUMERICS:
      raise ValueError(f"norm_numerics must be one of {NORM_NUMERICS}, got {self.norm_numerics!r}")
    if self.gdn_chunk_size < 1 or self.gdn_conv_kernel_dim < 1:
      raise ValueError("gdn_chunk_size and gdn_conv_kernel_dim must be positive")
    if self.load_balance_loss_weight > 0.0:
      # The aux loss would be computed per layer and never reach the loss -- a
      # different model from MaxText's, silently (the DeepSeek port refuses it too).
      raise NotImplementedError(
          "load_balance_loss_weight > 0 is not supported: the aux loss " "is not plumbed out of the layer stack"
      )
    # MaxText has more policies than the port runs; an unknown name would only
    # fail on the first trace, deep inside the scanned block.
    for field in ("remat_policy", "layer_remat_policy"):
      name = getattr(self, field)
      if name and name not in REMAT_POLICIES:
        raise ValueError(f"{field}={name!r} is not supported; the policies are " f"{sorted(REMAT_POLICIES)}")
    if not isinstance(self.shard, ShardPolicy):
      raise TypeError(f"shard must be a maxtext.experimental.mpmd.jax.sharding_mode.ShardPolicy, got {self.shard!r}")
    if self.gdn_core not in GDN_CORES:
      raise ValueError(f"gdn_core must be one of {GDN_CORES}, got {self.gdn_core!r}")
    if self.gdn_conv_impl not in GDN_CONV_IMPLS:
      raise ValueError(f"gdn_conv_impl must be one of {GDN_CONV_IMPLS}, got {self.gdn_conv_impl!r}")
    if self.gdn_scan_impl not in GDN_SCAN_IMPLS:
      raise ValueError(f"gdn_scan_impl must be one of {GDN_SCAN_IMPLS}, got {self.gdn_scan_impl!r}")
    if self.gdn_chunk_inverse not in GDN_CHUNK_INVERSES:
      raise ValueError(f"gdn_chunk_inverse must be one of {GDN_CHUNK_INVERSES}, got {self.gdn_chunk_inverse!r}")

  # --- derived --------------------------------------------------------------
  @property
  def num_blocks(self) -> int:
    return self.num_decoder_layers // self.inhomogeneous_layer_cycle_interval

  @property
  def rotary_dim(self) -> int:
    return int(self.head_dim * self.partial_rotary_factor)

  @property
  def gdn_key_dim(self) -> int:
    return self.gdn_num_key_heads * self.gdn_key_head_dim

  @property
  def gdn_value_dim(self) -> int:
    return self.gdn_num_value_heads * self.gdn_value_head_dim

  @property
  def gdn_v_per_k(self) -> int:
    """Value heads per key head (value head j reads key head j // v_per_k)."""
    return self.gdn_num_value_heads // self.gdn_num_key_heads

  @property
  def gdn_conv_dim(self) -> int:
    """Channels of the depthwise conv: [q_all | k_all | v_all]."""
    return 2 * self.gdn_key_dim + self.gdn_value_dim

  @property
  def gdn_qkvz_dim(self) -> int:
    return 2 * self.gdn_key_dim + 2 * self.gdn_value_dim

  @property
  def shared_expert_mlp_dim(self) -> int:
    return self.moe_mlp_dim

  @property
  def block_remat(self) -> bool:
    """Is each scanned block body inside a `jax.checkpoint`?"""
    return bool(self.scan_layers) and self.remat_granularity in ("block", "block+layer")

  @property
  def layer_remat(self) -> bool:
    """Is each layer inside its own `jax.checkpoint`?"""
    return bool(self.scan_layers) and self.remat_granularity in ("layer", "block+layer")

  @property
  def layer_remat_policy_name(self) -> str:
    return self.layer_remat_policy or self.remat_policy

  def is_attention_layer(self, i: int) -> bool:
    """Layer i (global or within a block) is gated full attention."""
    return (i + 1) % self.inhomogeneous_layer_cycle_interval == 0

  def replace(self, **kwargs) -> "Qwen35Config":
    return dataclasses.replace(self, **kwargs)


# --------------------------------------------------------------------- presets
# The tiny shape at which the port is bit-exact against MaxText on CPU: every
# structural feature of 397B-A17B (3:1 GDN:attention, GQA 2:1, 4 value heads per
# GDN key head, partial RoPE 0.25, top-k MoE + sigmoid-gated shared expert,
# untied head) at toy widths.
TINY = dict(
    vocab_size=256,
    emb_dim=64,
    num_decoder_layers=8,
    num_query_heads=4,
    num_kv_heads=2,
    head_dim=64,
    mrope_section=(3, 3, 2),
    gdn_num_key_heads=2,
    gdn_num_value_heads=8,
    gdn_key_head_dim=16,
    gdn_value_head_dim=16,
    gdn_chunk_size=16,
    moe_mlp_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    max_target_length=64,
    attention_kernel="dot_product",
)

# Per-layer dims of qwen3.5-35b-a3b (the class defaults are 397B-A17B's).
_Q35B = dict(
    emb_dim=2048,
    num_query_heads=16,
    num_kv_heads=2,
    head_dim=256,
    gdn_num_key_heads=16,
    gdn_num_value_heads=32,
    gdn_key_head_dim=128,
    gdn_value_head_dim=128,
    moe_mlp_dim=512,
    num_experts_per_tok=8,
)


def tiny(**overrides) -> Qwen35Config:
  """The tiny CPU config (bf16, dot-product attention), bit-exact vs MaxText."""
  return Qwen35Config(**{**TINY, **overrides})


def qwen35_397b(
    num_decoder_layers: int = 60, num_experts: int = 512, vocab_size: int = 248320, **overrides
) -> Qwen35Config:
  """qwen3.5-397b-a17b per-layer dims; depth, expert count and vocab are knobs.

  Shrinking `num_experts` keeps the active FLOPs per token (top-k is fixed).
  `num_decoder_layers` must be a multiple of the 4-layer block (checked).
  Vocab 32768 keeps the head off the critical stage.
  """
  return Qwen35Config(num_decoder_layers=num_decoder_layers, num_experts=num_experts, vocab_size=vocab_size, **overrides)


def qwen35_35b(
    num_decoder_layers: int = 40, num_experts: int = 256, vocab_size: int = 248320, **overrides
) -> Qwen35Config:
  """qwen3.5-35b-a3b per-layer dims; depth, expert count and vocab are knobs."""
  return Qwen35Config(
      **{
          **_Q35B,
          "num_decoder_layers": num_decoder_layers,
          "num_experts": num_experts,
          "vocab_size": vocab_size,
          **overrides,
      }
  )


# ------------------------------------------------------------------ MaxText
def resolve_attention_kernel(cfg, platform: str | None = None) -> str:
  """`attention_op.AttentionOp.apply_attention`'s train-mode dispatch.

  `autoselected` runs the bf16 dot-product path below 128 tokens AND on CPU;
  otherwise (and for `flash`) the TPU splash kernel, or tokamax's with
  `use_tokamax_splash`.  `src/maxtext/experimental/mpmd/jax/deepseek/config.py` omits the CPU clause.
  """
  kernel = cfg.attention
  platform = platform or jax.default_backend()
  if kernel == "dot_product":
    return "dot_product"
  if kernel in ("autoselected", "flash"):
    if kernel == "autoselected" and (int(cfg.max_target_length) < 128 or platform == "cpu"):
      return "dot_product"
    return "tokamax_splash" if getattr(cfg, "use_tokamax_splash", False) else "splash"
  raise ValueError(f"unsupported attention kernel for this port: {kernel}")


def _refuse(cond, what):
  if cond:
    raise NotImplementedError(f"Qwen3.5 port: {what}")


def maxtext_gdn_fields(cfg) -> dict:
  """The GDN settings MaxText's options `packing` and `gdn_chunk_inverse` of the pyconfig `cfg` give
  (`Qwen35Config.gdn_packing`, `gdn_chunk_inverse`)."""
  return dict(gdn_packing=bool(cfg.packing), gdn_chunk_inverse=str(cfg.gdn_chunk_inverse))


def from_maxtext(cfg, platform: str | None = None, *, norm_numerics: str = Qwen35Config.norm_numerics) -> Qwen35Config:
  """Builds a Qwen35Config from an initialized MaxText pyconfig.

  Raises on the MaxText features that would change the math and that this port
  does not implement, instead of training a different model without a word.

  `norm_numerics` is a property of the MaxText REVISION, not of its pyconfig,
  so it cannot be read off `cfg`: the default is MaxText's since 2026-09-10,
  this MaxText's; pass 'bf16' for an earlier one (`Qwen35Config.norm_numerics`).
  `qwen35_maxtext_parity_test` reads it off the MaxText it imports.  The GDN's
  settings MaxText's options give it reads off `cfg` (`maxtext_gdn_fields`).
  """
  block = getattr(cfg.decoder_block, "value", cfg.decoder_block)
  _refuse(str(block) != "qwen3_5", f"decoder_block={block!r}, expected qwen3_5")
  # The dense Qwen3.5 variants (num_experts=1, e.g. qwen3.5-9b): MaxText builds
  # an MlpBlock of width mlp_dim there, and the port has only the sparse MoE.
  _refuse(
      int(getattr(cfg, "num_experts", 1)) <= 1,
      f'num_experts={getattr(cfg, "num_experts", 1)} (the dense Qwen3.5 MLP, e.g. qwen3.5-9b: '
      "MaxText builds an MlpBlock of width mlp_dim)",
  )
  _refuse(
      getattr(cfg, "routed_score_func", "") not in ("", None),
      f"routed_score_func={cfg.routed_score_func!r} (Qwen routes on raw logits)",
  )
  _refuse(getattr(cfg, "routed_bias", False), "routed_bias=true")
  _refuse(float(getattr(cfg, "routed_scaling_factor", 1.0)) != 1.0, "routed_scaling_factor != 1")
  _refuse(getattr(cfg, "logits_via_embedding", False), "logits_via_embedding=true (tied head)")
  _refuse(getattr(cfg, "logits_dot_in_fp32", False), "logits_dot_in_fp32=true")
  # use_qk_norm=true makes MaxText's qwen3-hybrid attention build PLAIN RMSNorms
  # (ones init, no +1) for its q/k norms (attentions.py); the port applies the
  # zero-centred (1 + scale) norm the Qwen3.5 checkpoints are trained with.
  _refuse(getattr(cfg, "use_qk_norm", False), "use_qk_norm=true (plain q/k RMSNorms instead of the zero-centred ones)")
  # The dense MoE (sparse_matmul=false) drops tokens over expert capacity when
  # capacity_factor > 0; the port never drops.  MaxText's sparse_matmul path
  # ignores capacity_factor (layers/moe.py reads it only in dense_matmul and its
  # masks), so ragged + capacity_factor is MaxText's own no-drop math.
  _refuse(
      float(getattr(cfg, "capacity_factor", -1.0)) > 0 and not cfg.sparse_matmul,
      "capacity_factor > 0 with sparse_matmul=false (token dropping)",
  )
  # mlp_bias adds wi_0/wi_1/wo biases to the routed experts, leaves the port has
  # no use for: the converted tree would carry them and the forward ignore them.
  _refuse(getattr(cfg, "mlp_bias", False), "mlp_bias=true (expert biases)")
  # MaxText's trainer applies dropout only with enable_dropout (the Qwen3.5 ymls
  # set it false); the port has none.
  _refuse(
      float(getattr(cfg, "dropout_rate", 0.0)) > 0 and getattr(cfg, "enable_dropout", True),
      "dropout_rate > 0 with enable_dropout=true (the port has no dropout)",
  )
  _refuse(getattr(cfg, "quantization", "") not in ("", None), f'quantization={getattr(cfg, "quantization", "")!r}')
  # Neither adds a parameter, so `convert.check_params` cannot catch them:
  # activations_in_float32 runs the shared expert's (MlpBlock) and the dense
  # MoE's activations in fp32; use_untrainable_positional_embedding adds
  # sinusoidal position embeddings to the embedded tokens.
  _refuse(getattr(cfg, "activations_in_float32", False), "activations_in_float32=true (fp32 MLP / expert activations)")
  _refuse(
      getattr(cfg, "use_untrainable_positional_embedding", False),
      "use_untrainable_positional_embedding=true (sinusoidal position embeddings)",
  )
  # These add no parameter either: the rotary embeddings MaxText's gated attention picks ahead of
  # PartialRotaryEmbedding under use_mrope=false (attentions.Attention.init_rotary_embedding), LLaMA-3.1's (rope_type
  # llama3.1, or a llama3.1 model_name) and YaRN (rope_type yarn), both over all of head_dim, where the port has
  # only the partial RoPE.  Under use_mrope=true the mRoPE comes first and rope_type is unread.
  rope_type = getattr(cfg, "rope_type", None) or "default"
  rope_type = str(getattr(rope_type, "value", rope_type)).lower()  # str() of a (str, Enum) RopeType: 'RopeType.X'
  model_name = str(getattr(cfg, "model_name", "") or "")
  _refuse(
      not cfg.use_mrope and (rope_type != "default" or model_name.startswith("llama3.1")),
      f"use_mrope=false with rope_type={rope_type!r}, model_name={model_name!r} (MaxText rotates with its "
      "LLaMA-3.1 or YaRN RoPE there, not PartialRotaryEmbedding)",
  )
  # Before 2026-09-10 MaxText built shared_expert_gate without a weight_dtype,
  # so that kernel stayed fp32 while every other leaf followed weight_dtype; the
  # port types every leaf alike (`model.param_avals`), as later revisions do.
  # fp32 (base.yml and the Qwen3.5 configs) is the one setting both agree on.
  _refuse(
      jnp.dtype(getattr(cfg, "weight_dtype", jnp.float32)) != jnp.float32,
      f"weight_dtype={jnp.dtype(cfg.weight_dtype).name} (MaxText revisions disagree on "
      "the shared_expert_gate kernel dtype; only float32 is the same in all)",
  )
  # MaxText options that compute the GDN in other numerics than the port's JAX GDN: the Tokamax GDN kernel, its two-pass
  # bf16 refinement, and FP8 in_proj_qkvz / out_proj matmuls.  None adds a parameter, so the tree check cannot catch them.
  _refuse(getattr(cfg, "use_tokamax_gdn", False), "use_tokamax_gdn=true (the Tokamax GDN kernel's own rounding)")
  _refuse(getattr(cfg, "gdn_two_pass_bf16", False), "gdn_two_pass_bf16=true (the Tokamax GDN kernel's refinement)")
  _refuse(getattr(cfg, "gdn_fp8_projections", False), "gdn_fp8_projections=true (FP8 GDN projections)")
  gdn = maxtext_gdn_fields(cfg)
  sa = {
      k: getattr(cfg, k)
      for k in ("sa_block_q_dkv", "sa_block_kv_dkv", "sa_block_kv_dkv_compute")
      if getattr(cfg, k, None) is not None
  }
  return Qwen35Config(
      vocab_size=cfg.vocab_size,
      emb_dim=cfg.emb_dim,
      num_decoder_layers=cfg.num_decoder_layers,
      inhomogeneous_layer_cycle_interval=cfg.inhomogeneous_layer_cycle_interval,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      partial_rotary_factor=cfg.partial_rotary_factor,
      rope_min_timescale=cfg.rope_min_timescale,
      rope_max_timescale=cfg.rope_max_timescale,
      # use_mrope=false (Qwen3.8-2.4T's yml) with the default rope_type (anything else is refused above):
      # PartialRotaryEmbedding (attention.partial_rope); the sections are then unread (and checked by neither
      # MaxText nor Qwen35Config)
      use_mrope=bool(cfg.use_mrope),
      mrope_section=tuple(cfg.mrope_section),
      gdn_num_key_heads=cfg.gdn_num_key_heads,
      gdn_num_value_heads=cfg.gdn_num_value_heads,
      gdn_key_head_dim=cfg.gdn_key_head_dim,
      gdn_value_head_dim=cfg.gdn_value_head_dim,
      gdn_conv_kernel_dim=cfg.gdn_conv_kernel_dim,
      gdn_chunk_size=cfg.gdn_chunk_size,
      use_qk_norm_in_gdn=bool(cfg.use_qk_norm_in_gdn),
      **gdn,
      moe_mlp_dim=cfg.moe_mlp_dim,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      shared_experts=1 if cfg.shared_experts else 0,
      norm_topk_prob=bool(cfg.norm_topk_prob),
      mlp_activations=tuple(cfg.mlp_activations),
      moe_impl="ragged" if cfg.sparse_matmul else "dense",
      gmm_fp32_accum=bool(cfg.megablox or cfg.use_tokamax_gmm),
      moe_use_direct_token_gather=bool(cfg.moe_use_direct_token_gather),
      float32_weight_sum=bool(cfg.float32_weight_sum),
      float32_gate_logits=bool(cfg.float32_gate_logits),
      load_balance_loss_weight=float(cfg.load_balance_loss_weight),
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      matmul_precision=cfg.matmul_precision,
      cast_logits_to_fp32=bool(cfg.cast_logits_to_fp32),
      norm_numerics=norm_numerics,
      attention_kernel=resolve_attention_kernel(cfg, platform),
      sa_block_q=cfg.sa_block_q,
      sa_block_kv=cfg.sa_block_kv,
      sa_block_kv_compute=cfg.sa_block_kv_compute,
      sa_use_base2_exp=bool(cfg.sa_use_base2_exp),
      max_target_length=cfg.max_target_length,
      scan_layers=bool(cfg.scan_layers),
      remat_policy=cfg.remat_policy,
      **sa,
  )
