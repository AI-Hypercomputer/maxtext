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

"""DeepSeek-V3 sparse layers on one MPMD stage: FSDP x EP on a 2-D sub-mesh.

A pipeline stage owns a sub-mesh with axes `('fsdp', 'expert')`.  Weights live
in two layouts:

  storage  fp32 master weights, sharded like MaxText's `pipeline-large-moe`
           rules: non-routed weights shard their embed axis over
           ('fsdp', 'expert') (the LoRA ranks over 'fsdp'); routed experts
           shard the expert axis over 'expert' and the embed axis over 'fsdp'.
  compute  bf16, all-gathered once per step ahead of every microbatch:
           non-routed weights are replicated on the stage, routed experts keep
           [E / EP] experts per device.  This is MaxText's layout after its
           pipeline's per-repeat FSDP all-gather (`pipeline_fsdp_ag_per_repeat`),
           cast to bf16 *before* the gather instead of after it.

Every per-microbatch program is one `shard_map` over the stage mesh, so the
only collectives in the hot loop are ring-of-experts' token all-gather and
output reduce-scatter over 'expert' (MaxText `use_ring_of_experts=true`
`use_ragged_sort=true`, `RoutedMoE.sparse_matmul`).  Weight gradients are
accumulated per device, unreduced, in a layout with a leading "partial" axis,
and reduce-scattered back to the storage layout once per step.  Under
`shard_mode='explicit'` the stage meshes' axes are Explicit and the types say
so instead: the gathered weights are `reduced` over the axes their gradients
are partial over, the accumulators `unreduced` over them (StageConfig).

The expert and attention kernels are MaxText's own -- `ring_ragged_sort` /
`ring_ragged_unsort` (SparseCore ragged gather / gather-reduce), the megablox
`gmm`, and tokamax splash with the fused backward -- called with the argument
values MaxText derives from the reference flags, so both sides of the
comparison execute the same kernels.  MaxText is imported only where one of
its kernels runs: unsplit or under batch_split_impl 'native', with
`gmm_impl='ragged_dot'` and `use_ragged_sort=False` the stage programs are
pure JAX (plus tokamax for the tokamax_splash attention kernel); a batch
split on Lineage's layers (DeepSeek-V3's default, below) imports MaxText's
Lineage whatever those knobs are.

A second layer family runs on the same machinery: the Qwen3.5 family
(Qwen3.5-397B-A17B, Qwen3.8-2.4T-A95B; MaxText `decoder_block: qwen3_5`), when
`StageConfig.model` is a `maxtext.experimental.mpmd.jax.qwen35.config.Qwen35Config`.  Its layers
are MoE layers too, with one of two mixers in front of the experts: Gated
DeltaNet linear attention, or gated full attention on every
`inhomogeneous_layer_cycle_interval`-th layer.  The mixers and norms are the
`maxtext.experimental.mpmd.jax.qwen35` port's (as DeepSeek's are `maxtext.experimental.mpmd.jax.deepseek`'s), the routed
experts are this module's ring of experts with Qwen's router (top-k of the raw
logits, softmax over the k, renormalised), and the shared expert has its
sigmoid gate.  A chunk records each layer's kind (`ChunkSpec.kinds`), and its
weights are one subtree a layer, each leaf with a leading layer axis of one
(`chunk_shapes`): DeepSeek's [n, ...] stacks cost a Qwen3.8 chunk's backward
a copy of every layer's weight slice and every layer's weight gradient live at
once (a 2-layer chunk: 33.2 GiB of temporaries against 24.5).  Every leaf keeps
one leading layer axis, so the gathers, reduce-scatters, accumulators and plans
are the same code for both families.

Batch split (StageConfig.batch_split) runs a DeepSeek-V3 chunk's decoder
layers on MaxText's Lineage layers by default (ds_ep_lineage; the chunk's
programs are then global-view jits, ChunkPrograms), or on this module's
`split_layers` over the kernels above (batch_split_impl 'native'), where
Lineage's TPU kernels do not run (GPU, CPU: ds_ep_pipeline refuses 'native'
for DeepSeek-V3 on TPU).  Lineage builds DeepSeek-V3's layers only, so a
Qwen3.5-family chunk always batch-splits on `split_layers`, its mixer in
attention's place: 'native' is that family's default and only impl.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import math
import numbers
import threading
import warnings

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax.deepseek import attention
from maxtext.experimental.mpmd.jax.deepseek import layers
from maxtext.experimental.mpmd.jax.deepseek import rope
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import stage_ad
from maxtext.experimental.mpmd.jax.qwen35 import attention as q_attention
from maxtext.experimental.mpmd.jax.qwen35 import gdn as q_gdn
from maxtext.experimental.mpmd.jax.qwen35 import layers as q_layers
from maxtext.experimental.mpmd.jax.qwen35.config import Qwen35Config

FE = ("fsdp", "expert")
EP_AXIS = "expert"
GMM_IMPLS = ("megablox", "ragged_dot", "tokamax")
# The grouped-matmul backends gmm_impl 'tokamax' runs (StageConfig.tokamax_gmm):
#   'mosaic' -- Mosaic GPU for the forward and dlhs products, which skips the rows past the groups: on sm100
#     gpu_kernels.grouped_matmul (tokamax's Blackwell kernel reading the weight as it lies, tuned tiles), elsewhere
#     (and for dtypes it does not take) tokamax's own; tokamax has no Mosaic dW kernel on sm80 / sm100, so dW runs on
#     Triton -- but a dW with a starting value (in-place accumulation, an fp32 accumulator's too, or mb0's under
#     split_dw 'acc': _ragged_dw) on sm100 on gpu_kernels.ragged_dw's tcgen05 kernel where it takes the shapes.
#   'tokamax_mosaic' -- tokamax's Mosaic kernels everywhere (its heuristic tiles; its forward transposes the weight),
#     dW on Triton: none of gpu_kernels' Mosaic kernels (a dW with a starting value on ragged_dw's Triton one).
#   'triton' -- every product on Triton (sm80 and later; a dW with a starting value on ragged_dw's).
#   'xla' -- lax throughout: tokamax's own lax.ragged_dot, and ragged_dw's lax form for a dW with a starting value
#     (CPU tests).
# On a GB200 at DeepSeek-V3's EP 4 shapes (64 local experts, 32768 rows routed to them in a buffer of 65536) the
# forward products run 0.82 ms (1150 TFLOP/s) on 'mosaic', 1.59 ms on tokamax's Mosaic (0.66 of it the weight's
# transposed copy), 1.8 / 3.9 ms on Triton (k=7168 / n=2048, k=2048 / n=7168); dlhs 0.85 / 1.0 ms; Triton's dW 1.2 ms
# (785 TFLOP/s).  Qwen3.8's (128 experts, 40960 rows): forward 1.25-1.32 ms against tokamax's 2.7-3.0.
TOKAMAX_GMM = ("mosaic", "tokamax_mosaic", "triton", "xla")
# Stage programs take 'ppermute' / 'xla'; 'auto' (EP) is a pipeline-level
# choice that ds_ep_pipeline.Pipeline resolves per stage.
FSDP_COLLECTIVES = ("ppermute", "xla")
EP_COLLECTIVES = ("xla", "ppermute", "auto")
SHARD_MODES = ("auto", "explicit")
RS_AXES = ("joint", "each")
GATHER_AXES = ("joint", "each")
SPLIT_DW = ("acc", "sum")
EXPERT_WEIGHTS = ("stacked", "sliced")
# Whose code a batch-split chunk's decoder layers run (StageConfig.batch_split_impl): 'lineage', MaxText's Lineage
# layers (ds_ep_lineage), DeepSeek-V3's only; 'native', this module's own split_layers over its layer ports and
# MaxText's ring-of-experts / megablox kernels -- DeepSeek-V3's off TPU (ds_ep_pipeline.native_split_refusal), or
# the Qwen3.5 family's, its only batch split.
BATCH_SPLIT_IMPLS = ("lineage", "native")
# Where the routing weights multiply the routed experts' output (StageConfig.route_weights).
ROUTE_WEIGHTS = ("combine", "experts")
# A layer's routed-expert weight leaves (moe_experts' stacked).
EXPERT_LEAVES = ("wi_0", "wi_1", "wo")


def _is_spec(x):
  return isinstance(x, P)


# --------------------------------------------------------- megablox tilings
# A tiling is a megablox 9-tuple (fwd tm,tk,tn, dlhs tm,tk,tn, drhs tm,tk,tn),
# as MaxText builds it from the wi_tile_* / wo_tile_* flags (`RoutedMoE.sparse_matmul`'s
# get_wi_gmm_params / get_wo_gmm_params).
# The three kernels of one weight tile different k dims (D = emb_dim,
# F = moe_mlp_dim; the gmm tk is a contraction tile, the tgmm tk tiles dW's k):
#
#          fwd gmm   dlhs gmm (_dx)   drhs tgmm (_dw)
#   wi     k = D     k = F            k = D
#   wo     k = F     k = D            k = F
#
# megablox does not clamp a tk larger than its k: it rounds the tile count up
# (`_calculate_irregular_num_tiles`) and masks the tail, but the MXU still runs
# the dot on the whole [tm, tk] block -- tk=7168 on k=2048 is 3.5x the work,
# for bit-identical outputs.  Padded tiles, one (512, D, 512) x3 for both
# weights, do that in four of the nine kernels (gmm_wo, gmm_wi_{0,1}_dx,
# gmm_wo_dw).  In a TPU7x-8 capture (TPU:0 of the 2026-09-25 runs of
# DeepSeek-V3 layers against MaxText's SPMD pipeline), where all nine kernels
# do the same FLOPs: gmm_wo 6.42 ms vs gmm_wi_0 1.90 ms, gmm_wi_*_dx 6.50 vs
# gmm_wo_dx 1.88, gmm_wo_dw 7.86 vs gmm_wi_*_dw 2.60 -- ~376 ms/step of
# padding; the MaxText baseline, run with the same 7168 tiles, pads ~377
# ms/step.  On v6e the wo forward takes 8.42 ms at tk=7168 and 5.85 ms at
# tk=2048.
#
#   clamped    tk = the k it tiles, tm = tn = 512.  The default.
#   v6e        swept on TPU v6e-4 at D=7168, F=2048.  At PP2 x FSDP1 x EP2, 4
#              layers, 16 experts, gbs 16, M 8: 1607 ms/step with the padded
#              tiles -> 1328 ms/step (clamped: 1447; MaxText on the same kernels
#              and tiles: 1.91 -> 1.64 s/step, clamped 1.75).  Fixed numbers:
#              other shapes may pad.
TILES = {
    "clamped": lambda cfg: (
        (512, cfg.emb_dim, 512, 512, cfg.moe_mlp_dim, 512, 512, cfg.emb_dim, 512),
        (512, cfg.moe_mlp_dim, 512, 512, cfg.emb_dim, 512, 512, cfg.moe_mlp_dim, 512),
    ),
    "v6e": lambda cfg: (
        (1024, 3584, 2048, 1024, 2048, 2048, 256, 3584, 2048),
        (1024, 2048, 2048, 1024, 3584, 2048, 1024, 2048, 2048),
    ),
}


def split_acc_wi_tiling(wi):
  """The wi tiling a batch-split chunk (impl 'native') runs under split_dw
  'acc' where none is given: the drhs (dW) tm capped at 256.  Its dW kernel
  then also holds existing_out, a second [tk, tn] window double-buffered like
  the output: at the clamped (512, D, 512) and DeepSeek-V3's D that is 69.1 of
  TPU7x's 63.9 MiB of VMEM, at tm 256 it fits (megablox tgmm compiled for one
  TPU7x device, libtpu 0.0.46; tk D/2 or tn 256 fit too).  wo's dW kernel (tk
  F) fits either way."""
  wi = tuple(wi)
  return wi[:6] + (min(wi[6], 256),) + wi[7:]


# How StageConfig's warning about such a tiling starts (to filter it by).
TILING_WARNING = "megablox tile larger than the k it tiles"


def tiling_problems(cfg: DeepSeekConfig, wi_tiling, wo_tiling):
  """Every tk larger than the k it tiles (see TILES), as readable strings."""
  D, F = cfg.emb_dim, cfg.moe_mlp_dim
  out = []
  for w, tiling, ks in (("wi_{0,1}", wi_tiling, (D, F, D)), ("wo", wo_tiling, (F, D, F))):
    for kernel, i, k in zip(("", "_dx", "_dw"), (1, 4, 7), ks):
      if tiling[i] > k:
        out.append(f"gmm_{w}{kernel}: tk={tiling[i]} > k={k} ({tiling[i] / k:.2g}x the MXU work)")
  return out


def _tiling9(name, t):
  t = tuple(t)
  if len(t) != 9 or not all(isinstance(x, numbers.Integral) and x > 0 for x in t):
    raise ValueError(f"{name} must be 9 positive ints (fwd tm,tk,tn, dlhs tm,tk,tn, drhs tm,tk,tn), " f"got {t!r}")
  return tuple(int(x) for x in t)


# ---------------------------------------------------------------- the config
@dataclasses.dataclass(frozen=True)
class StageConfig:
  """Model + the knobs that belong to the stage programs, not the model.

  `model` is a DeepSeek-V3 config or a Qwen3.5-family one (`is_qwen`)."""

  model: DeepSeekConfig | Qwen35Config
  fsdp: int = 2
  ep: int = 2
  # megablox 9-tuples (see TILES).  None: TILES['clamped'] for `model`
  # (wi's under batch_split_impl 'native' with split_dw 'acc':
  # split_acc_wi_tiling; Lineage's layers run no megablox kernel), resolved
  # in __post_init__ (so dataclasses.replace(..., model=...) keeps the
  # tiling of the old model).
  wi_tiling: tuple | None = None
  wo_tiling: tuple | None = None
  # A tk larger than the k it tiles pads the MXU work (see TILES): warn, or
  # raise ValueError when True.
  strict_tiling: bool = False
  # The routed experts' grouped matmuls: 'megablox' (MaxText's kernels, with
  # the tilings above) or 'ragged_dot' (pure JAX, jax.lax.ragged_dot; the
  # tilings are unused).  Same math, same group_offset semantics.
  gmm_impl: str = "megablox"
  # gmm_impl 'tokamax': the backend of the forward and dlhs products (TOKAMAX_GMM); the dW product runs on
  # Triton unless this is 'xla' -- but under 'mosaic' on sm100 one with a starting value (inplace_dw and
  # inplace_dw_any_dtype, or mb0's under split_dw 'acc') on gpu_kernels.ragged_dw's tcgen05 kernel (_ragged_dw).
  # tokamax has no group_offset, so gmm_impl 'tokamax' runs the ring's experts on a compact buffer of this device's
  # rows only (`_local_plan`): no ragged sort, and a row of another shard's experts is never in a kernel's input.
  tokamax_gmm: str = "mosaic"
  # gmm_impl 'tokamax': accumulate the routed experts' weight gradients into the step's accumulator inside the dW
  # kernel (gpu_kernels.ragged_dw; ChunkPrograms' inplace_dw says where it applies) rather than writing each dW to
  # a buffer of its own and adding it in a pass of its own.
  inplace_dw: bool = True
  # ... also into an accumulator of a dtype other than the compute dtype (an fp32 one: Megatron's main_grad), on
  # every chunk but the last: the accumulated value leaves through the start's cotangent (tokamax_gmm's dw32), the
  # starts being primals of the backward's VJP (the saved-residual forward's, or bwd_core_recompute's).  Off by
  # default.  On sm_100 under tokamax_gmm 'mosaic' the kernel is gpu_kernels.ragged_dw's tcgen05 one (DW_MOSAIC_F32:
  # 1.77 ms at DeepSeek-V3's EP 8 shapes, a 2-node B200 step 1.6% faster than the dW plus an add); elsewhere, and
  # under 'tokamax_mosaic' / 'triton', its Triton one, slower than tokamax's dW plus an add pass (A100, dsv3-mini
  # expert shapes: 5.66 ms against 4.21 + ~1.5; a step 1% slower); under 'xla' lax's form (_ragged_dw).
  inplace_dw_any_dtype: bool = False
  # Zero the compact buffer's rows past its groups (_masked): after the dispatch, every grouped matmul's output and
  # each gradient entering one.  Every reader of those rows is bounded by the group sizes (the grouped-matmul
  # kernels) or indexes valid rows only (the combine's and the dispatch gradient's gathers by `pos`), so the masks
  # change no result; False skips their buffer-wide passes (the kernels leave the rows past the groups unwritten,
  # and nothing reads them).  At EP 1 with a dropless buffer every row is valid and they are skipped regardless.
  buffer_masks: bool = True
  # The per-token sums over a token's K choices -- the combine's weighted sum of its rows, and the dispatch's
  # gradient (the sum of its rows' gradients): 'xla', a fill-gather of T x K rows reduced over K (at EP 8 seven choices
  # in eight have no row here, and it still walks them), or 'triton', gpu_kernels.gather_sum, which loads only the
  # rows that exist (GPU; elsewhere it is the XLA sum).
  combine_impl: str = "xla"
  # Where the routing weights multiply the routed experts' output (ROUTE_WEIGHTS).  'combine': the combine's weighted
  # sum of a token's rows (MaxText's), whose backward reads the expert output rows for the weights' gradient, so a
  # saved-residual forward keeps them: [rows, emb], 896 MiB a DeepSeek-V3 layer at one sequence a device, EP 8 and
  # ragged_buffer_factor 2.  'experts': each row of the gated activation, before the wo product (Megatron-LM's, which
  # scales the activation by the probs), and the combine an unweighted sum -- a weight's gradient is then its row of
  # the activation dotted with that row's gradient, [rows, moe_mlp] arrays the wo product's backward has anyway, and no
  # expert output row is kept.  The same sums, rounded otherwise: the weighted activation is rounded to the compute
  # dtype before wo, as Megatron-LM rounds it.  It needs each buffer row's (token, choice) pair: gmm_impl 'tokamax'
  # (its compact buffer) or the argsort dispatch (use_ragged_sort False); MaxText's ring_ragged_sort keeps them inside
  # its kernels.
  route_weights: str = "combine"
  # Under lean residuals (saved residuals), whether the routed experts' grouped matmuls keep what their lhs is
  # recomputed from instead of the lhs (`Recompute`): the gathered tokens [T, emb] instead of the buffer rows
  # [rows, emb], and the gated product's inputs instead of the product.  True keeps the fewest bytes (a DeepSeek-V3
  # layer at one sequence a device, EP 8, ragged_buffer_factor 2: 0.7 GiB less) for a row gather and an elementwise
  # product more in each backward; False keeps the lhs, where the memory is there.
  lean_experts: bool = True
  # A DeepSeek chunk's layers' weights, its MoE layers' and its dense layers' alike: 'stacked' ([n, ...] leaves,
  # TPU's: megablox reads a layer's experts in place in the stack) or 'per_layer' (one subtree a layer, a layer axis
  # of one, as a Qwen chunk's).  Per layer, no layer's gradient is stacked into the accumulator in every backward (on
  # GB200, three [n, E / EP, D, F] concatenations a MoE chunk's backward, ~9% of a stage's step; chunk 0's three
  # DeepSeek-V3 dense layers' [3, D, mlp] ones ~0.6 s of the 64-B200 recipe's stage 0 step), each layer's dW
  # accumulating in its own GEMM instead; each accumulator block is a whole buffer, which in-place dW accumulation
  # needs; and a layer's weights are whole buffers, not slices of a stack, which the lean residuals' checkpoints
  # (layers.rms_norm_read) would copy through their optimization barriers in every backward.  Qwen3.5-family chunks
  # are per layer either way.
  layer_tree: str = "stacked"
  use_ragged_sort: bool = True
  random_routing: bool = True
  iota_embed: bool = True
  # dtype of the once-per-step gradient reduce-scatter.  (MaxText reduce-scatters
  # its compute copies' gradients whatever its grad_dtype: bf16, but for the
  # fp32 ones' -- with fp32 weights, an fp32 router's under float32_gate_logits
  # and a Qwen3.5-family model's norm scales under norm_numerics 'fp32'
  # (leaf_dtypes).)
  rs_dtype: jnp.dtype = jnp.bfloat16
  # The dtype the reduced gradient enters AdamW (and the clip) in, cast from
  # rs_dtype at the top of StagePrograms' update: MaxText grad_dtype, which its
  # trainer applies after the reduction -- rs_dtype bf16 with grad_dtype fp32
  # is MaxText's weight_dtype=float32 grad_dtype=float32, but for the routers'
  # kernels and a Qwen model's norm scales: MaxText reduces the gradients of
  # their fp32 compute copies in fp32, this pipeline in rs_dtype.  None: under
  # adam_math 'optax' each weight's dtype (MaxText's gradients take their
  # parameter's, so with fp32 weights that is the cast above), under 'fp32'
  # rs_dtype, no cast.  adam_math 'fp32' computes in fp32 whatever this is
  # (float32 changes nothing there; bfloat16 rounds the gradient first).  The
  # global norm stays the reduce-scatter's output's (its fp32 partial sums)
  # and the count leaves (count_leaves) keep their fp32.
  # MaxText casts fp32 gradients only: with bf16 weights its gradients stay
  # bf16 whatever its grad_dtype (keep None).
  grad_dtype: jnp.dtype | None = None
  # Accumulate weight gradients across microbatches in this dtype.
  acc_dtype: jnp.dtype = jnp.float32
  # Every chunk but the last saves only each layer's INPUT and recomputes the
  # layer inside its backward program (MaxText's remat 'full'): its
  # activations would otherwise have to outlive the forward program.  The
  # last chunk runs forward and backward in ONE program, so recomputing there
  # buys no memory: False skips it (same gradients, less compute), True keeps
  # MaxText's remat-everything FLOP count for a like-for-like comparison.
  remat_fused_last: bool = False
  # Every chunk but the last keeps its backward's residuals instead (Megatron-LM's default, no recompute): its
  # forward program runs the layers under jax.vjp and returns the VJP's residuals -- every array the backward
  # reads but the chunk's input and its compute weights, which the backward takes as arguments anyway -- and its
  # backward program evaluates that VJP on them.  No forward is computed twice (the structural recompute above
  # is about a quarter of a step's FLOPs), for the residuals' memory until the backward (ChunkPrograms.res_avals).
  save_residuals: bool = False
  # How the once-per-step FSDP all-gather / reduce-scatter are expressed.
  # 'ppermute': as collective-permutes over each size-2 mesh axis, which TPU
  #   runs as async DMAs that overlap compute in the same program on EVERY
  #   sub-mesh.  (libtpu only offloads all-gather / reduce-scatter to
  #   SparseCore -- the other way they become async -- when the program's
  #   chips are the whole slice or a solid rectangle without chip (0,0,0): it
  #   compiles any other device set against the whole slice's topology, and
  #   skips the offload when the program uses fewer devices than that ("does
  #   not use all devices"), so on a sub-mesh holding chip (0,0) they run
  #   synchronously and serialize with compute.  See sparsecore_offload_available.)
  # 'xla': lax.all_gather / psum_scatter.
  fsdp_collectives: str = "ppermute"
  # Ring-of-experts token all-gather / output reduce-scatter over 'expert':
  # 'xla' (lax collectives; SparseCore-offloaded and async where libtpu allows)
  # or 'ppermute' (async collective-permutes, for sub-meshes without offload).
  ep_collectives: str = "xla"
  # When weight gradients are reduced over FSDP:
  # 'step'        accumulate per-device partials unreduced across microbatches
  #               and reduce-scatter once per step (folded into later programs);
  # 'microbatch'  reduce-scatter every microbatch's gradient inside its own
  #               backward program and accumulate the sharded result -- the
  #               cadence MaxText's pipeline_fsdp_ag_per_repeat compiles to.
  grad_reduce: str = "step"
  # The AdamW update's arithmetic.  'fp32': in fp32 whatever the storage
  # dtype, stored back in it.  'optax': optax's chain as MaxText runs it --
  # clip_by_global_norm, scale_by_adam, add_decayed_weights,
  # scale_by_learning_rate, apply_updates -- every op in its leaf's dtype: with
  # bf16 weights the moments' decay, the bias correction and the learning rate
  # are bf16 (maxtext.experimental.mpmd.jax.deepseek.train holds that chain bit-exact against
  # MaxText).  For parity runs against MaxText with bf16 weights and gradients
  # (mu is always bf16 here; nu keeps its storage dtype); the global norm
  # still comes from this pipeline's fp32 partial sums, where MaxText's optax
  # reduces it in bf16 -- so the update is MaxText's bit for bit only while
  # the norm stays below the clip (or both norms round to one bf16 value).
  adam_math: str = "fp32"
  # The global-norm clip's threshold, MaxText's gradient_clipping_threshold; 0
  # does not clip (a parity run with bf16 gradients turns it off on both sides,
  # since MaxText rounds its norm to bf16 and this pipeline does not).
  grad_clip: float = 1.0
  # The ring of experts' dispatch buffer (MaxText ragged_buffer_factor,
  # RoutedMoE.get_ragged_buffer_size): <= 0 the worst case, every token of the
  # EP group x top-k rows a device (dropless; [EP x tokens x top-k, emb] bf16
  # is 7.5 GB at EP 16 and one 4k sequence a device); > 0 that many times the
  # balanced load, a device's own tokens x top-k.  A shard routed more than its
  # buffer drops the excess, as MaxText's truncated buffer does
  # (`_truncate_matrix` of the shard's group sizes).  Any factor: megablox gets
  # the buffer padded to its row tile (`gmm_row_tile`).  Needs use_ragged_sort
  # (the ring's SparseCore sort holds the buffer).
  ragged_buffer_factor: float = -1.0
  # Context parallelism: a microbatch's sequences cut into `cp` chunks, one
  # row of seq / cp tokens a device.  Rows shard fsdp-major over FE, so the cp
  # rows of a sequence sit on cp consecutive devices of one 'expert' ring:
  # device e holds chunk e % cp (`cp_chunk`).  Attention collects the other
  # chunks' kv latents from those devices (`cp_kv_context`); everything else
  # is per token.  1, 2 or 4, dividing ep.
  cp: int = 1
  # How the stage meshes type the step's weight copies and gradient
  # accumulators (jax's sharding in types):
  # 'auto'      Auto mesh axes, nothing typed; an accumulator carries a leading
  #             axis over the mesh axes its per-device partial sums are still
  #             to be reduced over (chunk_gacc_specs).
  # 'explicit'  Explicit mesh axes.  The gathered compute weights are typed
  #             P(*compute spec, reduced=<partial axes>) -- one copy on every
  #             device of those axes -- and an accumulator P(*compute spec,
  #             unreduced=<partial axes>): each device's partial sum, no leading
  #             axis, the gradient their sum over those axes.  The types sit on
  #             the programs' arguments and outputs; their bodies, collectives
  #             and folding are 'auto''s (check_vma=False).  Needs
  #             grad_reduce='step'.
  shard_mode: str = "auto"
  # How fsdp_collectives 'xla' reduce-scatters a gradient leaf stored over
  # both mesh axes (P(.., ('fsdp', 'expert'), ..): wq_a, wkv_a, out, the
  # shared experts, the embedding and the head): 'joint', one psum_scatter
  # over both; 'each', one a mesh axis, 'fsdp' first.  Compiled for a v5p
  # slice as one stage (v5p-128: fsdp 4 x expert 16; v5p-64: 2 x 16; libtpu
  # 0.0.46, the v5p recipe's flags) the joint one is a full-size all-reduce
  # over the stage and a slice -- twice a reduce-scatter's bytes, through a
  # full-size buffer -- where 'each' is two reduce-scatters, both on the
  # SparseCore; but for the embedding and the head, whose per-axis pair
  # libtpu runs on the TensorCore, synchronously: those keep the joint one
  # (_rs_joint_only).  Not timed on v5p: the joint all-reduce may use every
  # link of the torus, the 'fsdp' step of 'each' runs over a ring of 4
  # chips.  On v6e stages (2x2, 4x4) the joint one is one reduce-scatter, and
  # 'each' only adds a collective (on the v6e-4 four of the five DS-V3 layer
  # leaves took 9-63% longer, wkv_a 2.1 -> 0.9 ms); on a TPU7x-128 stage
  # (8 x 16) it is one too, but for wkv_a's.  So opt-in (ds_ep_pipeline
  # --rs-axes; a 2 x v5p-128 study's runs with --rs-axes each, 2026-10-14).
  # 'each' rounds the first axis's sum to rs_dtype before the second's is
  # added: within 2 bf16 ulps of 'joint' (measured).  Leaves stored over one
  # axis, and 'ppermute' (axis by axis anyway), reduce alike either way.
  rs_axes: str = "joint"
  # The gather side's counterpart: how fsdp_collectives 'xla' all-gathers a
  # leaf stored over both mesh axes -- 'joint', one lax.all_gather over both;
  # 'each', one a mesh axis, the minor ('expert') first, tiled, so the second
  # gathers whole blocks of the first and the result is the joint gather's,
  # bit for bit (a gather is a copy).  A v5p-128 stage of a 2-slice program
  # (2 x v5p-128 PP over DCN, libtpu 0.0.46, the v5p recipe's flags) compiles the
  # joint 64-way gather to a zero buffer, a dynamic-update-slice and a
  # full-size all-reduce over the stage (libtpu's all-gather decomposer, which
  # runs on multi-slice compiles only): about twice the bytes of a gather,
  # through a buffer the size of the result (12.6 ms on the SparseCore for the
  # [7168, 129280] head, measured).  'each' compiles to two native SparseCore
  # all-gathers there, over 16 and 4 devices (AOT of the runs' programs), and
  # the 2 x v5p-128 A/B timed them: 5.3 / 5.1 ms off update_s0_ag0 (the R2 /
  # R1 runs of PinModel; 2026-10).  On one slice the joint gather is native
  # already, as in a 2-slice v6e program (NATIVE_JOINT_GATHER_KINDS), where the
  # 2 x v6e-128 A/B timed 'each' slower on R2v (update_s0_ag0 18.3 -> 28.8 ms)
  # and neutral on R1v.  Leaves stored over one axis, and 'ppermute' (axis by
  # axis anyway), gather alike either way.  Opt-in: ds_ep_pipeline
  # --gather-axes.
  gather_axes: str = "joint"
  # Batch split, Lineage's: a microbatch's rows on each device run as two
  # halves, mb0 and mb1, through a chunk's MoE layers in phases: one half's
  # EP all-gather or reduce-scatter beside the other half's attention or
  # experts, so each EP collective has compute to run behind.  The backward
  # is the VJP of the chunk's whole split forward, recomputed from the
  # chunk's input, so a chunk keeps one residual, its input.  Two
  # neighbouring microbatches of the unsplit pipeline make one microbatch
  # here: it needs an even number of rows a device, cp 1 and no stash mode
  # (ds_ep_pipeline --batch-split), and, as neither implementation keeps
  # expert counts or an aux loss, no routed-bias update, sequence-wise aux
  # loss or MTP.  A chunk without a MoE layer (the embedding's or the
  # head's alone, or dense layers only) has nothing to split and runs the
  # plain path (ChunkPrograms.split), but not on Lineage's layers, which
  # refuse it.
  batch_split: bool = False
  # Whose code runs a batch-split chunk's decoder layers (BATCH_SPLIT_IMPLS).
  # None, the default, is the model's, resolved in __post_init__ (so a built
  # config names its impl): 'lineage' for DeepSeek-V3, 'native' for the
  # Qwen3.5 family, whose layers Lineage does not build (`is_qwen`).  (A
  # dataclasses.replace(..., model=...) across the families keeps the old
  # model's impl: pass batch_split_impl=None with it.)
  # 'lineage'  MaxText's Lineage layers (src/maxtext/experimental/lineage, the
  #            MLPerf DeepSeek-V3 batch-split stack) through ds_ep_lineage:
  #            its MLA and splash attention, router, SparseCore dispatch and
  #            combine and gmmv2 expert kernels, chained over the chunk's MoE
  #            layers as its driver chains them (the dense layers first,
  #            Lineage's dense layer each), and its own backward.  Lineage's
  #            code is global-view code that reads layouts from sharding
  #            types, so the chunk's programs are jits at global view on
  #            Explicit stage meshes (ChunkPrograms: the embedding, the head
  #            and every gather / reduce-scatter folded in run in shard_maps of
  #            their own): it needs shard_mode 'explicit' (so grad_reduce
  #            'step'), top-k routing (Lineage draws no random experts), a
  #            dropless dispatch (ragged_buffer_factor <= 0) and a model
  #            Lineage builds (ds_ep_lineage.check), and its forwards take no
  #            prefetch pin (Lineage's layers have no point to tie).  gmm_impl,
  #            the tilings, use_ragged_sort, ep_collectives, split_dw,
  #            expert_weights and route_weights mean nothing to its layers
  #            (ds_ep_lineage.IGNORED_STAGE_KNOBS).  DeepSeek-V3 only.
  # 'native'   `split_layers`: this module's own layers -- DeepSeek-V3's, or the
  #            Qwen3.5 family's with its mixer in attention's place -- with
  #            MaxText's ring-of-experts and megablox kernels in Lineage's four
  #            phases a layer, an optimization barrier closing each.
  #            DeepSeek-V3's where Lineage's TPU kernels do not run (GPU,
  #            CPU; ds_ep_pipeline refuses it on TPU), and the Qwen3.5
  #            family's only batch split.
  batch_split_impl: str | None = None
  # impl 'lineage': Lineage's MoE chunk size relative to a device's balanced
  # load of routed slots (ds_ep_lineage.CAPACITY_FACTOR; MaxText's DeepSeek-V3
  # Lineage recipe runs 3.0).  Dropless at any value: it sets how many trips
  # Lineage's expert loop takes.
  lineage_capacity_factor: float = 2.0
  # How a batch-split layer's backward combines the halves' routed-expert
  # weight gradients (impl 'native').  'acc': mb0's grouped matmuls hand
  # their weights on to mb1's (moe_experts pass_weights), so mb0's dW kernel
  # starts from mb1's dW and adds into it (megablox tgmm existing_out, in
  # place; Lineage's grad_w_acc).  'sum': each half's VJP gives its own dW and
  # XLA adds the two copies, an add fusion over every [E / EP, D, F] leaf
  # (ds_ep_pipeline --split-dw).  The other weights' halves are summed either
  # way.
  split_dw: str = "acc"
  # How the routed experts' megablox kernels read a layer's weights, which a
  # chunk's programs get stacked ([layers, ...] a leaf).  'stacked': in place,
  # the group sizes moved onto the layer's block (moe_experts' stacked).
  # 'sliced': from a slice, which XLA copies into a buffer of its own first,
  # because a custom call takes whole buffers: on TPU7x 4.8 ms of a 39 ms
  # forward at M=8, in every program, and the fused last chunk lays the slices
  # out once more (2 x TPU7x-4x4x4, 2026-10).  The same products either way, bit
  # for bit; stash modes read slices (ds_ep_pipeline --expert-weights).  Not
  # under batch_split_impl 'lineage', whose layers run no megablox kernel.
  expert_weights: str = "stacked"

  @property
  def lineage_split(self):
    """Do the batch-split chunks run Lineage's layers (batch_split, impl 'lineage')?"""
    return self.batch_split and self.batch_split_impl == "lineage"

  def __post_init__(self):
    if self.batch_split_impl is None:  # the model's: Lineage builds DeepSeek-V3's layers only
      object.__setattr__(self, "batch_split_impl", "native" if is_qwen(self.model) else "lineage")
    wi, wo = TILES["clamped"](self.model)
    if self.batch_split and self.batch_split_impl == "native" and self.split_dw == "acc":
      wi = split_acc_wi_tiling(wi)
    wi = _tiling9("wi_tiling", wi if self.wi_tiling is None else self.wi_tiling)
    wo = _tiling9("wo_tiling", wo if self.wo_tiling is None else self.wo_tiling)
    object.__setattr__(self, "wi_tiling", wi)
    object.__setattr__(self, "wo_tiling", wo)
    if self.batch_split_impl not in BATCH_SPLIT_IMPLS:
      raise ValueError(f"batch_split_impl must be one of {BATCH_SPLIT_IMPLS}, got {self.batch_split_impl!r}")
    if not (isinstance(self.lineage_capacity_factor, numbers.Real) and self.lineage_capacity_factor > 0):
      raise ValueError(f"lineage_capacity_factor must be a number > 0, got {self.lineage_capacity_factor!r}")
    if self.lineage_split:
      self._check_lineage()
    if self.gmm_impl not in GMM_IMPLS:
      raise ValueError(f"gmm_impl must be one of {GMM_IMPLS}, got {self.gmm_impl!r}")
    if self.fsdp_collectives not in FSDP_COLLECTIVES:
      raise ValueError(f"fsdp_collectives must be 'ppermute' or 'xla', got {self.fsdp_collectives!r}")
    if self.ep_collectives not in EP_COLLECTIVES:
      raise ValueError(
          f"ep_collectives must be 'xla' or 'ppermute' (or 'auto', which Pipeline "
          f"resolves per stage), got {self.ep_collectives!r}"
      )
    if self.ragged_buffer_factor > 0 and not self.use_ragged_sort and self.gmm_impl != "tokamax":
      raise ValueError(
          f"ragged_buffer_factor {self.ragged_buffer_factor} caps the ring's ragged-sort buffer: "
          "it needs use_ragged_sort (or gmm_impl 'tokamax', whose compact local buffer it caps)"
      )
    if self.gmm_impl == "tokamax" and self.use_ragged_sort:
      raise ValueError(
          "gmm_impl='tokamax' sorts this device's rows into a compact buffer of its own "
          "(_local_plan); MaxText's ring_ragged_sort is megablox's: use use_ragged_sort=False"
      )
    if self.tokamax_gmm not in TOKAMAX_GMM:
      raise ValueError(f"tokamax_gmm must be one of {TOKAMAX_GMM}, got {self.tokamax_gmm!r}")
    if self.route_weights not in ROUTE_WEIGHTS:
      raise ValueError(f"route_weights must be one of {ROUTE_WEIGHTS}, got {self.route_weights!r}")
    # (Lineage's layers weigh their experts' outputs in their own combine: under its batch split the knob is moot.)
    if self.route_weights == "experts" and self.use_ragged_sort and not self.lineage_split:
      raise ValueError(
          "route_weights 'experts' scales each buffer row by its (token, choice) pair's routing weight, "
          "and MaxText's ring_ragged_sort keeps the rows' pairs inside its kernels: use "
          "use_ragged_sort=False (the argsort dispatch, or gmm_impl 'tokamax''s compact buffer)"
      )
    if self.layer_tree not in LAYER_TREES:
      raise ValueError(f"layer_tree must be one of {LAYER_TREES}, got {self.layer_tree!r}")
    if self.cp not in (1, 2, 4) or self.ep % self.cp:
      raise ValueError(
          f"cp {self.cp!r}: want 1, 2 or 4 dividing ep={self.ep} (a sequence's chunks share an " "'expert' ring)"
      )
    # (Lineage's layers run no megablox kernel: under its batch split no tiling is ever used.)
    bad = tiling_problems(self.model, wi, wo) if self.gmm_impl == "megablox" and not self.lineage_split else []
    if bad:
      msg = (
          TILING_WARNING + " -- megablox rounds the tile count up and masks the tail, so the "
          "MXU runs the whole padded block (same outputs, more time; see ds_ep.TILES): " + "; ".join(bad)
      )
      if self.strict_tiling:
        raise ValueError(msg)
      warnings.warn(msg, stacklevel=3)
    # _ag2 / _rs2 exchange with ONE peer: exact only on mesh axes of size 1 or 2.
    if self.fsdp_collectives == "ppermute" and (self.fsdp not in (1, 2) or self.ep not in (1, 2)):
      raise ValueError("fsdp_collectives='ppermute' needs fsdp, ep in (1, 2); use 'xla' otherwise")
    if self.ep_collectives == "ppermute" and self.ep != 2:
      raise ValueError("ep_collectives='ppermute' needs ep == 2; use 'xla' otherwise")
    if self.grad_reduce not in ("step", "microbatch"):
      raise ValueError(f"grad_reduce must be step or microbatch, got {self.grad_reduce!r}")
    if self.adam_math not in ("fp32", "optax"):
      raise ValueError(f"adam_math must be 'fp32' or 'optax', got {self.adam_math!r}")
    if self.grad_dtype is not None and not jnp.issubdtype(self.grad_dtype, jnp.floating):
      raise ValueError(f"grad_dtype must be None or a floating dtype, got {self.grad_dtype!r}")
    if not (isinstance(self.grad_clip, (int, float)) and self.grad_clip >= 0):
      raise ValueError(f"grad_clip must be a number >= 0 (0: no clip), got {self.grad_clip!r}")
    if self.rs_axes not in RS_AXES:
      raise ValueError(f"rs_axes must be one of {RS_AXES}, got {self.rs_axes!r}")
    if self.gather_axes not in GATHER_AXES:
      raise ValueError(f"gather_axes must be one of {GATHER_AXES}, got {self.gather_axes!r}")
    if self.shard_mode not in SHARD_MODES:
      raise ValueError(f"shard_mode must be one of {SHARD_MODES}, got {self.shard_mode!r}")
    if self.shard_mode == "explicit" and self.grad_reduce != "step":
      raise ValueError(
          "shard_mode='explicit' types the accumulator of a step's per-device partial gradients "
          "unreduced: it needs grad_reduce='step'"
      )
    if self.batch_split and self.cp > 1:
      raise ValueError(
          f"batch_split halves the rows a device holds; context parallelism (cp={self.cp}) holds " "one: use cp 1"
      )
    if self.split_dw not in SPLIT_DW:
      raise ValueError(f"split_dw must be one of {SPLIT_DW}, got {self.split_dw!r}")
    if self.expert_weights not in EXPERT_WEIGHTS:
      raise ValueError(f"expert_weights must be one of {EXPERT_WEIGHTS}, got {self.expert_weights!r}")
    if self.shard_mode == "explicit" and self.fsdp * self.ep == 1:
      # jax 0.11.1 lowers a shard_map whose manual axes all have size 1 inline,
      # and its lowering then refuses an unreduced-typed output that is not a
      # sharding constraint (mlir.check_unreduced_constraint).
      raise ValueError(
          "shard_mode='explicit' needs a stage of more than one device (fsdp x ep > 1): a one-device "
          "stage has nothing to reduce, and jax lowers its shard_maps inline, which refuses their "
          "unreduced outputs"
      )
    if is_qwen(self.model):
      _check_qwen(self)

  def _check_lineage(self):
    """batch_split_impl 'lineage': ValueError naming everything Lineage's layers cannot run, all at once -- this
    stage's settings here, the model's by ds_ep_lineage.check(model).  Pure config: ds_ep_lineage imports this
    module (so it is imported here, lazily) but no Lineage, and check reads fields only, so a StageConfig validates
    on a MaxText without Lineage.  (ds_ep_lineage.Region runs the whole check, its stage part too, when a chunk's
    programs first trace.)  A Qwen3.5-family model is refused alone: Lineage has none of its layers, so the stage's
    settings are moot."""
    from maxtext.experimental.mpmd.jax import ds_ep_lineage

    qwen, bad = is_qwen(self.model), []
    if not qwen:
      if self.shard_mode != "explicit":
        bad.append(
            f"shard_mode {self.shard_mode!r}, want 'explicit': Lineage's code is global-view code that reads "
            "its layouts from sharding types, so a chunk's programs run at global view on Explicit stage "
            "meshes"
        )
      if self.grad_reduce != "step":
        bad.append(
            f"grad_reduce {self.grad_reduce!r}, want 'step': the programs add Lineage's per-device partial "
            "weight gradients into unreduced accumulators, reduce-scattered once a step"
        )
      if self.random_routing:
        bad.append(
            "random routing: Lineage routes top-k only, it draws no random experts (there is no RNG in it): "
            "random_routing=False (ds_ep_pipeline --routing topk)"
        )
      if self.ragged_buffer_factor > 0:
        bad.append(
            f"ragged_buffer_factor {self.ragged_buffer_factor}: Lineage is dropless, and > 0 drops the "
            "tokens past the ring's capped dispatch buffer, which it has no counterpart of (want <= 0)"
        )
      if self.cp > 1:
        bad.append(
            f"cp {self.cp}: the batch split halves the rows a device holds, context parallelism holds one "
            "(and Lineage's splash attention shards heads, never the sequence)"
        )
      if self.model.num_experts % self.ep:
        bad.append(f"num_experts {self.model.num_experts} is not a multiple of ep {self.ep}")
    try:
      ds_ep_lineage.check(self.model)
    except ValueError as e:
      bad.append(str(e))
    if bad:
      raise ValueError(
          "batch_split_impl 'lineage' cannot run this: "
          + "; ".join(bad)
          + (
              " (batch_split_impl 'native' -- the Qwen3.5 family's default, None -- runs its split_layers)"
              if qwen
              else " (off TPU, batch_split_impl 'native' runs split_layers instead)"
          )
      )


def is_qwen(cfg) -> bool:
  """Is `cfg` a Qwen3.5-family model (maxtext.experimental.mpmd.jax.qwen35: Qwen3.5, Qwen3.8), rather than DeepSeek-V3?"""
  return isinstance(cfg, Qwen35Config)


def _check_qwen(scfg: StageConfig):
  """What the Qwen3.5-family stage programs do not implement, refused here rather than traced wrong."""
  cfg = scfg.model
  if scfg.cp != 1:
    # The GDN's recurrent state and conv window would have to pass from one
    # chunk of a sequence to the next (MaxText's gdn_cp.py); cp_kv_context is MLA's.
    raise ValueError(
        f"cp={scfg.cp}: context parallelism is not implemented for the Qwen3.5 family " "(Gated DeltaNet state passing)"
    )
  if cfg.shared_experts != 1:
    raise ValueError(
        f"shared_experts={cfg.shared_experts}: the Qwen3.5-family stage programs build the one "
        "sigmoid-gated shared expert every Qwen3.5 / Qwen3.8 MoE layer has"
    )
  if cfg.moe_impl != "ragged":
    raise ValueError(
        f"moe_impl={cfg.moe_impl!r}: the routed experts are the ring of experts (sort + grouped "
        "matmuls), MaxText's sparse_matmul; there is no dense MoE here"
    )
  # These flags choose the port's (and MaxText's) MoE and head numerics, which the stage programs fix: moe_experts
  # sums the top k in fp32, named_gmm accumulates in fp32, head_loss casts the logits to fp32.  MaxText's
  # sparse_matmul reads float32_weight_sum and moe_use_direct_token_gather on its argsort dispatch only
  # (layers/moe.py) -- this one's counterpart, and what it runs at EP 1, where it refuses the ring and the ragged
  # sort -- and its ring + ragged-sort path ignores them, as use_ragged_sort's does here.  So float32_weight_sum=False
  # is refused wherever MaxText's dispatch is the argsort one: with use_ragged_sort off, or at ep 1.  The direct
  # gather only where this pipeline runs its argsort dispatch, with use_ragged_sort off: that one repeats and sorts
  # the tokens, so dx sums a token's k copies in the compute dtype, as MaxText's does without the flag.  The ragged
  # sort's backward (ring_ragged_sort's ragged_gather_reduce, at ep 1 as on the ring) sums them in fp32 and rounds
  # once, as MaxText's direct gather does (bit for bit on CPU: ds_ep_qwen_test.QwenDispatchVsMaxText), so at ep 1
  # it computes MaxText's float32_weight_sum=true moe_use_direct_token_gather=true.  The reverse is let through: a
  # flag-off model at ep 1 with the ragged sort gets that fp32 dx, where MaxText's EP-1 argsort dispatch sums in the
  # compute dtype (--ragged-sort auto turns the sort off at ep 1).
  if (not scfg.use_ragged_sort or scfg.ep == 1) and not cfg.float32_weight_sum:
    raise ValueError(
        "float32_weight_sum=False: MaxText's argsort dispatch (the counterpart with use_ragged_sort off, "
        "and at ep 1) then sums the top k in the compute dtype; the stage programs sum them in fp32 (only "
        "the ring + ragged-sort path ignores the flag, here as in MaxText)"
    )
  if not scfg.use_ragged_sort and cfg.moe_use_direct_token_gather:
    raise ValueError(
        "moe_use_direct_token_gather: MaxText's argsort dispatch (the counterpart with use_ragged_sort "
        "off) then sums dx over a token's k copies in fp32 (sort_activations.route); the argsort "
        "dispatch here repeats and sorts the tokens, and sums them in the compute dtype, as MaxText's "
        "does without the flag (the ragged sort's backward sums them in fp32, at ep 1 as on the ring, "
        "as the direct gather does)"
    )
  if not cfg.gmm_fp32_accum:
    raise ValueError(
        "gmm_fp32_accum=False (MaxText's ragged_dot fallback, megablox=false use_tokamax_gmm=false): "
        "both gmm_impl settings accumulate the grouped matmuls in fp32 (named_gmm)"
    )
  if cfg.gmm_exact_drhs and scfg.gmm_impl != "megablox":
    raise ValueError(
        "gmm_exact_drhs reproduces megablox's dW kernel (tgmm), which gmm_impl='megablox' runs; "
        "gmm_impl='ragged_dot' computes dW with an fp32 ragged_dot_general"
    )
  if not cfg.cast_logits_to_fp32:
    raise ValueError("cast_logits_to_fp32=False: the head casts the logits to fp32 (head_loss)")
  n = scfg.fsdp * scfg.ep
  if cfg.emb_dim % n or cfg.num_experts % scfg.ep:
    raise ValueError(
        f"emb_dim={cfg.emb_dim} must divide by fsdp*ep={n} and num_experts={cfg.num_experts} by "
        f"ep={scfg.ep} (the storage layout shards them so)"
    )


# ------------------------------------------------------------ the param tree
def layer_shapes(cfg: DeepSeekConfig, bias=False):
  """Shapes of one MoE decoder layer, in ds_ep's leaf names (MaxText's NNX tree, see deepseek/convert.py); `bias`:
  its router's routed bias (DeepSeekConfig.routed_bias), which picks the experts and is not trained (`count_leaves`)."""
  D, H, E = cfg.emb_dim, cfg.num_query_heads, cfg.num_experts
  F, Fs = cfg.moe_mlp_dim, cfg.shared_expert_mlp_dim * cfg.shared_experts
  return {
      "pre_self_attention_layer_norm": {"scale": (D,)},
      "post_self_attention_layer_norm": {"scale": (D,)},
      "self_attention": {
          "wq_a": {"kernel": (D, cfg.q_lora_rank)},
          "q_norm": {"scale": (cfg.q_lora_rank,)},
          "wq_b": {"kernel": (cfg.q_lora_rank, H, cfg.qk_head_dim)},
          "wkv_a": {"kernel": (D, cfg.kv_lora_rank + cfg.qk_rope_head_dim)},
          "kv_norm": {"scale": (cfg.kv_lora_rank,)},
          "wkv_b": {"kernel": (cfg.kv_lora_rank, H, cfg.qk_nope_head_dim + cfg.v_head_dim)},
          "out": {"kernel": (H, cfg.v_head_dim, D)},
      },
      "moe": {
          "gate": {"kernel": (D, E), **({"bias": (E,)} if bias else {})},
          "wi_0": (E, D, F),
          "wi_1": (E, D, F),
          "wo": (E, F, D),
          "shared_experts": {
              "wi_0": {"kernel": (D, Fs)},
              "wi_1": {"kernel": (D, Fs)},
              "wo": {"kernel": (Fs, D)},
          },
      },
  }


def dense_layer_shapes(cfg: DeepSeekConfig):
  """Shapes of one dense decoder layer, DeepSeek-V3's first `first_num_dense_layers`: layer_shapes' attention and
  norms, and a dense MLP of width `mlp_dim` (MaxText's `mlp`) in place of the experts."""
  D, F = cfg.emb_dim, cfg.mlp_dim
  shapes = {k: v for k, v in layer_shapes(cfg).items() if k != "moe"}
  shapes["mlp"] = {"wi_0": {"kernel": (D, F)}, "wi_1": {"kernel": (D, F)}, "wo": {"kernel": (F, D)}}
  return shapes


def dense_layer_storage_specs():
  """layer_storage_specs' attention and norms; the dense MLP stored as the shared expert is."""
  specs = {k: v for k, v in layer_storage_specs().items() if k != "moe"}
  specs["mlp"] = {"wi_0": {"kernel": P(FE, None)}, "wi_1": {"kernel": P(FE, None)}, "wo": {"kernel": P(None, FE)}}
  return specs


def layer_storage_specs(bias=False):
  """MaxText's pipeline-large-moe sharding rules on a ('fsdp','expert') stage."""
  return {
      "pre_self_attention_layer_norm": {"scale": P(None)},
      "post_self_attention_layer_norm": {"scale": P(None)},
      "self_attention": {
          "wq_a": {"kernel": P(FE, None)},
          "q_norm": {"scale": P(None)},
          "wq_b": {"kernel": P("fsdp", None, None)},
          "wkv_a": {"kernel": P(FE, None)},
          "kv_norm": {"scale": P(None)},
          "wkv_b": {"kernel": P("fsdp", None, None)},
          "out": {"kernel": P(None, None, FE)},
      },
      "moe": {
          "gate": {"kernel": P("fsdp", None), **({"bias": P(None)} if bias else {})},
          "wi_0": P("expert", "fsdp", None),
          "wi_1": P("expert", "fsdp", None),
          "wo": P("expert", None, "fsdp"),
          "shared_experts": {
              "wi_0": {"kernel": P(FE, None)},
              "wi_1": {"kernel": P(FE, None)},
              "wo": {"kernel": P(None, FE)},
          },
      },
  }


def mtp_shapes(cfg: DeepSeekConfig, bias=False):
  """One multi-token-prediction layer (multi_token_prediction.py, MultiTokenPredictionLayer): the norms of the main
  hidden state and of the next token's embedding, their [2 x emb -> emb] projection, an MoE decoder layer and its
  final norm -- and a copy of the token embedding, which MaxText shares with the decoder's input (the last chunk's
  copy and chunk 0's table get one gradient, their sum: ds_ep_pipeline.Pipeline).  The copy is never gathered: it
  computes in its storage blocks (`chunk_compute_specs`, `_sharded_embed`)."""
  D = cfg.emb_dim
  return {
      "embedding_norm": {"scale": (D,)},
      "hidden_state_norm": {"scale": (D,)},
      "projection": {"kernel": (2 * D, D)},
      "layer": layer_shapes(cfg, bias),
      "final_norm": {"scale": (D,)},
      "token_embedder": {"embedding": (cfg.vocab_size, D)},
  }


def mtp_storage_specs(bias=False):
  return {
      "embedding_norm": {"scale": P(None)},
      "hidden_state_norm": {"scale": P(None)},
      "projection": {"kernel": P(FE, None)},
      "layer": layer_storage_specs(bias),
      "final_norm": {"scale": P(None)},
      "token_embedder": {"embedding": P(None, FE)},
  }


# The Qwen3.5 family's two mixers, one entry a layer in `ChunkSpec.kinds`; each
# is also the key of that mixer's subtree inside a layer's subtree of a chunk's
# 'layers' tree (`_qwen_layers_tree`).
QWEN_KINDS = ("gdn", "attn")
# A DeepSeek-V3 layer's kind, MoE or dense, under StageConfig.layer_tree 'per_layer':
# the chunk's 'layers' (and 'dense_layers') are then one subtree a layer, as a Qwen
# chunk's (`_qwen_layers_tree`), with no mixer subtree of its own (MLA is in the
# common parts).
DEEPSEEK_KIND = "mla"
LAYER_TREES = ("stacked", "per_layer")


def layer_kinds(cfg, layers, per_layer=False):
  """`ChunkSpec.kinds` of the global layer ids `layers`: for DeepSeek () (its
  layers stacked) or, `per_layer` (StageConfig.layer_tree), DEEPSEEK_KIND a
  layer; for a Qwen3.5-family model 'attn' for a gated full-attention layer
  (`Qwen35Config.is_attention_layer`, every cycle-th) and 'gdn' for a Gated
  DeltaNet one."""
  if not is_qwen(cfg):
    return (DEEPSEEK_KIND,) * len(layers) if per_layer else ()
  return tuple("attn" if cfg.is_attention_layer(l) else "gdn" for l in layers)


def dense_layer_kinds(n, per_layer=False):
  """`ChunkSpec.dense_kinds` of a chunk's `n` dense layers: () (stacked) or,
  `per_layer` (StageConfig.layer_tree), DEEPSEEK_KIND a layer."""
  return (DEEPSEEK_KIND,) * n if per_layer else ()


def qwen_layer_shapes(cfg: Qwen35Config):
  """One Qwen3.5-family layer in parts: `common`, what every layer has (its two
  zero-centred norms and the MoE: router, routed experts, shared expert and its
  gate), and each mixer, `gdn` and `attn` (a layer has one of them).  The leaf names
  are `maxtext.experimental.mpmd.jax.qwen35`'s (MaxText's NNX tree), so the port's functions take the
  subtrees as they are."""
  D, X, F, Fs = cfg.emb_dim, cfg.num_experts, cfg.moe_mlp_dim, cfg.shared_expert_mlp_dim
  hv, hd = cfg.gdn_num_value_heads, cfg.head_dim
  return {
      "common": {
          "input_layernorm": {"scale": (D,)},
          "post_attention_layernorm": {"scale": (D,)},
          "mlp": {
              "routed_experts": {"gate": {"kernel": (D, X)}, "wi_0": (X, D, F), "wi_1": (X, D, F), "wo": (X, F, D)},
              "shared_expert": {"wi_0": {"kernel": (D, Fs)}, "wi_1": {"kernel": (D, Fs)}, "wo": {"kernel": (Fs, D)}},
              "shared_expert_gate": {"kernel": (D, 1)},
          },
      },
      "gdn": {
          "in_proj_qkvz": {"kernel": (D, cfg.gdn_qkvz_dim)},
          "in_proj_ba": {"kernel": (D, 2 * hv)},
          "conv1d": {"kernel": (cfg.gdn_conv_kernel_dim, 1, cfg.gdn_conv_dim)},
          "A_log": (hv,),
          "dt_bias": (hv,),
          "norm": {"rms_norm": {"scale": (cfg.gdn_value_head_dim,)}},
          "out_proj": {"kernel": (cfg.gdn_value_dim, D)},
      },
      "attn": {
          "query": {"kernel": (D, cfg.num_query_heads, 2 * hd)},
          "key": {"kernel": (D, cfg.num_kv_heads, hd)},
          "value": {"kernel": (D, cfg.num_kv_heads, hd)},
          "query_norm": {"scale": (hd,)},
          "key_norm": {"scale": (hd,)},
          "out": {"kernel": (cfg.num_query_heads * hd, D)},
      },
  }


def qwen_layer_storage_specs():
  """`qwen_layer_shapes`' storage specs, by DeepSeek's rules (`layer_storage_specs`):
  projections shard their embed axis over ('fsdp', 'expert'), the router over
  'fsdp', the routed experts their expert axis over 'expert' and their embed
  axis over 'fsdp'; what is used elementwise (norm scales, the depthwise conv
  kernel, A_log, dt_bias) is replicated."""
  return {
      "common": {
          "input_layernorm": {"scale": P(None)},
          "post_attention_layernorm": {"scale": P(None)},
          "mlp": {
              "routed_experts": {
                  "gate": {"kernel": P("fsdp", None)},
                  "wi_0": P("expert", "fsdp", None),
                  "wi_1": P("expert", "fsdp", None),
                  "wo": P("expert", None, "fsdp"),
              },
              "shared_expert": {
                  "wi_0": {"kernel": P(FE, None)},
                  "wi_1": {"kernel": P(FE, None)},
                  "wo": {"kernel": P(None, FE)},
              },
              "shared_expert_gate": {"kernel": P(FE, None)},
          },
      },
      "gdn": {
          "in_proj_qkvz": {"kernel": P(FE, None)},
          "in_proj_ba": {"kernel": P(FE, None)},
          "conv1d": {"kernel": P(None, None, None)},
          "A_log": P(None),
          "dt_bias": P(None),
          "norm": {"rms_norm": {"scale": P(None)}},
          "out_proj": {"kernel": P(None, FE)},
      },
      "attn": {
          "query": {"kernel": P(FE, None, None)},
          "key": {"kernel": P(FE, None, None)},
          "value": {"kernel": P(FE, None, None)},
          "query_norm": {"scale": P(None)},
          "key_norm": {"scale": P(None)},
          "out": {"kernel": P(None, FE)},
      },
  }


def _qwen_layers_tree(kinds, parts, stack):
  """A Qwen chunk's 'layers' tree from `parts` (`qwen_layer_shapes` or
  `qwen_layer_storage_specs`): one subtree a layer, keyed by its place in the
  chunk (`qwen_layer_key`), with the common parts and its own mixer's, every
  leaf with a leading layer axis of one (`stack(leaf, 1)`)."""
  out = {}
  for l, kind in enumerate(kinds):
    t = {k: jax.tree.map(lambda x: stack(x, 1), v, is_leaf=_is_part_leaf) for k, v in parts["common"].items()}
    if kind in parts:  # a DeepSeek layer's MLA is in its common parts
      t[kind] = jax.tree.map(lambda x: stack(x, 1), parts[kind], is_leaf=_is_part_leaf)
    out[qwen_layer_key(l)] = t
  return out


def qwen_layer_key(l):
  """The key of a Qwen chunk's l-th layer in its 'layers' tree: '00', '01', ..."""
  return f"{l:02d}"


def _is_part_leaf(x):
  return _is_spec(x) or _is_shape(x)


def _compute_of(spec):
  """Storage spec -> compute spec: drop every axis but 'expert' on routed leaves."""
  if spec and spec[0] == "expert":  # routed experts [E, ., .]
    return P("expert", *([None] * (len(spec) - 1)))
  return P(*([None] * len(spec)))


@dataclasses.dataclass(frozen=True)
class ChunkSpec:
  """One pipeline chunk: a contiguous run of decoder layers on one stage --
  `dense` dense layers, then the MoE layers `layers` -- whose weights stack as
  'dense_layers' and 'layers'.  The dense layers are the model's first: all in
  chunk 0, or, under a layer split that counts every layer, spread over the
  first chunks (ds_ep_pipeline.Pipeline).

  `kinds`: each MoE layer's mixer for a Qwen3.5-family model ('gdn' / 'attn',
  `layer_kinds`), for DeepSeek DEEPSEEK_KIND a layer under StageConfig.layer_tree
  'per_layer' and () stacked.  Non-empty, the chunk's 'layers' are one subtree a
  layer (`_qwen_layers_tree`) rather than stacks.  `dense_kinds` says the same of
  its 'dense_layers' (`dense_layer_kinds`: DEEPSEEK_KIND a layer under
  'per_layer')."""

  index: int
  layers: tuple  # global MoE layer ids (the routing keys'), dense layers not counted
  first: bool
  last: bool
  stage: int
  dense: int = 0  # dense layers run before the MoE layers (chunk 0: first_num_dense_layers)
  routed_bias: bool = False  # the MoE routers carry a routed bias (DeepSeekConfig.routed_bias)
  mtp: int = 0  # multi-token-prediction layers after the head (the last chunk: mtp_num_layers)
  kinds: tuple = ()  # each MoE layer's mixer ('gdn' / 'attn'; DEEPSEEK_KIND), () for stacked DeepSeek layers
  dense_kinds: tuple = ()  # each dense layer's (DEEPSEEK_KIND), () for stacked ones

  @property
  def n_moe(self):
    return len(self.layers)


def chunk_shapes(scfg: StageConfig, ch: ChunkSpec):
  cfg = scfg.model
  stack = lambda tree, n: jax.tree.map(lambda s: (n,) + s, tree, is_leaf=_is_shape)
  tree = {}
  if ch.kinds:
    parts = qwen_layer_shapes(cfg) if is_qwen(cfg) else {"common": layer_shapes(cfg, ch.routed_bias)}
    tree["layers"] = _qwen_layers_tree(ch.kinds, parts, lambda s, k: (k,) + s)
  elif ch.n_moe:
    tree["layers"] = stack(layer_shapes(cfg, ch.routed_bias), ch.n_moe)
  if ch.dense_kinds:
    tree["dense_layers"] = _qwen_layers_tree(ch.dense_kinds, {"common": dense_layer_shapes(cfg)}, lambda s, k: (k,) + s)
  elif ch.dense:
    tree["dense_layers"] = stack(dense_layer_shapes(cfg), ch.dense)
  if ch.first:
    tree["token_embedder"] = {"embedding": (cfg.vocab_size, cfg.emb_dim)}
  if ch.last:
    tree["decoder_norm"] = {"scale": (cfg.emb_dim,)}
    tree["logits_dense"] = {"kernel": (cfg.emb_dim, cfg.vocab_size)}
  if ch.mtp:
    tree["mtp"] = mtp_shapes(cfg, ch.routed_bias)
  return tree


def _is_shape(x):
  return isinstance(x, tuple) and all(isinstance(i, int) for i in x)


def chunk_storage_specs(ch: ChunkSpec):
  stack = lambda tree: jax.tree.map(lambda s: P(None, *s), tree, is_leaf=_is_spec)
  tree = {}
  if ch.kinds:
    parts = (
        {"common": layer_storage_specs(ch.routed_bias)} if ch.kinds[0] == DEEPSEEK_KIND else qwen_layer_storage_specs()
    )
    tree["layers"] = _qwen_layers_tree(ch.kinds, parts, lambda s, k: P(None, *s))
  elif ch.n_moe:
    tree["layers"] = stack(layer_storage_specs(ch.routed_bias))
  if ch.dense_kinds:
    tree["dense_layers"] = _qwen_layers_tree(
        ch.dense_kinds, {"common": dense_layer_storage_specs()}, lambda s, k: P(None, *s)
    )
  elif ch.dense:
    tree["dense_layers"] = stack(dense_layer_storage_specs())
  if ch.first:
    tree["token_embedder"] = {"embedding": P(None, FE)}
  if ch.last:
    tree["decoder_norm"] = {"scale": P(None)}
    tree["logits_dense"] = {"kernel": P(FE, None)}
  if ch.mtp:
    tree["mtp"] = mtp_storage_specs(ch.routed_bias)
  return tree


# The stacked subtrees of a chunk: a leading layer axis, no mesh axis on it.
STACKS = ("layers", "dense_layers")


def count_leaves(ch: ChunkSpec):
  """Per leaf of the chunk's weights (jax.tree.leaves order): is it a router's
  routed bias?  Those are not trained: their gradient slot carries their
  layers' expert counts instead -- kept in fp32 through the reduce-scatter and
  out of the gradient norm (`scatter_tree`) -- and the update adds
  routed_bias_update_rate x sign(mean load - load) (StagePrograms.update), as
  MaxText's loss-free balancing does (moe.load_balance_updates_from_counts)."""
  flat, _ = jax.tree_util.tree_flatten_with_path(chunk_storage_specs(ch), is_leaf=_is_spec)
  return tuple(getattr(path[-1], "key", None) == "bias" for path, _ in flat)


def count_mask(ch: ChunkSpec):
  """count_leaves as a tree shaped like the chunk's weights."""
  return jax.tree.unflatten(jax.tree.structure(chunk_storage_specs(ch), is_leaf=_is_spec), list(count_leaves(ch)))


def deferred_leaves(ch: ChunkSpec):
  """Per leaf of the chunk's weights (jax.tree.leaves order, the accumulator's too): does a split last chunk's W
  write its gradient (ChunkPrograms.build_split_last)?  Its layers' leaves -- under 'layers' and 'dense_layers' --
  but the routers' bias slots, which carry their layers' expert counts (count_leaves): the forward computes those,
  and they stay in its F (AI) with the head's, the decoder norm's, MTP's (and a first chunk's embedding's)."""
  flat, _ = jax.tree_util.tree_flatten_with_path(chunk_storage_specs(ch), is_leaf=_is_spec)
  return tuple(
      getattr(path[0], "key", None) in STACKS and not cnt for (path, _), cnt in zip(flat, count_leaves(ch), strict=True)
  )


def router_kernel_leaves(ch: ChunkSpec):
  """Per leaf of the chunk's weights (jax.tree.leaves order): is it a router's kernel (moe.gate.kernel)?"""
  flat, _ = jax.tree_util.tree_flatten_with_path(chunk_storage_specs(ch), is_leaf=_is_spec)
  return tuple([getattr(k, "key", None) for k in path[-2:]] == ["gate", "kernel"] for path, _ in flat)


def norm_scale_leaves(ch: ChunkSpec):
  """Per leaf of the chunk's weights (jax.tree.leaves order): is it a norm's scale?"""
  flat, _ = jax.tree_util.tree_flatten_with_path(chunk_storage_specs(ch), is_leaf=_is_spec)
  return tuple(getattr(path[-1], "key", None) == "scale" for path, _ in flat)


def leaf_dtypes(scfg: StageConfig, ch: ChunkSpec, kind):
  """Per leaf of the chunk's weights (jax.tree.leaves order), its dtype as
  `kind` -- 'storage' cfg.weight_dtype, 'compute' cfg.dtype -- but under
  float32_gate_logits a router's routed bias in fp32 for both, as MaxText
  keeps it (GateLogit's bias_dtype: in bf16 its 0.001 steps would round
  away), and its kernel's compute copy in fp32: GateLogit (and moe.route)
  casts the stored kernel to fp32, so with fp32 weights the router reads the
  master's values, not a bf16 copy's.  And a Qwen3.5-family model's norm
  scales compute in fp32 under norm_numerics 'fp32': MaxText's RMSNorm casts
  the stored scale to fp32, in the zero-centred norms (scale_offset) and in
  the GDN gated norm's (dtype float32) alike, so they too read the master's
  values."""
  cfg = scfg.model
  base = jnp.dtype(cfg.weight_dtype if kind == "storage" else cfg.dtype)
  f32 = jnp.dtype(jnp.float32)
  fp32_scales = kind == "compute" and is_qwen(cfg) and cfg.norm_numerics == "fp32"
  scales = norm_scale_leaves(ch) if fp32_scales else (False,) * len(count_leaves(ch))
  if not cfg.float32_gate_logits:
    return tuple(f32 if scale else base for scale in scales)
  return tuple(
      f32 if bias or scale or (kernel and kind == "compute") else base
      for bias, kernel, scale in zip(count_leaves(ch), router_kernel_leaves(ch), scales)
  )


def leaf_dtype_tree(scfg: StageConfig, ch: ChunkSpec, kind):
  """leaf_dtypes as a tree shaped like the chunk's weights."""
  return jax.tree.unflatten(
      jax.tree.structure(chunk_storage_specs(ch), is_leaf=_is_spec), list(leaf_dtypes(scfg, ch, kind))
  )


def _layer_compute(spec):
  # stacked layer leaves carry a leading None for the layer axis
  inner = P(*spec[1:])
  return P(None, *_compute_of(inner))


def chunk_compute_specs(ch: ChunkSpec):
  st = chunk_storage_specs(ch)
  out = {k: jax.tree.map(_layer_compute if k in STACKS else _compute_of, v, is_leaf=_is_spec) for k, v in st.items()}
  if ch.mtp:
    # MTP's embedding copy stays in its column blocks (no gather, no full-size gradient: _sharded_embed)
    out["mtp"]["token_embedder"] = dict(st["mtp"]["token_embedder"])
  return out


def _partial_axes(compute_spec):
  """Mesh axes over which a per-device weight gradient is still a partial sum."""
  used = {a for d in compute_spec if d is not None for a in ((d,) if isinstance(d, str) else d)}
  rest = tuple(a for a in FE if a not in used)
  return rest


def typed_spec(cs, kind):
  """A plain compute spec -> the same spec typed `kind` ('reduced' or
  'unreduced') over its partial axes (`_partial_axes`).  A typed spec refuses
  iteration and indexing, so only the programs' in / out specs and shardings
  take one: the bodies' helpers (_gather_leaf, _scatter_leaf) read the plain."""
  return P(*cs, **{kind: frozenset(_partial_axes(cs))})


def chunk_wc_specs(scfg: StageConfig, ch: ChunkSpec):
  """The gathered compute weights' specs as the programs take and return them:
  the compute specs, typed reduced over each leaf's partial axes under
  shard_mode 'explicit' (non-routed leaves over both mesh axes, routed experts
  over 'fsdp')."""
  cs = chunk_compute_specs(ch)
  if scfg.shard_mode != "explicit":
    return cs
  return jax.tree.map(lambda c: typed_spec(c, "reduced"), cs, is_leaf=_is_spec)


def chunk_gacc_specs(ch: ChunkSpec, explicit=False):
  """Per-device partial gradients: a leading axis over the not-yet-reduced
  axes -- or, `explicit`, the compute spec typed unreduced over them."""

  def one(cs):
    return typed_spec(cs, "unreduced") if explicit else P(_partial_axes(cs), *cs)

  return jax.tree.map(one, chunk_compute_specs(ch), is_leaf=_is_spec)


def acc_specs(scfg: StageConfig, ch: ChunkSpec):
  """Layout of the cross-microbatch gradient accumulator."""
  if scfg.grad_reduce == "microbatch":
    return chunk_storage_specs(ch)
  return chunk_gacc_specs(ch, scfg.shard_mode == "explicit")


def acc_shapes(scfg: StageConfig, ch: ChunkSpec):
  return chunk_shapes(scfg, ch) if scfg.grad_reduce == "microbatch" else chunk_gacc_shapes(scfg, ch)


def chunk_gacc_shapes(scfg: StageConfig, ch: ChunkSpec):
  """Global shapes of the per-device partials (chunk_gacc_specs): a leading
  axis of one block per device of the partial axes; under shard_mode
  'explicit' the chunk's own shapes (each device holds a compute-layout block,
  the same bytes)."""
  if scfg.shard_mode == "explicit":
    return chunk_shapes(scfg, ch)
  sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}

  def one(shape, cs):
    n = math.prod(sizes[a] for a in _partial_axes(cs))
    return (n,) + shape

  return jax.tree.map(one, chunk_shapes(scfg, ch), chunk_compute_specs(ch), is_leaf=_is_shape)


def num_params(scfg: StageConfig, chunks):
  tot = 0
  for ch in chunks:
    tot += sum(math.prod(s) for s in jax.tree.leaves(chunk_shapes(scfg, ch), is_leaf=_is_shape))
  return tot


# ------------------------------------------------------------ per-device math
def ragged_buffer_rows(scfg: StageConfig, tokens):
  """Rows of the ring's dispatch buffer a device for `tokens` all-gathered
  tokens (the EP group's), as MaxText sizes it (RoutedMoE.get_ragged_buffer_size
  of the balanced size (tokens // ep) x top-k): None for the worst case
  (dropless, every token x top-k: a buffer that large, or ragged_buffer_factor
  <= 0), else ragged_buffer_factor x a device's balanced load."""
  f, K, ep = scfg.ragged_buffer_factor, scfg.model.num_experts_per_tok, scfg.ep
  if f <= 0:
    return None
  rows = int((tokens // ep) * K * f)
  return None if rows >= tokens * K else rows


def truncate_group_sizes(gs, rows):
  """One shard's group sizes `gs` (its rows sorted by group) clamped to a
  buffer of `rows` rows: the groups past the buffer's end lose their rows,
  from the last row back -- MaxText's `_truncate_matrix` of the shard's
  column."""
  return jnp.diff(jnp.minimum(jnp.cumsum(gs), rows), prepend=0).astype(gs.dtype)


def gmm_row_tile(scfg: StageConfig):
  """The row count megablox needs a grouped matmul's lhs to be a multiple of:
  every tm of both tilings -- forward, dlhs and drhs, each of which tiles the
  rows -- so their least common multiple (512 for the clamped tiles)."""
  return math.lcm(*(t[i] for t in (scfg.wi_tiling, scfg.wo_tiling) for i in (0, 3, 6)))


@jax.custom_vjp
def _with_aux_loss(y, aux):
  """`y`, with the scalar loss term `aux` riding its backward: wherever `y` is
  differentiated, `aux` is too, with cotangent 1 -- its gradient joins the
  layer's without any loss output (Megatron-LM's MoEAuxLossAutoScaler)."""
  del aux
  return y


def _with_aux_loss_fwd(y, aux):
  return y, aux


def _with_aux_loss_bwd(aux, dy):
  return dy, jnp.ones_like(aux)


_with_aux_loss.defvjp(_with_aux_loss_fwd, _with_aux_loss_bwd)


def seq_aux_loss(scores, cfg: DeepSeekConfig):
  """Megatron-LM's sequence-wise aux loss of a sigmoid router, per sequence
  (moe.megatron_aux_loss_inputs + RoutedMoE.load_balance_loss): the fp32
  pre-bias scores normalized per token, the token fractions from a plain top-k
  of them (no bias, no groups) -> load_balance_loss_weight x E x sum over the
  experts of fraction x mean probability, [b] for scores [b, S, E]."""
  b, S, E = scores.shape
  K = cfg.num_experts_per_tok
  sc = scores.astype(jnp.float32)
  probs = sc / (jnp.sum(sc, axis=-1, keepdims=True) + 1e-20)
  _, idx = lax.top_k(sc, K)
  rows = jnp.broadcast_to(jnp.arange(b)[:, None, None], idx.shape)
  density = jnp.zeros((b, E), jnp.float32).at[rows, idx].add(1.0) / (S * K)
  return jnp.sum(density * jnp.mean(probs, axis=1), axis=-1) * E * cfg.load_balance_loss_weight


# In-place weight-gradient accumulation (gmm_impl 'tokamax'): while a backward that accumulates into the step's
# gradient accumulator is traced, {MoE layer index in the chunk: {expert leaf: its accumulator block}} of the chunk
# (`ChunkPrograms`' bwd_acc), and in the unsplit layer loops the layer being traced.  The routed experts' grouped
# matmul whose dW runs first in the layer's backward starts its dW kernel from that block (tokamax_gmm's dw0,
# gpu_kernels.ragged_dw), so the dW is added into the accumulator where it lies, as megablox's tgmm adds existing_out:
# no [E / EP, D, F] dW buffer of its own a weight and no pass that adds it.
_DW0 = contextvars.ContextVar("mpmd_dw0", default=None)
_DW0_LAYER = contextvars.ContextVar("mpmd_dw0_layer", default=None)


@contextlib.contextmanager
def _context(var, value):
  tok = var.set(value)
  try:
    yield
  finally:
    var.reset(tok)


def _dw0_for(l):
  """The accumulator blocks for MoE layer `l` of the chunk traced, or None."""
  by_layer = _DW0.get()
  return None if by_layer is None or l is None else by_layer.get(l)


def _ring_moe(w, x, key, scfg: StageConfig, stacked=None, aux_seqs=0, scores=None):
  """Routed experts, ring of experts, on device-local tokens `x [b, S, D]`
  -> (output, expert counts [E] int32 of the EP group's tokens' choices).
  With `aux_seqs` (the global batch's sequences; load_balance_loss_weight >
  0) the output carries this device's share of the sequence-wise aux loss
  (`seq_aux_loss` of its own sequences / aux_seqs, `_with_aux_loss`).

  MaxText's `RoutedMoE.sparse_matmul` under use_ring_of_experts: the all-gather
  and route, `RoutedMoE.permute` (ring_ragged_sort), the megablox gmm with
  group_offset, `RoutedMoE.unpermute` (ring_ragged_unsort) and the
  psum_scatter.  In four parts -- the router's scores, the
  dispatch (the tokens' all-gather), the experts and the combine (the
  reduce-scatter) -- which a batch-split layer (`split_layers`) runs in
  phases of their own; composed here they trace as one function.  `stacked`:
  moe_experts'.  `scores`: x's router scores (moe_scores), where the caller
  has them already (layer_fwd_counts under lean residuals).
  """
  scores = moe_scores(w, x, scfg) if scores is None else scores
  aux = jnp.sum(seq_aux_loss(scores, scfg.model)) / aux_seqs if aux_seqs else None
  xg, sg = moe_dispatch(x, scores, scfg)
  y, counts = moe_experts(w, xg, sg, key, scfg, stacked=stacked, with_counts=True, dw0=_dw0_for(_DW0_LAYER.get()))
  y = moe_combine(y, scfg)
  return (y if aux is None else _with_aux_loss(y, aux)), counts


def moe_scores(w, x, scfg: StageConfig):
  """The router's scores of device-local tokens `x [b, S, D]` -> [b, S, E]:
  its kernel alone (the routed bias joins after the score function, in
  moe_experts: GateLogit), in fp32 under float32_gate_logits (GateLogit's
  dtype, on the kernel's fp32 compute copy: leaf_dtypes).  A Qwen3.5-family
  router's scores are its raw logits."""
  cfg = scfg.model
  gcfg = cfg.replace(dtype=jnp.float32) if cfg.float32_gate_logits else cfg
  gate = layers.dense_general({"kernel": w["gate"]["kernel"]}, x, gcfg)
  if is_qwen(cfg):
    # Qwen routes on the raw logits: no score function, a softmax over the
    # top k only (qwen35/moe.py:route; moe_experts)
    return gate
  return layers.activation_fn(cfg.routed_score_func)(gate)


def moe_dispatch(x, scores, scfg: StageConfig, tag=None):
  """The ring of experts' dispatch: the tokens and their scores all-gathered
  over 'expert' -> (xg [EP b, S, D], sg [EP b, S, E]).  `tag`: a named scope
  inside ep_all_gather (a batch-split half's, mb0 / mb1), where a transform
  around the call still wraps ep_all_gather (xprof.scope_of reads it)."""
  with jax.named_scope("ep_all_gather"), _scope(tag):
    if scfg.ep_collectives == "ppermute" and scfg.ep == 2:
      xg = _ag2(x, EP_AXIS, 0)
      sg = _ag2(scores, EP_AXIS, 0)
    else:
      xg = lax.all_gather(x, EP_AXIS, axis=0, tiled=True)
      sg = lax.all_gather(scores, EP_AXIS, axis=0, tiled=True)
  # checkpoint_name is an identity (no HLO); only a stash forward's policy
  # reads the names (STASH_POLICIES).
  return checkpoint_name(xg, "moe_xg"), checkpoint_name(sg, "moe_sg")


def moe_combine(y, scfg: StageConfig, tag=None):
  """The ring of experts' combine: the gathered tokens' expert outputs
  reduce-scattered over 'expert' back to this device's tokens (`tag` as in
  moe_dispatch)."""
  with jax.named_scope("ep_reduce_scatter"), _scope(tag):
    if scfg.ep_collectives == "ppermute" and scfg.ep == 2:
      return _rs2(y, EP_AXIS, 0)
    return lax.psum_scatter(y, EP_AXIS, scatter_dimension=0, tiled=True)


def _scope(name):
  return jax.named_scope(name) if name else contextlib.nullcontext()


def moe_experts(w, xg, sg, key, scfg: StageConfig, pass_weights=False, stacked=None, with_counts=False, dw0=None):
  """This shard's routed experts over the gathered tokens `xg [EP b, S, D]`
  (scores `sg`): route, sort, the three grouped matmuls, unsort and the
  top-k weighted sum -> [EP b, S, D], this shard's part of every gathered
  token's output (moe_combine sums the shards').  Routing is MaxText's
  get_topk.  DeepSeek-V3's is RoutedMoE.deepseek_routing: the experts are
  picked on the scores plus the router's routed bias (if it has one), among
  each token's best groups (n_routing_groups), and weighed by the scores
  without the bias, sum-normalised and scaled by routed_scaling_factor
  (deepseek_scale_weights).  A Qwen3.5-family model's takes the top k of
  the raw logits (no groups, no routed bias), weighed by a softmax over
  those k in fp32, rounded to the compute dtype and renormalised under
  norm_topk_prob (qwen35/moe.py:route).  Under random_routing either family
  draws its k experts uniformly with replacement and weighs them by the raw
  scores (a Qwen model's raw logits), with no softmax or scaling, as
  get_topk returns early.  `with_counts`: -> (that, the expert counts [E]
  int32 of the gathered tokens' choices).

  `pass_weights`: also hand on the three expert weights as the grouped
  matmuls took them (cast to the compute dtype) -> (y, {'wi_0', 'wi_1',
  'wo'}).  A later moe_experts that runs on them in place of w's (a
  batch-split layer's mb1: split_layers) adds its weight gradients into this
  call's inside the dW kernel (named_gmm's acc) instead of beside it.

  `stacked` = (the chunk's stacked expert weights {'wi_0', 'wi_1', 'wo'},
  this layer's index in them): megablox reads the weights there, not from
  w's slices of them (named_gmm's stacked), where their dtype is the compute
  dtype.

  Under route_weights 'experts' the weights scale the rows of the gated
  activation before wo, and the unsort's sum over a token's choices is
  unweighted (the argsort dispatch's, or _tokamax_experts').  Under lean
  residuals wo's VJP keeps the gated product's inputs, not the product
  (`_wo_lhs`), and the wi products of the argsort dispatch keep the gathered
  tokens and the order, not the T x K sorted rows of them (`Recompute`)."""
  cfg, ep = scfg.model, scfg.ep
  _, S, D = xg.shape
  E, K = cfg.num_experts, cfg.num_experts_per_tok
  dt = cfg.dtype
  T = xg.shape[0] * S
  sg = sg.reshape(T, E)
  if scfg.random_routing:
    # moe.random_routing: uniform with replacement, weights are the raw scores and
    # deepseek_scale_weights is skipped (get_topk returns early) -- for Qwen the
    # raw logits, with no softmax.
    idx = jax.random.randint(key, (T * K,), 0, E, dtype=jnp.int32).reshape(T, K)
    wts = jnp.take_along_axis(sg, idx, axis=-1)
  else:
    from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe

    sel = sg
    if "bias" in w["gate"]:
      sel = sel + jnp.asarray(w["gate"]["bias"], sel.dtype)  # GateLogit: in the activation dtype
    # Qwen: the top k of the raw logits (get_topk; no groups, no routed bias)
    idx = lax.top_k(sel, K)[1] if is_qwen(cfg) else ds_moe.select_experts(sel, cfg)
    wts = jnp.take_along_axis(sg, idx, axis=-1)
    if is_qwen(cfg):
      # qwen35/moe.py:route (MaxText's get_topk for qwen3_5): a softmax over
      # the k selected logits in fp32, rounded, then renormalised in `dt`
      wts = jax.nn.softmax(wts.astype(jnp.float32), axis=-1).astype(dt)
      if cfg.norm_topk_prob:
        wts = wts / wts.sum(-1, keepdims=True)
    else:
      wts = wts / (wts.sum(-1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor
  counts = jnp.zeros((E,), jnp.int32).at[idx.reshape(-1)].add(1)
  xf = xg.reshape(T, D)
  shard = lax.axis_index(EP_AXIS)
  if scfg.gmm_impl == "tokamax":
    y, passed = _tokamax_experts(w, xf, idx, wts, shard, scfg, pass_weights, stacked, dw0)
    y = y.reshape(-1, S, D)
    out = (y, passed) if pass_weights else y
    return (out, counts) if with_counts else out
  interpret = jax.default_backend() != "tpu"
  if scfg.use_ragged_sort:
    # RoutedMoE.permute: SparseCore ragged sort in global expert order; this
    # shard's experts start at group_offset.
    from maxtext.kernels.ragged import ragged_sort

    rows = ragged_buffer_rows(scfg, T)
    # (x, group sizes, revert indices, argsort indices): ring_ragged_unsort takes the last optionally
    # (topk_argsort_indices, to skip an argsort in its backward: the same permutation, the same values).  Unsorted
    # without them, as MaxText's own default routing-maps policy (moe_routing_maps=remat) unsorts.
    xs, gs, rev, _ = ragged_sort.ring_ragged_sort(xf, idx, E, K, EP_AXIS, ep, buffer_size=rows)
    if rows is None:
      off, mm_gs = (shard * (E // ep)).astype(jnp.int32), gs
    else:
      # MaxText's truncated buffer (RoutedMoE.permute, use_truncated_buffer):
      # the buffer holds this shard's rows packed from row 0; its groups'
      # sizes, clamped to the buffer, and group_offset 0.  The unsort takes
      # the global sizes, as MaxText's.
      off = jnp.int32(0)
      mm_gs = truncate_group_sizes(lax.dynamic_slice_in_dim(gs, shard * (E // ep), E // ep), rows)
  else:
    # RoutedMoE.permute without ragged sort (MaxText requires it at EP=1): roll the expert
    # ids so this shard's experts are groups 0..E/EP-1, argsort, repeat +
    # custom-VJP gather; megablox then computes only the local groups.
    from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe

    rolled = ((idx - shard * (E // ep)) % E).reshape(-1)
    order = jnp.argsort(rolled)
    xs = ds_moe._sort_activations(jnp.repeat(xf, K, axis=0), order)
    gs = jnp.bincount(rolled, length=E).astype(jnp.int32)
    off, mm_gs = jnp.int32(0), gs

  passed = {}

  def mm(name, lhs, wname, tiling, keep=None):
    st = (stacked[0][wname], stacked[1]) if stacked is not None and stacked[0][wname].dtype == dt else None
    out = named_gmm(
        name,
        tiling,
        interpret,
        lhs,
        w[wname].astype(dt),
        mm_gs,
        off,
        impl=scfg.gmm_impl,
        acc=pass_weights,
        stacked=st,
        keep=keep,
    )
    if pass_weights:
      out, passed[wname] = out
    return out

  # megablox needs the rows a multiple of every tm (`gmm_row_tile`), which a
  # capped buffer -- int(balanced load x ragged_buffer_factor) rows -- rarely
  # is: zero rows past every group pad it, and the output is cut back (MaxText
  # pads its gmm input with max_utils.maybe_pad and slices the output alike).
  m = xs.shape[0]
  pad = -m % gmm_row_tile(scfg) if scfg.gmm_impl == "megablox" else 0
  if pad:
    xs = jnp.pad(xs, ((0, pad), (0, 0)))
  # (MaxText's ring_ragged_sort is a kernel of its own: its rows are kept as they are)
  keep_xs = (
      Recompute(functools.partial(_argsort_rows, K=K, pad=pad), (xf, order))
      if layers.lean() and scfg.lean_experts and not scfg.use_ragged_sort
      else None
  )
  xs = _tie("xs", xs)  # prefetch pins (an identity unless a pinned forward traces)
  h0 = checkpoint_name(mm("gmm_wi_0", xs, "wi_0", scfg.wi_tiling, keep_xs), "moe_h0")
  h1 = checkpoint_name(mm("gmm_wi_1", xs, "wi_1", scfg.wi_tiling, keep_xs), "moe_h1")
  # route_weights 'experts': each sorted row's weight (StageConfig refuses it with the ragged sort), the padding's 0
  w_row = (
      jnp.pad(jnp.take(wts.reshape(-1), order, axis=0).astype(jnp.float32), (0, pad))
      if scfg.route_weights == "experts"
      else None
  )
  act, keep_act = _wo_lhs(h0, h1, w_row, cfg.mlp_activations[0], dt, scfg.lean_experts)
  y = mm("gmm_wo", act, "wo", scfg.wo_tiling, keep_act)
  y = checkpoint_name(y[:m] if pad else y, "moe_y")
  y = _tie("y", y)
  if scfg.use_ragged_sort:
    y = ragged_sort.ring_ragged_unsort(y, gs, rev, K, E // ep, EP_AXIS, topk_weights=jnp.ravel(wts).astype(jnp.float32))
  else:
    # RoutedMoE.unpermute: inverse gather, then the fp32 weighted sum over top-k (its rows weighted already under
    # route_weights 'experts': the plain sum)
    y = ds_moe._sort_activations(y, jnp.argsort(order)).reshape(T, K, D).astype(jnp.float32)
    y = (y.sum(axis=1) if w_row is not None else jnp.einsum("tkd,tk->td", y, wts.astype(jnp.float32))).astype(dt)
  y = y.reshape(-1, S, D)
  out = (y, passed) if pass_weights else y
  return (out, counts) if with_counts else out


def _argsort_rows(xf, order, K, pad):
  """The argsort dispatch's sorted rows (moe_experts): token order[r] // K of `xf [T, D]` in row r -- the gather
  that sorting xf repeated K times is -- and `pad` zero rows after them (megablox's row tile)."""
  return jnp.pad(jnp.take(xf, order // K, axis=0), ((0, pad), (0, 0)))


@jax.tree_util.register_pytree_node_class
class Recompute:
  """A grouped matmul's lhs as its VJP keeps it under lean residuals (`keep` of named_gmm / tokamax_gmm): the arrays
  `args` it is recomputed from in the backward, `fn(*args)`, in place of the lhs itself.  The wi products' buffer
  rows [rows, emb] are kept as the gathered tokens [tokens, emb] they were gathered from (and the dispatch plan):
  fewer bytes wherever the buffer has more rows than there are tokens -- a DeepSeek-V3 layer at EP 8, one sequence a
  device and ragged_buffer_factor 2 keeps 448 MiB instead of 896 -- for a row gather more in the backward.  wo's
  gated activation is kept as the gated product's inputs, which the product's own VJP keeps anyway (it runs under
  jax.checkpoint), so its [rows, moe_mlp] costs nothing more.  `fn(*args)` must be the lhs bit for bit, since the
  backward's dW reads it in the lhs's place, and the lhs's gradient reaches `args` through the lhs's own producer (the
  dispatch's VJP, the gated product's): the grouped matmul gives `args` none.  A pytree, `args` its children and `fn`
  static, so that a custom VJP keeps it among its residuals."""

  def __init__(self, fn, args):
    self.fn, self.args = fn, tuple(args)

  def __call__(self):
    return self.fn(*self.args)

  def tree_flatten(self):
    return self.args, self.fn

  @classmethod
  def tree_unflatten(cls, fn, args):
    return cls(fn, args)


def _kept(lhs):
  """A grouped matmul's lhs from its residuals: the lhs, or recomputed from what was kept of it (`Recompute`)."""
  return lhs() if isinstance(lhs, Recompute) else lhs


def named_gmm(name, tiling, interpret, lhs, rhs, gs, off, impl="megablox", acc=False, stacked=None, keep=None):
  """The routed experts' grouped matmul, `lhs[r] @ rhs[g - off]` for every row
  r of a group g in [off, off + rhs.shape[0]) (rows sorted by group, `gs` the
  sizes of ALL groups; rows of other groups come out zero), with the kernels
  named for xprof: `name` forward, `name_dx` / `name_dw` backward.

  acc=True -> (out, rhs): `rhs` handed on, for another grouped matmul of the
  same weight to run on.  That one's weight gradient, the cotangent of the
  handed-on copy, is this call's dW kernel's starting value: megablox tgmm
  adds it as it stores each tile (existing_out, aliased to the output), so
  the two gradients make one buffer and no add follows.

  stacked=(w, l): `rhs` is block l of the chunk's stacked weight w [L,
  rhs.shape...], and the forward and dlhs kernels read it there -- w as one [L
  x rhs.shape[0], ...] array (a bitcast), the group sizes padded with empty
  groups so that this layer's land on its block (`_stacked_groups`) -- instead
  of from `rhs`, a slice of w that XLA copies into a buffer of its own before
  a megablox kernel (a custom call takes whole buffers: on TPU7x 4.8 ms of a
  39 ms forward at M=8).  `rhs` then carries only the weight gradient, the
  drhs kernel's output: its value is never read, so XLA drops the slice.  The
  same tiles, the same products: bit for bit on every row of the groups and in
  dW (ds_ep_test.Kernels).  megablox only: ragged_dot reads `rhs` (XLA lowers
  its dot itself, and with more groups it rounds differently).

  impl='megablox': megablox `ops.gmm` (its unquantized megablox path, bit for
  bit) with the 9-tuple `tiling`.  megablox's `backend.gmm` / `tgmm` are
  jitted, and the jit boundary drops the caller's name scope, so XLA named one
  layer's three kernels after whatever enclosed them (`gmm.1`, `shard_map.147`,
  `shard_map.148`).  A Pallas kernel's HLO instruction name -- its op name in
  the trace -- is the innermost named scope, so call the unjitted bodies under
  one.

  impl='ragged_dot': the same three products in pure JAX (`jax.lax.ragged_dot`
  / `ragged_dot_general`, fp32 accumulation like megablox); `tiling` and
  `interpret` are unused and MaxText is not imported.

  keep (a Recompute of `lhs`): what the VJP keeps of the lhs for the dW
  kernel, recomputed in the backward, instead of the lhs.
  """
  ws, layer = (None, None) if stacked is None else stacked
  if ws is not None and (tuple(ws.shape[1:]) != tuple(rhs.shape) or ws.dtype != rhs.dtype):
    raise ValueError(
        f"named_gmm stacked: block {layer} of a {ws.dtype}{list(ws.shape)} weight is not the "
        f"{rhs.dtype}{list(rhs.shape)} rhs"
    )
  if impl == "megablox":
    f = _megablox_gmm_acc if acc else _megablox_gmm
    return f(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep)
  if impl == "ragged_dot":
    return (_ragged_gmm_acc if acc else _ragged_gmm)(name, lhs, rhs, gs, off, keep)
  if impl == "tokamax":
    raise ValueError(
        "named_gmm: gmm_impl 'tokamax' runs on a compact local buffer, through tokamax_gmm "
        "(moe_experts' _tokamax_experts), not on the ring's sorted rows"
    )
  raise ValueError(f"named_gmm impl must be one of {GMM_IMPLS}, got {impl!r}")


def _stacked_groups(gs, off, n_local, layer, n_layers):
  """Group sizes and an offset that put groups [off, off + n_local) of `gs`
  on block `layer` of a stacked [n_layers x n_local, ...] rhs: the other
  layers' blocks get empty groups.  When `gs` holds just those groups (the
  truncated ring buffer: off 0), offset 0 and n_layers x n_local groups, as
  many as the rhs has, so megablox zeroes no rows it did not before; else one
  leading group of the rows before them (those of the groups < off), offset
  1, and megablox zeroes the rows outside the rhs's groups, as it did."""
  pad = lambda k: jnp.zeros((k,), gs.dtype)
  before, after = pad(layer * n_local), pad((n_layers - 1 - layer) * n_local)
  if gs.shape[0] == n_local:
    return jnp.concatenate([before, gs, after]), jnp.int32(0)
  lead = jnp.sum(jnp.where(jnp.arange(gs.shape[0]) < off, gs, 0)).astype(gs.dtype)
  local = lax.dynamic_slice_in_dim(gs, off, n_local)
  return jnp.concatenate([lead[None], before, local, after]), jnp.int32(1)


def _gmm_operands(rhs, gs, off, ws, layer):
  """(the rhs the forward and dlhs kernels read, its group sizes, its group
  offset): `rhs`, `gs`, `off` themselves, or under named_gmm's stacked the
  stacked weight `ws` as one [L x E_local, ...] array (`_stacked_groups`)."""
  if ws is None:
    return rhs, gs, off
  n_layers, n_local = ws.shape[:2]
  sizes, offset = _stacked_groups(gs, off, n_local, layer, n_layers)
  return ws.reshape(n_layers * n_local, *ws.shape[2:]), sizes, offset


@contextlib.contextmanager
def _kernel_scope(name, impl="megablox"):
  """`megablox/name` (`ragged_dot/name`): a transform wraps only the scope
  right after it, so `megablox` takes the backward's `transpose(jvp(...))` and
  the kernel keeps its name instead of becoming `transpose_jvp_gmm_wo_dx__`."""
  with jax.named_scope(impl), jax.named_scope(name):
    yield


# The keywords named_gmm passes to megablox's unjitted gmm / tgmm (private
# MaxText internals: checked once, on first use, by _megablox).
_MEGABLOX_KWARGS = {
    "gmm": (
        "lhs",
        "rhs",
        "group_sizes",
        "preferred_element_type",
        "tiling",
        "group_offset",
        "transpose_rhs",
        "interpret",
    ),
    "tgmm": (
        "lhs",
        "rhs",
        "group_sizes",
        "preferred_element_type",
        "tiling",
        "group_offset",
        "num_actual_groups",
        "existing_out",
        "interpret",
    ),
}


@functools.cache
def _megablox():
  """(gmm, tgmm): megablox's unjitted bodies, imported on first use, their
  signatures checked against the keywords named_gmm passes."""
  from maxtext.kernels.megablox import backend  # pylint: disable=import-outside-toplevel

  kw_ok = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
  out = []
  for fn_name, kws in _MEGABLOX_KWARGS.items():
    fn = getattr(getattr(backend, fn_name, None), "__wrapped__", None)
    if fn is None:
      raise RuntimeError(
          f"MaxText changed megablox: {backend.__name__}.{fn_name} ({backend.__file__}) "
          "is no longer a jitted function with an unjitted __wrapped__ body, which "
          "ds_ep.named_gmm calls"
      )
    params = inspect.signature(fn).parameters
    bad = [k for k in kws if k not in params or params[k].kind not in kw_ok]
    if bad:
      raise RuntimeError(
          f"MaxText changed megablox: {backend.__name__}.{fn_name}{inspect.signature(fn)} "
          f"({backend.__file__}) does not take {bad} by keyword, which "
          "ds_ep.named_gmm passes"
      )
    out.append(fn)
  return tuple(out)


# named_gmm's kernels.  `layer` and `ws` are its stacked=(ws, layer), (None,
# None) without; `ws` gets no gradient (its block's comes out as rhs's), so
# the backward returns None for it.  Under stacked the residuals hold `ws`,
# not `rhs`, whose shape and dtype are ws's block's (`_rhs_type`).  `keep`:
# named_gmm's, held in the lhs's place among the residuals (None: the lhs);
# it gets no gradient either.
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _megablox_gmm(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep):
  return _megablox_gmm_fwd(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep)[0]


def _megablox_gmm_fwd(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep):
  gmm, _ = _megablox()
  rk, gk, ok = _gmm_operands(rhs, gs, off, ws, layer)
  with _kernel_scope(name):
    out = gmm(
        lhs=lhs,
        rhs=rk,
        group_sizes=gk,
        preferred_element_type=lhs.dtype,
        tiling=tiling[:3],
        group_offset=ok,
        interpret=interpret,
    )
  return out, (lhs if keep is None else keep, rhs if ws is None else None, gs, off, ws)


def _rhs_type(rhs, ws):
  """(shape, dtype) of named_gmm's rhs from its residuals."""
  return (rhs.shape, rhs.dtype) if ws is None else (ws.shape[1:], ws.dtype)


def _megablox_gmm_bwd(name, tiling, interpret, layer, res, g, dw_in=None):
  """ops._gmm_bwd: dlhs = gmm(g, rhs^T), drhs = tgmm(lhs^T, g) (+ dw_in,
  inside the kernel: named_gmm's acc)."""
  gmm, tgmm = _megablox()
  lhs, rhs, gs, off, ws = res
  lhs = _kept(lhs)
  shape, dtype = _rhs_type(rhs, ws)
  rk, gk, ok = _gmm_operands(rhs, gs, off, ws, layer)
  with _kernel_scope(name + "_dx"):
    dx = gmm(
        lhs=g,
        rhs=rk,
        group_sizes=gk,
        preferred_element_type=lhs.dtype,
        tiling=tiling[3:6],
        group_offset=ok,
        transpose_rhs=True,
        interpret=interpret,
    )
  with _kernel_scope(name + "_dw"):
    dw = tgmm(
        lhs=lhs.swapaxes(0, 1),
        rhs=g,
        group_sizes=gs,
        preferred_element_type=dtype,
        tiling=tiling[-3:],
        group_offset=off,
        num_actual_groups=shape[0],
        existing_out=dw_in,
        interpret=interpret,
    )
  return dx, dw, None, None, None, None


_megablox_gmm.defvjp(_megablox_gmm_fwd, _megablox_gmm_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _megablox_gmm_acc(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep):
  return _megablox_gmm_fwd(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep)[0], rhs


def _megablox_gmm_acc_fwd(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep):
  out, res = _megablox_gmm_fwd(name, tiling, interpret, layer, lhs, rhs, gs, off, ws, keep)
  return (out, rhs), res


def _megablox_gmm_acc_bwd(name, tiling, interpret, layer, res, cts):
  g, dw_in = cts  # dw_in: the gradient of the handed-on rhs
  return _megablox_gmm_bwd(name, tiling, interpret, layer, res, g, dw_in)


_megablox_gmm_acc.defvjp(_megablox_gmm_acc_fwd, _megablox_gmm_acc_bwd)


def _local_groups(gs, off, n_local, m):
  """The rows of groups [off, off + n_local) in an m-row buffer sorted by
  group (sum(gs) <= m, as for megablox), as an [m, 1] mask, and group sizes
  that make ragged_dot visit exactly those groups: the local sizes, with the
  rows before them folded into the first and the rows after into the last
  (masked to zero, so they add nothing)."""
  ends = jnp.cumsum(gs)
  lo = (ends - gs)[off]
  hi = ends[off + n_local - 1]
  rows = lax.broadcasted_iota(jnp.int32, (m, 1), 0)
  mask = (rows >= lo) & (rows < hi)
  sizes = lax.dynamic_slice_in_dim(gs, off, n_local)
  return mask, sizes.at[0].add(lo).at[n_local - 1].add(m - hi)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ragged_gmm(name, lhs, rhs, gs, off, keep):
  return _ragged_gmm_fwd(name, lhs, rhs, gs, off, keep)[0]


def _ragged_gmm_fwd(name, lhs, rhs, gs, off, keep):
  mask, sizes = _local_groups(gs, off, rhs.shape[0], lhs.shape[0])
  with _kernel_scope(name, "ragged_dot"):
    out = lax.ragged_dot(jnp.where(mask, lhs, 0), rhs, sizes, preferred_element_type=jnp.float32)
  return jnp.where(mask, out, 0).astype(lhs.dtype), (lhs if keep is None else keep, rhs, gs, off)


def _ragged_gmm_bwd(name, res, g, dw_in=None):
  """dlhs = g @ rhs^T per group, drhs = lhs^T @ g per group (ragged
  contraction), + dw_in in fp32 before the rounding (named_gmm's acc, as
  megablox's tgmm adds existing_out)."""
  lhs, rhs, gs, off = res
  lhs = _kept(lhs)
  mask, sizes = _local_groups(gs, off, rhs.shape[0], lhs.shape[0])
  g = jnp.where(mask, g, 0)
  with _kernel_scope(name + "_dx", "ragged_dot"):
    dx = lax.ragged_dot(g, rhs.swapaxes(1, 2), sizes, preferred_element_type=jnp.float32)
  with _kernel_scope(name + "_dw", "ragged_dot"):
    dn = lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[]
    )
    dw = lax.ragged_dot_general(jnp.where(mask, lhs, 0), g, sizes, dn, preferred_element_type=jnp.float32)
    if dw_in is not None:
      dw = dw + dw_in.astype(jnp.float32)
  return jnp.where(mask, dx, 0).astype(lhs.dtype), dw.astype(rhs.dtype), None, None, None


_ragged_gmm.defvjp(_ragged_gmm_fwd, _ragged_gmm_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ragged_gmm_acc(name, lhs, rhs, gs, off, keep):
  return _ragged_gmm_fwd(name, lhs, rhs, gs, off, keep)[0], rhs


def _ragged_gmm_acc_fwd(name, lhs, rhs, gs, off, keep):
  out, res = _ragged_gmm_fwd(name, lhs, rhs, gs, off, keep)
  return (out, rhs), res


def _ragged_gmm_acc_bwd(name, res, cts):
  g, dw_in = cts
  return _ragged_gmm_bwd(name, res, g, dw_in)


_ragged_gmm_acc.defvjp(_ragged_gmm_acc_fwd, _ragged_gmm_acc_bwd)


# ------------------------------------------------- gmm_impl 'tokamax' (GPU)
# tokamax's grouped matmuls take no group_offset, so this device's rows of the
# ring's gathered tokens go into a compact buffer of their own -- the rows of
# its E / EP experts only, sorted by expert, other shards' rows never in it --
# instead of the ring's T x K sorted rows (`_local_plan`).  Dispatch into it
# and combine out of it are gathers both ways (`_dispatch_rows`,
# `_combine_rows`): each buffer row is one (token, choice) pair, so every
# backward is a gather too, and no scatter-add (atomics on GPU, an order that
# changes run to run) is left in the programs.


def _local_plan(idx, shard, n_local, rows):
  """This shard's part of the routing `idx [T, K]` (global expert ids; its
  experts are [shard x n_local, (shard + 1) x n_local)) -> (src [R] int32, the
  token each buffer row takes; gs [n_local] int32, the local experts' group
  sizes, clamped to the buffer; valid [R, 1] bool, rows inside a group; pos
  [T x K] int32, each (token, choice) pair's row, R for a pair of another
  shard's expert or one past the buffer; order [R] int32, the flat (token,
  choice) index of each row).  R = `rows`, or T x K (dropless) when None.  A
  stable argsort with other shards' experts keyed last puts the local pairs
  first, by expert, and in token order within an expert, as the ring's sort
  orders them."""
  T, K = idx.shape
  R = T * K if rows is None else min(rows, T * K)
  le = idx.reshape(-1) - shard * n_local
  local = (le >= 0) & (le < n_local)
  key = jnp.where(local, le, n_local).astype(jnp.int32)
  order = jnp.argsort(key, stable=True).astype(jnp.int32)
  cnt = jnp.bincount(key, length=n_local + 1)[:n_local].astype(jnp.int32)
  gs = truncate_group_sizes(cnt, R)
  n_valid = jnp.sum(gs)
  inv = jnp.zeros((T * K,), jnp.int32).at[order].set(jnp.arange(T * K, dtype=jnp.int32), unique_indices=True)
  pos = jnp.where(local & (inv < n_valid), inv, R)
  order = order[:R]
  valid = (jnp.arange(R, dtype=jnp.int32) < n_valid)[:, None]
  return (order // K).astype(jnp.int32), gs, valid, pos, order


def _take_rows(a, pos):
  """a[pos] over a's rows, zero for pos == len(a) (a gather; the fill keeps it
  in bounds without a padded copy of `a`)."""
  return jnp.take(a, pos, axis=0, mode="fill", fill_value=0)


def _masked(valid, x, fill):
  """jnp.where(valid, x, fill) -- or x itself where `valid` is None: a compact buffer every row of which is inside a
  group (EP 1 and a dropless buffer: _tokamax_experts' `full`), where the kernels write every row."""
  return x if valid is None else jnp.where(valid, x, fill)


def _token_sum(y, pos, w, T, K, dtype, impl):
  """sum_k w[t, k] y[pos[t, k]] for each token t (w None: unweighted) in fp32, cast to `dtype`; pos [T x K] is R
  (y's length) for a choice with no row.  `impl` StageConfig.combine_impl."""
  if impl == "triton":
    from maxtext.experimental.mpmd.jax import gpu_kernels

    return gpu_kernels.gather_sum(y, pos.reshape(T, K), w, out_dtype=dtype)
  yk = _take_rows(y, pos).reshape(T, K, y.shape[-1]).astype(jnp.float32)
  if w is None:
    return yk.sum(axis=1).astype(dtype)
  return jnp.einsum("tkd,tk->td", yk, w.astype(jnp.float32)).astype(dtype)


def _gather_rows(xf, src, valid):
  """Token src[r] of `xf [T, D]` in row r of the compact buffer, zero past the groups (`valid`)."""
  return _masked(valid, jnp.take(xf, src, axis=0), 0).astype(xf.dtype)


def _to_pairs(d_row, pos, order, T, K):
  """Per-row values `d_row [R]` placed at their rows' (token, choice) pairs -> [T, K] fp32, zero for a pair with no
  row inside the groups.  The rows inside the groups are those whose pair points back at them (pos), each at a pair
  of its own; the others' values go to distinct indices past the end, which drop them -- so the indices stay unique,
  no scatter-add is needed, and what the rows past the groups hold (other pairs', or the garbage an unmasked buffer
  leaves there) reaches no pair."""
  rows = jnp.arange(order.shape[0], dtype=order.dtype)
  here = jnp.take(pos, order, axis=0) == rows.astype(pos.dtype)
  return (
      jnp.zeros((T * K,), jnp.float32)
      .at[jnp.where(here, order, T * K + rows)]
      .set(d_row, mode="drop", unique_indices=True)
      .reshape(T, K)
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _dispatch_rows(xf, src, pos, valid, K, impl="xla"):
  """The compact buffer's rows: token src[r] of `xf [T, D]` in row r, zero past
  the groups.  Backward: a token's gradient is the sum of its rows' (pos),
  gathered, in fp32 -- not the scatter-add autodiff would emit (`impl`:
  StageConfig.combine_impl).  pos indexes the rows inside the groups only, so
  the gradient's rows past them are never read (and need no mask)."""
  return _gather_rows(xf, src, valid)


def _dispatch_rows_fwd(xf, src, pos, valid, K, impl="xla"):
  return _dispatch_rows(xf, src, pos, valid, K, impl), (src, pos, valid, xf.shape[0])


def _dispatch_rows_bwd(K, impl, res, g):
  src, pos, valid, T = res
  return _token_sum(g, pos, None, T, K, g.dtype, impl), None, None, None


_dispatch_rows.defvjp(_dispatch_rows_fwd, _dispatch_rows_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _combine_rows(y, wts, src, pos, valid, order, dtype, impl="xla"):
  """Each token's output from the buffer: sum over its choices k of wts[t, k] x
  y[pos[t, k]] in fp32 (a choice of another shard's expert reads zero), cast
  to `dtype` -- the ring's unsort and weighted sum (`impl`:
  StageConfig.combine_impl).  Backward: a row's gradient is its token's, times
  its weight (src, gathered); a weight's, its token's gradient dotted with its
  row -- under 'triton' computed for the R rows and placed at their (token,
  choice) by `order` rather than for all T x K choices (zero but for the rows
  here; the same dots, reduced in another order)."""
  T, K = wts.shape
  return _token_sum(y, pos, wts, T, K, dtype, impl)


def _combine_rows_fwd(y, wts, src, pos, valid, order, dtype, impl="xla"):
  return _combine_rows(y, wts, src, pos, valid, order, dtype, impl), (y, wts, src, pos, valid, order)


def _combine_rows_bwd(dtype, impl, res, g):
  y, wts, src, pos, valid, order = res
  T, K = wts.shape
  w_row = jnp.take(wts.reshape(-1), order, axis=0).astype(jnp.float32)
  g_row = jnp.take(g, src, axis=0)
  dy = _masked(valid, g_row.astype(jnp.float32) * w_row[:, None], 0).astype(y.dtype)
  if impl == "triton":
    dw = _to_pairs(jnp.einsum("rd,rd->r", y, g_row, preferred_element_type=jnp.float32), pos, order, T, K)
  else:
    yk = _take_rows(y, pos).reshape(T, K, y.shape[-1]).astype(jnp.float32)
    dw = jnp.einsum("tkd,td->tk", yk, g.astype(jnp.float32))
  return dy, dw.astype(wts.dtype), None, None, None, None


_combine_rows.defvjp(_combine_rows_fwd, _combine_rows_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _sum_rows(y, src, pos, valid, K, impl="xla"):
  """Each token's output from the buffer, unweighted: sum over its choices k of
  y[pos[t, k]] in fp32 (a choice with no row here adds nothing), rounded once
  to y's dtype -- the combine under route_weights 'experts', whose rows carry
  their weights already (`impl`: StageConfig.combine_impl).  _dispatch_rows'
  transpose: a row's gradient is its token's (src, gathered; zero past the
  groups), so the backward reads no row of `y` and the VJP keeps none."""
  return _token_sum(y, pos, None, pos.shape[0] // K, K, y.dtype, impl)


def _sum_rows_fwd(y, src, pos, valid, K, impl="xla"):
  return _sum_rows(y, src, pos, valid, K, impl), (src, valid)


def _sum_rows_bwd(K, impl, res, g):
  src, valid = res
  return _gather_rows(g, src, valid), None, None, None


_sum_rows.defvjp(_sum_rows_fwd, _sum_rows_bwd)


@jax.custom_vjp
def _row_weights(wts, pos, order):
  """Each buffer row's routing weight, wts[t, k] of its (token, choice) pair
  order[r] (wts [T, K] fp32): what route_weights 'experts' scales the rows of
  the gated activation by.  Backward: a weight's gradient is its row's,
  placed by `_to_pairs` -- zero for a pair with no row inside the groups --
  rather than autodiff's scatter-add, which would add the rows past the
  groups' (other pairs' weights, or garbage in an unmasked buffer) into their
  pairs."""
  return jnp.take(wts.reshape(-1), order, axis=0)


def _row_weights_fwd(wts, pos, order):
  return _row_weights(wts, pos, order), (pos, order, wts.shape[0], wts.shape[1])


def _row_weights_bwd(res, g):
  pos, order, T, K = res
  return _to_pairs(g, pos, order, T, K), None, None


_row_weights.defvjp(_row_weights_fwd, _row_weights_bwd)


def _tokamax():
  try:
    import tokamax
  except ImportError as e:
    raise ImportError(
        "gmm_impl='tokamax' needs tokamax (pip install tokamax; on GPU also jax-triton and "
        "triton for its Triton kernels)"
    ) from e
  return tokamax


def _tokamax_impls(impl):
  """(forward / dlhs backends, dW backends) for tokamax.ragged_dot_general:
  Mosaic GPU has no dW kernel on sm80 / sm100, so dW runs on Triton there."""
  if impl == "xla":
    return ("xla",), ("xla",)
  if impl in ("mosaic", "tokamax_mosaic"):
    return ("mosaic", "triton"), ("triton",)
  return ("triton",), ("triton",)


def _own_mosaic(impl, lhs, rhs):
  """Do this product's forward and dlhs run on gpu_kernels.grouped_matmul: tokamax_gmm 'mosaic' where it takes
  them (an sm100 GPU, bf16 both sides, weight dims multiples of 64) -- else on tokamax's backends (_tokamax_impls)."""
  if impl != "mosaic":
    return False
  from maxtext.experimental.mpmd.jax import gpu_kernels

  return gpu_kernels.grouped_matmul_supported(lhs, rhs)


def _ragged_dw(impl, lhs, g, gs, start):
  """start + lhs^T @ g per group, accumulated in place in start's dtype (gpu_kernels.ragged_dw, one rounding), on
  the kernel tokamax_gmm `impl` picks: ragged_dw's tcgen05 one under 'mosaic' only, as grouped_matmul (_own_mosaic;
  on sm100 where it takes the shapes, an fp32 start too), its Triton one under 'tokamax_mosaic' and 'triton', lax's
  form under 'xla'.  Every dW with a starting value runs here: _tokamax_gmm_bwd's (dw0, the handed-on copy's) and
  _dw32_backward's (an accumulator of its own dtype)."""
  from maxtext.experimental.mpmd.jax import gpu_kernels

  return gpu_kernels.ragged_dw(lhs, g, gs, start, mosaic=None if impl == "mosaic" else False, xla=impl == "xla")


def _tokamax_dot(lhs, rhs, gs, dims, out_dtype, impls):
  tk = _tokamax()
  return tk.ragged_dot_general(
      lhs, rhs, gs, ragged_dot_dimension_numbers=dims, preferred_element_type=out_dtype, implementation=impls
  )


_DEFAULT_DIMS = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[0]
)
_CONTRACTING_DIMS = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[]
)


def tokamax_gmm(name, impl, lhs, rhs, gs, valid, acc=False, stacked=None, dw0=None, keep=None):
  """`lhs[r] @ rhs[g]` for each row r of local group g of a compact buffer
  (`_local_plan`: rows sorted by group from row 0, `gs` the local groups'
  sizes, rows past them `valid` False, which come out zero), on tokamax's
  kernels (`impl`: StageConfig.tokamax_gmm), bf16 out of fp32 accumulation
  like megablox.  Named for xprof as named_gmm's (`tokamax/name`, `_dx`, `_dw`).

  acc=True -> (out, rhs), named_gmm's: the handed-on copy's weight gradient is
  added to this call's dW, in fp32, before the rounding.  stacked=(w, l): the
  forward reads block l of the chunk's stacked weight w in place (the group
  sizes padded with the other blocks' empty groups, named_gmm's trick), where
  a slice would be a copy the kernel's custom call needs; dlhs reads a
  transposed copy of the block either way (tokamax transposes rhs for it).

  dw0 (rhs's shape and dtype): the dW kernel's starting value -- the step
  accumulator's block for this weight (`_DW0`) -- so rhs's cotangent is dw0
  + this call's dW, accumulated in place (gpu_kernels.ragged_dw), as is a
  handed-on copy's (acc), which starts from that copy's cotangent.

  keep (a Recompute of `lhs`): named_gmm's -- what the VJP keeps of the lhs
  for the dW kernel, recomputed in the backward."""
  if dw0 is not None and dw0.shape != rhs.shape:
    raise ValueError(f"tokamax_gmm dw0: a {dw0.dtype}{list(dw0.shape)} start for a {rhs.dtype}{list(rhs.shape)} rhs")
  if dw0 is not None and dw0.dtype != rhs.dtype:
    # An accumulator in a dtype of its own (an fp32 one): a cotangent takes its primal's dtype, so the accumulated
    # value leaves as dw0's cotangent instead -- dw0 is then a primal of the caller's VJP (ChunkPrograms' saved
    # residuals, bwd_core_recompute) and rhs's cotangent is zero.  acc=True hands dw0 itself on, (out, dw0), in
    # place of the weight copy: the half that takes it (batch split's mb1) starts its dW from it, and this call's
    # dW kernel then starts from that half's accumulated value -- one chain, in place, no add pass.
    if stacked is not None:
      raise ValueError("tokamax_gmm: an accumulator start of its own dtype takes no stacked reads")
    return (_tokamax_gmm_dw32_pass if acc else _tokamax_gmm_dw32)(name, impl, None, lhs, rhs, gs, valid, dw0, keep)
  ws, layer = (None, None) if stacked is None else stacked
  if ws is not None and (tuple(ws.shape[1:]) != tuple(rhs.shape) or ws.dtype != rhs.dtype):
    raise ValueError(
        f"tokamax_gmm stacked: block {layer} of a {ws.dtype}{list(ws.shape)} weight is not the "
        f"{rhs.dtype}{list(rhs.shape)} rhs"
    )
  return (_tokamax_gmm_acc if acc else _tokamax_gmm)(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _tokamax_gmm(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep):
  return _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep)[0]


def _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep):
  fwd, _ = _tokamax_impls(impl)
  rk, gk, _ = _gmm_operands(rhs, gs, jnp.int32(0), ws, layer)
  with _kernel_scope(name, "tokamax"):
    if _own_mosaic(impl, lhs, rk):
      from maxtext.experimental.mpmd.jax import gpu_kernels
      # tiles for this block's groups, not the stacked view's (whose other blocks' groups are empty)
      cfg = gpu_kernels.mosaic_config(**gpu_kernels.mosaic_tiles(lhs.shape[0], rhs.shape[0]))
      out = gpu_kernels.grouped_matmul(lhs, rk, gk, out_dtype=lhs.dtype, config=cfg)
    else:
      out = _tokamax_dot(lhs, rk, gk, _DEFAULT_DIMS, lhs.dtype, fwd)
  # rows past the groups are left unwritten by the kernels: select them away
  # before anything multiplies them (garbage x 0 can be NaN)
  out = _masked(valid, out, 0).astype(lhs.dtype)
  return out, (lhs if keep is None else keep, rhs if ws is None else None, gs, valid, ws, dw0)


def _tokamax_gmm_bwd(name, impl, layer, res, g, dw_in=None):
  """dlhs = g @ rhs^T per group, dW = lhs^T @ g per group (ragged
  contraction).  With a starting value -- dw0 (the step accumulator's block),
  dw_in (acc: the handed-on copy's cotangent), or both, added first in fp32 --
  the dW kernel adds into it in place (_ragged_dw: gpu_kernels.ragged_dw on
  the kernel `impl` picks, one rounding); without, tokamax's dW kernel writes
  it fresh, in the weight's dtype."""
  fwd, dw_impls = _tokamax_impls(impl)
  lhs, rhs, gs, valid, ws, dw0 = res
  lhs = _kept(lhs)
  shape, dtype = _rhs_type(rhs, ws)
  rk = rhs if ws is None else ws[layer]
  g = _masked(valid, g, 0).astype(lhs.dtype)
  with _kernel_scope(name + "_dx", "tokamax"):
    if _own_mosaic(impl, g, rk):  # the weight as it lies, contracted on its last dim: no transposed copy
      from maxtext.experimental.mpmd.jax import gpu_kernels

      dx = gpu_kernels.grouped_matmul(g, rk, gs, transpose_rhs=True, out_dtype=lhs.dtype)
    else:
      dx = _tokamax_dot(g, rk.swapaxes(1, 2), gs, _DEFAULT_DIMS, lhs.dtype, fwd)
  start = (
      dw_in
      if dw0 is None
      else dw0
      if dw_in is None
      else (dw0.astype(jnp.float32) + dw_in.astype(jnp.float32)).astype(dtype)
  )
  with _kernel_scope(name + "_dw", "tokamax"):
    if start is None:
      dw = _tokamax_dot(lhs, g, gs, _CONTRACTING_DIMS, dtype, dw_impls)
    else:
      dw = _ragged_dw(impl, lhs, g, gs, start.astype(dtype))
  return _masked(valid, dx, 0).astype(lhs.dtype), dw.astype(dtype), None, None, None, None, None


_tokamax_gmm.defvjp(_tokamax_gmm_fwd, _tokamax_gmm_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _tokamax_gmm_acc(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep):
  return _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep)[0], rhs


def _tokamax_gmm_acc_fwd(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep):
  out, res = _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, ws, dw0, keep)
  return (out, rhs), res


def _tokamax_gmm_acc_bwd(name, impl, layer, res, cts):
  g, dw_in = cts
  return _tokamax_gmm_bwd(name, impl, layer, res, g, dw_in)


_tokamax_gmm_acc.defvjp(_tokamax_gmm_acc_fwd, _tokamax_gmm_acc_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _tokamax_gmm_dw32(name, impl, layer, lhs, rhs, gs, valid, acc, keep):
  """tokamax_gmm with the weight gradient accumulated in place into `acc` (its own dtype, e.g. fp32 -- Megatron's
  gradient-accumulation fusion into main_grad), which leaves as acc's cotangent; rhs's is zero."""
  return _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, None, None, keep)[0]


def _tokamax_gmm_dw32_fwd(name, impl, layer, lhs, rhs, gs, valid, acc, keep):
  out, res = _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, None, None, keep)
  return out, (res, acc)


def _dw32_backward(name, impl, res, g, start):
  """dlhs, and `start` + this call's dW accumulated in place in start's dtype (_ragged_dw: gpu_kernels.ragged_dw on
  the kernel `impl` picks) -> (dx, rhs's zero cotangent, the accumulated value)."""
  lhs, rhs, gs, valid, _, _ = res
  lhs = _kept(lhs)
  fwd, _ = _tokamax_impls(impl)
  g = _masked(valid, g, 0).astype(lhs.dtype)
  with _kernel_scope(name + "_dx", "tokamax"):
    if _own_mosaic(impl, g, rhs):
      from maxtext.experimental.mpmd.jax import gpu_kernels

      dx = gpu_kernels.grouped_matmul(g, rhs, gs, transpose_rhs=True, out_dtype=lhs.dtype)
    else:
      dx = _tokamax_dot(g, rhs.swapaxes(1, 2), gs, _DEFAULT_DIMS, lhs.dtype, fwd)
  with _kernel_scope(name + "_dw", "tokamax"):
    new_acc = _ragged_dw(impl, lhs, g, gs, start)
  return _masked(valid, dx, 0).astype(lhs.dtype), jnp.zeros_like(rhs), new_acc


def _tokamax_gmm_dw32_bwd(name, impl, layer, res_acc, g):
  res, acc = res_acc
  dx, drhs, new_acc = _dw32_backward(name, impl, res, g, acc)
  return dx, drhs, None, None, new_acc, None


_tokamax_gmm_dw32.defvjp(_tokamax_gmm_dw32_fwd, _tokamax_gmm_dw32_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _tokamax_gmm_dw32_pass(name, impl, layer, lhs, rhs, gs, valid, acc, keep):
  """_tokamax_gmm_dw32 that hands its accumulator start on -> (out, acc): the start's cotangent comes back
  holding the receiving call's accumulated value, which this call's dW kernel continues (tokamax_gmm's acc=True
  with an own-dtype dw0)."""
  return _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, None, None, keep)[0], acc


def _tokamax_gmm_dw32_pass_fwd(name, impl, layer, lhs, rhs, gs, valid, acc, keep):
  out, res = _tokamax_gmm_fwd(name, impl, layer, lhs, rhs, gs, valid, None, None, keep)
  return (out, acc), res


def _tokamax_gmm_dw32_pass_bwd(name, impl, layer, res, cts):
  g, acc_ct = cts  # acc_ct: the receiver's start + its dW
  dx, drhs, new_acc = _dw32_backward(name, impl, res, g, acc_ct)
  return dx, drhs, None, None, new_acc, None


_tokamax_gmm_dw32_pass.defvjp(_tokamax_gmm_dw32_pass_fwd, _tokamax_gmm_dw32_pass_bwd)


def _expert_act(h0, h1, w_row, act, dtype):
  """The routed experts' gated activation act(h0) * h1 in `dtype` (layers.gated_product's), each row then scaled by
  its routing weight `w_row [rows]` (fp32) and rounded again under route_weights 'experts'; w_row None: unscaled."""
  a = (layers.activation_fn(act)(h0) * h1).astype(dtype)
  return a if w_row is None else (a.astype(jnp.float32) * w_row[:, None]).astype(dtype)


def _wo_lhs(h0, h1, w_row, act, dtype, recompute=True):
  """wo's lhs, the gated activation (`_expert_act`), and what its grouped matmul's VJP keeps of it (`keep`).  Under
  lean residuals the activation runs under jax.checkpoint and the matmul keeps a Recompute of it, so neither VJP
  keeps more than h0, h1 and w_row (the [rows, moe_mlp] product is recomputed in the backward, elementwise) -- or,
  with `recompute` False (StageConfig.lean_experts), keeps the product itself beside them; else the plain product,
  kept as it is."""
  f = functools.partial(_expert_act, act=act, dtype=dtype)
  if not layers.lean():
    return f(h0, h1, w_row), None
  return jax.checkpoint(f)(h0, h1, w_row), (Recompute(f, (h0, h1, w_row)) if recompute else None)


def _tokamax_experts(w, xf, idx, wts, shard, scfg: StageConfig, pass_weights=False, stacked=None, dw0=None):
  """moe_experts' routed experts under gmm_impl 'tokamax': this shard's rows
  of the gathered tokens `xf [T, D]` (routing `idx`, `wts` [T, K]) in a
  compact buffer (`_local_plan`, ragged_buffer_rows of them; dropless by
  default), the three grouped matmuls on tokamax, and the fp32 sum of each
  token's rows back to [T, D], weighted by `wts` in the combine or, under
  route_weights 'experts', in wo's input rows (`_row_weights`) -> (y, the
  handed-on weights under pass_weights, else {}).  `dw0`: {expert leaf: the
  step accumulator's block}, the dW kernels' starting values (tokamax_gmm's
  dw0; `_DW0`).

  Under lean residuals the wi products keep the gathered tokens (and the
  plan) instead of the buffer's rows wherever these outnumber them, and gather
  the rows again in the backward (`Recompute`)."""
  cfg = scfg.model
  E, K = cfg.num_experts, cfg.num_experts_per_tok
  dt = cfg.dtype
  T = xf.shape[0]
  n_local = E // scfg.ep
  rows = ragged_buffer_rows(scfg, T)
  src, gs, valid, pos, order = _local_plan(idx, shard, n_local, rows)
  if (n_local == E and rows is None) or not scfg.buffer_masks:
    # EP 1, dropless: every (token, choice) pair is this shard's and has a row -- the buffer is exactly full, the
    # kernels write every row, and the masks below are the identity (one pass each spared); or masks off
    # (StageConfig.buffer_masks): nothing reads the rows past the groups
    valid = None
  xs = _dispatch_rows(xf, src, pos, valid, K, scfg.combine_impl)
  passed = {}

  def mm(name, lhs, wname, keep=None):
    st = (stacked[0][wname], stacked[1]) if stacked is not None and stacked[0][wname].dtype == dt else None
    out = tokamax_gmm(
        name,
        scfg.tokamax_gmm,
        lhs,
        w[wname].astype(dt),
        gs,
        valid,
        acc=pass_weights,
        stacked=st,
        dw0=None if dw0 is None else dw0[wname],
        keep=keep,
    )
    if pass_weights:
      out, passed[wname] = out
    return out

  keep_xs = Recompute(_gather_rows, (xf, src, valid)) if layers.lean() and scfg.lean_experts and xs.shape[0] > T else None
  xs = _tie("xs", xs)  # prefetch pins (an identity unless a pinned forward traces)
  h0 = checkpoint_name(mm("gmm_wi_0", xs, "wi_0", keep_xs), "moe_h0")
  h1 = checkpoint_name(mm("gmm_wi_1", xs, "wi_1", keep_xs), "moe_h1")
  w_row = _row_weights(wts.astype(jnp.float32), pos, order) if scfg.route_weights == "experts" else None
  act, keep_act = _wo_lhs(h0, h1, w_row, cfg.mlp_activations[0], dt, scfg.lean_experts)
  y = checkpoint_name(mm("gmm_wo", act, "wo", keep_act), "moe_y")
  y = _tie("y", y)
  if w_row is not None:
    return _sum_rows(y, src, pos, valid, K, scfg.combine_impl), passed
  return _combine_rows(y, wts, src, pos, valid, order, dt, scfg.combine_impl), passed


def sparsecore_offload_available(mesh, axes=None) -> bool:
  """Does libtpu offload (and so run async) an all-gather over `axes` (a mesh
  axis name or a tuple of them; None = all of them) on this sub-mesh?

  It offloads only when the program's chips are the whole slice or a solid
  rectangle whose minimum corner is not chip (0,0,0), in any device order:
  libtpu (0.0.46 and 0.0.48) compiles a program for a smaller subslice
  topology only for such a rectangle (TpuTopologyDescription::
  ComputeSubsliceForAssignment), and its SparseCore offload pass skips any
  program with fewer devices than its compile topology ("Skipping offloading
  collectives to SparseCore because the device assignment does not use all
  devices").  So a sub-mesh holding chip (0,0) -- stage 0 of
  ds_ep_pipeline's 'physical' layout on a v6e-256, any stage on chip (0,0) of
  a tpu7x-8 -- and any non-rectangular set run these collectives
  synchronously on the TensorCore; no flag changes it
  (ds_ep_pipeline.sync_collective_stages, layout 'physical_origin_last').
  Probe the axes the collective in question runs over: size-1 axes are
  dropped (a gather over them is a no-op, never offloaded), and with none left
  the answer is False.  The mesh's own devices decide, not the host's backend:
  a compile-only TPU topology's (AOT) answer is the same from any host.
  """
  if mesh.devices.flat[0].platform != "tpu":
    return False
  axes = mesh.axis_names if axes is None else (axes,) if isinstance(axes, str) else tuple(axes)
  axes = tuple(a for a in axes if mesh.shape[a] > 1)
  if not axes:
    return False
  ax = axes[0] if len(axes) == 1 else axes
  sm = jax.shard_map(
      lambda a: lax.all_gather(a, ax, axis=0, tiled=True), mesh=mesh, in_specs=P(ax), out_specs=P(), check_vma=False
  )
  n = math.prod(mesh.shape[a] for a in axes)
  a = jax.ShapeDtypeStruct((n * 1024, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P(ax)))
  return 'async_execution_thread="sparsecore"' in jax.jit(sm).lower(a).compile().as_text()


def _shared_mlp(params, inputs, cfg: DeepSeekConfig, name="shared"):
  """`deepseek/layers.py:mlp_block`, verbatim, plus the inert names
  `<name>_h0` / `<name>_h1` on its input projections (STASH_POLICIES).  A
  copy, so the independent port the parity tests check against stays as is.
  A dense layer's MLP is the same block (name 'mlp')."""
  hs = [
      checkpoint_name(layers.dense_general(params[f"wi_{idx}"], inputs, cfg), f"{name}_h{idx}")
      for idx in range(len(cfg.mlp_activations))
  ]
  if layers.lean() and tuple(cfg.mlp_activations[1:]) == ("linear",):
    activated = layers.gated_product(hs[0], hs[1], cfg.mlp_activations[0], cfg.dtype)
  else:
    activated = None
    for y, act in zip(hs, cfg.mlp_activations):
      y = layers.activation_fn(act)(y)
      activated = y if activated is None else activated * y
    activated = jnp.asarray(activated, cfg.dtype)
  return layers.dense_general(params["wo"], activated, cfg)


def cp_chunk(scfg: StageConfig):
  """Which chunk of its sequence this device's row is under context
  parallelism (0 without): its index on the 'expert' axis mod cp."""
  return lax.axis_index(EP_AXIS) % scfg.cp if scfg.cp > 1 else 0


def cp_kv_context(scfg: StageConfig):
  """attention.mla's kv_context under context parallelism: this row is chunk
  r of its sequence (`cp_chunk`); the sequence's other cp - 1 chunks sit on
  the neighbouring devices of its 'expert' ring.  Slots [chunk r+1, ...,
  chunk r+cp-1 (mod cp), own chunk r] of kv latents and roped keys: one
  static causal mask with offset (cp - 1) x s then holds for every device --
  the own chunk causal, every other slot wholly visible -- and the keys'
  segment ids take out the slots of the chunks after r (slot j's chunk
  precedes r iff r + j >= cp).  Only the chunks before r come in, latent and
  roped key as one payload (a collective-permute a slot, half of its pairs;
  autodiff sends their gradients back the same way); a slot taken out holds
  zeros.  Against the queries' segment ids (packed sequences) each slot's
  keys carry their chunk's own, permuted along, and -1 where it is taken out;
  without, the queries are one segment, 1 (attention.attention_op), and a
  slot's keys 1 or 0."""
  cp, ep = scfg.cp, scfg.ep

  def ctx(main, krope, seg=None):
    b, s, lora = main.shape
    r = cp_chunk(scfg)
    both = jnp.concatenate([main, krope.reshape(b, s, -1).astype(main.dtype)], axis=-1)
    slots, segs = [], []
    for j in range(1, cp):
      # chunk (q + j) % cp of each sequence -> the device holding its chunk q, where it comes first (q + j >= cp)
      perm = [(g * cp + (q + j) % cp, g * cp + q) for g in range(ep // cp) for q in range(cp) if q + j >= cp]
      slots.append(lax.ppermute(both, EP_AXIS, perm))
      seen = r + j >= cp
      segs.append(
          jnp.broadcast_to(seen.astype(jnp.int32), (b, s))
          if seg is None
          else jnp.where(seen, lax.ppermute(seg, EP_AXIS, perm), -1)
      )
    slots.append(both)
    segs.append(jnp.ones((b, s), jnp.int32) if seg is None else seg)
    kv = jnp.concatenate(slots, axis=1)
    return (
        kv[..., :lora],
        kv[..., lora:].reshape(b, cp * s, *krope.shape[2:]).astype(krope.dtype),
        jnp.concatenate(segs, axis=1),
        (cp - 1) * s,
    )

  return ctx


def layer_fwd_counts(w, x, key, scfg: StageConfig, stacked=None, aux_seqs=0):
  """One DeepSeek MoE decoder layer (`deepseek.DeepSeekMoELayer.__call__`) -- or a
  Qwen3.5-family one, its mixer in attention's place (`qwen_mixer_part`) --
  local -> (output, its router's expert counts [E] over the EP group's
  tokens).

  Under context parallelism (scfg.cp > 1) `x` is one row of seq / cp tokens,
  chunk `cp_chunk` of its sequence: its positions start at that chunk's, and
  attention reaches the sequence's other chunks (`cp_kv_context`).
  `stacked`: moe_experts'; `aux_seqs`: its aux loss rides the output
  (`_ring_moe`).

  Under lean residuals (a saved-residual forward) a DeepSeek layer computes
  the router's scores and the shared expert with its post-attention norm
  (attention_part's `read`), the projections that read hn: the VJP keeps h,
  not hn as their lhs."""
  wr = routed_tree(w, scfg)
  if layers.lean() and not is_qwen(scfg.model):
    h, hn, (scores, shared) = attention_part(
        w, x, scfg, read=lambda hn: (moe_scores(wr, hn, scfg), shared_part(w, hn, scfg))
    )
    routed, counts = _ring_moe(wr, hn, key, scfg, stacked, aux_seqs, scores=scores)
    return (routed + shared) + h, counts
  h, hn = attention_part(w, x, scfg)
  routed, counts = _ring_moe(wr, hn, key, scfg, stacked, aux_seqs)
  return (routed + shared_part(w, hn, scfg)) + h, counts


def layer_fwd(w, x, key, scfg: StageConfig, stacked=None, aux_seqs=0):
  """One MoE decoder layer, local (`layer_fwd_counts` without the counts)."""
  return layer_fwd_counts(w, x, key, scfg, stacked, aux_seqs)[0]


def routed_tree(w, scfg: StageConfig):
  """A MoE layer's router and routed experts, {'gate', 'wi_0', 'wi_1', 'wo'}
  -- what moe_scores and moe_experts read: DeepSeek's 'moe' subtree, a
  Qwen3.5-family layer's 'mlp' / 'routed_experts'."""
  return w["mlp"]["routed_experts"] if is_qwen(scfg.model) else w["moe"]


def shared_part(w, hn, scfg: StageConfig):
  """A MoE layer's shared expert on the experts' input `hn`: DeepSeek's
  (`_shared_mlp`), or a Qwen3.5-family layer's behind its sigmoid gate
  (qwen35/moe.py:sparse_moe_block)."""
  cfg = scfg.model
  if not is_qwen(cfg):
    return _shared_mlp(w["moe"]["shared_experts"], hn, cfg)
  moe = w["mlp"]
  gate = layers.dense_general(moe["shared_expert_gate"], hn, cfg)
  return jax.nn.sigmoid(gate) * _shared_mlp(moe["shared_expert"], hn, cfg)


def dense_layer_fwd(w, x, scfg: StageConfig):
  """One dense decoder layer (DeepSeek-V3's first ones, deepseek.py's dense
  branch): attention_part, then a dense MLP of width mlp_dim (`_shared_mlp`,
  MaxText's MlpBlock) in place of the experts.  It has no EP collective, so a
  batch-split chunk runs it on the whole microbatch, and its post-attention
  point ties no prefetch pin (pins count MoE layers).  Under lean residuals
  the MLP runs with the post-attention norm (attention_part's `read`)."""
  mlp = lambda hn: _shared_mlp(w["mlp"], hn, scfg.model, name="mlp")
  with _untied():
    if layers.lean() and not is_qwen(scfg.model):
      h, _, out = attention_part(w, x, scfg, read=mlp)
      return out + h
    h, hn = attention_part(w, x, scfg)
  return mlp(hn) + h


def attention_part(w, x, scfg: StageConfig, read=None):
  """layer_fwd up to the experts -> (h, hn): the residual stream after
  attention, and its post-attention norm, the experts' input -- a
  Qwen3.5-family layer's mixer instead (`qwen_mixer_part`).

  `read(hn)` (DeepSeek, under lean residuals): what else the layer computes
  from hn than the dispatch, i.e. the projections that read it and what
  follows them elementwise -> (h, hn, read(hn)), the norm and `read` in one
  checkpoint (layers.rms_norm_read), so that the VJP keeps h and recomputes
  hn for the projections' weight gradients.  hn's other reader, the
  dispatch's all-gather, needs nothing of it in the backward.  The prefetch
  pins' tie (`_tie`) then holds hn for the dispatch alone: it must stay out of
  the checkpoint, whose backward would recompute it, pinned weights and all."""
  if is_qwen(scfg.model):
    return qwen_mixer_part(w, x, scfg)
  cfg = scfg.model
  b, S, _ = x.shape
  positions = jnp.broadcast_to(cp_chunk(scfg) * S + jnp.arange(S, dtype=jnp.int32), (b, S))
  # Synthetic data: no packing, all ones -- under context parallelism none,
  # attention_op's then (cp_kv_context would permute these with the latents).
  segment_ids = None if scfg.cp > 1 else jnp.ones((b, S), jnp.int32)
  freqs = rope.freqs_cis_for(cfg, S * scfg.cp)
  attn = attention.mla(
      w["self_attention"],
      x,
      positions,
      segment_ids,
      freqs,
      cfg,
      kv_context=cp_kv_context(scfg) if scfg.cp > 1 else None,
      norm=w["pre_self_attention_layer_norm"],
  )
  h = x + attn
  if read is not None:
    hn, got = layers.rms_norm_read(w["post_self_attention_layer_norm"], h, cfg, lambda hn: (hn, read(hn)))
    return h, _tie("hn", hn), got
  hn = layers.rms_norm(w["post_self_attention_layer_norm"], h, cfg)
  hn = _tie("hn", hn)  # prefetch pins (an identity unless a pinned forward traces)
  return h, hn


def _sharded_embed(table, ids, scfg: StageConfig):
  """`embed` of this device's token ids `ids` [b, S] from the table's own column block (storage P(None, FE): [V,
  emb / devices]): every device looks up every device's tokens in its columns, and an all-to-all hands each device
  its rows' columns.  The same values as the gathered table's lookup, without the table's full-size copy and
  gradient on every device."""
  cfg = scfg.model
  t = table.astype(cfg.dtype)
  every = lax.all_gather(ids, FE, axis=0, tiled=True)
  cols = jnp.dot(jax.nn.one_hot(every, cfg.vocab_size, dtype=cfg.dtype), t) if scfg.iota_embed else t.at[every].get()
  return lax.all_to_all(cols, FE, 0, 2, tiled=True)


def mtp_loss(w, y, tokens, targets, denom, key, scfg: StageConfig, aux_seqs=0):
  """The multi-token-prediction loss of one microbatch (MultiTokenPredictionBlock
  at mtp_num_layers=1, mtp_reuse_input_embedding) and the MTP layer's expert
  counts: position t combines the main decoder's last hidden state `y` (before
  the decoder norm) with the embedding of token t + 1, runs the MTP layer and
  predicts token t + 2 through the shared head; the last position, which has
  no such token, is masked.  -> (mtp_loss_scaling_factor x the masked cross
  entropy summed / `denom`, counts).  `w` is the last chunk's weights (its
  'mtp' subtree and the head); tokens / targets [b, S]; `aux_seqs`: the MTP
  layer's aux loss rides its output, as the decoder's do."""
  cfg = scfg.model
  m = w["mtp"]
  b, S = tokens.shape
  shift = lambda a: jnp.concatenate([a[:, 1:], jnp.zeros((b, 1), a.dtype)], axis=1)  # roll_and_mask
  e = _sharded_embed(m["token_embedder"]["embedding"], shift(tokens), scfg)
  hin = jnp.concatenate(
      [layers.rms_norm(m["embedding_norm"], e, cfg), layers.rms_norm(m["hidden_state_norm"], y, cfg)], axis=-1
  )
  h, counts = layer_fwd_counts(m["layer"], layers.dense_general(m["projection"], hin, cfg), key, scfg, aux_seqs=aux_seqs)
  logits = layers.dense_general(w["logits_dense"], layers.rms_norm(m["final_norm"], h, cfg), cfg)
  logits = logits.astype(jnp.float32)
  lse = jax.nn.logsumexp(logits, axis=-1)
  tl = jnp.take_along_axis(logits, shift(targets)[..., None], axis=-1)[..., 0]
  valid = (jnp.arange(S) < S - 1).astype(jnp.float32)
  return cfg.mtp_loss_scaling_factor * jnp.sum((lse - tl) * valid) / denom, counts


def split_layers(ws, xa, xb, keys, scfg: StageConfig, stacked=None):
  """The layers `ws` (one weight tree each) over the two halves of a
  microbatch's rows, mb0 `xa` and mb1 `xb`, in Lineage's batch-split phase
  order, mb0 one attention block ahead:

      prologue  A0(0)
      layer l   P1: dispatch(mb0) || A1(l)        P2: experts(mb0) || dispatch(mb1)
                P3: combine(mb0) || experts(mb1)  P4: A0(l+1) || combine(mb1)

  A is attention_part and the router's scores -- a Qwen3.5-family layer's
  mixer, Gated DeltaNet or gated attention, in attention's place
  (qwen_mixer_part); dispatch and combine are the ring of experts' all-gather
  and reduce-scatter over 'expert'; experts are the routed experts
  (moe_experts) and the shared one (shared_part: a Qwen layer's behind its
  sigmoid gate).  An optimization
  barrier closes every phase, so one half's collective has the other half's
  compute to run behind.  The last layer's P4 has no attention of mb0 left:
  its combine of mb1 runs alone.  Each half's EP collectives carry its name
  (ep_all_gather/mb0, ...).  Prefetch pins tie mb0's points, which lead; mb1
  traces untied, so `_pinned` counts each point once a layer.  `keys[l]`:
  layer l's routing keys, one a half.  Under split_dw 'acc' mb0's experts
  hand their weights on to mb1's (moe_experts pass_weights) outside the
  barriers, so in the backward mb0's dW kernels add into mb1's dW, and no
  barrier holds that dW.  `stacked`: the stacked expert weights `ws` were
  taken from, layer l at index l (moe_experts').  -> (ya, yb)."""
  bar = lax.optimization_barrier
  half = lambda i: contextlib.nullcontext() if i == 0 else _untied()
  acc = scfg.split_dw == "acc"

  def attn(i, w, x):
    with half(i):
      h, hn = attention_part(w, x, scfg)
      return h, hn, moe_scores(routed_tree(w, scfg), hn, scfg)

  def dispatch(i, hn, scores):
    return moe_dispatch(hn, scores, scfg, f"mb{i}")

  def experts(i, l, xg, sg, key, hn, passed=None):
    """-> ((routed, shared), mb0's handed-on expert weights under 'acc', else
    {}); mb1 runs on `passed` where given."""
    w = ws[l]
    st = None if stacked is None else (stacked, l)
    with half(i):
      wr = routed_tree(w, scfg)
      d0 = _dw0_for(l)
      # an accumulator of its own dtype (fp32; bwd_core_recompute's acc_own): under 'acc' mb0 hands its start on
      # in place of the weight copies (tokamax_gmm's dw32 pass-through)
      own = d0 is not None and any(jnp.dtype(d0[k].dtype) != jnp.dtype(wr[k].dtype) for k in d0)
      wm = {**wr, **passed} if passed and not own else wr
      # mb1's experts run first in the layer's backward: their dW kernels start
      # from the accumulator's blocks (_DW0), and under 'acc' mb0's from mb1's
      if acc and i == 0:
        routed, out = moe_experts(wm, xg, sg, key, scfg, pass_weights=True, stacked=st, dw0=d0 if own else None)
      else:
        start = passed if own and acc and passed else (d0 if i == 1 else None)
        routed, out = moe_experts(wm, xg, sg, key, scfg, stacked=st, dw0=start), {}
      return (routed, shared_part(w, hn, scfg)), out

  def combine(i, y):
    return moe_combine(y, scfg, f"mb{i}")

  n = len(ws)
  # No pipeline chunk comes here without a layer: ChunkPrograms runs a chunk with no MoE layer on the plain path
  # (split = batch_split and n > 0).  A direct call with none gets its two halves back.
  if not n:
    return xa, xb
  ha, hna, sa = attn(0, ws[0], xa)
  for l in range(n):
    w, (ka, kb) = ws[l], keys[l]
    (ga, gsa), (hb, hnb, sb) = bar((dispatch(0, hna, sa), attn(1, w, xb)))
    ea_sha, passed = experts(0, l, ga, gsa, ka, hna)
    (ea, sha), (gb, gsb) = bar((ea_sha, dispatch(1, hnb, sb)))
    ca, (eb, shb) = bar((combine(0, ea), experts(1, l, gb, gsb, kb, hnb, passed)[0]))
    xa = (ca + sha) + ha
    if l + 1 < n:
      (ha, hna, sa), cb = bar((attn(0, ws[l + 1], xa), combine(1, eb)))
    else:
      cb = bar(combine(1, eb))
    xb = (cb + shb) + hb
  return xa, xb


def qwen_mixer_part(w, x, scfg: StageConfig):
  """A Qwen3.5-family layer (`maxtext.experimental.mpmd.jax.qwen35.model.decoder_layer`, MaxText's
  Qwen3_5DecoderLayer) up to the experts -> (h, hn), as attention_part: its
  mixer -- Gated DeltaNet, or gated full attention when `w` holds 'attn' --
  after a zero-centred RMSNorm, with its residual, then the post-attention
  zero-centred norm, the experts' input.  No packing: positions are 0..S-1,
  the attention's segment ids all one, and the GDN's none -- MaxText's GDN
  under packing=false.  Under its default packing MaxText runs its packed GDN
  on the batch's all-one segment ids, the same math in other roundings: fp32's
  in fp32 (maxtext_parity_test.AgainstMaxText.test_qwen_trains_as_maxtext_fp32),
  bf16's in a bf16 compute dtype, where the packed rule's fp32 differences round
  to other bf16 values and, with fp32 weights, the packed conv keeps the fp32
  kernel this one rounds to bf16.  So the parity protocol runs MaxText's
  trainer with packing=false (maxtext_parity's docstring)."""
  cfg = scfg.model
  b, S, _ = x.shape
  hx = q_layers.rms_norm(w["input_layernorm"], x, cfg)
  if "attn" in w:
    positions = jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32), (b, S))
    a = q_attention.gated_attention(w["attn"], hx, positions, jnp.ones((b, S), jnp.int32), cfg)
  else:
    a = q_gdn.gated_delta_net(w["gdn"], hx, None, cfg)
  h = x + a
  hn = q_layers.rms_norm(w["post_attention_layernorm"], h, cfg)
  hn = _tie("hn", hn)  # prefetch pins (an identity unless a pinned forward traces)
  return h, hn


def embed(w, tokens, scfg: StageConfig):
  cfg = scfg.model
  table = w["token_embedder"]["embedding"].astype(cfg.dtype)
  if scfg.iota_embed:  # Embed.__call__ under use_iota_embed
    return jnp.dot(jax.nn.one_hot(tokens, cfg.vocab_size, dtype=cfg.dtype), table)
  return table.at[tokens].get()


class TableRows:
  """A table's weight gradient as the rows it adds: `values` [n, ...] to the rows `ids` [n] of a table of `shape` (a
  repeated id adds each of its rows) -- the gathered embedding's (`embed_grad`).  Its dense form, autodiff's,
  zero-fills the whole table to scatter n rows into it, and an accumulator then adds every row of that: at
  DeepSeek-V3's vocabulary a [129280, 7168] table a device for a microbatch's 4096 rows.  The accumulating programs
  (`ChunkPrograms`' add_acc / new_acc) add the rows into the accumulator instead (`into`): into an fp32 one with a
  scatter-add of the rows as they come, into a bf16 one with a scatter-add of each id's rows summed in fp32 first (a
  sort of the n ids and a segment sum of the rows: nothing the table's size).  On an A100 the scatter-add of the rows
  as they come into a donated bf16 accumulator took 0.65 ms against the table's 5.4; the fp32 sums first are not
  timed on a GPU.  A leaf of the gradient tree, not a pytree: built and consumed in one trace, by the programs' own
  accumulate."""

  def __init__(self, ids, values, shape):
    self.ids, self.values, self.shape = ids, values, tuple(shape)

  def into(self, acc):
    """`acc` -- a table of `shape`, or an accumulator's block of one (leading axes of one) -- plus the rows, added in
    its dtype (where `acc` is donated, in place).  In fp32 (or wider) the rows scatter-add as they come, a repeated
    id's one after the other.  Below fp32 each id's rows are summed in fp32 and the sum added once: one add a row of
    the table a microbatch, as autodiff's table was added.  Added one by one, a bf16 sum rounds a row away once it is
    ~256 times the row (bf16's 8 significand bits), and at grad_reduce 'step' the accumulator takes a frequent token's
    rows from every microbatch of the step."""
    table = acc.reshape(self.shape)
    if jnp.finfo(acc.dtype).bits >= 32:
      return table.at[self.ids].add(self.values.astype(acc.dtype)).reshape(acc.shape)
    # each distinct id once, ascending, with the fp32 sum of its rows; the slots past them hold ids past the table, V
    # plus the slot (so no index repeats), whose zero sums the scatter drops
    n, V = self.ids.shape[0], self.shape[0]
    ids, inv = jnp.unique(self.ids, size=n, fill_value=V, return_inverse=True)
    ids = jnp.where(ids == V, V + jnp.arange(n, dtype=ids.dtype), ids)
    sums = jax.ops.segment_sum(self.values.astype(jnp.float32), inv.reshape(-1), num_segments=n)
    return (
        table.at[ids]
        .add(sums.astype(acc.dtype), mode="drop", indices_are_sorted=True, unique_indices=True)
        .reshape(acc.shape)
    )

  def dense(self, dtype=None):
    """A zero table of `dtype` (default the rows') with the rows added (`into`): in fp32 autodiff's table, by the same
    scatter; below fp32 with a repeated id's rows summed in fp32 first, where autodiff's scatter adds them one by one
    in `dtype`."""
    return self.into(jnp.zeros(self.shape, self.values.dtype if dtype is None else dtype))


def embed_grad(w, tokens, dh, scfg: StageConfig):
  """`embed`'s weight gradient for the cotangent `dh` of its output -> {'token_embedder': {'embedding': ...}}: the
  one-hot product's VJP (iota_embed), or, for the gather, its rows as they come (`TableRows`), in the table's dtype
  as the gather's VJP gives them.  Into an fp32 accumulator the gather's sums are the VJP's where no token repeats on
  a device: a repeated token's rows add into the accumulator one after the other, not summed first.  Into one below
  fp32 (bf16) a token's rows are summed in fp32 and added once a microbatch (`TableRows.into`)."""
  if scfg.iota_embed:
    _, vjp = jax.vjp(lambda we: embed(we, tokens, scfg), {"token_embedder": w["token_embedder"]})
    return vjp(dh)[0]
  table = w["token_embedder"]["embedding"]
  rows = TableRows(tokens.reshape(-1), dh.reshape(-1, dh.shape[-1]).astype(table.dtype), table.shape)
  return {"token_embedder": {"embedding": rows}}


def head_loss(w, h, targets, denom, scfg: StageConfig):
  cfg = scfg.model
  # Qwen's final norm is zero-centred (qwen35/model.py:head), DeepSeek's plain
  hn = (q_layers.rms_norm if is_qwen(cfg) else layers.rms_norm)(w["decoder_norm"], h, cfg)
  logits = layers.dense_general(w["logits_dense"], hn, cfg).astype(jnp.float32)
  lse = jax.nn.logsumexp(logits, axis=-1)
  tl = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
  return jnp.sum(lse - tl) / denom


def _require_resolved(scfg: StageConfig):
  """Stage programs run concrete collectives: EP's 'auto' is only meaningful
  to Pipeline, which resolves it per stage."""
  if scfg.fsdp_collectives not in ("ppermute", "xla") or scfg.ep_collectives not in ("xla", "ppermute"):
    raise ValueError(
        f"stage programs need fsdp_collectives and ep_collectives in (ppermute, xla), got "
        f"{scfg.fsdp_collectives!r} / {scfg.ep_collectives!r}: resolve auto per stage "
        "(ds_ep_pipeline.Pipeline does)"
    )


def _layer_key(layer, seed):
  return jax.random.fold_in(jax.random.fold_in(jax.random.key(0), layer), seed)


def _take(tree, i):
  return jax.tree.map(lambda a: a[i], tree)


def layer_weights(wl, l, kinds=()):
  """Layer l's weights out of a chunk's 'layers' (or 'dense_layers') tree `wl`:
  every leaf of its [n, ...] stacks at l (DeepSeek's stacked layers); for one
  subtree a layer (non-empty `kinds`: `ChunkSpec.kinds` / `dense_kinds`, a
  Qwen3.5-family chunk's or a DeepSeek one's under layer_tree 'per_layer') its
  subtree `wl[qwen_layer_key(l)]` without the layer axis of one -- the common
  leaves and its own mixer's."""
  if not kinds:
    return _take(wl, l)
  return _take(wl[qwen_layer_key(l)], 0)


def stack_layers(per_layer, kinds=()):
  """`layer_weights`' inverse: one tree a layer (each layer's dW, say) back into
  a chunk's 'layers' (or 'dense_layers') tree -- stacked on a leading axis, or,
  with `kinds`, each under its key with a layer axis of one."""
  if not kinds:
    return jax.tree.map(lambda *a: jnp.stack(a), *per_layer)
  return {qwen_layer_key(l): jax.tree.map(lambda a: a[None], t) for l, t in enumerate(per_layer)}


def _named(f, name):
  """Give a program a readable XLA module name (`jit_<name>`) for the profiler."""

  @functools.wraps(f)
  def g(*a, **k):
    return f(*a, **k)

  g.__name__ = g.__qualname__ = name
  return g


# ------------------------------------------------------------ stash policies
# What a stash forward keeps of each layer for its backward; everything else
# is recomputed there from the layer input, as today's backward recomputes
# all of it.  The names are the inert `checkpoint_name` tags: attention's in
# deepseek/attention.py:mla, the experts' in `_ring_moe` and `_shared_mlp`,
# and 'context', the name both splash kernels give their own residuals (the
# attention output and its logsumexp: attention.py passes it as
# residual_checkpoint_name), so no stash backward re-runs the splash forward
# (ds_ep_test.test_stash_recompute_set).  attention_out, context transposed,
# is not kept: the backward redoes the transpose.  (The dot_product and
# reference kernels name nothing; under them attention is recomputed from the
# kept q, k, v.)
# GiB per device per (layer, microbatch) at the v6e config (PP2 x FSDP1 x
# EP2, 16 experts, mbs 2, seq 4096; ds_ep_test.test_stash_bytes_v6e_dims), on
# top of the layer input (0.055) every mode keeps: attn 0.713 (q, k 192 MiB
# each, v 128, splash out 128 + logsumexp 2, out_proj 56, LoRA 32), attn_min
# 0.659, lite 1.354 (+ the expert up-projections' outputs h0, h1 256 MiB each,
# the gathered tokens 112, the shared expert's 16 each), lite_y 2.229 (+
# gmm_wo's [T*K, D] output 896), all 4.973.  Under all but 'all' whatever is
# not named is recomputed.
ATTN_NAMES = (
    "query_wa_proj",
    "mla_q",
    "kv_wa_proj",
    "mla_kv",
    "query_proj",
    "key_proj",
    "value_proj",
    "context",
    "out_proj",
)
MOE_NAMES = ("moe_xg", "moe_sg", "moe_h0", "moe_h1", "shared_h0", "shared_h1")
STASH_POLICIES = {
    "none": (),  # nothing: full remat (today)
    "attn": ATTN_NAMES,
    "attn_min": ATTN_NAMES[:-1],  # not out_proj
    "lite": ATTN_NAMES + MOE_NAMES,  # re-runs gmm_wo of the experts
    "lite_y": ATTN_NAMES + MOE_NAMES + ("moe_y",),  # re-runs no expert matmul
    "all": None,  # no recompute (diagnostics)
}


def stash_policy(name):
  """STASH_POLICIES[name] as a `jax.checkpoint` policy (None: save everything)."""
  names = STASH_POLICIES[name]
  return None if names is None else jax.checkpoint_policies.save_only_these_names(*names)


# ------------------------------------------------------- parts and composer
@dataclasses.dataclass(frozen=True)
class Part:
  """One piece of a per-microbatch program: `body(*args)` returns a tuple of
  outputs inside the stage's shard_map, with a spec and a sharding for each
  argument and output, and the positions of the arguments it donates.  A
  gather part ('G') lists the leaves it returns as `gathers`, ((chunk, leaf
  id), ...); a forward part with prefetch pins carries its program's pin plan
  as `pins`, ((point, layer, chunk, leaf id), ...) (`compose`).  A
  `global_view` part's body runs at global view instead, in the program's jit
  outside any shard_map (a Lineage chunk's F / B / A: ChunkPrograms), taking
  and returning arrays of its specs' types.  A forward part (`forward`)
  returns its chunk's output y, then its residuals -- the saved set, under
  saved residuals -- and in stash mode its stash; a backward part (`backward`)
  donates the dy, saved set and stash it reads (`fwd_like`, among `donate`),
  arrays of those outputs' shapes.  Side by side, an F&B pair, `compose` lists
  the backward's outputs first and keeps those donations off
  (`composed_order`, `composed_donations`)."""

  name: str
  body: object
  in_specs: tuple
  out_specs: tuple
  in_sh: tuple
  out_sh: tuple
  donate: tuple = ()
  gathers: tuple = ()
  pins: tuple = ()
  global_view: bool = False
  forward: bool = False
  backward: bool = False
  fwd_like: tuple = ()


# ------------------------------------------------------------ prefetch pins
# A gather folded into a forward -- a prefetch part's, or the self-prefetch's
# -- has no data dependence on the forward's compute, so XLA's scheduler alone
# decides where it runs.  On a v5p stage the SparseCore runs every offloaded
# collective and kernel of a program as one in-order chain, one at a time, and
# the scheduler put most folded gathers behind the forward's EP collectives,
# the large ones at the program's end (2 x v5p-128, 2026-10).  A pin ties a
# gathered leaf to a point of the forward: the leaf and the forward's value
# there pass one lax.optimization_barrier, so the gather must be done before
# the forward goes on, and the program returns the barrier's output.  A
# barrier is an identity: the values are the same either way.  The points, in
# each layer of the forward:
#   hn  the MoE input, after attention (layer_fwd)
#   xs  the sorted tokens, before the expert gmms (_ring_moe)
#   y   gmm_wo's output, before the unsort (_ring_moe)
PIN_POINTS = ("hn", "xs", "y")
# The v5p recipe's libtpu flags set --xla_tpu_aggressive_opt_barrier_removal=ENABLED,
# which removes every optimization barrier, these too: a program with pins is
# compiled with it DISABLED, per program (jax.jit's compiler_options), so the
# others keep their schedules.  Globally it would also keep the barriers XLA's
# own rematerialization leaves in the fused last chunk (fb_c3 / fba_c3: +2.4-2.6
# ms each in the AOT replay).
PIN_COMPILER_OPTIONS = {"xla_tpu_aggressive_opt_barrier_removal": "DISABLED"}


def pin_compiler_options(mesh):
  """The compiler options of a program with pins on `mesh`'s devices:
  PIN_COMPILER_OPTIONS for TPU devices (a compile-only topology's too), None
  elsewhere -- other backends keep optimization barriers anyway, and refuse
  libtpu's option."""
  return dict(PIN_COMPILER_OPTIONS) if mesh.devices.flat[0].platform == "tpu" else None


class _PinTrace(threading.local):
  """The pins of the forward this thread is tracing (precompile traces
  programs in threads): `regs` {(point, layer): [[(chunk, leaf id), value],
  ...]}, the values each point's barrier takes and hands back; how often each
  point was reached (`seen`), and the points that were (`hit`)."""

  def __init__(self):
    self.regs = self.seen = self.hit = None


_PINS = _PinTrace()


def _tie(point, v):
  """Pin point `point` of the layer being traced: `v` and the gathered leaves
  pinned there pass one optimization barrier, and the registered leaves are
  replaced by its outputs.  `v` alone outside a pinned forward: no op."""
  st = _PINS
  if st.regs is None:
    return v
  layer = st.seen.get(point, 0)
  st.seen[point] = layer + 1
  regs = st.regs.get((point, layer))
  if not regs:
    return v
  out = lax.optimization_barrier((v, *[r[1] for r in regs]))
  for r, o in zip(regs, out[1:]):
    r[1] = o
  st.hit.add((point, layer))
  return out[0]


@contextlib.contextmanager
def _untied():
  """Trace with `_tie` an identity inside, whatever pinned forward is being
  traced (a batch-split layer's mb1: `split_layers`)."""
  st = _PINS
  regs, st.regs = st.regs, None
  try:
    yield
  finally:
    st.regs = regs


@contextlib.contextmanager
def _pinned(regs, plan=None):
  """Trace a forward inside with `regs` ({(point, layer): [[(chunk, leaf id),
  value], ...]}) pinned at its points (`_tie`; the lists are updated in
  place).  Inside another pinned trace -- a composed program's forward that
  gathers some of its own leaves -- they join that one's.  The outermost
  checks on exit that every point with a pin was reached, and that every pin
  of `plan` ((point, layer, chunk, leaf id), ...), if given, was registered."""
  st = _PINS
  if st.regs is not None:
    for k, lst in regs.items():
      st.regs.setdefault(k, []).extend(lst)
    yield
    return
  st.regs, st.seen, st.hit = {k: list(v) for k, v in regs.items()}, {}, set()
  try:
    yield
    missing = sorted(set(k for k, v in st.regs.items() if v) - st.hit)
    if missing:
      raise ValueError(f"prefetch pins at points the forward never reached (point, layer): {missing}")
    got = {key for lst in st.regs.values() for key, _ in lst}
    lost = sorted((c, i) for _, _, c, i in plan or () if (c, i) not in got)
    if lost:
      raise ValueError(f"prefetch pins of leaves no part of the program gathers (chunk, leaf): {lost}")
  finally:
    st.regs = st.seen = st.hit = None


def _pinned_body(parts, spans):
  """`compose`'s body for a program with pins: the gather parts first, their
  leaves registered at the points the forward part pins them to; the forward
  traced with them pinned (it registers its own gathered leaves, if any,
  itself); the gather parts' outputs are the barriers'.  Any other part is
  traced after, outside the pins."""
  pinned = [k for k, p in enumerate(parts) if p.pins]
  if len(pinned) != 1:
    raise ValueError(
        f"a program pins the gathers of one forward part, not {len(pinned)}: " f"{[parts[k].name for k in pinned]}"
    )
  f = pinned[0]
  plan = parts[f].pins
  where = {(c, i): (pt, layer) for pt, layer, c, i in plan}
  outs, regs, slot = [None] * len(parts), {}, {}
  for k, (p, a) in enumerate(zip(parts, spans)):
    if p.gathers:
      (got,) = p.body(*a)
      outs[k] = list(got)
      for i, key in enumerate(p.gathers):
        if key in where:
          regs.setdefault(where[key], []).append([key, outs[k][i]])
          slot[key] = (k, i)
  with _pinned(regs, plan):
    outs[f] = tuple(parts[f].body(*spans[f]))
  for lst in regs.values():  # the entries the barriers updated in place
    for key, v in lst:
      k, i = slot[key]
      outs[k][i] = v
  res = []
  for k, (p, a) in enumerate(zip(parts, spans)):
    if outs[k] is None:
      outs[k] = tuple(p.body(*a))
    res.extend((outs[k],) if p.gathers else outs[k])
  return tuple(res)


def compose(mesh, parts, name, keep_barriers=False, compiler_options=None):
  """ONE shard_map and ONE jit running `parts` side by side.

  The arguments and outputs are the parts', concatenated in order, and each
  part's donations shift to its arguments' positions -- but an F&B pair (a
  forward beside a backward) returns its backward's outputs first
  (`composed_order`), and its backward donates no more than its accumulator
  (`composed_donations`).  The parts share no
  values, so XLA is free to interleave them: a forward with another chunk's
  gather (= fwd_prefetch), a backward with a reduce-scatter (= bwd_reduce),
  or the forward of one chunk with the backward of another (an F&B pair).
  A forward part with prefetch pins (`Part.pins`) ties the gathered leaves to
  its points instead (`_pinned_body`), and the program compiles with
  `pin_compiler_options` -- so on TPU a pinned program is a top-level jit
  only: jax refuses to stage a jit with compiler options inside another jit
  or jax.make_jaxpr (trace it with `.trace(...)`, compile it with
  `.lower(...)`).  `keep_barriers`: compile with those options whatever the
  parts (a batch-split chunk's phases are optimization barriers too);
  `compiler_options`: more of them, over those (a Lineage chunk's,
  ds_ep_lineage.compiler_options).

  A program with a `global_view` part (a Lineage chunk's F, B or A) is a
  mixed one: ONE jit at global view.  It calls every global_view part's body
  as it is (in an F&B pair the other chunk's part is one too: a stage runs
  Lineage's layers for all its chunks or for none) and runs each other part
  (a G or an R) in a shard_map of its own over the stage mesh -- the
  per-device body it would run in the one shard_map -- side by side, sharing
  no values as ever.  No pins there: Lineage's layers have no point to tie a
  gather to.
  """
  parts = tuple(parts)
  n_in = [len(p.in_specs) for p in parts]
  pins = any(p.pins for p in parts)
  mixed = any(p.global_view for p in parts)
  if mixed and pins:
    raise ValueError(
        f"{name}: prefetch pins tie points of ds_ep's own layers, not of a global-view part's "
        f"({[p.name for p in parts if p.global_view]}: Lineage's layers)"
    )
  order = composed_order(parts)
  if pins and order != tuple(range(len(parts))):
    raise ValueError(
        f"{name}: an F&B pair folds in no gather, so it pins none "
        f"({[p.name for p in parts if p.pins]}: prefetch pins)"
    )

  def body(*args):
    if pins:
      starts = [sum(n_in[:k]) for k in range(len(parts))]
      return _pinned_body(parts, [args[s : s + n] for s, n in zip(starts, n_in)])
    outs, i = [], 0
    for p, n in zip(parts, n_in):  # traced in the parts' order, whatever the outputs'
      f = p.body
      if mixed and not p.global_view:
        f = jax.shard_map(f, mesh=mesh, in_specs=p.in_specs, out_specs=p.out_specs, check_vma=False)
      outs.append(tuple(f(*args[i : i + n])))
      i += n
    return tuple(o for k in order for o in outs[k])

  donate = composed_donations(parts)
  cat = lambda field: tuple(x for p in parts for x in getattr(p, field))
  cat_out = lambda field: tuple(x for k in order for x in getattr(parts[k], field))
  opts = pin_compiler_options(mesh) if pins or keep_barriers else None
  if compiler_options:
    opts = {**(opts or {}), **compiler_options}
  prog = (
      body
      if mixed
      else jax.shard_map(body, mesh=mesh, in_specs=cat("in_specs"), out_specs=cat_out("out_specs"), check_vma=False)
  )
  return jax.jit(
      _named(prog, name),
      in_shardings=cat("in_sh"),
      out_shardings=cat_out("out_sh"),
      donate_argnums=donate,
      compiler_options=opts,
  )


def composed_order(parts):
  """The order of the parts whose outputs `compose`'s program returns, one part's after another: the parts' own, but
  in an F&B pair -- a backward (`Part.backward`) beside a forward (`Part.forward`) -- the backward's first, so the
  accumulator leaves it donates go to its own outputs before a forward output of their shape and dtype can take them
  (`composed_donations`)."""
  if not (any(p.forward for p in parts) and any(p.backward for p in parts)):
    return tuple(range(len(parts)))
  return tuple(sorted(range(len(parts)), key=lambda k: not parts[k].backward))


def composed_donations(parts):
  """The argument positions `compose`'s program donates: each part's `donate`, shifted to its arguments' place --
  but none of a backward's dy, saved set and stash (`Part.fwd_like`) beside a forward (`Part.forward`): an F&B pair.
  jit hands each output, in order, the buffer of the first donated argument of its shape and dtype on a device; the
  forward's y has dy's shape, and its saved set and stash have the backward's where the two chunks have the same
  layers.  With the forward's outputs listed first, jit aliased those buffers to them, and as XLA interleaves the two
  parts, it copied each buffer the backward still read before the forward overwrote it (A100, dsv3-mini, 16
  microbatches: ~1700 device copies a step, 85 ms, for the saved sets).  Undonated, they are freed when the program
  ends: one set's bytes, and dy's, held a program longer.  (Donated, dy would now go to the backward's own dx, but at
  gpu_port_test.InterleavedPairs' dims that costs XLA:CPU one copy more a pair.)  The accumulator stays donated, to be
  added to in place, and `compose` lists an F&B pair's backward outputs first (`composed_order`): the accumulator
  leaves go to the backward's own outputs before a forward output of their shape and dtype (a saved activation or a
  stash leaf of a weight's shape, as in fp32 at InterleavedPairs' dims) can take them.  The fused last chunk's part
  ('A') is no forward here: an A&B pair keeps the parts' order, and the A's dx takes the buffer of the x it donates,
  ahead of the backward's dy."""
  fwd = any(p.forward for p in parts)
  donate, i = [], 0
  for p in parts:
    donate.extend(i + d for d in p.donate if not (fwd and d in p.fwd_like))
    i += len(p.in_specs)
  return tuple(donate)


def pdesc(kind, **kw):
  """Hashable descriptor of one part (`ChunkPrograms.part(kind, **kw)`), for
  `ChunkPrograms.program`.  Defaults are dropped, so equal parts share a key;
  `peer=<ChunkPrograms>` takes the part from another chunk on the same mesh."""
  kw = {
      k: tuple(v) if isinstance(v, list) else v for k, v in kw.items() if v is not None and v is not False and v != "none"
  }
  return (kind, tuple(sorted(kw.items())))


@dataclasses.dataclass
class Stash:
  """One chunk's stash programs for one policy (`ChunkPrograms.build_stash`).

    fwd(w, x, seed)                          -> y, res, stash
    bwd(w, x, res, stash, dy, seed)          -> dx?, gacc           (stash, dy donated)
    bwda(w, x, res, stash, dy, gacc, seed)   -> dx?, gacc + dW      (and gacc)

  `stash` is a flat tuple, per layer in order, of the residuals the policy
  keeps, each with a leading per-device axis (spec P(FE, None, ...)).
  """

  policy: str
  specs: tuple  # PartitionSpec per stash array
  avals: tuple  # global ShapeDtypeStructs with shardings (precompile)
  layer_bytes: tuple  # per device and microbatch, per layer
  bytes: int  # their sum
  plans: tuple  # per layer: where each vjp constant comes from
  fwd_body: object
  bwd_core: object
  fwd: object = None
  bwd: object = None
  bwda: object = None


@dataclasses.dataclass
class Split:
  """One middle chunk's backward cut into I and W, for the zero-bubble schedules (`ChunkPrograms.build_split`): bwd's
  body -- the B part of the chunk's first gradient write -- traced to one flat per-device jaxpr and cut by data
  dependence (stage_ad.split_jaxpr).  I computes dx, the remat recompute of the forward included; W every
  accumulator leaf, from the values of I it reads (the crossing set) and the arguments of I it reads too (`kept`).
  One I feeds either W:

    I(w, x, res, dy, seed)            -> dx, crossing                part('I'): bwdi_c{c}
    W(crossing, kept)                 -> gacc                        part('W'): bwdw_c{c}             (acc False)
    W(crossing, kept, gacc)           -> gacc + dW                   part('W', acc=True): bwdwa_c{c}  (gacc donated)
    W(crossing, kept, gacc, sqs)      -> grads, stage sumsq          rs_self=n: bwdwa_c{c}_rsself     (gacc donated)

  Or the fused last chunk's program cut so (`last`, `ChunkPrograms.build_split_last`): fb's body, I its forward,
  head, loss and input gradient with the weight gradients of every accumulator leaf but its layers' (`i_leaves`: the
  head's, the decoder norm's, MTP's, the routers' bias slots, which hold their expert counts), W its layers' (the
  rest, `w_leaves`).  The chunk's accumulator is then two disjoint parts, each a flat tuple of its leaves in leaf
  order -- I's, written by its F (AI), and W's -- which the step merges once both are whole:

    AI(w, x, tgt[, tok], seed)                -> dx?, gk, loss, crossing          part('AI'): fbi_c{c}   (x donated)
    AI(w, x, tgt[, tok], gk, lacc, seed)      -> dx?, gk + dWk, lacc + loss, crossing   part('AI', acc=True): fbai_c{c}
    W(crossing, kept)                         -> gw                               part('W'): bwdw_c{c}
    W(crossing, kept, gw)                     -> gw + dWw                         part('W', acc=True): bwdwa_c{c}

  `acc`: the W this split's is, the chunk's first gradient write (False) or an accumulating one (True) -- the last
  chunk's AI too.  The accumulating halves are the first write's with the accumulator added to their outputs, leaf by
  leaf (`w_adds`; and the last chunk's loss to the loss accumulator): bwda's accumulate is bwd's plus the accumulator
  -- add_acc(g, dW) = g + new_acc(dW) on every leaf of a middle chunk and of the last chunk (the embedding's rows,
  TableRows, which add otherwise, are chunk 0's B's alone) -- so one trace makes both.  Not where the chunk accumulates
  its routed experts' dW in place (ChunkPrograms.inplace_dw: tokamax's dW kernels starting from the accumulator's
  blocks): there the accumulating W is cut from bwda's (fba's) own trace, and a middle chunk's reads the first write's
  I where that trace cuts the same I -- the same program, crossing set and kept arguments -- else its own (`i_acc`
  True: part('I', acc=True)); the last chunk's AI reads the accumulator's kept part, its own for each `acc`.

  `crossing`: a flat tuple, in the order W first reads them, each with a leading per-device axis (spec P(FE, None,
  ...)), as a stash: `crossing_local` the per-device ShapeDtypeStructs, `crossing_avals` the global ones with their
  shardings (precompile).  `kept`: ((group, leaf), ...), the leaves of I's arguments W reads -- group one of
  `groups` (I's arguments: 'w', 'x', 'res', 'dy', 'seed', or the last chunk's 'w', 'x', 'tgt'[, 'tok'], 'seed'),
  leaf the index among that argument's leaves -- which I does not donate (a step keeps them until W instead:
  `kept_of`); `kept_specs` / `kept_avals` theirs.  `i_src`: per invar of `i_jaxpr`, its position among the leaves of
  I's arguments, or the stage_ad.Const it is; `w_src`: per invar of `w_jaxpr` after the crossing set, ('kept', j),
  ('gacc', j) (the j-th leaf of the accumulator W writes: an in-place W's) or a Const.  `i_donate`: what I donates --
  what the B (or A) donates, but an argument W reads.  `bytes`: the crossing set's bytes per device; `kept_bytes`
  those of the kept x / res / dy leaves, which the fused backward would have freed (a step holds w, the seed and the
  batch anyway): one microbatch's W buffer, from its I to its W, is their sum.  `stats`: split_jaxpr's.
  """

  acc: bool
  i_acc: bool
  w_adds: bool
  i_jaxpr: object
  w_jaxpr: object
  i_src: tuple
  w_src: tuple
  crossing_local: tuple
  crossing_specs: tuple
  crossing_avals: tuple
  kept: tuple
  kept_specs: tuple
  kept_avals: tuple
  i_donate: tuple
  bytes: int
  kept_bytes: int
  stats: dict
  groups: tuple = ("w", "x", "res", "dy", "seed")
  last: bool = False
  i_leaves: tuple = ()  # the accumulator leaves I writes (the last chunk's AI), their positions in leaf order
  w_leaves: tuple | None = None  # ... those W writes: None, every one (a middle chunk's)

  def kept_of(self, i_args):
    """The values `kept` names, from I's arguments in `groups` order (w, x, res, dy, seed; the last chunk's w, x, tgt[,
    tok], seed): the arrays themselves (a step keeps the references, no copy)."""
    groups = {g: jax.tree.leaves(a) for g, a in zip(self.groups, i_args, strict=True)}
    return tuple(groups[g][i] for g, i in self.kept)


# `A_log = log A`, A ~ U(A_LOG_RANGE): MaxText's qwen3.py draws A ~ U(0, 16], kept off 0 here.
A_LOG_RANGE = (1e-9, 16.0)


def qwen_init_rule(path):
  """How `StagePrograms.init` draws a Qwen3.5-family leaf, by MaxText's rules
  where they shape the math: 'zeros' for the zero-centred norm scales (input,
  post-attention, final, q / k norms: applied as 1 + scale, so a one would
  double the gain), 'ones' for the GDN's gated output norm and dt_bias,
  'a_log' for A_log, else 'normal' (the kernels and the embedding, N(0,
  scale^2) as every DeepSeek leaf).  The zeros / ones / a_log choice is a
  copy of qwen35.init.leaf_rule's, and A_LOG_RANGE of its constant
  (ds_ep_qwen_test.QwenInit holds them to each other); leaf_rule's 'normal'
  draws by fan_in instead."""
  keys = tuple(str(getattr(k, "key", k)) for k in path)
  name = keys[-1]
  if name == "scale":
    return "ones" if keys[-3:-1] == ("norm", "rms_norm") else "zeros"
  if name == "A_log":
    return "a_log"
  if name == "dt_bias":
    return "ones"
  return "normal"


# ------------------------------------------------------------ stage programs
class ChunkPrograms:
  """The jitted programs of one chunk on its stage mesh.

    fwd(w, x, seed)                    -> y, res
    bwd(w, x, res, dy, seed)           -> dx, gacc           (first microbatch)
    bwda(w, x, res, dy, gacc, seed)    -> dx, gacc + dW      (gacc donated)
    fb / fba (last chunk)              -> dx, gacc, loss     (fused fwd + bwd)

  `x` is the chunk input (tokens for chunk 0), `res` the extra residuals (the
  inputs of layers 1..n-1, plus layer 0's for chunk 0).  Chunk 0 returns no dx.

  Those programs are fixed.  Everything else is composed from `part`s, by
  descriptor (`program`): the stash forward / backward of a policy
  (`build_stash`), a middle chunk's backward cut into I and W for the
  zero-bubble schedules (`build_split`) -- and the fused last chunk's program
  cut into AI and W (`build_split_last`) -- and any of them with another
  chunk's FSDP gather or reduce-scatter folded in, or paired with another
  chunk's forward.

  A batch-split chunk on Lineage's layers (StageConfig.lineage_split) has the
  same programs -- signatures, donations, residuals (chunk 0 keeps its
  embedding's output, the others nothing) -- but each is a jit at global view
  over the Explicit stage mesh, not a shard_map: Lineage's code opens its own
  shard_maps and reads layouts from the arrays' types (ds_ep_lineage).  The
  decoder layers are the chunk's ds_ep_lineage.Region, a custom_vjp whose
  primal and two rules are nested jits that every program of the chunk's
  structure shares, each traced once (so a structure with F and B programs
  traces its forward twice: at the primal's activation-bank size and the
  forward rule's); the embedding and the head are ds_ep's own per-device
  code, each in a shard_map of its own.  A backward is the VJP of the region,
  recomputed from the chunk's input inside the one program -- ds_ep's
  contract keeps one residual a chunk, its input; carrying Lineage's
  residuals from F to B instead would hold about 3.4-5.8 GiB per in-flight
  chunk per device at DeepSeek-V3 scale (estimated: a 128-device stage, 4096
  tokens a half a device) -- and the
  region's weight gradients come out typed as the accumulators are: each
  device's partial, unreduced, no leading axis.  Its composed programs are
  mixed (`compose`): the gathers and reduce-scatters folded in run in
  shard_maps of their own beside the global-view part.
  """

  def __init__(
      self,
      scfg: StageConfig,
      ch: ChunkSpec,
      mesh,
      mbs: int,
      seq: int,
      denom: float,
      lineage_sparsecore: bool | None = None,
  ):
    _require_resolved(scfg)
    self.scfg, self.ch, self.mesh = scfg, ch, mesh
    cfg = scfg.model
    self.n = len(ch.layers)
    # Dense layers -- the model's first: chunk 0's, or under a layer split
    # that counts every layer the first chunks' -- run before the MoE layers.
    self.nd = ch.dense
    # A batch-split chunk on Lineage's layers: global-view programs (above).
    # `lineage_sparsecore`: does its Region run Lineage's SparseCore regions
    # -- does libtpu offload the stage's collectives?  The pipeline's answer
    # (ds_ep_pipeline.Pipeline asks it as it asks the FSDP one, over the
    # whole stage, ahead of time too: libtpu's probe, agreed across
    # processes; the device rule only where the probe fails; or its
    # stage_lineage_sparsecore, which ds_ep_aot --transfers-only sets to the
    # rule's); None: ds_ep_lineage.region probes the mesh itself when a
    # program first traces.
    self.lineage = scfg.lineage_split
    self.lineage_sc = None if lineage_sparsecore is None else bool(lineage_sparsecore)
    # A batch-split chunk's phases are optimization barriers: on TPU its
    # programs compile with them kept, as a pinned one (pin_compiler_options).
    self._copts = pin_compiler_options(mesh) if scfg.batch_split else None
    if self.lineage:
      # Lineage orders its phases with program_order's barriers, which its
      # backward depends on; and its splash dkv kernel wants the scoped-VMEM
      # limit of its recipe (ds_ep_lineage.compiler_options: None off TPU).
      from maxtext.experimental.mpmd.jax import ds_ep_lineage  # lazily: it imports this module

      self._copts = {**(self._copts or {}), **(ds_ep_lineage.compiler_options(mesh) or {})} or None
    # The routed-bias update: each MoE layer's expert counts take its router
    # bias's gradient slot (count_leaves).  Without an update rate the bias
    # stays where it starts, and its slot keeps autodiff's zeros.
    self.counting = ch.routed_bias and cfg.routed_bias_update_rate > 0
    if self.counting and ch.kinds and not is_qwen(cfg):
      raise ValueError(
          "the routed-bias update (routed_bias_update_rate > 0) files a stacked DeepSeek chunk's expert "
          "counts (put_counts): use layer_tree 'stacked'"
      )
    # MTP: the fused last chunk also takes the microbatch's input tokens.
    self.mtp = ch.mtp
    # The sequence-wise aux loss: each MoE layer's output carries its share
    # (`_ring_moe`), the global batch's sequences its denominator.
    self.aux_seqs = 0
    if cfg.load_balance_loss_weight > 0:
      if not cfg.moe_use_megatron_seq_aux_loss or scfg.cp > 1:
        raise ValueError(
            "load_balance_loss_weight > 0: the pipeline runs Megatron-LM's sequence-wise aux loss "
            "(moe_use_megatron_seq_aux_loss), without context parallelism (a sequence on one device)"
        )
      self.aux_seqs = int(round(denom / seq))
    if ch.mtp and (ch.mtp != 1 or not ch.last or scfg.cp > 1):
      raise ValueError(
          f"chunk {ch.index}: one MTP layer (mtp_num_layers=1), on the last chunk, without context "
          f"parallelism, not mtp={ch.mtp} (last={ch.last}, cp={scfg.cp})"
      )
    # A chunk of no layer at all -- the embedding's or the head's alone (a layer split's 0 at an end) -- or of
    # dense layers only (chunk 0's, a layer split's 0 there) has no MoE layer to split: it runs the plain path even
    # under batch split (no expert collective to hide behind the other half) -- an empty one with no residual for
    # its backward (its embedding's VJP reads the tokens alone), where the split path would save its output for
    # nothing.
    self.empty = not self.n and not self.nd
    self.split = scfg.batch_split and self.n > 0
    if self.lineage and (self.counting or self.aux_seqs or self.mtp or not self.n):
      # ds_ep_lineage.Region hands back neither Lineage's expert counts nor its router logits (aux.group_sizes /
      # aux.logits would carry them), runs no MTP layer, and batch-splits MoE layers: say which of these it is
      why = [
          w
          for w, bad in (
              (
                  "a chunk without a MoE layer (Lineage batch-splits MoE layers, so every chunk needs one: a layer split "
                  "that counts every layer must not give a chunk dense layers only)",
                  not self.n,
              ),
              ("the routed-bias update (Lineage's expert counts are not wired up)", self.counting),
              ("the sequence-wise aux loss (Lineage's router logits are not wired up)", self.aux_seqs),
              (f"MTP (mtp {self.mtp}: no MTP layer on Lineage's layers)", self.mtp),
          )
          if bad
      ]
      raise NotImplementedError(
          f"chunk {ch.index}: batch_split on Lineage's layers (batch_split_impl 'lineage') "
          "cannot run " + "; ".join(why)
      )
    if scfg.batch_split and (self.counting or self.aux_seqs or self.mtp):
      # split_layers runs moe_experts on each half with neither counts nor an aux loss riding its output
      raise NotImplementedError(
          f"chunk {ch.index}: batch_split runs MoE layers without the routed-bias update, the "
          f"sequence-wise aux loss and MTP ({self.n} MoE layer(s), counting {self.counting}, "
          f"aux loss {bool(self.aux_seqs)}, mtp {self.mtp})"
      )
    ns = lambda spec: NamedSharding(mesh, spec)
    self.cspec = chunk_compute_specs(ch)
    # The weights' spec at the programs' boundaries (cspec typed reduced under
    # shard_mode 'explicit'); the bodies read cspec.
    self.wspec = chunk_wc_specs(scfg, ch)
    self.gspec = acc_specs(scfg, ch)
    # How the reduce-scatters read an accumulator and split their collectives
    # (_scatter_leaf): 'auto''s carries a leading partial axis, a
    # [1, *compute_local] block a device.
    self._rs = dict(lead=scfg.shard_mode != "explicit", per_axis=scfg.rs_axes == "each")
    # How the gathers split their collectives (_gather_leaf's per_axis).
    self._ag = scfg.gather_axes == "each"
    act = P(FE, None, None)
    tok = P(FE, None)
    self.act_spec, self.tok_spec = act, tok
    x_spec = tok if ch.first else act
    # The residuals a backward reads besides x: the inputs of its layers but
    # the first, dense and MoE (and chunk 0's first, its embedding); under
    # batch split only chunk 0's embedding, the backward recomputing the whole
    # split chunk.
    if scfg.batch_split:
      n_dev = math.prod(mesh.shape[a] for a in FE)
      if mbs % n_dev or (mbs // n_dev) % 2:
        raise ValueError(
            f"batch_split runs each device's rows as two halves: a microbatch of {mbs} rows over "
            f"{n_dev} devices is {mbs / n_dev:g} a device, want an even number"
        )
    n_res = (1 if ch.first else 0) if self.split else max(0, self.nd + self.n - (0 if ch.first else 1))
    self.n_res = n_res
    res_spec = (act,) * n_res
    self.res_spec = res_spec
    self.act_aval = jax.ShapeDtypeStruct((mbs, seq, cfg.emb_dim), cfg.dtype)
    self.tok_aval = jax.ShapeDtypeStruct((mbs, seq), jnp.int32)
    self.res_avals = tuple(jax.ShapeDtypeStruct(self.act_aval.shape, cfg.dtype, sharding=ns(act)) for _ in range(n_res))
    # one residual set's bytes a device (save_residuals: the VJP's residuals instead, set below)
    self.res_bytes = (
        n_res * (mbs // math.prod(mesh.shape[a] for a in FE)) * seq * cfg.emb_dim * jnp.dtype(cfg.dtype).itemsize
    )
    self._parts, self._programs, self._stash = {}, {}, {}
    self._splits = {}  # per acc (build_split)
    self.stash_avals, self.stash_bytes = {}, {}  # per policy (build_stash)
    loss_spec = P(FE)
    self.loss_spec = loss_spec
    if self.lineage:
      if scfg.save_residuals:
        raise ValueError(
            "save_residuals: not on Lineage's layers (batch_split_impl 'lineage'), whose region keeps "
            "residuals of its own"
        )
      self._init_lineage(x_spec, denom)
      return
    scalar = P()
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    acc_dt = scfg.acc_dtype
    layer_ids = ch.layers

    def stack_dw(dws, dwd):
      """The MoE and the dense layers' weight gradients, in layer order -> the chunk's layer trees (stacks, or one
      subtree a layer: `kinds`, `dense_kinds`)."""
      dw = {}
      if self.n:
        dw["layers"] = stack_layers(dws, ch.kinds)
      if self.nd:
        dw["dense_layers"] = stack_layers(dwd, ch.dense_kinds)
      return dw

    def dense_weights(w):
      """The chunk's dense layers' weights, one tree each, from its 'dense_layers' tree."""
      return [layer_weights(w["dense_layers"], l, ch.dense_kinds) for l in range(self.nd)]

    def with_counts(dwl, counts):
      """A MoE layer's gradient with its router bias's slot holding the layer's expert counts (fp32: exact)."""
      gate = dict(dwl["moe"]["gate"], bias=counts.astype(jnp.float32))
      return {**dwl, "moe": {**dwl["moe"], "gate": gate}}

    # A Qwen3.5-family chunk's layers are subtrees of their own (a layer axis
    # of one): a layer's expert leaf is already a whole buffer, read in place.
    stacked_reads = scfg.expert_weights == "stacked" and not ch.kinds and self.n > 0

    def experts_of(wls):
      """The chunk's stacked expert weights from its layer tree `wls`, for
      moe_experts' stacked; None under expert_weights 'sliced'."""
      return {k: wls["moe"][k] for k in EXPERT_LEAVES} if stacked_reads else None

    def at(st, l):
      return None if st is None else (st, l)

    def dense_run(wd, h):
      """The chunk's dense layers `wd` (one tree each) over all of h."""
      for wl in wd:
        h = dense_layer_fwd(wl, h, scfg)
      return h

    def split_run(wl, h, seed, stacked=None, wd=()):
      """The chunk's dense layers `wd` over all of h, then its MoE layers
      `wl` (one tree each) over h's rows on this device as two halves
      (split_layers), each layer's halves routed with keys of their own."""
      h = dense_run(wd, h)
      keys = [tuple(jax.random.fold_in(_layer_key(lid, seed), i) for i in (0, 1)) for lid in layer_ids]
      half = h.shape[0] // 2
      ya, yb = split_layers(wl, h[:half], h[half:], keys, scfg, stacked)
      return jnp.concatenate([ya, yb], axis=0)

    def run_layers(w, h, seed, save, remat=False, counts=False):
      """The chunk's dense layers, then its MoE layers, over h -> (the chunk's
      output, each layer's input if `save`, and with `counts` the expert
      counts of its MoE layers, in order)."""
      # Under stacked reads a layer's slices are taken inside the (remat)
      # function, which gets the stacked tree: its expert slices are then
      # never read (megablox reads the stacked leaves), and XLA drops them --
      # under remat too, whose barriers would otherwise take them.
      wd = dense_weights(w)
      if self.split:  # (no counts: refused above)
        if stacked_reads:

          def f(wls, h, wd):
            return split_run([_take(wls, l) for l in range(self.n)], h, seed, experts_of(wls), wd)

          args = (w["layers"], h, wd)
        else:

          def f(wl, h, wd):
            return split_run(wl, h, seed, wd=wd)

          args = ([layer_weights(w["layers"], l, ch.kinds) for l in range(self.n)], h, wd)
        if remat:
          f = jax.checkpoint(f)
        return f(*args), [h] if save else []
      hs, cnts = [], []
      for l in range(self.nd):
        if save:
          hs.append(h)
        f = functools.partial(dense_layer_fwd, scfg=scfg)
        if remat:
          f = jax.checkpoint(f)
        h = f(wd[l], h)
      fwd = layer_fwd_counts if counts else layer_fwd
      for l in range(self.n):
        if save:
          hs.append(h)
        if stacked_reads:

          def moe_f(wls, h, l=l):
            return fwd(_take(wls, l), h, _layer_key(layer_ids[l], seed), scfg, at(experts_of(wls), l), self.aux_seqs)

          args = (w["layers"], h)
        else:
          moe_f = functools.partial(fwd, key=_layer_key(layer_ids[l], seed), scfg=scfg, aux_seqs=self.aux_seqs)
          args = (layer_weights(w["layers"], l, ch.kinds), h)
        if remat:
          moe_f = jax.checkpoint(moe_f)
        with _context(_DW0_LAYER, l):  # in-place dW accumulation (_DW0): this layer's blocks
          if counts:
            h, cnt = moe_f(*args)
            cnts.append(cnt)
          else:
            h = moe_f(*args)
      return (h, hs, cnts) if counts else (h, hs)

    if scfg.grad_reduce == "microbatch":
      # Reduce-scatter this microbatch's partial gradient to the storage layout
      # right here (XLA overlaps the early-finished MoE part with the rest of
      # the backward) and accumulate the sharded, reduced result.
      ss_, cs_, cm_ = chunk_storage_specs(ch), self.cspec, count_mask(ch)
      # (the embedding's rows, TableRows, as their table: the reduce-scatter takes whole leaves)
      whole = lambda d: d.dense() if isinstance(d, TableRows) else d

      def reduced(dw):
        return jax.tree.map(
            lambda d, s_, c_, k_: _scatter_leaf(
                whole(d)[None],
                s_,
                c_,
                jnp.float32 if k_ else scfg.rs_dtype,
                scfg.fsdp_collectives,
                per_axis=self._rs["per_axis"],
            ).astype(acc_dt),
            dw,
            ss_,
            cs_,
            cm_,
            is_leaf=_arr_leaf,
        )

      # (add_acc(g, dw) is g + new_acc(dw) on every leaf, as below too: Split.w_adds builds an accumulating W so)
      def add_acc(gacc, dw):
        return jax.tree.map(lambda g, d: g + d, gacc, reduced(dw))

      def new_acc(dw):
        return reduced(dw)

    else:
      # A device's partial dW: its block of the leading partial axis ('auto'),
      # or the block of the unreduced-typed accumulator itself ('explicit').
      lift = (lambda d: d[None]) if self._rs["lead"] else (lambda d: d)

      def add_leaf(g, d):
        """g + a device's partial dW `d`: the embedding's rows (TableRows) scatter-added into g where it lies, any
        other dW added whole."""
        return d.into(g) if isinstance(d, TableRows) else g + lift(d).astype(acc_dt)

      # (add_acc(g, dw) is g + new_acc(dw) on every leaf but the embedding's rows, which only chunk 0 has:
      # Split.w_adds builds a middle chunk's accumulating W so)
      def add_acc(gacc, dw):
        return jax.tree.map(add_leaf, gacc, dw)

      def new_acc(dw):
        return jax.tree.map(lambda d: lift(d.dense(acc_dt) if isinstance(d, TableRows) else d).astype(acc_dt), dw)

    # In-place dW accumulation (gmm_impl 'tokamax', `_DW0`): a backward that
    # accumulates hands the accumulator's routed-expert blocks to the layers'
    # grouped matmuls, whose dW kernels add into them -- so those leaves of its
    # dW already hold the accumulator's value and are not added again.  Where
    # the accumulator's dtype is the compute weights' (a cotangent takes its
    # primal's dtype), under auto sharding at the step cadence, without
    # routed-bias counting (whose gate leaves carry counts).
    # Only where each layer's accumulator block is a whole buffer: a Qwen3.5-family
    # chunk's layer subtrees, or a DeepSeek chunk of one MoE layer.  Blocks of one
    # [n, ...] stack (a DeepSeek chunk of several layers) handed to the in-place
    # kernel came out wrong on GB200 (jax 0.12.0.1, jax-triton 0.5.0): the kernels'
    # aliased outputs land on views into the stack, inf / garbage even without
    # donation (gpu_port_test.InplaceDwStacked holds the minimal case and pins the guard).
    saved_chunk = scfg.save_residuals and not ch.last
    inplace_dw = (
        scfg.gmm_impl == "tokamax"
        and scfg.inplace_dw
        and scfg.shard_mode == "auto"
        and (bool(ch.kinds) or self.n == 1)
        and scfg.grad_reduce == "step"
        and not self.counting
        and self.n > 0
        and (
            jnp.dtype(acc_dt) == jnp.dtype(cfg.dtype)
            or (
                scfg.inplace_dw_any_dtype
                and not stacked_reads
                and not ch.last
                and (not self.split or scfg.split_dw == "acc")
            )
        )
        and (not saved_chunk or bool(ch.kinds) and not self.split)
    )
    # An accumulator of its own dtype (fp32): the accumulated value leaves as the start's cotangent, so the starts
    # are primals of the backward's VJP (tokamax_gmm's dw32, which takes no stacked reads) -- in the saved path's
    # forward and in bwd_core_recompute; not in the last chunk's fused loss-and-backward.  Under batch split the
    # halves chain it (split_dw 'acc': mb0 hands its start on to mb1, whose accumulated value mb0's dW continues).
    acc_own = inplace_dw and jnp.dtype(acc_dt) != jnp.dtype(cfg.dtype)
    self.inplace_dw = inplace_dw
    unlift = (lambda g: g[0]) if self._rs["lead"] else (lambda g: g)

    def routed_path(path):
      keys = [getattr(k, "key", getattr(k, "idx", None)) for k in path]
      return (
          len(keys) >= 3 and keys[0] == "layers" and keys[-1] in EXPERT_LEAVES and keys[-2] in ("moe", "routed_experts")
      )

    def dw0_blocks(gacc):
      """{MoE layer: {expert leaf: its block of `gacc`}} (`_DW0`), or None."""
      if not inplace_dw or gacc is None:
        return None
      gl = jax.tree.map(unlift, gacc["layers"])
      out = {}
      for l in range(self.n):
        r = routed_tree(layer_weights(gl, l, ch.kinds), scfg)
        out[l] = {k: r[k] for k in EXPERT_LEAVES}
      return out

    def add_acc_inplace(gacc, dw):
      return jax.tree_util.tree_map_with_path(
          lambda path, g, d: lift(d).astype(acc_dt) if routed_path(path) else add_leaf(g, d), gacc, dw
      )

    def with_dw0(core):
      """`core(*args, seed)` (a backward) traced with `gacc`'s blocks as its dW
      kernels' starting values -> (dx, gacc + dW)."""

      def acc_core(*args, gacc):
        blocks = dw0_blocks(gacc)
        with _context(_DW0, blocks):
          dx, dw = core(*args)
        return dx, (add_acc_inplace(gacc, dw) if blocks is not None else add_acc(gacc, dw))

      return acc_core

    def acc_starts(wl, blocks=None):
      """{MoE layer: {expert leaf: its accumulator start}}: `blocks` (the step accumulator's), else zeros."""
      if not inplace_dw:
        return None
      if blocks is not None:
        return blocks
      return {l: {k: jnp.zeros(routed_tree(wl[l], scfg)[k].shape, acc_dt) for k in EXPERT_LEAVES} for l in range(self.n)}

    def put_routed(t, blk):
      """Layer tree `t` with its routed experts' leaves replaced by `blk`."""
      if is_qwen(cfg):
        mlp = dict(t["mlp"])
        mlp["routed_experts"] = {**mlp["routed_experts"], **blk}
        return {**t, "mlp": mlp}
      return {**t, "moe": {**t["moe"], **blk}}

    if not ch.last and scfg.save_residuals:
      # Saved residuals (StageConfig.save_residuals): the forward runs the chunk's layers under jax.vjp and returns
      # the VJP's residual leaves -- the VJP is a pytree (a CustomNode over its residuals, a jaxpr in its treedef),
      # so its leaves cross the program boundary and the backward rebuilds it from them.  Leaves that ARE the
      # function's primals -- the layers' compute weights, the chunk's input -- are not outputs (the backward has
      # them as arguments: no copy of a weight a microbatch); every other leaf is, with a leading device axis so
      # that one spec, P(FE), places any local shape.  The structure comes from tracing the forward (_saved_meta,
      # filled by every trace: the probe below, then the lowerings).
      if self.counting or stacked_reads or self.aux_seqs:
        # say each setting it does not take, and what to set instead
        why = [
            w
            for w, bad in (
                (
                    "stacked expert reads -- the routed experts' weights read in place from the chunk's stacked "
                    "layer leaves, expert_weights 'stacked' (the default) on layer_tree 'stacked': set "
                    "expert_weights 'sliced' (--expert-weights sliced) or layer_tree 'per_layer' (--layer-tree "
                    "per_layer)",
                    stacked_reads,
                ),
                (
                    "the routed-bias update's expert counts (routed_bias_update_rate > 0): set it to 0 "
                    "(--bias-update-rate 0)",
                    self.counting,
                ),
                (
                    "the sequence-wise aux loss (load_balance_loss_weight > 0): set it to 0 (--lb-loss-weight 0)",
                    self.aux_seqs,
                ),
            )
            if bad
        ]
        raise ValueError(f"chunk {ch.index}: save_residuals (--save-residuals on) does not support " + "; nor ".join(why))
      saved = {}
      self._saved_meta = saved

      def layers_fn(seed):
        def f(wl, h, wd, accs=None):
          # accs (inplace_dw): fp32-or-whatever accumulator starts for each MoE layer's routed experts, primals of
          # the VJP whose cotangents are the accumulated values (tokamax_gmm's dw0 of its own dtype); placeholders
          # here, the step accumulator's blocks in the backward
          with _context(_DW0, accs):
            if self.split:
              return split_run(wl, h, seed, None, wd)
            for l in range(self.nd):
              h = dense_layer_fwd(wd[l], h, scfg)
            for l in range(self.n):
              with _context(_DW0_LAYER, l):
                h = layer_fwd(wl[l], h, _layer_key(layer_ids[l], seed), scfg, None, self.aux_seqs)
            return h

        return f

      def primals_of(w, x):
        h = embed(w, x, scfg) if ch.first else x
        wl = [layer_weights(w["layers"], l, ch.kinds) for l in range(self.n)]
        wd = dense_weights(w)
        return wl, h, wd

      def fwd_body(w, x, seed):
        wl, h, wd = primals_of(w, x)
        accs = acc_starts(wl)
        with layers.lean_residuals():
          y, vjp = jax.vjp(layers_fn(seed), wl, h, wd, accs)
        leaves, treedef = jax.tree_util.tree_flatten(vjp)
        prim = jax.tree_util.tree_leaves((wl, wd))
        accl = jax.tree_util.tree_leaves(accs)
        src, res = [], []
        for v in leaves:
          k = next((i for i, q in enumerate(prim) if v is q), None)
          r = next((i for i, q in enumerate(res) if v is q), None)
          a = next((i for i, q in enumerate(accl) if v is q), None)
          if k is not None:
            src.append(("w", k))
          elif a is not None:  # an accumulator start: the backward's own blocks
            src.append(("acc", a))
          elif v is h:
            src.append(("h",))
          elif v is x:
            src.append(("x",))
          elif v is seed:
            src.append(("seed",))
          elif r is not None:  # the same array twice in the VJP: one output
            src.append(("r", r))
          else:
            src.append(("r", len(res)))
            res.append(v)
        # (Under batch split a chunk without dense layers slices its input inside split_run, so the input's two
        # halves are residuals of their own beside x: one activation more a microbatch in flight.  Recognising them
        # would take split_run's halves as primals -- not done: saved residuals and batch split do not meet where
        # memory is short, at scale.)
        local = tuple(jax.ShapeDtypeStruct(r.shape, r.dtype) for r in res)
        if "treedef" not in saved:  # the probe's trace: the layout every later trace must reproduce
          saved.update(treedef=treedef, src=tuple(src), local=local)
        elif (saved["src"], saved["local"]) != (tuple(src), local):
          raise ValueError(
              f"chunk {ch.index}: save_residuals: a forward variant (folded collectives, pins) traced a "
              "residual layout other than the probe's; its backward would misread it"
          )
        return y, tuple(r[None] for r in res)

      # The probe: trace the forward once (abstractly) for the residuals' structure -> res_spec / res_avals.
      n_dev = math.prod(mesh.shape[a] for a in FE)
      w_av = jax.tree.map(
          lambda sh, sp, d: jax.ShapeDtypeStruct(sh, d, sharding=ns(sp)),
          chunk_shapes(scfg, ch),
          self.wspec,
          leaf_dtype_tree(scfg, ch, "compute"),
          is_leaf=_is_shape,
      )
      x_av = jax.ShapeDtypeStruct(
          self.tok_aval.shape if ch.first else self.act_aval.shape,
          jnp.int32 if ch.first else cfg.dtype,
          sharding=ns(x_spec),
      )
      seed_av = jax.ShapeDtypeStruct((), jnp.int32, sharding=ns(scalar))
      jax.eval_shape(
          jax.shard_map(
              fwd_body, mesh=mesh, in_specs=(self.wspec, x_spec, scalar), out_specs=(act, P(FE)), check_vma=False
          ),
          w_av,
          x_av,
          seed_av,
      )
      res_spec = tuple(P(FE, *([None] * len(r.shape))) for r in saved["local"])
      self.res_spec, self.n_res = res_spec, len(res_spec)
      self.res_avals = tuple(
          jax.ShapeDtypeStruct((n_dev, *r.shape), r.dtype, sharding=ns(sp)) for r, sp in zip(saved["local"], res_spec)
      )
      self.res_bytes = sum(math.prod(r.shape) * jnp.dtype(r.dtype).itemsize for r in saved["local"])

      def bwd_core(w, x, res, dy, seed):
        wl, h, wd = primals_of(w, x)
        prim = jax.tree_util.tree_leaves((wl, wd))
        # the accumulating backwards (with_dw0) hand the step accumulator's blocks in through _DW0; the first
        # backward starts from zeros
        accl = jax.tree_util.tree_leaves(acc_starts(wl, _DW0.get()))
        leaves = []
        for s_ in saved["src"]:
          leaves.append(
              {
                  "w": lambda: prim[s_[1]],
                  "h": lambda: h,
                  "x": lambda: x,
                  "seed": lambda: seed,
                  "acc": lambda: accl[s_[1]],
                  "r": lambda: res[s_[1]][0],
              }[s_[0]]()
          )
        vjp = jax.tree_util.tree_unflatten(saved["treedef"], leaves)
        dwl, dh, dwd, daccs = vjp(dy)
        dwl = list(dwl)
        if acc_own:
          # the routed experts' accumulated dW left through the starts' cotangents (an accumulator of its own
          # dtype: tokamax_gmm's dw32); in the compute dtype it is rhs's cotangent already
          dwl = [put_routed(dwl[l], daccs[l]) for l in range(self.n)]
        dw = stack_dw(dwl, list(dwd))
        if ch.first:
          dw.update(embed_grad(w, x, dh, scfg))
          return None, dw
        return dh, dw

    elif not ch.last:

      def fwd_body(w, x, seed):
        h = embed(w, x, scfg) if ch.first else x
        y, hs = run_layers(w, h, seed, save=True)
        hs = hs if ch.first else hs[1:]
        return y, tuple(hs)

    if not ch.last:
      self._fwd_body = fwd_body
      self.fwd = jax.jit(
          _named(sm(fwd_body, in_specs=(self.wspec, x_spec, scalar), out_specs=(act, res_spec)), f"fwd_c{ch.index}"),
          in_shardings=(jax.tree.map(ns, self.wspec, is_leaf=_is_spec), ns(x_spec), ns(scalar)),
          out_shardings=(ns(act), tuple(ns(r) for r in res_spec)),
          compiler_options=self._copts,
      )

      def bwd_core_recompute(w, x, res, dy, seed):
        hs = list(res) if ch.first else [x] + list(res)
        # The stacked expert weights are closed over, not differentiated: each
        # layer's gradient comes out as its slice's (named_gmm's stacked).
        st = experts_of(w["layers"]) if self.n else None
        wd = dense_weights(w)
        if self.split and acc_own:
          # the accumulator starts are primals (tokamax_gmm's dw32): the step accumulator's blocks (with_dw0's
          # _DW0) or zeros, traced here so their cotangents -- both halves' dW accumulated -- come out of the VJP
          wl0 = [layer_weights(w["layers"], l, ch.kinds) for l in range(self.n)]

          def run_acc(wl, h, wd, accs):
            with _context(_DW0, accs):
              return split_run(wl, h, seed, st, wd)

          _, vjp = jax.vjp(run_acc, wl0, hs[0], wd, acc_starts(wl0, _DW0.get()))
          dws, dy, dwd, daccs = vjp(dy)
          dws = [put_routed(dws[l], daccs[l]) for l in range(self.n)]
        elif self.split:
          _, vjp = jax.vjp(
              lambda wl, h, wd: split_run(wl, h, seed, st, wd),
              [layer_weights(w["layers"], l, ch.kinds) for l in range(self.n)],
              hs[0],
              wd,
          )
          dws, dy, dwd = vjp(dy)
        else:
          # MoE layers' inputs follow the dense layers' in hs
          dws, dwd = [None] * self.n, [None] * self.nd
          for l in reversed(range(self.n)):
            f = functools.partial(
                layer_fwd_counts if self.counting else layer_fwd,
                key=_layer_key(layer_ids[l], seed),
                scfg=scfg,
                stacked=at(st, l),
                aux_seqs=self.aux_seqs,
            )
            if self.counting:
              _, vjp, cnt = jax.vjp(f, layer_weights(w["layers"], l, ch.kinds), hs[self.nd + l], has_aux=True)
              dwl, dy = vjp(dy)
              dws[l] = with_counts(dwl, cnt)
            elif acc_own:
              # the accumulator starts are primals (tokamax_gmm's dw32): the step accumulator's blocks (with_dw0's
              # _DW0) or zeros, traced here so their cotangents -- the accumulated values -- come out of the VJP
              wl_l = layer_weights(w["layers"], l, ch.kinds)
              blocks = _DW0.get()
              start = (acc_starts([wl_l] * self.n) if blocks is None else blocks)[l]

              def f_acc(wl_, h_, acc_l, f=f, l=l):
                with _context(_DW0, {l: acc_l}), _context(_DW0_LAYER, l):
                  return f(wl_, h_)

              _, vjp = jax.vjp(f_acc, wl_l, hs[self.nd + l], start)
              dwl, dy, dacc = vjp(dy)
              dws[l] = put_routed(dwl, dacc)
            else:
              with _context(_DW0_LAYER, l):  # in-place dW accumulation (_DW0): this layer's blocks
                _, vjp = jax.vjp(f, layer_weights(w["layers"], l, ch.kinds), hs[self.nd + l])
              dws[l], dy = vjp(dy)
          for l in reversed(range(self.nd)):
            _, vjp = jax.vjp(functools.partial(dense_layer_fwd, scfg=scfg), wd[l], hs[l])
            dwd[l], dy = vjp(dy)
        dw = stack_dw(dws, dwd)
        if ch.first:
          dw.update(embed_grad(w, x, dy, scfg))
          return None, dw
        return dy, dw

      if not scfg.save_residuals:
        bwd_core = bwd_core_recompute
      out_dx = () if ch.first else (act,)
      self._bwd_core, self._new_acc, self._add_acc = bwd_core, new_acc, add_acc
      # (w, x, res, dy, seed, gacc=) -> (dx, gacc + dW), in place where inplace_dw
      self._bwd_acc = bwd_acc = with_dw0(bwd_core)

      def bwd_body(w, x, res, dy, seed):
        dx, dw = bwd_core(w, x, res, dy, seed)
        return (*(() if ch.first else (dx,)), new_acc(dw))

      def bwda_body(w, x, res, dy, gacc, seed):
        dx, acc = bwd_acc(w, x, res, dy, seed, gacc=gacc)
        return (*(() if ch.first else (dx,)), acc)

      w_sh = jax.tree.map(ns, self.wspec, is_leaf=_is_spec)
      g_sh = jax.tree.map(ns, self.gspec, is_leaf=_is_spec)
      self._w_sh, self._g_sh, self._x_spec = w_sh, g_sh, x_spec
      in_b = (self.wspec, x_spec, res_spec, act, scalar)
      in_b_sh = (w_sh, ns(x_spec), tuple(ns(r) for r in res_spec), ns(act), ns(scalar))
      out_b = (*out_dx, self.gspec)
      out_b_sh = (*(ns(s) for s in out_dx), g_sh)
      dy_don = (() if ch.first else (3,)) + ((2,) if scfg.save_residuals else ())
      self.bwd = jax.jit(
          _named(sm(bwd_body, in_specs=in_b, out_specs=out_b), f"bwd_c{ch.index}"),
          in_shardings=in_b_sh,
          out_shardings=out_b_sh,
          donate_argnums=dy_don,
          compiler_options=self._copts,
      )
      self.bwda = jax.jit(
          _named(sm(bwda_body, in_specs=in_b[:4] + (self.gspec, scalar), out_specs=out_b), f"bwda_c{ch.index}"),
          in_shardings=in_b_sh[:4] + (g_sh, ns(scalar)),
          out_shardings=out_b_sh,
          donate_argnums=dy_don + (4,),
          compiler_options=self._copts,
      )
    else:
      # MTP's positions with a token two ahead: all but each sequence's last.
      mtp_denom = denom * (seq - 1) / seq

      def loss_fn(w, x, tgt, tok, seed):
        h = embed(w, x, scfg) if ch.first else x
        if self.counting:
          y, _, counts = run_layers(w, h, seed, save=False, remat=scfg.remat_fused_last, counts=True)
        else:
          (y, _), counts = run_layers(w, h, seed, save=False, remat=scfg.remat_fused_last), []
        loss = head_loss(w, y, tgt, denom, scfg)
        if self.mtp:
          extra, cnt = mtp_loss(w, y, tok, tgt, mtp_denom, _layer_key(cfg.num_decoder_layers, seed), scfg, self.aux_seqs)
          loss, counts = loss + extra, counts + [cnt]
        return loss, counts

      def put_counts(dw, counts):
        """The routers' bias slots of the fused chunk's gradient <- its layers' expert counts (fp32)."""
        dw = dict(dw)
        if self.n:
          moe = dict(
              dw["layers"]["moe"],
              gate=dict(dw["layers"]["moe"]["gate"], bias=jnp.stack(counts[: self.n]).astype(jnp.float32)),
          )
          dw["layers"] = dict(dw["layers"], moe=moe)
        if self.mtp:
          dw["mtp"] = dict(dw["mtp"], layer=with_counts(dw["mtp"]["layer"], counts[-1]))
        return dw

      def fb_core(w, x, tgt, tok, seed):
        if ch.first:
          (loss, counts), dw = jax.value_and_grad(loss_fn, has_aux=True)(w, x, tgt, tok, seed)
          dx = None
        else:
          (loss, counts), (dw, dx) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(w, x, tgt, tok, seed)
        return dx, put_counts(dw, counts) if self.counting else dw, loss

      out_dx = () if ch.first else (act,)

      def fb_with(w, x, tgt, tok, seed):
        dx, dw, loss = fb_core(w, x, tgt, tok, seed)
        return (*(() if ch.first else (dx,)), new_acc(dw), loss[None])

      def fb_dx_dw(w, x, tgt, tok, seed):
        dx, dw, loss = fb_core(w, x, tgt, tok, seed)
        return (dx, loss), dw

      fb_acc = with_dw0(fb_dx_dw)

      def fba_with(w, x, tgt, tok, gacc, lacc, seed):
        (dx, loss), acc = fb_acc(w, x, tgt, tok, seed, gacc=gacc)
        return (*(() if ch.first else (dx,)), acc, lacc + loss[None])

      # With MTP the fused programs take the tokens after the targets.
      if self.mtp:
        fb_body, fba_body = fb_with, fba_with
      else:

        def fb_body(w, x, tgt, seed):
          return fb_with(w, x, tgt, None, seed)

        def fba_body(w, x, tgt, gacc, lacc, seed):
          return fba_with(w, x, tgt, None, gacc, lacc, seed)

      w_sh = jax.tree.map(ns, self.wspec, is_leaf=_is_spec)
      g_sh = jax.tree.map(ns, self.gspec, is_leaf=_is_spec)
      self._w_sh, self._g_sh, self._x_spec = w_sh, g_sh, x_spec
      self._fb_body, self._fba_body = fb_body, fba_body
      out_f = (*out_dx, self.gspec, loss_spec)
      out_f_sh = (*(ns(s) for s in out_dx), g_sh, ns(loss_spec))
      tk = (tok,) * (1 + bool(self.mtp))  # the targets [, the tokens]
      nt = len(tk) - 1
      self.fb = jax.jit(
          _named(sm(fb_body, in_specs=(self.wspec, x_spec, *tk, scalar), out_specs=out_f), f"fb_c{ch.index}"),
          in_shardings=(w_sh, ns(x_spec), *(ns(t) for t in tk), ns(scalar)),
          out_shardings=out_f_sh,
          donate_argnums=(1,) if not ch.first else (),
          compiler_options=self._copts,
      )
      self.fba = jax.jit(
          _named(
              sm(fba_body, in_specs=(self.wspec, x_spec, *tk, self.gspec, loss_spec, scalar), out_specs=out_f),
              f"fba_c{ch.index}",
          ),
          in_shardings=(w_sh, ns(x_spec), *(ns(t) for t in tk), g_sh, ns(loss_spec), ns(scalar)),
          out_shardings=out_f_sh,
          donate_argnums=((1, 3 + nt, 4 + nt) if not ch.first else (3 + nt, 4 + nt)),
          compiler_options=self._copts,
      )

  def _init_lineage(self, x_spec, denom):
    """__init__'s programs for a batch-split chunk on Lineage's layers (the class docstring): global-view bodies over
    the Explicit stage mesh, the decoder layers the chunk's ds_ep_lineage.Region, the embedding and the head
    ds_ep's per-device code in shard_maps of their own -- with the signatures, shardings and donations of the
    other chunks' fwd / bwd / bwda / fb / fba."""
    from maxtext.experimental.mpmd.jax import ds_ep_lineage  # lazily: it imports this module

    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    act, tok, scalar, loss_spec, res_spec = self.act_spec, self.tok_spec, P(), self.loss_spec, self.res_spec
    acc_dt, c = scfg.acc_dtype, ch.index

    def region():
      """The chunk's Region, looked up when a program traces: ds_ep_lineage.region builds one per chunk structure,
      mesh, capacity factor and SparseCore answer (`lineage_sc`), and with it the nested jits all its programs
      share, importing Lineage then -- so a pipeline builds, and refuses what it refuses, on a MaxText without
      Lineage."""
      return ds_ep_lineage.region(
          scfg, ch, mesh, capacity_factor=scfg.lineage_capacity_factor, sparsecore=self.lineage_sc
      )

    def layers_vjp(w, h, remat=False):
      """The chunk's decoder layers on h -> (y, their VJP: dy -> (dW 'layers', dW 'dense_layers' or None, dh)).
      The routing the region hands back (its aux) is not used: no counts, no aux loss (refused above)."""
      f = jax.checkpoint(region()) if remat else region()
      y, vjp, _ = jax.vjp(f, w["layers"], w.get("dense_layers"), h, has_aux=True)
      return y, vjp

    def with_layers(dw, dwl, dwd):
      return {**dw, "layers": dwl, **({"dense_layers": dwd} if self.nd else {})}

    # At global view an accumulator is typed unreduced like the region's gradients (each device's partial, no
    # leading axis), so these add each device's partial to its partial.  Never jnp.stack an unreduced value here:
    # at jit level jax 0.11.1 multiplies it by the unreduced axes' size product (the region concatenates its per-layer
    # gradients inside a shard_map).
    def new_acc(dw):
      return jax.tree.map(lambda d: d.astype(acc_dt), dw)

    def add_acc(gacc, dw):
      return jax.tree.map(lambda g, d: g + d.astype(acc_dt), gacc, dw)

    if ch.first:
      w_e = {"token_embedder": self.wspec["token_embedder"]}
      of_e = lambda w: {"token_embedder": w["token_embedder"]}
      embed_sm = sm(lambda we, x: embed(we, x, scfg), in_specs=(w_e, tok), out_specs=act)

      def embed_dw(we, x, dh):
        """This device's partial gradient of the embedding table (bwd_core's embedding VJP)."""
        _, vjp = jax.vjp(lambda we: embed(we, x, scfg), we)
        return vjp(dh)[0]

      embed_dw_sm = sm(embed_dw, in_specs=(w_e, tok, act), out_specs={"token_embedder": self.gspec["token_embedder"]})

    w_sh = jax.tree.map(ns, self.wspec, is_leaf=_is_spec)
    g_sh = jax.tree.map(ns, self.gspec, is_leaf=_is_spec)
    self._w_sh, self._g_sh, self._x_spec = w_sh, g_sh, x_spec
    self._new_acc, self._add_acc = new_acc, add_acc
    self._bwd_acc = None  # no in-place dW on Lineage's layers: the parts accumulate as core + add_acc
    out_dx = () if ch.first else (act,)
    if not ch.last:

      def fwd_body(w, x, seed):
        del seed  # Lineage routes top-k: no key
        h = embed_sm(of_e(w), x) if ch.first else x
        y = region()(w["layers"], w.get("dense_layers"), h)[0]
        return y, ((h,) if ch.first else ())

      def bwd_core(w, x, res, dy, seed):
        del seed
        _, vjp = layers_vjp(w, res[0] if ch.first else x)
        dwl, dwd, dh = vjp(dy)
        dw = with_layers({}, dwl, dwd)
        if ch.first:
          return None, {**dw, **embed_dw_sm(of_e(w), x, dh)}
        return dh, dw

      def bwd_body(w, x, res, dy, seed):
        dx, dw = bwd_core(w, x, res, dy, seed)
        return (*(() if ch.first else (dx,)), new_acc(dw))

      def bwda_body(w, x, res, dy, gacc, seed):
        dx, dw = bwd_core(w, x, res, dy, seed)
        return (*(() if ch.first else (dx,)), add_acc(gacc, dw))

      self._fwd_body, self._bwd_core = fwd_body, bwd_core
      self.fwd = jax.jit(
          _named(fwd_body, f"fwd_c{c}"),
          in_shardings=(w_sh, ns(x_spec), ns(scalar)),
          out_shardings=(ns(act), tuple(ns(r) for r in res_spec)),
          compiler_options=self._copts,
      )
      in_b_sh = (w_sh, ns(x_spec), tuple(ns(r) for r in res_spec), ns(act), ns(scalar))
      out_b_sh = (*(ns(s) for s in out_dx), g_sh)
      dy_don = () if ch.first else (3,)
      self.bwd = jax.jit(
          _named(bwd_body, f"bwd_c{c}"),
          in_shardings=in_b_sh,
          out_shardings=out_b_sh,
          donate_argnums=dy_don,
          compiler_options=self._copts,
      )
      self.bwda = jax.jit(
          _named(bwda_body, f"bwda_c{c}"),
          in_shardings=in_b_sh[:4] + (g_sh, ns(scalar)),
          out_shardings=out_b_sh,
          donate_argnums=dy_don + (4,),
          compiler_options=self._copts,
      )
      return
    head_keys = ("decoder_norm", "logits_dense")

    def head(wh, y, tgt):
      """This device's loss partial (head_loss), its partial gradient of the head and the loss's dy of its rows."""
      loss, vjp = jax.vjp(lambda wh, y: head_loss(wh, y, tgt, denom, scfg), wh, y)
      dwh, dy = vjp(jnp.ones_like(loss))
      return loss[None], dwh, dy

    head_sm = sm(
        head,
        in_specs=({k: self.wspec[k] for k in head_keys}, act, tok),
        out_specs=(loss_spec, {k: self.gspec[k] for k in head_keys}, act),
    )

    def fb_core(w, x, tgt, seed):
      """The fused last chunk: the layers' forward (recomputed in their backward under remat_fused_last, as
      run_layers' remat), the head, then the layers' VJP of the head's dy."""
      del seed
      h = embed_sm(of_e(w), x) if ch.first else x
      y, vjp = layers_vjp(w, h, remat=scfg.remat_fused_last)
      loss, dwh, dy = head_sm({k: w[k] for k in head_keys}, y, tgt)
      dwl, dwd, dh = vjp(dy)
      dw = with_layers(dwh, dwl, dwd)
      if ch.first:
        return None, {**dw, **embed_dw_sm(of_e(w), x, dh)}, loss
      return dh, dw, loss

    def fb_body(w, x, tgt, seed):
      dx, dw, loss = fb_core(w, x, tgt, seed)
      return (*(() if ch.first else (dx,)), new_acc(dw), loss)

    def fba_body(w, x, tgt, gacc, lacc, seed):
      dx, dw, loss = fb_core(w, x, tgt, seed)
      return (*(() if ch.first else (dx,)), add_acc(gacc, dw), lacc + loss)

    self._fb_body, self._fba_body = fb_body, fba_body
    out_f_sh = (*(ns(s) for s in out_dx), g_sh, ns(loss_spec))
    self.fb = jax.jit(
        _named(fb_body, f"fb_c{c}"),
        in_shardings=(w_sh, ns(x_spec), ns(tok), ns(scalar)),
        out_shardings=out_f_sh,
        donate_argnums=(1,) if not ch.first else (),
        compiler_options=self._copts,
    )
    self.fba = jax.jit(
        _named(fba_body, f"fba_c{c}"),
        in_shardings=(w_sh, ns(x_spec), ns(tok), g_sh, ns(loss_spec), ns(scalar)),
        out_shardings=out_f_sh,
        donate_argnums=(1, 3, 4) if not ch.first else (3, 4),
        compiler_options=self._copts,
    )

  # ---------------------------------------------- parts and the composer
  def part(self, kind, *, stash=None, acc=False, other=None, leaves=None, rs_self=None, self_leaves=None, pin=None):
    """One `Part` of this chunk's work on its stage (cached), for `compose`:

      'F'  forward           (w, x, seed) -> (y, res[, stash])
           self_leaves: the chunk's first forward with part of its own
           gather folded in, as fwd_self_prefetch:
           (w_rest, x, seed, storage_leaves) -> (y, res[, stash], compute_leaves)
           pin: prefetch pins, ((point, layer, chunk, leaf id), ...): the
           leaves this forward gathers (self_leaves) or a gather part of its
           program gathers, tied to points of its layers (PIN_POINTS; a remat
           forward only: a stash forward runs its layers inside jax.vjp, and
           every other kind refuses a pin)
      'B'  backward          (w, x, res[, stash], dy[, gacc], seed) -> (dx?, gacc)
           rs_self=n (with acc): the chunk's last backward with its own
           reduce-scatter, as bwda_reduce_self:
           (..., gacc, seed, n other sumsqs) -> (dx?, grads, stage sumsq)
      'A'  fused last chunk  (w, x, tgt[, gacc, lacc], seed) -> (dx?, gacc, loss)
      'I'  a middle chunk's split backward, its input gradient (`build_split`):
           (w, x, res, dy, seed) -> (dx, crossing), feeding either W; acc:
           the accumulating trace's own I, where in-place dW cuts another
           (Split.i_acc; else the same part as acc False's)
      'W'  ... its weight gradients   (crossing, kept[, gacc]) -> (gacc,)
           rs_self=n (with acc), as 'B''s: (..., gacc, n other sumsqs) -> (grads, stage sumsq)
           -- the last chunk's, of its layers' leaves (`build_split_last`):
           (crossing, kept[, gw]) -> (gw,), gw a tuple of those leaves
      'AI' the fused last chunk split (`build_split_last`): its forward,
           head, loss and input gradient, the weight gradients of its other
           leaves (gk, a tuple of them) -- (w, x, tgt[, tok][, gk, lacc],
           seed) -> (dx?, gk, loss, crossing)
      'G'  all-gather `leaves` of chunk `other`   (storage leaves,) -> (compute leaves,)
      'R'  reduce-scatter `leaves` of `other`     (gacc leaves,) -> (grads, sumsq)

    `stash` is a STASH_POLICIES name (None or 'none': recompute -- the bodies
    of the fixed programs); `acc` accumulates into a donated gacc (bwda /
    fba); `leaves=None` is all of them.  Donations are the fixed programs':
    dy (but chunk 0's), gacc, lacc, the last chunk's x, the stash, the saved
    set (saved residuals), the gacc leaves reduced -- never a forward's x,
    which its backward still needs; beside a forward, in an F&B pair, a
    backward's dy, stash and saved set are not (`composed_donations`), and
    its outputs come first (`composed_order`).  An I donates what its B does
    but an argument W reads (Split.i_donate), a W the accumulator only; an AI
    what its A does (x, gk, lacc), but an argument W reads.

    A Lineage chunk's F, B and A parts are `global_view` (the class
    docstring), with no stash and no pin; the gather its self_leaves forward
    folds in and the reduce-scatter of its rs_self backward run in shard_maps
    inside their bodies.
    """
    stash = None if stash == "none" else stash
    key = (
        kind,
        stash,
        acc,
        other,
        None if leaves is None else tuple(leaves),
        rs_self,
        None if self_leaves is None else tuple(self_leaves),
        tuple(map(tuple, pin)) if pin else None,
    )
    if key not in self._parts:
      self._parts[key] = self._make_part(*key)
    return self._parts[key]

  def _make_part(self, kind, stash, acc, other, leaves, rs_self, self_leaves, pin=None):
    if pin and self.lineage:
      raise ValueError(
          f"chunk {self.ch.index}: prefetch pins tie points of ds_ep's own layers (hn / xs / y), and a "
          "batch-split chunk on Lineage's layers (batch_split_impl 'lineage') has none: no pin "
          "(ds_ep_pipeline --prefetch-pin off)"
      )
    if pin and kind != "F":
      raise ValueError(f"prefetch pins tie points of a forward ('F'), not of a {kind!r} part")
    if pin and stash is not None:
      raise ValueError(
          f"prefetch pins tie points of a remat forward (its layers traced as they run), not of one in "
          f"stash mode {stash!r}: a stash forward runs each layer inside jax.vjp"
      )
    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    lv = lambda t: jax.tree.leaves(t, is_leaf=_is_spec)
    act, first, c = self.act_spec, ch.first, ch.index
    dt, mode = scfg.model.dtype, scfg.fsdp_collectives
    sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}
    out_dx = () if first else (act,)
    with_dx = lambda d: () if first else (d,)
    if kind in ("G", "R"):
      # oc: the plain compute specs the bodies read; ow, og: the gathered
      # leaves' and the gacc leaves' boundary specs (typed under 'explicit')
      os_, oc, ow, og = (
          lv(f(other))
          for f in (
              chunk_storage_specs,
              chunk_compute_specs,
              lambda o: chunk_wc_specs(scfg, o),
              lambda o: acc_specs(scfg, o),
          )
      )
      idx = tuple(range(len(os_))) if leaves is None else leaves
      os_, oc, ow, og = ([x[i] for i in idx] for x in (os_, oc, ow, og))
      ok = [count_leaves(other)[i] for i in idx]
      tag = "" if leaves is None else f"p{idx[0]}"
      if kind == "G":
        odt = leaf_dtypes(scfg, other, "compute")
        odt = [odt[i] for i in idx]

        def gather(st):
          return ([_gather_leaf(a, s_, c_, d_, mode, self._ag) for a, s_, c_, d_ in zip(st, os_, oc, odt)],)

        return Part(
            f"_ag{other.index}{tag}",
            gather,
            (os_,),
            (ow,),
            ([ns(x) for x in os_],),
            ([ns(x) for x in ow],),
            gathers=tuple((other.index, i) for i in idx),
        )

      def reduce(go):
        return scatter_tree(list(go), os_, oc, scfg.rs_dtype, sizes, mode, counts=ok, **self._rs)

      return Part(
          f"_rs{other.index}{tag}",
          reduce,
          (og,),
          (os_, P()),
          ([ns(x) for x in og],),
          ([ns(x) for x in os_], ns(P())),
          donate=(0,),
      )
    if kind == "A":
      if not ch.last or stash is not None:
        raise ValueError(f"'A' is the last chunk's fused forward + backward (chunk {c}, stash {stash})")
      nt = int(bool(self.mtp))  # MTP: the tokens after the targets
      in_specs = (self.wspec, self._x_spec) + (self.tok_spec,) * (1 + nt)
      in_sh = (self._w_sh, ns(self._x_spec)) + (ns(self.tok_spec),) * (1 + nt)
      out = (*out_dx, self.gspec, self.loss_spec)
      out_sh = (*(ns(s) for s in out_dx), self._g_sh, ns(self.loss_spec))
      x_don = () if first else (1,)
      if not acc:
        return Part(
            f"fb_c{c}", self._fb_body, in_specs + (P(),), out, in_sh + (ns(P()),), out_sh, x_don, global_view=self.lineage
        )
      return Part(
          f"fba_c{c}",
          self._fba_body,
          in_specs + (self.gspec, self.loss_spec, P()),
          out,
          in_sh + (self._g_sh, ns(self.loss_spec), ns(P())),
          out_sh,
          x_don + (3 + nt, 4 + nt),
          global_view=self.lineage,
      )
    if kind in ("I", "W", "AI"):
      extra = [n for n, v in (("other", other), ("leaves", leaves), ("self_leaves", self_leaves)) if v is not None]
      if kind == "AI" or (kind == "W" and ch.last):
        return self._split_last_part(kind, stash, acc, rs_self=rs_self, extra=extra)
      return self._split_part(kind, stash, acc, rs_self=rs_self, extra=extra)
    if ch.last:
      raise ValueError(f"chunk {c} is the last: its forward and backward are one part, 'A'")
    st = None if stash is None else self.build_stash(stash)
    sfx = "" if st is None else "_st"
    st_in = () if st is None else (st.specs,)
    st_sh = () if st is None else (tuple(ns(s) for s in st.specs),)
    pin = tuple(pin or ())
    if any(pt not in PIN_POINTS or not 0 <= layer < self.n for pt, layer, _, _ in pin):
      raise ValueError(
          f"chunk {c} ({self.n} MoE layer(s)): prefetch pins want a point of {PIN_POINTS} in one of "
          f"its MoE layers (a dense layer has none), got {pin}"
      )
    if kind == "F":
      body = self._fwd_body if st is None else st.fwd_body
      out = (act, self.res_spec) + st_in
      out_sh = (ns(act), tuple(ns(r) for r in self.res_spec)) + st_sh
      if self_leaves is None:
        return Part(
            f"fwd_c{c}{sfx}",
            body,
            (self.wspec, self._x_spec, P()),
            out,
            (self._w_sh, ns(self._x_spec), ns(P())),
            out_sh,
            pins=pin,
            global_view=self.lineage,
            forward=True,
        )
      ss, cs, ws = lv(chunk_storage_specs(ch)), lv(self.cspec), lv(self.wspec)
      tdef = jax.tree.structure(self.cspec, is_leaf=_is_spec)
      idx = self_leaves
      rest = tuple(i for i in range(len(cs)) if i not in idx)
      # the pins of this forward's own gathered leaves (the rest are its program's gather parts')
      mine = {(cc, i): (pt, layer) for pt, layer, cc, i in pin if cc == c and i in idx}

      cdt = leaf_dtypes(scfg, ch, "compute")

      def gather_own(stl):
        return [_gather_leaf(a, ss[i], cs[i], cdt[i], mode, self._ag) for a, i in zip(stl, idx)]

      if self.lineage:  # a global-view body: the gather, per-device code, in a shard_map of its own
        gather_own = jax.shard_map(
            gather_own, mesh=mesh, in_specs=([ss[i] for i in idx],), out_specs=[ws[i] for i in idx], check_vma=False
        )

      def self_pf(w_rest, x, seed, stl):
        got = gather_own(stl)
        full = [None] * len(cs)
        for i, v in zip(rest, w_rest):
          full[i] = v
        for i, v in zip(idx, got):
          full[i] = v
        if not mine:
          return (*body(jax.tree.unflatten(tdef, full), x, seed), got)
        regs = {}
        for k, i in enumerate(idx):
          if (c, i) in mine:
            regs.setdefault(mine[(c, i)], []).append([(c, i), got[k]])
        with _pinned(regs, pin):
          ys = body(jax.tree.unflatten(tdef, full), x, seed)
        for lst in regs.values():  # the entries the barriers updated in place
          for (_, i), v in lst:
            got[idx.index(i)] = v
        return (*ys, got)

      return Part(
          f"fwd_c{c}{sfx}_agself",
          self_pf,
          ([ws[i] for i in rest], self._x_spec, P(), [ss[i] for i in idx]),
          out + ([ws[i] for i in idx],),
          ([ns(ws[i]) for i in rest], ns(self._x_spec), ns(P()), [ns(ss[i]) for i in idx]),
          out_sh + ([ns(ws[i]) for i in idx],),
          pins=pin,
          global_view=self.lineage,
          forward=True,
      )
    if kind != "B":
      raise ValueError(f"unknown part kind {kind!r}")
    core = self._bwd_core if st is None else st.bwd_core
    k = 4 + len(st_in)  # w, x, res, [stash,] dy
    in_specs = (self.wspec, self._x_spec, self.res_spec) + st_in + (act,)
    in_sh = (self._w_sh, ns(self._x_spec), tuple(ns(r) for r in self.res_spec)) + st_sh + (ns(act),)
    # the saved set, the stash and dy (but chunk 0's): what a forward returns too, arrays of its outputs' shapes
    # (Part.fwd_like; gacc, where the part takes one, is donated after them)
    donate = ((2,) if scfg.save_residuals else ()) + tuple(range(3, k - 1)) + (() if first else (k - 1,))
    out = (*out_dx, self.gspec)
    out_sh = (*(ns(s) for s in out_dx), self._g_sh)
    if not acc:
      if rs_self is not None:
        raise ValueError("rs_self folds into an accumulating backward (acc=True)")

      def bwd(*a):
        dx, dw = core(*a[:k], a[k])
        return (*with_dx(dx), self._new_acc(dw))

      return Part(
          f"bwd_c{c}{sfx}",
          bwd,
          in_specs + (P(),),
          out,
          in_sh + (ns(P()),),
          out_sh,
          donate,
          global_view=self.lineage,
          backward=True,
          fwd_like=donate,
      )
    acc_core = self._bwd_acc if st is None else None

    def core_acc(a):
      """(dx, gacc + dW) of the part's arguments: in place where inplace_dw (not
      a stash backward's core)."""
      if acc_core is not None:
        return acc_core(*a[:k], a[k + 1], gacc=a[k])
      dx, dw = core(*a[:k], a[k + 1])
      return dx, self._add_acc(a[k], dw)

    if rs_self is None:

      def bwda(*a):
        dx, acc = core_acc(a)
        return (*with_dx(dx), acc)

      return Part(
          f"bwda_c{c}{sfx}",
          bwda,
          in_specs + (self.gspec, P()),
          out,
          in_sh + (self._g_sh, ns(P())),
          out_sh,
          donate + (k,),
          global_view=self.lineage,
          backward=True,
          fwd_like=donate,
      )
    ss, cs = chunk_storage_specs(ch), chunk_compute_specs(ch)
    if not self.lineage:

      def bwda_rs(*a):
        dx, acc = core_acc(a)
        g, sq = scatter_tree(acc, ss, cs, scfg.rs_dtype, sizes, mode, counts=count_mask(ch), **self._rs)
        return (*with_dx(dx), g, sq + sum(a[k + 2]))

    else:
      # A global-view body: the accumulate at global view (typed unreduced), the reduce-scatter -- per-device
      # code, as the R parts' -- in a shard_map of its own.
      def scatter(gacc, sqs):
        g, sq = scatter_tree(gacc, ss, cs, scfg.rs_dtype, sizes, mode, counts=count_mask(ch), **self._rs)
        return g, sq + sum(sqs)

      scatter = jax.shard_map(
          scatter, mesh=mesh, in_specs=(self.gspec, [P()] * rs_self), out_specs=(ss, P()), check_vma=False
      )

      def bwda_rs(*a):
        dx, acc = core_acc(a)
        return (*with_dx(dx), *scatter(acc, list(a[k + 2])))

    return Part(
        f"bwda_c{c}{sfx}_rsself",
        bwda_rs,
        in_specs + (self.gspec, P(), [P()] * rs_self),
        (*out_dx, ss, P()),
        in_sh + (self._g_sh, ns(P()), [ns(P())] * rs_self),
        (*(ns(s) for s in out_dx), jax.tree.map(ns, ss, is_leaf=_is_spec), ns(P())),
        donate + (k,),
        global_view=self.lineage,
        backward=True,
        fwd_like=donate,
    )

  def _split_part(self, kind, stash, acc, *, rs_self=None, extra=()):
    """`_make_part`'s 'I' and 'W': the halves of `build_split(acc)` as parts.  I evaluates the split's I jaxpr on
    its arguments' leaves and returns dx and the crossing set, each value with a leading per-device axis -- one part
    for both W's, unless in-place dW cut another I for the accumulating one (Split.i_acc); W evaluates its W jaxpr
    on the crossing set, the kept leaves (and an in-place W the accumulator's), adds the accumulator where it
    accumulates (Split.w_adds), and returns the accumulator -- with rs_self, reduce-scattered as bwda_reduce_self's.
    Each evaluates its jaxpr's equations as they are, under the part's shard_map (stage_ad.inline_eval), and closes
    over the constants it reads (stage_ad.Const)."""
    c, mesh, scfg, ch = self.ch.index, self.mesh, self.scfg, self.ch
    if stash is not None:
      raise ValueError(
          f"chunk {c}: the split backward ({kind!r}) cuts the recompute backward, not one in stash mode {stash!r}: a "
          "split chunk's microbatches run without a stash"
      )
    if extra:
      raise ValueError(f"part {kind!r} takes acc (and a W's rs_self) only, not {extra}")
    if rs_self is not None and (kind == "I" or not acc):
      raise ValueError(f"rs_self folds into an accumulating W (acc=True), not {'an I' if kind == 'I' else 'a first W'}")
    sp = self.build_split(acc)
    if kind == "I" and acc and not sp.i_acc:
      return self.part("I")  # the accumulating W reads the first write's I

    def ns(spec):
      return NamedSharding(mesh, spec)

    act = self.act_spec
    cross_sh = tuple(ns(s) for s in sp.crossing_specs)
    if kind == "I":
      i_jaxpr, i_src = sp.i_jaxpr, sp.i_src
      i_consts = list(getattr(i_jaxpr, "consts", ()))

      def bwdi(*a):
        flat = jax.tree.leaves(a)
        outs = stage_ad.inline_eval(i_jaxpr, i_consts, *[flat[s] if isinstance(s, int) else s.value for s in i_src])
        return outs[0], tuple(v[None] for v in outs[1:])

      return Part(
          f"bwdi_c{c}",
          bwdi,
          (self.wspec, self._x_spec, self.res_spec, act, P()),
          (act, sp.crossing_specs),
          (self._w_sh, ns(self._x_spec), tuple(ns(r) for r in self.res_spec), ns(act), ns(P())),
          (ns(act), cross_sh),
          sp.i_donate,
          backward=True,
          fwd_like=sp.i_donate,
      )
    w_jaxpr, w_src = sp.w_jaxpr, sp.w_src
    w_consts = list(getattr(w_jaxpr, "consts", ()))
    gtree = jax.tree.structure(self.gspec, is_leaf=_is_spec)

    def w_core(cross, kept, gacc=None):
      """The accumulator, from the crossing set, the kept leaves and (acc) the accumulator: the first write's W plus
      the accumulator, leaf by leaf, as bwda's add_acc adds it to new_acc's (w_adds), or in-place dW's own W."""
      vals = {"kept": kept, "gacc": jax.tree.leaves(gacc)}
      ins = [v[0] for v in cross] + [s.value if isinstance(s, stage_ad.Const) else vals[s[0]][s[1]] for s in w_src]
      out = jax.tree.unflatten(gtree, stage_ad.inline_eval(w_jaxpr, w_consts, *ins))
      return jax.tree.map(lambda g, d: g + d, gacc, out) if sp.w_adds else out

    in_specs = (sp.crossing_specs, sp.kept_specs)
    in_sh = (cross_sh, tuple(ns(s) for s in sp.kept_specs))
    # (the crossing set is not donated: no output of W has a crossing value's shape to alias, and a step drops it
    # once W is enqueued, so it is freed when W ends either way)
    if not acc:

      def bwdw(cross, kept):
        return (w_core(cross, kept),)

      return Part(f"bwdw_c{c}", bwdw, in_specs, (self.gspec,), in_sh, (self._g_sh,), backward=True)
    if rs_self is None:

      def bwdwa(cross, kept, gacc):
        return (w_core(cross, kept, gacc),)

      return Part(
          f"bwdwa_c{c}",
          bwdwa,
          in_specs + (self.gspec,),
          (self.gspec,),
          in_sh + (self._g_sh,),
          (self._g_sh,),
          (2,),
          backward=True,
      )
    ss, cs = chunk_storage_specs(ch), chunk_compute_specs(ch)
    sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}

    def bwdwa_rs(cross, kept, gacc, sqs):
      g, sq = scatter_tree(
          w_core(cross, kept, gacc),
          ss,
          cs,
          scfg.rs_dtype,
          sizes,
          scfg.fsdp_collectives,
          counts=count_mask(ch),
          **self._rs,
      )
      return g, sq + sum(sqs)

    return Part(
        f"bwdwa_c{c}_rsself",
        bwdwa_rs,
        in_specs + (self.gspec, [P()] * rs_self),
        (ss, P()),
        in_sh + (self._g_sh, [ns(P())] * rs_self),
        (jax.tree.map(ns, ss, is_leaf=_is_spec), ns(P())),
        (2,),
        backward=True,
    )

  def _split_last_part(self, kind, stash, acc, *, rs_self=None, extra=()):
    """`_make_part`'s 'AI' and the last chunk's 'W': the halves of `build_split_last(acc)` as parts.  AI evaluates the
    split's I jaxpr on its arguments' leaves and returns dx (but a first chunk's), the accumulator's kept part (gk, a
    tuple of its leaves in leaf order), the loss and the crossing set, each crossing value with a leading per-device
    axis -- accumulating, with gk and the loss accumulator added to its outputs where the split adds them
    (Split.w_adds), else read by the jaxpr itself (in-place dW: the cut of fba's own trace).  W evaluates its W jaxpr
    on the crossing set, the kept arguments (and an in-place W the accumulator's deferred part, gw, a tuple of those
    leaves), adds gw where the split adds it, and returns gw.  Each closes over the constants it reads
    (stage_ad.Const).  Neither folds a reduce-scatter in: the two parts merge before the stage's reduce."""
    c, mesh, ch = self.ch.index, self.mesh, self.ch
    if stash is not None:
      raise ValueError(f"chunk {c}: the fused last chunk keeps nothing, so no stash mode ({kind!r}, {stash!r})")
    if extra:
      raise ValueError(f"part {kind!r} of the last chunk takes acc only, not {extra}")
    if rs_self is not None:
      raise ValueError(
          f"chunk {c}: the split last chunk's {kind!r} folds no reduce-scatter (rs_self): its accumulator's two "
          "parts merge before the stage's reduce"
      )
    sp = self.build_split_last(acc)

    def ns(spec):
      return NamedSharding(mesh, spec)

    gl = jax.tree.leaves(self.gspec, is_leaf=_is_spec)
    gk_specs, gw_specs = tuple(gl[i] for i in sp.i_leaves), tuple(gl[i] for i in sp.w_leaves)
    gk_sh, gw_sh = tuple(ns(x) for x in gk_specs), tuple(ns(x) for x in gw_specs)
    cross_sh = tuple(ns(x) for x in sp.crossing_specs)
    if kind == "AI":
      nt, nk, n_dx = int(bool(self.mtp)), len(sp.i_leaves), 0 if ch.first else 1
      i_jaxpr, i_src = sp.i_jaxpr, sp.i_src
      i_consts = list(getattr(i_jaxpr, "consts", ()))
      adds = acc and sp.w_adds

      def fbi(*a):
        flat = jax.tree.leaves(a[: 3 + nt] + a[-1:] if adds else a)  # (w, x, tgt[, tok], seed): the first write's
        outs = stage_ad.inline_eval(i_jaxpr, i_consts, *[flat[s] if isinstance(s, int) else s.value for s in i_src])
        dx, gk, loss, cross = outs[:n_dx], outs[n_dx : n_dx + nk], outs[n_dx + nk], outs[n_dx + nk + 1 :]
        if adds:
          gk, loss = [g + d for g, d in zip(a[3 + nt], gk, strict=True)], a[4 + nt] + loss
        return (*dx, tuple(gk), loss, tuple(v[None] for v in cross))

      tk = (self.tok_spec,) * (1 + nt)
      out_dx = () if ch.first else (self.act_spec,)
      return Part(
          f"fb{'a' if acc else ''}i_c{c}",
          fbi,
          (self.wspec, self._x_spec, *tk) + ((gk_specs, self.loss_spec) if acc else ()) + (P(),),
          (*out_dx, gk_specs, self.loss_spec, sp.crossing_specs),
          (self._w_sh, ns(self._x_spec), *(ns(t) for t in tk))
          + ((gk_sh, ns(self.loss_spec)) if acc else ())
          + (ns(P()),),
          (*(ns(x) for x in out_dx), gk_sh, ns(self.loss_spec), cross_sh),
          sp.i_donate,
      )
    w_jaxpr, w_src = sp.w_jaxpr, sp.w_src
    w_consts = list(getattr(w_jaxpr, "consts", ()))

    def w_core(cross, kept, gw=None):
      """The accumulator's deferred part, from the crossing set, the kept leaves and (acc) that part: the first
      write's W plus it, leaf by leaf (w_adds), or in-place dW's own W."""
      vals = {"kept": kept, "gacc": gw}
      ins = [v[0] for v in cross] + [s.value if isinstance(s, stage_ad.Const) else vals[s[0]][s[1]] for s in w_src]
      out = stage_ad.inline_eval(w_jaxpr, w_consts, *ins)
      return tuple(g + d for g, d in zip(gw, out, strict=True)) if gw is not None and sp.w_adds else tuple(out)

    in_specs = (sp.crossing_specs, sp.kept_specs)
    in_sh = (cross_sh, tuple(ns(x) for x in sp.kept_specs))
    if not acc:

      def bwdw(cross, kept):
        return (w_core(cross, kept),)

      return Part(f"bwdw_c{c}", bwdw, in_specs, (gw_specs,), in_sh, (gw_sh,), backward=True)

    def bwdwa(cross, kept, gw):
      return (w_core(cross, kept, gw),)

    return Part(
        f"bwdwa_c{c}", bwdwa, in_specs + (gw_specs,), (gw_specs,), in_sh + (gw_sh,), (gw_sh,), (2,), backward=True
    )

  def program(self, desc):
    """The jitted program composing the parts `desc` names (cached by desc).

    `desc` is a tuple of `pdesc(kind, **part kwargs)`.  The program is named
    after its first part, with gather / reduce-scatter parts appended as the
    fixed programs name them (fwd_c0_ag1p3, bwda_c1_st_rs3) and any other part
    joined by '__' (an F&B pair: fwd_c2__bwda_c0_st).

    A remat backward hosting a reduce-scatter (B + R) is the fixed program
    `bwd_reduce`: the same values either way, but that one traces the
    reduce-scatter between the backward and its accumulate where compose
    traces it after, so only it lowers to that program byte for byte.
    Every other remat descriptor composes to its fixed program's text.  A
    Lineage chunk's B + R composes (a mixed program: `compose`), and so do its
    fixed fused programs (fwd_prefetch, ...), from the same parts.
    """
    if desc not in self._programs and self._is_bwd_reduce(desc) and not self.lineage:
      (_, b), (_, r) = desc
      r = dict(r)
      self._programs[desc] = self.bwd_reduce(r["other"], accumulate=dict(b).get("acc", False), leaves=r.get("leaves"))
    if desc not in self._programs:
      parts = []
      for kind, kw in desc:
        kw = dict(kw)
        pr = kw.pop("peer", self)
        if pr.mesh != self.mesh:
          raise ValueError(f"chunks {self.ch.index} and {pr.ch.index} are on different meshes")
        parts.append(pr.part(kind, **kw))
      name = parts[0].name + "".join(p.name if p.name.startswith("_") else "__" + p.name for p in parts[1:])
      self._programs[desc] = compose(
          self.mesh,
          parts,
          name,
          keep_barriers=self.scfg.batch_split,
          compiler_options=self._copts if self.lineage else None,
      )
    return self._programs[desc]

  @staticmethod
  def _is_bwd_reduce(desc):
    """Is `desc` a remat backward of this chunk (bwd / bwda) + a reduce-scatter?"""
    return (
        len(desc) == 2
        and desc[0][0] == "B"
        and desc[1][0] == "R"
        and {k for k, _ in desc[0][1]} <= {"acc"}
        and {k for k, _ in desc[1][1]} <= {"other", "leaves"}
    )

  # ------------------------------------------------------------- stash
  def build_stash(self, policy):
    """The `Stash` programs of `policy`, a STASH_POLICIES name (cached).

    Per layer, the forward takes `jax.vjp` of `jax.checkpoint(layer_fwd,
    policy)` and closure-converts the vjp (stage_ad.closure_convert_all), so
    everything its backward reads is an explicit constant.  Constants that
    are the layer's weights, its input or the seed are supplied again by the
    backward from its own arguments, and those that are not traced (the
    splash mask tables) are kept as they are; the rest -- exactly the named
    values the policy saves, or under 'all' every residual -- leave the
    shard_map as the stash.  The backward evaluates the converted vjps in
    reverse layer order (recomputing only what the policy did not save), then
    chunk 0's embedding vjp, as `bwd_core` does: dx and every dW are today's.

    Bit for bit so (ds_ep_test.StashAndCompose) for one-layer chunks as XLA
    compiles them by default, but for a chunk of two or more layers only with
    xla_allow_excess_precision off: by default XLA may keep a fused bf16
    intermediate in f32, and it fuses the stash and the recompute backward of
    such a chunk differently -- on CPU every dW leaf and dx then differ in
    about the last bf16 bit.  A pipeline test that wants stash and remat
    bitwise equal uses one-layer chunks, or compiles both with that option.
    ('all' is within one bf16 rounding, not bitwise: see the test.)

    The constants' layout and the converted vjps come from ONE eager
    `jax.eval_shape` of the forward, here, before anything is jitted: the
    backward can then be traced on its own (precompile traces in threads),
    and every later trace of the forward only checks the layout is the same.
    """
    if policy in self._stash:
      return self._stash[policy]
    if policy not in STASH_POLICIES or policy == "none":
      raise ValueError(
          f"stash policy must be one of {[k for k in STASH_POLICIES if k != 'none']}, "
          f"got {policy!r} (none: the recompute programs)"
      )
    if self.lineage:
      raise ValueError(
          f"stash policy {policy!r}: a batch-split chunk on Lineage's layers (batch_split_impl "
          "'lineage') runs them in one custom VJP whose residuals are Lineage's own, which a stash "
          "policy cannot name (batch_split takes the recompute programs only)"
      )
    if self.scfg.batch_split:
      raise ValueError(
          f"stash policy {policy!r}: a stash forward runs each layer inside jax.vjp, a batch-split "
          "chunk interleaves its layers' halves (batch_split takes the recompute programs only)"
      )
    if self.scfg.save_residuals:
      raise ValueError(
          f"stash policy {policy!r}: save_residuals already keeps every residual of the chunk's "
          "forward (its programs return the VJP's leaves, not the stash's layer inputs)"
      )
    if self.nd:
      raise ValueError(
          f"stash policy {policy!r}: chunk {self.ch.index} holds dense layers, which the stash programs "
          "do not build (they take MoE layers only: a pipeline with dense layers runs no stash policy)"
      )
    ch, mesh, scfg, n = self.ch, self.mesh, self.scfg, self.n
    if ch.last:
      raise ValueError(f"chunk {ch.index} is the last: its forward and backward are one program")
    if self.counting or self.aux_seqs:
      # its layers below run layer_fwd without the counts and without the aux loss riding their outputs
      raise NotImplementedError(
          f"chunk {ch.index}: stash mode runs MoE layers without the routed-bias update and "
          f"the sequence-wise aux loss only (counting {self.counting}, aux loss "
          f"{bool(self.aux_seqs)}): use the recompute programs (a schedule without a stash)"
      )
    ns = lambda spec: NamedSharding(mesh, spec)
    pol = stash_policy(policy)
    known = [None] * n  # per layer: (plan, converted vjp, literals, stash avals)

    def layer(l, wl, h, seed):
      lid = ch.layers[l]
      f = lambda wl_, h_: layer_fwd(wl_, h_, _layer_key(lid, seed), scfg)
      if pol is not None:
        f = jax.checkpoint(f, policy=pol, prevent_cse=False)
      y, vjp = jax.vjp(f, wl, h)
      conv, consts = stage_ad.closure_convert_all(vjp, jnp.zeros_like(y))
      wflat = jax.tree.leaves(wl)
      plan, res, literals = [], [], []
      for cst in consts:
        i = next((i for i, p in enumerate(wflat) if cst is p), None)
        if i is not None:
          hit = ("p", i)
        elif cst is h:
          hit = ("x", 0)
        elif cst is seed:
          hit = ("s", 0)
        elif not isinstance(cst, jax.core.Tracer):  # a constant of the trace itself
          hit = ("c", len(literals))
          literals.append(cst)
        else:
          j = next((j for j, r in enumerate(res) if r is cst), None)
          if j is None:
            j = len(res)
            res.append(cst)
          hit = ("r", j)
        plan.append(hit)
      avals = tuple((tuple(r.shape), jnp.dtype(r.dtype)) for r in res)
      if known[l] is None:  # the eager trace below
        known[l] = (tuple(plan), conv, literals, avals)
      elif (known[l][0], known[l][3]) != (tuple(plan), avals):
        raise RuntimeError(f"chunk {ch.index} layer {lid}: the {policy!r} stash layout changed " "on re-trace")
      return y, res

    def fwd_body(w, x, seed):
      h = embed(w, x, scfg) if ch.first else x
      hs, stash = [], []
      for l in range(n):
        hs.append(h)
        h, res = layer(l, layer_weights(w["layers"], l, ch.kinds), h, seed)
        stash.extend(r[None] for r in res)
      return h, tuple(hs if ch.first else hs[1:]), tuple(stash)

    def bwd_core(w, x, res, stash, dy, seed):
      hs = list(res) if ch.first else [x] + list(res)
      dws = [None] * n
      for l in reversed(range(n)):
        plan, conv, literals, _ = known[l]
        src = {
            "p": jax.tree.leaves(layer_weights(w["layers"], l, ch.kinds)),
            "x": (hs[l],),
            "s": (seed,),
            "c": literals,
            "r": [r[0] for r in stash[offs[l] : offs[l + 1]]],
        }
        dws[l], dy = conv(dy, *(src[kind][i] for kind, i in plan))
      dw = {"layers": stack_layers(dws, ch.kinds)} if n else {}  # an empty chunk 0: its embedding's alone
      if ch.first:
        dw.update(embed_grad(w, x, dy, scfg))
        return None, dw
      return dy, dw

    sds = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=ns(spec))
    # the compute weights as the programs get them: cfg.dtype, but leaf_dtypes' fp32 ones (an fp32 router under
    # float32_gate_logits, a Qwen model's norm scales under norm_numerics 'fp32')
    w_av = jax.tree.map(
        lambda s, sp, d: sds(s, d, sp),
        chunk_shapes(scfg, ch),
        self.wspec,
        leaf_dtype_tree(scfg, ch, "compute"),
        is_leaf=_is_shape,
    )
    x_av = (
        sds(self.tok_aval.shape, jnp.int32, self.tok_spec)
        if ch.first
        else sds(self.act_aval.shape, self.act_aval.dtype, self.act_spec)
    )
    jax.eval_shape(
        jax.shard_map(
            lambda w, x, seed: fwd_body(w, x, seed)[:2],
            mesh=mesh,
            in_specs=(self.wspec, self._x_spec, P()),
            out_specs=(self.act_spec, self.res_spec),
            check_vma=False,
        ),
        w_av,
        x_av,
        sds((), jnp.int32, P()),
    )
    offs = [0]
    for kn in known:
      offs.append(offs[-1] + len(kn[3]))
    avals = [a for kn in known for a in kn[3]]
    specs = tuple(P(FE, *([None] * len(s))) for s, _ in avals)
    n_dev = math.prod(mesh.shape[a] for a in FE)
    nbytes = lambda avs: sum(math.prod(s) * d.itemsize for s, d in avs)
    layer_bytes = tuple(nbytes(kn[3]) for kn in known)
    st = Stash(
        policy=policy,
        specs=specs,
        avals=tuple(sds((n_dev,) + s, d, sp) for (s, d), sp in zip(avals, specs)),
        layer_bytes=layer_bytes,
        bytes=sum(layer_bytes),
        plans=tuple(kn[0] for kn in known),
        fwd_body=fwd_body,
        bwd_core=bwd_core,
    )
    self._stash[policy] = st
    self.stash_avals[policy], self.stash_bytes[policy] = st.avals, st.bytes
    st.fwd = self.program((pdesc("F", stash=policy),))
    st.bwd = self.program((pdesc("B", stash=policy),))
    st.bwda = self.program((pdesc("B", stash=policy, acc=True),))
    return st

  # ------------------------------------------------------- split backward
  def _part_avals(self, kind, acc=False):
    """The abstract arguments of `part(kind, acc=acc)`, a tuple in its argument order of pytrees of
    ShapeDtypeStructs with this chunk's shardings: 'B' (recompute: no stash) (w, x, res, dy[, gacc], seed), 'A'
    (w, x, tgt[, tok][, gacc, lacc], seed) -- the weights in their compute dtypes (leaf_dtypes), the accumulator in
    acc_dtype."""
    mesh, scfg, ch = self.mesh, self.scfg, self.ch

    def sds(shape, dt, spec):
      return jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))

    w = jax.tree.map(
        lambda s, sp, d: sds(s, d, sp),
        chunk_shapes(scfg, ch),
        self.wspec,
        leaf_dtype_tree(scfg, ch, "compute"),
        is_leaf=_is_shape,
    )
    act = sds(self.act_aval.shape, self.act_aval.dtype, self.act_spec)
    tok = sds(self.tok_aval.shape, jnp.int32, self.tok_spec)
    x = tok if ch.first else act
    gacc = jax.tree.map(lambda s, sp: sds(s, scfg.acc_dtype, sp), acc_shapes(scfg, ch), self.gspec, is_leaf=_is_shape)
    seed = sds((), jnp.int32, P())
    if kind == "B" and not ch.last:
      return (w, x, tuple(self.res_avals), act) + ((gacc,) if acc else ()) + (seed,)
    if kind == "A" and ch.last:
      lacc = sds((scfg.fsdp * scfg.ep,), jnp.float32, self.loss_spec)
      return (w, x, tok) + ((tok,) if self.mtp else ()) + ((gacc, lacc) if acc else ()) + (seed,)
    raise ValueError(f"no abstract arguments for part {kind!r} of chunk {ch.index} (last {ch.last})")

  def build_split(self, acc=False):
    """The `Split` of this chunk's backward for the zero-bubble schedules: its I, and the W of `acc` -- the chunk's
    first gradient write (acc False) or an accumulating one (acc True) (cached).

    Traces `part('B')`, bwd's body, at its abstract arguments (`_part_avals`) to one flat per-device jaxpr
    (stage_ad.trace_body: jit and remat bodies evaluated in place, an inlined remat's operands behind its
    optimization barrier) and cuts it (stage_ad.split_jaxpr): I keeps dx, W computes every accumulator leaf.  So I
    is the forward's recompute and the input gradient -- every EP collective of the dx chain -- and W the weight
    gradients' matmuls beyond them, and, under grad_reduce 'microbatch', the reduce-scatter (new_acc's); under saved
    residuals the B is the VJP of the saved set (no recompute).  The halves run the fused backward's equations, each
    once, on the same operands: dx and every accumulator leaf are bwd's bit for bit wherever XLA compiles each
    operation as it does in the fused program -- not everywhere: XLA may rewrite a division into a multiplication by
    the reciprocal in one program and not in the other (ds_ep_test.SplitBackward, test_float32_gate_logits).

    The accumulating W is that W plus the accumulator (Split.w_adds), with no second trace: bwda's body is bwd's with
    add_acc(gacc, dW) for new_acc(dW), and on a middle chunk add_acc(g, dW) is g + new_acc(dW) on every leaf -- so
    one I feeds both W's, bwda's values bit for bit as above.  Under in-place dW accumulation (`inplace_dw`) bwda's
    dW kernels start from the accumulator's blocks: acc True traces and cuts `part('B', acc=True)` as well, its W
    reading those blocks, and takes acc False's I where the two cuts' I are one program -- the same jaxpr, inputs,
    crossing set, kept arguments and donations (Split.i_acc False) -- else its own.

    A middle chunk's only.  Refuses (ValueError) chunk 0, whose B the schedules keep whole (no input gradient to
    split off: it is all weight gradient to them); the last chunk, one fused program ('A'); a batch-split chunk on
    Lineage's layers, one custom VJP of Lineage's whole region; a chunk without layers; and a cut whose I would
    read the accumulator."""
    acc = bool(acc)
    if acc in self._splits:
      return self._splits[acc]
    ch = self.ch
    why = [
        w
        for w, bad in (
            ("chunk 0 (no input gradient to split off: the schedules keep its B whole)", ch.first),
            ("the last chunk (its forward and backward are one fused program, 'A')", ch.last),
            (
                "a batch-split chunk on Lineage's layers (batch_split_impl 'lineage': one custom VJP of Lineage's "
                "whole region)",
                self.lineage,
            ),
            ("a chunk without layers (nothing to split)", self.empty),
        )
        if bad
    ]
    if why:
      raise ValueError(f"chunk {ch.index}: no split backward (I / W) for " + "; ".join(why))
    if acc and not self.inplace_dw:
      sp = dataclasses.replace(self.build_split(False), acc=True, w_adds=True)
    else:
      sp = self._cut_backward(acc)
      if acc:  # in-place dW: bwda's own cut, on the first write's I where it is the same program
        first = self.build_split(False)
        if (
            (sp.kept, sp.i_donate, sp.crossing_local) == (first.kept, first.i_donate, first.crossing_local)
            and len(sp.i_src) == len(first.i_src)
            and all(
                s == t if isinstance(s, int) or isinstance(t, int) else stage_ad.same_values(s.value, t.value)
                for s, t in zip(sp.i_src, first.i_src)
            )
            and stage_ad.same_jaxpr(sp.i_jaxpr, first.i_jaxpr)
        ):
          sp = dataclasses.replace(sp, i_acc=False, i_jaxpr=first.i_jaxpr, i_src=first.i_src)
    self._splits[acc] = sp
    return sp

  def _cut_backward(self, acc):
    """`build_split`'s trace and cut of `part('B', acc=acc)` -> its Split (i_acc = acc, no w_adds)."""
    ch, mesh = self.ch, self.mesh
    avals = self._part_avals("B", acc)
    part = self.part("B", acc=acc)
    body = stage_ad.trace_body(part.body, avals, mesh, part.in_specs, part.out_specs)
    cut = stage_ad.split_jaxpr(body.jaxpr, 1)  # I keeps dx (the B's first output), W every accumulator leaf
    names = ("w", "x", "res", "dy") + (("gacc",) if acc else ()) + ("seed",)
    where, flat_av = [], []  # per leaf of the B's arguments: (group, its index there), its aval
    for n, a in zip(names, avals):
      for i, leaf in enumerate(jax.tree.leaves(a)):
        where.append((n, i))
        flat_av.append(leaf)
    flat_specs = jax.tree.leaves(part.in_specs, is_leaf=_is_spec)
    i_srcs = [body.src[k] for k in cut.i_inputs]
    read = sorted(where[s][1] for s in i_srcs if isinstance(s, int) and where[s][0] == "gacc")
    if read:
      raise ValueError(
          f"chunk {ch.index}: the cut's I reads accumulator leaves {read}: it would compute a value from the "
          "accumulator before its W, out of the order the accumulation keeps"
      )
    i_pos = {b: i for i, b in enumerate(b for b, (g, _) in enumerate(where) if g != "gacc")}
    i_src = tuple(i_pos[s] if isinstance(s, int) else s for s in i_srcs)
    i_jaxpr = cut.i_jaxpr.replace(invars=[body.jaxpr.invars[k] for k in cut.i_inputs])
    w_srcs = [body.src[k] for k in cut.w_inputs]
    kept = sorted({s for s in w_srcs if isinstance(s, int) and where[s][0] != "gacc"})  # B positions, in order
    w_src = tuple(
        s
        if isinstance(s, stage_ad.Const)
        else (("gacc", where[s][1]) if where[s][0] == "gacc" else ("kept", kept.index(s)))
        for s in w_srcs
    )
    kept_groups = {where[s][0] for s in kept}
    # I donates what the B donates of its arguments (dy; the saved set under saved residuals) but what W reads
    i_donate = tuple(d for d in part.donate if names[d] in ("res", "dy") and names[d] not in kept_groups)
    n_dev = math.prod(mesh.shape[a] for a in FE)
    local = tuple(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in cut.crossing)
    specs = tuple(P(FE, *([None] * len(a.shape))) for a in local)
    kept_avals = tuple(flat_av[s] for s in kept)

    def dev_bytes(a):
      return math.prod(a.sharding.shard_shape(a.shape)) * jnp.dtype(a.dtype).itemsize

    return Split(
        acc=acc,
        i_acc=acc,
        w_adds=False,
        i_jaxpr=i_jaxpr,
        w_jaxpr=cut.w_jaxpr,
        i_src=i_src,
        w_src=w_src,
        crossing_local=local,
        crossing_specs=specs,
        crossing_avals=tuple(
            jax.ShapeDtypeStruct((n_dev, *a.shape), a.dtype, sharding=NamedSharding(mesh, s))
            for a, s in zip(local, specs)
        ),
        kept=tuple(where[s] for s in kept),
        kept_specs=tuple(flat_specs[s] for s in kept),
        kept_avals=kept_avals,
        i_donate=i_donate,
        bytes=cut.stats["crossing_bytes"],
        kept_bytes=sum(dev_bytes(a) for s, a in zip(kept, kept_avals) if where[s][0] in ("x", "res", "dy")),
        stats=dict(cut.stats),
    )

  def build_split_last(self, acc=False):
    """The `Split` of the fused last chunk's program for the zero-bubble schedules that split it too (--zb-split-last):
    its AI -- the forward, the head, the loss and the input gradient, with the weight gradients of the leaves it keeps
    -- and the W of `acc`, its layers' weight gradients: the chunk's first gradient write (acc False) or an
    accumulating one (acc True) (cached).

    Traces `part('A')`, fb's body, at its abstract arguments (`_part_avals`) to one flat per-device jaxpr
    (stage_ad.trace_body, as `build_split`) and cuts it (stage_ad.split_jaxpr): I keeps dx, the loss and every
    accumulator leaf but the layers' (`deferred_leaves`: the head's, the decoder norm's, MTP's, the routers' bias
    slots, which carry their expert counts, and a first chunk's embedding's stay), W computes the layers' -- so the
    head's T x vocab logits and their cotangent never cross, and W is the layers' weight-gradient matmuls.  The halves
    run the fused program's equations, each once, on the same operands (an inlined remat -- remat_fused_last -- behind
    its optimization barrier): fb's values bit for bit wherever XLA compiles each operation as it does in the fused
    program.

    The accumulating halves add the accumulator's two parts (and the loss accumulator) to the first write's outputs,
    leaf by leaf (Split.w_adds), with no second trace: fba's body is fb's with add_acc(gacc, dW) for new_acc(dW) and
    lacc + loss, and add_acc(g, dW) is g + new_acc(dW) on every leaf of the last chunk.  Under in-place dW accumulation
    (`inplace_dw`) fba's dW kernels start from the accumulator's blocks: acc True traces and cuts `part('A',
    acc=True)` as well, its AI reading the accumulator's kept part and the loss accumulator, its W the deferred part.

    The last chunk's only.  Refuses (ValueError) any other chunk (its backward is `build_split`'s); a batch-split
    chunk on Lineage's layers, one custom VJP of Lineage's whole region; a chunk without layers -- its weight
    gradients, the head's and the norm's, are all AI's, with nothing to defer; and a cut whose AI would read the
    accumulator's deferred part, or whose W its kept part or the loss accumulator."""
    acc = bool(acc)
    if acc in self._splits:
      return self._splits[acc]
    ch = self.ch
    why = [
        w
        for w, bad in (
            ("not the last chunk (a middle chunk's backward is build_split's)", not ch.last),
            (
                "a batch-split chunk on Lineage's layers (batch_split_impl 'lineage': one custom VJP of Lineage's "
                "whole region)",
                self.lineage,
            ),
            (
                "a chunk without layers (the head's and the norm's weight gradients stay in AI: nothing to defer)",
                self.empty,
            ),
        )
        if bad
    ]
    if why:
      raise ValueError(f"chunk {ch.index}: no split last chunk (AI / W) for " + "; ".join(why))
    if acc and not self.inplace_dw:
      first = self.build_split_last(False)
      sp = dataclasses.replace(first, acc=True, i_acc=True, w_adds=True, i_donate=self._ai_donate(first.kept, True))
    else:
      sp = self._cut_last(acc)
    self._splits[acc] = sp
    return sp

  def _ai_donate(self, kept, acc):
    """What an AI part donates: the A part's x (but a first chunk's tokens), unless its W reads it (`kept`), and,
    accumulating, the accumulator's kept part and the loss accumulator it adds to."""
    nt = int(bool(self.mtp))
    x = () if self.ch.first or any(g == "x" for g, _ in kept) else (1,)
    return x + ((3 + nt, 4 + nt) if acc else ())

  def _cut_last(self, acc):
    """`build_split_last`'s trace and cut of `part('A', acc=acc)` -> its Split (last, i_acc = acc, no w_adds)."""
    ch, mesh = self.ch, self.mesh
    avals = self._part_avals("A", acc)
    part = self.part("A", acc=acc)
    body = stage_ad.trace_body(part.body, avals, mesh, part.in_specs, part.out_specs)
    defer = deferred_leaves(ch)
    w_leaves = tuple(i for i, d in enumerate(defer) if d)
    i_leaves = tuple(i for i, d in enumerate(defer) if not d)
    n_dx = 0 if ch.first else 1
    # A's outputs: dx (but a first chunk's), the accumulator's leaves, the loss.  I keeps all but the deferred leaves.
    if len(body.jaxpr.outvars) != n_dx + len(defer) + 1:
      raise AssertionError(f"chunk {ch.index}: A has {len(body.jaxpr.outvars)} outputs, want {n_dx + len(defer) + 1}")
    cut = stage_ad.split_jaxpr(body.jaxpr, [k for k in range(len(body.jaxpr.outvars)) if k - n_dx not in w_leaves])
    names = ("w", "x", "tgt") + (("tok",) if self.mtp else ()) + (("gacc", "lacc") if acc else ()) + ("seed",)
    where, flat_av = [], []  # per leaf of A's arguments: (group, its index there), its aval
    for n, a in zip(names, avals):
      for i, leaf in enumerate(jax.tree.leaves(a)):
        where.append((n, i))
        flat_av.append(leaf)
    flat_specs = jax.tree.leaves(part.in_specs, is_leaf=_is_spec)
    wset = set(w_leaves)
    i_srcs = [body.src[k] for k in cut.i_inputs]
    w_srcs = [body.src[k] for k in cut.w_inputs]
    bad = [
        f"AI reads the deferred accumulator leaf {where[s][1]}"
        for s in i_srcs
        if isinstance(s, int) and where[s][0] == "gacc" and where[s][1] in wset
    ]
    bad += [
        f"W reads {'the loss accumulator' if where[s][0] == 'lacc' else f'the kept accumulator leaf {where[s][1]}'}"
        for s in w_srcs
        if isinstance(s, int) and (where[s][0] == "lacc" or (where[s][0] == "gacc" and where[s][1] not in wset))
    ]
    if bad:
      raise ValueError(
          f"chunk {ch.index}: the cut of the fused last chunk crosses the accumulator's parts ({'; '.join(bad)}): each "
          "part has one writer, AI the kept leaves and W the deferred ones"
      )
    # AI's arguments: A's, but of the accumulator its kept part only, in leaf order (a tuple of its leaves)
    ai_pos = {wh: k for k, wh in enumerate(wh for wh in where if wh[0] != "gacc" or wh[1] not in wset)}
    i_src = tuple(ai_pos[where[s]] if isinstance(s, int) else s for s in i_srcs)
    i_jaxpr = cut.i_jaxpr.replace(invars=[body.jaxpr.invars[k] for k in cut.i_inputs])
    kept = sorted({s for s in w_srcs if isinstance(s, int) and where[s][0] != "gacc"})  # A positions, in order
    w_src = tuple(
        s
        if isinstance(s, stage_ad.Const)
        else (("gacc", w_leaves.index(where[s][1])) if where[s][0] == "gacc" else ("kept", kept.index(s)))
        for s in w_srcs
    )
    n_dev = math.prod(mesh.shape[a] for a in FE)
    local = tuple(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in cut.crossing)
    specs = tuple(P(FE, *([None] * len(a.shape))) for a in local)
    kept_avals = tuple(flat_av[s] for s in kept)

    def dev_bytes(a):
      return math.prod(a.sharding.shard_shape(a.shape)) * jnp.dtype(a.dtype).itemsize

    kept_wh = tuple(where[s] for s in kept)
    return Split(
        acc=acc,
        i_acc=acc,
        w_adds=False,
        i_jaxpr=i_jaxpr,
        w_jaxpr=cut.w_jaxpr,
        i_src=i_src,
        w_src=w_src,
        crossing_local=local,
        crossing_specs=specs,
        crossing_avals=tuple(
            jax.ShapeDtypeStruct((n_dev, *a.shape), a.dtype, sharding=NamedSharding(mesh, s))
            for a, s in zip(local, specs)
        ),
        kept=kept_wh,
        kept_specs=tuple(flat_specs[s] for s in kept),
        kept_avals=kept_avals,
        i_donate=self._ai_donate(kept_wh, acc),
        bytes=cut.stats["crossing_bytes"],
        kept_bytes=sum(dev_bytes(a) for s, a in zip(kept, kept_avals) if where[s][0] == "x"),
        stats=dict(cut.stats),
        groups=tuple(n for n in names if n not in ("gacc", "lacc")),
        last=True,
        i_leaves=i_leaves,
        w_leaves=w_leaves,
    )

  # ---------------------------------------------------- fused variants
  def fwd_prefetch(self, other: ChunkSpec, leaves=None):
    """fwd(w, x, seed, storage_leaves_other) -> y, res, compute_leaves_other.

    The forward of this chunk with the FSDP all-gather of (a subset of)
    another chunk's weights folded in, so the gather overlaps this chunk's
    compute instead of running as its own program.  `leaves` indexes the
    other chunk's flattened weight tree (None = all).  A Lineage chunk's (its
    forward a global-view body) is composed of the same parts, F + G.
    """
    if self.lineage:
      return self.program((pdesc("F"), pdesc("G", other=other, leaves=None if leaves is None else tuple(leaves))))
    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    os_ = jax.tree.leaves(chunk_storage_specs(other), is_leaf=_is_spec)
    oc = jax.tree.leaves(chunk_compute_specs(other), is_leaf=_is_spec)
    ow = jax.tree.leaves(chunk_wc_specs(scfg, other), is_leaf=_is_spec)
    idx = tuple(range(len(os_))) if leaves is None else tuple(leaves)
    os_, oc, ow = [os_[i] for i in idx], [oc[i] for i in idx], [ow[i] for i in idx]
    odt = leaf_dtypes(scfg, other, "compute")
    odt = [odt[i] for i in idx]
    fwd_body = self._fwd_body
    mode = scfg.fsdp_collectives

    def body(w, x, seed, st):
      y, res = fwd_body(w, x, seed)
      return y, res, [_gather_leaf(a, s_, c_, d_, mode, self._ag) for a, s_, c_, d_ in zip(st, os_, oc, odt)]

    tag = "" if leaves is None else "_" + ".".join(map(str, idx))
    return jax.jit(
        _named(
            sm(body, in_specs=(self.wspec, self._x_spec, P(), os_), out_specs=(self.act_spec, self.res_spec, ow)),
            f"fwd_c{ch.index}_ag{other.index}" + ("" if leaves is None else f"p{idx[0]}"),
        ),
        in_shardings=(self._w_sh, ns(self._x_spec), ns(P()), [ns(x) for x in os_]),
        out_shardings=(ns(self.act_spec), tuple(ns(r) for r in self.res_spec), [ns(x) for x in ow]),
        compiler_options=self._copts,
    )

  def fwd_self_prefetch(self, leaves):
    """This chunk's FIRST forward with part of its OWN all-gather folded in.

    (w_leaves_rest, x, seed, storage_leaves) -> (y, res, compute_leaves)

    `leaves` index the chunk's flattened weight tree (the routed experts, in
    practice): they are not needed until the MoE half of the layer, so their
    gather overlaps the embedding + attention compute instead of delaying the
    program's start.  The other leaves arrive already gathered.  A Lineage
    chunk's is its F part with these self_leaves, as the pipeline composes it.
    """
    if self.lineage:
      return self.program((pdesc("F", self_leaves=tuple(leaves)),))
    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    lv = lambda t: jax.tree.leaves(t, is_leaf=_is_spec)
    ss, cs, ws = lv(chunk_storage_specs(ch)), lv(self.cspec), lv(self.wspec)
    tdef = jax.tree.structure(self.cspec, is_leaf=_is_spec)
    idx = tuple(leaves)
    rest = tuple(i for i in range(len(cs)) if i not in idx)
    cdt = leaf_dtypes(scfg, ch, "compute")
    mode = scfg.fsdp_collectives
    fwd_body = self._fwd_body

    def body(w_rest, x, seed, st):
      got = [_gather_leaf(a, ss[i], cs[i], cdt[i], mode, self._ag) for a, i in zip(st, idx)]
      full = [None] * len(cs)
      for i, v in zip(rest, w_rest):
        full[i] = v
      for i, v in zip(idx, got):
        full[i] = v
      y, res = fwd_body(jax.tree.unflatten(tdef, full), x, seed)
      return y, res, got

    return jax.jit(
        _named(
            sm(
                body,
                in_specs=([ws[i] for i in rest], self._x_spec, P(), [ss[i] for i in idx]),
                out_specs=(self.act_spec, self.res_spec, [ws[i] for i in idx]),
            ),
            f"fwd_c{ch.index}_agself",
        ),
        in_shardings=([ns(ws[i]) for i in rest], ns(self._x_spec), ns(P()), [ns(ss[i]) for i in idx]),
        out_shardings=(ns(self.act_spec), tuple(ns(r) for r in self.res_spec), [ns(ws[i]) for i in idx]),
        compiler_options=self._copts,
    )

  def bwd_reduce(self, other: ChunkSpec, accumulate: bool, leaves=None):
    """bwd / bwda with (a subset of) another chunk's once-per-step gradient
    reduce-scatter folded in:  (..., gacc_leaves_other) ->
    (..., grads_leaves_other, sumsq_of_those).  A Lineage chunk's is
    composed of the same parts, B + R (`program`)."""
    if self.lineage:
      return self.program(
          (pdesc("B", acc=accumulate), pdesc("R", other=other, leaves=None if leaves is None else tuple(leaves)))
      )
    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    lv = lambda t: jax.tree.leaves(t, is_leaf=_is_spec)
    os_, oc, og = lv(chunk_storage_specs(other)), lv(chunk_compute_specs(other)), lv(acc_specs(scfg, other))
    idx = tuple(range(len(os_))) if leaves is None else tuple(leaves)
    os_, oc, og = [os_[i] for i in idx], [oc[i] for i in idx], [og[i] for i in idx]
    sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}
    rs_dt = scfg.rs_dtype
    mode = scfg.fsdp_collectives
    bwd_core, new_acc, add_acc = self._bwd_core, self._new_acc, self._add_acc
    first = ch.first
    act = self.act_spec

    ok = [count_leaves(other)[i] for i in idx]

    def reduce(go):
      g, sq = scatter_tree(list(go), os_, oc, rs_dt, sizes, mode, counts=ok, **self._rs)
      return g, sq

    if accumulate:

      def body(w, x, res, dy, gacc, seed, go):
        if self.inplace_dw:
          dx, acc = self._bwd_acc(w, x, res, dy, seed, gacc=gacc)
          g, sq = reduce(go)
        else:  # the program as it always traced: the reduce before the accumulate
          dx, dw = bwd_core(w, x, res, dy, seed)
          g, sq = reduce(go)
          acc = add_acc(gacc, dw)
        return (*(() if first else (dx,)), acc, g, sq)

      in_specs = (self.wspec, self._x_spec, self.res_spec, act, self.gspec, P(), og)
      in_sh = (
          self._w_sh,
          ns(self._x_spec),
          tuple(ns(r) for r in self.res_spec),
          ns(act),
          self._g_sh,
          ns(P()),
          [ns(x) for x in og],
      )
      donate = (() if first else (3,)) + (4, 6) + ((2,) if self.scfg.save_residuals else ())  # the saved set
    else:

      def body(w, x, res, dy, seed, go):
        dx, dw = bwd_core(w, x, res, dy, seed)
        g, sq = reduce(go)
        return (*(() if first else (dx,)), new_acc(dw), g, sq)

      in_specs = (self.wspec, self._x_spec, self.res_spec, act, P(), og)
      in_sh = (self._w_sh, ns(self._x_spec), tuple(ns(r) for r in self.res_spec), ns(act), ns(P()), [ns(x) for x in og])
      donate = (() if first else (3,)) + (5,) + ((2,) if self.scfg.save_residuals else ())
    out_specs = (*(() if first else (act,)), self.gspec, os_, P())
    out_sh = (*(() if first else (ns(act),)), self._g_sh, [ns(x) for x in os_], ns(P()))
    kind = "bwda" if accumulate else "bwd"
    return jax.jit(
        _named(
            sm(body, in_specs=in_specs, out_specs=out_specs),
            f"{kind}_c{ch.index}_rs{other.index}" + ("" if leaves is None else f"p{idx[0]}"),
        ),
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate,
        compiler_options=self._copts,
    )

  def bwda_reduce_self(self, n_other_sq):
    """The chunk's LAST backward with its own once-per-step reduce-scatter
    folded in (overlapping the backward's tail instead of trailing it):
    (w, x, res, dy, gacc, seed, other_sqs) -> (dx?, grads, stage_sumsq).
    `other_sqs` are the stage's other partial sums of squares, so the output
    is the stage total the optimizer's global-norm clip needs.  A Lineage
    chunk's is its B part with rs_self, as the pipeline composes it.
    """
    if self.lineage:
      return self.program((pdesc("B", acc=True, rs_self=n_other_sq),))
    ch, mesh, scfg = self.ch, self.mesh, self.scfg
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    ss, cs = chunk_storage_specs(ch), chunk_compute_specs(ch)
    sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}
    bwd_core, add_acc = self._bwd_core, self._add_acc
    first, act = ch.first, self.act_spec

    def body(w, x, res, dy, gacc, seed, sqs):
      dx, acc = self._bwd_acc(w, x, res, dy, seed, gacc=gacc)
      g, sq = scatter_tree(acc, ss, cs, scfg.rs_dtype, sizes, scfg.fsdp_collectives, counts=count_mask(ch), **self._rs)
      return (*(() if first else (dx,)), g, sq + sum(sqs))

    in_specs = (self.wspec, self._x_spec, self.res_spec, act, self.gspec, P(), [P()] * n_other_sq)
    in_sh = (
        self._w_sh,
        ns(self._x_spec),
        tuple(ns(r) for r in self.res_spec),
        ns(act),
        self._g_sh,
        ns(P()),
        [ns(P())] * n_other_sq,
    )
    out_specs = (*(() if first else (act,)), ss, P())
    out_sh = (*(() if first else (ns(act),)), jax.tree.map(ns, ss, is_leaf=_is_spec), ns(P()))
    return jax.jit(
        _named(sm(body, in_specs=in_specs, out_specs=out_specs), f"bwda_c{ch.index}_rsself"),
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=(() if first else (3,)) + (4,),
        compiler_options=self._copts,
    )


# ------------------------------------------------- once-per-step programs
def _axes_of(d):
  return () if d is None else ((d,) if isinstance(d, str) else tuple(d))


def _unsharded(spec, dim):
  """`spec` shards no mesh axis over dimension `dim` (a dim past its end
  included)."""
  return dim >= len(spec) or not _axes_of(spec[dim])


def _swap2(x, axis):
  """The other device's copy of `x` along a size-2 mesh axis (one async DMA)."""
  return lax.ppermute(x, axis, [(0, 1), (1, 0)])


def _trivial(axis):
  return lax.axis_size(axis) == 1


def _ag2(x, axis, dim):
  """all_gather(tiled) over a size-2 mesh axis, as one collective-permute."""
  if _trivial(axis):
    return x
  other = _swap2(x, axis)
  i = lax.axis_index(axis)
  # place both halves by dynamic offset -- no control flow, one write each
  n = x.shape[dim]
  full = list(x.shape)
  full[dim] = 2 * n
  out = jnp.zeros(full, x.dtype)
  out = lax.dynamic_update_slice_in_dim(out, x, i * n, dim)
  return lax.dynamic_update_slice_in_dim(out, other, (1 - i) * n, dim)


def _rs2(g, axis, dim):
  """psum_scatter(tiled) over a size-2 mesh axis: keep my half, add theirs."""
  if _trivial(axis):
    return g
  n = g.shape[dim] // 2
  i = lax.axis_index(axis)
  mine = lax.dynamic_slice_in_dim(g, i * n, n, dim)
  theirs = lax.dynamic_slice_in_dim(g, (1 - i) * n, n, dim)
  return mine + _swap2(theirs, axis)


# FSDP traffic runs under these scopes wherever it is folded (forward
# prefetch, update, backward): the collectives' op_name carries them, and
# xprof's "Framework Name Scope" row shows each as a labelled block.
@jax.named_scope("fsdp_all_gather")
def _gather_leaf(x, storage_spec, compute_spec, dtype, mode="xla", per_axis=False):
  """fp32 storage shard -> bf16 compute layout.  `per_axis`
  (StageConfig.gather_axes 'each'): a dimension stored over several mesh
  axes gathers one axis at a time under 'xla', as 'ppermute' does."""
  x = x.astype(dtype)
  for dim, (s, c) in enumerate(zip(storage_spec, compute_spec)):
    if s is not None and s != c:
      axes = _axes_of(s)
      if mode == "ppermute":
        # P(('fsdp','expert')) tiles fsdp-major: gather the minor axis first,
        # so each step is a plain two-block concatenation.  (Major-first would
        # halve the cross-chip bytes of these leaves, but its [2, 2, n] view
        # puts tiny dims into TPU's tiled layout: measured 45% slower.)
        for a in reversed(axes):
          x = _ag2(x, a, dim)
      elif per_axis and len(axes) > 1:
        # The minor axis first, tiled: P(('fsdp', 'expert')) tiles fsdp-major,
        # so the 'expert' gather leaves each device its fsdp row's blocks in
        # order, and the 'fsdp' gather puts the rows in order -- the joint
        # gather's result (why: StageConfig.gather_axes).
        for a in reversed(axes):
          if not _trivial(a):
            x = lax.all_gather(x, a, axis=dim, tiled=True)
      else:
        x = lax.all_gather(x, s, axis=dim, tiled=True)
  return x


# rs_axes 'each' on a v5p stage of 2 x v5p-128 PP (v5p-128 as fsdp 4 x expert 16,
# libtpu 0.0.46, its flags): libtpu offloads a reduce-scatter of a large leaf
# to the SparseCore (async) only if its extents suit it.  The vocab leaves
# ([129280, 7168] and its transpose; 129280 = 256 x 505) compiled to
# synchronous TensorCore reduce-scatters, both steps, in either orientation;
# at 129024 or 129536 rows (multiples of 512) both steps offloaded.  Of other
# unaligned row counts (x 7168), 8448-16640 offloaded as they were, 24832,
# 32512 and 49408 made the 'expert' step a full-size all-reduce, 98560 ran on
# the TensorCore (33024 offloaded).  Padding the vocab to 129536 buys nothing
# over the joint psum_scatter, whose all-reduce offloads at any size: XLA
# pads in a full-size pass of its own after the bf16 convert, so the padded
# pair holds the joint one's temporaries (one vocab leaf alone: 3.46 GiB,
# joint 3.45, unpadded 2.16) and adds that pass (3.1 ms a leaf on the
# v6e-4).  So 'each' leaves a leaf with an unsharded dim of 16384 or more
# that is no multiple of 512 to the joint psum_scatter (_rs_joint_only).
_RS_ALIGN, _RS_ALIGN_FROM = 512, 16384


def _rs_joint_only(shape, dim, storage_spec, compute_spec):
  """A leaf whose per-axis reduce-scatters along `dim` libtpu would run on
  the TensorCore: one of its unsharded dims is large and unaligned (the vocab:
  _RS_ALIGN)."""
  return any(
      d != dim and n >= _RS_ALIGN_FROM and n % _RS_ALIGN and _unsharded(storage_spec, d) and _unsharded(compute_spec, d)
      for d, n in enumerate(shape)
  )


@jax.named_scope("fsdp_reduce_scatter")
def _scatter_leaf(g, storage_spec, compute_spec, dtype, mode="xla", lead=True, per_axis=False):
  """Per-device partial [1, *compute_local] -> summed storage-local shard
  (`lead` False: the partial is [*compute_local], a device's block of an
  unreduced-typed accumulator).  `per_axis` (StageConfig.rs_axes 'each'): a
  dimension stored over both mesh axes reduce-scatters one axis at a time,
  but for the vocab leaves (_rs_joint_only)."""
  g = (g[0] if lead else g).astype(dtype)
  pa = _partial_axes(compute_spec)
  done = set()
  for dim, (s, c) in enumerate(zip(storage_spec, compute_spec)):
    if s is not None and s != c:
      axes = _axes_of(s)
      if mode == "ppermute":
        axes = tuple(a for a in axes if not _trivial(a))  # size-1 axes: nothing to reduce
      if mode == "ppermute" and not axes:
        pass
      elif mode == "ppermute" and len(axes) == 1:
        g = _rs2(g, axes[0], dim)
      elif mode == "ppermute":
        # Mirror of the gather: reduce the intra-chip MINOR axis first while
        # the array is big, so only the quarter crosses chips.
        shp = g.shape
        n = shp[dim] // 4
        y = g.reshape(shp[:dim] + (2, 2, n) + shp[dim + 1 :])
        y = _rs2(y, axes[1], dim + 1)
        y = _rs2(y, axes[0], dim)
        g = y.reshape(shp[:dim] + (n,) + shp[dim + 1 :])
      elif per_axis and len(axes) > 1 and not _rs_joint_only(g.shape, dim, storage_spec, compute_spec):
        # One psum_scatter a mesh axis, the major one first: P(('fsdp',
        # 'expert')) tiles fsdp-major, so the first leaves each device the
        # block the second splits (why: StageConfig.rs_axes).
        for a in axes:
          if not _trivial(a):
            g = lax.psum_scatter(g, a, scatter_dimension=dim, tiled=True)
      else:
        g = lax.psum_scatter(g, s, scatter_dimension=dim, tiled=True)
      done.update(_axes_of(s))
  rest = tuple(a for a in pa if a not in done)
  if rest:
    if mode == "ppermute":
      for a in rest:
        if not _trivial(a):
          g = g + _swap2(g, a)
    else:
      g = lax.psum(g, rest)
  return g


def _replication(storage_spec, sizes):
  used = {a for d in storage_spec if d is not None for a in ((d,) if isinstance(d, str) else d)}
  return math.prod(v for k, v in sizes.items() if k not in used)


def _arr_leaf(x):
  return not isinstance(x, (dict, list))


def leaf_groups(shapes_tree, n_groups, skip=()):
  """Partition a tree's leaves into `n_groups` sets of roughly equal bytes (indices in jax.tree.leaves order),
  leaving the leaves `skip` names out of every set."""
  sizes = [math.prod(s) for s in jax.tree.leaves(shapes_tree, is_leaf=_is_shape)]
  groups = [[] for _ in range(max(1, n_groups))]
  load = [0] * len(groups)
  skip = frozenset(skip)
  for i in sorted((i for i in range(len(sizes)) if i not in skip), key=lambda i: -sizes[i]):
    g = load.index(min(load))
    groups[g].append(i)
    load[g] += sizes[i]
  return [tuple(sorted(g)) for g in groups if g]


def routed_leaf_ids(ch: ChunkSpec):
  """The chunk's routed-expert leaves (jax.tree.leaves order of its weights): the leaves stored over 'expert' on
  their experts' axis -- the first, or the one after a stack's layer axis (the shared experts' leaves are stored
  over both mesh axes together, `FE`, not over 'expert' alone)."""
  specs = jax.tree.leaves(chunk_storage_specs(ch), is_leaf=_is_spec)
  return tuple(i for i, sp in enumerate(specs) if (len(sp) and sp[0] == "expert") or (len(sp) > 1 and sp[1] == "expert"))


def leaf_names(ch: ChunkSpec):
  """Readable names of chunk `ch`'s flat weight leaves (jax.tree.leaves
  order): each leaf's path in the parameter tree without the stacked layers'
  'layers' and a last 'kernel' / 'scale' -- 'moe.wi_0' (the routed experts),
  'moe.gate', 'moe.shared_experts.wo', 'self_attention.wq_a',
  'self_attention.out', 'post_self_attention_layer_norm', 'logits_dense',
  'decoder_norm', 'token_embedder.embedding'.  A DeepSeek layer leaf stacks
  the chunk's layers: one name, one gather, for all of them; a dense layer's
  leaves are 'dense.'-prefixed ('dense.mlp.wi_0', 'dense.self_attention.wq_a'),
  the MTP layer's 'mtp.' ('mtp.layer.moe.wi_0', 'mtp.projection').  A
  Qwen3.5-family chunk's leaves are a layer's each, named with its key in the
  chunk (`qwen_layer_key`): '00.mlp.routed_experts.wi_0', '01.gdn.in_proj_qkvz',
  '01.attn.query', '00.input_layernorm', ...; so are a DeepSeek chunk's under
  layer_tree 'per_layer' ('01.moe.wi_0', 'dense.00.mlp.wi_0')."""
  flat, _ = jax.tree_util.tree_flatten_with_path(chunk_storage_specs(ch), is_leaf=_is_spec)
  out = []
  for path, _ in flat:
    keys = [str(getattr(k, "key", k)) for k in path]
    keys = keys[1:] if keys[0] == "layers" else ["dense"] + keys[1:] if keys[0] == "dense_layers" else keys
    out.append(".".join(keys[:-1] if len(keys) > 1 and keys[-1] in ("kernel", "scale") else keys))
  return out


# ------------------------------------ prefetch pins: SPECs and the 'auto' plan
def parse_prefetch_pin(spec):
  """--prefetch-pin -> 'off', 'auto', or a SPEC's ((point, layer, name), ...).

  SPEC: `point[@layer]=name,name;...`, e.g. 'hn=moe.gate,moe.wi_0;xs=moe.wo,
  self_attention.out;y=logits_dense': the gathered leaves a name selects
  (`leaf_names`: the leaf of that name, or every leaf under it --
  'moe.shared_experts' its three, 'self_attention' all seven -- of every
  chunk, or with a chunk prefix of that chunk only: 'c3.moe.wi_0', 'c3' all
  of chunk 3) are pinned at that point (PIN_POINTS) of that layer of the
  host's forward (default 0, its first).  A leaf two names select takes the
  longer one's point (two of one length must agree: `pin_of`); a leaf none
  selects is left to XLA, and so is one with nothing to gather (a norm
  scale).  ValueError on anything else."""
  s = "off" if spec is None else str(spec).strip()
  if s in ("off", "auto"):
    return s
  out, seen = [], set()
  for item in filter(None, (x.strip() for x in s.split(";"))):
    where, eq, names = item.partition("=")
    point, at, layer = where.strip().partition("@")
    if not eq or point not in PIN_POINTS or (at and not layer.strip().isdigit()):
      raise ValueError(
          f"--prefetch-pin {spec!r}: {item!r} is not point[@layer]=name,... (points: "
          f'{", ".join(PIN_POINTS)}; e.g. hn=moe.wi_0;y=logits_dense)'
      )
    for n in (x.strip() for x in names.split(",")):
      if not n:
        raise ValueError(f"--prefetch-pin {spec!r}: an empty leaf name in {item!r}")
      if n in seen:
        raise ValueError(f"--prefetch-pin {spec!r}: {n!r} is pinned twice")
      seen.add(n)
      out.append((point, int(layer) if at else 0, n))
  if not out:
    raise ValueError(f"--prefetch-pin {spec!r}: off, auto, or point[@layer]=name,... (e.g. hn=moe.wi_0)")
  return tuple(out)


def pin_selects(n, name, chunk):
  """Does SPEC name `n` select leaf `name` of chunk `chunk` (parse_prefetch_pin)?"""
  return any(full == n or full.startswith(n + ".") for full in (name, f"c{chunk}.{name}"))


def pin_of(entries, name, chunk):
  """The (point, layer) the SPEC `entries` (parse_prefetch_pin) pin leaf
  `name` of chunk `chunk` at -- the longest name selecting it decides -- or
  None.  Two names of one length select a leaf only as a path and a
  chunk-qualified one ('moe.wo', 'c3.moe'): a ValueError where they pin it at
  different points, which neither length nor order could settle."""
  hits = [(len(n), n, (pt, layer)) for pt, layer, n in entries if pin_selects(n, name, chunk)]
  if not hits:
    return None
  top = max(k for k, _, _ in hits)
  at = {w for k, _, w in hits if k == top}
  if len(at) > 1:
    raise ValueError(
        f"--prefetch-pin: {sorted(n for k, n, _ in hits if k == top)}, names of one length, pin "
        f"{name} of chunk {chunk} at different points {sorted(at)}: lengthen the one that should decide"
    )
  return at.pop()


def has_gather(ch: ChunkSpec, i):
  """Does leaf i of chunk `ch` gather anything (its storage layout is not its
  compute layout)?  A norm scale is only cast: no pin."""
  ss = jax.tree.leaves(chunk_storage_specs(ch), is_leaf=_is_spec)[i]
  cs = jax.tree.leaves(chunk_compute_specs(ch), is_leaf=_is_spec)[i]
  return any(_axes_of(s) != _axes_of(c) for s, c in zip(ss, cs))


@dataclasses.dataclass(frozen=True)
class PinModel:
  """The costs `plan_pins` ('auto' prefetch pins) estimates a host forward's
  windows and its gathers with.  The defaults (PIN_MODEL) are calibrated on
  two profiled runs of PP over DCN on 2 x v5p-128 (fsdp 4 x expert 16 stages,
  libtpu 0.0.46, the v5p recipe's flags; 2026-10): R1 = E576, M=2, one
  4096-token row a device; R2 = E1024, M=4, cp 2.  PIN_MODEL_V6E is v6e's
  (PIN_MODELS: the model a device kind plans with).

    ring_gbps     a SparseCore all-gather over at most ring_max devices (v5p's
                  4-way 'fsdp', a ring of chips): GB/s received a chip (the
                  routed experts' 4-way gathers: 793 MB in 4.3 ms (R1), 1409 MB
                  in 7.6 (R2))
    plane_gbps    over ring_max + 1 to plane_max (the 16-way 'expert', a 4x4
                  plane; the EP token gathers: 881 MB in 2.84 ms)
    ring_max, plane_max  the buckets' largest group sizes (4 and 16 on v5p-128,
                  whose axes are 4, 16 and 64 devices)
    wide_gbps     over more: a native gather over a whole v5p-128 stage (64
                  devices), where libtpu does not make it an all-reduce -- with
                  prefag (--xla_tpu_prefer_async_allgather_to_allreduce), or in
                  a one-slice program.  Not measured (no run has timed one):
                  plane_gbps stands in, and 'auto''s plans under prefag rest on
                  it -- if such a gather is no faster than the all-reduce,
                  prefag + auto replays slower than auto without prefag
    allreduce_gbps  the all-reduce a multi-slice compile makes of a gather over
                  both axes (ds_ep.StageConfig.gather_axes), bytes on the wire a
                  chip (the [7168, 129280] head: 2 x 1.82 GB in 12.6 ms)
    attn_tflops, attn_ms  attention up to the MoE input ('hn'): MLA's matmul
                  and causal attention FLOPs a device at attn_tflops, plus
                  attn_ms of everything else (norms, rope, layout copies, the
                  cp permutes; 9.7 ms in R1, 5.9 in R2)
    topk_tokens_per_ms  the router's TopK over the EP group's tokens, up to
                  'xs' (4.22 ms for 65536 tokens in R1, 2.18 for 32768 in R2)
    gmm_tflops    the three expert gmms, up to 'y': megablox's FLOPs over the
                  tiles it visits -- a device's routed rows / tm plus one a
                  local expert (11.5 ms in R1, 10.2 in R2)
    safety        how far past its estimated compute a window may be filled: a
                  window's gathers run one at a time on the SparseCore and should
                  fit it, but a routed expert's gather in a window it overruns
                  a little beats leaving it -- or the head's -- to the end (R2:
                  a 7.6 ms gather in 5.9 ms of attention; 1.3 to 1.5 give the
                  same plans for both runs)
    overflow      where a gather no window holds goes: 'xla', left to XLA (no
                  gather of the v5p runs' plans was), or 'last', pinned at the
                  last window it may take -- the last layer's 'y', a forward's
                  own leaf its first layer's 'xs' -- after the windows are
                  filled.  The gathers v6e's costs leave out on R2v -- the second
                  routed expert of 0F1, the head of 0F2 -- XLA issues after the
                  gmms and waits for at once (AOT): the unpinned runs expose 5.9
                  of that 7.1 ms routed gather and 17.4 of the 19.5 ms per-axis
                  head (9.3 of 10.1 native), the A/B's pins at 'y' 1.8 and 10.4
                  (R2v's 0F2: 45.6 ms pinned, 53.8 unpinned)
  """

  ring_gbps: float = 185.0
  plane_gbps: float = 310.0
  wide_gbps: float = 310.0
  allreduce_gbps: float = 290.0
  attn_tflops: float = 292.0
  attn_ms: float = 2.1
  topk_tokens_per_ms: float = 15_200.0
  gmm_tflops: float = 408.0
  safety: float = 1.4
  ring_max: int = 4
  plane_max: int = 16
  overflow: str = "xla"

  def __post_init__(self):
    if self.overflow not in ("xla", "last"):
      raise ValueError(f"PinModel.overflow must be 'xla' or 'last', got {self.overflow!r}")


PIN_MODEL = PinModel()
# v6e's costs, measured the way PIN_MODEL's were (the same ops, medians over 4 chips x 3 profiled steps of the stage-0
# leader's trace) on the FSDP-overlap A/B on 2 x v6e-128 (2026-10-19, libtpu 0.0.46, the 'v6e' flag set, one stage a
# slice): R2v = E384, M=4, cp 2, fsdp 8 x expert 16; R1v = E128, M=2, one
# 4096-token row a device, fsdp 16 x expert 8.  A stage's mesh is v6e-128's 8 x 16 chips: its 8-device axis (R2v's
# 'fsdp', R1v's 'expert') gathers at half the rate of its 16-device one, whatever it moves, so the ring bucket is 8
# devices.
#   ring_gbps 88       8-way: the routed experts 617 MB in 6.97-7.05 ms (R2v), the EP tokens 411 MB in 4.63 (R1v),
#                      the head's per-axis 1622 MB in 18.3 (R2v, --gather-axes each)
#   plane_gbps 178     16-way: the EP tokens 440 MB in 2.38-2.44 ms (R2v), the routed experts 440 MB in 2.40-2.58
#                      (R1v), the head's per-axis 1738 MB in 9.5 (R1v, each)
#   wide_gbps 175      the native 128-way gathers of the vocab leaves (NATIVE_JOINT_GATHER_KINDS): 1839 MB in 10.1 ms
#                      (the head, R2v and R1v), 10.5-11.2 (the embedding, over its minor dim); out 233 MB in 1.32-1.38
#   allreduce_gbps 160 the 128-way all-reduces of the vocab leaves' gradients, 2 x 1.84 GB in 21.0-24.4 ms (no
#                      v6e forward gathers through one: Pipeline.joint_gather_allreduce)
#   attn 2.94 ms + FLOPs at 330 TFLOP/s: attention up to 'hn' 6.30 ms (R2v), 9.66 (R1v)
#   topk 87,700 tokens/ms: the TopK is XLA's sort of [EP x T, E] on v6e (no TopK custom call), 0.374 ms for 32768
#                      tokens x 384 experts (R2v), 0.072 x 128 (R1v); the larger stands for both
#   gmm 512 TFLOP/s:   the three gmms and the activation product between them, 4.91-4.99 ms (R2v), 7.02-7.03 (R1v)
#   overflow 'last':   R2v's 8-way routed gathers (7.0 ms) and its head (10.5 ms native, 19.6 per axis) outgrow the
#                      windows; left to XLA they are exposed (PinModel.overflow)
PIN_MODEL_V6E = PinModel(
    ring_gbps=88.0,
    plane_gbps=178.0,
    wide_gbps=175.0,
    allreduce_gbps=160.0,
    attn_tflops=330.0,
    attn_ms=2.94,
    topk_tokens_per_ms=87_700.0,
    gmm_tflops=512.0,
    ring_max=8,
    overflow="last",
)
# The PinModel each device kind's stages plan with (a v5p device reports 'TPU v5', a v6e one 'TPU v6 lite'); 'auto'
# plans any other kind with v5p's and warns (ds_ep_pipeline.Pipeline.pin_model).
PIN_MODELS = {"TPU v5": PIN_MODEL, "TPU v5p": PIN_MODEL, "TPU v6 lite": PIN_MODEL_V6E, "TPU v6e": PIN_MODEL_V6E}
# The device kinds whose multi-slice programs gather a leaf stored over both stage axes natively, with no
# all-reduce (ds_ep_pipeline.Pipeline.joint_gather_allreduce): a 2 x v6e-128 AOT (libtpu 0.0.46, the v5p recipe's
# flags) compiled the 128-way gathers of the vocab leaves, out, wq_a and the shared experts to native SparseCore
# all-gathers -- but wkv_a's, [7168, 576], 8 MB a layer -- where 2 x v5p-128 decomposes every one of them.
NATIVE_JOINT_GATHER_KINDS = ("TPU v6 lite", "TPU v6e")


def pin_windows(scfg: StageConfig, ch: ChunkSpec, tokens, model=PIN_MODEL, seq=None):
  """The compute windows of a forward of chunk `ch` with `tokens` tokens a
  device, as `plan_pins` fills them: [(point, layer, ms)] in the order the
  forward reaches the points -- per layer its attention (until 'hn'), the
  router's TopK (until 'xs') and the expert gmms (until 'y'), estimated with
  `model` (PinModel).  `seq`: the sequence length -- a device's tokens are
  rows of seq / cp, each attending within its own sequence -- by default
  tokens x cp (one row a device)."""
  cfg, t = scfg.model, tokens
  if is_qwen(cfg):
    raise ValueError(
        "prefetch pins 'auto' price each window from MLA's FLOPs and DeepSeek's measured rates: a "
        "Qwen3.5-family stage has no window model; pin by SPEC, or not at all (off)"
    )
  seq = t * scfg.cp if seq is None else seq
  D, H, F = cfg.emb_dim, cfg.num_query_heads, cfg.moe_mlp_dim
  qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
  q = D * cfg.q_lora_rank + cfg.q_lora_rank * H * qk if cfg.q_lora_rank else D * H * qk
  kv = D * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) + cfg.kv_lora_rank * H * (cfg.qk_nope_head_dim + cfg.v_head_dim)
  proj = 2 * t * (q + kv + H * cfg.v_head_dim * D)
  core = t * seq * H * (qk + cfg.v_head_dim)  # causal, per sequence: on average seq / 2 keys a query
  attn = model.attn_ms + (proj + core) / (model.attn_tflops * 1e9)
  topk = scfg.ep * t / model.topk_tokens_per_ms
  tiles_rows = t * cfg.num_experts_per_tok + cfg.num_experts // scfg.ep * scfg.wi_tiling[0]
  gmm = 3 * 2 * D * F * tiles_rows / (model.gmm_tflops * 1e9)
  return [(pt, layer, ms) for layer in range(ch.n_moe) for pt, ms in (("hn", attn), ("xs", topk), ("y", gmm))]


def pin_gather_ms(
    scfg: StageConfig, shape, storage_spec, compute_spec, joint_allreduce=False, model=PIN_MODEL, itemsize=2
):
  """The estimated SparseCore time (ms) of gathering one leaf into its
  compute copy of `itemsize` bytes an element (bf16; an fp32 router's 4),
  0 for one with nothing to gather.  Over one mesh axis a native all-gather; over
  both, under gather_axes 'joint', the all-reduce libtpu makes of it in a
  multi-slice program (`joint_allreduce`) or a native gather over all of
  them (over more than plane_max devices at PinModel.wide_gbps: measured on
  v6e, unmeasured on v5p), and under 'each' a native gather an axis, the
  minor first."""
  sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}
  out = itemsize * math.prod(
      n // math.prod(sizes[a] for a in _axes_of(c))
      for n, c in zip(shape, tuple(compute_spec) + (None,) * (len(shape) - len(compute_spec)))
  )

  def native(k, nbytes):
    gbps = model.ring_gbps if k <= model.ring_max else model.plane_gbps if k <= model.plane_max else model.wide_gbps
    return (k - 1) / k * nbytes / (gbps * 1e6)

  ms = 0.0
  for s, c in zip(storage_spec, compute_spec):
    axes = tuple(a for a in _axes_of(s) if a not in _axes_of(c) and sizes[a] > 1)
    n = math.prod(sizes[a] for a in axes)
    if not axes:
      continue
    if len(axes) == 1:
      ms += native(n, out)
    elif scfg.gather_axes == "each":
      cur = out / n
      for a in reversed(axes):
        cur *= sizes[a]
        ms += native(sizes[a], cur)
    elif joint_allreduce:
      ms += 2 * (n - 1) / n * out / (model.allreduce_gbps * 1e6)
    else:
      ms += native(n, out)
  return ms


def plan_pins(
    scfg: StageConfig, host: ChunkSpec, leaves, tokens, joint_allreduce=False, own=(), model=PIN_MODEL, seq=None
):
  """'auto' prefetch pins of one forward of chunk `host` that gathers `leaves`
  ([(ChunkSpec, leaf id)]: a prefetch part's, or the self-prefetch's) ->
  ((point, layer, chunk, leaf id), ...).

  The forward's windows (`pin_windows`: attention, the router, the gmms,
  layer by layer) are filled with the gathers (`pin_gather_ms`), largest
  first, each into the window it loads least -- the lowest (load + gather) /
  window -- among the windows it keeps within their compute times
  model.safety, the earlier window on a tie; a gather no window holds is left
  to XLA (model.overflow 'xla') or, once the windows are filled, pinned at
  the last window it may take ('last'); one with nothing to gather (a norm
  scale) is left to XLA.  Filling the
  windows in order instead, each with the largest gathers that fit, put R1's
  12.6 ms head gather into its 9.7 ms attention rather than its 11.5 ms of
  gmms (replayed: 65.5 ms against 58.3 for this plan).  `own` ((chunk, leaf
  id)s): leaves the forward itself uses (the self-prefetch's routed experts):
  at most one of them is pinned, in a window of the first layer before its
  gmms -- the rest are bound by their own gmms, and with the first one under
  way early XLA puts them in time (pinning a second into R1's attention
  replayed 51.2 ms against 48.9).  The windows and the gathers are estimates
  (PinModel; `tokens` and `seq` as pin_windows')."""
  cost = []
  for ch, i in leaves:
    shape = jax.tree.leaves(chunk_shapes(scfg, ch), is_leaf=_is_shape)[i]
    ss = jax.tree.leaves(chunk_storage_specs(ch), is_leaf=_is_spec)[i]
    cs = jax.tree.leaves(chunk_compute_specs(ch), is_leaf=_is_spec)[i]
    ms = pin_gather_ms(scfg, shape, ss, cs, joint_allreduce, model, itemsize=leaf_dtypes(scfg, ch, "compute")[i].itemsize)
    if ms > 0:
      cost.append((-ms, ch.index, i))
  windows = pin_windows(scfg, host, tokens, model, seq)
  if not windows:  # a chunk of dense layers: no point to pin at
    return ()
  load = [0.0] * len(windows)
  plan, n_own = [], 0
  for neg, c, i in sorted(cost):  # largest first; ties by (chunk, leaf)
    ms, mine = -neg, (c, i) in own
    if mine and n_own:
      continue
    fits = [
        ((load[k] + ms) / w, k)
        for k, (pt, layer, w) in enumerate(windows)
        if load[k] + ms <= w * model.safety and not (mine and (layer or pt == "y"))
    ]
    if fits:
      k = min(fits)[1]
      load[k] += ms
      plan.append((windows[k][0], windows[k][1], c, i))
      n_own += mine
  if model.overflow == "last":
    done = {(c, i) for _, _, c, i in plan}
    for neg, c, i in sorted(cost):
      mine = (c, i) in own
      if (c, i) in done or (mine and n_own):
        continue
      k = max(k for k, (pt, layer, _) in enumerate(windows) if not (mine and (layer or pt == "y")))
      load[k] -= neg
      plan.append((windows[k][0], windows[k][1], c, i))
      n_own += mine
  return tuple(sorted(plan))


def gather_tree(storage, sspec, cspec, dtype, mode="xla", per_axis=False):
  """Inside shard_map: fp32 storage shards -> bf16 compute layout (all-gather;
  `per_axis` as _gather_leaf's).  `dtype`: one for every leaf, or a tree of
  them (leaf_dtype_tree)."""
  if not isinstance(dtype, (dict, list)):
    dtype = jax.tree.map(lambda _: dtype, sspec, is_leaf=_is_spec)
  return jax.tree.map(
      lambda x, s, c, d: _gather_leaf(x, s, c, d, mode, per_axis), storage, sspec, cspec, dtype, is_leaf=_arr_leaf
  )


@jax.named_scope("grad_norm_all_reduce")
def _stage_sumsq(sq):
  """The stage's gradient sum of squares, summed over its devices (for the
  global-norm clip; the other stages' come in as arguments)."""
  return lax.psum(sum(jax.tree.leaves(sq)), FE)


def _leaf_sumsq(g, storage_spec, sizes, counts=False):
  """A reduced gradient leaf's share of the stage's sum of squares (none for
  an expert-count leaf: no gradient, count_leaves)."""
  if counts:
    return jnp.zeros((), jnp.float32)
  return jnp.sum(jnp.square(g.astype(jnp.float32))) / _replication(storage_spec, sizes)


def scatter_tree(gacc, sspec, cspec, dtype, sizes, mode="xla", lead=True, per_axis=False, counts=None):
  """Inside shard_map: per-device partials -> summed storage shards, plus the
  stage-wide sum of squares of the result (already psum'd).  `lead`,
  `per_axis`: as _scatter_leaf's.  `counts` (shaped like `gacc`, default
  none): the leaves holding expert counts (count_leaves), reduced in fp32 and
  left out of the sum of squares."""
  if counts is None:
    counts = jax.tree.map(lambda _: False, sspec, is_leaf=_is_spec)
  grads = jax.tree.map(
      lambda g, s, c, k: _scatter_leaf(g, s, c, jnp.float32 if k else dtype, mode, lead, per_axis),
      gacc,
      sspec,
      cspec,
      counts,
      is_leaf=_arr_leaf,
  )
  sq = jax.tree.map(lambda g, s, k: _leaf_sumsq(g, s, sizes, k), grads, sspec, counts, is_leaf=_arr_leaf)
  return grads, _stage_sumsq(sq)


class StagePrograms:
  """Once-per-step programs of one stage, over all of its chunks at once.

  gather(storage)                    -> compute        (bf16, all-gathered)
  reduce(gaccs)                      -> grads, sumsq   (reduce-scatter)
  update(params, opt, grads, gsq_all, lr) -> params, opt   (AdamW + clip)
  """

  def __init__(
      self, scfg: StageConfig, chunks, mesh, adam=None, n_stages=2, gather_slots=(0,), first_leaves=None, skip_leaves=None
  ):
    _require_resolved(scfg)
    self.scfg, self.chunks, self.mesh = scfg, chunks, mesh
    ns = lambda spec: NamedSharding(mesh, spec)
    sm = functools.partial(jax.shard_map, mesh=mesh, check_vma=False)
    sspec = [chunk_storage_specs(c) for c in chunks]
    cspec = [chunk_compute_specs(c) for c in chunks]
    wspec = [chunk_wc_specs(scfg, c) for c in chunks]  # cspec as the programs return it (typed: 'explicit')
    gspec = [acc_specs(scfg, c) for c in chunks]
    cmask = [count_mask(c) for c in chunks]  # the routers' biases (count_leaves)
    self.cmask = cmask
    rs = dict(lead=scfg.shard_mode != "explicit", per_axis=scfg.rs_axes == "each")  # as ChunkPrograms._rs
    self.sspec, self.cspec, self.wspec, self.gspec, self._rs = sspec, cspec, wspec, gspec, rs
    s_sh = jax.tree.map(ns, sspec, is_leaf=_is_spec)
    c_sh = jax.tree.map(ns, wspec, is_leaf=_is_spec)
    g_sh = jax.tree.map(ns, gspec, is_leaf=_is_spec)
    self.s_sh, self.c_sh, self.g_sh = s_sh, c_sh, g_sh
    dt = [leaf_dtype_tree(scfg, c, "compute") for c in chunks]  # cfg.dtype, but leaf_dtypes' fp32 ones
    ag = scfg.gather_axes == "each"  # _gather_leaf's per_axis, as ChunkPrograms._ag

    def gather_body(storage):
      return jax.tree.map(
          lambda x, s, c, d: _gather_leaf(x, s, c, d, scfg.fsdp_collectives, ag),
          storage,
          sspec,
          cspec,
          dt,
          is_leaf=lambda x: not isinstance(x, (dict, list)),
      )

    sid = chunks[0].stage
    self.gather = jax.jit(
        _named(sm(gather_body, in_specs=(sspec,), out_specs=wspec), f"gather_s{sid}"),
        in_shardings=(s_sh,),
        out_shardings=c_sh,
    )

    sizes = {"fsdp": scfg.fsdp, "expert": scfg.ep}
    rs_dt = scfg.rs_dtype

    def reduce_body(gaccs):
      if scfg.grad_reduce == "microbatch":
        grads = jax.tree.map(lambda x, k: x.astype(jnp.float32 if k else rs_dt), gaccs, cmask)
      else:
        grads = jax.tree.map(
            lambda g, s, c, k: _scatter_leaf(g, s, c, jnp.float32 if k else rs_dt, scfg.fsdp_collectives, **rs),
            gaccs,
            sspec,
            cspec,
            cmask,
            is_leaf=_arr_leaf,
        )
      sq = jax.tree.map(lambda g, s, k: _leaf_sumsq(g, s, sizes, k), grads, sspec, cmask, is_leaf=_arr_leaf)
      sumsq = _stage_sumsq(sq)
      return grads, sumsq

    self.reduce = jax.jit(
        _named(sm(reduce_body, in_specs=(gspec,), out_specs=(sspec, P())), f"reduce_s{sid}"),
        in_shardings=(g_sh,),
        out_shardings=(s_sh, ns(P())),
        donate_argnums=(0,),
    )

    mode = scfg.fsdp_collectives
    gs0 = tuple(gather_slots)
    # first_leaves: gather only these leaves of slot 0 here (as a flat list);
    # the rest of slot 0 is gathered by its first forward (fwd_self_prefetch).
    # skip_leaves: per slot, leaves never gathered -- the stage reads them in
    # place (Pipeline's `direct`: at fsdp 1 the routed experts, whose compute
    # copy would be their storage's bytes again).  `keep[i]`: the leaves of
    # slot i gathered here, as a flat list, where either applies; else None,
    # the whole tree.
    lv = lambda t: jax.tree.leaves(t, is_leaf=_is_spec)
    fl = None if first_leaves is None else tuple(first_leaves)
    self.first_leaves = fl
    sk = [frozenset(skip_leaves[i] if skip_leaves else ()) for i in range(len(chunks))]

    def keep_of(i):
      if (i != 0 or fl is None) and not sk[i]:
        return None
      base = fl if (i == 0 and fl is not None) else range(len(lv(sspec[i])))
      return tuple(j for j in base if j not in sk[i])

    keep = [keep_of(i) for i in range(len(chunks))]
    self.keep, self.skip_leaves = keep, [tuple(sorted(x)) for x in sk]

    def gather_slots_of(ps):
      out = []
      for i in gs0:
        if keep[i] is not None:
          st, ss_, cs_, ds_ = jax.tree.leaves(ps[i]), lv(sspec[i]), lv(cspec[i]), jax.tree.leaves(dt[i])
          out.append([_gather_leaf(st[j], ss_[j], cs_[j], ds_[j], mode, ag) for j in keep[i]])
        else:
          out.append(gather_tree(ps[i], sspec[i], cspec[i], dt[i], mode, ag))
      return out

    def gathered_specs(tree_of):
      return [[lv(tree_of[i])[j] for j in keep[i]] if keep[i] is not None else tree_of[i] for i in gs0]

    self._gather_slots_of, self._gathered_specs = gather_slots_of, gathered_specs
    self.gather_first = jax.jit(
        _named(sm(gather_slots_of, in_specs=(sspec,), out_specs=gathered_specs(wspec)), f"gather_s{sid}_init"),
        in_shardings=(s_sh,),
        out_shardings=gathered_specs(c_sh),
    )
    self._sm, self._ns, self._sizes, self._sid = sm, ns, sizes, sid
    self._final = {}

    a = dict(b1=0.9, b2=0.95, eps=1e-8, wd=0.1, clip=scfg.grad_clip, mu_dtype=jnp.bfloat16)
    a.update(adam or {})
    self.adam = a
    rate = getattr(scfg.model, "routed_bias_update_rate", 0.0)  # a Qwen3.5-family config has no routed bias
    flags = [k for c in chunks for k in count_leaves(c)]  # jax.tree.flatten order of the params list

    def bias_step(p, counts):
      """A router bias, from its layers' expert counts (stacked [.., E]):
      MaxText's load_balance_updates_from_counts, rate x sign(mean - load) --
      the counts' scale (the EP replicas' copies all summed) cancels."""
      c = counts.astype(jnp.float32)
      return (p.astype(jnp.float32) + rate * jnp.sign(jnp.mean(c, axis=-1, keepdims=True) - c)).astype(p.dtype)

    # StageConfig.grad_dtype: the reduced gradient (rs_dtype) enters AdamW in
    # it, cast ahead of the clip as MaxText casts after its reduction; the
    # count leaves never pass through it.
    gdt = scfg.grad_dtype

    def enter(g, p):
      """A reduced gradient as AdamW takes it: in grad_dtype; else under
      adam_math 'optax' in its weight's dtype, as MaxText's gradients take
      their parameter's (a bf16 reduce-scatter into an fp32 weight cast up, an
      fp32 --rs-dtype into a bf16 weight down); else as reduced."""
      if gdt is not None:
        return g.astype(gdt)
      return g.astype(p.dtype) if scfg.adam_math == "optax" else g

    def update(params, opt, grads, sumsqs, lr):
      gnorm = jnp.sqrt(sum(sumsqs))
      scale = jnp.minimum(1.0, a["clip"] / (gnorm + 1e-6)) if a["clip"] else 1.0
      count = opt["count"] + 1
      c1 = 1 - a["b1"] ** count.astype(jnp.float32)
      c2 = 1 - a["b2"] ** count.astype(jnp.float32)
      ps, tdef = jax.tree.flatten(params)
      ms, vs, gs = (tdef.flatten_up_to(t) for t in (opt["mu"], opt["nu"], grads))
      new_p, new_m, new_v = [], [], []
      if scfg.adam_math == "optax":
        # maxtext.experimental.mpmd.jax.deepseek.train's clip_by_global_norm / adamw_update / apply_updates, op for op.
        clip = gnorm < a["clip"]
        c1, c2 = 1 - a["b1"] ** count, 1 - a["b2"] ** count
        for p, m, v, g, k in zip(ps, ms, vs, gs, flags):
          if k:
            new_p.append(bias_step(p, g))
            new_m.append(m)
            new_v.append(v)
            continue
          g = enter(g, p)
          if a["clip"]:
            g = lax.select(clip, g, (g / gnorm.astype(g.dtype)) * a["clip"])
          nu_dtype = v.dtype
          m = (1 - a["b1"]) * g + a["b1"] * m
          v = (1 - a["b2"]) * (g**2) + a["b2"] * v
          u = (m / c1.astype(m.dtype)) / (jnp.sqrt(v / c2.astype(v.dtype)) + a["eps"])
          u = u + a["wd"] * p
          new_p.append((p + jnp.asarray(-lr, u.dtype) * u).astype(p.dtype))
          new_m.append(m.astype(a["mu_dtype"]))
          new_v.append(v.astype(nu_dtype))  # a wider gradient dtype would promote it; the programs fix its dtype
        un = lambda xs: jax.tree.unflatten(tdef, xs)
        return un(new_p), {"mu": un(new_m), "nu": un(new_v), "count": count}, gnorm
      # In fp32 whatever the storage dtype (cfg.weight_dtype), stored back in it.
      for p, m, v, g, k in zip(ps, ms, vs, gs, flags):
        if k:
          new_p.append(bias_step(p, g))
          new_m.append(m)
          new_v.append(v)
          continue
        g = enter(g, p).astype(jnp.float32) * scale
        m = a["b1"] * m.astype(jnp.float32) + (1 - a["b1"]) * g
        v32 = a["b2"] * v.astype(jnp.float32) + (1 - a["b2"]) * g * g
        p32 = p.astype(jnp.float32)
        upd = (m / c1) / (jnp.sqrt(v32 / c2) + a["eps"]) + a["wd"] * p32
        new_p.append((p32 - lr * upd).astype(p.dtype))
        new_m.append(m.astype(a["mu_dtype"]))
        new_v.append(v32.astype(v.dtype))
      un = lambda xs: jax.tree.unflatten(tdef, xs)
      return un(new_p), {"mu": un(new_m), "nu": un(new_v), "count": count}, gnorm

    opt_sh = {"mu": s_sh, "nu": s_sh, "count": ns(P())}
    self.opt_sh = opt_sh

    gs = tuple(gather_slots)
    self.gather_slots = gs

    def update_gather(params, opt, grads, sumsqs, lr):
      new_p, new_o, gnorm = update(params, opt, grads, sumsqs, lr)
      return new_p, new_o, gnorm, gather_slots_of(new_p)

    # AdamW is elementwise, so the whole update runs per device inside a
    # shard_map; the next step's all-gather of the chunks in `gather_slots`
    # rides in the same program and overlaps the (memory-bound) update.  On
    # the stage that starts the forward that is just its first chunk; a
    # stage with slack before its first forward can gather everything here.
    opt_spec = {"mu": sspec, "nu": sspec, "count": P()}
    self.update_gather = jax.jit(
        _named(
            sm(
                update_gather,
                in_specs=(sspec, opt_spec, sspec, [P()] * n_stages, P()),
                out_specs=(sspec, opt_spec, P(), gathered_specs(wspec)),
            ),
            f"update_s{sid}_ag" + ".".join(str(chunks[i].index) for i in gs),
        ),
        out_shardings=(s_sh, opt_sh, ns(P()), gathered_specs(c_sh)),
        donate_argnums=(0, 1, 2),
    )
    self.update = jax.jit(
        _named(update, f"update_s{sid}"), out_shardings=(s_sh, opt_sh, ns(P())), donate_argnums=(0, 1, 2)
    )

  def reduce_final(self, slots, n_other=None):
    """(gaccs of `slots`, partial sumsqs of the rest) -> (grads of `slots`, stage sumsq)."""
    slots = tuple(slots)
    if n_other is None:
      n_other = len(self.chunks) - len(slots)
    key = (slots, n_other)
    if key not in self._final:
      sm, ns, sizes, scfg = self._sm, self._ns, self._sizes, self.scfg
      ss = [self.sspec[i] for i in slots]
      cs = [self.cspec[i] for i in slots]
      gs = [self.gspec[i] for i in slots]

      cm = [self.cmask[i] for i in slots]

      def body(gaccs, sqs):
        if scfg.grad_reduce == "microbatch":  # already reduced: just the norm
          g = jax.tree.map(lambda x, k: x.astype(jnp.float32 if k else scfg.rs_dtype), gaccs, cm)
          sq_ = jax.tree.map(lambda x, s_, k: _leaf_sumsq(x, s_, sizes, k), g, ss, cm, is_leaf=_arr_leaf)
          sq = _stage_sumsq(sq_)
        else:
          g, sq = scatter_tree(gaccs, ss, cs, scfg.rs_dtype, sizes, scfg.fsdp_collectives, counts=cm, **self._rs)
        return g, sq + sum(sqs)

      self._final[key] = jax.jit(
          _named(
              sm(body, in_specs=(gs, [P()] * n_other), out_specs=(ss, P())),
              f"reduce_s{self._sid}_" + "_".join(str(self.chunks[i].index) for i in slots),
          ),
          in_shardings=(jax.tree.map(ns, gs, is_leaf=_is_spec), [ns(P())] * n_other),
          out_shardings=(jax.tree.map(ns, ss, is_leaf=_is_spec), ns(P())),
          donate_argnums=(0,),
      )
    return self._final[key]

  def init(self, key, scale=0.02, embed_key=None):
    """Random master weights in `cfg.weight_dtype`, created directly in the
    storage layout.  AdamW's `nu` takes the same dtype, as optax's does.

    Norm scales start at one, routed biases at zero, everything else
    ~N(0, scale^2) -- a Qwen3.5-family model's leaves by `qwen_init_rule`.
    `embed_key` (an MTP model's: chunk 0's table and the last chunk's copy
    must start equal, `mtp_shapes`): every token embedding from it, whichever
    chunk and stage holds it.
    """
    chunks, scfg = self.chunks, self.scfg
    qwen = is_qwen(scfg.model)

    def make(key, embed_key):
      out = []
      for ch in chunks:
        flat, tdef = jax.tree_util.tree_flatten_with_path(chunk_shapes(scfg, ch), is_leaf=_is_shape)
        ks = jax.random.split(jax.random.fold_in(key, ch.index), len(flat))
        vals = []
        for k, (path, s), wdt in zip(ks, flat, leaf_dtypes(scfg, ch, "storage")):
          name = getattr(path[-1], "key", None)
          rule = qwen_init_rule(path) if qwen else "ones" if name == "scale" else "zeros" if name == "bias" else "normal"
          if rule == "zeros":
            vals.append(jnp.zeros(s, wdt))
          elif rule == "ones":
            vals.append(jnp.ones(s, wdt))
          elif rule == "a_log":
            lo, hi = A_LOG_RANGE
            vals.append(jnp.log(jax.random.uniform(k, s, jnp.float32, minval=lo, maxval=hi)).astype(wdt))
          else:
            if embed_key is not None and name == "embedding":
              k = embed_key
            vals.append((scale * jax.random.normal(k, s, jnp.float32)).astype(wdt))
        out.append(jax.tree.unflatten(tdef, vals))
      return out

    return jax.jit(make, out_shardings=self.s_sh)(key, embed_key)

  def init_opt(self, params):
    zeros = jax.jit(
        lambda p: {
            "mu": jax.tree.map(lambda x: jnp.zeros(x.shape, self.adam["mu_dtype"]), p),
            "nu": jax.tree.map(jnp.zeros_like, p),
            "count": jnp.zeros((), jnp.int32),
        },
        out_shardings=self.opt_sh,
    )
    return zeros(params)
