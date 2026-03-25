#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""MLA Attention Layer."""

import math
from typing import Any, Optional, Tuple
import copy

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import layout
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

Layout = layout.Format
if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore

from flax import nnx

from maxtext.common.common_types import (
    Array,
    AxisIdxes,
    AxisNames,
    BATCH,
    BATCH_NO_EXP,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    D_KV,
    DType,
    EMBED,
    EP_AS_CONTEXT,
    HEAD,
    Q_LORA_UP_PROJ,
    KV_BATCH,
    KV_BATCH_NO_EXP,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_LORA_UP_PROJ,
    LENGTH,
    LENGTH_NO_EXP,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_KV_BATCH,
    PREFILL_LENGTH,
    AttentionType,
    DEFAULT_MASK_VALUE,
)

from maxtext.layers import nnx_wrappers
from maxtext.layers.attentions import Attention
from maxtext.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.inference import kvcache
from maxtext.inference import page_manager
from maxtext.inference import paged_attention
from maxtext.inference.kvcache import KVQuant
from maxtext.utils.sharding import create_sharding
from maxtext.utils.globals import EPS

####
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Constants
DEFAULT_MASK_VALUE = -1e9



def backward_qw_kernel(q_ref, w_ref, k_ref, d_score_ref, d_q_ref, d_w_ref,
                       k_scratch_ref, d_score_scratch_ref, sem, *,
                       bS):
    """Pallas kernel for d_q and d_w computation.
    
    Args:
        q_ref: Query block. Shape: (bT, H, D_padded). Memory: VMEM.
        w_ref: Weights block. Shape: (bT, H_padded). Memory: VMEM.
        k_ref: Key tensor (full array). Shape: (B, S_padded, D_padded). Memory: HBM (ANY).
        d_score_ref: Gradient of score (full array). Shape: (B, T_padded, S_padded). Memory: HBM (ANY).
        d_q_ref: Gradient of query (accumulator). Shape: (bT, H, D_padded). Memory: VMEM.
        d_w_ref: Gradient of weights (accumulator). Shape: (bT, H_padded). Memory: VMEM.
        k_scratch_ref: Scratch buffer for K block. Shape: (bS, D_padded). Memory: VMEM.
        d_score_scratch_ref: Scratch buffer for d_score block. Shape: (bT, bS). Memory: VMEM.
        sem: DMA Semaphore.
        bS: Block size for S dimension loop.
    """
    # Grid indices
    b_idx = pl.program_id(0)
    t_idx = pl.program_id(1)
    
    bT, H, D_padded = q_ref.shape
    
    H_padded = w_ref.shape[1]
    
    t_start = t_idx * bT
    
    # Initialize accumulators
    d_q_ref[...] = jnp.zeros_like(d_q_ref)
    d_w_ref[...] = jnp.zeros_like(d_w_ref)
    
    # Load q and w from VMEM to registers
    q_val = q_ref[...] # (bT, H, D_padded)
    w_val_padded = w_ref[...] # (bT, H_padded)
    w_val = w_val_padded[:, :H] # (bT, H)
    
    # Flatten q for matmul: (bT, H, D) -> (bT*H, D)
    q_flat = q_val.reshape(bT * H, D_padded)
    
    # S_padded is the second dim of k_ref
    S_padded = k_ref.shape[1]
    num_blocks = S_padded // bS
    
    def body(i, _):
        s_start = i * bS
        
        # Load K block: (B, S, D) -> (bS, D)
        dma_k = pltpu.make_async_copy(
            k_ref.at[b_idx, pl.ds(s_start, bS), :],
            k_scratch_ref,
            sem
        )
        # Load d_score block: (B, T, S) -> (bT, bS)
        dma_ds = pltpu.make_async_copy(
            d_score_ref.at[b_idx, pl.ds(t_start, bT), pl.ds(s_start, bS)],
            d_score_scratch_ref,
            sem
        )
        
        dma_k.start()
        dma_ds.start()
        dma_k.wait()
        dma_ds.wait()
        
        k_block = k_scratch_ref[...] # (bS, D_padded)
        d_score_block = d_score_scratch_ref[...] # (bT, bS)
        
        # Recompute Logits: (bT*H, D) @ (bS, D).T -> (bT*H, bS)
        logits_flat = jnp.dot(q_flat, k_block.T)
        logits = logits_flat.reshape(bT, H, bS) # (bT, H, bS)
        
        # ReLU mask
        relu_mask = logits > 0
        relu_logits = jnp.where(relu_mask, logits, 0.0)
        
        # Compute d_w contribution
        # d_w_acc += sum(d_score * relu_logits, axis=1)
        # d_score: (bT, bS) -> (bT, 1, bS)
        # relu_logits: (bT, H, bS)
        weighted_ds = d_score_block[:, None, :] * relu_logits # (bT, H, bS)
        d_w_partial = jnp.sum(weighted_ds, axis=2) # Sum over S -> (bT, H)
        
        # Accumulate d_w
        d_w_ref[:, :H] += d_w_partial
        
        # Compute d_logits for d_q
        # d_logits = d_score * w * (logits > 0)
        # w: (bT, H) -> (bT, H, 1)
        d_logits = d_score_block[:, None, :] * w_val[:, :, None] * relu_mask.astype(jnp.float32) # (bT, H, bS)
        
        # Compute d_q contribution
        # d_q_acc += d_logits @ k
        # d_logits -> (bT*H, bS)
        d_logits_flat = d_logits.reshape(bT * H, bS)
        d_q_partial = jnp.dot(d_logits_flat, k_block) # (bT*H, D_padded)
        d_q_partial_reshaped = d_q_partial.reshape(bT, H, D_padded)
        
        d_q_ref[...] += d_q_partial_reshaped
        
        return ()

    jax.lax.fori_loop(0, num_blocks, body, ())


def backward_k_kernel(k_ref, q_ref, w_ref, d_score_ref, d_k_ref,
                      q_scratch_ref, w_scratch_ref, d_score_scratch_ref, sem, *,
                      bT):
    """Pallas kernel for d_k computation.
    
    Args:
        k_ref: Key block. Shape: (bS, D_padded). Memory: VMEM.
        q_ref: Query tensor (full). Shape: (B, T_padded, H, D_padded). Memory: HBM (ANY).
        w_ref: Weights tensor (full). Shape: (B, T_padded, H_padded). Memory: HBM (ANY).
        d_score_ref: Gradient of score (full). Shape: (B, T_padded, S_padded). Memory: HBM (ANY).
        d_k_ref: Gradient of key (accumulator). Shape: (bS, D_padded). Memory: VMEM.
        q_scratch_ref: Scratch buffer for Q block. Shape: (bT, H, D_padded). Memory: VMEM.
        w_scratch_ref: Scratch buffer for W block. Shape: (bT, H_padded). Memory: VMEM.
        d_score_scratch_ref: Scratch buffer for d_score block. Shape: (bT, bS). Memory: VMEM.
        sem: DMA Semaphore.
        bT: Block size for T dimension loop.
    """
    # Grid indices
    b_idx = pl.program_id(0)
    s_idx = pl.program_id(1)
    
    bS, D_padded = k_ref.shape
    s_start = s_idx * bS
    
    # Initialize accumulator
    d_k_ref[...] = jnp.zeros_like(d_k_ref)
    
    # Load k from VMEM
    k_val = k_ref[...] # (bS, D_padded)
    
    # T_padded is the second dim of q_ref
    T_padded = q_ref.shape[1]
    H = q_ref.shape[2] # Real H
    num_blocks = T_padded // bT
    
    def body(i, _):
        t_start = i * bT
        
        # Load Q block: (B, T, H, D) -> (bT, H, D)
        dma_q = pltpu.make_async_copy(
            q_ref.at[b_idx, pl.ds(t_start, bT), :, :],
            q_scratch_ref,
            sem
        )
        # Load W block: (B, T, H_padded) -> (bT, H_padded)
        dma_w = pltpu.make_async_copy(
            w_ref.at[b_idx, pl.ds(t_start, bT), :],
            w_scratch_ref,
            sem
        )
        # Load d_score block: (B, T, S) -> (bT, bS)
        dma_ds = pltpu.make_async_copy(
            d_score_ref.at[b_idx, pl.ds(t_start, bT), pl.ds(s_start, bS)],
            d_score_scratch_ref,
            sem
        )
        
        dma_q.start()
        dma_w.start()
        dma_ds.start()
        dma_q.wait()
        dma_w.wait()
        dma_ds.wait()
        
        q_block = q_scratch_ref[...] # (bT, H, D_padded)
        w_block_padded = w_scratch_ref[...] # (bT, H_padded)
        w_block = w_block_padded[:, :H] # (bT, H)
        d_score_block = d_score_scratch_ref[...] # (bT, bS)
        
        # Recompute Logits: (bT, H, D) @ (bS, D).T -> (bT, H, bS)
        q_flat = q_block.reshape(bT * H, D_padded)
        logits_flat = jnp.dot(q_flat, k_val.T)
        logits = logits_flat.reshape(bT, H, bS)
        
        # ReLU mask
        relu_mask = logits > 0
        
        # Compute d_logits for d_k
        # d_logits = d_score * w * (logits > 0)
        d_logits = d_score_block[:, None, :] * w_block[:, :, None] * relu_mask.astype(jnp.float32) # (bT, H, bS)
        
        # Compute d_k contribution
        # d_k_acc += d_logits.T @ q
        # d_logits -> (bT*H, bS) -> (bS, bT*H)
        d_logits_flat = d_logits.reshape(bT * H, bS)
        # q_flat: (bT*H, D_padded)
        d_k_partial = jnp.dot(d_logits_flat.T, q_flat) # (bS, D_padded)
        d_k_ref[...] += d_k_partial
        
        return ()

    jax.lax.fori_loop(0, num_blocks, body, ())


def backward_computation(q: jnp.ndarray, k: jnp.ndarray, w: jnp.ndarray, d_score: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sets up and invokes the Pallas backward kernels.
    
    Args:
        q: Query tensor (B, T, H, D)
        k: Key tensor (B, S, D)
        w: Weights tensor (B, T, H)
        d_score: Gradient of score (B, T, S)
        
    Returns:
        d_q: (B, T, H, D)
        d_k: (B, S, D)
        d_w: (B, T, H)
    """
    B, T, H, D = q.shape
    _, S, _ = k.shape
    
    # Block sizes
    bT = 32
    bS = 512
    
    # Padding
    pad_d = (128 - (D % 128)) % 128
    D_padded = D + pad_d
    
    pad_s = (bS - (S % bS)) % bS
    S_padded = S + pad_s
    
    pad_t = (bT - (T % bT)) % bT
    T_padded = T + pad_t
    
    # Pad H in w to 128 for DMA alignment
    pad_h_w = (128 - (H % 128)) % 128
    H_padded_w = H + pad_h_w
    
    # Apply padding
    if pad_d > 0:
        q = jnp.pad(q, ((0,0), (0,0), (0,0), (0, pad_d)))
        k = jnp.pad(k, ((0,0), (0,0), (0, pad_d)))
    
    if pad_s > 0:
        k = jnp.pad(k, ((0,0), (0, pad_s), (0,0)))
        d_score = jnp.pad(d_score, ((0,0), (0,0), (0, pad_s)))
    
    if pad_t > 0:
        q = jnp.pad(q, ((0,0), (0, pad_t), (0,0), (0,0)))
        w = jnp.pad(w, ((0,0), (0, pad_t), (0,0)))
        d_score = jnp.pad(d_score, ((0,0), (0, pad_t), (0,0)))

    if pad_h_w > 0:
        # Pad w along H dimension (last dimension)
        w = jnp.pad(w, ((0,0), (0,0), (0, pad_h_w)))
        
    # --- Kernel 1: d_q and d_w ---
    grid_qw = (B, T_padded // bT)
    
    q_spec = pl.BlockSpec((None, bT, H, D_padded), lambda b, t: (b, t, 0, 0))
    w_spec = pl.BlockSpec((None, bT, H_padded_w), lambda b, t: (b, t, 0))
    k_spec_any = pl.BlockSpec(memory_space=None)
    d_score_spec_any = pl.BlockSpec(memory_space=None)
    
    d_q_spec = pl.BlockSpec((None, bT, H, D_padded), lambda b, t: (b, t, 0, 0))
    d_w_spec = pl.BlockSpec((None, bT, H_padded_w), lambda b, t: (b, t, 0))
    
    scratch_shapes_qw = [
        pltpu.VMEM((bS, D_padded), k.dtype),       # k_scratch
        pltpu.VMEM((bT, bS), d_score.dtype),       # d_score_scratch
        pltpu.SemaphoreType.DMA,                   # sem
    ]
    
    kernel_qw_fn = functools.partial(backward_qw_kernel, bS=bS)
    
    d_q, d_w = pl.pallas_call(
        kernel_qw_fn,
        out_shape=[
            jax.ShapeDtypeStruct((B, T_padded, H, D_padded), q.dtype),
            jax.ShapeDtypeStruct((B, T_padded, H_padded_w), w.dtype)
        ],
        grid=grid_qw,
        in_specs=[q_spec, w_spec, k_spec_any, d_score_spec_any],
        out_specs=[d_q_spec, d_w_spec],
        scratch_shapes=scratch_shapes_qw,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(q, w, k, d_score)
    
    # --- Kernel 2: d_k ---
    grid_k = (B, S_padded // bS)
    
    k_spec = pl.BlockSpec((None, bS, D_padded), lambda b, s: (b, s, 0))
    q_spec_any = pl.BlockSpec(memory_space=None)
    w_spec_any = pl.BlockSpec(memory_space=None)
    # d_score_spec_any reused
    
    d_k_spec = pl.BlockSpec((None, bS, D_padded), lambda b, s: (b, s, 0))
    
    scratch_shapes_k = [
        pltpu.VMEM((bT, H, D_padded), q.dtype),    # q_scratch
        pltpu.VMEM((bT, H_padded_w), w.dtype),     # w_scratch (padded)
        pltpu.VMEM((bT, bS), d_score.dtype),       # d_score_scratch
        pltpu.SemaphoreType.DMA,                   # sem
    ]
    
    kernel_k_fn = functools.partial(backward_k_kernel, bT=bT)
    
    d_k = pl.pallas_call(
        kernel_k_fn,
        out_shape=jax.ShapeDtypeStruct((B, S_padded, D_padded), k.dtype),
        grid=grid_k,
        in_specs=[k_spec, q_spec_any, w_spec_any, d_score_spec_any],
        out_specs=d_k_spec,
        scratch_shapes=scratch_shapes_k,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(k, q, w, d_score)
    
    # Slice back
    d_q = d_q[:, :T, :, :D]
    d_k = d_k[:, :S, :D]
    d_w = d_w[:, :T, :H] # Slice H back
    
    return d_q, d_k, d_w


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def indexer_computation_vjp(comp_fn, q: jnp.ndarray, k: jnp.ndarray, w: jnp.ndarray, mask: Optional[jnp.ndarray], top_k: int):
    return comp_fn(q, k, w, mask, top_k)

def _indexer_computation_fwd(comp_fn, q: jnp.ndarray, k: jnp.ndarray, w: jnp.ndarray, mask: Optional[jnp.ndarray], top_k: int):
    score, indices, index_mask = comp_fn(q, k, w, mask, top_k)
    return (score, indices, index_mask), (q, k, w)

def _indexer_computation_bwd(comp_fn, top_k: int, res, g):
    q, k, w = res
    g_score, g_indices, g_index_mask = g
    d_q, d_k, d_w = backward_computation(q, k, w, g_score)
    return d_q, d_k, d_w, None

indexer_computation_vjp.defvjp(_indexer_computation_fwd, _indexer_computation_bwd)

class Indexer(nnx.Module):
  """Indexer for DeepSeek Sparse Attention (DSA).

  This module implements the sparse attention indexer introduced in DeepSeek
  V3.2.
  It computes relevance scores to select the top-k most relevant tokens for
  attention.

  References:
    DeepSeek-AI, `DeepSeek-V3.2: Pushing the Frontier of Open Large Language
    Models
      <https://arxiv.org/pdf/2512.02556>`_, 2026
    Implementation:
    https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
  """

  def __init__(
      self,
      config: Any,
      rotary_embedding,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.rotary_embedding = rotary_embedding
    self.quant = quant
    self.kernel_init = kernel_init
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.n_heads = config.indexer_n_heads
    self.head_dim = config.indexer_head_dim
    self.indexer_topk = config.indexer_topk
    self.emb_dim = config.emb_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.q_lora_rank = config.q_lora_rank
    # scale head weights for numerical stability
    self.softmax_scale = self.head_dim**-0.5

    # Query Projection: Latent Query -> Indexer Query
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Projection: Input -> Shared Indexer Key
    self.wk = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Normalization with Bias
    self.k_norm = nnx.LayerNorm(num_features=self.head_dim, use_bias=True, dtype=self.weight_dtype, rngs=rngs)

    # Projection: Input -> Importance Weights for Heads
    # deepseek3.2 enforces FP32 and does not quantize, for precision and stability.
    self.weights_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "q_heads"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        quant=None,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

  def apply_partial_rope(
      self,
      inputs: Array,
      inputs_positions: Optional[Array | None] = None,
  ):
    """Applies partial RoPE to the indexer query or key

    The Indexer's RoPE implementation differs from MLA's in two key aspects:
    1. Split Order: Indexer splits the head dimension into [rope, nope], whereas MLA uses [nope, rope].
    2. Input Layout: Indexer uses concatenated layout (interleave=False), whereas MLA uses interleaved (interleave=True).

    Args:
      inputs: Input array of shape [batch, seqlen, indexer_n_heads, indexer_head_dim].
      positions: Position array of shape [batch, seqlen].

    Returns:
      Array with partial RoPE applied, with shape [batch, seqlen, indexer_n_heads, indexer_head_dim]
    """
    # indexer_head_dim -> [rope_head_dim, indexer_head_dim - rope_head_dim]
    x_pe, x_nope = jnp.split(inputs, [self.rope_head_dim], axis=-1)
    # x_pe [B, S, H, rope_head_dim], positions [B, S]
    x_pe = self.rotary_embedding(x_pe, position=inputs_positions)
    x = jnp.concatenate([x_pe, x_nope], axis=-1)
    return x

  def kernel(self, q_ref, k_ref, w_ref, mask_ref, o_score_ref, 
            k_scratch_ref, mask_scratch_ref, score_scratch_ref, sem, *, 
            has_mask, bS):
      """Pallas kernel for Indexer score computation.
      
      Args:
          q_ref: Query block. Shape: (bT, H, D). Memory: VMEM.
          k_ref: Key tensor (full array). Shape: (B, S_padded, D). Memory: HBM (ANY).
          w_ref: Weights block. Shape: (bT, H). Memory: VMEM.
          mask_ref: Mask tensor (full array). Shape: (B, T_padded, S_padded). Memory: HBM (ANY).
          o_score_ref: Output score tensor (full array). Shape: (B, T_padded, S_padded). Memory: HBM (ANY).
          k_scratch_ref: Scratch buffer for K block. Shape: (bS, D). Memory: VMEM.
          mask_scratch_ref: Scratch buffer for Mask block. Shape: (bT, bS). Memory: VMEM.
          score_scratch_ref: Scratch buffer for Score block. Shape: (bT, bS). Memory: VMEM.
          sem: DMA Semaphore.
          has_mask: Boolean, whether mask is provided.
          bS: Block size for S dimension loop.
      """
      # Grid indices
      b_idx = pl.program_id(0)
      t_idx = pl.program_id(1)
      
      bT, H, D = q_ref.shape
      
      t_start = t_idx * bT
      
      # S_padded is the second dim of k_ref
      S_padded = k_ref.shape[1]
      
      # Load q from VMEM to registers
      q_val = q_ref[...]
      # Flatten q for matmul: (bT, H, D) -> (bT*H, D)
      q_flat = q_val.reshape(bT * H, D)
      
      # Load w from VMEM to registers
      w_val = w_ref[...]
      
      # Loop over S blocks
      num_blocks = S_padded // bS
      
      def body(i, _):
          s_start = i * bS
          
          # Load K block from HBM (ANY) to VMEM (scratch)
          # k_ref is (B, S, D), we want (b_idx, s_start:s_start+bS, :)
          dma_k = pltpu.make_async_copy(
              k_ref.at[b_idx, pl.ds(s_start, bS), :],
              k_scratch_ref,
              sem
          )
          dma_k.start()
          dma_k.wait()
          
          k_block = k_scratch_ref[...]  # Shape: (bS, D)
          
          # Compute Logits: (bT*H, D) @ (bS, D).T -> (bT*H, bS)
          logits_flat = jnp.dot(q_flat, k_block.T)  # Shape: (bT*H, bS)
          
          # Reshape to (bT, H, bS)
          logits = logits_flat.reshape(bT, H, bS)
          
          # ReLU activation
          logits = jax.nn.relu(logits)
          
          # Weighted Sum: sum(logits * w) -> (bT, bS)
          weighted = logits * w_val[..., None]
          score_block = jnp.sum(weighted, axis=1)  # Shape: (bT, bS)
          
          # Apply Mask if present
          if has_mask:
              # mask_ref is (B, T, S), we want (b_idx, t_start:t_start+bT, s_start:s_start+bS)
              dma_mask = pltpu.make_async_copy(
                  mask_ref.at[b_idx, pl.ds(t_start, bT), pl.ds(s_start, bS)],
                  mask_scratch_ref,
                  sem
              )
              dma_mask.start()
              dma_mask.wait()
              
              mask_block = mask_scratch_ref[...]
              score_block = score_block + mask_block
              
          # Store Score Block to HBM (ANY)
          # First store to VMEM scratch
          score_scratch_ref[...] = score_block
          
          # Then copy to HBM
          dma_score = pltpu.make_async_copy(
              score_scratch_ref,
              o_score_ref.at[b_idx, pl.ds(t_start, bT), pl.ds(s_start, bS)],
              sem
          )
          dma_score.start()
          dma_score.wait()
          
          return ()

      jax.lax.fori_loop(0, num_blocks, body, ())

  def _computation_impl(self, q: jnp.ndarray, k: jnp.ndarray, w: jnp.ndarray, mask: Optional[jnp.ndarray], top_k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      """Sets up and invokes the Pallas kernel.
      
      Args:
          q: Query tensor (B, T, H, D)
          k: Key tensor (B, S, D)
          w: Weights tensor (B, T, H)
          mask: Optional mask tensor (B, T, S)
          top_k: Number of top elements
          
      Returns:
          score: (B, T, S)
          indices: (B, T, top_k)
          index_mask: (B, T, top_k) - Mask values at the selected indices
      """
      B, T, H, D = q.shape
      _, S, _ = k.shape
      
      # Block sizes
      bT = 32
      bS = 512
      
      # Pad D to multiple of 128 (TPU vector alignment)
      # TPU vector registers are 8x128 (for f32). The last dimension should be 128-aligned.
      pad_d = (128 - (D % 128)) % 128
      
      if pad_d > 0:
          q = jnp.pad(q, ((0,0), (0,0), (0,0), (0, pad_d)))
          k = jnp.pad(k, ((0,0), (0,0), (0, pad_d)))
      
      D_padded = D + pad_d
      
      # Pad S to multiple of bS
      pad_s = (bS - (S % bS)) % bS
      if pad_s > 0:
          k = jnp.pad(k, ((0,0), (0, pad_s), (0,0)))
          if mask is not None:
              mask = jnp.pad(mask, ((0,0), (0,0), (0, pad_s)), constant_values=DEFAULT_MASK_VALUE)
      
      S_padded = S + pad_s
      
      # Pad T to multiple of bT
      pad_t = (bT - (T % bT)) % bT
      if pad_t > 0:
          q = jnp.pad(q, ((0,0), (0, pad_t), (0,0), (0,0)))
          w = jnp.pad(w, ((0,0), (0, pad_t), (0,0)))
          if mask is not None:
              mask = jnp.pad(mask, ((0,0), (0, pad_t), (0,0)), constant_values=DEFAULT_MASK_VALUE)
      
      T_padded = T + pad_t
      
      # Grid: (B, T_padded // bT)
      grid = (B, T_padded // bT)
      
      # Block Specs
      # q: (B, T, H, D_padded) -> (bT, H, D_padded) in kernel (squeeze B)
      q_spec = pl.BlockSpec((None, bT, H, D_padded), lambda b, t: (b, t, 0, 0))
      # w: (B, T, H) -> (bT, H) in kernel (squeeze B)
      w_spec = pl.BlockSpec((None, bT, H), lambda b, t: (b, t, 0))
      
      # k: (B, S_padded, D_padded) -> Full array in HBM
      # We use ANY memory space, so we must pass the full array and slice manually in the kernel
      k_spec = pl.BlockSpec(memory_space=None)
      
      # mask
      has_mask = mask is not None
      if has_mask:
          # mask: (B, T, S) -> Full array in HBM
          mask_spec = pl.BlockSpec(memory_space=None)
      else:
          # Dummy mask to satisfy Pallas signature
          # Create a small dummy mask
          dummy_mask = jnp.zeros((1, 1), dtype=jnp.float32)
          mask_spec = pl.BlockSpec(memory_space=None)

      # Outputs
      # o_score: (B, T, S) -> Full array in HBM
      o_score_spec = pl.BlockSpec(memory_space=None)
      
      out_shape = jax.ShapeDtypeStruct((B, T_padded, S_padded), dtype=jnp.float32)
      
      # Scratch buffers for manual copy
      # We need VMEM buffers to copy data from ANY memory space
      scratch_shapes = [
          pltpu.VMEM((bS, D_padded), k.dtype),       # k_scratch
          pltpu.VMEM((bT, bS), jnp.float32),  # mask_scratch (assume float32)
          pltpu.VMEM((bT, bS), jnp.float32),  # score_scratch
          pltpu.SemaphoreType.DMA,            # sem
      ]
      
      # Call Pallas
      # Use functools.partial to bind static arguments to the kernel
      kernel_fn = functools.partial(self.kernel, has_mask=has_mask, bS=bS)
      
      # If has_mask is False, we pass the dummy mask to the kernel
      mask_arg = mask if has_mask else dummy_mask
      
      score = pl.pallas_call(
          kernel_fn,
          out_shape=out_shape,
          grid=grid,
          in_specs=[q_spec, k_spec, w_spec, mask_spec],
          out_specs=o_score_spec,
          scratch_shapes=scratch_shapes,
          compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
      )(q, k, w, mask_arg)
      
      # Slice back to original dimensions
      score = score[:, :T, :S]
      
      # Perform TopK outside the kernel
      # jax.lax.top_k is not supported inside Pallas TPU kernels (KernelType.TC)
      vals, indices = jax.lax.top_k(score, top_k)
      
      # Extract mask values at the selected indices
      if has_mask:
          mask_sliced = mask[:, :T, :S]
          index_mask = jnp.take_along_axis(mask_sliced, indices, axis=2)
      else:
          # If no mask was provided, return zeros (valid)
          index_mask = jnp.zeros((B, T, top_k), dtype=jnp.float32)
          
      return score, indices, index_mask

  def computation(self, q: jnp.ndarray, k: jnp.ndarray, w: jnp.ndarray, mask: Optional[jnp.ndarray], top_k: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      return indexer_computation_vjp(self._computation_impl, q, k, w, mask, top_k)

  def generate_mask(self, topk_indices, s):
    """
    Creates a mask for top-k indices.

    Args:
        topk_indices: [b, t, k] int - The indices to keep.
        s: int - The total size to select from.

    Returns:
        mask: [b, t, s] - `0.0` at topk_indices, `DEFAULT_MASK_VALUE` (large negative) elsewhere.
    """
    # 1. Create a range [0, 1, ..., s-1]
    # 2. Broadcast compare against [b, t, k] to get [b, t, k, s]
    # 3. Use .any() to see if a s-index is present in any of the k slots
    is_topk = (jnp.arange(s) == topk_indices[..., None]).any(axis=-2)
    # 4. Use where to select between 0.0 and the mask value
    # cast values to dtype
    val_true = jnp.array(0.0, dtype=self.dtype)
    val_false = jnp.array(DEFAULT_MASK_VALUE, dtype=self.dtype)
    return jnp.where(is_topk, val_true, val_false)

  def __call__(
      self,
      inputs_q: Array,
      low_rank_q: Array,
      inputs_kv: Array,
      inputs_positions: Optional[Array | None] = None,
      attention_mask: Optional[Array | None] = None,
  ):
    """Computes the index score to determine the top-k relevant tokens.

    This uses a ReLU-based similarity for QK with MQA-style broadcasting (shared K).
    It uses weighted aggregation over heads to produce a single score per token pair.

    Steps:
      1. Q = RoPE(Wq @ q_lora)
      2. K = RoPE(Norm(Wk @ X))
      3. Logits = ReLU(Q @ K.T)                      # Pairwise similarity
      4. Head_Weights = (W_proj @ X) * scale         # Dynamic head importance, scale for stability
      5. Score = Logits @ Head_Weights               # Aggregate heads
      6. Indices = ArgTopk(Score)

    Args:
      inputs_q: Input of shape [b, t, embed_dim].
      low_rank_q: Low-rank latent query representations of shape [b, t, q_lora_rank].
      inputs_kv: Input of shape [b, s, embed_dim], same as inputs_q
      inputs_positions: Position indices of shape [b, s].
      attention_mask: Optional attention mask of shape [b, t, s].
        Positions with `0.0` allow attention, while positions with
        `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
        Returns `None` if no masking is determined to be necessary based on
        the inputs and configuration.

    Returns:
      indexer_mask: A sparse mask [b, t, s] with 0.0 for top-k selected tokens
        and large negative values otherwise.
      topk_indices: Indices of the top-k selected tokens [b, t, k].
      indexer_score: The computed relevance scores [b, t, s].

    Notation:
      b: Batch size
      t: Query Sequence Length (Target), note t = s here
      s: Key/Value Sequence Length (Source)
      h: Number of Indexer Heads (indexer_n_heads)
      d: Indexer Head Dimension (indexer_head_dim)
    """
    # NOTE: If sequence length <= topk, indexer always selects all tokens.
    if self.config.max_target_length <= self.indexer_topk:
      return None, None, None

    bsz, seqlen, _ = inputs_q.shape  # s = t = seqlen
    # ==============================================================================
    # Gradient Isolation Strategy: Main Model vs. Indexer
    # ==============================================================================
    # This creates a barrier to train both components independently, and applies
    # for both Dense Warm-up and Sparse Training stages:
    #
    # Forward Pass:
    # - The Indexer receives a detached copy of the inputs (via `stop_gradient`)
    #   to independently calculate its scores and `indexer_loss`.
    #
    # Backward Pass (Main Model):
    # - The main model optimizes its weights based solely on the LM loss.
    # - The `indexer_mask` in the Attention layer prevents gradients from the main
    #   loss from flowing into the Indexer's weights.
    #
    # Backward Pass (Indexer):
    # - Gradients from the `indexer_loss` flow back to update the Indexer's weights.
    # - The `stop_gradient` applied to the inputs acts as a mathematical wall, dropping
    #   gradients to 0.0 and preventing the Indexer loss from altering the main model's
    #   earlier layers.
    inputs_q = jax.lax.stop_gradient(inputs_q)
    low_rank_q = jax.lax.stop_gradient(low_rank_q)
    inputs_kv = jax.lax.stop_gradient(inputs_kv)

    # Query Processing: Project from Latent low_rank_q
    q = self.wq_b(low_rank_q)  # [b, t, q_lora_rank] -> [b, t, h * d]
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)  # [b, t, h, d]
    q = self.apply_partial_rope(q, inputs_positions=inputs_positions)

    # Key Processing: Project from Input
    k = self.wk(inputs_kv)  # [b, s, embed_dim] -> [b, s, d]
    k = self.k_norm(k)
    k = k[:, :, None, :]  # [b, s, d] -> [b, s, 1, d]
    k = self.apply_partial_rope(k, inputs_positions=inputs_positions)
    k = k.squeeze(2)  # [b, s, 1, d] -> [b, s, d]

    if True:
      # early return
      print("use kernel implementation")
      weights = self.weights_proj(inputs_q)
      weights = weights * (self.n_heads**-0.5) * self.softmax_scale
      return self.computation(q, k, weights, attention_mask, self.config.index_topk)

    print("use JAX implementation")
    # Compute Index Scores
    # QK product: relu(q @ k.T), [b, t, s, h]
    # Similar to MQA, each key is shared by h query head
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    # Compute head weights: project from input, [b, t, embed_dim] -> [b, t, h]
    weights = self.weights_proj(inputs_q)
    # Weights scaling affect indexer_score, but does not affect topk_indices. Keep scaling for numerical stability.
    # https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L478-L480
    weights = weights * (self.n_heads**-0.5) * self.softmax_scale
    # Aggregate head-wise logits: logits @ weights
    indexer_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision)  # [b, t, s]

    # Apply attention mask before TopK
    if attention_mask is not None:
      indexer_score += attention_mask

    # TopK selection based on index score
    _, topk_indices = jax.lax.top_k(indexer_score, k=self.indexer_topk)  # topk_indices [b, t, k]

    # Create Sparse Index Mask: 0 and large negatives
    indexer_mask = self.generate_mask(topk_indices, seqlen)  # [b, t, s]

    # Re-apply attention mask after TopK: in case number of unmasked tokens < TopK
    if attention_mask is not None:
      indexer_mask += attention_mask

    return indexer_mask, topk_indices, indexer_score


def mla_as_linen(
    *,
    config: Config,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_target_length: int,
    mesh: Mesh,
    attention_kernel: str,
    inputs_q_shape: Tuple,
    inputs_kv_shape: Tuple,
    dtype: DType = jnp.float32,
    weight_dtype: DType = jnp.float32,
    max_prefill_predict_length: int = -1,
    dropout_rate: float = 0.0,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
    float32_qk_product: bool = False,  # computes logits in float32 for stability.
    float32_logits: bool = False,  # cast logits in float32 for stability.
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
    # Temperature tuning parameters used for Llama4
    temperature_tuning: bool = False,
    temperature_tuning_scale: float = 0.1,
    temperature_tuning_floor_scale: float = 8192.0,
    # Shard the query activation as the same as the key and value.
    # TODO: Find a better sharding axis name.
    # TODO: Further break down the Training and Inference axes for the q, k, v.
    prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
    ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
    out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
    ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
    prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
    decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
    prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
    decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    reshape_q: bool = False,
    is_nope_layer: bool = False,
    is_vision: bool = False,
    model_mode: str = MODEL_MODE_TRAIN,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    max_position_embeddings: int = 4096 * 4,
    original_max_position_embeddings: int = 4096,
    mscale: float = 1.0,  # scaling factor for softmax
    rope_factor: float = 40.0,  # rotary embedding factor
    name: str | None = None,
):
  """A factory function to create an MLA as a Linen module.

  This function serves as a bridge to use the NNX-based `MLA` within a
  Linen model.
  """
  return nnx_wrappers.to_linen(
      MLA,
      config=config,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      max_target_length=max_target_length,
      mesh=mesh,
      attention_kernel=attention_kernel,
      inputs_q_shape=inputs_q_shape,
      inputs_kv_shape=inputs_kv_shape,
      dtype=dtype,
      weight_dtype=weight_dtype,
      max_prefill_predict_length=max_prefill_predict_length,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      float32_qk_product=float32_qk_product,
      float32_logits=float32_logits,
      quant=quant,
      kv_quant=kv_quant,
      attention_type=attention_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
      sliding_window_size=sliding_window_size,
      use_ragged_attention=use_ragged_attention,
      ragged_block_size=ragged_block_size,
      use_qk_norm=use_qk_norm,
      query_pre_attn_scalar=query_pre_attn_scalar,
      use_bias_in_projections=use_bias_in_projections,
      temperature_tuning=temperature_tuning,
      temperature_tuning_scale=temperature_tuning_scale,
      temperature_tuning_floor_scale=temperature_tuning_floor_scale,
      prefill_query_axis_names=prefill_query_axis_names,
      prefill_key_axis_names=prefill_key_axis_names,
      prefill_value_axis_names=prefill_value_axis_names,
      query_axis_names=query_axis_names,
      key_axis_names=key_axis_names,
      value_axis_names=value_axis_names,
      ep_query_axis_names=ep_query_axis_names,
      ep_key_axis_names=ep_key_axis_names,
      ep_value_axis_names=ep_value_axis_names,
      input_axis_names=input_axis_names,
      ep_input_axis_names=ep_input_axis_names,
      out_axis_names=out_axis_names,
      ep_out_axis_names=ep_out_axis_names,
      prefill_input_axis_names=prefill_input_axis_names,
      decode_input_axis_names=decode_input_axis_names,
      prefill_out_axis_names=prefill_out_axis_names,
      decode_out_axis_names=decode_out_axis_names,
      prefill_cache_axis_order=prefill_cache_axis_order,
      ar_cache_axis_order=ar_cache_axis_order,
      compute_axis_order=compute_axis_order,
      reshape_q=reshape_q,
      is_nope_layer=is_nope_layer,
      is_vision=is_vision,
      model_mode=model_mode,
      q_lora_rank=q_lora_rank,
      kv_lora_rank=kv_lora_rank,
      qk_nope_head_dim=qk_nope_head_dim,
      qk_rope_head_dim=qk_rope_head_dim,
      v_head_dim=v_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      mscale=mscale,
      rope_factor=rope_factor,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )


class MLA(Attention):
  """Multi-Head Latent Attention (MLA) layer."""

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: Tuple,
      inputs_kv_shape: Tuple,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = False,  # cast logits in float32 for stability.
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
      # Temperature tuning parameters used for Llama4
      temperature_tuning: bool = False,
      temperature_tuning_scale: float = 0.1,
      temperature_tuning_floor_scale: float = 8192.0,
      # Shard the query activation as the same as the key and value.
      # TODO: Find a better sharding axis name.
      # TODO: Further break down the Training and Inference axes for the q, k, v.
      prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
      ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
      out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
      ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
      prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
      decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      reshape_q: bool = False,
      is_nope_layer: bool = False,
      is_vision: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      q_lora_rank: int = 0,
      kv_lora_rank: int = 512,
      qk_nope_head_dim: int = 128,
      qk_rope_head_dim: int = 64,
      v_head_dim: int = 128,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      mscale: float = 1.0,  # scaling factor for softmax
      rope_factor: float = 40.0,  # rotary embedding factor
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the MLA module.

    Args:
      config: The model configuration.
      ... and other configuration parameters for MLA attention.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    base_kv_cache = config.attention != "paged" and config.mla_naive_kvcache

    # Setting these before call to super because a field is used in super
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.mscale = mscale
    self.rope_factor = rope_factor

    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        use_ragged_attention=use_ragged_attention,
        ragged_block_size=ragged_block_size,
        use_qk_norm=use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        use_bias_in_projections=use_bias_in_projections,
        temperature_tuning=temperature_tuning,
        temperature_tuning_scale=temperature_tuning_scale,
        temperature_tuning_floor_scale=temperature_tuning_floor_scale,
        prefill_query_axis_names=prefill_query_axis_names,
        prefill_key_axis_names=prefill_key_axis_names,
        prefill_value_axis_names=prefill_value_axis_names,
        query_axis_names=query_axis_names,
        key_axis_names=key_axis_names,
        value_axis_names=value_axis_names,
        ep_query_axis_names=ep_query_axis_names,
        ep_key_axis_names=ep_key_axis_names,
        ep_value_axis_names=ep_value_axis_names,
        input_axis_names=input_axis_names,
        ep_input_axis_names=ep_input_axis_names,
        out_axis_names=out_axis_names,
        ep_out_axis_names=ep_out_axis_names,
        prefill_input_axis_names=prefill_input_axis_names,
        decode_input_axis_names=decode_input_axis_names,
        prefill_out_axis_names=prefill_out_axis_names,
        decode_out_axis_names=decode_out_axis_names,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
        reshape_q=reshape_q,
        is_nope_layer=is_nope_layer,
        is_vision=is_vision,
        model_mode=model_mode,
        base_kv_cache=base_kv_cache,
        rngs=rngs,
    )

    # Initialize Indexer
    self.use_indexer = config.use_indexer
    if self.use_indexer:
      # Need two versions of rope.
      # MLA applies yarn with interleave layout.
      # Indexer applies yarn with concatenate layout.
      indexer_rope = copy.copy(self.rotary_embedding)
      indexer_rope.interleave = False
      self.indexer = Indexer(
          config,
          rngs=rngs,
          rotary_embedding=indexer_rope,
          kernel_init=kernel_init,
          quant=quant,
          model_mode=model_mode,
      )

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.MlaKVCache_0 = self.init_mla_kv_caches(inputs_kv_shape) if model_mode != MODEL_MODE_TRAIN else None

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the MLA-specific projections."""
    # Assert required configuration parameters for MLA attention.
    assert (
        self.config.attention_type == AttentionType.MLA.value
    ), f"MLA requires MLA attention type {AttentionType.MLA.value}"
    assert self.kv_lora_rank > 0, "KV LoRA rank must be > 0"
    assert self.qk_nope_head_dim > 0, "QK NoPe head dim must be > 0"
    assert self.qk_rope_head_dim > 0, "QK RoPE head dim must be > 0"
    assert self.v_head_dim > 0, "V head dim must be > 0"
    assert self.num_query_heads == self.num_kv_heads, "MLA requires equal number of query and kv heads"
    assert not self.config.fused_qkv, "Fused QKV is not supported for MLA"

    if self.q_lora_rank == 0:
      # Standard Q projection (without LoRA).
      self.query = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    else:
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=self.q_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_lora_up_proj"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
      self.q_norm = RMSNorm(
          num_features=self.q_lora_rank,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
      self.wq_b = DenseGeneral(
          in_features_shape=self.q_lora_rank,
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("q_lora", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora_up_proj"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )
    self.kv_norm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.wkv_b = DenseGeneral(
        in_features_shape=self.kv_lora_rank,
        out_features_shape=(
            self.num_query_heads,
            (self.qk_nope_head_dim + self.v_head_dim),
        ),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

    # Setup paged attention op
    if self.config.attention == "paged":
      # Set head_dim to the max of qk_head_dim and v_head_dim. The current paged
      # attention kernel requires the head_dim to be the same for q, k, v.
      head_dim = max(self.qk_head_dim, self.v_head_dim)
      # Align head_dim to the pagedattn_head_dim_alignment if specified.
      if self.config.pagedattn_head_dim_alignment > 0:
        alignment = self.config.pagedattn_head_dim_alignment
        head_dim = (head_dim + alignment - 1) // alignment * alignment
      self.ds_paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=(self.config.max_target_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=(self.config.max_prefill_predict_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          rngs=self.rngs,
      )

  def mla_query_projection(
      self, inputs_q: Array, inputs_positions: Array, model_mode
  ) -> tuple[jax.Array, Optional[jax.Array]]:
    """Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0."""
    # specify query logical name
    if model_mode == MODEL_MODE_PREFILL:
      query_logical_name = self.prefill_query_axis_names
      wqa_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, Q_LORA_UP_PROJ)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      query_logical_name = self.ep_query_axis_names
      wqa_logical_name = (KV_BATCH_NO_EXP, LENGTH, Q_LORA_UP_PROJ)
    else:
      query_logical_name = self.query_axis_names
      wqa_logical_name = (KV_BATCH, LENGTH_NO_EXP, Q_LORA_UP_PROJ)
    query_sharding = create_sharding(self.mesh, query_logical_name)
    wqa_out_sharding = create_sharding(self.mesh, wqa_logical_name)
    # Set softmax scaling.
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    # Low-rank latent vector for queries. This is also accessed by indexer.
    low_rank_q = None

    if self.q_lora_rank == 0:
      q = self.query(inputs_q, out_sharding=query_sharding)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q, out_sharding=wqa_out_sharding)  # [B, L, q_lora_rank]
      low_rank_q = checkpoint_name(low_rank_q, "query_wa_proj")
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      low_rank_q = checkpoint_name(low_rank_q, "mla_q")
      q = self.wq_b(low_rank_q, out_sharding=query_sharding)  # [B, L, n_heads, qk_head_dim]

    # Partial RoPE: Split into non-positional and rotary parts.
    # last dimension: qk_nope_head_dim, qk_rope_head_dim
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_nope = self._maybe_shard_with_logical(q_nope, query_logical_name)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions=inputs_positions)
    q_pe = self._maybe_shard_with_logical(q_pe, query_logical_name)
    # Query projection is scaled by self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale
    query = self._maybe_shard_with_logical(query, query_logical_name)
    return query, low_rank_q

  def mla_get_key_value(self, low_rank_main, key_rope, model_mode):
    """get (key,value) pair from mla"""
    if model_mode == MODEL_MODE_PREFILL:
      key_logical_name = self.prefill_key_axis_names
      value_logical_name = self.prefill_value_axis_names
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      key_logical_name = self.ep_key_axis_names
      value_logical_name = self.ep_value_axis_names
    else:
      key_logical_name = self.key_axis_names
      value_logical_name = self.value_axis_names

    wkva_out_sharding = create_sharding(self.mesh, key_logical_name)
    kv_out = self.wkv_b(low_rank_main, out_sharding=wkva_out_sharding)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))
    key_nope = self._maybe_shard_with_logical(key_nope, key_logical_name)
    key_rope = self._maybe_shard_with_logical(key_rope, key_logical_name)

    key = jnp.concatenate([key_nope, key_rope], axis=-1)

    key = self._maybe_shard_with_logical(key, key_logical_name)
    value = self._maybe_shard_with_logical(value, value_logical_name)
    return key, value

  def init_mla_kv_caches(self, inputs_kv_shape: Tuple):
    """Initializes MlaKVCache.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      An MlaKVCache module instance.

    Raises:
      ValueError: If the configuration is invalid.

    """
    batch_size, _, _ = inputs_kv_shape
    # During initialization, seq_len of inputs_kv is max_target_length,
    # which is not always correct for some functions in MlaKVCache.
    # However, MlaKVCache internal cache shapes are based on max_prefill_length
    # and max_target_length, not the passed seq_len.
    # We can use a placeholder value. The correct fix might involve refactoring
    # MlaKVCache.
    placeholder_seq_len = 1

    return kvcache.MlaKVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=placeholder_seq_len,
        value_seq_len=placeholder_seq_len,
        key_head_size=self.kv_lora_rank,
        value_head_size=self.qk_rope_head_dim,
        dtype=self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        model_mode=self.model_mode,
        use_chunked_prefill=self.config.use_chunked_prefill,
        rngs=self.rngs,
    )

  def update_mla_kv_caches(self, low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk=None):
    """Updates the MLA (Multi-Head Latent Attention) KV caches.

    This method is specific to the MLA attention mechanism. It calls the
    `mla_kv_cache_as_linen` module to update and retrieve the caches, which
    store latent representations (`low_rank_main`) and RoPE-applied keys
    (`key_rope`). It then reconstructs the full key and value tensors from
    the cached components.

    Args:
      low_rank_main: The main latent component of the key.
      key_rope: The RoPE-applied component of the key.
      decoder_segment_ids: Segment IDs for decoder masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      previous_chunk: Information about previously processed chunks, for
        chunked prefill.

    Returns:
      A list containing two elements:
      - The prefill key-value cache, reconstructed from the MLA cache, or None.
      - The autoregressive key-value cache, reconstructed from the MLA cache, or None.
    """

    prefill_mla_cache, ar_mla_cache = self.MlaKVCache_0(
        key_latent=low_rank_main,
        key_rope=key_rope,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )

    if prefill_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids = prefill_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      prefill_kv_cache = key, value, decoder_segment_ids
    else:
      prefill_kv_cache = None

    if ar_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids, lengths = ar_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      ar_kv_cache = key, value, decoder_segment_ids, lengths
    else:
      ar_kv_cache = None
    return [prefill_kv_cache, ar_kv_cache]

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array, decoder_segment_ids, model_mode, previous_chunk):
    """MLA key/value projection with integrated rotary embedding."""
    if model_mode == MODEL_MODE_PREFILL:
      wka_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_LORA_UP_PROJ)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      wka_logical_name = (KV_BATCH_NO_EXP, LENGTH, KV_LORA_UP_PROJ)
    else:
      wka_logical_name = (KV_BATCH, LENGTH_NO_EXP, KV_LORA_UP_PROJ)
    wkva_out_sharding = create_sharding(self.mesh, wka_logical_name)
    low_rank = self.wkv_a(inputs, out_sharding=wkva_out_sharding)
    low_rank = checkpoint_name(low_rank, "kv_wa_proj")
    low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)
    low_rank_main = self.kv_norm(low_rank_main)
    low_rank_main = checkpoint_name(low_rank_main, "mla_kv")
    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions=inputs_positions)

    key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
    cached_values = [None, None]
    if self.config.attention != "paged" and model_mode != MODEL_MODE_TRAIN:
      if self.config.mla_naive_kvcache:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      else:
        cached_values = self.update_mla_kv_caches(
            low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk
        )

    return key, value, cached_values

  def calculate_indexer_loss(
      self,
      indexer_score: Array,
      query: Array,
      key: Array,
      attention_mask: Optional[Array | None],
      indexer_mask: Array,
      sparse_loss: bool,
      scaling_factor: float,
  ) -> Array:
    """Calculates the indexer KL divergence loss.

    This loss trains the indexer to predict which tokens are important by matching
    the distribution of true attention scores from the main model.

    The target distribution is derived through the following steps:
    1. Compute raw attention scores via Q @ K^T.
    2. Aggregate scores by summing across all attention heads.
    3. Apply L1-normalization across the sequence dimension.

    target_distribution = L1_Normalize(Sum_h(Softmax(Q @ K^T)))

    Reference:
    DeepSeek-V3.2 - https://arxiv.org/pdf/2512.02556

    Args:
      indexer_score: Scores predicted by indexer [batch, q_len, kv_len].
      query: Query tensor from main model [batch, q_len, heads, dim].
      key: Key tensor from main model [batch, kv_len, heads, dim].
      attention_mask: Attention mask [batch, q_len, kv_len] or None.
      indexer_mask: Indexer mask [batch, q_len, kv_len].
      sparse_loss: Whether to use sparse loss.
      scaling_factor: The scaling factor for the loss.

    Returns:
      The computed KL divergence loss.
    """
    # Detach main model components from the computational graph.
    # The indexer should match the main model, but the main model should not be influenced
    # by the indexer's learning progress via this loss in sparse training stage.
    # We also apply this during the Dense Warm-up stage to save compute and memory.
    query = jax.lax.stop_gradient(query)
    key = jax.lax.stop_gradient(key)

    # Compute attention scores: [b, t, h, d] @ [b, s, h, d] -> [b, h, t, s]
    attention_scores = jnp.einsum("bthd, bshd -> bhts", query, key, precision=self.config.matmul_precision)

    if sparse_loss:
      # indexer_mask is already pre-filtered with the attention_mask if any
      attention_scores = attention_scores + indexer_mask[:, None, :, :]
      indexer_score = indexer_score + indexer_mask
    elif attention_mask is not None:
      # indexer_score already applies attention_mask; updating attention_scores only
      attention_scores = attention_scores + attention_mask[:, None, :, :]

    # Use float32 for softmax numerical stability.
    attention_probs = jax.nn.softmax(attention_scores.astype(jnp.float32), axis=-1)
    indexer_probs = jax.nn.softmax(indexer_score.astype(jnp.float32), axis=-1)

    # Aggregate heads: [b, h, t, s] -> [b, t, s]
    attention_probs = jnp.sum(attention_probs, axis=1)
    # L1 normalize aggregated target distribution
    attention_probs = attention_probs / (jnp.sum(attention_probs, axis=-1, keepdims=True) + EPS)

    # KL Divergence: KL(attention || indexer)
    log_attention_probs = jnp.log(attention_probs + EPS)
    log_indexer_probs = jnp.log(indexer_probs + EPS)
    kl_per_token = attention_probs * (log_attention_probs - log_indexer_probs)
    indexer_loss = jnp.mean(jnp.sum(kl_per_token, axis=-1))

    return indexer_loss * scaling_factor

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Optional[Any] = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
  ) -> tuple[Array, Optional[Array]]:
    """Forward pass for MLA, reusing `AttentionOp` for the actual attention.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      model_mode: "train", "prefill", or "autoregressive".
      deterministic: Disables dropout if set to True.
      previous_chunk: Information about previously processed chunks for chunked prefill.
      slot: The batch slot index for paged attention.
      page_state: The current state of the paged attention manager.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.
      kv_cache: Optional key-value cache used when serving models with vLLM.
      attention_metadata: Optional attention-related metadata used when serving models with vLLM.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
      out_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.ep_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.ep_input_axis_names)
      out_logical_name = (BATCH_NO_EXP, LENGTH, HEAD, D_KV)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
      out_logical_name = (BATCH, LENGTH_NO_EXP, HEAD, D_KV)

    query, low_rank_q = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    if self.config.force_q_layout:
      query = layout.with_layout_constraint(query, DLL(major_to_minor=(0, 2, 3, 1)))
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    # Indexer Logic
    indexer_mask = None
    if self.use_indexer:
      if model_mode != MODEL_MODE_TRAIN:
        raise NotImplementedError("Sparse indexer has not implemented for inference yet.")
      # generate mask: with 0 and large negative, [b, 1, 1, q_len, kv_len] -> [b, q_len, kv_len]
      attention_mask = self.attention_op.generate_attention_mask(
          query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask
      )
      if attention_mask is not None:
        attention_mask = attention_mask.squeeze(axis=(1, 2))
      # apply indexer, indexer_mask [b, q_len, kv_len]
      indexer_mask, _, indexer_score = self.indexer(
          inputs_q=inputs_q,
          low_rank_q=low_rank_q,
          inputs_kv=inputs_kv,
          inputs_positions=inputs_positions,
          attention_mask=attention_mask,
      )

      if indexer_mask is not None and self.config.indexer_loss_scaling_factor > 0.0:
        indexer_loss = self.calculate_indexer_loss(
            indexer_score=indexer_score,
            query=query,
            key=key,
            attention_mask=attention_mask,
            indexer_mask=indexer_mask,
            sparse_loss=self.config.indexer_sparse_training,
            scaling_factor=self.config.indexer_loss_scaling_factor,
        )
        self.sow(nnx.Intermediate, "indexer_loss", indexer_loss)

    # Check if we need QK Clip stats
    use_qk_clip = self.model_mode == MODEL_MODE_TRAIN and self.config.use_qk_clip

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.ds_paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      unnormalized_out = unnormalized_out[..., : self.v_head_dim]
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      out = self.attention_op(
          query,
          key,
          value,
          decoder_segment_ids,
          model_mode,
          cached_values,
          indexer_mask=indexer_mask,
          record_max_logits=use_qk_clip,
      )

    out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
    if model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      out = self._maybe_shard_with_logical(out, self.ep_out_axis_names)
    else:
      out = self._maybe_shard_with_logical(out, self.out_axis_names)

    out_sharding = create_sharding(self.mesh, out_logical_name)
    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache
