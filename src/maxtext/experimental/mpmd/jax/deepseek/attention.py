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

"""Multi-head Latent Attention (MLA), matching MaxText's `attention_mla.MLA`.

Query/key/value are produced by low-rank (LoRA) projections with a partial
RoPE, and the softmax scale -- including the squared YaRN `mscale` -- is folded
into the query before the attention kernel runs (`MLA.mla_query_projection`).  The
kernel itself therefore sees a scale of 1.0.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from maxtext.experimental.mpmd.jax.deepseek import layers
from maxtext.experimental.mpmd.jax.deepseek import rope
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig

# common_types.DEFAULT_MASK_VALUE
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def _causal_segment_mask(query_len, kv_len, segment_ids, kv_segment_ids=None, offset=0):
  """`attention_op.AttentionOp.generate_attention_mask`, train mode.

  `offset` shifts the queries along the keys (query i is at key position
  i + offset, splash's CausalMask offset); `kv_segment_ids` are the keys' own
  segment ids where they are not the queries' (`segment_ids` serves both by
  default)."""
  causal = (
      jax.lax.broadcasted_iota(jnp.int32, (query_len, kv_len), 1)
      <= jax.lax.broadcasted_iota(jnp.int32, (query_len, kv_len), 0) + offset
  )
  keep = causal[jnp.newaxis, :, :]
  if segment_ids is not None:
    kv_seg = segment_ids if kv_segment_ids is None else kv_segment_ids
    same = (segment_ids[:, :, jnp.newaxis] == kv_seg[:, jnp.newaxis, :]) & (kv_seg[:, jnp.newaxis, :] >= 0)
    keep = keep & same
  return keep


def dot_product_attention(query, key, value, segment_ids, config: DeepSeekConfig, kv_segment_ids=None, offset=0):
  """`attention_op.AttentionOp.apply_attention_dot`, train mode.

  Everything -- the QK product, the mask fill, the softmax and the PV product --
  stays in bf16, because `float32_qk_product` and `float32_logits` are both
  false.  This is *not* the same arithmetic as the splash kernel, which does the
  softmax in fp32; use it only when comparing against a MaxText run that also
  took this branch (`attention=dot_product`, or `max_target_length < 128`).
  """
  precision = layers.precision_of(config)
  b, t, n, _ = query.shape
  n_kv = key.shape[2]
  g = n // n_kv

  grouped_q = jnp.reshape(query, (b, t, n_kv, g, query.shape[-1]))
  attn_weights = jnp.einsum("btkgd,bskd->bkgts", grouped_q, key, precision=precision)

  keep = _causal_segment_mask(t, key.shape[1], segment_ids, kv_segment_ids, offset)[:, jnp.newaxis, jnp.newaxis, :, :]
  attn_weights = jnp.where(keep, attn_weights, DEFAULT_MASK_VALUE)

  logits = jnp.reshape(attn_weights, (b, n, t, key.shape[1]))
  local_max = jnp.max(logits, axis=-1, keepdims=True)
  local_exps = jnp.exp(logits - local_max)
  local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)
  local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))  # [b, t, n, 1]

  local_exps = jnp.reshape(local_exps, (b, n_kv, g, t, key.shape[1]))
  out = jnp.einsum("bkgts,bskd->btkgd", local_exps, value, precision=precision)
  out = jnp.reshape(out, (b, t, n, out.shape[-1]))
  return out / local_sum


def reference_attention(query, key, value, segment_ids, precision=None, kv_segment_ids=None, offset=0):
  """A readable fp32 stand-in for the splash kernel (softmax and PV in fp32).

  `precision` is the QK product's (`layers.precision_or_none`).  With bf16
  operands any precision is exact, but an fp32 model's QK product at DEFAULT
  is a bf16 pass on TPU.  The PV product is always HIGHEST: its probabilities
  are fp32 whatever the model's dtype.  `offset` / `kv_segment_ids`: see
  `_causal_segment_mask`.
  """
  t, s = query.shape[1], key.shape[1]
  logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=precision, preferred_element_type=jnp.float32)
  keep = _causal_segment_mask(t, s, segment_ids, kv_segment_ids, offset)[:, jnp.newaxis, :, :]
  logits = jnp.where(keep, logits, DEFAULT_MASK_VALUE)

  m = jnp.max(logits, axis=-1, keepdims=True)
  p = jnp.exp(logits - m)
  p = p * (1.0 / jnp.sum(p, axis=-1, keepdims=True))
  out = jnp.einsum("bhqk,bkhd->bqhd", p, value.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  return out.astype(query.dtype)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7), static_argnames=("offset",))
def splash_attention(
    query, key, value, segment_ids, num_heads, block_q, block_kv, block_kv_compute, kv_segment_ids=None, offset=0
):
  """The TPU Pallas kernel MaxText autoselects, called the same way (`offset` /
  `kv_segment_ids`: see `_causal_segment_mask`)."""
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))

  q_len, kv_len = query.shape[2], key.shape[2]
  mask = splash_attention_mask.CausalMask(shape=(q_len, kv_len), offset=offset)
  multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * num_heads)
  block_sizes = splash_attention_kernel.BlockSizes(
      block_q=min(block_q, q_len),
      block_kv=min(block_kv, kv_len),
      block_kv_compute=min(block_kv_compute, kv_len),
      block_q_dkv=min(block_q, q_len),
      block_kv_dkv=min(block_kv, kv_len),
      block_kv_dkv_compute=min(block_kv_compute, kv_len),
      block_q_dq=min(block_q, q_len),
      block_kv_dq=min(block_kv, kv_len),
      use_fused_bwd_kernel=False,
  )
  kernel = splash_attention_kernel.make_splash_mha(
      mask=multi_head_mask,
      head_shards=1,
      q_seq_shards=1,
      block_sizes=block_sizes,
      attn_logits_soft_cap=None,
      residual_checkpoint_name="context",
      interpret=jax.default_backend() != "tpu",
  )
  segments = None
  if segment_ids is not None:
    segments = splash_attention_kernel.SegmentIds(segment_ids, segment_ids if kv_segment_ids is None else kv_segment_ids)
  out = jax.vmap(kernel, in_axes=(0, 0, 0, 0, None))(query, key, value, segments, None)
  return jnp.transpose(out, axes=(0, 2, 1, 3))


@functools.cache
def _tokamax_splash_package():
  """Where tokamax keeps its splash attention: tokamax._src.ops.splash_attention after 0.0.17, else
  tokamax._src.ops.experimental.tpu.splash_attention -- looked up in that order, as MaxText's attention_op and lineage
  kernel do, and once, so that the kernel and its masks always come from one tree."""
  import importlib

  err = None
  for pkg in ("tokamax._src.ops.splash_attention", "tokamax._src.ops.experimental.tpu.splash_attention"):
    try:
      importlib.import_module(pkg)
      return pkg
    except ModuleNotFoundError as e:
      if not e.name or not pkg.startswith(e.name):  # a dependency the tree imports is missing: say which
        raise
      err = e
  raise err


def tokamax_splash_module(name):
  """tokamax's splash-attention module `name` (splash_attention_kernel, splash_attention_mask, ...), from the tree
  `_tokamax_splash_package` finds."""
  import importlib

  return importlib.import_module(f"{_tokamax_splash_package()}.{name}")


def tokamax_splash_attention(query, key, value, segment_ids, config: DeepSeekConfig, kv_segment_ids=None, offset=0):
  """MaxText `use_tokamax_splash=true` (`AttentionOp._create_sa_config`, `AttentionOp.tpu_flash_attention`).

  Tokamax splash always runs the fused backward.  On TPU, like the jax splash
  kernel, it cannot be auto-partitioned: call it on device-local arrays, e.g.
  inside a `shard_map`.  Off TPU it runs in interpret mode, plain JAX that
  partitions like any other op.
  """
  from jax.experimental import pallas as pl

  tsk, tsm = tokamax_splash_module("splash_attention_kernel"), tokamax_splash_module("splash_attention_mask")

  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))
  q_len, kv_len = query.shape[2], key.shape[2]

  def cost(flops):
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=0) if flops >= 0 else None

  sa_config = tsk.SplashConfig(
      block_q=min(config.sa_block_q, q_len),
      block_kv=min(config.sa_block_kv, kv_len),
      block_kv_compute=min(config.sa_block_kv_compute, kv_len),
      block_q_dkv=min(config.sa_block_q_dkv or config.sa_block_q, q_len),
      block_kv_dkv=min(config.sa_block_kv_dkv or config.sa_block_kv, kv_len),
      block_kv_dkv_compute=min(config.sa_block_kv_dkv_compute or config.sa_block_kv_compute, kv_len),
      use_fused_bwd_kernel=True,
      fuse_reciprocal=True,
      # MaxText's sa_use_base2_exp, on by default; DeepSeekConfig says what it does to the softmax
      use_base2_exp=config.sa_use_base2_exp,
      residual_checkpoint_name="context",
      fwd_cost_estimate=cost(config.sa_cost_estimate_flops_fwd),
      bwd_cost_estimate=cost(config.sa_cost_estimate_flops_bwd),
      interpret=jax.default_backend() != "tpu",
  )
  kernel = tsk.make_splash_mha(tsm.CausalMask(shape=(q_len, kv_len), offset=offset), config=sa_config, q_seq_shards=1)
  segments = None
  if segment_ids is not None:
    segments = tsk.SegmentIds(segment_ids, segment_ids if kv_segment_ids is None else kv_segment_ids)
  out = jax.vmap(lambda q, k, v, s: kernel(q, k, v, s))(query, key, value, segments)
  return jnp.transpose(out, axes=(0, 2, 1, 3))


def attention_op(query, key, value, segment_ids, config: DeepSeekConfig, kv_segment_ids=None, offset=0):
  """The attention kernel `config.attention_kernel` names: causal attention of
  `query` [b, t, n, d] over `key` / `value` [b, s, n, .], query i at key
  position i + `offset`; `kv_segment_ids` are the keys' segment ids where they
  differ from the queries' `segment_ids` (context parallelism) -- given
  without them, against queries of one segment, 1: the keys' ids alone mask.

  A Pallas kernel (splash, tokamax) runs on device-local arrays: on TPU it is
  a Mosaic custom call, which XLA cannot partition, so on a sub-mesh of more
  than one Auto device it does not even compile, and off TPU tokamax's
  interpret mode does not trace at the top level of an explicit program.
  ds_ep calls `mla` from inside its own `shard_map`, where both run.
  """
  if segment_ids is None and kv_segment_ids is not None:
    # every kernel reads kv_segment_ids only beside segment_ids
    segment_ids = jnp.ones(query.shape[:2], kv_segment_ids.dtype)
  return _attention_kernel(query, key, value, segment_ids, config, kv_segment_ids, offset)


def cudnn_attention(query, key, value, segment_ids, config: DeepSeekConfig, kv_segment_ids=None, offset=0):
  """cuDNN's fused flash attention (GPU), causal, through jax's cuDNN API
  (`jax._src.cudnn.fused_attention_stablehlo`, what MaxText's `cudnn_flash_jax`
  calls): [b, t, n, d] operands as they are (BTNH), GQA natively, the softmax
  in fp32, scale 1.0 (MLA and Qwen fold it into the query).  The query / key
  head dim may differ from the value's (MLA's 192 / 128) and go up to 256
  (Qwen's) on sm90 and later with cuDNN >= 9.10; sm80 takes <= 128 and equal
  dims.  `jax.nn.dot_product_attention` would refuse MLA's value shape.

  Unpacked sequences only: the segment ids are taken as one segment a row
  (ds_ep's are all ones), and a query offset or separate key segment ids
  (context parallelism) are refused."""
  if offset or kv_segment_ids is not None:
    raise NotImplementedError(
        "attention_kernel 'cudnn': no query offset / key segment ids (context " "parallelism); use cp 1"
    )
  del segment_ids  # one segment a row: the causal mask is the whole mask
  from jax._src.cudnn import fused_attention_stablehlo as cudnn_fa

  return cudnn_fa.dot_product_attention(
      query, key, value, scale=1.0, mask_type=cudnn_fa.MaskType.CAUSAL, qkv_layout="BTNH"
  )


def _attention_kernel(query, key, value, segment_ids, config: DeepSeekConfig, kv_segment_ids=None, offset=0):
  if config.attention_kernel == "cudnn":
    return cudnn_attention(query, key, value, segment_ids, config, kv_segment_ids, offset)
  if config.attention_kernel == "tokamax_splash":
    return tokamax_splash_attention(query, key, value, segment_ids, config, kv_segment_ids, offset)
  if config.attention_kernel == "splash":
    return splash_attention(
        query,
        key,
        value,
        segment_ids,
        config.num_query_heads,
        config.sa_block_q,
        config.sa_block_kv,
        config.sa_block_kv_compute,
        kv_segment_ids,
        offset=offset,
    )
  if config.attention_kernel == "dot_product":
    return dot_product_attention(query, key, value, segment_ids, config, kv_segment_ids, offset)
  if config.attention_kernel == "reference":
    return reference_attention(query, key, value, segment_ids, layers.precision_or_none(config), kv_segment_ids, offset)
  raise ValueError(f"unknown attention_kernel: {config.attention_kernel}")


def mla(params, inputs, positions, segment_ids, freqs_cis, config: DeepSeekConfig, kv_context=None, norm=None):
  """`MLA.__call__` for MODEL_MODE_TRAIN.

  `kv_context`, for a sequence split over devices (context parallelism):
  `(kv latent [b, s, kv_lora], roped key [b, s, 1, rope], segment_ids) -> (the
  same over the keys these queries attend, their segment ids against
  `segment_ids` -- or, with None, against queries of one segment, 1 --, the
  queries' key offset)`, e.g. ds_ep's, which gathers the other chunks of the
  sequence.  The keys and values are then projected from the gathered latent.

  `norm`: the pre-attention RMSNorm's params, which `inputs` -- the residual
  stream, then -- go through first.  Under lean residuals
  (`layers.lean_residuals`) the norm and the two projections that read its
  output run in one checkpoint (`layers.rms_norm_read`), so that a VJP keeps
  the stream rather than the normalised activation; outside them this is the
  norm applied first.  Likewise the query's norm and its up-projection, and,
  without `kv_context`, the key / value latent's norm and its up-projection
  (with it the gathered latent comes between the two).

  Args:
    params: `wq_a`, `q_norm`, `wq_b` (or `query` when `q_lora_rank == 0`),
      `wkv_a`, `kv_norm`, `wkv_b`, `out`.
    inputs: `[b, s, emb_dim]`.
    positions: `[b, s]` token positions for RoPE.
    segment_ids: `[b, s]` packing segment ids, or None.
    freqs_cis: YaRN rotation table.
    config: model config.
    kv_context: see above, or None.
    norm: see above, or None.
  """
  b = inputs.shape[0]
  heads = config.num_query_heads

  def q_down(x):
    return layers.dense_general(params["query" if config.q_lora_rank == 0 else "wq_a"], x, config)

  def kv_down(x):
    return layers.dense_general(params["wkv_a"], x, config)

  if norm is not None and layers.lean():
    q_in, kv_in = layers.rms_norm_read(norm, inputs, config, lambda x: (q_down(x), kv_down(x)))
  else:
    if norm is not None:
      inputs = layers.rms_norm(norm, inputs, config)
    q_in, kv_in = q_down(inputs), None  # the key / value latent's projection where MaxText runs it, below

  # --- query: LoRA down -> RMSNorm -> LoRA up -> partial RoPE -> scale -------
  # `q_lora_rank == 0` is a plain projection, not a rank-0 LoRA
  # (`MLA._init_projections`, `MLA.mla_query_projection`).  Running the LoRA path at rank 0 would make
  # `wq_a` `[emb, 0]` and every query identically zero -- uniform attention,
  # silently -- so the two are different parameter trees, as in MaxText.
  if config.q_lora_rank == 0:
    q = q_in
  else:
    low_rank_q = jax.ad_checkpoint.checkpoint_name(q_in, "query_wa_proj")
    q = layers.rms_norm_read(
        params["q_norm"],
        low_rank_q,
        config,
        lambda lq: layers.dense_general(params["wq_b"], jax.ad_checkpoint.checkpoint_name(lq, "mla_q"), config),
    )
  q_nope, q_pe = jnp.split(q, [config.qk_nope_head_dim], axis=-1)
  q_pe = rope.apply_yarn(q_pe, positions, freqs_cis, config)
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * config.softmax_scale

  # --- key/value: shared LoRA latent, rope on the shared 1-head slice -------
  low_rank = kv_down(inputs) if kv_in is None else kv_in
  low_rank = jax.ad_checkpoint.checkpoint_name(low_rank, "kv_wa_proj")
  low_rank_main, low_rank_rope = jnp.split(low_rank, [config.kv_lora_rank], axis=-1)

  def kv_up(lk):
    return layers.dense_general(params["wkv_b"], lk, config)

  kv_out = None
  if kv_context is None and layers.lean():
    # the latent's norm and its up-projection in one checkpoint, as the query's: no VJP keeps the norm's output
    kv_out = layers.rms_norm_read(
        params["kv_norm"], low_rank_main, config, lambda lk: kv_up(jax.ad_checkpoint.checkpoint_name(lk, "mla_kv"))
    )
  else:
    # outside lean residuals the norm, then the projection below, where MaxText runs it; under context
    # parallelism the gathered latent comes between the two, and wkv_b's VJP keeps that
    low_rank_main = layers.rms_norm(params["kv_norm"], low_rank_main, config)
    low_rank_main = jax.ad_checkpoint.checkpoint_name(low_rank_main, "mla_kv")
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = rope.apply_yarn(key_rope, positions, freqs_cis, config)
  kv_segment_ids, offset = None, 0
  if kv_context is not None:
    low_rank_main, key_rope, kv_segment_ids, offset = kv_context(low_rank_main, key_rope, segment_ids)

  if kv_out is None:
    kv_out = kv_up(low_rank_main)
  key_nope, value = jnp.split(kv_out, [config.qk_nope_head_dim], axis=-1)
  key_rope = jnp.broadcast_to(key_rope, (b, key_rope.shape[1], heads, config.qk_rope_head_dim))
  key = jnp.concatenate([key_nope, key_rope], axis=-1)

  query = jax.ad_checkpoint.checkpoint_name(query, "query_proj")
  key = jax.ad_checkpoint.checkpoint_name(key, "key_proj")
  value = jax.ad_checkpoint.checkpoint_name(value, "value_proj")

  out = attention_op(query, key, value, segment_ids, config, kv_segment_ids, offset)
  # Under lean residuals the output takes no name: the out projection's VJP then keeps the very array the kernel's
  # VJP keeps (cuDNN's keeps its output), where the name, an array of its own, would be a second copy of it.  Only
  # remat policies read the name, and a saved-residual forward applies none.
  if not layers.lean():
    out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
  out = layers.dense_general(params["out"], out, config, axis=(-2, -1))
  return jax.ad_checkpoint.checkpoint_name(out, "out_proj")
