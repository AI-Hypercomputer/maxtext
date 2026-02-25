# Copyright 2023–2026 Google LLC
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


"""Alternative DeepSeek model definition with batch-split schedule."""

import dataclasses
import functools
import math
from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
from maxtext.kernels import megablox, sort_activations
from maxtext.layers import attention_op
from maxtext.layers import moe as moe_lib
from maxtext.layers import quantizations
import qwix.pallas as qpl
import tokamax


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(
        1,
        2,
        3,
    ),
)
def quantized_psum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool) -> jax.Array:
  """Forward: Standard BF16 Reduce-Scatter.

  Backward: Quantized FP8 All-Gather (DeepSeek optimization).

  Args:
    x: The input tensor.
    axis_name: The axis name for the psum_scatter/all_gather operation.
    scatter_dimension: The dimension along which to scatter.
    tiled: Whether the scatter/gather is tiled.

  Returns:
    The result of the reduce-scatter operation.
  """
  return _q_psum_scatter_fwd(x, axis_name, scatter_dimension, tiled)[0]


def _q_psum_scatter_fwd(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool) -> tuple[jax.Array, None]:
  out = jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=tiled)
  return out, None


def _q_psum_scatter_bwd(
    axis_name: str,
    scatter_dimension: int,
    tiled: bool,
    res: Any,
    grads: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  """Backward pass for quantized_psum_scatter.

  Performs a quantized All-Gather of the gradients.

  Args:
    axis_name: The axis name for the all_gather operation.
    scatter_dimension: The dimension along which the scatter occurred in the
      forward pass.
    tiled: Whether the gather is tiled.
    res: The residuals from the forward pass (_q_psum_scatter_fwd), containing
      (axis_name, scatter_dimension, tiled).
    grads: The gradients from the next layer, which are in BF16.

  Returns:
    The dequantized and all-gathered gradients.
  """
  del res
  # --- BACKWARD PASS (Dispatch) ---
  # 'grads' is the BF16 gradient arriving from the next layer.
  # We need to broadcast it back to all devices (All-Gather).

  grads_q = qpl.quantize(
      grads,
      jnp.float8_e5m2,
      channelwise_axes=[0],
  )

  gathered_qvals = jax.lax.all_gather(grads_q.qvalue, axis_name=axis_name, tiled=tiled, axis=scatter_dimension)

  return (qpl.dequantize(dataclasses.replace(grads_q, qvalue=gathered_qvals)),)


quantized_psum_scatter.defvjp(_q_psum_scatter_fwd, _q_psum_scatter_bwd)


def fetch_weights(params, dtype):
  """Fetches weights from params in the proper format for batch-split schedule."""
  return jax.tree.map(
      # If x is a LogicallyPartitioned array, then x.value is the underlying
      # array. If not, use the array directly.
      lambda x: jnp.asarray(getattr(x, "value", x)[...], dtype),
      (
          (
              (
                  params["pre_self_attention_layer_norm"]["scale"],
                  params["post_self_attention_layer_norm"]["scale"],
              ),
              (
                  params["self_attention"]["wq_a"]["kernel"],
                  params["self_attention"]["wq_b"]["kernel"],
                  params["self_attention"]["q_norm"]["scale"],
                  params["self_attention"]["wkv_a"]["kernel"],
                  params["self_attention"]["wkv_b"]["kernel"],
                  params["self_attention"]["kv_norm"]["scale"],
                  params["self_attention"]["out"]["kernel"],
              ),
          ),
          (
              (
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"],
              ),
              (
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_1"],
                  params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wo"],
              ),
              (
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wi_0"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wi_1"]["kernel"],
                  params["DeepSeekMoeBlock_0"]["shared_experts"]["wo"]["kernel"],
              ),
          ),
      ),
      is_leaf=lambda x: not isinstance(x, Sequence),
  )


@jax.named_scope("deepseek_batchsplit_split")
def split(x, split_factor=2):
  """Splits the input into `split_factor` parts along the batch dimension."""
  if split_factor == 1:
    return [x]
  if x is None:
    return [None] * split_factor
  else:
    x = jnp.reshape(x, (-1, split_factor) + x.shape[1:])
    return [x[:, i, ...] for i in range(split_factor)]


@jax.named_scope("deepseek_batchsplit_merge")
def merge(x, split_factor=2):
  """Merges the input microbatches back into a single tensor."""
  if split_factor == 1:
    return x[0]
  x = jnp.stack(x, axis=1)
  return jnp.reshape(x, (-1,) + x.shape[2:])


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
    policy,
):
  """Scans the layers with batch-split schedule."""

  def batch_split_scan_fn(inputs, weights, dpos, dseg):
    xs = batch_split_schedule(
        inputs,
        weights,
        dpos,
        dseg,
        model_mode=model_mode,
        mesh=mesh,
        quant=quant,
        cfg=cfg,
    )
    return xs, None

  batch_split_scan_fn_checkpointed = jax.checkpoint(
      batch_split_scan_fn,
      # No need to prevent CSE inside scan.
      prevent_cse=False,
      policy=policy,
  )
  weights = fetch_weights(params, cfg.dtype)
  # `jax.lax.scan` expects the leading dimension of weights to be the scan
  # dimension, but the weights are initialized/loaded with the param scan
  # axis as the scan dimension, so swap the axes.
  weights = jax.tree.map(lambda x: jnp.swapaxes(x, 0, cfg.param_scan_axis), weights)

  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  inputs = jax.shard_map(
      functools.partial(split, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=activation_pspec,
      out_specs=[activation_pspec] * cfg.batch_split_factor,
  )(inputs)
  dpos = split(positions, split_factor=cfg.batch_split_factor)
  dseg = split(segment_ids, split_factor=cfg.batch_split_factor)
  outputs, _ = jax.lax.scan(
      functools.partial(batch_split_scan_fn_checkpointed, dpos=dpos, dseg=dseg),
      inputs,
      weights,
  )
  outputs = jax.shard_map(
      functools.partial(merge, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=([activation_pspec] * cfg.batch_split_factor,),
      out_specs=activation_pspec,
  )(outputs)
  return outputs


def batch_split_schedule(
    inputs,
    weights,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  xs = [with_data_parallel_constraint(x, mesh) for x in inputs]
  xs = jax.ad_checkpoint.checkpoint_name(xs, "decoder_layer_input")

  attn_op = attention_op.AttentionOp(
      config=cfg,
      mesh=mesh,
      attention_kernel=cfg.attention,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      quant=quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      attention_type=cfg.attention_type,
  )
  norm_mla_ws, moe_ws = weights
  xs = mla_with_norms(
      xs,
      norm_mla_ws,
      positions,
      segment_ids,
      mesh=mesh,
      model_mode=model_mode,
      attn_op=attn_op,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      rope_max_timescale=cfg.rope_max_timescale,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      dtype=cfg.dtype,
  )

  xs = moe(
      xs,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      expert_axis_name="expert",
      use_gather_mosaic_kernel=False,
      config=cfg,
  )
  return xs


def staggered_call(fn, xs):
  for i, x in enumerate(xs):
    if i == len(xs) - 1:
      xs[i] = fn(x)
    else:
      xs[i], xs[i + 1] = jax.lax.optimization_barrier((fn(x), xs[i + 1]))
  return xs


def with_data_parallel_constraint(x, mesh):
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  return jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, activation_pspec))


def dot(x, y, axes=1):
  return jnp.tensordot(x, y, axes=axes)


def mla_with_norms(
    inputs,
    weights,
    decoder_positions,
    decoder_segment_ids,
    *,
    mesh,
    model_mode,
    attn_op,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_max_timescale,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    dtype,
):
  """Performs MLA with pre- and post-normalization."""
  (pre_attn_scale, post_attn_scale), attn_ws = weights

  def fn(args):
    x, dseg, dpos = args
    y = rms_norm(
        x,
        pre_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )
    out = x + with_data_parallel_constraint(
        mla(
            y,
            dpos,
            dseg,
            attn_ws,
            model_mode=model_mode,
            epsilon=normalization_layer_epsilon,
            kv_lora_rank=kv_lora_rank,
            kv_norm_epsilon=normalization_layer_epsilon,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_theta=rope_max_timescale,
            num_query_heads=num_query_heads,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            rope_factor=rope_factor,
            dtype=dtype,
            mscale=mscale,
            attention_op_fn=attn_op,
        ),
        mesh,
    )
    return out, rms_norm(
        out,
        post_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )

  return staggered_call(fn, list(zip(inputs, decoder_segment_ids, decoder_positions)))


def mla(
    inputs,
    positions,
    segment_ids,
    weights,
    *,
    model_mode,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    attention_op_fn,
    dtype,
):
  """Performs MLA."""
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query = query_projection(
      inputs,
      positions,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
  )
  query = jax.ad_checkpoint.checkpoint_name(query, "query_proj")
  key, value = kv_projection(
      inputs,
      positions,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
  )
  key = jax.ad_checkpoint.checkpoint_name(key, "key_proj")
  value = jax.ad_checkpoint.checkpoint_name(value, "value_proj")
  out = attention_op_fn(
      query,
      key,
      value,
      segment_ids,
      model_mode,
      cached_values=[None, None],
  )
  out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
  out = dot(out, out_weights, axes=2)
  out = jax.ad_checkpoint.checkpoint_name(out, "out_proj")
  return out


def query_projection(
    inputs_q,
    inputs_positions,
    wq_a_weights,
    wq_b_weights,
    q_norm_scale_weights,
    *,
    epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    mscale,
):
  """Performs query projection."""
  # Set softmax scaling.
  qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
  softmax_scale = qk_head_dim**-0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m

  # LoRA path
  low_rank_q = dot(inputs_q, wq_a_weights)
  low_rank_q = rms_norm(
      low_rank_q,
      q_norm_scale_weights,
      epsilon=epsilon,
      dtype=dtype,
  )
  low_rank_q = jax.ad_checkpoint.checkpoint_name(low_rank_q, "mla_q")
  q = dot(low_rank_q, wq_b_weights)

  # Split into non-positional and rotary parts.
  q_nope, q_pe = jnp.split(q, [qk_nope_head_dim], axis=-1)
  q_pe = yarn(
      q_pe,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale
  return query


def kv_projection(
    inputs,
    inputs_positions,
    wkv_a_weights,
    wkv_b_weights,
    kv_norm_scale_weights,
    *,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    qk_nope_head_dim,
    num_query_heads,
):
  """Performs KV projection."""
  low_rank = dot(inputs, wkv_a_weights)
  low_rank_main, low_rank_rope = jnp.split(low_rank, [kv_lora_rank], axis=-1)
  low_rank_main = rms_norm(
      low_rank_main,
      kv_norm_scale_weights,
      epsilon=kv_norm_epsilon,
      dtype=dtype,
  )
  low_rank_main = jax.ad_checkpoint.checkpoint_name(low_rank_main, "mla_kv")
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = yarn(
      key_rope,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )

  return get_key_value(
      low_rank_main,
      key_rope,
      wkv_b_weights,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
  )


def get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads):
  """Gets key and value from compressed KV latent vector and key rope."""
  kv_out = dot(low_rank_main, wkv_b_weights)

  # Split kv_out into key_nope and value parts.
  key_nope, value = jnp.split(kv_out, [qk_nope_head_dim], axis=-1)
  key_rope = jnp.broadcast_to(
      key_rope,
      (
          key_nope.shape[0],
          key_nope.shape[1],
          num_query_heads,
          key_rope.shape[3],
      ),
  )

  key = jnp.concatenate([key_nope, key_rope], axis=-1)

  return key, value


def rms_norm(x, scale, *, epsilon, dtype):
  """RMS normalization."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
  return jnp.einsum("i...k,...k->i...k", y, scale)


def yarn(
    inputs,
    positions,
    *,
    embedding_dims,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    fprop_dtype,
):
  """Performs YaRN rotary embedding."""
  # Initialize the swap and negate mask.
  indices = jnp.arange(embedding_dims)
  # [1, 0, 3, 2, 5, 4, ...]
  swap_indices = jnp.where(indices % 2 == 0, indices + 1, indices - 1)
  negation_mask = jnp.where(indices % 2 == 0, -1, 1)
  identity = jnp.eye(embedding_dims, dtype=jnp.int32)
  pairwise_swap_and_negate_mask = identity[swap_indices] * negation_mask

  # Calculate the frequencies.
  half_dim = embedding_dims // 2
  # Compute base frequencies for each (even-indexed) dimension.
  # (Note: We use jnp.arange with float32 for precision.)
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / embedding_dims))

  low = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  high = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  low = max(math.floor(low), 0)
  high = min(math.ceil(high), embedding_dims - 1)
  diff = high - low if high > low else 0.001
  linear_func = (jnp.arange(half_dim, dtype=jnp.float32) - low) / diff
  smooth = 1 - jnp.clip(linear_func, 0, 1)
  # The corrected frequency is a weighted mix of the scaled and base values.
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

  # Precompute frequencies for all positions by taking the outer product.
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
  # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
  freqs = jnp.outer(t, freqs)

  # Lookup the precomputed frequencies using the position indices.
  # self.freqs has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
  # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
  freqs = jnp.take(freqs, positions, axis=0)  # shape: [B, S, half_dim]
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]
  freqs = jnp.repeat(freqs, 2, axis=-1)  # shape: [B, S, 1, embedding_dims]
  # inputs @ mask: [B, S, N, embedding_dims] @ [embedding_dims, embedding_dims] -> [B, S, N, embedding_dims]
  output = inputs * jnp.cos(freqs) + jnp.matmul(inputs, pairwise_swap_and_negate_mask) * jnp.sin(freqs)
  return output.astype(fprop_dtype)


def moe(
    inputs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  xs, ys = list(zip(*inputs))
  ys = with_data_parallel_constraint(
      process_activations(
          ys,
          weights,
          mesh=mesh,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
      ),
      mesh,
  )
  return [x + y for x, y in zip(xs, ys)]


def expert_indices_and_weights(
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes expert indices for each token and their corresponding weights."""
  _, indices = jax.lax.top_k(
      gate_logits,
      k=num_experts_per_tok,
  )
  weights = jnp.take_along_axis(pre_bias_logits, indices, axis=-1)
  weights = routed_scaling_factor * (weights / weights.sum(-1, keepdims=True))
  return indices, weights


def expert_selection(
    x,
    routing_kernel,
    routing_bias,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
):
  """Selects experts for each token and calculates group sizes for each expert."""
  pre_bias_logits = jax.nn.sigmoid(dot(x, routing_kernel))
  logits = pre_bias_logits + routing_bias

  selected_experts, weights = expert_indices_and_weights(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
  )
  group_sizes = jnp.bincount(jnp.ravel(selected_experts), length=num_experts)
  return selected_experts, weights, group_sizes


def route(
    x,
    selected_experts,
    weights,
    group_sizes,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """All-gather tokens and then perform local routing."""
  # Communicate local results across the expert axis.
  x = jax.lax.all_gather(x, axis_name=expert_axis_name, tiled=True)
  weights = jax.lax.all_gather(weights, axis_name=expert_axis_name, tiled=True)
  selected_experts = jax.lax.all_gather(selected_experts, axis_name=expert_axis_name, tiled=True)
  group_sizes = jax.lax.psum(group_sizes, axis_name=expert_axis_name)

  # Sort the gathered tokens and weights.
  weights = jnp.ravel(weights)[jnp.argsort(jnp.ravel(selected_experts))]
  x = sort_activations.route(
      x,
      selected_experts,
      use_custom_mosaic_kernel=use_gather_mosaic_kernel,
  )

  return x, selected_experts, weights, group_sizes


def unroute(
    x,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """Undo `route()`."""
  # Unsort the output.
  x = sort_activations.unroute(
      x,
      selected_experts,
      use_custom_mosaic_kernel=use_gather_mosaic_kernel,
  )

  # Sum across expert shards.
  return jax.lax.psum_scatter(x, expert_axis_name, scatter_dimension=0, tiled=True)


def compute(x, w0, w1, wo, group_sizes, weights, *, config, mesh):
  """Processes routed tokens through the MLP."""

  def gmm(
      inputs,
      kernel,
      tiling,
      group_sizes,
      preferred_element_type,
      weight_gather_axes,
      input_buffer_count,
      combine_scopes,
  ):
    if config.use_qwix_quantization:
      output = megablox.gmm(
          lhs=inputs,
          rhs=kernel,
          group_sizes=group_sizes,
          preferred_element_type=preferred_element_type,
          tiling=tiling,
          use_qwix_quantization=config.use_qwix_quantization,
          use_tokamax_backend=config.use_tokamax_gmm,
          weight_gather_axes=weight_gather_axes,
          input_buffer_count=input_buffer_count,
          combine_scopes=combine_scopes,
      )
    else:
      output = tokamax.ragged_dot(
          lhs=inputs,
          rhs=kernel,
          group_sizes=group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=preferred_element_type,
          implementation="mosaic",
      )
    return output

  gmm_fn = functools.partial(gmm, group_sizes=group_sizes, preferred_element_type=config.dtype)
  wi_gather_axes = []
  wo_gather_axes = []

  wi_tile_size = (
      config.wi_tile_fwd_batch_seq,
      config.wi_tile_fwd_embed_dim,
      config.wi_tile_fwd_mlp_dim,
      config.wi_tile_dlhs_batch_seq,
      config.wi_tile_dlhs_embed_dim,
      config.wi_tile_dlhs_mlp_dim,
      config.wi_tile_drhs_batch_seq,
      config.wi_tile_drhs_embed_dim,
      config.wi_tile_drhs_mlp_dim,
  )
  wo_tile_size = (
      config.wo_tile_fwd_batch_seq,
      config.wo_tile_fwd_embed_dim,
      config.wo_tile_fwd_mlp_dim,
      config.wo_tile_dlhs_batch_seq,
      config.wo_tile_dlhs_embed_dim,
      config.wo_tile_dlhs_mlp_dim,
      config.wo_tile_drhs_batch_seq,
      config.wo_tile_drhs_embed_dim,
      config.wo_tile_drhs_mlp_dim,
  )
  wi_input_buffer_count = (
      config.wi_tile_fwd_buffer_count,
      config.wi_tile_dlhs_buffer_count,
      config.wi_tile_drhs_buffer_count,
  )
  wo_input_buffer_count = (
      config.wo_tile_fwd_buffer_count,
      config.wo_tile_dlhs_buffer_count,
      config.wo_tile_drhs_buffer_count,
  )

  wi_combine_scopes = config.wi_combine_scopes
  wo_combine_scopes = config.wo_combine_scopes
  if config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    w0_pspec = nn.logical_to_mesh_axes(gating_pspec)
    wo_pspec = nn.logical_to_mesh_axes(linear_pspec)
    ignored_axes = ("expert", "tensor", "tensor_transpose")

    def get_active_sharding_axes(pspec_dim_axes, tensor_dim_index):
      if pspec_dim_axes is None:
        return []
      axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
      active = []
      for ax in axes:
        if ax and ax not in ignored_axes and mesh.shape.get(ax, 1) > 1:
          active.append((ax, tensor_dim_index))
      return active

    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[0], 0))
    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[2], 2))

    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[0], 0))
    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[1], 1))

  layer_w0 = gmm_fn(
      x,
      w0,
      tiling=wi_tile_size,
      weight_gather_axes=wi_gather_axes,
      input_buffer_count=wi_input_buffer_count,
      combine_scopes=wi_combine_scopes,
  )
  layer_w1 = gmm_fn(
      x,
      w1,
      tiling=wi_tile_size,
      weight_gather_axes=wi_gather_axes,
      input_buffer_count=wi_input_buffer_count,
      combine_scopes=wi_combine_scopes,
  )
  layer_w0 = jax.ad_checkpoint.checkpoint_name(layer_w0, "mlpwi_0")
  layer_w1 = jax.ad_checkpoint.checkpoint_name(layer_w1, "mlpwi_1")
  intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
  intermediate_layer *= weights[:, None]
  layer_wo = gmm_fn(
      intermediate_layer,
      wo,
      tiling=wo_tile_size,
      weight_gather_axes=wo_gather_axes,
      input_buffer_count=wo_input_buffer_count,
      combine_scopes=wo_combine_scopes,
  )
  return layer_wo


def route_compute_unroute(
    xs,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    mesh,
):
  """Routes, processes, and unroutes activations."""
  orig_shape = xs[0].shape
  (
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn(inputs):
    # Shared expert.
    y = dot(jax.nn.silu(dot(inputs, shared_w0)) * dot(inputs, shared_w1), shared_wo)

    inputs = jnp.reshape(inputs, (-1, inputs.shape[-1]))
    selected_experts, weights, group_sizes = expert_selection(
        inputs,
        gate_kernel,
        gate_bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
    )
    x, selected_experts, weights, group_sizes = route(
        inputs,
        selected_experts,
        weights,
        group_sizes,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return x, y, selected_experts, weights, group_sizes

  def compute_fn(inputs):
    x, y, selected_experts, weights, group_sizes = inputs
    x = compute(
        x,
        routed_w0,
        routed_w1,
        routed_wo,
        group_sizes,
        weights,
        config=config,
        mesh=mesh,
    )
    return x, y, selected_experts

  def unroute_fn(inputs):
    x, y, selected_experts = inputs
    x = unroute(
        x,
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return jnp.reshape(x, orig_shape) + y

  xs = staggered_call(route_fn, xs)
  xs = staggered_call(compute_fn, xs)
  xs = staggered_call(unroute_fn, xs)
  return xs


def process_activations(
    xs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
):
  """Processes activations, which are fully sharded on the batch axis, with tensor/expert sharded weights."""
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  if config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    gating_pspec = nn.logical_to_mesh_axes(gating_pspec)
    linear_pspec = nn.logical_to_mesh_axes(linear_pspec)
  else:
    gating_pspec = jax.sharding.PartitionSpec(None, None, expert_axis_name)
    linear_pspec = jax.sharding.PartitionSpec(None, expert_axis_name, None)
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
          mesh=mesh,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * len(xs),
          (
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  gating_pspec,
                  gating_pspec,
                  linear_pspec,
              ),
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
              ),
          ),
      ),
      out_specs=activation_pspec,
      check_vma=False,
  )([x.astype(config.dtype) for x in xs], weights)
