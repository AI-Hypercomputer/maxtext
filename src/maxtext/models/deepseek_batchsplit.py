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


"""Alternative DeepSeek model definition with batch-split schedule.

The model logic and optimizations are very explicit in this implementation.
Weights are explicitly pre-fetched and gathered in the forward pass and gradients
are explicitly reduced and post-scattered in the backward pass. Optimization
barriers are used to enforce ordering of both large blocks of operations (e.g.
attention, dispatch, etc) and individual operations (e.g. AG+gather within
dispatch). In order to control remat, residuals from the forward pass are
explicitly stored and passed to the backward pass in a custom VJP over the
entire layer scan. The backward pass comprises of remat/bwd functions for each
forward pass function, with relevant residuals passed between them.
"""

import contextlib
import functools
import math
from typing import Sequence

import jax
import jax.numpy as jnp
from maxtext.kernels import attention, sort_activations

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask


def scheduling_group(group_id) -> contextlib.AbstractContextManager[None]:
  return jax.experimental.xla_metadata.set_xla_metadata(_scheduling_group_id=group_id)


def fetch_weights(params, dtype):
  """Fetches weights from params in the proper format for batch-split schedule."""
  return jax.tree.map(
      # If x is a LogicallyPartitioned array, then x.value is the underlying
      # array. If not, use the array directly.
      lambda x: jnp.asarray(getattr(x, "value", x)[...], dtype),
      (
          (
              params["pre_self_attention_layer_norm"]["scale"],
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
              params["post_self_attention_layer_norm"]["scale"],
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


def extract_layer_weights(all_weights, layer_idx, layer_axis):
  """Extracts the weights for given layer."""
  return jax.tree.map(
      lambda x: jax.lax.dynamic_index_in_dim(x, layer_idx, axis=layer_axis, keepdims=False),
      all_weights,
  )


def insert_layer_ws_grad(all_ws_grad, ws_grad, layer_idx, layer_axis):
  """Inserts the weight gradients for given layer."""
  return jax.tree.map(
      lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, layer_idx, axis=layer_axis),
      all_ws_grad,
      ws_grad,
  )


def gather_weights(weights, mesh):
  """all-gathers FSDP sharded weights."""

  def fn(weights):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = weights
    wq_a = jax.lax.pcast(wq_a, axis_name="data", to="reduced")
    wq_b = jax.lax.pcast(wq_b, axis_name="data", to="reduced")
    wkv_a = jax.lax.pcast(wkv_a, axis_name="data", to="reduced")
    wkv_b = jax.lax.pcast(wkv_b, axis_name="data", to="reduced")
    out = jax.lax.pcast(out, axis_name="data", to="reduced")
    gate = jax.lax.pcast(gate, axis_name="data", to="reduced")
    routed_wi_0 = jax.lax.pcast(routed_wi_0, axis_name="data", to="reduced")
    routed_wi_1 = jax.lax.pcast(routed_wi_1, axis_name="data", to="reduced")
    routed_wo = jax.lax.pcast(routed_wo, axis_name="data", to="reduced")
    shared_wi_0 = jax.lax.pcast(shared_wi_0, axis_name="data", to="reduced")
    shared_wi_1 = jax.lax.pcast(shared_wi_1, axis_name="data", to="reduced")
    shared_wo = jax.lax.pcast(shared_wo, axis_name="data", to="reduced")
    # Cast to reduced across expert axis for all weights that are replicated
    # across the expert axis. This transposes to an all-reduce across the expert
    # axis in the backward pass.
    wq_a = jax.lax.pcast(wq_a, axis_name="expert", to="reduced")
    wq_b = jax.lax.pcast(wq_b, axis_name="expert", to="reduced")
    wkv_a = jax.lax.pcast(wkv_a, axis_name="expert", to="reduced")
    wkv_b = jax.lax.pcast(wkv_b, axis_name="expert", to="reduced")
    out = jax.lax.pcast(out, axis_name="expert", to="reduced")
    gate = jax.lax.pcast(gate, axis_name="expert", to="reduced")
    shared_wi_0 = jax.lax.pcast(shared_wi_0, axis_name="expert", to="reduced")
    shared_wi_1 = jax.lax.pcast(shared_wi_1, axis_name="expert", to="reduced")
    shared_wo = jax.lax.pcast(shared_wo, axis_name="expert", to="reduced")
    # All-gather across FSDP axis. Setting to="reduced" transposes to a
    # reduce-scatter across the FSDP axis in the backward pass.
    wq_a = jax.lax.all_gather(wq_a, axis_name="fsdp", tiled=True, to="reduced")
    wq_b = jax.lax.all_gather(wq_b, axis_name="fsdp", tiled=True, to="reduced")
    wkv_a = jax.lax.all_gather(wkv_a, axis_name="fsdp", tiled=True, to="reduced")
    wkv_b = jax.lax.all_gather(wkv_b, axis_name="fsdp", tiled=True, to="reduced")
    out = jax.lax.all_gather(out, axis_name="fsdp", tiled=True, axis=2, to="reduced")
    gate = jax.lax.all_gather(gate, axis_name="fsdp", tiled=True, to="reduced")
    routed_wi_0 = jax.lax.all_gather(routed_wi_0, axis_name="fsdp", tiled=True, to="reduced")
    routed_wi_1 = jax.lax.all_gather(routed_wi_1, axis_name="fsdp", tiled=True, to="reduced")
    routed_wo = jax.lax.all_gather(routed_wo, axis_name="fsdp", tiled=True, to="reduced")
    shared_wi_0 = jax.lax.all_gather(shared_wi_0, axis_name="fsdp", tiled=True, to="reduced")
    shared_wi_1 = jax.lax.all_gather(shared_wi_1, axis_name="fsdp", tiled=True, to="reduced")
    shared_wo = jax.lax.all_gather(shared_wo, axis_name="fsdp", tiled=True, axis=1, to="reduced")
    return (
        (
            pre_attn_norm,
            (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
        ),
        (
            post_attn_norm,
            (gate, bias),
            (routed_wi_0, routed_wi_1, routed_wo),
            (shared_wi_0, shared_wi_1, shared_wo),
        ),
    )

  # Ensure weight AGs aren't fused with ops that use the weights.
  return jax.lax.optimization_barrier(
      jax.shard_map(
          fn,
          mesh=mesh,
          in_specs=(
              (
                  (
                      jax.sharding.PartitionSpec(None),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None, None),
                          jax.sharding.PartitionSpec(None),
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None, None),
                          jax.sharding.PartitionSpec(None),
                          jax.sharding.PartitionSpec(None, None, "fsdp"),
                      ),
                  ),
                  (
                      jax.sharding.PartitionSpec(None),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec(None),
                      ),
                      (
                          jax.sharding.PartitionSpec("fsdp", None, "expert"),
                          jax.sharding.PartitionSpec("fsdp", None, "expert"),
                          jax.sharding.PartitionSpec("fsdp", "expert", None),
                      ),
                      (
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec("fsdp", None),
                          jax.sharding.PartitionSpec(None, "fsdp"),
                      ),
                  ),
              ),
          ),
          out_specs=(
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, None, reduced={"data", "fsdp", "expert"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, "expert", reduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec(None, None, "expert", reduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec(None, "expert", None, reduced={"data", "fsdp"}),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  ),
              ),
          ),
          check_vma=True,
      )(weights)
  )


def reduce_scatter_ws_grad(ws_grad, mesh):
  """reduce-scatters weight gradients to FSDP sharding."""

  # Ensure grad RS/ARs aren't fused with ops that generated them.
  ws_grad = jax.lax.optimization_barrier(ws_grad)

  def fn(ws_grad):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = ws_grad
    # Reduce-scatter across FSDP axis.
    wq_a = jax.lax.psum_scatter(wq_a, axis_name="fsdp", tiled=True)
    wq_b = jax.lax.psum_scatter(wq_b, axis_name="fsdp", tiled=True)
    wkv_a = jax.lax.psum_scatter(wkv_a, axis_name="fsdp", tiled=True)
    wkv_b = jax.lax.psum_scatter(wkv_b, axis_name="fsdp", tiled=True)
    out = jax.lax.psum_scatter(out, axis_name="fsdp", tiled=True, scatter_dimension=2)
    gate = jax.lax.psum_scatter(gate, axis_name="fsdp", tiled=True)
    routed_wi_0 = jax.lax.psum_scatter(routed_wi_0, axis_name="fsdp", tiled=True)
    routed_wi_1 = jax.lax.psum_scatter(routed_wi_1, axis_name="fsdp", tiled=True)
    routed_wo = jax.lax.psum_scatter(routed_wo, axis_name="fsdp", tiled=True)
    shared_wi_0 = jax.lax.psum_scatter(shared_wi_0, axis_name="fsdp", tiled=True)
    shared_wi_1 = jax.lax.psum_scatter(shared_wi_1, axis_name="fsdp", tiled=True)
    shared_wo = jax.lax.psum_scatter(shared_wo, axis_name="fsdp", tiled=True, scatter_dimension=1)
    # All-reduce across expert axis.
    wq_a = jax.lax.psum(wq_a, axis_name="expert")
    wq_b = jax.lax.psum(wq_b, axis_name="expert")
    wkv_a = jax.lax.psum(wkv_a, axis_name="expert")
    wkv_b = jax.lax.psum(wkv_b, axis_name="expert")
    out = jax.lax.psum(out, axis_name="expert")
    gate = jax.lax.psum(gate, axis_name="expert")
    shared_wi_0 = jax.lax.psum(shared_wi_0, axis_name="expert")
    shared_wi_1 = jax.lax.psum(shared_wi_1, axis_name="expert")
    shared_wo = jax.lax.psum(shared_wo, axis_name="expert")
    return (
        (
            pre_attn_norm,
            (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
        ),
        (
            post_attn_norm,
            (gate, bias),
            (routed_wi_0, routed_wi_1, routed_wo),
            (shared_wi_0, shared_wi_1, shared_wo),
        ),
    )

  return jax.shard_map(
      fn,
      mesh=mesh,
      in_specs=(
          (
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, None, unreduced={"data", "fsdp", "expert"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, "expert", unreduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec(None, None, "expert", unreduced={"data", "fsdp"}),
                      jax.sharding.PartitionSpec(None, "expert", None, unreduced={"data", "fsdp"}),
                  ),
                  (
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                      jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  ),
              ),
          ),
      ),
      out_specs=(
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None, None, "fsdp", unreduced={"data"}),
              ),
          ),
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec("fsdp", None, "expert", unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, "expert", unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", "expert", None, unreduced={"data"}),
              ),
              (
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                  jax.sharding.PartitionSpec(None, "fsdp", unreduced={"data"}),
              ),
          ),
      ),
      check_vma=True,
  )(ws_grad)


def all_reduce_ws_grad_dcn(ws_grad, mesh):
  """all-reduces weight gradients across DCN axes."""

  # Ensure grad RS/ARs aren't fused with ops that generated them.
  ws_grad = jax.lax.optimization_barrier(ws_grad)

  def fn(ws_grad):
    (
        pre_attn_norm,
        (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
    ), (
        post_attn_norm,
        (gate, bias),
        (routed_wi_0, routed_wi_1, routed_wo),
        (shared_wi_0, shared_wi_1, shared_wo),
    ) = ws_grad
    # All-reduce across data axis.
    wq_a = jax.lax.psum(wq_a, axis_name="data")
    wq_b = jax.lax.psum(wq_b, axis_name="data")
    wkv_a = jax.lax.psum(wkv_a, axis_name="data")
    wkv_b = jax.lax.psum(wkv_b, axis_name="data")
    out = jax.lax.psum(out, axis_name="data")
    gate = jax.lax.psum(gate, axis_name="data")
    routed_wi_0 = jax.lax.psum(routed_wi_0, axis_name="data")
    routed_wi_1 = jax.lax.psum(routed_wi_1, axis_name="data")
    routed_wo = jax.lax.psum(routed_wo, axis_name="data")
    shared_wi_0 = jax.lax.psum(shared_wi_0, axis_name="data")
    shared_wi_1 = jax.lax.psum(shared_wi_1, axis_name="data")
    shared_wo = jax.lax.psum(shared_wo, axis_name="data")
    return (
        (
            pre_attn_norm,
            (wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out),
        ),
        (
            post_attn_norm,
            (gate, bias),
            (routed_wi_0, routed_wi_1, routed_wo),
            (shared_wi_0, shared_wi_1, shared_wo),
        ),
    )

  return jax.shard_map(
      fn,
      mesh=mesh,
      in_specs=(
          (
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                      jax.sharding.PartitionSpec(None, None, "fsdp", unreduced={"data"}),
                  ),
              ),
              (
                  jax.sharding.PartitionSpec(None),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, "expert", unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, "expert", unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", "expert", None, unreduced={"data"}),
                  ),
                  (
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec("fsdp", None, unreduced={"data"}),
                      jax.sharding.PartitionSpec(None, "fsdp", unreduced={"data"}),
                  ),
              ),
          ),
      ),
      out_specs=(
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None),
                  jax.sharding.PartitionSpec("fsdp", None, None),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec("fsdp", None),
                  jax.sharding.PartitionSpec("fsdp", None, None),
                  jax.sharding.PartitionSpec(None),
                  jax.sharding.PartitionSpec(None, None, "fsdp"),
              ),
          ),
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec("fsdp", None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec("fsdp", None, "expert"),
                  jax.sharding.PartitionSpec("fsdp", None, "expert"),
                  jax.sharding.PartitionSpec("fsdp", "expert", None),
              ),
              (
                  jax.sharding.PartitionSpec("fsdp", None),
                  jax.sharding.PartitionSpec("fsdp", None),
                  jax.sharding.PartitionSpec(None, "fsdp"),
              ),
          ),
      ),
      check_vma=True,
  )(ws_grad)


def init_splash_kernel(config):
  """Initializes the Splash kernel."""
  sa_config = attention.splash_attention_kernel.BlockSizes(
      block_q=min(config.sa_block_q, config.max_target_length),
      block_kv=min(config.sa_block_kv, config.max_target_length),
      block_kv_compute=min(config.sa_block_kv_compute, config.max_target_length),
      block_q_dkv=min(config.sa_block_q_dkv, config.max_target_length),
      block_kv_dkv=min(config.sa_block_kv_dkv, config.max_target_length),
      block_kv_dkv_compute=min(config.sa_block_kv_dkv_compute, config.max_target_length),
      block_q_dq=None if config.sa_use_fused_bwd_kernel else min(config.sa_block_q_dq, config.max_target_length),
      block_kv_dq=None if config.sa_use_fused_bwd_kernel else min(config.sa_block_kv_dq, config.max_target_length),
      use_fused_bwd_kernel=config.sa_use_fused_bwd_kernel,
      q_layout=attention.splash_attention_kernel.QKVLayout[config.sa_q_layout],
      k_layout=attention.splash_attention_kernel.QKVLayout[config.sa_k_layout],
      v_layout=attention.splash_attention_kernel.QKVLayout[config.sa_v_layout],
  )
  mask = splash_attention_mask.CausalMask(shape=(config.max_target_length, config.max_target_length))
  multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * config.num_query_heads)
  return attention.splash_attention_kernel.make_splash_mha(
      mask=multi_head_mask,
      head_shards=1,  # the size of the axis if sharding over heads
      q_seq_shards=1,  # axis for sequence sharding
      block_sizes=sa_config,
      attn_logits_soft_cap=None,
      residual_checkpoint_name="context",
      save_residuals=True,
  )


def tpu_flash_attention(
    query,
    key,
    value,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention."""
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))

  q_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  kv_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  lse_pspec = activation_pspec

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          q_pspec,
          kv_pspec,
          kv_pspec,
      ),
      out_specs=(
          q_pspec,
          lse_pspec,
      ),
      check_vma=False,
  )
  def wrap_flash_attention_manual(query, key, value):
    attention_output, logsumexp = jax.vmap(splash_kernel.manual_fwd, in_axes=(0, 0, 0, None, None), out_axes=(0, 0))(
        query,
        key,
        value,
        None,
        None,
    )
    return attention_output, logsumexp

  attention_output, logsumexp = wrap_flash_attention_manual(
      query,
      key,
      value,
  )
  return jnp.transpose(attention_output, axes=(0, 2, 1, 3)), logsumexp


def tpu_flash_attention_bwd(
    attention_out_grad,
    query,
    key,
    value,
    attention_output,
    logsumexp,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention backward."""
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))
  attention_output = jnp.transpose(attention_output, axes=(0, 2, 1, 3))
  attention_out_grad = jnp.transpose(attention_out_grad, axes=(0, 2, 1, 3))

  q_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  kv_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  lse_pspec = activation_pspec

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          (
              lse_pspec,  # logsumexp.
              q_pspec,
              kv_pspec,
              kv_pspec,
              q_pspec,  # attention_output.
          ),
          (
              q_pspec,
              lse_pspec,
          ),
      ),
      out_specs=(
          q_pspec,
          kv_pspec,
          kv_pspec,
      ),
      check_vma=False,
  )
  def wrap_flash_attention_manual_bwd(res, grad):
    logsumexp, query, key, value, attention_output = res
    attention_output_grad, _ = grad
    dq, dk, dv = jax.vmap(splash_kernel.manual_bwd, in_axes=(0, 0, 0, 0, 0, 0, None, None), out_axes=(0, 0, 0))(
        query,
        key,
        value,
        attention_output,
        logsumexp,
        attention_output_grad,
        None,
        None,
    )
    return dq, dk, dv

  dq, dk, dv = wrap_flash_attention_manual_bwd(
      (logsumexp, query, key, value, attention_output), (attention_out_grad, None)
  )
  dq = jnp.transpose(dq, axes=(0, 2, 1, 3))
  dk = jnp.transpose(dk, axes=(0, 2, 1, 3))
  dv = jnp.transpose(dv, axes=(0, 2, 1, 3))
  return dq, dk, dv


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    *,
    mesh,
    cfg,
    num_layers,
):
  """Scans the layers with batch-split schedule."""
  all_weights = fetch_weights(params, cfg.dtype)
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "expert"),
      None,
      None,
  )
  # The data mesh axis can be size 1, but we still want to keep it in the
  # partition spec because the code supports multi-slice runs and thus
  # expects the data axis to be present.
  inputs = jax.reshard(inputs, jax.sharding.NamedSharding(mesh, activation_pspec))
  yarn_freqs = initialize_yarn_freqs(
      positions=positions,
      embedding_dims=cfg.qk_rope_head_dim,
      rope_theta=cfg.rope_max_timescale,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      rope_factor=cfg.rope_factor,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  yarn_mask = initialize_yarn_mask(cfg.qk_rope_head_dim)
  splash_kernel = init_splash_kernel(cfg)

  @jax.custom_vjp
  def process_all_layers(inputs, all_weights, yarn_freqs):
    return process_all_layers_fwd(inputs, all_weights, yarn_freqs)[0]

  def process_all_layers_fwd(inputs, all_weights, yarn_freqs):
    def process_layer_scannable(carry, layer_idx, group_id):
      inputs, ws = carry
      # Prefetch weights for next layer.
      with scheduling_group(group_id=group_id):
        next_ws = gather_weights(extract_layer_weights(all_weights, layer_idx + 1, cfg.param_scan_axis), mesh)
      # Combine for previous layer's second microbatch.
      moe_inputs, routed_expert_out, shared_expert_out, selected_experts = inputs[1]
      inputs[1], unroute_res = unroute_ubatch_shard_mapped(
          moe_inputs,
          routed_expert_out,
          shared_expert_out,
          selected_experts,
          expert_axis_name="expert",
          use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
          target_length=cfg.max_target_length,
          mesh=mesh,
          activation_pspec=activation_pspec,
      )
      # Current layer computation.
      outputs, res = batch_split_schedule(
          inputs,
          ws,
          yarn_freqs,
          mesh=mesh,
          cfg=cfg,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
          pairwise_swap_and_negate_mask=yarn_mask,
      )
      # Offload to host memory.
      for residual_name in ("mlpwi_0", "mlpwi_1"):
        r = res.pop(residual_name)
        r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host")), r)
        res[residual_name] = r
      return (outputs, next_ws), (res, unroute_res)

    # Prologue: do first two layers and prefetch weights for third layer.
    with scheduling_group(group_id=40):
      first_ws = gather_weights(extract_layer_weights(all_weights, 0, cfg.param_scan_axis), mesh)
    with scheduling_group(group_id=41):
      second_ws = gather_weights(extract_layer_weights(all_weights, 1, cfg.param_scan_axis), mesh)
    second_inputs, first_res = batch_split_schedule(
        inputs,
        first_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    # Offload first layer residuals to host memory.
    for residual_name in ("mlpwi_0", "mlpwi_1"):
      r = first_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host")), r)
      first_res[residual_name] = r
    third_carry, (second_res, second_unroute_res) = process_layer_scannable((second_inputs, second_ws), 1, group_id=42)
    # Scan middle layers.
    last_last_carry, (middle_res, middle_unroute_res) = jax.lax.scan(
        functools.partial(process_layer_scannable, group_id=43),
        third_carry,
        jnp.arange(2, num_layers - 2),
    )
    # Epilogue: do last two layers without prefetching weights and finish second microbatch.
    (last_inputs, last_ws), (last_last_res, last_last_unroute_res) = process_layer_scannable(
        last_last_carry, num_layers - 2, group_id=44
    )
    moe_inputs, routed_expert_out, shared_expert_out, selected_experts = last_inputs[1]
    last_inputs[1], last_unroute_res = unroute_ubatch_shard_mapped(
        moe_inputs,
        routed_expert_out,
        shared_expert_out,
        selected_experts,
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        target_length=cfg.max_target_length,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    outputs, last_res = batch_split_schedule(
        last_inputs,
        last_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
    outputs[1], epilogue_unroute_res = unroute_ubatch_shard_mapped(
        moe_inputs,
        routed_expert_out,
        shared_expert_out,
        selected_experts,
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        target_length=cfg.max_target_length,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    return outputs, (
        first_res,
        second_res,
        middle_res,
        last_last_res,
        last_res,
        second_unroute_res,
        middle_unroute_res,
        last_last_unroute_res,
        last_unroute_res,
        epilogue_unroute_res,
        last_ws,
        all_weights,
        yarn_freqs,
    )

  def process_all_layers_bwd(res, g):
    (
        first_res,
        second_res,
        middle_res,
        last_last_res,
        last_res,
        second_unroute_res,
        middle_unroute_res,
        last_last_unroute_res,
        last_unroute_res,
        epilogue_unroute_res,
        last_ws,
        all_weights,
        yarn_freqs,
    ) = res
    initial_ws_grad = jax.tree.map(jnp.zeros_like, all_weights)

    def process_layer_bwd_scannable(carry, res_and_layer_idx, group_id):
      g, ws, next_next_ws_grad, next_ws_grad, all_layer_ws_grad = carry
      res, unroute_res, layer_idx = res_and_layer_idx
      # Prefetch weights and post-scatter weight grads.
      with scheduling_group(group_id=group_id):
        prev_ws = gather_weights(extract_layer_weights(all_weights, layer_idx - 1, cfg.param_scan_axis), mesh)
        next_ws_grad = reduce_scatter_ws_grad(next_ws_grad, mesh)
      next_next_ws_grad = all_reduce_ws_grad_dcn(next_next_ws_grad, mesh)
      all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, next_next_ws_grad, layer_idx + 2, cfg.param_scan_axis)
      # Get residuals from host.
      for residual_name in ("mlpwi_0", "mlpwi_1"):
        r = res.pop(residual_name)
        r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
        res[residual_name] = r

      # Current layer computation.
      g, ws_grad = batch_split_schedule_bwd(
          res,
          g,
          ws,
          yarn_freqs,
          mesh=mesh,
          cfg=cfg,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
          pairwise_swap_and_negate_mask=yarn_mask,
      )
      g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
          unroute_res["selected_experts"],
          g[1],
          expert_axis_name="expert",
          use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
          mesh=mesh,
          activation_pspec=activation_pspec,
      )
      return (g, prev_ws, next_ws_grad, ws_grad, all_layer_ws_grad), None

    # Prologue: do computation for last two layers and prefetch weights.
    with scheduling_group(group_id=50):
      prev_ws = gather_weights(extract_layer_weights(all_weights, num_layers - 2, cfg.param_scan_axis), mesh)
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        epilogue_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    g, ws_grad = batch_split_schedule_bwd(
        last_res,
        g,
        last_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        last_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    with scheduling_group(group_id=51):
      prev_prev_ws = gather_weights(extract_layer_weights(all_weights, num_layers - 3, cfg.param_scan_axis), mesh)
      ws_grad = reduce_scatter_ws_grad(ws_grad, mesh)
    # Get residuals from host.
    for residual_name in ("mlpwi_0", "mlpwi_1"):
      r = last_last_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
      last_last_res[residual_name] = r
    g, prev_ws_grad = batch_split_schedule_bwd(
        last_last_res,
        g,
        prev_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    g[1] = unroute_ubatch_remat_and_bwd_shard_mapped(
        last_last_unroute_res["selected_experts"],
        g[1],
        expert_axis_name="expert",
        use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    # Scan middle layers.
    (g, second_ws, fourth_ws_grad, third_ws_grad, all_layer_ws_grad), _ = jax.lax.scan(
        functools.partial(process_layer_bwd_scannable, group_id=52),
        (g, prev_prev_ws, ws_grad, prev_ws_grad, initial_ws_grad),
        (middle_res, middle_unroute_res, jnp.arange(2, num_layers - 2)),
        reverse=True,
    )
    # Epilogue: do third and second layer computation and post-scatter weight grads.
    (g, first_ws, third_ws_grad, second_ws_grad, all_layer_ws_grad), _ = process_layer_bwd_scannable(
        (g, second_ws, fourth_ws_grad, third_ws_grad, all_layer_ws_grad),
        (second_res, second_unroute_res, 1),
        group_id=53,
    )
    with scheduling_group(group_id=54):
      second_ws_grad = reduce_scatter_ws_grad(second_ws_grad, mesh)
    third_ws_grad = all_reduce_ws_grad_dcn(third_ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, third_ws_grad, 2, cfg.param_scan_axis)
    # Get residuals from host.
    for residual_name in ("mlpwi_0", "mlpwi_1"):
      r = first_res.pop(residual_name)
      r = jax.tree.map(lambda x: jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device")), r)
      first_res[residual_name] = r
    g, ws_grad = batch_split_schedule_bwd(
        first_res,
        g,
        first_ws,
        yarn_freqs,
        mesh=mesh,
        cfg=cfg,
        splash_kernel=splash_kernel,
        activation_pspec=activation_pspec,
        pairwise_swap_and_negate_mask=yarn_mask,
    )
    second_ws_grad = all_reduce_ws_grad_dcn(second_ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, second_ws_grad, 1, cfg.param_scan_axis)
    with scheduling_group(group_id=55):
      ws_grad = reduce_scatter_ws_grad(ws_grad, mesh)
    ws_grad = all_reduce_ws_grad_dcn(ws_grad, mesh)
    all_layer_ws_grad = insert_layer_ws_grad(all_layer_ws_grad, ws_grad, 0, cfg.param_scan_axis)
    return g, all_layer_ws_grad, None

  process_all_layers.defvjp(process_all_layers_fwd, process_all_layers_bwd)

  inputs = jax.shard_map(
      functools.partial(split, split_factor=cfg.batch_split_factor),
      mesh=mesh,
      in_specs=activation_pspec,
      out_specs=[activation_pspec] * cfg.batch_split_factor,
  )(inputs)
  yarn_freqs = split(yarn_freqs, split_factor=cfg.batch_split_factor)
  outputs = process_all_layers(inputs, all_weights, yarn_freqs)
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
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  norm_mla_ws, moe_ws = weights
  xs, attn_res = mla_with_norms(
      inputs,
      norm_mla_ws,
      positions,
      mesh=mesh,
      splash_kernel=splash_kernel,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  xs, moe_res = moe(
      xs,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      expert_axis_name="expert",
      use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
      config=cfg,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  return xs, {"layer_inputs": inputs, **attn_res, **moe_res}


def batch_split_schedule_bwd(
    residuals,
    outputs_grad,
    weights,
    positions,
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Performs the backward pass for a single layer."""
  norm_mla_ws, moe_ws = weights
  mla_out, mla_bwds = mla_with_norms_remat(
      residuals,
      norm_mla_ws,
      positions,
      mesh=mesh,
      splash_kernel=splash_kernel,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  residuals["mla_out"] = mla_out
  attn_out_grad, moe_ws_grad = moe_bwd(
      residuals,
      outputs_grad,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      expert_axis_name="expert",
      use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
      config=cfg,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  inputs_grad, norm_mla_ws_grad = mla_with_norms_bwd(attn_out_grad, mla_bwds)
  return inputs_grad, (norm_mla_ws_grad, moe_ws_grad)


def staggered_call(fn, xs):
  """Calls a function in a staggered manner while accumulating residuals."""
  res_dicts = []
  for i, x in enumerate(xs):
    x, res = fn(x)
    res_dicts.append(res)
    if i == len(xs) - 1:
      xs[i] = x
    else:
      xs[i], xs[i + 1] = jax.lax.optimization_barrier((x, xs[i + 1]))
  # Convert list of res dicts to dict of lists.
  return xs, jax.tree_util.tree_map(lambda *rs: list(rs), *res_dicts)


def dot(x, y, axes=1):
  return jnp.tensordot(x, y, axes=axes)


def mla_with_norms(
    inputs,
    weights,
    yarn_freqs,
    *,
    mesh,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs MLA with pre-normalization."""
  pre_attn_scale, attn_ws = weights

  def fn(args):
    x, yarn_freqs = args
    y = rms_norm(
        x,
        pre_attn_scale,
        epsilon=normalization_layer_epsilon,
        dtype=dtype,
        out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
    )
    mla_out, mla_res = mla(
        y,
        yarn_freqs,
        attn_ws,
        epsilon=normalization_layer_epsilon,
        kv_lora_rank=kv_lora_rank,
        kv_norm_epsilon=normalization_layer_epsilon,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
        dtype=dtype,
        mscale=mscale,
        splash_kernel=splash_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    # Prevent fusion with MoE ops, especially the RMS norm.
    # Unfortunately, this seems to be needed to avoid slight numerical differences
    # between the fwd pass and remat.
    return jax.lax.optimization_barrier(mla_out + x), mla_res

  return staggered_call(fn, list(zip(inputs, yarn_freqs)))


def mla_with_norms_remat(
    residuals,
    weights,
    yarn_freqs,
    *,
    mesh,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs remat for the mla_with_norms function."""
  xs = residuals.pop("layer_inputs")
  pre_attn_scale, attn_ws = weights

  def remat_fn(args):
    x, yarn_freqs, attn_out, lse = args
    y, pre_attn_rms_norm_bwd = jax.vjp(
        functools.partial(
            rms_norm,
            epsilon=normalization_layer_epsilon,
            dtype=dtype,
            out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
        ),
        x,
        pre_attn_scale,
    )
    mla_out, mla_bwds = mla_remat(
        (y, attn_out, lse),
        yarn_freqs,
        attn_ws,
        epsilon=normalization_layer_epsilon,
        kv_lora_rank=kv_lora_rank,
        kv_norm_epsilon=normalization_layer_epsilon,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
        dtype=dtype,
        mscale=mscale,
        splash_kernel=splash_kernel,
        mesh=mesh,
        activation_pspec=activation_pspec,
    )
    out = x + mla_out
    # Prevent fusion with MoE ops, especially the RMS norm.
    # Unfortunately, this seems to be needed to avoid slight numerical differences
    # between the fwd pass and remat.
    return jax.lax.optimization_barrier(out), (pre_attn_rms_norm_bwd, mla_bwds)

  bwds = [None] * len(xs)
  for i, x in enumerate(zip(xs, yarn_freqs, residuals.pop("attn_out"), residuals.pop("lse"))):
    xs[i], bwds[i] = remat_fn(x)
  return xs, bwds


def mla_with_norms_bwd(
    outputs_grad,
    bwds,
):
  """Performs the backward pass for the mla_with_norms function."""

  def bwd_fn(args):
    output_grad, (pre_attn_rms_norm_bwd, mla_bwds) = args
    x_grad, mla_out_grad = output_grad, output_grad
    y_grad, attn_ws_grad = mla_bwd(mla_out_grad, mla_bwds)
    x_grad_partial, pre_attn_scale_grad = pre_attn_rms_norm_bwd(y_grad)
    return x_grad + x_grad_partial, (pre_attn_scale_grad, attn_ws_grad)

  norm_mla_ws_grad = []
  for i, g in enumerate(outputs_grad):
    outputs_grad[i], ws_grad = bwd_fn((g, bwds[i]))
    norm_mla_ws_grad.append(ws_grad)
  (pre_attn_scale_grad, attn_ws_grad) = sum_grads(norm_mla_ws_grad)
  return outputs_grad, (pre_attn_scale_grad, attn_ws_grad)


def mla(
    inputs,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
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
      yarn_freqs,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  key, value = kv_projection(
      inputs,
      yarn_freqs,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  attn_out, lse = tpu_flash_attention(
      query,
      key,
      value,
      mesh=mesh,
      splash_kernel=splash_kernel,
      activation_pspec=activation_pspec,
  )
  out = dot(attn_out, out_weights, axes=2)
  return out, {"attn_out": attn_out, "lse": lse}


def mla_remat(
    residuals,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
):
  """Performs remat for the mla function."""
  inputs, attn_out, lse = residuals
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query, query_projection_bwd = jax.vjp(
      functools.partial(
          query_projection,
          epsilon=epsilon,
          qk_rope_head_dim=qk_rope_head_dim,
          max_position_embeddings=max_position_embeddings,
          original_max_position_embeddings=original_max_position_embeddings,
          rope_factor=rope_factor,
          pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
          dtype=dtype,
          qk_nope_head_dim=qk_nope_head_dim,
          mscale=mscale,
          mesh=mesh,
          activation_pspec=activation_pspec,
      ),
      inputs,
      yarn_freqs,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
  )
  (key, value), kv_projection_bwd = jax.vjp(
      functools.partial(
          kv_projection,
          kv_lora_rank=kv_lora_rank,
          kv_norm_epsilon=kv_norm_epsilon,
          pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
          dtype=dtype,
          qk_nope_head_dim=qk_nope_head_dim,
          num_query_heads=num_query_heads,
          mesh=mesh,
          activation_pspec=activation_pspec,
      ),
      inputs,
      yarn_freqs,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
  )
  out, out_projection_bwd = jax.vjp(
      functools.partial(
          dot,
          axes=2,
      ),
      attn_out,
      out_weights,
  )
  return out, (
      query_projection_bwd,
      kv_projection_bwd,
      functools.partial(
          tpu_flash_attention_bwd,
          query=query,
          key=key,
          value=value,
          attention_output=attn_out,
          logsumexp=lse,
          mesh=mesh,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
      ),
      out_projection_bwd,
  )


def mla_bwd(
    out_grad,
    bwds,
):
  """Performs the backward pass for the mla function."""
  query_projection_bwd, kv_projection_bwd, attn_op_bwd, out_projection_bwd = bwds
  attn_out_grad, out_weights_grad = out_projection_bwd(out_grad)
  # query_grad, key_grad, value_grad, _ = attention_op_bwd(attn_out_grad)
  query_grad, key_grad, value_grad = attn_op_bwd(attn_out_grad)
  inputs_grad_from_kv, _, wkv_a_weights_grad, wkv_b_weights_grad, kv_norm_scale_weights_grad = kv_projection_bwd(
      (key_grad, value_grad)
  )
  inputs_grad_from_q, _, wq_a_weights_grad, wq_b_weights_grad, q_norm_scale_weights_grad = query_projection_bwd(
      query_grad
  )
  return inputs_grad_from_kv + inputs_grad_from_q, (
      wq_a_weights_grad,
      wq_b_weights_grad,
      q_norm_scale_weights_grad,
      wkv_a_weights_grad,
      wkv_b_weights_grad,
      kv_norm_scale_weights_grad,
      out_weights_grad,
  )


def query_projection(
    inputs_q,
    yarn_freqs,
    wq_a_weights,
    wq_b_weights,
    q_norm_scale_weights,
    *,
    epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    pairwise_swap_and_negate_mask,
    dtype,
    mscale,
    mesh,
    activation_pspec,
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
      out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
  )
  q = dot(low_rank_q, wq_b_weights)

  # Split into non-positional and rotary parts.
  q_nope, q_pe = jnp.split(q, [qk_nope_head_dim], axis=-1)
  q_pe = yarn(
      q_pe,
      yarn_freqs,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      fprop_dtype=dtype,
  )
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale
  return query


def kv_projection(
    inputs,
    yarn_freqs,
    wkv_a_weights,
    wkv_b_weights,
    kv_norm_scale_weights,
    *,
    kv_lora_rank,
    kv_norm_epsilon,
    pairwise_swap_and_negate_mask,
    dtype,
    qk_nope_head_dim,
    num_query_heads,
    mesh,
    activation_pspec,
):
  """Performs KV projection."""
  low_rank = dot(inputs, wkv_a_weights)
  low_rank_main, low_rank_rope = jnp.split(low_rank, [kv_lora_rank], axis=-1)
  low_rank_main = rms_norm(
      low_rank_main,
      kv_norm_scale_weights,
      epsilon=kv_norm_epsilon,
      dtype=dtype,
      out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
  )
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = yarn(
      key_rope,
      yarn_freqs,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
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


def rms_norm(x, scale, *, epsilon, dtype, out_sharding=None):
  """RMS normalization."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
  return jnp.einsum("i...k,...k->i...k", y, scale, out_sharding=out_sharding)


def initialize_yarn_mask(embedding_dims):
  """Initializes YaRN mask."""
  indices = jnp.arange(embedding_dims)
  # [1, 0, 3, 2, 5, 4, ...]
  swap_indices = jnp.where(indices % 2 == 0, indices + 1, indices - 1)
  negation_mask = jnp.where(indices % 2 == 0, -1, 1)
  identity = jnp.eye(embedding_dims, dtype=jnp.int32)
  return identity[swap_indices] * negation_mask


def initialize_yarn_freqs(
    positions,
    embedding_dims,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mesh,
    activation_pspec,
):
  """Initializes YaRN frequencies."""
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
  freqs = freqs.at[positions].get(out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec))
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]
  return jnp.repeat(freqs, 2, axis=-1)  # shape: [B, S, 1, embedding_dims]


def yarn(
    inputs,
    freqs,
    *,
    pairwise_swap_and_negate_mask,
    fprop_dtype,
):
  """Performs YaRN rotary embedding."""
  # inputs @ mask: [B, S, N, embedding_dims] @ [embedding_dims, embedding_dims] -> [B, S, N, embedding_dims]
  output = inputs * jnp.cos(freqs) + jnp.matmul(inputs, pairwise_swap_and_negate_mask) * jnp.sin(freqs)
  return output.astype(fprop_dtype)


def shared_expert_and_route(
    inputs,
    post_attn_scale,
    shared_w0,
    shared_w1,
    shared_wo,
    gate_kernel,
    gate_bias,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
):
  """Computes the shared expert and routes the activations."""
  inputs = rms_norm(
      inputs,
      post_attn_scale,
      epsilon=normalization_layer_epsilon,
      dtype=dtype,
  )
  y = shared_expert(inputs, shared_w0, shared_w1, shared_wo)

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


def shared_expert(inputs, shared_w0, shared_w1, shared_wo):
  return dot(jax.nn.silu(dot(inputs, shared_w0)) * dot(inputs, shared_w1), shared_wo)


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
  weights = jax.lax.all_gather(weights, axis_name=expert_axis_name, tiled=True)
  selected_experts = jax.lax.all_gather(selected_experts, axis_name=expert_axis_name, tiled=True)
  weights = jnp.ravel(weights)[jnp.argsort(jnp.ravel(selected_experts))]
  group_sizes = jax.lax.psum(group_sizes, axis_name=expert_axis_name)

  x = route_impl(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
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
  return unroute_impl(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def route_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  return route_impl_fwd(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )[0]


def route_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Routes the activations and all-gathers across the expert axis."""
  x, selected_experts = jax.lax.optimization_barrier((x, selected_experts))
  x = jax.lax.all_gather(x, axis_name=expert_axis_name, tiled=True)
  x = sort_activations.route(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )
  x = jax.lax.optimization_barrier(x)
  return x, selected_experts


def route_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad):
  selected_experts = res
  return (
      unroute_impl_fwd(
          grad, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
      )[0],
      None,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def unroute_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  return unroute_impl_fwd(
      x, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
  )[0]


def unroute_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Unroutes the activations and reduce-scatters across the expert axis."""
  x, selected_experts = jax.lax.optimization_barrier((x, selected_experts))
  x = sort_activations.unroute(
      x,
      selected_experts,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )

  # Sum across expert shards.
  x = jax.lax.psum_scatter(x, expert_axis_name, scatter_dimension=0, tiled=True)
  x = jax.lax.optimization_barrier(x)
  return x, selected_experts


def unroute_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad):
  selected_experts = res
  return (
      route_impl_fwd(
          grad, selected_experts, expert_axis_name=expert_axis_name, use_gather_mosaic_kernel=use_gather_mosaic_kernel
      )[0],
      None,
  )


route_impl.defvjp(route_impl_fwd, route_impl_bwd)
unroute_impl.defvjp(unroute_impl_fwd, unroute_impl_bwd)


def compute_gating(x, w0, w1, group_sizes, *, dtype):
  """Computes the gating GMMs."""
  layer_w0 = jax.lax.ragged_dot(
      x,
      w0,
      group_sizes=group_sizes,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=dtype,
  )
  layer_w1 = jax.lax.ragged_dot(
      x,
      w1,
      group_sizes=group_sizes,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=dtype,
  )
  return layer_w0, layer_w1


def compute_linear(layer_w0, layer_w1, wo, group_sizes, weights, *, dtype):
  """Combines the outputs of the gating GMMs and computes the final GMM."""
  intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
  intermediate_layer *= weights[:, None]
  layer_wo = jax.lax.ragged_dot(
      intermediate_layer,
      wo,
      group_sizes=group_sizes,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=dtype,
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
    normalization_layer_epsilon,
    dtype,
):
  """Routes, processes, and unroutes activations."""
  target_length = xs[0].shape[1]
  (
      post_attn_scale,
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn(inputs):
    x, y, selected_experts, weights, group_sizes = shared_expert_and_route(
        inputs,
        post_attn_scale,
        shared_w0,
        shared_w1,
        shared_wo,
        gate_kernel,
        gate_bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
        normalization_layer_epsilon=normalization_layer_epsilon,
        dtype=dtype,
    )
    return (inputs, x, y, selected_experts, weights, group_sizes), {}

  def compute_fn(inputs):
    moe_inputs, x, y, selected_experts, weights, group_sizes = inputs
    layer_w0, layer_w1 = compute_gating_fn((x, group_sizes))
    x = compute_linear_fn((layer_w0, layer_w1, weights, group_sizes))
    return (moe_inputs, x, y, selected_experts), {"mlpwi_0": layer_w0, "mlpwi_1": layer_w1}

  def compute_gating_fn(inputs):
    x, group_sizes = inputs
    layer_w0, layer_w1 = compute_gating(
        x,
        routed_w0,
        routed_w1,
        group_sizes,
        dtype=dtype,
    )
    return layer_w0, layer_w1

  def compute_linear_fn(inputs):
    layer_w0, layer_w1, weights, group_sizes = inputs
    x = compute_linear(
        layer_w0,
        layer_w1,
        routed_wo,
        group_sizes,
        weights,
        dtype=dtype,
    )
    return x

  def unroute_fn(inputs):
    moe_inputs, x, y, selected_experts = inputs
    return unroute_ubatch_fn(
        moe_inputs,
        x,
        y,
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
        target_length=target_length,
    )

  xs, _ = staggered_call(route_fn, xs)
  xs, res = staggered_call(compute_fn, xs)
  # We don't need the residuals from unroute for the first microbatch since they are calculated earlier in the layer.
  xs[0], _ = unroute_fn(xs[0])
  return xs, res


def unroute_ubatch_shard_mapped(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
    mesh,
    activation_pspec,
):
  """Performs the unroute operation for a single microbatch in a shard map."""
  return jax.shard_map(
      functools.partial(
          unroute_ubatch_fn,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          target_length=target_length,
      ),
      mesh=mesh,
      in_specs=(
          activation_pspec,
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(activation_pspec[0], None),
      ),
      out_specs=(
          activation_pspec,
          {
              "selected_experts": jax.sharding.PartitionSpec(activation_pspec[0], None),
          },
      ),
      check_vma=True,
  )(
      moe_inputs,
      routed_expert_out,
      shared_expert_out,
      selected_experts,
  )


def unroute_ubatch_fn(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
):
  """Performs the unroute operation for a single microbatch."""
  routed_expert_out = unroute(
      routed_expert_out,
      selected_experts,
      expert_axis_name=expert_axis_name,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )
  expert_out = jnp.reshape(routed_expert_out, (-1, target_length, routed_expert_out.shape[-1])) + shared_expert_out
  return moe_inputs + expert_out, {"selected_experts": selected_experts}


def unroute_ubatch_remat_and_bwd_shard_mapped(
    selected_experts,
    outputs_grad,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    mesh,
    activation_pspec,
):
  """Performs remat and backward pass for unroute_ubatch in a shard map."""

  def unroute_ubatch_remat_and_bwd_fn(
      selected_experts,
      outputs_grad,
      *,
      expert_axis_name,
      use_gather_mosaic_kernel,
  ):
    unroute_bwd = unroute_ubatch_fn_remat(
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )
    return unroute_ubatch_fn_bwd(outputs_grad, unroute_bwd)

  return jax.shard_map(
      functools.partial(
          unroute_ubatch_remat_and_bwd_fn,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
      ),
      mesh=mesh,
      in_specs=(
          jax.sharding.PartitionSpec(activation_pspec[0], None),
          activation_pspec,
      ),
      out_specs=(
          activation_pspec,
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(*activation_pspec[:-1]),
          jax.sharding.PartitionSpec(activation_pspec[0], None),
      ),
      check_vma=True,
  )(
      selected_experts,
      outputs_grad,
  )


def unroute_ubatch_fn_remat(
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  # Since we never need the outputs of unroute in the backward pass, we can just the backward function of unroute directly.
  return functools.partial(
      route_impl_fwd,
      selected_experts=selected_experts,
      expert_axis_name=expert_axis_name,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )


def unroute_ubatch_fn_bwd(
    outputs_grad,
    unroute_bwd,
):
  moe_inputs_grad, expert_out_grad = outputs_grad, outputs_grad
  routed_expert_out_grad, shared_expert_out_grad = expert_out_grad, expert_out_grad
  routed_expert_out_grad = jnp.reshape(routed_expert_out_grad, (-1, routed_expert_out_grad.shape[-1]))
  routed_expert_out_grad, selected_experts_grad = unroute_bwd(routed_expert_out_grad)
  return moe_inputs_grad, routed_expert_out_grad, shared_expert_out_grad, selected_experts_grad


def sum_grads(grads):
  return functools.reduce(lambda x, y: jax.tree_util.tree_map(jnp.add, x, y), grads)


def route_compute_unroute_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
):
  """Performs the backward pass for route_compute_unroute."""
  xs = residuals.pop("mla_out")
  (
      post_attn_scale,
      (gate_kernel, gate_bias),
      (routed_w0, routed_w1, routed_wo),
      (shared_w0, shared_w1, shared_wo),
  ) = weights

  def route_fn_remat(inputs):
    (x, y, selected_experts, weights, group_sizes), shared_expert_and_route_bwd = jax.vjp(
        functools.partial(
            shared_expert_and_route,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            expert_axis_name=expert_axis_name,
            use_gather_mosaic_kernel=use_gather_mosaic_kernel,
            normalization_layer_epsilon=normalization_layer_epsilon,
            dtype=dtype,
        ),
        inputs,
        post_attn_scale,
        shared_w0,
        shared_w1,
        shared_wo,
        gate_kernel,
        gate_bias,
    )
    return (inputs, x, y, selected_experts, weights, group_sizes), shared_expert_and_route_bwd

  def route_fn_bwd(inputs):
    (
        inputs_grad1,
        x_grad,
        y_grad,
        selected_experts_grad,
        weights_grad,
        group_sizes_grad,
    ), shared_expert_and_route_bwd = inputs
    (
        inputs_grad2,
        post_attn_scale_grad,
        shared_w0_grad,
        shared_w1_grad,
        shared_wo_grad,
        gate_kernel_grad,
        gate_bias_grad,
    ) = shared_expert_and_route_bwd((x_grad, y_grad, selected_experts_grad, weights_grad, group_sizes_grad))
    return inputs_grad1 + inputs_grad2, (
        post_attn_scale_grad,
        shared_w0_grad,
        shared_w1_grad,
        shared_wo_grad,
        gate_kernel_grad,
        gate_bias_grad,
    )

  def compute_fn_remat(inputs):
    (moe_inputs, x, y, selected_experts, weights, group_sizes), layer_w0, layer_w1 = inputs
    _, compute_gating_bwd = compute_gating_fn_remat((x, group_sizes))
    x, compute_linear_bwd = compute_linear_fn_remat((layer_w0, layer_w1, weights, group_sizes))
    return (moe_inputs, x, y, selected_experts), (compute_gating_bwd, compute_linear_bwd)

  def compute_fn_bwd(inputs):
    (moe_inputs_grad, x_grad, y_grad, selected_experts_grad), (compute_gating_bwd, compute_linear_bwd) = inputs
    (layer_w0_grad, layer_w1_grad, routed_wo_grad, _, weights_grad) = compute_linear_bwd(x_grad)
    (x_grad, routed_w0_grad, routed_w1_grad, group_sizes_grad) = compute_gating_bwd((layer_w0_grad, layer_w1_grad))
    return (moe_inputs_grad, x_grad, y_grad, selected_experts_grad, weights_grad, group_sizes_grad), (
        routed_w0_grad,
        routed_w1_grad,
        routed_wo_grad,
    )

  def compute_gating_fn_remat(inputs):
    x, group_sizes = inputs
    (layer_w0, layer_w1), compute_gating_bwd = jax.vjp(
        functools.partial(
            compute_gating,
            dtype=dtype,
        ),
        x,
        routed_w0,
        routed_w1,
        group_sizes,
    )
    return (layer_w0, layer_w1), compute_gating_bwd

  def compute_linear_fn_remat(inputs):
    layer_w0, layer_w1, weights, group_sizes = inputs
    x, compute_linear_bwd = jax.vjp(
        functools.partial(
            compute_linear,
            dtype=dtype,
        ),
        layer_w0,
        layer_w1,
        routed_wo,
        group_sizes,
        weights,
    )
    return x, compute_linear_bwd

  def unroute_fn_remat(inputs):
    _, _, _, selected_experts = inputs
    return unroute_ubatch_fn_remat(
        selected_experts,
        expert_axis_name=expert_axis_name,
        use_gather_mosaic_kernel=use_gather_mosaic_kernel,
    )

  def unroute_fn_bwd(inputs):
    outputs_grad, bwds = inputs
    return unroute_ubatch_fn_bwd(outputs_grad, bwds)

  route_bwds = [None] * len(xs)
  for i, x in enumerate(xs):
    xs[i], route_bwds[i] = route_fn_remat(x)
  compute_bwds = [None] * len(xs)
  for i, x in enumerate(zip(xs, residuals.pop("mlpwi_0"), residuals.pop("mlpwi_1"))):
    xs[i], compute_bwds[i] = compute_fn_remat(x)
  unroute_bwd = unroute_fn_remat(xs[0])
  outputs_grad[0] = unroute_fn_bwd((outputs_grad[0], unroute_bwd))
  compute_ws_grad = []
  for i, g in enumerate(outputs_grad):
    g, ws_grad = compute_fn_bwd((g, compute_bwds[i]))
    outputs_grad[i] = g
    compute_ws_grad.append(ws_grad)
  route_ws_grad = []
  for i, g in enumerate(outputs_grad):
    g, ws_grad = route_fn_bwd((g, route_bwds[i]))
    outputs_grad[i] = g
    route_ws_grad.append(ws_grad)
  (routed_w0_grad, routed_w1_grad, routed_wo_grad) = sum_grads(compute_ws_grad)
  (post_attn_scale_grad, shared_w0_grad, shared_w1_grad, shared_wo_grad, gate_kernel_grad, gate_bias_grad) = sum_grads(
      route_ws_grad
  )
  return outputs_grad, (
      post_attn_scale_grad,
      (gate_kernel_grad, gate_bias_grad),
      (routed_w0_grad, routed_w1_grad, routed_wo_grad),
      (shared_w0_grad, shared_w1_grad, shared_wo_grad),
  )


def moe(
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
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          normalization_layer_epsilon=normalization_layer_epsilon,
          dtype=dtype,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * config.batch_split_factor,
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, expert_axis_name, None, reduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      out_specs=(
          [
              activation_pspec,
              (
                  activation_pspec,
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(activation_pspec[0], None),
              ),
          ],
          {
              "mlpwi_0": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
              "mlpwi_1": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
          },
      ),
      check_vma=True,
  )([x.astype(config.dtype) for x in xs], weights)


def moe_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs the backward pass for the moe function."""
  return jax.shard_map(
      functools.partial(
          route_compute_unroute_bwd,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          normalization_layer_epsilon=normalization_layer_epsilon,
          dtype=dtype,
      ),
      mesh=mesh,
      in_specs=(
          {
              "mla_out": [activation_pspec] * config.batch_split_factor,
              "mlpwi_0": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
              "mlpwi_1": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
          },
          [
              activation_pspec,
              (
                  activation_pspec,
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(activation_pspec[0], None),
              ),
          ],
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, expert_axis_name, None, reduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      out_specs=(
          [activation_pspec] * config.batch_split_factor,
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, unreduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, unreduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, expert_axis_name, None, unreduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, unreduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      check_vma=True,
  )(residuals, outputs_grad, weights)
