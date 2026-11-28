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

"""Biasless SwiGLU MLP block for Flint dense prefix and shared MoE experts."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops


def flint_mlp_forward(
    x_local: jax.Array,
    weights_shard: flint_types.FlintMLPWeights[jax.Array],
    cfg: flint_config.FlintConfig,
) -> jax.Array:
  """Computes biasless SwiGLU `(silu(x @ gate) * (x @ up)) @ down` locally.

  Executes inside `jax.shard_map(..., check_vma=False)` over the 3D mesh
  `('fsdp', 'cp', 'ep')`, where `gate_proj` and `up_proj` are sharded on axis 0
  and `down_proj` is sharded on axis 1.

  Args:
    x_local: Local token activations of shape `[..., hidden_size]`.
    weights_shard: Local `FlintMLPWeights` parameter shards.
    cfg: Validated `FlintConfig` controlling compute dtype and sharding.

  Returns:
    Local MLP output tensor with shape `x_local.shape` and dtype
    `cfg.train.compute_jnp_dtype`.
  """
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else None
  x_in = x_local.astype(comp_dt)
  if cfg.kernels.use_fp8:
    if cfg.kernels.fuse_layer_projections:
      intermediate_size = weights_shard.gate_proj.shape[-1]
      gate_up_shard = jnp.concatenate([weights_shard.gate_proj, weights_shard.up_proj], axis=-1)
      gate_up = ops.fp8_dense_dot_sharded(x_in, gate_up_shard, cfg, sharded_axis=0)
      gate = gate_up[..., :intermediate_size]
      up = gate_up[..., intermediate_size:]
    else:
      gate = ops.fp8_dense_dot_sharded(x_in, weights_shard.gate_proj, cfg, sharded_axis=0)
      up = ops.fp8_dense_dot_sharded(x_in, weights_shard.up_proj, cfg, sharded_axis=0)
    gate = jax.ad_checkpoint.checkpoint_name(gate, "mlp_gate_up")
    up = jax.ad_checkpoint.checkpoint_name(up, "mlp_gate_up")
    act = jax.nn.silu(gate) * up
    if cfg.kernels.transpose_dense_rhs_ag:
      return ops.fp8_dense_dot_sharded(
          act,
          jnp.swapaxes(weights_shard.down_proj, 0, 1),
          cfg,
          sharded_axis=0,
          rhs_is_transposed=True,
      )
    return ops.fp8_dense_dot_sharded(act, weights_shard.down_proj, cfg, sharded_axis=1)

  if cfg.kernels.fuse_layer_projections:
    inter = weights_shard.gate_proj.shape[-1]
    gu_shard = jnp.concatenate([weights_shard.gate_proj, weights_shard.up_proj], axis=-1)
    gu_proj = ops.gather_dense_weight(gu_shard, cfg, sharded_axis=0)
    gu = jnp.dot(x_in, gu_proj, precision=prec)
    gate = gu[..., :inter]
    up = gu[..., inter:]
  else:
    gate_proj = ops.gather_dense_weight(weights_shard.gate_proj, cfg, sharded_axis=0)
    up_proj = ops.gather_dense_weight(weights_shard.up_proj, cfg, sharded_axis=0)
    gate = jnp.dot(x_in, gate_proj, precision=prec)
    up = jnp.dot(x_in, up_proj, precision=prec)
  gate = jax.ad_checkpoint.checkpoint_name(gate, "mlp_gate_up")
  up = jax.ad_checkpoint.checkpoint_name(up, "mlp_gate_up")
  act = jax.nn.silu(gate) * up
  if cfg.kernels.transpose_dense_rhs_ag:
    down_proj_t = ops.gather_dense_weight(jnp.swapaxes(weights_shard.down_proj, 0, 1), cfg, sharded_axis=0)
    return jax.lax.dot_general(
        act,
        down_proj_t,
        (((act.ndim - 1,), (1,)), ((), ())),
        precision=prec,
    )
  down_proj = ops.gather_dense_weight(weights_shard.down_proj, cfg, sharded_axis=1)
  return jnp.dot(act, down_proj, precision=prec)
