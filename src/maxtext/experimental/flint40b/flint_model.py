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

"""Full `shard_map` pure-JAX model forward and `value_and_grad` for Flint."""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_layers_scan
from maxtext.experimental.flint40b import flint_loss
from maxtext.experimental.flint40b import flint_router
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops

P = PartitionSpec
_ALL_AXES = flint_config.ALL_AXES

ValueAndGradOut = tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    flint_types.FlintParams[jax.Array],
]


def _batch_pspecs() -> Any:
  """Returns the 2D `PartitionSpec` PyTree for `FlintBatch`."""
  b_spec = flint_sharding.batch_pspec()
  batch_cls: Any = flint_types.FlintBatch
  return batch_cls(
      tokens=b_spec,
      targets=b_spec,
      positions=b_spec,
      segment_ids=b_spec,
      loss_mask=b_spec,
  )


def batch_named_shardings(mesh: Mesh) -> Any:
  """Returns the 2D `NamedSharding` PyTree for `FlintBatch` on `mesh`."""
  b_sharding = NamedSharding(mesh, flint_sharding.batch_pspec())
  batch_cls: Any = flint_types.FlintBatch
  return batch_cls(
      tokens=b_sharding,
      targets=b_sharding,
      positions=b_sharding,
      segment_ids=b_sharding,
      loss_mask=b_sharding,
  )


def _batch_natural_to_lb(
    batch_local: flint_types.FlintBatch,
    cp_size: int,
) -> flint_types.FlintBatch:
  """Permutes local `FlintBatch` sequence shards from natural to LB CP order."""
  if cp_size <= 1:
    return batch_local
  return flint_types.FlintBatch(
      tokens=ops.cp_natural_to_lb(batch_local.tokens, cp_size, axis=1),
      targets=ops.cp_natural_to_lb(batch_local.targets, cp_size, axis=1),
      positions=ops.cp_natural_to_lb(batch_local.positions, cp_size, axis=1),
      segment_ids=ops.cp_natural_to_lb(batch_local.segment_ids, cp_size, axis=1),
      loss_mask=ops.cp_natural_to_lb(batch_local.loss_mask, cp_size, axis=1),
  )


def embed_tokens_local(
    tokens_local: jax.Array,
    embed_shard: jax.Array,
    cfg: flint_config.FlintConfig,
) -> jax.Array:
  """Gathers `embed_shard` in FP32 and indexes `tokens_local`.

  All-gathering `embed_shard` (`P(None, ('fsdp', 'cp', 'ep'))`) in `float32`
  before `jnp.take` keeps backward scatter-add and `psum_scatter` in `float32`.

  Args:
    tokens_local: Local integer token IDs `int32[B_loc, S_loc]`.
    embed_shard: Local `embed` shard `[vocab_size, hidden_size // num_devices]`.
    cfg: Validated `FlintConfig`.

  Returns:
    Local embedded activations `[B_loc, S_loc, hidden_size]` cast to
    `cfg.train.compute_jnp_dtype`.
  """
  embed_f32 = ops.gather_dense_weight(embed_shard, cfg, sharded_axis=1, dtype=jnp.float32, bwd_use_sc=False)
  tok_i32 = tokens_local.astype(jnp.int32)
  return jnp.take(embed_f32, tok_i32, axis=0, mode="fill", fill_value=jnp.nan).astype(cfg.train.compute_jnp_dtype)


@functools.lru_cache(maxsize=16)
def _make_forward_logits_jit(
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    upcast_lm_head_f32: bool,
) -> Callable[[flint_types.FlintParams[jax.Array], flint_types.FlintBatch], jax.Array]:
  """Builds and caches the `@jax.jit` `shard_map` forward-logits function."""
  p_specs = flint_sharding.param_pspecs(cfg)
  b_specs = _batch_pspecs()
  act_spec = flint_sharding.activation_pspec()

  p_shardings = flint_sharding.named_param_shardings(mesh, cfg)
  b_shardings = batch_named_shardings(mesh)
  out_sharding = NamedSharding(mesh, act_spec)

  def _local_logits(
      p_l: flint_types.FlintParams[jax.Array],
      b_l: flint_types.FlintBatch,
  ) -> jax.Array:
    cp_size = cfg.sharding.cp_size
    use_lb = cfg.kernels.use_lb_cp_layout and cp_size > 1
    if use_lb:
      b_l = _batch_natural_to_lb(b_l, cp_size)
    x_l = embed_tokens_local(b_l.tokens, p_l.embed, cfg)
    h_l, _, _ = flint_layers_scan.run_decoder(x_l, p_l.decoder, b_l.positions, b_l.segment_ids, b_l.loss_mask, cfg)
    normed_l = ops.flint_rms_norm(h_l, p_l.final_norm, cfg.model.rms_norm_eps)
    logits_l = flint_loss.compute_logits_f32(normed_l, p_l.lm_head, cfg, upcast_weight_f32=upcast_lm_head_f32)
    if use_lb:
      logits_l = ops.cp_lb_to_natural(logits_l, cp_size, axis=1)
    return logits_l

  shmapped = jax.shard_map(
      _local_logits,
      mesh=mesh,
      in_specs=(p_specs, b_specs),
      out_specs=act_spec,
      check_vma=False,
  )
  return jax.jit(
      shmapped,
      in_shardings=(p_shardings, b_shardings),
      out_shardings=out_sharding,
  )


def forward_logits(
    params: flint_types.FlintParams[jax.Array],
    batch: flint_types.FlintBatch,
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    *,
    upcast_lm_head_f32: bool = True,
) -> jax.Array:
  """Runs full sharded Flint forward pass returning `float32[B, S, V]` logits.

  Args:
    params: Sharded `FlintParams` on `mesh`.
    batch: Sharded `FlintBatch` on `mesh`.
    cfg: Validated `FlintConfig`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    upcast_lm_head_f32: If `True`, upcasts `normed_x` and `lm_head` to `float32`
      before vocabulary projection, matching `TorchFlintForCausalLM`.

  Returns:
    Sharded `float32[B, S, vocab_size]` logits (`P(('fsdp', 'ep'), 'cp',
    None)`).
  """
  fn = _make_forward_logits_jit(cfg, mesh, upcast_lm_head_f32)
  return fn(params, batch)


@functools.lru_cache(maxsize=16)
def _make_forward_intermediates_jit(
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    upcast_lm_head_f32: bool,
    return_router_aux: bool,
) -> Callable[
    [flint_types.FlintParams[jax.Array], flint_types.FlintBatch],
    tuple[
        jax.Array,
        tuple[jax.Array, ...],
        tuple[flint_types.FlintMoeAux | None, ...],
    ],
]:
  """Builds and caches `@jax.jit` `shard_map` forward-with-intermediates."""
  p_specs = flint_sharding.param_pspecs(cfg)
  b_specs = _batch_pspecs()
  act_spec = flint_sharding.activation_pspec()
  num_layers = cfg.model.num_hidden_layers

  aux_cls: Any = flint_types.FlintMoeAux
  aux_spec = aux_cls(
      expert_counts=P(),
      prob_sum=P(),
      topk_ids=act_spec,
      topk_weights=act_spec,
  )
  layer_h_specs = tuple(act_spec for _ in range(num_layers))
  layer_aux_specs = tuple(
      aux_spec if (return_router_aux and flint_config.is_moe_layer(i, cfg.model)) else None for i in range(num_layers)
  )
  out_specs = (act_spec, layer_h_specs, layer_aux_specs)

  p_shardings = flint_sharding.named_param_shardings(mesh, cfg)
  b_shardings = batch_named_shardings(mesh)
  out_shardings = jax.tree.map(lambda s: NamedSharding(mesh, s), out_specs)

  def _local_forward(
      p_l: flint_types.FlintParams[jax.Array],
      b_l: flint_types.FlintBatch,
  ) -> tuple[
      jax.Array,
      tuple[jax.Array, ...],
      tuple[flint_types.FlintMoeAux | None, ...],
  ]:
    cp_size = cfg.sharding.cp_size
    use_lb = cfg.kernels.use_lb_cp_layout and cp_size > 1
    if use_lb:
      b_l = _batch_natural_to_lb(b_l, cp_size)
    x_l = embed_tokens_local(b_l.tokens, p_l.embed, cfg)
    h_l, layer_hs, raw_aux = flint_layers_scan.run_decoder_with_intermediates(
        x_l,
        p_l.decoder,
        b_l.positions,
        b_l.segment_ids,
        b_l.loss_mask,
        cfg,
        return_router_aux=return_router_aux,
    )
    reduced_aux: list[flint_types.FlintMoeAux | None] = []
    for aux_i in raw_aux:
      if aux_i is None:
        reduced_aux.append(None)
      else:
        topk_ids = ops.cp_lb_to_natural(aux_i.topk_ids, cp_size, axis=1) if use_lb else aux_i.topk_ids
        topk_weights = ops.cp_lb_to_natural(aux_i.topk_weights, cp_size, axis=1) if use_lb else aux_i.topk_weights
        if cfg.sharding.num_devices > 1:
          counts = jax.lax.psum(aux_i.expert_counts, _ALL_AXES)
          probs = jax.lax.psum(aux_i.prob_sum, _ALL_AXES)
        else:
          counts = aux_i.expert_counts
          probs = aux_i.prob_sum
        reduced_aux.append(
            flint_types.FlintMoeAux(
                expert_counts=counts,
                prob_sum=probs,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        )

    normed_l = ops.flint_rms_norm(h_l, p_l.final_norm, cfg.model.rms_norm_eps)
    logits_l = flint_loss.compute_logits_f32(normed_l, p_l.lm_head, cfg, upcast_weight_f32=upcast_lm_head_f32)
    if use_lb:
      logits_l = ops.cp_lb_to_natural(logits_l, cp_size, axis=1)
      layer_hs = tuple(ops.cp_lb_to_natural(h_i, cp_size, axis=1) for h_i in layer_hs)
    return logits_l, layer_hs, tuple(reduced_aux)

  shmapped = jax.shard_map(
      _local_forward,
      mesh=mesh,
      in_specs=(p_specs, b_specs),
      out_specs=out_specs,
      check_vma=False,
  )
  return jax.jit(
      shmapped,
      in_shardings=(p_shardings, b_shardings),
      out_shardings=out_shardings,
  )


def forward_with_intermediates(
    params: flint_types.FlintParams[jax.Array],
    batch: flint_types.FlintBatch,
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    *,
    upcast_lm_head_f32: bool = True,
    return_router_aux: bool = True,
) -> tuple[
    jax.Array,
    tuple[jax.Array, ...],
    tuple[flint_types.FlintMoeAux | None, ...],
]:
  """Runs sharded Flint forward pass returning logits, layer hiddens, and aux.

  Args:
    params: Sharded `FlintParams` on `mesh`.
    batch: Sharded `FlintBatch` on `mesh`.
    cfg: Validated `FlintConfig`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    upcast_lm_head_f32: If `True`, evaluates `lm_head` matmul in `float32`.
    return_router_aux: If `True`, returns global-reduced `expert_counts` /
      `prob_sum` and sharded `topk_ids` / `topk_weights` per MoE layer; if
      `False`, returns `None` for every `layer_moe_aux` entry.

  Returns:
    Tuple `(logits_f32, layer_hiddens, layer_moe_aux)` of chronological length
    `cfg.model.num_hidden_layers`.
  """
  fn = _make_forward_intermediates_jit(cfg, mesh, upcast_lm_head_f32, return_router_aux)
  return fn(params, batch)


def shard_mapped_value_and_grad(
    params: flint_types.FlintParams[jax.Array],
    batch: flint_types.FlintBatch,
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    *,
    fp32_grads: bool = False,
) -> ValueAndGradOut:
  """Executes `check_vma=False`-safe `shard_map` forward and backward pass.

  Never differentiates through `jax.lax.psum` (avoiding the `xN` `psum`
  transpose bug under `check_vma=False`): `total_global_valid` is computed and
  stopped before `jax.value_and_grad`, local loss contributions are
  differentiated per rank/microbatch, and scalar `psum` +
  `psum_replicated_grads`
  execute strictly once after `value_and_grad` (and after the microbatch
  `lax.scan` when `cfg.train.grad_accum_steps > 1`).

  Args:
    params: Sharded `FlintParams` on `mesh`.
    batch: Sharded `FlintBatch` on `mesh`.
    cfg: Validated `FlintConfig`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    fp32_grads: When `True`, returns all gradient leaves in `jnp.float32` (used
      by `flint_train` for lossless FP32 norm/AdamW updates); when `False`,
      casts each gradient leaf back to its corresponding `params` leaf dtype.

  Returns:
    Tuple `((global_loss, global_ce, global_aux), grads)` where `global_aux` is
    the valid-token-weighted mean Switch load-balancing loss across MoE layers.
  """
  p_specs = flint_sharding.param_pspecs(cfg)
  b_specs = _batch_pspecs()
  out_specs = ((P(), P(), P()), p_specs)
  ga_steps = cfg.train.grad_accum_steps
  num_devices = cfg.sharding.num_devices

  def _local_step(
      p_l: flint_types.FlintParams[jax.Array],
      b_l: flint_types.FlintBatch,
  ) -> ValueAndGradOut:
    b_loc, s_loc = b_l.tokens.shape
    mb_b = b_loc // ga_steps
    msk_mb = b_l.loss_mask.reshape(ga_steps, mb_b, s_loc)
    mb_local_valid = jnp.sum((msk_mb > 0).astype(jnp.float32), axis=(1, 2))
    if num_devices > 1:
      mb_global_valid = jax.lax.psum(mb_local_valid, _ALL_AXES)
    else:
      mb_global_valid = mb_local_valid
    mb_global_valid = jax.lax.stop_gradient(mb_global_valid)
    raw_total_valid = jax.lax.stop_gradient(jnp.sum(mb_global_valid))
    total_valid_const = jax.lax.stop_gradient(jnp.maximum(raw_total_valid, 1.0))
    if cfg.kernels.use_lb_cp_layout and cfg.sharding.cp_size > 1:
      b_l = _batch_natural_to_lb(b_l, cfg.sharding.cp_size)

    def _microbatch_loss(
        params_local: flint_types.FlintParams[jax.Array],
        tok_l: jax.Array,
        tgt_l: jax.Array,
        pos_l: jax.Array,
        seg_l: jax.Array,
        msk_l: jax.Array,
        raw_mb_valid: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
      mb_valid_const = jax.lax.stop_gradient(jnp.maximum(raw_mb_valid, 1.0))
      mb_weight = jax.lax.stop_gradient(raw_mb_valid / total_valid_const)

      x_l = embed_tokens_local(tok_l, params_local.embed, cfg)
      h_l, counts_le, prob_sum_le = flint_layers_scan.run_decoder(x_l, params_local.decoder, pos_l, seg_l, msk_l, cfg)
      normed_l = ops.flint_rms_norm(h_l, params_local.final_norm, cfg.model.rms_norm_eps)
      local_ce_sum = flint_loss.chunked_lm_head_ce_sum(normed_l, params_local.lm_head, tgt_l, msk_l, cfg)
      local_ce_contrib = local_ce_sum / total_valid_const

      skip_aux = cfg.kernels.skip_router_aux_stats_when_zero_coef and cfg.train.router_aux_loss_coef <= 0.0
      if counts_le.shape[0] > 0 and not skip_aux:
        mb_aux_sum = flint_router.compute_local_moe_aux_loss(
            counts_le,
            prob_sum_le,
            mb_valid_const,
            num_experts=cfg.model.num_experts,
            top_k=cfg.model.num_experts_per_tok,
        )
        local_aux_contrib = (mb_aux_sum / float(counts_le.shape[0])) * mb_weight
      else:
        local_aux_contrib = jnp.zeros((), dtype=jnp.float32)

      if cfg.train.router_aux_loss_coef > 0.0:
        coef = jnp.asarray(cfg.train.router_aux_loss_coef, dtype=jnp.float32)
        local_total_contrib = local_ce_contrib + coef * local_aux_contrib
      else:
        local_total_contrib = local_ce_contrib
      return local_total_contrib, (
          jax.lax.stop_gradient(local_ce_contrib),
          jax.lax.stop_gradient(local_aux_contrib),
      )

    vg_fn = jax.value_and_grad(_microbatch_loss, has_aux=True)

    if ga_steps == 1:
      (local_total, (local_ce, local_aux)), raw_grads = vg_fn(
          p_l,
          b_l.tokens,
          b_l.targets,
          b_l.positions,
          b_l.segment_ids,
          b_l.loss_mask,
          mb_global_valid[0],
      )
      grads_f32 = jax.tree.map(lambda g: g.astype(jnp.float32), raw_grads)
    else:
      tok_mb = b_l.tokens.reshape(ga_steps, mb_b, s_loc)
      tgt_mb = b_l.targets.reshape(ga_steps, mb_b, s_loc)
      pos_mb = b_l.positions.reshape(ga_steps, mb_b, s_loc)
      seg_mb = b_l.segment_ids.reshape(ga_steps, mb_b, s_loc)

      z_scalar = jnp.zeros((), dtype=jnp.float32)
      grads_init_f32 = jax.tree.map(lambda p: jnp.zeros(p.shape, dtype=jnp.float32), p_l)

      def _ga_scan_body(
          carry: tuple[
              jax.Array,
              jax.Array,
              jax.Array,
              flint_types.FlintParams[jax.Array],
          ],
          mb_in: tuple[
              jax.Array,
              jax.Array,
              jax.Array,
              jax.Array,
              jax.Array,
              jax.Array,
          ],
      ) -> tuple[
          tuple[
              jax.Array,
              jax.Array,
              jax.Array,
              flint_types.FlintParams[jax.Array],
          ],
          None,
      ]:
        loss_acc, ce_acc, aux_acc, g_acc = carry
        tok_i, tgt_i, pos_i, seg_i, msk_i, mb_v_i = mb_in
        (lt_i, (lce_i, laux_i)), g_i = vg_fn(p_l, tok_i, tgt_i, pos_i, seg_i, msk_i, mb_v_i)
        g_next = jax.tree.map(lambda acc, g: acc + g.astype(jnp.float32), g_acc, g_i)
        return (loss_acc + lt_i, ce_acc + lce_i, aux_acc + laux_i, g_next), None

      (local_total, local_ce, local_aux, grads_f32), _ = jax.lax.scan(
          _ga_scan_body,
          (z_scalar, z_scalar, z_scalar, grads_init_f32),
          (tok_mb, tgt_mb, pos_mb, seg_mb, msk_mb, mb_global_valid),
      )

    if num_devices > 1:
      global_loss = jax.lax.psum(local_total, _ALL_AXES)
      global_ce = jax.lax.psum(local_ce, _ALL_AXES)
      global_aux = jax.lax.psum(local_aux, _ALL_AXES)
    else:
      global_loss = local_total
      global_ce = local_ce
      global_aux = local_aux

    grads_f32 = ops.psum_replicated_grads(grads_f32, p_specs, cfg.sharding)
    if fp32_grads:
      grads_out = grads_f32
    else:
      grads_out = jax.tree.map(lambda g, p: g.astype(p.dtype), grads_f32, p_l)
    return (global_loss, global_ce, global_aux), grads_out

  shmapped = jax.shard_map(
      _local_step,
      mesh=mesh,
      in_specs=(p_specs, b_specs),
      out_specs=out_specs,
      check_vma=False,
  )
  return shmapped(params, batch)


@functools.lru_cache(maxsize=16)
def make_value_and_grad_fn(
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
) -> Callable[
    [flint_types.FlintParams[jax.Array], flint_types.FlintBatch],
    ValueAndGradOut,
]:
  """Returns a `@jax.jit`-compiled `shard_map` `(loss_tuple, grads)` callable.

  Args:
    cfg: Validated `FlintConfig`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.

  Returns:
    Compiled callable `(params, batch) -> ((loss, ce_loss, aux_loss), grads)`.
  """
  p_shardings = flint_sharding.named_param_shardings(mesh, cfg)
  b_shardings = batch_named_shardings(mesh)
  rep_sharding = NamedSharding(mesh, P())
  out_shardings = (
      (rep_sharding, rep_sharding, rep_sharding),
      p_shardings,
  )

  def _vg(
      params: flint_types.FlintParams[jax.Array],
      batch: flint_types.FlintBatch,
  ) -> ValueAndGradOut:
    return shard_mapped_value_and_grad(params, batch, cfg, mesh, fp32_grads=False)

  return jax.jit(
      _vg,
      in_shardings=(p_shardings, b_shardings),
      out_shardings=out_shardings,
  )
