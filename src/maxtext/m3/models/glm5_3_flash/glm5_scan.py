# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path-transparent super-block scan and remat for GLM-5.3-Flash."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_mhc
from maxtext.m3.models.glm5_3_flash import glm5_moe


def layer_to_block_slot(
    layer_idx: int,
    first_k_dense_replace: int,
    num_hidden_layers: int,
) -> tuple[str, int, int | None]:
  """Maps flat layer index to (section, block_or_layer_idx, slot_idx)."""
  if num_hidden_layers <= 0:
    raise ValueError(f"num_hidden_layers must be positive: {num_hidden_layers}")
  if layer_idx < 0 or layer_idx >= num_hidden_layers:
    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_hidden_layers})")
  if first_k_dense_replace < 0 or first_k_dense_replace > num_hidden_layers:
    raise ValueError(f"first_k_dense_replace {first_k_dense_replace} invalid for" f" {num_hidden_layers} layers")
  if layer_idx < first_k_dense_replace:
    return ("prefix", layer_idx, None)
  num_super_blocks = (num_hidden_layers - first_k_dense_replace) // 4
  super_block_layers = 4 * num_super_blocks
  rel = layer_idx - first_k_dense_replace
  if rel < super_block_layers:
    return ("super_block", rel // 4, rel % 4)
  return ("suffix", rel - super_block_layers, None)


class Glm5SuperBlock(nnx.Module):
  """4-layer super-block: (MLA/MoE, KDA/MoE, KDA/MoE, KDA/MoE)."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    fk = config.first_k_dense_replace
    # Slots 0..3 correspond to layers fk..fk+3 (1 MLA + 3 KDA, all with MoE).
    self.slot_0 = glm5_mhc.Glm5DecoderLayer(config, fk + 0, rngs=rngs, sharding=sharding)
    self.slot_1 = glm5_mhc.Glm5DecoderLayer(config, fk + 1, rngs=rngs, sharding=sharding)
    self.slot_2 = glm5_mhc.Glm5DecoderLayer(config, fk + 2, rngs=rngs, sharding=sharding)
    self.slot_3 = glm5_mhc.Glm5DecoderLayer(config, fk + 3, rngs=rngs, sharding=sharding)

  def __call__(
      self,
      hidden_streams: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      remat_policy: Any = None,
      cur_lb: tuple[jax.Array, jax.Array, jax.Array] | None = None,
      return_router_logits: bool = True,
  ) -> tuple[
      jax.Array,
      jax.Array | None,
      jax.Array | None,
      jax.Array | None,
      tuple[jax.Array, jax.Array, jax.Array] | None,
  ]:
    cfg = self.config
    num_experts, top_k = cfg.n_routed_experts, cfg.num_experts_per_tok
    attn_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None

    def _call_slot(
        slot: glm5_mhc.Glm5DecoderLayer,
        streams: jax.Array,
        lb: tuple[jax.Array, jax.Array, jax.Array] | None,
    ):
      streams, r, rt, idx = slot(streams, decoder_segment_ids=decoder_segment_ids)
      if r is None or rt is None:
        raise ValueError("Super-block slots must all be MoE with non-None router logits and" " topk.")
      if lb is not None:
        lb = glm5_moe.accumulate_moe_lb_stats(
            lb,
            r,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attn_mask,
        )
      if return_router_logits:
        return streams, lb, r, rt, idx
      return streams, lb, idx

    slots = (self.slot_0, self.slot_1, self.slot_2, self.slot_3)
    router_logits = []
    router_topks = []
    indexer_topks = []
    for slot in slots:
      if remat_policy is not None:
        if not callable(remat_policy):
          raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
        exec_slot = jax.checkpoint(
            lambda s, lb, sl=slot: _call_slot(sl, s, lb),
            policy=remat_policy,
            prevent_cse=True,
        )
      else:

        def _exec_slot_direct(s: jax.Array, lb: Any, sl: Any = slot) -> Any:
          return _call_slot(sl, s, lb)

        exec_slot = _exec_slot_direct

      if return_router_logits:
        hidden_streams, cur_lb, r, rt, idx = exec_slot(hidden_streams, cur_lb)
        router_logits.append(r)
        router_topks.append(rt)
      else:
        hidden_streams, cur_lb, idx = exec_slot(hidden_streams, cur_lb)
      indexer_topks.append(idx)
    mla_idx = next(
        (idx for slot, idx in zip(slots, indexer_topks) if slot.layer_type == "full_attention"),
        None,
    )
    r_stack = jnp.stack(router_logits, axis=0) if return_router_logits else None
    rt_stack = jnp.stack(router_topks, axis=0) if return_router_logits else None
    return (
        hidden_streams,
        r_stack,
        rt_stack,
        mla_idx,
        cur_lb,
    )


def create_stacked_super_block(
    config: glm5_config.Glm5Config,
    num_super_blocks: int,
    *,
    rngs: nnx.Rngs,
    sharding: m3_sharding.Sharding | None = None,
    place_on_devices: bool,
) -> Glm5SuperBlock:
  """Constructs a stacked Glm5SuperBlock with leading dimension num_super_blocks."""
  if num_super_blocks <= 0:
    raise ValueError(f"num_super_blocks must be positive, got {num_super_blocks}")

  fk, lt, it = (
      config.first_k_dense_replace,
      config.layer_types,
      config.indexer_types,
  )
  if lt is None or it is None:
    raise ValueError(f"layer_types and indexer_types must not be None, got {lt=}, {it=}")
  if fk + 4 * num_super_blocks > config.num_hidden_layers:
    raise ValueError(
        f"Required {fk + 4 * num_super_blocks} layers for {num_super_blocks}"
        f" super blocks with first_k_dense_replace={fk} exceeds"
        f" num_hidden_layers={config.num_hidden_layers}."
    )
  for b in range(num_super_blocks):
    for s in range(4):
      idx = fk + 4 * b + s
      if lt[idx] != lt[fk + s] or it[idx] != it[fk + s]:
        raise ValueError(f"Non-periodic schedule at layer {idx}: {lt[idx]!r} / {it[idx]!r}.")

  def _init_single(r: nnx.Rngs) -> Glm5SuperBlock:
    return Glm5SuperBlock(config, rngs=r, sharding=sharding)

  forked_rngs = rngs.split(num_super_blocks)
  stacked_block = nnx.vmap(_init_single, in_axes=0, out_axes=0)(forked_rngs)
  graphdef, state = nnx.split(stacked_block)

  def _update_var(path, var):
    if isinstance(var, nnx.Variable):
      orig_sh = var.get_metadata("out_sharding", None)
      if orig_sh is None:
        if sharding is not None:
          raise ValueError(f"Expected sharding metadata on stacked variable at {path}, got" " None.")
        if place_on_devices:
          raise ValueError(
              "place_on_devices=True requires a non-empty NamedSharding mesh" f" on stacked parameters at {path}."
          )
        return var
      orig_spec = orig_sh.spec if isinstance(orig_sh, jax.sharding.NamedSharding) else orig_sh
      if not isinstance(orig_spec, jax.sharding.PartitionSpec):
        raise ValueError(f"Unsupported sharding metadata type at {path}: {type(orig_sh)}")
      new_spec = jax.sharding.PartitionSpec(None, *orig_spec)
      raw_val = var.get_raw_value()
      if place_on_devices:
        raw_sh = raw_val.sharding if isinstance(raw_val, jax.Array) else None
        mesh = (
            orig_sh.mesh
            if isinstance(orig_sh, jax.sharding.NamedSharding)
            else (raw_sh.mesh if isinstance(raw_sh, jax.sharding.NamedSharding) else None)
        )
        if mesh is None or mesh.empty:
          raise ValueError(
              "place_on_devices=True requires a non-empty NamedSharding mesh" f" on stacked parameters at {path}."
          )
        new_sh = jax.sharding.NamedSharding(mesh, new_spec)
        placed_val = jax.device_put(raw_val, new_sh)
        return var.replace(placed_val, out_sharding=new_spec)
      return var.replace(raw_val, out_sharding=new_spec)
    return var

  updated_state = jax.tree_util.tree_map_with_path(_update_var, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return nnx.merge(graphdef, updated_state)


def _slice_state(state: Any, i: int) -> Any:
  """Slices stacked super-block state at index i and strips the leading PartitionSpec axis."""

  def _slice_leaf(v: Any) -> Any:
    if isinstance(v, nnx.Variable):
      raw_val = v.get_raw_value()[i]
      orig_sh = (
          v.get_metadata("out_sharding", None)
          if hasattr(v, "get_metadata")
          else (v.out_sharding if hasattr(v, "out_sharding") else None)
      )
      if isinstance(orig_sh, jax.sharding.PartitionSpec) and len(orig_sh) > 0:
        return v.replace(raw_val, out_sharding=jax.sharding.PartitionSpec(*orig_sh[1:]))
      return v.replace(raw_val)
    return v[i]

  return jax.tree.map(_slice_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def scan_super_blocks(
    super_block: Glm5SuperBlock,
    hidden_streams: jax.Array,
    *,
    scan_layers: bool,
    decoder_segment_ids: jax.Array | None = None,
    remat_policy: Any = None,
    return_router_logits: bool,
    lb_stats: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[
    jax.Array,
    list[jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
    list[jax.Array],
    Any,
]:
  """Runs stacked Glm5SuperBlock via jax.lax.scan or unrolled loop."""
  if super_block.slot_0.attn_hc.base.ndim != 2:
    raise ValueError(
        "super_block is not stacked: scan_super_blocks requires a stacked"
        " Glm5SuperBlock with a leading num_super_blocks axis, got"
        f" slot_0.attn_hc.base.ndim={super_block.slot_0.attn_hc.base.ndim}"
    )

  cfg = super_block.config
  num_experts = cfg.n_routed_experts
  cur_lb_init = lb_stats if lb_stats is not None else glm5_moe.init_moe_lb_stats(num_experts)

  graphdef, params, non_params = nnx.split(super_block, nnx.Param, ...)
  leaves = jax.tree.leaves(params)
  if not leaves:
    raise ValueError("super_block has no parameters.")
  num_super_blocks = leaves[0].shape[0]

  def step_fn(carry_in: Any, xs_i: tuple[Any, Any]) -> tuple[Any, Any]:
    p_i, np_i = xs_i
    carry, cur_lb = carry_in
    block = nnx.merge(graphdef, p_i, np_i)
    carry, block_router_logits, block_router_topk, block_idx, cur_lb = block(
        carry,
        decoder_segment_ids=decoder_segment_ids,
        remat_policy=remat_policy,
        cur_lb=cur_lb,
        return_router_logits=return_router_logits,
    )
    r_out = block_router_logits if return_router_logits else None
    rt_out = block_router_topk if return_router_logits else None
    carry_out = (carry, cur_lb)
    return carry_out, (r_out, rt_out, block_idx)

  if remat_policy is not None and not callable(remat_policy):
    raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
  step_exec = step_fn

  init_carry = (hidden_streams, cur_lb_init)
  if scan_layers:
    (final_streams, final_lb), (scanned_r, scanned_rt, scanned_idx) = jax.lax.scan(
        step_exec, init_carry, (params, non_params)
    )
    all_router_logits = (
        [scanned_r[b, s] for b in range(num_super_blocks) for s in range(4)]
        if return_router_logits and scanned_r is not None
        else []
    )
    all_router_topk = (
        [scanned_rt[b, s] for b in range(num_super_blocks) for s in range(4)]
        if return_router_logits and scanned_rt is not None
        else []
    )
    all_idx = [scanned_idx[b] for b in range(num_super_blocks)] if scanned_idx is not None else None
  else:
    carry_state = init_carry
    all_router_logits = []
    all_router_topk = []
    all_idx_list = []
    for i in range(num_super_blocks):
      p_i = _slice_state(params, i)
      np_i = _slice_state(non_params, i)
      carry_state, (r_step, rt_step, idx_step) = step_exec(carry_state, (p_i, np_i))
      if return_router_logits and r_step is not None:
        for s in range(4):
          all_router_logits.append(r_step[s])
          all_router_topk.append(rt_step[s])
      all_idx_list.append(idx_step)
    final_streams, final_lb = carry_state
    all_idx = all_idx_list if any(x is not None for x in all_idx_list) else None

  return final_streams, all_router_logits, final_lb, all_router_topk, all_idx


def apply_glm5_remat(model: Any, *, policy: Any) -> Any:
  """Applies remat policy to model without altering parameter paths."""
  if policy is not None and not callable(policy):
    raise ValueError(f"policy must be None or callable: {policy!r}.")
  model.remat_policy = policy
  return model
