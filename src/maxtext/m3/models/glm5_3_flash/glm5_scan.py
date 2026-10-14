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


def layer_to_block_slot(
    layer_idx: int,
    first_k_dense_replace: int,
    num_super_blocks: int,
) -> tuple[str, int, int | None]:
  """Maps flat layer index to (section, block_or_layer_idx, slot_idx)."""
  if layer_idx < 0:
    raise ValueError(f"layer_idx must be non-negative: {layer_idx}")
  if num_super_blocks < 0:
    raise ValueError(f"num_super_blocks must be non-negative: {num_super_blocks}")
  if layer_idx < first_k_dense_replace:
    return ("prefix", layer_idx, None)
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
    # Slots 0..3 map to MLA (slot 0) and KDA (slots 1..3), all with MoE.
    self.slot_0 = glm5_mhc.Glm5DecoderLayer(config, 3, rngs=rngs, sharding=sharding)
    self.slot_1 = glm5_mhc.Glm5DecoderLayer(config, 4, rngs=rngs, sharding=sharding)
    self.slot_2 = glm5_mhc.Glm5DecoderLayer(config, 5, rngs=rngs, sharding=sharding)
    self.slot_3 = glm5_mhc.Glm5DecoderLayer(config, 6, rngs=rngs, sharding=sharding)

  def __call__(
      self,
      hidden_streams: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    hidden_streams, r0 = self.slot_0(hidden_streams, decoder_segment_ids=decoder_segment_ids)
    hidden_streams, r1 = self.slot_1(hidden_streams, decoder_segment_ids=decoder_segment_ids)
    hidden_streams, r2 = self.slot_2(hidden_streams, decoder_segment_ids=decoder_segment_ids)
    hidden_streams, r3 = self.slot_3(hidden_streams, decoder_segment_ids=decoder_segment_ids)
    if r0 is None or r1 is None or r2 is None or r3 is None:
      raise ValueError("Super-block slots must all be MoE with non-None router logits.")
    return hidden_streams, (r0, r1, r2, r3)


def create_stacked_super_block(
    config: glm5_config.Glm5Config,
    num_super_blocks: int,
    *,
    rngs: nnx.Rngs,
    sharding: m3_sharding.Sharding | None = None,
) -> Glm5SuperBlock:
  """Constructs a stacked Glm5SuperBlock with leading dimension num_super_blocks."""
  if num_super_blocks <= 0:
    raise ValueError(f"num_super_blocks must be positive, got {num_super_blocks}")

  blocks = [Glm5SuperBlock(config, rngs=rngs, sharding=sharding) for _ in range(num_super_blocks)]
  graphdefs, states = zip(*[nnx.split(b) for b in blocks])

  def stack_leaves(*leaves: Any) -> Any:
    first = leaves[0]
    vals = [leaf.value if isinstance(leaf, nnx.Variable) else leaf for leaf in leaves]
    stacked_val = jnp.stack(vals, axis=0)
    if isinstance(first, nnx.Variable):
      orig_sh = first.sharding
      if isinstance(orig_sh, jax.sharding.NamedSharding):
        new_spec = jax.sharding.PartitionSpec(None, *orig_sh.spec)
        new_sh = jax.sharding.NamedSharding(orig_sh.mesh, new_spec)
        return first.replace(jax.device_put(stacked_val, new_sh), sharding=new_sh)
      if isinstance(orig_sh, jax.sharding.PartitionSpec):
        new_spec = jax.sharding.PartitionSpec(None, *orig_sh)
        return first.replace(stacked_val, sharding=new_spec)
      return first.replace(stacked_val)
    return stacked_val

  stacked_state = jax.tree.map(stack_leaves, *states, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return nnx.merge(graphdefs[0], stacked_state)


def _slice_state(state: Any, i: int) -> Any:
  def _slice_leaf(v: Any) -> Any:
    if isinstance(v, nnx.Variable):
      return v.replace(v.value[i])
    return v[i]

  return jax.tree.map(_slice_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def scan_super_blocks(
    super_block: Glm5SuperBlock,
    hidden_streams: jax.Array,
    *,
    scan_layers: bool,
    decoder_segment_ids: jax.Array | None = None,
    remat_policy: Any = None,
) -> tuple[jax.Array, list[jax.Array]]:
  """Runs stacked Glm5SuperBlock via jax.lax.scan or unrolled loop."""
  graphdef, params, non_params = nnx.split(super_block, nnx.Param, ...)
  leaves = jax.tree.leaves(params)
  if not leaves:
    raise ValueError("super_block has no parameters.")
  num_super_blocks = leaves[0].shape[0]

  def step_fn(carry: jax.Array, xs_i: tuple[Any, Any]) -> tuple[jax.Array, jax.Array]:
    p_i, np_i = xs_i
    block = nnx.merge(graphdef, p_i, np_i)
    carry, (r0, r1, r2, r3) = block(carry, decoder_segment_ids=decoder_segment_ids)
    r_step = jnp.stack([r0, r1, r2, r3], axis=0)
    return carry, r_step

  if remat_policy is not None and remat_policy is not False:
    step_exec = jax.checkpoint(
        step_fn,
        policy=remat_policy if callable(remat_policy) else None,
    )
  else:
    step_exec = step_fn

  if scan_layers:
    final_streams, scanned_r = jax.lax.scan(step_exec, hidden_streams, (params, non_params))
    all_router_logits = [scanned_r[b, s] for b in range(num_super_blocks) for s in range(4)]
    return final_streams, all_router_logits
  else:
    carry = hidden_streams
    all_router_logits = []
    for i in range(num_super_blocks):
      p_i = _slice_state(params, i)
      np_i = _slice_state(non_params, i)
      carry, r_step = step_exec(carry, (p_i, np_i))
      for s in range(4):
        all_router_logits.append(r_step[s])
    return carry, all_router_logits


def apply_glm5_remat(model: Any, *, policy: Any = True) -> Any:
  """Applies remat policy to model without altering parameter paths."""
  model.remat_policy = policy
  return model
