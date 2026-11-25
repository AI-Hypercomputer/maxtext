# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero-extra-HBM 2-microbatch Dual-SparseCore schedule for m3 inhomogeneous decoders.

Splits global batch `B` into two interleaved microbatches `(mb0, mb1)` via a
zero-communication local slice (`x.reshape(B // 2, 2, ...)[:, i]`) so every FSDP
shard halves its local microbatch size (`B_local -> B_local // 2`) without cross-chip
resharding. Within each checkpointed sublayer:

  * `SC1` gathers (or STE-quantizes + gathers) routed-expert weights **once** and
    reduce-scatters routed-expert weight gradients **once** across both microbatches.
  * `mb0`'s `SC0` EP token dispatch and TensorCore grouped matmuls overlap with
    `mb1`'s attention/GDN + shared-expert compute via `jax.lax.optimization_barrier`
    and explicit `SC0`/`SC1` SparseCore pinning (`use_sparsecore_pinning`).
  * Only the post-combine `(h0, h1)` pair (`[B // 2, S, ...]` each) crosses sublayer
    `jax.checkpoint` boundaries, adding zero residual HBM while halving intra-layer
    activation temporaries.
"""

from collections.abc import Callable
from typing import Any

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp


def can_batch_split(x: jax.Array, batch_shards: int = 1) -> bool:
  """Returns True when `x.shape[0]` divides evenly into two sharded microbatches (`B >= 2 * batch_shards`)."""
  shards = max(1, int(batch_shards))
  return x.ndim >= 1 and x.shape[0] >= 2 * shards and (x.shape[0] % (2 * shards) == 0)


def _is_explicit_mesh(mesh: Any) -> bool:
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  return bool(axis_types) and all(t == jax.sharding.AxisType.Explicit for t in axis_types)


@jax.named_scope("m3_batch_split")
def split_microbatches(x: jax.Array, *, batch_axis: int = 0) -> tuple[jax.Array, jax.Array]:
  """Splits `x` along `batch_axis` into `(x[..., 0::2, ...], x[..., 1::2, ...])` with zero cross-shard copy."""
  ax = batch_axis % x.ndim
  b = x.shape[ax]
  if b < 2 or b % 2 != 0:
    raise ValueError(f"split_microbatches requires an even batch dimension >= 2 on axis {ax}; got shape {x.shape}.")
  reshaped_shape = (*x.shape[:ax], b // 2, 2, *x.shape[ax + 1 :])
  mesh = jax.sharding.get_abstract_mesh()
  if _is_explicit_mesh(mesh):
    cur_shd = getattr(jax.typeof(x), "sharding", None)
    cur_spec = tuple(getattr(cur_shd, "spec", ()))
    ext_spec = (*cur_spec, *([None] * (x.ndim - len(cur_spec))))
    out_spec = jax.sharding.PartitionSpec(*ext_spec[:ax], ext_spec[ax], None, *ext_spec[ax + 1 :])
    reshaped = jax.lax.reshape(x, reshaped_shape, out_sharding=out_spec)
  else:
    reshaped = jnp.reshape(x, reshaped_shape)
  mb0 = jax.lax.index_in_dim(reshaped, 0, axis=ax + 1, keepdims=False)
  mb1 = jax.lax.index_in_dim(reshaped, 1, axis=ax + 1, keepdims=False)
  return mb0, mb1


@jax.named_scope("m3_batch_merge")
def merge_microbatches(mb0: jax.Array, mb1: jax.Array, *, batch_axis: int = 0) -> jax.Array:
  """Merges interleaved microbatches `(mb0, mb1)` back along `batch_axis` with zero cross-shard copy."""
  if mb0.shape != mb1.shape:
    raise ValueError(f"merge_microbatches requires matching shapes; got {mb0.shape} vs {mb1.shape}.")
  ax = batch_axis % mb0.ndim
  stacked = jnp.stack([mb0, mb1], axis=ax + 1)
  merged_shape = (*mb0.shape[:ax], mb0.shape[ax] * 2, *mb0.shape[ax + 1 :])
  mesh = jax.sharding.get_abstract_mesh()
  if _is_explicit_mesh(mesh):
    cur_shd = getattr(jax.typeof(mb0), "sharding", None)
    cur_spec = tuple(getattr(cur_shd, "spec", ()))
    ext_spec = (*cur_spec, *([None] * (mb0.ndim - len(cur_spec))))
    out_spec = jax.sharding.PartitionSpec(*ext_spec)
    return jax.lax.reshape(stacked, merged_shape, out_sharding=out_spec)
  return jnp.reshape(stacked, merged_shape)


def split_sublayer_kwargs(dyn_kw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  """Splits per-batch dynamic sublayer keyword arguments into `(kw0, kw1)`."""
  kw0: dict[str, Any] = {}
  kw1: dict[str, Any] = {}
  for key, val in dyn_kw.items():
    if val is None:
      continue
    if key == "position_embeddings" and isinstance(val, tuple) and len(val) == 2:
      cos0, cos1 = split_microbatches(val[0], batch_axis=0)
      sin0, sin1 = split_microbatches(val[1], batch_axis=0)
      kw0[key] = (cos0, sin0)
      kw1[key] = (cos1, sin1)
    elif key == "positions" and isinstance(val, jax.Array):
      b_ax = 1 if val.ndim == 3 else 0
      kw0[key], kw1[key] = split_microbatches(val, batch_axis=b_ax)
    elif isinstance(val, jax.Array) and val.ndim >= 1:
      kw0[key], kw1[key] = split_microbatches(val, batch_axis=0)
    else:
      kw0[key] = val
      kw1[key] = val
  return kw0, kw1


def merge_dual_aux(
    aux0: dict[str, Any],
    aux1: dict[str, Any],
    *,
    compact_router_aux: bool,
) -> dict[str, Any]:
  """Combines microbatch auxiliary router dictionaries `(aux0, aux1)` into a full-batch aux dict."""
  if compact_router_aux:
    return {
        "tokens_per_expert_sum": aux0["tokens_per_expert_sum"] + aux1["tokens_per_expert_sum"],
        "router_prob_sum": aux0["router_prob_sum"] + aux1["router_prob_sum"],
        "total_rows": aux0["total_rows"] + aux1["total_rows"],
    }
  merged: dict[str, Any] = {
      "router_indices": merge_microbatches(aux0["router_indices"], aux1["router_indices"], batch_axis=0),
      "routing_weights": merge_microbatches(aux0["routing_weights"], aux1["routing_weights"], batch_axis=0),
      "router_probs": merge_microbatches(aux0["router_probs"], aux1["router_probs"], batch_axis=0),
  }
  if "selected_block_indices" in aux0 and "selected_block_indices" in aux1:
    merged["selected_block_indices"] = merge_microbatches(
        aux0["selected_block_indices"], aux1["selected_block_indices"], batch_axis=0
    )
  return merged


def run_dual_decoder_layer_core(
    layer: Any,
    h0: jax.Array,
    h1: jax.Array,
    kw0: dict[str, Any],
    kw1: dict[str, Any],
    *,
    return_aux: bool,
    compact_router_aux: bool,
    shard_batch_fn: Callable[[jax.Array, Any], jax.Array],
    compact_aux_fn: Callable[..., dict[str, jax.Array]],
    group_base: int = 100,
) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any] | None]:
  """Executes one inhomogeneous sublayer (`GDN + MoE` or `QSA + MoE`) on `(h0, h1)` with Dual-SC overlap."""
  need_selected_blocks = return_aux and not compact_router_aux

  def _run_attn_and_pre_moe(
      h_in: jax.Array,
      kw: dict[str, Any],
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any] | None]:
    h_cur = checkpoint_name(checkpoint_name(h_in, "decoder_layer_input"), "hidden_states")
    ple = getattr(layer, "ple", None)
    ple_embeds = kw.get("ple_embeds")
    ple_input_ids = kw.get("ple_input_ids")
    if ple is not None and (ple_embeds is not None or ple_input_ids is not None):
      h_cur = h_cur + ple(
          h_cur,
          ple_input_ids,
          conv_mask=kw.get("conv_mask"),
          precomputed_embeds=ple_embeds,
      )
    mixed_a, inject_a = layer.attn_hyper_connection.before_block(h_cur)
    if layer.layer_type == "linear_attention":
      attn_out = layer.linear_attn(mixed_a, attention_mask=kw.get("conv_mask"))
      s_idx = None
    else:
      qsa_out = layer.self_attn(
          mixed_a,
          position_ids=kw.get("positions"),
          attention_mask=kw.get("conv_mask"),
          segment_ids=kw.get("segment_ids"),
          return_selected_blocks=need_selected_blocks,
          position_embeddings=kw.get("position_embeddings"),
      )
      attn_out, s_idx = qsa_out if need_selected_blocks else (qsa_out, None)
    attn_out = checkpoint_name(attn_out, "self_attn")
    h_post_attn = shard_batch_fn(layer.attn_hyper_connection.combine(h_cur, attn_out, inject_a), layer.sharding)
    mixed_m, inject_m = layer.mlp_hyper_connection.before_block(h_post_attn)
    r_w, r_idx, r_probs = layer.mlp.gate(mixed_m)
    shared_raw = layer.mlp.shared_expert(mixed_m)
    shared_gate = jax.nn.sigmoid(layer.mlp.shared_expert_gate(mixed_m).astype(jnp.float32)).astype(shared_raw.dtype)
    sh_out = shared_gate * shared_raw
    b_mb, s_mb, _ = mixed_m.shape
    r_w_3d = r_w.reshape(b_mb, s_mb, layer.mlp.num_experts_per_tok)
    r_idx_3d = r_idx.reshape(b_mb, s_mb, layer.mlp.num_experts_per_tok)
    r_probs_3d = r_probs.reshape(b_mb, s_mb, layer.mlp.num_experts)
    if not return_aux:
      aux_half = None
    elif compact_router_aux:
      seg_ids = kw.get("segment_ids")
      r_mask = kw.get("router_aux_mask")
      eff_mask = r_mask if r_mask is not None else ((seg_ids != 0) if seg_ids is not None else None)
      aux_half = compact_aux_fn(r_probs_3d, r_idx_3d, mask=eff_mask)
    else:
      aux_half = {
          "router_indices": r_idx_3d,
          "routing_weights": r_w_3d,
          "router_probs": r_probs_3d,
          **({"selected_block_indices": s_idx} if s_idx is not None else {}),
      }
    return h_post_attn, mixed_m, inject_m, r_w, r_idx, sh_out, aux_half

  del group_base
  h0_post, mixed0, inject0, rw0, ridx0, sh0, aux0 = _run_attn_and_pre_moe(h0, kw0)
  (h0_post, mixed0, rw0, ridx0, sh0), h1 = jax.lax.optimization_barrier(((h0_post, mixed0, rw0, ridx0, sh0), h1))
  h1_post, mixed1, inject1, rw1, ridx1, sh1, aux1 = _run_attn_and_pre_moe(h1, kw1)
  routed0, routed1 = layer.mlp.experts.call_dual(mixed0, ridx0, rw0, mixed1, ridx1, rw1)
  routed0, (routed1, sh1) = jax.lax.optimization_barrier((routed0, (routed1, sh1)))
  mlp_out0 = (routed0 + sh0).reshape(mixed0.shape)
  mlp_out1 = (routed1 + sh1).reshape(mixed1.shape)
  h0_out = shard_batch_fn(layer.mlp_hyper_connection.combine(h0_post, mlp_out0, inject0), layer.sharding)
  h1_out = shard_batch_fn(layer.mlp_hyper_connection.combine(h1_post, mlp_out1, inject1), layer.sharding)

  if not return_aux:
    return (h0_out, h1_out), None
  assert aux0 is not None and aux1 is not None
  return (h0_out, h1_out), merge_dual_aux(aux0, aux1, compact_router_aux=compact_router_aux)
