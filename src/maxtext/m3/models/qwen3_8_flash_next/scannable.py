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

"""Scannable heterogeneous 4-layer cycle block and layer definitions."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen3_8_flash_next import gdn, hyper_connection, moe, ple, qsa_attention  # pylint: disable=no-name-in-module


def build_ple_layer(config: Any, layer_idx: int = 1, *, rngs: nnx.Rngs) -> ple.Qwen3_8PLELayer | None:
  """Instantiates a Qwen3_8PLELayer if configured at layer_idx, else None."""
  if not (bool(config.use_ple_layer) and layer_idx == int(config.ple_layer_id)):
    return None
  c, dt, wdt = config, jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
  return ple.Qwen3_8PLELayer(
      eos_token_id=int(c.ple_eos_token_id),
      emb_dim=int(c.emb_dim),
      hc_count=int(c.hc_count),
      vocab_size=int(c.vocab_size),
      ple_embed_dim=int(c.ple_embed_dim),
      ngram_size=int(c.ngram_size),
      heads_per_ngram=int(c.heads_per_ngram),
      ngram_vocab_size_base=int(c.ngram_vocab_size_base),
      make_ngram_vocab_size_divisible_by=int(c.make_ngram_vocab_size_divisible_by),
      seed=int(c.ple_seed),
      ple_conv_kernel_size=int(c.ple_conv_kernel_size),
      eps=float(c.normalization_layer_epsilon),
      dtype=dt,
      weight_dtype=wdt,
      rngs=rngs,
  )


class Qwen3_8DecoderLayer(nnx.Module):
  """Single decoder layer with Hyper-Connection residuals and MoE."""

  def __init__(self, config: Any, layer_idx: int, *, include_ple: bool = True, rngs: nnx.Rngs):
    c, emb, dt, wdt = config, int(config.emb_dim), jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    ckw = {"eps": float(c.normalization_layer_epsilon), "dtype": dt, "weight_dtype": wdt, "rngs": rngs}
    hc_kw = {"emb_dim": emb, "hc_count": int(c.hc_count), "hc_lowrank": int(c.hc_lowrank), **ckw}
    cyc = int(c.inhomogeneous_layer_cycle_interval)
    self.layer_type = "linear_attention" if (layer_idx % cyc) != (cyc - 1) else "full_attention"
    self.attn_hyper_connection = hyper_connection.Qwen3_8GatedResidual(use_combine=True, **hc_kw)
    if self.layer_type == "linear_attention":
      self.linear_attn = gdn.Qwen3_8GatedDeltaNet(
          emb_dim=emb,
          num_k_heads=int(c.gdn_num_key_heads),
          num_v_heads=int(c.gdn_num_value_heads),
          **ckw,
          head_k_dim=int(c.gdn_key_head_dim),
          head_v_dim=int(c.gdn_value_head_dim),
          chunk_size=int(c.gdn_chunk_size),
          conv_kernel_size=int(c.gdn_conv_kernel_dim),
          output_gate_activation=str(c.gdn_output_gate_activation),
      )
    else:
      self.self_attn = qsa_attention.Qwen3_8SparseAttention(
          emb_dim=emb,
          num_query_heads=int(c.num_query_heads),
          num_kv_heads=int(c.num_kv_heads),
          head_dim=int(c.head_dim),
          partial_rotary_factor=float(c.partial_rotary_factor),
          rope_theta=float(c.rope_max_timescale),
          **ckw,
          mrope_section=tuple(c.mrope_section),
          index_n_heads=int(c.index_n_heads),
          index_kv_heads=int(c.index_kv_heads),
          index_head_dim=int(c.index_head_dim),
          token_budget=int(c.index_select_budget),
          compress_ratio=int(c.index_select_block_size),
      )
    self.mlp_hyper_connection = hyper_connection.Qwen3_8GatedResidual(use_combine=True, **hc_kw)
    self.mlp = moe.Qwen3_8SparseMoeBlock(
        emb_dim=emb,
        shared_expert_intermediate_size=int(c.mlp_dim),
        moe_mlp_dim=int(c.moe_mlp_dim),
        routed_scaling_factor=float(c.routed_scaling_factor),
        num_experts=int(c.num_experts),
        rngs=rngs,
        num_experts_per_tok=int(c.num_experts_per_tok),
        norm_topk_prob=bool(c.norm_topk_prob),
        dtype=dt,
        weight_dtype=wdt,
    )
    self.ple = build_ple_layer(c, layer_idx, rngs=rngs) if include_ple else None

  def __call__(
      self,
      hidden_states: jax.Array,
      *,
      positions: jax.Array | None = None,
      conv_mask: jax.Array | None = None,
      ple_input_ids: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_aux: bool = False,
  ) -> Any:
    """Executes single decoder layer forward pass."""
    if self.ple is not None and ple_input_ids is not None:
      hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, conv_mask=conv_mask)
    mixed, inject = self.attn_hyper_connection.before_block(hidden_states)
    if self.layer_type == "linear_attention":
      attn_out, s_idx = self.linear_attn(mixed, attention_mask=conv_mask), None
    else:
      attn_out, s_idx = self.self_attn(
          mixed, position_ids=positions, attention_mask=conv_mask, segment_ids=segment_ids, return_selected_blocks=True
      )
    hidden_states = self.attn_hyper_connection.combine(hidden_states, attn_out, inject)
    mixed, inject = self.mlp_hyper_connection.before_block(hidden_states)
    mlp_out, r_w, r_idx, r_probs = self.mlp(mixed, return_router_topk=True)
    hidden_states = self.mlp_hyper_connection.combine(hidden_states, mlp_out, inject)
    aux = {
        "router_indices": r_idx,
        "routing_weights": r_w,
        "router_probs": r_probs,
        **({"selected_block_indices": s_idx} if s_idx is not None else {}),
    }
    return (hidden_states, aux) if return_aux else hidden_states


class Qwen3_8CycleBlock(nnx.Module):
  """4-layer cycle block composed of 3 GDN layers and 1 QSA layer."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    self.gdn_0, self.gdn_1, self.gdn_2, self.qsa_3 = (
        Qwen3_8DecoderLayer(config, i, include_ple=False, rngs=rngs) for i in range(4)
    )


def create_scanned_cycles(config: Any, num_cycles: int, *, rngs: nnx.Rngs) -> Qwen3_8CycleBlock:
  """Creates scanned cycle block with stacked parameters along cycle axis."""
  if int(config.inhomogeneous_layer_cycle_interval) != 4:
    raise ValueError(
        f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
        f"{config.inhomogeneous_layer_cycle_interval}."
    )
  if int(config.num_decoder_layers) % 4 != 0:
    raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {config.num_decoder_layers}.")
  if bool(config.use_ple_layer) and int(config.ple_layer_id) != 1:
    raise ValueError(f"Scanned cycles only support ple_layer_id == 1, got {config.ple_layer_id}.")
  sp = [nnx.split(Qwen3_8CycleBlock(config, rngs=rngs), nnx.Param, ...) for _ in range(num_cycles)]
  return nnx.merge(sp[0][0], *(jax.tree.map(lambda *x: jnp.stack(x, 0), *(s[i] for s in sp)) for i in (1, 2)))


def apply_scanned_cycles(
    cycles: Qwen3_8CycleBlock,
    ple_layer: ple.Qwen3_8PLELayer | None,
    hidden_states: jax.Array,
    *,
    num_cycles: int,
    positions: jax.Array | None = None,
    conv_mask: jax.Array | None = None,
    ple_input_ids: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    return_aux: bool = False,
) -> Any:
  """Executes scan over stacked cycle weights with conditional PLE at cycle 0."""
  c_def, c_params, c_rest = nnx.split(cycles, nnx.Param, ...)
  p_def, p_state = nnx.split(ple_layer) if ple_layer is not None else (None, None)

  def ple_step(h, cyc_idx):
    if p_def is not None and ple_input_ids is not None:
      return jax.lax.cond(
          cyc_idx == 0, lambda x: x + nnx.merge(p_def, p_state)(x, ple_input_ids, conv_mask=conv_mask), lambda x: x, h
      )
    return h

  def scan_fn(carry, xs):
    blk, kw = nnx.merge(c_def, xs[0], xs[1]), {"positions": positions, "conv_mask": conv_mask, "segment_ids": segment_ids}
    if return_aux:
      h, a0 = blk.gdn_0(carry, return_aux=True, **kw)
      h, a1 = blk.gdn_1(ple_step(h, xs[2]), return_aux=True, **kw)
      h, a2 = blk.gdn_2(h, return_aux=True, **kw)
      h, a3 = blk.qsa_3(h, return_aux=True, **kw)
      return h, (a0, a1, a2, a3)
    h = ple_step(blk.gdn_0(carry, **kw), xs[2])
    return blk.qsa_3(blk.gdn_2(blk.gdn_1(h, **kw), **kw), **kw), None

  res = jax.lax.scan(scan_fn, hidden_states, (c_params, c_rest, jnp.arange(num_cycles)))
  if return_aux:
    return res[0], [jax.tree.map(lambda x, c=c: x[c], res[1][k]) for c in range(num_cycles) for k in range(4)]
  return res[0]


def copy_unscanned_to_scanned(unscanned_decoder: Any, scanned_decoder: Any) -> None:
  """Copies and stacks weights from unscanned decoder layers into scanned cycles."""
  u, s = getattr(unscanned_decoder, "decoder", unscanned_decoder), getattr(scanned_decoder, "decoder", scanned_decoder)
  if hasattr(unscanned_decoder, "lm_head") and hasattr(scanned_decoder, "lm_head"):
    nnx.update(scanned_decoder.lm_head, nnx.state(unscanned_decoder.lm_head))

  def get_sub(l, a):
    return (l.attn_hyper_connection, getattr(l, a), l.mlp_hyper_connection, l.mlp)

  for k, name in enumerate(["gdn_0", "gdn_1", "gdn_2", "qsa_3"]):
    a = "linear_attn" if k < 3 else "self_attn"
    st = [nnx.state(get_sub(getattr(u, f"layers_{4 * c + k}"), a)) for c in range(s.num_cycles)]
    nnx.update(get_sub(getattr(s.cycles, name), a), jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *st))
  for m in ("token_embedder", "hyper_connection_mixer"):
    nnx.update(getattr(s, m), nnx.state(getattr(u, m)))
  u_ple, s_ple = getattr(getattr(u, "layers_1", None), "ple", None), getattr(s, "ple", None)
  if (u_ple is not None) != (s_ple is not None):
    raise ValueError(f"PLE presence mismatch: unscanned={u_ple is not None}, scanned={s_ple is not None}.")
  if u_ple is not None:
    nnx.update(s_ple, nnx.state(u_ple))
