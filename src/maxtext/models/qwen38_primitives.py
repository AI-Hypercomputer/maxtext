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

"""Qwen3.8-Flash-Next grouped norms, gated residuals, and PLE math.

N-gram hashing requires signed int64 arithmetic. Enable JAX x64 before
constructing model parameters; silently truncating the checkpoint buffers
to int32 changes lookup IDs.
"""

import math
import jax
import jax.numpy as jnp


def grouped_norm(x, weight, group_size, epsilon):
  """Apply checkpoint zero-centered RMSNorm independently to each group."""
  dtype = x.dtype
  grouped = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, group_size)
  grouped *= jax.lax.rsqrt(jnp.mean(grouped * grouped, axis=-1, keepdims=True) + epsilon)
  return (grouped.reshape(x.shape) * (1 + weight.astype(jnp.float32))).astype(dtype)


def linear(x, weight):
  """Apply an HF-oriented linear weight with highest matmul precision."""
  return jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST)


def gated_residual(x, weights, hidden_size, hc_count, epsilon):
  """Read normalized streams and compute per-stream residual injection gates."""
  normalized = grouped_norm(x, weights["hc_norm.weight"], hidden_size, epsilon)
  mix = jax.nn.silu(linear(normalized, weights["input_mix_weight_down.weight"]) / hc_count)
  mix = jax.nn.sigmoid(linear(mix, weights["input_mix_weight_up.weight"]))
  mixed = jnp.mean((mix * normalized).reshape(*x.shape[:-1], hc_count, hidden_size), axis=-2)
  if "block_inject_weight.weight" not in weights:
    return mixed
  inject = 2 * jax.nn.sigmoid(linear(normalized, weights["block_inject_weight.weight"]) / hc_count)
  return mixed, x, inject


def ngram_ids(input_ids, multipliers, vocab_sizes, offsets, eos_token_id, ngram_size, heads_per_ngram):
  """Hash raw token history with signed int64 overflow and EOS resets."""
  if not jax.config.x64_enabled:
    raise ValueError("Qwen3.8 PLE hash requires JAX_ENABLE_X64=true")
  ids = input_ids.astype(jnp.int64)
  shifts = [ids]
  for shift in range(1, ngram_size):
    shifted = jnp.pad(ids, ((0, 0), (shift, 0)), constant_values=eos_token_id)[:, : ids.shape[1]]
    valid = jnp.ones_like(ids, dtype=jnp.bool_)
    for previous in range(1, shift + 1):
      prev = jnp.pad(ids, ((0, 0), (previous, 0)), constant_values=eos_token_id)[:, : ids.shape[1]]
      valid &= prev != eos_token_id
    shifts.append(jnp.where(valid, shifted, eos_token_id))
  blocks = []
  mixed = ids * multipliers[0]
  for order in range(2, ngram_size + 1):
    mixed = jnp.bitwise_xor(mixed, shifts[order - 1] * multipliers[order - 1])
    lo = (order - 2) * heads_per_ngram
    hi = lo + heads_per_ngram
    blocks.append(jnp.remainder(mixed[..., None], vocab_sizes[lo:hi]) + offsets[lo:hi])
  return jnp.concatenate(blocks, axis=-1)


def ple_forward(
    x, embeddings, weights, hidden_size, hc_count, dilation, epsilon, conv_history=None, mask=None, return_history=False
):
  """Inject contextual n-gram values with a cached dilated convolution."""
  key = grouped_norm(linear(embeddings, weights["key_proj.weight"]), weights["norm_key.weight"], hidden_size, epsilon)
  query = grouped_norm(x, weights["norm_query.weight"], hidden_size, epsilon)
  shape = (*x.shape[:-1], hc_count, hidden_size)
  gate = jnp.sum(key.reshape(shape) * query.reshape(shape), axis=-1, keepdims=True) / math.sqrt(hidden_size)
  gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
  value = linear(embeddings, weights["value_proj.weight"])
  gated = (jax.nn.sigmoid(gate) * value[..., None, :]).reshape(x.shape)
  normalized = grouped_norm(gated, weights["norm_conv.weight"], hidden_size, epsilon)
  if mask is not None:
    gated *= mask[..., None]
    normalized *= mask[..., None]
  kernel = weights["conv1d.weight"][:, 0, :]
  history_size = (kernel.shape[-1] - 1) * dilation
  if conv_history is None:
    conv_history = jnp.zeros((x.shape[0], history_size, x.shape[-1]), normalized.dtype)
  combined = jnp.concatenate([conv_history, normalized], axis=1)
  convolved = jnp.zeros_like(combined)
  for tap in range(kernel.shape[-1]):
    shift = (kernel.shape[-1] - 1 - tap) * dilation
    padded = jnp.pad(combined, ((0, 0), (shift, 0), (0, 0)))[:, : combined.shape[1], :]
    convolved += padded * kernel[:, tap]
  output = gated + jax.nn.silu(convolved[:, history_size:])
  if return_history:
    lengths = jnp.full((x.shape[0],), x.shape[1], jnp.int32) if mask is None else jnp.sum(mask, axis=1).astype(jnp.int32)
    indices = lengths[:, None] + jnp.arange(history_size)[None, :]
    history = jnp.take_along_axis(combined, indices[..., None], axis=1)
    return output, history
  return output


def subtree(weights, prefix):
  """Extract weights under a reference module name."""
  return {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)}


def moe_forward(x, weights, cfg):
  """Compute routed and shared experts with bounded per-token gather memory."""
  shape = x.shape
  flat = x.reshape(-1, cfg.hidden_size)
  probabilities = jax.nn.softmax(linear(flat, weights["gate.weight"]).astype(jnp.float32), axis=-1)
  scores, indices = jax.lax.top_k(probabilities, cfg.num_experts_per_tok)
  if cfg.norm_topk_prob:
    scores /= jnp.sum(scores, axis=-1, keepdims=True)

  # Scan tokens to bound expert gather memory to top_k weights, rather than
  # [batch*sequence, top_k, intermediate, hidden].
  def token_fn(args):
    token, experts, routing = args
    gate_up = weights["experts.gate_up_proj"][experts]
    projections = jnp.einsum("eih,h->ei", gate_up, token, precision=jax.lax.Precision.HIGHEST)
    gate, up = jnp.split(projections, 2, axis=-1)
    expert_out = jnp.einsum(
        "ehi,ei->eh", weights["experts.down_proj"][experts], jax.nn.silu(gate) * up, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.sum(expert_out * routing[:, None], axis=0)

  output = jax.lax.map(token_fn, (flat, indices, scores.astype(flat.dtype)))
  shared = subtree(weights, "shared_expert.")
  intermediate = jax.nn.silu(linear(flat, shared["gate_proj.weight"])) * linear(flat, shared["up_proj.weight"])
  output += linear(intermediate, shared["down_proj.weight"]) * jax.nn.sigmoid(
      linear(flat, weights["shared_expert_gate.weight"])
  )
  return output.reshape(shape)


def gdn_forward(x, weights, cfg, state=None, mask=None, return_state=False):
  """Run causal convolution and gated delta recurrence, optionally caching state."""
  batch, length, _ = x.shape
  projected = linear(x, weights["in_proj_qkv.weight"])
  if mask is not None:
    projected *= mask[..., None]
  kernel = weights["conv1d.weight"][:, 0, :]
  history_size = kernel.shape[-1] - 1
  history = jnp.zeros((batch, history_size, projected.shape[-1]), projected.dtype) if state is None else state["conv"]
  combined = jnp.concatenate([history, projected], axis=1)
  convolved = jnp.zeros_like(combined)
  for tap in range(kernel.shape[-1]):
    shift = kernel.shape[-1] - 1 - tap
    convolved += jnp.pad(combined, ((0, 0), (shift, 0), (0, 0)))[:, : combined.shape[1]] * kernel[:, tap]
  convolved = jax.nn.silu(convolved[:, history_size:])
  kd = cfg.linear_num_key_heads * cfg.linear_key_head_dim
  q, k, v = jnp.split(convolved, [kd, 2 * kd], axis=-1)
  q = q.reshape(batch, length, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
  k = k.reshape(q.shape)
  v = v.reshape(batch, length, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
  repeats = cfg.linear_num_value_heads // cfg.linear_num_key_heads
  q = jnp.repeat(q, repeats, axis=2).astype(jnp.float32)
  k = jnp.repeat(k, repeats, axis=2).astype(jnp.float32)
  q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
  k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
  q /= math.sqrt(cfg.linear_key_head_dim)
  beta = jax.nn.sigmoid(linear(x, weights["in_proj_b.weight"])).astype(jnp.float32)
  decay = -jnp.exp(weights["A_log"].astype(jnp.float32)) * jax.nn.softplus(
      linear(x, weights["in_proj_a.weight"]).astype(jnp.float32) + weights["dt_bias"]
  )
  initial = (
      jnp.zeros((batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim), jnp.float32)
      if state is None
      else state["recurrent"]
  )
  active = jnp.ones((batch, length), jnp.bool_) if mask is None else mask.astype(jnp.bool_)

  def step(state, args):
    qt, kt, vt, bt, gt, valid = args
    old_state = state
    state = state * jnp.exp(gt)[..., None, None]
    delta = (vt - jnp.einsum("bhk,bhkv->bhv", kt, state)) * bt[..., None]
    state = state + kt[..., None] * delta[..., None, :]
    state = jnp.where(valid[:, None, None, None], state, old_state)
    return state, jnp.einsum("bhk,bhkv->bhv", qt, state)

  final_state, core = jax.lax.scan(step, initial, tuple(jnp.swapaxes(a, 0, 1) for a in (q, k, v, beta, decay, active)))
  core = jnp.swapaxes(core, 0, 1).astype(x.dtype)
  core = core * jax.lax.rsqrt(jnp.mean(core.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + cfg.rms_norm_eps)
  z = linear(x, weights["in_proj_z.weight"]).reshape(core.shape)
  gate = jax.nn.sigmoid(z) if cfg.output_gate_type == "sigmoid" else jax.nn.silu(z)
  core = core * weights["norm.weight"] * gate
  output = linear(core.reshape(batch, length, -1), weights["out_proj.weight"])
  if return_state:
    lengths = jnp.sum(active, axis=1).astype(jnp.int32)
    history = jnp.take_along_axis(combined, (lengths[:, None] + jnp.arange(history_size)[None, :])[..., None], axis=1)
    return output, {"conv": history, "recurrent": final_state}
  return output


def rotary(x, positions, cfg):
  """Apply partial text RoPE in the source dtype and checkpoint ordering."""
  dim = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0))
  inverse = 1 / (cfg.rope_parameters["rope_theta"] ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
  angles = positions.astype(jnp.float32)[..., None] * inverse
  angles = jnp.concatenate([angles, angles], axis=-1)
  a = x[..., :dim]
  rotated = jnp.concatenate([-a[..., dim // 2 :], a[..., : dim // 2]], axis=-1)
  return jnp.concatenate(
      [
          a * jnp.cos(angles).astype(x.dtype)[..., None, :] + rotated * jnp.sin(angles).astype(x.dtype)[..., None, :],
          x[..., dim:],
      ],
      axis=-1,
  )


def attention_forward(x, weights, cfg, positions=None, mask=None, state=None, return_state=False):
  """Run micro-block QSA using selected keys and an optional fixed-size cache."""
  batch, length, _ = x.shape
  if positions is None:
    positions = jnp.broadcast_to(jnp.arange(length), (batch, length))
  active = jnp.ones((batch, length), jnp.bool_) if mask is None else mask.astype(jnp.bool_)
  qgate = linear(x, weights["q_proj.weight"]).reshape(batch, length, cfg.num_attention_heads, 2 * cfg.head_dim)
  q, gate = jnp.split(qgate, 2, axis=-1)
  k = linear(x, weights["k_proj.weight"]).reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
  v = linear(x, weights["v_proj.weight"]).reshape(k.shape)
  q = rotary(grouped_norm(q, weights["q_norm.weight"], cfg.head_dim, cfg.rms_norm_eps), positions, cfg)
  k = rotary(grouped_norm(k, weights["k_norm.weight"], cfg.head_dim, cfg.rms_norm_eps), positions, cfg)
  index = subtree(weights, "indexer.")
  qk = linear(x, index["index_qk_proj.weight"])
  iq, raw = jnp.split(qk, [cfg.indexer_n_heads * cfg.indexer_head_dim], axis=-1)
  iq = iq.reshape(batch, length, cfg.indexer_n_heads, cfg.indexer_head_dim)
  iq = rotary(grouped_norm(iq, index["q_layernorm.weight"], cfg.indexer_head_dim, cfg.rms_norm_eps), positions, cfg)
  if state is not None:
    batch_indices = jnp.arange(batch)[:, None]
    write_positions = jnp.where(active, positions, state["key"].shape[1])
    keys = state["key"].at[batch_indices, write_positions].set(k, mode="drop")
    values = state["value"].at[batch_indices, write_positions].set(v, mode="drop")
    raw = state["index_key"].at[batch_indices, write_positions].set(raw, mode="drop")
    valid = state["valid"].at[batch_indices, write_positions].set(active, mode="drop")
    k, v = keys, values
  else:
    valid = active
  new_state = {"key": k, "value": v, "index_key": raw, "valid": valid}
  kv_length = k.shape[1]
  ratio = cfg.indexer_compress_ratio
  blocks = (kv_length + ratio - 1) // ratio
  padding = blocks * ratio - kv_length
  pooled = jnp.pad(raw.astype(jnp.float32), ((0, 0), (0, padding), (0, 0))).reshape(batch, blocks, ratio, -1).mean(axis=2)
  pooled = grouped_norm(pooled.astype(x.dtype), index["k_layernorm.weight"], cfg.indexer_head_dim, cfg.rms_norm_eps)
  block_positions = jnp.broadcast_to(jnp.arange(blocks) * ratio, (batch, blocks))
  pooled = rotary(pooled[:, :, None, :], block_positions, cfg).squeeze(2)
  index_scores = jnp.einsum(
      "bqhd,bnd->bqnh", iq.astype(jnp.float32), pooled.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
  )
  index_scores = jnp.sum(jax.nn.relu(index_scores), axis=-1) / math.sqrt(cfg.indexer_head_dim)
  complete = (jnp.arange(blocks)[None, None, :] + 1) * ratio <= (positions[:, :, None] + 1)
  block_valid = jnp.pad(valid, ((0, 0), (0, padding))).reshape(batch, blocks, ratio).all(axis=-1)
  complete &= block_valid[:, None, :]
  index_scores = jnp.where(complete, index_scores, -jnp.inf)
  _, chosen = jax.lax.top_k(index_scores, min(cfg.indexer_budget // ratio, blocks))
  selected_tokens = (chosen[..., None] * ratio + jnp.arange(ratio)).reshape(batch, length, -1)
  selected_valid = jnp.repeat(jnp.take_along_axis(complete, chosen, axis=-1), ratio, axis=-1)
  tail_start = ((positions + 1) // ratio) * ratio
  tail = tail_start[..., None] + jnp.arange(ratio)
  selected_tokens = jnp.concatenate([selected_tokens, tail], axis=-1)
  selected_valid = jnp.concatenate([selected_valid, jnp.ones_like(tail, dtype=jnp.bool_)], axis=-1)
  selected_valid &= (selected_tokens < kv_length) & (selected_tokens <= positions[..., None])
  safe_tokens = jnp.clip(selected_tokens, 0, kv_length - 1)
  selected_valid &= jnp.take_along_axis(
      jnp.broadcast_to(valid[:, None, :], (batch, length, kv_length)), safe_tokens, axis=-1
  )

  def attend_query(args):
    query, indices, visible, read_gate = args
    selected_k = jnp.take_along_axis(k, indices[..., None, None], axis=1)
    selected_v = jnp.take_along_axis(v, indices[..., None, None], axis=1)
    selected_k = jnp.repeat(selected_k, cfg.num_attention_heads // cfg.num_key_value_heads, axis=2)
    selected_v = jnp.repeat(selected_v, cfg.num_attention_heads // cfg.num_key_value_heads, axis=2)
    scores = jnp.einsum("bhd,bkhd->bhk", query, selected_k, precision=jax.lax.Precision.HIGHEST) / math.sqrt(cfg.head_dim)
    scores = jnp.where(visible[:, None, :], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    return jnp.einsum("bhk,bkhd->bhd", probs, selected_v, precision=jax.lax.Precision.HIGHEST) * jax.nn.sigmoid(read_gate)

  # Sequential query processing bounds gathered K/V memory for long prefills.
  output = jax.lax.map(attend_query, tuple(jnp.swapaxes(a, 0, 1) for a in (q, safe_tokens, selected_valid, gate)))
  output = jnp.swapaxes(output, 0, 1)
  output = linear(output.reshape(batch, length, -1), weights["o_proj.weight"])
  return (output, new_state) if return_state else output


def decoder_forward(x, ids, weights, cfg, layer_idx, ple_embeddings=None):
  """Compute an unscanned layer for contiguous text validation inputs."""
  if layer_idx + 1 in cfg.ple_layer_ids:
    ple = subtree(weights, "ple.")
    if ple_embeddings is None:
      indices = ngram_ids(
          ids,
          ple["ple_embedding.layer_multipliers"],
          ple["ple_embedding.ngram_heads_vocab_sizes"],
          ple["ple_embedding.ngram_heads_offsets"],
          cfg.eos_token_id,
          cfg.ngram_size,
          cfg.heads_per_ngram,
      )
      ple_embeddings = ple["ple_embedding.ngram_embedding.weight"][indices].reshape(*ids.shape, cfg.ple_embed_dim)
    x = x + ple_forward(x, ple_embeddings, ple, cfg.hidden_size, cfg.hc_count, cfg.ngram_size, cfg.rms_norm_eps)
  mixed, residual, inject = gated_residual(
      x, subtree(weights, "attn_hyper_connection."), cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps
  )
  if cfg.layer_types[layer_idx] == "linear_attention":
    result = gdn_forward(mixed, subtree(weights, "linear_attn."), cfg)
  else:
    result = attention_forward(mixed, subtree(weights, "self_attn."), cfg)
  x = residual + (result[..., None, :] * inject[..., None]).reshape(x.shape)
  mixed, residual, inject = gated_residual(
      x, subtree(weights, "mlp_hyper_connection."), cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps
  )
  result = moe_forward(mixed, subtree(weights, "mlp."), cfg)
  return residual + (result[..., None, :] * inject[..., None]).reshape(x.shape)


def decoder_forward_cached(x, ids, weights, cfg, layer_idx, cache, positions, mask, ple_lookup=None):
  """Compute a cached text step and return fixed-size recurrent and PLE state."""
  new_cache = dict(cache)
  if layer_idx + 1 in cfg.ple_layer_ids:
    ple = subtree(weights, "ple.")
    tokens = jnp.where(mask, ids, cfg.eos_token_id).astype(cache["tokens"].dtype)
    combined = jnp.concatenate([cache["tokens"], tokens], axis=1)
    indices = ngram_ids(
        combined,
        ple["ple_embedding.layer_multipliers"],
        ple["ple_embedding.ngram_heads_vocab_sizes"],
        ple["ple_embedding.ngram_heads_offsets"],
        cfg.eos_token_id,
        cfg.ngram_size,
        cfg.heads_per_ngram,
    )[:, 2:]
    embeddings = ple_lookup(indices).reshape(*ids.shape, cfg.ple_embed_dim)
    delta, history = ple_forward(
        x,
        embeddings,
        ple,
        cfg.hidden_size,
        cfg.hc_count,
        cfg.ngram_size,
        cfg.rms_norm_eps,
        conv_history=cache["ple_conv"],
        mask=mask,
        return_history=True,
    )
    x = x + delta
    lengths = jnp.sum(mask, axis=1).astype(jnp.int32)
    new_cache["tokens"] = jnp.take_along_axis(combined, lengths[:, None] + jnp.arange(2)[None, :], axis=1)
    new_cache["ple_conv"] = history
  mixed, residual, inject = gated_residual(
      x, subtree(weights, "attn_hyper_connection."), cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps
  )
  if cfg.layer_types[layer_idx] == "linear_attention":
    result, new_cache["attention"] = gdn_forward(
        mixed, subtree(weights, "linear_attn."), cfg, state=cache["attention"], mask=mask, return_state=True
    )
  else:
    result, new_cache["attention"] = attention_forward(
        mixed,
        subtree(weights, "self_attn."),
        cfg,
        positions=positions,
        mask=mask,
        state=cache["attention"],
        return_state=True,
    )
  x = residual + (result[..., None, :] * inject[..., None]).reshape(x.shape)
  mixed, residual, inject = gated_residual(
      x, subtree(weights, "mlp_hyper_connection."), cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps
  )
  result = moe_forward(mixed, subtree(weights, "mlp."), cfg)
  return residual + (result[..., None, :] * inject[..., None]).reshape(x.shape), new_cache
