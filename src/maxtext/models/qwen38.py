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

"""Unscanned Qwen3.8 text decoder for bringup correctness verification.

Weights retain HF orientation. PLE source shards remain separate leaves, so
conversion never concatenates the complete 51B-parameter table.
"""

from types import SimpleNamespace
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from maxtext.models.qwen38_primitives import decoder_forward, decoder_forward_cached, gated_residual, ngram_ids


def validate_contiguous_inputs(segments, positions, starts):
  """Reject unsupported packed/left-padded/offset inputs rather than miscompute."""

  segments = np.asarray(segments)
  positions = np.asarray(positions)
  starts = np.asarray(starts)
  if not np.isin(segments, [0, 1]).all():
    raise ValueError("Qwen3.8 bringup supports one unpacked segment per batch item")
  valid = segments > 0
  if np.any(valid[:, 1:] & ~valid[:, :-1]):
    raise ValueError("Qwen3.8 bringup requires contiguous right-padded inputs")
  expected = starts[:, None] + np.arange(positions.shape[1])[None, :]
  if np.any((positions != expected) & valid):
    raise ValueError("Qwen3.8 bringup requires contiguous text positions")


def validate_lookup_rows(found):
  # JAX callbacks receive CPU JAX arrays in newer versions. Dispatching a JAX
  # reduction inside a CPU callback can deadlock the backend; use NumPy only.
  if not np.asarray(found).all():
    raise ValueError("PLE sparse fixture is missing requested global row IDs")


def text_config(config):
  """Translate MaxText dimensions into the reference text architecture."""
  return SimpleNamespace(
      hidden_size=config.emb_dim,
      hc_count=config.qwen38_hc_count,
      hc_lowrank=config.qwen38_hc_lowrank,
      num_attention_heads=config.num_query_heads,
      num_key_value_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      rms_norm_eps=config.normalization_layer_epsilon,
      num_experts=config.num_experts,
      num_experts_per_tok=config.num_experts_per_tok,
      norm_topk_prob=config.norm_topk_prob,
      moe_intermediate_size=config.moe_mlp_dim,
      shared_expert_intermediate_size=config.moe_mlp_dim,
      linear_num_key_heads=config.gdn_num_key_heads,
      linear_num_value_heads=config.gdn_num_value_heads,
      linear_key_head_dim=config.gdn_key_head_dim,
      linear_value_head_dim=config.gdn_value_head_dim,
      linear_conv_kernel_dim=config.gdn_conv_kernel_dim,
      output_gate_type="sigmoid",
      layer_types=tuple(
          "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(config.num_decoder_layers)
      ),
      ple_layer_ids=(2,),
      ple_embed_dim=config.emb_dim,
      ngram_size=3,
      heads_per_ngram=8,
      eos_token_id=248044,
      indexer_budget=2048,
      indexer_n_heads=4,
      indexer_kv_heads=1,
      indexer_head_dim=128,
      indexer_compress_ratio=4,
      rope_parameters={"rope_theta": config.rope_max_timescale, "partial_rotary_factor": config.partial_rotary_factor},
  )


def hc_shapes(hidden, count, rank, combine=True):
  """Return raw checkpoint shapes for a gated residual or final mixer."""
  expanded = hidden * count
  shapes = {
      "hc_norm.weight": (expanded,),
      "input_mix_weight_down.weight": (rank, expanded),
      "input_mix_weight_up.weight": (expanded, rank),
  }
  if combine:
    shapes["block_inject_weight.weight"] = (count, expanded)
  return shapes


def layer_shapes(config, index):
  """Return checkpoint-oriented weight shapes for one unscanned layer."""
  cfg = text_config(config)
  h = cfg.hidden_size
  shapes = {}
  for branch in ["attn_hyper_connection", "mlp_hyper_connection"]:
    shapes.update({f"{branch}.{k}": v for k, v in hc_shapes(h, cfg.hc_count, cfg.hc_lowrank).items()})
  e = cfg.num_experts
  m = cfg.moe_intermediate_size
  shapes.update(
      {
          "mlp.gate.weight": (e, h),
          "mlp.experts.gate_up_proj": (e, 2 * m, h),
          "mlp.experts.down_proj": (e, h, m),
          "mlp.shared_expert.gate_proj.weight": (m, h),
          "mlp.shared_expert.up_proj.weight": (m, h),
          "mlp.shared_expert.down_proj.weight": (h, m),
          "mlp.shared_expert_gate.weight": (1, h),
      }
  )
  if (index + 1) % 4:
    k = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    attn = {
        "in_proj_qkv.weight": (2 * k + v, h),
        "in_proj_z.weight": (v, h),
        "in_proj_a.weight": (cfg.linear_num_value_heads, h),
        "in_proj_b.weight": (cfg.linear_num_value_heads, h),
        "conv1d.weight": (2 * k + v, 1, cfg.linear_conv_kernel_dim),
        "A_log": (cfg.linear_num_value_heads,),
        "dt_bias": (cfg.linear_num_value_heads,),
        "norm.weight": (cfg.linear_value_head_dim,),
        "out_proj.weight": (h, v),
    }
    shapes.update({f"linear_attn.{k}": v for k, v in attn.items()})
  else:
    d = cfg.head_dim
    attn = {
        "q_proj.weight": (2 * cfg.num_attention_heads * d, h),
        "k_proj.weight": (cfg.num_key_value_heads * d, h),
        "v_proj.weight": (cfg.num_key_value_heads * d, h),
        "o_proj.weight": (h, cfg.num_attention_heads * d),
        "q_norm.weight": (d,),
        "k_norm.weight": (d,),
        "indexer.index_qk_proj.weight": (640, h),
        "indexer.q_layernorm.weight": (128,),
        "indexer.k_layernorm.weight": (128,),
    }
    shapes.update({f"self_attn.{k}": v for k, v in attn.items()})
  if index == 1:
    expanded = h * cfg.hc_count
    shapes.update(
        {
            "ple.key_proj.weight": (expanded, h),
            "ple.value_proj.weight": (h, h),
            "ple.norm_key.weight": (expanded,),
            "ple.norm_query.weight": (expanded,),
            "ple.norm_conv.weight": (expanded,),
            "ple.conv1d.weight": (expanded, 1, 4),
            "ple.ple_embedding.layer_multipliers": (3,),
            "ple.ple_embedding.ngram_heads_vocab_sizes": (16,),
            "ple.ple_embedding.ngram_heads_offsets": (16,),
        }
    )
    if config.qwen38_ple_compact_rows:
      shapes["ple.ple_embedding.ngram_embedding.weight"] = (config.qwen38_ple_compact_rows, 160)
      shapes["ple.ple_embedding.row_ids"] = (config.qwen38_ple_compact_rows,)
    else:
      for shard in range(128):
        shapes[f"ple.ple_embedding.ngram_embedding.shard_{shard}.weight"] = (2500012, 160)
  return shapes


def is_integer_weight(name):
  """Identify exact PLE hash metadata and sparse global row identifiers."""
  return name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets", "row_ids"))


def cache_tree(values, batch_axis):
  """Wrap individual cache arrays with batch-axis metadata for MaxEngine."""
  return nnx.Dict(
      {
          k: cache_tree(v, batch_axis)
          if isinstance(v, dict)
          else nnx.Cache(v, sharding=(batch_axis,) + (None,) * (v.ndim - 1))
          for k, v in values.items()
      }
  )


def cache_values(tree):
  """Unbox the nested per-array NNX cache into a functional state dict."""
  return {k: cache_values(v) if isinstance(v, nnx.Dict) else v.get_value() for k, v in tree.items()}


def update_cache(tree, values):
  """Update fixed-shape per-array caches after a model step."""
  for key, value in values.items():
    if isinstance(value, dict):
      update_cache(tree[key], value)
    else:
      tree[key].set_value(value)


def weight_axes(name, shape):
  """Describe logical sharding axes for raw checkpoint-oriented weights."""
  if is_integer_weight(name):
    return (None,) * len(shape)
  if "ngram_embedding" in name:
    # Each original shard has 2,500,012 rows, indivisible by common tensor meshes.
    # Replicate rows and shard its 160 columns over the existing FSDP axis.
    return (None, "embed_vocab")
  if name == "mlp.experts.gate_up_proj":
    return ("exp", "mlp_moe", "embed_moe")
  if name == "mlp.experts.down_proj":
    return ("exp", "embed_moe", "mlp_moe")
  if len(shape) == 2:
    return ("mlp", "embed")
  return (None,) * len(shape)


def init_weight(name, shape, dtype, rngs):
  """Initialize norms and projections without zeroing the entire model."""
  if is_integer_weight(name):
    return jnp.zeros(shape, jnp.int64)
  if name == "linear_attn.norm.weight":
    return jnp.ones(shape, dtype)
  if ".norm" in name or name.startswith("hc_norm") or name.endswith(("A_log", "dt_bias")):
    return jnp.zeros(shape, dtype)
  return jax.random.normal(rngs.params(), shape, dtype) * 0.02


class Qwen38Mixer(nnx.Module):
  """Collapse the final hyper-connection streams with learned read gates."""

  def __init__(self, config, rngs, **kwargs):
    self.cfg = text_config(config)
    self.w = nnx.Dict(
        {
            k: nnx.Param(init_weight(k, v, config.weight_dtype, rngs), sharding=weight_axes(k, v))
            for k, v in hc_shapes(config.emb_dim, config.qwen38_hc_count, config.qwen38_hc_lowrank, False).items()
        }
    )

  def __call__(self, x, **kwargs):
    return gated_residual(
        x, {k: v.get_value() for k, v in self.w.items()}, self.cfg.hidden_size, self.cfg.hc_count, self.cfg.rms_norm_eps
    )


class Qwen38DecoderLayer(nnx.Module):
  """Unscanned text decoder with exact sparse fixtures and native caches."""

  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    if config.scan_layers:
      raise NotImplementedError("Qwen3.8 initial bringup requires scan_layers=false")
    if config.use_multimodal:
      raise NotImplementedError(
          "Qwen3.8 initial bringup validates the text backbone; multimodal input is not yet supported"
      )
    if not jax.config.x64_enabled:
      raise ValueError("Qwen3.8 PLE requires JAX_ENABLE_X64=true")
    self.cfg = text_config(config)
    self.layer_idx = layer_idx
    self.w = nnx.Dict(
        {
            k: nnx.Param(init_weight(k, shape, config.weight_dtype, rngs), sharding=weight_axes(k, shape))
            for k, shape in layer_shapes(config, layer_idx).items()
        }
    )
    if model_mode in ("prefill", "autoregressive"):
      batch = (
          1
          if model_mode == "prefill"
          else int(config.per_device_batch_size * (mesh.size if mesh is not None else jax.device_count()))
      )
      if (layer_idx + 1) % 4:
        channels = (
            2 * config.gdn_num_key_heads * config.gdn_key_head_dim
            + config.gdn_num_value_heads * config.gdn_value_head_dim
        )
        cache = {
            "attention": {
                "conv": jnp.zeros((batch, config.gdn_conv_kernel_dim - 1, channels), config.dtype),
                "recurrent": jnp.zeros(
                    (batch, config.gdn_num_value_heads, config.gdn_key_head_dim, config.gdn_value_head_dim), jnp.float32
                ),
            }
        }
      else:
        shape = (batch, config.max_target_length, config.num_kv_heads, config.head_dim)
        cache = {
            "attention": {
                "key": jnp.zeros(shape, config.dtype),
                "value": jnp.zeros(shape, config.dtype),
                "index_key": jnp.zeros((batch, config.max_target_length, 128), config.dtype),
                "valid": jnp.zeros((batch, config.max_target_length), jnp.bool_),
            }
        }
      if layer_idx == 1:
        cache.update(
            tokens=jnp.full((batch, 2), 248044, jnp.int32),
            ple_conv=jnp.zeros((batch, 9, config.emb_dim * config.qwen38_hc_count), config.dtype),
        )
      cache["count"] = jnp.zeros((batch,), jnp.int32)
      self.cache = cache_tree(cache, "cache_batch_prefill" if model_mode == "prefill" else "cache_batch")

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot=None,
      kv_cache=None,
      decoder_input_tokens=None,
      **kwargs,
  ):
    weights = {k: v.get_value() for k, v in self.w.items()}
    embeddings = None

    def lookup(ids):
      prefix = "ple.ple_embedding."
      if prefix + "row_ids" in weights:
        rows = weights[prefix + "row_ids"]
        positions = jnp.searchsorted(rows, ids)
        safe = jnp.minimum(positions, rows.shape[0] - 1)
        valid = (positions < rows.shape[0]) & (rows[safe] == ids)

        jax.debug.callback(validate_lookup_rows, valid)
        embeddings = weights[prefix + "ngram_embedding.weight"][safe]
      else:
        embeddings = jnp.zeros((*ids.shape, 160), inputs.dtype)
        for shard in range(128):
          local = ids - shard * 2500012
          values = weights[prefix + f"ngram_embedding.shard_{shard}.weight"][jnp.clip(local, 0, 2500011)]
          embeddings += jnp.where(((local >= 0) & (local < 2500012))[..., None], values, 0)
      return embeddings

    if model_mode in ("prefill", "autoregressive"):
      cache = cache_values(self.cache)
      if model_mode == "prefill":
        cache = jax.tree.map(jnp.zeros_like, cache)
        if self.layer_idx == 1:
          cache["tokens"] = jnp.full_like(cache["tokens"], 248044)
      mask = (
          jnp.ones_like(decoder_input_tokens, dtype=jnp.bool_) if decoder_segment_ids is None else decoder_segment_ids > 0
      )
      if self.layer_idx == 0:
        segments = mask.astype(jnp.int32) if decoder_segment_ids is None else decoder_segment_ids
        jax.debug.callback(validate_contiguous_inputs, segments, decoder_positions, cache["count"])
      result, new_cache = decoder_forward_cached(
          inputs, decoder_input_tokens, weights, self.cfg, self.layer_idx, cache, decoder_positions, mask, lookup
      )
      new_cache["count"] = cache["count"] + jnp.sum(mask, axis=1).astype(jnp.int32)
      update_cache(self.cache, new_cache)
      return result, None
    if self.layer_idx == 0:
      segments = jnp.ones_like(decoder_input_tokens) if decoder_segment_ids is None else decoder_segment_ids
      jax.debug.callback(
          validate_contiguous_inputs, segments, decoder_positions, jnp.zeros((inputs.shape[0],), jnp.int32)
      )
    if self.layer_idx == 1:
      ids = ngram_ids(
          decoder_input_tokens,
          weights["ple.ple_embedding.layer_multipliers"],
          weights["ple.ple_embedding.ngram_heads_vocab_sizes"],
          weights["ple.ple_embedding.ngram_heads_offsets"],
          self.cfg.eos_token_id,
          3,
          8,
      )
      embeddings = lookup(ids).reshape(*decoder_input_tokens.shape, self.cfg.ple_embed_dim)
    return decoder_forward(inputs, decoder_input_tokens, weights, self.cfg, self.layer_idx, embeddings), None
