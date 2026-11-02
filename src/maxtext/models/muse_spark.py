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

"""Muse Spark model layers for MaxText."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config, MODEL_MODE_TRAIN
from maxtext.layers import attentions
from maxtext.layers import linears
from maxtext.layers import normalizations
from maxtext.layers.quantizations import AqtQuantization as Quant

_RESIDUAL_GATE_EPS, _NOPE_LAYER_INTERVAL = 1e-3, 4


class MuseSparkResidualGate(nnx.Module):
  """Pure-NNX high-precision residual gate."""

  def __init__(
      self,
      num_features: int,
      temperature: float = 0.3,
      eps: float = _RESIDUAL_GATE_EPS,
      weight_dtype: Any = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self.num_features, self.temperature, self.eps = (num_features, temperature, eps)
    hp_dtype = jnp.promote_types(weight_dtype, jnp.float32)
    self.gate = nnx.Param(jax.nn.initializers.zeros(rngs.params(), (num_features,), hp_dtype), out_sharding=("norm",))

  def __call__(self, residual: jax.Array, sublayer: jax.Array) -> jax.Array:
    compute_dtype = jnp.promote_types(residual.dtype, jnp.float32)
    w = jnp.asarray(self.gate.get_value(), compute_dtype) / self.temperature
    beta = jax.nn.sigmoid(w)
    alpha = jnp.sqrt(jnp.maximum(self.eps, jax.nn.sigmoid(-w) * (1.0 + beta)))
    out = alpha * jnp.asarray(residual, compute_dtype) + beta * jnp.asarray(sublayer, compute_dtype)
    return jnp.asarray(out, residual.dtype)


class MuseSparkTextMLP(nnx.Module):
  """Muse Spark SwiGLU MLP with wi_0, wi_1, wo."""

  def __init__(self, config: Config, mesh: Mesh | None = None, quant: Quant | None = None, *, rngs: nnx.Rngs) -> None:
    dkw = {
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "shard_mode": config.shard_mode,
        "quant": quant,
        "mesh": mesh,
        "use_bias": False,
        "matmul_precision": config.matmul_precision,
        "parameter_memory_host_offload": config.parameter_memory_host_offload,
    }
    self.wi_0 = linears.DenseGeneral(config.emb_dim, config.mlp_dim, kernel_axes=("embed", "mlp"), rngs=rngs, **dkw)
    self.wi_1 = linears.DenseGeneral(config.emb_dim, config.mlp_dim, kernel_axes=("embed", "mlp"), rngs=rngs, **dkw)
    self.wo = linears.DenseGeneral(config.mlp_dim, config.emb_dim, kernel_axes=("mlp", "embed"), rngs=rngs, **dkw)

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.wo(jax.nn.silu(self.wi_0(x)) * self.wi_1(x))


class MuseSparkDecoderLayer(nnx.Module):
  """Muse Spark decoder layer compatible with NNXDecoder sequential and scanned callers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      layer_idx: int,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: Quant | None = None,
      rngs: nnx.Rngs,
  ) -> None:
    self.config, self.mesh, self.layer_idx, self.quant = (config, mesh, layer_idx, quant)
    hp_dtype, temp = jnp.promote_types(config.weight_dtype, jnp.float32), config.residual_gate_temperature
    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "shard_mode": config.shard_mode,
        "parameter_memory_host_offload": config.parameter_memory_host_offload,
        "rngs": rngs,
    }
    self.input_layernorm = normalizations.CenteredRMSNorm(config.emb_dim, **norm_kw)
    is_nope = (config.num_decoder_layers - 1 - layer_idx) % _NOPE_LAYER_INTERVAL == 0
    attn_kw = {
        "config": config,
        "num_query_heads": config.num_query_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "max_target_length": config.max_target_length,
        "mesh": mesh,
        "attention_kernel": config.attention,
        "quant": quant,
        "inputs_q_shape": (1, 1, config.emb_dim),
        "inputs_kv_shape": (1, 1, config.emb_dim),
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "is_nope_layer": is_nope,
        "model_mode": model_mode,
        "rngs": rngs,
    }
    self.self_attention = attentions.Attention(**attn_kw)
    self.post_attention_layernorm = normalizations.MuseSparkRMSNorm(
        config.emb_dim, **{**norm_kw, "epsilon": config.post_norm_eps, "with_scale": False}
    )
    gate_kw = {"temperature": temp, "eps": _RESIDUAL_GATE_EPS, "weight_dtype": hp_dtype, "rngs": rngs}
    self.post_attn_gate = MuseSparkResidualGate(config.emb_dim, **gate_kw)
    self.pre_ffn_norm = normalizations.CenteredRMSNorm(config.emb_dim, **norm_kw)
    mlp = MuseSparkTextMLP(config=config, mesh=mesh, quant=quant, rngs=rngs)
    self.is_dense_layer = layer_idx < config.first_num_dense_layers
    if self.is_dense_layer:
      self.mlp = mlp
    else:
      self.moe_block = mlp
    self.post_ffn_norm = normalizations.InputScaledRMSNorm(config.emb_dim, **{**norm_kw, "epsilon": config.post_norm_eps})
    self.post_ffn_gate = MuseSparkResidualGate(config.emb_dim, **gate_kw)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      kv_cache: jax.Array | None = None,
      **kwargs: Any,
  ) -> tuple[jax.Array, jax.Array | None]:
    hidden = self.input_layernorm(inputs)
    attn_out, kv_cache = self.self_attention(
        hidden,
        hidden,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
    )
    hidden = self.post_attn_gate(inputs, self.post_attention_layernorm(attn_out))
    ffn = (self.mlp if self.is_dense_layer else self.moe_block)(self.pre_ffn_norm(hidden))
    return self.post_ffn_gate(hidden, self.post_ffn_norm(ffn)), kv_cache
