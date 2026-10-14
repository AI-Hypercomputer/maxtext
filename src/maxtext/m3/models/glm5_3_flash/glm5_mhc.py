# Copyright 2023-2026 Google LLC
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

"""Manifold-Constrained Hyper-Connections (mHC) and Decoder Layer for GLM-5.3-Flash."""

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_dsa_mla
from maxtext.m3.models.glm5_3_flash import glm5_kda
from maxtext.m3.models.glm5_3_flash import glm5_moe


class Glm5UnweightedRMSNorm(nnx.Module):
  """Unweighted RMSNorm: x * rsqrt(mean(x^2) + eps)."""

  def __init__(self, eps: float):
    self.eps = eps

  def __call__(self, x: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(v + self.eps)).astype(x.dtype)


class Glm5HyperConnection(nnx.Module):
  """Manifold-Constrained Hyper-Connections (mHC) with 20 Sinkhorn iterations."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    del rngs
    self.config, self.hc_mult = config, config.mhc_expansion_rate
    self.hc_sinkhorn_iters = config.mhc_sinkhorn_iters
    self.input_norm = Glm5UnweightedRMSNorm(eps=config.rms_norm_eps)

    p_dt = getattr(jnp, config.weight_dtype)
    mix = (2 + self.hc_mult) * self.hc_mult
    self.fn = nnx.Param(
        jnp.zeros((self.hc_mult * config.hidden_size, mix), dtype=p_dt),
        sharding=sharding("mhc_fn", ("embed", None)) if sharding else None,
    )
    self.base = nnx.Param(
        jnp.zeros((mix,), dtype=p_dt),
        sharding=sharding("mhc_base", (None,)) if sharding else None,
    )
    self.scale = nnx.Param(
        jnp.zeros((3,), dtype=p_dt),
        sharding=sharding("mhc_scale", (None,)) if sharding else None,
    )

  def __call__(self, hidden_streams: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    hc, eps = self.hc_mult, self.config.mhc_eps
    b, s = hidden_streams.shape[:2]
    flat = self.input_norm(hidden_streams.reshape(b, s, -1).astype(jnp.float32))
    fn_w = self.fn[...].astype(jnp.float32)
    pre_w, post_w, comb_w = jnp.split(jnp.matmul(flat, fn_w).astype(jnp.float32), [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = jnp.split(self.base[...].astype(jnp.float32), [hc, 2 * hc], axis=-1)
    pre_s, post_s, comb_s = self.scale[...].astype(jnp.float32)

    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.reshape(b, s, hc, hc) * comb_s + comb_b.reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
      for ax in (-1, -2):
        comb = comb / (jnp.sum(comb, axis=ax, keepdims=True) + eps)

    col = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2).astype(hidden_streams.dtype)
    return post, comb, col

  def apply_residual(
      self,
      residual: jax.Array,
      sublayer_out: jax.Array,
      post: jax.Array,
      comb: jax.Array,
  ) -> jax.Array:
    dt = residual.dtype
    branch1 = post.astype(dt)[..., None] * sublayer_out[:, :, None, :]
    return branch1 + jnp.matmul(jnp.swapaxes(comb.astype(dt), -1, -2), residual)


class Glm5HyperHead(nnx.Module):
  """Final GLM-5.3-Flash HC-stream collapse via unweighted mean across streams."""

  def __call__(self, hidden_streams: jax.Array) -> jax.Array:
    return jnp.mean(hidden_streams, axis=2)


class Glm5DecoderLayer(nnx.Module):
  """GLM-5.3-Flash Transformer block with 4-stream mHC and heterogeneous attn/mlp."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      layer_idx: int,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if not 0 <= layer_idx < config.num_hidden_layers:
      raise ValueError(f"layer_idx {layer_idx} out of range [0, {config.num_hidden_layers}).")
    self.config, self.layer_idx = config, layer_idx
    if config.layer_types is None or config.indexer_types is None:
      raise ValueError("config.layer_types and config.indexer_types must not be None.")
    block_type = config.layer_types[layer_idx]
    self.block_type = block_type
    p_dt = getattr(jnp, config.weight_dtype)

    if block_type == "linear_attention":
      self.self_attn = glm5_kda.Glm5KdaAttention(config, sharding=sharding, rngs=rngs)
    elif block_type == "full_attention":
      self.self_attn = glm5_dsa_mla.Glm5DsaMlaAttention(
          config,
          layer_idx,
          indexer_type=config.indexer_types[layer_idx],
          sharding=sharding,
          rngs=rngs,
      )
    else:
      raise ValueError(f"Unknown block_type: {block_type}")

    self.is_moe = layer_idx >= config.first_k_dense_replace
    self.mlp = (
        glm5_moe.Glm5MoE(config, sharding=sharding, rngs=rngs)
        if self.is_moe
        else glm5_moe.Glm5MLP(config, config.intermediate_size, sharding=sharding, rngs=rngs)
    )

    def _norm() -> glm5_dsa_mla.Glm5RMSNorm:
      return glm5_dsa_mla.Glm5RMSNorm(
          config.hidden_size,
          eps=config.rms_norm_eps,
          param_dtype=p_dt,
          sharding=sharding,
      )

    self.input_layernorm, self.post_attention_layernorm = _norm(), _norm()
    self.attn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)
    self.ffn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)

  def __call__(
      self,
      hidden_streams: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array | None]:
    p, c, col = self.attn_hc(hidden_streams)
    a_out = self.self_attn(self.input_layernorm(col), decoder_segment_ids=decoder_segment_ids)
    hidden_streams = self.attn_hc.apply_residual(hidden_streams, a_out, p, c)

    p, c, col = self.ffn_hc(hidden_streams)
    normed_mlp_in = self.post_attention_layernorm(col)
    if isinstance(self.mlp, glm5_moe.Glm5MoE):
      m_out, r_logits = self.mlp(normed_mlp_in)
    else:
      m_out, r_logits = self.mlp(normed_mlp_in), None
    return self.ffn_hc.apply_residual(hidden_streams, m_out, p, c), r_logits
