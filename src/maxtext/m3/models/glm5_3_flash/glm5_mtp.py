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

"""Layer-45 Multi-Token Prediction (MTP) block for GLM-5.3-Flash.

Consumes post-final-norm backbone hidden states (norm(hc_head(streams))),
matching Megatron and DeepSeek-V3 NextN MTP specifications.
"""

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_dsa_mla
from maxtext.m3.models.glm5_3_flash import glm5_moe


class Glm5Mtp(nnx.Module):
  """Layer-45 Multi-Token Prediction (MTP) block for GLM-5.3-Flash.

  Matches SGLang Glm5NextForConditionalGenerationNextN and DeepSeek-V3
  NextN specifications, consuming post-final-norm backbone hidden
  states out.hidden_states (norm(hc_head(hidden_streams))).
  """

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sharding = sharding
    h = config.hidden_size
    eps = config.rms_norm_eps
    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)

    self.enorm = glm5_dsa_mla.Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.hnorm = glm5_dsa_mla.Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    eh_meta = {"sharding": sharding("eh_proj", ("embed", None))} if sharding else {}
    self.eh_proj = nnx.Linear(
        2 * h,
        h,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=eh_meta,
        rngs=rngs,
    )
    self.input_layernorm = glm5_dsa_mla.Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.self_attn = glm5_dsa_mla.Glm5DsaMlaAttention(
        config,
        layer_idx=glm5_config.MTP_LAYER_INDEX,
        indexer_type=config.mtp_indexer_type,
        sharding=sharding,
        rngs=rngs,
    )
    self.post_attention_layernorm = glm5_dsa_mla.Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.mlp = glm5_moe.Glm5MoE(config, sharding=sharding, rngs=rngs)
    self.shared_head_norm = glm5_dsa_mla.Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)

  def __call__(
      self,
      hidden_states: jax.Array,
      next_token_ids: jax.Array,
      embed_tokens: nnx.Embed,
      lm_head: nnx.Linear | None,
      decoder_segment_ids: jax.Array | None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes MTP logits and router logits for next-token target prediction.

    Args:
      hidden_states: Post-HyperHead mean collapse and post-final-norm hidden
        states out.hidden_states [B, S, H], matching Megatron and DeepSeek-V3
        NextN specifications.
      next_token_ids: Input tokens for the MTP step [B, S] (i.e. targets).
      embed_tokens: Shared token embedding module.
      lm_head: Shared linear projection to vocabulary logits, or None to return
        shared_head_norm(z) [B, S, H] for vocab tiling without materializing
        full logits.
      decoder_segment_ids: Optional sequence segmentation IDs [B, S].

    Returns:
      (mtp_out, router_logits, router_topk) tuple where mtp_out is mtp_logits
      of shape [B, S, V] if lm_head is provided, or normalized hidden states
      of shape [B, S, H] if lm_head is None. router_logits has shape
      [B * S, num_experts], and router_topk has shape [B, S, top_k].
    """
    emb = embed_tokens(next_token_ids)
    x = self.eh_proj(jnp.concatenate([self.enorm(emb), self.hnorm(hidden_states)], axis=-1))
    attn_out = self.self_attn(self.input_layernorm(x), decoder_segment_ids=decoder_segment_ids)
    if isinstance(attn_out, tuple):
      attn_out = attn_out[0]
    y = x + attn_out
    mlp_out, (router_logits, router_topk) = self.mlp(self.post_attention_layernorm(y))
    z = y + mlp_out
    normed_z = self.shared_head_norm(z)
    if lm_head is None:
      return normed_z, router_logits, router_topk
    head_dt = lm_head.kernel.dtype
    mtp_logits = lm_head(normed_z.astype(head_dt))
    return mtp_logits, router_logits, router_topk
