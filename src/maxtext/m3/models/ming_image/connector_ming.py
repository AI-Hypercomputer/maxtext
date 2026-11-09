# Copyright 2023–2026 Google LLC
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

"""Ming-Image condition head: the Qwen2 connector and the learnable-token / directvlm projections (Flax NNX).

Port of the tail of `BailingMM2NativeForConditionalGeneration.get_condition_embeds_for_image_gen` (Ming-Image
`modeling_bailingmm2.py`) for text-to-image, `img_gen_scales=[16]`:

1. `MingConditioner.query_tokens` (`[256, 2048]`) replace the word embeddings of the 256 `<image_patch>` tokens that
   `build_t2i_inputs` appends after the prompt (`image_mask`); the LLM (`MingTextEncoder`) routes them by
   `mlp.image_gate`.
2. directvlm: LLM `hidden_states[5]`, `[12]`, `[20]` (`[20]` = after the final norm) are concatenated (6144),
   multiplied by `prompt_mask` and projected by `proj_directvlm` (diffusers RMSNorm eps 1e-5 + Linear 6144 -> 3840)
   into `cap_feats_2 [B, L, 3840]`. The reference truncates the sequence to `directvlm_max_valid_ind` = the longest
   prompt in the batch (and feeds the DiT all those positions, unmasked: positions past a shorter sample's prompt are
   `proj_directvlm(0)` = the bias); here the length is static (`L`) and rows outside the prompt are likewise the bias.
   Keep the first `max(prompt_mask.sum(-1))` rows (for B = 1: exactly the prompt tokens) to match the reference.
3. learnable tokens: the final-normed LLM hidden state at `gen_mask` (`[B, 256, 2048]`) -> `proj_in` (-> 1536) ->
   `MingConnector` (Qwen2, bidirectional, no mask) -> its final-normed last hidden state -> `proj_out` (-> 2560)
   = `cap_feats [B, 256, 2560]` (no L2 normalization: `connector_norm` is false).

`get_condition_embeds` runs the whole chain.
"""

from collections.abc import Sequence
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.sharding import Sharding
from maxtext.m3.models.ming_image.text_encoder_ming import (
    BATCH,
    EMBED,
    LENGTH,
    MingMLP,
    MingRMSNorm,
    _linear,
    attention_mask_4d,
    default_inv_freq,
    dot_product_attention,
    rotate_half,
)
from maxtext.m3.models.wan.utils_wan import constrain, param_metadata

# NNX modules take model-specific `__call__` arguments; architecture kwargs mirror the HF configs.
# pylint: disable=arguments-differ,too-many-positional-arguments

# ==============================================================================
# Qwen2 connector
# ==============================================================================


class MingConnectorAttention(nnx.Module):
  """`Qwen2Attention` without causal masking: biased q/k/v, unbiased o_proj, full-head RoPE, GQA."""

  def __init__(self, hidden, num_heads, num_kv_heads, head_dim, *, dtype, weights_dtype, precision, rngs, sharding):
    self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
    self.precision = precision
    self.sharding = sharding
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.q_proj = _linear(hidden, num_heads * head_dim, "attn_q", ("embed", "heads"), use_bias=True, **kw)
    self.k_proj = _linear(hidden, num_kv_heads * head_dim, "attn_k", ("embed", "kv_heads"), use_bias=True, **kw)
    self.v_proj = _linear(hidden, num_kv_heads * head_dim, "attn_v", ("embed", "kv_heads"), use_bias=True, **kw)
    self.o_proj = _linear(num_heads * head_dim, hidden, "attn_out", ("heads", "embed"), use_bias=False, **kw)

  def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array) -> jax.Array:
    batch, length, _ = x.shape
    query = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
    key = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
    value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
    query = query * cos[:, :, None] + rotate_half(query) * sin[:, :, None]
    key = key * cos[:, :, None] + rotate_half(key) * sin[:, :, None]
    out = dot_product_attention(query, key, value, mask, precision=self.precision)
    return self.o_proj(out.reshape(batch, length, self.num_heads * self.head_dim))


class MingConnectorLayer(nnx.Module):
  """`Qwen2DecoderLayer`: pre-norm self-attention and SwiGLU MLP residual blocks."""

  def __init__(self, cfg: dict, *, dtype, weights_dtype, precision, rngs, sharding):
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.sharding = sharding
    self.self_attn = MingConnectorAttention(
        cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], **kw
    )
    self.mlp = MingMLP(cfg["hidden_size"], cfg["intermediate_size"], **kw)
    self.input_layernorm = MingRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"], weights_dtype, sharding)
    self.post_attention_layernorm = MingRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"], weights_dtype, sharding)

  def __call__(self, x, cos, sin, mask):
    x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
    x = constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))
    x = x + self.mlp(self.post_attention_layernorm(x))
    return constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))


class MingConnector(nnx.Module):
  """Flax NNX port of the Ming-Image connector: `Qwen2Model` run bidirectionally on `inputs_embeds`.

  The reference loads `connector/` as `Qwen2ForCausalLM`, sets every `self_attn.is_causal = False` and calls it with
  `inputs_embeds` and an all-ones 4D additive mask (a no-op), so attention is full and unmasked; it uses
  `hidden_states[-1]`, i.e. the output after the final `model.norm`. `embed_tokens` / `lm_head` are unused and not
  ported. Constructor arguments are the `connector/config.json` (Qwen2Config) keys. Param paths mirror HF
  `model.<path>`.

  Sharding: parameter axes `embed`, `heads` (12 q heads), `kv_heads` (2), `mlp` (8960), `norm`; activation axes
  `activation_batch`, `activation_length`, `activation_embed`, `activation_mlp` (no head-axis activation constraint,
  since 12 heads do not divide 8 devices).
  """

  def __init__(
      self,
      hidden_size: int = 1536,
      intermediate_size: int = 8960,
      num_hidden_layers: int = 28,
      num_attention_heads: int = 12,
      num_key_value_heads: int = 2,
      head_dim: Optional[int] = None,
      rms_norm_eps: float = 1e-6,
      rope_theta: float = 1e6,
      hidden_act: str = "silu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    if hidden_act != "silu":
      raise NotImplementedError(f"Only silu is supported, got {hidden_act!r}.")
    self.dtype = jnp.dtype(dtype)
    self.sharding = sharding
    weights_dtype = jnp.dtype(weights_dtype)
    self.head_dim = head_dim or hidden_size // num_attention_heads
    self.rope_theta = float(rope_theta)
    cfg = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": self.head_dim,
        "rms_norm_eps": rms_norm_eps,
    }
    kw = {"dtype": self.dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs}
    self.layers = nnx.List([MingConnectorLayer(cfg, sharding=sharding, **kw) for _ in range(num_hidden_layers)])
    self.norm = MingRMSNorm(hidden_size, rms_norm_eps, weights_dtype, sharding)

  def __call__(self, inputs_embeds: jax.Array, attention_mask: Optional[jax.Array] = None) -> jax.Array:
    """`inputs_embeds [B, L, hidden]` -> final-normed last hidden state `[B, L, hidden]` (bidirectional attention).

    `attention_mask [B, L]` optionally masks padded keys (the reference passes none).
    """
    batch, length, _ = inputs_embeds.shape
    x = constrain(self.sharding, inputs_embeds.astype(self.dtype), "residual", (BATCH, LENGTH, EMBED))
    inv_freq = default_inv_freq(self.rope_theta, self.head_dim)
    angles = np.arange(length, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = np.concatenate([angles, angles], axis=-1)[None]
    cos, sin = jnp.asarray(np.cos(angles), self.dtype), jnp.asarray(np.sin(angles), self.dtype)
    if attention_mask is None:
      attention_mask = jnp.ones((batch, length), jnp.bool_)
    mask = attention_mask_4d(attention_mask, causal=False)
    for layer in self.layers:
      x = layer(x, cos, sin, mask)
    return self.norm(x)


# ==============================================================================
# Conditioning head
# ==============================================================================


class MingDiffusersRMSNorm(nnx.Module):
  """`diffusers.models.normalization.RMSNorm` (with weight): fp32 statistics; cast to a half-precision weight dtype."""

  def __init__(self, dim: int, eps: float, weights_dtype=jnp.float32, sharding: Sharding | None = None):
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((dim,), weights_dtype), **param_metadata(sharding, "norm_scale", ("norm",)))

  def __call__(self, x: jax.Array) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + self.eps)
    weight = self.weight[...]
    if weight.dtype in (jnp.float16, jnp.bfloat16):
      x = x.astype(weight.dtype)
    return x * weight


class MingConditioner(nnx.Module):
  """The Ming-Image condition head (`mlp/model.safetensors`): query tokens, `proj_in` / `proj_out`, `proj_directvlm`.

  Constructor arguments are the `mlp/config.json` keys (DiT-side keys `use_identity_mlp` / `text_encoder_norm` are
  accepted and ignored) plus `llm_hidden_size` / `connector_hidden_size`. All params are replicated under sharding
  (axes `None` / `norm`).
  """

  def __init__(
      self,
      img_gen_scales: Sequence[int] = (16,),
      diffusion_c_input_dim: int = 2560,
      diffusion_inner_dim: int = 3840,
      selected_hidden_states_layers: Optional[Sequence[int]] = (5, 12, 20),
      use_vlm_directvlm_condition: bool = True,
      use_learnable_token_condition: bool = True,
      connector_norm: bool = False,
      use_identity_mlp: bool = True,
      text_encoder_norm: bool = False,
      llm_hidden_size: int = 2048,
      connector_hidden_size: int = 1536,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    del use_identity_mlp, text_encoder_norm  # Consumed by the DiT side.
    self.dtype = jnp.dtype(dtype)
    weights_dtype = jnp.dtype(weights_dtype)
    self.img_gen_scales = tuple(img_gen_scales)
    self.selected_hidden_states_layers = (
        tuple(selected_hidden_states_layers) if selected_hidden_states_layers is not None else (-1,)
    )
    self.use_vlm_directvlm_condition = use_vlm_directvlm_condition
    self.use_learnable_token_condition = use_learnable_token_condition
    self.connector_norm = connector_norm
    self.num_query_tokens = sum(s * s for s in self.img_gen_scales)
    self.last_scale_tokens = self.img_gen_scales[-1] ** 2
    kw = {"dtype": self.dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs}
    replicated = (None, None)
    self.query_tokens = nnx.Param(
        nnx.initializers.normal(0.02)(rngs.params(), (self.num_query_tokens, llm_hidden_size), weights_dtype),
        **param_metadata(sharding, "query_tokens", replicated),
    )
    if use_learnable_token_condition:
      self.proj_in = _linear(
          llm_hidden_size, connector_hidden_size, "proj_in", replicated, use_bias=True, sharding=sharding, **kw
      )
      self.proj_out = _linear(
          connector_hidden_size, diffusion_c_input_dim, "proj_out", replicated, use_bias=True, sharding=sharding, **kw
      )
    if use_vlm_directvlm_condition:
      directvlm_dim = llm_hidden_size * len(self.selected_hidden_states_layers)
      self.proj_directvlm = nnx.List(
          [
              MingDiffusersRMSNorm(directvlm_dim, 1e-5, weights_dtype, sharding),
              _linear(
                  directvlm_dim, diffusion_inner_dim, "proj_directvlm", replicated, use_bias=True, sharding=sharding, **kw
              ),
          ]
      )

  def directvlm(self, hidden_states, prompt_mask: jax.Array) -> jax.Array:
    """`cap_feats_2 [B, L, diffusion_inner_dim]` from the selected LLM hidden states.

    Args:
      hidden_states: `[B, L, len(layers) * llm_hidden]` (already concatenated) or a sequence of `[B, L, llm_hidden]`
        in `selected_hidden_states_layers` order.
      prompt_mask: `[B, L]` directvlm tokens (`build_t2i_inputs()["prompt_mask"]`); other rows are zeroed before the
        projection (so they come out as the `proj_directvlm` bias), like the reference.
    """
    if not isinstance(hidden_states, jax.Array):
      hidden_states = jnp.concatenate(list(hidden_states), axis=-1)
    x = hidden_states * prompt_mask[..., None].astype(hidden_states.dtype)
    return self.proj_directvlm[1](self.proj_directvlm[0](x))

  def learnable(self, query_hidden: jax.Array, connector: MingConnector) -> jax.Array:
    """`cap_feats [B, s*s, diffusion_c_input_dim]` from the query tokens' final-normed LLM hidden states.

    Args:
      query_hidden: `[B, num_query_tokens, llm_hidden]` (all scales, in order; only the last scale is used).
      connector: the `MingConnector`.
    """
    x = self.proj_in(query_hidden[:, -self.last_scale_tokens :])
    x = self.proj_out(connector(x))
    if self.connector_norm:
      x = x / jnp.maximum(jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True), 1e-12).astype(x.dtype)
    return x


def gather_gen_tokens(hidden: jax.Array, gen_mask: jax.Array, num_tokens: int) -> jax.Array:
  """`[B, num_tokens, D]`: the rows of `hidden [B, L, D]` where `gen_mask [B, L]` is set, in order (`masked_select`)."""
  idx = jnp.argsort(~gen_mask.astype(jnp.bool_), axis=-1, stable=True)[:, :num_tokens]
  return jnp.take_along_axis(hidden, idx[..., None], axis=1)


def get_condition_embeds(text_encoder, connector: MingConnector, conditioner: MingConditioner, inputs: dict):
  """`(cap_feats [B, 256, 2560], cap_feats_2 [B, L, 3840])` for batched `build_t2i_inputs` arrays.

  `inputs`: `input_ids [B, L]`, `attention_mask [B, L]`, `position_ids [3, B, L]`, `image_mask`, `gen_mask`,
  `prompt_mask [B, L]`. Either output is None when its condition is disabled. See the module docstring for how
  `cap_feats_2` relates to the reference's truncated length.
  """
  layers = conditioner.selected_hidden_states_layers if conditioner.use_vlm_directvlm_condition else ()
  selected, last = text_encoder(
      inputs["input_ids"],
      inputs["position_ids"],
      inputs["attention_mask"],
      inputs["image_mask"],
      query_embeds=conditioner.query_tokens[...],
      output_layers=layers,
  )
  cap_feats = cap_feats_2 = None
  if conditioner.use_vlm_directvlm_condition:
    cap_feats_2 = conditioner.directvlm(selected, inputs["prompt_mask"])
  if conditioner.use_learnable_token_condition:
    query_hidden = gather_gen_tokens(last, inputs["gen_mask"], conditioner.num_query_tokens)
    cap_feats = conditioner.learnable(query_hidden, connector)
  return cap_feats, cap_feats_2


# ==============================================================================
# HF -> MaxText parameter mappings
# ==============================================================================


def _transpose(x, target_shape):
  del target_shape
  return x.T


def connector_param_mapping(hf_config: dict) -> tuple[dict, dict]:
  """`({maxtext_key: hf_key}, {maxtext_key: hook})` for `MingConnector` from `connector/` (Qwen2ForCausalLM)."""
  mapping, hooks = {"norm-weight": "model.norm.weight"}, {}
  for i in range(hf_config.get("num_hidden_layers", 28)):
    mt, hf = f"layers-{i}", f"model.layers.{i}"
    for proj in ("q_proj", "k_proj", "v_proj"):
      mapping[f"{mt}-self_attn-{proj}-kernel"] = f"{hf}.self_attn.{proj}.weight"
      hooks[f"{mt}-self_attn-{proj}-kernel"] = _transpose
      mapping[f"{mt}-self_attn-{proj}-bias"] = f"{hf}.self_attn.{proj}.bias"
    mapping[f"{mt}-self_attn-o_proj-kernel"] = f"{hf}.self_attn.o_proj.weight"
    hooks[f"{mt}-self_attn-o_proj-kernel"] = _transpose
    for proj in ("gate_proj", "up_proj", "down_proj"):
      mapping[f"{mt}-mlp-{proj}-kernel"] = f"{hf}.mlp.{proj}.weight"
      hooks[f"{mt}-mlp-{proj}-kernel"] = _transpose
    for norm in ("input_layernorm", "post_attention_layernorm"):
      mapping[f"{mt}-{norm}-weight"] = f"{hf}.{norm}.weight"
  return mapping, hooks


def conditioner_param_mapping(hf_config: dict) -> tuple[dict, dict]:
  """`({maxtext_key: hf_key}, {maxtext_key: hook})` for `MingConditioner` from `mlp/model.safetensors`."""
  scales = list(hf_config.get("img_gen_scales", [16]))
  if len(scales) != 1:
    raise NotImplementedError(f"Only a single img_gen_scale is supported by the mapping, got {scales}.")
  mapping = {"query_tokens": f"query_tokens_dict.{scales[0]}x{scales[0]}"}
  hooks = {}
  if hf_config.get("use_learnable_token_condition", True):
    for proj in ("proj_in", "proj_out"):
      mapping[f"{proj}-kernel"], hooks[f"{proj}-kernel"] = f"{proj}.weight", _transpose
      mapping[f"{proj}-bias"] = f"{proj}.bias"
  if hf_config.get("use_vlm_directvlm_condition", True):
    mapping["proj_directvlm-0-weight"] = "proj_directvlm.0.weight"
    mapping["proj_directvlm-1-kernel"], hooks["proj_directvlm-1-kernel"] = "proj_directvlm.1.weight", _transpose
    mapping["proj_directvlm-1-bias"] = "proj_directvlm.1.bias"
  return mapping, hooks
