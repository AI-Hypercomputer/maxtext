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

"""DeepSeek-OCR-2 vision encoder and connector models."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Tuple
from maxtext.common.common_types import Config
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.linears import MlpBlock
from maxtext.layers.attentions import Attention
from maxtext.layers import quantizations
from maxtext.layers.quantizations import AqtQuantization as Quant
from jax.sharding import Mesh
from maxtext.configs.pyconfig import HyperParameters

# ==============================================================================
# Helper functions for SAM ViT-B Relative Position Bias
# ==============================================================================


def get_rel_pos(q_size: int, k_size: int, rel_pos: jax.Array) -> jax.Array:
  """Get relative positional embeddings."""
  max_rel_dist = int(2 * max(q_size, k_size) - 1)
  if rel_pos.shape[0] != max_rel_dist:
    rel_pos_resized = jax.image.resize(rel_pos, (max_rel_dist, rel_pos.shape[1]), method="linear")
  else:
    rel_pos_resized = rel_pos

  q_coords = jnp.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
  k_coords = jnp.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
  relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
  relative_coords = relative_coords.astype(jnp.int32)

  return rel_pos_resized[relative_coords]


def add_decomposed_rel_pos(
    q: jax.Array,
    rel_pos_h: jax.Array,
    rel_pos_w: jax.Array,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> Tuple[jax.Array, jax.Array]:
  """Calculate decomposed Relative Positional Embeddings."""
  q_h, q_w = q_size
  k_h, k_w = k_size
  Rh = get_rel_pos(q_h, k_h, rel_pos_h)
  Rw = get_rel_pos(q_w, k_w, rel_pos_w)

  B, HW, dim = q.shape
  r_q = q.reshape(B, q_h, q_w, dim)
  rel_h = jnp.einsum("bhwc,hkc->bhwk", r_q, Rh)
  rel_w = jnp.einsum("bhwc,wkc->bhwk", r_q, Rw)

  rel_h = jnp.expand_dims(rel_h, -1)
  rel_w = jnp.expand_dims(rel_w, -2)
  rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
  rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

  return rel_h, rel_w


# ==============================================================================
# SAM ViT-B Components
# ==============================================================================


class SAMAttention(nnx.Module):
  """Multi-head Attention block with relative position embeddings for SAM."""

  def __init__(
      self,
      dim: int,
      num_heads: int,
      qkv_bias: bool,
      use_rel_pos: bool,
      input_size: Optional[Tuple[int, int]],
      rngs: nnx.Rngs,
  ):
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim**-0.5
    self.use_rel_pos = use_rel_pos

    self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
    self.proj = nnx.Linear(dim, dim, use_bias=True, rngs=rngs)

    if self.use_rel_pos:
      assert input_size is not None
      self.rel_pos_h = nnx.Param(jnp.zeros((2 * input_size[0] - 1, self.head_dim)))
      self.rel_pos_w = nnx.Param(jnp.zeros((2 * input_size[1] - 1, self.head_dim)))

  def __call__(self, x: jax.Array) -> jax.Array:
    B, H, W, _ = x.shape
    qkv = self.qkv(x.reshape(B, H * W, -1))
    qkv = qkv.reshape(B, H * W, 3, self.num_heads, self.head_dim)
    qkv = qkv.transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if self.use_rel_pos:
      q_flat = q.reshape(B * self.num_heads, H * W, self.head_dim)
      rel_h, rel_w = add_decomposed_rel_pos(q_flat, self.rel_pos_h.value, self.rel_pos_w.value, (H, W), (H, W))

      rel_h = rel_h.reshape(B, self.num_heads, H * W, H, 1)
      rel_w = rel_w.reshape(B, self.num_heads, H * W, 1, W)

      attn_bias = rel_h + rel_w
      attn_bias = attn_bias.reshape(B, self.num_heads, H * W, H * W)

      logits = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale
      logits = logits + attn_bias
      attn = jax.nn.softmax(logits, axis=-1)
      out = jnp.einsum("bhij,bhjd->bhid", attn, v)
    else:
      logits = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale
      attn = jax.nn.softmax(logits, axis=-1)
      out = jnp.einsum("bhij,bhjd->bhid", attn, v)

    out = out.transpose(0, 2, 1, 3).reshape(B, H, W, -1)
    out = self.proj(out)
    return out


def window_partition(x: jax.Array, window_size: int) -> Tuple[jax.Array, Tuple[int, int]]:
  """Partition into non-overlapping windows."""
  B, H, W, C = x.shape
  pad_h = (window_size - H % window_size) % window_size
  pad_w = (window_size - W % window_size) % window_size
  if pad_h > 0 or pad_w > 0:
    x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
  Hp, Wp = H + pad_h, W + pad_w

  x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
  x = x.transpose(0, 1, 3, 2, 4, 5)
  windows = x.reshape(-1, window_size, window_size, C)
  return windows, (Hp, Wp)


def window_unpartition(windows: jax.Array, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> jax.Array:
  """Window unpartition."""
  Hp, Wp = pad_hw
  H, W = hw
  B = windows.shape[0] // (Hp * Wp // window_size // window_size)
  x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
  x = x.transpose(0, 1, 3, 2, 4, 5)
  x = x.reshape(B, Hp, Wp, -1)
  if Hp > H or Wp > W:
    x = x[:, :H, :W, :]
  return x


class SAMBlock(nnx.Module):
  """Transformer block for SAM."""

  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float,
      qkv_bias: bool,
      use_rel_pos: bool,
      window_size: int,
      input_size: Optional[Tuple[int, int]],
      rngs: nnx.Rngs,
  ):
    self.norm1 = nnx.LayerNorm(num_features=dim, epsilon=1e-5, rngs=rngs)
    self.attn = SAMAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        use_rel_pos=use_rel_pos,
        input_size=input_size if window_size == 0 else (window_size, window_size),
        rngs=rngs,
    )
    self.norm2 = nnx.LayerNorm(num_features=dim, epsilon=1e-5, rngs=rngs)
    self.lin1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
    self.lin2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
    self.window_size = window_size

  def __call__(self, x: jax.Array) -> jax.Array:
    shortcut = x
    x = self.norm1(x)

    if self.window_size > 0:
      H, W = x.shape[1], x.shape[2]
      x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)

    if self.window_size > 0:
      x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    mlp_out = self.lin2(jax.nn.gelu(self.lin1(self.norm2(x))))
    x = x + mlp_out
    return x


def get_abs_pos_sam(abs_pos, tgt_size):
  """Interpolate absolute position embeddings."""
  src_size = abs_pos.shape[1]
  if src_size != tgt_size:
    new_pos_embed = jax.image.resize(abs_pos, (1, tgt_size, tgt_size, abs_pos.shape[3]), method="bicubic")
    return new_pos_embed
  else:
    return abs_pos


class SAMViTB(nnx.Module):
  """SAM ViT-B image encoder."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    # SAM ViT-B hardcoded parameters
    img_size = 1024
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    out_chans = 256
    qkv_bias = True
    use_abs_pos = True
    use_rel_pos = True
    window_size = 14
    global_attn_indexes = (2, 5, 8, 11)

    self.patch_embed = nnx.Conv(
        in_features=in_chans,
        out_features=embed_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="VALID",
        rngs=rngs,
    )

    if use_abs_pos:
      self.pos_embed = nnx.Param(jnp.zeros((1, img_size // patch_size, img_size // patch_size, embed_dim)))
    else:
      self.pos_embed = None

    self.blocks = nnx.List([])
    for i in range(depth):
      block = SAMBlock(
          dim=embed_dim,
          num_heads=num_heads,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          use_rel_pos=use_rel_pos,
          window_size=window_size if i not in global_attn_indexes else 0,
          input_size=(img_size // patch_size, img_size // patch_size),
          rngs=rngs,
      )
      self.blocks.append(block)
      setattr(self, f"block_{i}", block)

    # Neck
    self.neck_conv1 = nnx.Conv(embed_dim, out_chans, kernel_size=(1, 1), use_bias=False, rngs=rngs)
    self.neck_ln1 = nnx.LayerNorm(num_features=out_chans, rngs=rngs)
    self.neck_conv2 = nnx.Conv(out_chans, out_chans, kernel_size=(3, 3), padding="SAME", use_bias=False, rngs=rngs)
    self.neck_ln2 = nnx.LayerNorm(num_features=out_chans, rngs=rngs)

    self.net_2 = nnx.Conv(
        256, 512, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs
    )
    self.net_3 = nnx.Conv(
        512, 896, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.patch_embed(x)

    if self.pos_embed is not None:
      H_prime = x.shape[1]
      pos_embed_resized = get_abs_pos_sam(self.pos_embed.value, H_prime)
      x = x + pos_embed_resized

    for i in range(len(self.blocks)):
      x = self.blocks[i](x)

    x = self.neck_conv1(x)
    x = self.neck_ln1(x)
    x = self.neck_conv2(x)
    x = self.neck_ln2(x)

    x2 = self.net_2(x)
    x3 = self.net_3(x2)

    return x3


# ==============================================================================
# Qwen2 Decoder as Encoder Components
# ==============================================================================


class Qwen2EncoderLayer(nnx.Module):
  """Qwen2 decoder layer modified for encoding (supports bidirectional mask)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant],
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    query_pre_attn_scalar = config.head_dim**-0.5
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=(config.per_device_batch_size, config.max_target_length, config.emb_dim),
        inputs_kv_shape=(config.per_device_batch_size, config.max_target_length, config.emb_dim),
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_bias_in_projections=config.attention_bias,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode="train",
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode="train",
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jax.Array,
      bidirectional_mask: jax.Array,
      decoder_positions: jax.Array,
      deterministic: bool,
  ) -> jax.Array:
    lnx = self.pre_self_attention_layer_norm(inputs)

    attention_lnx, _ = self.self_attention(
        lnx,
        lnx,
        inputs_positions=decoder_positions,
        deterministic=deterministic,
        model_mode="train",
        bidirectional_mask=bidirectional_mask,
    )

    intermediate_inputs = inputs + attention_lnx
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)

    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    layer_output = intermediate_inputs + mlp_lnx
    return layer_output


class Qwen2Decoder2Encoder(nnx.Module):
  """Qwen2 decoder used as an encoder with learnable queries."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant],
      rngs: nnx.Rngs,
  ):
    self.config = config

    # Create connector config
    pydantic_config = config._pydantic_config
    update_dict = {
        "emb_dim": config.vision_connector_emb_dim,
        "num_query_heads": config.vision_connector_num_query_heads,
        "num_kv_heads": config.vision_connector_num_kv_heads,
        "mlp_dim": config.vision_connector_mlp_dim,
        "num_decoder_layers": config.vision_connector_num_layers,
        "head_dim": config.vision_connector_emb_dim // config.vision_connector_num_query_heads,
        "attention_type": "global",
        "rope_max_timescale": 1000000,
    }
    if hasattr(pydantic_config, "model_copy"):
      new_pydantic_config = pydantic_config.model_copy(update=update_dict)
    else:
      new_pydantic_config = pydantic_config.copy(update=update_dict)
    self.connector_config = HyperParameters(new_pydantic_config)

    self.query_768 = nnx.Embed(num_embeddings=144, features=self.connector_config.emb_dim, rngs=rngs)
    self.query_1024 = nnx.Embed(num_embeddings=256, features=self.connector_config.emb_dim, rngs=rngs)
    self.norm = RMSNorm(
        num_features=self.connector_config.emb_dim,
        dtype=self.connector_config.dtype,
        weight_dtype=self.connector_config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.connector_config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.layers = nnx.List([])
    for i in range(self.connector_config.num_decoder_layers):
      layer = Qwen2EncoderLayer(
          config=self.connector_config,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
      self.layers.append(layer)
      setattr(self, f"layer_{i}", layer)

  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    B, H, W, C = x.shape
    x_flat = x.reshape(B, H * W, C)

    n_query = H * W
    if n_query == 144:
      queries = self.query_768.embedding
    elif n_query == 256:
      queries = self.query_1024.embedding
    else:
      raise ValueError(f"Unsupported query size: {n_query}")

    queries_batch = jnp.broadcast_to(queries[None, :, :], (B, queries.shape[0], queries.shape[1]))
    x_combined = jnp.concatenate([x_flat, queries_batch], axis=1)

    mask_image = jnp.ones((B, n_query), dtype=jnp.bool_)
    mask_query = jnp.zeros((B, n_query), dtype=jnp.bool_)
    bidirectional_mask = jnp.concatenate([mask_image, mask_query], axis=1)

    decoder_positions = jnp.arange(2 * n_query)[None, :]

    y = x_combined
    for layer in self.layers:
      y = layer(y, bidirectional_mask, decoder_positions, deterministic)
    y = self.norm(y)

    return y[:, n_query:, :]


# ==============================================================================
# Projector and Top-level Vision Encoder
# ==============================================================================


class MlpProjector(nnx.Module):
  """Linear projector to map connector features to language model dimension."""

  def __init__(self, config: Config, mesh: Mesh, rngs: nnx.Rngs):
    self.linear = nnx.Linear(config.vision_connector_emb_dim, config.emb_dim, use_bias=True, rngs=rngs)
    self.view_seperator = nnx.Param(jax.random.normal(rngs.params(), (config.emb_dim,)) * (config.emb_dim**-0.5))

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.linear(x)
    if x.ndim == 4:
      separator = jnp.broadcast_to(self.view_seperator.value, (*x.shape[:-2], 1, x.shape[-1]))
      return jnp.concatenate([x, separator], axis=-2)
    if x.ndim == 3:
      separator = jnp.broadcast_to(self.view_seperator.value, (x.shape[0], 1, x.shape[-1]))
      return jnp.concatenate([x, separator], axis=-2)
    return x


class DeepseekOCR2VisionEncoder(nnx.Module):
  """Full vision tower for DeepSeek-OCR-2 (SAM + Qwen2)."""

  def __init__(self, config: Config, mesh: Mesh, rngs: nnx.Rngs):
    self.sam_model = SAMViTB(config, mesh, rngs)
    self.qwen2_model = Qwen2Decoder2Encoder(config, mesh, None, rngs)
    self.crop_size = 768

  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    if x.ndim == 6:
      B, N, crops, H, W, C = x.shape
      if crops < 1:
        raise ValueError("DeepSeek-OCR-2 vision input must include the global view.")

      global_images = x[:, :, 0].reshape(B * N, H, W, C)
      global_embeddings = self.qwen2_model(self.sam_model(global_images), deterministic)

      if crops == 1:
        return global_embeddings.reshape(B, N, global_embeddings.shape[1], global_embeddings.shape[2])

      crop_images = x[:, :, 1:, : self.crop_size, : self.crop_size, :]
      crop_images = crop_images.reshape(B * N * (crops - 1), self.crop_size, self.crop_size, C)
      crop_embeddings = self.qwen2_model(self.sam_model(crop_images), deterministic)
      crop_tokens = crop_embeddings.shape[1]
      crop_dim = crop_embeddings.shape[2]
      crop_embeddings = crop_embeddings.reshape(B, N, crops - 1, crop_tokens, crop_dim)
      crop_embeddings = crop_embeddings.reshape(B, N, (crops - 1) * crop_tokens, crop_dim)

      return jnp.concatenate(
          [crop_embeddings, global_embeddings.reshape(B, N, global_embeddings.shape[1], global_embeddings.shape[2])],
          axis=2,
      )
    elif x.ndim == 4:
      return self.qwen2_model(self.sam_model(x), deterministic)
    else:
      raise ValueError(f"Expected 4D or 6D input, got {x.ndim}D")
