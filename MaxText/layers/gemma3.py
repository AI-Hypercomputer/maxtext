"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.debug
import jax.numpy as jnp

from flax import linen as nn

from MaxText.common_types import Config
from MaxText.layers import attentions
from MaxText.layers import quantizations
from MaxText.layers.attentions import AttentionType, Attention
from MaxText.layers.linears import MlpBlock
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.quantizations import AqtQuantization as Quant


GEMMA3_ATTENTION_PATTERN = (
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  layer_id %= len(GEMMA3_ATTENTION_PATTERN)
  return GEMMA3_ATTENTION_PATTERN[layer_id]


def get_query_pre_attn_scalar(config) -> float:
  """Returns the scalar to multiply the query by before attention."""
  if config.model_name in ["gemma3-4b", "gemma3-12b"]:
    return config.head_dim**-0.5
  elif config.model_name == "gemma3-27b":
    return (config.base_emb_dim // config.base_num_query_heads) ** -0.5
  else:
    raise ValueError(f"Unsupported model name: {config.model_name}")


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]  # pylint: disable=unpacking-non-sequence

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


class MlpBlockViT(nn.Module):
  """Transformer MLP / feed-forward block."""

  block_id: int
  dtype_mm: str
  mlp_dim: int | None = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    """Applies Transformer MlpBlock module."""
    inits = {"kernel_init": nn.initializers.xavier_uniform(), "bias_init": nn.initializers.normal(stddev=1e-6)}

    d = x.shape[-1]
    x = nn.Dense(features=self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(
        features=d,
        dtype=self.dtype_mm,
        **inits,
    )(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  block_id: int
  dtype_mm: str
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    y = nn.LayerNorm()(x)

    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlockViT(
        block_id=self.block_id,
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  depth: int
  dtype_mm: str
  remat_policy: str
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False

  @nn.compact
  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(2,),  # 0=self, 2=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      x = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth,
      )(
          block_id=0,
          name="encoderblock",
          dtype_mm=self.dtype_mm,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
      )(
          x, deterministic
      )
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            block_id=lyr,
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        x = block_cur(x, deterministic)
    x: jax.Array = nn.LayerNorm(name="encoder_norm")(x)
    return x


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  shape: tuple[int, ...]
  weight_name: str = "w"
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    return jnp.einsum(eqn, x, w)


class VisionEmbedder(nn.Module):
  """Projects image embeddings to the embedding space of the text encoder."""

  config: Config
  mesh: Mesh
  vision_proj_dim: int = 1152

  def setup(self):
    if self.vision_proj_dim:
      self.mm_soft_embedding_norm = rms_norm(self.vision_proj_dim)
      self.mm_input_projection = Einsum((self.vision_proj_dim, self.config.emb_dim))

  def encode_vision(self, x: jax.Array) -> jax.Array:
    x = self.mm_soft_embedding_norm(x)
    x = self.mm_input_projection("...tm,md->...td", x)
    return x

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.encode_vision(x)


class VisionExit(nn.Module):
  """The vision exit layer.

  Possibly downsample the soft tokens to a required output length.

  Attributes:
    output_length: The embed will be spatially avg-pooled to this output length.
  """

  output_length: int = 256

  def __call__(self, x):
    cur_length = x.shape[1]
    if cur_length == self.output_length:
      return x
    cur_width = int(cur_length**0.5)
    assert cur_width**2 == cur_length
    output_width = int(self.output_length**0.5)
    assert output_width**2 == self.output_length, f"Cannot pool {x.shape=} to {self.output_length}=!"
    batch_size = x.shape[0]
    embed_dim = x.shape[-1]
    x = jnp.reshape(x, (batch_size, cur_width, cur_width, embed_dim))
    assert not cur_width % output_width, f"{cur_width=} {output_width=}"
    window = cur_width // output_width
    window_shape = (window, window)
    x = nn.avg_pool(x, window_shape=window_shape, strides=window_shape)
    batch_size, height, width, embed_dim = x.shape
    return jnp.reshape(x, (batch_size, height * width, embed_dim))


class Gemma3VisionEncoderLayer(nn.Module):
  """gemma 3 vision encoder layer"""

  config: Config
  mesh: Mesh
  patch_size: tuple[int, int] = (14, 14)
  width: int = 1152
  mlp_dim: int | None = 4304  # Defaults to 4x input dim
  depth: int = 27
  num_heads: int = 16
  posemb: str = "learn"  # Can also be "sincos2d"
  dropout: float = 0.0
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)

  def _get_posemb(
      self,
      typ: str,
      *,
      seqshape: tuple[int, int],
      width: int,
      name: str,
      dtype: jnp.dtype = jnp.float32,
  ):
    """Returns the position embedding."""
    if typ == "learn":
      shape_product = seqshape[0] * seqshape[1]
      return self.param(
          name,
          nn.initializers.normal(stddev=1 / (width**0.5)),
          (1, shape_product, width),
          dtype,
      )
    elif typ == "sincos2d":
      return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype)
    else:
      raise ValueError(f"Unknown posemb type: {typ}")

  @nn.compact
  def __call__(self, inputs, deterministic, train=False):
    """ViT model that transforms image inputs to image embeddings.
    Args:
      inputs: jnp.array shaped [B, N, H, W, C], e.g. [4, 1, 896, 896, 3]
    Returns:
      jnp.array for image embeddings, shaped [B, N, P, D], e.g. [4, 1, 256, 2560]
    """
    cfg = self.config
    b, n, h, w, c = inputs.shape
    x = jnp.reshape(inputs, [b * n, h, w, c])
    # Gemma3 uses conv2d with stride 14 and kernel size 14 to extract patches.
    x = nn.Conv(features=1152, kernel_size=(14, 14), strides=14, padding="VALID", name="embedding")(x)
    bn, h, w, c = x.shape
    x = jnp.reshape(x, [bn, h * w, c])

    # Add posemb before adding extra token.
    x = x + self._get_posemb(
        self.posemb,
        seqshape=(h, w),
        width=c,
        name="pos_embedding",
        dtype=x.dtype,
    )

    x = nn.Dropout(rate=self.dropout)(x, not train)

    # Transformer encoder to extract image features.
    x = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=cfg.scan_layers,
        remat_policy=cfg.remat_policy_for_vit,
        dtype_mm=cfg.dtype_mm,
        name="Transformer",
    )(x, deterministic=deterministic)

    # Gemma3 use a vision exit layer to downsample the soft tokens to a required output length.
    x = VisionExit(output_length=256)(x)
    bn, l, c = x.shape
    x = jnp.reshape(x, [b, n, l, c])
    return x


# Gemma3 Decoder Layer
class Gemma3DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None
  attention_type: AttentionType = AttentionType.LOCAL_SLIDING

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      bidirectional_mask=None,
  ):
    cfg = self.config
    mesh = self.mesh
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = rms_norm(
        num_features=inputs.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        kernel_axes=("norm",),
    )(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
    query_pre_attn_scalar = get_query_pre_attn_scalar(cfg)

    attention_layer = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        attention_type=self.attention_type,
        sliding_window_size=cfg.sliding_window_size,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 3 models use query, key normalizations
        query_pre_attn_scalar=query_pre_attn_scalar,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
    )
    if cfg.use_post_attn_norm:
      attention_lnx = rms_norm(
          num_features=attention_lnx.shape[-1],
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="post_self_attention_norm",
          kernel_axes=("norm",),
      )(attention_lnx)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    attention_lnx += inputs
    residual = attention_lnx

    attn_output = rms_norm(
        num_features=attention_lnx.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_ffw_norm",
        kernel_axes=("norm",),
    )(attention_lnx)

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(attn_output, deterministic=deterministic)

    if cfg.use_post_ffw_norm:
      mlp_lnx = rms_norm(
          num_features=mlp_lnx.shape[-1],
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="post_ffw_norm",
          kernel_axes=("norm",),
      )(mlp_lnx)

    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
    next_layer_addition = mlp_lnx + residual
    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
