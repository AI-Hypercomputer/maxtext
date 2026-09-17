import functools
import operator
from typing import Any, Callable, Sequence, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import Config, DType, ShardMode, AttentionType
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.embeddings import PositionalEmbedding, Embed
from maxtext.layers.initializers import nd_dense_init, NdInitializer
from maxtext.layers.vae import Encoder, Decoder
from maxtext.utils import max_utils


class UnaffinedLayerNorm(nnx.Module):
  """Layer normalization without learnable affine parameters."""

  def __init__(self, epsilon: float = 1e-6):
    self.epsilon = epsilon

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + self.epsilon)


def zeros_nd_init() -> NdInitializer:
  """Returns an initializer that initializes to zeros, compatible with DenseGeneral."""
  def init_fn(key, shape, dtype, in_axis, out_axis):
    return jax.nn.initializers.zeros(key, shape, dtype)
  return init_fn


class TimestepEmbedder(nnx.Module):
  """Embeds scalar timesteps into vector representations."""

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs):
    self.hidden_size = hidden_size
    self.frequency_embedding_size = frequency_embedding_size
    self.pos_emb = PositionalEmbedding(embedding_dims=frequency_embedding_size)
    self.linear1 = DenseGeneral(in_features_shape=frequency_embedding_size, out_features_shape=hidden_size, use_bias=True, rngs=rngs)
    self.linear2 = DenseGeneral(in_features_shape=hidden_size, out_features_shape=hidden_size, use_bias=True, rngs=rngs)

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    t_reshaped = t[:, jnp.newaxis]
    temb = self.pos_emb(seq_len=1, position=t_reshaped)
    temb = temb[:, 0, :]
    
    # Flip sine and cosine embeddings to match HF flip_sin_to_cos=True
    half_dim = temb.shape[-1] // 2
    temb = jnp.concatenate([temb[:, half_dim:], temb[:, :half_dim]], axis=-1)
    
    temb = self.linear1(temb)
    temb = jax.nn.silu(temb)
    temb = self.linear2(temb)
    return temb


class LabelEmbedder(nnx.Module):
  """Embeds categorical labels into vector representations."""

  def __init__(self, num_classes: int, hidden_size: int, use_cfg_embedding: bool, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.num_embeddings = num_classes + (1 if use_cfg_embedding else 0)
    self.embedding = Embed(
        num_embeddings=self.num_embeddings,
        num_features=hidden_size,
        config=config,
        mesh=mesh,
        rngs=rngs
    )
    
  def __call__(self, labels: jnp.ndarray) -> jnp.ndarray:
    return self.embedding(labels)


class DiTBlock(nnx.Module):
  """DiT Block with Adaptive Layer Norm (adaLN-Zero)."""

  def __init__(self, config: Config, mesh: Mesh, hidden_size: int, num_heads: int, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.hidden_size = hidden_size
    
    self.norm1 = UnaffinedLayerNorm()
    self.attn = Attention(
        config=config,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=hidden_size // num_heads,
        max_target_length=config.max_target_length,
        attention_kernel=config.attention,
        inputs_q_shape=(1, config.max_target_length, hidden_size),
        inputs_kv_shape=(1, config.max_target_length, hidden_size),
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        attention_type=AttentionType.FULL,
        is_nope_layer=True, # DiT doesn't use RoPE
        use_bias_in_projections=True,
        query_pre_attn_scalar=1.0 / ((hidden_size // num_heads) ** 0.5),
        rngs=rngs,
    )
    self.norm2 = UnaffinedLayerNorm(epsilon=1e-5)
    self.mlp = MlpBlock(
        config=config,
        mesh=mesh,
        in_features=hidden_size,
        intermediate_dim=hidden_size * 4,
        activations=[functools.partial(jax.nn.gelu, approximate=True)],
        use_pre_norm=False,
        use_bias=True,
        rngs=rngs,
    )
    
    self.adaLN_modulation = DenseGeneral(
        in_features_shape=hidden_size,
        out_features_shape=6 * hidden_size,
        use_bias=True,
        kernel_init=zeros_nd_init(),
        rngs=rngs
    )

  def __call__(self, x: jnp.ndarray, c: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    modulation = self.adaLN_modulation(c)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)
    
    shift_msa = shift_msa[:, jnp.newaxis, :]
    scale_msa = scale_msa[:, jnp.newaxis, :]
    gate_msa = gate_msa[:, jnp.newaxis, :]
    shift_mlp = shift_mlp[:, jnp.newaxis, :]
    scale_mlp = scale_mlp[:, jnp.newaxis, :]
    gate_mlp = gate_mlp[:, jnp.newaxis, :]
    
    normed_x = self.norm1(x)
    modulated_x = normed_x * (1 + scale_msa) + shift_msa
    
    attn_out, _ = self.attn(
        inputs_q=modulated_x,
        inputs_kv=modulated_x,
        deterministic=deterministic,
    )
    
    x = x + gate_msa * attn_out
    
    normed_x = self.norm2(x)
    modulated_x = normed_x * (1 + scale_mlp) + shift_mlp
    mlp_out = self.mlp(modulated_x, deterministic=deterministic)
    
    x = x + gate_mlp * mlp_out
    
    return x


class FinalLayer(nnx.Module):
  """The final projection layer of DiT."""

  def __init__(self, hidden_size: int, patch_size: int, out_channels: int, *, rngs: nnx.Rngs):
    self.norm_final = UnaffinedLayerNorm()
    self.linear = DenseGeneral(
        in_features_shape=hidden_size,
        out_features_shape=patch_size * patch_size * out_channels,
        use_bias=True,
        rngs=rngs
    )
    self.adaLN_modulation = DenseGeneral(
        in_features_shape=hidden_size,
        out_features_shape=2 * hidden_size,
        use_bias=True,
        kernel_init=zeros_nd_init(),
        rngs=rngs
    )

  def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    modulation = self.adaLN_modulation(c)
    shift, scale = jnp.split(modulation, 2, axis=-1)
    
    shift = shift[:, jnp.newaxis, :]
    scale = scale[:, jnp.newaxis, :]
    
    normed_x = self.norm_final(x)
    modulated_x = normed_x * (1 + scale) + shift
    
    output = self.linear(modulated_x)
    return output


class DiT(nnx.Module):
  """Diffusion Transformer (DiT) Model."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.hidden_size = config.base_emb_dim
    
    patch_size = config.patch_size_for_vit
    in_channels = config.num_channels_for_vit
    out_channels = getattr(config, 'out_channels_for_vit', in_channels)
    patch_dim = patch_size * patch_size * in_channels
    
    self.patch_embed = DenseGeneral(in_features_shape=patch_dim, out_features_shape=self.hidden_size, use_bias=True, rngs=rngs)
    
    seq_len = config.max_target_length
    self.pos_embed = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(rngs.params(), (1, seq_len, self.hidden_size), config.dtype),
    )
    
    self.timestep_embedder = TimestepEmbedder(hidden_size=self.hidden_size, rngs=rngs)
    self.label_embedder = LabelEmbedder(
        num_classes=config.vocab_size,
        hidden_size=self.hidden_size,
        use_cfg_embedding=True,
        config=config,
        mesh=mesh,
        rngs=rngs
    )
    
    self.blocks = nnx.List([
        DiTBlock(config=config, mesh=mesh, hidden_size=self.hidden_size, num_heads=config.base_num_query_heads, rngs=rngs)
        for _ in range(config.base_num_decoder_layers)
    ])
    
    self.final_layer = FinalLayer(hidden_size=self.hidden_size, patch_size=patch_size, out_channels=out_channels, rngs=rngs)
    
    self.vae_decoder = Decoder(layers_per_block=3, rngs=rngs)

  def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
    x = self.patch_embed(x)
    x = x + self.pos_embed.get_value()
    
    t_emb = self.timestep_embedder(t)
    y_emb = self.label_embedder(y)
    
    c = jax.nn.silu(t_emb + y_emb)
    
    for block in self.blocks:
      x = block(x, c, deterministic=deterministic)
      
    x = self.final_layer(x, c)
    
    return x



