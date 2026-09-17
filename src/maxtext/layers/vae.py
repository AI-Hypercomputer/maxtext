import functools
from typing import Any, Callable, Sequence, Tuple

from flax import nnx
import jax
import jax.numpy as jnp


class Downsample(nnx.Module):
  """Downsamples input by factor of 2."""

  def __init__(self, in_channels: int, out_channels: int | None = None, *, rngs: nnx.Rngs):
    self.in_channels = in_channels
    self.out_channels = out_channels or in_channels
    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=self.out_channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.conv(x)


class Upsample(nnx.Module):
  """Upsamples input by factor of 2."""

  def __init__(self, in_channels: int, out_channels: int | None = None, *, rngs: nnx.Rngs):
    self.in_channels = in_channels
    self.out_channels = out_channels or in_channels
    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=self.out_channels,
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    b, h, w, c = x.shape
    x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
    return self.conv(x)


class AttentionBlock(nnx.Module):
  """Self-attention block for 2D feature maps."""

  def __init__(self, channels: int, groups: int = 32, *, rngs: nnx.Rngs):
    self.channels = channels
    self.norm = nnx.GroupNorm(num_groups=groups, num_features=channels, rngs=rngs)
    self.q = nnx.Linear(in_features=channels, out_features=channels, rngs=rngs)
    self.k = nnx.Linear(in_features=channels, out_features=channels, rngs=rngs)
    self.v = nnx.Linear(in_features=channels, out_features=channels, rngs=rngs)
    self.proj_out = nnx.Linear(in_features=channels, out_features=channels, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    b, h, w, c = x.shape
    h_in = x
    
    x = self.norm(x)
    x = x.reshape(b, h * w, c)
    
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    
    # Compute attention
    attn = jnp.einsum("btd,bsd->bts", q, k) * (c**-0.5)
    attn = jax.nn.softmax(attn, axis=-1)
    
    h_out = jnp.einsum("bts,bsd->btd", attn, v)
    h_out = self.proj_out(h_out)
    
    h_out = h_out.reshape(b, h, w, c)
    
    return h_in + h_out


class ResnetBlock(nnx.Module):
  """Resnet block for VAE."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int | None = None,
      groups: int = 32,
      *,
      rngs: nnx.Rngs,
  ):
    out_channels = out_channels or in_channels
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.norm1 = nnx.GroupNorm(num_groups=groups, num_features=in_channels, rngs=rngs)
    self.conv1 = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )
    self.norm2 = nnx.GroupNorm(num_groups=groups, num_features=out_channels, rngs=rngs)
    self.conv2 = nnx.Conv(
        in_features=out_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )

    if in_channels != out_channels:
      self.nin_shortcut = nnx.Conv(
          in_features=in_channels,
          out_features=out_channels,
          kernel_size=(1, 1),
          padding="SAME",
          rngs=rngs,
      )
    else:
      self.nin_shortcut = None

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    h = self.norm1(h)
    h = jax.nn.silu(h)
    h = self.conv1(h)

    h = self.norm2(h)
    h = jax.nn.silu(h)
    h = self.conv2(h)

    if self.nin_shortcut is not None:
      x = self.nin_shortcut(x)

    return x + h


class MidBlock(nnx.Module):
  """Mid block of Encoder/Decoder."""

  def __init__(self, channels: int, groups: int = 32, *, rngs: nnx.Rngs):
    self.resnet1 = ResnetBlock(in_channels=channels, out_channels=channels, groups=groups, rngs=rngs)
    self.attn = AttentionBlock(channels=channels, groups=groups, rngs=rngs)
    self.resnet2 = ResnetBlock(in_channels=channels, out_channels=channels, groups=groups, rngs=rngs)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = self.resnet1(x)
    x = self.attn(x)
    x = self.resnet2(x)
    return x


class Encoder(nnx.Module):
  def __init__(
      self,
      in_channels: int = 3,
      latent_channels: int = 4,
      block_out_channels: Sequence[int] = (128, 256, 512, 512),
      layers_per_block: int = 2,
      groups: int = 32,
      *,
      rngs: nnx.Rngs,
  ):
    self.conv_in = nnx.Conv(
        in_features=in_channels,
        out_features=block_out_channels[0],
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )
    
    # Down blocks
    self.down_blocks = nnx.List()
    current_channels = block_out_channels[0]
    for i, out_c in enumerate(block_out_channels):
      for _ in range(layers_per_block):
        self.down_blocks.append(
            ResnetBlock(in_channels=current_channels, out_channels=out_c, groups=groups, rngs=rngs)
        )
        current_channels = out_c
      if i != len(block_out_channels) - 1:
        self.down_blocks.append(Downsample(in_channels=current_channels, rngs=rngs))
        
    self.mid = MidBlock(channels=current_channels, groups=groups, rngs=rngs)
    
    self.norm_out = nnx.GroupNorm(num_groups=groups, num_features=current_channels, rngs=rngs)
    self.conv_out = nnx.Conv(
        in_features=current_channels,
        out_features=2 * latent_channels, # mean and logvar
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = self.conv_in(x)
    for block in self.down_blocks:
      x = block(x)
    x = self.mid(x)
    x = self.norm_out(x)
    x = jax.nn.silu(x)
    x = self.conv_out(x)
    return x


class Decoder(nnx.Module):
  def __init__(
      self,
      out_channels: int = 3,
      latent_channels: int = 4,
      block_out_channels: Sequence[int] = (512, 512, 256, 128),
      layers_per_block: int = 2,
      groups: int = 32,
      *,
      rngs: nnx.Rngs,
  ):
    self.post_quant_conv = nnx.Conv(
        in_features=latent_channels,
        out_features=latent_channels,
        kernel_size=(1, 1),
        padding="SAME",
        rngs=rngs,
    )
    
    self.conv_in = nnx.Conv(
        in_features=latent_channels,
        out_features=block_out_channels[0],
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )
    
    self.mid = MidBlock(channels=block_out_channels[0], groups=groups, rngs=rngs)
    
    # Up blocks
    self.up_blocks = nnx.List()
    current_channels = block_out_channels[0]
    for i, out_c in enumerate(block_out_channels):
      for _ in range(layers_per_block):
        self.up_blocks.append(
            ResnetBlock(in_channels=current_channels, out_channels=out_c, groups=groups, rngs=rngs)
        )
        current_channels = out_c
      if i != len(block_out_channels) - 1:
        self.up_blocks.append(Upsample(in_channels=current_channels, rngs=rngs))
        
    self.norm_out = nnx.GroupNorm(num_groups=groups, num_features=current_channels, rngs=rngs)
    self.conv_out = nnx.Conv(
        in_features=current_channels,
        out_features=out_channels,
        kernel_size=(3, 3),
        padding="SAME",
        rngs=rngs,
    )

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = self.post_quant_conv(z)
    z = self.conv_in(z)
    z = self.mid(z)
    for block in self.up_blocks:
      z = block(z)
    z = self.norm_out(z)
    z = jax.nn.silu(z)
    z = self.conv_out(z)
    return z
