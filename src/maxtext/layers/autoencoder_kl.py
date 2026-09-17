import functools
from typing import Any, Callable, Sequence, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.layers.vae import Encoder, Decoder


class AutoencoderKL(nnx.Module):
  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 3,
      latent_channels: int = 4,
      encoder_block_out_channels: Sequence[int] = (128, 256, 512, 512),
      decoder_block_out_channels: Sequence[int] = (512, 512, 256, 128),
      layers_per_block: int = 2,
      groups: int = 32,
      *,
      rngs: nnx.Rngs,
  ):
    self.encoder = Encoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        block_out_channels=encoder_block_out_channels,
        layers_per_block=layers_per_block,
        groups=groups,
        rngs=rngs,
    )
    self.decoder = Decoder(
        out_channels=out_channels,
        latent_channels=latent_channels,
        block_out_channels=decoder_block_out_channels,
        layers_per_block=layers_per_block,
        groups=groups,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray, sample_posterior: bool = False, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
    moments = self.encoder(x)
    mean, logvar = jnp.split(moments, 2, axis=-1)
    
    if sample_posterior:
      if rngs is None:
        raise ValueError("rngs must be provided for sampling posterior")
      std = jnp.exp(0.5 * logvar)
      eps = jax.random.normal(rngs.random(), mean.shape)
      z = mean + std * eps
    else:
      z = mean
      
    return self.decoder(z)
