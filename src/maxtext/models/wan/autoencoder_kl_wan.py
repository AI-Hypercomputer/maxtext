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

from typing import Tuple, List, Sequence, Union, Optional

import flax
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax import tree_util
from flax import nnx
from diffusers.configuration_utils import ConfigMixin
from maxtext.utils import max_logging
from .wan_utils import FlaxModelMixin, get_activation
from maxtext.common import common_types
from diffusers.utils import BaseOutput
from jax import tree_util


@flax.struct.dataclass
class FlaxDecoderOutput(BaseOutput):
  sample: jnp.ndarray


@flax.struct.dataclass
class FlaxAutoencoderKLOutput(BaseOutput):
  latent_dist: "FlaxDiagonalGaussianDistribution"


class FlaxDiagonalGaussianDistribution(object):

  def __init__(self, parameters, deterministic=False):
    self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
    self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = jnp.exp(0.5 * self.logvar)
    self.var = jnp.exp(self.logvar)
    if self.deterministic:
      self.var = self.std = jnp.zeros_like(self.mean)

  def sample(self, key):
    return self.mean + self.std * jax.random.normal(key, self.mean.shape)

  def kl(self, other=None):
    if self.deterministic:
      return jnp.array([0.0])
    if other is None:
      return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])
    return 0.5 * jnp.sum(
        jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
        axis=[1, 2, 3],
    )

  def nll(self, sample, axis=[1, 2, 3]):
    if self.deterministic:
      return jnp.array([0.0])
    logtwopi = jnp.log(2.0 * jnp.pi)
    return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

  def mode(self):
    return self.mean


class WanDiagonalGaussianDistribution(FlaxDiagonalGaussianDistribution):
  """Diagonal Gaussian distribution with JAX pytree registration for VAE latents."""


def _wan_diag_gauss_dist_flatten(dist):
  return (dist.mean, dist.logvar, dist.std, dist.var), (dist.deterministic,)


def _wan_diag_gauss_dist_unflatten(aux, children):
  mean, logvar, std, var = children
  deterministic = aux[0]
  obj = WanDiagonalGaussianDistribution.__new__(WanDiagonalGaussianDistribution)
  obj.mean = mean
  obj.logvar = logvar
  obj.std = std
  obj.var = var
  obj.deterministic = deterministic
  return obj


tree_util.register_pytree_node(
    WanDiagonalGaussianDistribution,
    _wan_diag_gauss_dist_flatten,
    _wan_diag_gauss_dist_unflatten,
)

BlockSizes = common_types.BlockSizes

CACHE_T = 2
try:
  flax.config.update("flax_always_shard_variable", False)
except LookupError:
  pass


def _update_cache(cache, idx, value):
  if cache is None:
    return None
  return cache[:idx] + (value,) + cache[idx + 1 :]


# Helper to ensure kernel_size, stride, padding are tuples of 3 integers
def _canonicalize_tuple(x: Union[int, Sequence[int]], rank: int, name: str) -> Tuple[int, ...]:
  """Canonicalizes a value to a tuple of integers."""
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(f"Argument '{name}' must be an integer or a sequence of {rank} integers. Got {x}")


class RepSentinel:

  def __eq__(self, other):
    return isinstance(other, RepSentinel)


tree_util.register_pytree_node(RepSentinel, lambda x: ((), None), lambda _, __: RepSentinel())


class WanCausalConv3d(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,  # rngs are required for initializing parameters,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      padding: Union[int, Tuple[int, int, int]] = 0,
      use_bias: bool = True,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    padding_tuple = _canonicalize_tuple(padding, 3, "padding")

    self._causal_padding = (
        (0, 0),  # Batch dimension - no padding
        (2 * padding_tuple[0], 0),  # Depth dimension - causal padding (pad only before)
        (padding_tuple[1], padding_tuple[1]),  # Height dimension - symmetric padding
        (padding_tuple[2], padding_tuple[2]),  # Width dimension - symmetric padding
        (0, 0),  # Channel dimension - no padding
    )
    # Store the amount of padding needed *before* the depth dimension for caching logic
    self._depth_padding_before = self._causal_padding[1][0]  # 2 * padding_tuple[0]
    self.mesh = mesh

    # Weight sharding (Kernel is sharded along output channels)
    num_fsdp_devices = mesh.shape["vae_spatial"] if mesh is not None and "vae_spatial" in mesh.shape else 1
    kernel_sharding = (None, None, None, None, None)
    if num_fsdp_devices > 1 and out_channels % num_fsdp_devices == 0:
      kernel_sharding = (None, None, None, None, "vae_spatial")

    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=self.kernel_size,
        strides=self.stride,
        use_bias=use_bias,
        padding="VALID",  # Handle padding manually
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), kernel_sharding),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array, cache_x: Optional[jax.Array] = None, idx=-1) -> jax.Array:
    if self.mesh is not None and "vae_spatial" in self.mesh.shape:
      spatial_sharding = NamedSharding(self.mesh, P("redundant", None, None, "vae_spatial", None))
      if spatial_sharding is not None:
        x = jax.lax.with_sharding_constraint(x, spatial_sharding)

    current_padding = list(self._causal_padding)
    padding_needed = self._depth_padding_before

    if cache_x is not None and padding_needed > 0:
      assert cache_x.shape[0] == x.shape[0] and cache_x.shape[2:] == x.shape[2:]
      cache_len = cache_x.shape[1]
      x = jnp.concatenate([cache_x, x], axis=1)

      padding_needed -= cache_len
      if padding_needed < 0:
        x = x[:, -padding_needed:, ...]
        current_padding[1] = (0, 0)
      else:
        current_padding[1] = (padding_needed, 0)

    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(x, padding_to_apply, mode="constant", constant_values=0.0)
    else:
      x_padded = x

    out = self.conv(x_padded)
    return out


class WanRMS_norm(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      channel_first: bool = True,
      images: bool = True,
      eps: float = 1e-6,
      use_bias: bool = False,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ):
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)
    self.eps = eps
    self.channel_first = channel_first
    self.scale = dim**0.5
    self.dtype = dtype
    # Initialize gamma as parameter
    self.gamma = nnx.Param(jnp.ones(shape, dtype=weights_dtype))
    if use_bias:
      self.bias = nnx.Param(jnp.zeros(shape, dtype=weights_dtype))
    else:
      self.bias = 0

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.astype(self.dtype)
    normalized = jnp.linalg.norm(x, ord=2, axis=(1 if self.channel_first else -1), keepdims=True)
    normalized = x / jnp.maximum(normalized, self.eps)
    normalized = normalized * self.scale * self.gamma
    if self.bias:
      return (normalized + self.bias.value).astype(self.dtype)
    return normalized.astype(self.dtype)


class WanUpsample(nnx.Module):

  def __init__(self, scale_factor: Tuple[float, float], method: str = "nearest"):
    # scale_factor for (H, W)
    # JAX resize works on spatial dims, H, W assuming (N, D, H, W, C) or (N, H, W, C)
    self.scale_factor = scale_factor
    self.method = method

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    in_shape = x.shape
    assert len(in_shape) == 4, "This module only takes tensors with shape of 4."
    n, h, w, c = in_shape
    target_h = int(h * self.scale_factor[0])
    target_w = int(w * self.scale_factor[1])
    if (
        self.method == "nearest"
        and self.scale_factor[0] == int(self.scale_factor[0])
        and self.scale_factor[1] == int(self.scale_factor[1])
    ):
      scale_h = int(self.scale_factor[0])
      scale_w = int(self.scale_factor[1])
      out = jnp.repeat(jnp.repeat(x, scale_h, axis=1), scale_w, axis=2)
    else:
      if self.method == "nearest":
        max_logging.log(
            f"Warning: WanUpsample2D nearest method requested but scale_factor {self.scale_factor} is not integer. Falling back to jax.image.resize."
        )
      out = jax.image.resize(x.astype(jnp.float32), (n, target_h, target_w, c), method=self.method)
      out = out.astype(input_dtype)
    return out


class Identity(nnx.Module):

  def __call__(self, x):
    return x


class ZeroPaddedConv2D(nnx.Module):
  """
  Module for adding padding before conv.
  Currently it does not add any padding.
  """

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    rank = len(kernel_size) if isinstance(kernel_size, (tuple, list)) else 2
    kernel_sharding = (None,) * (rank + 2)
    self.conv = nnx.Conv(
        dim,
        dim,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), kernel_sharding),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x):
    return self.conv(x)


class WanResample(nnx.Module):

  def __init__(
      self,
      dim: int,
      mode: str,
      rngs: nnx.Rngs,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    self.mode = mode
    self.dtype = dtype
    self.time_conv = nnx.data(None)

    if mode == "upsample2d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
    elif mode == "upsample3d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim * 2,
          kernel_size=(3, 1, 1),
          padding=(1, 0, 0),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample2d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample3d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim,
          kernel_size=(3, 1, 1),
          stride=(2, 1, 1),
          padding=(0, 0, 0),
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      self.resample = Identity()

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    # Input x: (N, D, H, W, C), assume C = self.dim
    b, t, h, w, c = x.shape
    assert c == self.dim
    x = x.astype(self.dtype)

    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, RepSentinel())
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and not isinstance(feat_cache[idx], RepSentinel):
            # cache last frame of last two chunk
            cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and isinstance(feat_cache[idx], RepSentinel):
            cache_x = jnp.concatenate([jnp.zeros(cache_x.shape, dtype=cache_x.dtype), cache_x], axis=1)
          if isinstance(feat_cache[idx], RepSentinel):
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1
          x = x.reshape(b, t, h, w, 2, c)
          # x = jnp.stack([x[:, :, :, :, 0, :], x[:, :, :, :, 1, :]], axis=1)
          x = x.transpose(0, 1, 4, 2, 3, 5)
          x = x.reshape(b, t * 2, h, w, c)
    t = x.shape[1]
    x = x.reshape(b * t, h, w, c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(b, t, h_new, w_new, c_new)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, jnp.copy(x))
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -1:, :, :, :])
          x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1

    return x, feat_cache, feat_idx


class WanResidualBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.nonlinearity = get_activation(non_linearity)

    # layers
    self.norm1 = WanRMS_norm(
        dim=in_dim, rngs=rngs, images=False, channel_first=False, dtype=dtype, weights_dtype=weights_dtype
    )
    self.conv1 = WanCausalConv3d(
        rngs=rngs,
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm2 = WanRMS_norm(
        dim=out_dim, rngs=rngs, images=False, channel_first=False, dtype=dtype, weights_dtype=weights_dtype
    )
    self.conv2 = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.conv_shortcut = (
        WanCausalConv3d(
            rngs=rngs,
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        if in_dim != out_dim
        else Identity()
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    # Apply shortcut connection
    h = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv1(x, feat_cache[idx], idx)
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv2(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv2(x)
    x = x + h
    return x, feat_cache, feat_idx


class WanAttentionBlock(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim

    self.norm = WanRMS_norm(rngs=rngs, dim=dim, channel_first=False, dtype=dtype, weights_dtype=weights_dtype)
    self.to_qkv = nnx.Conv(
        in_features=dim,
        out_features=dim * 3,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out")),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )
    self.proj = nnx.Conv(
        in_features=dim,
        out_features=dim,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "conv_in", None)),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  @jax.named_scope("WanVAEAttentionBlock")
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    identity = x
    batch_size, time, height, width, channels = x.shape

    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)

    qkv = self.to_qkv(x)  # Output: (N*T, H, W, C * 3)
    qkv = qkv.reshape(batch_size * time, 1, -1, channels * 3)
    qkv = jnp.transpose(qkv, (0, 1, 3, 2))
    q, k, v = jnp.split(qkv, 3, axis=-2)
    q = jnp.transpose(q, (0, 1, 3, 2))
    k = jnp.transpose(k, (0, 1, 3, 2))
    v = jnp.transpose(v, (0, 1, 3, 2))

    x = jax.nn.dot_product_attention(q, k, v)
    x = jnp.squeeze(x, 1).reshape(batch_size * time, height, width, channels)

    # output projection
    x = self.proj(x)
    # Reshape back
    x = x.reshape(batch_size, time, height, width, channels)

    return x + identity, feat_cache, feat_idx


class WanMidBlock(nnx.Module):

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      num_layers: int = 1,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    resnets = [
        WanResidualBlock(
            in_dim=dim,
            out_dim=dim,
            rngs=rngs,
            dropout=dropout,
            non_linearity=non_linearity,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
    ]
    attentions = []
    for _ in range(num_layers):
      attentions.append(
          WanAttentionBlock(dim=dim, rngs=rngs, mesh=mesh, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
      )
      resnets.append(
          WanResidualBlock(
              in_dim=dim,
              out_dim=dim,
              rngs=rngs,
              dropout=dropout,
              non_linearity=non_linearity,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
    self.attentions = nnx.data(attentions)
    self.resnets = nnx.data(resnets)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    x, feat_cache, feat_idx = self.resnets[0](x, feat_cache, feat_idx)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x, feat_cache, feat_idx = attn(x, feat_cache, feat_idx)
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
    return x, feat_cache, feat_idx


class WanUpBlock(nnx.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      num_res_blocks: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      upsample_mode: Optional[str] = None,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    # Create layers list
    resnets = []
    # Add residual blocks and attention if needed
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
      resnets.append(
          WanResidualBlock(
              in_dim=current_dim,
              out_dim=out_dim,
              dropout=dropout,
              non_linearity=non_linearity,
              rngs=rngs,
              mesh=mesh,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      current_dim = out_dim
    self.resnets = nnx.data(resnets)

    # Add upsampling layer if needed.
    self.upsamplers = nnx.data(None)
    if upsample_mode is not None:
      self.upsamplers = [
          WanResample(
              dim=out_dim,
              mode=upsample_mode,
              rngs=rngs,
              mesh=mesh,
              weights_dtype=weights_dtype,
              dtype=dtype,
              precision=precision,
          )
      ]

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    for resnet in self.resnets:
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)

    if self.upsamplers is not None:
      x, feat_cache, feat_idx = self.upsamplers[0](x, feat_cache, feat_idx)
    return x, feat_cache, feat_idx


class WanEncoder3d(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      attn_scales=[],
      temperal_downsample=[True, True, False],
      dropout=0.0,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_downsample = temperal_downsample
    self.nonlinearity = get_activation(non_linearity)

    # dimensions
    dims = [dim * u for u in [1] + dim_mult]
    scale = 1.0

    # init block
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=3,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # downsample blocks
    self.down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      for _ in range(num_res_blocks):
        self.down_blocks.append(
            WanResidualBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )
        )
        if scale in attn_scales:
          self.down_blocks.append(
              WanAttentionBlock(
                  dim=out_dim, rngs=rngs, mesh=mesh, dtype=dtype, weights_dtype=weights_dtype, precision=precision
              )
          )
        in_dim = out_dim

      # downsample block
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.down_blocks.append(
            WanResample(
                out_dim, mode=mode, rngs=rngs, mesh=mesh, dtype=dtype, weights_dtype=weights_dtype, precision=precision
            )
        )
        scale /= 2.0
    self.down_blocks = nnx.data(self.down_blocks)

    # middle_blocks
    self.mid_block = WanMidBlock(
        dim=out_dim,
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # output blocks
    self.norm_out = WanRMS_norm(
        out_dim, channel_first=False, images=False, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype
    )
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=z_dim,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)
    for layer in self.down_blocks:
      x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)

    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)

    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_out(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_out(x)
    return x, feat_cache, jnp.array(feat_idx, dtype=jnp.int32)


class WanDecoder3d(nnx.Module):
  r"""
  A 3D decoder module.
  Args:
    dim (int): The base number of channels in the first layer.
    z_dim (int): The dimensionality of the latent space.
    dim_mult (list of int): Multipliers for the number of channels in each block.
    num_res_blocks (int): Number of residual blocks in each block.
    attn_scales (list of float): Scales at which to apply attention mechanisms.
    temperal_upsample (list of bool): Whether to upsample temporally in each block.
    dropout (float): Dropout rate for the dropout layers.
    non_linearity (str): Type of non-linearity to use.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult: List[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales=List[float],
      temperal_upsample=[False, True, True],
      dropout=0.0,
      non_linearity: str = "silu",
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_upsample = temperal_upsample

    self.nonlinearity = get_activation(non_linearity)

    # dimensions
    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)

    # init block
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # middle_blocks
    self.mid_block = WanMidBlock(
        dim=dims[0],
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # upsample blocks
    self.up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      if i > 0:
        in_dim = in_dim // 2

      # Determine if we need upsampling
      upsample_mode = None
      if i != len(dim_mult) - 1:
        upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
      # Create and add the upsampling block
      up_block = WanUpBlock(
          in_dim=in_dim,
          out_dim=out_dim,
          num_res_blocks=num_res_blocks,
          dropout=dropout,
          upsample_mode=upsample_mode,
          non_linearity=non_linearity,
          rngs=rngs,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      self.up_blocks.append(up_block)

      # Update scale for next iteration
      if upsample_mode is not None:
        scale *= 2.0
    self.up_blocks = nnx.data(self.up_blocks)

    # output blocks
    self.norm_out = WanRMS_norm(
        dim=out_dim, images=False, rngs=rngs, channel_first=False, dtype=dtype, weights_dtype=weights_dtype
    )
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=3,
        kernel_size=3,
        padding=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)

    ## middle
    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)
    ## upsamples
    for up_block in self.up_blocks:
      x, feat_cache, feat_idx = up_block(x, feat_cache, feat_idx)

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # cache last frame of the last two chunk
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_out(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_out(x)
    return x, feat_cache, jnp.array(feat_idx, dtype=jnp.int32)


class AutoencoderKLWanCache:

  def __init__(self, module):
    self.module = module

    def _count_conv3d(module):
      count = 0
      node_types = nnx.graph.iter_graph([module])
      for _, value in node_types:
        if isinstance(value, WanCausalConv3d):
          count += 1
      return count

    self._conv_num = _count_conv3d(self.module.decoder)
    self._enc_conv_num = _count_conv3d(self.module.encoder)
    self.init_cache()

  def init_cache(self):
    """Resets cache dictionaries and indices"""
    self._feat_map = (None,) * self._conv_num
    # cache encode
    self._enc_feat_map = (None,) * self._enc_conv_num


def _wan_cache_flatten(cache):
  return (cache._feat_map, cache._enc_feat_map), (cache._conv_num, cache._enc_conv_num)


def _wan_cache_unflatten(aux, children):
  conv_num, enc_conv_num = aux
  feat_map, enc_feat_map = children
  # Create a dummy object or one without module reference for JIT internal use
  # We can't easily reconstruct 'module' but we don't need it for init_cache anymore
  # if we store counts in aux.
  # However, __init__ expects module.
  # We will bypass __init__ for unflattening.
  obj = AutoencoderKLWanCache.__new__(AutoencoderKLWanCache)
  obj._conv_num = conv_num
  obj._enc_conv_num = enc_conv_num
  obj._feat_map = feat_map
  obj._enc_feat_map = enc_feat_map
  obj.module = None  # module is not needed inside the trace for the cache logic now
  return obj


tree_util.register_pytree_node(AutoencoderKLWanCache, _wan_cache_flatten, _wan_cache_unflatten)


class AutoencoderKLWan(nnx.Module, FlaxModelMixin, ConfigMixin):

  def __init__(
      self,
      rngs: nnx.Rngs,
      base_dim: int = 96,
      z_dim: int = 16,
      dim_mult: Tuple[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_downsample: List[bool] = [False, True, True],
      dropout: float = 0.0,
      latents_mean: List[float] = [
          -0.7571,
          -0.7089,
          -0.9113,
          0.1075,
          -0.1745,
          0.9653,
          -0.1517,
          1.5508,
          0.4134,
          -0.0715,
          0.5517,
          -0.3632,
          -0.1922,
          -0.9497,
          0.2503,
          -0.2921,
      ],
      latents_std: List[float] = [
          2.8184,
          1.4541,
          2.3275,
          2.6558,
          1.2196,
          1.7708,
          2.6052,
          2.0743,
          3.2687,
          2.1526,
          2.8652,
          1.5579,
          1.6382,
          1.1253,
          2.8251,
          1.9160,
      ],
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      vae_decode_chunk: int = 1,
      vae_encode_chunk: int = 4,
  ):
    self.z_dim = z_dim
    assert vae_decode_chunk >= 1 or vae_decode_chunk == -1, f"vae_decode_chunk must be >= 1 or -1, got {vae_decode_chunk}"
    assert vae_encode_chunk >= 1 or vae_encode_chunk == -1, f"vae_encode_chunk must be >= 1 or -1, got {vae_encode_chunk}"
    self.vae_decode_chunk = vae_decode_chunk
    self.vae_encode_chunk = vae_encode_chunk
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]
    self.temporal_downsample_factor = 2 ** sum(temperal_downsample)

    if self.vae_encode_chunk != -1:
      assert (
          self.vae_encode_chunk % self.temporal_downsample_factor == 0
      ), f"vae_encode_chunk ({self.vae_encode_chunk}) must be a multiple of the temporal downsampling factor ({self.temporal_downsample_factor})."

    self.latents_mean = latents_mean
    self.latents_std = latents_std

    self.encoder = WanEncoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim * 2,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        dropout=dropout,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim * 2,
        out_channels=z_dim * 2,
        kernel_size=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    self.post_quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=z_dim,
        kernel_size=1,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    self.decoder = WanDecoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=self.temporal_upsample,
        dropout=dropout,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.mesh = mesh

  def _encode(self, x: jax.Array, feat_cache: AutoencoderKLWanCache):
    feat_cache.init_cache()
    if x.shape[-1] != 3:
      # reshape channel last for JAX
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert x.shape[-1] == 3, f"Expected input shape (N, D, H, W, 3), got {x.shape}"
    x = x.astype(self.encoder.conv_in.conv.dtype)

    t = x.shape[1]
    CHUNK_SIZE = self.vae_encode_chunk
    if CHUNK_SIZE == -1:
      # Round up the remaining sequence duration to the next multiple of downsampling factor
      T_rest = max(1, t - 1)
      factor = self.temporal_downsample_factor
      CHUNK_SIZE = ((T_rest + factor - 1) // factor) * factor
    # Number of chunk iterations: 1 for init frame, then ceil((t-1)/CHUNK_SIZE) for the rest
    iter_ = 1 + ((t - 1 + CHUNK_SIZE - 1) // CHUNK_SIZE) if t > 1 else 1
    enc_feat_map = feat_cache._enc_feat_map

    spatial_sharding = (
        NamedSharding(self.mesh, P("redundant", None, None, "vae_spatial", None))
        if self.mesh is not None and "vae_spatial" in self.mesh.shape
        else None
    )

    def finalize(out, enc_feat_map):
      feat_cache._enc_feat_map = enc_feat_map
      enc = self.quant_conv(out)
      return enc

    # First iteration (i=0): size 1
    with jax.named_scope("AutoencoderKLWan_encode_chunk_0"):
      chunk_0 = x[:, :1, ...]
      out_0, enc_feat_map, _ = self.encoder(chunk_0, feat_cache=enc_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_0 = jax.lax.with_sharding_constraint(out_0, spatial_sharding)

    if iter_ <= 1:
      return finalize(out_0, enc_feat_map)

    # We must adjust enc_feat_map from None/'Rep'/'zeros' for scan shapes.
    # By running chunk 1 outside the scan, the PyTree shapes will reach their stable state.
    with jax.named_scope("AutoencoderKLWan_encode_chunk_1"):
      chunk_1 = x[:, 1 : (1 + CHUNK_SIZE), ...]
      out_1, enc_feat_map, _ = self.encoder(chunk_1, feat_cache=enc_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_1 = jax.lax.with_sharding_constraint(out_1, spatial_sharding)

    if iter_ <= 2:
      out = jnp.concatenate([out_0, out_1], axis=1)
      if spatial_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, spatial_sharding)
      return finalize(out, enc_feat_map)

    # Prepare the remaining chunks to be scanned over
    x_rest = x[:, 1 + CHUNK_SIZE :, ...]
    T_rest = x_rest.shape[1]

    # Pad T_rest up to the next multiple of CHUNK_SIZE
    pad_amount = (-T_rest) % CHUNK_SIZE
    if pad_amount > 0:
      pad_shape = (x_rest.shape[0], pad_amount, *x_rest.shape[2:])
      x_rest_padded = jnp.concatenate([x_rest, jnp.zeros(pad_shape, dtype=x_rest.dtype)], axis=1)
    else:
      x_rest_padded = x_rest
    T_padded = T_rest + pad_amount
    num_scan_iters = T_padded // CHUNK_SIZE

    x_scannable = x_rest_padded.reshape(
        x_rest_padded.shape[0],
        num_scan_iters,
        CHUNK_SIZE,
        x_rest_padded.shape[2],
        x_rest_padded.shape[3],
        x_rest_padded.shape[4],
    )
    x_scannable = jnp.transpose(x_scannable, (1, 0, 2, 3, 4, 5))

    graphdef, state = nnx.split(self.encoder)

    @jax.named_scope("AutoencoderKLWan_encode_scan_body")
    def scan_fn(carry, chunk):
      current_feat_map = carry
      local_encoder = nnx.merge(graphdef, state)
      if spatial_sharding is not None:
        chunk = jax.lax.with_sharding_constraint(chunk, spatial_sharding)
      out_chunk, next_feat_map, _ = local_encoder(chunk, feat_cache=current_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_chunk = jax.lax.with_sharding_constraint(out_chunk, spatial_sharding)
      next_feat_map = jax.tree_util.tree_map(
          lambda x: jax.lax.with_sharding_constraint(x, spatial_sharding)
          if spatial_sharding is not None and hasattr(x, "shape") and x.ndim == len(spatial_sharding.spec)
          else x,
          next_feat_map,
      )
      return next_feat_map, out_chunk

    enc_feat_map, out_rest = jax.lax.scan(scan_fn, enc_feat_map, x_scannable)
    # out_rest shape: (iter_-2, B, T', H, W, C) -> transpose back
    out_rest = jnp.transpose(out_rest, (1, 0, 2, 3, 4, 5))
    # reshape to (B, (iter_-2)*T', H, W, C)
    out_rest = out_rest.reshape(out_rest.shape[0], -1, out_rest.shape[3], out_rest.shape[4], out_rest.shape[5])
    # Trim padding from the output
    out_rest = out_rest[:, : T_rest // self.temporal_downsample_factor, ...]

    out = jnp.concatenate([out_0, out_1, out_rest], axis=1)
    if spatial_sharding is not None:
      out = jax.lax.with_sharding_constraint(out, spatial_sharding)
    return finalize(out, enc_feat_map)

  @jax.named_scope("AutoencoderKLWan_encode")
  def encode(
      self, x: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxAutoencoderKLOutput, Tuple[FlaxDiagonalGaussianDistribution]]:
    """Encode video into latent distribution."""
    h = self._encode(x, feat_cache)
    posterior = WanDiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior,)
    return FlaxAutoencoderKLOutput(latent_dist=posterior)

  def _decode(
      self, z: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    feat_cache.init_cache()
    iter_ = z.shape[1]
    z = z.astype(self.post_quant_conv.conv.dtype)
    x = self.post_quant_conv(z)

    dec_feat_map = feat_cache._feat_map
    spatial_sharding = (
        NamedSharding(self.mesh, P("redundant", None, None, "vae_spatial", None))
        if self.mesh is not None and "vae_spatial" in self.mesh.shape
        else None
    )

    # First chunk (i=0)
    with jax.named_scope("AutoencoderKLWan_decode_chunk_0"):
      if spatial_sharding is not None:
        chunk_in_0 = jax.lax.with_sharding_constraint(x[:, 0:1, ...], spatial_sharding)
      out_0, dec_feat_map, _ = self.decoder(chunk_in_0, feat_cache=dec_feat_map, feat_idx=0)
      if spatial_sharding is not None:
        out_0 = jax.lax.with_sharding_constraint(out_0, spatial_sharding)

    if iter_ > 1:
      # Run chunk 1 outside scan to properly form the cache shape
      with jax.named_scope("AutoencoderKLWan_decode_chunk_1"):
        if spatial_sharding is not None:
          chunk_in_1 = jax.lax.with_sharding_constraint(x[:, 1:2, ...], spatial_sharding)
        out_chunk_1, dec_feat_map, _ = self.decoder(chunk_in_1, feat_cache=dec_feat_map, feat_idx=0)
        if spatial_sharding is not None:
          out_chunk_1 = jax.lax.with_sharding_constraint(out_chunk_1, spatial_sharding)

      out_1 = out_chunk_1
      out_list = [out_0, out_1]

      if iter_ > 2:
        x_rest = x[:, 2:, ...]
        T_rest = x_rest.shape[1]
        K = self.vae_decode_chunk
        if K == -1:
          K = max(1, T_rest)

        # Pad T_rest up to the next multiple of K so the scan has uniform chunks.
        # This avoids data-dependent branches and dynamic shapes under JIT.
        pad_amount = (-T_rest) % K  # 0 when already divisible
        if pad_amount > 0:
          pad_shape = (x_rest.shape[0], pad_amount, *x_rest.shape[2:])
          x_rest_padded = jnp.concatenate([x_rest, jnp.zeros(pad_shape, dtype=x_rest.dtype)], axis=1)
        else:
          x_rest_padded = x_rest
        T_padded = T_rest + pad_amount
        num_chunks = T_padded // K

        # Reshape to (num_chunks, B, K, H, W, C) for scan
        x_scannable = x_rest_padded.reshape(
            x_rest_padded.shape[0], num_chunks, K, x_rest_padded.shape[2], x_rest_padded.shape[3], x_rest_padded.shape[4]
        )
        x_scannable = jnp.transpose(x_scannable, (1, 0, 2, 3, 4, 5))

        graphdef, state = nnx.split(self.decoder)

        @jax.named_scope("AutoencoderKLWan_decode_scan_body")
        def scan_fn(carry, chunk_in):
          current_feat_map = carry
          local_decoder = nnx.merge(graphdef, state)
          if spatial_sharding is not None:
            chunk_in = jax.lax.with_sharding_constraint(chunk_in, spatial_sharding)
          out_chunk, next_feat_map, _ = local_decoder(chunk_in, feat_cache=current_feat_map, feat_idx=0)
          if spatial_sharding is not None:
            out_chunk = jax.lax.with_sharding_constraint(out_chunk, spatial_sharding)
          next_feat_map = jax.tree_util.tree_map(
              lambda x: jax.lax.with_sharding_constraint(x, spatial_sharding)
              if spatial_sharding is not None and hasattr(x, "shape") and x.ndim == len(spatial_sharding.spec)
              else x,
              next_feat_map,
          )
          return next_feat_map, out_chunk

        dec_feat_map, out_rest = jax.lax.scan(scan_fn, dec_feat_map, x_scannable)
        out_rest = jnp.transpose(out_rest, (1, 0, 2, 3, 4, 5))
        out_rest = out_rest.reshape(out_rest.shape[0], -1, out_rest.shape[3], out_rest.shape[4], out_rest.shape[5])
        # Trim padding from the output
        out_rest = out_rest[:, : T_rest * self.temporal_downsample_factor, ...]
        out_list.append(out_rest)

      out = jnp.concatenate(out_list, axis=1)
      if spatial_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, spatial_sharding)
    else:
      out = out_0

    feat_cache._feat_map = dec_feat_map

    out = jnp.clip(out, min=-1.0, max=1.0)
    if not return_dict:
      return (out,)

    return FlaxDecoderOutput(sample=out)

  @jax.named_scope("AutoencoderKLWan_decode")
  def decode(
      self, z: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True
  ) -> Union[FlaxDecoderOutput, jax.Array]:
    if z.shape[-1] != self.z_dim:
      # reshape channel last for JAX
      z = jnp.transpose(z, (0, 2, 3, 4, 1))
      assert z.shape[-1] == self.z_dim, f"Expected input shape (N, D, H, W, {self.z_dim}, got {z.shape}"
    decoded = self._decode(z, feat_cache).sample
    if not return_dict:
      return (decoded,)
    return FlaxDecoderOutput(sample=decoded)
