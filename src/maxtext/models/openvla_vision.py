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

"""OpenVLA (Prismatic `dinosiglip-vit-so-224px`) fused vision backbone and projector.

Reference: https://huggingface.co/openvla/openvla-7b (modeling_prismatic.py, timm==0.9.10).

The image is encoded by two timm ViTs that run side by side on the same 224x224 image, each with
its own normalization (the two normalized copies are channel-stacked by the preprocessor into a
single [H, W, 6] array):
  * DINOv2 ViT-L/14 with 4 register tokens (`vit_large_patch14_reg4_dinov2.lvd142m`), LayerScale.
  * SigLIP So400m/14 (`vit_so400m_patch14_siglip_224`), no class token.
Prismatic takes the patch tokens of the *second-to-last* block of each ViT (no final norm, no
attention pooling head), concatenates them on the feature axis (1024 + 1152 = 2176) and projects
them into the LLM embedding space with a 3-layer GELU MLP (`no-align+fused-gelu-mlp`).

Only the blocks that contribute to the output are instantiated (23 / 26 blocks); the last block,
final norm and SigLIP attention-pool weights of the HF checkpoint are unused by OpenVLA.
"""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.layers.linears import DenseGeneral


@dataclasses.dataclass(frozen=True)
class ViTSpec:
  """Static architecture of one timm ViT tower."""

  hidden_size: int
  mlp_dim: int
  num_heads: int
  num_layers: int  # number of blocks actually used (second-to-last block output)
  num_reg_tokens: int
  use_cls_token: bool
  use_layer_scale: bool
  patch_size: int = 14
  image_size: int = 224
  num_channels: int = 3
  layer_norm_eps: float = 1e-6

  @property
  def num_patches(self) -> int:
    return (self.image_size // self.patch_size) ** 2

  @property
  def num_prefix_tokens(self) -> int:
    return int(self.use_cls_token) + self.num_reg_tokens


# `vit_large_patch14_reg4_dinov2.lvd142m` has 24 blocks; Prismatic uses the output of block 22.
DINOV2_SPEC = ViTSpec(
    hidden_size=1024,
    mlp_dim=4096,
    num_heads=16,
    num_layers=23,
    num_reg_tokens=4,
    use_cls_token=True,
    use_layer_scale=True,
)
# `vit_so400m_patch14_siglip_224` has 27 blocks; Prismatic uses the output of block 25.
SIGLIP_SPEC = ViTSpec(
    hidden_size=1152,
    mlp_dim=4304,
    num_heads=16,
    num_layers=26,
    num_reg_tokens=0,
    use_cls_token=False,
    use_layer_scale=False,
)
OPENVLA_VISION_DIM = DINOV2_SPEC.hidden_size + SIGLIP_SPEC.hidden_size  # 2176
OPENVLA_NUM_IMAGE_TOKENS = DINOV2_SPEC.num_patches  # 256


class ViTAttention(nnx.Module):
  """timm `Attention`: fused qkv projection with bias, full (non-causal) softmax attention."""

  def __init__(self, config: Config, spec: ViTSpec, *, rngs: nnx.Rngs):
    self.num_heads = spec.num_heads
    self.head_dim = spec.hidden_size // spec.num_heads
    self.dtype = config.dtype_mm
    self.precision = config.matmul_precision
    self.qkv = DenseGeneral(
        in_features_shape=spec.hidden_size,
        out_features_shape=(3, self.num_heads, self.head_dim),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.proj = DenseGeneral(
        in_features_shape=(self.num_heads, self.head_dim),
        out_features_shape=spec.hidden_size,
        axis=(-2, -1),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    qkv = self.qkv(x)  # [B, S, 3, H, D]
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = q * (self.head_dim**-0.5)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision=self.precision, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", probs, v, precision=self.precision)
    return self.proj(out)


class ViTMlp(nnx.Module):
  """timm `Mlp`: fc1 -> exact GELU -> fc2."""

  def __init__(self, config: Config, spec: ViTSpec, *, rngs: nnx.Rngs):
    self.fc1 = DenseGeneral(
        in_features_shape=spec.hidden_size,
        out_features_shape=spec.mlp_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.fc2 = DenseGeneral(
        in_features_shape=spec.mlp_dim,
        out_features_shape=spec.hidden_size,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class ViTBlock(nnx.Module):
  """timm pre-norm ViT `Block`, with optional LayerScale (DINOv2)."""

  def __init__(self, config: Config, spec: ViTSpec, *, rngs: nnx.Rngs):
    self.use_layer_scale = spec.use_layer_scale
    self.norm1 = nnx.LayerNorm(
        num_features=spec.hidden_size,
        epsilon=spec.layer_norm_eps,
        dtype=config.dtype_mm,
        param_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.attn = ViTAttention(config, spec, rngs=rngs)
    self.norm2 = nnx.LayerNorm(
        num_features=spec.hidden_size,
        epsilon=spec.layer_norm_eps,
        dtype=config.dtype_mm,
        param_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.mlp = ViTMlp(config, spec, rngs=rngs)
    if self.use_layer_scale:
      self.ls1 = nnx.Param(jnp.ones((spec.hidden_size,), dtype=config.weight_dtype))
      self.ls2 = nnx.Param(jnp.ones((spec.hidden_size,), dtype=config.weight_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    y = self.attn(self.norm1(x))
    if self.use_layer_scale:
      y = y * self.ls1[...].astype(y.dtype)
    x = x + y
    y = self.mlp(self.norm2(x))
    if self.use_layer_scale:
      y = y * self.ls2[...].astype(y.dtype)
    return x + y


class OpenVLAViT(nnx.Module):
  """A timm VisionTransformer truncated to `spec.num_layers` blocks, returning patch tokens only."""

  def __init__(self, config: Config, spec: ViTSpec, *, rngs: nnx.Rngs):
    self.spec = spec
    self.dtype = config.dtype_mm
    init = nnx.initializers.normal(stddev=0.02)
    self.patch_embed = nnx.Conv(
        in_features=spec.num_channels,
        out_features=spec.hidden_size,
        kernel_size=(spec.patch_size, spec.patch_size),
        strides=(spec.patch_size, spec.patch_size),
        padding="VALID",
        use_bias=True,
        dtype=config.dtype_mm,
        param_dtype=config.weight_dtype,
        precision=config.matmul_precision,
        rngs=rngs,
    )
    self.pos_embed = nnx.Param(init(rngs.params(), (1, spec.num_patches, spec.hidden_size), config.weight_dtype))
    if spec.use_cls_token:
      self.cls_token = nnx.Param(init(rngs.params(), (1, 1, spec.hidden_size), config.weight_dtype))
    if spec.num_reg_tokens:
      self.reg_token = nnx.Param(init(rngs.params(), (1, spec.num_reg_tokens, spec.hidden_size), config.weight_dtype))
    for i in range(spec.num_layers):
      setattr(self, f"blocks_{i}", ViTBlock(config, spec, rngs=rngs))

  def __call__(self, images: jax.Array) -> jax.Array:
    """images: [B, H, W, 3] normalized pixels -> [B, num_patches, hidden_size]."""
    b = images.shape[0]
    x = self.patch_embed(images.astype(self.dtype))  # [B, H/p, W/p, D]
    x = jnp.reshape(x, (b, -1, self.spec.hidden_size))
    # timm `_pos_embed`: DINOv2 (no_embed_class=True) adds pos-embed to patches before prepending prefix
    # tokens; SigLIP has no prefix tokens, so the same order applies to both towers.
    x = x + self.pos_embed[...].astype(x.dtype)
    prefix = []
    if self.spec.use_cls_token:
      prefix.append(jnp.broadcast_to(self.cls_token[...].astype(x.dtype), (b, 1, self.spec.hidden_size)))
    if self.spec.num_reg_tokens:
      prefix.append(
          jnp.broadcast_to(self.reg_token[...].astype(x.dtype), (b, self.spec.num_reg_tokens, self.spec.hidden_size))
      )
    if prefix:
      x = jnp.concatenate(prefix + [x], axis=1)
    for i in range(self.spec.num_layers):
      x = getattr(self, f"blocks_{i}")(x)
    return x[:, self.spec.num_prefix_tokens :]


class OpenVLAVisionEncoder(nnx.Module):
  """Prismatic fused DINOv2 + SigLIP backbone: [B, (N,) H, W, 6] -> [B, N, 256, 2176]."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.featurizer = OpenVLAViT(config, DINOV2_SPEC, rngs=rngs)
    self.fused_featurizer = OpenVLAViT(config, SIGLIP_SPEC, rngs=rngs)

  def __call__(self, inputs: jax.Array, deterministic: bool = True) -> jax.Array:
    del deterministic  # no dropout in the vision towers
    if inputs.ndim == 4:
      inputs = inputs[:, None]
    b, n, h, w, c = inputs.shape
    x = jnp.reshape(inputs, (b * n, h, w, c))
    dino = self.featurizer(x[..., :3])
    siglip = self.fused_featurizer(x[..., 3:])
    patches = jnp.concatenate([dino, siglip], axis=-1)
    return jnp.reshape(patches, (b, n, patches.shape[1], patches.shape[2]))


class OpenVLAProjector(nnx.Module):
  """Prismatic `fused-gelu-mlp` projector: 2176 -> 8704 -> GELU -> 4096 -> GELU -> 4096."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    dims = (OPENVLA_VISION_DIM, 4 * OPENVLA_VISION_DIM, config.emb_dim, config.emb_dim)
    for i in range(3):
      setattr(
          self,
          f"fc{i + 1}",
          DenseGeneral(
              in_features_shape=dims[i],
              out_features_shape=dims[i + 1],
              dtype=config.dtype_mm,
              weight_dtype=config.weight_dtype,
              use_bias=True,
              matmul_precision=config.matmul_precision,
              kernel_axes=("embed", "mlp"),
              rngs=rngs,
          ),
      )

  def __call__(self, x: jax.Array) -> jax.Array:
    x = jax.nn.gelu(self.fc1(x), approximate=False)
    x = jax.nn.gelu(self.fc2(x), approximate=False)
    return self.fc3(x)
