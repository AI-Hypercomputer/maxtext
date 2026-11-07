# Copyright 2026 Google LLC
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

"""OLMo 3.5 / Molmo2 vision tower, connector, splicing, and checkpoint conversion.

Implements the exact vision stack from ``allenai/OLMo-core`` (#916):
  1. ``OLMo35VisionEncoder``: ``SigLIP2-SO400M/14-378`` truncated to 25 blocks
     (no CLS token, no pre/post-LN, C-first ``14x14x3 = 588`` patch projection
     with bias, learned ``[729, 1152]`` positional embedding, ``LayerNorm(eps=1e-6)``
     with bias, 16x72 MHA with bias, 4304 ``gelu_pytorch_tanh`` MLP with bias),
     concatenating hidden states from blocks ``(24, 18)`` -> ``[B, N_crops * 729, 2304]``.
     Supports per-block remat, crop chunking, and automatic Context-Parallel
     crop sharding (each CP rank encodes ``N_crops // CP`` crops before AllGather).
  2. ``OLMo35VisionConnector``: gathers 4-patch groups via ``pooled_patches_idx``
     (``-1`` slots masked), pools via ``attention_meanq`` (masked mean query,
     FP32 ``_few_key_attention`` with finite mask fill), and projects via a
     bias-free SwiGLU MLP (``1152 -> 9728 -> emb_dim``).
  3. ``merge_olmo35_image_embeddings``: adds (``+=``) connector features at
     ``<im_patch>`` positions *after* ``sqrt(emb_dim)`` and ``embedding_norm``,
     supporting Context Parallelism via global prefix counts across ``context``.
"""

from __future__ import annotations

import functools
from typing import Any

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import Config
from maxtext.layers import linears
from maxtext.utils import sharding

# Default Molmo2 / Dolma2 special token IDs when appended to dolma2-tokenizer (100,278 base -> 100,352 padded):
# 100278: <im_start>, 100279: <im_end>, 100280: <im_patch>, 100281: <im_col>, 100282: <low_res_im_start>, 100283: <|image|>
DEFAULT_OLMO_IMAGE_PATCH_TOKEN_ID = 100280


def _normal_kernel_init(stddev: float = 0.02):
  """MaxText ``DenseGeneral`` initializer accepting ``(key, shape, dtype, in_axis, out_axis)``."""
  base_init = jax.nn.initializers.normal(stddev=stddev)

  def _init(key, shape, dtype=jnp.float32, *_):
    return base_init(key, shape, dtype)

  return _init


def _vit_kernel_axes(in_f: int, out_f: int, config: Config, mesh: Mesh | None) -> tuple[str | None, str | None]:
  """Selects divisible logical sharding axes for ViT and Connector ``DenseGeneral`` kernels."""
  if mesh is None:
    return (None, None)
  for axis_name in ("vit_embed_cp", "vit_embed"):
    spec = sharding.logical_to_mesh_axes((axis_name,), mesh, rules=config.logical_axis_rules)
    if spec is None or not spec.partitions or spec.partitions[0] is None:
      continue
    phys_axes = sharding.mesh_axes_for_dim(spec.partitions[0])
    prod = 1
    for axis in phys_axes:
      prod *= mesh.shape.get(axis, 1)
    if prod <= 1:
      continue
    if in_f % prod == 0:
      return (axis_name, None)
    if out_f % prod == 0:
      return (None, axis_name)
  return (None, None)


def _batch_mesh_spec_entry(config: Config, mesh: Mesh | None):
  """Returns active ``mesh`` axis (or tuple of axes) for ``activation_batch``, or ``None``."""
  if mesh is None:
    return None
  spec = sharding.logical_to_mesh_axes(("activation_batch",), mesh, rules=config.logical_axis_rules)
  if spec is None or not spec.partitions or not spec.partitions[0]:
    return None
  return spec.partitions[0]


class _ViTDense(linears.DenseGeneral):
  """``DenseGeneral`` that explicitly FSDP-gathers sharded ViT/Connector weights to ``P()`` before ``dot_general``.

  Prevents XLA GSPMD from propagating ``vit_embed`` / ``vit_embed_cp`` weight sharding onto activation
  feature dimensions (which would otherwise trigger involuntary full-tensor replication across ``context * expert``).
  """

  def __call__(self, inputs: jax.Array, *args, **kwargs) -> jax.Array:
    if self.mesh is None or self.quant is not None or args or kwargs:
      return super().__call__(inputs, *args, **kwargs)
    x = jnp.asarray(inputs, self.dtype)
    k = jnp.asarray(self.kernel[...], self.dtype)
    rep = NamedSharding(self.mesh, PartitionSpec())
    k = jax.lax.with_sharding_constraint(k, rep)
    prec = jax.lax.Precision(self.matmul_precision)
    y = jax.lax.dot_general(x, k, (((x.ndim - 1,), (0,)), ((), ())), precision=prec)
    if self.bias is not None:
      b = jax.lax.with_sharding_constraint(jnp.asarray(self.bias[...], self.dtype), rep)
      y = y + b
    return y


def patchify_hwc_to_olmo_cfirst(images_hwc: jax.Array, patch_size: int = 14) -> jax.Array:
  """Converts ``[B, N_crops, H, W, 3]`` images into OLMo-core's C-first patches ``[B, N_crops, (H/p)*(W/p), 3*p*p]``."""
  b, n_crops, h, w, c = images_hwc.shape
  p = patch_size
  if h % p != 0 or w % p != 0:
    raise ValueError(f"Image spatial dims ({h}, {w}) must be divisible by patch_size ({p}).")
  gh, gw = h // p, w // p
  # OLMo-core flattens each patch in PyTorch NCHW order: (3, p, p) -> 3 * p * p.
  patches = images_hwc.reshape(b, n_crops, gh, p, gw, p, c)
  patches = patches.transpose(0, 1, 2, 4, 6, 3, 5)
  return patches.reshape(b, n_crops, gh * gw, c * p * p)


class _ViTLayerNorm(nnx.Module):
  """FP32 LayerNorm with weight and bias matching PyTorch ``nn.LayerNorm(eps=1e-6)``."""

  def __init__(self, dim: int, eps: float, dtype: Any, weight_dtype: Any, *, rngs: nnx.Rngs):
    self.dim = dim
    self.eps = eps
    self.dtype = dtype
    self.scale = nnx.Param(jnp.ones((dim,), dtype=weight_dtype))
    self.bias = nnx.Param(jnp.zeros((dim,), dtype=weight_dtype))

  def __call__(self, x: jax.Array) -> jax.Array:
    x_f = x.astype(jnp.float32)
    mean = jnp.mean(x_f, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f - mean), axis=-1, keepdims=True)
    normed = (x_f - mean) * jax.lax.rsqrt(var + self.eps)
    out = normed * self.scale[...].astype(jnp.float32) + self.bias[...].astype(jnp.float32)
    return out.astype(self.dtype)


class OLMo35ViTAttention(nnx.Module):
  """Bidirectional multi-head self-attention for SigLIP2 ViT blocks."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.embed_dim = int(config.hidden_size_for_vit)
    self.num_heads = int(config.num_attention_heads_for_vit)
    if self.embed_dim % self.num_heads != 0:
      raise ValueError(f"hidden_size_for_vit ({self.embed_dim}) must be divisible by heads ({self.num_heads}).")
    self.head_dim = self.embed_dim // self.num_heads
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=True,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.wq = _lin(self.embed_dim, self.embed_dim)
    self.wk = _lin(self.embed_dim, self.embed_dim)
    self.wv = _lin(self.embed_dim, self.embed_dim)
    self.wo = _lin(self.embed_dim, self.embed_dim)

  def __call__(self, x: jax.Array) -> jax.Array:
    *prefix, n_patches, _ = x.shape
    h, d = self.num_heads, self.head_dim
    q = self.wq(x).reshape(*prefix, n_patches, h, d)
    k = self.wk(x).reshape(*prefix, n_patches, h, d)
    v = self.wv(x).reshape(*prefix, n_patches, h, d)
    scale = d**-0.5
    scores = jnp.einsum(
        "...qhd,...khd->...hqk",
        q.astype(self.dtype),
        k.astype(self.dtype),
        precision=self.config.matmul_precision,
        preferred_element_type=jnp.float32,
    ) * scale
    probs = jax.nn.softmax(scores, axis=-1).astype(self.dtype)
    attn_out = jnp.einsum(
        "...hqk,...khd->...qhd",
        probs,
        v.astype(self.dtype),
        precision=self.config.matmul_precision,
        preferred_element_type=jnp.float32,
    ).astype(self.dtype)
    return self.wo(attn_out.reshape(*prefix, n_patches, self.embed_dim))


class OLMo35ViTMLP(nnx.Module):
  """Two-layer ViT MLP with ``gelu_pytorch_tanh`` activation."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    emb_dim = int(config.hidden_size_for_vit)
    mlp_dim = int(config.intermediate_size_for_vit)
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)
    self.w1 = _ViTDense(
        in_features_shape=emb_dim,
        out_features_shape=mlp_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=init,
        kernel_axes=_vit_kernel_axes(emb_dim, mlp_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.w2 = _ViTDense(
        in_features_shape=mlp_dim,
        out_features_shape=emb_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=init,
        kernel_axes=_vit_kernel_axes(mlp_dim, emb_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    h = self.w1(x)
    # PyTorch F.gelu(..., approximate="tanh") computes tanh approximation in input dtype / fp32.
    h = jax.nn.gelu(h.astype(jnp.float32), approximate=True).astype(self.dtype)
    return self.w2(h)


class OLMo35ViTBlock(nnx.Module):
  """Pre-LN SigLIP2 ViT residual block."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    dim = int(config.hidden_size_for_vit)
    eps = float(config.normalization_layer_epsilon)
    self.attn_norm = _ViTLayerNorm(dim, eps, config.dtype_mm, config.weight_dtype, rngs=rngs)
    self.attn = OLMo35ViTAttention(config, mesh, rngs=rngs)
    self.ffn_norm = _ViTLayerNorm(dim, eps, config.dtype_mm, config.weight_dtype, rngs=rngs)
    self.ffn = OLMo35ViTMLP(config, mesh, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x + self.attn(self.attn_norm(x))
    x = x + self.ffn(self.ffn_norm(x))
    return x


class OLMo35VisionEncoder(nnx.Module):
  """SigLIP2-SO400M/14-378 25-block vision tower returning concatenated multi-layer features."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.patch_size = int(config.patch_size_for_vit)
    img_size = config.image_size_for_vit
    if isinstance(img_size, list):
      img_h, img_w = int(img_size[0]), int(img_size[1])
    else:
      img_h = img_w = int(img_size)
    self.num_patches = (img_h // self.patch_size) * (img_w // self.patch_size)
    self.patch_pixels = 3 * self.patch_size * self.patch_size
    self.embed_dim = int(config.hidden_size_for_vit)
    self.num_layers = int(config.num_hidden_layers_for_vit)
    self.feature_layers = tuple(int(i) for i in config.olmo_vit_feature_layers)
    for idx in self.feature_layers:
      if not 0 <= idx < self.num_layers:
        raise ValueError(f"olmo_vit_feature_layers entry {idx} out of range [0, {self.num_layers}).")
    self.dtype = config.dtype_mm

    self.patch_embedding = _ViTDense(
        in_features_shape=self.patch_pixels,
        out_features_shape=self.embed_dim,
        use_bias=True,
        dtype=self.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=_normal_kernel_init(0.02),
        kernel_axes=_vit_kernel_axes(self.patch_pixels, self.embed_dim, config, mesh),
        matmul_precision=config.matmul_precision,
        mesh=mesh,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    pos_init = jax.nn.initializers.normal(stddev=self.embed_dim**-0.5)
    self.positional_embedding = nnx.Param(
        pos_init(rngs.params(), (self.num_patches, self.embed_dim), config.weight_dtype)
    )
    for i in range(self.num_layers):
      setattr(self, f"blocks_{i}", OLMo35ViTBlock(config, mesh, rngs=rngs))

  def _encode_flat_crops(self, crops: jax.Array, act_pspec: PartitionSpec | None = None) -> jax.Array:
    """Encodes ``[..., M, num_patches, patch_pixels]`` into ``[..., M, num_patches, len(feature_layers) * embed_dim]``."""
    x = self.patch_embedding(crops.astype(self.dtype))
    x = x + self.positional_embedding[...].astype(self.dtype)
    if act_pspec is not None and self.mesh is not None:
      x = sharding.maybe_shard_with_pspec(x, act_pspec, self.mesh, self.config.shard_mode)
    remat = bool(self.config.olmo_vit_remat_blocks)
    wanted = set(self.feature_layers)
    saved: dict[int, jax.Array] = {}
    for i in range(self.num_layers):
      block = getattr(self, f"blocks_{i}")
      if remat:
        graphdef, state = nnx.split(block)

        def _run_block(st, inp, gd=graphdef):
          mod = nnx.merge(gd, st)
          out_b = mod(inp)
          if act_pspec is not None and self.mesh is not None:
            out_b = sharding.maybe_shard_with_pspec(out_b, act_pspec, self.mesh, self.config.shard_mode)
          return out_b

        x = jax.checkpoint(_run_block, prevent_cse=True)(state, x)
      else:
        x = block(x)
        if act_pspec is not None and self.mesh is not None:
          x = sharding.maybe_shard_with_pspec(x, act_pspec, self.mesh, self.config.shard_mode)
      if i in wanted:
        saved[i] = x
    return jnp.concatenate([saved[i] for i in self.feature_layers], axis=-1)

  def _encode_crops_maybe_chunked(self, crops: jax.Array, *, cp_sharded: bool = False) -> jax.Array:
    """Encodes ``[B, N_crops, P, D_in]`` while preserving batch (and optional CP) sharding across scan."""
    b_dim, n_crops, p, d_in = crops.shape
    chunk = int(self.config.olmo_vit_crop_chunk_size)
    batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if (cp_sharded and self.mesh is not None) else 1
    crops_per_rank = n_crops // cp_size if cp_size > 1 else n_crops

    if chunk > 0 and crops_per_rank > chunk and crops_per_rank % chunk == 0:
      n_chunks = crops_per_rank // chunk
      remat = bool(self.config.olmo_vit_remat_blocks)
      graphdef, state = nnx.split(self)
      if cp_size > 1:
        reshaped = crops.reshape(b_dim, cp_size, n_chunks, chunk, p, d_in)
        reshaped = sharding.maybe_shard_with_pspec(
            reshaped,
            PartitionSpec(batch_axes, cp_axis, None, None, None, None),
            self.mesh,
            self.config.shard_mode,
        )
        chunked = jnp.transpose(reshaped, (2, 0, 1, 3, 4, 5))
        chunk_pspec = PartitionSpec(batch_axes, cp_axis, None, None, None)

        def _scan_body_cp(carry, crop_chunk):
          mod = nnx.merge(graphdef, carry)
          crop_chunk = sharding.maybe_shard_with_pspec(
              crop_chunk, chunk_pspec, self.mesh, self.config.shard_mode
          )
          return carry, mod._encode_flat_crops(crop_chunk, act_pspec=chunk_pspec)  # pylint: disable=protected-access

        scan_fn_cp = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body_cp) if remat else _scan_body_cp
        _, out_chunks = jax.lax.scan(scan_fn_cp, state, chunked)
        out = jnp.transpose(out_chunks, (1, 2, 0, 3, 4, 5)).reshape(b_dim, n_crops * p, -1)
      else:
        # Chunk along axis 1 (n_crops) and place n_chunks at axis 0 for scan so axis 1 (B)
        # remains batch-sharded across devices inside every scan iteration.
        chunked = jnp.swapaxes(crops.reshape(b_dim, n_chunks, chunk, p, d_in), 0, 1)
        chunk_pspec = PartitionSpec(batch_axes, None, None, None) if self.mesh is not None else None

        def _scan_body(carry, crop_chunk):
          mod = nnx.merge(graphdef, carry)
          return carry, mod._encode_flat_crops(crop_chunk, act_pspec=chunk_pspec)  # pylint: disable=protected-access

        scan_fn = functools.partial(jax.checkpoint, prevent_cse=False)(_scan_body) if remat else _scan_body
        _, out_chunks = jax.lax.scan(scan_fn, state, chunked)
        out = jnp.swapaxes(out_chunks, 0, 1).reshape(b_dim, n_crops * p, -1)
    else:
      flat_pspec = (
          PartitionSpec(batch_axes, cp_axis if cp_size > 1 else None, None, None)
          if self.mesh is not None
          else None
      )
      out = self._encode_flat_crops(crops, act_pspec=flat_pspec).reshape(b_dim, n_crops * p, -1)
    return out

  def __call__(self, input_images: jax.Array, deterministic: bool = True) -> jax.Array:
    del deterministic
    if input_images.ndim == 5:
      patches = patchify_hwc_to_olmo_cfirst(input_images, self.patch_size)
    elif input_images.ndim == 4:
      patches = input_images
    else:
      raise ValueError(
          f"OLMo35VisionEncoder expects [B, N_crops, P, 588] or [B, N_crops, H, W, 3], got {input_images.shape}."
      )

    cp_axis = self.config.context_sharding
    cp_size = self.mesh.shape.get(cp_axis, 1) if self.mesh is not None else 1
    n_crops = patches.shape[1]
    if bool(self.config.olmo_vit_cp_shard_crops) and cp_size > 1 and n_crops % cp_size == 0:
      # Every CP rank shares the same sequence's `n_crops` images. Shard `patches` on `cp_axis`
      # in GSPMD so each CP rank encodes `n_crops // cp_size` crops while ViT weights stay
      # FSDP/EP-sharded (`vit_embed` / `vit_embed_cp`), then AllGather features across `cp_axis`.
      batch_axes = _batch_mesh_spec_entry(self.config, self.mesh)
      patches = sharding.maybe_shard_with_pspec(
          patches, PartitionSpec(batch_axes, cp_axis, None, None), self.mesh, self.config.shard_mode
      )
      out = self._encode_crops_maybe_chunked(patches, cp_sharded=True)
      out = sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, cp_axis, None), self.mesh, self.config.shard_mode
      )
      return sharding.maybe_shard_with_pspec(
          out, PartitionSpec(batch_axes, None, None), self.mesh, self.config.shard_mode
      )

    if self.mesh is not None:
      patches = sharding.maybe_shard_with_logical(
          patches, ("activation_batch", None, None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
    return self._encode_crops_maybe_chunked(patches)


def _few_key_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    valid: jax.Array | None,
) -> jax.Array:
  """FP32 group pooling attention matching ``olmo_core.nn.vision.connector._few_key_attention``.

  Args:
    q: ``[..., H, D]``
    k: ``[..., P, H, D]``
    v: ``[..., P, H, D]``
    valid: ``[..., P]`` bool, ``True`` where key is valid (``pooled_patches_idx >= 0``).
  Returns:
    ``[..., H, D]`` in ``q.dtype``, zero on groups where all slots are ``-1``.
  """
  scores = jnp.sum(q.astype(jnp.float32)[..., None, :, :] * k.astype(jnp.float32), axis=-1) * (q.shape[-1] ** -0.5)
  if valid is not None:
    # Finite fill (finfo.min) keeps softmax and its backward finite on all-(-1) padding rows.
    fmin = jnp.finfo(jnp.float32).min
    scores = jnp.where(valid[..., None], scores, fmin)
  probs = jax.nn.softmax(scores, axis=-2)
  if valid is not None:
    probs = jnp.where(valid[..., None], probs, 0.0)
  return jnp.sum(probs[..., None] * v.astype(jnp.float32), axis=-3).astype(q.dtype)


class OLMo35PoolingCrossAttention(nnx.Module):
  """``attention_meanq`` 2x2 patch-group cross-attention pooling."""

  def __init__(self, config: Config, mesh: Mesh, input_dim: int, *, rngs: nnx.Rngs):
    self.config = config
    self.emb_dim = int(config.hidden_size_for_vit)
    self.num_heads = int(config.num_attention_heads_for_vit)
    self.head_dim = self.emb_dim // self.num_heads
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=True,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.wq = _lin(input_dim, self.emb_dim)
    self.wk = _lin(input_dim, self.emb_dim)
    self.wv = _lin(input_dim, self.emb_dim)
    self.wo = _lin(self.emb_dim, self.emb_dim)

  def __call__(self, query: jax.Array, context: jax.Array, valid: jax.Array | None) -> jax.Array:
    """Pools ``context`` ``[..., P, input_dim]`` with ``query`` ``[..., input_dim]`` -> ``[..., emb_dim]``."""
    *prefix, p, _ = context.shape
    q = self.wq(query).reshape(*prefix, self.num_heads, self.head_dim)
    k = self.wk(context).reshape(*prefix, p, self.num_heads, self.head_dim)
    v = self.wv(context).reshape(*prefix, p, self.num_heads, self.head_dim)
    out = _few_key_attention(q, k, v, valid)
    return self.wo(out.reshape(*prefix, self.emb_dim))


class OLMo35ConnectorMLP(nnx.Module):
  """Bias-free SwiGLU projector ``w2(silu(w1(x)) * w3(x))`` into LM ``emb_dim``."""

  def __init__(self, config: Config, mesh: Mesh, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
    self.dtype = config.dtype_mm
    init = _normal_kernel_init(0.02)

    def _lin(in_f: int, out_f: int):
      return _ViTDense(
          in_features_shape=in_f,
          out_features_shape=out_f,
          use_bias=False,
          dtype=self.dtype,
          weight_dtype=config.weight_dtype,
          kernel_init=init,
          kernel_axes=_vit_kernel_axes(in_f, out_f, config, mesh),
          matmul_precision=config.matmul_precision,
          mesh=mesh,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.w1 = _lin(input_dim, hidden_dim)
    self.w3 = _lin(input_dim, hidden_dim)
    self.w2 = _lin(hidden_dim, output_dim)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jax.nn.silu(self.w1(x).astype(jnp.float32)).astype(self.dtype)
    return self.w2(gate * self.w3(x))


class OLMo35VisionConnector(nnx.Module):
  """Molmo2 / OLMo 3.5 gather-pool-project vision connector."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.dtype = config.dtype_mm
    num_layers_used = len(config.olmo_vit_feature_layers)
    pooling_input_dim = num_layers_used * int(config.hidden_size_for_vit)
    hidden_dim = int(config.olmo_connector_mlp_hidden_size)
    self.pooling = OLMo35PoolingCrossAttention(config, mesh, pooling_input_dim, rngs=rngs)
    self.projector = OLMo35ConnectorMLP(
        config,
        mesh,
        int(config.hidden_size_for_vit),
        hidden_dim,
        int(config.emb_dim),
        rngs=rngs,
    )

  def _forward_connector(self, image_features: jax.Array, pooled_patches_idx: jax.Array) -> jax.Array:
    b, n_total, dim = image_features.shape
    n_pooled, pool_size = pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]
    valid = pooled_patches_idx >= 0
    safe_idx = jnp.clip(pooled_patches_idx, 0, max(n_total - 1, 0))
    # Gather strictly within axis 1 (n_total) per batch item so axis 0 (B) stays shard-local.
    gather_idx = safe_idx.reshape(b, n_pooled * pool_size, 1)
    to_pool = jnp.take_along_axis(image_features, gather_idx, axis=1).reshape(b, n_pooled, pool_size, dim)
    to_pool = jnp.where(valid[..., None], to_pool, jnp.zeros((), dtype=to_pool.dtype))

    denom = jnp.maximum(jnp.sum(valid.astype(jnp.int32), axis=-1, keepdims=True), 1).astype(to_pool.dtype)
    query = jnp.sum(to_pool, axis=2) / denom
    pooled = self.pooling(query, to_pool, valid)
    projected = self.projector(pooled)
    row_valid = jnp.any(valid, axis=-1, keepdims=True)
    return jnp.where(row_valid, projected, jnp.zeros((), dtype=projected.dtype))

  def __call__(self, image_features: jax.Array, pooled_patches_idx: jax.Array | None = None) -> jax.Array:
    """Pools ``image_features`` ``[B, N_total_patches, 2304]`` via ``pooled_patches_idx`` ``[B, N_pooled, 4]``."""
    b, n_total, _ = image_features.shape
    if pooled_patches_idx is None:
      pool_size = int(self.config.olmo_connector_pool_size)
      if n_total % pool_size != 0:
        raise ValueError(
            f"pooled_patches_idx was not provided and total patches ({n_total}) is not divisible by {pool_size}."
        )
      n_pooled = n_total // pool_size
      pooled_patches_idx = jnp.broadcast_to(
          jnp.arange(n_total, dtype=jnp.int32).reshape(1, n_pooled, pool_size),
          (b, n_pooled, pool_size),
      )
    else:
      pooled_patches_idx = pooled_patches_idx.astype(jnp.int32)
    if self.mesh is not None:
      image_features = sharding.maybe_shard_with_logical(
          image_features, ("activation_batch", None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
      pooled_patches_idx = sharding.maybe_shard_with_logical(
          pooled_patches_idx, ("activation_batch", None, None), self.mesh, self.config.shard_mode, self.config.logical_axis_rules
      )
    if not getattr(self.config, "olmo_vit_remat_blocks", True):
      return self._forward_connector(image_features, pooled_patches_idx)

    graphdef, state = nnx.split(self)

    @functools.partial(jax.checkpoint, prevent_cse=True)
    def _remat_connector(st, feats, p_idx):
      mod = nnx.merge(graphdef, st)
      return mod._forward_connector(feats, p_idx)  # pylint: disable=protected-access

    return _remat_connector(state, image_features, pooled_patches_idx)


def merge_olmo35_image_embeddings(
    text_embeddings: jax.Array,
    decoder_input_tokens: jax.Array,
    image_embeddings: jax.Array,
    *,
    patch_token_id: int,
    mesh: Mesh | None,
    config: Config,
) -> jax.Array:
  """Adds ``image_embeddings`` ``[B, N_pooled, D]`` (``+=``) at ``<im_patch>`` positions after ``embedding_norm``.

  Supports Context Parallelism (``CP > 1``): ``decoder_input_tokens`` and ``text_embeddings``
  are sharded on ``("activation_batch", "activation_norm_length", ...)``, while
  ``image_embeddings`` holds all ``N_pooled`` pooled tokens for each sequence in the batch.
  """
  is_patch = decoder_input_tokens == int(patch_token_id)
  n_pooled = image_embeddings.shape[1]
  cp_axis = config.context_sharding
  cp_size = mesh.shape.get(cp_axis, 1) if mesh is not None else 1

  if cp_size > 1 and mesh is not None:
    batch_axes = _batch_mesh_spec_entry(config, mesh)
    emb_spec = PartitionSpec(batch_axes, cp_axis, None)
    tok_spec = PartitionSpec(batch_axes, cp_axis)
    img_spec = PartitionSpec(batch_axes, None, None)

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(emb_spec, tok_spec, img_spec),
        out_specs=emb_spec,
        check_vma=False,
    )
    def _cp_add(y_loc, is_patch_loc, img_emb_loc):
      b_loc, t_loc, _ = y_loc.shape
      is_patch_full = jax.lax.all_gather(is_patch_loc, cp_axis, axis=1, tiled=True)
      full_ranks = jnp.cumsum(is_patch_full.astype(jnp.int32), axis=1) - 1
      c_idx = jax.lax.axis_index(cp_axis)
      local_ranks = jax.lax.dynamic_slice_in_dim(full_ranks, c_idx * t_loc, t_loc, axis=1)
      safe_ranks = jnp.clip(local_ranks, 0, max(n_pooled - 1, 0))
      gathered = jnp.take_along_axis(img_emb_loc, safe_ranks[..., None], axis=1)
      delta = jnp.where(is_patch_loc[..., None], gathered.astype(y_loc.dtype), jnp.zeros((), dtype=y_loc.dtype))
      return y_loc + delta

    return _cp_add(text_embeddings, is_patch, image_embeddings)

  ranks = jnp.cumsum(is_patch.astype(jnp.int32), axis=1) - 1
  safe_ranks = jnp.clip(ranks, 0, max(n_pooled - 1, 0))
  gathered = jnp.take_along_axis(image_embeddings, safe_ranks[..., None], axis=1)
  delta = jnp.where(is_patch[..., None], gathered.astype(text_embeddings.dtype), jnp.zeros((), dtype=text_embeddings.dtype))
  return text_embeddings + delta


def olmo_core_vision_to_maxtext_params(olmo_state: dict[str, Any], num_layers: int = 25) -> dict[str, Any]:
  """Maps OLMo-core ``MultimodalLM`` ``vision.*`` and ``connector.*`` tensors (numpy/array) to MaxText flat paths.

  Keys follow ``olmo_core.nn.maxtext.convert``'s ``/``-joined unscanned convention under
  ``vision_encoder/OLMo35VisionEncoder_0/...`` and ``vision_encoder/OLMo35VisionConnector_0/...``.
  """
  enc = "vision_encoder/OLMo35VisionEncoder_0"
  con = "vision_encoder/OLMo35VisionConnector_0"
  out: dict[str, Any] = {
      f"{enc}/patch_embedding/kernel": olmo_state["vision.patch_embedding.weight"].T,
      f"{enc}/patch_embedding/bias": olmo_state["vision.patch_embedding.bias"],
      f"{enc}/positional_embedding": olmo_state["vision.positional_embedding"],
  }
  for i in range(num_layers):
    src = f"vision.blocks.{i}"
    dst = f"{enc}/blocks_{i}"
    for norm_name in ("attn_norm", "ffn_norm"):
      out[f"{dst}/{norm_name}/scale"] = olmo_state[f"{src}.{norm_name}.weight"]
      out[f"{dst}/{norm_name}/bias"] = olmo_state[f"{src}.{norm_name}.bias"]
    for proj in ("wq", "wk", "wv", "wo"):
      out[f"{dst}/attn/{proj}/kernel"] = olmo_state[f"{src}.attn.{proj}.weight"].T
      out[f"{dst}/attn/{proj}/bias"] = olmo_state[f"{src}.attn.{proj}.bias"]
    for proj in ("w1", "w2"):
      out[f"{dst}/ffn/{proj}/kernel"] = olmo_state[f"{src}.ffn.{proj}.weight"].T
      out[f"{dst}/ffn/{proj}/bias"] = olmo_state[f"{src}.ffn.{proj}.bias"]

  for proj in ("wq", "wk", "wv", "wo"):
    out[f"{con}/pooling/{proj}/kernel"] = olmo_state[f"connector.pooling.{proj}.weight"].T
    out[f"{con}/pooling/{proj}/bias"] = olmo_state[f"connector.pooling.{proj}.bias"]
  for proj in ("w1", "w2", "w3"):
    out[f"{con}/projector/{proj}/kernel"] = olmo_state[f"connector.projector.{proj}.weight"].T
  return out


def maxtext_vision_to_olmo_core_params(mt_state: dict[str, Any], num_layers: int = 25) -> dict[str, Any]:
  """Inverse of :func:`olmo_core_vision_to_maxtext_params` (bit-exact transpose/rename)."""
  enc = "vision_encoder/OLMo35VisionEncoder_0"
  con = "vision_encoder/OLMo35VisionConnector_0"
  out: dict[str, Any] = {
      "vision.patch_embedding.weight": mt_state[f"{enc}/patch_embedding/kernel"].T,
      "vision.patch_embedding.bias": mt_state[f"{enc}/patch_embedding/bias"],
      "vision.positional_embedding": mt_state[f"{enc}/positional_embedding"],
  }
  for i in range(num_layers):
    src = f"{enc}/blocks_{i}"
    dst = f"vision.blocks.{i}"
    for norm_name in ("attn_norm", "ffn_norm"):
      out[f"{dst}.{norm_name}.weight"] = mt_state[f"{src}/{norm_name}/scale"]
      out[f"{dst}.{norm_name}.bias"] = mt_state[f"{src}/{norm_name}/bias"]
    for proj in ("wq", "wk", "wv", "wo"):
      out[f"{dst}.attn.{proj}.weight"] = mt_state[f"{src}/attn/{proj}/kernel"].T
      out[f"{dst}.attn.{proj}.bias"] = mt_state[f"{src}/attn/{proj}/bias"]
    for proj in ("w1", "w2"):
      out[f"{dst}.ffn.{proj}.weight"] = mt_state[f"{src}/ffn/{proj}/kernel"].T
      out[f"{dst}.ffn.{proj}.bias"] = mt_state[f"{src}/ffn/{proj}/bias"]

  for proj in ("wq", "wk", "wv", "wo"):
    out[f"connector.pooling.{proj}.weight"] = mt_state[f"{con}/pooling/{proj}/kernel"].T
    out[f"connector.pooling.{proj}.bias"] = mt_state[f"{con}/pooling/{proj}/bias"]
  for proj in ("w1", "w2", "w3"):
    out[f"connector.projector.{proj}.weight"] = mt_state[f"{con}/projector/{proj}/kernel"].T
  return out
