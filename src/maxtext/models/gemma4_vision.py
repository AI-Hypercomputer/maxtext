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


"""Vision transformer implementation for Gemma4."""

from typing import cast
import functools
import operator
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config, AttentionType
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers
from maxtext.layers import normalizations


# =============================================================================
# Gemma-4 vision clipped-linears (Navi upstream contribution)
# -----------------------------------------------------------------------------
# The Gemma-4 E2B/E4B vision tower ships per-projection activation clip bounds in
# the reference (HF) checkpoint: for each of the 7 vision projections
# (self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj) in each of the 16 encoder
# blocks, a scalar {input_min,input_max,output_min,output_max} = 16*7*4 = 448
# checkpoint tensors. The reference forward clamps each projection's input by
# [input_min,input_max] and its output by [output_min,output_max]. Omitting the
# clamps produces large activation drift in the image span (empirically KL 4-17
# on a 340-token teacher-forced parity harness), because a handful of vision
# activations blow up without the trained saturation. Upstream MaxText marks
# E2B/E4B multimodal "not yet supported" and does not model these bounds.
#
# This module adds them as OPT-IN, checkpoint-resident, NON-TRAINABLE scalars,
# gated on ``config.use_clipped_linears_for_vit`` (exact no-op when False).
# Design: plain ``nnx.Param`` bounds (so they map to the canonical ``params``
# collection and round-trip through the nnx->linen->orbax checkpoint path), plus
# a leaf-PATH optimizer-freeze mask so the 448 bounds are excluded from optimizer
# updates and weight decay without a custom nnx.Variable subclass (subclasses are
# renamed by the linen bridge and silently dropped from the saved checkpoint).
# A NaN sentinel + ``validate_clip_bounds`` hard-fails on a missing/non-finite
# bound rather than silently degrading to an identity clamp.

# Leaf-name tokens that identify a clip-bound scalar in a flattened params tree.
_CLIP_LEAF_TOKENS = ("q_clip", "k_clip", "v_clip", "o_clip", "gate_clip", "up_clip", "down_clip")
_CLIP_BOUND_NAMES = ("input_min", "input_max", "output_min", "output_max")


def _mk_clip_bound(init_val=jnp.nan):
  """A checkpoint-resident scalar clip bound as a plain ``nnx.Param`` (maps to the
  canonical ``params`` collection; NaN sentinel marks an unloaded bound)."""
  return nnx.Param(jnp.asarray(init_val, dtype=jnp.float32))


def _is_clip_bound_path(path) -> bool:
  """True iff a flattened-params key path addresses a clip-bound scalar."""
  s = "/".join(str(getattr(p, "key", p)) for p in path) if not isinstance(path, str) else path
  return any(tok in s for tok in _CLIP_LEAF_TOKENS) and any(b in s for b in _CLIP_BOUND_NAMES)


def clip_optimizer_freeze_mask(params_tree):
  """Bool pytree (same structure as ``params_tree``): True for TRAINABLE leaves,
  False for the immutable clip bounds. Feed to ``optax.masked``/``multi_transform``
  so the bounds get ``set_to_zero()`` updates. Path-based, so it survives the
  nnx->linen->orbax round-trip regardless of leaf type erasure."""
  flat = jax.tree_util.tree_flatten_with_path(params_tree)[0]
  leaves_mask = [not _is_clip_bound_path(path) for path, _ in flat]
  treedef = jax.tree_util.tree_structure(params_tree)
  return jax.tree_util.tree_unflatten(treedef, leaves_mask)


def _clip_bound_value(bound, dtype):
  """Read a clip-bound scalar as a constant: stop_gradient defends the bounds from receiving
  gradient at the clamp use-site (defense in depth alongside the optimizer freeze mask), so clip-bound
  gradients can never contaminate gradient statistics even if the freeze mask were misconfigured."""
  return jax.lax.stop_gradient(bound.value).astype(dtype)


def _clip_in(x, cb):
  """clamp(x, input_min, input_max) in x's dtype; no-op if ``cb`` is None. Bounds are stop_gradient'd."""
  if cb is None:
    return x
  xd = x.dtype
  return jnp.clip(x, _clip_bound_value(cb.input_min, xd), _clip_bound_value(cb.input_max, xd))


def _clip_out(y, cb):
  """clamp(y, output_min, output_max) in y's dtype; no-op if ``cb`` is None. Bounds are stop_gradient'd."""
  if cb is None:
    return y
  yd = y.dtype
  return jnp.clip(y, _clip_bound_value(cb.output_min, yd), _clip_bound_value(cb.output_max, yd))


class _ClipBounds(nnx.Module):
  """Holds the four checkpoint-resident scalar clip bounds as ``nnx.Param`` leaves."""

  def __init__(self):
    self.input_min = _mk_clip_bound()
    self.input_max = _mk_clip_bound()
    self.output_min = _mk_clip_bound()
    self.output_max = _mk_clip_bound()


def _make_clip_state():
  """Four NaN-sentinel scalar bounds (checkpoint-resident, non-trainable)."""
  return _ClipBounds()


def validate_clip_bounds(cb, where=""):
  """Hard-fail: every bound finite + scalar, and input_min<=input_max, output_min<=output_max.
  Raises ValueError otherwise. No-op if ``cb`` is None."""
  if cb is None:
    return
  vals = {}
  for nm in ("input_min", "input_max", "output_min", "output_max"):
    v = getattr(cb, nm).value
    if getattr(v, "shape", ()) not in ((), (1,)):
      raise ValueError(f"Gemma4 vision clip bound '{nm}'{(' @ '+where) if where else ''} has non-scalar "
                       f"shape {v.shape}; expected scalar.")
    fv = float(jnp.reshape(v, (-1,))[0])
    if not bool(jnp.isfinite(jnp.asarray(fv))):
      raise ValueError(f"Gemma4 vision clip bound '{nm}'{(' @ '+where) if where else ''} = {fv} is non-finite "
                       f"(missing/NaN/Inf). use_clipped_linears_for_vit=True declares a FINITE clipped model; "
                       f"refusing to fall back to an identity clamp.")
    vals[nm] = fv
  if vals["input_min"] > vals["input_max"]:
    raise ValueError(f"Gemma4 vision clip bound{(' @ '+where) if where else ''}: input_min={vals['input_min']} "
                     f"> input_max={vals['input_max']} (a clamp with min>max would empty the interval).")
  if vals["output_min"] > vals["output_max"]:
    raise ValueError(f"Gemma4 vision clip bound{(' @ '+where) if where else ''}: output_min={vals['output_min']} "
                     f"> output_max={vals['output_max']} (a clamp with min>max would empty the interval).")


# Expected clipped-module accounting for a Gemma-4 vision tower: 16 encoder blocks x 7 projections
# (attention q/k/v/o = 4 modules holding q/k/v/o clip states + mlp gate/up/down = 3) -> but the clip
# STATE lives on 2 modules per block (the Gemma4Attention holding q/k/v/o_clip, and the
# Gemma4ClippedMlpBlock holding gate/up/down_clip). The scalar bound count is the invariant that matters:
# 16 blocks x 7 projections x 4 bounds = 448. We validate the projection-level clip states (16x7 = 112).
EXPECTED_CLIP_PROJECTIONS = 112   # 16 blocks x 7 projections (q,k,v,o,gate,up,down)
EXPECTED_CLIP_BOUNDS = 448        # 112 projections x 4 scalar bounds


def validate_all_vision_clip_bounds(model, *, expected_projections=EXPECTED_CLIP_PROJECTIONS,
                                    expected_bounds=EXPECTED_CLIP_BOUNDS):
  """Post-checkpoint-load, pre-first-JIT validation of ALL Gemma-4 vision clip bounds. Fail-closed.

  Walks the model graph, collects every ``_ClipBounds`` module (each carries the 4 scalars for one
  projection), validates each (finite, scalar, min<=max), and asserts EXACT counts:
  ``expected_projections`` clip-state modules and ``expected_bounds`` scalar leaves. Raises ValueError on
  any deficiency. The exact-count check is what prevents a traversal that accidentally finds zero modules
  from silently "passing".
  """
  from flax import nnx  # pylint: disable=import-outside-toplevel
  n_proj = 0
  n_bounds = 0
  for path, mod in nnx.iter_graph(model):
    # A clip-state module has exactly the four bound leaves.
    if (hasattr(mod, "input_min") and hasattr(mod, "input_max")
        and hasattr(mod, "output_min") and hasattr(mod, "output_max")
        and not isinstance(mod, (int, float))):
      where = "/".join(str(getattr(p, "key", p)) for p in path) if isinstance(path, (list, tuple)) else str(path)
      validate_clip_bounds(mod, where)
      n_proj += 1
      n_bounds += 4
  if n_proj != expected_projections:
    raise ValueError(
        f"Gemma-4 vision clip validation found {n_proj} clip-state modules, expected exactly "
        f"{expected_projections} (16 blocks x 7 projections). A mismatch means the clip bounds were not "
        f"loaded/mapped correctly; refusing to run with a partially-clipped vision tower."
    )
  if n_bounds != expected_bounds:
    raise ValueError(
        f"Gemma-4 vision clip validation found {n_bounds} clip-bound scalars, expected exactly {expected_bounds}."
    )
  return n_proj, n_bounds





def factorized_posemb(posemb: jax.Array, positions_xy: jax.Array, precision) -> jax.Array:
  """Computes factorized position embedding from (x, y) coordinates.

  Args:
    posemb: The factorized position embedding parameters.
    positions_xy: The (x, y) coordinates for each patch.
    precision: The precision for the einsum operation.

  Returns:
    The computed position embeddings.
  """
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
  nan = jnp.logical_and(nan, positions_xy[..., None] != -1)
  pos_oh = jnp.where(nan, jnp.nan, one_hot)
  pe_seq = jnp.einsum("...is,sid->i...d", pos_oh, posemb, precision=precision).astype(posemb.dtype)
  return jnp.sum(pe_seq, axis=0)


def patchify(images: jax.Array, patch_size: int) -> tuple[jax.Array, jax.Array]:
  """Patchifies images and returns patches and (x, y) coordinates.

  Args:
    images: The input images of shape [..., H, W, C].
    patch_size: The size of each square patch.

  Returns:
    A tuple containing:
      - patches: The extracted patches of shape [..., num_patches, patch_size * patch_size * C].
      - positions_xy: The (x, y) coordinates of the top-left corner of each patch,
        of shape [..., num_patches, 2].
  """
  # Using jax.lax.reshape and transpose instead of einshape for simplicity
  *b, h, w, c = images.shape

  p = patch_size
  q = patch_size

  # ... h w c -> ... (h//p) p (w//q) q c
  reshaped_images = jax.lax.reshape(images, tuple(b) + (h // p, p, w // q, q, c))
  # ... (h//p) p (w//q) q c -> ... (h//p) (w//q) p q c
  transposed_images = jnp.transpose(
      reshaped_images, axes=tuple(range(len(b))) + (len(b), len(b) + 2, len(b) + 1, len(b) + 3, len(b) + 4)
  )
  # ... (h//p) (w//q) p q c -> ... ((h//p)*(w//q)) (p*q*c)
  patches = jax.lax.reshape(transposed_images, tuple(b) + ((h // p) * (w // q), p * q * c))

  xy = jnp.meshgrid(jnp.arange(w // patch_size), jnp.arange(h // patch_size))
  positions_xy = jnp.stack(xy, axis=-1)
  # yxc -> (yx)c
  positions_xy = jnp.reshape(positions_xy, (-1, 2))

  return patches, jnp.broadcast_to(positions_xy, tuple(b) + positions_xy.shape)


class VisionEntry(nnx.Module):
  """The vision entry layer."""

  def __init__(
      self,
      d_model: int,
      patch_size: int,
      pos_emb_shape_yx: tuple[int, int],
      normalize_input_range: bool = False,
      *,
      rngs: nnx.Rngs,
      dtype,
      weight_dtype,
      matmul_precision,
  ):
    self.d_model = d_model
    self.patch_size = patch_size
    self.pos_emb_shape_yx = pos_emb_shape_yx
    self.normalize_input_range = normalize_input_range
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.matmul_precision = matmul_precision

    self.input_projection = linears.DenseGeneral(
        in_features_shape=self.patch_size * self.patch_size * 3,
        out_features_shape=self.d_model,
        use_bias=False,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
        axis=-1,
        rngs=rngs,
    )

    assert self.pos_emb_shape_yx[-1] == 2, f"{self.pos_emb_shape_yx=}"

    pos_emb_init = nnx.initializers.normal(stddev=0.02)
    self.pos_emb_param = nnx.Param(
        pos_emb_init(
            rngs.params(),
            (self.pos_emb_shape_yx[0], self.pos_emb_shape_yx[1], self.d_model),
            jnp.float32,
        )
    )

  def __call__(
      self,
      images_or_patches: jax.Array,
      positions_xy: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Processes input images or patches and applies projection and position embeddings."""
    if positions_xy is None:
      # If positions_xy is not provided, we assume the input are images
      # in the format [..., H, W, C] and need to be patchified.
      patches, positions_xy = patchify(images_or_patches, self.patch_size)
    else:
      # If positions_xy is provided, we assume the input are already patches.
      patches = images_or_patches
      assert patches.ndim == 3, f"Expected patches to have 3 dimensions, but got {patches.ndim}"
      assert positions_xy is not None, "positions_xy must be provided when images_or_patches are already patchified"
      # Ensure positions_xy has the correct batch dimension
      if positions_xy.shape[0] == patches.shape[0]:
        pass  # positions_xy already has batch dimension
      elif positions_xy.ndim == 2:
        positions_xy = jnp.broadcast_to(positions_xy, (patches.shape[0],) + positions_xy.shape)
      else:
        raise ValueError(f"Unexpected positions_xy shape: {positions_xy.shape}")

    if self.normalize_input_range:
      patches = 2 * (patches - 0.5)

    x = self.input_projection(patches)

    pos_embed = factorized_posemb(cast(jax.Array, self.pos_emb_param.value), positions_xy, self.matmul_precision).astype(
        x.dtype
    )

    return x + pos_embed, positions_xy


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    rotary_fraction: float | None = None,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies multidimensional RoPE. Based on Gemma 4 implementation.

  Args:
    inputs: The input array to apply RoPE to.
    positions: The positional information. Can be 1D or ND.
    base_frequency: The base frequency for the sinusoidal functions.
    rotary_fraction: The fraction of the hidden dimension to apply RoPE to. If None,
      applies to the full dimension.
    scale_factor: A scale factor applied to the sinusoidal arguments.

  Returns:
    The input array with multidimensional RoPE applied.
  """

  # Internal _apply_rope logic
  def _apply_rope(x_in: jax.Array, pos: jax.Array, base_freq: int, scale: float) -> jax.Array:
    # x_in: [B, L, N, H]
    # pos: [B, L] or similar
    dim = x_in.shape[-1]
    half_dim = dim // 2
    fraction = 2 * jnp.arange(0, half_dim) / dim
    timescale = base_freq**fraction

    # position shape logic
    reshaped_pos = pos[..., jnp.newaxis, jnp.newaxis]
    sinusoid_inp = reshaped_pos / timescale

    sin_half = jnp.sin(sinusoid_inp).astype(x_in.dtype)
    cos_half = jnp.cos(sinusoid_inp).astype(x_in.dtype)

    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)

    x1, x2 = jnp.split(x_in, 2, axis=-1)
    rotated_x = jnp.concatenate((-x2, x1), axis=-1)

    return (x_in * cos) + (rotated_x * sin)

  if positions.ndim + 2 == inputs.ndim:
    if rotary_fraction is None or rotary_fraction == 1.0:
      return _apply_rope(
          inputs,
          positions,
          base_frequency,
          scale_factor,
      )
    dim_to_rope = int(rotary_fraction * inputs.shape[-1])
    if dim_to_rope == inputs.shape[-1]:
      return _apply_rope(
          inputs,
          positions,
          base_frequency,
          scale_factor,
      )
    if dim_to_rope == 0:
      return inputs
    x1 = inputs[..., :dim_to_rope]
    x2 = inputs[..., dim_to_rope:]
    x1 = _apply_rope(
        x1,
        positions,
        base_frequency,
        scale_factor,
    )
    return jnp.concatenate([x1, x2], axis=-1)

  ndim = positions.shape[-1]
  num_input_channels = inputs.shape[-1]
  num_rotated_channels = num_input_channels
  if rotary_fraction is not None:
    num_rotated_channels = int(round(num_rotated_channels * rotary_fraction))
  num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

  assert num_rotated_channels_per_dim > 0, f"Requirement not satisfied: 2 * {ndim=} <= {num_input_channels=}."

  split_points = [(k + 1) * num_rotated_channels_per_dim for k in range(ndim)]
  if rotary_fraction is None:
    split_points = split_points[:-1]
  assert all(
      isinstance(sp, int) for sp in split_points
  ), f"Expected all split points to be integers, but got {split_points}"
  x_parts = jnp.split(inputs, split_points, axis=-1)
  y_parts = [
      _apply_rope(
          x_parts[k],
          positions[..., k],
          base_frequency,
          scale_factor,
      )
      for k in range(ndim)
  ]

  if rotary_fraction is not None:
    y_parts.append(x_parts[-1])

  return jnp.concatenate(y_parts, axis=-1)


def avg_pool_by_positions(
    x: jax.Array,
    *,
    positions_xy: jax.Array,
    length: int,
    precision,
) -> tuple[jax.Array, jax.Array]:
  """Performs 2D spatial pooling according to patch positions.

  Args:
    x: The input features of shape [B, L, D].
    positions_xy: The (x, y) coordinates of each patch of shape [B, L, 2].
    length: The desired output sequence length after pooling.
    precision: The precision for the einsum operation.

  Returns:
    A tuple containing:
      - output: The pooled features of shape [B, length, D].
      - mask: A boolean mask indicating valid pooled positions.
  """
  k = max(1, int((x.shape[1] // length) ** 0.5))
  assert k * k * length == x.shape[1], f"Cannot pool {x.shape=} to {length=}"

  max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
  kernel_idxs = jnp.floor_divide(positions_xy, k)
  flat_kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
  weights = jax.nn.one_hot(flat_kernel_idx, length) / k**2
  output = jnp.einsum("bLl,bLd->bld", weights, x, precision=precision)
  mask = jnp.logical_not((weights == 0).all(axis=1))
  return output, mask


class VisionExit(nnx.Module):
  """Vision exit layer with scaling and optional spatial pooling."""

  def __init__(self, d_model: int, output_length: int | tuple[int, ...] = 256, *, rngs: nnx.Rngs, precision):
    self.d_model = d_model
    self.output_length = output_length
    self.precision = precision

  def _maybe_downsample(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    """Downsamples the vision features if required by the output length."""
    cur_length = x.shape[1]

    POSITIONS_PAD_VALUE = -1

    if cur_length == length:
      if positions_xy is None:
        mask = jnp.ones(x.shape[:-1], dtype=jnp.bool_)
      else:
        mask = jnp.logical_not((positions_xy == POSITIONS_PAD_VALUE).all(axis=-1))
      return x, mask

    if positions_xy is not None:
      x_pooled, mask = avg_pool_by_positions(x, positions_xy=positions_xy, length=length, precision=self.precision)
      return x_pooled, mask

    cur_width = int(cur_length**0.5)
    if cur_width**2 != cur_length:
      raise ValueError(f"x.shape[1]={cur_length} must be a perfect square.")

    output_width = int(length**0.5)
    if output_width**2 != length:
      raise ValueError(f"{length=} must be a perfect square.")

    if cur_width % output_width != 0:
      raise ValueError(f"{cur_width=} must be divisible by {output_width=}.")

    x_2d = x.reshape(x.shape[0], cur_width, cur_width, x.shape[-1])

    window = cur_width // output_width
    window_shape = (window, window)
    x_2d = nnx.avg_pool(x_2d, window_shape=window_shape, strides=window_shape)

    x_pooled = x_2d.reshape(x.shape[0], length, x.shape[-1])
    mask = jnp.ones(x_pooled.shape[:-1], dtype=jnp.bool_)
    return x_pooled, mask

  def _single_call(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    """Processes the features for a single target length."""
    x, mask = self._maybe_downsample(x, positions_xy=positions_xy, length=length)

    x = x * jnp.sqrt(self.d_model)

    return x, mask

  def __call__(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      output_length_overrides: tuple[int, ...] | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array | None], ...]:
    """Applies vision exit processing, optionally downsampling to requested output lengths."""
    lens = (self.output_length,) if isinstance(self.output_length, int) else self.output_length
    if output_length_overrides is not None:
      lens = output_length_overrides

    return tuple(self._single_call(x, positions_xy=positions_xy, length=length) for length in lens)


class Gemma4VisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Gemma 4 vision."""

  def __init__(
      self,
      base_frequency: int,
      rotary_fraction: float | None = None,
      scale_factor: float = 1.0,
  ):
    self.base_frequency = base_frequency
    self.rotary_fraction = rotary_fraction
    self.scale_factor = scale_factor

  def __call__(self, inputs: jax.Array, positions: jax.Array) -> jax.Array:
    """Applies rotary position embeddings to the inputs."""
    return apply_multidimensional_rope(
        inputs,
        positions,
        base_frequency=self.base_frequency,
        rotary_fraction=self.rotary_fraction,
        scale_factor=self.scale_factor,
    )


class Gemma4Attention(attentions.Attention):
  """Gemma 4 specific Attention module.

  When ``use_clipped_linears`` is enabled, the q/k/v/o projections apply the
  per-projection activation clip bounds carried in the Gemma-4 vision checkpoint
  (input clamp before the matmul, output clamp after). The clamps are wired by
  overriding the base ``Attention`` projection methods, so the underlying
  ``DenseGeneral`` weights and their checkpoint key paths are unchanged. When the
  flag is off, every override is an exact delegate to the base implementation.
  """

  def init_rotary_embedding(self) -> Gemma4VisionRotaryEmbedding:
    """Initializes the rotary position embedding module for Gemma 4 vision."""
    return Gemma4VisionRotaryEmbedding(
        base_frequency=self.config.rope_theta_for_vit if hasattr(self.config, "rope_theta_for_vit") else 100,
        rotary_fraction=None,  # Or assume it from config if available
    )

  def enable_vision_clip_bounds(self):
    """Attach the four checkpoint-resident clip-bound scalars for each of q/k/v/o.

    Called once by ``Gemma4EncoderBlock`` after construction when
    ``config.use_clipped_linears_for_vit`` is set. Idempotent.
    """
    if getattr(self, "_use_clipped_linears", False):
      return
    # Fail-closed: the clipped path clamps q/k/v independently, which requires SEPARATE q/k/v projections.
    # A fused QKV projection would apply a single clamp and silently bypass the per-projection clip semantics.
    if bool(getattr(self.config, "fused_qkv", False)):
      raise ValueError(
          "Gemma-4 vision clipped-linears require fused_qkv=False: the checkpoint carries distinct "
          "q/k/v activation clip bounds that must be applied per-projection. Refusing to silently clip a "
          "fused QKV projection."
      )
    self._use_clipped_linears = True
    self.q_clip = _make_clip_state()
    self.k_clip = _make_clip_state()
    self.v_clip = _make_clip_state()
    self.o_clip = _make_clip_state()

  def validate_clip_bounds(self):
    if not getattr(self, "_use_clipped_linears", False):
      return
    validate_clip_bounds(self.q_clip, "q_proj")
    validate_clip_bounds(self.k_clip, "k_proj")
    validate_clip_bounds(self.v_clip, "v_proj")
    validate_clip_bounds(self.o_clip, "o_proj")

  # --- projection overrides: clamp(input) -> DenseGeneral -> clamp(output) ---
  def query_projection(self, inputs_q, out_sharding=None):
    if not getattr(self, "_use_clipped_linears", False):
      return super().query_projection(inputs_q, out_sharding=out_sharding)
    x = _clip_in(inputs_q, self.q_clip)
    y = self.query(x, out_sharding=out_sharding)
    return _clip_out(y, self.q_clip)

  def kv_projection(self, inputs_kv, proj_name, out_sharding=None):
    if not getattr(self, "_use_clipped_linears", False):
      return super().kv_projection(inputs_kv, proj_name=proj_name, out_sharding=out_sharding)
    if proj_name == "key":
      cb, module = self.k_clip, self.key
    elif proj_name == "value":
      cb, module = self.v_clip, self.value
    else:
      raise ValueError(f"proj_name must be 'key' or 'value', but got {proj_name}")
    x = _clip_in(inputs_kv, cb)
    y = module(x, out_sharding=out_sharding)
    return _clip_out(y, cb)

  def out_projection(self, out, out_sharding=None):
    if not getattr(self, "_use_clipped_linears", False):
      return super().out_projection(out, out_sharding=out_sharding)
    x = _clip_in(out, self.o_clip)
    y = self.out(x, out_sharding=out_sharding)
    return _clip_out(y, self.o_clip)


class Gemma4ClippedMlpBlock(linears.MlpBlock):
  """MlpBlock that applies the Gemma-4 vision per-projection activation clip bounds
  to the gate (wi_0), up (wi_1) and down (wo) projections.

  Only the non-fused activation path is supported (E2B/E4B use
  ``activations=("gelu", "linear")`` with ``fused_mlp=False``), because gate and up
  carry distinct clip bounds. When ``use_clipped_linears`` is off, this delegates to
  the base ``MlpBlock`` unchanged.
  """

  def __init__(self, *args, use_clipped_linears=False, **kwargs):
    super().__init__(*args, **kwargs)
    self._use_clipped_linears = bool(use_clipped_linears)
    if self._use_clipped_linears:
      self.gate_clip = _make_clip_state()  # wi_0
      self.up_clip = _make_clip_state()    # wi_1
      self.down_clip = _make_clip_state()  # wo

  def validate_clip_bounds(self):
    if not self._use_clipped_linears:
      return
    validate_clip_bounds(self.gate_clip, "gate_proj")
    validate_clip_bounds(self.up_clip, "up_proj")
    validate_clip_bounds(self.down_clip, "down_proj")

  def __call__(self, inputs, decode=False, deterministic=False,
               intermediate_sharding=None, out_sharding=None):
    if not self._use_clipped_linears:
      return super().__call__(inputs, decode=decode, deterministic=deterministic,
                              intermediate_sharding=intermediate_sharding, out_sharding=out_sharding)
    cfg = self.config
    if getattr(cfg, "fused_mlp", False):
      # Clipped vision MLP requires the unfused path so gate/up get their own output clamps.
      raise ValueError("Gemma4ClippedMlpBlock requires fused_mlp=False (per-projection clip bounds).")
    if self.mlp_layer_norm is not None:
      inputs = self.mlp_layer_norm(inputs)
    clips = [self.gate_clip, self.up_clip]  # order matches activations ("gelu", "linear") == (gate, up)
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
      module = getattr(self, dense_name)
      x = _clip_in(inputs, clips[idx])
      x = module(x, out_sharding=intermediate_sharding)
      x = _clip_out(x, clips[idx])
      x = linears.checkpoint_name(x, "mlp" + dense_name)
      if cfg.activations_in_float32:
        x = x.astype(jnp.float32)
      x = linears._convert_to_activation_function(act_fn)(x)
      activations.append(x)
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    x = self.dropout(x, deterministic=deterministic)
    x = self._maybe_shard_with_logical(x, self.intermediate_logical)
    x = _clip_in(x, self.down_clip)
    output = self.wo(x, out_sharding=out_sharding)
    output = _clip_out(output, self.down_clip)
    output = linears.checkpoint_name(output, "mlpwo")
    return output



class Gemma4EncoderBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    # Standard RMSNorm
    self.pre_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.post_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Attention with Gemma 4 specifics
    # We need dummy shapes for initialization
    batch_size = config.per_device_batch_size

    # Handle both square (int) and rectangular (list/tuple) image sizes
    if isinstance(config.image_size_for_vit, (list, tuple)):
      img_h, img_w = config.image_size_for_vit
      seq_len = (img_h // config.patch_size_for_vit) * (img_w // config.patch_size_for_vit)
    else:
      seq_len = (config.image_size_for_vit // config.patch_size_for_vit) ** 2

    dummy_shape = (batch_size, seq_len, config.hidden_size_for_vit)

    self.attention = Gemma4Attention(
        config=config,
        num_query_heads=config.num_attention_heads_for_vit,
        num_kv_heads=config.num_attention_heads_for_vit,
        head_dim=config.hidden_size_for_vit // config.num_attention_heads_for_vit,
        max_target_length=seq_len,
        mesh=mesh,
        attention_kernel=config.attention_for_vit,
        inputs_q_shape=dummy_shape,
        inputs_kv_shape=dummy_shape,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        dropout_rate=config.dropout_rate,
        attention_type=AttentionType.FULL,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        is_vision=True,
        rngs=self.rngs,
    )
    # Opt-in Gemma-4 vision clipped-linears: attach the q/k/v/o checkpoint-resident
    # clip bounds and route the projections through the clamp-in/clamp-out overrides.
    self._use_clipped_linears = bool(getattr(config, "use_clipped_linears_for_vit", False))
    if self._use_clipped_linears:
      self.attention.enable_vision_clip_bounds()

    self.pre_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.post_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    mlp_cls = Gemma4ClippedMlpBlock if self._use_clipped_linears else linears.MlpBlock
    mlp_kwargs = {"use_clipped_linears": True} if self._use_clipped_linears else {}
    self.mlp = mlp_cls(
        config=config,
        mesh=mesh,
        in_features=config.hidden_size_for_vit,
        intermediate_dim=config.intermediate_size_for_vit,
        activations=("gelu", "linear"),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        intermediate_dropout_rate=config.dropout_rate,
        rngs=self.rngs,
        **mlp_kwargs,
    )

  def __call__(
      self,
      x: jax.Array,
      positions: jax.Array | None = None,
      deterministic: bool = False,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    """Applies the encoder block (MHSA + MLP) to the inputs.

    When ``decoder_segment_ids`` is provided, patches carrying distinct segment
    ids cannot attend to each other. This is used by the padded-patch path
    (valid patches = segment 1, padded/sentinel patches = segment 2) so that the
    phantom pad patches are masked out of vision self-attention.
    """
    x_normed = self.pre_attention_norm(x)
    # Pass positions to attention for RoPE (+ optional segment mask for padded patches).
    x_attn, _ = self.attention(
        x_normed,
        x_normed,
        inputs_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    x_attn = self.post_attention_norm(x_attn)
    x_after_attn = x_attn + x

    x_ffw_normed = self.pre_ffw_norm(x_after_attn)
    x_ffw = self.mlp(x_ffw_normed, deterministic=deterministic)
    x_ffw = self.post_ffw_norm(x_ffw)
    x_after_ffw = x_ffw + x_after_attn
    return x_after_ffw


class Gemma4VisionEncoderLayer(nnx.Module):
  """Gemma 4 Vision Encoder Layer."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    # Input Projection (VisionEntry)
    self.vision_entry = VisionEntry(
        d_model=config.hidden_size_for_vit,
        patch_size=config.patch_size_for_vit,
        pos_emb_shape_yx=(config.num_position_embeddings_for_vit, 2),
        normalize_input_range=True,
        rngs=self.rngs,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
    )

    # Encoder Blocks
    for i in range(config.num_hidden_layers_for_vit):
      layer = Gemma4EncoderBlock(config, mesh, rngs=self.rngs)
      # Register submodules for NNX
      setattr(self, f"layer_{i}", layer)

    # Vision Exit
    self.vision_exit = VisionExit(
        d_model=config.hidden_size_for_vit,
        output_length=config.vision_output_length,
        rngs=self.rngs,
        precision=config.matmul_precision,
    )
    self.std_bias = nnx.Param(
        nnx.initializers.zeros(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype), sharding=(None,)
    )
    self.std_scale = nnx.Param(
        nnx.initializers.ones(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype), sharding=(None,)
    )

  def __call__(
      self,
      inputs: jax.Array,
      deterministic: bool = False,
      image_position_ids: jax.Array | None = None,
  ):
    """Applies the vision encoder layer.

    Two contracts:

    (A) Legacy all-valid (``image_position_ids is None``): ``inputs`` are raw images
        [B, N, H, W, C] (or [B, H, W, C]); patchify -> full unmasked attention -> pool by the
        derived positions -> return embeddings only (4D array [B, N, K, D]).

    (B) Padded-patch dynamic-N (``image_position_ids is not None``): ``inputs`` are ALREADY
        patchified pixel_values with shape [B, L, P*P*C] (or [B, N, L, P*P*C]) and
        ``image_position_ids`` is [B, L, 2] (or [B, N, L, 2]) with -1 sentinel rows marking padded
        patches. The pre-patchified patches + REAL positions are fed to VisionEntry, per-patch
        ``decoder_segment_ids`` (valid=1, pad=2) mask the phantom pad patches out of self-attention,
        pooling uses the real positions (``avg_pool_by_positions`` maps a -1 patch to a zero-weight
        bucket), and the VisionExit validity mask is returned. Returns a 2-tuple
        ``(embeddings[B, N, K, D], image_masks[B*N, K])`` where ``image_masks.sum()`` is the number
        of valid pooled tokens, threaded to ``merge_mm_embeddings.token_masks`` so exactly the valid
        pooled tokens land in the image placeholders.
    """
    if image_position_ids is None:
      # ---- Legacy path: raw images -> patchify -> full (unmasked) attention ----
      # Fail-closed contract guard: the legacy path requires FULL images ([B,N,H,W,C] or [B,H,W,C]).
      # Pre-patchified inputs ([B,L,F] or [B,N,L,F]) only make sense with per-patch image_position_ids
      # (the padded-patch path). Refuse to silently mis-handle pre-patchified input as a full image, which
      # would either crash cryptically or produce wrong pooling.
      if inputs.ndim not in (4, 5):
        raise ValueError(
            "Gemma4 vision legacy path expects full images with shape [B, H, W, C] or [B, N, H, W, C]; "
            f"got inputs.ndim={inputs.ndim} (shape {tuple(inputs.shape)}). Pre-patchified pixel_values must be "
            "accompanied by image_position_ids (the padded-patch path). Refusing to silently take the legacy "
            "all-valid vision path."
        )
      if inputs.ndim == 4:
        inputs = jnp.expand_dims(inputs, 1)
      b, n, h, w, c = inputs.shape
      inputs_flat = jnp.reshape(inputs, (b * n, h, w, c))

      x, positions_xy = self.vision_entry(inputs_flat)

      for i in range(self.config.num_hidden_layers_for_vit):
        layer = getattr(self, f"layer_{i}")
        x = layer(x, positions=positions_xy, deterministic=deterministic)

      vision_exit_results = self.vision_exit(x, positions_xy=positions_xy)
      (embeddings, _) = vision_exit_results[0]

      embeddings = (embeddings - self.std_bias.value.astype(embeddings.dtype)) * self.std_scale.value.astype(
          embeddings.dtype
      )

      # Unflatten batch and num_images
      final_x = jnp.reshape(embeddings, (b, n, embeddings.shape[1], embeddings.shape[2]))
      return final_x

    # ---- Padded-patch dynamic-N path: pre-patchified patches + sentinel positions ----
    # inputs: [B, L, P*P*C] pre-patchified pixel_values. Support an optional per-image N dim
    # [B, N, L, F] -> flatten to [B*N, L, F] to match positions.
    if inputs.ndim == 4:
      b, n, l, f = inputs.shape
      patches = jnp.reshape(inputs, (b * n, l, f))
      pos = jnp.reshape(image_position_ids, (b * n, l, 2))
    else:
      assert inputs.ndim == 3, f"padded-patch path expects pre-patchified [B, L, F] patches, got {inputs.shape}"
      b, l, f = inputs.shape
      n = 1
      patches = inputs
      pos = image_position_ids
      if pos.ndim == 2:
        pos = jnp.broadcast_to(pos, (b, l, 2))

    pos = pos.astype(jnp.int32)

    # VisionEntry consumes pre-patchified patches + REAL positions (incl -1 sentinels).
    x, positions_xy = self.vision_entry(patches, positions_xy=pos)

    # Segment ids: valid patch (any coord != -1) -> 1, padded sentinel patch -> 2. Distinct segments
    # cannot attend to each other, masking the phantom pad patches out of self-attention.
    is_pad = (positions_xy == -1).all(axis=-1)  # [B*N, L]
    decoder_segment_ids = jnp.where(is_pad, 2, 1).astype(jnp.int32)

    for i in range(self.config.num_hidden_layers_for_vit):
      layer = getattr(self, f"layer_{i}")
      x = layer(
          x,
          positions=positions_xy,
          deterministic=deterministic,
          decoder_segment_ids=decoder_segment_ids,
      )

    # Pool with REAL positions; avg_pool_by_positions returns (embeddings, validity_mask).
    vision_exit_results = self.vision_exit(x, positions_xy=positions_xy)
    (embeddings, image_masks) = vision_exit_results[0]  # embeddings [B*N, K, D], mask [B*N, K]

    embeddings = (embeddings - self.std_bias.value.astype(embeddings.dtype)) * self.std_scale.value.astype(
        embeddings.dtype
    )

    final_x = jnp.reshape(embeddings, (b, n, embeddings.shape[1], embeddings.shape[2]))
    if image_masks is None:
      image_masks = jnp.ones((b * n, embeddings.shape[1]), dtype=jnp.int32)
    else:
      # merge_mm_embeddings does argsort(-token_mask); use int32 so negation/sort is well-defined.
      image_masks = image_masks.astype(jnp.int32)

    return final_x, image_masks


class Gemma4VisionProjector(nnx.Module):
  """A layer that projects image embeddings to the embedding space of the text encoder."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.projection = linears.DenseGeneral(
        in_features_shape=config.hidden_size_for_vit,
        out_features_shape=config.emb_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )

    self.norm = normalizations.RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Projects vision embeddings into the text embedding space."""
    x_normed = self.norm(x)
    x_projected = self.projection(x_normed)
    return x_projected


def gemma4_vision_encoder_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Wraps the Gemma 4 Vision Encoder as a Linen module."""
  return nnx_wrappers.to_linen(
      Gemma4VisionEncoderLayer,
      config=config,
      mesh=mesh,
      name="Gemma4VisionEncoderLayer",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
