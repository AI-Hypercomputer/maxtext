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

r"""Self-contained Weaver Mixture-of-Transformers architecture.

This module implements the Weaver architecture from top to bottom in one
place without routing through a shared ``Attention`` or ``Decoder`` class:

1. ``WeaverConfig``: Typed dataclass configuration.
2. Sequence packing & stream helpers (``WeaverPackingMetadata``,
   ``build_weaver_packing_metadata``, ``unpack_streams``,
   ``reinterleave_streams``).
3. Sample-fenced causal & bidirectional attention masks + Pallas Splash masks.
4. 3D M-RoPE rotary position embeddings (``rotate_half``,
   ``apply_rotary_pos_emb``, ``compute_3d_mrope_cos_sin``).
5. Dual-pathway attention kernels & ``WeaverDualAttention`` module using pure
   ``flax.nnx`` leaf layers (``nnx.Linear``, ``nnx.RMSNorm``).
6. ``WeaverMLP`` feed-forward block (``silu`` SwiGLU and ``relu2`` variants).
7. ``WeaverMoTDecoderLayer`` dual-pathway Mixture-of-Transformers decoder
   block with canonical weight alias properties.
8. ``WeaverOmniTransformer`` full 36-layer joint multimodal diffusion backbone
   integrating ``embed_tokens``, ``proj_in``, ``time_embedder``, stacked
   ``WeaverMoTDecoderLayer`` blocks (``scan_layers=False`` and
   ``scan_layers=True``), ``norm``, ``norm_moe_gen``, and ``proj_out``.

    ========================================================================
                   WEAVER DECODER LAYER DATA FLOW
    ========================================================================

      und_seq [U, M]                                gen_seq [G, M]
           │                                              │
           ├──────────────┐                ┌──────────────┤
           │      input_layernorm   input_layernorm_moe_gen
           │              │                │              │
           │              └──── WeaverDualAttention ──────┘
           │                    (causal und | full gen)
           │              ┌────────────────┬──────────────┐
           │              │                │              │
           ▼              ▼                ▼              ▼
      residual_und = und_seq + und_attn   residual_gen = gen_seq + gen_attn
           │                                              │
    post_attention_layernorm          post_attention_layernorm_moe_gen
           │                                              │
          mlp                                        mlp_moe_gen
           │                                              │
           ▼                                              ▼
      und_out = residual_und + mlp_out    gen_out = residual_gen + mlp_out
      [U, M]                                              [G, M]
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Any, Literal, overload

from flax import nnx
import jax
from jax import sharding
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from maxtext.kernels.attention import jax_flash_attention
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info
from maxtext.layers import nnx_scan
import numpy as np


# ==============================================================================
# 0. Configuration & Pure-NNX Leaf Layer Builders
# ==============================================================================


def _resolve_jax_dtype(dtype: Any) -> Any:
  """Resolves a string or dtype object to a JAX dtype."""
  if isinstance(dtype, str):
    return getattr(jnp, dtype, jnp.dtype(dtype))
  return dtype


@dataclasses.dataclass
class WeaverConfig:
  """Configuration for Weaver Mixture-of-Transformers blocks and backbone."""

  hidden_size: int = 4096
  head_dim: int = 128
  num_attention_heads: int = 32
  num_key_value_heads: int = 8
  intermediate_size: int = 12288
  num_hidden_layers: int = 36
  vocab_size: int = 151936
  latent_channels: int = 48
  patch_size: int = 2
  time_embed_in_channels: int = 256
  timestep_scale: float = 0.001
  rms_norm_eps: float = 1e-6
  hidden_act: str = "silu"
  attention_bias: bool = False
  qk_norm_for_text: bool = True
  qk_norm_for_diffusion: bool = True
  use_und_k_norm_for_gen: bool = False
  attention_kernel: str = "dot_product"
  use_ragged_ops: bool = False
  mrope_section: tuple[int, int, int] = (24, 20, 20)
  rope_theta: float = 5_000_000.0
  scan_layers: bool = False
  param_scan_axis: int = 0
  remat_policy: str = "none"
  embed_kernel_axes: tuple[None | str, ...] = ("vocab", "embed")
  proj_in_kernel_axes: tuple[None | str, ...] = ("patch", "embed")
  time_mlp_in_kernel_axes: tuple[None | str, ...] = ("time_embed", "embed")
  time_mlp_out_kernel_axes: tuple[None | str, ...] = ("embed", "embed")
  proj_out_kernel_axes: tuple[None | str, ...] = ("embed", "patch")
  q_kernel_axes: tuple[None | str, ...] = ("embed", "heads")
  kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads")
  o_kernel_axes: tuple[None | str, ...] = ("heads", "embed")
  qk_norm_kernel_axes: tuple[None | str, ...] = ()
  mlp_up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp")
  mlp_down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed")
  norm_kernel_axes: tuple[None | str, ...] = ("norm",)
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32

  def __post_init__(self) -> None:
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          "num_attention_heads must be divisible by num_key_value_heads, "
          f"got {self.num_attention_heads=} {self.num_key_value_heads=}"
      )
    self.dtype = _resolve_jax_dtype(self.dtype)
    self.weight_dtype = _resolve_jax_dtype(self.weight_dtype)

  @classmethod
  def from_maxtext_config(cls, config: Any) -> WeaverConfig:
    """Constructs a ``WeaverConfig`` from a MaxText ``pyconfig`` object."""
    if isinstance(config, cls):
      return config

    mlp_acts = getattr(config, "mlp_activations", ["silu", "linear"])
    hidden_act = "relu2" if "relu2" in mlp_acts else "silu"

    raw_attn = str(getattr(config, "attention", "dot_product"))
    attn_kernel = raw_attn if raw_attn in ("dot_product", "flash", "splash") else "dot_product"

    mrope_raw = getattr(config, "mrope_section", (24, 20, 20))
    mrope_tuple = (int(mrope_raw[0]), int(mrope_raw[1]), int(mrope_raw[2]))

    return cls(
        hidden_size=int(getattr(config, "emb_dim", getattr(config, "base_emb_dim", 4096))),
        head_dim=int(getattr(config, "head_dim", 128)),
        num_attention_heads=int(getattr(config, "num_query_heads", getattr(config, "base_num_query_heads", 32))),
        num_key_value_heads=int(getattr(config, "num_kv_heads", getattr(config, "base_num_kv_heads", 8))),
        intermediate_size=int(getattr(config, "mlp_dim", getattr(config, "base_mlp_dim", 12288))),
        num_hidden_layers=int(getattr(config, "num_decoder_layers", getattr(config, "base_num_decoder_layers", 36))),
        vocab_size=int(getattr(config, "vocab_size", 151936)),
        latent_channels=int(getattr(config, "latent_channels", 48)),
        patch_size=int(getattr(config, "patch_size", 2)),
        time_embed_in_channels=int(getattr(config, "time_embed_in_channels", 256)),
        timestep_scale=float(getattr(config, "timestep_scale", 0.001)),
        rms_norm_eps=float(getattr(config, "normalization_layer_epsilon", 1e-6)),
        hidden_act=hidden_act,
        attention_bias=bool(getattr(config, "attention_bias", False)),
        qk_norm_for_text=bool(getattr(config, "qk_norm_for_text", getattr(config, "use_qk_norm", True))),
        qk_norm_for_diffusion=bool(getattr(config, "qk_norm_for_diffusion", True)),
        use_und_k_norm_for_gen=bool(getattr(config, "use_und_k_norm_for_gen", False)),
        attention_kernel=attn_kernel,
        use_ragged_ops=bool(getattr(config, "use_ragged_attention", False)),
        mrope_section=mrope_tuple,
        rope_theta=float(getattr(config, "rope_max_timescale", 5_000_000.0)),
        scan_layers=bool(getattr(config, "scan_layers", False)),
        param_scan_axis=int(getattr(config, "param_scan_axis", 0)),
        remat_policy=str(getattr(config, "remat_policy", "none")),
        dtype=_resolve_jax_dtype(getattr(config, "dtype", jnp.float32)),
        weight_dtype=_resolve_jax_dtype(getattr(config, "weight_dtype", jnp.float32)),
    )


def _adapt_kernel_init(
    kernel_init: Callable[..., jax.Array] | None,
) -> nnx.Initializer:
  """Adapts a 3-arg NNX initializer or 5-arg NdInitializer to a 3-arg NNX initializer."""
  if kernel_init is None:
    return nnx.initializers.lecun_normal()

  def _wrapped(key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32) -> jax.Array:
    try:
      return kernel_init(key, shape, dtype)
    except TypeError:
      in_axis = (0,)
      out_axis = tuple(range(1, len(shape)))
      return kernel_init(key, shape, dtype, in_axis, out_axis)

  return _wrapped


def _make_embed(
    num_embeddings: int,
    features: int,
    *,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    embedding_init: Callable[..., jax.Array] | None = None,
    kernel_axes: tuple[None | str, ...] = ("vocab", "embed"),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "embed_kernel_init",
    rngs: nnx.Rngs,
) -> nnx.Embed:
  """Constructs a pure ``nnx.Embed`` with logical-axis annotations."""
  base_init = _adapt_kernel_init(embedding_init or nnx.initializers.normal(stddev=0.02))
  if sharding_hook is not None:
    base_init = sharding_hook(base_init, hook_name)
  meta_init = nnx.with_metadata(
      base_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  embed = nnx.Embed(
      num_embeddings=num_embeddings,
      features=features,
      dtype=dtype,
      param_dtype=weight_dtype,
      embedding_init=meta_init,
      rngs=rngs,
  )
  embed.kernel_axes = kernel_axes
  return embed


def _make_linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = False,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_init: Callable[..., jax.Array] | None = None,
    kernel_axes: tuple[None | str, ...] = (),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "linear_kernel_init",
    rngs: nnx.Rngs,
) -> nnx.Linear:
  """Constructs a pure ``nnx.Linear`` with logical-axis annotations."""
  base_init = _adapt_kernel_init(kernel_init)
  if sharding_hook is not None:
    base_init = sharding_hook(base_init, hook_name)
  meta_init = nnx.with_metadata(
      base_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  bias_axes = kernel_axes[-1:] if kernel_axes else ()
  bias_init = nnx.with_metadata(
      nnx.initializers.zeros_init(),
      out_sharding=bias_axes,
      kernel_axes=bias_axes,
      eager_sharding=False,
  )
  layer = nnx.Linear(
      in_features=in_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=weight_dtype,
      kernel_init=meta_init,
      bias_init=bias_init,
      rngs=rngs,
  )
  layer.kernel_axes = kernel_axes
  return layer


def _make_rmsnorm(
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: tuple[None | str, ...] = (),
    sharding_hook: Callable[[Any, str], Any] | None = None,
    hook_name: str = "norm_scale_init",
    rngs: nnx.Rngs,
) -> nnx.RMSNorm:
  """Constructs a pure ``nnx.RMSNorm`` with logical-axis annotations."""
  base_scale_init: Any = nnx.initializers.ones_init()
  if sharding_hook is not None:
    hooked = sharding_hook(1, hook_name)
    if callable(hooked):
      base_scale_init = hooked
  meta_scale_init = nnx.with_metadata(
      base_scale_init,
      out_sharding=kernel_axes,
      kernel_axes=kernel_axes,
      eager_sharding=False,
  )
  norm = nnx.RMSNorm(
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      param_dtype=weight_dtype,
      scale_init=meta_scale_init,
      rngs=rngs,
  )
  norm.kernel_axes = kernel_axes
  return norm


# ==============================================================================
# 1. Packing Metadata and Stream Helpers
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class WeaverPackingMetadata:
  """Boundary metadata and stream indexing for packed multimodal sequences.

  Attributes:
    causal_q_offsets: Cumulative offsets for understanding (UND) queries, shape
      [num_samples + 1]. Defines sample boundaries in the UND stream.
    full_q_offsets: Cumulative offsets for generation (GEN) queries, shape
      [num_samples + 1]. Defines sample boundaries in the GEN stream.
    sample_kv_offsets: Cumulative offsets for total tokens per sample in the
      interleaved sequence, shape [num_samples + 1]. Defines sample boundaries
      for the combined KV context.
    packed_und_token_indexes: Global token indices belonging to the UND stream,
      shape [num_und_tokens].
    packed_gen_token_indexes: Global token indices belonging to the GEN stream,
      shape [num_gen_tokens].
    max_causal_len: Maximum number of UND tokens across all samples.
    max_full_len: Maximum number of GEN tokens across all samples.
    max_sample_len: Maximum number of total tokens across all samples.
    num_samples: Number of batch instances B packed into the sequence.
    total_tokens: Total sequence length N_total = N_und + N_gen.
    num_und_tokens: Total number of UND tokens across all samples.
    num_gen_tokens: Total number of GEN tokens across all samples.
  """

  causal_q_offsets: jax.Array
  full_q_offsets: jax.Array
  sample_kv_offsets: jax.Array
  packed_und_token_indexes: jax.Array
  packed_gen_token_indexes: jax.Array
  max_causal_len: int
  max_full_len: int
  max_sample_len: int
  num_samples: int
  total_tokens: int
  num_und_tokens: int
  num_gen_tokens: int


def build_weaver_packing_metadata(
    sample_und_lens: Sequence[int],
    sample_gen_lens: Sequence[int],
    packed_und_token_indexes: jax.Array | None = None,
    packed_gen_token_indexes: jax.Array | None = None,
) -> WeaverPackingMetadata:
  """Constructs WeaverPackingMetadata from per-sample token counts.

  If index arrays are not provided, assumes default contiguous layout per
  sample: each sample b has its UND tokens followed by its GEN tokens.

  Args:
    sample_und_lens: Number of UND tokens for each sample in the batch.
    sample_gen_lens: Number of GEN tokens for each sample in the batch.
    packed_und_token_indexes: Optional explicit global indices for UND tokens.
    packed_gen_token_indexes: Optional explicit global indices for GEN tokens.

  Returns:
    A fully populated WeaverPackingMetadata dataclass.
  """
  num_samples = len(sample_und_lens)
  if len(sample_gen_lens) != num_samples:
    raise ValueError(
        f"Length of sample_und_lens ({num_samples}) does not match " f"sample_gen_lens ({len(sample_gen_lens)})"
    )

  und_lens = list(sample_und_lens)
  gen_lens = list(sample_gen_lens)
  sample_lens = [u + g for u, g in zip(und_lens, gen_lens)]

  causal_q_offsets_list = [0]
  full_q_offsets_list = [0]
  sample_kv_offsets_list = [0]

  for u, g, s in zip(und_lens, gen_lens, sample_lens):
    causal_q_offsets_list.append(causal_q_offsets_list[-1] + u)
    full_q_offsets_list.append(full_q_offsets_list[-1] + g)
    sample_kv_offsets_list.append(sample_kv_offsets_list[-1] + s)

  total_tokens = sample_kv_offsets_list[-1]
  num_und_tokens = causal_q_offsets_list[-1]
  num_gen_tokens = full_q_offsets_list[-1]

  if packed_und_token_indexes is None or packed_gen_token_indexes is None:
    und_idx_list = []
    gen_idx_list = []
    current_offset = 0
    for u, g in zip(und_lens, gen_lens):
      und_idx_list.extend(range(current_offset, current_offset + u))
      gen_idx_list.extend(range(current_offset + u, current_offset + u + g))
      current_offset += u + g
    packed_und_token_indexes = jnp.array(und_idx_list, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.array(gen_idx_list, dtype=jnp.int32)
  else:
    packed_und_token_indexes = jnp.asarray(packed_und_token_indexes, dtype=jnp.int32)
    packed_gen_token_indexes = jnp.asarray(packed_gen_token_indexes, dtype=jnp.int32)

  max_causal = max(und_lens) if und_lens else 0
  max_full = max(gen_lens) if gen_lens else 0
  max_sample = max(sample_lens) if sample_lens else 0

  return WeaverPackingMetadata(
      causal_q_offsets=jnp.array(causal_q_offsets_list, dtype=jnp.int32),
      full_q_offsets=jnp.array(full_q_offsets_list, dtype=jnp.int32),
      sample_kv_offsets=jnp.array(sample_kv_offsets_list, dtype=jnp.int32),
      packed_und_token_indexes=packed_und_token_indexes,
      packed_gen_token_indexes=packed_gen_token_indexes,
      max_causal_len=max_causal,
      max_full_len=max_full,
      max_sample_len=max_sample,
      num_samples=num_samples,
      total_tokens=total_tokens,
      num_und_tokens=num_und_tokens,
      num_gen_tokens=num_gen_tokens,
  )


def unpack_streams(
    packed_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    use_ragged_ops: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Unpacks a single 1D packed sequence into understanding and generation streams.

  Args:
    packed_tokens: Tensor of shape (N_total, D_model) containing all packed
      tokens.
    packed_und_token_indexes: 1D array of indices for UND tokens.
    packed_gen_token_indexes: 1D array of indices for GEN tokens.
    use_ragged_ops: Whether to use TPU SparseCore ragged gather kernels if
      available. Default is False (standard JAX indexing).

  Returns:
    und_tokens: Tensor of shape (N_und, D_model).
    gen_tokens: Tensor of shape (N_gen, D_model).
  """
  del use_ragged_ops
  und_tokens = packed_tokens[packed_und_token_indexes]
  gen_tokens = packed_tokens[packed_gen_token_indexes]
  return und_tokens, gen_tokens


def reinterleave_streams(
    und_tokens: jax.Array,
    gen_tokens: jax.Array,
    packed_und_token_indexes: jax.Array,
    packed_gen_token_indexes: jax.Array,
    total_tokens: int,
    use_ragged_ops: bool = False,
) -> jax.Array:
  """Re-interleaves sliced understanding and generation outputs into the global packed order.

  Args:
    und_tokens: Tensor of shape (N_und, D_model) from the understanding stream.
    gen_tokens: Tensor of shape (N_gen, D_model) from the generation stream.
    packed_und_token_indexes: 1D array of indices where UND tokens belong in the
      packed sequence.
    packed_gen_token_indexes: 1D array of indices where GEN tokens belong in the
      packed sequence.
    total_tokens: Total sequence length N_total.
    use_ragged_ops: Whether to use TPU SparseCore ragged scatter kernels if
      available. Default is False (standard JAX dynamic tensor scatter).

  Returns:
    A single merged tensor of shape (N_total, D_model) matching the original
    global layout.
  """
  del use_ragged_ops
  trailing_shape = und_tokens.shape[1:]
  output_tokens = jnp.zeros((total_tokens, *trailing_shape), dtype=und_tokens.dtype)
  if und_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_und_token_indexes].set(und_tokens)
  if gen_tokens.shape[0] > 0:
    output_tokens = output_tokens.at[packed_gen_token_indexes].set(gen_tokens)
  return output_tokens


# ==============================================================================
# 2. Attention Mask Construction
# ==============================================================================


def build_causal_understanding_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
) -> jax.Array:
  """Builds sample-specific lower-triangular causal attention mask for Kernel 1.

  Cross-sample interference is strictly prohibited via offset fences:
  tokens can only attend to earlier tokens within the same sample.

  Args:
    causal_q_offsets: Array of shape [num_samples + 1] giving cumulative UND
      token counts.
    num_und_tokens: Total number of UND tokens N_und.

  Returns:
    A boolean array of shape [num_und_tokens, num_und_tokens], where True
    indicates valid attention and False indicates masked-out positions.
  """
  if num_und_tokens == 0:
    return jnp.zeros((0, 0), dtype=jnp.bool_)

  indices = jnp.arange(num_und_tokens)
  sample_ids = jnp.sum(indices[:, None] >= causal_q_offsets[None, 1:], axis=-1)

  same_sample = sample_ids[:, None] == sample_ids[None, :]
  is_causal = indices[:, None] >= indices[None, :]
  return same_sample & is_causal


def build_full_generative_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
) -> jax.Array:
  """Builds bidirectional full generative attention mask for Kernel 2.

  Enables generation queries to attend to all preceding context (both
  textual/UND and visual/GEN tokens) within the same sample while strictly
  prohibiting cross-sample interference.

  Args:
    full_q_offsets: Array of shape [num_samples + 1] giving cumulative GEN query
      counts.
    sample_kv_offsets: Array of shape [num_samples + 1] giving cumulative total
      token counts.
    num_gen_tokens: Total number of GEN query tokens N_gen.
    total_tokens: Total sequence length N_total (combined UND + GEN).

  Returns:
    A boolean array of shape [num_gen_tokens, total_tokens], where True
    indicates valid attention and False indicates masked-out positions.
  """
  if num_gen_tokens == 0 or total_tokens == 0:
    return jnp.zeros((num_gen_tokens, total_tokens), dtype=jnp.bool_)

  q_indices = jnp.arange(num_gen_tokens)
  kv_indices = jnp.arange(total_tokens)

  q_sample_ids = jnp.sum(q_indices[:, None] >= full_q_offsets[None, 1:], axis=-1)
  kv_sample_ids = jnp.sum(kv_indices[:, None] >= sample_kv_offsets[None, 1:], axis=-1)

  return q_sample_ids[:, None] == kv_sample_ids[None, :]


def build_causal_understanding_splash_mask(
    causal_q_offsets: jax.Array,
    num_und_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 1 (Causal Understanding)."""
  dense_mask = build_causal_understanding_mask(causal_q_offsets, num_und_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def build_full_generative_splash_mask(
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    num_gen_tokens: int,
    total_tokens: int,
):
  """Builds a Pallas Splash Attention Mask for Kernel 2 (Full Generative)."""
  dense_mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, num_gen_tokens, total_tokens)
  return splash_attention_mask.NumpyMask(np.asarray(dense_mask, dtype=np.bool_))


def compile_weaver_splash_mask(
    mask_array: jax.Array | np.ndarray,
    block_q: int = 128,
    block_kv: int = 128,
):
  """Compiles a dense boolean mask into Pallas Splash Attention MaskInfo."""
  mask_np = np.asarray(mask_array, dtype=np.bool_)
  q_len, kv_len = mask_np.shape
  if q_len % block_q != 0 or kv_len % block_kv != 0:
    raise ValueError(
        f"Sequence lengths ({q_len}, {kv_len}) must be multiples of block sizes"
        f" ({block_q}, {block_kv}) for Splash Attention compilation."
    )
  return splash_attention_mask_info.process_dynamic_mask(mask_np, (block_q, block_kv))


# ==============================================================================
# 3. Rotary Position Embedding Helpers (3D M-RoPE)
# ==============================================================================


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dimensions of the input tensor."""
  half_dim = x.shape[-1] // 2
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    unsqueeze_dim: int | None = 1,
) -> jax.Array:
  """Applies Rotary Position Embedding (RoPE / 3D M-RoPE) to an activation tensor."""
  if unsqueeze_dim is not None:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

  x_f32 = x.astype(jnp.float32)
  cos_f32 = cos.astype(jnp.float32)
  sin_f32 = sin.astype(jnp.float32)

  rotated = (x_f32 * cos_f32) + (rotate_half(x_f32) * sin_f32)
  return rotated.astype(x.dtype)


def compute_3d_mrope_cos_sin(
    position_ids_3d: jax.Array,
    head_dim: int,
    mrope_section: tuple[int, int, int] = (24, 20, 20),
    rope_theta: float = 1000000.0,
) -> tuple[jax.Array, jax.Array]:
  """Computes interleaved 3D M-RoPE cosine and sine tensors from 3D coordinates."""
  if sum(mrope_section) != head_dim // 2:
    raise ValueError(f"mrope_section {mrope_section} sum must equal head_dim // 2" f" ({head_dim // 2})")

  freq_dim = head_dim // 2
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, freq_dim, 1, dtype=jnp.float32) * 2 / head_dim))

  pos_temporal = position_ids_3d[:, 0, None].astype(jnp.float32)
  freqs = pos_temporal * inv_freq[None, :]

  pos_height = position_ids_3d[:, 1, None].astype(jnp.float32)
  pos_width = position_ids_3d[:, 2, None].astype(jnp.float32)

  h_limit = mrope_section[1] * 3
  w_limit = mrope_section[2] * 3

  h_indices = jnp.arange(1, min(h_limit, freq_dim), 3)
  w_indices = jnp.arange(2, min(w_limit, freq_dim), 3)

  freqs = freqs.at[:, h_indices].set(pos_height * inv_freq[h_indices])
  freqs = freqs.at[:, w_indices].set(pos_width * inv_freq[w_indices])

  emb = jnp.concatenate([freqs, freqs], axis=-1)
  return jnp.cos(emb), jnp.sin(emb)


# ==============================================================================
# 4. Dual-Attention Functional Kernels
# ==============================================================================


def causal_understanding_attention(
    q_und: jax.Array,
    k_und: jax.Array,
    v_und: jax.Array,
    causal_q_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
) -> jax.Array:
  """Kernel 1: Causal Understanding Attention with cross-sample fences."""
  n_und, num_heads, head_dim = q_und.shape
  if n_und == 0:
    return jnp.zeros_like(q_und)

  num_kv_heads = k_und.shape[1]
  if num_heads != num_kv_heads:
    repeat_factor = num_heads // num_kv_heads
    k_und = jnp.repeat(k_und, repeat_factor, axis=1)
    v_und = jnp.repeat(v_und, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    block_q = 128
    block_kv = 128
    if n_und % block_q != 0:
      raise ValueError(
          "FlashAttention requires sequence length divisible by block_q"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]
    k_reshaped = k_und.transpose(1, 0, 2)[jnp.newaxis, ...]
    v_reshaped = v_und.transpose(1, 0, 2)[jnp.newaxis, ...]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_und.dtype)

  elif attention_kernel == "splash":
    block_q = 128
    block_kv = 128
    if n_und % block_q != 0:
      raise ValueError(
          "Splash Attention requires sequence length divisible by block size"
          f" ({block_q}), but got N_und={n_und}. Use"
          " attention_kernel='dot_product' for unaligned lengths."
      )
    splash_mask = build_causal_understanding_splash_mask(causal_q_offsets, n_und)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_und * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)
    k_reshaped = k_und.transpose(1, 0, 2)
    v_reshaped = v_und.transpose(1, 0, 2)
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)
    out = splash_out.transpose(1, 0, 2)
    return out.astype(q_und.dtype)

  elif attention_kernel == "dot_product":
    scores = jnp.einsum("qhd,khd->hqk", q_und, k_und) * scale
    mask = build_causal_understanding_mask(causal_q_offsets, n_und)
    scores = jnp.where(mask[None, :, :], scores, mask_value)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_und.dtype)
    return jnp.einsum("hqk,khd->qhd", attn_weights, v_und)
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )


def full_generative_attention(
    q_gen: jax.Array,
    k_all: jax.Array,
    v_all: jax.Array,
    full_q_offsets: jax.Array,
    sample_kv_offsets: jax.Array,
    scale: float | None = None,
    mask_value: float = -1e9,
    attention_kernel: str = "dot_product",
) -> jax.Array:
  """Kernel 2: Full Generative Attention with cross-sample fences."""
  n_gen, num_heads, head_dim = q_gen.shape
  total_tokens = k_all.shape[0]
  if n_gen == 0 or total_tokens == 0:
    return jnp.zeros_like(q_gen)

  num_kv_heads = k_all.shape[1]
  if num_heads != num_kv_heads:
    repeat_factor = num_heads // num_kv_heads
    k_all = jnp.repeat(k_all, repeat_factor, axis=1)
    v_all = jnp.repeat(v_all, repeat_factor, axis=1)

  if scale is None:
    scale = 1.0 / math.sqrt(head_dim)

  if attention_kernel == "flash":
    block_q = 128
    block_kv = 128
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "FlashAttention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)[jnp.newaxis, ...]
    k_reshaped = k_all.transpose(1, 0, 2)[jnp.newaxis, ...]
    v_reshaped = v_all.transpose(1, 0, 2)[jnp.newaxis, ...]
    flash_out = jax_flash_attention.flash_attention_block_masked(
        q=q_reshaped,
        k=k_reshaped,
        v=v_reshaped,
        segment_ids=None,
        block_q=block_q,
        block_kv=block_kv,
        mask=mask,
        mask_value=mask_value,
    )
    out = jnp.squeeze(flash_out, axis=0).transpose(1, 0, 2)
    return out.astype(q_gen.dtype)

  elif attention_kernel == "splash":
    block_q = 128
    block_kv = 128
    if n_gen % block_q != 0 or total_tokens % block_kv != 0:
      raise ValueError(
          "Splash Attention requires sequence lengths divisible by block sizes"
          f" (block_q={block_q}, block_kv={block_kv}), but got N_gen={n_gen},"
          f" N_total={total_tokens}. Use attention_kernel='dot_product' for"
          " unaligned lengths."
      )
    splash_mask = build_full_generative_splash_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    sa_config = dataclasses.replace(
        splash_attention_kernel.SplashConfig.get_default(),
        block_q=block_q,
        block_kv=block_kv,
    )
    splash_kernel_fn = splash_attention_kernel.make_splash_mha_single_device(
        mask=splash_mask,
        config=sa_config,
        downcast_smem_data=False,
        partial_mask_blocks_dtype=np.int32,
    )
    q_scaled = q_gen * scale
    q_reshaped = q_scaled.transpose(1, 0, 2)
    k_reshaped = k_all.transpose(1, 0, 2)
    v_reshaped = v_all.transpose(1, 0, 2)
    splash_out = splash_kernel_fn(q_reshaped, k_reshaped, v_reshaped)
    out = splash_out.transpose(1, 0, 2)
    return out.astype(q_gen.dtype)

  elif attention_kernel == "dot_product":
    scores = jnp.einsum("qhd,khd->hqk", q_gen, k_all) * scale
    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, n_gen, total_tokens)
    scores = jnp.where(mask[None, :, :], scores, mask_value)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q_gen.dtype)
    return jnp.einsum("hqk,khd->qhd", attn_weights, v_all)
  else:
    raise ValueError(
        f"Unsupported attention_kernel '{attention_kernel}'. " "Expected one of ['dot_product', 'flash', 'splash']."
    )


# ==============================================================================
# 5. Weaver Dual-Attention Module (Pure Flax NNX)
# ==============================================================================


class WeaverDualAttention(nnx.Module):
  """Dual-pathway packed attention for Weaver using pure ``flax.nnx`` layers.

  Composes ``nnx.Linear`` and ``nnx.RMSNorm`` directly with zero wrapper layer
  classes.
  """

  def __init__(
      self,
      dim: int | WeaverConfig,
      num_heads: int | None = None,
      num_kv_heads: int | None = None,
      head_dim: int | None = None,
      *,
      qk_norm_for_text: bool = True,
      qk_norm_for_diffusion: bool = True,
      use_und_k_norm_for_gen: bool = False,
      use_bias: bool = False,
      rms_norm_eps: float = 1e-6,
      attention_kernel: str = "dot_product",
      use_ragged_ops: bool = False,
      q_kernel_axes: tuple[None | str, ...] = ("embed", "heads"),
      kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads"),
      o_kernel_axes: tuple[None | str, ...] = ("heads", "embed"),
      qk_norm_kernel_axes: tuple[None | str, ...] = (),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes ``WeaverDualAttention`` from explicit arguments or ``WeaverConfig``."""
    del mesh
    if isinstance(dim, WeaverConfig):
      cfg = dim
      dim = cfg.hidden_size
      num_heads = cfg.num_attention_heads
      num_kv_heads = cfg.num_key_value_heads
      head_dim = cfg.head_dim
      qk_norm_for_text = cfg.qk_norm_for_text
      qk_norm_for_diffusion = cfg.qk_norm_for_diffusion
      use_und_k_norm_for_gen = cfg.use_und_k_norm_for_gen
      use_bias = cfg.attention_bias
      rms_norm_eps = cfg.rms_norm_eps
      attention_kernel = cfg.attention_kernel
      use_ragged_ops = cfg.use_ragged_ops
      q_kernel_axes = cfg.q_kernel_axes
      kv_kernel_axes = cfg.kv_kernel_axes
      o_kernel_axes = cfg.o_kernel_axes
      qk_norm_kernel_axes = cfg.qk_norm_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if num_heads is None or num_kv_heads is None or head_dim is None:
      raise ValueError("num_heads, num_kv_heads, and head_dim are required when dim is an int.")

    if attention_kernel not in ("dot_product", "flash", "splash"):
      raise ValueError(
          f"Unsupported attention_kernel '{attention_kernel}'. "
          "Supported kernels are ['dot_product', 'flash', 'splash']."
      )

    self.attention_kernel = attention_kernel
    self.use_ragged_ops = use_ragged_ops
    self.dim = dim
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.qk_norm_for_text = qk_norm_for_text
    self.qk_norm_for_diffusion = qk_norm_for_diffusion
    self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
    self.q_kernel_axes = q_kernel_axes
    self.kv_kernel_axes = kv_kernel_axes
    self.o_kernel_axes = o_kernel_axes
    self.qk_norm_kernel_axes = qk_norm_kernel_axes
    self.sharding_hook = sharding_hook or (lambda x, name: x)
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scaling = 1.0 / math.sqrt(head_dim)

    # Understanding Pathway Projections (pure nnx.Linear)
    self.q_proj = _make_linear(
        dim,
        num_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=q_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="q_proj_kernel_init",
        rngs=rngs,
    )
    self.k_proj = _make_linear(
        dim,
        num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=kv_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="k_proj_kernel_init",
        rngs=rngs,
    )
    self.v_proj = _make_linear(
        dim,
        num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=kv_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="v_proj_kernel_init",
        rngs=rngs,
    )
    self.o_proj = _make_linear(
        num_heads * head_dim,
        dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=o_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="o_proj_kernel_init",
        rngs=rngs,
    )

    # Understanding QK Normalization (pure nnx.RMSNorm)
    if qk_norm_for_text:
      self.q_norm = _make_rmsnorm(
          head_dim,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="qk_norm_scale_init",
          rngs=rngs,
      )
      self.k_norm = _make_rmsnorm(
          head_dim,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="qk_norm_scale_init",
          rngs=rngs,
      )
    else:
      self.q_norm = None
      self.k_norm = None

    # Generation Pathway Projections (pure nnx.Linear)
    self.q_proj_gen = _make_linear(
        dim,
        num_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=q_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="q_proj_gen_kernel_init",
        rngs=rngs,
    )
    self.k_proj_gen = _make_linear(
        dim,
        num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=kv_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="k_proj_gen_kernel_init",
        rngs=rngs,
    )
    self.v_proj_gen = _make_linear(
        dim,
        num_kv_heads * head_dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=kv_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="v_proj_gen_kernel_init",
        rngs=rngs,
    )
    self.o_proj_gen = _make_linear(
        num_heads * head_dim,
        dim,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=o_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="o_proj_gen_kernel_init",
        rngs=rngs,
    )

    # Generation QK Normalization (pure nnx.RMSNorm)
    if qk_norm_for_diffusion:
      self.q_norm_gen = _make_rmsnorm(
          head_dim,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="qk_norm_scale_init",
          rngs=rngs,
      )
      self.k_norm_gen = _make_rmsnorm(
          head_dim,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="qk_norm_scale_init",
          rngs=rngs,
      )
    else:
      self.q_norm_gen = None
      self.k_norm_gen = None

    # Cross-Attention UND K Normalization (pure nnx.RMSNorm)
    if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
      self.k_norm_und_for_gen = _make_rmsnorm(
          head_dim,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=qk_norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="qk_norm_scale_init",
          rngs=rngs,
      )
    else:
      self.k_norm_und_for_gen = None

  @classmethod
  def from_config(
      cls,
      config: WeaverConfig,
      *,
      rngs: nnx.Rngs,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
  ) -> WeaverDualAttention:
    """Constructs ``WeaverDualAttention`` from a ``WeaverConfig``."""
    return cls(config, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs)

  @overload
  def __call__(
      self,
      tokens: jax.Array,
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: Literal[True] = True,
  ) -> jax.Array:
    ...

  @overload
  def __call__(
      self,
      tokens: tuple[jax.Array, jax.Array],
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    ...

  @overload
  def __call__(
      self,
      tokens: jax.Array,
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: Literal[False],
  ) -> tuple[jax.Array, jax.Array]:
    ...

  def __call__(
      self,
      tokens: jax.Array | tuple[jax.Array, jax.Array],
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
      *,
      reinterleave: bool = True,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Executes dual-pathway attention on understanding and generation tokens.

    Input shapes and layouts:
      tokens:
        * Packed mode: 2D array of shape ``[N_total, dim]`` in global packed
          order, where ``N_total = N_und + N_gen``.
        * Unpacked mode: Tuple ``(und_tokens, gen_tokens)`` with shapes
          ``[N_und, dim]`` and ``[N_gen, dim]``.
      metadata: ``WeaverPackingMetadata`` containing cumulative sample
        boundaries (``causal_q_offsets``: ``[B + 1]``, ``full_q_offsets``:
        ``[B + 1]``, ``sample_kv_offsets``: ``[B + 1]``) and 1D index arrays
        (``packed_und_token_indexes``: ``[N_und]``,
        ``packed_gen_token_indexes``: ``[N_gen]``).
      cos: Optional 3D M-RoPE cosine table of shape ``[N_total, head_dim]``.
      sin: Optional 3D M-RoPE sine table of shape ``[N_total, head_dim]``.
      reinterleave: When True and ``tokens`` is a 2D array, re-interleaves the
        outputs back into global packed order.

    Output shapes and layouts:
      * When ``reinterleave=True`` and ``tokens`` is 2D: Transformed packed array
        of shape ``[N_total, dim]`` in global packed order.
      * When ``reinterleave=False`` or ``tokens`` is a tuple: Tuple
        ``(und_out, gen_out)`` of shapes ``[N_und, dim]`` and ``[N_gen, dim]``.
    """
    hook = self.sharding_hook
    if isinstance(tokens, tuple):
      und_tokens, gen_tokens = tokens
      should_reinterleave = False
    else:
      und_tokens, gen_tokens = unpack_streams(
          tokens,
          metadata.packed_und_token_indexes,
          metadata.packed_gen_token_indexes,
          use_ragged_ops=self.use_ragged_ops,
      )
      should_reinterleave = reinterleave

    n_und = und_tokens.shape[0]
    n_gen = gen_tokens.shape[0]
    total_tokens = metadata.total_tokens

    # 1. Linear Projections
    if n_und > 0:
      q_und = self.q_proj(und_tokens).reshape(n_und, self.num_heads, self.head_dim)
      k_und = self.k_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
      v_und = self.v_proj(und_tokens).reshape(n_und, self.num_kv_heads, self.head_dim)
    else:
      q_und = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_und = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    if n_gen > 0:
      q_gen = self.q_proj_gen(gen_tokens).reshape(n_gen, self.num_heads, self.head_dim)
      k_gen = self.k_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
      v_gen = self.v_proj_gen(gen_tokens).reshape(n_gen, self.num_kv_heads, self.head_dim)
    else:
      q_gen = jnp.zeros((0, self.num_heads, self.head_dim), dtype=self.dtype)
      k_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)
      v_gen = jnp.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype)

    # 2. QK Normalization
    if self.q_norm is not None and n_und > 0:
      q_und = self.q_norm(q_und)
    if self.k_norm is not None and n_und > 0:
      k_und = self.k_norm(k_und)

    if self.q_norm_gen is not None and n_gen > 0:
      q_gen = self.q_norm_gen(q_gen)
    if self.k_norm_gen is not None and n_gen > 0:
      k_gen = self.k_norm_gen(k_gen)

    # 3. Apply 3D M-RoPE
    k_und_pre_rope = k_und
    if cos is not None and sin is not None:
      if n_und > 0:
        cos_und = cos[metadata.packed_und_token_indexes]
        sin_und = sin[metadata.packed_und_token_indexes]
        q_und = apply_rotary_pos_emb(q_und, cos_und, sin_und, unsqueeze_dim=1)
        k_und = apply_rotary_pos_emb(k_und, cos_und, sin_und, unsqueeze_dim=1)
      if n_gen > 0:
        cos_gen = cos[metadata.packed_gen_token_indexes]
        sin_gen = sin[metadata.packed_gen_token_indexes]
        q_gen = apply_rotary_pos_emb(q_gen, cos_gen, sin_gen, unsqueeze_dim=1)
        k_gen = apply_rotary_pos_emb(k_gen, cos_gen, sin_gen, unsqueeze_dim=1)

    q_und = checkpoint_name(hook(q_und, "query"), "query_proj")
    k_und = checkpoint_name(hook(k_und, "key"), "key_proj")
    v_und = checkpoint_name(hook(v_und, "value"), "value_proj")
    q_gen = checkpoint_name(hook(q_gen, "gen_query"), "gen_query_proj")
    k_gen = checkpoint_name(hook(k_gen, "gen_key"), "gen_key_proj")
    v_gen = checkpoint_name(hook(v_gen, "gen_value"), "gen_value_proj")

    # 4. Cross-Attention UND K Normalization (if configured)
    if self.k_norm_und_for_gen is not None and n_und > 0:
      k_und_norm_for_gen = self.k_norm_und_for_gen(k_und_pre_rope)
      if cos is not None and sin is not None:
        k_und_for_gen = apply_rotary_pos_emb(k_und_norm_for_gen, cos_und, sin_und, unsqueeze_dim=1)
      else:
        k_und_for_gen = k_und_norm_for_gen
    else:
      k_und_for_gen = k_und

    # 5. Assemble Interleaved K_all and V_all for Kernel 2
    k_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    v_all = jnp.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
    if n_und > 0:
      k_all = k_all.at[metadata.packed_und_token_indexes].set(k_und_for_gen)
      v_all = v_all.at[metadata.packed_und_token_indexes].set(v_und)
    if n_gen > 0:
      k_all = k_all.at[metadata.packed_gen_token_indexes].set(k_gen)
      v_all = v_all.at[metadata.packed_gen_token_indexes].set(v_gen)

    # 6. Kernel 1: Causal Understanding Attention
    if n_und > 0:
      und_attn_context = causal_understanding_attention(
          q_und=q_und,
          k_und=k_und,
          v_und=v_und,
          causal_q_offsets=metadata.causal_q_offsets,
          scale=self.scaling,
          attention_kernel=self.attention_kernel,
      )
      und_attn_context = checkpoint_name(hook(und_attn_context, "attn_out"), "attention_out")
      und_attn_flat = und_attn_context.reshape(n_und, self.num_heads * self.head_dim)
      und_out = hook(self.o_proj(und_attn_flat), "post_attn")
    else:
      und_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    # 7. Kernel 2: Full Generative Attention
    if n_gen > 0:
      gen_attn_context = full_generative_attention(
          q_gen=q_gen,
          k_all=k_all,
          v_all=v_all,
          full_q_offsets=metadata.full_q_offsets,
          sample_kv_offsets=metadata.sample_kv_offsets,
          scale=self.scaling,
          attention_kernel=self.attention_kernel,
      )
      gen_attn_context = checkpoint_name(hook(gen_attn_context, "gen_attn_out"), "gen_attention_out")
      gen_attn_flat = gen_attn_context.reshape(n_gen, self.num_heads * self.head_dim)
      gen_out = hook(self.o_proj_gen(gen_attn_flat), "gen_post_attn")
    else:
      gen_out = jnp.zeros((0, self.dim), dtype=self.dtype)

    if should_reinterleave:
      return reinterleave_streams(
          und_tokens=und_out,
          gen_tokens=gen_out,
          packed_und_token_indexes=metadata.packed_und_token_indexes,
          packed_gen_token_indexes=metadata.packed_gen_token_indexes,
          total_tokens=metadata.total_tokens,
          use_ragged_ops=self.use_ragged_ops,
      )

    return und_out, gen_out


WeaverAttention = WeaverDualAttention


# ==============================================================================
# 6. Feed-Forward Network (Pure Flax NNX)
# ==============================================================================


class WeaverMLP(nnx.Module):
  """Feed-forward network for a single Weaver tower using pure ``nnx.Linear``.

  Supports two activation variants:
  * ``silu``  - gated SwiGLU: ``down(silu(gate(x)) * up(x))`` (weaver-mini)
  * ``relu2`` - squared ReLU: ``down(relu(up(x)) ** 2)`` (weaver-max)
  """

  def __init__(
      self,
      hidden_size: int | WeaverConfig,
      intermediate_size: int | None = None,
      *,
      hidden_act: str = "silu",
      use_bias: bool = False,
      up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp"),
      down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed"),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes the Weaver feed-forward block."""
    del mesh
    if isinstance(hidden_size, WeaverConfig):
      cfg = hidden_size
      hidden_size = cfg.hidden_size
      intermediate_size = cfg.intermediate_size
      hidden_act = cfg.hidden_act
      up_kernel_axes = cfg.mlp_up_kernel_axes
      down_kernel_axes = cfg.mlp_down_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if intermediate_size is None:
      raise ValueError("intermediate_size is required when hidden_size is an int.")

    if hidden_act not in ("silu", "relu2"):
      raise ValueError(f"Weaver only supports `hidden_act` values 'silu' and 'relu2', got {hidden_act!r}.")

    self.hidden_act = hidden_act
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.up_kernel_axes = up_kernel_axes
    self.down_kernel_axes = down_kernel_axes
    self.sharding_hook = sharding_hook or (lambda x, name: x)

    self.gate_proj = (
        _make_linear(
            hidden_size,
            intermediate_size,
            use_bias=use_bias,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=kernel_init,
            kernel_axes=up_kernel_axes,
            sharding_hook=sharding_hook,
            hook_name="gate_proj_kernel_init",
            rngs=rngs,
        )
        if hidden_act == "silu"
        else None
    )
    self.up_proj = _make_linear(
        hidden_size,
        intermediate_size,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=up_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="up_proj_kernel_init",
        rngs=rngs,
    )
    self.down_proj = _make_linear(
        intermediate_size,
        hidden_size,
        use_bias=use_bias,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=down_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="down_proj_kernel_init",
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies the feed-forward transformation.

    Input shape and layout:
      x: Activation array of shape ``[N, hidden_size]``, where ``N`` is the
        number of tokens in the tower (e.g. ``N_und`` or ``N_gen``).

    Output shape and layout:
      Transformed activations of shape ``[N, hidden_size]``.
    """
    hook = self.sharding_hook
    if self.hidden_act == "relu2":
      hidden = jnp.square(jax.nn.relu(self.up_proj(x)))
    else:
      hidden = jax.nn.silu(self.gate_proj(x)) * self.up_proj(x)
    hidden = checkpoint_name(hook(hidden, "mlpwi"), "mlpwi")
    return hook(self.down_proj(hidden), "post_mlp")


# ==============================================================================
# 7. Dual-Pathway Mixture-of-Transformers Decoder Layer (Pure Flax NNX)
# ==============================================================================


class WeaverMoTDecoderLayer(nnx.Module):
  """Dual-pathway Weaver Mixture-of-Transformers decoder layer.

  Encapsulates two parallel transformer towers that share one fused
  dual-attention step:

  * Understanding pathway: ``input_layernorm``, ``to_q`` / ``to_k`` / ``to_v``
    / ``to_out``, ``norm_q`` / ``norm_k``, ``post_attention_layernorm``,
    ``mlp``.
  * Generation pathway: ``input_layernorm_moe_gen``, ``add_q_proj`` /
    ``add_k_proj`` / ``add_v_proj`` / ``to_add_out``, ``norm_added_q`` /
    ``norm_added_k``, ``post_attention_layernorm_moe_gen``, ``mlp_moe_gen``.
  """

  def __init__(
      self,
      hidden_size: int | WeaverConfig,
      head_dim: int | None = None,
      num_attention_heads: int | None = None,
      num_key_value_heads: int | None = None,
      intermediate_size: int | None = None,
      *,
      attention_bias: bool = False,
      rms_norm_eps: float = 1e-6,
      hidden_act: str = "silu",
      qk_norm_for_text: bool = True,
      qk_norm_for_diffusion: bool = True,
      use_und_k_norm_for_gen: bool = False,
      attention_kernel: str = "dot_product",
      use_ragged_ops: bool = False,
      q_kernel_axes: tuple[None | str, ...] = ("embed", "heads"),
      kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads"),
      o_kernel_axes: tuple[None | str, ...] = ("heads", "embed"),
      qk_norm_kernel_axes: tuple[None | str, ...] = (),
      mlp_up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp"),
      mlp_down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed"),
      norm_kernel_axes: tuple[None | str, ...] = ("norm",),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes the Weaver Mixture-of-Transformers decoder layer."""
    del mesh
    if isinstance(hidden_size, WeaverConfig):
      cfg = hidden_size
      hidden_size = cfg.hidden_size
      head_dim = cfg.head_dim
      num_attention_heads = cfg.num_attention_heads
      num_key_value_heads = cfg.num_key_value_heads
      intermediate_size = cfg.intermediate_size
      attention_bias = cfg.attention_bias
      rms_norm_eps = cfg.rms_norm_eps
      hidden_act = cfg.hidden_act
      qk_norm_for_text = cfg.qk_norm_for_text
      qk_norm_for_diffusion = cfg.qk_norm_for_diffusion
      use_und_k_norm_for_gen = cfg.use_und_k_norm_for_gen
      attention_kernel = cfg.attention_kernel
      use_ragged_ops = cfg.use_ragged_ops
      q_kernel_axes = cfg.q_kernel_axes
      kv_kernel_axes = cfg.kv_kernel_axes
      o_kernel_axes = cfg.o_kernel_axes
      qk_norm_kernel_axes = cfg.qk_norm_kernel_axes
      mlp_up_kernel_axes = cfg.mlp_up_kernel_axes
      mlp_down_kernel_axes = cfg.mlp_down_kernel_axes
      norm_kernel_axes = cfg.norm_kernel_axes
      dtype = cfg.dtype
      weight_dtype = cfg.weight_dtype

    if head_dim is None or num_attention_heads is None or num_key_value_heads is None or intermediate_size is None:
      raise ValueError(
          "head_dim, num_attention_heads, num_key_value_heads, and intermediate_size "
          "are required when hidden_size is an int."
      )

    self.hidden_size = hidden_size
    self.head_dim = head_dim
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.intermediate_size = intermediate_size
    self.rms_norm_eps = rms_norm_eps
    self.hidden_act = hidden_act
    self.dtype = dtype
    self.sharding_hook = sharding_hook or (lambda x, name: x)

    self.self_attn = WeaverDualAttention(
        dim=hidden_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        qk_norm_for_text=qk_norm_for_text,
        qk_norm_for_diffusion=qk_norm_for_diffusion,
        use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        use_bias=attention_bias,
        rms_norm_eps=rms_norm_eps,
        attention_kernel=attention_kernel,
        use_ragged_ops=use_ragged_ops,
        q_kernel_axes=q_kernel_axes,
        kv_kernel_axes=kv_kernel_axes,
        o_kernel_axes=o_kernel_axes,
        qk_norm_kernel_axes=qk_norm_kernel_axes,
        sharding_hook=sharding_hook,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        rngs=rngs,
    )

    def _norm() -> nnx.RMSNorm:
      return _make_rmsnorm(
          hidden_size,
          epsilon=rms_norm_eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_axes=norm_kernel_axes,
          sharding_hook=sharding_hook,
          hook_name="norm_scale_init",
          rngs=rngs,
      )

    def _mlp() -> WeaverMLP:
      return WeaverMLP(
          hidden_size=hidden_size,
          intermediate_size=intermediate_size,
          hidden_act=hidden_act,
          up_kernel_axes=mlp_up_kernel_axes,
          down_kernel_axes=mlp_down_kernel_axes,
          sharding_hook=sharding_hook,
          dtype=dtype,
          weight_dtype=weight_dtype,
          kernel_init=kernel_init,
          rngs=rngs,
      )

    # Understanding tower.
    self.input_layernorm = _norm()
    self.post_attention_layernorm = _norm()
    self.mlp = _mlp()

    # Generation tower.
    self.input_layernorm_moe_gen = _norm()
    self.post_attention_layernorm_moe_gen = _norm()
    self.mlp_moe_gen = _mlp()

  @classmethod
  def from_config(
      cls,
      config: WeaverConfig,
      *,
      rngs: nnx.Rngs,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
  ) -> WeaverMoTDecoderLayer:
    """Constructs ``WeaverMoTDecoderLayer`` from a ``WeaverConfig``."""
    return cls(config, sharding_hook=sharding_hook, kernel_init=kernel_init, rngs=rngs)

  # ---------------------------------------------------------------------
  # Canonical aliases onto the shared attention submodule.
  # ---------------------------------------------------------------------

  @property
  def to_q(self) -> nnx.Linear:
    """Understanding query projection (``self_attn.q_proj``)."""
    return self.self_attn.q_proj

  @property
  def to_k(self) -> nnx.Linear:
    """Understanding key projection (``self_attn.k_proj``)."""
    return self.self_attn.k_proj

  @property
  def to_v(self) -> nnx.Linear:
    """Understanding value projection (``self_attn.v_proj``)."""
    return self.self_attn.v_proj

  @property
  def to_out(self) -> nnx.Linear:
    """Understanding output projection (``self_attn.o_proj``)."""
    return self.self_attn.o_proj

  @property
  def norm_q(self) -> nnx.RMSNorm | None:
    """Understanding query normalization (``self_attn.q_norm``)."""
    return self.self_attn.q_norm

  @property
  def norm_k(self) -> nnx.RMSNorm | None:
    """Understanding key normalization (``self_attn.k_norm``)."""
    return self.self_attn.k_norm

  @property
  def add_q_proj(self) -> nnx.Linear:
    """Generation query projection (``self_attn.q_proj_gen``)."""
    return self.self_attn.q_proj_gen

  @property
  def add_k_proj(self) -> nnx.Linear:
    """Generation key projection (``self_attn.k_proj_gen``)."""
    return self.self_attn.k_proj_gen

  @property
  def add_v_proj(self) -> nnx.Linear:
    """Generation value projection (``self_attn.v_proj_gen``)."""
    return self.self_attn.v_proj_gen

  @property
  def to_add_out(self) -> nnx.Linear:
    """Generation output projection (``self_attn.o_proj_gen``)."""
    return self.self_attn.o_proj_gen

  @property
  def norm_added_q(self) -> nnx.RMSNorm | None:
    """Generation query normalization (``self_attn.q_norm_gen``)."""
    return self.self_attn.q_norm_gen

  @property
  def norm_added_k(self) -> nnx.RMSNorm | None:
    """Generation key normalization (``self_attn.k_norm_gen``)."""
    return self.self_attn.k_norm_gen

  @property
  def k_norm_und_for_gen(self) -> nnx.RMSNorm | None:
    """Understanding key normalization used by generation queries."""
    return self.self_attn.k_norm_und_for_gen

  def __call__(
      self,
      und_seq: jax.Array,
      gen_seq: jax.Array,
      metadata: WeaverPackingMetadata,
      cos: jax.Array | None = None,
      sin: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Runs one dual-pathway Mixture-of-Transformers decoder layer step.

    Input shapes and layouts:
      und_seq: Understanding token sequence of shape ``[N_und, hidden_size]``.
      gen_seq: Generation token sequence of shape ``[N_gen, hidden_size]``.
      metadata: ``WeaverPackingMetadata`` containing cumulative sample
        boundaries (``causal_q_offsets``: ``[B + 1]``, ``full_q_offsets``:
        ``[B + 1]``, ``sample_kv_offsets``: ``[B + 1]``) and 1D index arrays
        (``packed_und_token_indexes``: ``[N_und]``,
        ``packed_gen_token_indexes``: ``[N_gen]``).
      cos: Optional 3D M-RoPE cosine table of shape ``[N_total, head_dim]`` in
        global packed order (``N_total = N_und + N_gen``).
      sin: Optional 3D M-RoPE sine table of shape ``[N_total, head_dim]`` in
        global packed order (``N_total = N_und + N_gen``).

    Output shapes and layouts:
      Tuple ``(und_out, gen_out)`` where:
        * und_out: Transformed understanding tokens of shape
          ``[N_und, hidden_size]``.
        * gen_out: Transformed generation tokens of shape
          ``[N_gen, hidden_size]``.
    """
    hook = self.sharding_hook

    # 1. Pre-attention normalization (independent per tower).
    und_norm = hook(self.input_layernorm(und_seq), "attn_input")
    gen_norm = hook(self.input_layernorm_moe_gen(gen_seq), "attn_input")

    # 2. Fused dual attention.
    und_attn_out, gen_attn_out = self.self_attn(
        (und_norm, gen_norm),
        metadata,
        cos,
        sin,
    )

    # 3. Attention residual.
    residual_und = hook(und_seq + und_attn_out, "post_attn")
    residual_gen = hook(gen_seq + gen_attn_out, "gen_post_attn")

    # 4. Pre-MLP normalization, feed-forward, and MLP residual.
    mlp_in_und = hook(self.post_attention_layernorm(residual_und), "mlp_input")
    mlp_in_gen = hook(self.post_attention_layernorm_moe_gen(residual_gen), "mlp_input")
    mlp_out_und = self.mlp(mlp_in_und)
    mlp_out_gen = self.mlp_moe_gen(mlp_in_gen)

    return hook(residual_und + mlp_out_und, "post_mlp"), hook(residual_gen + mlp_out_gen, "gen_post_mlp")


# ==============================================================================
# 8. Weaver Omni-Transformer Full Backbone Architecture (Pure Flax NNX)
# ==============================================================================


def patchify_latents(latents: jax.Array, patch_size: int = 2) -> jax.Array:
  """Patchifies 5D video/image latents ``[B, C, T, H, W]`` into ``[B, S_gen, p*p*C]``.

  Matches the spatial 2D patchification layout ``b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)``.

  Args:
    latents: Latent tensor of shape ``[B, C, T, H, W]``.
    patch_size: Spatial patch size ``p`` along height and width.

  Returns:
    Patchified token sequence of shape ``[B, T * (H // p) * (W // p), p * p * C]``.
  """
  if latents.ndim != 5:
    raise ValueError(f"latents must be a 5D array [B, C, T, H, W], got shape {latents.shape}.")
  bsz, c_lat, t_lat, h_lat, w_lat = latents.shape
  if h_lat % patch_size != 0 or w_lat % patch_size != 0:
    raise ValueError(f"Latent spatial dimensions ({h_lat}, {w_lat}) must be divisible by patch_size ({patch_size}).")
  h_patch = h_lat // patch_size
  w_patch = w_lat // patch_size
  s_gen = t_lat * h_patch * w_patch
  patch_dim = patch_size * patch_size * c_lat

  reshaped = latents.reshape(bsz, c_lat, t_lat, h_patch, patch_size, w_patch, patch_size)
  permuted = jnp.einsum("bcthpwq->bthwpqc", reshaped)
  return permuted.reshape(bsz, s_gen, patch_dim)


def unpatchify_latents(
    patches: jax.Array,
    t_lat: int,
    h_lat: int,
    w_lat: int,
    *,
    patch_size: int = 2,
    latent_channels: int = 48,
) -> jax.Array:
  """Unpatchifies generation tokens ``[B, S_gen, p*p*C]`` or ``[B*S_gen, p*p*C]`` to ``[B, C, T, H, W]``.

  Matches the inverse spatial layout ``(b t h w) (p1 p2 c) -> b c t (h p1) (w p2)``.

  Args:
    patches: Patch token tensor of shape ``[B, S_gen, p*p*C]`` or ``[B * S_gen, p*p*C]``.
    t_lat: Number of latent temporal frames ``T``.
    h_lat: Latent spatial height ``H``.
    w_lat: Latent spatial width ``W``.
    patch_size: Spatial patch size ``p`` along height and width.
    latent_channels: Number of latent channels ``C``.

  Returns:
    Unpatchified latent tensor of shape ``[B, C, T, H, W]``.
  """
  if h_lat % patch_size != 0 or w_lat % patch_size != 0:
    raise ValueError(f"Latent spatial dimensions ({h_lat}, {w_lat}) must be divisible by patch_size ({patch_size}).")
  h_patch = h_lat // patch_size
  w_patch = w_lat // patch_size
  s_gen = t_lat * h_patch * w_patch
  patch_dim = patch_size * patch_size * latent_channels

  if patches.shape[-1] != patch_dim:
    raise ValueError(f"Expected patch feature dim {patch_dim}, got {patches.shape[-1]}.")

  if patches.ndim == 2:
    if patches.shape[0] % s_gen != 0:
      raise ValueError(f"Total patch tokens ({patches.shape[0]}) must be divisible by s_gen ({s_gen}).")
    bsz = patches.shape[0] // s_gen
  elif patches.ndim == 3:
    bsz = patches.shape[0]
    if patches.shape[1] != s_gen:
      raise ValueError(f"Expected s_gen={s_gen} along axis 1, got {patches.shape[1]}.")
  else:
    raise ValueError(f"patches must be 2D or 3D, got shape {patches.shape}.")

  reshaped = patches.reshape(bsz, t_lat, h_patch, w_patch, patch_size, patch_size, latent_channels)
  permuted = jnp.einsum("bthwpqc->bcthpwq", reshaped)
  return permuted.reshape(bsz, latent_channels, t_lat, h_lat, w_lat)


def get_weaver_timestep_embedding(
    timesteps: jax.Array,
    num_channels: int = 256,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> jax.Array:
  """Computes sinusoidal timestep embeddings in ``float32``.

  Args:
    timesteps: Timestep tensor of shape ``[N]`` or ``[B, T]``.
    num_channels: Number of sinusoidal embedding channels (default 256).
    flip_sin_to_cos: Whether to order ``[cos, sin]`` instead of ``[sin, cos]``.
    downscale_freq_shift: Frequency shift denominator offset.
    scale: Multiplier applied to sinusoidal arguments.
    max_period: Controls minimum frequency of the embeddings.

  Returns:
    Float32 sinusoidal embeddings of shape ``(*timesteps.shape, num_channels)``.
  """
  half_dim = num_channels // 2
  exponent = -math.log(float(max_period)) * jnp.arange(0, half_dim, dtype=jnp.float32)
  exponent = exponent / (half_dim - downscale_freq_shift)

  emb = jnp.exp(exponent)
  emb = timesteps[..., None].astype(jnp.float32) * emb

  emb = scale * emb
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

  if flip_sin_to_cos:
    emb = jnp.concatenate([emb[..., half_dim:], emb[..., :half_dim]], axis=-1)

  if num_channels % 2 == 1:
    emb = jnp.pad(emb, ((0, 0),) * (emb.ndim - 1) + ((0, 1),))

  return emb


class WeaverTimeEmbedder(nnx.Module):
  """Sinusoidal + 2-layer SiLU MLP diffusion timestep embedder using pure ``nnx.Linear``."""

  def __init__(
      self,
      in_channels: int = 256,
      time_embed_dim: int = 4096,
      *,
      timestep_scale: float = 0.001,
      use_bias: bool = True,
      in_kernel_axes: tuple[None | str, ...] = ("time_embed", "embed"),
      out_kernel_axes: tuple[None | str, ...] = ("embed", "embed"),
      sharding_hook: Callable[[Any, str], Any] | None = None,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes the diffusion timestep embedder."""
    self.in_channels = in_channels
    self.time_embed_dim = time_embed_dim
    self.timestep_scale = timestep_scale
    self.use_bias = use_bias
    self.dtype = _resolve_jax_dtype(dtype)
    self.weight_dtype = _resolve_jax_dtype(weight_dtype)

    self.mlp_0 = _make_linear(
        in_channels,
        time_embed_dim,
        use_bias=use_bias,
        dtype=jnp.float32,
        weight_dtype=self.weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=in_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="time_mlp_0_kernel_init",
        rngs=rngs,
    )
    self.mlp_2 = _make_linear(
        time_embed_dim,
        time_embed_dim,
        use_bias=use_bias,
        dtype=jnp.float32,
        weight_dtype=self.weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=out_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="time_mlp_2_kernel_init",
        rngs=rngs,
    )

  @property
  def linear_1(self) -> nnx.Linear:
    """First linear projection in the timestep MLP (``mlp_0``)."""
    return self.mlp_0

  @property
  def linear_2(self) -> nnx.Linear:
    """Second linear projection in the timestep MLP (``mlp_2``)."""
    return self.mlp_2

  def __call__(self, timesteps: jax.Array) -> jax.Array:
    """Embeds scalar/per-frame diffusion timesteps into ``time_embed_dim``."""
    scaled_t = timesteps.astype(jnp.float32) * self.timestep_scale
    t_emb = get_weaver_timestep_embedding(
        scaled_t,
        num_channels=self.in_channels,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
    )
    hidden = jax.nn.silu(self.mlp_0(t_emb))
    return self.mlp_2(hidden).astype(self.dtype)


def build_weaver_3d_position_ids(
    batch_size: int,
    s_und: int,
    t_lat: int,
    h_patch: int,
    w_patch: int,
) -> jax.Array:
  """Constructs packed 3D M-RoPE coordinates ``[N_total, 3]`` for ``[text, vision]`` samples.

  For each sample:
  * Understanding (text) tokens have 1D positions ``(i, i, i)`` for ``i in 0..s_und-1``.
  * Generation (vision) tokens have 3D positions ``(s_und + t, h, w)`` across the
    ``(t_lat, h_patch, w_patch)`` patch grid.

  Args:
    batch_size: Number of samples ``B``.
    s_und: Number of understanding (text) tokens per sample.
    t_lat: Number of temporal latent frames ``T``.
    h_patch: Number of spatial height patches ``H // p``.
    w_patch: Number of spatial width patches ``W // p``.

  Returns:
    Integer coordinate array of shape ``[B * (s_und + t_lat * h_patch * w_patch), 3]``.
  """
  pos_und = jnp.broadcast_to(jnp.arange(s_und, dtype=jnp.int32)[:, None], (s_und, 3))

  t_grid = jnp.repeat(jnp.arange(t_lat, dtype=jnp.int32) + s_und, h_patch * w_patch)
  h_grid = jnp.tile(jnp.repeat(jnp.arange(h_patch, dtype=jnp.int32), w_patch), t_lat)
  w_grid = jnp.tile(jnp.arange(w_patch, dtype=jnp.int32), t_lat * h_patch)
  pos_gen = jnp.stack([t_grid, h_grid, w_grid], axis=-1)

  sample_pos = jnp.concatenate([pos_und, pos_gen], axis=0)
  return jnp.tile(sample_pos, (batch_size, 1))


class WeaverOmniTransformer(nnx.Module):
  """Complete Weaver joint Mixture-of-Transformers diffusion backbone.

  Integrates:
  * ``embed_tokens``: Text token embedding table (``nnx.Embed``).
  * ``proj_in`` (alias ``vae2llm``): Patchified latent input projection (``nnx.Linear``).
  * ``time_embedder``: Sinusoidal + 2-layer SiLU MLP timestep embedder (``WeaverTimeEmbedder``).
  * ``layers`` / ``scanned_layers``: Stack of ``WeaverMoTDecoderLayer`` blocks supporting
    both ``scan_layers=False`` and ``scan_layers=True``.
  * ``norm``: Final understanding pathway RMSNorm (``nnx.RMSNorm``).
  * ``norm_moe_gen``: Final generation pathway RMSNorm (``nnx.RMSNorm``).
  * ``proj_out`` (alias ``llm2vae``): Output projection back to latent patch space (``nnx.Linear``).
  """

  def __init__(
      self,
      config: WeaverConfig | Any | int = 4096,
      *,
      head_dim: int | None = None,
      num_attention_heads: int | None = None,
      num_key_value_heads: int | None = None,
      intermediate_size: int | None = None,
      num_hidden_layers: int | None = None,
      vocab_size: int | None = None,
      latent_channels: int | None = None,
      patch_size: int | None = None,
      time_embed_in_channels: int | None = None,
      timestep_scale: float | None = None,
      rms_norm_eps: float | None = None,
      hidden_act: str | None = None,
      attention_bias: bool | None = None,
      qk_norm_for_text: bool | None = None,
      qk_norm_for_diffusion: bool | None = None,
      use_und_k_norm_for_gen: bool | None = None,
      attention_kernel: str | None = None,
      use_ragged_ops: bool | None = None,
      mrope_section: tuple[int, int, int] | None = None,
      rope_theta: float | None = None,
      scan_layers: bool | None = None,
      param_scan_axis: int | None = None,
      remat_policy: str | None = None,
      dtype: Any | None = None,
      weight_dtype: Any | None = None,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      mesh: sharding.Mesh | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
      rngs: nnx.Rngs,
  ):
    """Initializes ``WeaverOmniTransformer`` from ``WeaverConfig``, MaxText ``Config``, or kwargs."""
    del mesh
    if isinstance(config, WeaverConfig):
      cfg = dataclasses.replace(config)
    elif isinstance(config, int):
      cfg = WeaverConfig(hidden_size=config)
    else:
      cfg = WeaverConfig.from_maxtext_config(config)

    overrides: dict[str, Any] = {}
    for field_name, val in (
        ("head_dim", head_dim),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("intermediate_size", intermediate_size),
        ("num_hidden_layers", num_hidden_layers),
        ("vocab_size", vocab_size),
        ("latent_channels", latent_channels),
        ("patch_size", patch_size),
        ("time_embed_in_channels", time_embed_in_channels),
        ("timestep_scale", timestep_scale),
        ("rms_norm_eps", rms_norm_eps),
        ("hidden_act", hidden_act),
        ("attention_bias", attention_bias),
        ("qk_norm_for_text", qk_norm_for_text),
        ("qk_norm_for_diffusion", qk_norm_for_diffusion),
        ("use_und_k_norm_for_gen", use_und_k_norm_for_gen),
        ("attention_kernel", attention_kernel),
        ("use_ragged_ops", use_ragged_ops),
        ("mrope_section", mrope_section),
        ("rope_theta", rope_theta),
        ("scan_layers", scan_layers),
        ("param_scan_axis", param_scan_axis),
        ("remat_policy", remat_policy),
        ("dtype", dtype),
        ("weight_dtype", weight_dtype),
    ):
      if val is not None:
        overrides[field_name] = val
    if overrides:
      cfg = dataclasses.replace(cfg, **overrides)

    self.config = cfg
    self.hidden_size = cfg.hidden_size
    self.head_dim = cfg.head_dim
    self.num_attention_heads = cfg.num_attention_heads
    self.num_key_value_heads = cfg.num_key_value_heads
    self.intermediate_size = cfg.intermediate_size
    self.num_hidden_layers = cfg.num_hidden_layers
    self.vocab_size = cfg.vocab_size
    self.latent_channels = cfg.latent_channels
    self.patch_size = cfg.patch_size
    self.patch_dim = cfg.patch_size * cfg.patch_size * cfg.latent_channels
    self.time_embed_in_channels = cfg.time_embed_in_channels
    self.timestep_scale = cfg.timestep_scale
    self.rms_norm_eps = cfg.rms_norm_eps
    self.hidden_act = cfg.hidden_act
    self.mrope_section = cfg.mrope_section
    self.rope_theta = cfg.rope_theta
    self.scan_layers = cfg.scan_layers
    self.param_scan_axis = cfg.param_scan_axis
    self.remat_policy = cfg.remat_policy
    self.use_ragged_ops = cfg.use_ragged_ops
    self.dtype = cfg.dtype
    self.weight_dtype = cfg.weight_dtype
    self.sharding_hook = sharding_hook or (lambda x, name: x)

    # 1. Text token embedding table.
    self.embed_tokens = _make_embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.hidden_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.embed_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="embed_tokens_kernel_init",
        rngs=rngs,
    )

    # 2. Latent patch input projection (vae2llm).
    self.proj_in = _make_linear(
        self.patch_dim,
        cfg.hidden_size,
        use_bias=True,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=cfg.proj_in_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="proj_in_kernel_init",
        rngs=rngs,
    )

    # 3. Diffusion timestep embedder.
    self.time_embedder = WeaverTimeEmbedder(
        in_channels=cfg.time_embed_in_channels,
        time_embed_dim=cfg.hidden_size,
        timestep_scale=cfg.timestep_scale,
        in_kernel_axes=cfg.time_mlp_in_kernel_axes,
        out_kernel_axes=cfg.time_mlp_out_kernel_axes,
        sharding_hook=sharding_hook,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_init=kernel_init,
        rngs=rngs,
    )

    # 4. Stacked Mixture-of-Transformers decoder blocks.
    if cfg.scan_layers:
      self.layers = None
      self.scanned_layers = nnx_scan.create_scanned_layers(
          lambda layer_rngs: WeaverMoTDecoderLayer(
              cfg,
              sharding_hook=sharding_hook,
              kernel_init=kernel_init,
              rngs=layer_rngs,
          ),
          length=cfg.num_hidden_layers,
          param_scan_axis=cfg.param_scan_axis,
          metadata_axis_name="layers",
          rngs=rngs,
      )
    else:
      self.layers = nnx.List(
          [
              WeaverMoTDecoderLayer(
                  cfg,
                  sharding_hook=sharding_hook,
                  kernel_init=kernel_init,
                  rngs=rngs,
              )
              for _ in range(cfg.num_hidden_layers)
          ]
      )
      self.scanned_layers = None

    # 5. Final pathway RMSNorms.
    self.norm = _make_rmsnorm(
        cfg.hidden_size,
        epsilon=cfg.rms_norm_eps,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.norm_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="norm_scale_init",
        rngs=rngs,
    )
    self.norm_moe_gen = _make_rmsnorm(
        cfg.hidden_size,
        epsilon=cfg.rms_norm_eps,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=cfg.norm_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="norm_moe_gen_scale_init",
        rngs=rngs,
    )

    # 6. Output projection to latent patch space (llm2vae).
    self.proj_out = _make_linear(
        cfg.hidden_size,
        self.patch_dim,
        use_bias=True,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_init=kernel_init,
        kernel_axes=cfg.proj_out_kernel_axes,
        sharding_hook=sharding_hook,
        hook_name="proj_out_kernel_init",
        rngs=rngs,
    )

  @classmethod
  def from_config(
      cls,
      config: WeaverConfig | Any,
      *,
      rngs: nnx.Rngs,
      mesh: sharding.Mesh | None = None,
      sharding_hook: Callable[[Any, str], Any] | None = None,
      kernel_init: Callable[..., jax.Array] | None = None,
      **overrides: Any,
  ) -> WeaverOmniTransformer:
    """Constructs ``WeaverOmniTransformer`` from a ``WeaverConfig`` or MaxText ``Config``."""
    return cls(
        config,
        mesh=mesh,
        sharding_hook=sharding_hook,
        kernel_init=kernel_init,
        rngs=rngs,
        **overrides,
    )

  @property
  def vae2llm(self) -> nnx.Linear:
    """Canonical alias for the latent patch input projection (``proj_in``)."""
    return self.proj_in

  @property
  def llm2vae(self) -> nnx.Linear:
    """Canonical alias for the latent patch output projection (``proj_out``)."""
    return self.proj_out

  @overload
  def __call__(
      self,
      input_ids: jax.Array,
      latents: jax.Array,
      timesteps: jax.Array,
      *,
      position_ids: jax.Array | None = None,
      metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: Literal[False] = False,
  ) -> jax.Array:
    ...

  @overload
  def __call__(
      self,
      input_ids: jax.Array,
      latents: jax.Array,
      timesteps: jax.Array,
      *,
      position_ids: jax.Array | None = None,
      metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: Literal[True],
  ) -> tuple[jax.Array, jax.Array]:
    ...

  def __call__(
      self,
      input_ids: jax.Array,
      latents: jax.Array,
      timesteps: jax.Array,
      *,
      position_ids: jax.Array | None = None,
      metadata: WeaverPackingMetadata | None = None,
      return_hidden_states: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Executes the full Weaver Omni-Transformer forward pass.

    Args:
      input_ids: Understanding (text) token IDs of shape ``[B, S_und]``.
      latents: Noisy video/image latent tensor of shape ``[B, C, T, H, W]``.
      timesteps: Diffusion timesteps of shape ``[B]``, ``[B, 1]``, or ``[B, T]``.
      position_ids: Optional 3D M-RoPE position IDs of shape ``[N_total, 3]``
        or ``[3, N_total]``. Computed automatically from ``(B, S_und, T, H//p, W//p)``
        when ``None``.
      metadata: Optional pre-built ``WeaverPackingMetadata``. Constructed
        automatically for ``[S_und] * B`` and ``[S_gen] * B`` when ``None``.
      return_hidden_states: When ``True``, returns ``(preds_vision, last_hidden_state)``
        where ``last_hidden_state`` has shape ``[N_total, hidden_size]``.

    Returns:
      Predicted velocity/noise latent tensor of shape ``[B, C, T, H, W]`` (or a
      tuple ``(preds_vision, last_hidden_state)`` when ``return_hidden_states=True``).
    """
    if input_ids.ndim != 2:
      raise ValueError(f"input_ids must be 2D [B, S_und], got shape {input_ids.shape}.")
    if latents.ndim != 5:
      raise ValueError(f"latents must be 5D [B, C, T, H, W], got shape {latents.shape}.")

    bsz, s_und = input_ids.shape
    bsz_lat, c_lat, t_lat, h_lat, w_lat = latents.shape
    if bsz != bsz_lat:
      raise ValueError(f"Batch size mismatch between input_ids ({bsz}) and latents ({bsz_lat}).")
    if c_lat != self.latent_channels:
      raise ValueError(f"Expected latent_channels={self.latent_channels}, got {c_lat}.")

    p = self.patch_size
    h_patch = h_lat // p
    w_patch = w_lat // p
    s_gen = t_lat * h_patch * w_patch

    # 1. Embed understanding (text) tokens -> [B * S_und, hidden_size].
    text_embeds = self.embed_tokens(input_ids).astype(self.dtype)
    und_seq = text_embeds.reshape(bsz * s_und, self.hidden_size)

    # 2. Patchify latents, project to hidden_size, and add timestep embeddings -> [B * S_gen, hidden_size].
    patches = patchify_latents(latents.astype(self.dtype), patch_size=p)
    vision_embeds = self.proj_in(patches)

    time_embeds = self.time_embedder(timesteps).astype(self.dtype)
    if time_embeds.ndim == 2:
      time_embeds = time_embeds[:, None, :]
    elif time_embeds.ndim == 3 and time_embeds.shape[1] == t_lat and s_gen != t_lat:
      time_embeds = jnp.repeat(time_embeds, h_patch * w_patch, axis=1)

    gen_tokens = vision_embeds + time_embeds
    gen_seq = gen_tokens.reshape(bsz * s_gen, self.hidden_size)

    # 3. Build packing metadata and 3D M-RoPE rotary tables.
    if metadata is None:
      metadata = build_weaver_packing_metadata([s_und] * bsz, [s_gen] * bsz)

    if position_ids is None:
      pos_3d = build_weaver_3d_position_ids(
          batch_size=bsz,
          s_und=s_und,
          t_lat=t_lat,
          h_patch=h_patch,
          w_patch=w_patch,
      )
    else:
      if position_ids.ndim == 2 and position_ids.shape[0] == 3 and position_ids.shape[1] == metadata.total_tokens:
        pos_3d = position_ids.T
      else:
        pos_3d = position_ids

    cos, sin = compute_3d_mrope_cos_sin(
        pos_3d,
        head_dim=self.head_dim,
        mrope_section=self.mrope_section,
        rope_theta=self.rope_theta,
    )
    cos = cos.astype(self.dtype)
    sin = sin.astype(self.dtype)

    # 4. Apply stacked Mixture-of-Transformers decoder blocks.
    if self.scan_layers:
      assert self.scanned_layers is not None
      remat = self.remat_policy not in ("none", "")
      und_seq, gen_seq = nnx_scan.apply_scanned_layers(
          self.scanned_layers,
          (und_seq, gen_seq),
          length=self.num_hidden_layers,
          param_scan_axis=self.param_scan_axis,
          apply_fn=lambda layer, carry: layer(carry[0], carry[1], metadata, cos, sin),
          remat=remat,
      )
    else:
      assert self.layers is not None
      for layer in self.layers:
        und_seq, gen_seq = layer(und_seq, gen_seq, metadata, cos, sin)

    # 5. Final pathway RMSNorms.
    und_seq = self.norm(und_seq)
    gen_seq = self.norm_moe_gen(gen_seq)

    # 6. Project generation tokens to patch space and unpatchify to [B, C, T, H, W].
    gen_patches = self.proj_out(gen_seq)
    preds_vision = unpatchify_latents(
        gen_patches,
        t_lat=t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        patch_size=p,
        latent_channels=c_lat,
    )

    if return_hidden_states:
      last_hidden_state = reinterleave_streams(
          und_tokens=und_seq,
          gen_tokens=gen_seq,
          packed_und_token_indexes=metadata.packed_und_token_indexes,
          packed_gen_token_indexes=metadata.packed_gen_token_indexes,
          total_tokens=metadata.total_tokens,
          use_ragged_ops=self.use_ragged_ops,
      )
      return preds_vision, last_hidden_state

    return preds_vision
