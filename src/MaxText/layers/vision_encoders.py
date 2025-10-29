import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx

from MaxText.common_types import Array, Config, DType, AttentionType
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.initializers import nd_dense_init, NdInitializer
from MaxText.layers.embeddings import Qwen3OmniMoeVisionPosEmbedInterpolate
from MaxText.layers.attentions import Attention
from MaxText.layers.packing_utils import compute_tokens_per_video, generate_segment_ids_from_counts


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionPatchMerger(nnx.Module):
    config: Config
    hidden_size: int
    use_postshuffle_norm: bool
    dtype: DType
    weight_dtype: DType
    kernel_init: NdInitializer
    rngs: nnx.Rngs

    ln_q: nnx.LayerNorm
    mlp_0: DenseGeneral
    mlp_2: DenseGeneral

    def __init__(
        self,
        config: Config,
        use_postshuffle_norm: bool = False,
        dtype: DType = jnp.float32,
        weight_dtype: DType = jnp.float32,
        kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.use_postshuffle_norm = use_postshuffle_norm
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.kernel_init = kernel_init
        self.rngs = rngs

        # Calculate hidden_size after spatial merge
        spatial_merge_size = config.spatial_merge_size_for_vit
        base_hidden_size = config.hidden_size_for_vit
        out_hidden_size = config.out_hidden_size_for_vit

        self.hidden_size = base_hidden_size * (spatial_merge_size**2)

        # LayerNorm before MLP
        ln_features = self.hidden_size if use_postshuffle_norm else base_hidden_size
        self.ln_q = nnx.LayerNorm(
            num_features=ln_features,
            epsilon=1e-6,
            dtype=dtype,
            rngs=rngs,
        )

        # MLP layers: Linear -> GELU -> Linear
        self.mlp_0 = DenseGeneral(
            in_features_shape=self.hidden_size,
            out_features_shape=self.hidden_size,
            use_bias=True,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.mlp_2 = DenseGeneral(
            in_features_shape=self.hidden_size,
            out_features_shape=out_hidden_size,
            use_bias=True,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, hidden: Array) -> Array:
        """
        Args:
            hidden: Input tensor of shape (seq_len, hidden_size) - packed sequences

        Returns:
            Output tensor of shape (seq_len, out_hidden_size)
        """
        # Apply layer norm
        if self.use_postshuffle_norm:
            hidden = self.ln_q(hidden.reshape(-1, self.hidden_size))
        else:
            hidden = self.ln_q(hidden)

        # Ensure correct shape for MLP
        hidden = hidden.reshape(-1, self.hidden_size)

        # MLP: Linear -> GELU -> Linear
        hidden = self.mlp_0(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = self.mlp_2(hidden)

        return hidden


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionMLP(nnx.Module):
    config: Config
    hidden_size: int
    intermediate_size: int
    dtype: DType
    weight_dtype: DType
    kernel_init: NdInitializer
    rngs: nnx.Rngs

    linear_fc1: DenseGeneral
    linear_fc2: DenseGeneral

    def __init__(
        self,
        config: Config,
        dtype: DType = jnp.float32,
        weight_dtype: DType = jnp.float32,
        kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.kernel_init = kernel_init
        self.rngs = rngs

        self.hidden_size = config.hidden_size_for_vit
        self.intermediate_size = config.intermediate_size_for_vit

        self.linear_fc1 = DenseGeneral(
            in_features_shape=self.hidden_size,
            out_features_shape=self.intermediate_size,
            use_bias=True,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.linear_fc2 = DenseGeneral(
            in_features_shape=self.intermediate_size,
            out_features_shape=self.hidden_size,
            use_bias=True,
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, hidden_state: Array) -> Array:
        """
        Args:
            hidden_state: Input tensor of shape (..., hidden_size) - supports packed sequences

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = jax.nn.gelu(hidden_state)
        hidden_state = self.linear_fc2(hidden_state)
        return hidden_state


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionPatchEmbed(nnx.Module):
    config: Config
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    embed_dim: int
    dtype: DType
    weight_dtype: DType
    rngs: nnx.Rngs

    proj: nnx.Conv

    def __init__(
        self,
        config: Config,
        dtype: DType = jnp.float32,
        weight_dtype: DType = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.rngs = rngs

        self.patch_size = config.patch_size_for_vit
        self.temporal_patch_size = config.temporal_patch_size_for_vit
        self.in_channels = config.num_channels_for_vit
        self.embed_dim = config.hidden_size_for_vit

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)

        self.proj = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.embed_dim,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=weight_dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        """
        Args:
            hidden_states: Flattened input tensor that will be reshaped to
                          (batch_size, temporal_patch_size, patch_size, patch_size, in_channels)

        Returns:
            Output tensor of shape (seq_len, embed_dim) - flattened packed sequences
        """
        # Get target dtype from projection weights
        target_dtype = self.proj.kernel.value.dtype

        # Compute batch size from total elements
        batch_size = hidden_states.shape[0] // (
            self.temporal_patch_size
            * self.patch_size
            * self.patch_size
            * self.in_channels
        )

        # Reshape input: (batch, in_channels, temporal, height, width)
        hidden_states = hidden_states.reshape(
            batch_size,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        # Transpose to JAX conv format: (batch, temporal, height, width, in_channels)
        hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = hidden_states.astype(target_dtype)

        # Apply 3D conv
        hidden_states = self.proj(hidden_states)

        # Flatten to packed sequences: (seq_len, embed_dim)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)

        return hidden_states


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionAttention(nnx.Module):
    config: Config
    attn: Attention

    def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
        self.config = config
        head_dim = self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
        # Vision uses full SA, no kv cache
        self.attn = Attention(
            config=self.config,
            num_query_heads=self.config.num_attention_heads_for_vit,
            num_kv_heads=self.config.num_attention_heads_for_vit,
            head_dim=head_dim,
            max_target_length=self.config.num_position_embeddings_for_vit,
            attention_kernel="dot_product",
            inputs_q_shape=(1, 1, self.config.hidden_size_for_vit),
            inputs_kv_shape=(1, 1, self.config.hidden_size_for_vit),
            float32_qk_product=self.config.float32_qk_product,
            float32_logits=self.config.float32_logits,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            mesh=mesh,
            dropout_rate=0.0,
            attention_type=AttentionType.FULL,
            is_nope_layer=False,
            use_bias_in_projections=True,
            is_vision=True,
            use_qk_norm=False,
            query_pre_attn_scalar=1.0 / jnp.sqrt(head_dim),
            model_mode="train",
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        grid_thw: Optional[Array] = None,
        decoder_segment_ids: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Array:
        """
        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_size) - packed sequences
            grid_thw: Grid specification for rotary embeddings, shape (num_images, 3)
            decoder_segment_ids: Segment IDs for packed sequences, shape (1, seq_len) or (seq_len,)
            deterministic: Whether to use deterministic mode (disable dropout)

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        # Attention layer expects (batch, seq_len, hidden_size)
        # We use batch=1 with packed sequences in the sequence dimension
        hidden_states_batched = hidden_states[jnp.newaxis, :, :]

        # Ensure segment IDs have batch dimension for attention layer
        if decoder_segment_ids is not None and decoder_segment_ids.ndim == 1:
            decoder_segment_ids = decoder_segment_ids[jnp.newaxis, :]

        # Pass through attention
        output = self.attn(
            inputs_q=hidden_states_batched,
            inputs_kv=hidden_states_batched,
            grid_thw=grid_thw,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
        )

        # Remove batch dimension: (1, seq_len, hidden_size) -> (seq_len, hidden_size)
        return output[0]


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionBlock(nnx.Module):
    config: Config
    ln1: nnx.LayerNorm
    ln2: nnx.LayerNorm
    attn: Qwen3OmniMoeVisionAttention
    mlp: DenseGeneral
    mlp_out: DenseGeneral

    def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
        self.config = config
        hs = self.config.hidden_size_for_vit
        self.ln1 = nnx.LayerNorm(num_features=hs, epsilon=1e-6, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=hs, epsilon=1e-6, rngs=rngs)
        self.attn = Qwen3OmniMoeVisionAttention(config=config, mesh=mesh, rngs=rngs)
        interm = self.config.intermediate_size_for_vit
        self.mlp = DenseGeneral(
            in_features_shape=hs, out_features_shape=interm, use_bias=True, rngs=rngs
        )
        self.mlp_out = DenseGeneral(
            in_features_shape=interm, out_features_shape=hs, use_bias=True, rngs=rngs
        )

    def __call__(
        self,
        x: Array,
        grid_thw: Optional[Array] = None,
        decoder_segment_ids: Optional[Array] = None,
    ) -> Array:
        """
        Args:
            x: Input tensor of shape (seq_len, hidden_size) - packed sequences
            grid_thw: Grid specification for rotary embeddings
            decoder_segment_ids: Segment IDs for packed sequences

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        x = x + self.attn(
            self.ln1(x), grid_thw=grid_thw, decoder_segment_ids=decoder_segment_ids
        )
        y = self.ln2(x)
        y = self.mlp(y)
        y = jax.nn.gelu(y)
        y = self.mlp_out(y)
        return x + y


@dataclasses.dataclass(repr=False)
class Qwen3OmniMoeVisionEncoder(nnx.Module):
    config: Config
    # modules
    patch_embed: Qwen3OmniMoeVisionPatchEmbed
    pos_embed_interpolate: Qwen3OmniMoeVisionPosEmbedInterpolate
    blocks: nnx.List
    # optional deep taps
    merger_list: nnx.List
    final_merger: "Qwen3OmniMoeVisionPatchMerger"

    # constants
    spatial_merge_size: int
    deep_idx: Tuple[int, ...]

    def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
        self.config = config
        self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(config=config, rngs=rngs)

        num_pos = config.num_position_embeddings_for_vit
        hs = config.hidden_size_for_vit
        self.spatial_merge_size = config.spatial_merge_size_for_vit

        # Initialize positional embedding interpolation module
        self.pos_embed_interpolate = Qwen3OmniMoeVisionPosEmbedInterpolate(
            num_position_embeddings=num_pos,
            hidden_size=hs,
            spatial_merge_size=self.spatial_merge_size,
            rngs=rngs,
        )

        depth = config.num_hidden_layers_for_vit

        self.blocks = nnx.List(
            [Qwen3OmniMoeVisionBlock(config=config, mesh=mesh, rngs=rngs) for _ in range(depth)]
        )

        self.deep_idx = tuple(config.deepstack_visual_indexes_for_vit)
        self.merger_list = nnx.List(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    config=config, use_postshuffle_norm=True, rngs=rngs
                )
                for _ in self.deep_idx
            ]
        )

        self.final_merger = Qwen3OmniMoeVisionPatchMerger(
            config=config, use_postshuffle_norm=False, rngs=rngs
        )

    def __call__(
        self, hidden_states: Array, grid_thw: Array, deterministic: bool = True
    ):
        """
        Args:
            hidden_states: Flattened visual tokens BEFORE embedding - packed sequences
            grid_thw: [N,3] with (T,H,W) per sample
            deterministic: Whether to use deterministic mode

        Returns:
            Tuple of:
            - final_output: shape (seq_len, out_hidden_size) - packed sequences
            - deep_features: List of intermediate features, each of shape (seq_len, out_hidden_size)
        """
        # Patch embedding: flat -> (seq_len, hidden_size)
        x = self.patch_embed(hidden_states)

        # Add positional embeddings: (seq_len, hidden_size)
        pos = self.pos_embed_interpolate(grid_thw)
        x = x + pos

        # Generate segment IDs for packed vision sequences
        # Compute tokens per video/image BEFORE spatial merging
        # (spatial merging happens after attention blocks via PatchMerger)
        # tokens = temporal * height * width
        tokens_per_video = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).astype(jnp.int32)
        # Generate segment IDs: each token knows which image/video it belongs to
        decoder_segment_ids = generate_segment_ids_from_counts(tokens_per_video)

        # Process blocks and collect outputs for deep features
        h_traj = []
        for blk in self.blocks:
            x = blk(x, grid_thw=grid_thw, decoder_segment_ids=decoder_segment_ids)
            h_traj.append(x)

        # Extract deep features at specified indices
        deep_feats = [
            self.merger_list[i](h_traj[idx])
            for i, idx in enumerate(self.deep_idx)
        ]

        # Final merger
        x = self.final_merger(x)
        return x, deep_feats
