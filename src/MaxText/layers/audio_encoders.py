import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from MaxText.common_types import Config, Array, AttentionType, MODEL_MODE_TRAIN
from MaxText.layers import nnx_wrappers, linears
from MaxText.layers.attentions import Attention
from MaxText.layers.embeddings import SinusoidsPositionEmbedding
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.initializers import nd_dense_init, variable_to_logically_partitioned
from MaxText.layers.packing_utils import generate_segment_ids_from_counts

# Model hyperparameters
DEFAULT_N_WINDOW: int = 50
DEFAULT_N_WINDOW_INFER: int = 400
NUM_CONV_LAYERS: int = 3
DEFAULT_MAX_TIMESCALE: float = 10000.0

# Static compilation parameters (for JIT shape/bound computation)
DEFAULT_MAX_SAMPLE_LEN: int = 10000
DEFAULT_CONV_CHUNKSIZE: int = 500


# Helper functions for audio encoder
def compute_max_chunk_len_after_cnn(n_window: int) -> int: 
    """Compute maximum chunk length after CNN layers given n_window.

    For a full chunk of size (n_window * 2), compute output after NUM_CONV_LAYERS
    stride-2 convolutions.
    """
    chunk_size = n_window * 2
    output_len = chunk_size
    for _ in range(NUM_CONV_LAYERS):
        output_len = (output_len - 1) // 2 + 1
    return output_len

def audio_encoder_get_feat_extract_output_lengths(input_lengths: Array, n_window: int = DEFAULT_N_WINDOW) -> Array:
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder.

    This matches PyTorch's _get_feat_extract_output_lengths implementation which processes
    audio in chunks of size (n_window * 2) and applies 3 stride-2 convolutions.

    Args:
        input_lengths: Input sequence lengths, shape (batch_size,)
        n_window: Window size parameter (default 50, making chunk_size = 100)

    Returns:
        output_lengths_after_all_convs: Array of shape (batch_size,)
    """
    chunk_size = n_window * 2
    # Compute output length per full chunk through 3 convolutions
    output_per_chunk = chunk_size
    for _ in range(NUM_CONV_LAYERS):
        output_per_chunk = (output_per_chunk - 1) // 2 + 1

    # Process remainder (< chunk_size) through convolutions
    input_lengths_leave = input_lengths % chunk_size
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    remainder_output = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1

    # Total output = full chunks + remainder
    output_lengths = remainder_output + (input_lengths // chunk_size) * output_per_chunk
    return output_lengths


def compute_chunk_lengths(feature_lengths: Array, n_window: int) -> tuple[Array, Array]:
    """Compute chunk lengths for variable-length audio features.

    Args:
        feature_lengths: Length of each audio feature sequence
        n_window: Window size parameter

    Returns:
        Tuple of (chunk_lengths, chunk_num) where chunk_lengths contains the length
        of each chunk and chunk_num contains the number of chunks per sample
    """
    chunk_window_size = n_window * 2

    chunk_num = jnp.ceil(feature_lengths / chunk_window_size).astype(jnp.int32)
    total_chunks = jnp.sum(chunk_num)

    chunk_lengths = jnp.full((total_chunks,), chunk_window_size, dtype=jnp.int32)

    padded = jnp.pad(chunk_num, (1, 0), constant_values=-1)
    tail_chunk_index = jnp.cumsum(padded)[1:]

    remainders = feature_lengths % chunk_window_size
    chunk_lengths = chunk_lengths.at[tail_chunk_index].set(remainders)
    chunk_lengths = jnp.where(chunk_lengths == 0, chunk_window_size, chunk_lengths)

    return chunk_lengths, chunk_num


def prepare_audio_chunks(
    input_features: Array,
    feature_lengths: Array,
    chunk_lengths: Array,
    chunk_num: Array,
    n_window: int = DEFAULT_N_WINDOW  # STATIC
) -> tuple[Array, Array]:
    """Split and pad audio chunks matching PyTorch's logic (JIT-compatible)."""
    batch_size = input_features.shape[0]
    num_mel_bins = input_features.shape[1]
    max_chunk_len = n_window * 2  # STATIC: inferred from n_window
    total_chunks = chunk_lengths.shape[0]

    # Compute cumulative chunk indices to map global chunk_idx -> (sample_idx, local_chunk_idx)
    chunk_num_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(chunk_num)])

    def extract_chunk(chunk_idx):
        """Extract and pad a single chunk."""
        # Find which sample this chunk belongs to
        sample_idx = jnp.searchsorted(chunk_num_cumsum, chunk_idx, side='right') - 1

        # Find local chunk index within that sample
        local_chunk_idx = chunk_idx - chunk_num_cumsum[sample_idx]

        # Compute start position in the sample
        start_pos = local_chunk_idx * max_chunk_len

        # Get the actual chunk length and sample length
        chunk_len = chunk_lengths[chunk_idx]
        sample_len = feature_lengths[sample_idx]

        # Extract chunk using dynamic slice
        chunk = jax.lax.dynamic_slice(
            input_features[sample_idx],
            start_indices=(0, start_pos),
            slice_sizes=(num_mel_bins, max_chunk_len)
        )

        # Create mask for valid positions in chunk
        positions = jnp.arange(max_chunk_len)
        valid_mask = (positions < chunk_len) & ((start_pos + positions) < sample_len)

        # Zero out invalid positions
        chunk = jnp.where(valid_mask[None, :], chunk, 0.0)

        return chunk

    # Vectorize over all chunks
    padded_feature = jax.vmap(extract_chunk)(jnp.arange(total_chunks))

    # Compute mask after CNN
    feature_lengths_after_cnn = audio_encoder_get_feat_extract_output_lengths(chunk_lengths, n_window)
    max_len_after_cnn = compute_max_chunk_len_after_cnn(n_window)
    padded_mask_after_cnn = jnp.arange(max_len_after_cnn)[None, :] < feature_lengths_after_cnn[:, None]

    return padded_feature, padded_mask_after_cnn


def generate_segment_ids(chunk_num: Array) -> Array:
    """Generate segment IDs for packed audio chunks.

    Args:
        chunk_num: Number of chunks per audio sample, shape (batch_size,)

    Returns:
        segment_ids: Segment ID for each chunk, shape (total_chunks,)
                    e.g., chunk_num=[2, 3, 1] -> segment_ids=[0, 0, 1, 1, 1, 2]
    """
    # Use the shared utility function
    return generate_segment_ids_from_counts(chunk_num)


class AudioMLP(nnx.Module):
    """MLP block for AudioEncoderLayer. """
     
    def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
        self.config = config
        self.rngs = rngs
        self.audio_encoder_layer_mlp_fc1 = linears.DenseGeneral(
            in_features_shape=self.config.d_model_for_audio,
            out_features_shape=self.config.encoder_ffn_dim_for_audio,
            dtype=self.config.dtype_mm,
            use_bias=True,
            matmul_precision=self.config.matmul_precision,
            rngs=self.rngs,
        )
        self.audio_encoder_layer_mlp_fc2 = linears.DenseGeneral(
            in_features_shape=self.config.encoder_ffn_dim_for_audio,
            out_features_shape=self.config.d_model_for_audio,
            dtype=self.config.dtype_mm,
            use_bias=True,
            matmul_precision=self.config.matmul_precision,
            rngs=self.rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.audio_encoder_layer_mlp_fc1(hidden_states)
        hidden_states = nnx.gelu(hidden_states, approximate=False)
        hidden_states = self.audio_encoder_layer_mlp_fc2(hidden_states)
        return hidden_states


class AudioEncoderLayer(nnx.Module):
    """Transformer encoder layer for audio model."""

    def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
        self.config = config
        self.mesh = mesh
        self.rngs = rngs

        self.hidden_states_shape = (
            self.config.per_device_batch_size,
            self.config.max_source_positions_for_audio,
            self.config.d_model_for_audio,
        )

        self.input_layer_norm = nnx.LayerNorm(
            num_features=self.config.d_model_for_audio,
            epsilon=1e-5,
            dtype=self.config.dtype_mm,
            rngs=self.rngs,
        )

        self.self_attention_audio = Attention(
            config=self.config,
            num_query_heads=self.config.encoder_attention_heads_for_audio,
            num_kv_heads=self.config.encoder_attention_heads_for_audio,
            head_dim=self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio,
            max_target_length=self.config.max_source_positions_for_audio,
            attention_kernel="dot_product",
            inputs_q_shape=self.hidden_states_shape,
            inputs_kv_shape=self.hidden_states_shape,
            float32_qk_product=self.config.float32_qk_product,
            float32_logits=self.config.float32_logits,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            mesh=self.mesh,
            dropout_rate=self.config.attention_dropout_for_audio,
            name="self_attention_audio",
            attention_type=AttentionType.FULL,
            is_nope_layer=True,  # No rotary position embeddings for audio
            use_bias_in_projections=True,
            use_qk_norm=False,
            query_pre_attn_scalar=1 / math.sqrt(self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio),
            model_mode=MODEL_MODE_TRAIN,
            rngs=self.rngs,
        )

        self.post_attention_layer_norm = nnx.LayerNorm(
            num_features=self.config.d_model_for_audio,
            epsilon=1e-5,
            dtype=self.config.dtype_mm,
            rngs=self.rngs,
        )

        self.AudioMLP = AudioMLP(config=self.config, rngs=self.rngs)

    def __call__(
        self,
        hidden_states: Array,
        decoder_segment_ids: Array | None = None,
        deterministic: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)
        hidden_states = self.self_attention_audio(
            inputs_q=hidden_states,
            inputs_kv=hidden_states,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layer_norm(hidden_states)
        hidden_states = self.AudioMLP(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AudioEncoder(nnx.Module):
    """Transformer encoder consisting of multiple AudioEncoderLayer layers.

    Attributes:
        config: Config containing model parameters
        mesh: Mesh, JAX device mesh (used for sharding)
    """

    def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
        self.config = config
        self.mesh = mesh
        self.rngs = rngs

        for lyr in range(self.config.encoder_layers_for_audio):
            layer_name = f"layers_{lyr}"
            layer = AudioEncoderLayer(
                config=self.config,
                mesh=self.mesh,
                rngs=self.rngs,
            )
            setattr(self, layer_name, layer)

    def __call__(
        self,
        hidden_states: Array,
        decoder_segment_ids: Array | None = None,
        deterministic: bool = False,
    ):
        for lyr in range(self.config.encoder_layers_for_audio):
            layer_name = f"layers_{lyr}"
            layer = getattr(self, layer_name)
            hidden_states = layer(
                hidden_states,
                decoder_segment_ids=decoder_segment_ids,
                deterministic=deterministic,
            )
        return hidden_states


class AudioModel(nnx.Module):
    """Audio model for processing audio inputs.

    This model processes audio features through convolutional layers followed
    by transformer encoder layers.

    Attributes:
        config: Config containing model parameters
        mesh: Mesh, JAX device mesh (used for sharding)
    """

    def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
        self.config = config
        self.mesh = mesh
        self.rngs = rngs

        self.positional_embedding = SinusoidsPositionEmbedding(
            length=self.config.max_source_positions_for_audio,
            channels=self.config.d_model_for_audio,
            max_timescale=DEFAULT_MAX_TIMESCALE,
        )

        self.layernorm_pre = nnx.LayerNorm(
            num_features=self.config.d_model_for_audio,
            epsilon=1e-5,
            dtype=self.config.dtype_mm,
            rngs=self.rngs,
        )

        self.layernorm_post = nnx.LayerNorm(
            num_features=self.config.d_model_for_audio,
            epsilon=1e-5,
            dtype=self.config.dtype_mm,
            rngs=self.rngs,
        )

        self.conv2d1 = nnx.Conv(
            in_features=1,
            out_features=self.config.downsample_hidden_size_for_audio,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.config.dtype_mm,
            param_dtype=self.config.weight_dtype,
            rngs=self.rngs,
        )

        self.conv2d2 = nnx.Conv(
            in_features=self.config.downsample_hidden_size_for_audio,
            out_features=self.config.downsample_hidden_size_for_audio,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.config.dtype_mm,
            param_dtype=self.config.weight_dtype,
            rngs=self.rngs,
        )

        self.conv2d3 = nnx.Conv(
            in_features=self.config.downsample_hidden_size_for_audio,
            out_features=self.config.downsample_hidden_size_for_audio,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.config.dtype_mm,
            param_dtype=self.config.weight_dtype,
            rngs=self.rngs,
        )

        conv_out_dim = self.config.downsample_hidden_size_for_audio * ((((self.config.num_mel_bins_for_audio + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = DenseGeneral(
            in_features_shape=conv_out_dim,
            out_features_shape=self.config.d_model_for_audio,
            use_bias=False,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
            rngs=self.rngs,
        )

        self.AudioEncoder = AudioEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs)

        self.proj1 = DenseGeneral(
            in_features_shape=self.config.d_model_for_audio,
            out_features_shape=self.config.d_model_for_audio,
            use_bias=True,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
            rngs=self.rngs,
        )

        self.proj2 = DenseGeneral(
            in_features_shape=self.config.d_model_for_audio,
            out_features_shape=self.config.output_dim_for_audio,
            use_bias=True,
            dtype=self.config.dtype_mm,
            weight_dtype=self.config.weight_dtype,
            kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
            rngs=self.rngs,
        )

    def __call__(
        self,
        audio_features: Array,
        audio_lengths: Array,
        output_attentions: None | bool = None,
        output_hidden_states: None | bool = None,
        return_dict: None | bool = None,
        deterministic: None | bool = False,
    ) -> Array:
        """Forward pass of the audio model.

        Args:
            audio_features: Input audio features of shape (batch_size, num_mel_bins, max_audio_length)
            audio_lengths: Actual lengths of each audio sample in the batch, shape (batch_size,)
            output_attentions: Whether to output attention weights (not currently used)
            output_hidden_states: Whether to output hidden states (not currently used)
            return_dict: Whether to return a dict (not currently used)
            deterministic: Whether to use deterministic mode (disables dropout)

        Returns:
            Encoded audio features of shape (batch_size * num_chunks, seq_len_after_conv, output_dim)
        """
        chunk_lengths, chunk_num = compute_chunk_lengths(audio_lengths, self.config.n_window_for_audio)
        padded_feature, padded_mask_after_cnn = prepare_audio_chunks(
            audio_features, audio_lengths, chunk_lengths, chunk_num, self.config.n_window_for_audio
        )

        # Generate segment IDs for packed chunks (shape: [total_chunks])
        chunk_segment_ids = generate_segment_ids(chunk_num)

        num_chunks = padded_feature.shape[0]
        padded_feature_with_channel = padded_feature[..., None]

        num_batches = (num_chunks + self.config.conv_chunksize_for_audio - 1) // self.config.conv_chunksize_for_audio
        padded_num_chunks = num_batches * self.config.conv_chunksize_for_audio

        pad_amount = padded_num_chunks - num_chunks
        if pad_amount > 0:
            pad_shape = (pad_amount, self.config.num_mel_bins_for_audio, padded_feature.shape[2], 1)
            padding = jnp.zeros(pad_shape, dtype=padded_feature_with_channel.dtype)
            padded_feature_with_channel = jnp.concatenate([padded_feature_with_channel, padding], axis=0)

        def process_conv_batch(carry, start_idx):
            batch_slice = jax.lax.dynamic_slice(
                padded_feature_with_channel,
                start_indices=(start_idx, 0, 0, 0),
                slice_sizes=(self.config.conv_chunksize_for_audio, self.config.num_mel_bins_for_audio, padded_feature.shape[2], 1)
            )
            x = self.conv2d1(batch_slice)
            x = jax.nn.gelu(x)
            x = self.conv2d2(x)
            x = jax.nn.gelu(x)
            x = self.conv2d3(x)
            x = jax.nn.gelu(x)
            return carry, x

        start_indices = jnp.arange(num_batches) * self.config.conv_chunksize_for_audio
        _, padded_embed = jax.lax.scan(process_conv_batch, None, start_indices)

        padded_embed = padded_embed.reshape(-1, *padded_embed.shape[2:])
        padded_embed = padded_embed[:num_chunks]

        b, f, t, c = padded_embed.shape
        padded_embed = padded_embed.transpose(0, 2, 1, 3)
        padded_embed = padded_embed.reshape(b, t, f * c)
        padded_embed = self.conv_out(padded_embed)

        seq_len = padded_embed.shape[1]
        pos_emb = self.positional_embedding(seq_len)
        pos_emb = jnp.broadcast_to(pos_emb[None, :, :], (b, seq_len, self.config.d_model_for_audio))
        hidden_states = padded_embed + pos_emb

        # Expand segment IDs to match sequence length: [num_chunks] -> [num_chunks, seq_len]
        # Each position in a chunk gets the same segment ID
        decoder_segment_ids = jnp.broadcast_to(
            chunk_segment_ids[:, None], (num_chunks, seq_len)
        )

        hidden_states = self.layernorm_pre(hidden_states)
        hidden_states = self.AudioEncoder(
            hidden_states,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
        )
        hidden_states = self.layernorm_post(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)

        hidden_states = jnp.where(padded_mask_after_cnn[:, :, None], hidden_states, 0.0)

        return hidden_states


def audiomodel_as_linen(config: Config, mesh: Mesh) -> nnx.Module:
    """Convert AudioModel to Linen module for integration with the rest of the codebase."""
    return nnx_wrappers.to_linen(
        AudioModel,
        config=config,
        mesh=mesh,
        name="AudioModel",
        abstract_init=False,
        metadata_fn=variable_to_logically_partitioned,
    )

# def create_block_diagonal_mask(cu_seqlens: Array, max_length: int, dtype: jnp.dtype = jnp.float32) -> Array:
#     """Create block-diagonal attention mask from cumulative sequence lengths.

#     This creates a mask where positions within the same segment (defined by cu_seqlens
#     boundaries) can attend to each other, but cannot attend across segments.
#     Positions beyond cu_seqlens[-1] are assigned a special segment ID and will be
#     masked by a separate validity mask.

#     Args:
#         cu_seqlens: Cumulative sequence lengths of shape (num_segments + 1,).
#                    E.g., [0, 8, 16, 24] defines 3 segments of length 8 each.
#         max_length: Maximum sequence length for the mask. Should be >= cu_seqlens[-1].
#                    This must be a static Python int for JIT compilation.
#         dtype: Data type for the mask (default: float32)

#     Returns:
#         Attention mask of shape (1, 1, max_length, max_length) in additive format
#         where 0.0 means "can attend" and finfo(dtype).min means "cannot attend".

#     Example:
#         >>> cu_seqlens = jnp.array([0, 8, 16])
#         >>> mask = create_block_diagonal_mask(cu_seqlens, 16)
#         >>> # Positions 0-7 can attend to 0-7, positions 8-15 can attend to 8-15
#         >>> # but positions 0-7 cannot attend to 8-15 and vice versa
#     """
#     # For each position, determine which segment it belongs to
#     position_ids = jnp.arange(max_length)
#     # segment_ids[i] = j means position i is in segment j
#     # (between cu_seqlens[j] and cu_seqlens[j+1])
#     # Mark positions beyond cu_seqlens[-1] with a special segment ID (-1)
#     actual_length = cu_seqlens[-1]
#     valid_positions = position_ids < actual_length

#     segment_ids_raw = jnp.searchsorted(cu_seqlens, position_ids, side='right') - 1
#     # Assign invalid segment ID to positions beyond actual length
#     segment_ids = jnp.where(valid_positions, segment_ids_raw, -1)

#     # Create block-diagonal mask: positions can attend if they have same segment_id
#     # Positions with segment_id=-1 will all have same ID, but will be masked by validity mask
#     block_diagonal_mask = segment_ids[:, None] == segment_ids[None, :]  # (max_length, max_length)

#     # Convert boolean mask to additive mask for attention: True->0, False->finfo(dtype).min
#     min_value = jnp.finfo(dtype).min
#     # Add batch and head dimensions: (1, 1, max_length, max_length)
#     attn_mask = jnp.where(block_diagonal_mask, 0.0, min_value)
#     attn_mask = attn_mask[None, None, :, :].astype(dtype)

#     return attn_mask


# def compute_cu_seqlens(
#     sample_lens_after_cnn: Array,
#     max_chunk_len_after_cnn: int,  # STATIC
#     n_window: int = DEFAULT_N_WINDOW,
#     n_window_infer: int = DEFAULT_N_WINDOW_INFER,
#     max_sample_len: int = DEFAULT_MAX_SAMPLE_LEN  # STATIC
# ) -> Array:
#     """Compute cumulative sequence lengths for windowed attention. Not JIT-compatible."""
#     window_aftercnn = max_chunk_len_after_cnn * (n_window_infer // (n_window * 2))
#     num_full_windows = sample_lens_after_cnn // window_aftercnn
#     remainders = sample_lens_after_cnn % window_aftercnn

#     max_windows_per_sample = (max_sample_len // window_aftercnn) + 1
#     batch_size = sample_lens_after_cnn.shape[0]

#     def create_chunks_for_sample(sample_idx):
#         n_full = num_full_windows[sample_idx]
#         remainder = remainders[sample_idx]
#         window_indices = jnp.arange(max_windows_per_sample)

#         chunk_len = jnp.where(
#             window_indices < n_full,
#             window_aftercnn,
#             jnp.where((window_indices == n_full) & (remainder > 0), remainder, 0)
#         )
#         return chunk_len

#     all_chunks = jax.vmap(create_chunks_for_sample)(jnp.arange(batch_size))
#     chunk_lens_flat = all_chunks.reshape(-1)

#     # Pack non-zero values densely (not JIT-compatible due to dynamic slicing)
#     nonzero_mask = chunk_lens_flat > 0
#     total_nonzero = jnp.sum(nonzero_mask)
#     indices = jnp.where(nonzero_mask, jnp.cumsum(nonzero_mask) - 1, len(chunk_lens_flat))

#     packed = jnp.zeros(len(chunk_lens_flat) + 1, dtype=jnp.int32)
#     packed = packed.at[indices].set(jnp.where(nonzero_mask, chunk_lens_flat, 0))
#     chunk_lens_actual = packed[:total_nonzero]

#     cu_chunk_lens = jnp.concatenate([jnp.array([0], dtype=jnp.int32), chunk_lens_actual])
#     return jnp.cumsum(cu_chunk_lens)