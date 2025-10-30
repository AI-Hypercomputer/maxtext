import jax
import jax.numpy as jnp

from MaxText.common_types import Array
from MaxText.layers.packing_utils import generate_segment_ids_from_counts

# Helper functions for audio encoder
def compute_max_chunk_len_after_cnn(n_window: int, num_conv_layers: int) -> int:
    """Compute maximum chunk length after CNN layers given n_window.

    For a full chunk of size (n_window * 2), compute output after num_conv_layers
    stride-2 convolutions.

    Args:
        n_window: Window size parameter
        num_conv_layers: Number of convolutional layers with stride 2
    """
    chunk_size = n_window * 2
    output_len = chunk_size
    for _ in range(num_conv_layers):
        output_len = (output_len - 1) // 2 + 1
    return output_len

def audio_encoder_get_feat_extract_output_lengths(input_lengths: Array, n_window: int, num_conv_layers: int) -> Array:
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder.

    This matches PyTorch's _get_feat_extract_output_lengths implementation which processes
    audio in chunks of size (n_window * 2) and applies stride-2 convolutions.

    Args:
        input_lengths: Input sequence lengths, shape (batch_size,)
        n_window: Window size parameter (e.g., 50, making chunk_size = 100)
        num_conv_layers: Number of convolutional layers with stride 2

    Returns:
        output_lengths_after_all_convs: Array of shape (batch_size,)
    """
    chunk_size = n_window * 2
    # Compute output length per full chunk through convolutions
    output_per_chunk = chunk_size
    for _ in range(num_conv_layers):
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
    n_window: int,
    num_conv_layers: int
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
    feature_lengths_after_cnn = audio_encoder_get_feat_extract_output_lengths(chunk_lengths, n_window, num_conv_layers)
    max_len_after_cnn = compute_max_chunk_len_after_cnn(n_window, num_conv_layers)
    padded_mask_after_cnn = jnp.arange(max_len_after_cnn)[None, :] < feature_lengths_after_cnn[:, None]

    return padded_feature, padded_mask_after_cnn

