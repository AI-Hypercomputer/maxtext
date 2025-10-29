# Copyright 2023–2025 Google LLC
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

"""Utilities for handling packed sequences across modalities."""

import jax.numpy as jnp
from MaxText.common_types import Array


def generate_segment_ids_from_counts(counts: Array) -> Array:
    """Generate segment IDs from counts of items per segment.

    This is a generic utility that works for any packed sequence scenario
    (audio chunks, vision tokens, text tokens, etc.).

    Args:
        counts: Number of items per segment, shape (num_segments,)
                e.g., [2, 3, 1] means segment 0 has 2 items, segment 1 has 3, segment 2 has 1

    Returns:
        segment_ids: Segment ID for each item, shape (total_items,)
                    e.g., counts=[2, 3, 1] -> segment_ids=[0, 0, 1, 1, 1, 2]

    Examples:
        >>> counts = jnp.array([2, 3, 1])
        >>> generate_segment_ids_from_counts(counts)
        Array([0, 0, 1, 1, 1, 2], dtype=int32)

        >>> counts = jnp.array([3])  # Single segment with 3 items
        >>> generate_segment_ids_from_counts(counts)
        Array([0, 0, 0], dtype=int32)
    """
    num_segments = counts.shape[0]
    total_items = jnp.sum(counts)

    # Create segment IDs by repeating segment indices
    # Use cumsum to find boundaries
    counts_cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])

    # For each item position, find which segment it belongs to
    item_indices = jnp.arange(total_items)
    segment_ids = jnp.searchsorted(counts_cumsum, item_indices, side='right') - 1

    return segment_ids


def compute_tokens_per_video(grid_thw: Array, spatial_merge_size: int) -> Array:
    """Compute the number of tokens per video/image after spatial merging.

    Args:
        grid_thw: Array of shape [num_videos, 3] with (temporal, height, width) per video
        spatial_merge_size: Spatial merge factor (e.g., 2 means 2x2 patches merge into 1)

    Returns:
        tokens_per_video: Number of tokens per video, shape (num_videos,)

    Examples:
        >>> grid_thw = jnp.array([[1, 24, 24], [1, 12, 12]])  # 2 images
        >>> compute_tokens_per_video(grid_thw, spatial_merge_size=2)
        Array([144, 36], dtype=int32)  # 1*12*12=144, 1*6*6=36
    """
    tokens_per_video = (
        grid_thw[:, 0]  # temporal
        * (grid_thw[:, 1] // spatial_merge_size)  # merged height
        * (grid_thw[:, 2] // spatial_merge_size)  # merged width
    )
    return tokens_per_video.astype(jnp.int32)
