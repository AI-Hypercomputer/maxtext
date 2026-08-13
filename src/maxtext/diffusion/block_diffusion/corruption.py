# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random token corruption for block-diffusion training."""

from typing import NamedTuple

import numpy as np


_SUPPORTED_CONTRACTS = {
    ("same_position", "all_masked"),
    ("shifted", "seed_and_mask"),
}


class BlockDiffusionCorruptionResult(NamedTuple):
  """Corrupted token canvas and its two independent supervision masks."""

  inputs: np.ndarray
  corruption_mask: np.ndarray
  targets_loss_mask: np.ndarray


def corrupt_tokens(
    inputs: np.ndarray,
    validity_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    block_size: int,
    mask_id: int,
    min_noise: float = 1.0e-3,
    logit_alignment: str = "same_position",
    canvas_policy: str = "all_masked",
    axis: int = 1,
) -> BlockDiffusionCorruptionResult:
  """Masks valid tokens independently within each logical diffusion block.

  ``inputs`` and ``validity_mask`` must have the same shape. With the default
  ``axis=1``, that shape is ``[batch, sequence, ...]``; ``axis`` selects the
  sequence dimension when a different layout is used. The returned arrays
  have the same shape as ``inputs``.
  """
  inputs = np.asarray(inputs)
  validity_mask = np.asarray(validity_mask, dtype=np.bool_)
  if inputs.shape != validity_mask.shape:
    raise ValueError(f"inputs and validity_mask must have identical shapes, got {inputs.shape} and {validity_mask.shape}")
  if inputs.ndim == 0 or not -inputs.ndim <= axis < inputs.ndim:
    raise ValueError(f"axis {axis} is invalid for inputs with {inputs.ndim} dimensions")
  if inputs.shape[axis] == 0:
    raise ValueError("the diffusion sequence axis must be nonempty")
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}")
  if mask_id < 0:
    raise ValueError(f"mask_id must be nonnegative, got {mask_id}")
  if not 0.0 < min_noise <= 1.0:
    raise ValueError(f"min_noise must satisfy 0 < min_noise <= 1, got {min_noise}")
  if (logit_alignment, canvas_policy) not in _SUPPORTED_CONTRACTS:
    raise ValueError(
        "Block diffusion supports only same_position/all_masked or shifted/seed_and_mask; "
        f"got {logit_alignment}/{canvas_policy}"
    )
  if canvas_policy == "seed_and_mask" and block_size < 2:
    raise ValueError("seed_and_mask requires block_size to be at least 2")

  axis %= inputs.ndim
  moved_validity = np.moveaxis(validity_mask, axis, -1)
  rows = moved_validity.reshape(-1, inputs.shape[axis])
  block_count = (rows.shape[-1] + block_size - 1) // block_size
  padded_length = block_count * block_size
  padded_rows = np.pad(rows, ((0, 0), (0, padded_length - rows.shape[-1])))
  eligible_blocks = padded_rows.reshape(rows.shape[0], block_count, block_size)

  seed_loss_blocks = np.zeros_like(eligible_blocks)
  if canvas_policy == "seed_and_mask":
    seed_positions = np.arange(block_size) == 0
    seed_loss_blocks = eligible_blocks & seed_positions
    seed_loss_blocks[:, 0, :] = False
    eligible_blocks &= ~seed_positions

  noise = rng.uniform(min_noise, 1.0, size=eligible_blocks.shape[:-1] + (1,))
  selected_blocks = eligible_blocks & (rng.random(size=eligible_blocks.shape) < noise)

  # A nonempty eligible block must always contribute at least one target.
  needs_fallback = eligible_blocks.any(axis=-1) & ~selected_blocks.any(axis=-1)
  fallback_scores = np.where(eligible_blocks, rng.random(size=eligible_blocks.shape), -1.0)
  fallback_offsets = np.argmax(fallback_scores, axis=-1)
  fallback = np.arange(block_size) == fallback_offsets[..., None]
  selected_blocks |= needs_fallback[..., None] & fallback

  corruption_rows = selected_blocks.reshape(rows.shape[0], padded_length)[:, : rows.shape[-1]]
  loss_rows = (selected_blocks | seed_loss_blocks).reshape(rows.shape[0], padded_length)[:, : rows.shape[-1]]
  corruption_mask = np.moveaxis(corruption_rows.reshape(moved_validity.shape), -1, axis)
  targets_loss_mask = np.moveaxis(loss_rows.reshape(moved_validity.shape), -1, axis)
  return BlockDiffusionCorruptionResult(
      inputs=np.where(corruption_mask, inputs.dtype.type(mask_id), inputs),
      corruption_mask=corruption_mask,
      targets_loss_mask=targets_loss_mask,
  )
