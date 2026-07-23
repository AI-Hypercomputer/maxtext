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

"""MaxText adapters for target-aligned Tunix diffusion SFT."""

from collections.abc import Mapping
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.diffusion import scoring
from tunix.diffusion import types as diffusion_types


_REQUIRED_FIELDS = (
    "inputs",
    "inputs_position",
    "inputs_segmentation",
    "targets",
    "targets_position",
    "targets_segmentation",
    "completion_mask",
    "corruption_mask",
    "targets_loss_mask",
)


def _concrete_numpy(value):
    if isinstance(value, jax.core.Tracer):
        return None
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        return None
    return np.asarray(value)


def _validate_batch_masks(
    positions,
    validity_mask,
    completion_mask,
    corruption_mask,
    loss_weights,
    *,
    alignment,
    block_size,
):
    shapes = {
        "positions": tuple(positions.shape),
        "validity_mask": tuple(validity_mask.shape),
        "completion_mask": tuple(completion_mask.shape),
        "corruption_mask": tuple(corruption_mask.shape),
        "targets_loss_mask": tuple(loss_weights.shape),
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"diffusion SFT masks must have identical shapes; received {shapes}"
        )
    concrete = [
        _concrete_numpy(value)
        for value in (
            positions,
            validity_mask,
            completion_mask,
            corruption_mask,
            loss_weights,
        )
    ]
    if any(value is None for value in concrete):
        return
    concrete_positions = np.asarray(concrete[0])
    validity, completion, corruption = (
        np.asarray(value, dtype=bool) for value in concrete[1:4]
    )
    weights = np.asarray(concrete[4])
    weighted = weights != 0
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("targets_loss_mask must contain finite nonnegative weights")
    if np.any(completion & ~validity):
        raise ValueError("completion_mask must be a subset of valid target positions")
    if np.any(corruption & ~completion):
        raise ValueError("corruption_mask must be a subset of completion_mask")
    if np.any(weighted & ~completion):
        raise ValueError("targets_loss_mask must be a subset of completion_mask")
    allowed = corruption.copy()
    if alignment == "shifted":
        allowed |= (concrete_positions > 0) & (concrete_positions % block_size == 0)
    if np.any(weighted & ~allowed):
        raise ValueError(
            "diffusion SFT loss weights must own corrupted targets or shifted block anchors"
        )


def create_batch_adapter(config):
    """Builds a raw MaxText batch adapter for Tunix diffusion SFT."""
    alignment = config.block_diffusion_logit_alignment
    block_size = int(config.block_diffusion_block_size)

    def adapt(raw_batch: Mapping[str, Any]) -> diffusion_types.DiffusionTokenBatch:
        missing = sorted(name for name in _REQUIRED_FIELDS if name not in raw_batch)
        if missing:
            raise ValueError(
                f"block-diffusion SFT requires explicit batch fields; missing {missing}"
            )
        targets = jnp.asarray(raw_batch["targets"])
        positions = jnp.asarray(raw_batch["targets_position"], dtype=jnp.int32)
        validity_mask = jnp.asarray(raw_batch["targets_segmentation"]) != 0
        completion_mask = jnp.asarray(raw_batch["completion_mask"], dtype=jnp.bool_)
        corruption_mask = jnp.asarray(raw_batch["corruption_mask"], dtype=jnp.bool_)
        raw_loss_weights = jnp.asarray(
            raw_batch["targets_loss_mask"], dtype=jnp.float32
        )
        _validate_batch_masks(
            positions,
            validity_mask,
            completion_mask,
            corruption_mask,
            raw_loss_weights,
            alignment=alignment,
            block_size=block_size,
        )
        allowed = corruption_mask
        if alignment == "shifted":
            allowed |= (positions > 0) & (positions % block_size == 0)
        loss_weights = jnp.where(
            validity_mask & completion_mask & allowed, raw_loss_weights, 0.0
        )
        return diffusion_types.DiffusionTokenBatch.create(
            model_inputs={
                "input_tokens": jnp.asarray(raw_batch["inputs"]),
                "input_positions": jnp.asarray(
                    raw_batch["inputs_position"], dtype=jnp.int32
                ),
                "input_segmentation": jnp.asarray(raw_batch["inputs_segmentation"]),
                "targets": targets,
                "target_positions": positions,
                "target_segmentation": jnp.asarray(raw_batch["targets_segmentation"]),
            },
            target_ids=targets,
            loss_weights=loss_weights,
        )

    return adapt


def create_target_aligned_logits_fn(config):
    """Builds a MaxText scorer satisfying Tunix's diffusion logits contract."""
    alignment = config.block_diffusion_logit_alignment
    enable_dropout = bool(config.enable_dropout)

    def logits_fn(model: nnx.Module, model_inputs: diffusion_types.ModelInputs):
        base_model = getattr(model, "base", model)
        logits = base_model(
            decoder_input_tokens=model_inputs["input_tokens"],
            decoder_positions=model_inputs["input_positions"],
            decoder_segment_ids=model_inputs["input_segmentation"],
            enable_dropout=enable_dropout,
            decoder_target_tokens=model_inputs["targets"],
            decoder_target_mask=model_inputs["target_segmentation"],
        )
        return scoring.align_logits_to_targets(
            logits,
            alignment,
            model_inputs["target_positions"],
            model_inputs["target_segmentation"] != 0,
        )

    return logits_fn
