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

"""Tests for the MaxText-to-Tunix diffusion SFT adapter."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.diffusion import scoring
from maxtext.integration.tunix import diffusion_sft


pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]


def _config(alignment="shifted"):
    return SimpleNamespace(
        block_diffusion_block_size=4,
        block_diffusion_logit_alignment=alignment,
        enable_dropout=False,
    )


def _raw_batch():
    positions = jnp.arange(8, dtype=jnp.int32)[None, :]
    segmentation = jnp.ones((1, 8), dtype=jnp.int32)
    return {
        "inputs": jnp.asarray([[10, 11, 99, 99, 14, 99, 99, 99]], dtype=jnp.int32),
        "inputs_position": positions,
        "inputs_segmentation": segmentation,
        "targets": jnp.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=jnp.int32),
        "targets_position": positions,
        "targets_segmentation": segmentation,
        "completion_mask": jnp.asarray([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32),
        "corruption_mask": jnp.asarray([[0, 0, 1, 1, 0, 1, 1, 1]], dtype=jnp.int32),
        "targets_loss_mask": jnp.asarray([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32),
    }


def test_batch_adapter_preserves_shifted_anchor_weight():
    batch = diffusion_sft.create_batch_adapter(_config())(_raw_batch())

    np.testing.assert_array_equal(batch.target_ids, _raw_batch()["targets"])
    np.testing.assert_array_equal(batch.loss_weights, [[0, 0, 1, 1, 1, 1, 1, 1]])
    assert batch.loss_weights.dtype == jnp.float32


def test_batch_adapter_rejects_unowned_clean_target():
    raw = _raw_batch()
    raw["corruption_mask"] = raw["corruption_mask"].at[0, 3].set(0)

    with pytest.raises(ValueError, match="corrupted targets or shifted block anchors"):
        diffusion_sft.create_batch_adapter(_config())(raw)


def test_logits_adapter_uses_target_alignment():
    raw_logits = jnp.arange(1 * 8 * 3, dtype=jnp.float32).reshape(1, 8, 3)
    calls = []

    class Model:

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return raw_logits

    batch = diffusion_sft.create_batch_adapter(_config())(_raw_batch())
    actual = diffusion_sft.create_target_aligned_logits_fn(_config())(
        Model(), batch.model_inputs
    )
    expected = scoring.align_logits_to_targets(
        raw_logits,
        "shifted",
        batch.model_inputs["target_positions"],
        batch.model_inputs["target_segmentation"] != 0,
    )

    np.testing.assert_array_equal(actual, expected)
    assert calls[0]["enable_dropout"] is False
    np.testing.assert_array_equal(
        calls[0]["decoder_input_tokens"], _raw_batch()["inputs"]
    )
