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
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.diffusion.block_diffusion import target_alignment
from maxtext.integration.tunix import diffusion_sft


pytestmark = [pytest.mark.post_training]


def _config(alignment="shifted", completion_only=True):
  return SimpleNamespace(
      causal_block_size=4,
      block_diffusion_logit_alignment=alignment,
      sft_train_on_completion_only=completion_only,
      enable_dropout=False,
  )


def _raw_batch(array_module=np):
  positions = array_module.arange(8, dtype=np.int32)[None, :]
  segmentation = array_module.ones((1, 8), dtype=np.int32)
  return {
      "inputs": array_module.asarray([[10, 11, 99, 99, 14, 99, 99, 99]], dtype=np.int32),
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets": array_module.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=np.int32),
      "targets_position": positions,
      "targets_segmentation": segmentation,
      "completion_mask": array_module.asarray([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.int32),
      "corruption_mask": array_module.asarray([[0, 0, 1, 1, 0, 1, 1, 1]], dtype=np.int32),
      "targets_loss_mask": array_module.asarray([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.int32),
  }


def test_batch_adapter_preserves_shifted_anchor_weight():
  batch = diffusion_sft.create_batch_adapter(_config())(_raw_batch())

  np.testing.assert_array_equal(batch.target_ids, _raw_batch()["targets"])
  np.testing.assert_array_equal(batch.loss_weights, [[0, 0, 1, 1, 1, 1, 1, 1]])
  assert batch.loss_weights.dtype == jnp.float32


def test_batch_adapter_rejects_unowned_clean_target():
  raw = _raw_batch()
  raw["corruption_mask"][0, 3] = 0

  with pytest.raises(ValueError, match="corrupted targets or shifted block anchors"):
    diffusion_sft.create_batch_adapter(_config())(raw)


@pytest.mark.parametrize("field", ["targets_position", "targets_loss_mask"])
def test_batch_adapter_rejects_none_required_field(field):
  raw = _raw_batch()
  raw[field] = None

  with pytest.raises(ValueError, match=field):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_batch_adapter_preserves_full_token_cft_supervision():
  raw = _raw_batch()
  raw["corruption_mask"] = np.ones_like(raw["corruption_mask"])
  raw["targets_loss_mask"] = np.ones_like(raw["targets_loss_mask"])

  batch = diffusion_sft.create_batch_adapter(_config(alignment="same_position", completion_only=False))(raw)

  np.testing.assert_array_equal(batch.loss_weights, jnp.ones((1, 8)))


def test_completion_only_adapter_rejects_prompt_supervision():
  raw = _raw_batch()
  raw["corruption_mask"][0, 0] = 1
  raw["targets_loss_mask"][0, 0] = 1

  with pytest.raises(ValueError, match="configured supervision scope"):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_completion_only_adapter_rejects_future_prompt_in_block():
  raw = _raw_batch()
  raw["completion_mask"][0, 5] = 0
  raw["corruption_mask"][0, 5] = 0
  raw["targets_loss_mask"][0, 5] = 0

  with pytest.raises(ValueError, match="prompt token after a completion token"):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_concrete_numpy_does_not_materialize_jax_arrays():
  value = jnp.ones((1, 2))

  with mock.patch.object(diffusion_sft.np, "asarray") as asarray:
    assert diffusion_sft._concrete_numpy(value) is None  # pylint: disable=protected-access

  asarray.assert_not_called()


def test_batch_adapter_accepts_jax_arrays_without_eager_value_validation():
  batch = diffusion_sft.create_batch_adapter(_config())(_raw_batch(jnp))

  np.testing.assert_array_equal(batch.target_ids, _raw_batch()["targets"])


def test_batch_adapter_rejects_mismatched_mask_shapes():
  raw = _raw_batch()
  raw["completion_mask"] = raw["completion_mask"][:, :-1]

  with pytest.raises(ValueError, match="must have identical shapes"):
    diffusion_sft.create_batch_adapter(_config())(raw)


@pytest.mark.parametrize("invalid_weight", [-1.0, np.nan])
def test_batch_adapter_rejects_invalid_loss_weights(invalid_weight):
  raw = _raw_batch()
  raw["targets_loss_mask"] = raw["targets_loss_mask"].astype(np.float32)
  raw["targets_loss_mask"][0, 2] = invalid_weight

  with pytest.raises(ValueError, match="finite nonnegative weights"):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_batch_adapter_rejects_completion_outside_validity():
  raw = _raw_batch()
  raw["targets_segmentation"][0, 2] = 0

  with pytest.raises(ValueError, match="subset of valid target positions"):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_batch_adapter_rejects_loss_weight_outside_supervision():
  raw = _raw_batch()
  raw["targets_loss_mask"][0, 0] = 1

  with pytest.raises(ValueError, match="targets_loss_mask must be a subset"):
    diffusion_sft.create_batch_adapter(_config())(raw)


def test_logits_adapter_uses_target_alignment():
  raw_logits = jnp.arange(1 * 8 * 3, dtype=jnp.float32).reshape(1, 8, 3)
  calls = []

  class Model:

    def __call__(self, **kwargs):
      calls.append(kwargs)
      return raw_logits

  batch = diffusion_sft.create_batch_adapter(_config())(_raw_batch())
  actual = diffusion_sft.create_target_aligned_logits_fn(_config())(Model(), batch.model_inputs)
  expected = target_alignment.align_logits_to_targets(
      raw_logits,
      "shifted",
      batch.model_inputs["target_positions"],
      batch.model_inputs["target_segmentation"] != 0,
  )

  np.testing.assert_array_equal(actual, expected)
  assert calls[0]["enable_dropout"] is False
  np.testing.assert_array_equal(calls[0]["decoder_input_tokens"], _raw_batch()["inputs"])
