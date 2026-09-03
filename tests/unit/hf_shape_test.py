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

"""Tests for hf_shape.py shape maps (checkpoint_conversion HF export)."""

import unittest
from unittest import mock

import pytest

pytestmark = [pytest.mark.decoupled_target]


from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping


def _gemma4_e2b_config(standardize=False, num_text_layers=4, num_vision_layers=3):
  """A config dict shaped like the HF ``gemma4-e2b`` config.

  Small layer counts keep the test fast; key *coverage* does not depend on the
  layer count. Vision dims mirror ``configs/models/gemma4-e2b.yml``.
  """
  return {
      "text_config": {
          "num_hidden_layers": num_text_layers,
          "hidden_size": 640,
          "intermediate_size": 2048,
          "num_attention_heads": 4,
          "num_key_value_heads": 1,
          "num_global_key_value_heads": 1,
          "head_dim": 256,
          "global_head_dim": 512,
          "vocab_size": 262144,
          "hidden_size_per_layer_input": 256,
          "vocab_size_per_layer_input": 262144,
          "num_kv_shared_layers": 2,
          "layer_types": (["sliding_attention"] * (num_text_layers - 1)) + ["full_attention"],
          "use_double_wide_mlp": True,
      },
      "vision_config": {
          "hidden_size": 768,
          "intermediate_size": 3072,
          "num_hidden_layers": num_vision_layers,
          "num_attention_heads": 12,
          "head_dim": 64,
          "num_key_value_heads": 12,
          "position_embedding_size": 10240,
          "patch_size": 16,
          "num_channels": 3,
          "standardize": standardize,
      },
  }


class Gemma4SmallHfShapeTest(unittest.TestCase):
  """GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE (gemma4-e2b / gemma4-e4b) multimodal export."""

  def _mapped_hf_targets(self, config, use_multimodal=True, v_norm_with_scale=False, use_clipped_linears_for_vit=False):
    maxtext_config = mock.Mock()
    maxtext_config.use_multimodal = use_multimodal
    maxtext_config.v_norm_with_scale = v_norm_with_scale
    maxtext_config.use_clipped_linears_for_vit = use_clipped_linears_for_vit
    mapping = param_mapping.GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    return {v for v in mapping.values() if isinstance(v, str)}

  def test_shape_map_covers_every_mapped_vision_target(self):
    """Fail-closed: every HF target the param map emits must have a shape.

    ``utils._process`` raises ``ValueError('HF path ... not found in
    hf_shape_map')`` for any mapped target absent from the shape map. Before the
    vision block was added, the small-model shape map emitted zero
    ``model.vision_tower.*`` keys, so ``to_huggingface use_multimodal=true`` died
    on the first vision tensor. This asserts the coverage that keeps it alive.
    """
    config = _gemma4_e2b_config(standardize=False)
    shape_map = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config)
    targets = self._mapped_hf_targets(config)

    vision_targets = {t for t in targets if "vision_tower" in t or "embed_vision" in t}
    self.assertGreater(len(vision_targets), 0, "mapping produced no vision targets to check")

    missing = sorted(t for t in targets if t not in shape_map)
    self.assertEqual(missing, [], f"{len(missing)} mapped HF targets missing from shape map: {missing[:10]}")

  def test_vision_tower_keys_present(self):
    """The canonical vision entry, projector, and per-layer keys are emitted."""
    config = _gemma4_e2b_config(standardize=False)
    shape_map = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config)

    self.assertIn("model.vision_tower.patch_embedder.input_proj.weight", shape_map)
    self.assertIn("model.vision_tower.patch_embedder.position_embedding_table", shape_map)
    self.assertIn("model.embed_vision.embedding_projection.weight", shape_map)
    self.assertIn("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", shape_map)
    self.assertIn("model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight", shape_map)

  def test_clip_bound_keys_present_and_scalar(self):
    """Clip-bound targets are emitted and are scalar ``[]`` (not ``[1]``).

    ``GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING`` maps four activation clip bounds
    per clipped-linear vision projection (q/k/v/o + gate/up/down) under
    ``use_clipped_linears_for_vit``. The shape map covers them so the multimodal
    export composes with that mapping. A rank-1 ``[1]`` shape would make the HF
    loader reinitialize the bound to a non-finite sentinel, so it must be ``[]``.
    """
    config = _gemma4_e2b_config(standardize=False)
    shape_map = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config)
    n_vision_layers = config["vision_config"]["num_hidden_layers"]
    clip_keys = [k for k in shape_map if k.endswith((".input_min", ".input_max", ".output_min", ".output_max"))]
    # 7 clipped projections x 4 bounds x n_vision_layers.
    self.assertEqual(len(clip_keys), 7 * 4 * n_vision_layers)
    for k in clip_keys:
      self.assertEqual(shape_map[k], [], f"clip bound {k} must be scalar [] (got {shape_map[k]})")
    # Spot-check the exact HF target format the param map emits.
    self.assertIn("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min", shape_map)
    self.assertIn("model.vision_tower.encoder.layers.0.mlp.down_proj.output_max", shape_map)

  def test_shape_map_covers_clipped_mapping_targets(self):
    """Fail-closed with clipped-linears ON: every HF target the param map emits
    (including the clip bounds, when the running param map supports them) must
    exist in the shape map.

    The clip-bound mapping is added by the E2B/E4B multimodal feature; on a base
    without it the param map emits no clip targets and this reduces to the plain
    coverage check. When it is present, the shape map must cover all 448 bounds.
    """
    config = _gemma4_e2b_config(standardize=False)
    shape_map = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config)
    targets = self._mapped_hf_targets(config, use_clipped_linears_for_vit=True)
    missing = sorted(t for t in targets if t not in shape_map)
    self.assertEqual(missing, [], f"{len(missing)} mapped targets missing from shape map: {missing[:10]}")
    clip_targets = {t for t in targets if t.endswith((".input_min", ".input_max", ".output_min", ".output_max"))}
    if clip_targets:  # param map supports clipped-linears (E2B/E4B feature present)
      n_vision_layers = config["vision_config"]["num_hidden_layers"]
      self.assertEqual(len(clip_targets), 7 * 4 * n_vision_layers)

  def test_std_keys_only_when_standardize(self):
    """std_scale / std_bias appear only under ``standardize=true`` (E2B/E4B ship
    ``standardize=false`` and store no std keys)."""
    off = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(_gemma4_e2b_config(standardize=False))
    on = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(_gemma4_e2b_config(standardize=True))
    self.assertNotIn("model.vision_tower.std_scale", off)
    self.assertNotIn("model.vision_tower.std_bias", off)
    self.assertIn("model.vision_tower.std_scale", on)
    self.assertIn("model.vision_tower.std_bias", on)

  def test_text_only_config_emits_no_vision_keys(self):
    """With no vision_config the shape map is text-only (no regression to the
    text-only export path)."""
    config = _gemma4_e2b_config()
    config.pop("vision_config")
    shape_map = hf_shape.GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config)
    vision_keys = [k for k in shape_map if "vision_tower" in k or "embed_vision" in k]
    self.assertEqual(vision_keys, [])


if __name__ == "__main__":
  unittest.main()
