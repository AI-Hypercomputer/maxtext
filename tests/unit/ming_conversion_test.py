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

"""Regression coverage for Ming conversion layout and reversible tensor transforms."""

from types import SimpleNamespace
import numpy as np
import pytest
from maxtext.checkpoint_conversion.utils.param_mapping import (
    MING_MAXTEXT_TO_HF_PARAM_MAPPING,
    MING_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    _ming_scan_layout,
)
from maxtext.models import ming


def make_config(layers=14, audio=True):
  """Small conversion metadata with a dense prefix and partial final cycle."""
  return SimpleNamespace(
      num_decoder_layers=layers,
      inhomogeneous_layer_cycle_interval=4,
      first_num_dense_layers=1,
      num_experts=3,
      use_audio=audio,
      encoder_layers_for_audio=2,
      num_conv_layers_for_audio=3,
  )


def flatten(value):
  """Flattens expert/block lists of HF names."""
  if isinstance(value, str):
    return [value]
  return [name for item in value for name in flatten(item)]


@pytest.mark.parametrize("layers", [1, 4, 8, 14, 24])
def test_scanned_mapping_covers_same_parameters(layers):
  """Scanning preserves every HF tensor exactly once, including remainder layers."""
  config = make_config(layers)
  unscanned = MING_MAXTEXT_TO_HF_PARAM_MAPPING(None, config)
  scanned = MING_MAXTEXT_TO_HF_PARAM_MAPPING(None, config, scan_layers=True)
  expected = [name for value in unscanned.values() for name in flatten(value)]
  actual = [name for value in scanned.values() for name in flatten(value)]
  assert len(actual) == len(set(actual))
  assert sorted(actual) == sorted(expected)
  assert _ming_scan_layout(config) == ming.scan_layout(config)


def test_scanned_expert_and_block_order():
  """Expert lists use [expert][block] order rather than interleaving axes."""
  mapping = MING_MAXTEXT_TO_HF_PARAM_MAPPING(None, make_config(), scan_layers=True)
  key = "params-decoder-scanned_blocks-layers_0-moe_block-MoeBlock_0-wi_0"
  assert mapping[key] == [
      [f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight" for layer in [4, 8]] for expert in range(3)
  ]
  assert mapping["params-decoder-layers_0-mlp-wi_0-kernel"] == "model.layers.0.mlp.gate_proj.weight"
  assert mapping["params-decoder-layers_13-self_attention-wq_a-kernel"] == "model.layers.13.attention.q_a_proj.weight"


@pytest.mark.parametrize("scan", [False, True])
def test_hooks_round_trip(scan):
  """Exercise reshaped linear projections and grouped/audio conv kernels in both directions."""
  config = make_config()
  to_mt = MING_MAXTEXT_TO_HF_PARAM_HOOK_FN(None, config, scan_layers=scan)
  to_hf = MING_MAXTEXT_TO_HF_PARAM_HOOK_FN(None, config, scan_layers=scan, saving_to_hf=True)
  samples = [
      ("params-decoder-layers_3-self_attention-wq_b-kernel", (12, 5), (5, 3, 4)),
      ("params-decoder-layers_0-kda-q_conv1d-kernel", (6, 1, 4), (4, 1, 6)),
      ("params-audio_encoder-MingAudioProjector_0-conv_0-kernel", (6, 5, 3), (3, 5, 6)),
      ("params-decoder-layers_1-moe_block-MoeBlock_0-wi_0", (8, 5), (5, 8)),
  ]
  if scan:
    samples[-1] = ("params-decoder-scanned_blocks-layers_0-moe_block-MoeBlock_0-wi_0", (8, 5), (5, 8))
  for key, hf_shape, mt_shape in samples:
    original = np.arange(np.prod(hf_shape), dtype=np.float32).reshape(hf_shape)
    converted = to_mt[key](original, mt_shape)
    assert converted.shape == mt_shape
    np.testing.assert_array_equal(to_hf[key](converted, hf_shape), original)


def test_text_only_mapping_omits_audio():
  mapping = MING_MAXTEXT_TO_HF_PARAM_MAPPING(None, make_config(audio=False))
  assert not any("audio_encoder" in key for key in mapping)
