"""Unit tests for the ``packing_max_segments_per_sample`` config.

Covers:

* ``megatron_min_segment_length`` branching on ``reset_attention_mask`` and the
  configurable divisor (default 25, ``<=0`` disables merging).
* ``MMapDataset.packing_max_segments_per_sample`` default value, preserving the
  prior hardcoded behavior.

Run with: ``python -m pytest tests/unit/packing_max_segments_test.py -v``
"""

from types import SimpleNamespace

import pytest

from maxtext.configs.types import MMapDataset
from maxtext.input_pipeline.input_pipeline_utils import megatron_min_segment_length


def _cfg(reset_attention_mask: bool, divisor: int, max_target_length: int = 4096):
  return SimpleNamespace(
      reset_attention_mask=reset_attention_mask,
      packing_max_segments_per_sample=divisor,
      max_target_length=max_target_length,
  )


class TestMegatronMinSegmentLength:
  """``megatron_min_segment_length`` returns ``max_target_length // divisor`` only when merging is active."""

  def test_default_divisor_matches_prior_hardcoded_25(self):
    # Prior behavior used a hardcoded 25; the default config value preserves that.
    cfg = _cfg(reset_attention_mask=True, divisor=25, max_target_length=4096)
    assert megatron_min_segment_length(cfg) == 4096 // 25

  @pytest.mark.parametrize(
      "divisor,max_target_length,expected",
      [
          (50, 4096, 4096 // 50),
          (10, 8192, 8192 // 10),
          (1, 4096, 4096),
          (100, 4097, 4097 // 100),  # integer division (truncates)
      ],
  )
  def test_custom_divisor(self, divisor, max_target_length, expected):
    cfg = _cfg(reset_attention_mask=True, divisor=divisor, max_target_length=max_target_length)
    assert megatron_min_segment_length(cfg) == expected

  @pytest.mark.parametrize("divisor", [0, -1, -25])
  def test_non_positive_divisor_disables_merging(self, divisor):
    cfg = _cfg(reset_attention_mask=True, divisor=divisor)
    assert megatron_min_segment_length(cfg) == 0

  @pytest.mark.parametrize("divisor", [0, 25, 50])
  def test_reset_attention_mask_false_returns_zero(self, divisor):
    # reset_attention_mask=False short-circuits regardless of divisor.
    cfg = _cfg(reset_attention_mask=False, divisor=divisor)
    assert megatron_min_segment_length(cfg) == 0


class TestPackingMaxSegmentsPerSampleConfig:
  """``MMapDataset`` exposes ``packing_max_segments_per_sample`` with the documented default."""

  def test_default_is_25_preserving_prior_behavior(self):
    # The new field replaces the hardcoded 25 in megatron_min_segment_length;
    # an unset default must preserve the prior threshold for existing configs.
    cfg = MMapDataset()
    assert cfg.packing_max_segments_per_sample == 25

  def test_custom_value_is_accepted(self):
    cfg = MMapDataset(packing_max_segments_per_sample=64)
    assert cfg.packing_max_segments_per_sample == 64

  def test_zero_disables_merging_round_trip(self):
    # Round-trip the documented "disable merging" sentinel through the config
    # and confirm megatron_min_segment_length agrees.
    cfg = MMapDataset(packing_max_segments_per_sample=0, reset_attention_mask=True)
    helper_cfg = _cfg(
        reset_attention_mask=cfg.reset_attention_mask,
        divisor=cfg.packing_max_segments_per_sample,
        max_target_length=4096,
    )
    assert megatron_min_segment_length(helper_cfg) == 0
