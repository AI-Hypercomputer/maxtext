# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the HLO collective text helpers.

The fixture lines are captured from real jax.jit lowerings of shard_map
collectives, plus the asynchronous -start/-done forms compiled HLO uses.
"""

from absl.testing import absltest

from tests.utils import hlo_test_utils

_PPERMUTE = (
    "  ppermute.1 = f32[2,16]{1,0} collective-permute(shard_map.2), channel_id=1,"
    " source_target_pairs={{0,1},{1,2},{2,3},{3,0}}"
)
_PPERMUTE_START = (
    "  collective-permute-start.1 = (f32[2,16]{1,0}, f32[2,16]{1,0}) collective-permute-start(shard_map.2),"
    " channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}"
)
_ALL_TO_ALL = (
    "  all_to_all.5 = bf16[8,8,4]{2,1,0} all-to-all(shard_map.2), channel_id=1,"
    " replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}"
)
_SEGMENT_ID_ALL_GATHER = (
    "  all_gather.1 = s32[4,512]{1,0} all-gather(broadcast.1), channel_id=1,"
    " replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true"
)
_KV_ALL_GATHER = (
    "  all-gather.3 = bf16[4,512,8,128]{3,2,1,0} all-gather(param.1), channel_id=2,"
    " replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true"
)
_KV_ALL_GATHER_START = (
    "  all-gather-start.1 = (bf16[4,128,8,128]{3,2,1,0}, bf16[4,512,8,128]{3,2,1,0})"
    " all-gather-start(param.1), channel_id=3, replica_groups={{0,1,2,3}}, dimensions={1}"
)
_KV_ALL_GATHER_DONE = "  all-gather-done.1 = bf16[4,512,8,128]{3,2,1,0} all-gather-done(all-gather-start.1)"
_FUSION_WITH_COLLECTIVE_OPERAND = "  fusion.1 = bf16[4,512]{1,0} fusion(all-gather-done.1), kind=kLoop"
_NON_SEQUENCE_DIM_ALL_GATHER = (
    "  all-gather.9 = bf16[8,512,16]{2,1,0} all-gather(param.2), channel_id=4,"
    " replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true"
)


class HloTestUtilsTest(absltest.TestCase):
  """Tests for collective_lines and attention_sequence_all_gather_lines."""

  def test_collective_lines_matches_sync_and_async_forms(self):
    hlo_text = "\n".join([_PPERMUTE, _PPERMUTE_START, _ALL_TO_ALL])
    self.assertLen(hlo_test_utils.collective_lines(hlo_text, "collective-permute"), 2)
    self.assertLen(hlo_test_utils.collective_lines(hlo_text, "all-to-all"), 1)
    self.assertLen(hlo_test_utils.collective_lines(hlo_text, "all-gather"), 0)

  def test_collective_lines_counts_each_async_collective_once(self):
    hlo_text = "\n".join([_KV_ALL_GATHER_START, _KV_ALL_GATHER_DONE, _FUSION_WITH_COLLECTIVE_OPERAND])
    self.assertLen(hlo_test_utils.collective_lines(hlo_text, "all-gather"), 1)

  def test_sequence_all_gather_lines_excludes_segment_id_gathers(self):
    hlo_text = "\n".join([_SEGMENT_ID_ALL_GATHER, _KV_ALL_GATHER])
    lines = hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, (512,))
    self.assertLen(lines, 1)
    self.assertIn("bf16", lines[0])

  def test_sequence_all_gather_lines_counts_segment_id_gathers_for_s32(self):
    hlo_text = "\n".join([_SEGMENT_ID_ALL_GATHER, _KV_ALL_GATHER])
    lines = hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, (512,), dtypes=("s32",))
    self.assertLen(lines, 1)
    self.assertIn("s32", lines[0])

  def test_sequence_all_gather_lines_detects_full_shape_in_async_tuple(self):
    lines = hlo_test_utils.attention_sequence_all_gather_lines(_KV_ALL_GATHER_START, (512,))
    self.assertLen(lines, 1)

  def test_sequence_all_gather_lines_ignores_other_sequence_lengths(self):
    self.assertLen(hlo_test_utils.attention_sequence_all_gather_lines(_KV_ALL_GATHER, (1024,)), 0)

  def test_sequence_all_gather_lines_ignores_non_sequence_gather_dimensions(self):
    self.assertLen(hlo_test_utils.attention_sequence_all_gather_lines(_NON_SEQUENCE_DIM_ALL_GATHER, (512,)), 0)


if __name__ == "__main__":
  absltest.main()
