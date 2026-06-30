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

"""Unit tests for the Orbax v1 Context / policy builders."""

import unittest

from absl.testing import absltest
from maxtext.common import checkpoint_context
from orbax.checkpoint import v1 as ocp_v1


_GB = 1024**3


class TestSaveDecisionPolicy(unittest.TestCase):
  """build_save_decision_policy mirrors the v0 manager's selection logic."""

  def test_fixed_interval_by_default(self):
    policies = ocp_v1.training.save_decision_policies
    policy = checkpoint_context.build_save_decision_policy(save_interval_steps=7)
    self.assertIsInstance(policy, policies.FixedIntervalPolicy)

  def test_continuous(self):
    policies = ocp_v1.training.save_decision_policies
    policy = checkpoint_context.build_save_decision_policy(
        save_interval_steps=7, enable_continuous_checkpointing=True
    )
    self.assertIsInstance(policy, policies.ContinuousCheckpointingPolicy)

  def test_autocheckpoint_is_any_of_preemption_or_interval(self):
    policies = ocp_v1.training.save_decision_policies
    policy = checkpoint_context.build_save_decision_policy(
        save_interval_steps=7, enable_autocheckpoint=True
    )
    self.assertIsInstance(policy, policies.AnySavePolicy)

  def test_continuous_takes_precedence_over_autocheckpoint(self):
    # Matches v0: the continuous branch is checked first.
    policies = ocp_v1.training.save_decision_policies
    policy = checkpoint_context.build_save_decision_policy(
        save_interval_steps=7,
        enable_continuous_checkpointing=True,
        enable_autocheckpoint=True,
    )
    self.assertIsInstance(policy, policies.ContinuousCheckpointingPolicy)


class TestPreservationPolicy(unittest.TestCase):

  def test_latest_n(self):
    policy = checkpoint_context.build_preservation_policy(max_num_checkpoints_to_keep=5)
    self.assertIsInstance(policy, ocp_v1.training.preservation_policies.LatestN)


class TestBuildContext(unittest.TestCase):
  """build_context maps flat flags onto the right Context fields."""

  def test_storage_format_and_file_size(self):
    ctx = checkpoint_context.build_context(
        use_ocdbt=False, use_zarr3=False, ocdbt_target_data_file_size_bytes=2048
    )
    self.assertFalse(ctx.array.saving.use_ocdbt)
    self.assertFalse(ctx.array.saving.use_zarr3)
    self.assertEqual(ctx.array.saving.ocdbt_target_data_file_size, 2048)
    self.assertEqual(ctx.array.saving.storage_options.chunk_byte_size, 2048)

  def test_concurrent_gb_to_bytes_both_directions(self):
    ctx = checkpoint_context.build_context(checkpoint_storage_concurrent_gb=96)
    self.assertEqual(ctx.memory.write_concurrent_bytes, 96 * _GB)
    self.assertEqual(ctx.memory.read_concurrent_bytes, 96 * _GB)

  def test_async_timeout(self):
    ctx = checkpoint_context.build_context(async_timeout_secs=3600)
    self.assertEqual(ctx.asynchronous.timeout_secs, 3600)

  def test_todelete_full_path(self):
    ctx = checkpoint_context.build_context(todelete_full_path="trash")
    self.assertEqual(ctx.deletion.gcs_deletion_options.todelete_full_path, "trash")

  def test_single_replica_restore_enables_load_and_broadcast(self):
    ctx = checkpoint_context.build_context(
        enable_single_replica_ckpt_restoring=True, replica_axis_index=1
    )
    self.assertTrue(ctx.array.loading.use_load_and_broadcast)
    self.assertEqual(ctx.array.loading.load_and_broadcast_options.replica_axis_index, 1)
    self.assertEqual(
        ctx.array.loading.load_and_broadcast_options.broadcast_memory_limit_bytes,
        1024 * 1024 * 1000,
    )

  def test_single_replica_off_by_default(self):
    ctx = checkpoint_context.build_context()
    self.assertFalse(ctx.array.loading.use_load_and_broadcast)

  def test_colocated_python_sets_pathways_impl(self):
    ctx = checkpoint_context.build_context(colocated_python_checkpointing=True)
    self.assertIsNotNone(ctx.pathways.checkpointing_impl)

  def test_checkpoint_layout(self):
    ctx = checkpoint_context.build_context(
        checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
    )
    self.assertEqual(ctx.checkpoint_layout, ocp_v1.options.CheckpointLayout.SAFETENSORS)

  def test_defaults_leave_ocdbt_zarr3_on(self):
    ctx = checkpoint_context.build_context()
    self.assertTrue(ctx.array.saving.use_ocdbt)
    self.assertTrue(ctx.array.saving.use_zarr3)


if __name__ == "__main__":
  absltest.main()
