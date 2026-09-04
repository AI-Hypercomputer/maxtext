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
"""Tests for SparseCore capability detection and the MoE offload plumbing.

Everything here runs on CPU: the detection helpers answer for a *target* chip
named by ``compile_topology``, and :func:`sparsecore.offload` degrades to a
no-op off TPU. The numerical equivalence of the offload itself is covered by
the TPU tests in ``moe_test.py``.
"""

from absl.testing import absltest
from absl.testing import parameterized

from jax.sharding import PartitionSpec

from maxtext.configs import pyconfig
from maxtext.utils import sharding as sharding_utils
from maxtext.utils import sparsecore

from tests.utils.test_helpers import get_test_config_path


class ParseOffloadTargetsTest(parameterized.TestCase):
  """Tests for parsing the ``moe_sparse_core_offload_targets`` config string."""

  def test_empty_string_disables_every_target(self):
    self.assertEqual(sparsecore.parse_offload_targets(""), frozenset())

  def test_all_expands_to_every_target(self):
    self.assertEqual(sparsecore.parse_offload_targets("all"), frozenset(sparsecore.OFFLOAD_TARGETS))

  @parameterized.parameters(*sparsecore.OFFLOAD_TARGETS)
  def test_single_target(self, target):
    self.assertEqual(sparsecore.parse_offload_targets(target), frozenset([target]))

  def test_comma_separated_subset_ignores_whitespace(self):
    parsed = sparsecore.parse_offload_targets(f" {sparsecore.RAGGED_SORT}, {sparsecore.EP_COLLECTIVES} ,")
    self.assertEqual(parsed, frozenset([sparsecore.RAGGED_SORT, sparsecore.EP_COLLECTIVES]))

  def test_unknown_target_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown SparseCore offload target"):
      sparsecore.parse_offload_targets("fsdp_all_gather,tensor_core")


class SparseCoreDetectionTest(parameterized.TestCase):
  """Tests that SparseCore presence is read off JAX's chip table."""

  @parameterized.named_parameters(
      ("v5p", "v5p-8", True),
      ("v6e", "v6e-16", True),
      ("tpu7x", "tpu7x-256", True),
      ("v4", "v4-8", False),
      ("v5e", "v5e-16", False),
  )
  def test_has_sparse_core_for_compile_topology(self, compile_topology, expected):
    self.assertEqual(sparsecore.has_sparse_core(compile_topology=compile_topology), expected)

  def test_gpu_topology_has_no_sparse_core(self):
    self.assertFalse(sparsecore.has_sparse_core(compile_topology="a3"))

  @parameterized.parameters(*sparsecore._NON_TPU_HARDWARE)  # pylint: disable=protected-access
  def test_non_tpu_hardware_has_no_sparse_core(self, hardware):
    # The hardware check wins even when the topology names a SparseCore chip.
    self.assertFalse(sparsecore.has_sparse_core(compile_topology="tpu7x-256", hardware=hardware))

  def test_unknown_topology_is_not_an_error(self):
    self.assertFalse(sparsecore.has_sparse_core(compile_topology="not-a-real-topology"))

  def test_sparse_core_info_reports_core_count(self):
    info = sparsecore.sparse_core_info(compile_topology="v5p-8")
    self.assertIsNotNone(info)
    self.assertGreater(info.num_cores, 0)

  def test_offload_is_a_no_op_off_tpu(self):
    # CPU/GPU runs must be able to trace a TPU config unchanged.
    if sparsecore.is_tpu_runtime():
      self.skipTest("This asserts the off-TPU fallback.")
    with sparsecore.offload(True):
      pass
    with sparsecore.offload(False):
      pass


class AllGatherAxesBetweenPspecsTest(parameterized.TestCase):
  """Tests for deriving the all-gathers that take one PartitionSpec to another."""

  def test_identical_pspecs_need_no_gather(self):
    pspec = PartitionSpec("expert", None, "mlp")
    self.assertEqual(sharding_utils.all_gather_axes_between_pspecs(pspec, pspec, 3), [])

  def test_single_dropped_axis(self):
    gathers = sharding_utils.all_gather_axes_between_pspecs(
        PartitionSpec("fsdp", None, "mlp"), PartitionSpec(None, None, "mlp"), 3
    )
    self.assertEqual(gathers, [(0, ("fsdp",))])

  def test_dropped_axes_on_several_dims(self):
    gathers = sharding_utils.all_gather_axes_between_pspecs(
        PartitionSpec(("expert", "fsdp"), "fsdp_transpose", "mlp"), PartitionSpec("expert", None, "mlp"), 3
    )
    self.assertEqual(gathers, [(0, ("fsdp",)), (1, ("fsdp_transpose",))])

  def test_shorter_pspec_is_padded_with_replicated_dims(self):
    gathers = sharding_utils.all_gather_axes_between_pspecs(PartitionSpec("fsdp"), PartitionSpec(), 3)
    self.assertEqual(gathers, [(0, ("fsdp",))])

  def test_gaining_an_axis_is_not_a_pure_all_gather(self):
    self.assertIsNone(
        sharding_utils.all_gather_axes_between_pspecs(
            PartitionSpec(None, None, "mlp"), PartitionSpec("fsdp", None, "mlp"), 3
        )
    )

  def test_dropping_a_major_axis_is_not_a_pure_all_gather(self):
    # `all_gather(tiled=True)` concatenates in device order, so only the minor
    # (suffix) axes of a dim can be gathered away.
    self.assertIsNone(
        sharding_utils.all_gather_axes_between_pspecs(
            PartitionSpec(("fsdp", "expert"), None, "mlp"), PartitionSpec("expert", None, "mlp"), 3
        )
    )

  def test_swapped_axes_are_not_a_pure_all_gather(self):
    self.assertIsNone(
        sharding_utils.all_gather_axes_between_pspecs(PartitionSpec("fsdp", None), PartitionSpec("expert", None), 2)
    )


class SparseCoreConfigValidationTest(parameterized.TestCase):
  """Tests that the config rejects offload requests the hardware cannot serve."""

  def _config(self, **overrides):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="sparsecore_config_test",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        **overrides,
    )

  def test_default_is_disabled(self):
    config = self._config()
    self.assertEqual(config.moe_sparse_core_offload_targets, "")

  def test_accepted_on_a_sparse_core_topology(self):
    config = self._config(
        compile_topology="tpu7x-256", compile_topology_num_slices=1, moe_sparse_core_offload_targets="all"
    )
    self.assertEqual(config.moe_sparse_core_offload_targets, "all")

  def test_rejected_on_a_topology_without_a_sparse_core(self):
    with self.assertRaisesRegex(ValueError, "requires a TPU with a .*SparseCore"):
      self._config(compile_topology="v4-8", compile_topology_num_slices=1, moe_sparse_core_offload_targets="ragged_sort")

  def test_rejected_on_cpu(self):
    with self.assertRaisesRegex(ValueError, "requires a TPU with a .*SparseCore"):
      self._config(hardware="cpu", moe_sparse_core_offload_targets="ep_collectives")

  def test_unknown_target_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Unknown SparseCore offload target"):
      self._config(
          compile_topology="tpu7x-256", compile_topology_num_slices=1, moe_sparse_core_offload_targets="everything"
      )


if __name__ == "__main__":
  absltest.main()
