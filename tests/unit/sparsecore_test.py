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

import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
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

  def test_offload_traces_off_tpu(self):
    # CPU/GPU runs must be able to trace a TPU config unchanged.
    with sparsecore.offload(True):
      pass
    with sparsecore.offload(False):
      pass


class CollectiveCapabilityTest(parameterized.TestCase):
  """Tests the gate that keeps MaxText from annotating a collective XLA would abort on."""

  @parameterized.named_parameters(
      ("v5p", "v5p-8", False),
      ("v6e", "v6e-16", False),
      ("tpu7x", "tpu7x-256", True),
  )
  def test_all_gather_offload_needs_ironwood(self, compile_topology, expected):
    self.assertEqual(
        sparsecore.supports_collective_offload(sparsecore.ALL_GATHER, compile_topology=compile_topology), expected
    )

  def test_ragged_all_to_all_offloads_on_any_sparse_core_chip(self):
    self.assertTrue(sparsecore.supports_collective_offload(sparsecore.RAGGED_ALL_TO_ALL, compile_topology="v5p-8"))

  @parameterized.named_parameters(("v5p", "v5p-8", False), ("tpu7x", "tpu7x-256", True))
  def test_reduce_scatter_follows_all_gather(self, compile_topology, expected):
    # Its transpose is an all-gather, which would be annotated too.
    self.assertEqual(
        sparsecore.supports_collective_offload(sparsecore.REDUCE_SCATTER, compile_topology=compile_topology), expected
    )

  def test_no_collective_offloads_without_a_sparse_core(self):
    for collective in (sparsecore.ALL_GATHER, sparsecore.RAGGED_ALL_TO_ALL, sparsecore.REDUCE_SCATTER):
      self.assertFalse(sparsecore.supports_collective_offload(collective, compile_topology="v4-8"))
      self.assertFalse(sparsecore.supports_collective_offload(collective, hardware="cpu"))

  def test_fsdp_target_is_dropped_where_all_gather_cannot_offload(self):
    self.assertEqual(
        sparsecore.supported_offload_targets("all", compile_topology="v5p-8"),
        frozenset([sparsecore.EP_COLLECTIVES, sparsecore.RAGGED_SORT]),
    )

  def test_every_target_survives_on_ironwood(self):
    self.assertEqual(
        sparsecore.supported_offload_targets("all", compile_topology="tpu7x-256"),
        frozenset(sparsecore.OFFLOAD_TARGETS),
    )

  def test_targets_that_need_no_collective_are_kept(self):
    self.assertEqual(
        sparsecore.supported_offload_targets(sparsecore.RAGGED_SORT, compile_topology="v5p-8"),
        frozenset([sparsecore.RAGGED_SORT]),
    )

  def test_unknown_target_still_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown SparseCore offload target"):
      sparsecore.supported_offload_targets("tensor_core", compile_topology="tpu7x-256")

  def test_disabled_needs_no_hardware(self):
    self.assertEqual(sparsecore.supported_offload_targets("", hardware="cpu"), frozenset())


class ComputeTypeContextTest(parameterized.TestCase):
  """Tests that a compute-type context manager is found on the installed JAX.

  ``jax.experimental.compute_on.compute_on`` changed from a context manager into
  a function transform in JAX 0.11.1, so the resolution has to be checked
  against whatever JAX is actually installed rather than assumed.
  """

  def test_offload_annotates_the_lowered_hlo(self):
    """The end-to-end check: ops traced inside `offload` carry the compute type.

    This is the assertion that has to run on CPU. The compute-type API is the
    part of this feature most likely to move under us, and every test that could
    catch it moving used to be TPU-gated, so a JAX upgrade broke the annotation
    with CI green.
    """

    def annotated(x):
      with sparsecore.offload(True):
        return x + 1

    def plain(x):
      return x + 1

    x = jnp.zeros((8, 8))
    self.assertIn('_xla_compute_type = "sparseoffload"', jax.jit(annotated).lower(x).as_text())
    self.assertNotIn("sparseoffload", jax.jit(plain).lower(x).as_text())

  def test_offload_skips_a_collective_the_target_chip_cannot_run(self):
    def annotated(x):
      with sparsecore.offload(True, collective=sparsecore.ALL_GATHER, compile_topology="v5p-8"):
        return x + 1

    self.assertNotIn("sparseoffload", jax.jit(annotated).lower(jnp.zeros((8, 8))).as_text())

  def test_a_context_manager_was_resolved(self):
    self.assertIsNotNone(
        sparsecore._COMPUTE_TYPE_CONTEXT,  # pylint: disable=protected-access
        "No compute-type context manager found on this JAX; SparseCore offloading would silently do nothing.",
    )

  def test_resolved_context_manager_accepts_the_compute_type(self):
    context = sparsecore._COMPUTE_TYPE_CONTEXT  # pylint: disable=protected-access
    with context(sparsecore.SPARSE_CORE_COMPUTE_TYPE):
      pass

  def test_function_transform_spelling_is_not_used_as_a_context_manager(self):
    """A 0.11.1-style `compute_on` must fall back, not be called positionally."""

    def function_transform(f=None, *, compute_type, out_memory_spaces, compiler_options=None):
      del f, compute_type, out_memory_spaces, compiler_options
      raise AssertionError("the function-transform spelling must not be used as a context manager")

    with mock.patch.object(sparsecore.compute_on, "compute_on", function_transform):
      resolved = sparsecore._resolve_compute_type_context()  # pylint: disable=protected-access
    self.assertIsNotNone(resolved)
    self.assertIsNot(resolved, function_transform)
    with resolved(sparsecore.SPARSE_CORE_COMPUTE_TYPE):
      pass

  def test_context_manager_spelling_is_used_directly(self):
    """The pre-0.11.1 public context manager is preferred when present."""

    @contextlib.contextmanager
    def context_manager(compute_type):
      del compute_type
      yield

    with mock.patch.object(sparsecore.compute_on, "compute_on", context_manager):
      self.assertIs(sparsecore._resolve_compute_type_context(), context_manager)  # pylint: disable=protected-access


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

  def test_target_the_chip_cannot_serve_is_accepted_and_ignored(self):
    # A SparseCore that cannot offload an all-gather must not fail the run: the
    # same config has to stay usable across chip generations.
    config = self._config(
        compile_topology="v5p-8", compile_topology_num_slices=1, moe_sparse_core_offload_targets="fsdp_all_gather"
    )
    self.assertEqual(config.moe_sparse_core_offload_targets, "fsdp_all_gather")
    self.assertEqual(
        sparsecore.supported_offload_targets(config.moe_sparse_core_offload_targets, config.compile_topology), frozenset()
    )

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
