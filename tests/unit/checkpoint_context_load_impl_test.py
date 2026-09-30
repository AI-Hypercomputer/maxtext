# Copyright 2026 Google LLC
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

"""Unit tests for pathways_load_impl in checkpoint_context.build_context."""

import unittest
from unittest import mock

from absl.testing import absltest
from maxtext.common import checkpoint_context
from orbax.checkpoint import pathways as ocp_pathways
from orbax.checkpoint.experimental.v1._src.serialization import registration


class CheckpointContextLoadImplTest(unittest.TestCase):
  """Tier A CPU tests verifying pathways_load_impl mappings and backward-compatibility."""

  def test_load_impl_mappings_and_fallbacks(self):
    cases_evaluated = 0

    # 1. Default None preserves current NO_DISPATCHER when colocated_python_checkpointing=False
    ctx = checkpoint_context.build_context()
    self.assertEqual(ctx.pathways.checkpointing_impl, ocp_pathways.CheckpointingImpl.NO_DISPATCHER)
    cases_evaluated += 1

    # 2. Default None preserves current COLOCATED_PYTHON when colocated_python_checkpointing=True
    ctx = checkpoint_context.build_context(colocated_python_checkpointing=True)
    self.assertEqual(ctx.pathways.checkpointing_impl, ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON)
    cases_evaluated += 1

    # 3. Explicit "no_dispatcher" overrides colocated_python_checkpointing=True
    ctx = checkpoint_context.build_context(
        colocated_python_checkpointing=True, pathways_load_impl="no_dispatcher"
    )
    self.assertEqual(ctx.pathways.checkpointing_impl, ocp_pathways.CheckpointingImpl.NO_DISPATCHER)
    cases_evaluated += 1

    # 4. Explicit "persistence" maps to CheckpointingImpl.PERSISTENCE
    ctx = checkpoint_context.build_context(pathways_load_impl="persistence")
    self.assertEqual(ctx.pathways.checkpointing_impl, ocp_pathways.CheckpointingImpl.PERSISTENCE)
    cases_evaluated += 1

    # 5. Explicit "colocated_python" maps to CheckpointingImpl.COLOCATED_PYTHON
    ctx = checkpoint_context.build_context(pathways_load_impl="colocated_python")
    self.assertEqual(ctx.pathways.checkpointing_impl, ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON)
    cases_evaluated += 1

    # 6. Resolution through Orbax v1 registration under mocked Pathways backend
    with mock.patch(
        "orbax.checkpoint.experimental.v1._src.synchronization.multihost.is_pathways_backend",
        return_value=True,
    ):
      res_pers = registration.resolve_pathways_checkpointing_impl(
          checkpoint_context.build_context(pathways_load_impl="persistence")
      )
      self.assertEqual(res_pers, ocp_pathways.CheckpointingImpl.PERSISTENCE)
      res_colo = registration.resolve_pathways_checkpointing_impl(
          checkpoint_context.build_context(pathways_load_impl="colocated_python")
      )
      self.assertEqual(res_colo, ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON)
      cases_evaluated += 2

    # 7. Invalid string raises ValueError
    for bad_val in ("unknown", "", "PERSISTENCE", "colocated"):
      with self.assertRaises(ValueError):
        checkpoint_context.build_context(pathways_load_impl=bad_val)  # pyrefly: ignore[bad-argument-type]
      cases_evaluated += 1

    # Load-bearing execution counter: ensure every case was exercised
    self.assertEqual(cases_evaluated, 11)


if __name__ == "__main__":
  absltest.main()
