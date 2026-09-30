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

"""Unit tests for warm-start pathways_load_impl plumbing in common/checkpointing.py."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import absltest
from maxtext.common import checkpointing


class CheckpointingWarmStartLoadImplTest(unittest.TestCase):
  """Tier A CPU tests for warm-start Pathways load impl derivation and wiring."""

  def test_derive_pathways_load_impl_environment_and_config(self):
    cases_evaluated = 0

    # 1. Unset env -> returns None regardless of config
    with mock.patch.dict(os.environ, {}, clear=True):
      cfg = SimpleNamespace(pathways_checkpointing_impl="colocated_python")
      self.assertIsNone(checkpointing._derive_pathways_load_impl(cfg))
      self.assertIsNone(checkpointing._derive_pathways_load_impl(None))
      cases_evaluated += 2

    # 2. Env set to "0" -> returns None
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "0"}):
      cfg = SimpleNamespace(pathways_checkpointing_impl="colocated_python")
      self.assertIsNone(checkpointing._derive_pathways_load_impl(cfg))
      cases_evaluated += 1

    # 3. Env set to "1" with config -> returns configured impl
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      cfg_colo = SimpleNamespace(pathways_checkpointing_impl="colocated_python")
      self.assertEqual(checkpointing._derive_pathways_load_impl(cfg_colo), "colocated_python")

      cfg_pers = SimpleNamespace(pathways_checkpointing_impl="persistence")
      self.assertEqual(checkpointing._derive_pathways_load_impl(cfg_pers), "persistence")

      # Supports dict configs
      dict_cfg = {"pathways_checkpointing_impl": "colocated_python"}
      self.assertEqual(checkpointing._derive_pathways_load_impl(dict_cfg), "colocated_python")

      # Fallback to "persistence" when config is None or missing field
      self.assertEqual(checkpointing._derive_pathways_load_impl(None), "persistence")
      self.assertEqual(checkpointing._derive_pathways_load_impl(SimpleNamespace()), "persistence")
      cases_evaluated += 5

    self.assertEqual(cases_evaluated, 8)

  @mock.patch("maxtext.common.checkpoint_context.build_context")
  @mock.patch("orbax.checkpoint.v1.load")
  @mock.patch("maxtext.common.train_state_nnx.to_checkpoint_dict", return_value={})
  @mock.patch("maxtext.common.checkpointing._restored_linen_to_nnx", return_value="dummy_restored")
  def test_load_linen_checkpoint_passes_derived_impl(
      self, mock_restored, mock_to_dict, mock_ocp_load, mock_build_context
  ):
    # Case A: ENABLE_PATHWAYS_PERSISTENCE=1 -> routes config.pathways_checkpointing_impl
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      cfg = SimpleNamespace(pathways_checkpointing_impl="colocated_python", enable_diloco=False)
      res = checkpointing._load_linen_checkpoint_into_nnx(
          path="/dummy/ckpt",
          abstract_nnx_state=mock.MagicMock(),
          checkpoint_storage_concurrent_gb=96,
          use_ocdbt=False,
          use_zarr3=False,
          config=cfg,
      )
      self.assertEqual(res, "dummy_restored")
      mock_build_context.assert_called_with(
          use_ocdbt=False,
          use_zarr3=False,
          checkpoint_storage_concurrent_gb=96,
          partial_load=True,
          enable_single_replica_ckpt_restoring=False,
          pathways_load_impl="colocated_python",
      )

    # Case B: ENABLE_PATHWAYS_PERSISTENCE unset -> passes pathways_load_impl=None
    with mock.patch.dict(os.environ, {}, clear=True):
      cfg = SimpleNamespace(pathways_checkpointing_impl="colocated_python", enable_diloco=False)
      checkpointing._load_linen_checkpoint_into_nnx(
          path="/dummy/ckpt",
          abstract_nnx_state=mock.MagicMock(),
          checkpoint_storage_concurrent_gb=96,
          use_ocdbt=False,
          use_zarr3=False,
          config=cfg,
      )
      mock_build_context.assert_called_with(
          use_ocdbt=False,
          use_zarr3=False,
          checkpoint_storage_concurrent_gb=96,
          partial_load=True,
          enable_single_replica_ckpt_restoring=False,
          pathways_load_impl=None,
      )

  @mock.patch("maxtext.common.checkpoint_context.build_context")
  @mock.patch("orbax.checkpoint.v1.load", return_value="dummy_unboxed_state")
  def test_load_full_state_passes_derived_impl(self, mock_ocp_load, mock_build_context):
    with mock.patch.dict(os.environ, {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      cfg = SimpleNamespace(pathways_checkpointing_impl="persistence")
      dummy_abstract = mock.MagicMock()  # not nnx.State
      res = checkpointing._load_full_state_from_path(
          path="/dummy/full_state",
          abstract_unboxed_pre_state=dummy_abstract,
          checkpoint_conversion_fn=None,
          source_checkpoint_layout="orbax",
          checkpoint_storage_concurrent_gb=96,
          use_ocdbt=False,
          use_zarr3=False,
          maxtext_config=cfg,
      )
      self.assertEqual(res, "dummy_unboxed_state")
      mock_build_context.assert_called_with(
          use_ocdbt=False,
          use_zarr3=False,
          checkpoint_storage_concurrent_gb=96,
          checkpoint_layout=mock.ANY,
          enable_single_replica_ckpt_restoring=False,
          pathways_load_impl="persistence",
      )


if __name__ == "__main__":
  absltest.main()
