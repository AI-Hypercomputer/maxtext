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

"""Unit tests for Elastic Training utility functions."""

import unittest
import pytest

from maxtext.utils import elastic_utils
from maxtext.utils import gcs_utils
import pathwaysutils


class FakeGcsUtils:
  """Fake implementation for gcs_utils functions."""

  def __init__(self):
    self.directories = {}
    self.files = set()
    self.deleted_directories = []

  def gcs_list_directories(self, path):
    if path in self.directories:
      return self.directories[path]
    if path.endswith("/") and path[:-1] in self.directories:
      return self.directories[path[:-1]]
    return []

  def gcs_glob_pattern(self, pattern):
    # Very simple glob implementation for testing
    prefix = pattern.replace("*", "")
    return [f for f in self.files if f.startswith(prefix)]

  def gcs_delete_directory(self, path):
    self.deleted_directories.append(path)

  @staticmethod
  def add_trailing_slash(path):
    return gcs_utils.add_trailing_slash(path)

  @staticmethod
  def parse_gcs_bucket_and_prefix(path):
    return gcs_utils.parse_gcs_bucket_and_prefix(path)


class FakeManager:
  """Fake implementation for pathwaysutils.elastic.manager.Manager."""

  def __init__(self):
    self.active_slice_indices = set()
    self.elastic_retry_called = False
    self.elastic_retry_kwargs = {}

  def elastic_retry(self, **kwargs):
    self.elastic_retry_called = True
    self.elastic_retry_kwargs = kwargs


class FakePathwaysUtils:
  """Fake implementation for pathwaysutils."""

  def __init__(self):
    self.is_pathways_used = True

  def is_pathways_backend_used(self):
    return self.is_pathways_used


class FakeLogging:
  """Fake implementation for max_logging."""

  def __init__(self):
    self.logs = []

  def log(self, message):
    self.logs.append(message)


class FakeJax:
  """Fake implementation for jax."""

  def __init__(self):
    self.devices_list = []

  def devices(self, *args, **kwargs):
    return self.devices_list


class FakeDevice:
  """Fake Device object."""

  def __init__(self, slice_index=0):
    self.slice_index = slice_index


class FakeConfig:
  """Fake configuration object."""

  def __init__(self):
    self.elastic_enabled = True
    self.checkpoint_dir = "gs://test_bucket/checkpoints"
    self.elastic_max_retries = 3
    self.elastic_timeout_seconds = 100


@pytest.mark.cpu_only
class ElasticUtilsTest(unittest.TestCase):
  """Unit tests for Elastic Training utility functions."""

  def setUp(self):
    super().setUp()
    # Save original dependencies
    self.original_pathwaysutils = elastic_utils.pathwaysutils
    self.original_jax = elastic_utils.jax
    self.original_gcs_utils = elastic_utils.gcs_utils
    self.original_max_logging = elastic_utils.max_logging
    self.original_manager_class = pathwaysutils.elastic.manager.Manager

    # Initialize fakes
    self.fake_gcs_utils = FakeGcsUtils()
    self.fake_pathwaysutils = FakePathwaysUtils()
    self.fake_logging = FakeLogging()
    self.fake_jax = FakeJax()
    self.fake_manager = FakeManager()

    # Inject fakes into elastic_utils namespace
    elastic_utils.pathwaysutils = self.fake_pathwaysutils
    elastic_utils.jax = self.fake_jax
    elastic_utils.gcs_utils = self.fake_gcs_utils
    elastic_utils.max_logging = self.fake_logging

    # Hook up pathwaysutils.elastic.manager.Manager to return our fake_manager
    pathwaysutils.elastic.manager.Manager = lambda *args, **kwargs: self.fake_manager

    # Reset global state for testing
    elastic_utils.elastic_manager = None

  def tearDown(self):
    # Restore original dependencies
    elastic_utils.pathwaysutils = self.original_pathwaysutils
    elastic_utils.jax = self.original_jax
    elastic_utils.gcs_utils = self.original_gcs_utils
    elastic_utils.max_logging = self.original_max_logging
    pathwaysutils.elastic.manager.Manager = self.original_manager_class
    super().tearDown()

  def test_elastic_enabled(self):
    """Tests elastic_enabled."""
    config = FakeConfig()
    self.fake_pathwaysutils.is_pathways_used = True
    config.elastic_enabled = True
    self.assertTrue(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = False
    self.assertFalse(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = True
    self.fake_pathwaysutils.is_pathways_used = False
    self.assertFalse(elastic_utils.elastic_enabled(config))

  def test_clean_up_checkpoints_no_checkpoints(self):
    """Tests clean_up_checkpoints when no checkpoints exist."""
    self.fake_gcs_utils.directories = {"gs://test_bucket/checkpoints": []}
    elastic_utils.clean_up_checkpoints("gs://test_bucket/checkpoints")
    self.assertEqual(len(self.fake_gcs_utils.deleted_directories), 0)

  def test_clean_up_checkpoints_incomplete(self):
    """Tests clean_up_checkpoints when the latest checkpoint is incomplete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.directories = {checkpoint_dir: ["1", "2", "10"]}
    # No commit_success for "10"
    elastic_utils.clean_up_checkpoints(checkpoint_dir)
    self.assertIn(f"{checkpoint_dir}/10/", self.fake_gcs_utils.deleted_directories)
    self.assertNotIn(f"{checkpoint_dir}/1/", self.fake_gcs_utils.deleted_directories)
    self.assertNotIn(f"{checkpoint_dir}/2/", self.fake_gcs_utils.deleted_directories)

  def test_clean_up_checkpoints_complete(self):
    """Tests clean_up_checkpoints when the latest checkpoint is complete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.directories = {checkpoint_dir: ["1", "2", "10"]}
    self.fake_gcs_utils.files.add(f"{checkpoint_dir}/10/commit_success_0")
    elastic_utils.clean_up_checkpoints(checkpoint_dir)
    self.assertEqual(len(self.fake_gcs_utils.deleted_directories), 0)

  def test_live_devices_no_pathways(self):
    """Tests live_devices when pathways is not used."""
    self.fake_pathwaysutils.is_pathways_used = False
    device0 = FakeDevice(slice_index=0)
    self.fake_jax.devices_list = [device0]

    devices = elastic_utils.live_devices()
    self.assertEqual(devices, [device0])

  def test_elastic_retry_disabled(self):
    """Tests elastic_retry when disabled but pathways is used."""
    self.fake_pathwaysutils.is_pathways_used = True
    config = FakeConfig()
    config.elastic_enabled = False
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled: False, pathways"
        " backend used: True"
    )
    with self.assertRaisesRegex(ValueError, msg):
      elastic_utils.elastic_retry(config)

  def test_elastic_retry_no_pathways(self):
    """Tests elastic_retry when enabled but pathways is not used."""
    self.fake_pathwaysutils.is_pathways_used = False
    config = FakeConfig()
    config.elastic_enabled = True
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled: True, pathways"
        " backend used: False"
    )
    with self.assertRaisesRegex(ValueError, msg):
      elastic_utils.elastic_retry(config)

  def test_chain_callbacks(self):
    """Tests chain_callbacks."""
    # Test with no functions
    chained_fn_empty = elastic_utils.chain_callbacks()
    chained_fn_empty()  # Should not fail

    # Test with multiple functions
    call_order = []

    def fn1():
      call_order.append(1)

    def fn2():
      call_order.append(2)

    chained_fn = elastic_utils.chain_callbacks(fn1, fn2)
    chained_fn()
    self.assertEqual(call_order, [1, 2])


if __name__ == "__main__":
  unittest.main()
