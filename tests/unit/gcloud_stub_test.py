# Copyright 2025 Google LLC
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

"""Decoupling unit tests for MaxText GCloud stubs.

These tests are written to pass whether optional deps (JetStream, cloud_tpu_diagnostics)
are installed or not, and they focus only on decoupling behavior.
"""

import importlib
import os
import unittest
from unittest import mock

import pytest

from maxtext.common import gcloud_stub
from maxtext.utils import gcs_utils


@pytest.mark.cpu_only
class GCloudStubTest(unittest.TestCase):
  # pylint: disable=protected-access

  def test_is_decoupled_parsing(self):
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "true"}):
      self.assertTrue(gcloud_stub.is_decoupled())
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertFalse(gcloud_stub.is_decoupled())
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "FALSE"}):
      self.assertFalse(gcloud_stub.is_decoupled())

  def test_gcs_storage_is_stub_when_decoupled(self):
    # gcs_storage() explicitly prefers stubs when decoupled.
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      storage = gcloud_stub.gcs_storage()
      self.assertTrue(hasattr(storage, "Client"))
      self.assertTrue(getattr(storage, "_IS_STUB", False))

  def test_jetstream_contract_in_decoupled_mode(self):
    """In decoupled mode, jetstream() returns 5 objects with expected API.

    They may be real modules (_IS_STUB=False) or stubs (_IS_STUB=True),
    depending on whether JetStream is installed.
    """
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      config_lib, engine_api, token_utils, tokenizer_api, token_params_ns = gcloud_stub.jetstream()

      self.assertIsInstance(getattr(config_lib, "_IS_STUB", None), bool)
      self.assertIsInstance(getattr(engine_api, "_IS_STUB", None), bool)
      self.assertIsInstance(getattr(token_utils, "_IS_STUB", None), bool)
      self.assertIsInstance(getattr(tokenizer_api, "_IS_STUB", None), bool)
      self.assertIsInstance(getattr(token_params_ns, "_IS_STUB", None), bool)

      self.assertTrue(hasattr(engine_api, "Engine"))
      self.assertTrue(hasattr(engine_api, "ResultTokens"))

  def test_jetstream_returns_stubs_when_deps_missing_and_decoupled(self):
    """Force JetStream lookup to fail -> stubs returned in decoupled mode."""
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      with mock.patch("maxtext.common.gcloud_stub.importlib.util.find_spec", return_value=None):
        config_lib, engine_api, token_utils, tokenizer_api, token_params_ns = gcloud_stub.jetstream()

        self.assertTrue(getattr(config_lib, "_IS_STUB", False))
        self.assertTrue(getattr(engine_api, "_IS_STUB", False))
        self.assertTrue(getattr(token_utils, "_IS_STUB", False))
        self.assertTrue(getattr(tokenizer_api, "_IS_STUB", False))
        self.assertTrue(getattr(token_params_ns, "_IS_STUB", False))

        self.assertTrue(hasattr(engine_api, "Engine"))
        self.assertTrue(hasattr(engine_api, "ResultTokens"))

  def test_cloud_diagnostics_contract_in_decoupled_mode(self):
    """cloud_diagnostics() returns 4-tuple; content can be real or stub."""
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      diag, debug_cfg, diag_cfg, stack_cfg = gcloud_stub.cloud_diagnostics()
      self.assertIsNotNone(diag)
      self.assertIsNotNone(debug_cfg)
      self.assertIsNotNone(diag_cfg)
      self.assertIsNotNone(stack_cfg)

  def test_cloud_diagnostics_returns_stub_object_when_missing_and_decoupled(self):
    """Force stub branch -> diag is stub object with .run()."""
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      with mock.patch("maxtext.common.gcloud_stub._import_or_stub") as _ios:
        _ios.side_effect = lambda import_fn, stub_fn, **kwargs: stub_fn()
        diag, debug_cfg, diag_cfg, stack_cfg = gcloud_stub.cloud_diagnostics()

      self.assertTrue(hasattr(diag, "run"))
      self.assertIsNotNone(debug_cfg)
      self.assertIsNotNone(diag_cfg)
      self.assertIsNotNone(stack_cfg)

  def test_monitoring_modules_returns_stub_tuple_when_decoupled_and_missing(self):
    # Force stub path regardless of installed deps.
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      with mock.patch("maxtext.common.gcloud_stub._import_or_stub") as _ios:
        _ios.side_effect = lambda import_fn, stub_fn, **kwargs: stub_fn()
        monitoring_v3, metric_pb2, monitored_resource_pb2, google_api_error, is_stub = gcloud_stub.monitoring_modules()
      self.assertTrue(is_stub)
      self.assertIsNotNone(monitoring_v3)
      self.assertIsNotNone(metric_pb2)
      self.assertIsNotNone(monitored_resource_pb2)
      self.assertIsNotNone(google_api_error)

  def test_goodput_modules_returns_stub_tuple_when_decoupled_and_missing(self):
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      with mock.patch("maxtext.common.gcloud_stub._import_or_stub") as _ios:
        _ios.side_effect = lambda import_fn, stub_fn, **kwargs: stub_fn()
        goodput, monitoring, is_stub = gcloud_stub.goodput_modules()
      self.assertTrue(is_stub)
      self.assertIsNotNone(goodput)
      self.assertIsNotNone(monitoring)

  def test_vertex_tensorboard_modules_returns_stub_tuple_when_decoupled(self):
    # vertex_tensorboard_modules uses stub_if_decoupled policy: should always be stubbed.
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      manager, is_stub = gcloud_stub.vertex_tensorboard_modules()
      self.assertTrue(is_stub)
      self.assertIsNotNone(manager)

  def test_vertex_tensorboard_components_alias(self):
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      self.assertIsNotNone(gcloud_stub.vertex_tensorboard_components())
      self.assertIsNotNone(gcloud_stub.vertex_tensorboard_modules())

  # ---- Decoupling call-site tests ----

  def test_gcs_utils_guard_is_noop_when_decoupled_and_stub(self):
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      with mock.patch.object(gcs_utils, "storage") as mock_storage:
        mock_storage._IS_STUB = True
        self.assertFalse(gcs_utils._gcs_guard("unit-test"))

  def test_gcs_utils_guard_raises_when_not_decoupled_and_stub(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(gcs_utils, "storage") as mock_storage:
        mock_storage._IS_STUB = True
        with self.assertRaises(RuntimeError):
          gcs_utils._gcs_guard("unit-test")

  def test_maxengine_config_create_exp_maxengine_signature_decoupled(self):
    # Import lazily under decoupled mode (safe even without JetStream installed).
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      maxengine_config = importlib.import_module("maxtext.inference.maxengine.maxengine_config")
      importlib.reload(maxengine_config)

      mock_devices = mock.MagicMock()
      mock_config = mock.MagicMock()

      with mock.patch("maxtext.inference.maxengine.maxengine.MaxEngine") as mock_engine:
        maxengine_config.create_exp_maxengine(mock_devices, mock_config)
        mock_engine.assert_called_once_with(mock_config)

  def test_maxengine_config_create_exp_maxengine_signature_not_decoupled(self):
    # Import safely (under decoupled) then flip behavior only for the call.
    with mock.patch.dict(os.environ, {"DECOUPLE_GCLOUD": "TRUE"}):
      maxengine_config = importlib.import_module("maxtext.inference.maxengine.maxengine_config")
      importlib.reload(maxengine_config)

    with mock.patch.object(maxengine_config, "is_decoupled", return_value=False):
      mock_devices = mock.MagicMock()
      mock_config = mock.MagicMock()
      with mock.patch("maxtext.inference.maxengine.maxengine.MaxEngine") as mock_engine:
        maxengine_config.create_exp_maxengine(mock_devices, mock_config)
        mock_engine.assert_called_once_with(config=mock_config, devices=mock_devices)


if __name__ == "__main__":
  unittest.main()
