# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for maxengine_server."""

import os
import unittest
from unittest import mock
import pytest

pytest.importorskip("jetstream.core.server_lib", reason="jetstream.core.server_lib not fully installed")

import jetstream.core.server_lib  # pylint: disable=unused-import

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine_server
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


@pytest.mark.cpu_only
class MaxEngineServerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize basic test configurations

    self.base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    self.config = pyconfig.initialize(
        ["", self.base_config_path],
        run_name="test_server",
        enable_checkpointing=False,
    )

  @mock.patch("maxtext.inference.maxengine.maxengine_server.gcloud_stub")
  @mock.patch("maxtext.inference.maxengine.maxengine_server.maxengine_config")
  @mock.patch("jetstream.core.server_lib")
  @mock.patch("grpc.insecure_server_credentials")
  def test_main_insecure_default(self, mock_insecure_creds, mock_server_lib, mock_maxengine_config, mock_gcloud_stub):
    mock_config_lib = mock.MagicMock()
    mock_gcloud_stub.jetstream.return_value = (
        mock_config_lib,
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )
    mock_gcloud_stub.is_decoupled.return_value = False

    maxengine_server.main(self.config)
    # Verify insecure credentials were instantiated and run
    mock_insecure_creds.assert_called_once()
    mock_server_lib.run.assert_called_once()
    kwargs = mock_server_lib.run.call_args[1]
    self.assertEqual(kwargs["credentials"], mock_insecure_creds.return_value)

  @mock.patch("maxtext.inference.maxengine.maxengine_server.gcloud_stub")
  @mock.patch("maxtext.inference.maxengine.maxengine_server.maxengine_config")
  @mock.patch("jetstream.core.server_lib")
  @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"test_key_cert_data")
  @mock.patch("grpc.ssl_server_credentials")
  def test_main_secure_tls(self, mock_ssl_creds, mock_open, mock_server_lib, mock_maxengine_config, mock_gcloud_stub):
    # Re-initialize config with secure TLS enabled
    self.config = pyconfig.initialize(
        ["", self.base_config_path],
        run_name="test_server",
        enable_checkpointing=False,
        grpc_tls_certificate_path="/path/to/cert",
        grpc_tls_private_key_path="/path/to/key",
    )

    mock_config_lib = mock.MagicMock()
    mock_gcloud_stub.jetstream.return_value = (
        mock_config_lib,
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )
    mock_gcloud_stub.is_decoupled.return_value = False

    maxengine_server.main(self.config)
    # Verify cert and private key files were read
    mock_open.assert_has_calls(
        [
            mock.call("/path/to/key", "rb"),
            mock.call("/path/to/cert", "rb"),
        ],
        any_order=True,
    )
    # Verify SSL credentials were constructed and provided to the server run method
    mock_ssl_creds.assert_called_once_with([(b"test_key_cert_data", b"test_key_cert_data")])
    mock_server_lib.run.assert_called_once()
    kwargs = mock_server_lib.run.call_args[1]
    self.assertEqual(kwargs["credentials"], mock_ssl_creds.return_value)

  def test_main_partially_configured_tls_raises_error(self):
    # 1. Missing private key
    self.config = pyconfig.initialize(
        ["", self.base_config_path],
        run_name="test_server",
        enable_checkpointing=False,
        grpc_tls_certificate_path="/path/to/cert",
        grpc_tls_private_key_path="",
    )
    with self.assertRaises(ValueError) as ctx:
      maxengine_server.main(self.config)
    self.assertIn(
        "Both 'grpc_tls_certificate_path' and 'grpc_tls_private_key_path' must be provided",
        str(ctx.exception),
    )

    # 2. Missing certificate
    self.config = pyconfig.initialize(
        ["", self.base_config_path],
        run_name="test_server",
        enable_checkpointing=False,
        grpc_tls_certificate_path="",
        grpc_tls_private_key_path="/path/to/key",
    )
    with self.assertRaises(ValueError) as ctx:
      maxengine_server.main(self.config)
    self.assertIn(
        "Both 'grpc_tls_certificate_path' and 'grpc_tls_private_key_path' must be provided",
        str(ctx.exception),
    )


if __name__ == "__main__":
  unittest.main()
