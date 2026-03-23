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

"""Unit tests for maxtext.eval.runner.server_manager."""

import signal
import subprocess
import unittest
from unittest import mock

from maxtext.eval.runner.server_manager import VllmServerManager


def _make_manager(**kwargs) -> VllmServerManager:
  defaults = dict(
      hf_model_path="/fake/model",
      host="localhost",
      port=8000,
      tensor_parallel_size=4,
      max_model_len=4096,
  )
  defaults.update(kwargs)
  return VllmServerManager(**defaults)


class TestVllmServerManagerCommand(unittest.TestCase):

  def _start_with_mock(self, mgr: VllmServerManager):
    """Start the manager with Popen and health-check mocked out."""
    with mock.patch("subprocess.Popen") as mock_popen, \
         mock.patch.object(mgr, "_wait_until_healthy"):
      mock_proc = mock.MagicMock()
      mock_proc.poll.return_value = None
      mock_popen.return_value = mock_proc
      mgr.start()
      return mock_popen.call_args[0][0]  # the cmd list

  def test_required_args_present(self):
    mgr = _make_manager()
    cmd = self._start_with_mock(mgr)
    self.assertIn("vllm.entrypoints.openai.api_server", " ".join(cmd))
    self.assertIn("--model", cmd)
    self.assertIn("/fake/model", cmd)
    self.assertIn("--tensor-parallel-size", cmd)
    self.assertIn("4", cmd)
    self.assertIn("--device", cmd)
    self.assertIn("tpu", cmd)

  def test_max_num_batched_tokens_included_when_set(self):
    mgr = _make_manager(max_num_batched_tokens=2048)
    cmd = self._start_with_mock(mgr)
    self.assertIn("--max-num-batched-tokens", cmd)
    idx = cmd.index("--max-num-batched-tokens")
    self.assertEqual(cmd[idx + 1], "2048")

  def test_max_num_batched_tokens_omitted_when_none(self):
    mgr = _make_manager(max_num_batched_tokens=None)
    cmd = self._start_with_mock(mgr)
    self.assertNotIn("--max-num-batched-tokens", cmd)

  def test_max_num_seqs_included_when_set(self):
    mgr = _make_manager(max_num_seqs=256)
    cmd = self._start_with_mock(mgr)
    self.assertIn("--max-num-seqs", cmd)
    idx = cmd.index("--max-num-seqs")
    self.assertEqual(cmd[idx + 1], "256")

  def test_max_num_seqs_omitted_when_none(self):
    mgr = _make_manager(max_num_seqs=None)
    cmd = self._start_with_mock(mgr)
    self.assertNotIn("--max-num-seqs", cmd)

  def test_extra_vllm_args_appended(self):
    mgr = _make_manager(extra_vllm_args=["--gpu-memory-utilization", "0.9"])
    cmd = self._start_with_mock(mgr)
    self.assertIn("--gpu-memory-utilization", cmd)
    self.assertIn("0.9", cmd)

  def test_hf_token_forwarded_in_env(self):
    mgr = _make_manager(env={"HF_TOKEN": "hf_test123"})
    with mock.patch("subprocess.Popen") as mock_popen, \
         mock.patch.object(mgr, "_wait_until_healthy"):
      mock_proc = mock.MagicMock()
      mock_proc.poll.return_value = None
      mock_popen.return_value = mock_proc
      mgr.start()
      _, kwargs = mock_popen.call_args
      self.assertEqual(kwargs["env"]["HF_TOKEN"], "hf_test123")

  def test_env_merges_with_os_environ(self):
    mgr = _make_manager(env={"MY_VAR": "value"})
    with mock.patch("subprocess.Popen") as mock_popen, \
         mock.patch.object(mgr, "_wait_until_healthy"), \
         mock.patch.dict("os.environ", {"EXISTING": "yes"}):
      mock_proc = mock.MagicMock()
      mock_proc.poll.return_value = None
      mock_popen.return_value = mock_proc
      mgr.start()
      _, kwargs = mock_popen.call_args
      self.assertIn("EXISTING", kwargs["env"])
      self.assertIn("MY_VAR", kwargs["env"])


class TestVllmServerManagerLifecycle(unittest.TestCase):

  def test_stop_sends_sigterm(self):
    mgr = _make_manager()
    mock_proc = mock.MagicMock()
    mock_proc.poll.return_value = None
    mgr._proc = mock_proc
    mgr.stop()
    mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)

  def test_stop_is_noop_when_not_started(self):
    mgr = _make_manager()
    mgr.stop()  # should not raise

  def test_stop_called_on_context_exit(self):
    mgr = _make_manager()
    with mock.patch.object(mgr, "start"), mock.patch.object(mgr, "stop") as mock_stop:
      with mgr:
        pass
    mock_stop.assert_called_once()

  def test_stop_called_on_exception_in_context(self):
    mgr = _make_manager()
    with mock.patch.object(mgr, "start"), mock.patch.object(mgr, "stop") as mock_stop:
      try:
        with mgr:
          raise RuntimeError("boom")
      except RuntimeError:
        pass
    mock_stop.assert_called_once()

  def test_base_url(self):
    mgr = _make_manager(host="0.0.0.0", port=9000)
    self.assertEqual(mgr.base_url, "http://0.0.0.0:9000")


if __name__ == "__main__":
  unittest.main()
