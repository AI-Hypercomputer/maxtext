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

"""Unit tests for maxtext.eval.runner.server_manager.VllmServerManager."""

import os
import unittest
from unittest import mock

from maxtext.eval.runner.server_manager import VllmServerManager


def _make_manager(**kwargs) -> VllmServerManager:
  defaults = dict(
      model_path="/fake/model",
      host="localhost",
      port=8000,
      tensor_parallel_size=4,
      max_model_len=4096,
  )
  defaults.update(kwargs)
  return VllmServerManager(**defaults)


def _start_capturing_llm_kwargs(mgr: VllmServerManager, rank: int = 0) -> dict:
  """Call mgr.start() with vLLM/uvicorn/JAX mocked; return kwargs passed to LLM(...)."""
  mock_llm_cls = mock.MagicMock()
  mock_vllm = mock.MagicMock()
  mock_vllm.LLM = mock_llm_cls
  mock_uvicorn = mock.MagicMock()

  with mock.patch.dict("sys.modules", {"vllm": mock_vllm, "uvicorn": mock_uvicorn}), \
       mock.patch("jax.process_index", return_value=rank), \
       mock.patch("threading.Thread", return_value=mock.MagicMock()), \
       mock.patch("maxtext.eval.runner.server_manager._build_app", return_value=mock.MagicMock()), \
       mock.patch.object(mgr, "_wait_until_healthy"):
    mgr.start()

  return mock_llm_cls.call_args.kwargs


class TestVllmServerManagerConfig(unittest.TestCase):
  """Tests for vLLM LLM constructor kwargs built by start()."""

  def test_required_vllm_kwargs(self):
    mgr = _make_manager(tensor_parallel_size=4, max_model_len=8192)
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertEqual(kwargs["model"], "/fake/model")
    self.assertEqual(kwargs["tensor_parallel_size"], 4)
    self.assertEqual(kwargs["max_model_len"], 8192)
    self.assertEqual(kwargs["device"], "tpu")

  def test_maxtext_adapter_mode_sets_hf_overrides(self):
    mgr = _make_manager(
        checkpoint_path="gs://bucket/run/0/items",
        maxtext_model_name="llama3.1-8b",
    )
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertIn("hf_overrides", kwargs)
    self.assertEqual(kwargs["hf_overrides"]["architectures"], ["MaxTextForCausalLM"])

  def test_maxtext_adapter_mode_sets_additional_config(self):
    mgr = _make_manager(
        checkpoint_path="gs://bucket/run/0/items",
        maxtext_model_name="llama3.1-8b",
    )
    kwargs = _start_capturing_llm_kwargs(mgr)
    add_cfg = kwargs["additional_config"]["maxtext_config"]
    self.assertEqual(add_cfg["load_parameters_path"], "gs://bucket/run/0/items")
    self.assertEqual(add_cfg["model_name"], "llama3.1-8b")

  def test_hf_mode_sets_load_format_auto(self):
    mgr = _make_manager()  # no checkpoint_path → HF mode
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertEqual(kwargs.get("load_format"), "auto")
    self.assertNotIn("hf_overrides", kwargs)
    self.assertNotIn("additional_config", kwargs)

  def test_max_num_batched_tokens_forwarded(self):
    mgr = _make_manager(max_num_batched_tokens=2048)
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertEqual(kwargs["max_num_batched_tokens"], 2048)

  def test_max_num_batched_tokens_omitted_when_none(self):
    mgr = _make_manager(max_num_batched_tokens=None)
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertNotIn("max_num_batched_tokens", kwargs)

  def test_max_num_seqs_forwarded(self):
    mgr = _make_manager(max_num_seqs=256)
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertEqual(kwargs["max_num_seqs"], 256)

  def test_max_num_seqs_omitted_when_none(self):
    mgr = _make_manager(max_num_seqs=None)
    kwargs = _start_capturing_llm_kwargs(mgr)
    self.assertNotIn("max_num_seqs", kwargs)

  def test_env_applied_to_os_environ_before_llm_init(self):
    mgr = _make_manager(env={"_TEST_EVAL_TOKEN": "abc123"})
    env_at_init = {}

    def capture_env(**kwargs):  # pylint: disable=unused-argument
      env_at_init.update(os.environ)
      return mock.MagicMock()

    mock_llm_cls = mock.MagicMock(side_effect=capture_env)
    mock_vllm = mock.MagicMock()
    mock_vllm.LLM = mock_llm_cls

    with mock.patch.dict("sys.modules", {"vllm": mock_vllm, "uvicorn": mock.MagicMock()}), \
         mock.patch("jax.process_index", return_value=0), \
         mock.patch("threading.Thread", return_value=mock.MagicMock()), \
         mock.patch("maxtext.eval.runner.server_manager._build_app", return_value=mock.MagicMock()), \
         mock.patch.object(mgr, "_wait_until_healthy"), \
         mock.patch.dict("os.environ", {}, clear=False):
      mgr.start()

    self.assertEqual(env_at_init.get("_TEST_EVAL_TOKEN"), "abc123")

  def test_missing_maxtext_model_name_raises(self):
    with self.assertRaises(ValueError):
      VllmServerManager(model_path="/fake/model", checkpoint_path="gs://bucket/0/items")


class TestVllmServerManagerHttp(unittest.TestCase):
  """Tests that the HTTP server is started only on rank-0."""

  def _start_capturing_thread_calls(self, mgr, rank):
    mock_llm_cls = mock.MagicMock()
    mock_vllm = mock.MagicMock()
    mock_vllm.LLM = mock_llm_cls
    mock_thread_cls = mock.MagicMock(return_value=mock.MagicMock())

    with mock.patch.dict("sys.modules", {"vllm": mock_vllm, "uvicorn": mock.MagicMock()}), \
         mock.patch("jax.process_index", return_value=rank), \
         mock.patch("threading.Thread", mock_thread_cls), \
         mock.patch("maxtext.eval.runner.server_manager._build_app", return_value=mock.MagicMock()), \
         mock.patch.object(mgr, "_wait_until_healthy"):
      mgr.start()

    return mock_thread_cls

  def test_rank0_starts_http_server_thread(self):
    mgr = _make_manager()
    mock_thread_cls = self._start_capturing_thread_calls(mgr, rank=0)
    mock_thread_cls.assert_called_once()
    _, kwargs = mock_thread_cls.call_args
    self.assertTrue(kwargs.get("daemon"))

  def test_non_rank0_does_not_start_http_server(self):
    mgr = _make_manager()
    mock_thread_cls = self._start_capturing_thread_calls(mgr, rank=1)
    mock_thread_cls.assert_not_called()


class TestVllmServerManagerLifecycle(unittest.TestCase):

  def test_stop_signals_uvicorn_should_exit(self):
    mgr = _make_manager()
    mock_server = mock.MagicMock()
    mock_thread = mock.MagicMock()
    mock_thread.is_alive.return_value = False
    mgr._uvicorn_server = mock_server
    mgr._server_thread = mock_thread
    with mock.patch("jax.process_index", return_value=0):
      mgr.stop()
    self.assertTrue(mock_server.should_exit)

  def test_stop_clears_references(self):
    mgr = _make_manager()
    mgr._llm = mock.MagicMock()
    mgr._uvicorn_server = mock.MagicMock()
    mgr._server_thread = mock.MagicMock()
    mgr._server_thread.is_alive.return_value = False
    with mock.patch("jax.process_index", return_value=0):
      mgr.stop()
    self.assertIsNone(mgr._llm)
    self.assertIsNone(mgr._uvicorn_server)
    self.assertIsNone(mgr._server_thread)

  def test_stop_is_noop_when_not_started(self):
    mgr = _make_manager()
    with mock.patch("jax.process_index", return_value=0):
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
