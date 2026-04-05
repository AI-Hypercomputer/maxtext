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

"""Unit tests for maxtext.eval.runner.server_manager._build_app."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_mock_output(generated_text="hello", prompt_token_ids=(1, 2, 3), generated_token_ids=(4, 5)):
  """Build a SimpleNamespace mimicking a vLLM RequestOutput object."""
  return SimpleNamespace(
      prompt_token_ids=list(prompt_token_ids),
      prompt_logprobs=None,
      outputs=[
          SimpleNamespace(
              text=generated_text,
              token_ids=list(generated_token_ids),
              logprobs=None,
              finish_reason="stop",
          )
      ],
  )


def _make_mock_llm(generated_text="hello", prompt_token_ids=(1, 2, 3), generated_token_ids=(4, 5)):
  """Return a mock vLLM LLM object whose generate() returns a single RequestOutput.

  The tokenizer returned by ``get_tokenizer()`` decodes each token ID to the
  string ``f"tok{tok_id}"``.
  """
  mock_output = _make_mock_output(
      generated_text=generated_text,
      prompt_token_ids=prompt_token_ids,
      generated_token_ids=generated_token_ids,
  )

  mock_tokenizer = MagicMock()
  mock_tokenizer.decode.side_effect = lambda ids: "".join(f"tok{i}" for i in ids)
  mock_tokenizer.apply_chat_template.return_value = "rendered_prompt"

  mock_llm = MagicMock()
  mock_llm.generate.return_value = [mock_output]
  mock_llm.get_tokenizer.return_value = mock_tokenizer
  return mock_llm


class TestBuildApp(unittest.TestCase):
  """Tests for the FastAPI app returned by _build_app(llm)."""

  def setUp(self):
    """Patch SamplingParams at the module level used by server_manager."""
    self.mock_llm = _make_mock_llm()
    self.mock_sampling_params_cls = MagicMock(return_value=MagicMock())

    # Patch at the import location used inside _build_app.
    self._sp_patcher = patch(
        "vllm.sampling_params.SamplingParams",
        self.mock_sampling_params_cls,
    )
    self._vllm_patcher = patch.dict(
        "sys.modules",
        {
            "vllm": MagicMock(),
            "vllm.sampling_params": MagicMock(SamplingParams=self.mock_sampling_params_cls),
        },
    )
    self._vllm_patcher.start()
    self._sp_patcher.start()

    from maxtext.eval.runner.server_manager import _build_app
    from starlette.testclient import TestClient

    self.app = _build_app(self.mock_llm)
    self.client = TestClient(self.app)

  def tearDown(self):
    self._sp_patcher.stop()
    self._vllm_patcher.stop()

  def test_health_endpoint(self):
    resp = self.client.get("/health")
    self.assertEqual(resp.status_code, 200)
    self.assertEqual(resp.json(), {"status": "ok"})

  def test_completions_basic(self):
    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "hi", "max_tokens": 10},
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertIn("choices", data)
    self.assertEqual(len(data["choices"]), 1)
    self.assertEqual(data["choices"][0]["text"], "hello")

  def test_completions_list_prompt(self):
    mock_llm = _make_mock_llm(generated_text="world")
    mock_llm.generate.return_value = [
        _make_mock_output(generated_text="alpha"),
        _make_mock_output(generated_text="beta"),
    ]
    mock_llm.get_tokenizer.return_value = self.mock_llm.get_tokenizer()

    from maxtext.eval.runner.server_manager import _build_app
    from starlette.testclient import TestClient

    app = _build_app(mock_llm)
    client = TestClient(app)

    resp = client.post(
        "/v1/completions",
        json={"model": "m", "prompt": ["first", "second"], "max_tokens": 5},
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(len(data["choices"]), 2)
    self.assertEqual(data["choices"][0]["text"], "alpha")
    self.assertEqual(data["choices"][1]["text"], "beta")

  def test_completions_no_logprobs(self):
    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "test", "max_tokens": 5},
    )
    data = resp.json()
    self.assertIsNone(data["choices"][0]["logprobs"])

  def test_completions_with_logprobs_echo_false(self):
    mock_output = _make_mock_output(
        generated_text="hi",
        prompt_token_ids=[1, 2],
        generated_token_ids=[4, 5],
    )
    mock_output.outputs[0].logprobs = [
        {4: SimpleNamespace(logprob=-0.5)},
        {5: SimpleNamespace(logprob=-1.2)},
    ]
    self.mock_llm.generate.return_value = [mock_output]

    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "ab", "max_tokens": 5, "logprobs": 1},
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    lp = data["choices"][0]["logprobs"]
    self.assertIsNotNone(lp)
    self.assertEqual(len(lp["tokens"]), 2)
    self.assertAlmostEqual(lp["token_logprobs"][0], -0.5, places=4)
    self.assertAlmostEqual(lp["token_logprobs"][1], -1.2, places=4)

  def test_completions_with_logprobs_echo_true(self):
    mock_output = _make_mock_output(
        generated_text=" world",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
    )
    mock_output.prompt_logprobs = [
        None,
        {2: SimpleNamespace(logprob=-0.3)},
        {3: SimpleNamespace(logprob=-0.7)},
    ]
    mock_output.outputs[0].logprobs = [
        {4: SimpleNamespace(logprob=-0.9)},
        {5: SimpleNamespace(logprob=-1.1)},
    ]
    self.mock_llm.generate.return_value = [mock_output]

    resp = self.client.post(
        "/v1/completions",
        json={
            "model": "m",
            "prompt": "tok1tok2tok3",
            "max_tokens": 5,
            "logprobs": 1,
            "echo": True,
        },
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    lp = data["choices"][0]["logprobs"]
    self.assertIsNotNone(lp)
    # echo=True → prompt tokens (3) + generated tokens (2) = 5 total.
    self.assertEqual(len(lp["tokens"]), 5)

  def test_chat_completions_basic(self):
    resp = self.client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 20,
        },
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertIn("choices", data)
    self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
    self.assertEqual(data["choices"][0]["message"]["content"], "hello")

  def test_chat_completions_applies_template(self):
    resp = self.client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 10,
        },
    )
    self.assertEqual(resp.status_code, 200)
    tokenizer = self.mock_llm.get_tokenizer()
    tokenizer.apply_chat_template.assert_called()
    call_args = tokenizer.apply_chat_template.call_args
    # The messages list should have been forwarded to apply_chat_template.
    passed_messages = call_args[0][0] if call_args[0] else call_args[1].get("conversation")
    self.assertIsNotNone(passed_messages)


if __name__ == "__main__":
  unittest.main()
