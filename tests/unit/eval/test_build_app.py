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

"""Unit tests for eval server_manager.build_app."""

from __future__ import annotations

import asyncio
import sys
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
  """Return a mock vLLM LLM object whose generate() returns a single RequestOutput."""
  mock_output = _make_mock_output(
      generated_text=generated_text,
      prompt_token_ids=prompt_token_ids,
      generated_token_ids=generated_token_ids,
  )

  mock_tokenizer = MagicMock()
  mock_tokenizer.decode.side_effect = lambda ids: "".join(f"tok{i}" for i in ids)
  mock_tokenizer.apply_chat_template.return_value = [101, 102, 103]

  mock_llm = MagicMock()
  mock_llm.generate.return_value = [mock_output]
  mock_llm.get_tokenizer.return_value = mock_tokenizer
  return mock_llm


class TestBuildApp(unittest.TestCase):
  """Tests for the FastAPI returned by _build_app(llm)."""

  def setUp(self):
    """Patch SamplingParams at the module level used by server_manager."""
    self.mock_llm = _make_mock_llm()
    self.mock_sampling_params_cls = MagicMock(return_value=MagicMock())

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

    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

    self.app = _build_app(self.mock_llm)
    self.client = TestClient(self.app)

  def tearDown(self):
    self._sp_patcher.stop()
    self._vllm_patcher.stop()

  def test_health_endpoint(self):
    resp = self.client.get("/health")
    self.assertEqual(resp.status_code, 200)
    self.assertEqual(resp.json(), {"status": "ok"})

  def test_harmony_prompt_helper_uses_vllm_harmony_utilities(self):
    from maxtext.eval.runner.server_manager import _render_harmony_chat_prompt  # pylint: disable=import-outside-toplevel

    harmony_utils = MagicMock()
    harmony_utils.get_system_message.return_value = "canonical-system"
    harmony_utils.parse_chat_inputs_to_harmony_messages.return_value = ["user-message"]
    harmony_utils.render_for_completion.return_value = [1, 2, 3]
    messages = [{"role": "user", "content": "question"}]
    with patch.dict(
        sys.modules,
        {"vllm.entrypoints.openai.parser.harmony_utils": harmony_utils},
    ):
      token_ids = _render_harmony_chat_prompt(messages, "high")

    self.assertEqual(token_ids, [1, 2, 3])
    harmony_utils.get_system_message.assert_called_once_with(
        reasoning_effort="high",
        browser_description=None,
        python_description=None,
        with_custom_tools=True,
    )
    harmony_utils.parse_chat_inputs_to_harmony_messages.assert_called_once_with(messages)
    harmony_utils.render_for_completion.assert_called_once_with(["canonical-system", "user-message"])

  def test_harmony_stop_ids_use_older_vllm_helper_when_available(self):
    from maxtext.eval.runner.server_manager import _get_harmony_stop_token_ids  # pylint: disable=import-outside-toplevel

    harmony_utils = SimpleNamespace(get_stop_tokens_for_assistant_actions=lambda: [99, 100])
    with patch.dict(sys.modules, {"vllm.entrypoints.openai.parser.harmony_utils": harmony_utils}):
      self.assertEqual(_get_harmony_stop_token_ids(), [99, 100])

  def test_harmony_stop_ids_are_unset_for_newer_vllm_api(self):
    from maxtext.eval.runner.server_manager import _get_harmony_stop_token_ids  # pylint: disable=import-outside-toplevel

    # vLLM #44009 removed both the helper and its OpenAI frontend override.
    harmony_utils = SimpleNamespace()
    with patch.dict(sys.modules, {"vllm.entrypoints.openai.parser.harmony_utils": harmony_utils}):
      self.assertEqual(_get_harmony_stop_token_ids(), [])

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

    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

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
    passed_messages = call_args[0][0] if call_args[0] else call_args[1].get("conversation")
    self.assertIsNotNone(passed_messages)
    self.assertTrue(call_args.kwargs["tokenize"])
    generated_prompts = self.mock_llm.generate.call_args[0][0]
    self.assertEqual(generated_prompts, [{"prompt_token_ids": [101, 102, 103]}])

  def test_harmony_chat_uses_vllm_rendering_and_returns_only_final_content(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

    mock_llm = _make_mock_llm(
        generated_text="<raw harmony output>",
        prompt_token_ids=[10, 11],
        generated_token_ids=[21, 22, 23],
    )
    mock_llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="gpt_oss"),
        max_model_len=128,
    )
    self.mock_sampling_params_cls.reset_mock()
    with (
        patch("maxtext.eval.runner.server_manager._get_harmony_stop_token_ids", return_value=[99, 100]),
        patch("maxtext.eval.runner.server_manager._render_harmony_chat_prompt", return_value=[10, 11]) as render,
        patch(
            "maxtext.eval.runner.server_manager._parse_harmony_chat_output",
            return_value=("private reasoning", "Answer: 42", False),
        ) as parse,
    ):
      client = TestClient(_build_app(mock_llm))
      response = client.post(
          "/v1/chat/completions",
          json={
              "model": "gpt-oss-20b",
              "messages": [{"role": "user", "content": "question"}],
              "max_tokens": 20,
              "reasoning_effort": "high",
              "include_raw_output": True,
          },
      )

    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["choices"][0]["message"]["content"], "Answer: 42")
    self.assertEqual(data["choices"][0]["message"]["reasoning"], "private reasoning")
    self.assertEqual(data["diagnostics"]["raw_output"], "<raw harmony output>")
    self.assertNotIn("<raw harmony output>", data["choices"][0]["message"]["content"])
    render.assert_called_once_with([{"role": "user", "content": "question"}], "high")
    parse.assert_called_once_with([21, 22, 23])
    mock_llm.get_tokenizer.assert_not_called()
    sampling_kwargs = self.mock_sampling_params_cls.call_args.kwargs
    self.assertEqual(sampling_kwargs["stop_token_ids"], [99, 100])

  def test_harmony_reasoning_and_raw_output_are_opt_in(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

    mock_llm = _make_mock_llm(generated_text="raw", generated_token_ids=[21])
    mock_llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="gpt_oss"),
        max_model_len=128,
    )
    self.mock_sampling_params_cls.reset_mock()
    with (
        patch("maxtext.eval.runner.server_manager._get_harmony_stop_token_ids", return_value=[]),
        patch("maxtext.eval.runner.server_manager._render_harmony_chat_prompt", return_value=[10]),
        patch(
            "maxtext.eval.runner.server_manager._parse_harmony_chat_output",
            return_value=("private", "final", False),
        ),
    ):
      response = TestClient(_build_app(mock_llm)).post(
          "/v1/chat/completions",
          json={
              "model": "gpt-oss-20b",
              "messages": [{"role": "user", "content": "question"}],
              "include_reasoning": False,
          },
      )

    message = response.json()["choices"][0]["message"]
    self.assertEqual(message["content"], "final")
    self.assertNotIn("reasoning", message)
    self.assertNotIn("diagnostics", response.json())
    self.assertNotIn("stop_token_ids", self.mock_sampling_params_cls.call_args.kwargs)

  def test_chat_context_budget_caps_max_tokens(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

    mock_llm = _make_mock_llm(prompt_token_ids=[1, 2, 3])
    mock_llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="other"),
        max_model_len=5,
    )
    self.mock_sampling_params_cls.reset_mock()
    response = TestClient(_build_app(mock_llm)).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 10},
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(self.mock_sampling_params_cls.call_args.kwargs["max_tokens"], 2)

  def test_chat_template_error_is_a_bad_request_without_silent_fallback(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel
    from starlette.testclient import TestClient  # pylint: disable=import-outside-toplevel

    mock_llm = _make_mock_llm()
    mock_llm.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    mock_llm.get_tokenizer.return_value.apply_chat_template.side_effect = TypeError("bad template")
    response = TestClient(_build_app(mock_llm)).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "reasoning_effort": "high",
        },
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn("bad template", response.json()["detail"])
    mock_llm.generate.assert_not_called()

  def test_completions_token_id_prompt_normalised_as_single(self):
    """A list-of-ints prompt must be treated as one token-ID prompt, not a batch of ints."""
    mock_output = _make_mock_output(generated_text="out", prompt_token_ids=[1, 2, 3], generated_token_ids=[7])
    self.mock_llm.generate.return_value = [mock_output]

    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": [1, 2, 3], "max_tokens": 5},
    )
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(len(data["choices"]), 1)
    self.assertEqual(data["choices"][0]["text"], "out")
    # generate() must be called with a single-element list containing the token-ID list
    call_prompts = self.mock_llm.generate.call_args[0][0]
    self.assertEqual(call_prompts, [[1, 2, 3]])

  def test_completions_top_logprobs_populated(self):
    """When logprobs is requested, top_logprobs entries must be non-None dicts."""
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    mock_output = _make_mock_output(generated_text="hi", prompt_token_ids=[1], generated_token_ids=[4, 5])
    mock_output.outputs[0].logprobs = [
        {4: SimpleNamespace(logprob=-0.5), 6: SimpleNamespace(logprob=-1.0)},
        {5: SimpleNamespace(logprob=-0.8)},
    ]
    self.mock_llm.generate.return_value = [mock_output]

    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": "x", "max_tokens": 5, "logprobs": 2},
    )
    self.assertEqual(resp.status_code, 200)
    lp = resp.json()["choices"][0]["logprobs"]
    self.assertIsNotNone(lp)
    self.assertIsInstance(lp["top_logprobs"][0], dict)
    self.assertGreater(len(lp["top_logprobs"][0]), 0)

  def test_completions_echo_with_token_id_prompt(self):
    """echo=True with a token-ID prompt must decode the prompt and prepend it to text."""
    mock_output = _make_mock_output(generated_text=" world", prompt_token_ids=[1, 2, 3], generated_token_ids=[4])
    self.mock_llm.generate.return_value = [mock_output]

    resp = self.client.post(
        "/v1/completions",
        json={"model": "m", "prompt": [1, 2, 3], "max_tokens": 5, "echo": True},
    )
    self.assertEqual(resp.status_code, 200)
    text = resp.json()["choices"][0]["text"]
    # tokenizer.decode([1, 2, 3]) → "tok1tok2tok3" (see _make_mock_llm side_effect)
    self.assertTrue(text.startswith("tok1tok2tok3"), f"Expected decoded prompt prefix, got: {text!r}")
    self.assertIn(" world", text)


class TestChatCompletionsBatching(unittest.IsolatedAsyncioTestCase):
  """Tests for _ChatBatchQueue: concurrent /v1/chat/completions requests must
  coalesce into one llm.generate() call instead of serializing one-at-a-time
  (the bottleneck that caused the surrounding server to only ever process a
  single sequence regardless of concurrent client load).
  """

  def setUp(self):
    self.mock_sampling_params_cls = MagicMock(side_effect=lambda **kw: kw)
    self._vllm_patcher = patch.dict(
        "sys.modules",
        {
            "vllm": MagicMock(),
            "vllm.sampling_params": MagicMock(SamplingParams=self.mock_sampling_params_cls),
        },
    )
    self._vllm_patcher.start()
    self.addCleanup(self._vllm_patcher.stop)

  def _make_llm(self, generate_side_effect):
    mock_llm = MagicMock()
    mock_llm.get_tokenizer.return_value.apply_chat_template.side_effect = lambda messages, **kw: [
        int(messages[0]["content"][-1])
    ]
    mock_llm.generate.side_effect = generate_side_effect
    return mock_llm

  async def _post_chat(self, app, content):
    import httpx  # pylint: disable=import-outside-toplevel

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
      return await client.post(
          "/v1/chat/completions",
          json={"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 5},
      )

  async def test_concurrent_requests_coalesce_into_one_generate_call(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel

    calls = []

    def fake_generate(prompts, _sampling_params):
      calls.append(list(prompts))
      return [
          _make_mock_output(generated_text=f"reply-to-msg{p['prompt_token_ids'][0]}")
          for p in prompts
      ]

    mock_llm = self._make_llm(fake_generate)
    # Wide window so all 5 concurrent requests land in the same pending batch
    # before the timer fires -- this asserts real coalescing, not a timing fluke.
    app = _build_app(mock_llm, chat_batch_wait_s=0.2, chat_batch_max_size=64)

    responses = await asyncio.gather(*[self._post_chat(app, f"msg{i}") for i in range(5)])

    self.assertEqual(len(calls), 1, f"expected 1 batched generate() call, got {len(calls)}: {calls}")
    self.assertEqual(len(calls[0]), 5)
    for i, resp in enumerate(responses):
      self.assertEqual(resp.status_code, 200)
      self.assertEqual(resp.json()["choices"][0]["message"]["content"], f"reply-to-msg{i}")

  async def test_batch_flushes_early_at_max_size(self):
    """A full batch must flush immediately, not wait out the full time window."""
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel

    calls = []

    def fake_generate(prompts, _sampling_params):
      calls.append(list(prompts))
      return [_make_mock_output(generated_text="ok") for _ in prompts]

    mock_llm = self._make_llm(fake_generate)
    # 5s window: if early-flush-at-max-batch didn't work, the wait_for below
    # would time out instead of the requests completing almost immediately.
    app = _build_app(mock_llm, chat_batch_wait_s=5.0, chat_batch_max_size=2)

    responses = await asyncio.wait_for(
        asyncio.gather(*[self._post_chat(app, f"m{i}") for i in range(4)]),
        timeout=2.0,
    )
    self.assertTrue(all(r.status_code == 200 for r in responses))
    self.assertEqual(len(calls), 2)
    self.assertTrue(all(len(c) == 2 for c in calls))

  async def test_generate_exception_fails_every_request_in_the_batch(self):
    """One failing batched generate() call must 500 every request in it, not hang any."""
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel

    mock_llm = self._make_llm(ValueError("boom: simulated engine failure"))
    app = _build_app(mock_llm, chat_batch_wait_s=0.05, chat_batch_max_size=64)

    responses = await asyncio.wait_for(
        asyncio.gather(*[self._post_chat(app, f"m{i}") for i in range(3)]),
        timeout=2.0,
    )
    self.assertTrue(all(r.status_code == 500 for r in responses))
    self.assertTrue(all("boom" in r.json().get("detail", "") for r in responses))

  async def test_admission_control_rejects_request_beyond_resolved_concurrency(self):
    from maxtext.eval.runner.server_manager import _build_app  # pylint: disable=import-outside-toplevel

    mock_llm = self._make_llm(lambda prompts, _: [_make_mock_output() for _ in prompts])
    app = _build_app(
        mock_llm,
        chat_batch_wait_s=0.2,
        chat_batch_max_size=64,
        request_concurrency=1,
    )
    responses = await asyncio.gather(self._post_chat(app, "m1"), self._post_chat(app, "m2"))
    self.assertEqual(sorted(response.status_code for response in responses), [200, 429])


if __name__ == "__main__":
  unittest.main()
