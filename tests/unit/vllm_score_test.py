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

"""CPU scoring regression tests; no vLLM installation or model is needed."""

import functools
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

from absl import flags
import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module", name="score")
def _score_module():
  """Load the real module with isolated flags and hardware dependencies.

  Restore sys.modules after loading, so other tests still import real dependencies.
  Use a private module name to avoid caching stubs as maxtext.inference.vllm_score.
  """
  names = (
      "maxtext",
      "maxtext.common",
      "maxtext.common.common_types",
      "maxtext.configs",
      "maxtext.configs.pyconfig",
      "maxtext.inference",
      "maxtext.inference.vllm_decode",
      "maxtext.integration",
      "maxtext.integration.vllm",
      "maxtext.integration.vllm.maxtext_vllm_adapter",
      "maxtext.utils",
      "maxtext.utils.max_logging",
      "jax",
      "transformers",
      "vllm",
      "vllm.inputs",
      "vllm.sampling_params",
  )
  modules = {name: ModuleType(name) for name in names}
  for name, module in modules.items():
    module.__path__ = []
    parent, _, child = name.rpartition(".")
    if parent in modules:
      setattr(modules[parent], child, module)
  modules["maxtext.common.common_types"].Config = object
  modules["maxtext.utils.max_logging"].log = lambda _: None
  modules["vllm"].LLM = object
  modules["vllm.inputs"].TokensPrompt = dict
  modules["vllm.sampling_params"].SamplingParams = SimpleNamespace
  private_flags = flags.FlagValues()
  flag_functions = {
      name: functools.partial(getattr(flags, name), flag_values=private_flags)
      for name in ("DEFINE_string", "DEFINE_integer", "DEFINE_enum", "DEFINE_bool", "register_validator")
  }
  path = Path(__file__).resolve().parents[2] / "src/maxtext/inference/vllm_score.py"
  spec = importlib.util.spec_from_file_location("_vllm_score_unit_test", path)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules), mock.patch.multiple(flags, FLAGS=private_flags, **flag_functions):
    spec.loader.exec_module(module)
  private_flags(["vllm_score_test"])
  return module


class FakeTokenizer:
  """Character tokenizer with explicit fixtures for boundary merges and BOS/EOS."""

  eos_token_id = 255

  def __len__(self):
    return 256

  def encode(self, text, add_special_tokens=False):
    ids = [200] if text == "mergeA" else [ord(c) for c in text]
    return [254] + ids if add_special_tokens else ids

  def decode(self, ids):
    return "".join(chr(token_id) for token_id in ids)


@pytest.fixture(name="engine")
def _engine():
  return SimpleNamespace(
      get_tokenizer=FakeTokenizer,
      llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_logprobs=20)),
      generate=mock.Mock(side_effect=AssertionError("Unexpected inference request")),
  )


def _logprobs(values):
  return {token_id: SimpleNamespace(logprob=value) for token_id, value in values.items()}


@pytest.mark.parametrize("special", [False, True])
def test_numeric_string_is_text_and_integer_is_token_id(score, special):
  """With a text prompt, tokenize candidate '1' as text but use candidate 1 directly as a token ID."""
  tokenizer = FakeTokenizer()
  prompt_ids = tokenizer.encode("prompt ", add_special_tokens=special)
  assert score.candidate_token_id(tokenizer, "prompt ", prompt_ids, "1", special) == 49
  assert score.candidate_token_id(tokenizer, "prompt ", prompt_ids, 1, special) == 1


@pytest.mark.parametrize("prompt,candidate", [("prompt ", "AB"), ("merge", "A"), ("prompt ", "")])
def test_rejects_non_single_token_continuations(score, prompt, candidate):
  """Reject empty or multi-token continuations and tokens that merge across the prompt boundary."""
  tokenizer = FakeTokenizer()
  with pytest.raises(ValueError, match="not a single token"):
    score.candidate_token_id(tokenizer, prompt, tokenizer.encode(prompt), candidate)


@pytest.mark.parametrize("candidate", [-1, 256, True, 1.5])
def test_rejects_invalid_candidate_ids(score, engine, candidate):
  """Reject out-of-vocabulary IDs, booleans, and floats before issuing inference requests."""
  with pytest.raises(ValueError):
    score.candidate_token_id(FakeTokenizer(), "p", [112], candidate)
  with pytest.raises(ValueError):
    score.VllmScorer(engine).candidate_logprobs([[112]], [[candidate]])
  engine.generate.assert_not_called()


def test_rejects_prompt_with_appended_eos(score):
  """Reject candidate scoring when special-token handling appends EOS to the prompt."""
  with pytest.raises(ValueError, match="appended EOS"):
    score.candidate_token_id(FakeTokenizer(), "p", [112, 255], "A", add_special_tokens=True)


def test_auto_falls_back_only_for_missing_prompts_and_preserves_order(score, engine):
  """Score only prompts with missing top-k candidates exactly, preserving candidate order."""
  engine.generate.side_effect = None
  engine.generate.return_value = [
      SimpleNamespace(prompt_logprobs=[None, _logprobs({66: -4.0})]),
      SimpleNamespace(prompt_logprobs=[None, _logprobs({65: -2.0})]),
  ]
  results = score.VllmScorer(engine).candidate_logprobs(
      [[10], [20], [30]],
      [[66, 65], [66, 65], []],
      top_k_logprobs=[{65: -1.0, 66: -3.0}, {65: -9.0}, {}],
  )
  np.testing.assert_array_equal(results[0], [-3.0, -1.0])
  np.testing.assert_array_equal(results[1], [-4.0, -2.0])
  assert results[2].size == 0
  requests = engine.generate.call_args.args[0]
  assert requests == [{"prompt_token_ids": [20, 66]}, {"prompt_token_ids": [20, 65]}]
  assert engine.generate.call_count == 1


def test_top_k_rejects_missing_candidates_without_exact_inference(score, engine):
  """Raise on candidates missing from top-k without falling back to exact inference."""
  with pytest.raises(RuntimeError, match="Candidates outside top"):
    score.VllmScorer(engine).candidate_logprobs([[10]], [[65, 66]], method="top_k", top_k_logprobs=[{65: -1.0}])
  engine.generate.assert_not_called()


def test_exact_returns_candidate_scores_in_input_order(score, engine):
  """Append each candidate to its prompt, read that last token's logprob, and return scores in input order."""
  engine.generate.side_effect = None
  engine.generate.return_value = [
      SimpleNamespace(prompt_logprobs=[None, _logprobs({10: -99.0}), _logprobs({99: -0.1, 66: -3.0})]),
      SimpleNamespace(prompt_logprobs=[None, _logprobs({10: -99.0}), _logprobs({99: -0.1, 65: -1.0})]),
      SimpleNamespace(prompt_logprobs=[None, _logprobs({99: -0.1, 67: -2.0})]),
  ]
  results = score.VllmScorer(engine).candidate_logprobs([[9, 10], [20]], [[66, 65], [67]], method="exact")
  np.testing.assert_array_equal(results[0], [-3.0, -1.0])
  np.testing.assert_array_equal(results[1], [-2.0])
  assert engine.generate.call_args.args[1].prompt_logprobs == 1
  assert engine.generate.call_args.args[0] == [
      {"prompt_token_ids": [9, 10, 66]},
      {"prompt_token_ids": [9, 10, 65]},
      {"prompt_token_ids": [20, 67]},
  ]


@pytest.mark.parametrize("returned", [None, [None, _logprobs({66: -1.0})]])
def test_exact_rejects_missing_candidate_logprob(score, engine, returned):
  """Raise when an exact-scoring response omits the appended candidate's logprob."""
  engine.generate.side_effect = None
  engine.generate.return_value = [SimpleNamespace(prompt_logprobs=returned)]
  with pytest.raises(RuntimeError, match="candidate token 65"):
    score.VllmScorer(engine).exact_candidate_logprobs([[10]], [[65]])


@pytest.mark.parametrize("top_k", [0, 21])
def test_rejects_top_k_outside_engine_limit(score, engine, top_k):
  """Reject nonpositive top-k values and values exceeding the engine's logprob limit."""
  with pytest.raises(ValueError, match="top_k must be positive"):
    score.VllmScorer(engine).next_token_logprobs([[10]], top_k=top_k)
  engine.generate.assert_not_called()


def test_candidate_probabilities_are_stable_for_large_negative_scores(score):
  """Normalize very negative scores without underflow and handle an empty candidate set."""
  # pylint: disable=protected-access
  np.testing.assert_allclose(score._candidate_probabilities(np.array([-1000.0, -1001.0])), [0.7310585786, 0.2689414214])
  assert score._candidate_probabilities(np.array([])).size == 0


def test_score_requests_reuses_top_k_and_reports_prediction(score, engine):
  """Reuse top-k scores to report sorted tokens, candidate probabilities, and the winning option."""
  engine.generate.side_effect = None
  engine.generate.return_value = [
      SimpleNamespace(outputs=[SimpleNamespace(logprobs=[_logprobs({66: -2.0, 65: -1.0})])]),
      SimpleNamespace(outputs=[SimpleNamespace(logprobs=[_logprobs({67: -0.5})])]),
  ]
  requests = [
      {"prompt": "p", "prompt_ids": [112], "candidates": ["B", "A"], "candidate_ids": [66, 65]},
      {"prompt": "q", "prompt_ids": [113], "candidates": [], "candidate_ids": []},
  ]
  results = score.score_requests(score.VllmScorer(engine), requests)
  assert engine.generate.call_count == 1
  assert results[0]["prediction"] == "A"
  assert results[0]["greedy_token"] == {"id": 65, "text": "A", "logprob": -1.0}
  assert [token["id"] for token in results[0]["top_k"]] == [65, 66]
  assert [candidate["candidate"] for candidate in results[0]["candidates"]] == ["B", "A"]
  np.testing.assert_allclose(
      [candidate["prob_among_candidates"] for candidate in results[0]["candidates"]], [0.2689414214, 0.7310585786]
  )
  assert results[1]["greedy_token"]["id"] == 67
  assert "prediction" not in results[1]
  assert "candidates" not in results[1]
