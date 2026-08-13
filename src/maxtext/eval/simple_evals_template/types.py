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

"""
Original implementation from https://github.com/openai/simple-evals.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, overload

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class SamplerResponse:
  """
  Response from a sampler.
  """

  response_text: str
  actual_queried_message_list: MessageList
  response_metadata: dict[str, Any]


class SamplerBase:
  """
  Base class for defining a sampling model, which can be evaluated,
  or used as part of the grading process.
  """

  def __call__(
      self,
      message_list: MessageList,
  ) -> SamplerResponse:
    raise NotImplementedError


@dataclass
class EvalResult:
  """
  Result of running an evaluation (usually consisting of many samples)
  """

  score: float | None  # top-line metric
  metrics: dict[str, float] | None  # other metrics
  htmls: list[str]  # strings of valid HTML
  convos: list[MessageList]  # sampled conversations
  metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen


@dataclass
class SingleEvalResult:
  """
  Result of evaluating a single sample
  """

  score: float | None
  metrics: dict[str, float] = field(default_factory=dict)
  html: str | None = None
  convo: MessageList | None = None  # sampled conversation
  example_level_metadata: dict[str, Any] | None = None  # Extra data such as rubric scores or sollen


class Eval:
  """
  Base class for defining an evaluation.
  """

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    raise NotImplementedError
