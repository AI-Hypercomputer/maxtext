# Copyright 2023–2025 Google LLC
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

"""
This module defines the Pydantic models used for request and response
validation in the OpenAI-compatible API server. These models ensure that
the data exchanged with clients conforms to the expected structure for
endpoints like completions and chat completions.
"""

import time
import uuid
from typing import List, Optional, Union, Dict, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator


class SamplingParams(BaseModel):
  """
  Defines the common sampling parameters that are shared across different types of
  generation requests, such as standard completions and chat-based completions.

  Attributes:
      max_tokens: The maximum number of tokens to generate.
      temperature: The sampling temperature.
      top_p: The nucleus sampling probability.
      top_k: The top-k sampling integer.
      stream: Whether to stream the response.
      stop: A string or list of strings that will stop the generation.
      seed: A seed for deterministic sampling.
  """

  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  top_p: Optional[float] = None
  top_k: Optional[int] = None
  stream: Optional[bool] = False
  stop: Optional[Union[str, List[str]]] = None
  seed: Optional[int] = None


class CompletionRequest(SamplingParams):
  """
  Represents a request for a standard text completion, inheriting sampling
  parameters from `SamplingParams`.

  Attributes:
      model: The ID of the model to use for the completion.
      prompt: The prompt(s) to generate completions for, which can be a string,
              a list of strings, a list of token IDs, or a list of lists of token IDs.
      echo: Whether to echo the prompt back in the response.
      logprobs: The number of top log probabilities to return for each token.
  """

  model: str
  prompt: Union[str, List[str], List[int], List[List[int]]]
  echo: Optional[bool] = False
  logprobs: Optional[int] = None

  @field_validator("logprobs")
  def validate_logprobs(self, v):
    if v is not None and v < 0:
      raise ValueError("logprobs must be a non-negative integer if provided.")
    return v


class LogProbsPayload(BaseModel):
  """
  A data structure to hold the log probability information for a sequence of tokens,
  formatted to be compatible with OpenAI's API.

  Attributes:
      tokens: The string representation of each token.
      token_logprobs: The log probability of each token.
      top_logprobs: A list of dictionaries mapping other tokens to their log
                    probabilities at each position.
      text_offset: The character offset of each token in the text.
  """

  tokens: List[str]
  token_logprobs: List[Optional[float]]
  top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
  text_offset: List[int]


class CompletionChoice(BaseModel):
  """
  Represents a single choice (a possible completion) in a `CompletionResponse`.

  Attributes:
      text: The generated text for this choice.
      index: The index of this choice in the list of choices.
      logprobs: An optional payload containing log probability information.
      finish_reason: The reason the model stopped generating tokens (e.g., 'stop', 'length').
  """

  text: str
  index: int
  logprobs: Optional[LogProbsPayload] = None
  finish_reason: str = "stop"


class Usage(BaseModel):
  """
  Provides information about the number of tokens used in a request.

  Attributes:
      prompt_tokens: The number of tokens in the input prompt.
      completion_tokens: The number of tokens in the generated completion.
      total_tokens: The total number of tokens used.
  """

  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


# Define a TypeVar for the choice models
ChoiceType = TypeVar("ChoiceType")


class BaseCompletionResponse(BaseModel, Generic[ChoiceType]):
  """
  A generic base response model using Python's Generic type. It shares all
  common fields for API responses and uses a TypeVar for the 'choices' list
  to accommodate different types of choices (e.g., for standard vs. chat completions).

  Attributes:
      id: A unique identifier for the response.
      object: The type of the object (e.g., 'text_completion').
      created: The timestamp when the response was created.
      model: The model that generated the response.
      choices: A list of choices, the type of which is determined by `ChoiceType`.
      usage: Token usage statistics.
  """

  id: str
  object: str
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: List[ChoiceType]
  usage: Usage


class CompletionResponse(BaseCompletionResponse[CompletionChoice]):
  """
  The specific response object for a standard completion request. It inherits
  from the generic base and specifies `CompletionChoice` as its choice type.
  It also provides default values for the 'id' and 'object' fields.
  """

  id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
  object: str = "text_completion"


class ChatMessage(BaseModel):
  """
  Represents a single message within a chat conversation.

  Attributes:
      role: The role of the message's author (e.g., 'user', 'assistant').
      content: The text content of the message.
  """

  role: str
  content: str


class ChatCompletionRequest(SamplingParams):
  """
  Represents a request for a chat-based completion, where the input is a
  sequence of messages. Inherits sampling parameters from `SamplingParams`.

  Attributes:
      model: The ID of the model to use.
      messages: A list of `ChatMessage` objects representing the conversation history.
      logprobs: Whether to return log probabilities.
      top_logprobs: The number of top log probabilities to return if `logprobs` is true.
  """

  model: str
  messages: List[ChatMessage]
  logprobs: Optional[bool] = False
  top_logprobs: Optional[int] = None


class ChatCompletionChoice(BaseModel):
  """
  Represents a single choice in a `ChatCompletionResponse`.

  Attributes:
      index: The index of this choice.
      message: The `ChatMessage` generated by the model.
      finish_reason: The reason the model stopped generating.
      logprobs: An optional payload with log probability information.
  """

  index: int
  message: ChatMessage
  finish_reason: str = "stop"
  logprobs: Optional[LogProbsPayload] = None


class ChatCompletionResponse(BaseCompletionResponse[ChatCompletionChoice]):
  """
  The specific response object for a chat completion request. It inherits from
  the generic base, specifies `ChatCompletionChoice` as its choice type, and
  provides chat-specific default values for 'id' and 'object'.
  """

  id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
  object: str = "chat.completion"
