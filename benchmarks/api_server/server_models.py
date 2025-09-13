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


import time
import uuid
from typing import List, Optional, Union, Dict, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator


class SamplingParams(BaseModel):
    """Common sampling parameters shared by Completion and ChatCompletion requests."""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

class CompletionRequest(SamplingParams):
    """Inherits all fields from SamplingParams."""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    echo: Optional[bool] = False
    logprobs: Optional[int] = None

    @field_validator("logprobs")
    def validate_logprobs(cls, v):
        if v is not None and v < 0:
            raise ValueError("logprobs must be a non-negative integer if provided.")
        return v

class LogProbsPayload(BaseModel):
    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
    text_offset: List[int]

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[LogProbsPayload] = None
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Define a TypeVar for the choice models
ChoiceType = TypeVar('ChoiceType')

class BaseCompletionResponse(BaseModel, Generic[ChoiceType]):
    """A generic base response model using Python's Generic type.
    
    It shares all common fields and uses a TypeVar for the 'choices' list.
    """
    id: str
    object: str
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChoiceType]
    usage: Usage

class CompletionResponse(BaseCompletionResponse[CompletionChoice]):
    """Inherits from the generic base, specifying CompletionChoice.
    
    We only need to override the 'id' and 'object' fields with their specific defaults.
    """
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(SamplingParams):
    """Inherits all fields from SamplingParams."""
    model: str
    messages: List[ChatMessage]
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Optional[LogProbsPayload] = None

class ChatCompletionResponse(BaseCompletionResponse[ChatCompletionChoice]):
    """Inherits from the generic base, specifying ChatCompletionChoice.
    
    We only need to override the 'id' and 'object' fields with their specific defaults.
    """
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
