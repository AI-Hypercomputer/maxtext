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

"""Abstract base classes for benchmark datasets."""

from __future__ import annotations

import abc
from typing import NamedTuple


class SampleRequest(NamedTuple):
  """A single inference request with its ground-truth reference.

  Attributes:
    prompt: The full text prompt to send to the model (after chat templating).
    reference: Ground-truth answer/label used by the scorer.
    metadata: Optional dict of extra fields forwarded to the scorer
              (e.g. {"subject": "college_math"} for per-subject MMLU stats).
  """

  prompt: str
  reference: str
  metadata: dict | None = None


class BenchmarkDataset(abc.ABC):
  """Abstract base class for benchmark datasets."""
  name: str

  @abc.abstractmethod
  def sample_requests(
      self,
      num_samples: int | None,
      tokenizer,
  ) -> list[SampleRequest]:
    """Load the dataset and return a list of SampleRequests.

    Args:
      num_samples: If not None, truncate to this number of samples.
      tokenizer: A HuggingFace tokenizer used for chat templating.  Implementations
                 that do not require tokenization may ignore this parameter.

    Returns:
      List of SampleRequest objects ready for inference.
    """
