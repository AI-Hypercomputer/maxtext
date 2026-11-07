# Copyright 2026 Google LLC
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

"""Common audio generation contract and model-specific implementation dispatch."""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from maxtext.inference.component_runtime import GenerationContext
from maxtext.layers import audio_decoders


@dataclass
class AudioGenerationRequest:
  """Text-generation outputs and optional captured features for one speech stream."""

  prompt_ids: np.ndarray
  generated_ids: Sequence[int]
  features: dict[str, Any] = field(default_factory=dict)
  processor_outputs: Any = None
  output_path: str | None = None


@dataclass
class AudioGenerationResult:
  """An implementation-independent waveform; codec tokens are optional diagnostics."""

  waveform: np.ndarray
  sample_rate: int
  diagnostics: dict[str, Any] = field(default_factory=dict)


class AudioGenerator(Protocol):
  """A generator owns its conditioning and generation procedure, including any codec stages."""

  def generate(self, request: AudioGenerationRequest) -> AudioGenerationResult:
    """Generates audio for a single stream."""


def validate_audio_generation(config) -> None:
  """Checks the selected implementation's feature requirements before model loading."""
  audio_decoders.get_feature_spec(config).validate(num_layers=config.num_decoder_layers, scan_layers=config.scan_layers)


def create_audio_generator(config, context: GenerationContext) -> AudioGenerator:
  """Selects a generator without imposing a shared model architecture or codec loop."""
  output_type = audio_decoders.get_audio_output_type(config)
  if output_type == "qwen3_omni":
    from maxtext.inference.qwen3_omni_talker_pipeline import (  # pylint: disable=import-outside-toplevel
        Qwen3OmniTalkerPipeline,
    )

    return Qwen3OmniTalkerPipeline(config, context)
  raise ValueError("Audio generation requires enable_audio_output=true.")
