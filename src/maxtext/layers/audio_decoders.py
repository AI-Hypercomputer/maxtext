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

"""Model-specific audio component construction behind a shared entry point.

Components are attached directly to Transformer using the returned names. There is
no wrapper module, so existing checkpoint paths remain unchanged.
"""

from maxtext.common.decoder_features import DecoderFeatureSpec


_AUTO_AUDIO_TYPES = {"qwen3-omni-30b-a3b": "qwen3_omni"}
_SUPPORTED_AUDIO_TYPES = frozenset(_AUTO_AUDIO_TYPES.values())


def get_audio_output_type(config) -> str | None:
  """Resolves an enabled audio capability, preserving older model configurations."""
  if not getattr(config, "enable_audio_output", False):
    return None
  output_type = getattr(config, "audio_output_type", "auto")
  if output_type == "auto":
    output_type = _AUTO_AUDIO_TYPES.get(config.model_name)
  if output_type not in _SUPPORTED_AUDIO_TYPES:
    raise ValueError(f"Unsupported audio_output_type={output_type!r} for model_name={config.model_name!r}.")
  return output_type


def get_feature_spec(config) -> DecoderFeatureSpec:
  """Declares the text-model features consumed by the selected audio implementation."""
  if get_audio_output_type(config) == "qwen3_omni":
    return DecoderFeatureSpec(input_embeddings=True, hidden_layers=(config.accept_hidden_layer,))
  return DecoderFeatureSpec()


def create_audio_components(config, mesh, quant, *, model_mode, rngs) -> dict:
  """Constructs named model components; algorithms and inference state live elsewhere."""
  if get_audio_output_type(config) == "qwen3_omni":
    from maxtext.models.qwen3_omni_talker import Qwen3OmniTalker  # pylint: disable=import-outside-toplevel
    from maxtext.models.qwen3_omni_code2wav import Qwen3OmniCode2Wav  # pylint: disable=import-outside-toplevel

    return {
        "talker": Qwen3OmniTalker(config, mesh, quant, model_mode=model_mode, rngs=rngs),
        "code2wav": Qwen3OmniCode2Wav(config, mesh, rngs=rngs),
    }
  return {}
