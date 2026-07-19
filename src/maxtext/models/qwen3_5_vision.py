# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 Vision model tower NNX subclasses.

These classes subclass the Qwen3-Omni vision tower components to provide
clean class type names (Qwen3_5MoeVision...) in the Flax NNX metadata,
ensuring that the JAX parameter keys stored in checkpoints do not contain
the word 'Omni'.
"""

from maxtext.models.qwen3 import Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeVisionProjector


class Qwen3_5MoeVisionEncoder(Qwen3OmniMoeVisionEncoder):
  """Subclass of Qwen3OmniMoeVisionEncoder for Qwen3.5 VL models.

  Inherits all core vision tower layers (patch embedding, position embedding,
  rotary embeddings, attention, and transformer blocks) without modification.
  """


class Qwen3_5MoeVisionProjector(Qwen3OmniMoeVisionProjector):
  """Subclass of Qwen3OmniMoeVisionProjector for Qwen3.5 VL models.

  Inherits the final projection/merger layers without modification.
  """
