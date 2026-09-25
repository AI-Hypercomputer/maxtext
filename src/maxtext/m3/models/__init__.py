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

"""m3 model garden: one self-contained directory per model family.

Each family owns its architecture end to end:

    models/
        qwen3/
            modeling_qwen3.py  # attention, MLP, decoder layer, model

This module holds the single dispatch point (create_model) and the model
registry. There is no shared Decoder, no shared attention class, and no
model-name branching.
"""

from collections.abc import Callable
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model


MODEL_REGISTRY: dict[str, Callable] = {
    "qwen3-0.6b": create_qwen3_model,
}


def create_model(config, mesh, **kwargs):
  """Dispatches to a registered m3 constructor using driver configuration."""
  if config.model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Model {config.model_name!r} does not support the m3 backend (use_m3_model=True). "
        f"Registered models: {', '.join(sorted(MODEL_REGISTRY))}."
    )
  return MODEL_REGISTRY[config.model_name](config, mesh, **kwargs)
