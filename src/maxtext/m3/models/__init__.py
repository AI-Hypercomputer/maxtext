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
        qwen3/        modeling_qwen3.py        attention, MLP, decoder layer, model
        qwen3_moe/    modeling_qwen3_moe.py    routed MoE block and layer on Qwen3's attention
        qwen3_vl/     modeling_qwen3_vl.py     vision tower and MRoPE attention on Qwen3's stack
        deepseek_v3/  modeling_deepseek_v3.py  MLA, YaRN, sigmoid/bias routing, shared experts

This module holds the single dispatch point (`create_model`) and the model
registry. There is no shared `Decoder`, no shared attention class, and no
model-name branching: the registry maps a `model_name` to a constructor and
that is the whole of it.
"""

from collections.abc import Callable

from flax import nnx

from maxtext.m3.models.deepseek_v3.modeling_deepseek_v3 import create_deepseek_v3_model
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3_moe.modeling_qwen3_moe import create_qwen3_moe_model
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model

MODEL_REGISTRY: dict[str, Callable] = {
    "qwen3-0.6b": create_qwen3_model,
    "qwen3-30b-a3b": create_qwen3_moe_model,
    "qwen3-235b-a22b": create_qwen3_moe_model,
    "qwen3-vl-2b": create_qwen3_vl_model,
    "deepseek3-671b": create_deepseek_v3_model,
    "deepseek3-tiny": create_deepseek_v3_model,
}


def create_model(config, mesh, **kwargs) -> nnx.Module:
  """Builds the registered m3 model for `config.model_name`.

  Must be called inside `jax.set_mesh(mesh)`: parameters are placed from their
  sharding metadata as they are created (see `maxtext.m3.core.sharding`).
  Flax's eager sharding is switched on for the duration, because importing
  legacy MaxText (`maxtext.layers.nnx_wrappers`) turns it off process-wide,
  after which `kernel_metadata={"out_sharding": ...}` is neither applied nor
  enforced and every parameter silently comes out replicated.

  Args:
    config: Legacy `pyconfig` surface.
    mesh: Passed through to the constructor; the ambient mesh is what places
      parameters.
    **kwargs: Constructor keywords -- `rngs`, `sharding`, and legacy ones the
      models ignore.
  """
  if config.model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Model {config.model_name!r} does not support the m3 backend (use_m3_model=True). "
        f"Registered models: {', '.join(sorted(MODEL_REGISTRY))}."
    )
  with nnx.use_eager_sharding(True):
    return MODEL_REGISTRY[config.model_name](config, mesh, **kwargs)
