# Copyright 2023–2026 Google LLC
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
"""Linen-wrapped NNX modules for tests that still drive layers through Linen `init` / `apply`."""

from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.pipeline import NNXCircularPipeline, NNXPipeline
from maxtext.models.deepseek import DeepSeekMoELayer
from maxtext.models.simple_layer import SimpleDecoderLayer


def to_linen(nnx_class, *, name=None, **kwargs):
  """Wraps an NNX module class as a Linen module, with its arguments passed through."""
  return nnx_wrappers.to_linen(
      nnx_class, name=name, metadata_fn=variable_to_logically_partitioned, abstract_init=False, **kwargs
  )


SimpleDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    SimpleDecoderLayer, base_metadata_fn=variable_to_logically_partitioned
)
DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer, base_metadata_fn=variable_to_logically_partitioned
)
Pipeline = nnx_wrappers.to_linen_class(NNXPipeline, base_metadata_fn=variable_to_logically_partitioned)
CircularPipeline = nnx_wrappers.to_linen_class(NNXCircularPipeline, base_metadata_fn=variable_to_logically_partitioned)


def create_pipeline(config, layers=None, mesh=None, remat_policy=None):
  """Returns the Linen-wrapped NNX pipeline for the config; `layers` builds one stage from an `nnx.Rngs`."""
  cls = CircularPipeline if config.pipeline_fsdp_ag_per_repeat else Pipeline
  return cls(config=config, stage_factory=layers, mesh=mesh, remat_policy=remat_policy)
