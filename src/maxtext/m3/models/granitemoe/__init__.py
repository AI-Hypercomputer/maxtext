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

"""GraniteMoe pure-NNX model package for MaxText `m3`."""

from maxtext.m3.models.granitemoe.checkpoint_mappings_granitemoe import export_model_to_hf_weights
from maxtext.m3.models.granitemoe.checkpoint_mappings_granitemoe import GRANITEMOE_HF_WEIGHTS_TO_SHAPE
from maxtext.m3.models.granitemoe.checkpoint_mappings_granitemoe import GRANITEMOE_MAXTEXT_TO_HF_PARAM_HOOK_FN
from maxtext.m3.models.granitemoe.checkpoint_mappings_granitemoe import GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING
from maxtext.m3.models.granitemoe.checkpoint_mappings_granitemoe import load_hf_weights_into_model
from maxtext.m3.models.granitemoe.modeling_granitemoe import compute_granitemoe_lb_loss
from maxtext.m3.models.granitemoe.modeling_granitemoe import compute_granitemoe_loss
from maxtext.m3.models.granitemoe.modeling_granitemoe import create_granitemoe_model
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeAttention
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeConfig
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeDecoderLayer
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeForwardOutput
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeKVCache
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeMoE
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeModel
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeRMSNorm
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeSharding
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter
from maxtext.m3.models.granitemoe.modeling_granitemoe import granitemoe_loss_fn
from maxtext.m3.models.granitemoe.modeling_granitemoe import init_kv_cache
from maxtext.m3.models.granitemoe.utils_granitemoe import calculate_granitemoe_tflops_per_device
from maxtext.m3.models.granitemoe.utils_granitemoe import configure_granitemoe_engine
from maxtext.m3.models.granitemoe.utils_granitemoe import granitemoe_engine_loss_fn

__all__ = [
    "GRANITEMOE_HF_WEIGHTS_TO_SHAPE",
    "GRANITEMOE_MAXTEXT_TO_HF_PARAM_HOOK_FN",
    "GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING",
    "GraniteMoeAttention",
    "GraniteMoeConfig",
    "GraniteMoeDecoderLayer",
    "GraniteMoeExperts",
    "GraniteMoeForwardOutput",
    "GraniteMoeKVCache",
    "GraniteMoeMoE",
    "GraniteMoeModel",
    "GraniteMoeRMSNorm",
    "GraniteMoeSharding",
    "GraniteMoeTopKRouter",
    "calculate_granitemoe_tflops_per_device",
    "compute_granitemoe_lb_loss",
    "compute_granitemoe_loss",
    "configure_granitemoe_engine",
    "create_granitemoe_model",
    "export_model_to_hf_weights",
    "granitemoe_engine_loss_fn",
    "granitemoe_loss_fn",
    "init_kv_cache",
    "load_hf_weights_into_model",
]
