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

"""Multi-Pass Precision Emulation subpackage (`codesign.multi_pass`)."""

from maxtext.codesign.multi_pass.checkpoint_utils import (
    apply_multipass_to_model_weights,
    apply_multipass_to_pytree,
    evaluate_multipass_error,
    infer_weight_contracting_axes,
    inspect_model_or_pytree_multipass,
    load_multipass_metadata,
    save_multipass_metadata,
)
from maxtext.codesign.multi_pass.multipass_config import (
    DEFAULT_EMULATION_CONFIG_KEYS,
    DEFAULT_MULTIPASS_CONFIG_KEYS,
    DtypeSpec,
    EmulationConfig,
    MULTIPASS_FLAG_DOCUMENTATION,
    MULTIPASS_PRESETS,
    MultipassEmulationConfig,
    MultipassRuleConfig,
    OperandMultipassSpec,
    TARGET_PRESETS,
    extract_layer_index,
    get_registered_pass_slicer,
    parse_active_passes,
    parse_dtype_list,
    parse_layer_indices,
    register_pass_slicer,
    resolve_cross_term_pairs,
)
from maxtext.codesign.multi_pass.multipass_kernel import (
    MultipassSlice,
    decompose_tensor_into_passes,
    multipass_dot_general,
    multipass_einsum,
    multipass_gmm_operands,
    quantize_to_dtype_spec,
    reconstruct_multipass_eager,
    reconstruct_multipass_tensor,
    resolve_axis,
)

__all__ = [
    "DEFAULT_EMULATION_CONFIG_KEYS",
    "DEFAULT_MULTIPASS_CONFIG_KEYS",
    "DtypeSpec",
    "EmulationConfig",
    "MULTIPASS_FLAG_DOCUMENTATION",
    "MULTIPASS_PRESETS",
    "MultipassEmulationConfig",
    "MultipassRuleConfig",
    "MultipassSlice",
    "OperandMultipassSpec",
    "TARGET_PRESETS",
    "apply_multipass_to_model_weights",
    "apply_multipass_to_pytree",
    "decompose_tensor_into_passes",
    "evaluate_multipass_error",
    "extract_layer_index",
    "get_registered_pass_slicer",
    "infer_weight_contracting_axes",
    "inspect_model_or_pytree_multipass",
    "load_multipass_metadata",
    "multipass_dot_general",
    "multipass_einsum",
    "multipass_gmm_operands",
    "parse_active_passes",
    "parse_dtype_list",
    "parse_layer_indices",
    "quantize_to_dtype_spec",
    "reconstruct_multipass_eager",
    "reconstruct_multipass_tensor",
    "register_pass_slicer",
    "resolve_axis",
    "resolve_cross_term_pairs",
    "save_multipass_metadata",
]
