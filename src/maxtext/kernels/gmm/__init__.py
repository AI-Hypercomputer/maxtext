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

"""Decoupled grouped matrix multiplication (GMM) backends and quantization dispatch."""

from maxtext.kernels.gmm.backends import (
    DLHS_RAGGED_DOT_DIM_NUMS,
    DRHS_RAGGED_DOT_DIM_NUMS,
    GmmBackend,
    JaxRaggedDotBackend,
    MegabloxBackend,
    TeGroupedGemmBackend,
    TokamaxV1Backend,
    TokamaxV2Backend,
    scale_like,
    select_gmm_backend,
)
from maxtext.kernels.gmm.dispatch import grouped_matmul
from maxtext.kernels.gmm.quantize import (
    TcUnsortCfg,
    check_tc_unsort_bwd_supported,
    dlhs_scale_grad_by_rhs_scale,
    lhs_rhs_share_fixed_scale,
    prepare_lhs_scale,
    quantized_gmm,
    resolve_quantization_rule,
)

__all__ = [
    "DLHS_RAGGED_DOT_DIM_NUMS",
    "DRHS_RAGGED_DOT_DIM_NUMS",
    "GmmBackend",
    "JaxRaggedDotBackend",
    "MegabloxBackend",
    "TcUnsortCfg",
    "TeGroupedGemmBackend",
    "TokamaxV1Backend",
    "TokamaxV2Backend",
    "check_tc_unsort_bwd_supported",
    "dlhs_scale_grad_by_rhs_scale",
    "grouped_matmul",
    "lhs_rhs_share_fixed_scale",
    "prepare_lhs_scale",
    "quantized_gmm",
    "resolve_quantization_rule",
    "scale_like",
    "select_gmm_backend",
]
