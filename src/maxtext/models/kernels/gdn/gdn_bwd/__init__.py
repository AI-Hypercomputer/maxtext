# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modular package for Gated Delta Net (GDN) backward pass."""

from .api import (
    _gdn_decoupled_conv1d_bwd,
    _gdn_decoupled_conv1d_fwd,
    _gdn_fused_conv1d_bwd,
    _gdn_fused_conv1d_fwd,
    _run_local_gdn_decoupled_fwd,
    _run_local_gdn_fused_fwd,
    decoupled_conv1d_gdn_bwd_computation,
    decoupled_conv1d_gdn_bwd_kernel,
    gdn_decoupled_conv1d,
    gdn_fused_conv1d,
    gdn_kernel,
    pallas_fused_conv1d_gdn_bwd_computation,
    pallas_fused_conv1d_gdn_bwd_kernel,
)
from .bwd_memory_ref import make_bwd_block_specs
from .compute_bwd_gdn import (
    _compute_forward_conv_and_states,
    chunk_forward,
    chunk_forward_with_tinv,
    chunk_state_forward_with_cached_tinv,
    pure_jax_decoupled_conv1d_gdn,
    pure_jax_fused_conv1d_gdn,
)
from .compute_conv1d_bwd import (
    conv1d_silu_bwd,
    conv1d_silu_fwd,
)
from .pallas_mosaic_tpu_bwd import (
    _bwd_gdn_pipeline_body,
    _pallas_gdn_bwd_kernel_single_group,
    pallas_gdn_bwd_computation,
    pallas_gdn_bwd_kernel,
)
from .runtime_utils import (
    _invert_triangular_matrix_bwd,
    _invert_triangular_matrix_fwd,
    ensure_cpu_interpret_registered,
    invert_triangular_matrix,
)

__all__ = [
    "chunk_forward",
    "chunk_forward_with_tinv",
    "chunk_state_forward_with_cached_tinv",
    "_compute_forward_conv_and_states",
    "pure_jax_decoupled_conv1d_gdn",
    "pure_jax_fused_conv1d_gdn",
    "pallas_gdn_bwd_kernel",
    "pallas_gdn_bwd_computation",
    "_bwd_gdn_pipeline_body",
    "_pallas_gdn_bwd_kernel_single_group",
    "decoupled_conv1d_gdn_bwd_kernel",
    "decoupled_conv1d_gdn_bwd_computation",
    "pallas_fused_conv1d_gdn_bwd_kernel",
    "pallas_fused_conv1d_gdn_bwd_computation",
    "conv1d_silu_fwd",
    "conv1d_silu_bwd",
    "gdn_decoupled_conv1d",
    "gdn_fused_conv1d",
    "gdn_kernel",
    "_gdn_decoupled_conv1d_fwd",
    "_gdn_decoupled_conv1d_bwd",
    "_gdn_fused_conv1d_fwd",
    "_gdn_fused_conv1d_bwd",
    "_run_local_gdn_decoupled_fwd",
    "_run_local_gdn_fused_fwd",
    "ensure_cpu_interpret_registered",
    "invert_triangular_matrix",
    "_invert_triangular_matrix_fwd",
    "_invert_triangular_matrix_bwd",
    "make_bwd_block_specs",
]
