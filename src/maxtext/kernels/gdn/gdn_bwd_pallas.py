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

"""Canonical Gated Delta Net (GDN) backward pass facade re-exporting from .gdn_bwd."""

from .gdn_bwd.api import (
    _gdn_decoupled_conv1d_bwd,
    _gdn_decoupled_conv1d_fwd,
    _run_local_gdn_decoupled_fwd,
    decoupled_conv1d_gdn_bwd_kernel,
    gdn_decoupled_conv1d,
    pallas_gdn_bwd_kernel,
)
from .gdn_bwd.bwd_memory_ref import make_bwd_block_specs
from .gdn_bwd.compute_conv1d_bwd import (
    conv1d_silu_bwd,
    conv1d_silu_fwd,
)
from .gdn_bwd.jax_compute_gdn_states import (
    _compute_forward_conv_and_states,
    chunk_forward,
    chunk_forward_with_tinv,
    chunk_state_forward,
    pure_jax_decoupled_conv1d_gdn,
)
from .gdn_bwd.pallas_mosaic_tpu_bwd import (
    _bwd_gdn_pipeline_body,
    _pallas_gdn_bwd_kernel_single_group,
    GDNBackwardConfig,
)
from .gdn_bwd.runtime_utils import (
    _invert_triangular_matrix_bwd,
    _invert_triangular_matrix_fwd,
    ensure_cpu_interpret_registered,
    invert_triangular_matrix,
)

__all__ = [
    "GDNBackwardConfig",
    "_bwd_gdn_pipeline_body",
    "_compute_forward_conv_and_states",
    "_gdn_decoupled_conv1d_bwd",
    "_gdn_decoupled_conv1d_fwd",
    "_invert_triangular_matrix_bwd",
    "_invert_triangular_matrix_fwd",
    "_pallas_gdn_bwd_kernel_single_group",
    "_run_local_gdn_decoupled_fwd",
    "chunk_forward",
    "chunk_forward_with_tinv",
    "chunk_state_forward",
    "conv1d_silu_bwd",
    "conv1d_silu_fwd",
    "decoupled_conv1d_gdn_bwd_kernel",
    "ensure_cpu_interpret_registered",
    "gdn_decoupled_conv1d",
    "invert_triangular_matrix",
    "make_bwd_block_specs",
    "pallas_gdn_bwd_kernel",
    "pure_jax_decoupled_conv1d_gdn",
]
