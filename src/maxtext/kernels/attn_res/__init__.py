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

"""Fused Pallas TPU kernels for Kimi-K3 Block Attention Residuals (AttnRes)."""

from maxtext.kernels.attn_res.pallas_kernel import (
    AttnResKernelConfig,
    attn_res_bwd_local,
    attn_res_fwd_local,
    attn_res_pallas,
    attn_res_rms_norm_pallas,
    can_use_attn_res_pallas,
)

__all__ = [
    "AttnResKernelConfig",
    "attn_res_bwd_local",
    "attn_res_fwd_local",
    "attn_res_pallas",
    "attn_res_rms_norm_pallas",
    "can_use_attn_res_pallas",
]
