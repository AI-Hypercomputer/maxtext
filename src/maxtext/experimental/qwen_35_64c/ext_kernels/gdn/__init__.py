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

"""Fused Conv1D + Gated DeltaNet Pallas TPU kernels."""

from maxtext.experimental.qwen_35_64c.ext_kernels.gdn.gdn_bwd_pallas import decoupled_conv1d_gdn_bwd_kernel
from maxtext.experimental.qwen_35_64c.ext_kernels.gdn.gdn_bwd_pallas import gdn_decoupled_conv1d
from maxtext.experimental.qwen_35_64c.ext_kernels.gdn.gdn_bwd_pallas import pallas_gdn_bwd_kernel

__all__ = [
    "decoupled_conv1d_gdn_bwd_kernel",
    "gdn_decoupled_conv1d",
    "pallas_gdn_bwd_kernel",
]
