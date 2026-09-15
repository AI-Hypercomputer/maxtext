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
# ==============================================================================

"""Local GDN forward kernel package with triangular inverse matrix caching."""

from . import compute_conv1d
from . import compute_gdn
from . import config
from . import gdn_bwd
from . import gdn_bwd_pallas
from . import memory_ref
from . import metadata
from . import tiling
from . import vmem_ldst
from . import wrapper
from .gdn_bwd_pallas import gdn_decoupled_conv1d
from .gdn_bwd_pallas import pallas_gdn_bwd_kernel

gdn_fused_conv1d = gdn_decoupled_conv1d
gdn_kernel = gdn_decoupled_conv1d

__all__ = [
    "compute_conv1d",
    "compute_gdn",
    "config",
    "gdn_bwd",
    "gdn_bwd_pallas",
    "gdn_decoupled_conv1d",
    "gdn_fused_conv1d",
    "gdn_kernel",
    "memory_ref",
    "metadata",
    "pallas_gdn_bwd_kernel",
    "tiling",
    "vmem_ldst",
    "wrapper",
]
