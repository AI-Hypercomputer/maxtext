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

"""Minimal MegaBlox v2 Pallas GMM & TGMM kernels."""

from maxtext.experimental.qwen_35_64c.ext_kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.experimental.qwen_35_64c.ext_kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2

__all__ = ["gmm_v2", "tgmm_v2"]
