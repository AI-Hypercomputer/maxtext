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

"""Multimodal reference loader and golden-logit generation utilities for DeepSeek-V4.1-Flash."""

from tests.utils.deepseek_v41_mm_golden import generate_mm_golden_logits
from tests.utils.deepseek_v41_mm_golden import reference_loader

__all__ = [
    "generate_mm_golden_logits",
    "reference_loader",
]
