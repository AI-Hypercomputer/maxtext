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

"""Locates the Kimi-K3 HuggingFace reference checkout used by parity tests.

The reference is a large, untracked local checkout. Tests that compare against
it skip cleanly when it is absent so the suite stays green without it.
"""

import os
import unittest

KIMI_K3_REFERENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "kimi-k3-hf-reference"))


def reference_available() -> bool:
  """True if the Kimi-K3 HF reference checkout is present."""
  return os.path.isfile(os.path.join(KIMI_K3_REFERENCE_DIR, "modeling_kimi_linear.py"))


requires_kimi_k3_reference = unittest.skipUnless(
    reference_available(),
    f"Kimi-K3 HF reference checkout not found at {KIMI_K3_REFERENCE_DIR}. "
    "Clone the moonshotai Kimi-K3 repo there to run parity tests.",
)
