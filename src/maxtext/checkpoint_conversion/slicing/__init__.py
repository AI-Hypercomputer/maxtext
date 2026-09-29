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

"""Model / checkpoint slicing tools.

- `slice_model`: find the first depth -> expert -> width slice of an onboarded
  MaxText model that fits a per-device HBM budget ("What should I try?").
- `slice_hf_checkpoint`: apply a `slice_model` trim plan directly to a
  Hugging Face checkpoint (`config.json` + `.safetensors`).
- `reducer`: legal reductions of a MaxText config ("How can this model get smaller?").
- `slice_utils`: MaxText / JAX compile, HBM, and checkpoint plumbing
  ("How do I ask MaxText to compile or save it?").
"""
