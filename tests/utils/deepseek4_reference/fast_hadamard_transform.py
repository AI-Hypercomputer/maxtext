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

"""Pure-torch stand-in for the CUDA `fast_hadamard_transform` package."""

import functools

import torch


@functools.lru_cache(maxsize=8)
def hadamard_matrix(d: int, dtype: torch.dtype) -> torch.Tensor:
  """Unnormalized Sylvester Hadamard matrix (natural order, symmetric)."""
  assert d > 0 and d & (d - 1) == 0, f"dim {d} must be a power of 2"
  h = torch.ones(1, 1, dtype=dtype)
  while h.size(0) < d:
    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
  return h


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
  """Returns (x @ H_d) * scale along the last dim, computed in >= fp32."""
  cdt = torch.promote_types(x.dtype, torch.float32)
  h = hadamard_matrix(x.size(-1), cdt).to(x.device)
  return (torch.matmul(x.to(cdt), h) * scale).to(x.dtype)
