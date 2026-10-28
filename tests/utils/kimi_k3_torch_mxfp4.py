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

"""MXFP4-packed routed experts for the PyTorch Kimi-K3 reference (test oracle only).

The full model does not fit in host RAM with bf16 experts (~5.6 TB), so for the 93-layer
oracle each routed expert's `w1/w2/w3` `nn.Linear` is swapped for `Mxfp4Linear`, which keeps
the released `weight_packed` / `weight_scale` uint8 tensors (same names and shapes as the
safetensors keys, so `load_state_dict` consumes the checkpoint directly) and decodes the
weight on every call. The decode is bit-exact with the NumPy converter codec and computes in
fp32 before casting to the activation dtype, i.e. exactly what the dense oracle did when it
dequantized at load time -- so the forward is bit-identical to the dense oracle.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_mxfp4_torch(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
  """Decodes uint8 `packed [..., n/2]` + E8M0 `scale [..., n/32]` to `[..., n]` in `dtype`."""
  if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
    raise TypeError(f"packed/scale must be uint8, got {packed.dtype}/{scale.dtype}")
  lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=packed.device)
  low = lut[(packed & 0x0F).long()]
  high = lut[(packed >> 4).long()]
  values = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
  n = values.shape[-1]
  if n != scale.shape[-1] * MXFP4_GROUP_SIZE:
    raise ValueError(f"unpacked width {n} != {MXFP4_GROUP_SIZE} x {scale.shape[-1]} scale groups")
  factor = torch.exp2(scale.to(torch.float32) - E8M0_BIAS)
  out = values.reshape(*values.shape[:-1], -1, MXFP4_GROUP_SIZE) * factor.unsqueeze(-1)
  return out.reshape(*values.shape[:-1], n).to(dtype)


class Mxfp4Linear(nn.Module):
  """Bias-free `nn.Linear` replacement holding MXFP4 `weight_packed` / `weight_scale` buffers."""

  def __init__(self, in_features: int, out_features: int, device=None):
    super().__init__()
    if in_features % MXFP4_GROUP_SIZE:
      raise ValueError(f"in_features {in_features} must be a multiple of {MXFP4_GROUP_SIZE}")
    self.in_features = in_features
    self.out_features = out_features
    self.register_buffer("weight_packed", torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device))
    self.register_buffer(
        "weight_scale", torch.empty(out_features, in_features // MXFP4_GROUP_SIZE, dtype=torch.uint8, device=device)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(  # pylint: disable=not-callable
        x, dequantize_mxfp4_torch(self.weight_packed, self.weight_scale, x.dtype)
    )

  def extra_repr(self) -> str:
    return f"in_features={self.in_features}, out_features={self.out_features}, format=mxfp4"


def patch_experts_mxfp4(model: nn.Module) -> int:
  """Swaps every routed expert's `w1/w2/w3` for `Mxfp4Linear` in place; returns the count.

  Works on a model built under `torch.device("meta")` (the buffers are created on the same
  device as the weight they replace, to be filled by `load_state_dict(..., assign=True)`).
  """
  count = 0
  for module in model.modules():
    experts = getattr(module, "experts", None)
    if not isinstance(experts, nn.ModuleList):
      continue
    for expert in experts:
      for name in ("w1", "w2", "w3"):
        lin = getattr(expert, name, None)
        if not isinstance(lin, nn.Linear):
          continue
        setattr(expert, name, Mxfp4Linear(lin.in_features, lin.out_features, device=lin.weight.device))
        count += 1
  return count
