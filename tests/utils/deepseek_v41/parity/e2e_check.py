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

"""DeepSeek-V4.1 e2e output policy wrapper enforcing CPU logits ceilings and row registry."""

from collections.abc import Mapping
import dataclasses
import types

from tests.utils.parity import e2e_check as _base

TokensTol = _base.TokensTol
RouterAgreementTol = _base.RouterAgreementTol
OutputTol = _base.OutputTol
check_router_agreement = _base.check_router_agreement
compare_one = _base.compare_one
check_outputs = _base.check_outputs
negative_run_status = _base.negative_run_status

# Approved DeepSeek-V4.1 CPU full-model fp32 logits parity ceilings (KL divergence and top-1 agreement).
LOGITS_KL_CEILING = 1e-6
LOGITS_TOP1_FLOOR = 1.0


@dataclasses.dataclass(frozen=True, init=False)
class LogitsTol(_base.LogitsTol):
  """DeepSeek-V4.1 `LogitsTol` requiring `kl_max` and enforcing CPU logits ceilings."""

  def __init__(self, kl_mean_max: float, kl_max: float, top1_min: float) -> None:
    super().__init__(kl_mean_max=kl_mean_max, top1_min=top1_min, kl_max=kl_max)
    if self.kl_mean_max > LOGITS_KL_CEILING or kl_max > LOGITS_KL_CEILING or self.top1_min < LOGITS_TOP1_FLOOR:
      raise ValueError(
          "LogitsTol exceeds approved CPU logits ceilings "
          f"(kl_mean_max <= {LOGITS_KL_CEILING}, kl_max <= {LOGITS_KL_CEILING}, "
          f"top1_min >= {LOGITS_TOP1_FLOOR}): {self}"
      )


CPU_LOGITS_TOL = LogitsTol(
    kl_mean_max=LOGITS_KL_CEILING,
    kl_max=LOGITS_KL_CEILING,
    top1_min=LOGITS_TOP1_FLOOR,
)

# Immutable registry mapping DeepSeek-V4.1 check row IDs to approved LogitsTol bounds.
# Add new rows by editing this MappingProxyType directly.
_APPROVED_LOGITS_ROWS: Mapping[str, LogitsTol] = types.MappingProxyType(
    {
        "T13": CPU_LOGITS_TOL,
    }
)


def logits_tol_for_row(row_id: str) -> LogitsTol:
  """Returns the registered `LogitsTol` for DeepSeek-V4.1 check row `row_id`."""
  if row_id in _APPROVED_LOGITS_ROWS:
    return _APPROVED_LOGITS_ROWS[row_id]
  registered = ", ".join(sorted(_APPROVED_LOGITS_ROWS))
  raise ValueError(f"Unregistered logits tolerance row {row_id!r}; registered rows: [{registered}]")
