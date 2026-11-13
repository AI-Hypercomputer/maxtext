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

"""Plain-JAX rematerialization policy helpers and loop-barrier utilities."""

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
from typing import Any

from jax import ad_checkpoint
from jax import checkpoint_policies as cp
from jax.experimental import xla_metadata

# Supported `remat_policy` config values (case-insensitive).
REMAT_POLICIES: tuple[str, ...] = (
    "none",
    "full",
    "minimal",
    "minimal_with_context",
    "minimal_flash",
    "minimal_with_quantization",
    "minimal_with_context_and_quantization",
    "save_dot_with_context_except_mlp",
    "save_dot_except_mlpwi",
    "save_dot_except_mlp",
    "save_qkv_proj",
    "save_out_proj",
    "qkv_proj_offloaded",
    "minimal_offloaded",
    "matmul_without_batch",
    "offload_matmul_without_batch",
    "hidden_state_with_offload",
    "custom",
)


@dataclasses.dataclass(frozen=True)
class Remat:
  """Resolved checkpointing policy and expected activation tag names."""

  policy: Callable[..., Any] | None = cp.nothing_saveable
  names: tuple[str, ...] | None = None
  is_custom: bool = False
  saved_names: tuple[str, ...] = ()
  offloaded_names: tuple[str, ...] = ()


def policy_callable(remat: Remat | Callable[..., Any] | None) -> Callable[..., Any] | None:
  """Extracts the `jax.checkpoint` policy callable from `remat`, or `None` when remat is disabled."""
  if remat is None:
    return None
  if isinstance(remat, Remat):
    return cp.nothing_saveable if remat.policy is None else remat.policy
  if callable(remat):
    return remat
  raise TypeError(f"Expected Remat, callable policy, or None; got {remat!r}.")


def scan_safe_policy(remat: Remat | Callable[..., Any] | None) -> Callable[..., Any] | None:
  """Returns a policy callable keeping offloaded residuals on device across trip-count-one scans."""
  pol = policy_callable(remat)
  if pol is None or pol is cp.nothing_saveable:
    return pol

  def _safe_policy(prim: Any, *args: Any, **params: Any) -> Any:
    decision = pol(prim, *args, **params)
    if isinstance(decision, ad_checkpoint.Offloadable):
      return ad_checkpoint.Saveable
    return decision

  return _safe_policy


@contextlib.contextmanager
def trip_count_one_scan_barrier(enabled: bool = True) -> Iterator[None]:
  """Preserves a trip-count-1 `jax.lax.scan` while loop as an XLA remat memory barrier."""
  if enabled:
    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
      yield
  else:
    yield
