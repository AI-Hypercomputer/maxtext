# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remat (gradient checkpointing) policies selected by config name."""

from typing import Callable, Optional, Sequence

import jax
from jax import checkpoint_policies as cp

# `remat_policy` config values (case-insensitive).
REMAT_POLICIES = (
    "none",
    "full",
    "matmul_without_batch",
    "offload_matmul_without_batch",
    "custom",
    "hidden_state_with_offload",
)


def remat_policy(
    name: Optional[str],
    names_which_can_be_saved: Optional[Sequence[str]] = None,
    names_which_can_be_offloaded: Optional[Sequence[str]] = None,
) -> Optional[Callable]:
  """Returns the `jax.checkpoint` policy for a `remat_policy` config value, or None for no remat.

  Models take the returned policy at construction and wrap their per-layer forward with
  `nnx.remat(..., policy=policy)` when it is not None (path-transparent, so parameter paths are unchanged).

  Args:
    name: one of `REMAT_POLICIES` (None means "none").
    names_which_can_be_saved: `checkpoint_name` tags kept on device ("custom" only).
    names_which_can_be_offloaded: `checkpoint_name` tags offloaded to pinned host memory ("custom" only).
  """
  name = "none" if name is None else name.lower()
  saved = list(names_which_can_be_saved or [])
  offloaded = list(names_which_can_be_offloaded or [])
  match name:
    case "none":
      return None
    case "full":  # save nothing, recompute the whole layer in the backward pass
      return cp.nothing_saveable
    case "matmul_without_batch":
      return cp.checkpoint_dots_with_no_batch_dims
    case "offload_matmul_without_batch":
      return cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host")
    case "custom":
      return cp.save_and_offload_only_these_names(
          names_which_can_be_saved=saved,
          names_which_can_be_offloaded=offloaded,
          offload_src="device",
          offload_dst="pinned_host",
      )
    case "hidden_state_with_offload":
      return jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=["hidden_states", "self_attn", "cross_attn"],
          offload_src="device",
          offload_dst="pinned_host",
      )
  raise ValueError(f"Unknown remat_policy {name!r}; expected one of {REMAT_POLICIES}.")
