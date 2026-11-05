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

"""Metrics aggregation for multi-turn coding RL (`code_rl/*` and `judge/*`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def compute_code_rl_metrics(
    turn1_passed: Sequence[bool],
    final_passed: Sequence[bool],
    assistant_lengths: Sequence[int],
    max_response_length: int = 8192,
) -> dict[str, float]:
  """Computes multi-turn coding RL solve rates, repair gain, and token length stats.

  Args:
    turn1_passed: Boolean pass indicator at Turn 1 for each trajectory.
    final_passed: Boolean pass indicator at the final turn for each trajectory.
    assistant_lengths: Total assistant response tokens per trajectory.
    max_response_length: Response token budget used to compute truncation rate.

  Returns:
    Dictionary with keys:
      - `code_rl/turn1_solve`
      - `code_rl/final_solve`
      - `code_rl/repair_gain` (`final_solve - turn1_solve`)
      - `code_rl/mean_assistant_tokens`
      - `code_rl/p90_assistant_tokens`
      - `code_rl/truncation_rate`
  """
  t1 = np.asarray(turn1_passed, dtype=np.float32)
  fin = np.asarray(final_passed, dtype=np.float32)
  lens = np.asarray(assistant_lengths, dtype=np.float32)
  if len(fin) == 0:
    return {
        "code_rl/turn1_solve": 0.0,
        "code_rl/final_solve": 0.0,
        "code_rl/repair_gain": 0.0,
        "code_rl/mean_assistant_tokens": 0.0,
        "code_rl/p90_assistant_tokens": 0.0,
        "code_rl/truncation_rate": 0.0,
    }
  if len(t1) != len(fin) or len(lens) != len(fin):
    raise ValueError("turn1_passed, final_passed, and assistant_lengths must have equal length")
  t1_solve = float(np.mean(t1))
  fin_solve = float(np.mean(fin))
  return {
      "code_rl/turn1_solve": t1_solve,
      "code_rl/final_solve": fin_solve,
      "code_rl/repair_gain": fin_solve - t1_solve,
      "code_rl/mean_assistant_tokens": float(np.mean(lens)),
      "code_rl/p90_assistant_tokens": float(np.percentile(lens, 90)),
      "code_rl/truncation_rate": float(np.mean(lens >= max_response_length)),
  }


def extract_trajectory_code_rl_metrics(
    trajectories: Sequence[Mapping[str, Any] | Any],
    max_response_length: int = 8192,
) -> dict[str, float]:
  """Extracts `code_rl/*` metrics directly from a batch of agentic trajectories."""
  turn1_passed: list[bool] = []
  final_passed: list[bool] = []
  assistant_lengths: list[int] = []

  for item in trajectories:
    traj = item.traj if hasattr(item, "traj") else item
    if not isinstance(traj, Mapping):
      continue
    extras = traj.get("env_extras") or {}
    turn_records = list(extras.get("turn_records") or [])
    if turn_records:
      first_tr = turn_records[0]
      if hasattr(first_tr, "test_result") and hasattr(first_tr.test_result, "all_passed"):
        t1_pass = bool(first_tr.test_result.all_passed)
      elif isinstance(first_tr, Mapping):
        tr_res = first_tr.get("test_result") or {}
        t1_pass = bool(tr_res.get("all_passed", False) if isinstance(tr_res, Mapping) else False)
      else:
        t1_pass = False
    else:
      t1_pass = False

    judge_meta = traj.get("judge")
    if isinstance(judge_meta, Mapping) and "passed" in judge_meta:
      fin_pass = bool(judge_meta["passed"])
    elif "passed" in extras:
      fin_pass = bool(extras["passed"])
    else:
      fin_pass = float(traj.get("trajectory_reward", 0.0)) > 0.0

    if isinstance(judge_meta, Mapping) and "assistant_len" in judge_meta:
      assist_len = int(judge_meta["assistant_len"])
    else:
      conv_masks = traj.get("conversation_masks")
      if conv_masks is not None:
        assist_len = int(np.sum(np.asarray(conv_masks) != 0))
      else:
        conv_tokens = traj.get("conversation_tokens") or []
        assist_len = len(conv_tokens)

    turn1_passed.append(t1_pass)
    final_passed.append(fin_pass)
    assistant_lengths.append(assist_len)

  return compute_code_rl_metrics(
      turn1_passed=turn1_passed,
      final_passed=final_passed,
      assistant_lengths=assistant_lengths,
      max_response_length=max_response_length,
  )


def get_code_rl_metrics(
    prompts: Any = None,
    completions: Any = None,
    rewards: np.ndarray | Sequence[float] = (),
    advantages: Any = None,
    max_response_length: int = 8192,
    **kwargs: Any,
) -> dict[str, tuple[float | int, Callable[..., Any]]]:
  """Tunix-compatible custom metric function for coding RL runs."""
  del prompts, advantages
  rewards_arr = np.asarray(rewards, dtype=np.float64)
  if rewards_arr.size > 0:
    solve_all = 1 if bool(np.all(rewards_arr > 0.1)) else 0
    solve_none = 1 if bool(np.all(rewards_arr <= 1e-6)) else 0
    solve_partial = 1 if (not solve_all and not solve_none) else 0
    solve_ratio = float(np.mean(rewards_arr > 0.1))
  else:
    solve_all = 0
    solve_none = 0
    solve_partial = 0
    solve_ratio = 0.0

  max_response_length = int(kwargs.get("max_response_length", max_response_length))
  if "trajectories" in kwargs and kwargs["trajectories"] is not None:
    code_metrics = extract_trajectory_code_rl_metrics(kwargs["trajectories"], max_response_length=max_response_length)
  else:
    final_passed = list(kwargs.get("final_passed", [bool(r > 0.1) for r in rewards_arr.ravel()]))
    turn1_passed = list(kwargs.get("turn1_passed", final_passed))
    if "assistant_lengths" in kwargs and kwargs["assistant_lengths"] is not None:
      assistant_lengths = [int(x) for x in kwargs["assistant_lengths"]]
    elif completions is not None:
      assistant_lengths = [len(str(c).split()) if isinstance(c, str) else len(c) for c in completions]
    else:
      assistant_lengths = [0 for _ in range(len(final_passed))]
    code_metrics = compute_code_rl_metrics(
        turn1_passed=turn1_passed,
        final_passed=final_passed,
        assistant_lengths=assistant_lengths,
        max_response_length=max_response_length,
    )

  out: dict[str, tuple[float | int, Callable[..., Any]]] = {
      "rewards/solve_all": (solve_all, np.mean),
      "rewards/solve_none": (solve_none, np.mean),
      "rewards/solve_partial": (solve_partial, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
      "reward/turn1_solve": (float(code_metrics["code_rl/turn1_solve"]), np.mean),
      "reward/final_solve": (float(code_metrics["code_rl/final_solve"]), np.mean),
      "reward/repair_gain": (float(code_metrics["code_rl/repair_gain"]), np.mean),
      "length/assistant_tokens_mean": (float(code_metrics["code_rl/mean_assistant_tokens"]), np.mean),
      "length/assistant_tokens_p90": (float(code_metrics["code_rl/p90_assistant_tokens"]), np.mean),
      "length/truncation_rate": (float(code_metrics["code_rl/truncation_rate"]), np.mean),
      "length/truncation_rate_at_8192": (float(code_metrics["code_rl/truncation_rate"]), np.mean),
  }
  for key, val in code_metrics.items():
    out[key] = (float(val), np.mean)
  return out
