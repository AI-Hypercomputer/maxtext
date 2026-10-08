# Copyright 2026 Google LLC
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

"""Deals whole test classes to CI shards.

A test's duration depends on which test ran before it on the same shard: the first test of a
compile family pays the XLA compile, later ones reuse it. pytest-split deals single tests and its
deal changes whenever the selected tests or the recorded durations change, so a different test
pays in the nightly run and in a PR run and the per-test duration check fires falsely. Dealing
whole classes keeps that cost on the same test in every run.

Used through the `--class-splits N --class-group G` options registered in tests/conftest.py. Both
options take integers only: a path-valued option would make pytest treat the value as a test path
during early argument parsing, and tests/conftest.py would then not be loaded in time to register
the options. The durations are read from DURATIONS_FILE in the working directory, the file the CI
workflow downloads next to the checkout.
"""

import json
import os

import pytest

# `{nodeid: seconds}` written by tools/dev/convert_durations.py from the nightly baseline.
DURATIONS_FILE = ".test_durations"


def group_key(nodeid):
  """Returns the split unit a nodeid belongs to: its class, or the function itself for module-level tests.

  'tests/unit/moe_test.py::RoutedMoeTest::test_x[p]' -> 'tests/unit/moe_test.py::RoutedMoeTest'
  'tests/unit/moe_test.py::test_fn[p]' -> 'tests/unit/moe_test.py::test_fn'
  Parameters are stripped first so that a parameter value containing '::' cannot shift the key.
  """
  base = nodeid
  if base.endswith("]") and "[" in base:
    base = base[: base.index("[")]
  parts = base.split("::")
  if len(parts) < 2:
    return parts[0]
  return f"{parts[0]}::{parts[1]}"


def load_durations(path):
  """Reads a `{nodeid: seconds}` JSON file.

  Returns `(durations, warning)`; `warning` is empty when the file was usable. A missing or
  unreadable file is not fatal: the caller falls back to equal weights, which still gives a
  deterministic deal that keeps classes together.
  """
  if not os.path.isfile(path):
    return {}, f"durations file {path!r} not found; using equal weights"
  try:
    with open(path, "r", encoding="utf-8") as f:
      raw = json.load(f)
  except (OSError, ValueError) as e:
    return {}, f"durations file {path!r} could not be read ({e}); using equal weights"
  if not isinstance(raw, dict):
    return {}, f"durations file {path!r} is not a JSON object; using equal weights"
  durations = {}
  for key, value in raw.items():
    try:
      durations[str(key)] = float(value)
    except (TypeError, ValueError):
      continue
  if not durations:
    return {}, f"durations file {path!r} holds no usable entries; using equal weights"
  return durations, ""


def _weights_and_members(nodeids, durations):
  """Returns `(weight per group key, members per group key as (collection index, nodeid))`.

  An item with no recorded duration is charged the mean of the durations recorded for the items
  being dealt (1.0 when none are recorded), so entries for tests that were deselected before the
  deal cannot influence it.
  """
  recorded = [durations[n] for n in nodeids if n in durations]
  fallback = sum(recorded) / len(recorded) if recorded else 1.0
  weights = {}
  members = {}
  for index, nodeid in enumerate(nodeids):
    key = group_key(nodeid)
    weights[key] = weights.get(key, 0.0) + durations.get(nodeid, fallback)
    members.setdefault(key, []).append((index, nodeid))
  return weights, members


def deal_groups(nodeids, durations, splits):
  """Deals whole group keys into `splits` shards, heaviest key first.

  Deterministic: keys are ordered by (-weight, key), each key goes to the currently lightest shard
  with ties broken by the lowest shard index, and inside a shard the items keep their collection
  order. Returns one list of nodeids per shard.
  """
  if splits < 1:
    raise ValueError(f"splits must be >= 1, got {splits}")
  weights, members = _weights_and_members(nodeids, durations)
  totals = [0.0] * splits
  shards = [[] for _ in range(splits)]
  for key in sorted(weights, key=lambda k: (-weights[k], k)):
    target = min(range(splits), key=lambda i: (totals[i], i))
    totals[target] += weights[key]
    shards[target].extend(members[key])
  return [[nodeid for _, nodeid in sorted(shard)] for shard in shards]


class ClassSplitPlugin:
  """Keeps only the classes dealt to this shard. Registered by tests/conftest.py."""

  def __init__(self, config, durations_path=DURATIONS_FILE):
    self.splits = config.getoption("class_splits")
    self.group = config.getoption("class_group")
    self.durations_path = durations_path
    if self.splits is None or self.group is None:
      raise pytest.UsageError("--class-splits and --class-group must be given together")
    if self.splits < 1:
      raise pytest.UsageError(f"--class-splits must be >= 1, got {self.splits}")
    if not 1 <= self.group <= self.splits:
      raise pytest.UsageError(f"--class-group must be between 1 and {self.splits}, got {self.group}")

  @staticmethod
  def _write_line(config, text):
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
      reporter.write_line(text)
    else:  # xdist worker, or -p no:terminal
      print(text)

  # trylast: run after pytest's -m deselection and after the marker hooks in tests/conftest.py.
  @pytest.hookimpl(trylast=True)
  def pytest_collection_modifyitems(self, config, items):
    """Keeps only the items dealt to this shard's group, deselecting the rest."""
    if not items:
      return
    durations, warning = load_durations(self.durations_path)
    if warning:
      self._write_line(config, f"[class-split] warning: {warning}")
    shards = deal_groups([item.nodeid for item in items], durations, self.splits)
    keep = set(shards[self.group - 1])
    kept = [item for item in items if item.nodeid in keep]
    dropped = [item for item in items if item.nodeid not in keep]
    items[:] = kept
    if dropped:
      config.hook.pytest_deselected(items=dropped)
    classes = len({group_key(item.nodeid) for item in kept})
    recorded = sum(1 for item in kept if item.nodeid in durations)
    self._write_line(
        config,
        f"[class-split] Running group {self.group}/{self.splits}: {len(kept)} tests in {classes} classes, "
        f"{recorded} with a recorded duration",
    )
