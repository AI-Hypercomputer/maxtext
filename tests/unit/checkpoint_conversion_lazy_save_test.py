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

"""Goalpost: `to_maxtext --lazy_load_tensors` must stay lazy through the checkpoint save.

`LazyTensor` leaves are meant to be loaded just-in-time as Orbax writes them, so that
converting a model never needs the whole checkpoint in host RAM. Today the single-device
save path materializes every leaf up front (`materialize_lazy_weights`, added because the
Orbax v1 checkpointer rejects the proxies), which put the 93-layer Kimi-K3 convert at an
estimated ~4 TB peak. These tests pin down the required behaviour:

  * correctness: a tree of `LazyTensor`s (plus eager leaves, incl. uint8) saved through
    `save_weights_to_checkpoint` restores bit-exactly through the unmodified MaxText
    loader `checkpointing.load_params_from_path`;
  * laziness: peak memory during the save stays near the write budget
    (`checkpoint_storage_concurrent_gb`), far below the lazy total. Measured two ways,
    because any copy of a loaded leaf (e.g. `np.array(leaf)`) escapes a tracker that only
    watches the loader's own return value:
      - `tracemalloc` peak: numpy reports its data buffers to tracemalloc, so every numpy
        copy is counted; deterministic, but blind to C++-side (XLA / TensorStore) buffers;
      - Linux peak-RSS delta (`VmHWM` after resetting it via `/proc/self/clear_refs`):
        counts every allocation, noisier, so it uses a looser bound;
  * every lazy leaf is loaded exactly once, and none are still alive after the save;
  * leaves are loaded concurrently (loading is CPU/IO-bound per leaf, so a serial
    loader makes large converts needlessly slow).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
import weakref

import jax
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext import LazyTensor
from maxtext.checkpoint_conversion.utils import utils as conv_utils
from maxtext.common import checkpointing

_NUM_LAZY = 32
_LEAF_SHAPE = (2048, 1024)  # 8 MiB of float32 per leaf -> 256 MiB of lazy leaves in total
_LEAF_BYTES = int(np.prod(_LEAF_SHAPE)) * 4
_LAZY_TOTAL_BYTES = _NUM_LAZY * _LEAF_BYTES
_BUDGET_BYTES = 4 * _LEAF_BYTES  # write budget: 4 leaves (32 MiB), i.e. 1/8 of the lazy total
# tracemalloc bound: budget, plus one leaf in flight, plus one transient copy of it.
_TRACEMALLOC_LIMIT_BYTES = _BUDGET_BYTES + 2 * _LEAF_BYTES
# RSS bound: three quarters of the lazy total. Eager materialization peaks at >= the lazy total,
# so this still catches it, while leaving headroom for allocator noise (per-thread malloc
# arenas of the loader pool, tracemalloc's own bookkeeping), which grows when this test shares a
# process with other suites. `_TRACEMALLOC_LIMIT_BYTES` is the tight bound.
_RSS_LIMIT_BYTES = _LAZY_TOTAL_BYTES * 3 // 4
_MIB = 2**20


def _reset_peak_rss() -> bool:
  """Resets this process's VmHWM to its current RSS (Linux >= 4.0). Returns success."""
  if not sys.platform.startswith("linux"):
    return False
  try:
    with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
      f.write("5")
    return True
  except OSError:
    return False


def _proc_status_kib(field: str) -> int:
  with open("/proc/self/status", encoding="ascii") as f:
    for line in f:
      if line.startswith(field + ":"):
        return int(line.split()[1])
  raise KeyError(field)


class _LiveBytesTracker:
  """Counts loads of each lazy leaf and bytes of loader-returned arrays still alive."""

  def __init__(self):
    self._lock = threading.Lock()
    self.live = 0
    self.peak = 0
    self.loads: dict[str, int] = {}
    self.active = 0
    self.max_active = 0

  def begin_load(self) -> None:
    with self._lock:
      self.active += 1
      self.max_active = max(self.max_active, self.active)

  def end_load(self) -> None:
    with self._lock:
      self.active -= 1

  def _release(self, nbytes: int) -> None:
    with self._lock:
      self.live -= nbytes

  def track(self, name: str, arr: np.ndarray) -> np.ndarray:
    with self._lock:
      self.live += arr.nbytes
      self.peak = max(self.peak, self.live)
      self.loads[name] = self.loads.get(name, 0) + 1
    weakref.finalize(arr, self._release, arr.nbytes)
    return arr


def _expected(i: int) -> np.ndarray:
  # Generate float32 directly: a float64 temporary would inflate the tracemalloc peak.
  return np.random.default_rng(i).standard_normal(_LEAF_SHAPE, dtype=np.float32)


def _build_tree(tracker: _LiveBytesTracker):
  """MaxText-shaped params: lazy float32 layer weights plus eager embedder / uint8 leaves."""
  layers = {}
  for i in range(_NUM_LAZY):
    name = f"layers_{i}"

    def load(i=i, name=name):
      tracker.begin_load()
      try:
        arr = _expected(i)
        time.sleep(0.02)  # stand-in for per-leaf read/transform latency
      finally:
        tracker.end_load()
      return tracker.track(name, arr)

    layers[name] = {"kernel": LazyTensor(load, _LEAF_SHAPE, np.float32, name=name)}
  eager = {
      "embedding": np.random.default_rng(1000).standard_normal((64, 32)).astype(np.float32),
      "packed": np.random.default_rng(1001).integers(0, 256, (16, 8), dtype=np.uint8),
  }
  return {"decoder": layers, "token_embedder": eager}


def _expected_tree():
  return {
      "decoder": {f"layers_{i}": {"kernel": _expected(i)} for i in range(_NUM_LAZY)},
      "token_embedder": {
          "embedding": np.random.default_rng(1000).standard_normal((64, 32)).astype(np.float32),
          "packed": np.random.default_rng(1001).integers(0, 256, (16, 8), dtype=np.uint8),
      },
  }


def _abstract(tree):
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding), tree)


def _save(path, weights):
  conv_utils.save_weights_to_checkpoint(
      path,
      weights,
      device_count=1,
      use_ocdbt=True,
      use_zarr3=True,
      checkpoint_storage_concurrent_gb=_BUDGET_BYTES / 1e9,
  )


class LazyCheckpointSaveTest(unittest.TestCase):
  """Saves one lazy tree per test class and checks restore, memory and load counts."""

  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    cls.out = os.path.join(cls.tmp.name, "ckpt")
    cls.tracker = _LiveBytesTracker()
    # Warm-up save of a tiny eager tree so one-time Orbax / TensorStore / thread-pool
    # initialization is not charged to the measured save.
    _save(os.path.join(cls.tmp.name, "warmup"), {"w": np.zeros((8, 8), np.float32)})

    weights = _build_tree(cls.tracker)
    cls.rss_measured = _reset_peak_rss()
    rss_before_kib = _proc_status_kib("VmRSS") if cls.rss_measured else 0
    tracemalloc.start()
    try:
      _save(cls.out, weights)
      _, cls.tracemalloc_peak = tracemalloc.get_traced_memory()
    finally:
      tracemalloc.stop()
    cls.rss_peak_delta = (_proc_status_kib("VmHWM") - rss_before_kib) * 1024 if cls.rss_measured else 0

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_restores_bit_exact_through_maxtext_loader(self):
    want = _expected_tree()
    # The converter saves `params={"params": weights}`; a Linen-style abstract names the collection.
    got = checkpointing.load_params_from_path(
        os.path.join(self.out, "0", "items"), _abstract({"params": want}), 1, True, True
    )["params"]
    flat_got, _ = jax.tree_util.tree_flatten_with_path(got)
    flat_want = dict(jax.tree_util.tree_flatten_with_path(want)[0])
    self.assertEqual(len(flat_got), len(flat_want))
    for path, arr in flat_got:
      np.testing.assert_array_equal(np.asarray(arr), flat_want[path], err_msg=jax.tree_util.keystr(path))
      self.assertEqual(np.asarray(arr).dtype, flat_want[path].dtype, jax.tree_util.keystr(path))

  def test_tracemalloc_peak_bounded_by_write_budget(self):
    self.assertLessEqual(
        self.tracemalloc_peak,
        _TRACEMALLOC_LIMIT_BYTES,
        f"tracemalloc peak {self.tracemalloc_peak / _MIB:.0f} MiB during save of "
        f"{_LAZY_TOTAL_BYTES / _MIB:.0f} MiB of lazy leaves; limit {_TRACEMALLOC_LIMIT_BYTES / _MIB:.0f} MiB "
        f"(budget {_BUDGET_BYTES / _MIB:.0f} MiB + 2 leaves). The save is materializing eagerly.",
    )

  def test_peak_rss_delta_well_below_lazy_total(self):
    if not self.rss_measured:
      self.skipTest("peak-RSS reset via /proc/self/clear_refs unavailable (Linux only)")
    self.assertLessEqual(
        self.rss_peak_delta,
        _RSS_LIMIT_BYTES,
        f"peak RSS grew {self.rss_peak_delta / _MIB:.0f} MiB during save of "
        f"{_LAZY_TOTAL_BYTES / _MIB:.0f} MiB of lazy leaves; limit {_RSS_LIMIT_BYTES / _MIB:.0f} MiB. "
        "The save is materializing eagerly.",
    )

  def test_each_lazy_leaf_loaded_exactly_once(self):
    self.assertEqual(set(self.tracker.loads), {f"layers_{i}" for i in range(_NUM_LAZY)})
    self.assertEqual(set(self.tracker.loads.values()), {1}, self.tracker.loads)

  def test_all_lazy_leaves_released_after_save(self):
    self.assertEqual(self.tracker.live, 0)

  def test_lazy_leaves_load_concurrently(self):
    self.assertGreater(self.tracker.max_active, 1, "lazy leaves were loaded one at a time")


_SUBPROCESS_SAVE = r"""
import sys
import numpy as np
from maxtext.checkpoint_conversion.to_maxtext import LazyTensor
from maxtext.checkpoint_conversion.utils import utils as conv_utils

want = np.arange(64, dtype=np.float32).reshape(8, 8)
weights = {  # every leaf lazy: no eager leaf to set up the save's commit plumbing
    "a": LazyTensor(want.copy, (8, 8), np.float32, name="a"),
    "b": LazyTensor(lambda: want.copy() + 1, (8, 8), np.float32, name="b"),
}
conv_utils.save_weights_to_checkpoint(sys.argv[1], weights, device_count=1, use_ocdbt=True, use_zarr3=True)
"""

_SUBPROCESS_LOAD = r"""
import sys
import jax
import numpy as np
from maxtext.common import checkpointing

assert "maxtext.checkpoint_conversion.utils.utils" not in sys.modules
sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
spec = jax.ShapeDtypeStruct((8, 8), np.float32, sharding=sharding)
got = checkpointing.load_params_from_path(sys.argv[1] + "/0/items", {"params": {"a": spec, "b": spec}}, 1, True, True)
want = np.arange(64, dtype=np.float32).reshape(8, 8)
np.testing.assert_array_equal(np.asarray(got["params"]["a"]), want)
np.testing.assert_array_equal(np.asarray(got["params"]["b"]), want + 1)
print("LOAD_OK")
"""


class LazyCheckpointFreshProcessTest(unittest.TestCase):
  """Save and load each in a fresh interpreter, as `to_maxtext` and a training job would.

  In-process tests can pass by accident (e.g. state left behind by an earlier save);
  this catches hangs in an all-lazy save and checkpoints that only load when the
  conversion code is imported.
  """

  def _run(self, script, *args):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    return subprocess.run(
        [sys.executable, "-c", script, *args], env=env, capture_output=True, text=True, timeout=300, check=False
    )

  def test_all_lazy_tree_saves_and_loads_with_stock_handler(self):
    with tempfile.TemporaryDirectory() as tmp:
      out = os.path.join(tmp, "ckpt")
      saved = self._run(_SUBPROCESS_SAVE, out)
      self.assertEqual(saved.returncode, 0, saved.stderr[-3000:])
      loaded = self._run(_SUBPROCESS_LOAD, out)
      self.assertEqual(loaded.returncode, 0, loaded.stderr[-3000:])
      self.assertIn("LOAD_OK", loaded.stdout)
      self.assertNotIn("Failed to resolve handler", loaded.stderr)


if __name__ == "__main__":
  unittest.main()
