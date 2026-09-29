# Copyright 2026 Google LLC
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

"""Save-time fingerprints in `training_engine.checkpointing`: real Orbax I/O, real on-disk corruption."""

# pylint: disable=protected-access

import glob
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import jax

try:
  # Must precede backend initialization (the fixtures module below creates arrays at import).
  jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError:  # Backend already initialized by an earlier module; the sharding test skips.
  pass

# pylint: disable=wrong-import-position
from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.training_engine import checkpointing
import numpy as np
import tensorstore as ts
from tests.unit.training_engine_checkpoint_roundtrip_test import _build, _config, _train_step
# pylint: enable=wrong-import-position

_fp = checkpointing._tree_fingerprint


class CheckpointFingerprintTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)

  def _save_trained(self):
    """Saves 2 non-zero updates at step 2; returns the live model/optimizer."""
    model, optimizer = _build(seed=0)
    for _ in range(2):
      _train_step(model, optimizer)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.assertTrue(
        manager.save_checkpoint(step=2, checkpoint_state=checkpointing.CheckpointState(model=model, optimizer=optimizer))
    )
    manager.wait_until_finished()
    manager.close()
    return model, optimizer

  def _restore_into(self, model, optimizer=None):
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(manager.close)
    return manager.restore_checkpoint(checkpointing.CheckpointState(model=model, optimizer=optimizer))

  def test_round_trip_verifies_and_records_both_items(self):
    model, optimizer = self._save_trained()
    want = {
        "model_params": _fp(nnx.state(model)),
        "optimizer_state": _fp(nnx.state(optimizer, nnx.optimizer.OptState)),
    }
    fresh_model, fresh_opt = _build(seed=123)
    # Non-vacuous: the recorded value is not what an untrained model would produce ...
    self.assertNotEqual(want["model_params"], _fp(nnx.state(fresh_model)))

    step, _, metadata = self._restore_into(fresh_model, fresh_opt)

    self.assertEqual(step, 2)
    self.assertEqual(metadata[checkpointing._FINGERPRINT_KEY], want)
    # ... and the live state after restore hashes to it.
    self.assertEqual(_fp(nnx.state(fresh_model)), want["model_params"])
    self.assertEqual(_fp(nnx.state(fresh_opt, nnx.optimizer.OptState)), want["optimizer_state"])

  def test_single_element_bit_flip_on_disk_raises_naming_item(self):
    self._save_trained()
    (kernel_dir,) = glob.glob(os.path.join(self.ckpt_dir, "2", "model_params", "*linear1*kernel*"))
    # A valid, readable array whose bytes differ from what was saved: Orbax restores it happily.
    arr = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": kernel_dir}}).result()
    arr[1, 3] = np.asarray(arr[1, 3].read().result()) + np.float32(1e-3)
    fresh_model, _ = _build(seed=123)
    before = _fp(nnx.state(fresh_model))

    with self.assertLogs("absl", level="INFO") as logs:
      with self.assertRaisesRegex(checkpointing.CheckpointRestoreError, "'model_params' fingerprint"):
        self._restore_into(fresh_model)
    # Verification runs before `nnx.update`: the live model was not overwritten with corrupt bytes.
    self.assertEqual(_fp(nnx.state(fresh_model)), before)
    # A failed check must never log success, in any form.
    self.assertFalse([r for r in logs.records if r.getMessage().startswith("Verified checkpoint")])

  def test_checkpoint_without_fingerprints_still_restores(self):
    model, _ = self._save_trained()
    meta_path = os.path.join(self.ckpt_dir, "2", "_CHECKPOINT_METADATA")
    with open(meta_path, encoding="utf-8") as f:
      meta = json.load(f)
    del meta["custom_metadata"][checkpointing._FINGERPRINT_KEY]
    with open(meta_path, "w", encoding="utf-8") as f:
      json.dump(meta, f)
    fresh_model, fresh_opt = _build(seed=123)

    with self.assertLogs("absl", level="INFO") as logs:
      step, _, metadata = self._restore_into(fresh_model, fresh_opt)

    self.assertEqual(step, 2)
    self.assertNotIn(checkpointing._FINGERPRINT_KEY, metadata)
    self.assertEqual(_fp(nnx.state(fresh_model)), _fp(nnx.state(model)))
    # The 397B-B expected path (legacy 0922 checkpoint): check_logs.py greps exactly this line.
    self.assertEqual(len(_matches(logs, _LEGACY_RE)), 1)
    self.assertEqual(_matches(logs, _VERIFIED_RE), [])

  def test_save_and_verified_lines_carry_the_recomputed_fingerprints(self):
    with self.assertLogs("absl", level="INFO") as save_logs:
      model, optimizer = self._save_trained()
    want = (_fp(nnx.state(model)), _fp(nnx.state(optimizer, nnx.optimizer.OptState)))
    (saved,) = _matches(save_logs, _SAVE_RE)
    self.assertEqual(saved.group("step"), "2")
    self.assertEqual(_hex(saved, "model_params", "optimizer_state"), want)

    fresh_model, fresh_opt = _build(seed=123)
    with self.assertLogs("absl", level="INFO") as restore_logs:
      self._restore_into(fresh_model, fresh_opt)
    (verified,) = _matches(restore_logs, _VERIFIED_RE)
    self.assertEqual(verified.group("step"), "2")
    # Values are those recomputed from the restored bytes (== live state) and match the save line verbatim.
    self.assertEqual(
        _hex(verified, "model_params", "optimizer_state"),
        (_fp(nnx.state(fresh_model)), _fp(nnx.state(fresh_opt, nnx.optimizer.OptState))),
    )
    self.assertEqual(verified.group("fps"), saved.group("fps"))
    self.assertEqual(_matches(restore_logs, _LEGACY_RE), [])

  def test_verified_line_lists_only_items_actually_verified(self):
    model, _ = self._save_trained()  # Saved with an optimizer ...
    fresh_model, _ = _build(seed=123)
    with self.assertLogs("absl", level="INFO") as logs:
      self._restore_into(fresh_model)  # ... restored without one: optimizer_state is never read.
    (verified,) = _matches(logs, _VERIFIED_RE)
    self.assertEqual(verified.group("fps"), f"model_params={_fp(nnx.state(model)):#010x}")


_FPS = r"(?P<fps>(?:\w+=0x[0-9a-f]{8})(?: \w+=0x[0-9a-f]{8})*)"
_SAVE_RE = re.compile(rf"^Checkpoint fingerprints_v1 step=(?P<step>\d+) {_FPS} \(\d+\.\d{{2}}s\)$")
_VERIFIED_RE = re.compile(
    rf"^Verified checkpoint fingerprints_v1 step=(?P<step>\d+) {_FPS} \(restore \d+\.\ds, verify \d+\.\d{{2}}s\)$"
)
_LEGACY_RE = re.compile(
    r"^Checkpoint at step 2 predates save-time fingerprints; skipping verification \(restore \d+\.\ds\)\.$"
)


def _matches(logs, pattern):
  return [m for m in (pattern.match(r.getMessage()) for r in logs.records) if m]


def _hex(match, *items):
  values = dict(pair.split("=") for pair in match.group("fps").split(" "))
  assert list(values) == list(items), values
  return tuple(int(values[item], 16) for item in items)


class TreeFingerprintTest(unittest.TestCase):

  def _tree(self):
    return {
        # bf16 with a trailing 1, like the 397B `shared_expert_gate.kernel` [4096, 15, 1].
        "gate": jnp.linspace(-2.0, 3.0, 8 * 3, dtype=jnp.bfloat16).reshape(8, 3, 1),
        "w": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) * 0.37,
        "count": jnp.arange(8, dtype=jnp.int32),
        "key": jax.random.split(jax.random.key(7), 8),
        "is_skipped": jnp.array([True, False] * 4),  # bool, as in the skip_step_on_spikes opt state
    }

  @unittest.skipIf(jax.device_count() < 4, "needs 4 CPU devices; run this file on its own")
  def test_invariant_to_sharding(self):
    tree = self._tree()
    mesh = Mesh(np.array(jax.devices()[:4]), ("x",))
    sharded = jax.device_put(tree, NamedSharding(mesh, P("x")))
    self.assertEqual(len(sharded["w"].sharding.device_set), 4)

    self.assertEqual(_fp(sharded), _fp(tree))

  def test_detects_swapped_leaves_moved_elements_and_resized_zero_tensors(self):
    tree = self._tree()
    a, b = jnp.full((8, 4), 1.5), jnp.full((8, 4), -0.25)

    self.assertNotEqual(_fp({"wi_0": a, "wi_1": b}), _fp({"wi_0": b, "wi_1": a}))
    # Two shards' rows written to each other's slots: same multiset of values, different layout.
    self.assertNotEqual(_fp(dict(tree, w=tree["w"][jnp.array([4, 5, 6, 7, 0, 1, 2, 3])])), _fp(tree))
    self.assertNotEqual(_fp({"z": jnp.zeros(4)}), _fp({"z": jnp.zeros(8)}))
    flipped = dict(tree, gate=tree["gate"].at[5, 1, 0].set(tree["gate"][5, 1, 0] + 1))
    self.assertNotEqual(_fp(flipped), _fp(tree))
    self.assertNotEqual(_fp(dict(tree, is_skipped=~tree["is_skipped"])), _fp(tree))


if __name__ == "__main__":
  unittest.main()
