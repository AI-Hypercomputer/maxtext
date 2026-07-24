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

"""Unit tests for threaded DiLoCo components."""

import os
import re
import sys
import unittest
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # pylint: disable=wrong-import-order
import jax.numpy as jnp
import optax

from maxtext.configs import pyconfig
from maxtext.trainers.diloco.threaded_diloco import (
    _delayed_response_step,
    _reshard_tree,
    _save_checkpoint_serialized,
    _validate_checkpoint_alignment,
    make_learner_config,
    make_step_fns,
    make_streaming_mean_fns,
    stream_learner_mean,
)
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager
from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator


class ThreadedDilocoUnitTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Need to add src to path if not already there, but maxtext imports usually assume src is in path.
    # We will initialize config with base.yml
    self.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="test",
        enable_diloco=True,
        enable_streaming_diloco=True,
        num_diloco_replicas=2,
    )

  def test_make_learner_config(self):
    learner_config = make_learner_config(self.config, learner_idx=1, num_learners=2)

    # Check that diloco is removed from mesh_axes
    self.assertNotIn("diloco", learner_config.mesh_axes)

    # Check logical_axis_rules
    for _, physical_axes in learner_config.logical_axis_rules:
      if isinstance(physical_axes, str):
        self.assertNotEqual(physical_axes, "diloco")
      elif isinstance(physical_axes, (list, tuple)):
        self.assertNotIn("diloco", physical_axes)

    # Check other flags
    self.assertTrue(learner_config.enable_local_data_loading)
    self.assertEqual(learner_config.learner_idx, 1)
    self.assertEqual(learner_config.num_learners, 2)
    self.assertFalse(learner_config.enable_streaming_diloco)
    self.assertFalse(learner_config.enable_diloco)

  def test_transport_manager_basic(self):
    manager = ThreadedTransportManager(num_learners=2)

    # Test learner to syncer
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="l0_s1_f1")
    manager.send_to_syncer(learner_idx=1, step=1, fragment_id=1, data="l1_s1_f1")

    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "l0_s1_f1")
    self.assertEqual(manager.recv_from_learner(learner_idx=1, step=1, fragment_id=1), "l1_s1_f1")

    # Test syncer to learner
    manager.send_to_learner(learner_idx=0, step=1, fragment_id=1, data="s_l0_s1_f1")
    manager.send_to_learner(learner_idx=1, step=1, fragment_id=1, data="s_l1_s1_f1")

    self.assertEqual(manager.recv_from_syncer(learner_idx=0, step=1, fragment_id=1), "s_l0_s1_f1")
    self.assertEqual(manager.recv_from_syncer(learner_idx=1, step=1, fragment_id=1), "s_l1_s1_f1")

  def test_transport_manager_blocking(self):
    manager = ThreadedTransportManager(num_learners=1)
    results = {}

    def worker():
      results["data"] = manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1)

    t = threading.Thread(target=worker)
    t.start()

    # Sleep to ensure worker is blocked
    time.sleep(0.1)
    self.assertTrue(t.is_alive())
    self.assertNotIn("data", results)

    # Send data
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="blocked_data")
    t.join(timeout=1.0)

    self.assertFalse(t.is_alive())
    self.assertEqual(results["data"], "blocked_data")

  def test_checkpoint_helper_skips_step_zero_and_waits_for_real_save(self):
    checkpoint_manager = mock.Mock()
    checkpoint_lock = threading.Lock()

    with mock.patch("maxtext.trainers.diloco.threaded_diloco.checkpointing.maybe_save_checkpoint") as maybe_save:
      _save_checkpoint_serialized(
          checkpoint_manager,
          state=object(),
          config=self.config,
          data_iterator=None,
          checkpoint_lock=checkpoint_lock,
          step=0,
          skip_step_zero=True,
      )
      maybe_save.assert_not_called()
      checkpoint_manager.wait_until_finished.assert_not_called()

      state = object()
      _save_checkpoint_serialized(
          checkpoint_manager,
          state=state,
          config=self.config,
          data_iterator=None,
          checkpoint_lock=checkpoint_lock,
          step=1,
          skip_step_zero=True,
      )
      maybe_save.assert_called_once_with(
          checkpoint_manager=checkpoint_manager,
          state=state,
          config=self.config,
          data_iterator=None,
          step=1,
      )
      checkpoint_manager.wait_until_finished.assert_called_once_with()

      checkpoint_manager.reset_mock()
      maybe_save.reset_mock()
      forced_state = object()
      _save_checkpoint_serialized(
          checkpoint_manager,
          state=forced_state,
          config=self.config,
          data_iterator=None,
          checkpoint_lock=checkpoint_lock,
          step=2,
          force=True,
      )
      maybe_save.assert_called_once_with(
          checkpoint_manager=checkpoint_manager,
          state=forced_state,
          config=self.config,
          data_iterator=None,
          step=2,
          force=True,
      )
      checkpoint_manager.wait_until_finished.assert_called_once_with()

  def test_delayed_response_step_skips_pre_resume_messages(self):
    start_step = 10_000
    self.assertIsNone(_delayed_response_step(10_001, 2, 1, start_step))
    self.assertIsNone(_delayed_response_step(10_002, 2, 1, start_step))
    self.assertEqual(_delayed_response_step(10_003, 2, 1, start_step), 10_001)

    # With a two-step fragment interval, the first post-resume response is the
    # first delayed step that is both newer than the checkpoint and a sync step.
    self.assertIsNone(_delayed_response_step(10_003, 2, 2, start_step))
    self.assertEqual(_delayed_response_step(10_004, 2, 2, start_step), 10_002)

  def test_checkpoint_alignment_validation(self):
    aligned = SimpleNamespace(
        enable_checkpointing=True,
        diloco_sync_period=36,
        num_diloco_fragments=17,
        checkpoint_period=10_000,
        save_checkpoint_on_completion=True,
        steps=20,
    )
    _validate_checkpoint_alignment(aligned)

    bad_period = SimpleNamespace(**{**vars(aligned), "checkpoint_period": 9_999})
    with self.assertRaisesRegex(ValueError, "checkpoint_period"):
      _validate_checkpoint_alignment(bad_period)

    bad_completion = SimpleNamespace(**{**vars(aligned), "steps": 21})
    with self.assertRaisesRegex(ValueError, "completion checkpoint"):
      _validate_checkpoint_alignment(bad_completion)

    continuous = SimpleNamespace(**{**vars(aligned), "enable_continuous_checkpointing": True})
    with self.assertRaisesRegex(ValueError, "unsupported modes"):
      _validate_checkpoint_alignment(continuous)


def _build_fake_params(mesh, num_layers=8, hidden=4, value=1.0):
  """Create a fake param tree with scanned 'layers' and non-scanned 'embed' params."""
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return {
      "layers": {"w": jax.device_put(jnp.full((num_layers, hidden), value), sharding)},
      "embed": jax.device_put(jnp.full((hidden,), value), sharding),
  }


def _build_manipulator(params, num_layers=8, num_transformer_frags=4):
  """Build a FragmentedTreeManipulator for fake params."""
  layers_per_frag = num_layers // num_transformer_frags
  fragment_to_layer_indices = {
      i + 1: jnp.array(list(range(i * layers_per_frag, (i + 1) * layers_per_frag))) for i in range(num_transformer_frags)
  }
  scanned_regex = re.compile(r"/(?:layers|blocks|moe_layers|dense_layers|layers_outside_pipeline)(?:/|$)")
  keypath_to_is_scanned = {}
  for keypath, _ in jax.tree_util.tree_flatten_with_path(params)[0]:
    parts = [str(k.key) if hasattr(k, "key") else str(k) for k in keypath]
    sp = "/" + "/".join(parts)
    keypath_to_is_scanned[jax.tree_util.keystr(keypath)] = bool(scanned_regex.search(sp))
  return FragmentedTreeManipulator(
      keypath_to_is_scanned=keypath_to_is_scanned,
      fragment_to_layer_indices=fragment_to_layer_indices,
      num_fragments=num_transformer_frags + 1,
      param_scan_axis=0,
  )


def _flat_params_shardings(params):
  """Return {keystr: NamedSharding} for all leaves."""
  return {jax.tree_util.keystr(k): v.sharding for k, v in jax.tree_util.tree_flatten_with_path(params)[0]}


class _FakeLearnerTransport:
  """Deterministic learner-fragment source for streaming reduction tests."""

  def __init__(self, fragments, events, expected_step=12, expected_fragment_id=3):
    self.fragments = fragments
    self.events = events
    self.expected_step = expected_step
    self.expected_fragment_id = expected_fragment_id

  def recv_from_learner(self, learner_idx, step, fragment_id):
    assert step == self.expected_step
    assert fragment_id == self.expected_fragment_id
    self.events.append(("recv", learner_idx))
    return self.fragments[learner_idx]


class SyncerComputeTest(unittest.TestCase):
  """Reproduces the syncer-side compute path (fragment extraction → outer step → scatter)
  on a CPU mesh with fake scanned-layer params.

  This exercises the same fragment-extraction/scatter code that hit a Pathways-specific
  tiling/layout crash on scanned fragments (see SyncerPathwaysBugReproTest below for the
  crash reproduction and the fix). Running on CPU verifies logical correctness cheaply,
  since the layout bug this class does NOT catch only manifests on real Pathways hardware.
  """

  NUM_LAYERS = 8
  NUM_FRAGS = 4
  HIDDEN = 4
  NUM_LEARNERS = 2

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Need at least 2 CPU devices; set XLA_FLAGS")
    self.mesh = jax.sharding.Mesh(np.array(devices[:2]), ("model",))

  # ------------------------------------------------------------------
  # FragmentedTreeManipulator round-trip
  # ------------------------------------------------------------------

  def test_fragment_roundtrip_all_fragments(self):
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    for frag_idx in range(manipulator.num_fragments):
      flat_frag = manipulator.get_flat_fragment(params, frag_idx)
      restored = manipulator.apply_flat_fragment(params, frag_idx, flat_frag)
      for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_allclose(np.array(a), np.array(b), err_msg=f"round-trip failed for frag {frag_idx}")

  def test_fragment_sizes_are_correct(self):
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    layers_per_frag = self.NUM_LAYERS // self.NUM_FRAGS
    # Fragment 0: non-scanned only (embed)
    frag0 = manipulator.get_flat_fragment(params, 0)
    self.assertIn("['embed']", frag0)
    self.assertNotIn("['layers']['w']", frag0)
    # Fragment >0: scanned only, with layers_per_frag rows
    for f in range(1, manipulator.num_fragments):
      fragf = manipulator.get_flat_fragment(params, f)
      self.assertNotIn("['embed']", fragf)
      w = fragf["['layers']['w']"]
      self.assertEqual(w.shape[0], layers_per_frag)

  def test_scanned_scatter_donates_full_array_and_preserves_values(self):
    params = _build_fake_params(self.mesh, value=2.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    original_layers = params["layers"]["w"]
    expected = np.asarray(original_layers).copy()
    indices = np.asarray(manipulator.fragment_to_layer_indices[1])
    expected[indices] = -1.0

    replacement = {
        "['layers']['w']": jax.device_put(
            jnp.full((len(indices), self.HIDDEN), -1.0, dtype=original_layers.dtype),
            original_layers.sharding,
        )
    }
    updated = manipulator.apply_flat_fragment(
        params,
        fragment_idx=1,
        flat_fragment=replacement,
        use_null_layout_jit=True,
        donate_full_array=True,
    )
    updated_layers = jax.block_until_ready(updated["layers"]["w"])

    np.testing.assert_allclose(np.asarray(updated_layers), expected)
    self.assertTrue(original_layers.is_deleted(), "scanned full array was not donated")

  def test_scanned_scatter_donation_supports_nonzero_scan_axis(self):
    """Qwen scanned parameters use a nonzero scan axis."""
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    original_layers = jax.device_put(
        jnp.arange(2 * self.NUM_LAYERS * self.HIDDEN, dtype=jnp.float32).reshape(2, self.NUM_LAYERS, self.HIDDEN),
        sharding,
    )
    params = {"layers": {"w": original_layers}}
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    manipulator.param_scan_axis = 1
    indices = np.asarray(manipulator.fragment_to_layer_indices[2])
    expected = np.asarray(original_layers).copy()
    expected[:, indices, :] = -5.0
    replacement = {
        "['layers']['w']": jax.device_put(
            jnp.full((2, len(indices), self.HIDDEN), -5.0, dtype=original_layers.dtype),
            sharding,
        )
    }

    updated = manipulator.apply_flat_fragment(
        params,
        fragment_idx=2,
        flat_fragment=replacement,
        use_null_layout_jit=True,
        donate_full_array=True,
    )
    updated_layers = jax.block_until_ready(updated["layers"]["w"])

    np.testing.assert_allclose(np.asarray(updated_layers), expected)
    self.assertTrue(original_layers.is_deleted())

  # ------------------------------------------------------------------
  # Coordinator streaming mean
  # ------------------------------------------------------------------

  def test_streaming_mean_matches_reference_and_preserves_dtypes(self):
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

    def make_fragment(value):
      return {
          "low_precision": jax.device_put(
              jnp.asarray([value, value + 1], dtype=jnp.bfloat16),
              sharding,
          ),
          "nested": {
              "full_precision": jax.device_put(
                  jnp.asarray([2 * value], dtype=jnp.float32),
                  sharding,
              )
          },
      }

    fragments = [make_fragment(value) for value in (1, 3, 8)]
    target_shardings = jax.tree_util.tree_map(lambda value: value.sharding, fragments[0])
    reduction_fns = make_streaming_mean_fns(
        fragments[0],
        target_shardings,
        num_learners=len(fragments),
    )
    events = []

    def fake_reshard(fragment, requested_shardings, *, donate):
      self.assertTrue(donate)
      self.assertEqual(requested_shardings, target_shardings)
      events.append(("reshard", len(events)))
      return fragment

    averaged = stream_learner_mean(
        _FakeLearnerTransport(fragments, events),
        num_learners=len(fragments),
        step=12,
        fragment_id=3,
        target_shardings=target_shardings,
        reduction_fns=reduction_fns,
        reshard_fn=fake_reshard,
    )

    self.assertEqual(averaged["low_precision"].dtype, jnp.bfloat16)
    self.assertEqual(averaged["nested"]["full_precision"].dtype, jnp.float32)
    np.testing.assert_allclose(np.asarray(averaged["low_precision"], dtype=np.float32), [4.0, 5.0])
    np.testing.assert_allclose(np.asarray(averaged["nested"]["full_precision"]), [8.0])
    for value, target in zip(
        jax.tree_util.tree_leaves(averaged),
        jax.tree_util.tree_leaves(target_shardings),
    ):
      self.assertEqual(value.sharding, target)
    self.assertEqual(sum(event[0] == "reshard" for event in events), len(fragments))

  def test_streaming_mean_single_learner_has_no_replica_dimension(self):
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    fragment = {"weight": jax.device_put(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), sharding)}
    target_shardings = {"weight": sharding}
    reduction_fns = make_streaming_mean_fns(fragment, target_shardings, num_learners=1)

    averaged = stream_learner_mean(
        _FakeLearnerTransport(
            [fragment],
            [],
            expected_step=12,
            expected_fragment_id=3,
        ),
        num_learners=1,
        step=12,
        fragment_id=3,
        target_shardings=target_shardings,
        reduction_fns=reduction_fns,
        reshard_fn=lambda value, _, *, donate: value,
    )

    self.assertEqual(averaged["weight"].shape, (2, 3))
    np.testing.assert_array_equal(np.asarray(averaged["weight"]), np.arange(6).reshape(2, 3))

  def test_identity_reshard_adopts_coordinator_tree_without_plugin_call(self):
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    fragment = {"weight": jax.device_put(jnp.arange(4), sharding)}
    target_shardings = {"weight": sharding}

    with mock.patch("maxtext.trainers.diloco.threaded_diloco.pathways_reshard.reshard") as sidechannel_reshard:
      result = _reshard_tree(fragment, target_shardings, donate=True)

    self.assertIs(result, fragment)
    sidechannel_reshard.assert_not_called()

  def test_identity_outgoing_reshard_forces_independent_payload(self):
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    fragment = {"weight": jax.device_put(jnp.arange(4), sharding)}
    target_shardings = {"weight": sharding}
    real_device_put = jax.device_put

    with (
        mock.patch("maxtext.trainers.diloco.threaded_diloco.pathways_reshard.reshard") as sidechannel_reshard,
        mock.patch(
            "maxtext.trainers.diloco.threaded_diloco.jax.device_put",
            wraps=real_device_put,
        ) as device_put,
    ):
      result = _reshard_tree(fragment, target_shardings, donate=False)

    sidechannel_reshard.assert_not_called()
    device_put.assert_called_once_with(fragment, target_shardings, may_alias=False)
    self.assertIsNot(result, fragment)
    self.assertFalse(fragment["weight"].is_deleted())
    np.testing.assert_array_equal(np.asarray(result["weight"]), np.arange(4))

  def test_streaming_mean_reduces_each_fragment_before_receiving_the_next(self):
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    fragments = [{"weight": jax.device_put(jnp.asarray([value], dtype=jnp.float32), sharding)} for value in (1, 3, 8)]
    target_shardings = {"weight": sharding}
    events = []

    def fake_reshard(fragment, requested_shardings, *, donate):
      self.assertIs(requested_shardings, target_shardings)
      self.assertTrue(donate)
      value = int(np.asarray(fragment["weight"])[0])
      events.append(("reshard", value))
      return fragment

    def initialize_sum(fragment):
      value = int(np.asarray(fragment["weight"])[0])
      events.append(("initialize", value))
      return fragment

    def add_to_sum(running_sum, fragment):
      value = int(np.asarray(fragment["weight"])[0])
      events.append(("add", value))
      return jax.tree_util.tree_map(lambda total, item: total + item, running_sum, fragment)

    def finish_mean(running_sum):
      events.append(("finish", None))
      return jax.tree_util.tree_map(lambda total: total / len(fragments), running_sum)

    averaged = stream_learner_mean(
        _FakeLearnerTransport(fragments, events),
        num_learners=len(fragments),
        step=12,
        fragment_id=3,
        target_shardings=target_shardings,
        reduction_fns=(initialize_sum, add_to_sum, finish_mean),
        reshard_fn=fake_reshard,
    )

    self.assertEqual(
        events,
        [
            ("recv", 0),
            ("reshard", 1),
            ("initialize", 1),
            ("recv", 1),
            ("reshard", 3),
            ("add", 3),
            ("recv", 2),
            ("reshard", 8),
            ("add", 8),
            ("finish", None),
        ],
    )
    np.testing.assert_allclose(np.asarray(averaged["weight"]), [4.0])

  # ------------------------------------------------------------------
  # make_step_fns: compute_grad
  # ------------------------------------------------------------------

  def test_compute_grad_uses_averaged_coordinator_fragment(self):
    """The outer pseudo-gradient consumes a fragment already averaged on the coordinator."""
    params = _build_fake_params(self.mesh, value=1.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    fps = _flat_params_shardings(params)
    outer_optimizer = optax.sgd(learning_rate=0.1, momentum=0.0, nesterov=False)

    frag_idx = 1
    outer_frag = manipulator.get_flat_fragment(params, frag_idx, use_null_layout_jit=True)
    averaged_frag = {
        key: jax.device_put(jnp.full(value.shape, 0.9, value.dtype), value.sharding) for key, value in outer_frag.items()
    }
    trace_dict = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in outer_frag.items()}
    compute_grad, _ = make_step_fns(self.mesh, fps, outer_frag, trace_dict, outer_optimizer)

    grad = jax.block_until_ready(compute_grad(outer_frag, averaged_frag))
    for v in jax.tree_util.tree_leaves(grad):
      np.testing.assert_allclose(np.array(v), 0.1, atol=1e-5)

  # ------------------------------------------------------------------
  # make_step_fns: apply_outer_step
  # ------------------------------------------------------------------

  def test_apply_outer_step_moves_params(self):
    """Outer SGD (no momentum) with lr=1.0 should set new_params = params - grad."""
    params = _build_fake_params(self.mesh, value=1.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    fps = _flat_params_shardings(params)
    lr = 1.0
    outer_optimizer = optax.sgd(learning_rate=lr, momentum=0.0, nesterov=False)

    frag_idx = 1
    outer_frag = manipulator.get_flat_fragment(params, frag_idx, use_null_layout_jit=True)
    opt_state = outer_optimizer.init(outer_frag)
    trace_dict = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in outer_frag.items()}
    _, apply_outer_step = make_step_fns(self.mesh, fps, outer_frag, trace_dict, outer_optimizer)

    # grad of 0.1 → new_params should be 1.0 - 0.1 = 0.9
    grad = {
        key: jax.device_put(jnp.full(value.shape, 0.1, value.dtype), value.sharding) for key, value in outer_frag.items()
    }
    new_frag, _ = jax.block_until_ready(apply_outer_step(grad, opt_state, outer_frag))
    for v in jax.tree_util.tree_leaves(new_frag):
      np.testing.assert_allclose(np.array(v), 0.9, atol=1e-5)

  # ------------------------------------------------------------------
  # Full syncer compute cycle: one sync period covering all fragments
  # ------------------------------------------------------------------

  def test_full_syncer_compute_one_period(self):
    """Run a complete syncer period: for each fragment, extract → compute_grad →
    apply_outer_step → scatter back.  After the period params must have changed
    and the syncer state must be self-consistent."""
    num_layers, num_frags, hidden = self.NUM_LAYERS, self.NUM_FRAGS, self.HIDDEN
    params = _build_fake_params(self.mesh, num_layers=num_layers, hidden=hidden, value=2.0)
    manipulator = _build_manipulator(params, num_layers, num_frags)
    fps = _flat_params_shardings(params)
    outer_optimizer = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
    outer_opt_state = outer_optimizer.init(params)
    outer_opt_state = (
        optax.TraceState(
            trace=jax.tree_util.tree_map(
                lambda value, param: jax.device_put(value, param.sharding),
                outer_opt_state[0].trace,
                params,
            )
        ),
        outer_opt_state[1],
    )

    # Build the fragment-specialized coordinator functions once, as the syncer does.
    step_fns = {}
    for frag_idx in range(manipulator.num_fragments):
      frag_dict = manipulator.get_flat_fragment(params, frag_idx, use_null_layout_jit=True)
      trace_dict = manipulator.get_flat_fragment(
          outer_opt_state[0].trace,
          frag_idx,
          use_null_layout_jit=True,
      )
      step_fns[frag_idx] = make_step_fns(
          self.mesh,
          fps,
          frag_dict,
          trace_dict,
          outer_optimizer,
      )

    # One full period: each parameter sees the mean of learner values 2.0 and
    # 1.6, already reduced to a coordinator fragment of 1.8.
    with jax.set_mesh(self.mesh):
      for frag_idx in range(manipulator.num_fragments):
        outer_frag = manipulator.get_flat_fragment(
            params,
            frag_idx,
            use_null_layout_jit=True,
        )
        trace_frag = manipulator.get_flat_fragment(
            outer_opt_state[0].trace,
            frag_idx,
            use_null_layout_jit=True,
        )
        opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())
        averaged_inner = {key: jax.device_put(value * 0.9, value.sharding) for key, value in outer_frag.items()}

        compute_grad, apply_outer_step = step_fns[frag_idx]
        pseudo_grad = jax.block_until_ready(compute_grad(outer_frag, averaged_inner))
        new_frag, new_opt_state_frag = jax.block_until_ready(apply_outer_step(pseudo_grad, opt_state_frag, outer_frag))
        params = manipulator.apply_flat_fragment(
            params,
            frag_idx,
            new_frag,
            use_null_layout_jit=True,
            donate_full_array=True,
        )
        new_trace_full = manipulator.apply_flat_fragment(
            outer_opt_state[0].trace,
            frag_idx,
            new_opt_state_frag[0].trace,
            use_null_layout_jit=True,
            donate_full_array=True,
        )
        params, new_trace_full = jax.block_until_ready((params, new_trace_full))
        outer_opt_state = (optax.TraceState(trace=new_trace_full), outer_opt_state[1])

    # After the full period params must have decreased (outer step moved them)
    for v in jax.tree_util.tree_leaves(params):
      self.assertTrue(np.all(np.array(v) < 2.0), "Params should decrease after outer step")

    # Optimizer trace must have the same tree structure as params
    trace_leaves = jax.tree_util.tree_leaves(outer_opt_state[0].trace)
    param_leaves = jax.tree_util.tree_leaves(params)
    self.assertEqual(len(trace_leaves), len(param_leaves))
    for t, p in zip(trace_leaves, param_leaves):
      self.assertEqual(t.shape, p.shape)


class SyncerPathwaysBugReproTest(unittest.TestCase):
  """Regression tests for the Pathways-specific eager ``jnp.take`` failure.

  Production passes ``use_null_layout_jit=True`` for coordinator scanned
  fragments. Despite that compatibility argument name, the implementation no
  longer assumes a null physical layout: it compiles the take and places each
  call into the executable-selected ``input_formats``.

  On Pathways, EAGER calls to jnp.take with the default mode='raise' raise:
    NotImplementedError: The 'raise' mode to jnp.take is not supported.

  Inside a @jax.jit body jnp.take is only *traced* (args are jax.core.Tracer objects),
  so it goes through XLA compilation rather than Pathways's eager dispatch — those are fine.

  CPU normally permits the eager call, so these tests simulate the Pathways
  restriction by rejecting non-traced inputs.
  """

  NUM_LAYERS = 8
  NUM_FRAGS = 4
  HIDDEN = 4

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2)
    self.mesh = jax.sharding.Mesh(np.array(devices[:2]), ("model",))
    self._real_take = jnp.take

  def _pathways_take(self, *args, **kwargs):
    """Simulates Pathways: eager jnp.take with mode='raise' is unsupported.
    Inside JIT (traced inputs) it is fine — XLA handles it; only eager calls fail.
    """
    is_traced = args and isinstance(args[0], jax.core.Tracer)
    mode = kwargs.get("mode", "raise")
    if not is_traced and mode == "raise":
      raise NotImplementedError("The 'raise' mode to jnp.take is not supported.")
    return self._real_take(*args, **kwargs)

  def test_bare_take_on_scanned_fragment_fails_eagerly(self):
    """The unadapted compatibility path remains unsafe for Pathways CPU arrays."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      # Fragment 0: non-scanned, no jnp.take — safe on Pathways.
      frag0 = manipulator.get_flat_fragment(params, fragment_idx=0)
      self.assertIn("['embed']", frag0)

      # Fragment 1: scanned, eager jnp.take — crashes on Pathways.
      with self.assertRaises(NotImplementedError):
        manipulator.get_flat_fragment(params, fragment_idx=1)

  def test_layout_adapted_take_avoids_eager_take(self):
    """The production adapter traces take and canonicalizes its physical input format."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      with jax.set_mesh(self.mesh):
        frag1 = manipulator.get_flat_fragment(params, fragment_idx=1, use_null_layout_jit=True)
      self.assertIn("['layers']['w']", frag1)


if __name__ == "__main__":
  unittest.main()
