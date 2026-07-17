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
from unittest import mock

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # pylint: disable=wrong-import-order
import jax.numpy as jnp
import optax

from maxtext.configs import pyconfig
from maxtext.trainers.diloco.threaded_diloco import make_learner_config, make_step_fns, _normalize_to_null_layout
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

  def test_transport_manager_out_of_order(self):
    manager = ThreadedTransportManager(num_learners=1)

    # Send out of order
    manager.send_to_syncer(learner_idx=0, step=2, fragment_id=1, data="step2")
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="step1")

    # Receive in order
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "step1")
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=2, fragment_id=1), "step2")

  def test_transport_manager_blocking(self):
    manager = ThreadedTransportManager(num_learners=1)
    results = {}

    def worker():
      results['data'] = manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1)

    t = threading.Thread(target=worker)
    t.start()

    # Sleep to ensure worker is blocked
    time.sleep(0.1)
    self.assertTrue(t.is_alive())
    self.assertNotIn('data', results)

    # Send data
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="blocked_data")
    t.join(timeout=1.0)

    self.assertFalse(t.is_alive())
    self.assertEqual(results['data'], "blocked_data")

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
      i + 1: jnp.array(list(range(i * layers_per_frag, (i + 1) * layers_per_frag)))
      for i in range(num_transformer_frags)
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
  return {
      jax.tree_util.keystr(k): v.sharding
      for k, v in jax.tree_util.tree_flatten_with_path(params)[0]
  }


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
    self.mesh = jax.sharding.Mesh(
        np.array(devices[: self.NUM_LEARNERS]).reshape(self.NUM_LEARNERS, 1),
        ("diloco", "model"),
    )

  # ------------------------------------------------------------------
  # _normalize_to_null_layout
  # ------------------------------------------------------------------

  def test_normalize_preserves_values(self):
    params = _build_fake_params(self.mesh, value=3.14)
    normalized = _normalize_to_null_layout(params)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(normalized)):
      np.testing.assert_allclose(np.array(a), np.array(b))

  def test_normalize_is_idempotent(self):
    params = _build_fake_params(self.mesh, value=2.71)
    once = _normalize_to_null_layout(params)
    twice = _normalize_to_null_layout(once)
    for a, b in zip(jax.tree_util.tree_leaves(once), jax.tree_util.tree_leaves(twice)):
      np.testing.assert_allclose(np.array(a), np.array(b))

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

  # ------------------------------------------------------------------
  # make_step_fns: compute_grad
  # ------------------------------------------------------------------

  def test_compute_grad_averages_learners(self):
    """pseudo-grad = outer_params - mean(inner_params) across learners."""
    params = _build_fake_params(self.mesh, value=1.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    fps = _flat_params_shardings(params)
    outer_optimizer = optax.sgd(learning_rate=0.1, momentum=0.0, nesterov=False)

    frag_idx = 1
    outer_frag = manipulator.get_flat_fragment(params, frag_idx)
    # learner 0: params = 1.0, learner 1: params = 0.8 → average = 0.9 → grad = 1.0 - 0.9 = 0.1
    stacked_sharding = {
        k: jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", *fps[k].spec))
        for k in outer_frag
    }
    stacked_frag = {
        k: jax.device_put(jnp.stack([v, v * 0.8], axis=0), stacked_sharding[k])
        for k, v in outer_frag.items()
    }
    trace_dict = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in outer_frag.items()}
    compute_grad, _ = make_step_fns(self.mesh, fps, outer_frag, trace_dict, outer_optimizer)

    grad = compute_grad(outer_frag, stacked_frag)
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
    outer_frag = manipulator.get_flat_fragment(params, frag_idx)
    outer_frag = _normalize_to_null_layout(outer_frag)
    opt_state = _normalize_to_null_layout(outer_optimizer.init(outer_frag))
    trace_dict = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in outer_frag.items()}
    _, apply_outer_step = make_step_fns(self.mesh, fps, outer_frag, trace_dict, outer_optimizer)

    # grad of 0.1 → new_params should be 1.0 - 0.1 = 0.9
    grad = {k: jnp.full_like(v, 0.1) for k, v in outer_frag.items()}
    new_frag, _ = apply_outer_step(grad, opt_state, outer_frag)
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

    params = _normalize_to_null_layout(params)
    outer_opt_state = _normalize_to_null_layout(outer_opt_state)

    # Precompute stacked learner fragment for each fragment index.
    # Learner 0: params = 2.0, learner 1: params = 1.8
    def make_stacked(outer_frag):
      stacked_sharding = {
          k: jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", *fps[k].spec))
          for k in outer_frag
      }
      return {
          k: jax.device_put(jnp.stack([v, v * 0.9], axis=0), stacked_sharding[k])
          for k, v in outer_frag.items()
      }

    params_full_sharding = jax.tree_util.tree_map(lambda x: x.sharding, params)

    # Build step fns for each fragment (mimics the precompute block in _run_syncer_loop)
    step_fns = {}
    with jax.set_mesh(self.mesh):
      for f_idx in range(manipulator.num_fragments):
        if f_idx == 0:
          frag_dict = manipulator.get_flat_fragment(params, f_idx)
          trace_dict = manipulator.get_flat_fragment(outer_opt_state[0].trace, f_idx)
        else:
          indices = manipulator.fragment_to_layer_indices[f_idx]
          frag_dict = {}
          trace_dict = {}
          for kpath, v in jax.tree_util.tree_flatten_with_path(params)[0]:
            ks = jax.tree_util.keystr(kpath)
            if manipulator.keypath_to_is_scanned.get(ks, False):
              frag_shape = (len(indices),) + v.shape[1:]
              frag_dict[ks] = jax.ShapeDtypeStruct(frag_shape, v.dtype)
              trace_dict[ks] = jax.ShapeDtypeStruct(frag_shape, v.dtype)
        step_fns[f_idx] = make_step_fns(self.mesh, fps, frag_dict, trace_dict, outer_optimizer)

    # One full period: process all fragments
    with jax.set_mesh(self.mesh):
      for frag_idx in range(manipulator.num_fragments):
        outer_frag = _normalize_to_null_layout(manipulator.get_flat_fragment(params, frag_idx))
        trace_frag = _normalize_to_null_layout(
            manipulator.get_flat_fragment(outer_opt_state[0].trace, frag_idx)
        )
        opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

        stacked_inner = make_stacked(outer_frag)
        stacked_inner = _normalize_to_null_layout(stacked_inner)

        compute_grad, apply_outer_step = step_fns[frag_idx]
        pseudo_grad = _normalize_to_null_layout(compute_grad(outer_frag, stacked_inner))
        new_frag, new_opt_state_frag = apply_outer_step(pseudo_grad, opt_state_frag, outer_frag)
        new_frag = _normalize_to_null_layout(new_frag)
        new_trace = _normalize_to_null_layout(new_opt_state_frag[0].trace)

        params = manipulator.apply_flat_fragment(params, frag_idx, new_frag)
        params = _normalize_to_null_layout(jax.device_put(params, params_full_sharding))
        new_trace_full = manipulator.apply_flat_fragment(
            outer_opt_state[0].trace, frag_idx, new_trace
        )
        new_trace_full = _normalize_to_null_layout(jax.device_put(new_trace_full, params_full_sharding))
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
  """Regression test for the Pathways-specific jnp.take failure that used to crash the
  syncer on real Pathways hardware (fixed by always passing use_null_layout_jit=True for
  scanned fragments in _run_syncer_loop; see threaded_diloco.py).

  On Pathways, EAGER calls to jnp.take with the default mode='raise' raise:
    NotImplementedError: The 'raise' mode to jnp.take is not supported.

  Inside a @jax.jit body jnp.take is only *traced* (args are jax.core.Tracer objects),
  so it goes through XLA compilation rather than Pathways's eager dispatch — those are fine.

  Before the fix, the syncer's _run_syncer_loop called get_flat_fragment() WITHOUT
  use_null_layout_jit=True, so for any scanned fragment (index > 0) it hit the EAGER bare
  jnp.take and crashed on Pathways. On CPU this worked fine — which is why SyncerComputeTest
  alone couldn't catch it — so this test simulates Pathways by patching jnp.take to raise
  only on non-traced (eager) inputs, and keeps the old buggy call pattern below as a
  regression guard against reintroducing it.
  """

  NUM_LAYERS = 8
  NUM_FRAGS = 4
  HIDDEN = 4

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2)
    self.mesh = jax.sharding.Mesh(
        np.array(devices[:2]).reshape(2, 1), ("diloco", "model")
    )
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
    """get_flat_fragment without use_null_layout_jit calls jnp.take eagerly on real
    arrays, which Pathways rejects.  This is the exact call made in _run_syncer_loop."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      # Fragment 0: non-scanned, no jnp.take — safe on Pathways.
      frag0 = manipulator.get_flat_fragment(params, fragment_idx=0)
      self.assertIn("['embed']", frag0)

      # Fragment 1: scanned, eager jnp.take — crashes on Pathways.
      with self.assertRaises(NotImplementedError):
        manipulator.get_flat_fragment(params, fragment_idx=1)

  def test_null_layout_jit_path_avoids_eager_take(self):
    """With use_null_layout_jit=True the call goes through _make_take_jit_null which
    wraps jnp.take in a @jax.jit.  The take is only traced (Tracer input), not executed
    eagerly, so the Pathways restriction does not fire."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      with jax.set_mesh(self.mesh):
        frag1 = manipulator.get_flat_fragment(params, fragment_idx=1, use_null_layout_jit=True)
      self.assertIn("['layers']['w']", frag1)

  def test_syncer_loop_body_crashes_on_scanned_fragments(self):
    """Reproduces the pre-fix code path that _run_syncer_loop used to run (calling
    get_flat_fragment without use_null_layout_jit):
        outer_params_frag = _normalize_to_null_layout(
            manipulator.get_flat_fragment(syncer_state.params, frag_idx)
        )
    Fragment 0 (non-scanned) is safe; any fragment > 0 (scanned layers) crashes on
    Pathways. _run_syncer_loop now always passes use_null_layout_jit=True for scanned
    fragments (see test_null_layout_jit_path_avoids_eager_take above), so this test only
    documents/guards against the bug rather than describing current behavior."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      # Fragment 0: no jnp.take — safe.
      _normalize_to_null_layout(manipulator.get_flat_fragment(params, 0))

      # All scanned fragments crash — this is the bug.
      for frag_idx in range(1, manipulator.num_fragments):
        with self.assertRaises(NotImplementedError, msg=f"frag_idx={frag_idx} should crash"):
          _normalize_to_null_layout(manipulator.get_flat_fragment(params, frag_idx))


if __name__ == "__main__":
  unittest.main()
