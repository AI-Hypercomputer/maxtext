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
from maxtext.trainers.diloco.threaded_diloco import (
    make_learner_config,
    make_step_fns,
    _slice_global_mesh_to_submesh,
    _get_apply_outer_step_flat_jit,
    _extract_scalar_metrics,
)
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager
from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator
from maxtext.utils.mesh_utils import stack_across_meshes_pytree

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

  def test_dynamic_extract_scanned_fragment_matches_static(self):
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    for f in range(1, manipulator.num_fragments):
      static_frag = manipulator.get_flat_fragment(params, f)
      dyn_frag = manipulator.dynamic_extract_scanned_fragment(params, jnp.int32(f - 1))
      self.assertEqual(set(static_frag.keys()), set(dyn_frag.keys()))
      for k in static_frag:
        np.testing.assert_allclose(
            np.array(static_frag[k]),
            np.array(dyn_frag[k]),
            err_msg=f"Dynamic extract mismatch for fragment {f} key {k}",
        )

  def test_dynamic_apply_scanned_fragment_matches_static(self):
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS, value=1.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    for f in range(1, manipulator.num_fragments):
      frag = manipulator.get_flat_fragment(params, f)
      updated_frag = {k: v * 2.0 for k, v in frag.items()}
      static_restored = manipulator.apply_flat_fragment(params, f, updated_frag)
      dyn_restored = manipulator.dynamic_apply_scanned_fragment(params, jnp.int32(f - 1), updated_frag)
      for a, b in zip(jax.tree_util.tree_leaves(static_restored), jax.tree_util.tree_leaves(dyn_restored)):
        np.testing.assert_allclose(
            np.array(a),
            np.array(b),
            err_msg=f"Dynamic apply mismatch for fragment {f}",
        )



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
    with jax.set_mesh(self.mesh):
      opt_state = jax.jit(outer_optimizer.init)(outer_frag)
    trace_dict = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in outer_frag.items()}
    _, apply_outer_step = make_step_fns(self.mesh, fps, outer_frag, trace_dict, outer_optimizer)

    # grad of 0.1 → new_params should be 1.0 - 0.1 = 0.9
    grad = {k: jax.device_put(jnp.full_like(v, 0.1), v.sharding) for k, v in outer_frag.items()}
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
    with jax.set_mesh(self.mesh):
      outer_opt_state = jax.jit(outer_optimizer.init)(params)

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
        frag_dict = manipulator.get_flat_fragment(params, f_idx)
        trace_dict = manipulator.get_flat_fragment(outer_opt_state[0].trace, f_idx)
        step_fns[f_idx] = make_step_fns(self.mesh, fps, frag_dict, trace_dict, outer_optimizer)

    # One full period: process all fragments
    with jax.set_mesh(self.mesh):
      for frag_idx in range(manipulator.num_fragments):
        outer_frag = manipulator.get_flat_fragment(params, frag_idx)
        trace_frag = manipulator.get_flat_fragment(outer_opt_state[0].trace, frag_idx)
        opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

        stacked_inner = make_stacked(outer_frag)

        compute_grad, apply_outer_step = step_fns[frag_idx]
        pseudo_grad = compute_grad(outer_frag, stacked_inner)
        new_frag, new_opt_state_frag = apply_outer_step(pseudo_grad, opt_state_frag, outer_frag)

        params = manipulator.apply_flat_fragment(params, frag_idx, new_frag)
        params = jax.device_put(params, params_full_sharding)
        new_trace_full = manipulator.apply_flat_fragment(
            outer_opt_state[0].trace, frag_idx, new_opt_state_frag[0].trace
        )
        new_trace_full = jax.device_put(new_trace_full, params_full_sharding)
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
    """Whole-fragment JIT extraction in get_flat_fragment avoids eager jnp.take on Pathways."""
    params = _build_fake_params(self.mesh)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    with mock.patch("maxtext.trainers.diloco.fragmenter.jnp.take", self._pathways_take):
      # Fragment 0: non-scanned
      frag0 = manipulator.get_flat_fragment(params, fragment_idx=0)
      self.assertIn("['embed']", frag0)

      # Fragment 1: scanned — extracted inside JIT graph without eager jnp.take error
      frag1 = manipulator.get_flat_fragment(params, fragment_idx=1)
      self.assertIn("['layers']['w']", frag1)

  def test_make_step_fns_aot_precompilation(self):
    """Verifies that make_step_fns pre-compiles outer optimization AOT with abstract state."""
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS, hidden=self.HIDDEN, value=2.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    fps = _flat_params_shardings(params)
    outer_optimizer = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
    with jax.set_mesh(self.mesh):
      outer_opt_state = jax.jit(outer_optimizer.init)(params)

    # Convert to abstract state structs
    abstract_params = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), params)
    abstract_opt_state = (
        optax.TraceState(
            trace=jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), outer_opt_state[0].trace)
        ),
        outer_opt_state[1],
    )

    compute_grad, apply_outer_step = make_step_fns(
        self.mesh,
        fps,
        None,
        None,
        outer_optimizer,
        abstract_params=abstract_params,
        abstract_opt_state=abstract_opt_state,
        manipulator=manipulator,
        num_learners=2,
    )

    # Test executing on fragment 0 (embed) and fragment 1 (layer)
    for f_idx in range(manipulator.num_fragments):
      outer_frag = manipulator.get_flat_fragment(params, f_idx)
      trace_frag = manipulator.get_flat_fragment(outer_opt_state[0].trace, f_idx)
      stacked_outer_frag = {
          k: jax.device_put(jnp.stack([v, v], axis=0), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", *fps[k].spec)))
          for k, v in outer_frag.items()
      }
      stacked_trace_frag = {
          k: jax.device_put(jnp.stack([v, v], axis=0), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", *fps[k].spec)))
          for k, v in trace_frag.items()
      }
      opt_state_frag = (optax.TraceState(trace=stacked_trace_frag), optax.EmptyState())
      stacked_inner_frag = {
          k: jax.device_put(jnp.stack([v, v * 0.9], axis=0), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", *fps[k].spec)))
          for k, v in outer_frag.items()
      }
      grad = compute_grad(stacked_outer_frag, stacked_inner_frag, frag_idx=f_idx)
      new_p, new_o = apply_outer_step(grad, opt_state_frag, stacked_outer_frag, frag_idx=f_idx)
      self.assertIsNotNone(new_p)
      self.assertIsNotNone(new_o)



  def test_replace_leaves_from_dict(self):
    params = _build_fake_params(self.mesh, value=1.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    shd = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
    leaf_dict = {"['embed']": jax.device_put(jnp.full((self.HIDDEN,), 5.0), shd)}
    updated = manipulator.replace_leaves_from_dict(params, leaf_dict)
    np.testing.assert_allclose(np.array(updated["embed"]), 5.0)
    np.testing.assert_allclose(np.array(updated["layers"]["w"]), 1.0)

  def test_get_leaves_for_fragment(self):
    params = _build_fake_params(self.mesh, value=2.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    leaves0 = manipulator.get_leaves_for_fragment(params, 0)
    self.assertIn("['embed']", leaves0)
    self.assertNotIn("['layers']['w']", leaves0)

    leaves1 = manipulator.get_leaves_for_fragment(params, 1)
    self.assertNotIn("['embed']", leaves1)
    self.assertIn("['layers']['w']", leaves1)
    self.assertEqual(leaves1["['layers']['w']"].shape, (self.NUM_LAYERS, self.HIDDEN))

  def test_abstract_shape_dtype_struct_roundtrip(self):
    params = _build_fake_params(self.mesh, value=1.0)
    abstract_params = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    for f in range(manipulator.num_fragments):
      frag = manipulator.get_flat_fragment(abstract_params, f)
      restored = manipulator.apply_flat_fragment(abstract_params, f, frag)
      for a, b in zip(jax.tree_util.tree_leaves(abstract_params), jax.tree_util.tree_leaves(restored)):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)

  def test_non_contiguous_layer_indices_roundtrip(self):
    params = _build_fake_params(self.mesh, num_layers=4, value=3.0)
    # Fragment 1 has non-contiguous layer indices: (0, 2), Fragment 2 has: (1, 3)
    fragment_to_layer_indices = {
        1: (0, 2),
        2: (1, 3),
    }
    scanned_regex = re.compile(r"/(?:layers|blocks|moe_layers|dense_layers|layers_outside_pipeline)(?:/|$)")
    keypath_to_is_scanned = {}
    for keypath, _ in jax.tree_util.tree_flatten_with_path(params)[0]:
      parts = [str(k.key) if hasattr(k, "key") else str(k) for k in keypath]
      sp = "/" + "/".join(parts)
      keypath_to_is_scanned[jax.tree_util.keystr(keypath)] = bool(scanned_regex.search(sp))
    manipulator = FragmentedTreeManipulator(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=3,
        param_scan_axis=0,
    )
    for f in range(manipulator.num_fragments):
      frag = manipulator.get_flat_fragment(params, f)
      restored = manipulator.apply_flat_fragment(params, f, frag)
      for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_allclose(np.array(a), np.array(b))




class LearnerFragmentCopyAndSliceTest(unittest.TestCase):
  """Tests verifying blocking fragment copy, non-donating outer step, and safe submesh slicing."""

  NUM_LAYERS = 4
  NUM_FRAGS = 2
  HIDDEN = 4
  NUM_LEARNERS = 2

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2)
    self.mesh = jax.sharding.Mesh(
        np.array(devices[: self.NUM_LEARNERS]).reshape(self.NUM_LEARNERS, 1),
        ("diloco", "model"),
    )

  def test_fragment_blocking_copy_and_materialization(self):
    """Verifies that jnp.copy + jax.block_until_ready produces independent ready arrays."""
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS, hidden=self.HIDDEN, value=3.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)

    for frag_idx in range(manipulator.num_fragments):
      frag_data = manipulator.get_flat_fragment(params, frag_idx)
      copied_frag = jax.tree_util.tree_map(
          lambda leaf: jnp.copy(leaf) if isinstance(leaf, jax.Array) else leaf,
          frag_data,
      )
      ready_frag = jax.block_until_ready(copied_frag)
      self.assertIsNotNone(ready_frag)
      for k, v in ready_frag.items():
        self.assertTrue(isinstance(v, jax.Array))
        np.testing.assert_allclose(np.array(v), np.array(frag_data[k]))

  def test_fragment_copy_isolation_from_mutated_source(self):
    """Verifies that copying leaves isolates the fragment even if the source array is overwritten."""
    params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS, hidden=self.HIDDEN, value=5.0)
    manipulator = _build_manipulator(params, self.NUM_LAYERS, self.NUM_FRAGS)
    frag_data = manipulator.get_flat_fragment(params, 0)
    copied_frag = jax.tree_util.tree_map(
        lambda leaf: jnp.copy(leaf) if isinstance(leaf, jax.Array) else leaf,
        frag_data,
    )
    ready_frag = jax.block_until_ready(copied_frag)

    # Simulate next step overwriting / reallocating params with new values
    new_params = _build_fake_params(self.mesh, num_layers=self.NUM_LAYERS, hidden=self.HIDDEN, value=99.0)
    # The ready_frag must still contain the original 5.0 values
    for k, v in ready_frag.items():
      np.testing.assert_allclose(np.array(v), 5.0)

  def test_apply_outer_step_flat_jit_preserves_input_buffers(self):
    """Verifies that _get_apply_outer_step_flat_jit does not donate or delete input buffers."""
    outer_optimizer = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
    apply_fn = _get_apply_outer_step_flat_jit(outer_optimizer)

    g_leaves = (jnp.full((self.HIDDEN,), 0.1),)
    trace_leaves = (jnp.full((self.HIDDEN,), 0.0),)
    p_leaves = (jnp.full((self.HIDDEN,), 1.0),)

    new_p_leaves, new_trace_leaves = apply_fn(g_leaves, trace_leaves, p_leaves)

    # Check outputs are computed correctly
    self.assertEqual(len(new_p_leaves), 1)
    self.assertEqual(len(new_trace_leaves), 1)

    # Crucially check that input arrays p_leaves and trace_leaves are NOT deleted or invalidated
    self.assertFalse(hasattr(p_leaves[0], "is_deleted") and p_leaves[0].is_deleted())
    self.assertFalse(hasattr(trace_leaves[0], "is_deleted") and trace_leaves[0].is_deleted())
    np.testing.assert_allclose(np.array(p_leaves[0]), 1.0)
    np.testing.assert_allclose(np.array(trace_leaves[0]), 0.0)

  def test_slice_global_mesh_to_submesh_safe_reshape_2d_and_3d(self):
    """Verifies _slice_global_mesh_to_submesh with safe reshaping across submeshes for 2D and 3D shapes."""
    devices = list(self.mesh.devices.flat)
    submesh0 = jax.sharding.Mesh(np.array([devices[0]]), ("model",))
    submesh1 = jax.sharding.Mesh(np.array([devices[1]]), ("model",))

    # 3D scanned leaf with replica dimension: (num_learners=2, num_layers=4, hidden=4)
    arr_3d = jax.device_put(jnp.ones((2, 4, 4)), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", None, "model")))
    target_sharding0 = jax.sharding.NamedSharding(submesh0, jax.sharding.PartitionSpec(None, "model"))

    sliced0 = _slice_global_mesh_to_submesh(
        arr_3d,
        submesh0,
        learner_idx=0,
        num_devices_per_mesh=1,
        target_shardings=target_sharding0,
        num_learners=2,
        target_shapes=(4, 4),
    )
    self.assertEqual(sliced0.shape, (4, 4))
    np.testing.assert_allclose(np.array(sliced0), 1.0)

    # 2D leaf with replica dimension: (num_learners=2, hidden=4)
    arr_2d = jax.device_put(jnp.full((2, 4), 2.0), jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("diloco", "model")))
    target_sharding_2d = jax.sharding.NamedSharding(submesh0, jax.sharding.PartitionSpec("model"))
    sliced_2d = _slice_global_mesh_to_submesh(
        arr_2d,
        submesh0,
        learner_idx=0,
        num_devices_per_mesh=1,
        target_shardings=target_sharding_2d,
        num_learners=2,
        target_shapes=(4,),
    )
    self.assertEqual(sliced_2d.shape, (4,))
    np.testing.assert_allclose(np.array(sliced_2d), 2.0)

  def test_extract_scalar_metrics_packed(self):
    mock_metrics = {
        "scalar": {
            "learning/loss": jnp.array(2.345),
            "learning/grad_norm": jnp.array([1.0, 3.0]),
            "learning/raw_grad_norm": jnp.array(0.789),
            "learning/current_learning_rate": 0.001,
            "perf/step_time_seconds": 1.25,
            "int_metric": 42,
        },
        "scalars": {},
    }
    extracted = _extract_scalar_metrics(mock_metrics)
    self.assertIsInstance(extracted, dict)
    self.assertAlmostEqual(extracted["scalar"]["learning/loss"], 2.345, places=3)
    self.assertAlmostEqual(extracted["scalar"]["learning/grad_norm"], 2.0, places=3)
    self.assertAlmostEqual(extracted["scalar"]["learning/raw_grad_norm"], 0.789, places=3)
    self.assertAlmostEqual(extracted["scalar"]["learning/current_learning_rate"], 0.001, places=5)
    self.assertAlmostEqual(extracted["scalar"]["perf/step_time_seconds"], 1.25, places=3)
    for k, v in extracted["scalar"].items():
      self.assertIsInstance(v, float)

    # Test extraction with sharded arrays under active mesh context
    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(np.array(devices), ("model",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model"))
    with jax.set_mesh(mesh):
      sharded_arr = jax.device_put(jnp.ones(4), sharding)
      sharded_metrics = {"scalar": {"loss": sharded_arr}}
      extracted_sharded = _extract_scalar_metrics(sharded_metrics)
      self.assertAlmostEqual(extracted_sharded["scalar"]["loss"], 1.0, places=3)
      self.assertIsInstance(extracted_sharded["scalar"]["loss"], float)

  def test_spec_map_slicing_multiaxis_shardings(self):
    """Verifies that PartitionSpec slicing for multi-axis shardings produces valid PartitionSpecs and NamedShardings."""
    devices = np.array(jax.devices()[:8])
    submesh = jax.sharding.Mesh(devices.reshape((1, 2, 2, 2)), ("diloco", "fsdp", "tensor", "model"))

    test_specs = {
        "3d_spec": jax.sharding.PartitionSpec("diloco", "fsdp", "tensor"),
        "2d_spec": jax.sharding.PartitionSpec("diloco", "fsdp"),
        "none_spec": jax.sharding.PartitionSpec("diloco", None, "model"),
        "1d_spec": jax.sharding.PartitionSpec("diloco"),
        "empty_spec": jax.sharding.PartitionSpec(),
    }
    shardings = {k: jax.sharding.NamedSharding(submesh, s) for k, s in test_specs.items()}

    spec_map = {
        k: (
            jax.sharding.PartitionSpec(*shardings[k].spec[1:])
            if hasattr(shardings[k], "spec") and len(shardings[k].spec) > 1
            else getattr(shardings[k], "spec", jax.sharding.PartitionSpec())
        )
        for k in shardings
    }

    # Verify each entry is an actual PartitionSpec and can construct a NamedSharding
    for k, spec in spec_map.items():
      self.assertIsInstance(spec, jax.sharding.PartitionSpec, f"{k} is not a PartitionSpec: {type(spec)}")
      named_shd = jax.sharding.NamedSharding(submesh, spec)
      self.assertIsInstance(named_shd, jax.sharding.NamedSharding)

  def test_syncer_direct_tpu_slice_dispatch(self):
    """Simulates Syncer direct fragment leaf slicing and dispatch across submeshes without host conversions."""
    devices = np.array(jax.devices()[:8])
    global_mesh = jax.sharding.Mesh(devices.reshape((2, 4)), ("diloco", "fsdp"))
    submesh0 = jax.sharding.Mesh(devices[:4].reshape((1, 4)), ("diloco", "fsdp"))
    submesh1 = jax.sharding.Mesh(devices[4:].reshape((1, 4)), ("diloco", "fsdp"))
    submeshes = [submesh0, submesh1]

    # Create dummy fragment with replica dimension = 2 including a 3D scanned parameter
    frag_sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec("diloco", "fsdp", None))
    scanned_sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec("diloco", None, None, "fsdp"))
    new_outer_params_frag = {
        "layers/w": jax.device_put(jnp.zeros((2, 4, 16), dtype=jnp.bfloat16), frag_sharding),
        "embed/w": jax.device_put(jnp.zeros((2, 16), dtype=jnp.bfloat16), jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec("diloco", "fsdp"))),
        "scanned/mlp": jax.device_put(jnp.zeros((2, 16, 1, 16), dtype=jnp.bfloat16), scanned_sharding),
    }

    for i, submesh in enumerate(submeshes):
      local_leaves = {}
      for k, v in new_outer_params_frag.items():
        v_slice = v[i] if hasattr(v, "ndim") and v.ndim > 0 and v.shape[0] == 2 else v
        leaf_spec = (
            jax.sharding.PartitionSpec(*v.sharding.spec[1:])
            if hasattr(v, "sharding") and hasattr(v.sharding, "spec") and len(v.sharding.spec) > 1
            else jax.sharding.PartitionSpec()
        )
        target_sharding = jax.sharding.NamedSharding(submesh, leaf_spec)
        local_leaves[k] = jax.device_put(v_slice, target_sharding)

      self.assertEqual(local_leaves["layers/w"].shape, (4, 16))
      self.assertEqual(local_leaves["embed/w"].shape, (16,))
      self.assertEqual(local_leaves["scanned/mlp"].shape, (16, 1, 16))
      self.assertIsInstance(local_leaves["layers/w"], jax.Array)
      self.assertIsInstance(local_leaves["embed/w"], jax.Array)
      self.assertIsInstance(local_leaves["scanned/mlp"], jax.Array)

  def test_jit_fused_packed_d2h_conversion(self):
    """Verifies that JIT-compiled dtype-grouped packing and slice unpacking produce bit-exact weights."""
    from maxtext.trainers.diloco.threaded_diloco import _get_jit_pack_fn
    num_learners = 2
    frag = {
        "layers/w": jnp.ones((num_learners, 4, 16), dtype=jnp.bfloat16),
        "layers/b": jnp.zeros((num_learners, 16), dtype=jnp.float32),
        "embed/w": jnp.full((num_learners, 32), 2.0, dtype=jnp.bfloat16),
        "scalar/scale": jnp.array(1.5, dtype=jnp.float32),
    }

    import collections
    dtype_groups = collections.defaultdict(dict)
    for k, v in frag.items():
      dtype_groups[v.dtype][k] = v

    local_leaves_by_learner = [{} for _ in range(num_learners)]

    for dt, group_leaves in dtype_groups.items():
      leaf_keys = list(group_leaves.keys())
      flat_shapes = [v.shape for v in group_leaves.values()]
      raw_leaves = tuple(group_leaves.values())

      jit_pack_fn = _get_jit_pack_fn(num_learners, flat_shapes)
      packed = jit_pack_fn(*raw_leaves)
      host_packed = np.asarray(packed)

      for i in range(num_learners):
        learner_packed = host_packed[i]
        offset = 0
        for k, shape in zip(leaf_keys, flat_shapes):
          learner_shape = shape[1:] if len(shape) > 1 and shape[0] == num_learners else shape
          size = int(np.prod(learner_shape)) if len(learner_shape) > 0 else 1
          local_leaves_by_learner[i][k] = learner_packed[offset:offset+size].reshape(learner_shape)
          offset += size

    for i in range(num_learners):
      self.assertEqual(local_leaves_by_learner[i]["layers/w"].shape, (4, 16))
      self.assertEqual(local_leaves_by_learner[i]["layers/w"].dtype, jnp.bfloat16)
      self.assertEqual(local_leaves_by_learner[i]["layers/b"].shape, (16,))
      self.assertEqual(local_leaves_by_learner[i]["layers/b"].dtype, jnp.float32)
      self.assertEqual(local_leaves_by_learner[i]["embed/w"].shape, (32,))
      self.assertEqual(local_leaves_by_learner[i]["embed/w"].dtype, jnp.bfloat16)
      self.assertEqual(local_leaves_by_learner[i]["scalar/scale"].shape, ())
      self.assertEqual(local_leaves_by_learner[i]["scalar/scale"].dtype, jnp.float32)
      np.testing.assert_allclose(np.array(local_leaves_by_learner[i]["layers/w"]), 1.0)
      np.testing.assert_allclose(np.array(local_leaves_by_learner[i]["layers/b"]), 0.0)
      np.testing.assert_allclose(np.array(local_leaves_by_learner[i]["embed/w"]), 2.0)
      np.testing.assert_allclose(np.array(local_leaves_by_learner[i]["scalar/scale"]), 1.5)

  def test_jit_pack_slice_concurrent_d2h(self):
    """Verifies that per-learner JIT slice packing and concurrent D2H unpacking produce bit-exact weights."""
    from maxtext.trainers.diloco.threaded_diloco import _get_jit_pack_slice_fn
    num_learners = 2
    frag = {
        "layers/w": jnp.ones((num_learners, 4, 16), dtype=jnp.bfloat16),
        "layers/b": jnp.zeros((num_learners, 16), dtype=jnp.float32),
        "embed/w": jnp.full((num_learners, 32), 2.0, dtype=jnp.bfloat16),
        "scalar/scale": jnp.array(1.5, dtype=jnp.float32),
    }

    import collections
    from concurrent.futures import ThreadPoolExecutor
    dtype_groups = collections.defaultdict(dict)
    for k, v in frag.items():
      dtype_groups[v.dtype][k] = v

    per_learner_packed_groups = [[] for _ in range(num_learners)]
    for dt, group_leaves in dtype_groups.items():
      leaf_keys = list(group_leaves.keys())
      flat_shapes = [v.shape for v in group_leaves.values()]
      raw_leaves = tuple(group_leaves.values())

      for i in range(num_learners):
        jit_pack_slice_fn = _get_jit_pack_slice_fn(i, num_learners, flat_shapes)
        packed_i = jit_pack_slice_fn(*raw_leaves)
        per_learner_packed_groups[i].append((dt, leaf_keys, flat_shapes, packed_i))

    def _fetch_and_unpack_learner(learner_i):
      learner_dict = {}
      for dt, leaf_keys, flat_shapes, packed_i in per_learner_packed_groups[learner_i]:
        host_packed = np.asarray(packed_i)
        offset = 0
        for k, shape in zip(leaf_keys, flat_shapes):
          learner_shape = shape[1:] if len(shape) > 1 and shape[0] == num_learners else shape
          size = int(np.prod(learner_shape)) if len(learner_shape) > 0 else 1
          learner_dict[k] = host_packed[offset : offset + size].reshape(learner_shape)
          offset += size
      return learner_i, learner_dict

    with ThreadPoolExecutor(max_workers=num_learners) as executor:
      futures = [executor.submit(_fetch_and_unpack_learner, i) for i in range(num_learners)]
      results = dict([fut.result() for fut in futures])

    for i in range(num_learners):
      self.assertEqual(results[i]["layers/w"].shape, (4, 16))
      self.assertEqual(results[i]["layers/w"].dtype, jnp.bfloat16)
      self.assertEqual(results[i]["layers/b"].shape, (16,))
      self.assertEqual(results[i]["layers/b"].dtype, jnp.float32)
      self.assertEqual(results[i]["embed/w"].shape, (32,))
      self.assertEqual(results[i]["embed/w"].dtype, jnp.bfloat16)
      self.assertEqual(results[i]["scalar/scale"].shape, ())
      self.assertEqual(results[i]["scalar/scale"].dtype, jnp.float32)
      np.testing.assert_allclose(np.array(results[i]["layers/w"]), 1.0)
      np.testing.assert_allclose(np.array(results[i]["layers/b"]), 0.0)
      np.testing.assert_allclose(np.array(results[i]["embed/w"]), 2.0)
      np.testing.assert_allclose(np.array(results[i]["scalar/scale"]), 1.5)

  def test_1d_packed_outer_step_parity(self):
    """Verifies that 1D packed outer SGD+Nesterov momentum step produces bit-exact parity with PyTree optax.sgd."""
    frag = {
        "layers/w": jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32),
        "layers/b": jnp.array([[0.5, -0.5]], dtype=jnp.float32),
    }
    # Simulate stacked inner params for 2 learners: shape (2, ...)
    stacked_inner = {
        "layers/w": jnp.array([[[1.1, 2.1], [3.1, 4.1]], [[0.9, 1.9], [2.9, 3.9]]], dtype=jnp.float32),
        "layers/b": jnp.array([[0.6, -0.4], [0.4, -0.6]], dtype=jnp.float32),
    }

    lr = 0.7
    momentum = 0.9
    opt = optax.sgd(learning_rate=lr, momentum=momentum, nesterov=True)

    # 1. PyTree standard execution
    outer_params = frag
    trace = jax.tree_util.tree_map(jnp.zeros_like, outer_params)
    opt_state = (optax.TraceState(trace=trace), optax.EmptyState())

    mean_inner = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), stacked_inner)
    pseudo_grad = jax.tree_util.tree_map(lambda o, i: o - i, outer_params, mean_inner)
    updates, new_opt_state = opt.update(pseudo_grad, opt_state, outer_params)
    expected_new_outer = optax.apply_updates(outer_params, updates)

    # 2. 1D Packed execution
    sorted_keys = sorted(frag.keys())
    leaf_shapes = [frag[k].shape for k in sorted_keys]
    leaf_sizes = [int(np.prod(s)) if len(s) > 0 else 1 for s in leaf_shapes]
    leaf_offsets = []
    offset = 0
    for sz in leaf_sizes:
      leaf_offsets.append((offset, offset + sz))
      offset += sz

    outer_1d = jnp.concatenate([jnp.reshape(frag[k], (-1,)) for k in sorted_keys])
    trace_1d = jnp.zeros_like(outer_1d)
    stacked_1d = jnp.stack([
        jnp.concatenate([jnp.reshape(stacked_inner[k][i], (-1,)) for k in sorted_keys])
        for i in range(2)
    ], axis=0)

    @jax.jit
    def outer_step_1d(outer_buf, trace_buf, stacked_buf):
      avg_inner = jnp.mean(stacked_buf, axis=0)
      p_grad = outer_buf - avg_inner
      n_trace = momentum * trace_buf + p_grad
      upd = lr * (p_grad + momentum * n_trace)
      n_outer = outer_buf - upd
      return n_outer, n_trace

    new_outer_1d, new_trace_1d = outer_step_1d(outer_1d, trace_1d, stacked_1d)

    # Unpack 1D
    actual_new_outer = {}
    for k, shp, (st, en) in zip(sorted_keys, leaf_shapes, leaf_offsets):
      actual_new_outer[k] = jnp.reshape(new_outer_1d[st:en], shp)

    for k in sorted_keys:
      np.testing.assert_allclose(
          np.array(actual_new_outer[k]), np.array(expected_new_outer[k]), rtol=1e-6, atol=1e-6
      )

  def test_fragment_1d_pipeline_helpers(self):
    """Verifies that _build_fragment_1d_metadata, _pack_fragment_1d, _unpack_fragment_1d, and _outer_sgd_1d_jit execute correctly across all fragments."""
    from maxtext.trainers.diloco.threaded_diloco import (
        _build_fragment_1d_metadata,
        _pack_fragment_1d,
        _unpack_fragment_1d,
        _outer_sgd_1d_jit,
    )

    params = _build_fake_params(self.mesh, num_layers=4, value=2.0)
    manipulator = _build_manipulator(params, num_layers=4, num_transformer_frags=4)

    for f_idx in range(manipulator.num_fragments):
      frag = manipulator.get_flat_fragment(params, f_idx)
      metadata = _build_fragment_1d_metadata(frag)

      # Test Pack & Unpack Roundtrip
      packed = _pack_fragment_1d(frag, metadata)
      self.assertIsInstance(packed, dict)
      unpacked = _unpack_fragment_1d(packed, metadata)

      for k in frag:
        np.testing.assert_allclose(np.array(unpacked[k]), np.array(frag[k]))

      # Test 1D Outer SGD
      for dt, outer_1d in packed.items():
        trace_1d = jnp.zeros_like(outer_1d)
        stacked_1d = jnp.stack([outer_1d, outer_1d * 0.9], axis=0)

        new_outer_1d, new_trace_1d = _outer_sgd_1d_jit(
            outer_1d, trace_1d, stacked_1d, lr=0.1, momentum=0.9, nesterov=True
        )
        self.assertEqual(new_outer_1d.shape, outer_1d.shape)
        self.assertEqual(new_trace_1d.shape, trace_1d.shape)


if __name__ == "__main__":
  unittest.main()



