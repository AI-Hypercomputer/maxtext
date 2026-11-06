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

"""Tests for defer_small_all_reduces: the stacked partial expert counts, reduced once after the layer loop, equal the
values the per-layer all-reduces give."""

import os
import subprocess
import sys
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.layers import moe
from tests.utils.test_helpers import get_test_config_path
from tests.utils import linen_wrappers

_REQUIRED_CPU_DEVICES = 8

# A ring-of-experts config that validates without a device mesh (as configs_value_test's MoE fallback test).
_RING_CONFIG = {
    "run_name": "test",
    "num_experts": 8,
    "base_mlp_dim": 64,
    "base_moe_mlp_dim": 64,
    "override_logical_axis_rules": True,
    "use_ring_of_experts": True,
    "use_ragged_sort": True,
    "ragged_buffer_factor": 1.5,
}


def _cfg(rate=0.01, defer=True, gradient_accumulation_steps=1):
  return SimpleNamespace(
      routed_bias_update_rate=rate, defer_small_all_reduces=defer, gradient_accumulation_steps=gradient_accumulation_steps
  )


def _counts_to_update(global_counts, rate):
  """calculate_load_balance_updates' arithmetic on already-reduced counts."""
  average_load = jnp.sum(global_counts) / global_counts.shape[-1]
  return jnp.sign(average_load - global_counts) * rate


def _per_layer_reference(local_counts, cfg):
  """What the layer emits without deferral (PR #5401 semantics): the counts psum'd over the shards and summed
  over the ring-of-experts token chunks, then one update rate * sign(mean - counts) (load_balance_updates_from_counts).

  local_counts: (num_layers, num_parts, num_chunks, E) int32.
  """
  out = []
  for layer in range(local_counts.shape[0]):
    global_counts = jnp.sum(local_counts[layer], axis=(0, 1))  # the per-layer psum, summed over chunks
    out.append(moe.load_balance_updates_from_counts(global_counts, global_counts.shape[-1], cfg.routed_bias_update_rate))
  return jnp.stack(out)


class FinalizeDeferredBiasSignalTest(unittest.TestCase):
  """Pure array tests (any device count)."""

  def _local_counts(self, num_layers=3, num_parts=4, num_chunks=2, num_experts=8, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, 50, size=(num_layers, num_parts, num_chunks, num_experts)), jnp.int32)

  def test_equals_per_layer_chunk_average(self):
    """Counts reduced over shards and chunks, then one update, as the chunk loop does without deferral."""
    local = self._local_counts()
    cfg = _cfg()
    got = moe.finalize_deferred_bias_signal(local, cfg)
    ref = _per_layer_reference(local, cfg)
    self.assertEqual(got.shape, (3, 8))
    self.assertEqual(got.dtype, ref.dtype)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_gradient_accumulation_returns_counts(self):
    """gradient_accumulation_steps > 1: the summed int32 counts, as the non-deferred layer emits under GA, so that
    summing over microbatches and one load_balance_updates_from_counts equals the sign of the global-batch counts."""
    cfg = _cfg(gradient_accumulation_steps=2)
    micro = [self._local_counts(seed=10), self._local_counts(seed=11)]
    got = [moe.finalize_deferred_bias_signal(m, cfg) for m in micro]
    for g, m in zip(got, micro):
      self.assertEqual(g.dtype, jnp.int32)
      np.testing.assert_array_equal(np.asarray(g), np.asarray(jnp.sum(m, axis=(-3, -2))))
    step_update = moe.load_balance_updates_from_counts(got[0] + got[1], 8, cfg.routed_bias_update_rate)
    np.testing.assert_array_equal(np.asarray(step_update), np.asarray(_per_layer_reference(micro[0] + micro[1], _cfg())))

  def test_matches_calculate_load_balance_updates(self):
    """One chunk: the deferred update equals calculate_load_balance_updates on the concatenated shards' indices."""
    rng = np.random.default_rng(5)
    indices = jnp.asarray(rng.integers(0, 8, size=(4, 16, 2)), jnp.int32)  # (parts, tokens, top_k)
    local = jnp.stack([moe.calculate_expert_counts(indices[i], 8) for i in range(4)])[:, None, :]  # (parts, 1, E)
    got = moe.finalize_deferred_bias_signal(local, _cfg(rate=0.01))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(moe.calculate_load_balance_updates(indices, 8, 0.01)))

  def test_single_chunk_and_unscanned_layer(self):
    """One token chunk, and the unscanned (MTP) layer shape (num_parts, num_chunks, E) without a layer axis."""
    cfg = _cfg()
    local = self._local_counts(num_chunks=1, seed=2)
    np.testing.assert_array_equal(
        np.asarray(moe.finalize_deferred_bias_signal(local, cfg)), np.asarray(_per_layer_reference(local, cfg))
    )
    single = self._local_counts(num_layers=1, seed=3)
    np.testing.assert_array_equal(
        np.asarray(moe.finalize_deferred_bias_signal(single[0], cfg)),
        np.asarray(_per_layer_reference(single, cfg)[0]),
    )

  def test_finalize_intermediates_rewrites_only_bias_leaves(self):
    local = self._local_counts(seed=4)
    flags = jnp.array([[False, True], [False, False], [False, False]])
    tree = {
        "decoder": {"moe_layers": {"moe_bias_updates": (local,), "moe_has_overflow": (flags,)}},
        "mtp_block": {"mtp_layer_1": {"moe_bias_updates": (local[0],)}},
    }
    cfg = _cfg()
    out = moe.finalize_deferred_intermediates(tree, cfg)
    np.testing.assert_array_equal(
        np.asarray(out["decoder"]["moe_layers"]["moe_bias_updates"][0]), np.asarray(_per_layer_reference(local, cfg))
    )
    np.testing.assert_array_equal(
        np.asarray(out["mtp_block"]["mtp_layer_1"]["moe_bias_updates"][0]),
        np.asarray(_per_layer_reference(local[:1], cfg)[0]),
    )
    self.assertIs(out["decoder"]["moe_layers"]["moe_has_overflow"][0], flags)
    # Flag off: the tree is returned as is.
    self.assertIs(moe.finalize_deferred_intermediates(tree, _cfg(defer=False)), tree)


class DeferConfigTest(unittest.TestCase):
  """The flag's default and its validation."""

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="defer_small_ar_test", enable_checkpointing=False, **kw
    )

  def test_default_off(self):
    self.assertFalse(self._init().defer_small_all_reduces)

  def test_requires_ring_of_experts_sparse_matmul(self):
    with self.assertRaisesRegex(ValueError, "defer_small_all_reduces requires use_ring_of_experts"):
      self._init(defer_small_all_reduces=True, use_ring_of_experts=False, sparse_matmul=True)

  def test_rejects_layer_fallback(self):
    with self.assertRaisesRegex(ValueError, "defer_small_all_reduces does not support moe_dropless_fallback='layer'"):
      types.MaxTextConfig(**_RING_CONFIG, defer_small_all_reduces=True, moe_dropless_fallback="layer")

  def test_accepts_ring_of_experts_with_step_fallback(self):
    for mode in (None, "step"):
      with self.subTest(moe_dropless_fallback=mode):
        config = types.MaxTextConfig(**_RING_CONFIG, defer_small_all_reduces=True, moe_dropless_fallback=mode)
        self.assertTrue(config.defer_small_all_reduces)


@pytest.mark.cpu_only
def test_deferred_reduction_on_cpu_mesh():
  """Runs the mesh tests below in a subprocess with 8 forced CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_REQUIRED_CPU_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "DEFER_SMALL_AR_MESH_TESTS_PASSED" in result.stdout


def _count_prims(jaxpr, names, inside_scan=False):
  """Returns (#name eqns inside a scan body, #name eqns outside) over a closed jaxpr, recursively."""
  inside, outside = 0, 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in names:
      if inside_scan:
        inside += 1
      else:
        outside += 1
    for v in eqn.params.values():
      for sub in v if isinstance(v, (list, tuple)) else (v,):
        sub_jaxpr = getattr(sub, "jaxpr", sub)
        if hasattr(sub_jaxpr, "eqns"):
          i, o = _count_prims(sub_jaxpr, names, inside_scan or eqn.primitive.name == "scan")
          inside, outside = inside + i, outside + o
  return inside, outside


class DeferredMeshTest(unittest.TestCase):
  """A scanned stack of shard_map 'layers' on a (fsdp=4, expert=2) CPU mesh, shaped like the ring-of-experts path:
  tokens are sharded over fsdp and replicated over expert, counts are psum'd over fsdp only. The overflow flag is
  emitted unreduced per device and reduced once with jnp.any in both variants, as main's layer does since the
  moe_dropless_fallback="step" memory-neutral change; it checks that the deferred counts leave it intact."""

  __test__ = False

  num_layers, num_chunks, num_experts, rate = 3, 2, 8, 0.01

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < _REQUIRED_CPU_DEVICES:
      self.skipTest("needs 8 CPU devices; run through test_deferred_reduction_on_cpu_mesh")
    self.mesh = Mesh(np.array(jax.devices("cpu")[:8]).reshape(4, 2), ("fsdp", "expert"))
    rng = np.random.default_rng(0)
    # (layers, tokens, top_k) expert ids, tokens sharded over fsdp in the shard_map.
    self.indices = jnp.asarray(rng.integers(0, self.num_experts, size=(self.num_layers, 64, 2)), jnp.int32)
    # Local overflow on exactly one device (fsdp=2, expert=1) in layer 1 only.
    ov = np.zeros((self.num_layers, 4, 2), np.int32)
    ov[1, 2, 1] = 1
    self.local_overflow = jnp.asarray(ov)

  def _run(self, cfg, defer, bias_out_spec=None, flag_out_spec=None):
    """Returns ((bias signal, any overflow, per-layer overflow), jaxpr) of the scanned stack, deferred or not."""
    mesh, n_chunks, e = self.mesh, self.num_chunks, self.num_experts

    def layer(idx, ov):
      chunks = jnp.split(idx, n_chunks, axis=0)
      fsdp_i, ep_i = jax.lax.axis_index("fsdp"), jax.lax.axis_index("expert")
      local_flag = jnp.reshape(ov[fsdp_i, ep_i] > 0, (1,))  # this device's flag, unreduced (as on main)
      if defer:
        counts = jnp.stack([moe.calculate_expert_counts(c, e) for c in chunks])[None]  # (1, n_chunks, E)
        return counts, local_flag
      # PR #5401: psum'd counts summed over chunks, then one sign (load_balance_updates_from_counts).
      counts = sum(moe.calculate_expert_counts(c, e, axis_names=("fsdp",)) for c in chunks)
      signal = moe.load_balance_updates_from_counts(counts, e, cfg.routed_bias_update_rate)
      return signal, local_flag

    bias_spec = bias_out_spec if bias_out_spec is not None else (P("fsdp", None, None) if defer else P())
    flag_spec = flag_out_spec if flag_out_spec is not None else P(("fsdp", "expert"))
    smap = jax.shard_map(
        layer, mesh=mesh, in_specs=(P("fsdp", None), P()), out_specs=(bias_spec, flag_spec), check_vma=False
    )

    def model(indices, overflow):
      _, (bias, flags) = jax.lax.scan(lambda carry, xs: (carry, smap(*xs)), None, (indices, overflow))
      tree = (
          moe.finalize_deferred_intermediates({"moe_bias_updates": (bias,)}, cfg)
          if defer
          else {"moe_bias_updates": (bias,)}
      )
      return tree["moe_bias_updates"][0], jnp.any(flags), jnp.any(flags, axis=tuple(range(1, flags.ndim)))

    jaxpr = jax.make_jaxpr(model)(self.indices, self.local_overflow)
    return jax.jit(model)(self.indices, self.local_overflow), jaxpr

  def test_deferred_equals_per_layer(self):
    cfg = _cfg(rate=self.rate)
    (ref_bias, ref_any, ref_per_layer), ref_jaxpr = self._run(cfg, defer=False)
    (bias, any_flag, per_layer), jaxpr = self._run(cfg, defer=True)
    np.testing.assert_array_equal(np.asarray(bias), np.asarray(ref_bias))
    self.assertTrue(bool(ref_any))
    self.assertEqual(bool(any_flag), bool(ref_any))
    np.testing.assert_array_equal(np.asarray(per_layer), np.asarray(ref_per_layer))
    np.testing.assert_array_equal(np.asarray(per_layer), np.array([False, True, False]))
    # The reference has one count psum per chunk inside the scan; the deferred model has none there.
    self.assertEqual(_count_prims(ref_jaxpr.jaxpr, ("psum", "psum2", "psum_invariant"))[0], self.num_chunks)
    self.assertEqual(_count_prims(jaxpr.jaxpr, ("psum", "psum2", "psum_invariant"))[0], 0)

  def test_replicated_out_spec_is_wrong(self):
    """Negative control: emitting the unreduced partials under P() (the pre-deferral out_spec) loses the other
    devices' counts (and a flag under P() loses the other devices' flags), so the test above would catch that
    mistake."""
    cfg = _cfg(rate=self.rate)
    (ref_bias, ref_any, _), _ = self._run(cfg, defer=False)
    (bias, any_flag, _), _ = self._run(cfg, defer=True, bias_out_spec=P(None, None, None), flag_out_spec=P(None))
    self.assertFalse(np.array_equal(np.asarray(bias), np.asarray(ref_bias)))
    self.assertNotEqual(bool(any_flag), bool(ref_any))

  def test_routed_moe_layer_deferred_equals_per_layer(self):
    """The real RoutedMoE ring-of-experts layer (EP=2, fsdp=4, jax.jit on CPU): with the flag on, the finalized bias
    update, the any-overflow flag and the layer output equal the flag-off values, with and without token chunks and
    with and without a ragged buffer that overflows."""
    # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning
    from maxtext.layers.initializers import nd_dense_init
    from maxtext.utils import maxtext_utils

    def run(defer, chunks, rbf):
      cfg = pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="defer_small_ar_layer",
          enable_checkpointing=False,
          model_name="mixtral-8x7b",
          override_model_config=True,
          base_emb_dim=256,
          base_mlp_dim=128,
          base_moe_mlp_dim=128,
          dtype="float32",
          weight_dtype="float32",
          megablox=True,
          sparse_matmul=True,
          per_device_batch_size=2,
          ici_expert_parallelism=2,
          use_ring_of_experts=True,
          max_target_length=64,
          use_ragged_sort=True,
          ragged_buffer_factor=rbf,
          num_moe_token_chunks=chunks,
          routed_bias=True,
          routed_bias_update_rate=0.01,
          routed_score_func="sigmoid",
          decoder_block="deepseek",
          defer_small_all_reduces=defer,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = linen_wrappers.to_linen(
          moe.RoutedMoE,
          name="MoeBlock",
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
      )
      batch = int(cfg.per_device_batch_size) * jax.device_count()
      x = jax.random.normal(jax.random.PRNGKey(1), (batch, cfg.max_target_length, cfg.base_emb_dim), jnp.float32)
      rng = jax.random.PRNGKey(0)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        variables = jax.jit(lambda h: model.init({"params": rng, "dropout": rng}, h))(x)

        def f(p, h):
          (out, _, bias), inter = model.apply({"params": p}, h, mutable=["intermediates"])
          bias = moe.finalize_deferred_intermediates({"moe_bias_updates": (bias,)}, cfg)["moe_bias_updates"][0]
          flags = maxtext_utils.collect_intermediates_by_suffix(inter, "moe_has_overflow")
          return out, bias, jnp.any(jnp.stack([jnp.any(v) for v in flags]))

        return jax.jit(f)(variables["params"], x)

    for chunks in (1, 2):
      for rbf in (0.3, -1.0):
        with self.subTest(chunks=chunks, rbf=rbf):
          out_off, bias_off, ov_off = run(False, chunks, rbf)
          out_on, bias_on, ov_on = run(True, chunks, rbf)
          np.testing.assert_array_equal(np.asarray(out_on), np.asarray(out_off))
          np.testing.assert_array_equal(np.asarray(bias_on), np.asarray(bias_off))
          self.assertEqual(bias_on.shape, (8,))
          self.assertGreater(int(np.count_nonzero(np.asarray(bias_on))), 0)
          self.assertEqual(bool(ov_on), bool(ov_off))
          self.assertEqual(bool(ov_on), rbf > 0)


if __name__ == "__main__":
  DeferredMeshTest.__test__ = True
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeferredMeshTest)
  res = unittest.TextTestRunner(verbosity=2).run(suite)
  if res.wasSuccessful() and res.testsRun == 3 and not res.skipped:
    print("DEFER_SMALL_AR_MESH_TESTS_PASSED")
  sys.exit(0 if res.wasSuccessful() else 1)
