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

"""CPU check of chain_pretrain.run_chain bookkeeping (run by hand: python3 chain_pretrain_cpu_check.py).

A toy 4-unit chain with the same (h, params, bias, batch) -> (out, indexer losses, lb losses) contract as
chain_pretrain.unit_fn checks that:
  1. the chained forward + reverse VJP reproduces direct autodiff of the full loss (lm + indexer + lb), per leaf;
  2. a run killed mid-forward or mid-backward and restarted from --resume_dir gives bitwise the same result;
  3. bf16 arrays round-trip through the resume store.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_pretrain as cp  # pylint: disable=g-import-not-at-top,wrong-import-position

import jax  # pylint: disable=wrong-import-order,wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-order,wrong-import-position
import numpy as np  # pylint: disable=wrong-import-order,wrong-import-position

UNITS = ["S1", "B0", "B1", "S3"]
MB, SEQ, RATE, EMB, VOCAB, N_MB = 2, 8, 2, 4, 16, 3


def init_params() -> dict:
  keys = jax.random.split(jax.random.PRNGKey(1), 5)
  return {
      "S1": {"emb": jax.random.normal(keys[0], (VOCAB, EMB)), "w": 0.5 * jax.random.normal(keys[1], (EMB, EMB))},
      "B0": {"w": 0.5 * jax.random.normal(keys[2], (EMB, EMB))},
      "B1": {"w": 0.5 * jax.random.normal(keys[3], (EMB, EMB))},
      "S3": {"head": jax.random.normal(keys[4], (EMB, VOCAB))},
  }


def toy_fn(unit: str):
  """Same signature and outputs as chain_pretrain.unit_fn: (out, indexer losses, lb losses)."""

  def fn(h, p, b, inputs, targets, segs, pos):  # pylint: disable=too-many-positional-arguments
    del b, pos
    if unit == "S1":
      y = p["emb"][inputs]
      h = jnp.broadcast_to(y[:, :, None, :], (y.shape[0], y.shape[1], RATE, EMB))
      h = jnp.tanh(h @ p["w"])
      return h, jnp.stack([jnp.mean(h**2)]), jnp.zeros((0,))
    if unit == "S3":
      logits = jnp.mean(h, axis=2) @ p["head"]
      logp = jax.nn.log_softmax(logits)
      xent = -jnp.take_along_axis(logp, targets[..., None], -1)[..., 0]
      return jnp.sum(xent * (segs != 0)), jnp.stack([1e-2 * jnp.mean(logits**2)]), jnp.zeros((0,))
    h_out = h + jnp.tanh(h @ p["w"])
    return h_out, jnp.stack([jnp.mean(h_out**2)]), jnp.stack([jnp.mean(jnp.abs(h_out)), jnp.mean(h_out)])

  return fn


class _SV:

  def __init__(self, params):
    self.params, self.bias = params, {}


def make_data():
  tok = np.asarray(jax.random.randint(jax.random.PRNGKey(0), (N_MB * MB, SEQ + 1), 0, VOCAB))
  return {
      "inputs": tok[:, :-1],
      "targets": tok[:, 1:],
      "segs": np.ones((N_MB * MB, SEQ), np.int32),
      "pos": np.zeros((N_MB * MB, SEQ), np.int32),
  }


def chain(params, data, store=None, crash_on_restore=None, units=tuple(UNITS)):
  """Runs cp.run_chain on the toy model; restore number `crash_on_restore` raises to simulate preemption."""
  calls = [0]

  def restore(unit):
    calls[0] += 1
    if crash_on_restore is not None and calls[0] == crash_on_restore:
      raise RuntimeError(f"simulated preemption at restore #{calls[0]} ({unit})")
    return _SV(params[unit]), 0.0

  def batch_fn(i):
    sl = slice(i * MB, (i + 1) * MB)
    return tuple(jnp.asarray(data[k][sl]) for k in ("inputs", "targets", "segs", "pos"))

  return cp.run_chain(
      units,
      restore,
      batch_fn,
      jnp.asarray,
      n_mb=N_MB,
      h_shape=(MB, SEQ, RATE, EMB),
      total_weights=float(data["segs"].sum()),
      make_fn=lambda sv, unit: toy_fn(unit),
      h_dtype=jnp.float32,
      store=store,
      mem_fn=dict,
  )


def direct(params, data):
  """Full-model loss as train.py forms it (lb averaged per microbatch, as the chain does) and its gradient."""

  def loss_fn(params):
    xent, idx, lb = 0.0, [], []
    for i in range(N_MB):
      sl = slice(i * MB, (i + 1) * MB)
      batch = tuple(jnp.asarray(data[k][sl]) for k in ("inputs", "targets", "segs", "pos"))
      h = jnp.zeros((MB, SEQ, RATE, EMB))
      for u in UNITS:
        h, ix, l = toy_fn(u)(h, params[u], {}, *batch)
        idx.append(ix)
        lb.append(l)
      xent += h
    lm = xent / data["segs"].sum()
    ix, l = jnp.concatenate(idx), jnp.concatenate(lb)
    return lm + jnp.mean(ix) + jnp.mean(l), (lm, jnp.mean(ix), jnp.mean(l))

  (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  return aux, grads


class RunChainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_enable_x64", False)
    self.params, self.data = init_params(), make_data()

  def test_chain_matches_direct_autodiff(self):
    res = chain(self.params, self.data)
    (lm, ix, lb), grads = direct(self.params, self.data)
    np.testing.assert_allclose(res["lm_loss"], float(lm), rtol=1e-6)
    np.testing.assert_allclose(res["indexer_loss"], float(ix), rtol=1e-6)
    np.testing.assert_allclose(res["moe_lb_loss_microbatched"], float(lb), rtol=1e-6)
    ref_norm = float(np.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grads))))
    np.testing.assert_allclose(res["raw_grad_norm"], ref_norm, rtol=1e-5)
    # Per unit: same leaf count and norm as autodiff (keys must not collide across units).
    for unit in UNITS:
      ref_u = float(np.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grads[unit]))))
      got_u = [r["grad_norm"] for r in res["units"] if r["unit"] == unit and r["pass"] == "bwd"][0]
      np.testing.assert_allclose(got_u, ref_u, rtol=1e-5, err_msg=unit)

  def test_resume_after_preemption_is_bitwise_identical(self):
    ref = chain(self.params, self.data)
    # Restores: S1 B0 B1 S3 (fwd) then S3 B1 B0 S1 (bwd). Crash in forward (#3) and in backward (#6).
    for crash_at in (3, 6):
      with tempfile.TemporaryDirectory() as d:
        store = cp.ResumeStore(d)
        with self.assertRaises(RuntimeError):
          chain(self.params, self.data, store=store, crash_on_restore=crash_at)
        res = chain(self.params, self.data, store=store)
        resumed = [(r["unit"], r["pass"]) for r in res["units"] if r.get("resumed")]
        self.assertEqual(len(resumed), crash_at - 1, resumed)
        for k in ("lm_loss", "indexer_loss", "moe_lb_loss_microbatched", "raw_grad_norm"):
          self.assertEqual(res[k], ref[k], f"{k} crash_at={crash_at}")
        self.assertEqual(res["top_grad_leaves"], ref["top_grad_leaves"])

  def test_forward_only_prefix(self):
    res = chain(self.params, self.data, units=UNITS[:2])
    self.assertNotIn("raw_grad_norm", res)
    self.assertEqual(res["n_indexer_entries"], 2)

  def test_bf16_store_round_trip(self):
    x = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (3, 5)), dtype=jnp.bfloat16)
    with tempfile.TemporaryDirectory() as d:
      store = cp.ResumeStore(d)
      store.save_array("x", x)
      y = store.load_array("x", jnp.bfloat16)
    self.assertEqual(y.dtype, x.dtype)
    np.testing.assert_array_equal(y.view(np.uint16), x.view(np.uint16))


if __name__ == "__main__":
  unittest.main()
