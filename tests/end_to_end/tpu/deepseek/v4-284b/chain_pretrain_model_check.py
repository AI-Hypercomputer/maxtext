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

"""CPU check that the chained pretrain step equals train.py on a real DeepSeek-V4 model (run by hand).

JAX_PLATFORMS=cpu PYTHONPATH=src:. python3 tests/end_to_end/tpu/deepseek/v4-284b/chain_pretrain_model_check.py

A deepseek4-tiny-shaped model (3 hash prefix layers + N_BLOCKS scanned 2-layer blocks, scan_layers=True, fp32) with
random weights, random tid2eid and nonzero router biases. Reference: train.py loss_fn under jax.value_and_grad (the
diff_wrapper train_step uses) and train_step's own metrics. Chain: chain_pretrain.run_chain with chain_pretrain.unit_fn
on the 1-block skeleton the TPU job uses, each unit loading its own slice of the same weights. Checks:
  1. lm/indexer/moe_lb losses and the global raw grad norm match train.py to fp32 rounding (microbatch = batch);
  2. every gradient leaf of every unit matches the matching block slice of the full-model gradient, and every
     nonzero full-model gradient leaf (per block) is produced by exactly one unit;
  3. the production setting (microbatch 1) gives the same losses and grad norm;
  4. chain_layerwise.kl_per_token equals forward_pass_logit_checker's jax-path KL (clip_logits_epsilon=1e-6).
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_layerwise as cl  # pylint: disable=g-import-not-at-top,wrong-import-position
import chain_pretrain as cp  # pylint: disable=g-import-not-at-top,wrong-import-position

from flax import nnx  # pylint: disable=wrong-import-order,wrong-import-position
import jax  # pylint: disable=wrong-import-order,wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-order,wrong-import-position
from maxtext.common import train_state_nnx  # pylint: disable=wrong-import-position
from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=wrong-import-position
from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.layers import moe  # pylint: disable=wrong-import-position
from maxtext.layers import quantizations  # pylint: disable=wrong-import-position
from maxtext.models import models  # pylint: disable=wrong-import-position
from maxtext.trainers.pre_train import train as pre_train  # pylint: disable=wrong-import-position
from maxtext.utils import maxtext_utils  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-order,wrong-import-position
import optax  # pylint: disable=wrong-import-order,wrong-import-position

views = cl.views
PREFIX = 3
N_BLOCKS = 5
SEQ, BATCH, VOCAB = 256, 4, 256
RATIOS = [0, 0, 4] + [128, 4] * N_BLOCKS


def make_cfg(n_layers: int, per_device_batch: int = 1):
  """deepseek4-tiny with n_layers decoder layers at the check config (fp32, scanned, compressed attention)."""
  kwargs = {
      "model_name": "deepseek4-tiny",
      "override_model_config": True,
      "run_name": "ds4_chain_model_check",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "attention": "dot_product",
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 1.0,
      "sparse_matmul": True,
      "megablox": False,
      # train.py loss_fn truncates the batch to micro_batch_size_to_train_on, so the reference needs the full batch.
      "per_device_batch_size": per_device_batch,
      "max_target_length": SEQ,
      "scan_layers": True,
      "param_scan_axis": 1,
      "base_num_decoder_layers": n_layers,
      "compress_ratios": RATIOS[:n_layers],
      "vocab_size": VOCAB,
      "load_balance_loss_weight": 0.01,
      "enable_dropout": False,
  }
  return pyconfig.initialize(["", os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")], **kwargs)


def build(cfg, mesh):
  return models.Transformer(
      cfg,
      mesh,
      quantizations.configure_quantization(cfg),
      model_mode=MODEL_MODE_TRAIN,
      rngs=nnx.Rngs(params=0, dropout=0),
  )


def randomize(model, num_experts: int, seed: int = 19) -> None:
  """Random weights and tid2eid tables, and nonzero router biases so routing depends on them."""
  rng = np.random.default_rng(seed)
  for _, var in nnx.to_flat_state(nnx.state(model)):
    if isinstance(var, nnx.RngState):
      continue
    val = var.get_value()
    if isinstance(var, moe.Tid2EidVar):
      var.set_value(jnp.asarray(rng.integers(0, num_experts, size=val.shape).astype(np.asarray(val).dtype)))
    elif isinstance(var, moe.MoEBiasVar):
      var.set_value(jnp.asarray((rng.standard_normal(val.shape) * 0.5).astype(np.float32), dtype=val.dtype))
    elif isinstance(var, nnx.Param) and jnp.issubdtype(val.dtype, jnp.floating):
      var.set_value(jnp.asarray((rng.standard_normal(val.shape) * 0.08).astype(np.float32), dtype=val.dtype))


def block_axis(full_shape, skel_shape) -> int:
  axes = [a for a, (f, s) in enumerate(zip(full_shape, skel_shape)) if f == N_BLOCKS and s == 1]
  assert len(axes) == 1 and len(full_shape) == len(skel_shape), (full_shape, skel_shape)
  return axes[0]


def load_unit(skel, full_values: dict, blk: int) -> None:
  """Copies prefix/global leaves and scanned block `blk` of the full model into the 1-block skeleton."""
  for path, var in nnx.to_flat_state(nnx.state(skel)):
    if isinstance(var, nnx.RngState):
      continue
    src = full_values[path]
    dst_shape = tuple(var.get_value().shape)
    if "scanned_blocks" in [str(p) for p in path]:
      src = np.take(src, [blk], axis=block_axis(src.shape, dst_shape))
    assert src.shape == dst_shape, (path, src.shape, dst_shape)
    var.set_value(jnp.asarray(src, dtype=var.get_value().dtype))


def reference(cfg, mesh, model, batch) -> dict:
  """train.py loss_fn losses and per-leaf gradients, plus train_step's raw_grad_norm."""
  gdef, params, rest = nnx.split(model, nnx.Param, ...)

  def f(p):
    m = nnx.merge(gdef, p, jax.tree.map(lambda t: t, rest), copy=True)
    return pre_train.loss_fn(m, cfg, dict(batch), None, None, is_train=True)

  with views.maxtext_context(cfg, mesh):
    (_, aux), g = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
  leaves = {jax.tree_util.keystr(p): np.asarray(v, np.float64) for p, v in jax.tree_util.tree_flatten_with_path(g)[0]}
  res = {
      "lm_loss": float(aux["xent_sum"] / aux["total_weights"]),
      "indexer_loss": float(aux["indexer_loss"]),
      "moe_lb_loss": float(aux["moe_lb_loss"]),
      "raw_grad_norm": float(np.sqrt(sum(np.sum(v**2) for v in leaves.values()))),
      "leaves": leaves,
  }
  # Runs last: train_step updates the router biases of `model` in place.
  tgdef, tstate = nnx.split(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.sgd(0.0), wrt=nnx.Param)))
  _, metrics = pre_train.train_step(tgdef, cfg, None, None, tstate, dict(batch))
  res["train_step_raw_grad_norm"] = float(metrics["scalar"]["learning/raw_grad_norm"])
  return res


def run_chain(cfg_skel, mesh, full_values: dict, data: dict, mb: int) -> dict:
  """chain_pretrain.run_chain over S1, B0..B{N-2}, S3 with the per-unit grad records read back from the store."""
  skel = build(cfg_skel, mesh)
  units = ["S1"] + [f"B{b}" for b in range(N_BLOCKS - 1)] + ["S3"]
  blk_of = {"S1": 0, "S3": N_BLOCKS - 1, **{f"B{b}": b for b in range(N_BLOCKS - 1)}}

  def restore(unit):
    load_unit(skel, full_values, blk_of[unit])
    return views.MaxTextSubgroups(skel, cfg_skel, mesh), 0.0

  def batch_fn(i):
    return tuple(jnp.asarray(data[k][i * mb : (i + 1) * mb]) for k in ("inputs", "targets", "segs", "pos"))

  store = cp.ResumeStore(tempfile.mkdtemp(prefix=f"chain_model_check_mb{mb}_"))
  try:
    res = cp.run_chain(
        units,
        restore,
        batch_fn,
        jnp.asarray,
        n_mb=BATCH // mb,
        h_shape=(mb, SEQ, cfg_skel.mhc_expansion_rate, cfg_skel.emb_dim),
        total_weights=float(data["segs"].sum()),
        ctx=lambda: views.maxtext_context(cfg_skel, mesh),
        h_dtype=jnp.float32,
        mem_fn=dict,
        store=store,
    )
    res["grad_sq"] = {u: store.load_json(f"bwd_{u}.json")["grad_sq"] for u in units}
  finally:
    shutil.rmtree(store.root)
  flat = jax.tree_util.tree_flatten_with_path(nnx.split(skel, nnx.Param, ...)[1])[0]
  res["skel_shapes"] = {jax.tree_util.keystr(p): tuple(v.shape) for p, v in flat}
  res["blk_of"] = blk_of
  return res


def per_leaf(ref_leaves: dict, got: dict):
  """(rel err, unit, leaf, chain norm, ref norm) per unit leaf, and full-model leaves not covered exactly once."""
  rows, covered = [], {}
  for unit, sq in got["grad_sq"].items():
    for key, v in sq.items():
      leaf = key.split("/", 1)[1]
      ref, blk = ref_leaves[leaf], None
      if "scanned_blocks" in leaf:
        blk = got["blk_of"][unit]
        ref = np.take(ref, [blk], axis=block_axis(ref.shape, got["skel_shapes"][leaf]))
      covered[(leaf, blk)] = covered.get((leaf, blk), 0) + 1
      r = float(np.sqrt(np.sum(ref**2)))
      rows.append((abs(np.sqrt(v) - r) / (r + 1e-30), unit, leaf, float(np.sqrt(v)), r))
  uncovered = []
  for leaf, ref in ref_leaves.items():
    if "scanned_blocks" in leaf:
      a = block_axis(ref.shape, got["skel_shapes"][leaf])
      parts = [(b, np.take(ref, [b], axis=a)) for b in range(N_BLOCKS)]
    else:
      parts = [(None, ref)]
    for blk, part in parts:
      if covered.get((leaf, blk), 0) != 1:
        uncovered.append((leaf, blk, covered.get((leaf, blk), 0), float(np.sqrt(np.sum(part**2)))))
  return rows, uncovered


def rel(a: float, b: float) -> float:
  return abs(a - b) / (abs(b) + 1e-30)


class ChainPretrainModelCheck(unittest.TestCase):
  """Chained step vs train.py on the same random tiny DeepSeek-V4 weights."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    n_layers = PREFIX + 2 * N_BLOCKS
    cfg_full, cfg_skel = make_cfg(n_layers, BATCH), make_cfg(PREFIX + 2)
    mesh = maxtext_utils.get_mesh_from_config(cfg_full)
    data = cp.synthetic_batch(VOCAB, BATCH, SEQ)
    batch = {
        "inputs": data["inputs"],
        "inputs_position": data["pos"],
        "inputs_segmentation": data["segs"],
        "targets": data["targets"],
        "targets_position": data["pos"],
        "targets_segmentation": data["segs"],
    }
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    with views.maxtext_context(cfg_full, mesh):
      full = build(cfg_full, mesh)
      randomize(full, cfg_full.num_experts)
    # Snapshot values before reference(): train_step updates the router biases in place.
    cls.full_values = {
        p: np.array(v.get_value()) for p, v in nnx.to_flat_state(nnx.state(full)) if not isinstance(v, nnx.RngState)
    }
    cls.ref = reference(cfg_full, mesh, full, batch)
    cls.chain = {mb: run_chain(cfg_skel, mesh, cls.full_values, data, mb) for mb in (BATCH, 1)}

  def test_losses_and_grad_norm(self):
    got, ref = self.chain[BATCH], self.ref
    self.assertLess(rel(got["lm_loss"], ref["lm_loss"]), 1e-6)
    self.assertLess(rel(got["indexer_loss"], ref["indexer_loss"]), 1e-5)
    self.assertLess(rel(got["moe_lb_loss_microbatched"], ref["moe_lb_loss"]), 1e-6)
    self.assertLess(rel(got["raw_grad_norm"], ref["raw_grad_norm"]), 1e-6)
    self.assertLess(rel(got["raw_grad_norm"], ref["train_step_raw_grad_norm"]), 1e-6)

  def test_per_leaf_gradients(self):
    gnorm = self.ref["raw_grad_norm"]
    rows, uncovered = per_leaf(self.ref["leaves"], self.chain[BATCH])
    big = [r for r in rows if r[4] > 1e-4 * gnorm]
    self.assertGreater(len(big), 100)
    worst = max(big)
    self.assertLess(worst[0], 1e-4, f"worst leaf {worst}")
    # Leaves a unit never produces must carry no gradient in the full model either (rounding-level at most).
    for leaf, blk, count, norm in uncovered:
      self.assertLess(norm, 1e-9 * gnorm, f"{leaf} block {blk}: produced by {count} units, ref norm {norm}")

  def test_microbatch_one(self):
    got, ref = self.chain[1], self.ref
    self.assertLess(rel(got["lm_loss"], ref["lm_loss"]), 1e-6)
    self.assertLess(rel(got["indexer_loss"], ref["indexer_loss"]), 1e-5)
    self.assertLess(rel(got["raw_grad_norm"], ref["raw_grad_norm"]), 1e-4)

  def test_kl_metric_matches_checker(self):
    rng = np.random.default_rng(3)
    for scale in (1.0, 5.0, 20.0):
      m = (rng.standard_normal((64, 1000)) * scale).astype(np.float32)
      g = (m + rng.standard_normal((64, 1000)) * scale * 0.3).astype(np.float32)
      mp = jnp.clip(jax.nn.softmax(jnp.asarray(m), axis=-1), min=1e-6)
      gp = jnp.clip(jax.nn.softmax(jnp.asarray(g), axis=-1), min=1e-6)
      gp, mp = gp / jnp.sum(gp, -1, keepdims=True), mp / jnp.sum(mp, -1, keepdims=True)
      want = np.asarray(jnp.sum(jax.scipy.special.kl_div(gp, mp), axis=-1))
      np.testing.assert_allclose(cl.kl_per_token(m, g), want, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
