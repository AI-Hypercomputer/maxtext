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

"""Chained full-model layerwise train step of DeepSeek-V4-Flash (284B) at the e2e pretrain config.

Reproduces step 1 of the 2_test_deepseek.sh pretrain stage (synthetic batch from PRNGKey(0), 64 x 4096, bf16) one
unit at a time on a small slice. Forward: S1 (embed + layers 0..2) -> B0..B18 -> S3 (layers 41..42 + head +
cross-entropy) per microbatch, keeping unit inputs on host. Backward: units in reverse with the real cotangent of
loss = lm_loss + indexer_loss + moe_lb_loss, accumulating each unit's weight gradients over microbatches. Reports
lm_loss / indexer_loss / moe_lb_loss and raw_grad_norm (comparable to the e2e step-1 metrics) plus per-unit gradient
norms, wall clock and memory.

--resume_dir writes every unit's output, cotangent and losses to a local or gs:// dir and skips finished units on
restart, so a preempted run loses at most one unit. --aux_safetensors replaces the hash-routing tables and router
biases with those of another run (e.g. the state a resumed e2e run restored), to replay that run's exact inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_layerwise as cl  # pylint: disable=g-import-not-at-top,wrong-import-position

from absl import logging as absl_logging  # pylint: disable=wrong-import-order,wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-order,wrong-import-position
import jax  # pylint: disable=wrong-import-order,wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-order,wrong-import-position
from maxtext.layers import mhc  # pylint: disable=wrong-import-position
from maxtext.layers.attention_mla import indexer_losses  # pylint: disable=wrong-import-position
from maxtext.trainers.pre_train import train_compile  # pylint: disable=wrong-import-position
from maxtext.utils import max_utils  # pylint: disable=wrong-import-position
from maxtext.utils import maxtext_utils  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-order,wrong-import-position
from safetensors.numpy import load_file as load_safetensors  # pylint: disable=wrong-import-order,wrong-import-position

vl, views = cl.vl, cl.views
TRAIN = views.MODEL_MODE_TRAIN


def pretrain_kwargs(args) -> dict:
  """2_test_deepseek.sh pretrain-stage flags (matmul precision and indexer_topk stay at their yml defaults)."""
  return {
      "attention": "dot_product",
      "use_tokamax_splash": False,
      "sparse_matmul": True,
      "use_tokamax_gmm": False,
      "megablox": args.megablox,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "per_device_batch_size": 1,
      "max_target_length": args.seq,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 1.0,
  }


def synthetic_batch(vocab_size: int, batch: int, seq: int) -> dict[str, np.ndarray]:
  """SyntheticDataIterator's batch: uniform tokens from PRNGKey(0), shifted targets, all-ones segmentation."""
  tokens = np.asarray(jax.random.randint(jax.random.PRNGKey(0), (batch, seq + 1), 0, vocab_size, dtype=jnp.int32))
  return {
      "inputs": tokens[:, :-1],
      "targets": tokens[:, 1:],
      "segs": np.ones((batch, seq), np.int32),
      "pos": np.broadcast_to(np.arange(seq, dtype=np.int32), (batch, seq)),
  }


def _cat(xs) -> jax.Array:
  return jnp.concatenate([jnp.atleast_1d(x).astype(jnp.float32).ravel() for x in xs]) if xs else jnp.zeros((0,))


def _aux_losses(m) -> tuple[jax.Array, jax.Array]:
  """Per-layer indexer and load-balance losses sown during the unit, collected as in train.py loss_fn."""
  idx_state = nnx.pop(m, indexer_losses)
  inter = nnx.pop(m, nnx.Intermediate).to_pure_dict()
  inter["indexer_losses"] = idx_state.to_pure_dict()
  idx = maxtext_utils.collect_intermediates_by_suffix(inter, "indexer_loss")
  lb = maxtext_utils.collect_intermediates_by_suffix(inter, "moe_lb_loss")
  return _cat(idx), _cat(lb)


def unit_fn(sv, unit: str):
  """(h, params, bias, batch) -> (h_out or xent_sum, indexer losses, lb losses) for one unit."""
  cfg = sv.cfg

  def fn(h, p, b, inputs, targets, segs, pos):  # pylint: disable=too-many-positional-arguments
    m = sv._merge(p, b)  # pylint: disable=protected-access
    d = m.decoder
    if unit == "S1":
      y = d._apply_embedding(m.token_embedder, inputs, pos, True, TRAIN)  # pylint: disable=protected-access
      h = mhc.get_functions(cfg.mhc_expansion_rate)[0](y)
      for i in range(cfg.first_num_hash_layers):
        h, _ = getattr(d, f"layers_{i}")(
            h, segs, pos, True, TRAIN, previous_chunk=None, slot=None, decoder_input_tokens=inputs
        )
    else:
      h, _, _ = d._apply_layers_sequentially(  # pylint: disable=protected-access
          d.scanned_blocks,
          h,
          segs,
          pos,
          True,
          TRAIN,
          length=sv.num_blocks,
          metadata_axis_name="scanned_blocks",
          previous_chunk=None,
          slot=None,
          decoder_input_tokens=inputs,
      )
    out = h
    if unit == "S3":
      logits = d.apply_output_head(m.token_embedder, d.hc_head(h), True, TRAIN)
      xent, _ = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(targets, cfg.vocab_size), z_loss=0.0)
      out = jnp.sum(xent * (segs != 0))
    idx, lb = _aux_losses(m)
    return out, idx, lb

  return fn


def vjp_accumulate(fn):
  """jit(acc, h, p, b, batch, ct) -> (acc + dparams in fp32, dh); acc is donated."""

  def run(acc, h, p, b, inputs, targets, segs, pos, ct):  # pylint: disable=too-many-positional-arguments
    _, pull = jax.vjp(lambda h_, p_: fn(h_, p_, b, inputs, targets, segs, pos), h, p)
    dh, dp = pull(ct)
    return jax.tree.map(lambda a, g: a + g.astype(jnp.float32), acc, dp), dh

  return jax.jit(run, donate_argnums=(0,))


def tree_sq_norms(tree) -> dict[str, float]:
  """Squared L2 norm per leaf, keyed by the leaf's tree path."""
  flat, _ = jax.tree_util.tree_flatten_with_path(tree)
  return {jax.tree_util.keystr(path): float(jnp.sum(jnp.square(v))) for path, v in flat}


def apply_aux_override(model, cfg, unit: str, aux: dict[str, np.ndarray]) -> list[str]:
  """Replaces the restored hash-routing tables (S1) or router biases (scanned units) of `unit` with `aux`.

  `aux` holds tid2eid_layers_{l} (vocab, k) and gate_bias_layers_{j} (n_blocks, num_experts), the layout of a
  scan_layers=True train state. Abstract (AOT) leaves are shape-checked only.
  """
  state = nnx.state(model)
  entries = views.flat_state(state)
  todo = {}
  if unit == "S1":
    for l in range(cfg.first_num_hash_layers):
      todo[f"Tid2EidVar-decoder-layers_{l}-mlp-MoeBlock_0-tid2eid"] = aux[f"tid2eid_layers_{l}"]
  else:
    block = cl.UNITS.index(unit) - 1
    for j in range(2):
      key = f"MoEBiasVar-decoder-scanned_blocks-layers_{j}-mlp-MoeBlock_0-gate-bias"
      todo[key] = aux[f"gate_bias_layers_{j}"][block : block + 1]
  for key, arr in todo.items():
    var = entries[key]
    old = var.get_value()
    if tuple(arr.shape) != tuple(old.shape):
      raise ValueError(f"aux override {key}: shape {arr.shape} != model {old.shape}")
    if isinstance(old, jax.ShapeDtypeStruct):
      continue
    new = np.asarray(arr).astype(old.dtype)
    sharding = getattr(old, "sharding", None)
    var.set_value(jax.device_put(new, sharding) if sharding is not None else new)
  nnx.update(model, state)
  return sorted(todo)


class ResumeStore:
  """Per-unit chain state in a local or gs:// dir: arrays as .npy, small records as .json (the commit marker)."""

  def __init__(self, root: str, tmp_dir: str = "/tmp"):
    self.root = root.rstrip("/")
    self.gcs = self.root.startswith("gs://")
    self.tmp_dir = tmp_dir
    if not self.gcs:
      os.makedirs(self.root, exist_ok=True)

  def _path(self, name: str) -> str:
    return f"{self.root}/{name}"

  def exists(self, name: str) -> bool:
    if self.gcs:
      cmd = ["gcloud", "storage", "objects", "describe", self._path(name)]
      return subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    return os.path.exists(self._path(name))

  def _put_file(self, local: str, name: str) -> None:
    if self.gcs:
      subprocess.run(["gcloud", "storage", "cp", "-q", local, self._path(name)], check=True)
    else:
      shutil.copyfile(local, self._path(name) + ".tmp")
      os.replace(self._path(name) + ".tmp", self._path(name))

  def save_array(self, name: str, arr: np.ndarray) -> None:
    raw = arr.view(np.uint16) if arr.dtype == jnp.bfloat16 else arr
    fd, local = tempfile.mkstemp(suffix=".npy", dir=self.tmp_dir)
    try:
      with os.fdopen(fd, "wb") as f:
        np.save(f, raw)
      self._put_file(local, name + ".npy")
    finally:
      os.remove(local)

  def load_array(self, name: str, dtype) -> np.ndarray:
    """Reads an array written by save_array, restoring its dtype (bf16 is stored as uint16)."""
    if self.gcs:
      fd, local = tempfile.mkstemp(suffix=".npy", dir=self.tmp_dir)
      os.close(fd)
      try:
        subprocess.run(["gcloud", "storage", "cp", "-q", self._path(name + ".npy"), local], check=True)
        raw = np.load(local)
      finally:
        os.remove(local)
    else:
      raw = np.load(self._path(name + ".npy"))
    return raw.view(jnp.bfloat16) if np.dtype(dtype) == jnp.bfloat16 else raw.astype(dtype, copy=False)

  def save_json(self, name: str, obj) -> None:
    data = json.dumps(obj).encode()
    if self.gcs:
      subprocess.run(["gcloud", "storage", "cp", "-q", "-", self._path(name)], input=data, check=True)
    else:
      with open(self._path(name) + ".tmp", "wb") as f:
        f.write(data)
      os.replace(self._path(name) + ".tmp", self._path(name))

  def load_json(self, name: str):
    if self.gcs:
      cmd = ["gcloud", "storage", "cat", self._path(name)]
      return json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
    with open(self._path(name), encoding="utf-8") as f:
      return json.load(f)


class _AsyncSaver:
  """Saves unit records in order on one background thread; the .json marker is written after its arrays."""

  def __init__(self, store: ResumeStore | None):
    self.store = store
    self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if store else None
    self.futures = []

  def submit(self, arrays: dict[str, np.ndarray], marker: str, record: dict) -> None:
    if self.store is None:
      return

    def task():
      for k, v in arrays.items():
        self.store.save_array(k, v)
      self.store.save_json(marker, record)

    self.futures.append(self.pool.submit(task))

  def wait(self) -> None:
    for f in self.futures:
      f.result()
    self.futures = []

  def drain(self) -> None:
    """Finishes queued saves, ignoring their errors (used while another exception propagates)."""
    for f in self.futures:
      with contextlib.suppress(Exception):
        f.result()
    self.futures = []


def run_chain(
    units: list[str],
    restore: Callable,
    batch_fn: Callable,
    put: Callable,
    *,
    store: ResumeStore | None = None,
    **kwargs,
) -> dict:
  """Runs the chained forward (and backward) pass and returns losses, raw_grad_norm and per-unit records.

  restore(unit) -> (sv with .params / .bias, restore seconds); batch_fn(i) -> (inputs, targets, segs, pos) of
  microbatch i on device; put(host array) -> device array; kwargs as in _run_chain (make_fn(sv, unit) -> unit
  function as in unit_fn). With `store`, finished units are skipped and their outputs, cotangents and losses read
  back from it; if the run fails, records of the units that did finish are flushed before the error propagates.
  """
  saver = _AsyncSaver(store)
  try:
    return _run_chain(units, restore, batch_fn, put, store=store, saver=saver, **kwargs)
  except BaseException:
    saver.drain()
    raise


def _run_chain(
    units: list[str],
    restore: Callable,
    batch_fn: Callable,
    put: Callable,
    *,
    n_mb: int,
    h_shape: tuple[int, ...],
    total_weights: float,
    store: ResumeStore | None,
    saver: _AsyncSaver,
    ctx: Callable = contextlib.nullcontext,
    make_fn: Callable = unit_fn,
    h_dtype=jnp.bfloat16,
    skip_backward: bool = False,
    mem_fn: Callable = cl.device_mem_gib,
) -> dict:
  """Body of run_chain."""
  mb = h_shape[0]
  prev = {u: (units[i - 1] if i else None) for i, u in enumerate(units)}
  nxt = {u: (units[i + 1] if i + 1 < len(units) else None) for i, u in enumerate(units)}
  do_backward = not skip_backward and units[-1] == "S3"
  keep_inputs = store is None and do_backward
  zeros_in = np.zeros((n_mb * mb,) + tuple(h_shape[1:]), h_dtype)

  def unit_input(unit: str, h_prev):
    if unit == units[0]:
      return zeros_in
    return h_prev if h_prev is not None else store.load_array(f"h_{prev[unit]}", h_dtype)

  unit_inputs: dict[str, np.ndarray] = {}
  h_host = None
  xent_sum, idx_vals, lb_vals, timings = 0.0, {}, {}, []
  for unit in units:
    if store is not None and store.exists(f"fwd_{unit}.json"):
      rec = store.load_json(f"fwd_{unit}.json")
      xent_sum = rec["xent_sum"]
      idx_vals[unit], lb_vals[unit] = np.asarray(rec["idx"], np.float32), np.asarray(rec["lb"], np.float32)
      h_host = None
      timings.append({"unit": unit, "pass": "fwd", "resumed": True})
      vl.log(f"FWD {unit}: resumed from {store.root}")
      continue
    h_in_all = unit_input(unit, h_host)
    sv, t_restore = restore(unit)
    fwd = jax.jit(make_fn(sv, unit))
    if keep_inputs:
      unit_inputs[unit] = h_in_all
    outs, idx_u, lb_u = [], [], []
    t1 = time.time()
    with ctx():
      for i in range(n_mb):
        out, idx, lb = fwd(put(h_in_all[i * mb : (i + 1) * mb]), sv.params, sv.bias, *batch_fn(i))
        if unit == "S3":
          xent_sum += float(out)
        else:
          outs.append(np.asarray(out))
        idx_u.append(np.asarray(idx))
        lb_u.append(np.asarray(lb))
    h_host = np.concatenate(outs) if unit != "S3" else None
    idx_vals[unit], lb_vals[unit] = np.stack(idx_u), np.stack(lb_u)
    saver.submit(
        {f"h_{unit}": h_host} if h_host is not None else {},
        f"fwd_{unit}.json",
        {"xent_sum": xent_sum, "idx": idx_vals[unit].tolist(), "lb": lb_vals[unit].tolist()},
    )
    rec = {"unit": unit, "pass": "fwd", "restore_s": t_restore, "run_s": time.time() - t1, "rss_peak_gib": vl.rss_gib()}
    rec.update(mem_fn())
    timings.append(rec)
    vl.log(f"FWD {unit}: restore {t_restore:.1f}s run {rec['run_s']:.1f}s hbm_peak {rec.get('hbm_peak_gib', 0):.1f} GiB")
    del sv, fwd, h_in_all
    gc.collect()
  saver.wait()

  n_idx = sum(v.shape[1] for v in idx_vals.values())
  n_lb = sum(v.shape[1] for v in lb_vals.values())
  lm_loss = xent_sum / total_weights
  indexer_loss = float(sum(v.sum() for v in idx_vals.values()) / (n_idx * n_mb)) if n_idx else 0.0
  moe_lb_loss = float(sum(v.sum() for v in lb_vals.values()) / (n_lb * n_mb)) if n_lb else 0.0
  results = {
      "lm_loss": lm_loss,
      "indexer_loss": indexer_loss,
      "moe_lb_loss_microbatched": moe_lb_loss,
      "loss": lm_loss + indexer_loss + moe_lb_loss,
      "n_indexer_entries": n_idx,
      "n_lb_entries": n_lb,
  }
  vl.log(f"FORWARD LOSSES {json.dumps(results)}")

  if do_backward:
    ct_host = None
    grad_sq = {}
    for unit in reversed(units):
      if store is not None and store.exists(f"bwd_{unit}.json"):
        grad_sq.update(store.load_json(f"bwd_{unit}.json")["grad_sq"])
        ct_host = None
        timings.append({"unit": unit, "pass": "bwd", "resumed": True})
        vl.log(f"BWD {unit}: resumed from {store.root}")
        continue
      if unit != "S3" and ct_host is None:
        ct_host = store.load_array(f"ct_{nxt[unit]}", h_dtype)
      h_in_all = unit_inputs.pop(unit) if keep_inputs else unit_input(unit, None)
      sv, t_restore = restore(unit)
      bwd = vjp_accumulate(make_fn(sv, unit))
      acc = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32, device=p.sharding), sv.params)
      new_ct = []
      t1 = time.time()
      with ctx():
        for i in range(n_mb):
          sl = slice(i * mb, (i + 1) * mb)
          ct_idx = jnp.full(idx_vals[unit].shape[1:], 1.0 / (n_idx * n_mb), jnp.float32)
          ct_lb = jnp.full(lb_vals[unit].shape[1:], 1.0 / (n_lb * n_mb) if n_lb else 0.0, jnp.float32)
          ct_main = jnp.float32(1.0 / total_weights) if unit == "S3" else put(ct_host[sl])
          acc, dh = bwd(acc, put(h_in_all[sl]), sv.params, sv.bias, *batch_fn(i), (ct_main, ct_idx, ct_lb))
          if unit != units[0]:
            new_ct.append(np.asarray(dh))
      ct_host = np.concatenate(new_ct) if unit != units[0] else None
      unit_sq = {f"{unit}/{k}": v for k, v in tree_sq_norms(acc).items() if v > 0}
      grad_sq.update(unit_sq)
      unit_norm = float(np.sqrt(sum(unit_sq.values())))
      saver.submit(
          {f"ct_{unit}": ct_host} if ct_host is not None else {},
          f"bwd_{unit}.json",
          {"grad_sq": unit_sq, "grad_norm": unit_norm},
      )
      rec = {"unit": unit, "pass": "bwd", "restore_s": t_restore, "run_s": time.time() - t1, "rss_peak_gib": vl.rss_gib()}
      rec.update(mem_fn(), grad_norm=unit_norm)
      timings.append(rec)
      vl.log(f"BWD {unit}: restore {t_restore:.1f}s run {rec['run_s']:.1f}s grad_norm {unit_norm:.4f}")
      del sv, bwd, acc, h_in_all
      gc.collect()
    saver.wait()
    results["raw_grad_norm"] = float(np.sqrt(sum(grad_sq.values())))
    results["top_grad_leaves"] = sorted(((k, float(np.sqrt(v))) for k, v in grad_sq.items()), key=lambda kv: -kv[1])[:10]
    vl.log(f"RAW_GRAD_NORM {results['raw_grad_norm']:.4f}")
  results["units"] = timings
  return results


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seq", type=int, default=4096)
  parser.add_argument("--global_batch", type=int, default=64, help="e2e: per_device_batch_size=1 x 64 chips")
  parser.add_argument("--microbatch", type=int, default=1)
  parser.add_argument("--megablox", type=lambda s: s.lower() == "true", default=True)
  parser.add_argument("--unscanned_ckpt", default=vl.DEFAULT_UNSCANNED_CKPT)
  parser.add_argument("--tid2eid_path", default=vl.DEFAULT_TID2EID_PATH)
  parser.add_argument("--units", default=",".join(cl.UNITS), help="prefix of the chain, for smoke runs")
  parser.add_argument("--skip_backward", action="store_true")
  parser.add_argument("--out_json", default="")
  parser.add_argument("--override", action="append", default=[], help="extra pyconfig key=value")
  parser.add_argument("--aot_topology", default="", help="e.g. v5p-8: only AOT-compile S1/B0/S3 forward and VJP")
  parser.add_argument("--resume_dir", default="", help="local or gs:// dir for per-unit outputs; skips finished units")
  parser.add_argument(
      "--aux_safetensors",
      default="",
      help="safetensors with tid2eid_layers_{l} / gate_bias_layers_{j} that replace the restored routing state",
  )
  args = parser.parse_args()
  absl_logging.set_verbosity(absl_logging.WARNING)
  units = args.units.split(",")
  if units != list(cl.UNITS[: len(units)]):
    raise ValueError(f"--units must be a prefix of {cl.UNITS}")
  if args.aot_topology:
    units = [u for u in units if u in ("S1", "B0", "S3")]
  jax.config.update("jax_compilation_cache_dir", "/tmp/jax_chain_cache")
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

  t_start = time.time()
  cfg = cl.make_chain_config(pretrain_kwargs(args), aot_topology=args.aot_topology, overrides=args.override)
  vl.replicate_batch_axis(cfg)
  mesh = train_compile.get_topology_mesh(cfg) if args.aot_topology else views.maxtext_utils.get_mesh_from_config(cfg)
  model = vl.init_model_on_cpu(cfg, mesh)
  mb, n_mb = args.microbatch, args.global_batch // args.microbatch
  data = synthetic_batch(cfg.vocab_size, args.global_batch, args.seq)
  total_weights = float(data["segs"].sum())
  h_shape = (mb, args.seq, cfg.mhc_expansion_rate, cfg.emb_dim)
  vl.log(
      f"Config: seq={args.seq} batch={args.global_batch} microbatch={mb} megablox={cfg.megablox} "
      f"indexer_topk={cfg.indexer_topk} matmul_precision={cfg.matmul_precision} mesh={dict(mesh.shape)} "
      f"tokens[0,:4]={data['inputs'][0, :4].tolist()} aux={args.aux_safetensors or 'checkpoint'}"
  )
  rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  aux = None
  if args.aux_safetensors:
    local = args.aux_safetensors
    if local.startswith("gs://"):
      local = os.path.join("/tmp", os.path.basename(local))
      subprocess.run(["gcloud", "storage", "cp", args.aux_safetensors, local], check=True)
    aux = load_safetensors(local)

  def restore(unit):
    t0 = time.time()
    vl.restore_subgroup_weights(
        model,
        cfg,
        mesh,
        unit,
        1,
        unscanned_ckpt=args.unscanned_ckpt,
        tid2eid_path=args.tid2eid_path,
        abstract=bool(args.aot_topology),
        with_globals=unit == "S1",
    )
    if aux is not None:
      vl.log(f"{unit}: routing state replaced from {args.aux_safetensors}: {apply_aux_override(model, cfg, unit, aux)}")
    return views.MaxTextSubgroups(model, cfg, mesh), time.time() - t0

  if args.aot_topology:
    for unit in units:
      sv, _ = restore(unit)
      fn = unit_fn(sv, unit)
      h_abs = jax.ShapeDtypeStruct(h_shape, jnp.bfloat16, sharding=rep)
      b_abs = tuple(jax.ShapeDtypeStruct((mb, args.seq), jnp.int32, sharding=rep) for _ in range(4))
      with views.maxtext_context(cfg, mesh):
        out = jax.eval_shape(fn, h_abs, sv.params, sv.bias, *b_abs)
        ct = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=rep), out)
        acc = jax.tree.map(lambda p: jax.ShapeDtypeStruct(p.shape, jnp.float32, sharding=p.sharding), sv.params)
        for kind, f, f_args in (
            ("fwd", jax.jit(fn), (h_abs, sv.params, sv.bias, *b_abs)),
            ("vjp_acc", vjp_accumulate(fn), (acc, h_abs, sv.params, sv.bias, *b_abs, ct)),
        ):
          mem = f.lower(*f_args).compile().memory_analysis()
          vl.log(
              f"AOT OK {unit}.{kind} ({args.aot_topology}): args {mem.argument_size_in_bytes / 2**30:.2f} out "
              f"{mem.output_size_in_bytes / 2**30:.2f} alias {mem.alias_size_in_bytes / 2**30:.2f} temp "
              f"{mem.temp_size_in_bytes / 2**30:.2f} GiB per device"
          )
      del sv, fn
      gc.collect()
    vl.log(f"AOT PASSED for {units}.")
    return

  store = None
  if args.resume_dir:
    store = ResumeStore(args.resume_dir)
    # Refuse to resume a different configuration: a stale state dir silently replacing inputs is exactly the
    # failure this harness exists to catch.
    run_cfg = {
        k: getattr(args, k)
        for k in ("seq", "global_batch", "microbatch", "megablox", "unscanned_ckpt", "tid2eid_path", "override")
    }
    run_cfg.update(aux_safetensors=args.aux_safetensors, mesh={k: int(v) for k, v in mesh.shape.items()})
    if store.exists("run_config.json"):
      if store.load_json("run_config.json") != json.loads(json.dumps(run_cfg)):
        raise ValueError(f"{args.resume_dir} holds a run with a different config: {store.load_json('run_config.json')}")
      vl.log(f"Resuming from {args.resume_dir}")
    else:
      store.save_json("run_config.json", run_cfg)

  def batch_fn(i):
    sl = slice(i * mb, (i + 1) * mb)
    return tuple(jax.device_put(data[k][sl], rep) for k in ("inputs", "targets", "segs", "pos"))

  results = run_chain(
      units,
      restore,
      batch_fn,
      lambda x: jax.device_put(x, rep),
      n_mb=n_mb,
      h_shape=h_shape,
      total_weights=total_weights,
      ctx=lambda: views.maxtext_context(cfg, mesh),
      skip_backward=args.skip_backward,
      store=store,
  )
  results["total_s"] = time.time() - t_start
  if args.out_json:
    with open(args.out_json, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
  print("CHAIN_PRETRAIN_RESULTS " + json.dumps({k: v for k, v in results.items() if k != "units"}), flush=True)


if __name__ == "__main__":
  main()
