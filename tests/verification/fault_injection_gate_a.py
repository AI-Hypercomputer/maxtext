"""Gate A fault-injection probe: production Stage 1 logit check on CPU.

Mirrors tests/utils/forward_pass_logit_checker.py (from_pretrained, jitted nnx
forward, clip 1e-6 KL) and runs arms that corrupt the loaded model in place.

Env:
  ARMS     comma list of clean,f1,f2 (f3 = clean arm under float32_gate_logits=false)
  TAG      output prefix, e.g. /home/.../fault/out/a1
  GOLDENS  comma list of golden jsonl paths (first data point of each is used)
  F1_LAYERS  e.g. 20-27; F1_FRAC fraction of routed experts swapped (default 0.4)
Argv: MaxText config args (base.yml + overrides).
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import jsonlines
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import linears
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.forward_pass_logit_checker import get_data

CLIP = 1e-6
TID2EID = "/tmp/tid2eid_284b.safetensors"  # gs://maxtext-deepseek/deepseek4-284b/2026-09-17/tid2eid.safetensors


def log(msg):
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kl_per_pos(model_logits, golden_logits):
  """Checker formula: clip softmax at 1e-6, renormalize, sum kl_div over vocab."""
  v = min(model_logits.shape[-1], golden_logits.shape[-1])
  q = jnp.clip(jax.nn.softmax(model_logits[:, :v].astype(jnp.float32), -1), min=CLIP)
  p = jnp.clip(jax.nn.softmax(golden_logits[:, :v].astype(jnp.float32), -1), min=CLIP)
  q = q / q.sum(-1, keepdims=True)
  p = p / p.sum(-1, keepdims=True)
  return np.asarray(jnp.sum(jax.scipy.special.kl_div(p, q), -1))


def summarize(kl):
  i = int(np.argmax(kl))
  return {
      "max": float(kl[i]),
      "argmax": i,
      "median": float(np.median(kl)),
      "mean": float(np.mean(kl)),
      "p99": float(np.quantile(kl, 0.99)),
      "n_gt_0.42": int((kl > 0.42).sum()),
  }


def make_eval(config, graphdef):
  @jax.jit
  def step(state, ids, pos, seg):
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      m = nnx.merge(graphdef, state)
      return m(
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          encoder_images=None,
          enable_dropout=False,
      )

  return step


def f1_targets(state, layers, num_experts):
  out = []
  for path, var in nnx.to_flat_state(state):
    names = [str(p) for p in path]
    if names[-1] not in ("wi_0", "wi_1", "wo") or "shared_experts" in names:
      continue
    layer = [n for n in names if n.startswith("layers_")]
    if not layer or int(layer[0].split("_")[1]) not in layers:
      continue
    val = var.get_value()
    if val.ndim == 3 and val.shape[0] == num_experts:
      out.append(("/".join(names), var))
  return out


def inject_f1(targets, num_experts, frac, seed=0):
  """Swap a cyclic permutation of `frac` of the experts (same ids across wi_0/wi_1/wo)."""
  saved, info = [], {}
  by_layer = {}
  for name, var in targets:
    layer = [n for n in name.split("/") if n.startswith("layers_")][0]
    by_layer.setdefault(layer, []).append((name, var))
  rng = np.random.default_rng(seed)
  for layer, items in sorted(by_layer.items()):
    sel = np.sort(rng.choice(num_experts, int(round(frac * num_experts)), replace=False))
    src = np.roll(sel, 1)
    info[layer] = sel.tolist()
    for name, var in items:
      old = var.get_value()
      new = old.at[sel].set(old[src])
      var.set_value(new)
      saved.append((var, old))
      diff = int(np.asarray(jnp.any(new != old, axis=(1, 2))).sum())
      log(f"F1 {name} shape={old.shape} experts_changed={diff}")
  return saved, info


def shared_mlps(model):
  return [
      m
      for _, m in nnx.iter_graph(model)
      if isinstance(m, linears.MlpBlock) and getattr(m, "activations_limit", None) is not None
  ]


def sideload_tid2eid(model, config, path=TID2EID):
  """CPU from_pretrained frees tid2eid before restore and never refills it; load it from the sideload file."""
  from safetensors.numpy import load_file  # pylint: disable=import-outside-toplevel

  tables = load_file(path)
  for i in range(config.first_num_hash_layers):
    var = getattr(model.decoder, f"layers_{i}").mlp.MoeBlock_0.tid2eid
    old = var.get_value()
    t = tables[f"layers_{i}"]
    assert t.shape == old.shape and int(t.max()) < config.num_experts, (t.shape, old.shape)
    log(f"tid2eid layers_{i}: was_deleted={old.is_deleted()} -> sideloaded {t.shape}")
    var.set_value(jnp.asarray(t.astype(np.float32), dtype=old.dtype))
  deleted = [
      "/".join(map(str, p))
      for p, v in nnx.to_flat_state(nnx.state(model))
      if isinstance(v, nnx.Variable) and isinstance(v.get_value(), jax.Array) and v.get_value().is_deleted()
  ]
  assert not deleted, f"deleted leaves after restore: {deleted[:10]}"


def main():
  arms = os.environ["ARMS"].split(",")
  tag = os.environ["TAG"]
  goldens = os.environ["GOLDENS"].split(",")
  lo, hi = map(int, os.environ.get("F1_LAYERS", "20-27").split("-"))
  frac = float(os.environ.get("F1_FRAC", "0.4"))
  config = pyconfig.initialize_pydantic([sys.argv[0]] + sys.argv[1:])
  log(
      f"cfg float32_gate_logits={config.float32_gate_logits} indexer_topk={config.indexer_topk} "
      f"mlp_activations_limit={config.mlp_activations_limit} dtype={config.dtype}"
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
  t0 = time.time()
  model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  log(f"loaded model in {time.time() - t0:.0f}s")
  sideload_tid2eid(model, config)
  graphdef, state = nnx.split(model)

  data = []
  for g in goldens:
    with jsonlines.open(g) as f:
      point = next(iter(f))
    ids, seg, pos, glog, seq_len, _ = get_data(point, config)
    data.append((os.path.basename(g), ids, seg, pos, np.asarray(glog), seq_len))
  # All goldens must share the prompt so one forward serves all of them.
  for d in data[1:]:
    assert np.array_equal(d[1], data[0][1]), f"prompt mismatch: {d[0]}"
  _, ids, seg, pos, _, seq_len = data[0]
  log(f"seq_len={seq_len} goldens={[d[0] for d in data]}")

  results = {}
  for arm in arms:
    saved, extra = [], {}
    gdef = graphdef
    if arm == "f1":
      tg = f1_targets(state, set(range(lo, hi + 1)), config.num_experts)
      assert len(tg) == 3 * (hi - lo + 1), f"F1 found {len(tg)} targets: {[t[0] for t in tg]}"
      saved, extra["swapped"] = inject_f1(tg, config.num_experts, frac)
    elif arm == "f2":
      mlps = shared_mlps(model)
      log(f"F2 clamped shared MlpBlocks: {len(mlps)}")
      assert mlps, "no clamped shared MlpBlock found"
      extra["n_unclamped"] = len(mlps)
      for m in mlps:
        m.activations_limit = None
      gdef, _ = nnx.split(model)
    t1 = time.time()
    logits = make_eval(config, gdef)(state, ids, pos, seg)
    logits = np.asarray(logits[0, :seq_len].astype(jnp.float32))
    log(f"arm={arm} forward {time.time() - t1:.0f}s")
    np.save(f"{tag}_{arm}_logits.npy", logits)
    results[arm] = {"extra": extra}
    for name, *_, glog, _ in data:
      kl = kl_per_pos(logits, glog[:seq_len])
      np.save(f"{tag}_{arm}_{name}_kl.npy", kl)
      s = summarize(kl)
      results[arm][name] = s
      log(
          f"SUMMARY arm={arm} golden={name} "
          + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
      )
    for var, old in saved:
      var.set_value(old)
    if arm == "f2":
      for m in mlps:
        m.activations_limit = config.mlp_activations_limit
    with open(f"{tag}_results.json", "w") as f:
      json.dump(results, f, indent=1)
  log("DONE")


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  main()
