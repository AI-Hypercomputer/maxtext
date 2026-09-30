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

"""Real-weight DeepSeek-V4-Flash (284B) layer-subgroup forward/backward verifier.

Restores only the active subgroup leaves from the unscanned Orbax checkpoint
(plus tid2eid.safetensors for hash-routed layers), executes the production
MaxText subgroup on CPU (float32) or TPU (bfloat16/float32), and compares
forward outputs, router/indexer top-k selections, input VJPs (dh_in), non-expert
weight gradients (dW), and 256-expert Rademacher gradient sketches against a
reference bundle produced by the official DeepSeek-V4 reference runner.

Subgroups:
  S1: layers 0..2 (unrolled prefix: SWA+hash layers 0..1, CSA+indexer+hash layer 2)
  S2: layers 3..6 (2 scanned HCA+CSA blocks with top-k routed MoE)
  S3: layers 41..42 + hc_head + decoder_norm + logits_dense + masked cross-entropy
"""

from __future__ import annotations

import argparse
import collections
from concurrent import futures
import contextlib
import dataclasses
import functools
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if os.path.isdir(os.path.join(REPO_ROOT, "src", "maxtext")):
  sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
  sys.path.insert(1, REPO_ROOT)

from absl import logging as absl_logging
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.layers import attention_compressed
from maxtext.layers import moe
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import sharding as mt_sharding
import numpy as np
import orbax.checkpoint as ocp
from safetensors.torch import load_file
from tests.utils import deepseek4_layerwise as views
import torch

N_PROJ = 8
DEFAULT_UNSCANNED_CKPT = "gs://maxtext-deepseek/deepseek4-284b/2026-09-17/unscanned/0/items"
DEFAULT_TID2EID_PATH = "gs://maxtext-deepseek/deepseek4-284b/2026-09-17/tid2eid.safetensors"


def rss_gib() -> float:
  """Returns peak resident set size of the current process in GiB."""
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def log(msg: str) -> None:
  """Prints a timestamped log line with current peak RSS."""
  print(f"[{time.strftime('%H:%M:%S')} rss_peak={rss_gib():.1f}GiB] {msg}", flush=True)


def stable_seed(text: str) -> int:
  """Returns a deterministic 63-bit integer seed from sha256(text)."""
  return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63)


def rademacher(leaf_name: str, j: int, shape: tuple[int, ...]) -> torch.Tensor:
  """Generates +/-1 int8 Rademacher probe j for leaf_name matching SKETCH.md."""
  g = torch.Generator(device="cpu").manual_seed(stable_seed(f"{leaf_name}#{j}"))
  return torch.randint(0, 2, tuple(shape), generator=g, dtype=torch.int8) * 2 - 1


def compare_arrs(actual, reference, label: str = "") -> dict[str, float | str]:
  """Computes max_abs, rel_l2, and cosine similarity in float64."""
  actual = np.asarray(actual, dtype=np.float64).flatten()
  reference = np.asarray(reference, dtype=np.float64).flatten()
  diff = actual - reference
  ref_norm = float(np.linalg.norm(reference))
  act_norm = float(np.linalg.norm(actual))
  diff_norm = float(np.linalg.norm(diff))
  rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else (0.0 if act_norm == 0 else float("inf"))
  if act_norm > 0 and ref_norm > 0:
    cos = float(np.dot(actual, reference) / (act_norm * ref_norm))
  else:
    cos = 1.0 if act_norm == ref_norm else 0.0
  max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
  return {"label": label, "max_abs": max_abs, "rel_l2": rel_l2, "cos": cos}


def compare_sketches(got_sketch, ref_sketch, label: str = "") -> dict[str, dict]:
  """Compares [256, 9] expert gradient sketches (col 0 norm + cols 1..8 projections)."""
  got = np.asarray(got_sketch, dtype=np.float64)
  ref = np.asarray(ref_sketch, dtype=np.float64)
  return {
      "overall": compare_arrs(got, ref, label=f"{label}.overall"),
      "col0_norm": compare_arrs(got[:, 0], ref[:, 0], label=f"{label}.col0_norm"),
      "cols1_8_projs": compare_arrs(got[:, 1:], ref[:, 1:], label=f"{label}.cols1_8_projs"),
  }


def topk_slot_overlap(actual: np.ndarray, reference: np.ndarray) -> float:
  """Mean fraction of reference top-k index set per token also chosen by actual."""
  k = reference.shape[-1]
  a = np.asarray(actual).reshape(-1, k)
  r = np.asarray(reference).reshape(-1, k)
  overlaps = []
  for row_a, row_r in zip(a, r):
    sa = set(int(x) for x in row_a if x >= 0)
    sr = set(int(x) for x in row_r if x >= 0)
    overlaps.append(1.0 if not sr else len(sa & sr) / len(sr))
  return float(np.mean(overlaps))


def _meta_tree(path: str):
  m = ocp.PyTreeCheckpointer().metadata(path)
  t = getattr(m, "item_metadata", m)
  return getattr(t, "tree", t)


def _flat_tree(tree: dict) -> dict[tuple[str, ...], object]:
  """Flattens a nested dict into tuple-keyed leaves."""
  out = {}

  def rec(d, prefix):
    for k, v in d.items():
      if isinstance(v, dict):
        rec(v, prefix + (k,))
      else:
        out[prefix + (k,)] = v

  rec(tree, ())
  return out


def _unflat_tree(d: dict[tuple[str, ...], object]) -> dict:
  root = {}
  for kp, v in d.items():
    cur = root
    for k in kp[:-1]:
      cur = cur.setdefault(k, {})
    cur[kp[-1]] = v
  return root


def orbax_partial_restore(path: str, sel: dict[tuple[str, ...], object]) -> dict[tuple[str, ...], np.ndarray]:
  """Restores only the selected leaves from an Orbax PyTree checkpoint onto CPU."""
  cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
  item, rargs = {}, {}
  for kp, m in sel.items():
    gs = tuple(m.shape)
    item[kp] = jax.ShapeDtypeStruct(gs, m.dtype, sharding=cpu_sharding)
    rargs[kp] = ocp.ArrayRestoreArgs(sharding=cpu_sharding, global_shape=gs, dtype=m.dtype, strict=True)
  ckptr = ocp.PyTreeCheckpointer()
  out = ckptr.restore(
      path,
      args=ocp.args.PyTreeRestore(item=_unflat_tree(item), restore_args=_unflat_tree(rargs), partial_restore=True),
  )
  return _flat_tree(out)


def _var_sharding(var, mesh) -> jax.sharding.NamedSharding:
  """Returns NamedSharding for an NNX Variable, accounting for param_scan_axis and indivisible shapes."""
  sharding_obj = mt_sharding.get_nnx_var_named_sharding_with_scan_axis(var, mesh).get_value()
  if isinstance(sharding_obj, jax.sharding.NamedSharding):
    pspec = mt_sharding.adjust_pspec_for_indivisible_shapes(sharding_obj.spec, var.get_value().shape, mesh)
    return jax.sharding.NamedSharding(mesh, pspec)
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _set_entry(entries: dict, key: str, arr, populated: set[str]) -> None:
  if key not in entries:
    raise KeyError(f"Missing model state entry: {key}")
  var = entries[key]
  target = var.get_value()
  if tuple(arr.shape) != tuple(target.shape):
    raise ValueError(f"Shape mismatch for {key}: got {arr.shape}, expected {target.shape}")
  var.set_value(jnp.asarray(arr, dtype=target.dtype))
  populated.add(key)


@contextlib.contextmanager
def capture_topk_any_mesh(records: list):
  """Records MoE and CSA indexer top-k indices on both 1-device CPU and multi-device TPU meshes."""
  if jax.device_count() == 1:
    with views.capture_maxtext_topk(records):
      yield records
    return
  orig_topk = moe.RoutedMoE.get_topk
  orig_indexer = attention_compressed.DeepseekV4Indexer.__call__

  def get_topk(self, gate_logits, *args, **kwargs):
    weights, indices = orig_topk(self, gate_logits, *args, **kwargs)
    cb = functools.partial(views._record, records, "moe", bool(self.is_hash_routing))  # pylint: disable=protected-access
    jax.debug.callback(cb, indices, gate_logits, ordered=False)
    return weights, indices

  def indexer_call(self, *args, **kwargs):
    out = orig_indexer(self, *args, **kwargs)
    jax.debug.callback(functools.partial(views._record, records, "indexer", None), out[0], ordered=False)  # pylint: disable=protected-access
    return out

  moe.RoutedMoE.get_topk = get_topk
  attention_compressed.DeepseekV4Indexer.__call__ = indexer_call
  try:
    yield records
  finally:
    moe.RoutedMoE.get_topk = orig_topk
    attention_compressed.DeepseekV4Indexer.__call__ = orig_indexer


def ensure_local_bundle(bundle_dir: str, sg_name: str, num_cotangents: int) -> str:
  """Downloads bundle files from GCS to /tmp if bundle_dir is a gs:// URI."""
  if not bundle_dir.startswith("gs://"):
    return os.path.join(bundle_dir, sg_name)
  tag = hashlib.sha256(bundle_dir.encode()).hexdigest()[:10]
  local_dir = os.path.join("/tmp/ds4_layerwise_bundles", tag, sg_name)
  os.makedirs(local_dir, exist_ok=True)
  needed = ["manifest.json", "forward.safetensors", "expert_sketch.safetensors"]
  for i in range(num_cotangents):
    needed.append(f"vjp_k{i}.safetensors")
  if sg_name == "S3":
    needed.append("vjp_loss.safetensors")
  for fname in needed:
    dst = os.path.join(local_dir, fname)
    if not os.path.exists(dst):
      src = f"{bundle_dir.rstrip('/')}/{sg_name}/{fname}"
      log(f"Downloading {src} -> {dst}...")
      subprocess.run(["gcloud", "storage", "cp", src, dst], check=True)
  return local_dir


def restore_subgroup_weights(
    model,
    mt_config,
    mesh,
    subgroup_name: str,
    num_blocks: int,
    *,
    unscanned_ckpt: str,
    tid2eid_path: str,
    abstract: bool = False,
) -> set[str]:
  """Restores active subgroup weights from unscanned_ckpt (+ tid2eid) and shards onto mesh.

  With `abstract`, jit-argument leaves (Param, MoEBiasVar) become sharded ShapeDtypeStructs for AOT
  compilation against a device-less topology mesh; other Variables stay concrete (trace constants).
  """
  log(f"Restoring weights for {subgroup_name} from {unscanned_ckpt}...")
  cpu_dev = jax.devices("cpu")[0]
  state = nnx.state(model)
  entries = views.flat_state(state)
  fm = _flat_tree(_meta_tree(unscanned_ckpt))

  if subgroup_name == "S1":
    u_layers = [0, 1, 2]
    needs_globals = False
  elif subgroup_name == "S2":
    u_layers = [3, 4, 5, 6]
    needs_globals = False
  elif subgroup_name == "S3":
    u_layers = [41, 42]
    needs_globals = True
  else:
    raise ValueError(f"Unknown subgroup {subgroup_name}")

  needed_u = {f"layers_{l}" for l in u_layers}
  if needs_globals:
    needed_u.update(["token_embedder", "decoder_norm", "hc_head", "logits_dense"])

  sel = {}
  for kp, m in fm.items():
    for u in needed_u:
      if u in kp or (u == "token_embedder" and kp[:3] == ("params", "params", "token_embedder")):
        sel[kp] = m
        break

  log(f"Selected {len(sel)} leaves from unscanned checkpoint. Restoring to host CPU...")
  t0 = time.time()
  restored = orbax_partial_restore(unscanned_ckpt, sel)
  log(f"Restored {len(restored)} leaves in {time.time() - t0:.1f}s.")

  layer_leaves = collections.defaultdict(dict)
  globals_leaves = {}
  for kp, arr in restored.items():
    arr = np.asarray(arr)
    if "token_embedder" in kp:
      globals_leaves["token_embedder"] = arr
    elif "decoder_norm" in kp:
      globals_leaves["decoder_norm"] = arr
    elif "logits_dense" in kp:
      globals_leaves["logits_dense"] = arr
    elif "hc_head" in kp:
      globals_leaves[f"hc_head_{kp[-1]}"] = arr
    else:
      for l in u_layers:
        if f"layers_{l}" in kp:
          idx = kp.index(f"layers_{l}")
          layer_leaves[l][kp[idx + 1 :]] = arr
          break

  if subgroup_name == "S1":
    tmp_tid = "/tmp/tid2eid_284b.safetensors"
    if not os.path.exists(tmp_tid):
      log(f"Downloading {tid2eid_path} -> {tmp_tid}...")
      subprocess.run(["gcloud", "storage", "cp", tid2eid_path, tmp_tid], check=True)
    t_tid = load_file(tmp_tid)
    max_eid = int(t_tid["layers_0"].max())
    if max_eid >= mt_config.num_experts:
      raise ValueError(f"Invalid tid2eid max expert id {max_eid} >= num_experts {mt_config.num_experts}")
    for l in (0, 1, 2):
      layer_leaves[l]["tid2eid"] = t_tid[f"layers_{l}"].numpy()
    log(f"Loaded 284B tid2eid (max_eid={max_eid}).")

  populated: set[str] = set()
  with jax.default_device(cpu_dev):
    if needs_globals:
      _set_entry(entries, "params-token_embedder-embedding", globals_leaves["token_embedder"], populated)
      _set_entry(entries, "params-decoder-decoder_norm-scale", globals_leaves["decoder_norm"], populated)
      _set_entry(entries, "params-decoder-hc_head-hc_base", globals_leaves["hc_head_hc_base"], populated)
      _set_entry(entries, "params-decoder-hc_head-hc_fn", globals_leaves["hc_head_hc_fn"], populated)
      _set_entry(entries, "params-decoder-hc_head-hc_scale", globals_leaves["hc_head_hc_scale"], populated)
      if "params-decoder-logits_dense-kernel" in entries:
        _set_entry(entries, "params-decoder-logits_dense-kernel", globals_leaves["logits_dense"], populated)

    if subgroup_name == "S1":
      for l in (0, 1, 2):
        for subpath, arr in layer_leaves[l].items():
          if subpath == "tid2eid":
            k = f"Tid2EidVar-decoder-layers_{l}-mlp-MoeBlock_0-tid2eid"
          else:
            k = f"params-decoder-layers_{l}-" + "-".join(subpath)
          _set_entry(entries, k, arr, populated)
      expected_active = {k for k in entries if any(f"-layers_{l}-" in k and "scanned_blocks" not in k for l in (0, 1, 2))}
    else:
      for j in range(2):
        target_layers = [u_layers[2 * b + j] for b in range(num_blocks)]
        subpaths = list(layer_leaves[target_layers[0]].keys())
        for subpath in subpaths:
          if subpath == ("mlp", "MoeBlock_0", "gate", "bias"):
            stacked = np.stack([layer_leaves[l][subpath] for l in target_layers], axis=0)
            k = f"MoEBiasVar-decoder-scanned_blocks-layers_{j}-mlp-MoeBlock_0-gate-bias"
          else:
            stacked = np.stack([layer_leaves[l][subpath] for l in target_layers], axis=1)
            k = f"params-decoder-scanned_blocks-layers_{j}-" + "-".join(subpath)
          _set_entry(entries, k, stacked, populated)
      expected_active = {k for k in entries if "scanned_blocks" in k}
      if needs_globals:
        expected_active |= {
            k for k in entries if any(g in k for g in ("token_embedder", "decoder_norm", "hc_head", "logits_dense"))
        }

  missing_active = sorted(expected_active - populated)
  if missing_active:
    raise RuntimeError(f"Active subgroup leaves left unpopulated: {missing_active}")

  log(f"Populated {len(populated)} active leaves (0 missing). Sharding active leaves across mesh {dict(mesh.shape)}...")
  with views.maxtext_context(mt_config, mesh):
    for k, var in entries.items():
      is_arg = isinstance(var, (nnx.Param, moe.MoEBiasVar))
      if abstract and is_arg:
        value = var.get_value() if k in populated else np.zeros((), np.float32)
        sharding = (
            _var_sharding(var, mesh) if k in populated else jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        )
        var.set_value(jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding))
      elif abstract:
        if k not in populated:
          var.set_value(np.zeros((), dtype=np.float32))
      elif k in populated:
        sharding = _var_sharding(var, mesh)
        var.set_value(jax.device_put(var.get_value(), sharding))
      else:
        var.set_value(jnp.zeros((), dtype=jnp.float32))

  nnx.update(model, state)
  del restored, layer_leaves, globals_leaves
  gc.collect()
  log("Weight restoration and device placement complete.")
  return populated


def expert_key_mapping(sg_name: str, num_blocks: int):
  """Returns (maxtext_key, native_layer_id, native_w_name, block_index) for routed experts."""
  mapping = []
  if sg_name == "S1":
    for l in (0, 1, 2):
      mapping.append((f"params-decoder-layers_{l}-mlp-MoeBlock_0-wi_0", l, "w1", False))
      mapping.append((f"params-decoder-layers_{l}-mlp-MoeBlock_0-wi_1", l, "w3", False))
      mapping.append((f"params-decoder-layers_{l}-mlp-MoeBlock_0-wo", l, "w2", False))
  elif sg_name in ("S2", "S3"):
    u_layers = [3, 4, 5, 6] if sg_name == "S2" else [41, 42]
    for j in range(2):
      for b in range(num_blocks):
        actual_l = u_layers[2 * b + j]
        mapping.append((f"params-decoder-scanned_blocks-layers_{j}-mlp-MoeBlock_0-wi_0", actual_l, "w1", b))
        mapping.append((f"params-decoder-scanned_blocks-layers_{j}-mlp-MoeBlock_0-wi_1", actual_l, "w3", b))
        mapping.append((f"params-decoder-scanned_blocks-layers_{j}-mlp-MoeBlock_0-wo", actual_l, "w2", b))
  return mapping


def _sketch_one_expert(args):
  """Computes [norm, <R_0, dW>, ..., <R_7, dW>] for one expert weight leaf."""
  e, dW_e, actual_l, w_name, shape = args
  row = np.zeros(1 + N_PROJ, dtype=np.float64)
  dW_t = torch.from_numpy(np.ascontiguousarray(dW_e)).to(torch.float64).flatten()
  norm = float(dW_t.norm())
  row[0] = norm
  if norm > 0:
    leaf_name = f"layers.{actual_l}.ffn.experts.{e}.{w_name}.weight"
    for j in range(N_PROJ):
      R_j = rademacher(leaf_name, j, shape).flatten().to(torch.float64)
      row[1 + j] = float(torch.dot(R_j, dW_t))
  return e, row


def compute_expert_sketches(dparams_flat: dict[str, np.ndarray], sg_name: str, num_blocks: int) -> dict[str, np.ndarray]:
  """Computes [256, 9] float64 sketches for all expert leaves in the subgroup."""
  sketches = {}
  with futures.ThreadPoolExecutor(max_workers=32) as pool:
    for mt_key, actual_l, w_name, b_idx in expert_key_mapping(sg_name, num_blocks):
      val_np = np.asarray(dparams_flat[mt_key], dtype=np.float32)
      if b_idx is not False:
        val_np = val_np[:, b_idx]
      sketch_out = np.zeros((256, 1 + N_PROJ), dtype=np.float64)
      shape = (2048, 4096) if w_name in ("w1", "w3") else (4096, 2048)
      tasks = [(e, val_np[e].T, actual_l, w_name, shape) for e in range(256)]
      for e, row in pool.map(_sketch_one_expert, tasks):
        sketch_out[e] = row
      sketches[f"layers.{actual_l}.ffn.experts.{w_name}.sketch"] = sketch_out
  return sketches


def _gib(num_bytes: int) -> float:
  return num_bytes / 2**30


def aot_compile_subgroup(fns: dict, sv, *, h_shape: tuple[int, ...], act_dtype) -> None:
  """AOT-compiles each fn in `fns` (h, params, bias) -> out and its VJP for the topology mesh `sv.mesh`."""
  rep = jax.sharding.NamedSharding(sv.mesh, jax.sharding.PartitionSpec())
  h = jax.ShapeDtypeStruct(h_shape, act_dtype, sharding=rep)
  topo = sv.cfg.compile_topology
  with views.maxtext_context(sv.cfg, sv.mesh):
    for name, fn in fns.items():
      out = jax.eval_shape(fn, h, sv.params, sv.bias)
      ct = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=rep), out)

      def vjp(h_, p_, b_, v_, fn=fn):
        return jax.vjp(fn, h_, p_, b_)[1](v_)

      for kind, f, f_args in (("fwd", fn, (h, sv.params, sv.bias)), ("vjp", vjp, (h, sv.params, sv.bias, ct))):
        t0 = time.time()
        mem = jax.jit(f).lower(*f_args).compile().memory_analysis()
        log(
            f"AOT OK {name}.{kind} ({topo}) in {time.time() - t0:.1f}s: args {_gib(mem.argument_size_in_bytes):.2f} "
            f"GiB, out {_gib(mem.output_size_in_bytes):.2f} GiB, temp {_gib(mem.temp_size_in_bytes):.2f} GiB per device"
        )
  log(f"AOT PASSED: {sorted(fns)} forward and VJP compile for {topo}.")


def main() -> None:
  """CLI entry point for real-weight layer-subgroup forward/backward verification."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--subgroup", required=True, choices=["S1", "S2", "S3"])
  parser.add_argument("--bundle_dir", required=True)
  parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  parser.add_argument("--num_cotangents", type=int, default=1)
  parser.add_argument("--unscanned_ckpt", default=DEFAULT_UNSCANNED_CKPT)
  parser.add_argument("--tid2eid_path", default=DEFAULT_TID2EID_PATH)
  parser.add_argument("--out_json", default=None)
  parser.add_argument("--assert_pass", action="store_true")
  parser.add_argument("--megablox", default="auto", choices=["auto", "true", "false"])
  parser.add_argument(
      "--aot_topology", default="", help="e.g. v5p-8: only AOT-compile forward/VJP for this TPU topology."
  )
  args_cli = parser.parse_args()
  absl_logging.set_verbosity(absl_logging.WARNING)
  # TPU uses the pretrain-stage MoE kernel (megablox GMM, 2_test_deepseek.sh); CPU cannot compile Mosaic
  # kernels, so it uses jax.lax.ragged_dot.
  on_tpu = bool(args_cli.aot_topology) or jax.default_backend() == "tpu"
  use_megablox = args_cli.megablox == "true" or (args_cli.megablox == "auto" and on_tpu)
  topo_kwargs = {"compile_topology": args_cli.aot_topology, "compile_topology_num_slices": 1}

  sg_name = args_cli.subgroup
  act_jnp_dtype = jnp.bfloat16 if args_cli.dtype == "bfloat16" else jnp.float32
  bundle_path = ensure_local_bundle(args_cli.bundle_dir, sg_name, args_cli.num_cotangents)
  manifest_path = os.path.join(bundle_path, "manifest.json")
  fwd_bundle_path = os.path.join(bundle_path, "forward.safetensors")
  sketch_bundle_path = os.path.join(bundle_path, "expert_sketch.safetensors")

  log(
      f"--- Starting MaxText verification for {sg_name} (dtype={args_cli.dtype}, "
      f"backend={jax.default_backend()}, devices={jax.device_count()}) ---"
  )
  log(f"Bundle dir: {bundle_path}")

  fwd_tensors = load_file(fwd_bundle_path)
  sketch_tensors = load_file(sketch_bundle_path)
  with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

  seq = 4096
  num_layers = 5 if sg_name in ("S1", "S3") else 7
  num_blocks = 1 if sg_name in ("S1", "S3") else 2
  num_devices = jax.device_count()

  ma = manifest["model_args"]
  fields = {f.name for f in dataclasses.fields(views.ref_model.ModelArgs)}
  ma = {k: v for k, v in ma.items() if k in fields}
  ref_args = dataclasses.replace(views.ref_model.ModelArgs(**ma), max_batch_size=1)

  t0_init = time.time()
  log(
      f"Initializing MaxText model for {sg_name} (num_layers={num_layers}, dtype={args_cli.dtype}, devices={num_devices})..."
  )
  mt_config = views.make_config(
      ref_args,
      seq=seq,
      dtype=args_cli.dtype,
      num_layers=num_layers,
      scan_layers=True,
      model_name="deepseek4-284b",
      ici_fsdp_parallelism=-1,
      per_device_batch_size=1,
      megablox=use_megablox,
      **(topo_kwargs if args_cli.aot_topology else {}),
  )
  # Replicate batch=1 activations while keeping all model parameters FSDP-sharded across the mesh.
  object.__setattr__(
      mt_config,
      "logical_axis_rules",
      tuple(
          (
              k,
              tuple(ax for ax in (v if isinstance(v, (list, tuple)) else (v,)) if ax not in ("fsdp", "fsdp_transpose"))
              if "batch" in k
              else v,
          )
          for k, v in mt_config.logical_axis_rules
      ),
  )
  if args_cli.aot_topology:
    mesh = train_compile.get_topology_mesh(mt_config)
  else:
    mesh = views.maxtext_utils.get_mesh_from_config(mt_config)
  log(f"megablox={use_megablox}, mesh={dict(mesh.shape)}")
  cpu_dev = jax.devices("cpu")[0]
  # Eager init runs on a 1-CPU mesh: a topology mesh has no devices, and a TPU mesh would stack unsharded scanned
  # layers on one chip's HBM. Modules still capture `mesh`; restore_subgroup_weights places active leaves onto it.
  init_mesh = jax.sharding.Mesh(
      np.array([cpu_dev]).reshape((1,) * len(mesh.axis_names)), mesh.axis_names, axis_types=mesh.axis_types
  )
  with jax.default_device(cpu_dev), views.maxtext_context(mt_config, init_mesh):
    model = views.models.Transformer(
        mt_config,
        mesh,
        views.quantizations.configure_quantization(mt_config),
        model_mode=views.MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=0),
    )
  log(f"Model skeleton initialized on CPU in {time.time() - t0_init:.1f}s.")

  active_keys = restore_subgroup_weights(
      model,
      mt_config,
      mesh,
      sg_name,
      num_blocks,
      unscanned_ckpt=args_cli.unscanned_ckpt,
      tid2eid_path=args_cli.tid2eid_path,
      abstract=bool(args_cli.aot_topology),
  )

  subgroup_views = views.MaxTextSubgroups(model, mt_config, mesh)
  params, bias = subgroup_views.params, subgroup_views.bias

  h_in = jnp.asarray(fwd_tensors["h_in"].float().numpy(), dtype=act_jnp_dtype)
  tokens = jnp.asarray(fwd_tensors["input_ids"].numpy(), dtype=jnp.int32)
  segs = jnp.ones((1, seq), dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (1, seq))

  targets = loss_weights = None
  if sg_name == "S3":
    targets = jnp.asarray(fwd_tensors["targets"].numpy(), dtype=jnp.int32)
    loss_weights = jnp.asarray(fwd_tensors["loss_weights"].float().numpy(), dtype=jnp.float32)

  if args_cli.aot_topology:
    sv = subgroup_views
    aot_fns = {"blocks": lambda h_, p_, b_: sv.blocks(h_, p_, b_, tokens, segs, pos)}
    if sg_name == "S1":
      aot_fns = {"s1": lambda h_, p_, b_: sv.s1(h_, p_, b_, tokens, segs, pos)}
    elif sg_name == "S3":
      aot_fns["s3"] = lambda h_, p_, b_: sv.s3(h_, p_, b_, tokens, segs, pos, targets, loss_weights)
    aot_compile_subgroup(aot_fns, sv, h_shape=h_in.shape, act_dtype=act_jnp_dtype)
    return

  records = []
  log("Running forward pass and capturing top-k selections...")
  t0_fwd = time.time()

  out = out_blocks = bwd_fn = bwd_blocks = bwd_loss = None
  with views.maxtext_context(mt_config, mesh), capture_topk_any_mesh(records):
    if sg_name == "S1":
      out = jax.jit(lambda h_, p_, b_: subgroup_views.s1(h_, p_, b_, tokens, segs, pos))(h_in, params, bias)
      jax.block_until_ready(out)
    elif sg_name == "S2":
      out = jax.jit(lambda h_, p_, b_: subgroup_views.blocks(h_, p_, b_, tokens, segs, pos))(h_in, params, bias)
      jax.block_until_ready(out)
    elif sg_name == "S3":
      out_blocks = jax.jit(lambda h_, p_, b_: subgroup_views.blocks(h_, p_, b_, tokens, segs, pos))(h_in, params, bias)
      jax.block_until_ready(out_blocks)

  if sg_name == "S1":
    bwd_fn = jax.jit(
        lambda h_, p_, b_, v_: jax.vjp(
            lambda h__, p__, b__: subgroup_views.s1(h__, p__, b__, tokens, segs, pos), h_, p_, b_
        )[1](v_)
    )
  elif sg_name == "S2":
    bwd_fn = jax.jit(
        lambda h_, p_, b_, v_: jax.vjp(
            lambda h__, p__, b__: subgroup_views.blocks(h__, p__, b__, tokens, segs, pos), h_, p_, b_
        )[1](v_)
    )
  elif sg_name == "S3":
    with views.maxtext_context(mt_config, mesh):
      out = jax.jit(lambda h_, p_, b_: subgroup_views.s3(h_, p_, b_, tokens, segs, pos, targets, loss_weights))(
          h_in, params, bias
      )
      jax.block_until_ready(out)
    bwd_blocks = jax.jit(
        lambda h_, p_, b_, v_: jax.vjp(
            lambda h__, p__, b__: subgroup_views.blocks(h__, p__, b__, tokens, segs, pos), h_, p_, b_
        )[1](v_)
    )
    bwd_loss = jax.jit(
        lambda h_, p_, b_, ct_: jax.vjp(
            lambda h__, p__, b__: subgroup_views.s3(h__, p__, b__, tokens, segs, pos, targets, loss_weights),
            h_,
            p_,
            b_,
        )[1](ct_)
    )

  fwd_s = time.time() - t0_fwd
  log(f"Forward completed in {fwd_s:.1f}s.")

  results = {
      "subgroup": sg_name,
      "dtype": args_cli.dtype,
      "backend": jax.default_backend(),
      "device_count": num_devices,
      "forward": {},
      "topk": {},
      "topk_slot_overlap": {},
      "backward": {},
      "timings": {"fwd_s": fwd_s},
  }

  if sg_name in ("S1", "S2"):
    h_out_ref = fwd_tensors["h_out"].float().numpy()
    h_cmp = compare_arrs(np.asarray(out, dtype=np.float32), h_out_ref, "h_out")
    results["forward"]["h_out"] = h_cmp
    log(f"h_out: max_abs={h_cmp['max_abs']:.4e} rel_l2={h_cmp['rel_l2']:.4e} cos={h_cmp['cos']:.6f}")
  else:
    logits, xent_sum = out
    with views.maxtext_context(mt_config, mesh):
      m_temp = subgroup_views._merge(params, bias)  # pylint: disable=protected-access
      hidden = m_temp.decoder.hc_head(out_blocks)
      head_in_mt = m_temp.decoder.decoder_norm(hidden)

    results["forward"]["h_out"] = compare_arrs(
        np.asarray(out_blocks, dtype=np.float32), fwd_tensors["h_out"].float().numpy(), "h_out"
    )
    results["forward"]["head_in"] = compare_arrs(
        np.asarray(head_in_mt, dtype=np.float32), fwd_tensors["head_in"].float().numpy(), "head_in"
    )
    results["forward"]["xent_sum"] = compare_arrs(
        np.asarray(xent_sum, dtype=np.float32), fwd_tensors["xent_sum"].float().numpy(), "xent_sum"
    )
    results["forward"]["total_weights"] = compare_arrs(
        np.asarray(jnp.sum(loss_weights), dtype=np.float32),
        fwd_tensors["total_weights"].float().numpy(),
        "total_weights",
    )

    lf = jnp.asarray(logits, dtype=jnp.float32)
    logsumexp_mt = jax.scipy.special.logsumexp(lf, axis=-1)
    argmax_mt = jnp.argmax(lf, axis=-1).astype(jnp.int32)
    logits_target_mt = jnp.take_along_axis(lf, jnp.expand_dims(targets, -1), axis=-1).squeeze(-1)
    logits_last_pos_mt = lf[:, -1, :]

    results["forward"]["logits_logsumexp"] = compare_arrs(
        logsumexp_mt, fwd_tensors["logits_logsumexp"].float().numpy(), "logits_logsumexp"
    )
    results["forward"]["logits_argmax_agreement"] = float(
        np.mean(np.asarray(argmax_mt) == fwd_tensors["logits_argmax"].numpy())
    )
    results["forward"]["logits_target"] = compare_arrs(
        logits_target_mt, fwd_tensors["logits_target"].float().numpy(), "logits_target"
    )
    results["forward"]["logits_last_pos"] = compare_arrs(
        logits_last_pos_mt, fwd_tensors["logits_last_pos"].float().numpy(), "logits_last_pos"
    )
    fwd_r = results["forward"]
    log(
        f"S3 forward checks: h_out rel_l2={fwd_r['h_out']['rel_l2']:.4e} "
        f"head_in rel_l2={fwd_r['head_in']['rel_l2']:.4e} "
        f"xent_sum rel_l2={fwd_r['xent_sum']['rel_l2']:.4e} "
        f"argmax_agree={fwd_r['logits_argmax_agreement']:.4f}"
    )

  log("Checking routing and indexer top-k agreements...")
  if sg_name == "S1":
    ref_l, idx_l = [0, 1, 2], [2]
  elif sg_name == "S2":
    ref_l, idx_l = [3, 4, 5, 6], [4, 6]
  else:
    ref_l, idx_l = [41, 42], [42]

  moe_raw = [r for r in records if r[0] == "moe"]
  moe_step = len(moe_raw) // len(ref_l) if (moe_raw and len(moe_raw) % len(ref_l) == 0) else 1
  moe_recs = moe_raw[::moe_step]
  idx_raw = [r for r in records if r[0] == "indexer"]
  idx_step = len(idx_raw) // len(idx_l) if (idx_raw and len(idx_raw) % len(idx_l) == 0) else 1
  idx_recs = idx_raw[::idx_step]

  for rec, l in zip(moe_recs, ref_l):
    mt_indices = rec[2][:1]
    ref_indices = fwd_tensors[f"router.{l}.indices"].numpy()
    results["topk"][f"router.{l}.indices"] = views.topk_agreement(mt_indices, ref_indices)
    results["topk_slot_overlap"][f"router.{l}.indices"] = topk_slot_overlap(mt_indices, ref_indices)

  for rec, l in zip(idx_recs, idx_l):
    mt_idx = rec[2][:1]
    ref_idx = fwd_tensors[f"indexer.{l}.topk_idxs"].numpy()
    # Reference Indexer.forward adds seqlen to valid (>= 0) block indices; undo offset.
    ref_idx = np.where(ref_idx >= 0, ref_idx - seq, ref_idx)
    results["topk"][f"indexer.{l}.topk_idxs"] = views.topk_agreement(mt_idx, ref_idx)
    results["topk_slot_overlap"][f"indexer.{l}.topk_idxs"] = topk_slot_overlap(mt_idx, ref_idx)

  for k, v in results["topk"].items():
    ov = results["topk_slot_overlap"][k]
    log(f"Top-k {k}: exact_row_agree = {v:.4f}, slot_set_overlap = {ov:.4f}")

  num_cotangents = args_cli.num_cotangents
  vjp_tags = [f"k{i}" for i in range(num_cotangents)]
  if sg_name == "S3":
    vjp_tags.append("loss")

  hf_cfg_dict = views.hf_config_from_ref_args(ref_args, num_layers=num_layers)
  layer_map = {3: 41, 4: 42} if sg_name == "S3" else None

  for tag in vjp_tags:
    log(f"--- Running VJP for {tag} ---")
    t0_vjp = time.time()
    vjp_file = os.path.join(bundle_path, f"vjp_{tag}.safetensors")
    vjp_tensors = load_file(vjp_file)

    with views.maxtext_context(mt_config, mesh):
      if tag == "loss":
        ct_loss = (jnp.zeros_like(logits), jnp.ones_like(xent_sum))
        dh_in, dparams, dbias = bwd_loss(h_in, params, bias, ct_loss)  # pylint: disable=not-callable
      else:
        target_out = out_blocks if sg_name == "S3" else out
        v_cotangent = jnp.asarray(vjp_tensors["v"].float().numpy(), dtype=target_out.dtype)
        if sg_name == "S3":
          dh_in, dparams, dbias = bwd_blocks(h_in, params, bias, v_cotangent)  # pylint: disable=not-callable
        else:
          dh_in, dparams, dbias = bwd_fn(h_in, params, bias, v_cotangent)  # pylint: disable=not-callable
      jax.block_until_ready(dh_in)

    vjp_time = time.time() - t0_vjp
    log(f"VJP {tag} computed in {vjp_time:.1f}s.")

    dh_in_ref = vjp_tensors["dh_in"].float().numpy()
    dh_in_cmp = compare_arrs(np.asarray(dh_in, dtype=np.float32), dh_in_ref, f"dh_in_{tag}")
    log(f"dh_in {tag}: max_abs={dh_in_cmp['max_abs']:.4e} rel_l2={dh_in_cmp['rel_l2']:.4e} cos={dh_in_cmp['cos']:.6f}")

    log(f"Converting active non-expert dparams for {tag} to native keys...")
    mt_flat_active = {}
    for path, leaf in nnx.to_flat_state(dparams):
      k = views._flat_key("params", path)  # pylint: disable=protected-access
      if k in active_keys:
        val = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
        mt_flat_active[k] = np.asarray(val, dtype=np.float32)
    for path, leaf in nnx.to_flat_state(dbias):
      k = views._flat_key("MoEBiasVar", path)  # pylint: disable=protected-access
      if k in active_keys:
        val = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
        mt_flat_active[k] = np.asarray(val, dtype=np.float32)

    # Filter out routed expert stacks before native_from_maxtext (sketched separately below).
    mt_flat_non_expert = {
        k: v
        for k, v in mt_flat_active.items()
        if not k.endswith(("-MoeBlock_0-wi_0", "-MoeBlock_0-wi_1", "-MoeBlock_0-wo"))
    }
    native_dparams = views.native_from_maxtext(
        mt_flat_non_expert,
        mt_config,
        hf_cfg_dict,
        scan_layers=True,
        layer_map=layer_map,
        fix_stack_axis=True,
    )

    non_expert_comparisons = {}
    dW_ref_keys = [k for k in vjp_tensors if k.startswith("dW.")]
    non_expert_ref_keys = [k for k in dW_ref_keys if "experts" not in k or "shared_experts" in k]

    worst_rel_l2 = 0.0
    worst_key = ""
    worst_nonzero_rel_l2 = 0.0
    worst_nonzero_key = ""
    for k in non_expert_ref_keys:
      native_key = k[3:]
      ref_val = vjp_tensors[k].float().numpy()
      if native_key in native_dparams:
        got_val = native_dparams[native_key]
        cmp_res = compare_arrs(got_val, ref_val, native_key)
        non_expert_comparisons[native_key] = cmp_res
        if cmp_res["rel_l2"] > worst_rel_l2:
          worst_rel_l2 = cmp_res["rel_l2"]
          worst_key = native_key
        if float(np.linalg.norm(ref_val)) > 1e-6 and cmp_res["rel_l2"] > worst_nonzero_rel_l2:
          worst_nonzero_rel_l2 = cmp_res["rel_l2"]
          worst_nonzero_key = native_key
      else:
        log(f"WARNING: native key {native_key} not produced by native_from_maxtext!")
        non_expert_comparisons[native_key] = {"missing": True}

    log(
        f"Non-expert leaves compared: {len(non_expert_comparisons)}/{len(non_expert_ref_keys)}. "
        f"Worst rel_l2: {worst_rel_l2:.4e} ({worst_key}); "
        f"worst non-tiny-norm rel_l2: {worst_nonzero_rel_l2:.4e} ({worst_nonzero_key})"
    )

    log(f"Computing expert sketches for {tag}...")
    t0_sk = time.time()
    got_sketches = compute_expert_sketches(mt_flat_active, sg_name, num_blocks)
    sketch_time = time.time() - t0_sk
    log(f"Expert sketches computed in {sketch_time:.1f}s.")

    sketch_comparisons = {}
    for sk_key, got_sk in got_sketches.items():
      ref_sk_key = f"{tag}.{sk_key}"
      ref_sk = sketch_tensors[ref_sk_key].numpy()
      sk_cmp = compare_sketches(got_sk, ref_sk, ref_sk_key)
      sketch_comparisons[sk_key] = sk_cmp
      c0 = sk_cmp["col0_norm"]
      c18 = sk_cmp["cols1_8_projs"]
      log(
          f"  {sk_key}: col0_norm rel_l2={c0['rel_l2']:.4e} cols1_8_projs rel_l2={c18['rel_l2']:.4e} cos={c18['cos']:.6f}"
      )

    results["backward"][tag] = {
        "vjp_s": vjp_time,
        "sketch_s": sketch_time,
        "dh_in": dh_in_cmp,
        "non_expert": {
            "total_leaves": len(non_expert_ref_keys),
            "worst_leaf": worst_key,
            "worst_rel_l2": worst_rel_l2,
            "worst_nonzero_leaf": worst_nonzero_key,
            "worst_nonzero_rel_l2": worst_nonzero_rel_l2,
            "leaves": non_expert_comparisons,
        },
        "expert_sketches": sketch_comparisons,
    }
    # Release this cotangent's device gradients before the next VJP; v5p-8 HBM fits only one set.
    del dh_in, dparams, dbias

  results["peak_rss_gib"] = rss_gib()
  log(f"--- Verification for {sg_name} completed. Peak RSS: {results['peak_rss_gib']:.1f} GiB ---")

  if args_cli.out_json:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_json)), exist_ok=True)
    with open(args_cli.out_json, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
    log(f"Saved results to {args_cli.out_json}")

  fwd_r = results["forward"]
  print("\n" + "=" * 80)
  print(f"VERIFICATION SUMMARY: {sg_name} ({args_cli.dtype}, {jax.default_backend()}x{num_devices})")
  print("=" * 80)
  print(f"Forward h_out rel_l2: {fwd_r['h_out']['rel_l2']:.4e}, cos: {fwd_r['h_out']['cos']:.6f}")
  if sg_name == "S3":
    print(f"Forward head_in rel_l2: {fwd_r['head_in']['rel_l2']:.4e}")
    print(f"Forward xent_sum rel_l2: {fwd_r['xent_sum']['rel_l2']:.4e}")
    print(f"Logits argmax agreement: {fwd_r['logits_argmax_agreement']:.4f}")
  print("Top-k agreements (exact_row / slot_overlap):")
  for k, v in results["topk"].items():
    print(f"  {k}: {v:.4f} / {results['topk_slot_overlap'][k]:.4f}")
  for tag, b_res in results["backward"].items():
    dh_r = b_res["dh_in"]
    ne_r = b_res["non_expert"]
    print(f"Backward {tag}:")
    print(f"  dh_in rel_l2: {dh_r['rel_l2']:.4e}, cos: {dh_r['cos']:.6f}")
    print(
        f"  non-expert leaves ({ne_r['total_leaves']}): worst rel_l2 = {ne_r['worst_rel_l2']:.4e} ({ne_r['worst_leaf']})"
    )
    for sk_k, sk_v in b_res["expert_sketches"].items():
      c0 = sk_v["col0_norm"]
      c18 = sk_v["cols1_8_projs"]
      print(f"  {sk_k}: norm rel_l2 = {c0['rel_l2']:.4e}, projs rel_l2 = {c18['rel_l2']:.4e}, cos = {c18['cos']:.6f}")
  print(f"Peak RSS: {results['peak_rss_gib']:.1f} GiB")
  print("=" * 80 + "\n")

  if args_cli.assert_pass:
    max_fwd = 5e-2 if args_cli.dtype == "bfloat16" else 5e-4
    max_bwd = 1.5e-1 if args_cli.dtype == "bfloat16" else 5e-3
    min_ov = 0.90 if args_cli.dtype == "bfloat16" else 0.98
    fwd_rel = fwd_r["h_out"]["rel_l2"]
    if not fwd_rel <= max_fwd:  # NaN-safe.
      raise AssertionError(f"{sg_name} forward h_out rel_l2 {fwd_rel:.4e} > {max_fwd:.4e}")
    for k, ov in results["topk_slot_overlap"].items():
      if not ov >= min_ov:
        raise AssertionError(f"{sg_name} top-k slot overlap {k} = {ov:.4f} < {min_ov:.4f}")
    for tag, b_res in results["backward"].items():
      dh_rel = b_res["dh_in"]["rel_l2"]
      if not dh_rel <= max_bwd:
        raise AssertionError(f"{sg_name} backward {tag} dh_in rel_l2 {dh_rel:.4e} > {max_bwd:.4e}")
    log(f"ALL ASSERTIONS PASSED for {sg_name} ({args_cli.dtype}).")


if __name__ == "__main__":
  main()
