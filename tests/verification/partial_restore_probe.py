"""Probe partial / sliced restore from DSV4 Orbax OCDbt checkpoints.

Usage: python partial_restore_probe.py MODE
  MODE in {u_partial, s_orbax_slice, s_ts_slice, s_full_one, chunks}
"""

import asyncio
import os
import sys
import time

import jax
import numpy as np
import orbax.checkpoint as ocp
import tensorstore as ts

U = "gs://maxtext-deepseek/deepseek4-284b/2026-09-17/unscanned/0/items"
S = "gs://maxtext-deepseek/deepseek4-284b/2026-09-17/scanned/0/items"
N_SCAN = 20
SCAN_AXIS = 1
BLOCKS = 2
CPU = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])


def meta_tree(path):
  m = ocp.PyTreeCheckpointer().metadata(path)
  t = getattr(m, "item_metadata", m)
  return getattr(t, "tree", t)


def flat(tree):
  out = {}

  def rec(d, prefix):
    for k, v in d.items():
      if isinstance(v, dict):
        rec(v, prefix + (k,))
      else:
        out[prefix + (k,)] = v

  rec(tree, ())
  return out


def unflat(d):
  root = {}
  for kp, v in d.items():
    cur = root
    for k in kp[:-1]:
      cur = cur.setdefault(k, {})
    cur[kp[-1]] = v
  return root


def ts_bytes():
  """Sum of tensorstore kvstore read-byte counters."""
  tot = {}
  for m in ts.experimental_collect_matching_metrics("/tensorstore/kvstore/"):
    name = m["name"]
    if "bytes_read" in name or name.endswith("/read_bytes"):
      for v in m.get("values", []):
        val = v.get("value", 0)
        if isinstance(val, (int, float)):
          tot[name] = tot.get(name, 0) + val
  return tot


def diff_bytes(a, b):
  return {k: b.get(k, 0) - a.get(k, 0) for k in b if b.get(k, 0) - a.get(k, 0)}


def orbax_restore(path, sel, global_shapes=None, strict=True):
  """Restore selected leaves (dict kp->ArrayMetadata) as CPU jax arrays."""
  item, rargs = {}, {}
  for kp, m in sel.items():
    gs = tuple(global_shapes[kp]) if global_shapes else tuple(m.shape)
    item[kp] = jax.ShapeDtypeStruct(gs, m.dtype, sharding=CPU)
    rargs[kp] = ocp.ArrayRestoreArgs(sharding=CPU, global_shape=gs, dtype=m.dtype, strict=strict)
  ckptr = ocp.PyTreeCheckpointer()
  out = ckptr.restore(
      path, args=ocp.args.PyTreeRestore(item=unflat(item), restore_args=unflat(rargs), partial_restore=True)
  )
  return flat(out)


def report(name, arrs, t0, b0):
  dt = time.time() - t0
  nbytes = sum(np.asarray(a).nbytes for a in arrs.values())
  print(f"[{name}] leaves={len(arrs)} host_bytes={nbytes/2**30:.3f}GiB " f"wall={dt:.1f}s", flush=True)
  print(f"[{name}] ts_read_bytes_delta={diff_bytes(b0, ts_bytes())}", flush=True)


def is_u_target(kp):
  if kp[:3] == ("params", "params", "token_embedder"):
    return True
  return len(kp) > 3 and kp[:3] == ("params", "params", "decoder") and kp[3] in ("layers_0", "layers_1", "layers_2")


def u_partial():
  fm = flat(meta_tree(U))
  sel = {k: v for k, v in fm.items() if is_u_target(k)}
  print(f"[u_partial] selected {len(sel)} / {len(fm)} leaves", flush=True)
  t0, b0 = time.time(), ts_bytes()
  arrs = orbax_restore(U, sel)
  report("u_partial", arrs, t0, b0)
  print("\n== restored layers_2 leaves ==")
  for kp, a in sorted(arrs.items()):
    if "layers_2" in kp:
      print(f"  {'/'.join(kp)}\t{a.shape}\t{a.dtype}")
  tid = [kp for kp in fm if any("tid2eid" in str(k).lower() for k in kp)]
  print(f"\n== Tid2Eid leaves in U metadata: {len(tid)} ==")
  for kp in tid:
    if kp in arrs:
      a = np.asarray(arrs[kp])
      print(f"  {'/'.join(kp)} min={a.min()} max={a.max()} " f"nunique={np.unique(a).size}")
  for layer in ("layers_0", "layers_1", "layers_2"):
    kp = ("params", "params", "decoder", layer, "mlp", "MoeBlock_0", "gate", "bias")
    print(f"  {layer} gate/bias present: {kp in fm}")
  emb = np.asarray(arrs[("params", "params", "token_embedder", "embedding")])
  print(f"  token_embedder sample [0,:4]={emb[0,:4].astype(np.float32)}")


def s_targets():
  fm = flat(meta_tree(S))
  sel = {k: v for k, v in fm.items() if "scanned_blocks" in k}
  gshapes = {}
  for kp, m in sel.items():
    s = list(m.shape)
    assert s[SCAN_AXIS] == N_SCAN, (kp, s)
    s[SCAN_AXIS] = BLOCKS
    gshapes[kp] = tuple(s)
  return fm, sel, gshapes


def s_orbax_slice():
  """strict=False + smaller global_shape: orbax restricts TS domain to prefix."""
  _, sel, gshapes = s_targets()
  t0, b0 = time.time(), ts_bytes()
  arrs = orbax_restore(S, sel, gshapes, strict=False)
  report("s_orbax_slice", arrs, t0, b0)
  bad = [(kp, a.shape) for kp, a in arrs.items() if a.shape != gshapes[kp]]
  print(f"[s_orbax_slice] shape mismatches: {bad}")
  np.save(
      "/tmp/_probe_wq_a_orbax.npy",
      np.asarray(
          arrs[("params", "params", "decoder", "scanned_blocks", "layers_0", "self_attention", "wq_a", "kernel")]
      ).astype(np.float32),
  )


def ts_spec(path, kp):
  return {
      "driver": "zarr3",
      "kvstore": {
          "driver": "ocdbt",
          "base": path.rstrip("/") + "/",
          "path": ".".join(kp),
      },
  }


async def _ts_read(path, sel, gshapes):
  ctx = ts.Context({"cache_pool": {"total_bytes_limit": 0}, "file_io_concurrency": {"limit": 128}})

  async def one(kp):
    t = await ts.open(ts_spec(path, kp), open=True, read=True, context=ctx)
    idx = [slice(None)] * t.rank
    if gshapes is not None:
      idx[SCAN_AXIS] = slice(0, gshapes[kp][SCAN_AXIS])
    return kp, await t[tuple(idx)].read()

  res = await asyncio.gather(*[one(kp) for kp in sel])
  return dict(res)


def s_ts_slice():
  """Direct tensorstore open on the OCDbt kvstore + slice read."""
  _, sel, gshapes = s_targets()
  t0, b0 = time.time(), ts_bytes()
  arrs = asyncio.run(_ts_read(S, sel, gshapes))
  report("s_ts_slice", arrs, t0, b0)
  bad = [(kp, a.shape) for kp, a in arrs.items() if a.shape != gshapes[kp]]
  print(f"[s_ts_slice] shape mismatches: {bad}")
  # Cross-check scan mapping: S scanned_blocks/layers_j[:, b] == U layers_{3+2b+j}.
  wq = ("self_attention", "wq_a", "kernel")
  sub = {}
  for b in range(BLOCKS):
    for j in range(2):
      sub[("params", "params", "decoder", f"layers_{3+2*b+j}") + wq] = None
  ug = asyncio.run(_ts_read(U, sub, None))
  for b in range(BLOCKS):
    for j in range(2):
      s_arr = np.asarray(arrs[("params", "params", "decoder", "scanned_blocks", f"layers_{j}") + wq])[:, b]
      u_arr = np.asarray(ug[("params", "params", "decoder", f"layers_{3+2*b+j}") + wq])
      print(f"  S blocks[{b}].layers_{j} wq_a == U layers_{3+2*b+j}: " f"{np.array_equal(s_arr, u_arr)}")
  if os.path.exists("/tmp/_probe_wq_a_orbax.npy"):
    o = np.load("/tmp/_probe_wq_a_orbax.npy")
    t_ = np.asarray(arrs[("params", "params", "decoder", "scanned_blocks", "layers_0") + wq]).astype(np.float32)
    print(f"  orbax-slice == ts-slice (wq_a layers_0): {np.array_equal(o, t_)}")


def xcheck():
  """Scan mapping + orbax-vs-ts equality on the wq_a family only."""
  _, sel, gshapes = s_targets()
  wq = ("self_attention", "wq_a", "kernel")
  sel = {k: v for k, v in sel.items() if k[-3:] == wq}
  arrs = asyncio.run(_ts_read(S, sel, gshapes))
  sub = {("params", "params", "decoder", f"layers_{3+2*b+j}") + wq: None for b in range(BLOCKS) for j in range(2)}
  ug = asyncio.run(_ts_read(U, sub, None))
  for b in range(BLOCKS):
    for j in range(2):
      s_arr = np.asarray(arrs[("params", "params", "decoder", "scanned_blocks", f"layers_{j}") + wq])[:, b]
      u_arr = np.asarray(ug[("params", "params", "decoder", f"layers_{3+2*b+j}") + wq])
      print(f"  S blocks[{b}].layers_{j} wq_a == U layers_{3+2*b+j}: " f"{np.array_equal(s_arr, u_arr)}")
  o = np.load("/tmp/_probe_wq_a_orbax.npy")
  t_ = np.asarray(arrs[("params", "params", "decoder", "scanned_blocks", "layers_0") + wq]).astype(np.float32)
  print(f"  orbax-slice == ts-slice (wq_a layers_0): {np.array_equal(o, t_)}")


def s_full_one():
  """Full (unsliced) restore of one leaf family: scanned wq_a kernels."""
  fm = flat(meta_tree(S))
  sel = {k: v for k, v in fm.items() if "scanned_blocks" in k and k[-3:] == ("self_attention", "wq_a", "kernel")}
  t0, b0 = time.time(), ts_bytes()
  arrs = orbax_restore(S, sel)
  report("s_full_one", arrs, t0, b0)


async def _chunks(path, kps):
  for kp in kps:
    t = await ts.open(ts_spec(path, kp), open=True, read=True)
    cl = t.chunk_layout
    print(
        f"  {'/'.join(kp[3:])}\tshape={t.shape}\tread_chunk="
        f"{cl.read_chunk.shape}\twrite_chunk={cl.write_chunk.shape}\t"
        f"codecs={t.codec.to_json().get('codecs') if t.codec else None}"
    )


def chunks():
  fm = flat(meta_tree(S))
  kps = [k for k in fm if "scanned_blocks" in k and k[-1] in ("wi_0", "wo", "kernel", "scale", "bias")][:12]
  asyncio.run(_chunks(S, kps))
  fu = flat(meta_tree(U))
  asyncio.run(_chunks(U, [k for k in fu if "layers_2" in k][:4]))


_T0 = time.time()


if __name__ == "__main__":
  print("orbax", ocp.__version__, "jax", jax.__version__, flush=True)
  {
      "u_partial": u_partial,
      "s_orbax_slice": s_orbax_slice,
      "s_ts_slice": s_ts_slice,
      "s_full_one": s_full_one,
      "chunks": chunks,
      "xcheck": xcheck,
  }[sys.argv[1]]()
  import resource

  print(
      f"[{sys.argv[1]}] peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f}GiB "
      f"total_wall={time.time()-_T0:.1f}s",
      flush=True,
  )
