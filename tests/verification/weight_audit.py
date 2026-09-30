"""Audit public MaxText DeepSeek-V4-Flash checkpoint vs native bf16 safetensors.

Uses the PRODUCTION converter mapping/hooks (param_mapping.py) and the production
hook-application code paths (to_huggingface._get_model_mappings,
utils.process_maxtext_param, to_maxtext._get_hf_loading_function).

Usage: python weight_audit.py [MODE]
  MODE: full (default) | dry (accounting only) | smoke (globals + layer 0)
"""

import asyncio
import collections
import hashlib
import json
import os
import re
import resource
import sys
import time

WT_SRC = "/home/yaoyuchen_google_com/wt-layerwise/src"
AUDIT = "/home/yaoyuchen_google_com/golden_gen/layerwise/audit"
sys.path.insert(0, WT_SRC)
sys.path.insert(1, "/home/yaoyuchen_google_com/golden_gen/layerwise")

import jax  # pylint: disable=g-import-not-at-top
import ml_dtypes
import numpy as np
import tensorstore as ts

import maxtext

assert os.path.realpath(maxtext.__file__).startswith("/home/yaoyuchen_google_com/wt-layerwise/"), maxtext.__file__

from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import utils as conv_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING

import partial_restore_probe as prp

MODEL = "deepseek4-284b"
NATIVE = "/home/yaoyuchen_google_com/hf_models/deepseek4-284b-bf16"
U = prp.U
S = prp.S
TID_BUCKET = "maxtext-deepseek"
TID_PATH = "deepseek4-284b/2026-09-17/tid2eid.safetensors"
N_LAYERS = 43
FWD_SAMPLE_LAYERS = (0, 1, 2, 3, 4, 42)
S_BLOCKS = (0, 19)
T0 = time.time()


def log(msg):
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
  print(f"[{time.time()-T0:8.1f}s rss_peak={rss:7.1f}GiB] {msg}", flush=True)


# ---------------------------------------------------------------- safetensors
_ST_DT = {
    "BF16": ml_dtypes.bfloat16,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "F64": np.float64,
    "BOOL": np.bool_,
}


def st_header(buf):
  n = int(np.frombuffer(buf[:8], dtype="<u8")[0])
  hdr = json.loads(bytes(buf[8 : 8 + n]))
  hdr.pop("__metadata__", None)
  return hdr, 8 + n


def st_tensor(buf, hdr, base, key):
  h = hdr[key]
  a, b = h["data_offsets"]
  return np.frombuffer(buf[base + a : base + b], dtype=_ST_DT[h["dtype"]]).reshape(h["shape"])


class Native:
  """mmap-backed reader for native sharded safetensors."""

  def __init__(self, root):
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
      self.weight_map = json.load(f)["weight_map"]
    self.root = root
    self._shards = {}

  def _shard(self, name):
    if name not in self._shards:
      buf = np.memmap(os.path.join(self.root, name), dtype=np.uint8, mode="r")
      hdr, base = st_header(buf)
      self._shards[name] = (buf, hdr, base)
    return self._shards[name]

  def dtype(self, key):
    _, hdr, _ = self._shard(self.weight_map[key])
    return hdr[key]["dtype"]

  def get(self, key):
    buf, hdr, base = self._shard(self.weight_map[key])
    return st_tensor(buf, hdr, base, key)


# ---------------------------------------------------------------- compare
def bits(a):
  a = np.ascontiguousarray(np.asarray(a))
  assert a.dtype == ml_dtypes.bfloat16, a.dtype
  return a.view(np.uint16)


def compare(got, ref):
  """Exact bf16 bit-pattern compare. Returns (status, info)."""
  got = np.asarray(got)
  ref = np.asarray(ref)
  info = {
      "got_shape": list(got.shape),
      "ref_shape": list(ref.shape),
      "got_dtype": str(got.dtype),
      "ref_dtype": str(ref.dtype),
  }
  if got.shape != ref.shape:
    return "shape_mismatch", info
  if got.dtype != ml_dtypes.bfloat16:
    rt = got.astype(ml_dtypes.bfloat16)
    info["lossless_cast_to_bf16"] = bool(np.array_equal(rt.astype(np.float64), got.astype(np.float64)))
    got = rt
  if np.array_equal(bits(got), bits(ref)):
    return "equal", info
  d = np.abs(got.astype(np.float32) - ref.astype(np.float32))
  info["max_abs_diff"] = float(np.nanmax(d))
  info["n_diff"] = int(np.count_nonzero(bits(got) != bits(ref)))
  return "UNEQUAL", info


# ---------------------------------------------------------------- setup
def build_config():
  argv = [
      "weight_audit",
      f"{WT_SRC}/maxtext/configs/base.yml",
      f"model_name={MODEL}",
      "scan_layers=False",
      "weight_dtype=bfloat16",
      "skip_jax_distributed_system=True",
      "run_name=weight_audit",
      f"base_output_directory={AUDIT}/unused_out",
      "enable_checkpointing=False",
  ]
  return pyconfig.initialize(argv)


def mt_key(kp):
  # ('params','params','decoder',...) -> 'params-decoder-...'
  return "-".join(kp[1:])


def group_of(key):
  m = re.match(r"^[A-Za-z0-9]+-decoder-layers_(\d+)-", key)
  return int(m.group(1)) if m else "globals"


def flat_keys(k):
  return list(k) if isinstance(k, tuple) else [k]


def flat_hf(v):
  if v is None:
    return []
  if isinstance(v, (list, tuple)):
    out = []
    for x in v:
      out.extend(flat_hf(x))
    return out
  return [v]


def pattern(k):
  return re.sub(r"\.\d+\.", ".N.", re.sub(r"^(layers|mtp)\.\d+\.", r"\1.N.", k))


# ---------------------------------------------------------------- tid2eid
def tid2eid_check(native, summary):
  kv = ts.KvStore.open({"driver": "gcs", "bucket": TID_BUCKET}).result()
  raw = kv.read(TID_PATH).result().value
  buf = np.frombuffer(raw, dtype=np.uint8)
  hdr, base = st_header(buf)
  log(f"tid2eid sideload: {len(raw)} bytes, keys=" f"{[(k, v['dtype'], v['shape']) for k, v in hdr.items()]}")
  res = {"sideload_keys": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in hdr.items()}}
  for i in range(3):
    nk = f"layers.{i}.ffn.gate.tid2eid"
    ref = native.get(nk)
    ref_f = ref.astype(np.float64)
    is_int = bool(np.all(ref_f == np.round(ref_f)))
    ref_i = ref_f.astype(np.int64)
    cands = [k for k in hdr if re.search(rf"(^|\D){i}(\D|$)", k)]
    if len(hdr) == 1:
      cands = list(hdr)
    ent = {
        "native_dtype": native.dtype(nk),
        "native_shape": list(ref.shape),
        "native_is_integer_valued": is_int,
        "native_min": int(ref_i.min()),
        "native_max": int(ref_i.max()),
        "sideload_candidates": cands,
        "matches": {},
    }
    for ck in cands:
      s = st_tensor(buf, hdr, base, ck)
      s_arr = s[i] if (s.ndim == 3 and len(hdr) == 1) else s
      sf = s_arr.astype(np.float64)
      eq = (s_arr.shape == ref.shape) and bool(np.array_equal(sf.astype(np.int64), ref_i))
      ent["matches"][ck] = {
          "shape": list(s_arr.shape),
          "dtype": str(s.dtype),
          "exact_equal_after_cast": eq,
          "sideload_integer_valued": bool(np.all(sf == np.round(sf))),
      }
    log(f"tid2eid layer {i}: {json.dumps(ent)}")
    res[nk] = ent
  summary["tid2eid"] = res


# ---------------------------------------------------------------- config diff
def config_diff(cfg, summary):
  with open(os.path.join(NATIVE, "config.json")) as f:
    hf = json.load(f)
  conv = HF_MODEL_CONFIGS[MODEL].to_dict()
  rows, mism = [], []
  g = lambda c, k: c.get(k)
  for i in range(N_LAYERS):
    cr_u = cfg.compress_ratios[i]
    cr_s = (
        cfg.compress_ratios[i]
        if i < cfg.first_num_hash_layers
        else (128 if (i - cfg.first_num_hash_layers) % 2 == 0 else 4)
    )
    mt = {
        "compress_ratio_unscanned": cr_u,
        "compress_ratio_scanned": cr_s,
        "hash_unscanned": i < cfg.first_num_hash_layers,
        "hash_scanned": i < cfg.first_num_hash_layers,
        "sliding_window": cfg.sliding_window_size,
        "rope_theta": (cfg.compressed_rope_max_timescale if cr_u > 0 else cfg.rope_max_timescale),
        "index_topk": cfg.indexer_topk,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "swiglu_limit": cfg.mlp_activations_limit,
    }
    ref = {}
    for name, c in (("hf", hf), ("conv", conv)):
      cr = g(c, "compress_ratios")[i]
      ref[name] = {
          "compress_ratio": cr,
          "hash": i < g(c, "num_hash_layers"),
          "sliding_window": g(c, "sliding_window"),
          "rope_theta": g(c, "compress_rope_theta") if cr > 0 else g(c, "rope_theta"),
          "index_topk": g(c, "index_topk"),
          "routed_scaling_factor": g(c, "routed_scaling_factor"),
          "num_experts_per_tok": g(c, "num_experts_per_tok"),
          "swiglu_limit": g(c, "swiglu_limit"),
      }
    pairs = [
        ("compress_ratio_unscanned", "compress_ratio"),
        ("compress_ratio_scanned", "compress_ratio"),
        ("hash_unscanned", "hash"),
        ("hash_scanned", "hash"),
        ("sliding_window", "sliding_window"),
        ("rope_theta", "rope_theta"),
        ("index_topk", "index_topk"),
        ("routed_scaling_factor", "routed_scaling_factor"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("swiglu_limit", "swiglu_limit"),
    ]
    for a, b in pairs:
      for name in ("hf", "conv"):
        if mt[a] is None or ref[name][b] is None or float(mt[a]) != float(ref[name][b]):
          mism.append({"layer": i, "field": a, "maxtext": mt[a], f"{name}_config": ref[name][b]})
    rows.append({"layer": i, "maxtext": mt, "hf_config_json": ref["hf"], "converter_hf_config": ref["conv"]})
  print(
      "layer | cr_U cr_S hf_cr conv_cr | hash_U hf_hash | win hf_win | " "theta hf_theta | topk | rsf | k | swiglu(mt/hf)"
  )
  for r in rows:
    m, h, c = r["maxtext"], r["hf_config_json"], r["converter_hf_config"]
    print(
        f"{r['layer']:5d} | {m['compress_ratio_unscanned']:4} "
        f"{m['compress_ratio_scanned']:4} {h['compress_ratio']:5} "
        f"{c['compress_ratio']:7} | {str(m['hash_unscanned']):6} "
        f"{str(h['hash']):7} | {m['sliding_window']:3} {h['sliding_window']:6} | "
        f"{m['rope_theta']:6} {h['rope_theta']:8} | {m['index_topk']}/"
        f"{h['index_topk']} | {m['routed_scaling_factor']}/"
        f"{h['routed_scaling_factor']} | {m['num_experts_per_tok']}/"
        f"{h['num_experts_per_tok']} | {m['swiglu_limit']}/{h['swiglu_limit']}"
    )
  rs = hf.get("rope_scaling", {})
  rope = {
      "rope_type": (getattr(cfg.rope_type, "value", cfg.rope_type), rs.get("type")),
      "factor": (cfg.rope_factor, rs.get("factor")),
      "beta_fast": (cfg.beta_fast, rs.get("beta_fast")),
      "beta_slow": (cfg.beta_slow, rs.get("beta_slow")),
      "original_max_position_embeddings": (
          cfg.original_max_position_embeddings,
          rs.get("original_max_position_embeddings"),
      ),
      "max_position_embeddings": (cfg.max_position_embeddings, hf.get("max_position_embeddings")),
      "len_compress_ratios": (len(cfg.compress_ratios), len(hf["compress_ratios"])),
  }
  for k, (a, b) in rope.items():
    ok = str(a) == str(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b))
    print(f"  rope/global {k}: maxtext={a} hf={b} {'OK' if ok else 'MISMATCH'}")
    if not ok:
      mism.append({"layer": "global", "field": k, "maxtext": a, "hf_config": b})
  print(f"CONFIG_DIFF mismatches={len(mism)}")
  for m in mism:
    print(f"  MISMATCH {m}")
  summary["config_diff"] = {"rows": rows, "rope": {k: list(map(str, v)) for k, v in rope.items()}, "mismatches": mism}


# ---------------------------------------------------------------- main audit
def main(mode):
  summary = {"mode": mode, "U": U, "S": S, "native": NATIVE}
  cfg = build_config()
  hf_cfg = HF_MODEL_CONFIGS[MODEL].to_dict()
  native = Native(NATIVE)
  native_keys = set(native.weight_map)
  log(f"native keys={len(native_keys)}")

  # Production mappings, exactly as to_huggingface / to_maxtext build them.
  maps = to_huggingface._get_model_mappings(MODEL, False, hf_cfg, cfg)
  pm_hf, hooks_hf, shape_map = (maps["param_mapping"], maps["hook_fn_mapping"], maps["shape_mapping"])
  pm_mt = PARAM_MAPPING[MODEL](hf_cfg, cfg, False)
  hooks_mt = HOOK_FNS[MODEL](hf_cfg, cfg, False, saving_to_hf=False)

  meta = prp.flat(prp.meta_tree(U))
  u_leaves = {mt_key(kp): (kp, m) for kp, m in meta.items() if kp[0] == "params"}
  non_params = [kp for kp in meta if kp[0] != "params"]
  log(f"U leaves: params={len(u_leaves)} non_params={non_params}")
  ckpt_keys = set(u_leaves)

  # ---- key-set accounting
  flat_map = set()
  for k in pm_hf:
    flat_map.update(flat_keys(k))
  mt_unmapped = sorted(ckpt_keys - flat_map)
  map_absent = sorted(flat_map - ckpt_keys)
  filtered = [k for k in pm_hf if all(x in ckpt_keys for x in flat_keys(k))]
  none_mapped = sorted(k for k in filtered if pm_hf[k] is None)
  hf_all = set()
  for v in pm_mt.values():
    hf_all.update(flat_hf(v))
  hf_present = set()
  for k in filtered:
    hf_present.update(flat_hf(pm_hf[k]))
  # Collection-alias: production maps bias/tid2eid under MoEBiasVar/Tid2EidVar.
  alias = {}
  for k in map_absent:
    a = re.sub(r"^(MoEBiasVar|Tid2EidVar)-", "params-", k)
    if a != k and a in ckpt_keys:
      alias[a] = k
  hf_alias = set()
  for a, k in alias.items():
    hf_alias.update(flat_hf(pm_mt[k]))
  n_unc = native_keys - hf_present
  unc_groups = collections.Counter(pattern(k) for k in n_unc)
  unc_after_alias = native_keys - hf_present - hf_alias
  acct = {
      "native_total": len(native_keys),
      "mapping_hf_keys_total": len(hf_all),
      "mapping_hf_keys_not_in_native": sorted(hf_all - native_keys),
      "N_mapped_present": len(hf_present & native_keys),
      "native_not_covered_count": len(n_unc),
      "native_not_covered_by_pattern": dict(sorted(unc_groups.items())),
      "native_not_covered_after_collection_alias": len(unc_after_alias),
      "native_not_covered_after_alias_by_pattern": dict(
          sorted(collections.Counter(pattern(k) for k in unc_after_alias).items())
      ),
      "maxtext_leaves": len(ckpt_keys),
      "maxtext_leaves_not_in_mapping": mt_unmapped,
      "maxtext_leaves_mapped_to_None": none_mapped,
      "mapping_keys_absent_from_ckpt_count": len(map_absent),
      "mapping_keys_absent_from_ckpt_by_pattern": dict(
          sorted(collections.Counter(re.sub(r"layers_\d+", "layers_N", k) for k in map_absent).items())
      ),
      "collection_alias_pairs": len(alias),
      "filtered_map_entries": len(filtered),
  }
  summary["accounting"] = acct
  for k, v in acct.items():
    if isinstance(v, list) and len(v) > 12:
      log(f"ACCT {k}: count={len(v)} first={v[:6]} ... last={v[-3:]}")
    else:
      log(f"ACCT {k}: {v}")

  config_diff(cfg, summary)
  tid2eid_check(native, summary)
  if mode == "dry":
    return finish(summary)

  # ---- per-layer streaming compare
  native_status = {}  # native key -> status record
  mt_status = {}  # maxtext key -> status
  fwd_status = {}
  keep_for_s = {}  # (layer, rest) -> numpy, for scanned check
  keep_layers = {3 + 2 * b + j for b in S_BLOCKS for j in range(2)}
  groups = ["globals"] + list(range(N_LAYERS if mode == "full" else 1))
  for grp in groups:
    tg = time.time()
    sel = {kp: m for k, (kp, m) in u_leaves.items() if group_of(k) == grp}
    arrs = prp.orbax_restore(U, sel)
    state = {mt_key(kp): a for kp, a in arrs.items()}
    del arrs
    t_rest = time.time() - tg
    n_eq = n_bad = 0
    work = [k for k in filtered if group_of(flat_keys(k)[0]) == grp]
    work += [a for a in alias if group_of(a) == grp]
    for k in work:
      if k in alias:
        pm_local = {k: pm_hf[alias[k]]}
        out = conv_utils.process_maxtext_param(k, state[k], pm_local, hooks_hf, shape_map, cfg)
        tag = "via_collection_alias"
      else:
        if pm_hf[k] is None:
          for x in flat_keys(k):
            mt_status[x] = "mapped_to_None"
          continue
        w = [state[x] for x in k] if isinstance(k, tuple) else state[k]
        out = conv_utils.process_maxtext_param(k, w, pm_hf, hooks_hf, shape_map, cfg)
        tag = "direct"
      key_ok = True
      for hf_path, got in out:
        if hf_path not in native_keys:
          native_status[hf_path] = {"status": "missing_in_native", "mt": str(k)}
          key_ok = False
          continue
        st, info = compare(got, native.get(hf_path))
        info.update({"status": st, "mt": str(k), "path": tag})
        native_status[hf_path] = info
        if st == "equal":
          n_eq += 1
        else:
          n_bad += 1
          key_ok = False
          log(f"  NOT EQUAL {hf_path} <- {k}: {info}")
      for x in flat_keys(k):
        mt_status[x] = ("equal" if key_ok else "NOT_EQUAL") + f"({tag})"
    for k in state:
      mt_status.setdefault(k, "not_compared")

    # mapped-to-None leaves: record their values (production hook: ones).
    for k in none_mapped:
      if group_of(k) == grp:
        a = np.asarray(state[k]).astype(np.float32)
        mt_status[k] = f"mapped_to_None(all_ones={bool(np.all(a == 1.0))})"

    # Forward direction (native -> MaxText) on sampled groups.
    n_fwd_eq = n_fwd_bad = 0
    if grp == "globals" or grp in FWD_SAMPLE_LAYERS:
      for k, src in pm_mt.items():
        if isinstance(k, tuple) or k not in state or group_of(k) != grp:
          continue
        shape = tuple(state[k].shape)
        load_fn = to_maxtext._get_hf_loading_function(src, native.get, hooks_mt.get(k), shape, cfg, k)
        got = load_fn()
        st, info = compare(got, np.asarray(state[k]))
        h = hooks_mt.get(k)
        info["hook"] = getattr(h, "__qualname__", str(h)) if h else "identity"
        info["status"] = st
        fwd_status[k] = info
        if st == "equal":
          n_fwd_eq += 1
        else:
          n_fwd_bad += 1
          log(f"  FWD NOT EQUAL {k}: {info}")
      for a, k in alias.items():
        if group_of(a) == grp:
          load_fn = to_maxtext._get_hf_loading_function(
              pm_mt[k], native.get, hooks_mt.get(k), tuple(state[a].shape), cfg, k
          )
          st, info = compare(load_fn(), np.asarray(state[a]))
          info.update({"status": st, "hook": "identity", "via_alias_of": k})
          fwd_status[a] = info
          n_fwd_eq += st == "equal"
          n_fwd_bad += st != "equal"

    if grp in keep_layers:
      pre = f"params-decoder-layers_{grp}-"
      for k, a in state.items():
        keep_for_s[(grp, k[len(pre) :])] = np.asarray(a)
    del state
    log(
        f"group={grp} leaves={len(sel)} restore={t_rest:.1f}s native_eq={n_eq} "
        f"native_bad={n_bad} fwd_eq={n_fwd_eq} fwd_bad={n_fwd_bad} "
        f"wall={time.time()-tg:.1f}s"
    )

  summary["native_status"] = native_status
  summary["maxtext_status"] = mt_status
  summary["forward_status"] = fwd_status
  if mode == "full":
    scanned_check(keep_for_s, summary)
  return finish(summary)


def scanned_check(keep, summary):
  meta = prp.flat(prp.meta_tree(S))
  fam = {kp: m for kp, m in meta.items() if "scanned_blocks" in kp}
  log(f"S scanned leaf families={len(fam)}")
  ctx = ts.Context({"cache_pool": {"total_bytes_limit": 0}, "file_io_concurrency": {"limit": 128}})

  async def one(kp):
    t = await ts.open(prp.ts_spec(S, kp), open=True, read=True, context=ctx)
    assert t.shape[1] == 20, (kp, t.shape)
    return kp, await t[:, list(S_BLOCKS)].read()

  async def run():
    return dict(await asyncio.gather(*[one(kp) for kp in fam]))

  t0 = time.time()
  got = asyncio.run(run())
  log(f"S slice read {time.time()-t0:.1f}s")
  res, n_eq, n_bad = {}, 0, 0
  for kp, a in got.items():
    j = int(kp[4].split("_")[1])
    rest = "-".join(kp[5:])
    for bi, b in enumerate(S_BLOCKS):
      layer = 3 + 2 * b + j
      u = keep.get((layer, rest))
      if u is None:
        st = "missing_in_U"
      else:
        st = "equal" if (a[:, bi].shape == u.shape and np.array_equal(bits(a[:, bi]), bits(u))) else "UNEQUAL"
      res[f"{'/'.join(kp[3:])}[:, {b}] vs U layers_{layer}"] = st
      n_eq += st == "equal"
      n_bad += st != "equal"
      if st != "equal":
        log(f"  S {kp} block {b}: {st}")
  log(f"SCANNED_CHECK families={len(fam)} blocks={S_BLOCKS} equal={n_eq} " f"not_equal={n_bad}")
  summary["scanned_check"] = {
      "families": len(fam),
      "blocks": list(S_BLOCKS),
      "equal": n_eq,
      "not_equal": n_bad,
      "per_key": res,
  }


def finish(summary):
  ns = summary.get("native_status", {})
  cnt = collections.Counter(v["status"] for v in ns.values())
  ms = collections.Counter(re.sub(r"\(.*", "", v) for v in summary.get("maxtext_status", {}).values())
  fs = collections.Counter(v["status"] for v in summary.get("forward_status", {}).values())
  fh = collections.Counter((v["hook"], v["status"]) for v in summary.get("forward_status", {}).values())
  summary["totals"] = {
      "native_compared_status": dict(cnt),
      "maxtext_status": dict(ms),
      "forward_status": dict(fs),
      "forward_by_hook": {f"{h}|{s}": n for (h, s), n in fh.items()},
      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
      "runtime_s": time.time() - T0,
  }
  a = summary["accounting"]
  lines = [
      f"FINAL native_total={a['native_total']} N_mapped_present="
      f"{a['N_mapped_present']} native_not_covered={a['native_not_covered_count']}"
      f" native_not_covered_after_alias="
      f"{a['native_not_covered_after_collection_alias']}",
      f"FINAL not_covered_by_pattern={a['native_not_covered_by_pattern']}",
      f"FINAL maxtext_leaves={a['maxtext_leaves']} not_in_mapping="
      f"{len(a['maxtext_leaves_not_in_mapping'])} mapped_to_None="
      f"{len(a['maxtext_leaves_mapped_to_None'])} alias_pairs="
      f"{a['collection_alias_pairs']}",
      f"FINAL native_compare={dict(cnt)}",
      f"FINAL maxtext_status={dict(ms)}",
      f"FINAL forward={dict(fs)} by_hook={summary['totals']['forward_by_hook']}",
      f"FINAL tid2eid="
      + json.dumps(
          {
              k: {c: m["exact_equal_after_cast"] for c, m in v["matches"].items()}
              for k, v in summary["tid2eid"].items()
              if k.startswith("layers")
          }
      ),
      f"FINAL config_mismatches={len(summary['config_diff']['mismatches'])}",
      f"FINAL scanned_check=" + json.dumps({k: v for k, v in summary.get("scanned_check", {}).items() if k != "per_key"}),
      f"FINAL runtime_s={summary['totals']['runtime_s']:.1f} peak_rss_gib=" f"{summary['totals']['peak_rss_gib']:.1f}",
  ]
  summary["final_lines"] = lines
  os.makedirs(f"{AUDIT}/logs", exist_ok=True)
  suffix = "" if summary["mode"] == "full" else f"_{summary['mode']}"
  out = f"{AUDIT}/logs/weight_audit_summary{suffix}.json"
  data = json.dumps(summary, indent=1, sort_keys=True, default=str).encode()
  with open(out, "wb") as f:
    f.write(data)
  sha = hashlib.sha256(data).hexdigest()
  with open(out + ".sha256", "w") as f:
    f.write(f"{sha}  {os.path.basename(out)}\n")
  for l in lines:
    print(l, flush=True)
  print(f"FINAL summary={out} sha256={sha}", flush=True)


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "full")
