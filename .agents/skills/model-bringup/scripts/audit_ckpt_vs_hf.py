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

"""Audits routed-expert weights in an unscanned or scanned MaxText OCDbt checkpoint against HF source weights.

Checks a [:, :256, :256] (unscanned) or [:, :, :256, :256] (scanned) slice
across all experts and layers/blocks.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time

import numpy as np

DEFAULT_WEIGHT_MAP = "wi_0=gate_proj,wi_1=up_proj,wo=down_proj"
R = C = 256


def _parse_layers(spec: str) -> list[int]:
  out = []
  for part in spec.split(","):
    if "-" in part:
      lo, hi = part.split("-", 1)
      out.extend(range(int(lo), int(hi) + 1))
    else:
      out.append(int(part))
  return sorted(set(out))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


async def _read_block(base_uri: str, key: str, scanned: bool) -> np.ndarray:
  import tensorstore as ts  # pylint: disable=import-outside-toplevel

  spec = {
      "driver": "zarr3",
      "kvstore": {"driver": "ocdbt", "base": base_uri, "path": key},
  }
  arr = await ts.open(spec, open=True, read=True)
  sl = arr[:, :, :R, :C] if scanned else arr[:, :R, :C]
  return np.asarray(await sl.read(), np.float32)


async def main() -> None:
  parser = argparse.ArgumentParser(description="Audit OCDbt checkpoint expert weights against HF safetensors.")
  parser.add_argument("--ckpt-uri", required=True, help="GCS URI ending in /0/items/")
  parser.add_argument(
      "--hf-dir",
      required=True,
      help="Local directory containing model.safetensors.index.json",
  )
  parser.add_argument("--kind", choices=["unscanned", "scanned"], required=True)
  parser.add_argument(
      "--layers",
      default="0-60",
      help="Layer range for unscanned, e.g. 0-60 or 40,41,42",
  )
  parser.add_argument(
      "--num-prefix-layers",
      type=int,
      default=0,
      help="Unscanned prefix layers in scanned ckpt",
  )
  parser.add_argument(
      "--scan-group-size",
      type=int,
      default=1,
      help="Layers per scan block in scanned ckpt",
  )
  parser.add_argument(
      "--hf-key-template",
      default="model.layers.{layer}.mlp.experts.{expert}.{w_short}",
      help="HF safetensors key prefix template before .weight / .scale",
  )
  parser.add_argument(
      "--weight-map",
      default=DEFAULT_WEIGHT_MAP,
      help="Comma-separated mt_name=hf_name expert projection mapping",
  )
  parser.add_argument(
      "--dequant-script-dir",
      default="",
      help="Optional directory containing a custom dequantization module",
  )
  parser.add_argument(
      "--dequant-fn",
      default="",
      help="Optional module:function for custom block dequantization",
  )
  parser.add_argument(
      "--tol",
      type=float,
      default=1e-2,
      help="Relative L2 threshold for flagging a corrupt expert",
  )
  args = parser.parse_args()
  try:
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError("safetensors and torch are required to audit HF expert weights") from exc

  hfw = dict(item.split("=", 1) for item in args.weight_map.split(",") if item)
  if args.dequant_script_dir:
    sys.path.insert(0, args.dequant_script_dir)
  dequant_fn = None
  if args.dequant_fn:
    mod_name, fn_name = args.dequant_fn.split(":", 1)
    dequant_fn = getattr(importlib.import_module(mod_name), fn_name)

  with open(f"{args.hf_dir}/model.safetensors.index.json", encoding="utf-8") as idx_f:
    wm = json.load(idx_f)["weight_map"]
  handles = {}

  def get_hf_ref(layer: int, expert: int, w_short: str) -> np.ndarray:
    k = args.hf_key_template.format(layer=layer, expert=expert, w_short=w_short)
    w_key = k + ".weight"
    fn = wm[w_key]
    if fn not in handles:
      handles[fn] = safe_open(f"{args.hf_dir}/{fn}", framework="pt", device="cpu")
    if k + ".scale" in wm and dequant_fn is not None:
      s_fn = wm[k + ".scale"]
      if s_fn not in handles:
        handles[s_fn] = safe_open(f"{args.hf_dir}/{s_fn}", framework="pt", device="cpu")
      q = handles[fn].get_slice(w_key)[slice(0, C), slice(0, R // 2)]
      s = handles[s_fn].get_slice(k + ".scale")[slice(0, C), slice(0, R // 32)]
      return dequant_fn(q, s).float().numpy().T
    t = handles[fn].get_slice(w_key)[slice(0, C), slice(0, R)]
    return t.to(torch.float32).numpy().T

  def audit_layer(layer: int, ck_dict: dict[str, np.ndarray]) -> list[int]:
    first_key = next(iter(hfw))
    bad = set()
    for e in range(ck_dict[first_key].shape[0]):
      for n, w in hfw.items():
        if _rel_l2(ck_dict[n][e], get_hf_ref(layer, e, w)) > args.tol:
          bad.add(e)
    return sorted(bad)

  total_bad_layers = 0
  if args.kind == "unscanned":
    for layer in _parse_layers(args.layers):
      t0 = time.time()
      prefix = f"params.params.decoder.layers_{layer}.mlp.MoeBlock_0."
      ck = dict(
          zip(
              hfw,
              await asyncio.gather(*[_read_block(args.ckpt_uri, prefix + n, False) for n in hfw]),
          )
      )
      bad = audit_layer(layer, ck)
      total_bad_layers += bool(bad)
      print(
          f"AUDIT unscanned L{layer:02d} n_bad={len(bad)} ids={bad[:32]}" f" ({time.time()-t0:.1f}s)",
          flush=True,
      )
  else:
    for layer in range(args.num_prefix_layers):
      t0 = time.time()
      prefix = f"params.params.decoder.layers_{layer}.mlp.MoeBlock_0."
      ck = dict(
          zip(
              hfw,
              await asyncio.gather(*[_read_block(args.ckpt_uri, prefix + n, False) for n in hfw]),
          )
      )
      bad = audit_layer(layer, ck)
      total_bad_layers += bool(bad)
      print(
          f"AUDIT scanned-prefix L{layer:02d} n_bad={len(bad)} ids={bad[:32]}" f" ({time.time()-t0:.1f}s)",
          flush=True,
      )
    first_key = next(iter(hfw))
    for j in range(args.scan_group_size):
      t0 = time.time()
      prefix = f"params.params.decoder.scanned_blocks.layers_{j}.mlp.MoeBlock_0."
      full = dict(
          zip(
              hfw,
              await asyncio.gather(*[_read_block(args.ckpt_uri, prefix + n, True) for n in hfw]),
          )
      )
      print(f"AUDIT scanned layers_{j} read in {time.time()-t0:.1f}s", flush=True)
      for bi in range(full[first_key].shape[1]):
        layer = args.num_prefix_layers + args.scan_group_size * bi + j
        bad = audit_layer(layer, {n: full[n][:, bi] for n in hfw})
        total_bad_layers += bool(bad)
        print(
            f"AUDIT scanned L{layer:02d} (j={j} b={bi}) n_bad={len(bad)}" f" ids={bad[:32]}",
            flush=True,
        )

  verdict = "PASS" if total_bad_layers == 0 else "FAIL"
  print(f"VALUE_AUDIT_VERDICT={verdict} bad_layers={total_bad_layers}", flush=True)
  sys.exit(0 if total_bad_layers == 0 else 1)


if __name__ == "__main__":
  asyncio.run(main())
