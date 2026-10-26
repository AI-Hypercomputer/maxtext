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

"""Model-agnostic layerwise activation bisection and optional FP64 single-layer weight check."""

import argparse
import os
import sys
from typing import Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bisect_utils  # pylint: disable=wrong-import-position


def discover_layers(hf_npz: Any, mt_npz: Any, out_template: str, max_scan: int = 256) -> list[int]:
  """Discovers layer indices present in both NPZ archives for the given key template."""
  layers = []
  for idx in range(max_scan):
    key = out_template.format(layer=idx)
    if key in hf_npz and key in mt_npz:
      layers.append(idx)
  return layers


def bisect_activations(  # pylint: disable=too-many-positional-arguments
    hf_npz: Any,
    mt_npz: Any,
    layers: list[int],
    in_template: str = "L{layer:02d}_in",
    out_template: str = "L{layer:02d}_out",
    subkey_templates: list[str] | None = None,
    rtol: float = 1e-2,
) -> tuple[list[dict[str, Any]], int | None]:
  """Compares per-layer and optional sub-layer activations without requiring model weights."""
  rows: list[dict[str, Any]] = []
  first_bad_layer: int | None = None

  for layer in layers:
    in_key = in_template.format(layer=layer)
    out_key = out_template.format(layer=layer)
    if out_key not in hf_npz or out_key not in mt_npz:
      continue

    in_metrics = (
        bisect_utils.tensor_diff_metrics(hf_npz[in_key], mt_npz[in_key])
        if (in_key in hf_npz and in_key in mt_npz)
        else {"max_abs": 0.0, "rel_l2": 0.0, "cos_sim": 1.0, "worst_pos": 0}
    )
    out_metrics = bisect_utils.tensor_diff_metrics(hf_npz[out_key], mt_npz[out_key])
    amp = float(out_metrics["rel_l2"]) / max(float(in_metrics["rel_l2"]), 1e-12)

    sub_results: dict[str, dict[str, float | int]] = {}
    for tmpl in subkey_templates or []:
      skey = tmpl.format(layer=layer)
      if skey in hf_npz and skey in mt_npz:
        sub_results[skey] = bisect_utils.tensor_diff_metrics(hf_npz[skey], mt_npz[skey])

    out_rel_l2 = float(out_metrics["rel_l2"])
    is_bad = (not np.isfinite(out_rel_l2)) or (out_rel_l2 > rtol)
    if is_bad and first_bad_layer is None:
      first_bad_layer = layer

    rows.append(
        {
            "layer": layer,
            "in_rel_l2": float(in_metrics["rel_l2"]),
            "out_rel_l2": float(out_metrics["rel_l2"]),
            "out_max_abs": float(out_metrics["max_abs"]),
            "cos_sim": float(out_metrics["cos_sim"]),
            "worst_pos": int(out_metrics["worst_pos"]),
            "amplification": amp,
            "subkeys": sub_results,
            "status": "DIVERGED" if is_bad else "OK",
        }
    )
  return rows, first_bad_layer


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hf-acts", required=True, help="NPZ with HuggingFace reference activations")
  parser.add_argument("--mt-acts", required=True, help="NPZ with MaxText activations")
  parser.add_argument(
      "--layers",
      default=None,
      help="Comma-separated layer indices (default: auto-discover)",
  )
  parser.add_argument(
      "--in-key-template",
      default="L{layer:02d}_in",
      help="Input activation key template",
  )
  parser.add_argument(
      "--out-key-template",
      default="L{layer:02d}_out",
      help="Output activation key template",
  )
  parser.add_argument(
      "--subkey-templates",
      default="",
      help="Optional comma-separated sub-layer templates",
  )
  parser.add_argument(
      "--rtol",
      type=float,
      default=1e-2,
      help="Relative L2 threshold for flagging divergence",
  )
  parser.add_argument(
      "--ckpt-uri",
      default=None,
      help="Optional Orbax Zarr3/OCDbt items URI for FP64 weight check",
  )
  parser.add_argument(
      "--adapter",
      default="default",
      help="Adapter 'module:Class' for optional FP64 weight check",
  )
  parser.add_argument(
      "--scanned",
      action="store_true",
      help="Whether checkpoint layers are stacked (scan_layers=True)",
  )
  parser.add_argument(
      "--scan-axis",
      type=int,
      default=1,
      help="Layer scan axis when --scanned is set (default: 1, matching MaxText param_scan_axis)",
  )
  parser.add_argument(
      "--eps",
      type=float,
      default=1e-6,
      help="Normalization epsilon for optional adapter check",
  )
  args = parser.parse_args()

  hf_npz = np.load(args.hf_acts)
  mt_npz = np.load(args.mt_acts)
  layers = (
      [int(x.strip()) for x in args.layers.split(",") if x.strip()]
      if args.layers
      else discover_layers(hf_npz, mt_npz, args.out_key_template)
  )
  sub_templates = [s.strip() for s in args.subkey_templates.split(",") if s.strip()]

  rows, first_bad = bisect_activations(
      hf_npz,
      mt_npz,
      layers,
      args.in_key_template,
      args.out_key_template,
      sub_templates,
      rtol=args.rtol,
  )

  print(
      f"{'Layer':<6} | {'In RelL2':<12} | {'Out RelL2':<12} | {'MaxAbs':<12} | {'CosSim':<10} | {'WorstPos':<8} | Status"
  )
  print("-" * 82)
  for r in rows:
    print(
        f"{r['layer']:<6} | {r['in_rel_l2']:<12.4e} | {r['out_rel_l2']:<12.4e} | "
        f"{r['out_max_abs']:<12.4e} | {r['cos_sim']:<10.6f} | {r['worst_pos']:<8} | {r['status']}"
    )
    for skey, sm in r["subkeys"].items():
      print(f"  -> {skey}: rel_l2={sm['rel_l2']:.4e}, max_abs={sm['max_abs']:.4e}, worst_pos={sm['worst_pos']}")

  if first_bad is not None:
    print(f"\nFirst layer exceeding rtol={args.rtol:.1e}: layer {first_bad}")

  if args.ckpt_uri:
    ckpt_items = args.ckpt_uri.rstrip("/") + "/"
    adapter = bisect_utils.resolve_adapter(args.adapter)
    target_layers = [first_bad] if (first_bad is not None and not args.layers) else layers
    print(f"\nRunning FP64 single-layer weight check ({adapter.__class__.__name__}, scanned={args.scanned}):")
    for layer in target_layers:
      in_k = args.in_key_template.format(layer=layer)
      out_k = args.out_key_template.format(layer=layer)
      weights = adapter.load_layer_weights(ckpt_items, layer, scanned=args.scanned, scan_axis=args.scan_axis)
      iso_out = adapter.forward_layer_fp64(hf_npz[in_k], weights, eps=args.eps)
      iso_m = bisect_utils.tensor_diff_metrics(hf_npz[out_k], iso_out)
      print(f"  Layer {layer}: isolated_rel_l2={iso_m['rel_l2']:.4e}, max_abs={iso_m['max_abs']:.4e}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
