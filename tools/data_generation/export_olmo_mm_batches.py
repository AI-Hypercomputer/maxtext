#!/usr/bin/env python3
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

"""Export deterministic global multimodal batches from an ``OLMo-core`` config to ``.npz``.

Produces ``batch_00000.npz``, ``batch_00001.npz``, ... (plus ``manifest.json``) readable by
MaxText's ``dataset_type=olmo_mm`` input pipeline via ``olmo_mm_batches_dir=<output_dir>``.
Supports both local paths and ``gs://`` URIs via ``etils.epath``.

Example usage:

    python tools/data_generation/export_olmo_mm_batches.py \\
        --olmo_config_path /path/to/joint_8k_config.json \\
        --output_dir gs://my-bucket/olmo35/batches/joint_8k \\
        --num_batches 16 \\
        --path_remap_from /weka \\
        --path_remap_to /mnt/weka \\
        --work_dir /tmp/olmo_mm_work
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional, Sequence

from etils import epath
import numpy as np

# Allow running directly from repo root with src/ on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))

from maxtext.input_pipeline import olmo_mm_data_processing as mm_dp  # noqa: E402

log = logging.getLogger(__name__)


def _parse_bool(val: str | bool) -> bool:
  if isinstance(val, bool):
    return val
  low = val.strip().lower()
  if low in ("1", "true", "yes", "y"):
    return True
  if low in ("0", "false", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean string, got {val!r}")


def export_batches(
    *,
    olmo_config_path: str,
    output_dir: str,
    num_batches: int,
    work_dir: str = "/tmp/olmo_mm_export_work",
    global_batch_tokens: Optional[int] = None,
    pack_buffer_size: Optional[int] = None,
    max_target_length: Optional[int] = None,
    max_crops_per_seq: Optional[int] = None,
    max_pooled_patches_per_seq: Optional[int] = None,
    path_remap_from: str = "",
    path_remap_to: str = "",
    num_workers: int = 0,
    seed_offset: int = 0,
    start_batch_index: int = 0,
    overwrite: bool = True,
) -> dict[str, Any]:
  """Iterate an ``OLMo-core`` ``MixtureDataLoader`` and write MaxText ``.npz`` batch archives."""
  if num_batches <= 0:
    raise ValueError(f"num_batches must be positive, got {num_batches}")

  loader, raw_cfg = mm_dp.build_olmo_core_mixture_loader(
      olmo_config_path,
      work_dir=work_dir,
      shard_index=0,
      shard_count=1,
      path_remap_from=path_remap_from,
      path_remap_to=path_remap_to,
      global_batch_tokens=global_batch_tokens,
      pack_buffer_size=pack_buffer_size,
      num_workers=num_workers,
      batch_prefetch_depth=0,
      seed_offset=seed_offset,
  )

  dl_cfg = raw_cfg.get("data_loader", {})
  model_cfg = raw_cfg.get("model", {})
  vision_cfg = model_cfg.get("vision", {})
  tm_cfg = raw_cfg.get("train_module", {})

  seq_len = int(max_target_length or dl_cfg["sequence_length"])
  crops_cap = int(max_crops_per_seq or dl_cfg.get("pack_max_crops") or 64)
  pooled_cap = int(max_pooled_patches_per_seq or (crops_cap * 196))

  img_sz = vision_cfg.get("image_default_input_size", [378, 378])
  img_h, img_w = (int(img_sz[0]), int(img_sz[1])) if isinstance(img_sz, (list, tuple)) else (int(img_sz), int(img_sz))
  patch_sz = int(vision_cfg.get("image_patch_size", 14))
  n_patches = (img_h // patch_sz) * (img_w // patch_sz)
  patch_dim = 3 * patch_sz * patch_sz

  tm_pad_id = tm_cfg.get("padding_input_token_id")
  padding_input_token_id = int(tm_pad_id) if tm_pad_id is not None else 100257

  out_root = epath.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)

  it = iter(loader)
  batch_summaries: list[dict[str, Any]] = []
  t0 = time.perf_counter()

  for step_idx in range(int(start_batch_index), int(start_batch_index) + int(num_batches)):
    out_file = out_root / f"batch_{step_idx:05d}.npz"
    try:
      raw_batch = next(it)
    except StopIteration:
      loader.reshuffle()
      it = iter(loader)
      raw_batch = next(it)

    if not overwrite and out_file.exists():
      log.info("Skipping existing %s (overwrite=False)", out_file)
      continue

    mt_batch = mm_dp.olmo_core_batch_to_maxtext_batch(
        raw_batch,
        max_target_length=seq_len,
        max_crops_per_seq=crops_cap,
        max_pooled_patches_per_seq=pooled_cap,
        n_patches_per_crop=n_patches,
        patch_dim=patch_dim,
        pool_size=4,
        dtype_mm="bfloat16",
        reset_positions_on_segment=True,
        padding_input_token_id=padding_input_token_id,
        use_loss_weights=True,
        include_router_token_mask=True,
    )

    lw = mt_batch["loss_weights"]
    rtm = mt_batch["router_token_mask"]
    n_im_patch = int(np.sum(mt_batch["inputs"] == 100280))
    n_valid = int(np.sum(rtm > 0))
    n_supervised = int(np.sum(lw > 0))
    meta = {
        "batch_index": step_idx,
        "sequence_length": seq_len,
        "batch_size_seqs": int(mt_batch["inputs"].shape[0]),
        "valid_tokens": n_valid,
        "supervised_tokens": n_supervised,
        "im_patch_tokens": n_im_patch,
        "loss_weight_sum": float(np.sum(lw)),
    }
    mm_dp.save_maxtext_batch_npz(out_file, mt_batch, metadata=meta)
    batch_summaries.append(meta)
    log.info(
        "Wrote %s: shape=%s valid_tok=%d supervised_tok=%d im_patch=%d (%.1fs)",
        out_file,
        tuple(mt_batch["inputs"].shape),
        n_valid,
        n_supervised,
        n_im_patch,
        time.perf_counter() - t0,
    )

  manifest = {
      "olmo_config_path": str(olmo_config_path),
      "output_dir": str(output_dir),
      "sequence_length": seq_len,
      "max_crops_per_seq": crops_cap,
      "max_pooled_patches_per_seq": pooled_cap,
      "padding_input_token_id": padding_input_token_id,
      "num_exported_batches": len(batch_summaries),
      "batches": batch_summaries,
  }
  (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
  return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
  parser = argparse.ArgumentParser(
      description="Export deterministic OLMo-core multimodal mixture batches to MaxText .npz files."
  )
  parser.add_argument("--olmo_config_path", required=True, help="Path to OLMo-core experiment config.json.")
  parser.add_argument("--output_dir", required=True, help="Output directory (local or gs://) for batch_XXXXX.npz.")
  parser.add_argument("--num_batches", type=int, required=True, help="Number of global batches to export.")
  parser.add_argument("--work_dir", default="/tmp/olmo_mm_export_work", help="Scratch work_dir for OLMo-core loader.")
  parser.add_argument(
      "--global_batch_tokens",
      type=int,
      default=None,
      help="Optional override for data_loader.global_batch_size (in tokens).",
  )
  parser.add_argument(
      "--pack_buffer_size",
      type=int,
      default=None,
      help="Optional override for data_loader.pack_buffer_size (in packed sequences; useful for 256K export).",
  )
  parser.add_argument(
      "--max_target_length",
      type=int,
      default=None,
      help="Optional override for sequence length (defaults to data_loader.sequence_length).",
  )
  parser.add_argument(
      "--max_crops_per_seq",
      type=int,
      default=None,
      help="Vision crop padding ceiling per sequence (defaults to data_loader.pack_max_crops).",
  )
  parser.add_argument(
      "--max_pooled_patches_per_seq",
      type=int,
      default=None,
      help="Pooled patch padding ceiling per sequence (defaults to max_crops_per_seq * 196).",
  )
  parser.add_argument("--path_remap_from", default="", help="Source path prefix to rewrite (e.g. /weka).")
  parser.add_argument("--path_remap_to", default="", help="Destination path prefix (e.g. /mnt/weka).")
  parser.add_argument("--num_workers", type=int, default=0, help="Prefetch worker thread count.")
  parser.add_argument("--seed_offset", type=int, default=0, help="XOR offset added to data_loader.seed.")
  parser.add_argument("--start_batch_index", type=int, default=0, help="Starting index for batch_XXXXX.npz filenames.")
  parser.add_argument("--overwrite", type=_parse_bool, default=True, help="Overwrite existing .npz batch files.")
  args = parser.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  manifest = export_batches(
      olmo_config_path=args.olmo_config_path,
      output_dir=args.output_dir,
      num_batches=args.num_batches,
      work_dir=args.work_dir,
      global_batch_tokens=args.global_batch_tokens,
      pack_buffer_size=args.pack_buffer_size,
      max_target_length=args.max_target_length,
      max_crops_per_seq=args.max_crops_per_seq,
      max_pooled_patches_per_seq=args.max_pooled_patches_per_seq,
      path_remap_from=args.path_remap_from,
      path_remap_to=args.path_remap_to,
      num_workers=args.num_workers,
      seed_offset=args.seed_offset,
      start_batch_index=args.start_batch_index,
      overwrite=args.overwrite,
  )
  print(json.dumps({"status": "OK", "num_exported_batches": manifest["num_exported_batches"], "output_dir": args.output_dir}))


if __name__ == "__main__":
  main()
