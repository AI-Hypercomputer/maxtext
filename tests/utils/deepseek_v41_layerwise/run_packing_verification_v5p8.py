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

"""Standalone sequence packing isolation verification script on TPU v5p-8."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
from typing import Any

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from tests.utils.deepseek_v41_layerwise import lw_l2
from tests.utils.deepseek_v41_layerwise import lw_l4
from tests.utils.deepseek_v41_layerwise import lw_model
from tests.utils.deepseek_v41_layerwise import lw_programs
from tests.utils.deepseek_v41_layerwise import lw_stages

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses CLI flags with argparse and marks absl flags parsed."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--layer", type=int, default=1, help="Layer index to verify packing controls (default 1).")
  parser.add_argument("--hf-ckpt-dir", default="/mnt/disks/data/hf_selective", help="HF checkpoint directory.")
  parser.add_argument("--tokenizer-path", default="/mnt/disks/data/ds41_tokenizer", help="Tokenizer path.")
  parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length (default 2048).")
  parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default 1).")
  parser.add_argument("--output-json", default="", help="Output JSON path for packing verification telemetry.")

  args = parser.parse_args(argv[1:])
  flags.FLAGS.mark_as_parsed()
  return args


def build_packing_config(
    *,
    seq_len: int = 2048,
    batch_size: int = 1,
    tokenizer_path: str = "/mnt/disks/data/ds41_tokenizer",
    num_devices: int = 4,
) -> Any:
  """Builds MaxText config for layerwise packing verification."""
  args = {
      "run_name": "ds41_packing_verification",
      "base_output_directory": os.path.join(tempfile.gettempdir(), "ds41_packing"),
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash",
      "override_model_config": True,
      "use_multimodal": True,
      "vision_encoder_block": "deepseek41",
      "freeze_vision_encoder_params": True,
      "tokenizer_type": "huggingface",
      "tokenizer_path": tokenizer_path,
      "max_target_length": seq_len,
      "per_device_batch_size": batch_size / num_devices,
      "ici_fsdp_parallelism": num_devices,
      "ici_expert_parallelism": 1,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "dtype_mm": "bfloat16",
      "matmul_precision": "default",
      "float32_gate_logits": True,
      "ds41_activation_quant": False,
      "sparse_matmul": False,
      "capacity_factor": -1,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 0.0,
  }
  return pyconfig.initialize(
      [sys.argv[0], str(REPO_ROOT / "src" / "maxtext" / "configs" / "base.yml")],
      **args,
  )


def run_packing_verification(args: argparse.Namespace) -> dict[str, Any]:
  """Executes L2 and L4 sequence packing negative controls and asserts FIRED status."""
  t0 = time.time()
  num_dev = jax.device_count()
  config = build_packing_config(
      seq_len=args.seq_len,
      batch_size=args.batch_size,
      tokenizer_path=args.tokenizer_path,
      num_devices=num_dev,
  )
  mesh = lw_programs.build_mesh(config)
  hf_dir = pathlib.Path(args.hf_ckpt_dir)

  b, s = args.batch_size, args.seq_len
  rng = np.random.default_rng(42)
  tokens = rng.integers(100, 1000, size=(b, s), dtype=np.int32)
  positions = np.arange(s, dtype=np.int32)[None, :]
  img_mask = np.zeros((b, s), dtype=bool)

  # Full layer skeleton to obtain active Engram rows
  print(f"[PACKING] Initializing Layer {args.layer} skeleton...")
  full_parts, _ = lw_stages.load_layer(config, mesh, args.layer, hf_dir, None)
  layer_mod = lw_model.merge_fresh(full_parts.graphdef, full_parts.weights, full_parts.rest)

  active = np.unique(
      np.asarray(layer_mod.engram_hasher.val(input_ids=jnp.asarray(tokens), image_mask=jnp.asarray(img_mask)))
  )
  num_rows = getattr(layer_mod.engram, "num_rows", layer_mod.engram.embed[...].shape[0])
  controls = np.setdiff1d(np.arange(num_rows), active)[:8]

  rows = lw_stages.EngramRows(active, controls, tokens, img_mask, include_packing_rows=True)
  parts, plan = lw_stages.load_layer(config, mesh, args.layer, hf_dir, rows)

  lw_model.release(full_parts.weights)
  del full_parts, layer_mod

  sample_inputs = lw_l2.SampleInputs(
      positions=positions,
      tokens=tokens,
      image_mask=img_mask,
      text_pos_mask=~img_mask,
      image_pos_mask=img_mask,
      decoder_segment_ids=np.ones((b, s), np.int32),
  )

  dim = config.base_emb_dim
  carry = {
      "h": jnp.zeros((b, s, dim), jnp.bfloat16),
      "pre_mix": jnp.zeros((b, s, dim), jnp.bfloat16),
  }

  program = lw_programs.GroupProgram(config, mesh, (args.layer,), (parts.graphdef,), (plan,))
  replays = [{"moe_topk": jnp.zeros((b, s, config.num_experts_per_tok), jnp.int32)}]

  # 1. Forward L2 Packing Control
  print("[PACKING] Running L2 Forward Packing Control...")
  # pylint: disable=protected-access
  l2_pack = lw_l2._packing_control(program, [parts.weights], [parts.rest], carry, sample_inputs, replays, 0)
  if l2_pack["status"] != "FIRED":
    raise AssertionError(f"L2 packing control failed to fire: {l2_pack}")
  print(f"[PACKING] L2 Forward Packing Control FIRED: max_abs_diff_ones={l2_pack['max_abs_diff_ones']:.4e}")

  # 2. Backward L4 VJP Packing Control
  print("[PACKING] Running L4 Backward VJP Packing Control...")
  vjp_entry = jax.jit(lw_programs.layer_vjp_entry(program))
  cotangents = {"h": jnp.ones_like(carry["h"]), "pre_mix": jnp.ones_like(carry["pre_mix"])}
  l4_pack = lw_l4._packing_control(
      vjp_entry, parts.weights, parts.rest, carry, sample_inputs, replays[0], cotangents, args.layer
  )
  if l4_pack["status"] != "FIRED":
    raise AssertionError(f"L4 packing control failed to fire: {l4_pack}")
  print(f"[PACKING] L4 Backward VJP Packing Control FIRED: grad_frob_ones={l4_pack['grad_frob_ones']:.4e}")

  elapsed = time.time() - t0
  report = {
      "layer": args.layer,
      "batch_size": args.batch_size,
      "seq_len": args.seq_len,
      "l2_pack": l2_pack,
      "l4_pack": l4_pack,
      "passed": True,
      "elapsed_seconds": elapsed,
  }
  return report


def main(args: argparse.Namespace) -> None:
  report = run_packing_verification(args)
  if args.output_json:
    out_path = pathlib.Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"[PACKING] Telemetry written to {out_path}")
  print("[PACKING] Sequence packing isolation verification passed successfully!")


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
