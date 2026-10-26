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

"""Quantizes an unquantized DeepSeek-V4.1 HF-layout checkpoint into the released FP8 / FP4 layout.

Same tensors the released DeepSeek-V4.1-Flash checkpoint quantizes, same formats: FP8 E4M3 weights with one E8M0
(power-of-two) scale per 32 x 32 block for linears, packed FP4 E2M1 (int8 storage, two codes per byte, low nibble
first) with one E8M0 scale per 1 x 32 block for routed experts, and FP8 rows with 1 x 32 scales for the engram
tables. Used to test the dequantizing loaders at mini scale.

  python -m tests.utils.deepseek_v41_reference.quantize_checkpoint --input_dir /tmp/mini --output_dir /tmp/mini_q
"""

import argparse
import glob
import json
import os
import re
import shutil

import torch

_FP4_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP8_LINEARS = re.compile(r"(attn\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)|shared_experts\.w[123]|engram\.wkv)\.weight$")
_FP4_EXPERTS = re.compile(r"ffn\.experts\.\d+\.w[123]\.weight$")
_ENGRAM_TABLE = re.compile(r"engram\.embed\.weight$")


def _pow2_scale(amax: torch.Tensor, fmax: float) -> torch.Tensor:
  return torch.exp2(torch.ceil(torch.log2(torch.clamp(amax, min=1e-30) / fmax)))


def quantize_fp8(w: torch.Tensor, block_rows: int, block_cols: int):
  rows, cols = w.shape
  wf = w.float().reshape(rows // block_rows, block_rows, cols // block_cols, block_cols)
  scale = _pow2_scale(wf.abs().amax(dim=(1, 3)), 448.0)
  q = (wf / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(rows, cols)
  return q, scale.to(torch.float8_e8m0fnu)


def quantize_fp4(w: torch.Tensor):
  rows, cols = w.shape
  wf = w.float().reshape(rows, cols // 32, 32)
  scale = _pow2_scale(wf.abs().amax(dim=-1), 6.0)
  x = (wf / scale[..., None]).reshape(rows, cols)
  mag = (x.abs()[..., None] - _FP4_E2M1_VALUES).abs().argmin(dim=-1)
  codes = (mag | ((x < 0) & (mag > 0)).long() << 3).to(torch.uint8)
  packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
  return packed, scale.to(torch.float8_e8m0fnu)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--input_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  args = parser.parse_args()
  from safetensors.torch import load_file, save_file  # pylint: disable=import-outside-toplevel

  os.makedirs(args.output_dir, exist_ok=True)
  weight_map = {}
  for path in sorted(glob.glob(os.path.join(args.input_dir, "*.safetensors"))):
    out = {}
    for name, t in load_file(path).items():
      if _FP4_EXPERTS.search(name):
        out[name], out[name[: -len("weight")] + "scale"] = quantize_fp4(t)
      elif _FP8_LINEARS.search(name):
        out[name], out[name[: -len("weight")] + "scale"] = quantize_fp8(t, 32, 32)
      elif _ENGRAM_TABLE.search(name):
        out[name], out[name[: -len("weight")] + "scale"] = quantize_fp8(t, 1, 32)
      else:
        out[name] = t
    fname = os.path.basename(path)
    save_file(out, os.path.join(args.output_dir, fname), metadata={"format": "pt"})
    weight_map.update({k: fname for k in out})
    print(f"wrote {fname}", flush=True)
  with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
  with open(os.path.join(args.input_dir, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
  config["quantization_config"] = {
      "quant_method": "fp8",
      "activation_scheme": "dynamic",
      "weight_block_size": [32, 32],
      "scale_fmt": "ue8m0",
      "expert_dtype": "fp4",
  }
  with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
  for extra in ("tokenizer.json", "tokenizer_config.json"):
    if os.path.exists(os.path.join(args.input_dir, extra)):
      shutil.copyfile(os.path.join(args.input_dir, extra), os.path.join(args.output_dir, extra))


if __name__ == "__main__":
  main()
