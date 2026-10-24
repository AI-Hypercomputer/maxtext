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

"""Slices a real-weight mini subset of inclusionAI/Ming-3.0-audio-tiny.

The text decoder repeats a 4-layer cycle (KDA, KDA, KDA, MLA) and only layer 0
uses a dense MLP, so the first 4 layers form the minimal representative subset:

  layer 0: KDA + dense SwiGLU
  layer 1: KDA + MoE (128 experts, top-8, 1 shared)
  layer 2: KDA + MoE
  layer 3: gated MLA + MoE

The whisper audio encoder, audio projector, embeddings, final norm and lm_head
are kept in full. The MTP layer (index 24) is dropped.

Usage:
  HF_HOME=/dev/shm/$USER python3 tests/utils/ming/prepare_mini.py \
      --output=/dev/shm/hf_mini/ming-3.0-audio-tiny_4layers --num_layers=4
"""

import argparse
import json
import os
import re
import shutil

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

MING_HF_MODEL_ID = "inclusionAI/Ming-3.0-audio-tiny"
_COPY_FILES = (
    "config.json",
    "configuration_bailing_moe_v3.py",
    "configuration_bailing_moe_v3_audio.py",
    "modeling_bailing_moe_v3.py",
    "modeling_bailing_moe_v3_audio.py",
    "processing_bailing_moe_v3_audio.py",
    "processor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def keep_key(key: str, num_layers: int) -> bool:
  m = re.match(r"model\.layers\.(\d+)\.", key)
  if m:
    return int(m.group(1)) < num_layers
  return True  # embeddings, norm, lm_head, audio encoder, audio projector


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_id", default=MING_HF_MODEL_ID)
  parser.add_argument("--output", required=True)
  parser.add_argument("--num_layers", type=int, default=4)
  args = parser.parse_args()

  src = snapshot_download(args.model_id, allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors"])
  os.makedirs(args.output, exist_ok=True)
  for f in _COPY_FILES:
    shutil.copy(os.path.join(src, f), os.path.join(args.output, f))

  with open(os.path.join(src, "model.safetensors.index.json"), encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  keys = sorted(k for k in weight_map if keep_key(k, args.num_layers))
  tensors = {}
  for shard in sorted(set(weight_map[k] for k in keys)):
    with safe_open(os.path.join(src, shard), "pt") as sf:
      for k in keys:
        if weight_map[k] == shard:
          tensors[k] = sf.get_tensor(k).contiguous()
  save_file(tensors, os.path.join(args.output, "model.safetensors"), metadata={"format": "pt"})

  with open(os.path.join(args.output, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
  cfg["text_config"]["num_hidden_layers"] = args.num_layers
  cfg["text_config"]["num_nextn_predict_layers"] = 0
  with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
  print(f"Saved {len(tensors)} tensors with {args.num_layers} text layers to {args.output}")


if __name__ == "__main__":
  main()
