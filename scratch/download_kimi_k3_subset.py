# Copyright 2026 Google LLC
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

"""Phase 1: Partial HuggingFace Shard Downloader for Kimi K3.

Downloads only the minimum required .safetensors shards from moonshotai/Kimi-K3
to cover:
- config.json & model.safetensors.index.json
- model.embed_tokens.weight
- model.norm.weight & lm_head.weight
- model.layers.0.* (Layer 0: KDA + MoE)
- model.layers.3.* (Layer 3: MLA + MoE)
"""

import json
import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "moonshotai/Kimi-K3"
LOCAL_DIR = "scratch/hf_kimi_k3_subset"


def download_file(filename: str) -> str:
  """Downloads a single file from the HF repo into LOCAL_DIR."""
  print(f"Downloading {filename}...", flush=True)

  path = hf_hub_download(
      repo_id=REPO_ID,
      filename=filename,
      local_dir=LOCAL_DIR,
      local_dir_use_symlinks=False,
  )
  print(f"  Saved to: {path}")
  return path


def main():
  os.makedirs(LOCAL_DIR, exist_ok=True)

  # 1. Download config.json, model.safetensors.index.json, and all Python code files
  print("=== Step 1: Downloading config, index, and Python modeling files ===")
  try:
    config_path = download_file("config.json")
  except Exception as e:
    print(f"Error downloading config.json from {REPO_ID}: {e}")
    print("Checking available repo files...")
    files = list_repo_files(REPO_ID)
    print("Repo files:", files[:20])
    sys.exit(1)

  try:
    index_path = download_file("model.safetensors.index.json")
  except Exception as e:
    print(f"Error downloading model.safetensors.index.json: {e}")
    files = list_repo_files(REPO_ID)
    print("Repo files:", files[:20])
    sys.exit(1)

  py_files = [
      "configuration_kimi_k3.py",
      "modeling_kimi_k3.py",
      "modeling_kimi_linear.py",
      "encoding_k3.py",
      "media_utils.py",
      "tokenization_kimi.py",
  ]
  for pf in py_files:
    try:
      download_file(pf)
    except Exception as e:
      print(f"Warning: could not download {pf}: {e}")

  # 2. Parse index.json to find required shards
  print("\n=== Step 2: Parsing index.json to identify required shards ===")
  with open(index_path, "r") as f:
    index_data = json.load(f)

  weight_map = index_data.get("weight_map", {})
  print(f"Total tensors in weight_map: {len(weight_map)}")

  # We need shards for:
  # - model.embed_tokens.weight
  # - model.norm.weight
  # - lm_head.weight
  # - model.layers.0.*
  # - model.layers.3.*
  required_patterns = [
      "model.embed_tokens.weight",
      "model.norm.weight",
      "lm_head.weight",
      "model.layers.0.",
      "model.layers.3.",
  ]

  required_shards = set()
  matched_tensors = []

  for tensor_name, shard_file in weight_map.items():
    for pattern in required_patterns:
      if pattern in tensor_name:
        required_shards.add(shard_file)
        matched_tensors.append((tensor_name, shard_file))
        break

  print(f"\nFound {len(matched_tensors)} matching tensors across {len(required_shards)} shards:")
  for shard in sorted(required_shards):
    tensors_in_shard = [t for t, s in matched_tensors if s == shard]
    print(f"  - {shard}: {len(tensors_in_shard)} tensors")
    for t in tensors_in_shard[:5]:
      print(f"      {t}")
    if len(tensors_in_shard) > 5:
      print(f"      ... and {len(tensors_in_shard) - 5} more")

  # 3. Download the required shards
  print(f"\n=== Step 3: Downloading {len(required_shards)} required shard(s) ===")
  for shard in sorted(required_shards):
    download_file(shard)

  print("\n=== Phase 1 Complete! ===")
  print(f"All required shards downloaded to {LOCAL_DIR}")


if __name__ == "__main__":
  main()
