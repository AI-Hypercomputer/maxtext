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

"""Extract the representative four-layer prefix without downloading PLE.

This creates a validation fixture, not a generally loadable full HF model.
PLE rows are collected separately from the pinned source during golden runs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import urllib.request

from maxtext.checkpoint_conversion.utils.safetensors_range import read_range, RemoteSafetensors

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def prepare_source(source):
  """Fetch pinned index/config and headers, making the slice command standalone."""
  source.mkdir(parents=True, exist_ok=True)
  metadata = source / "hub_metadata.json"
  if metadata.exists() and json.loads(metadata.read_text())["sha"] != REVISION:
    raise ValueError("Source metadata belongs to a different checkpoint revision")
  metadata.write_text(json.dumps({"sha": REVISION, "model_id": MODEL_ID}))
  for name in ["config.json", "model.safetensors.index.json"]:
    if not (source / name).exists():
      with urllib.request.urlopen(
          f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", timeout=120
      ) as response:
        (source / name).write_bytes(response.read())
  if (source / "headers.json").exists() and (source / "selected.json").exists():
    return
  index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
  keys = [
      key
      for key in index
      if key == "lm_head.weight"
      or (
          key.startswith("model.language_model.")
          and (".layers." not in key or any(key.startswith(f"model.language_model.layers.{i}.") for i in range(4)))
      )
  ]

  def fetch_header(filename):
    reader = RemoteSafetensors(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}")
    return filename, {"header": reader.header, "data_start": reader.data_start}

  with ThreadPoolExecutor(max_workers=4) as pool:
    headers = dict(pool.map(fetch_header, sorted({index[key] for key in keys})))
  selected = {key: {"file": index[key], **headers[index[key]]["header"][key]} for key in keys}
  (source / "headers.json").write_text(json.dumps(headers))
  (source / "selected.json").write_text(json.dumps(selected))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", default="/dev/shm/qwen38_bringup/source")
  parser.add_argument("--output", default="/dev/shm/hf_mini/qwen3.8-flash-next_4layers")
  args = parser.parse_args()
  source, output = Path(args.source), Path(args.output)
  prepare_source(source)
  output.mkdir(parents=True, exist_ok=True)
  revision = json.loads((source / "hub_metadata.json").read_text())["sha"]
  headers = json.loads((source / "headers.json").read_text())
  selected = json.loads((source / "selected.json").read_text())
  tensors = [(key, spec) for key, spec in selected.items() if "ngram_embedding" not in key]
  total = sum(spec["data_offsets"][1] - spec["data_offsets"][0] for _, spec in tensors)
  available = __import__("shutil").disk_usage(output).free
  if available < total + 2**30:
    raise RuntimeError(f"Mini slice needs {total} bytes plus 1 GiB headroom; available {available}")

  def extract(item):
    number, (key, spec) = item
    name = f"model-{number:05d}-of-{len(tensors):05d}.safetensors"
    target = output / name
    size = spec["data_offsets"][1] - spec["data_offsets"][0]
    header = json.dumps({key: {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [0, size]}}).encode()
    header += b" " * (-len(header) % 8)
    expected = 8 + len(header) + size
    if not target.exists() or target.stat().st_size != expected:
      partial = target.with_suffix(".partial")
      url = f'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{revision}/{spec["file"]}'
      start = headers[spec["file"]]["data_start"] + spec["data_offsets"][0]
      with partial.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for offset in range(0, size, 16 * 1024 * 1024):
          for attempt in range(3):
            try:
              data = read_range(url, start + offset, min(16 * 1024 * 1024, size - offset))
              break
            except (OSError, ValueError):
              if attempt == 2:
                raise
          f.write(data)
      partial.replace(target)
    print(f"{number}/{len(tensors)} {key} {size/2**20:.1f} MiB", flush=True)
    return key, name

  with ThreadPoolExecutor(max_workers=4) as pool:
    weight_map = dict(pool.map(extract, enumerate(tensors, 1)))
  (output / "model.safetensors.index.json").write_text(
      json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
  )
  cfg = json.loads((source / "config.json").read_text())
  cfg["text_config"]["num_hidden_layers"] = 4
  cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:4]
  (output / "config.json").write_text(json.dumps(cfg, indent=2))
  (output / "ple_source.json").write_text(
      json.dumps(
          {
              "model_id": "Qwen/Qwen3.8-Flash-Next",
              "revision": revision,
              "table_tensors": {k: v for k, v in selected.items() if "ngram_embedding" in k},
              "headers": headers,
          }
      )
  )
  for name in [
      "tokenizer.json",
      "tokenizer_config.json",
      "vocab.json",
      "merges.txt",
      "generation_config.json",
      "chat_template.jinja",
  ]:
    with urllib.request.urlopen(
        f"https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{revision}/{name}", timeout=120
    ) as response:
      data = response.read()
    (output / name).write_bytes(data)
  print(
      f"Completed {len(tensors)} tensors; {total/2**30:.3f} GiB. PLE must be attached via exact sparse lookup.",
      flush=True,
  )


if __name__ == "__main__":
  main()
