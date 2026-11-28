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

"""Generates synthetic ArrayRecord shards for the input pipeline benchmark.

- `text`: `Example` protos with a `text` feature. Document lengths follow a
  log-normal distribution (C4-like), so `FirstFit` packing pulls several
  variable-length records per training sequence.
- `multimodal`: `Example` protos with `question`, `answer`, and encoded JPEG
  `image` bytes that span multiple 64 KiB Riegeli blocks.

Each modality directory gets `dataset_info.json` (shard, record, and byte counts
measured from the written files) and `shard_record_counts.json`, a BP-1
pre-indexed manifest that maps shard path to record count.
"""

import argparse
from concurrent import futures
import glob
import io
import json
import os

import numpy as np

from maxtext.input_pipeline.protos import example_pb2

try:
  from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
  from array_record import ArrayRecordWriter

SHARD_PREFIX = "train.array_record"
DATASET_INFO = "dataset_info.json"
MANIFEST = "shard_record_counts.json"

_VOCAB = (
    "transformer attention gradient sequence token embedding layer normalization activation parameter "
    "throughput latency bandwidth storage bucket sharding pipeline prefetch multiprocess worker "
    "accelerator matrix tensor computation distributed allreduce checkpoint convergence optimization "
    "schedule dataset record"
).split()


def _text_record(rng: np.random.Generator, doc_idx: int, mean_words: int) -> bytes:
  num_words = int(np.clip(rng.lognormal(mean=np.log(mean_words), sigma=0.55), 16, 2400))
  text = f"Document {doc_idx}: " + " ".join(rng.choice(_VOCAB, size=num_words)) + "."
  ex = example_pb2.Example()
  ex.features.feature["text"].bytes_list.value.append(text.encode("utf-8"))
  return ex.SerializeToString()


def _jpeg_bytes(rng: np.random.Generator, width: int, height: int) -> bytes:
  from PIL import Image  # pylint: disable=import-outside-toplevel

  xx, yy = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
  base = np.stack([xx, yy, (xx + yy) * 0.5], axis=-1)
  arr = np.clip(base * 0.5 + rng.integers(0, 120, size=(height, width, 3)), 0, 255).astype(np.uint8)
  buf = io.BytesIO()
  Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=90)
  return buf.getvalue()


def _multimodal_record(idx: int, image: bytes) -> bytes:
  ex = example_pb2.Example()
  ex.features.feature["question"].bytes_list.value.append(f"Describe the structure of sample {idx}.".encode("utf-8"))
  ex.features.feature["answer"].bytes_list.value.append(f"Sample {idx} is a synthetic RGB test frame.".encode("utf-8"))
  ex.features.feature["image"].bytes_list.value.append(image)
  return ex.SerializeToString()


def shard_paths(mod_dir: str) -> list[str]:
  return sorted(glob.glob(os.path.join(mod_dir, f"{SHARD_PREFIX}-*")))


def _cached_info(mod_dir: str, expected: dict) -> dict | None:
  path = os.path.join(mod_dir, DATASET_INFO)
  try:
    with open(path, "r", encoding="utf-8") as f:
      info = json.load(f)
  except (OSError, json.JSONDecodeError):
    return None
  if all(info.get(k) == v for k, v in expected.items()) and len(shard_paths(mod_dir)) == expected["num_shards"]:
    return info
  return None


def _write_dataset(mod_dir: str, num_shards: int, records_per_shard: int, writer_options: str, make_record, extra: dict):
  """Writes shards in parallel and returns measured dataset info."""
  os.makedirs(mod_dir, exist_ok=True)
  expected = {"num_shards": num_shards, "records_per_shard": records_per_shard, "writer_options": writer_options, **extra}
  cached = _cached_info(mod_dir, expected)
  if cached is not None:
    return cached
  for p in shard_paths(mod_dir):
    os.unlink(p)

  def _write(shard_idx: int) -> tuple[str, int, int]:
    path = os.path.join(mod_dir, f"{SHARD_PREFIX}-{shard_idx:05d}-of-{num_shards:05d}")
    writer = ArrayRecordWriter(path, writer_options)
    payload = 0
    for r in range(records_per_shard):
      rec = make_record(shard_idx * records_per_shard + r)
      payload += len(rec)
      writer.write(rec)
    writer.close()
    return path, os.path.getsize(path), payload

  with futures.ThreadPoolExecutor(max_workers=min(16, num_shards)) as ex:
    results = list(ex.map(_write, range(num_shards)))
  total_records = num_shards * records_per_shard
  info = {
      **expected,
      "dir": mod_dir,
      "glob_pattern": os.path.join(mod_dir, f"{SHARD_PREFIX}-*"),
      "total_records": total_records,
      "total_file_bytes": sum(r[1] for r in results),
      "total_record_bytes": sum(r[2] for r in results),
      "mean_record_bytes": sum(r[2] for r in results) / max(1, total_records),
  }
  with open(os.path.join(mod_dir, DATASET_INFO), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
  with open(os.path.join(mod_dir, MANIFEST), "w", encoding="utf-8") as f:
    json.dump({path: records_per_shard for path, _, _ in results}, f, indent=2)
  return info


def generate_text_dataset(
    output_dir: str,
    num_shards: int = 16,
    records_per_shard: int = 256,
    writer_options: str = "group_size:1",
    mean_words: int = 450,
    seed: int = 42,
) -> dict:
  """Generates `text` shards under `<output_dir>/text` and returns dataset info."""
  pool_rng = np.random.default_rng(seed)
  pool = [_text_record(pool_rng, i, mean_words) for i in range(512)]
  return _write_dataset(
      os.path.join(output_dir, "text"),
      num_shards,
      records_per_shard,
      writer_options,
      lambda i: pool[(i * 131 + i // records_per_shard) % len(pool)],
      {"modality": "text", "mean_words": mean_words, "seed": seed},
  )


def generate_multimodal_dataset(
    output_dir: str,
    num_shards: int = 16,
    records_per_shard: int = 64,
    writer_options: str = "group_size:1",
    image_size: int = 896,
    seed: int = 123,
) -> dict:
  """Generates `multimodal` shards under `<output_dir>/multimodal` and returns dataset info."""
  pool_rng = np.random.default_rng(seed)
  images = [_jpeg_bytes(pool_rng, image_size, image_size) for _ in range(16)]
  return _write_dataset(
      os.path.join(output_dir, "multimodal"),
      num_shards,
      records_per_shard,
      writer_options,
      lambda i: _multimodal_record(i, images[i % len(images)]),
      {"modality": "multimodal", "image_size": image_size, "seed": seed},
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--modality", choices=("text", "multimodal", "both"), default="text")
  parser.add_argument("--num_shards", type=int, default=16)
  parser.add_argument("--text_records_per_shard", type=int, default=256)
  parser.add_argument("--multimodal_records_per_shard", type=int, default=64)
  parser.add_argument("--writer_options", default="group_size:1")
  args = parser.parse_args()
  if args.modality in ("text", "both"):
    print(
        json.dumps(
            generate_text_dataset(args.output_dir, args.num_shards, args.text_records_per_shard, args.writer_options)
        )
    )
  if args.modality in ("multimodal", "both"):
    print(
        json.dumps(
            generate_multimodal_dataset(
                args.output_dir, args.num_shards, args.multimodal_records_per_shard, args.writer_options
            )
        )
    )


if __name__ == "__main__":
  main()
