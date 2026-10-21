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

"""Generate synthetic text and multimodal ArrayRecord shards for GCS QPS benchmarking.

Creates realistic `.array_record` datasets for:
1. Text-based pre-training (`text`):
   - Variable-length documents sampled from a log-normal length distribution
     (mimicking C4 / RedPajama / Dolma) so that PyGrain's `FirstFitPackIterDataset`
     packs multiple raw ArrayRecord documents (~2.3-3.5 records) per `max_target_length`
     training sequence.
2. Multimodal vision training (`multimodal`):
   - `tf.train.Example` records containing `"question"`, `"answer"`, and `"image"`
     (encoded RGB JPEG/PNG image bytes spanning multiple 64 KiB Riegeli blocks).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import io
import json
import os
from typing import Sequence

from array_record.python.array_record_module import ArrayRecordWriter
import numpy as np
from PIL import Image

from maxtext.input_pipeline.protos import example_pb2

VOCAB_WORDS = (
    "transformer",
    "attention",
    "gradient",
    "sequence",
    "token",
    "embedding",
    "layer",
    "normalization",
    "activation",
    "parameter",
    "throughput",
    "latency",
    "bandwidth",
    "storage",
    "bucket",
    "sharding",
    "pipeline",
    "prefetch",
    "multiprocess",
    "worker",
    "accelerator",
    "matrix",
    "tensor",
    "computation",
    "distributed",
    "allreduce",
    "checkpoint",
    "convergence",
    "optimization",
    "schedule",
    "dataset",
    "record",
)


def _make_text_record(rng: np.random.Generator, doc_idx: int, mean_words: int = 450) -> bytes:
  """Creates a serialized `tf.train.Example` with log-normal document length."""
  num_words = int(np.clip(rng.lognormal(mean=np.log(mean_words), sigma=0.55), 64, 2400))
  chosen = rng.choice(VOCAB_WORDS, size=num_words, replace=True)
  text = f"Document {doc_idx}: " + " ".join(chosen) + "."

  ex = example_pb2.Example()
  ex.features.feature["text"].bytes_list.value.append(text.encode("utf-8"))
  return ex.SerializeToString()


def _make_encoded_image_bytes(
    rng: np.random.Generator,
    width: int,
    height: int,
    image_format: str = "JPEG",
    jpeg_quality: int = 90,
) -> bytes:
  """Creates realistic high-entropy encoded image bytes."""
  x = np.linspace(0, 255, width, dtype=np.float32)
  y = np.linspace(0, 255, height, dtype=np.float32)
  xx, yy = np.meshgrid(x, y)
  base = np.stack([xx, yy, (xx + yy) * 0.5], axis=-1)
  noise = rng.integers(0, 120, size=(height, width, 3), dtype=np.uint8)
  arr = np.clip(base * 0.5 + noise, 0, 255).astype(np.uint8)

  buf = io.BytesIO()
  img = Image.fromarray(arr, mode="RGB")
  if image_format.upper() == "PNG":
    img.save(buf, format="PNG", compress_level=1)
  else:
    img.save(buf, format="JPEG", quality=jpeg_quality)
  return buf.getvalue()


def _make_multimodal_record(
    record_idx: int,
    image_pool: Sequence[bytes],
) -> bytes:
  """Creates a serialized `tf.train.Example` with question, answer, and image bytes."""
  img_bytes = image_pool[record_idx % len(image_pool)]
  prompt = (
      f"Analyze the visual patterns and objects in sample {record_idx} "
      "and summarize the primary structural details."
  )
  answer = (
      f"Sample {record_idx} depicts a synthetic high-resolution RGB test frame "
      "used to benchmark multimodal PyGrain + ArrayRecord storage QPS."
  )

  ex = example_pb2.Example()
  ex.features.feature["question"].bytes_list.value.append(prompt.encode("utf-8"))
  ex.features.feature["answer"].bytes_list.value.append(answer.encode("utf-8"))
  ex.features.feature["image"].bytes_list.value.append(img_bytes)
  return ex.SerializeToString()


def _check_cached_manifest(
    mod_dir: str,
    num_shards: int,
    records_per_shard: int,
    writer_options: str,
) -> dict | None:
  manifest_path = os.path.join(mod_dir, "manifest.json")
  if not os.path.isfile(manifest_path):
    return None
  try:
    with open(manifest_path, "r", encoding="utf-8") as f:
      meta = json.load(f)
    if (
        meta.get("num_shards") == num_shards
        and meta.get("records_per_shard") == records_per_shard
        and meta.get("writer_options") == writer_options
        and len(glob.glob(os.path.join(mod_dir, "train.array_record-*"))) == num_shards
    ):
      return meta
  except (OSError, json.JSONDecodeError):
    pass
  return None


def _clean_old_shards(mod_dir: str) -> None:
  for p in glob.glob(os.path.join(mod_dir, "train.array_record-*")):
    try:
      os.unlink(p)
    except OSError:
      pass


def generate_text_dataset(
    output_dir: str,
    num_shards: int = 64,
    records_per_shard: int = 256,
    group_size: int = 1,
    compression: str = "uncompressed",
    mean_words: int = 450,
    seed: int = 42,
    force_regenerate: bool = False,
) -> dict:
  """Generates synthetic text pre-training ArrayRecord shards."""
  text_dir = os.path.join(output_dir, "text")
  os.makedirs(text_dir, exist_ok=True)
  writer_options = f"group_size:{group_size},{compression}"

  if not force_regenerate:
    cached = _check_cached_manifest(text_dir, num_shards, records_per_shard, writer_options)
    if cached is not None:
      return cached

  _clean_old_shards(text_dir)

  # Pre-generate a pool of 512 variable-length serialized text Examples for fast multi-shard creation
  pool_rng = np.random.default_rng(seed)
  record_pool = [_make_text_record(pool_rng, i, mean_words=mean_words) for i in range(512)]

  def _write_shard(shard_idx: int) -> tuple[str, int]:
    shard_path = os.path.join(
        text_dir,
        f"train.array_record-{shard_idx:05d}-of-{num_shards:05d}",
    )
    writer = ArrayRecordWriter(shard_path, writer_options)
    for r in range(records_per_shard):
      global_idx = shard_idx * records_per_shard + r
      writer.write(record_pool[(global_idx * 131 + shard_idx) % len(record_pool)])
    writer.close()
    return shard_path, os.path.getsize(shard_path)

  with ThreadPoolExecutor(max_workers=min(16, num_shards)) as ex:
    results = list(ex.map(_write_shard, range(num_shards)))

  total_bytes = sum(sz for _, sz in results)
  total_records = num_shards * records_per_shard
  meta = {
      "modality": "text",
      "dir": text_dir,
      "glob_pattern": os.path.join(text_dir, "train.array_record-*"),
      "num_shards": num_shards,
      "records_per_shard": records_per_shard,
      "total_records": total_records,
      "total_bytes": total_bytes,
      "avg_record_bytes": total_bytes / max(1, total_records),
      "writer_options": writer_options,
  }
  with open(os.path.join(text_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
  return meta


def generate_multimodal_dataset(
    output_dir: str,
    num_shards: int = 64,
    records_per_shard: int = 64,
    group_size: int = 1,
    compression: str = "uncompressed",
    image_width: int = 896,
    image_height: int = 896,
    image_format: str = "JPEG",
    seed: int = 123,
    force_regenerate: bool = False,
) -> dict:
  """Generates synthetic multimodal vision SFT ArrayRecord shards."""
  mm_dir = os.path.join(output_dir, "multimodal")
  os.makedirs(mm_dir, exist_ok=True)
  writer_options = f"group_size:{group_size},{compression}"

  if not force_regenerate:
    cached = _check_cached_manifest(mm_dir, num_shards, records_per_shard, writer_options)
    if cached is not None:
      return cached

  _clean_old_shards(mm_dir)

  pool_rng = np.random.default_rng(seed)
  image_pool = [
      _make_encoded_image_bytes(pool_rng, image_width, image_height, image_format=image_format)
      for _ in range(16)
  ]

  def _write_shard(shard_idx: int) -> tuple[str, int]:
    shard_path = os.path.join(
        mm_dir,
        f"train.array_record-{shard_idx:05d}-of-{num_shards:05d}",
    )
    writer = ArrayRecordWriter(shard_path, writer_options)
    for r in range(records_per_shard):
      global_idx = shard_idx * records_per_shard + r
      writer.write(_make_multimodal_record(global_idx, image_pool))
    writer.close()
    return shard_path, os.path.getsize(shard_path)

  with ThreadPoolExecutor(max_workers=min(16, num_shards)) as ex:
    results = list(ex.map(_write_shard, range(num_shards)))

  total_bytes = sum(sz for _, sz in results)
  total_records = num_shards * records_per_shard
  meta = {
      "modality": "multimodal",
      "dir": mm_dir,
      "glob_pattern": os.path.join(mm_dir, "train.array_record-*"),
      "num_shards": num_shards,
      "records_per_shard": records_per_shard,
      "total_records": total_records,
      "total_bytes": total_bytes,
      "avg_record_bytes": total_bytes / max(1, total_records),
      "avg_image_bytes": float(np.mean([len(b) for b in image_pool])),
      "writer_options": writer_options,
  }
  with open(os.path.join(mm_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
  return meta


def main():
  parser = argparse.ArgumentParser(description="Generate synthetic ArrayRecord datasets for QPS benchmarking.")
  parser.add_argument("--output_dir", type=str, required=True, help="Local directory to write ArrayRecord shards.")
  parser.add_argument(
      "--modality",
      type=str,
      choices=("text", "multimodal", "both"),
      default="both",
      help="Which dataset modality to generate.",
  )
  parser.add_argument("--num_shards", type=int, default=64, help="Number of .array_record shards per modality.")
  parser.add_argument("--text_records_per_shard", type=int, default=256, help="Records per shard for text dataset.")
  parser.add_argument(
      "--multimodal_records_per_shard", type=int, default=64, help="Records per shard for multimodal dataset."
  )
  parser.add_argument("--group_size", type=int, default=1, help="ArrayRecord writer group_size.")
  parser.add_argument("--compression", type=str, default="uncompressed", help="ArrayRecord writer compression.")
  parser.add_argument("--image_width", type=int, default=896, help="Synthetic image width for multimodal dataset.")
  parser.add_argument("--image_height", type=int, default=896, help="Synthetic image height for multimodal dataset.")
  args = parser.parse_args()

  if args.modality in ("text", "both"):
    stats = generate_text_dataset(
        args.output_dir,
        num_shards=args.num_shards,
        records_per_shard=args.text_records_per_shard,
        group_size=args.group_size,
        compression=args.compression,
    )
    print(f"[TEXT] Generated {stats['num_shards']} shards ({stats['total_records']} records) in {stats['dir']}")

  if args.modality in ("multimodal", "both"):
    stats = generate_multimodal_dataset(
        args.output_dir,
        num_shards=args.num_shards,
        records_per_shard=args.multimodal_records_per_shard,
        group_size=args.group_size,
        compression=args.compression,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    print(
        f"[MULTIMODAL] Generated {stats['num_shards']} shards ({stats['total_records']} records) in {stats['dir']}"
    )


if __name__ == "__main__":
  main()
