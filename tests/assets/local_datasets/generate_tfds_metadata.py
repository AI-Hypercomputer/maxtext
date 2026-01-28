#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Generate minimal TFDS metadata for existing local C4 ArrayRecord shards.

Usage:
  python local_datasets/generate_tfds_metadata.py \
      --root local_datasets/c4_en_dataset_minimal \
      --version 3.1.0 \
      --force

This script creates a tiny TFDS builder and outputs the ``dataset_info.json`` and 
``features.json`` files.

After running, you can point TFDS to ``--root`` and load with
``dataset_name='c4/en:3.1.0'``.
"""
from __future__ import annotations
import os
import json
import argparse
import tensorflow_datasets as tfds  # type: ignore


def write_metadata(root: str, version_dir: str, dataset_version: str, force: bool = False) -> None:
  """Write TFDS ``dataset_info.json`` and ``features.json`` for local C4 shards."""

  info_path = os.path.join(version_dir, "dataset_info.json")
  if os.path.exists(info_path):
    if force:
      os.remove(info_path)
      print("Removed existing dataset_info.json due to --force.")
    else:
      print("dataset_info.json already exists; skipping overwrite (use --force to regenerate).")
      return

  # Discover shards (we assume they exist and are correct; counts are fixed)
  num_shards_train = 8
  num_shards_val = 2
  exact_train_records = 1000
  exact_val_records = 200

  train_records_per_shard = exact_train_records // num_shards_train
  val_records_per_shard = exact_val_records // num_shards_val
  train_shard_lengths = [train_records_per_shard] * num_shards_train
  val_shard_lengths = [val_records_per_shard] * num_shards_val

  train_split = tfds.core.SplitInfo(name="train", shard_lengths=train_shard_lengths, num_bytes=0)
  val_split = tfds.core.SplitInfo(name="validation", shard_lengths=val_shard_lengths, num_bytes=0)

  class _LocalC4Builder(tfds.core.GeneratorBasedBuilder):
    """Tiny builder used only to materialize TFDS metadata on disk."""

    VERSION = tfds.core.Version(dataset_version)
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name="en", version=VERSION, description="Local minimal C4 EN subset")]

    def _info(self) -> tfds.core.DatasetInfo:  # type: ignore[override]
      info = tfds.core.DatasetInfo(
          builder=self,
          description="Local minimal C4 English subset.",
          features=tfds.features.FeaturesDict({"text": tfds.features.Text()}),
          homepage="https://www.tensorflow.org/datasets/catalog/c4",
          citation="",
      )
      info.set_splits({"train": train_split, "validation": val_split})
      return info

    def _split_generators(self, dl_manager):  # type: ignore[override]
      """No actual generation; data already exists on disk."""
      del dl_manager
      return []

    def _generate_examples(self):  # type: ignore[override]
      """No example generation; placeholder to satisfy API."""
      yield from ()

  builder = _LocalC4Builder(data_dir=root)
  info = builder.info

  # Write canonical files (features.json + dataset_info.json)
  info.write_to_directory(version_dir)
  print(f"Wrote TFDS dataset_info & features to {version_dir}")

  info_path = os.path.join(version_dir, "dataset_info.json")
  try:
    with open(info_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    if isinstance(data.get("splits"), dict):
      data["splits"] = list(data["splits"].values())
      with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
  except (OSError, json.JSONDecodeError) as e:
    print(f"Warning: Could not patch splits in dataset_info.json: {e}")


def main() -> None:
  """CLI entry point for generating TFDS metadata."""
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--root",
      required=True,
      help="Root directory containing c4/en/<version> shards",
  )
  ap.add_argument(
      "--version",
      default="3.1.0",
      help="Target version to expose via TFDS",
  )
  ap.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing dataset_info.json if present",
  )
  args = ap.parse_args()

  # Use the version directory directly
  version_dir = os.path.join(args.root, "c4", "en", args.version)
  if not os.path.isdir(version_dir):
    raise FileNotFoundError(f"Version directory not found: {version_dir}")
  write_metadata(args.root, version_dir, args.version, force=args.force)
  print("Done.")


if __name__ == "__main__":
  main()
