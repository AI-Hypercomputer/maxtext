#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a HuggingFace dataset to MaxText Grain ArrayRecord format.

Flattens ShareGPT conversations into plain text with role markers,
serializes as tf.train.Example protos (using raw protobuf, no TF dependency),
and writes sharded ArrayRecord files.

Dependencies: datasets, array-record, protobuf (no tensorflow required).

Usage:
    python prepare_data.py \
        --dataset hiyouga/glaive-function-calling-v2-sharegpt \
        --output gs://my-bucket/data/glaive-fc-v2 \
        --num-shards 64

    # Local output (then upload manually):
    python prepare_data.py --output /tmp/glaive-data --num-shards 8
"""

import argparse
import os
import subprocess
import tempfile

from datasets import load_dataset, concatenate_datasets
from array_record.python.array_record_module import ArrayRecordWriter


# Protobuf serialization (wire-compatible with tf.train.Example)
# MaxText's ParseFeatures calls example_pb2.Example().ParseFromString(element),
# which expects the standard tf.train.Example proto wire format. We encode it
# directly with raw protobuf, no TensorFlow or compiled proto files needed.

def _varint_bytes(value):
    """Encode an integer as a protobuf varint."""
    pieces = []
    while value > 0x7F:
        pieces.append((value & 0x7F) | 0x80)
        value >>= 7
    pieces.append(value & 0x7F)
    return bytes(pieces)


def _length_delimited(field_number, data):
    """Encode a length-delimited protobuf field."""
    tag = _varint_bytes((field_number << 3) | 2)
    return tag + _varint_bytes(len(data)) + data


def make_example(text: str) -> bytes:
    """Serialize text as a tf.train.Example proto.

    Wire format:
        Example { features { feature { key: "text", value { bytes_list { value: [text] } } } } }
    """
    text_bytes = text.encode("utf-8")
    # BytesList: field 1 = bytes value
    bytes_list = _length_delimited(1, text_bytes)
    # Feature: field 1 = bytes_list (oneof)
    feature_value = _length_delimited(1, bytes_list)
    # Features.FeatureEntry (map): field 1 = key (string), field 2 = value (Feature)
    key_field = _length_delimited(1, b"text")
    value_field = _length_delimited(2, feature_value)
    feature_entry = key_field + value_field
    # Features: field 1 = feature (map entries)
    features = _length_delimited(1, feature_entry)
    # Example: field 1 = features
    example = _length_delimited(1, features)
    return example


# ShareGPT role -> role marker
ROLE_MAP = {
    "system": "<|system|>",
    "human": "<|user|>",
    "user": "<|user|>",
    "gpt": "<|assistant|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "function": "<|tool|>",
    "function_call": "<|tool_call|>",
    "observation": "<|observation|>",
}


def flatten_conversation(conversations: list[dict]) -> str:
    """Flatten a ShareGPT conversation into a single text string with role markers.

    Args:
        conversations: List of {"from": role, "value": text} dicts.

    Returns:
        Flattened text with role markers.
    """
    parts = []
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        marker = ROLE_MAP.get(role, f"<|{role}|>")
        parts.append(f"{marker}\n{value}")
    return "\n".join(parts)


def write_arrayrecord_shard(records: list[bytes], output_path: str):
    """Write serialized records to a single ArrayRecord file."""
    writer = ArrayRecordWriter(output_path, "group_size:1")
    for record in records:
        writer.write(record)
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to MaxText Grain ArrayRecord format"
    )
    parser.add_argument(
        "--dataset",
        default="hiyouga/glaive-function-calling-v2-sharegpt",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Dataset config names to load (default: all configs)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (local dir or gs:// URI)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of ArrayRecord shards to create",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max examples to process (for testing)",
    )
    args = parser.parse_args()

    # Determine output location
    is_gcs = args.output.startswith("gs://")
    if is_gcs:
        local_dir = tempfile.mkdtemp(prefix="maxtext-data-")
        gcs_dir = args.output.rstrip("/")
    else:
        local_dir = args.output
        os.makedirs(local_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.configs:
        datasets_list = []
        for config in args.configs:
            print(f"  Loading config: {config}")
            ds = load_dataset(args.dataset, config, split="train")
            datasets_list.append(ds)
        dataset = concatenate_datasets(datasets_list)
    else:
        dataset = load_dataset(args.dataset, split="train")

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"  Total examples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")

    # Determine the conversation column
    conv_column = None
    for candidate in ("conversations", "messages", "text"):
        if candidate in dataset.column_names:
            conv_column = candidate
            break

    if conv_column is None:
        raise ValueError(
            f"No conversation column found. Available: {dataset.column_names}"
        )
    print(f"  Using column: {conv_column}")

    # Preview first example
    first = dataset[0]
    if conv_column in ("conversations", "messages"):
        sample_text = flatten_conversation(first[conv_column])
    else:
        sample_text = first[conv_column]
    print(f"\n  Preview (first 500 chars):\n  {'─' * 60}")
    for line in sample_text[:500].split("\n"):
        print(f"  {line}")
    print(f"  {'─' * 60}\n")

    # Convert and shard
    examples_per_shard = len(dataset) // args.num_shards
    remainder = len(dataset) % args.num_shards

    total_written = 0
    shard_paths = []

    for shard_idx in range(args.num_shards):
        start = shard_idx * examples_per_shard + min(shard_idx, remainder)
        end = start + examples_per_shard + (1 if shard_idx < remainder else 0)

        shard_name = f"train.array_record-{shard_idx:05d}-of-{args.num_shards:05d}"
        shard_path = os.path.join(local_dir, shard_name)
        shard_paths.append(shard_path)

        records = []
        for i in range(start, end):
            example = dataset[i]
            if conv_column in ("conversations", "messages"):
                text = flatten_conversation(example[conv_column])
            else:
                text = example[conv_column]

            if text.strip():
                records.append(make_example(text))

        write_arrayrecord_shard(records, shard_path)
        total_written += len(records)
        print(
            f"  Shard {shard_idx + 1}/{args.num_shards}: "
            f"{len(records)} records -> {shard_name}"
        )

    print(f"\nTotal records written: {total_written}")

    # Upload to GCS if needed
    if is_gcs:
        print(f"\nUploading to {gcs_dir}/")
        for shard_path in shard_paths:
            shard_name = os.path.basename(shard_path)
            gcs_path = f"{gcs_dir}/{shard_name}"
            subprocess.run(
                ["gcloud", "storage", "cp", shard_path, gcs_path],
                check=True,
            )
            print(f"  Uploaded {shard_name}")

        # Clean up temp dir
        import shutil
        shutil.rmtree(local_dir)
        print(f"\nDone. Use in MaxText config:")
        print(f"  grain_train_files={gcs_dir}/train.array_record*")
    else:
        print(f"\nDone. Files in: {local_dir}")
        print(f"  grain_train_files={local_dir}/train.array_record*")


if __name__ == "__main__":
    main()
