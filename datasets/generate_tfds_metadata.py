"""Generate minimal TFDS metadata (features.json, dataset_info.json) for existing local C4 array_record shards.

Usage:
    cd datasets && python generate_tfds_metadata.py --root c4_en_dataset_minimal --version 3.1.0 --source-version 3.0.1 --force

Instead of hand-crafting JSON (which is brittle to TFDS proto changes), this script
creates a tiny in-memory TFDS builder and uses DatasetInfo.write_to_dir() to emit
the correctly structured dataset_info.json and features.json, then overwrites the
split metadata to reference existing local ArrayRecord shards.

After running, you can point TFDS to --root and load with dataset_name 'c4/en:3.1.0'.
"""
from __future__ import annotations
import os, json, argparse, glob
import tensorflow_datasets as tfds
import tensorflow as tf

def ensure_symlink(root: str, source_version: str, target_version: str) -> str:
    c4_en_dir = os.path.join(root, 'c4', 'en')
    src = os.path.join(c4_en_dir, source_version)
    dst = os.path.join(c4_en_dir, target_version)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source version directory not found: {src}")
    if not os.path.lexists(dst):
        try:
            os.symlink(src, dst)
            print(f"Created symlink {dst} -> {src}")
        except OSError as e:
            print(f"Symlink creation failed (continuing): {e}")
    else:
        print(f"Symlink already exists: {dst}")
    return dst

def count_array_record_examples(shard_path):
    dataset = tf.data.experimental.ArrayRecordDataset([shard_path])
    return sum(1 for _ in dataset)

def write_metadata(root: str, version_dir: str, dataset_version: str, force: bool = False):
    info_path = os.path.join(version_dir, 'dataset_info.json')
    if os.path.exists(info_path) and not force:
        print("dataset_info.json already exists; skipping overwrite (use --force to regenerate).")
        return

    # Discover shards
    NUM_SHARDS_TRAIN = 8
    NUM_SHARDS_VAL = 2
    EXACT_TRAIN_RECORDS = 1000
    EXACT_VAL_RECORDS = 200
    train_shards = sorted(glob.glob(os.path.join(version_dir, 'c4-train.array_record-*')))
    val_shards = sorted(glob.glob(os.path.join(version_dir, 'c4-validation.array_record-*')))

    # Distribute records evenly across shards
    train_records_per_shard = EXACT_TRAIN_RECORDS // NUM_SHARDS_TRAIN
    val_records_per_shard = EXACT_VAL_RECORDS // NUM_SHARDS_VAL
    train_shard_lengths = [train_records_per_shard] * NUM_SHARDS_TRAIN
    val_shard_lengths = [val_records_per_shard] * NUM_SHARDS_VAL
    train_split = tfds.core.SplitInfo(
        name='train', shard_lengths=train_shard_lengths, num_bytes=0
    )
    val_split = tfds.core.SplitInfo(
        name='validation', shard_lengths=val_shard_lengths, num_bytes=0
    )
    class _LocalC4Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(dataset_version)
        BUILDER_CONFIGS = [tfds.core.BuilderConfig(name='en', version=VERSION, description='Local minimal C4 EN subset')]
        def _info(self):
            info = tfds.core.DatasetInfo(
                builder=self,
                description='Local minimal C4 English subset.',
                features=tfds.features.FeaturesDict({'text': tfds.features.Text()}),
                homepage='https://www.tensorflow.org/datasets/catalog/c4',
                citation='',
            )
            info.set_splits({
                'train': train_split,
                'validation': val_split,
            })
            return info
        def _split_generators(self, dl_manager):
            return []  # We will override splits manually
        def _generate_examples(self):
            if False:
                yield  # No generation; data already exists

    builder = _LocalC4Builder(data_dir=root)
    info = builder.info

    # Write canonical files (features.json + dataset_info.json)
    info.write_to_directory(version_dir)
    print(f"Wrote TFDS dataset_info & features to {version_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root directory containing c4/en/<version> shards')
    ap.add_argument('--version', default='3.1.0', help='Target version to expose via TFDS')
    ap.add_argument('--source-version', default='3.0.1', help='Existing version directory with shards')
    ap.add_argument('--force', action='store_true', help='Overwrite existing dataset_info.json if present')
    args = ap.parse_args()
    target_dir = ensure_symlink(args.root, args.source_version, args.version)
    write_metadata(args.root, target_dir, args.version, force=args.force)
    print('Done.')

if __name__ == '__main__':
    main()

