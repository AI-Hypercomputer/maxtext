"""Precompute Megatron-LM index .npy files from .idx/.bin datasets.

Generates document_index.npy, sample_index.npy, and shuffle_index.npy
that can be loaded by MegatronNpyDataSource at training time, skipping
the index-building step at training initialization.

Usage::

    # Single dataset
    python tools/data_processing/mmap_index_builder.py \
        --input /data/megatron_dataset/ \
        --output-dir /cache/indices/ \
        --seq-length 2048 --num-samples 1000000

    # Blending multiple datasets (concurrent build)
    python tools/data_processing/mmap_index_builder.py blend \
        --datasets '/data/ds_a,0.7;/data/ds_b,0.3' \
        --output-dir /cache/blend_indices \
        --seq-length 2048 --total-samples 1000000
"""

import logging
import os
import sys

# When run directly (python tools/data_processing/mmap_index_builder.py), ensure
# the src directory is on sys.path so the maxtext package can be resolved.
_src_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _src_dir not in sys.path:
  sys.path.insert(0, _src_dir)

# Re-export all public symbols from the canonical location so that
# existing ``from tools.data_processing.mmap_index_builder import ...``
# statements continue to work without changes.
from maxtext.input_pipeline._mmap_index_utils import (  # pylint: disable=unused-import  # noqa: F401 -- public re-exports
    build_document_index,
    build_indices,
    build_sample_index,
    build_shuffle_index,
    compute_index_hash,
    compute_num_epochs,
    convert,
    convert_blend,
    get_document_sizes,
    resolve_shard_prefixes,
    should_separate_last_epoch,
)

log = logging.getLogger(__name__)

# Backwards-compatible alias: old callers use ``discover_shards``
discover_shards = resolve_shard_prefixes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_convert_args(parser):
  """Add arguments for the single-dataset convert mode."""
  parser.add_argument(
      "--input",
      required=True,
      nargs="+",
      dest="input_paths",
      help="Path prefix(es) or directory(ies) containing .idx/.bin files.",
  )
  parser.add_argument("--output-dir", required=True, help="Directory to write .npy output files.")
  parser.add_argument("--seq-length", required=True, type=int, help="Sequence length for sample construction.")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--num-samples", type=int, default=None, help="Total number of training samples to generate.")
  group.add_argument("--num-epochs", type=int, default=None, help="Number of epochs to generate samples for.")
  parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
  parser.add_argument("--split", type=str, default=None, help="Comma-separated split ratios (e.g. '0.9,0.05,0.05').")
  parser.add_argument("--split-index", type=int, default=0, help="Index into the split ratios.")
  parser.add_argument("--add-extra-token", type=int, default=1, help="Extra token for next-token prediction (default 1).")


def _add_blend_args(parser):
  """Add arguments for the blend mode."""
  parser.add_argument(
      "--datasets",
      required=True,
      type=str,
      help="Semicolon-separated dataset specs: 'input_path,weight;input_path2,weight2;...'.",
  )
  parser.add_argument(
      "--output-dir",
      required=True,
      help=(
          "Root directory for .npy output (sub-dirs created per dataset); also "
          "contains dataset_index.npy and dataset_sample_index.npy for blend_index_dir."
      ),
  )
  parser.add_argument("--seq-length", required=True, type=int, help="Sequence length.")
  parser.add_argument(
      "--total-samples", required=True, type=int, help="Total training samples (= train_steps x global_batch_size)."
  )
  parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
  parser.add_argument(
      "--margin", type=float, default=0.5, help="Overprovisioning margin %% (default 0.5, matching Megatron)."
  )
  parser.add_argument(
      "--max-workers", type=int, default=None, help="Max concurrent builds (default: number of datasets)."
  )
  parser.add_argument("--split", type=str, default=None, help="Comma-separated split ratios.")
  parser.add_argument("--split-index", type=int, default=0, help="Index into the split ratios.")
  parser.add_argument("--add-extra-token", type=int, default=1, help="Extra token (default 1).")


def _run_convert(args):
  """Execute the 'convert' subcommand."""
  paths = convert(
      input_paths=args.input_paths,
      output_dir=args.output_dir,
      seq_length=args.seq_length,
      num_samples=args.num_samples,
      num_epochs=args.num_epochs,
      seed=args.seed,
      split=args.split,
      split_index=args.split_index,
      add_extra_token=args.add_extra_token,
  )
  for name, path in paths.items():
    log.info("Wrote %s -> %s", name, path)


def _run_blend(args):
  """Execute the 'blend' subcommand."""
  specs_raw = args.datasets.strip().split(";")
  dataset_specs = []
  for i, raw in enumerate(specs_raw):
    raw = raw.strip()
    if not raw:
      continue
    parts = raw.rsplit(",", 1)
    if len(parts) != 2:
      raise ValueError(f"Invalid dataset spec '{raw}': expected 'input_path,weight'")
    input_path, weight_str = parts[0].strip(), parts[1].strip()
    dataset_specs.append(
        {
            "input": [input_path],
            "weight": float(weight_str),
            "output_dir": os.path.join(args.output_dir, f"dataset_{i}"),
        }
    )

  results = convert_blend(
      dataset_specs=dataset_specs,
      total_samples=args.total_samples,
      seq_length=args.seq_length,
      seed=args.seed,
      margin=args.margin,
      max_workers=args.max_workers,
      split=args.split,
      split_index=args.split_index,
      add_extra_token=args.add_extra_token,
      blend_index_output_dir=args.output_dir,
  )
  for i, r in enumerate(results):
    log.info("Dataset %d: buffer_samples=%d", i, r["buffer_samples"])
    for name, path in r["paths"].items():
      log.info("  %s -> %s", name, path)
  log.info("Blend dispatch -> %s", os.path.join(args.output_dir, "dataset_index.npy"))
  log.info("Blend dispatch -> %s", os.path.join(args.output_dir, "dataset_sample_index.npy"))


def main():
  """CLI entry point with subcommands for single-dataset and blend modes."""
  import argparse  # pylint: disable=import-outside-toplevel

  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  # Check if the first positional arg is a known subcommand.
  # If not, fall back to the legacy flat-argument interface so that
  # existing scripts using ``--input ... --output-dir ...`` keep working.
  subcommands = {"convert", "blend"}
  if len(sys.argv) > 1 and sys.argv[1] in subcommands:
    parser = argparse.ArgumentParser(
        description="Precompute Megatron-LM index .npy files from .idx/.bin datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert", help="Build indices for a single dataset.")
    _add_convert_args(convert_parser)

    blend_parser = subparsers.add_parser("blend", help="Build indices for a blended mixture of datasets.")
    _add_blend_args(blend_parser)

    args = parser.parse_args()
    if args.command == "blend":
      _run_blend(args)
    else:
      _run_convert(args)
  else:
    # Legacy mode: flat argparse without subcommands
    parser = argparse.ArgumentParser(
        description="Precompute Megatron-LM index .npy files from .idx/.bin datasets.",
    )
    _add_convert_args(parser)
    _run_convert(parser.parse_args())


if __name__ == "__main__":
  main()
