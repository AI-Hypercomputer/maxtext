"""Megatron-compatible weighted blending for Grain map datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from maxtext.input_pipeline import _mmap_index_utils


logger = logging.getLogger(__name__)

_DATASET_INDEX_SUFFIX = "dataset_index.npy"
_DATASET_SAMPLE_INDEX_SUFFIX = "dataset_sample_index.npy"
_RECOVERABLE_CACHE_ERRORS = (FileNotFoundError, OSError, ValueError, EOFError)


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: np.ndarray,
    num_datasets: int,
    size: int,
) -> None:
  """Populate blend indices using Megatron's greedy error minimization.

  This is a direct Python implementation of
  ``megatron/core/datasets/helpers.cpp::build_blending_indices``.  In
  particular, ``np.argmax`` keeps the first (lowest dataset ID) on ties.
  """
  if size < 0:
    raise ValueError(f"size must be non-negative, got {size}")
  if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
    raise ValueError("Blend index arrays must both have shape (size,)")
  if num_datasets <= 0:
    raise ValueError(f"num_datasets must be positive, got {num_datasets}")

  weights = np.asarray(weights, dtype=np.float64)
  if weights.shape != (num_datasets,) or np.any(weights <= 0):
    raise ValueError("weights must contain one positive value per dataset")
  if not np.isclose(weights.sum(), 1.0, rtol=1e-4, atol=1e-6):
    raise ValueError(f"weights must sum to 1, got {weights.tolist()}")

  current_samples = np.zeros(num_datasets, dtype=np.int64)
  for sample_id in range(size):
    errors = weights * max(float(sample_id), 1.0) - current_samples
    dataset_id = int(np.argmax(errors))
    dataset_index[sample_id] = dataset_id
    dataset_sample_index[sample_id] = current_samples[dataset_id]
    current_samples[dataset_id] += 1


def _normalize_datasets(
    map_datasets: Sequence,
    weights: Sequence[float],
    dataset_lengths: Sequence[int] | None = None,
):
  if len(map_datasets) != len(weights) or not map_datasets:
    raise ValueError("At least one dataset and one corresponding weight are required")
  if dataset_lengths is not None and len(dataset_lengths) != len(map_datasets):
    raise ValueError("dataset_lengths must match map_datasets before zero-weight filtering")
  weights = np.asarray(weights, dtype=np.float64)
  if np.any(weights < 0) or weights.sum() <= 0:
    raise ValueError(f"weights must be non-negative with a positive total, got {weights.tolist()}")
  keep = weights > 0
  datasets = [dataset for dataset, keep_dataset in zip(map_datasets, keep) if keep_dataset]
  lengths = (
      [int(length) for length, keep_dataset in zip(dataset_lengths, keep) if keep_dataset]
      if dataset_lengths is not None
      else None
  )
  weights = weights[keep]
  return datasets, weights / weights.sum(), lengths


def _infer_size(weights: np.ndarray, lengths: Sequence[int]) -> int:
  if len(lengths) != len(weights) or any(length <= 0 for length in lengths):
    raise ValueError(f"Dataset lengths must be positive and match weights, got {lengths}")
  size = int(math.floor(min(length / weight for length, weight in zip(lengths, weights))))
  if size <= 0:
    raise ValueError("Blended dataset would be empty")
  return size


def _cache_paths(cache_dir: str, weights: np.ndarray, lengths: Sequence[int], size: int, split: str):
  payload = json.dumps(
      {"weights": weights.tolist(), "lengths": list(lengths), "size": size, "split": split},
      sort_keys=True,
      separators=(",", ":"),
  )
  key = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
  root = Path(cache_dir)
  return root / f"{key}-{_DATASET_INDEX_SUFFIX}", root / f"{key}-{_DATASET_SAMPLE_INDEX_SUFFIX}"


def _validate_indices(dataset_index, dataset_sample_index, lengths, size):
  """Reject stale or corrupted blend indices before they reach Grain workers."""
  if dataset_index.ndim != 1 or dataset_sample_index.ndim != 1:
    raise ValueError("Blend index arrays must be one-dimensional")
  if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
    raise ValueError("Cached blend indices have an unexpected size")
  if not np.issubdtype(dataset_index.dtype, np.integer) or not np.issubdtype(dataset_sample_index.dtype, np.integer):
    raise ValueError("Blend index arrays must have integer dtypes")
  if np.any(dataset_index < 0) or np.any(dataset_index >= len(lengths)):
    raise ValueError("Cached blend dataset_index is out of range")
  for dataset_id, length in enumerate(lengths):
    used = dataset_sample_index[dataset_index == dataset_id]
    if not used.size:
      continue
    if np.any(used < 0) or int(np.max(used)) >= length:
      raise ValueError(f"Cached blend indices exceed dataset {dataset_id} length")
    if not np.array_equal(used, np.arange(used.size, dtype=used.dtype)):
      raise ValueError(f"Cached blend sample indices are not contiguous for dataset {dataset_id}")


def _find_index_pair_in_dir(index_dir: str, split: str) -> tuple[Path, Path]:
  """Find either the documented fixed pair or Megatron-style split-specific files."""
  root = Path(index_dir)
  direct_pair = root / _DATASET_INDEX_SUFFIX, root / _DATASET_SAMPLE_INDEX_SUFFIX
  if direct_pair[0].is_file() and direct_pair[1].is_file():
    return direct_pair
  if not root.is_dir():
    raise FileNotFoundError(f"Blend index directory does not exist: {index_dir}")

  pairs = []
  for dataset_path in sorted(root.glob(f"*{_DATASET_INDEX_SUFFIX}")):
    sample_path = dataset_path.with_name(dataset_path.name.replace(_DATASET_INDEX_SUFFIX, _DATASET_SAMPLE_INDEX_SUFFIX))
    if sample_path.is_file():
      pairs.append((dataset_path, sample_path))
  split_pairs = [pair for pair in pairs if f"-BlendedDataset-{split}-" in pair[0].name]
  if len(split_pairs) == 1:
    return split_pairs[0]
  if len(split_pairs) > 1:
    raise ValueError(f"Multiple blend index pairs match split '{split}' in {index_dir}")
  if len(pairs) == 1:
    return pairs[0]
  raise FileNotFoundError(f"Could not find a blend index pair in {index_dir}")


def build_and_save_blend_indices(
    output_dir: str,
    weights: Sequence[float],
    dataset_lengths: Sequence[int],
    size: int,
) -> dict[str, Path]:
  """Write the fixed-name blend pair accepted by ``blend_index_dir`` at runtime.

  ``weights`` are normalized once here, exactly as ``MegatronBlendedDataSource``
  does when it receives the already-normalized mixture weights from the input
  parser.  This keeps an offline pair bit-identical to a runtime-built pair.
  """
  _, normalized_weights, normalized_lengths = _normalize_datasets([object()] * len(weights), weights, dataset_lengths)
  if normalized_lengths is None:
    raise ValueError("dataset_lengths are required to build blend indices")
  if size <= 0:
    raise ValueError(f"size must be positive, got {size}")
  dataset_index = np.zeros(size, dtype=np.int16)
  dataset_sample_index = np.zeros(size, dtype=np.int64)
  build_blending_indices(dataset_index, dataset_sample_index, normalized_weights, len(normalized_lengths), size)
  _validate_indices(dataset_index, dataset_sample_index, normalized_lengths, size)
  root = Path(output_dir)
  root.mkdir(parents=True, exist_ok=True)
  paths = {
      "dataset_index": root / _DATASET_INDEX_SUFFIX,
      "dataset_sample_index": root / _DATASET_SAMPLE_INDEX_SUFFIX,
  }
  _mmap_index_utils.save_npy_atomic(paths["dataset_index"], dataset_index)
  _mmap_index_utils.save_npy_atomic(paths["dataset_sample_index"], dataset_sample_index)
  return paths


class MegatronBlendedDataSource:
  """Random-access blend whose global order is identical to Megatron's."""

  def __init__(
      self,
      map_datasets: Sequence,
      weights: Sequence[float],
      size: int | None = None,
      dataset_lengths: Sequence[int] | None = None,
      cache_dir: str | None = None,
      blend_index_dir: str | None = None,
      split: str = "train",
  ):
    self._datasets, self._weights, filtered_lengths = _normalize_datasets(map_datasets, weights, dataset_lengths)
    self._lengths = filtered_lengths if filtered_lengths is not None else [len(dataset) for dataset in self._datasets]
    self._size = _infer_size(self._weights, self._lengths) if size is None else int(size)
    if self._size <= 0:
      raise ValueError(f"size must be positive, got {self._size}")

    self._dataset_index, self._dataset_sample_index = self._load_or_build(cache_dir, blend_index_dir, split)

  def _load_or_build(self, cache_dir, blend_index_dir, split):
    candidates = []
    if blend_index_dir:
      try:
        candidates.append(_find_index_pair_in_dir(blend_index_dir, split))
      except _RECOVERABLE_CACHE_ERRORS as error:
        logger.warning("Could not discover prebuilt blend indices in %s: %s", blend_index_dir, error)
    if cache_dir:
      candidates.append(_cache_paths(cache_dir, self._weights, self._lengths, self._size, split))

    for dataset_path, sample_path in candidates:
      try:
        dataset_index = np.load(dataset_path, allow_pickle=False, mmap_mode="r")
        dataset_sample_index = np.load(sample_path, allow_pickle=False, mmap_mode="r")
        _validate_indices(dataset_index, dataset_sample_index, self._lengths, self._size)
        return dataset_index, dataset_sample_index
      except _RECOVERABLE_CACHE_ERRORS as error:
        logger.warning("Ignoring invalid blend indices %s / %s: %s", dataset_path, sample_path, error)
        continue

    dataset_index = np.zeros(self._size, dtype=np.int16)
    dataset_sample_index = np.zeros(self._size, dtype=np.int64)
    build_blending_indices(dataset_index, dataset_sample_index, self._weights, len(self._datasets), self._size)
    _validate_indices(dataset_index, dataset_sample_index, self._lengths, self._size)

    if cache_dir and _mmap_index_utils.is_primary_process():
      dataset_path, sample_path = _cache_paths(cache_dir, self._weights, self._lengths, self._size, split)
      try:
        os.makedirs(cache_dir, exist_ok=True)
        _mmap_index_utils.save_npy_atomic(dataset_path, dataset_index)
        _mmap_index_utils.save_npy_atomic(sample_path, dataset_sample_index)
      except OSError as error:
        logger.warning("Failed to save blend cache in %s; continuing with in-memory indices: %s", cache_dir, error)
    return dataset_index, dataset_sample_index

  def __len__(self):
    return self._size

  def __getitem__(self, idx):
    if idx < 0:
      idx += self._size
    if idx < 0 or idx >= self._size:
      raise IndexError(f"Index {idx} out of range for blend of size {self._size}")
    return self._datasets[int(self._dataset_index[idx])][int(self._dataset_sample_index[idx])]
