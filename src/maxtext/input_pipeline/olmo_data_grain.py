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

"""OLMo numpy fixed-seq-length dataset on top of Grain.

A Grain ``RandomAccessDataSource`` over the AI2 OLMo virtual token stream
plus a deterministic global-shuffle sampler. See
``docs/guides/data_input_pipeline/olmo_grain.md`` for an overview.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, List, Optional

import numpy as np

import grain

from maxtext.input_pipeline.olmo_data import (
    OlmoNpyIndex,
    global_to_local,
    is_clean_instance,
)


class OlmoNpyDataSource(grain.sources.RandomAccessDataSource):
  """Random-access view of an OLMo numpy mix as a stream of token windows.

  Files are opened lazily and cached as ``np.memmap`` per worker. Open mmaps
  are reference-counted by :class:`_MmapCache` so we don't blow past
  ``ulimit -n`` when iterating over the full 950-file mix.

  The data source is **process-safe**: every Grain worker subprocess builds
  its own ``_MmapCache`` after the fork. No shared mutable state.

  Args:
    index: The :class:`OlmoNpyIndex` describing the mix. Path strings must be
      reachable from the data-loading host (typically a GCSFUSE mount path
      like ``/mnt/<your-mount>/...``).
    path_remap: Optional dict to rewrite ``index.files[i].path``. Useful when
      the index was built with ``gs://`` paths and you want to read from a
      gcsfuse mount, or vice versa. A path is rewritten if it starts with
      any key in this dict.
    max_open_files: Soft cap on the number of mmaps held open in the
      per-worker cache. The cache is LRU.
  """

  def __init__(
      self,
      index: OlmoNpyIndex,
      *,
      path_remap: Optional[Dict[str, str]] = None,
      max_open_files: int = 64,
  ):
    self._index = index
    self._dtype = np.dtype(index.dtype)
    self._sequence_length = index.sequence_length
    self._path_remap = dict(path_remap or {})
    self._mmaps = _MmapCache(max_open_files=max_open_files)

  # ---- Grain's RandomAccessDataSource interface --------------------------- #

  def __len__(self) -> int:
    return self._index.total_instances

  def __getitem__(self, instance_id: int) -> Dict[str, Any]:
    file_idx, token_offset = global_to_local(self._index, instance_id)
    file_entry = self._index.files[file_idx]
    arr = self._mmaps.get(self._resolve_path(file_entry.path), self._dtype)
    # Always copy: the memmap is opened read-only (mode="r"), and we need to
    # hand a writable, picklable array back to Grain so transforms can
    # mutate it freely and worker processes can serialize it without
    # dragging the memmap object along.
    tokens = np.array(arr[token_offset : token_offset + self._sequence_length], copy=True)
    return {
        "tokens": tokens,
        "instance_id": int(instance_id),
        "file_id": int(file_idx),
    }

  # ---- Helpers ------------------------------------------------------------ #

  def _resolve_path(self, path: str) -> str:
    for prefix, replacement in self._path_remap.items():
      if path.startswith(prefix):
        return replacement + path[len(prefix) :]
    return path

  def __getstate__(self):
    # Mmap caches don't survive pickling; rebuild after unpickle.
    state = self.__dict__.copy()
    state["_mmaps"] = None
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._mmaps = _MmapCache(max_open_files=64)

  def __repr__(self) -> str:
    """Stable repr — Grain compares ``repr(data_source)`` between the
    checkpoint and the live source on resume. The default repr embeds the
    object id, which breaks resume across separate construction. We hash
    the index fingerprint + seq + path-remap instead so equivalent sources
    compare equal as strings."""
    return (
        f"OlmoNpyDataSource(fingerprint={self._index.fingerprint!r}, "
        f"seq={self._sequence_length}, dtype={self._dtype.str!r}, "
        f"remap={sorted(self._path_remap.items())!r})"
    )


class _MmapCache:
  """Tiny LRU-ish cache of open ``np.memmap`` handles."""

  def __init__(self, max_open_files: int = 64):
    self._max = max_open_files
    self._mmaps: Dict[str, np.memmap] = {}
    self._lock = threading.Lock()

  def get(self, path: str, dtype: np.dtype) -> np.memmap:
    """Return a cached ``np.memmap`` for ``path``, opening it lazily."""
    with self._lock:
      arr = self._mmaps.get(path)
      if arr is not None:
        # touch: re-insert at end for LRU ordering.
        del self._mmaps[path]
        self._mmaps[path] = arr
        return arr
      # Open the file as 1-D memmap; for raw .npy (no header) we need to know
      # length, but np.memmap can derive it from file size when shape=(-1,).
      arr = np.memmap(path, dtype=dtype, mode="r")
      self._mmaps[path] = arr
      while len(self._mmaps) > self._max:
        # Evict oldest.
        oldest = next(iter(self._mmaps))
        del self._mmaps[oldest]
      return arr


class OlmoIndexSampler:
  """Global-shuffle sampler over an OLMo numpy mix.

  Mirrors OLMo-core's :class:`NumpyDataLoaderBase` shuffle math: a single
  Fisher-Yates over ``[0, total_instances)`` keyed by ``hash(seed, epoch)``,
  then partitioned across ``shard_count`` hosts.

  Implements Grain's ``Sampler`` protocol — i.e. ``__getitem__`` returning
  :class:`grain.python.RecordMetadata`. Grain calls
  ``sampler[index]`` for each global step; the sampler is responsible for
  mapping that to the actual record_key fed to ``data_source[record_key]``.

  Indexing semantics:

  * ``index`` here is a *per-host* (per-data-loader) global step counter
    starting at 0 and advancing without bound (we support infinite epochs).
  * ``epoch = index // num_local_instances_per_epoch`` selects which
    permutation to use; ``in_epoch = index % num_local_instances_per_epoch``
    selects the position within this host's shard of that permutation.

  Checkpointing is trivial: the only mutable state is "which epoch's
  permutation is currently cached" (a perf optimization). The user-visible
  position is just the index passed to ``__getitem__``.

  Args:
    total_instances: ``index.total_instances`` from the OLMo index.
    seed: Base seed for the shuffle.
    shard_index: Zero-based index of this data-loading host. Typically
      ``jax.process_index()``.
    shard_count: Number of data-loading hosts. Typically
      ``jax.process_count()``.
    shuffle: If ``False``, instances are emitted in linear order — useful
      for debugging.
    initial_step: Per-host batch step at which the *training run* should
      resume. ``__getitem__(local_idx)`` returns the record at absolute
      position ``local_idx + initial_step``. Use this to resume a run from
      a saved trainer step without saving Grain's iterator state — our
      sampler is a pure function of its inputs, so the (seed, shard,
      absolute step) tuple fully determines the next record.
  """

  def __init__(
      self,
      *,
      total_instances: int,
      seed: int,
      shard_index: int = 0,
      shard_count: int = 1,
      shuffle: bool = True,
      initial_step: int = 0,
  ):
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
      raise ValueError(f"Invalid shard config: shard_index={shard_index} of {shard_count}")
    if total_instances <= 0:
      raise ValueError(f"total_instances must be positive, got {total_instances}")
    if initial_step < 0:
      raise ValueError(f"initial_step must be non-negative, got {initial_step}")
    self._total = int(total_instances)
    self._seed = int(seed)
    self._shard_index = int(shard_index)
    self._shard_count = int(shard_count)
    self._shuffle = bool(shuffle)
    self._initial_step = int(initial_step)
    # Cache the shuffled-and-sharded indices for the most-recently-touched
    # epoch. Cheap to recompute on epoch boundaries; expensive to keep many
    # epochs resident at once for the full 724 M-instance mix.
    self._cached_epoch: Optional[int] = None
    self._cached_shard_indices: Optional[np.ndarray] = None
    self._cache_lock = threading.Lock()

  # ---- Public API --------------------------------------------------------- #

  @property
  def num_instances(self) -> int:
    return self._total

  @property
  def num_local_instances_per_epoch(self) -> int:
    """Instances assigned to *this* host per epoch (drops trailing remainder)."""
    return self._total // self._shard_count

  def shuffled_global_indices(self, *, seed: int, epoch: int) -> np.ndarray:
    """Build the full shuffled list for ``(seed, epoch)``.

    For the production 724 M-instance mix this allocates ~5.8 GB at uint64
    (numpy's default for ``permutation``). For production we should swap to
    an on-disk memmap scheme like olmo-core's
    ``build_and_save_global_indices``. Sized for unit tests + the initial
    smoke training run for now.
    """
    if not self._shuffle:
      return np.arange(self._total, dtype=np.uint64)
    rng = np.random.default_rng(_combine_seed_epoch(seed, epoch))
    order = rng.permutation(self._total)
    return order.astype(np.uint64, copy=False)

  def shard_indices(self, *, seed: int, epoch: int) -> np.ndarray:
    """Slice the global shuffled order down to this host's share."""
    full = self.shuffled_global_indices(seed=seed, epoch=epoch)
    n_per = self.num_local_instances_per_epoch
    start = self._shard_index * n_per
    end = start + n_per
    return full[start:end]

  def _shard_indices_for_epoch(self, epoch: int) -> np.ndarray:
    with self._cache_lock:
      if self._cached_epoch == epoch and self._cached_shard_indices is not None:
        return self._cached_shard_indices
      shard = self.shard_indices(seed=self._seed, epoch=epoch)
      self._cached_epoch = epoch
      self._cached_shard_indices = shard
      return shard

  def __getstate__(self):
    # threading.Lock can't be pickled, and the per-epoch cache is a pure perf
    # optimization — drop both before serialization to forked Grain workers.
    state = self.__dict__.copy()
    state["_cache_lock"] = None
    state["_cached_epoch"] = None
    state["_cached_shard_indices"] = None
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._cache_lock = threading.Lock()

  # ---- Sampler protocol --------------------------------------------------- #

  def __getitem__(self, index: int) -> grain.RecordMetadata:
    """Map a per-host global step ``index`` to the next record to fetch.

    The lookup applies ``initial_step`` as a transparent offset: the caller
    sees a fresh stream starting at index 0, but the underlying record
    pointer is at absolute position ``index + initial_step``. That's the
    mechanism that lets resume work without persisting any iterator state.
    """
    if index < 0:
      raise IndexError(f"sampler index must be non-negative, got {index}")
    n_per = self.num_local_instances_per_epoch
    if n_per == 0:
      raise IndexError(
          f"No instances assigned to shard {self._shard_index}/{self._shard_count} " f"(total_instances={self._total})"
      )
    absolute = index + self._initial_step
    epoch = absolute // n_per
    in_epoch = absolute % n_per
    shard = self._shard_indices_for_epoch(epoch)
    record_key = int(shard[in_epoch])
    return grain.RecordMetadata(index=index, record_key=record_key)

  # Grain >=0.2.16 expects either a finite ``__len__`` or that the sampler
  # raises ``IndexError`` on out-of-bounds. We support infinite training and
  # never raise IndexError for non-negative indices, so we omit ``__len__``.

  def __repr__(self) -> str:
    """Stable repr — Grain compares ``repr(sampler)`` between the checkpoint
    and the live sampler to validate the sampler is unchanged on resume.

    We deliberately **exclude** ``initial_step`` from the repr: a sampler
    rebuilt with a different ``initial_step`` produces a different absolute
    position via offset arithmetic, but it's still the *same logical sampler*
    over the same data. Including the step here would break interop with
    Grain's iterator-state checkpointing path (different reprs reject each
    other). The repr captures only the immutable config that defines the
    sample space; the offset is just a starting cursor.
    """
    return (
        f"OlmoIndexSampler(total_instances={self._total}, seed={self._seed}, "
        f"shard_index={self._shard_index}, shard_count={self._shard_count}, "
        f"shuffle={self._shuffle})"
    )


def _combine_seed_epoch(seed: int, epoch: int) -> int:
  """Stable 64-bit mix of (seed, epoch) for the per-epoch shuffle RNG.

  Uses SHA-256 truncated to 64 bits — no fixed points (unlike a raw multiply
  by a constant when seed=epoch=0), and avoids the numpy uint64 multiplication
  overflow warnings that dog SplitMix-style mixers in pure numpy.
  """
  digest = hashlib.sha256(f"olmo-shuffle:{int(seed)}:{int(epoch)}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "little")


class NgramFilterTransform(grain.transforms.Map):
  """Add an ``instance_mask`` field per OLMo-core's repetition filter.

  ``instance_mask = True`` if the instance is "clean" (kept fully in the
  loss); ``False`` if it has too-repetitive periodic spans (zero-out at
  loss time). We don't drop the instance — that would mess with sharding —
  matching OLMo-core's behavior.
  """

  def __init__(
      self,
      *,
      max_period: int = 13,
      min_period: int = 1,
      max_count: int = 32,
      mask_value: int = -1,
  ):
    self._max_period = int(max_period)
    self._min_period = int(min_period)
    self._max_count = int(max_count)
    self._mask_value = int(mask_value)

  def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``instance_mask`` to ``element`` based on the n-gram filter."""
    tokens = element["tokens"]
    clean = is_clean_instance(
        tokens,
        repetition_max_period=self._max_period,
        repetition_min_period=self._min_period,
        repetition_max_count=self._max_count,
        mask_value=self._mask_value,
    )
    out = dict(element)
    out["instance_mask"] = bool(clean)
    return out


class ShiftToInputsTargets(grain.transforms.Map):
  """Convert a ``tokens`` array into the keys MaxText's pretrain trainer expects.

  Produces, for a single instance of length ``L = sequence_length``:

  * ``inputs``: ``tokens.astype(int32)``, shape ``(L,)``
  * ``targets``: ``tokens`` shifted left by one, padded with 0 at position
    ``L-1``, shape ``(L,)``
  * ``inputs_position``: ``[0, 1, ..., L-1]`` int32
  * ``inputs_segmentation``: ``int32`` ones, shape ``(L,)`` — single segment
  * ``targets_segmentation``: ``int32`` ones, shape ``(L,)`` with the last
    position zeroed (loss masked at the padded position); the entire row is
    zero if ``instance_mask`` is False (n-gram filter flagged the instance).

  Outputs are the full ``L`` tokens (not ``L-1``) because the TPU splash
  attention kernel requires ``q_seq_len`` divisible by 512; producing length
  ``L-1`` would break that invariant for typical OLMo ``L=8192``.

  The OLMo dataset has no document boundaries inside an instance — sequences
  span doc boundaries with no special masking — so ``segmentation`` and
  ``position`` are trivially uniform within an instance.
  """

  def map(self, element: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``tokens`` into ``inputs`` / ``targets`` / segmentation tensors."""
    tokens = element["tokens"].astype(np.int32, copy=False)
    L = tokens.shape[0]  # == sequence_length from the index
    instance_mask = bool(element.get("instance_mask", True))
    seg_value = np.int32(1) if instance_mask else np.int32(0)

    # Output rank-2 (batch, seq) tensors of length L (= max_target_length).
    # The TPU splash-attention kernel requires q_seq_len to be divisible by
    # 512, which means the trainer-side seq length must be the full L —
    # using ``tokens[:-1]`` (length L-1) breaks that invariant.
    #
    # For next-token prediction we still want ``targets[i] = tokens[i+1]``,
    # so we shift and pad the last position with 0 then *mask it out* via
    # ``targets_segmentation[L-1] = 0``. The trainer's segmentation-aware
    # loss skips positions where targets_segmentation == 0, so the padded
    # last token contributes nothing to the loss. Information loss is
    # 1 token per ``L``-token instance (~0.012% at L=8192).
    inputs = tokens
    targets = np.empty(L, dtype=np.int32)
    targets[:-1] = tokens[1:]
    targets[-1] = 0  # pad; loss masked below

    targets_seg = np.full(L, seg_value, dtype=np.int32)
    targets_seg[-1] = 0  # never compute loss on the boundary position

    return {
        "inputs": inputs,
        "targets": targets,
        "inputs_position": np.arange(L, dtype=np.int32),
        "inputs_segmentation": np.ones(L, dtype=np.int32),
        "targets_segmentation": targets_seg,
    }


def make_olmo_grain_data_loader(
    index: OlmoNpyIndex,
    *,
    seed: int,
    batch_size: int,
    shard_index: int,
    shard_count: int,
    apply_ngram_filter: bool = True,
    shift_to_inputs_targets: bool = True,
    path_remap: Optional[Dict[str, str]] = None,
    grain_worker_count: int = 0,
    grain_worker_buffer_size: int = 1,
    initial_step: int = 0,
):
  """Build a Grain ``DataLoader`` for OLMo-style fixed-seq-length training.

  Args:
    index: Loaded :class:`OlmoNpyIndex`.
    seed: Shuffle seed (paired with the implicit per-step ``epoch =
      step // n_per_host`` to drive the per-epoch permutation).
    batch_size: Per-host batch size (i.e. global_batch / shard_count).
    shard_index: This host's data-loading rank.
    shard_count: Total data-loading hosts.
    apply_ngram_filter: Add :class:`NgramFilterTransform` (recommended).
    shift_to_inputs_targets: Add :class:`ShiftToInputsTargets` so the loader
      yields the ``inputs``/``targets`` shape MaxText's trainer expects.
    path_remap: Pass-through to :class:`OlmoNpyDataSource`.
    grain_worker_count: ``0`` runs in-process; otherwise Grain forks workers.
    grain_worker_buffer_size: Per-worker batch prefetch.
    initial_step: Start the *underlying sampler* at this absolute step.
      The Grain DataLoader still iterates from its own 0, but every record
      lookup is shifted by ``initial_step``. Set this to ``train_step *
      batch_size`` on resume to pick up the data stream where it left off
      *without* needing Grain's iterator-state checkpointing.

  Returns:
    A ``grain.DataLoader``.
  """
  source = OlmoNpyDataSource(index, path_remap=path_remap)
  sampler = OlmoIndexSampler(
      total_instances=index.total_instances,
      seed=seed,
      shard_index=shard_index,
      shard_count=shard_count,
      initial_step=initial_step,
  )

  ops: List[Any] = []
  if apply_ngram_filter:
    ops.append(NgramFilterTransform())
  if shift_to_inputs_targets:
    ops.append(ShiftToInputsTargets())
  ops.append(grain.transforms.Batch(batch_size=batch_size, drop_remainder=True))

  # Grain expects ``shard_options`` on the DataLoader (sharding used to live
  # on the Sampler). Our sampler already does the shard-by-rank slicing, but
  # Grain still requires this object to validate checkpoint compatibility.
  shard_options = grain.sharding.ShardOptions(shard_index=shard_index, shard_count=shard_count, drop_remainder=True)
  return grain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=ops,
      shard_options=shard_options,
      worker_count=grain_worker_count,
      worker_buffer_size=grain_worker_buffer_size,
  )
