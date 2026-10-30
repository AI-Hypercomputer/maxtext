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

"""5-key M3Batch contract, zero-copy 6-key iterator adapter, and pre-shift document bin-packing."""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Any, Literal, NotRequired, TypedDict

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

_CORE_6KEY_NAMES = frozenset(
    {
        "inputs",
        "targets",
        "inputs_position",
        "targets_position",
        "inputs_segmentation",
        "targets_segmentation",
        "positions",
        "segment_ids",
        "weights",
        "extras",
    }
)


class M3Batch(TypedDict):
  """Canonical 5-key batch schema (+ optional multimodal `extras`) for m3 trainers."""

  inputs: jax.Array
  targets: jax.Array
  positions: jax.Array
  segment_ids: jax.Array
  weights: jax.Array
  extras: NotRequired[dict[str, Any]]


def adapt_dict_to_m3_batch(raw: Mapping[str, Any]) -> M3Batch:
  """Zero-copy conversion from a 6-key MaxText batch (or M3Batch dict) into `M3Batch`.

  Reuses `inputs` and `targets` array buffers directly without copying token arrays, maps
  `inputs_position` -> `positions` and `inputs_segmentation` -> `segment_ids`, derives float32
  per-token `weights` from explicit `'weights'` or `targets_segmentation != 0`, and collects any
  additional multimodal keys into `extras`.
  """
  inputs = raw["inputs"]
  targets = raw["targets"]
  batch_shape = inputs.shape[:2]

  if "positions" in raw:
    positions = raw["positions"]
  elif "inputs_position" in raw:
    positions = raw["inputs_position"]
  else:
    seq_len = batch_shape[1]
    xp = jnp if isinstance(inputs, jax.Array) else np
    positions = xp.broadcast_to(xp.arange(seq_len, dtype=xp.int32)[None, :], batch_shape)

  if "segment_ids" in raw:
    segment_ids = raw["segment_ids"]
  elif "inputs_segmentation" in raw:
    segment_ids = raw["inputs_segmentation"]
  else:
    xp = jnp if isinstance(inputs, jax.Array) else np
    segment_ids = xp.ones(batch_shape, dtype=xp.int32)

  if "weights" in raw:
    xp = jnp if isinstance(raw["weights"], jax.Array) else np
    weights = raw["weights"].astype(xp.float32)
  elif "targets_segmentation" in raw:
    t_seg = raw["targets_segmentation"]
    xp = jnp if isinstance(t_seg, jax.Array) else np
    weights = (t_seg != 0).astype(xp.float32)
  else:
    xp = jnp if isinstance(segment_ids, jax.Array) else np
    weights = (segment_ids != 0).astype(xp.float32)

  batch: M3Batch = {
      "inputs": inputs,
      "targets": targets,
      "positions": positions,
      "segment_ids": segment_ids,
      "weights": weights,
  }
  extras: dict[str, Any] = {}
  if "extras" in raw and isinstance(raw["extras"], Mapping):
    extras.update(raw["extras"])
  for k, v in raw.items():
    if k not in _CORE_6KEY_NAMES:
      extras[k] = v
  if extras:
    batch["extras"] = extras
  return batch


class _AdaptedM3Iterator:
  """Zero-copy iterator wrapper forwarding stateful Grain checkpoint methods."""

  def __init__(self, inner: Any):
    self._inner = inner

  def __iter__(self) -> "_AdaptedM3Iterator":
    return self

  def __next__(self) -> M3Batch:
    return adapt_dict_to_m3_batch(next(self._inner))

  def get_state(self) -> Any:
    if hasattr(self._inner, "get_state"):
      return self._inner.get_state()
    raise AttributeError(f"Wrapped iterator {type(self._inner).__name__} does not implement `get_state()`.")

  def set_state(self, state: Any) -> None:
    if hasattr(self._inner, "set_state"):
      self._inner.set_state(state)
      return
    raise AttributeError(f"Wrapped iterator {type(self._inner).__name__} does not implement `set_state()`.")

  def save(self, *args: Any, **kwargs: Any) -> Any:
    return self._inner.save(*args, **kwargs)

  def restore(self, *args: Any, **kwargs: Any) -> Any:
    return self._inner.restore(*args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._inner, name)


def adapt_iterator_to_m3_batch(iterator: Iterator[Mapping[str, Any]]) -> _AdaptedM3Iterator:
  """Wraps any 6-key MaxText data iterator to yield 5-key `M3Batch` dicts without copying token buffers."""
  return _AdaptedM3Iterator(iterator)


@dataclasses.dataclass(frozen=True)
class PreparedDocument:
  """Pre-shifted single-document causal training example ready for deterministic bin-packing."""

  inputs: np.ndarray
  targets: np.ndarray
  positions: np.ndarray
  weights: np.ndarray
  extras: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

  @property
  def length(self) -> int:
    return int(self.inputs.shape[0])


def _compute_mrope_positions_1d_or_3d(
    input_tokens: np.ndarray,
    grid_thw: np.ndarray | None,
    image_token_id: int = 151655,
    spatial_merge_size: int = 2,
) -> np.ndarray:
  """Computes `[L_eff]` 1D positions, or `[L_eff, 3]` 3D M-RoPE `(t, h, w)` offsets when `grid_thw` is provided."""
  seq_len = int(input_tokens.shape[0])
  if grid_thw is None:
    return np.arange(seq_len, dtype=np.int32)

  grids = np.asarray(grid_thw, dtype=np.int32).reshape(-1, 3)
  positions = np.zeros((seq_len, 3), dtype=np.int32)
  pos_cursor = 0
  idx = 0
  grid_idx = 0
  while idx < seq_len:
    if input_tokens[idx] == image_token_id and grid_idx < len(grids):
      t_dim, h_dim, w_dim = (int(grids[grid_idx, 0]), int(grids[grid_idx, 1]), int(grids[grid_idx, 2]))
      h_merged = max(1, h_dim // spatial_merge_size)
      w_merged = max(1, w_dim // spatial_merge_size)
      num_vis = t_dim * h_merged * w_merged
      end_idx = min(seq_len, idx + num_vis)
      actual = end_idx - idx
      t_grid, h_grid, w_grid = np.meshgrid(
          np.arange(t_dim, dtype=np.int32),
          np.arange(h_merged, dtype=np.int32),
          np.arange(w_merged, dtype=np.int32),
          indexing="ij",
      )
      coords = np.stack([t_grid.ravel(), h_grid.ravel(), w_grid.ravel()], axis=-1)[:actual] + pos_cursor
      positions[idx:end_idx] = coords
      pos_cursor = int(coords[-1].max()) + 1 if actual > 0 else pos_cursor
      idx = end_idx
      grid_idx += 1
    else:
      positions[idx] = pos_cursor
      pos_cursor += 1
      idx += 1
  return positions


def prepare_document_for_packing(
    tokens: Sequence[int] | np.ndarray,
    loss_mask: Sequence[float] | np.ndarray | None = None,
    *,
    grid_thw: np.ndarray | None = None,
    image_token_id: int = 151655,
    spatial_merge_size: int = 2,
) -> PreparedDocument:
  """Slices intra-document causal `(inputs, targets, weights)` BEFORE bin-packing.

  Because `inputs = tokens[:-1]`, `targets = tokens[1:]`, and `weights = loss_mask[1:]` are computed
  on each individual unpadded document before packing:
  1. Adjacent packed documents can never leak tokens across document boundaries.
  2. Terminal `EOS` target tokens retain `weights == 1.0` even when `pad_id == eos_id`, while
     prompt/SFT prefix tokens (`loss_mask == 0.0`) remain strictly masked (`weights == 0.0`) and
     bins padded after packing always carry `segment_ids == 0` and `weights == 0.0`.
  """
  tok_arr = np.asarray(tokens, dtype=np.int32)
  if tok_arr.ndim != 1 or tok_arr.shape[0] < 2:
    raise ValueError(f"`tokens` must be a 1D sequence of at least 2 tokens; got shape {tok_arr.shape}.")

  inputs = tok_arr[:-1]
  targets = tok_arr[1:]
  if loss_mask is None:
    weights = np.ones_like(targets, dtype=np.float32)
  else:
    mask_arr = np.asarray(loss_mask, dtype=np.float32)
    if mask_arr.shape != tok_arr.shape:
      raise ValueError(f"`loss_mask` shape {mask_arr.shape} must match `tokens` shape {tok_arr.shape}.")
    weights = mask_arr[1:]

  positions = _compute_mrope_positions_1d_or_3d(
      inputs,
      grid_thw=grid_thw,
      image_token_id=image_token_id,
      spatial_merge_size=spatial_merge_size,
  )
  extras: dict[str, np.ndarray] = {}
  if grid_thw is not None:
    extras["grid_thw"] = np.asarray(grid_thw, dtype=np.int32)
  return PreparedDocument(inputs=inputs, targets=targets, positions=positions, weights=weights, extras=extras)


def pack_prepared_documents(
    documents: Sequence[PreparedDocument],
    seq_len: int,
    *,
    strategy: Literal["first_fit", "best_fit"] = "first_fit",
    pad_id: int = 0,
    max_segments_per_seq: int = -1,
) -> list[dict[str, np.ndarray]]:
  """Deterministically packs pre-shifted `PreparedDocument`s into fixed-length `seq_len` sequences.

  Documents exceeding `seq_len` are deterministically chunked into `<= seq_len` slices before bin
  assignment. Each packed document inside a bin receives a distinct positive 1-based `segment_id`
  (`1, 2, ...`), while trailing unfilled positions have `segment_ids == 0` and `weights == 0.0`.
  """
  if seq_len <= 0:
    raise ValueError(f"`seq_len` must be positive; got {seq_len}.")
  if strategy not in ("first_fit", "best_fit"):
    raise ValueError(f"Unsupported packing strategy {strategy!r}; expected 'first_fit' or 'best_fit'.")

  use_3d_pos = any(doc.positions.ndim == 2 for doc in documents)
  chunks: list[PreparedDocument] = []
  for doc in documents:
    for offset in range(0, doc.length, seq_len):
      end = min(doc.length, offset + seq_len)
      pos = doc.positions[offset:end]
      if use_3d_pos and pos.ndim == 1:
        pos = np.repeat(pos[:, None], 3, axis=-1)
      chunks.append(
          PreparedDocument(
              inputs=doc.inputs[offset:end],
              targets=doc.targets[offset:end],
              positions=pos,
              weights=doc.weights[offset:end],
              extras=doc.extras,
          )
      )

  bins: list[list[PreparedDocument]] = []
  bin_used: list[int] = []

  for chunk in chunks:
    length = chunk.length
    chosen_idx: int | None = None
    if strategy == "first_fit":
      for b_idx, used in enumerate(bin_used):
        seg_ok = max_segments_per_seq <= 0 or len(bins[b_idx]) < max_segments_per_seq
        if seg_ok and used + length <= seq_len:
          chosen_idx = b_idx
          break
    else:  # best_fit
      best_rem = seq_len + 1
      for b_idx, used in enumerate(bin_used):
        seg_ok = max_segments_per_seq <= 0 or len(bins[b_idx]) < max_segments_per_seq
        rem = seq_len - (used + length)
        if seg_ok and 0 <= rem < best_rem:
          best_rem = rem
          chosen_idx = b_idx

    if chosen_idx is None:
      bins.append([chunk])
      bin_used.append(length)
    else:
      bins[chosen_idx].append(chunk)
      bin_used[chosen_idx] += length

  packed_rows: list[dict[str, np.ndarray]] = []
  pos_shape = (seq_len, 3) if use_3d_pos else (seq_len,)
  for bin_docs in bins:
    inputs_row = np.full((seq_len,), pad_id, dtype=np.int32)
    targets_row = np.full((seq_len,), pad_id, dtype=np.int32)
    positions_row = np.zeros(pos_shape, dtype=np.int32)
    segment_ids_row = np.zeros((seq_len,), dtype=np.int32)
    weights_row = np.zeros((seq_len,), dtype=np.float32)

    cursor = 0
    for seg_idx, doc in enumerate(bin_docs, start=1):
      n = doc.length
      inputs_row[cursor : cursor + n] = doc.inputs
      targets_row[cursor : cursor + n] = doc.targets
      positions_row[cursor : cursor + n] = doc.positions
      segment_ids_row[cursor : cursor + n] = seg_idx
      weights_row[cursor : cursor + n] = doc.weights
      cursor += n

    packed_rows.append(
        {
            "inputs": inputs_row,
            "targets": targets_row,
            "positions": positions_row,
            "segment_ids": segment_ids_row,
            "weights": weights_row,
        }
    )
  return packed_rows


class M3PackedDataIterator:
  """Deterministic, host-checkpointable iterator yielding packed `M3Batch` batches."""

  def __init__(
      self,
      documents: Sequence[Sequence[int] | np.ndarray | Mapping[str, Any]],
      *,
      batch_size: int,
      seq_len: int,
      strategy: Literal["first_fit", "best_fit"] = "first_fit",
      pad_id: int = 0,
      seed: int = 0,
      repeat: bool = True,
      mesh: Mesh | None = None,
      batch_axes: tuple[str, ...] = ("data", "fsdp", "expert"),
  ):
    if batch_size <= 0:
      raise ValueError(f"`batch_size` must be positive; got {batch_size}.")
    self._raw_docs = list(documents)
    if not self._raw_docs:
      raise ValueError("`documents` must be non-empty.")
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.strategy = strategy
    self.pad_id = pad_id
    self.seed = seed
    self.repeat = repeat
    self._mesh = mesh
    self._batch_axes = batch_axes
    self._epoch = 0
    self._row_cursor = 0
    self._packed_rows = self._build_epoch_rows(self._epoch)

  def _prepare_one(self, item: Sequence[int] | np.ndarray | Mapping[str, Any]) -> PreparedDocument:
    if isinstance(item, Mapping):
      return prepare_document_for_packing(
          item["tokens"],
          loss_mask=item.get("loss_mask"),
          grid_thw=item.get("grid_thw"),
      )
    return prepare_document_for_packing(item)

  def _build_epoch_rows(self, epoch: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(self.seed + epoch)
    order = rng.permutation(len(self._raw_docs))
    prepared = [self._prepare_one(self._raw_docs[int(i)]) for i in order]
    return pack_prepared_documents(
        prepared,
        self.seq_len,
        strategy=self.strategy,
        pad_id=self.pad_id,
    )

  def _place_batch(self, stacked: dict[str, np.ndarray]) -> M3Batch:
    if self._mesh is None:
      return {k: jnp.asarray(v) for k, v in stacked.items()}  # type: ignore[return-value]
    live_axes = tuple(ax for ax in self._batch_axes if ax in self._mesh.axis_names)
    batch_spec = P(live_axes if len(live_axes) > 1 else (live_axes[0] if live_axes else None))
    sharding = NamedSharding(self._mesh, batch_spec)
    return {k: jax.device_put(v, sharding) for k, v in stacked.items()}  # type: ignore[return-value]

  def __iter__(self) -> "M3PackedDataIterator":
    return self

  def __next__(self) -> M3Batch:
    collected: list[dict[str, np.ndarray]] = []
    while len(collected) < self.batch_size:
      if self._row_cursor >= len(self._packed_rows):
        if not self.repeat:
          raise StopIteration
        self._epoch += 1
        self._row_cursor = 0
        self._packed_rows = self._build_epoch_rows(self._epoch)
      collected.append(self._packed_rows[self._row_cursor])
      self._row_cursor += 1

    keys = ("inputs", "targets", "positions", "segment_ids", "weights")
    stacked = {k: np.stack([row[k] for row in collected], axis=0) for k in keys}
    return self._place_batch(stacked)

  def get_state(self) -> dict[str, int]:
    """Returns exact deterministic epoch and row cursor state for Orbax / Grain checkpointing."""
    return {"epoch": int(self._epoch), "row_cursor": int(self._row_cursor), "seed": int(self.seed)}

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Restores exact iterator position so resumed steps yield bit-identical packed batches."""
    self.seed = int(state.get("seed", self.seed))
    self._epoch = int(state["epoch"])
    self._row_cursor = int(state["row_cursor"])
    self._packed_rows = self._build_epoch_rows(self._epoch)


class M3SyntheticDataIterator:
  """Deterministic checkpointable synthetic `M3Batch` generator for CPU/TPU runs and tests."""

  def __init__(
      self,
      *,
      batch_size: int,
      seq_len: int,
      vocab_size: int,
      seed: int = 0,
      mesh: Mesh | None = None,
      batch_axes: tuple[str, ...] = ("data", "fsdp", "expert"),
  ):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.vocab_size = vocab_size
    self.seed = seed
    self._step = 0
    self._mesh = mesh
    self._batch_axes = batch_axes

  def __iter__(self) -> "M3SyntheticDataIterator":
    return self

  def __next__(self) -> M3Batch:
    rng = np.random.default_rng(self.seed + self._step * 1009)
    self._step += 1
    tokens = rng.integers(1, max(2, self.vocab_size), size=(self.batch_size, self.seq_len + 1), dtype=np.int32)
    positions = np.broadcast_to(np.arange(self.seq_len, dtype=np.int32)[None, :], (self.batch_size, self.seq_len)).copy()
    segment_ids = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
    weights = np.ones((self.batch_size, self.seq_len), dtype=np.float32)
    raw = {
        "inputs": tokens[:, :-1],
        "targets": tokens[:, 1:],
        "positions": positions,
        "segment_ids": segment_ids,
        "weights": weights,
    }
    if self._mesh is None:
      return {k: jnp.asarray(v) for k, v in raw.items()}  # type: ignore[return-value]
    live_axes = tuple(ax for ax in self._batch_axes if ax in self._mesh.axis_names)
    sharding = NamedSharding(self._mesh, P(live_axes if len(live_axes) > 1 else (live_axes[0] if live_axes else None)))
    return {k: jax.device_put(v, sharding) for k, v in raw.items()}  # type: ignore[return-value]

  def get_state(self) -> dict[str, int]:
    return {"step": int(self._step), "seed": int(self.seed)}

  def set_state(self, state: Mapping[str, Any]) -> None:
    self.seed = int(state.get("seed", self.seed))
    self._step = int(state["step"])
