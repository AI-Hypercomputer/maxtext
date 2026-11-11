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

"""MaxText input pipeline adapter for OLMo-core multimodal mid-training mixes (``dataset_type=olmo_mm``).

Supports two interoperable execution modes with identical batch tensor semantics:

1. **Live ``OLMo-core`` ``MixtureDataLoader`` mode (``olmo_mm_config_path``)**:
   Deserializes an uploaded ``OLMo-core`` experiment ``config.json`` (e.g. ``bridge_8k_config.json``,
   ``joint_8k_config.json``, or ``mt_lc_256k_config.json``), remaps storage prefixes
   (``olmo_path_remap_from`` -> ``olmo_path_remap_to``), registers the ``merged_mt_lc`` composable
   text-replay source classes when needed, and yields per-host sharded batches directly in MaxText's
   ``data`` dictionary format.

2. **Pre-exported ``.npz`` batch mode (``olmo_mm_batches_dir``)**:
   Reads pre-baked global batches (``batch_00000.npz``, ...) from local disk or GCS (``gs://...``)
   and deterministically slices each data-loading host's shard, enabling TPU pod validation without
   requiring ``OLMo-core`` or raw media mounts on every worker.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import functools as ft
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import types
from typing import Any, List, Optional, Tuple
import zipfile

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as np_format

from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline.olmo_grain_data_processing import _detect_resumed_step
from maxtext.utils import max_logging


def remap_paths_in_obj(obj: Any, remap_from: str, remap_to: str) -> Any:
  """Recursively rewrite string prefixes ``remap_from`` -> ``remap_to`` inside nested config dicts/lists."""
  if not remap_from or not remap_to:
    return obj
  if isinstance(obj, str):
    if obj.startswith(remap_from):
      return remap_to + obj[len(remap_from) :]
    return obj
  if isinstance(obj, list):
    return [remap_paths_in_obj(v, remap_from, remap_to) for v in obj]
  if isinstance(obj, tuple):
    return tuple(remap_paths_in_obj(v, remap_from, remap_to) for v in obj)
  if isinstance(obj, dict):
    return {k: remap_paths_in_obj(v, remap_from, remap_to) for k, v in obj.items()}
  return obj


_ACTIVE_PATH_REMAPS: list[tuple[str, str]] = []
_PATH_REMAP_HOOKS_INSTALLED: bool = False


def _translate_remapped_path(p: Any) -> Any:
  if not _ACTIVE_PATH_REMAPS:
    return p
  if isinstance(p, str):
    for src_prefix, dst_prefix in _ACTIVE_PATH_REMAPS:
      if p.startswith(src_prefix):
        return dst_prefix + p[len(src_prefix) :]
    return p
  if isinstance(p, os.PathLike):
    s = os.fspath(p)
    if isinstance(s, str):
      for src_prefix, dst_prefix in _ACTIVE_PATH_REMAPS:
        if s.startswith(src_prefix):
          translated = dst_prefix + s[len(src_prefix) :]
          return type(p)(translated) if isinstance(p, Path) else translated
  return p


def install_path_remap_prefix(remap_from: str, remap_to: str) -> None:
  """Install an idempotent runtime prefix translator for ``OLMo-core`` dataset path lookups.

  ``OLMo-core`` multimodal datasets reference ``/weka/...`` both via module-level constants in
  ``olmo_core.data.multimodal.paths`` and via literal string file paths stored inside Arrow rows.
  When ``remap_from != remap_to`` (for instance ``/weka`` -> a local ``gcsfuse`` virtual root),
  this helper updates environment defaults, updates ``olmo_core.data.multimodal.paths`` if loaded,
  and wraps file-open/stat helpers so inner row paths resolve identically without mutating data.
  """
  global _PATH_REMAP_HOOKS_INSTALLED
  src = remap_from.rstrip("/")
  dst = remap_to.rstrip("/")
  if not src or not dst or src == dst:
    return

  for idx, (existing_src, _) in enumerate(_ACTIVE_PATH_REMAPS):
    if existing_src == src:
      _ACTIVE_PATH_REMAPS[idx] = (src, dst)
      break
  else:
    _ACTIVE_PATH_REMAPS.append((src, dst))

  default_env_paths = {
      "MOLMO_DATA_DIR": "/weka/oe-training-default/mm-olmo",
      "OE_ENCODER_DATA_DIR": "/weka/oe-training-default/oe-encoder",
      "TEXT_RICH_CAPTION_DIR": "/weka/oe-training-default/mm-olmo/molmo3_datasets/text_rich_caption",
      "PIXMO_POINTS_V2_DIR": "/weka/oe-training-default/chrisc/pixmo-points-with-masks-v17",
      "NVIDIA_SYNTH_OCR_DIR": "/weka/oe-training-default/jasonr/dataset/ocr/nvidia-ocr-synthetic-multilingual-v1",
      "SYNTH_RECEIPTS_OCR_DIR": "/weka/oe-training-default/jasonr/dataset/ocr/synthetic-receipts-ocr",
      "NEMOTRON_IMAGE_DIR": "/weka/oe-training-default/jasonr/dataset/nemotron-image",
  }
  for env_key, default_val in default_env_paths.items():
    cur_val = os.environ.get(env_key, default_val)
    if cur_val.startswith(src):
      os.environ[env_key] = dst + cur_val[len(src) :]

  paths_mod = sys.modules.get("olmo_core.data.multimodal.paths")
  if paths_mod is not None:
    for attr in dir(paths_mod):
      val = getattr(paths_mod, attr, None)
      if isinstance(val, str) and val.startswith(src):
        setattr(paths_mod, attr, dst + val[len(src) :])

  if _PATH_REMAP_HOOKS_INSTALLED:
    return

  orig_builtins_open = builtins.open

  @ft.wraps(orig_builtins_open)
  def _wrapped_open(file, *args, **kwargs):
    return orig_builtins_open(_translate_remapped_path(file), *args, **kwargs)

  builtins.open = _wrapped_open  # type: ignore[assignment]

  for fn_name in ("exists", "isfile", "isdir", "islink", "realpath", "abspath"):
    orig_fn = getattr(os.path, fn_name)

    @ft.wraps(orig_fn)
    def _wrapped_path_fn(p, *args, _orig=orig_fn, **kwargs):
      return _orig(_translate_remapped_path(p), *args, **kwargs)

    setattr(os.path, fn_name, _wrapped_path_fn)

  for fn_name in ("stat", "lstat", "listdir", "scandir", "open"):
    orig_os_fn = getattr(os, fn_name)

    @ft.wraps(orig_os_fn)
    def _wrapped_os_fn(p, *args, _orig=orig_os_fn, **kwargs):
      return _orig(_translate_remapped_path(p), *args, **kwargs)

    setattr(os, fn_name, _wrapped_os_fn)

  orig_glob = glob.glob

  @ft.wraps(orig_glob)
  def _wrapped_glob(pathname, *args, **kwargs):
    return orig_glob(_translate_remapped_path(pathname), *args, **kwargs)

  glob.glob = _wrapped_glob  # type: ignore[assignment]

  try:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    orig_img_open = Image.open

    @ft.wraps(orig_img_open)
    def _wrapped_img_open(fp, *args, **kwargs):
      return orig_img_open(_translate_remapped_path(fp), *args, **kwargs)

    Image.open = _wrapped_img_open  # type: ignore[assignment]
  except ImportError:
    pass

  _PATH_REMAP_HOOKS_INSTALLED = True


def _set_work_dirs_recursive(obj: Any, work_dir: str) -> Any:
  """Rewrite ``work_dir`` entries inside nested OLMo-core dataset config dicts to ``work_dir``."""
  if isinstance(obj, list):
    return [_set_work_dirs_recursive(v, work_dir) for v in obj]
  if isinstance(obj, dict):
    out = {}
    for k, v in obj.items():
      if k == "work_dir" and isinstance(v, str):
        out[k] = work_dir
      else:
        out[k] = _set_work_dirs_recursive(v, work_dir)
    return out
  return obj


def register_merged_mt_lc_shim() -> None:
  """Register ``merged_mt_lc`` classes in ``sys.modules`` if not already importable.

  Ai2's Stage 3 (``mt_lc_256k_config.json``) serializes two recipe-local ``InstanceSourceConfig``
  subclasses (``merged_mt_lc.LegacyDataStreamInstanceSourceConfig`` and
  ``merged_mt_lc.ProportionalInterleavedInstanceSourceConfig``) inside
  ``dataset.sources['text_midtraining'].source``. Registering compatible implementations allows
  ``MixtureDataLoaderConfig.from_dict(...)`` and ``ComposableTextReplayConfig.build(...)`` to
  instantiate the exact interleaved 8K-chunked Dolmino + 256K Longmino stream out of the box.
  """
  if "merged_mt_lc" in sys.modules:
    return
  try:
    __import__("merged_mt_lc")
    return
  except ImportError:
    pass

  from olmo_core.aliases import PathOrStr  # pylint: disable=import-outside-toplevel
  from olmo_core.data.composable.instance_source import (  # pylint: disable=import-outside-toplevel
      Instance,
      InstanceSource,
      InstanceSourceConfig,
  )
  from olmo_core.data.data_loader import NumpyDataLoaderConfig  # pylint: disable=import-outside-toplevel
  from olmo_core.data.numpy_dataset import NumpyDatasetConfig  # pylint: disable=import-outside-toplevel

  mod = types.ModuleType("merged_mt_lc")

  class LegacyDataStreamInstanceSource(InstanceSource):
    """Wraps a legacy shuffled ``NumpyDatasetConfig`` stream as a fixed-length ``InstanceSource``."""

    def __init__(
        self,
        *,
        dataset_cfg: NumpyDatasetConfig,
        data_loader_seed: int,
        legacy_global_batch_size: int,
        output_sequence_length: int,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
      super().__init__(
          work_dir=work_dir,
          sequence_length=int(output_sequence_length),
          max_sequence_length=int(output_sequence_length),
          label=label,
      )
      ds_cfg = dataset_cfg.copy()
      ds_cfg.work_dir = str(work_dir)
      self._dataset = ds_cfg.build()
      self._dataset.prepare()
      self._data_loader_seed = int(data_loader_seed)
      self._legacy_global_batch_size = int(legacy_global_batch_size)
      loader = NumpyDataLoaderConfig(
          global_batch_size=self._legacy_global_batch_size,
          seed=self._data_loader_seed,
          work_dir=str(work_dir),
      ).build(self._dataset)
      loader.reshuffle(epoch=1, in_memory=False)
      self._global_indices = loader.get_global_indices()
      in_seq = int(self._dataset.sequence_length)
      out_seq = int(output_sequence_length)
      if out_seq % in_seq != 0:
        raise ValueError(f"output_sequence_length={out_seq} must be a multiple of dataset.sequence_length={in_seq}")
      self._chunks_per_instance = out_seq // in_seq
      self._length = len(self._global_indices) // self._chunks_per_instance

    @ft.cached_property
    def fingerprint(self) -> str:
      h = hashlib.sha256()
      h.update(
          json.dumps(
              [
                  "LegacyDataStreamInstanceSource",
                  self._dataset.fingerprint,
                  self._data_loader_seed,
                  self._legacy_global_batch_size,
                  self.sequence_length,
              ],
              sort_keys=True,
          ).encode("utf-8")
      )
      return h.hexdigest()

    def __len__(self) -> int:
      return self._length

    def __getitem__(self, idx: int) -> Instance:
      idx = self.validate_index(idx)
      k = self._chunks_per_instance
      if k == 1:
        item = self._dataset[int(self._global_indices[idx])]
        out: dict[str, Any] = {"input_ids": np.asarray(item["input_ids"], dtype=np.int64)}
        if "label_mask" in item:
          out["label_mask"] = np.asarray(item["label_mask"], dtype=np.bool_)
        if "instance_mask" in item:
          out["instance_mask"] = bool(item["instance_mask"])
        return out  # type: ignore[return-value]

      chunk_len = int(self._dataset.sequence_length)
      tokens = np.empty(self.sequence_length, dtype=np.int64)
      label_mask = np.ones(self.sequence_length, dtype=np.bool_)
      extra_denom = 0
      base = idx * k
      for c in range(k):
        item = self._dataset[int(self._global_indices[base + c])]
        sl = slice(c * chunk_len, (c + 1) * chunk_len)
        tokens[sl] = np.asarray(item["input_ids"], dtype=np.int64)
        if c > 0:
          # Never predict token 0 of chunk c from the final token of unrelated chunk c - 1.
          label_mask[c * chunk_len] = False
        if "label_mask" in item:
          label_mask[sl] &= np.asarray(item["label_mask"], dtype=np.bool_)
        if not bool(item.get("instance_mask", True)):
          # Filtered 8K chunk contributes zero CE loss while keeping its (chunk_len - 1) denominator mass.
          active_in_chunk = int(label_mask[c * chunk_len + 1 : (c + 1) * chunk_len].sum())
          label_mask[sl] = False
          extra_denom += active_in_chunk
      return {  # type: ignore[return-value]
          "input_ids": tokens,
          "label_mask": label_mask,
          "loss_denominator_extra_tokens": extra_denom,
      }

    def children(self):
      return ()

  @dataclass
  class LegacyDataStreamInstanceSourceConfig(InstanceSourceConfig):
    dataset: NumpyDatasetConfig
    data_loader_seed: int
    legacy_global_batch_size: int
    output_sequence_length: int
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> LegacyDataStreamInstanceSource:
      return LegacyDataStreamInstanceSource(
          dataset_cfg=self.dataset,
          data_loader_seed=self.data_loader_seed,
          legacy_global_batch_size=self.legacy_global_batch_size,
          output_sequence_length=self.output_sequence_length,
          work_dir=work_dir,
          label=self.label,
      )

  class ProportionalInterleavedInstanceSource(InstanceSource):
    """Interleaves child ``InstanceSource``s proportionally to their lengths and supports ``.locate(idx)``."""

    def __init__(
        self,
        *sources: InstanceSource,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
      if not sources:
        raise ValueError("ProportionalInterleavedInstanceSource requires at least one source.")
      seq_len = sources[0].sequence_length
      max_seq_len = sources[0].max_sequence_length
      for s in sources:
        if s.sequence_length != seq_len or s.max_sequence_length != max_seq_len:
          raise ValueError("All interleaved sources must share sequence_length and max_sequence_length.")
      super().__init__(
          work_dir=work_dir,
          sequence_length=seq_len,
          max_sequence_length=max_seq_len,
          label=label,
      )
      self._sources = tuple(sources)
      lengths = [len(s) for s in self._sources]
      self._total_len = sum(lengths)
      # Build exact deterministic proportional interleave schedule via largest-remainder / Bresenham deficit.
      src_ids = np.empty(self._total_len, dtype=np.int32)
      local_ids = np.empty(self._total_len, dtype=np.int64)
      emitted = [0] * len(lengths)
      acc = [0] * len(lengths)
      n_src = len(lengths)
      total = self._total_len
      for pos in range(total):
        best_s = -1
        best_val = -1
        for s_idx in range(n_src):
          if emitted[s_idx] >= lengths[s_idx]:
            continue
          acc[s_idx] += lengths[s_idx]
          if acc[s_idx] > best_val:
            best_val = acc[s_idx]
            best_s = s_idx
        acc[best_s] -= total
        src_ids[pos] = best_s
        local_ids[pos] = emitted[best_s]
        emitted[best_s] += 1
      self._src_ids = src_ids
      self._local_ids = local_ids

    @property
    def sources(self) -> Tuple[InstanceSource, ...]:
      return self._sources

    @ft.cached_property
    def fingerprint(self) -> str:
      h = hashlib.sha256()
      h.update(b"class=ProportionalInterleavedInstanceSource,")
      for s in self._sources:
        h.update(f"source={s.fingerprint},".encode("utf-8"))
      return h.hexdigest()

    def __len__(self) -> int:
      return self._total_len

    def locate(self, idx: int) -> Tuple[int, int]:
      idx = self.validate_index(idx)
      return int(self._src_ids[idx]), int(self._local_ids[idx])

    def __getitem__(self, idx: int) -> Instance:
      s_idx, l_idx = self.locate(idx)
      return self._sources[s_idx][l_idx]

    def children(self):
      return self._sources

  @dataclass
  class ProportionalInterleavedInstanceSourceConfig(InstanceSourceConfig):
    sources: List[InstanceSourceConfig]
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> ProportionalInterleavedInstanceSource:
      return ProportionalInterleavedInstanceSource(
          *[s.build(work_dir=work_dir) for s in self.sources],
          work_dir=work_dir,
          label=self.label,
      )

  LegacyDataStreamInstanceSource.Config = LegacyDataStreamInstanceSourceConfig
  ProportionalInterleavedInstanceSource.Config = ProportionalInterleavedInstanceSourceConfig

  LegacyDataStreamInstanceSourceConfig.__annotations__ = {
      "dataset": NumpyDatasetConfig,
      "data_loader_seed": int,
      "legacy_global_batch_size": int,
      "output_sequence_length": int,
      "label": Optional[str],
  }
  ProportionalInterleavedInstanceSourceConfig.__annotations__ = {
      "sources": List[InstanceSourceConfig],
      "label": Optional[str],
  }
  for cls in (
      LegacyDataStreamInstanceSource,
      LegacyDataStreamInstanceSourceConfig,
      ProportionalInterleavedInstanceSource,
      ProportionalInterleavedInstanceSourceConfig,
  ):
    cls.__module__ = "merged_mt_lc"

  mod.PathOrStr = PathOrStr  # type: ignore[attr-defined]
  mod.Instance = Instance  # type: ignore[attr-defined]
  mod.InstanceSource = InstanceSource  # type: ignore[attr-defined]
  mod.InstanceSourceConfig = InstanceSourceConfig  # type: ignore[attr-defined]
  mod.NumpyDatasetConfig = NumpyDatasetConfig  # type: ignore[attr-defined]
  mod.Optional = Optional  # type: ignore[attr-defined]
  mod.List = List  # type: ignore[attr-defined]
  mod.LegacyDataStreamInstanceSource = LegacyDataStreamInstanceSource  # type: ignore[attr-defined]
  mod.LegacyDataStreamInstanceSourceConfig = LegacyDataStreamInstanceSourceConfig  # type: ignore[attr-defined]
  mod.ProportionalInterleavedInstanceSource = ProportionalInterleavedInstanceSource  # type: ignore[attr-defined]
  mod.ProportionalInterleavedInstanceSourceConfig = ProportionalInterleavedInstanceSourceConfig  # type: ignore[attr-defined]
  sys.modules["merged_mt_lc"] = mod


def _tensor_to_numpy_mm(arr: Any, dtype_mm: Any) -> np.ndarray:
  """Convert a PyTorch or NumPy image tensor (including ``torch.bfloat16``) to ``jnp.dtype(dtype_mm)``."""
  target_dtype = jnp.dtype(dtype_mm)
  if hasattr(arr, "detach"):
    import torch  # pylint: disable=import-outside-toplevel

    t = arr.detach().cpu()
    if t.dtype == torch.bfloat16:
      np_bf16 = t.contiguous().view(torch.uint16).numpy().view(jnp.dtype("bfloat16"))
      return np_bf16.astype(target_dtype, copy=False)
    return t.numpy().astype(target_dtype, copy=False)
  return np.asarray(arr, dtype=target_dtype)


def _tensor_to_numpy(arr: Any, dtype: Any) -> np.ndarray:
  if hasattr(arr, "detach"):
    return arr.detach().cpu().numpy().astype(dtype, copy=False)
  return np.asarray(arr, dtype=dtype)


def olmo_core_batch_to_maxtext_batch(
    batch: dict[str, Any],
    *,
    max_target_length: int,
    max_crops_per_seq: int,
    max_pooled_patches_per_seq: int,
    n_patches_per_crop: int = 729,
    patch_dim: int = 588,
    pool_size: int = 4,
    dtype_mm: Any = jnp.bfloat16,
    reset_positions_on_segment: bool = True,
    padding_input_token_id: int = -1,
    pad_token_id: int = 100277,
    use_loss_weights: bool = True,
    include_router_token_mask: bool = True,
) -> dict[str, np.ndarray]:
  """Convert an ``OLMo-core`` multimodal collator/loader batch into an exact MaxText training batch.

  Preserves exact parity with ``MultimodalOLMoDDPTrainModule`` and ``MultimodalOLMoDDPModel``:
  - ``labels`` from ``MultimodalCollator`` / ``ComposableTextReplayDataset`` are already pre-shifted
    for next-token prediction (we do NOT shift ``input_ids`` a second time).
  - ``router_token_mask`` marks non-padding tokens (``True`` for real tokens, ``False`` for right-padding).
    When ``padding_input_token_id >= 0`` (e.g. ``100257`` in ``mt_lc_256k``), padding positions in
    ``inputs`` are replaced with ``padding_input_token_id``.
  - ``inputs_segmentation`` assigns contiguous run IDs ``1, 2, ...`` over ``document_ids`` (if present),
    else ``example_ids`` (if present), else ``where(router_token_mask, 0, -1)``. Trailing ``-1`` padding
    runs receive their own isolated positive segment ID so self-attention never produces all-masked NaNs
    while staying isolated from real documents.
  - ``targets_segmentation`` zeroes ignored positions (``labels < 0``), right-padding (``~router_token_mask``),
    and padding documents (``raw_doc_ids < 0``).
  - ``loss_weights`` preserves positive ``loss_masks`` on all valid non-padding positions (including
    repetition-filtered ``labels == -100, loss_masks == 1.0`` tokens so the global CE denominator matches
    ``(loss_masks * (loss_masks > 0)).sum()`` in ``OLMo-core``).
  """
  raw_in = _tensor_to_numpy(batch["input_ids"], np.int32)
  raw_labels = _tensor_to_numpy(batch["labels"], np.int64)
  bsz, seq_in = raw_in.shape
  target_seq = int(max_target_length)
  if seq_in > target_seq:
    raise ValueError(f"Batch sequence length {seq_in} exceeds config.max_target_length={target_seq}.")

  if "router_token_mask" in batch and batch["router_token_mask"] is not None:
    raw_rtm = _tensor_to_numpy(batch["router_token_mask"], np.bool_)
  else:
    raw_rtm = np.ones((bsz, seq_in), dtype=np.bool_)

  if batch.get("document_ids") is not None:
    raw_doc = _tensor_to_numpy(batch["document_ids"], np.int64)
  elif batch.get("example_ids") is not None:
    raw_doc = _tensor_to_numpy(batch["example_ids"], np.int64)
  else:
    raw_doc = np.where(raw_rtm, 0, -1).astype(np.int64)

  if batch.get("loss_masks") is not None:
    raw_lm = _tensor_to_numpy(batch["loss_masks"], np.float32)
  else:
    raw_lm = (raw_labels >= 0).astype(np.float32)

  if seq_in < target_seq:
    pad_len = target_seq - seq_in
    raw_in = np.pad(raw_in, ((0, 0), (0, pad_len)), constant_values=np.int32(pad_token_id))
    raw_labels = np.pad(raw_labels, ((0, 0), (0, pad_len)), constant_values=np.int64(-100))
    raw_rtm = np.pad(raw_rtm, ((0, 0), (0, pad_len)), constant_values=False)
    raw_doc = np.pad(raw_doc, ((0, 0), (0, pad_len)), constant_values=np.int64(-1))
    raw_lm = np.pad(raw_lm, ((0, 0), (0, pad_len)), constant_values=np.float32(0.0))

  if int(padding_input_token_id) >= 0:
    inputs = np.where(raw_rtm, raw_in, np.int32(padding_input_token_id)).astype(np.int32, copy=False)
  else:
    inputs = raw_in.astype(np.int32, copy=False)

  run_change = np.ones((bsz, target_seq), dtype=np.int32)
  if target_seq > 1:
    run_change[:, 1:] = (raw_doc[:, 1:] != raw_doc[:, :-1]).astype(np.int32)
  inputs_segmentation = np.cumsum(run_change, axis=1, dtype=np.int32)

  pos_base = np.broadcast_to(np.arange(target_seq, dtype=np.int32)[None, :], (bsz, target_seq))
  if reset_positions_on_segment:
    seg_start_idx = np.maximum.accumulate(np.where(run_change != 0, pos_base, 0), axis=1).astype(np.int32)
    positions = (pos_base - seg_start_idx).astype(np.int32)
  else:
    positions = pos_base.copy()

  valid_token = raw_rtm & (raw_doc >= 0)
  targets = np.where(raw_labels >= 0, raw_labels, 0).astype(np.int32)
  targets_segmentation = np.where((raw_labels >= 0) & valid_token, inputs_segmentation, 0).astype(np.int32)

  mm_np_dtype = jnp.dtype(dtype_mm)
  raw_images = batch.get("images")
  if raw_images is not None:
    if isinstance(raw_images, list):
      # Unpack CompactImages or list[(C_i, P, D), ...] if encountered.
      for item_img in raw_images:
        if len(item_img) > 0:
          n_patches_per_crop = int(item_img.shape[1])
          patch_dim = int(item_img.shape[2])
          break
      c_in = max((len(x) for x in raw_images), default=0)
      img_arr = np.zeros((bsz, c_in, n_patches_per_crop, patch_dim), dtype=mm_np_dtype)
      for i, item_img in enumerate(raw_images):
        if len(item_img) > 0:
          img_arr[i, : len(item_img)] = _tensor_to_numpy_mm(item_img, mm_np_dtype)
    else:
      raw_img_arr = _tensor_to_numpy_mm(raw_images, mm_np_dtype)
      if raw_img_arr.ndim == 3:
        # Compact 3D images: (total_crops, n_patches, patch_dim) paired with image_crop_counts: (bsz,).
        n_patches_per_crop = int(raw_img_arr.shape[1])
        patch_dim = int(raw_img_arr.shape[2])
        raw_counts = batch.get("image_crop_counts")
        if raw_counts is None:
          raise ValueError(
              f"Compact 3D batch['images'] with shape {raw_img_arr.shape} requires batch['image_crop_counts']."
          )
        crop_counts = _tensor_to_numpy(raw_counts, np.int64).reshape(-1)
        if len(crop_counts) != bsz:
          raise ValueError(f"Expected image_crop_counts length {bsz}, got {len(crop_counts)}.")
        if int(crop_counts.sum()) != int(raw_img_arr.shape[0]):
          raise ValueError(
              f"Sum of image_crop_counts ({int(crop_counts.sum())}) != images.shape[0] ({raw_img_arr.shape[0]})."
          )
        c_in = int(crop_counts.max()) if bsz > 0 else 0
        img_arr = np.zeros((bsz, c_in, n_patches_per_crop, patch_dim), dtype=mm_np_dtype)
        offset = 0
        for i, cnt in enumerate(crop_counts):
          cnt_i = int(cnt)
          if cnt_i > 0:
            img_arr[i, :cnt_i] = raw_img_arr[offset : offset + cnt_i]
            offset += cnt_i
      elif raw_img_arr.ndim == 4:
        n_patches_per_crop = int(raw_img_arr.shape[2])
        patch_dim = int(raw_img_arr.shape[3])
        c_in = int(raw_img_arr.shape[1])
        img_arr = raw_img_arr
      else:
        raise ValueError(f"Unexpected batch['images'] ndim={raw_img_arr.ndim}, shape={raw_img_arr.shape}.")
  else:
    c_in = 0
    img_arr = np.zeros((bsz, 0, n_patches_per_crop, patch_dim), dtype=mm_np_dtype)

  if c_in > int(max_crops_per_seq):
    raise ValueError(
        f"Batch has {c_in} crops per sequence, exceeding config.olmo_max_crops_per_seq={max_crops_per_seq}."
    )
  images = np.zeros((bsz, int(max_crops_per_seq), n_patches_per_crop, patch_dim), dtype=mm_np_dtype)
  if c_in > 0:
    images[:, :c_in] = img_arr

  raw_pooled = batch.get("pooled_patches_idx")
  if raw_pooled is not None:
    pooled_arr = _tensor_to_numpy(raw_pooled, np.int32)
    p_in = pooled_arr.shape[1] if pooled_arr.ndim >= 2 else 0
    if pooled_arr.ndim == 3:
      pool_size = pooled_arr.shape[2]
  else:
    p_in = 0
    pooled_arr = np.full((bsz, 0, pool_size), -1, dtype=np.int32)

  if p_in > int(max_pooled_patches_per_seq):
    raise ValueError(
        f"Batch has {p_in} pooled patches per sequence, exceeding "
        f"config.olmo_max_pooled_patches_per_seq={max_pooled_patches_per_seq}."
    )
  pooled_patches_idx = np.full((bsz, int(max_pooled_patches_per_seq), pool_size), -1, dtype=np.int32)
  if p_in > 0:
    pooled_patches_idx[:, :p_in] = pooled_arr

  out: dict[str, np.ndarray] = {
      "inputs": inputs,
      "inputs_position": positions,
      "inputs_segmentation": inputs_segmentation,
      "targets": targets,
      "targets_position": positions,
      "targets_segmentation": targets_segmentation,
      "images": images,
      "pooled_patches_idx": pooled_patches_idx,
  }
  if use_loss_weights:
    out["loss_weights"] = (np.maximum(raw_lm, 0.0) * valid_token.astype(np.float32)).astype(np.float32)
  if include_router_token_mask:
    out["router_token_mask"] = valid_token.astype(np.int32)
  return out


def save_maxtext_batch_npz(
    path: str | os.PathLike[str] | epath.Path,
    batch: dict[str, np.ndarray],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
  """Serialize a MaxText multimodal batch to ``.npz`` (supports local paths and ``gs://`` URIs).

  Stores per-sequence active crops in compact 3D ``uint16`` ``bfloat16`` bit-pattern
  (``images_compact_bf16_u16`` + ``image_crop_counts``) and ``pooled_patches_idx`` trimmed to
  the active pooled-patch rows so standard NumPy ``.npz`` archives stay compact and portable.
  """
  images = batch["images"]
  pooled = batch["pooled_patches_idx"]
  bsz = int(images.shape[0])
  n_patches = int(images.shape[2])
  patch_dim = int(images.shape[3])

  valid_pool_rows = np.any(pooled >= 0, axis=(0, 2))
  used_pooled = int(np.flatnonzero(valid_pool_rows)[-1] + 1) if np.any(valid_pool_rows) else 1

  images_bf16 = np.asarray(images, dtype=jnp.dtype("bfloat16"))
  images_u16 = images_bf16.view(np.uint16)

  crop_counts = np.zeros((bsz,), dtype=np.int32)
  for i in range(bsz):
    max_idx_i = int(pooled[i, :used_pooled].max()) if used_pooled > 0 else -1
    c_idx_i = (max_idx_i // max(1, n_patches) + 1) if max_idx_i >= 0 else 0
    nz_cols_i = np.any(images_u16[i] != 0, axis=(1, 2))
    c_img_i = int(np.flatnonzero(nz_cols_i)[-1] + 1) if np.any(nz_cols_i) else 0
    crop_counts[i] = max(c_idx_i, c_img_i)

  total_crops = int(crop_counts.sum())
  compact_u16 = np.empty((total_crops, n_patches, patch_dim), dtype=np.uint16)
  offset = 0
  for i in range(bsz):
    cnt = int(crop_counts[i])
    if cnt > 0:
      compact_u16[offset : offset + cnt] = images_u16[i, :cnt]
      offset += cnt

  payload: dict[str, Any] = {
      "inputs": np.asarray(batch["inputs"], dtype=np.int32),
      "inputs_position": np.asarray(batch["inputs_position"], dtype=np.int32),
      "inputs_segmentation": np.asarray(batch["inputs_segmentation"], dtype=np.int32),
      "targets": np.asarray(batch["targets"], dtype=np.int32),
      "targets_position": np.asarray(batch["targets_position"], dtype=np.int32),
      "targets_segmentation": np.asarray(batch["targets_segmentation"], dtype=np.int32),
      "images_compact_bf16_u16": compact_u16,
      "image_crop_counts": crop_counts,
      "pooled_patches_idx": np.asarray(pooled[:, :used_pooled], dtype=np.int32),
  }
  if "loss_weights" in batch and batch["loss_weights"] is not None:
    payload["loss_weights"] = np.asarray(batch["loss_weights"], dtype=np.float32)
  if "router_token_mask" in batch and batch["router_token_mask"] is not None:
    payload["router_token_mask"] = np.asarray(batch["router_token_mask"], dtype=np.int32)
  if metadata is not None:
    payload["metadata_json"] = np.asarray(json.dumps(metadata))

  out_path = epath.Path(path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  path_str = os.fspath(out_path)
  if not path_str.startswith("gs://"):
    with builtins.open(path_str, "wb") as f:
      np.savez(f, **payload)
  else:
    buf = io.BytesIO()
    np.savez(buf, **payload)
    out_path.write_bytes(buf.getvalue())


_GCS_STORAGE_CLIENT: Any = None


def _get_gcs_storage_client() -> Any:
  global _GCS_STORAGE_CLIENT
  if _GCS_STORAGE_CLIENT is None:
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    _GCS_STORAGE_CLIENT = storage.Client()
  return _GCS_STORAGE_CLIENT


def _open_seekable_binary_stream(path_str: str) -> Any:
  if not path_str.startswith("gs://"):
    return builtins.open(path_str, "rb")
  rest = path_str[len("gs://") :]
  bucket_name, blob_name = rest.split("/", 1)
  return _get_gcs_storage_client().bucket(bucket_name).blob(blob_name).open("rb", chunk_size=8 * 1024 * 1024)


def _read_stored_npy_header(
    zf: zipfile.ZipFile, raw_f: Any, key: str
) -> Optional[Tuple[Tuple[int, ...], np.dtype, int]]:
  """Return ``(shape, dtype, data_offset)`` for an uncompressed ``ZIP_STORED`` ``.npy`` entry, or ``None``."""
  member_name = f"{key}.npy"
  if member_name not in zf.NameToInfo:
    return None
  info = zf.getinfo(member_name)
  if info.compress_type != zipfile.ZIP_STORED:
    return None
  for probe_cap in (4096, 65536):
    raw_f.seek(info.header_offset)
    probe = raw_f.read(min(info.file_size + 512, probe_cap))
    if len(probe) < 30 or probe[:4] != b"PK\x03\x04":
      return None
    fname_len, extra_len = struct.unpack_from("<HH", probe, 26)
    npy_start = 30 + fname_len + extra_len
    if npy_start >= len(probe):
      raw_f.seek(info.header_offset + npy_start)
      npy_bytes = raw_f.read(min(info.file_size, probe_cap))
    else:
      npy_bytes = probe[npy_start:]
    bio = io.BytesIO(npy_bytes)
    try:
      version = np_format.read_magic(bio)
      if version == (1, 0):
        shape, fortran, dtype = np_format.read_array_header_1_0(bio)
      elif version in ((2, 0), (3, 0)):
        shape, fortran, dtype = np_format.read_array_header_2_0(bio)
      else:
        return None
      if fortran or dtype.hasobject:
        return None
      return tuple(int(d) for d in shape), dtype, int(info.header_offset + npy_start + bio.tell())
    except EOFError:
      continue
  return None


def _read_contiguous_npy_rows(
    zf: zipfile.ZipFile, raw_f: Any, key: str, start: Optional[int], stop: Optional[int]
) -> Optional[np.ndarray]:
  meta = _read_stored_npy_header(zf, raw_f, key)
  if meta is None:
    return None
  shape, dtype, data_offset = meta
  if not shape:
    raw_f.seek(data_offset)
    buf = raw_f.read(dtype.itemsize)
    if len(buf) != dtype.itemsize:
      raise EOFError(f"Unexpected EOF reading scalar {key}.npy")
    return np.frombuffer(buf, dtype=dtype).reshape(()).copy()
  n0 = shape[0]
  s0, s1, step = slice(start, stop, 1).indices(n0)
  if step != 1:
    return None
  row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
  nbytes = max(0, s1 - s0) * row_bytes
  if nbytes == 0:
    return np.empty((0, *shape[1:]), dtype=dtype)
  raw_f.seek(data_offset + s0 * row_bytes)
  buf = bytearray(nbytes)
  mv = memoryview(buf)
  read_so_far = 0
  while read_so_far < nbytes:
    chunk = raw_f.read(nbytes - read_so_far)
    if not chunk:
      raise EOFError(f"Unexpected EOF reading slice [{s0}:{s1}] of {key}.npy")
    mv[read_so_far : read_so_far + len(chunk)] = chunk
    read_so_far += len(chunk)
  return np.frombuffer(buf, dtype=dtype).reshape((s1 - s0, *shape[1:]))


def _try_load_maxtext_batch_npz_seekable(
    path_str: str,
    *,
    max_crops_per_seq: int,
    max_pooled_patches_per_seq: int,
    mm_np_dtype: np.dtype,
    sl: slice,
    use_loss_weights: bool,
    include_router_token_mask: bool,
) -> Optional[dict[str, np.ndarray]]:
  """Fast byte-range reader for uncompressed ``.npz`` archives (avoids full-file download when slicing rows)."""
  if sl.step not in (None, 1):
    return None
  try:
    raw_f = _open_seekable_binary_stream(path_str)
  except Exception:  # pylint: disable=broad-except
    return None
  try:
    if not getattr(raw_f, "seekable", lambda: False)():
      return None
    with zipfile.ZipFile(raw_f, "r") as zf:
      inp_meta = _read_stored_npy_header(zf, raw_f, "inputs")
      if inp_meta is None or len(inp_meta[0]) != 2:
        return None
      full_bsz, _ = inp_meta[0]
      r_start, r_stop, _ = sl.indices(full_bsz)

      inputs_arr = _read_contiguous_npy_rows(zf, raw_f, "inputs", r_start, r_stop)
      if inputs_arr is None:
        return None
      inputs = np.asarray(inputs_arr, dtype=np.int32)
      bsz, seq_len = inputs.shape

      if "images_compact_bf16_u16.npy" in zf.NameToInfo:
        counts_raw = _read_contiguous_npy_rows(zf, raw_f, "image_crop_counts", None, None)
        compact_meta = _read_stored_npy_header(zf, raw_f, "images_compact_bf16_u16")
        if counts_raw is None or compact_meta is None or len(compact_meta[0]) != 3:
          return None
        all_counts = np.asarray(counts_raw, dtype=np.int64)
        n_patches, patch_dim = compact_meta[0][1], compact_meta[0][2]
        c_in = int(all_counts[r_start:r_stop].max()) if bsz > 0 else 0
        if c_in > int(max_crops_per_seq):
          raise ValueError(
              f"Saved batch {path_str} has {c_in} crops > config.olmo_max_crops_per_seq={max_crops_per_seq}."
          )
        images = np.zeros((bsz, int(max_crops_per_seq), n_patches, patch_dim), dtype=mm_np_dtype)
        if bsz > 0 and c_in > 0:
          offsets = np.zeros(len(all_counts) + 1, dtype=np.int64)
          np.cumsum(all_counts, out=offsets[1:])
          c_start = int(offsets[r_start])
          c_end = int(offsets[r_stop])
          sub_raw = _read_contiguous_npy_rows(zf, raw_f, "images_compact_bf16_u16", c_start, c_end)
          if sub_raw is None:
            return None
          sub_u16 = np.ascontiguousarray(sub_raw, dtype=np.uint16)
          sub_mm = sub_u16.view(jnp.dtype("bfloat16")).astype(mm_np_dtype, copy=False)
          local_off = 0
          for j in range(bsz):
            cnt = int(all_counts[r_start + j])
            if cnt > 0:
              images[j, :cnt] = sub_mm[local_off : local_off + cnt]
              local_off += cnt
      else:
        img_raw = _read_contiguous_npy_rows(zf, raw_f, "images_bf16_u16", r_start, r_stop)
        if img_raw is None:
          return None
        img_u16 = np.ascontiguousarray(img_raw, dtype=np.uint16)
        img_sub = img_u16.view(jnp.dtype("bfloat16")).astype(mm_np_dtype, copy=False)
        c_in, n_patches, patch_dim = img_sub.shape[1], img_sub.shape[2], img_sub.shape[3]
        if c_in > int(max_crops_per_seq):
          raise ValueError(
              f"Saved batch {path_str} has {c_in} crops > config.olmo_max_crops_per_seq={max_crops_per_seq}."
          )
        images = np.zeros((bsz, int(max_crops_per_seq), n_patches, patch_dim), dtype=mm_np_dtype)
        images[:, :c_in] = img_sub

      pooled_raw = _read_contiguous_npy_rows(zf, raw_f, "pooled_patches_idx", r_start, r_stop)
      if pooled_raw is None:
        return None
      pooled_sub = np.asarray(pooled_raw, dtype=np.int32)
      p_in, pool_size = pooled_sub.shape[1], pooled_sub.shape[2]
      if p_in > int(max_pooled_patches_per_seq):
        raise ValueError(
            f"Saved batch {path_str} has {p_in} pooled patches > "
            f"config.olmo_max_pooled_patches_per_seq={max_pooled_patches_per_seq}."
        )
      pooled_patches_idx = np.full((bsz, int(max_pooled_patches_per_seq), pool_size), -1, dtype=np.int32)
      pooled_patches_idx[:, :p_in] = pooled_sub

      in_pos = _read_contiguous_npy_rows(zf, raw_f, "inputs_position", r_start, r_stop)
      in_seg = _read_contiguous_npy_rows(zf, raw_f, "inputs_segmentation", r_start, r_stop)
      tgts = _read_contiguous_npy_rows(zf, raw_f, "targets", r_start, r_stop)
      tgt_pos = _read_contiguous_npy_rows(zf, raw_f, "targets_position", r_start, r_stop)
      tgt_seg_raw = _read_contiguous_npy_rows(zf, raw_f, "targets_segmentation", r_start, r_stop)
      if any(arr is None for arr in (in_pos, in_seg, tgts, tgt_pos, tgt_seg_raw)):
        return None
      targets_seg = np.asarray(tgt_seg_raw, dtype=np.int32)
      out: dict[str, np.ndarray] = {
          "inputs": inputs,
          "inputs_position": np.asarray(in_pos, dtype=np.int32),
          "inputs_segmentation": np.asarray(in_seg, dtype=np.int32),
          "targets": np.asarray(tgts, dtype=np.int32),
          "targets_position": np.asarray(tgt_pos, dtype=np.int32),
          "targets_segmentation": targets_seg,
          "images": images,
          "pooled_patches_idx": pooled_patches_idx,
      }
      if use_loss_weights:
        if "loss_weights.npy" in zf.NameToInfo:
          lw_raw = _read_contiguous_npy_rows(zf, raw_f, "loss_weights", r_start, r_stop)
          if lw_raw is None:
            return None
          out["loss_weights"] = np.asarray(lw_raw, dtype=np.float32)
        else:
          out["loss_weights"] = (targets_seg != 0).astype(np.float32)
      if include_router_token_mask:
        if "router_token_mask.npy" in zf.NameToInfo:
          rtm_raw = _read_contiguous_npy_rows(zf, raw_f, "router_token_mask", r_start, r_stop)
          if rtm_raw is None:
            return None
          out["router_token_mask"] = np.asarray(rtm_raw, dtype=np.int32)
        else:
          out["router_token_mask"] = np.ones((bsz, seq_len), dtype=np.int32)
      return out
  except ValueError:
    raise
  except Exception:  # pylint: disable=broad-except
    return None
  finally:
    raw_f.close()


def load_maxtext_batch_npz(
    path: str | os.PathLike[str] | epath.Path,
    *,
    max_crops_per_seq: int,
    max_pooled_patches_per_seq: int,
    dtype_mm: Any = jnp.bfloat16,
    row_slice: Optional[slice] = None,
    use_loss_weights: bool = True,
    include_router_token_mask: bool = True,
) -> dict[str, np.ndarray]:
  """Load a pre-exported ``.npz`` batch from local disk or GCS and pad vision arrays to config caps."""
  sl = row_slice if row_slice is not None else slice(None)
  mm_np_dtype = jnp.dtype(dtype_mm)
  p_obj = epath.Path(path)
  path_str = os.fspath(p_obj)
  fast_loaded = _try_load_maxtext_batch_npz_seekable(
      path_str,
      max_crops_per_seq=max_crops_per_seq,
      max_pooled_patches_per_seq=max_pooled_patches_per_seq,
      mm_np_dtype=mm_np_dtype,
      sl=sl,
      use_loss_weights=use_loss_weights,
      include_router_token_mask=include_router_token_mask,
  )
  if fast_loaded is not None:
    return fast_loaded
  npz_source: Any = path_str if not path_str.startswith("gs://") else io.BytesIO(p_obj.read_bytes())
  with np.load(npz_source, allow_pickle=False) as z:
    inputs = np.asarray(z["inputs"][sl], dtype=np.int32)
    bsz, seq_len = inputs.shape
    if "images_compact_bf16_u16" in z:
      all_counts = np.asarray(z["image_crop_counts"], dtype=np.int64)
      row_indices = list(range(*sl.indices(len(all_counts))))
      c_in = max((int(all_counts[r]) for r in row_indices), default=0)
      compact_arr = z["images_compact_bf16_u16"]
      n_patches, patch_dim = int(compact_arr.shape[1]), int(compact_arr.shape[2])
      if c_in > int(max_crops_per_seq):
        raise ValueError(
            f"Saved batch {path} has {c_in} crops > config.olmo_max_crops_per_seq={max_crops_per_seq}."
        )
      images = np.zeros((bsz, int(max_crops_per_seq), n_patches, patch_dim), dtype=mm_np_dtype)
      if bsz > 0 and c_in > 0:
        offsets = np.zeros(len(all_counts) + 1, dtype=np.int64)
        np.cumsum(all_counts, out=offsets[1:])
        if sl.step in (None, 1):
          c_start = int(offsets[row_indices[0]])
          c_end = int(offsets[row_indices[-1] + 1])
          sub_u16 = np.ascontiguousarray(compact_arr[c_start:c_end], dtype=np.uint16)
          sub_mm = sub_u16.view(jnp.dtype("bfloat16")).astype(mm_np_dtype, copy=False)
          local_off = 0
          for j, r in enumerate(row_indices):
            cnt = int(all_counts[r])
            if cnt > 0:
              images[j, :cnt] = sub_mm[local_off : local_off + cnt]
              local_off += cnt
        else:
          for j, r in enumerate(row_indices):
            cnt = int(all_counts[r])
            if cnt > 0:
              sub_u16 = np.ascontiguousarray(compact_arr[offsets[r] : offsets[r + 1]], dtype=np.uint16)
              images[j, :cnt] = sub_u16.view(jnp.dtype("bfloat16")).astype(mm_np_dtype, copy=False)
    else:
      img_u16 = np.ascontiguousarray(z["images_bf16_u16"][sl], dtype=np.uint16)
      img_sub = img_u16.view(jnp.dtype("bfloat16")).astype(mm_np_dtype, copy=False)
      c_in, n_patches, patch_dim = img_sub.shape[1], img_sub.shape[2], img_sub.shape[3]
      if c_in > int(max_crops_per_seq):
        raise ValueError(
            f"Saved batch {path} has {c_in} crops > config.olmo_max_crops_per_seq={max_crops_per_seq}."
        )
      images = np.zeros((bsz, int(max_crops_per_seq), n_patches, patch_dim), dtype=mm_np_dtype)
      images[:, :c_in] = img_sub

    pooled_sub = np.asarray(z["pooled_patches_idx"][sl], dtype=np.int32)
    p_in, pool_size = pooled_sub.shape[1], pooled_sub.shape[2]
    if p_in > int(max_pooled_patches_per_seq):
      raise ValueError(
          f"Saved batch {path} has {p_in} pooled patches > "
          f"config.olmo_max_pooled_patches_per_seq={max_pooled_patches_per_seq}."
      )
    pooled_patches_idx = np.full((bsz, int(max_pooled_patches_per_seq), pool_size), -1, dtype=np.int32)
    pooled_patches_idx[:, :p_in] = pooled_sub

    targets_seg = np.asarray(z["targets_segmentation"][sl], dtype=np.int32)
    out: dict[str, np.ndarray] = {
        "inputs": inputs,
        "inputs_position": np.asarray(z["inputs_position"][sl], dtype=np.int32),
        "inputs_segmentation": np.asarray(z["inputs_segmentation"][sl], dtype=np.int32),
        "targets": np.asarray(z["targets"][sl], dtype=np.int32),
        "targets_position": np.asarray(z["targets_position"][sl], dtype=np.int32),
        "targets_segmentation": targets_seg,
        "images": images,
        "pooled_patches_idx": pooled_patches_idx,
    }
    if use_loss_weights:
      if "loss_weights" in z:
        out["loss_weights"] = np.asarray(z["loss_weights"][sl], dtype=np.float32)
      else:
        out["loss_weights"] = (targets_seg != 0).astype(np.float32)
    if include_router_token_mask:
      if "router_token_mask" in z:
        out["router_token_mask"] = np.asarray(z["router_token_mask"][sl], dtype=np.int32)
      else:
        out["router_token_mask"] = np.ones((bsz, seq_len), dtype=np.int32)
  return out


def _resolve_vision_caps(config) -> Tuple[int, int, int, int, int]:
  img_sz = config.image_size_for_vit
  img_h, img_w = tuple(img_sz) if isinstance(img_sz, (list, tuple)) else (int(img_sz), int(img_sz))
  patch = int(config.patch_size_for_vit)
  channels = int(config.num_channels_for_vit)
  n_patches = (img_h // patch) * (img_w // patch)
  patch_dim = channels * patch * patch
  pool_h, pool_w = (img_h // patch + 1) // 2, (img_w // patch + 1) // 2
  n_crops = int(getattr(config, "olmo_max_crops_per_seq", 64))
  n_pooled = (
      int(config.olmo_max_pooled_patches_per_seq)
      if int(getattr(config, "olmo_max_pooled_patches_per_seq", 0)) > 0
      else n_crops * pool_h * pool_w
  )
  pool_size = int(getattr(config, "olmo_connector_pool_size", 4))
  return n_crops, n_pooled, n_patches, patch_dim, pool_size


def build_olmo_core_mixture_loader(
    config_path: str | os.PathLike[str] | epath.Path,
    *,
    work_dir: str,
    shard_index: int = 0,
    shard_count: int = 1,
    path_remap_from: str = "",
    path_remap_to: str = "",
    global_batch_tokens: Optional[int] = None,
    pack_buffer_size: Optional[int] = None,
    num_workers: int = 0,
    batch_prefetch_depth: Optional[int] = None,
    seed_offset: int = 0,
) -> Tuple[Any, dict[str, Any]]:
  """Build a live ``OLMo-core`` ``MixtureDataLoader`` from a serialized experiment ``config.json``."""
  if path_remap_from and path_remap_to:
    install_path_remap_prefix(path_remap_from, path_remap_to)
  register_merged_mt_lc_shim()
  from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig  # pylint: disable=import-outside-toplevel
  from olmo_core.data.multimodal.collator import MultimodalCollator  # pylint: disable=import-outside-toplevel
  from olmo_core.data.multimodal.mixture_data_loader import (  # pylint: disable=import-outside-toplevel
      MixtureDataLoader,
      MixtureDataLoaderConfig,
  )

  raw_cfg = json.loads(epath.Path(config_path).read_text())
  if path_remap_from and path_remap_to:
    raw_cfg = remap_paths_in_obj(raw_cfg, path_remap_from, path_remap_to)

  Path(work_dir).mkdir(parents=True, exist_ok=True)
  ds_dict = _set_work_dirs_recursive(dict(raw_cfg["dataset"]), work_dir)
  dl_dict = _set_work_dirs_recursive(dict(raw_cfg["data_loader"]), work_dir)
  if global_batch_tokens is not None:
    dl_dict["global_batch_size"] = int(global_batch_tokens)
  if pack_buffer_size is not None:
    dl_dict["pack_buffer_size"] = int(pack_buffer_size)
  if seed_offset:
    dl_dict["seed"] = int(dl_dict.get("seed", 0)) ^ int(seed_offset)
  # Keep worker loading in-process or thread-based inside JAX host processes to avoid fork+JAX deadlocks.
  dl_dict["prefetch_processes"] = False
  dl_dict["batch_prefetch_process"] = False
  seq_len_cfg = int(dl_dict.get("sequence_length") or 0)
  if int(num_workers) <= 0:
    dl_dict["prefetch_workers"] = 0
    dl_dict["batch_prefetch_depth"] = 0
  else:
    dl_dict["prefetch_workers"] = int(num_workers)
    default_depth = 0 if seq_len_cfg >= 65536 else min(int(dl_dict.get("batch_prefetch_depth") or 1), 1)
    dl_dict["batch_prefetch_depth"] = (
        max(0, int(batch_prefetch_depth)) if batch_prefetch_depth is not None else default_depth
    )
    in_flight_cap = max(16, int(num_workers) * 4) if seq_len_cfg >= 65536 else max(32, int(num_workers) * 16)
    dl_dict["prefetch_max_in_flight"] = min(int(dl_dict.get("prefetch_max_in_flight") or in_flight_cap), in_flight_cap)

  ds_cfg = MultimodalMixtureConfig.from_dict(ds_dict)
  dl_cfg = MixtureDataLoaderConfig.from_dict(dl_dict)
  dataset = ds_cfg.build()
  dp_world = int(shard_count)
  dp_rank = int(shard_index)
  if dp_world == 1 and dp_rank == 0:
    loader = dl_cfg.build(dataset)
  else:
    if (
        dl_cfg.sequence_length <= 0
        or dl_cfg.global_batch_size <= 0
        or dl_cfg.global_batch_size % (dl_cfg.sequence_length * dp_world) != 0
    ):
      raise ValueError(
          f"global_batch_size ({dl_cfg.global_batch_size}) must be a positive multiple of "
          f"sequence_length ({dl_cfg.sequence_length}) * shard_count ({dp_world})."
      )
    pad_token_id = dataset.tokenizer.pad_token_id
    if pad_token_id is None:
      pad_token_id = dataset.tokenizer.eos_token_id
    if pad_token_id is None:
      raise ValueError("Tokenizer in OLMo-core multimodal mixture must define pad_token_id or eos_token_id.")
    loader = MixtureDataLoader(
        dataset.datasets,
        dataset.weights,
        MultimodalCollator(
            pad_token_id=pad_token_id,
            pad_sequence_length=dl_cfg.sequence_length,
            batch_metadata=dl_cfg.batch_metadata,
            image_dtype=dl_cfg.image_dtype.as_pt() if dl_cfg.image_dtype is not None else None,
            compact_images=dl_cfg.compact_images,
        ),
        work_dir=dl_cfg.work_dir,
        global_batch_size=dl_cfg.global_batch_size,
        seed=dl_cfg.seed,
        epoch_instances=dl_cfg.epoch_instances,
        pack=dl_cfg.pack,
        pack_max_crops=dl_cfg.pack_max_crops,
        pack_buffer_size=dl_cfg.pack_buffer_size,
        pack_image_weight=dl_cfg.pack_image_weight,
        continuous_stream=dl_cfg.continuous_stream,
        est_tokens_per_example=dl_cfg.est_tokens_per_example,
        prefetch_workers=dl_cfg.prefetch_workers,
        prefetch_max_in_flight=dl_cfg.prefetch_max_in_flight,
        prefetch_processes=dl_cfg.prefetch_processes,
        defer_packed_image_copy=(
            False if dl_cfg.defer_packed_image_copy is None else dl_cfg.defer_packed_image_copy
        ),
        batch_prefetch_depth=dl_cfg.batch_prefetch_depth,
        batch_prefetch_process=dl_cfg.batch_prefetch_process,
        max_consecutive_data_errors=dl_cfg.max_consecutive_data_errors,
        max_total_data_errors=dl_cfg.max_total_data_errors,
        dp_world_size=dp_world,
        dp_rank=dp_rank,
        dataset_names=dataset.names,
        allow_legacy_state_without_dataset_fingerprints=(
            dl_cfg.allow_legacy_state_without_dataset_fingerprints
        ),
        source_groups=dl_cfg.source_groups,
        group_sequence_quotas=dl_cfg.group_sequence_quotas,
    )
  loader.reshuffle(epoch=1)
  return loader, raw_cfg


class _PreExportedNpzHostLoader:
  """Per-host iterable yielding slices of pre-exported ``batch_XXXXX.npz`` global batches."""

  def __init__(
      self,
      *,
      batches_dir: str,
      shard_index: int,
      shard_count: int,
      per_host_batch: int,
      max_crops_per_seq: int,
      max_pooled_patches_per_seq: int,
      dtype_mm: Any,
      use_loss_weights: bool,
      initial_batch_step: int = 0,
  ):
    root = epath.Path(batches_dir)
    if not root.exists():
      raise FileNotFoundError(f"olmo_mm_batches_dir does not exist: {batches_dir}")
    files = sorted(p for p in root.iterdir() if p.name.endswith(".npz"))
    if not files:
      raise ValueError(f"No .npz batch files found in olmo_mm_batches_dir={batches_dir}")
    self._files = files
    self._row_slice = slice(shard_index * per_host_batch, (shard_index + 1) * per_host_batch)
    self._max_crops = int(max_crops_per_seq)
    self._max_pooled = int(max_pooled_patches_per_seq)
    self._dtype_mm = dtype_mm
    self._use_loss_weights = bool(use_loss_weights)
    self._step = int(initial_batch_step)
    self._shard_count = int(shard_count)
    self._per_host_batch = int(per_host_batch)
    self._cache_sliced = os.environ.get("OLMO_MM_CACHE_SLICED_NPZ", "0") == "1" and len(self._files) <= 32
    self._sliced_cache: dict[int, dict[str, np.ndarray]] = {}

  def __iter__(self):
    return self

  def __next__(self) -> dict[str, np.ndarray]:
    file_idx = self._step % len(self._files)
    path = self._files[file_idx]
    self._step += 1
    if self._cache_sliced and file_idx in self._sliced_cache:
      return self._sliced_cache[file_idx]
    batch = load_maxtext_batch_npz(
        path,
        max_crops_per_seq=self._max_crops,
        max_pooled_patches_per_seq=self._max_pooled,
        dtype_mm=self._dtype_mm,
        row_slice=self._row_slice,
        use_loss_weights=self._use_loss_weights,
        include_router_token_mask=True,
    )
    if batch["inputs"].shape[0] != self._per_host_batch:
      raise ValueError(
          f"Batch {path} yielded {batch['inputs'].shape[0]} rows for slice {self._row_slice}, "
          f"expected per_host_batch={self._per_host_batch} (shard_count={self._shard_count})."
      )
    if self._cache_sliced:
      self._sliced_cache[file_idx] = batch
    return batch


class _LiveMixtureHostLoader:
  """Per-host iterable wrapping ``OLMo-core`` ``MixtureDataLoader`` and converting batches on the fly."""

  def __init__(
      self,
      *,
      config_path: str,
      work_dir: str,
      shard_index: int,
      shard_count: int,
      per_host_batch: int,
      max_target_length: int,
      max_crops_per_seq: int,
      max_pooled_patches_per_seq: int,
      n_patches_per_crop: int,
      patch_dim: int,
      pool_size: int,
      dtype_mm: Any,
      reset_positions_on_segment: bool,
      padding_input_token_id: int,
      use_loss_weights: bool,
      path_remap_from: str,
      path_remap_to: str,
      num_workers: int,
      seed_offset: int = 0,
      initial_batch_step: int = 0,
  ):
    global_batch_tokens = int(per_host_batch) * int(shard_count) * int(max_target_length)
    self._loader, raw_cfg = build_olmo_core_mixture_loader(
        config_path,
        work_dir=work_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        path_remap_from=path_remap_from,
        path_remap_to=path_remap_to,
        global_batch_tokens=global_batch_tokens,
        num_workers=num_workers,
        seed_offset=seed_offset,
    )
    tm_pad_id = raw_cfg.get("train_module", {}).get("padding_input_token_id")
    if int(padding_input_token_id) >= 0:
      self._padding_input_token_id = int(padding_input_token_id)
    elif tm_pad_id is not None:
      self._padding_input_token_id = int(tm_pad_id)
    else:
      self._padding_input_token_id = -1

    self._max_target_length = int(max_target_length)
    self._max_crops = int(max_crops_per_seq)
    self._max_pooled = int(max_pooled_patches_per_seq)
    self._n_patches = int(n_patches_per_crop)
    self._patch_dim = int(patch_dim)
    self._pool_size = int(pool_size)
    self._dtype_mm = dtype_mm
    self._reset_positions = bool(reset_positions_on_segment)
    self._use_loss_weights = bool(use_loss_weights)

    self._iter = iter(self._loader)
    for _ in range(int(initial_batch_step)):
      try:
        next(self._iter)
      except StopIteration:
        self._loader.reshuffle()
        self._iter = iter(self._loader)
        next(self._iter)

  def __iter__(self):
    return self

  def __next__(self) -> dict[str, np.ndarray]:
    try:
      raw_batch = next(self._iter)
    except StopIteration:
      self._loader.reshuffle()
      self._iter = iter(self._loader)
      raw_batch = next(self._iter)
    return olmo_core_batch_to_maxtext_batch(
        raw_batch,
        max_target_length=self._max_target_length,
        max_crops_per_seq=self._max_crops,
        max_pooled_patches_per_seq=self._max_pooled,
        n_patches_per_crop=self._n_patches,
        patch_dim=self._patch_dim,
        pool_size=self._pool_size,
        dtype_mm=self._dtype_mm,
        reset_positions_on_segment=self._reset_positions,
        padding_input_token_id=self._padding_input_token_id,
        use_loss_weights=self._use_loss_weights,
        include_router_token_mask=True,
    )


def _make_mm_loader_for_host(
    config,
    *,
    process_indices: List[int],
    is_eval: bool = False,
):
  this_proc = jax.process_index()
  shard_index = process_indices.index(this_proc)
  shard_count = len(process_indices)

  global_load = config.global_batch_size_to_load_eval if is_eval else config.global_batch_size_to_load
  num_procs = jax.process_count()
  per_host_batch = global_load // num_procs
  if per_host_batch * num_procs != global_load:
    raise ValueError(f"global_batch_size_to_load={global_load} is not divisible by jax.process_count()={num_procs}")

  n_crops, n_pooled, n_patches, patch_dim, pool_size = _resolve_vision_caps(config)
  use_loss_weights = bool(getattr(config, "olmo_use_loss_weights", False))
  resumed_step = 0 if is_eval else _detect_resumed_step(config)

  batches_dir = (getattr(config, "olmo_mm_batches_dir", "") or "").strip()
  cfg_path = (getattr(config, "olmo_mm_config_path", "") or "").strip()
  if batches_dir:
    max_logging.log(
        f"OLMo MM pre-exported batch loader: dir={batches_dir} shard={shard_index}/{shard_count} "
        f"per_host_batch={per_host_batch} seq={config.max_target_length} crops={n_crops} pooled={n_pooled} "
        f"resumed_step={resumed_step}"
    )
    return _PreExportedNpzHostLoader(
        batches_dir=batches_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        per_host_batch=per_host_batch,
        max_crops_per_seq=n_crops,
        max_pooled_patches_per_seq=n_pooled,
        dtype_mm=config.dtype_mm,
        use_loss_weights=use_loss_weights,
        initial_batch_step=resumed_step,
    )

  if not cfg_path:
    raise ValueError(
        "When dataset_type=olmo_mm, set either config.olmo_mm_batches_dir (pre-exported .npz batches) "
        "or config.olmo_mm_config_path (OLMo-core multimodal config.json)."
    )

  work_dir = (getattr(config, "olmo_mm_work_dir", "") or "/tmp/olmo_mm_work").strip()
  max_logging.log(
      f"OLMo MM live MixtureDataLoader: config={cfg_path} work_dir={work_dir} shard={shard_index}/{shard_count} "
      f"per_host_batch={per_host_batch} seq={config.max_target_length} crops={n_crops} pooled={n_pooled} "
      f"resumed_step={resumed_step}"
  )
  return _LiveMixtureHostLoader(
      config_path=cfg_path,
      work_dir=work_dir,
      shard_index=shard_index,
      shard_count=shard_count,
      per_host_batch=per_host_batch,
      max_target_length=config.max_target_length,
      max_crops_per_seq=n_crops,
      max_pooled_patches_per_seq=n_pooled,
      n_patches_per_crop=n_patches,
      patch_dim=patch_dim,
      pool_size=pool_size,
      dtype_mm=config.dtype_mm,
      reset_positions_on_segment=bool(getattr(config, "olmo_reset_positions_on_segment", True)),
      padding_input_token_id=int(getattr(config, "olmo_mm_padding_input_token_id", -1)),
      use_loss_weights=use_loss_weights,
      path_remap_from=getattr(config, "olmo_path_remap_from", "") or "",
      path_remap_to=getattr(config, "olmo_path_remap_to", "") or "",
      num_workers=int(getattr(config, "olmo_mm_num_workers", 0)),
      seed_offset=0x1F1F1F1F if is_eval else 0,
      initial_batch_step=resumed_step,
  )


def make_olmo_mm_train_iterator(config, global_mesh, process_indices):
  """Train iterator for ``dataset_type=olmo_mm``."""
  loader = _make_mm_loader_for_host(config, process_indices=process_indices, is_eval=False)
  return multihost_dataloading.MultiHostDataLoadIterator(
      loader,
      global_mesh,
      config.generate_padding_batch_train,
      expansion_loading_factor_for_grain=-1,
  )


def make_olmo_mm_eval_iterator(config, global_mesh, process_indices):
  """Eval iterator for ``dataset_type=olmo_mm``."""
  loader = _make_mm_loader_for_host(config, process_indices=process_indices, is_eval=True)
  return multihost_dataloading.MultiHostDataLoadIterator(
      loader,
      global_mesh,
      config.generate_padding_batch_eval,
      expansion_loading_factor_for_grain=-1,
  )
