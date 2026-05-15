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

"""MaxText trainer adapter for the OLMo numpy fixed-seq-length pipeline.

The trainer expects ``dataset_type`` to map to two factory functions
``(make_<type>_train_iterator, make_<type>_eval_iterator)`` that take
``(config, mesh, process_indices)`` and return a
:class:`MultiHostDataLoadIterator`.

This module provides those for ``dataset_type=olmo_grain``. The hard work
lives in :mod:`maxtext.input_pipeline.olmo_data_grain` (data source +
sampler + transforms); here we just wire it to MaxText's config + the
multihost dataloading wrapper.

Notes
-----

* **Sequence length match**: ``config.max_target_length`` must match the
  ``sequence_length`` recorded in the index JSON. Mismatches raise at load
  time.
* **Path remap**: AI2's index typically holds ``gs://`` URIs. For training,
  we read via a GCSFUSE mount on each TPU host. The
  ``olmo_path_remap_from`` / ``olmo_path_remap_to`` config pair rewrites
  the prefix at runtime.
* **Sharding**: each data-loading host is assigned a non-overlapping shard
  of the global instance space via ``OlmoIndexSampler``. We use
  ``process_indices.index(jax.process_index())`` as the local shard index
  (matches the pattern in :mod:`grain_data_processing`).
"""

from __future__ import annotations

from typing import List

import jax
from etils import epath

from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline.olmo_data import load_index
from maxtext.input_pipeline.olmo_data_grain import make_olmo_grain_data_loader
from maxtext.utils import max_logging


def _build_path_remap(config) -> dict:
  src = getattr(config, "olmo_path_remap_from", "") or ""
  dst = getattr(config, "olmo_path_remap_to", "") or ""
  if src and dst:
    return {src: dst}
  if src or dst:
    raise ValueError("olmo_path_remap_from and olmo_path_remap_to must both be set or both empty.")
  return {}


def _detect_resumed_step(config) -> int:
  """Return the step number of the latest checkpoint, or 0 for a fresh run.

  Used so the Grain DataLoader can resume reading at the same offset where
  the model checkpoint was saved (``initial_step = step * batch_size``).
  Uses :class:`etils.epath.Path` so the lookup works against both local
  paths and GCS (``gs://...``) — checkpoints commonly land straight in
  GCS, where ``os.path.isdir`` would silently return False.
  """
  if not getattr(config, "enable_checkpointing", False):
    return 0
  ckpt_dir = getattr(config, "checkpoint_dir", "") or ""
  if not ckpt_dir:
    return 0
  path = epath.Path(ckpt_dir)
  if not path.exists() or not path.is_dir():
    return 0
  steps = [int(p.name) for p in path.iterdir() if p.name.isdigit()]
  return max(steps) if steps else 0


def _make_loader_for_host(
    config,
    *,
    process_indices: List[int],
    seed: int,
):
  """Construct an OLMo grain DataLoader for the current data-loading host."""
  index = load_index(config.olmo_index_path)
  if index.sequence_length != config.max_target_length:
    raise ValueError(
        f"OLMo index sequence_length={index.sequence_length} but "
        f"config.max_target_length={config.max_target_length}. Either rebuild "
        f"the index with the matching seq length or update the config."
    )

  this_proc = jax.process_index()
  shard_index = process_indices.index(this_proc)
  shard_count = len(process_indices)

  per_host_batch = config.global_batch_size_to_load // shard_count
  if per_host_batch * shard_count != config.global_batch_size_to_load:
    raise ValueError(
        f"global_batch_size_to_load={config.global_batch_size_to_load} is not " f"divisible by shard_count={shard_count}"
    )

  # Resume = step counter from the latest checkpoint (if any) × per-host
  # batch. Our sampler is stateless, so this single integer is enough to
  # rejoin the stream — no Grain iterator-state serialization needed.
  resumed_step = _detect_resumed_step(config)
  initial_step = resumed_step * per_host_batch

  max_logging.log(
      f"OLMo grain loader: index={config.olmo_index_path} "
      f"total_instances={index.total_instances:,} "
      f"shard={shard_index}/{shard_count} per_host_batch={per_host_batch} "
      f"seq={index.sequence_length} resumed_step={resumed_step} "
      f"initial_step={initial_step}"
  )

  # Worker count and per-worker buffer reuse the standard grain flags. The
  # ``-1`` value of ``grain_worker_count`` is the auto-tuning sentinel for
  # the standard pipeline; we don't auto-tune yet, so treat it as 0
  # (in-process) for safety.
  worker_count = max(int(getattr(config, "grain_worker_count", 0) or 0), 0)
  worker_buffer = int(getattr(config, "grain_per_worker_buffer_size", 1) or 1)

  return make_olmo_grain_data_loader(
      index,
      seed=seed,
      batch_size=per_host_batch,
      shard_index=shard_index,
      shard_count=shard_count,
      apply_ngram_filter=getattr(config, "olmo_apply_ngram_filter", True),
      shift_to_inputs_targets=True,
      path_remap=_build_path_remap(config),
      grain_worker_count=worker_count,
      grain_worker_buffer_size=worker_buffer,
      initial_step=initial_step,
  )


def make_olmo_grain_train_iterator(config, global_mesh, process_indices):
  """Train iterator for ``dataset_type=olmo_grain``."""
  if not getattr(config, "olmo_index_path", ""):
    raise ValueError(
        "When dataset_type=olmo_grain, please set config.olmo_index_path to "
        "the JSON produced by tools/data_generation/build_olmo_npy_index.py."
    )
  loader = _make_loader_for_host(
      config,
      process_indices=process_indices,
      seed=int(getattr(config, "data_shuffle_seed", 0)),
  )
  return multihost_dataloading.MultiHostDataLoadIterator(
      loader,
      global_mesh,
      config.generate_padding_batch_train,
      expansion_loading_factor_for_grain=config.expansion_factor_real_data,
  )


def make_olmo_grain_eval_iterator(config, global_mesh, process_indices):
  """Eval iterator for ``dataset_type=olmo_grain``.

  Currently reuses the train data with a different seed: the OLMo mix is a
  pretraining corpus with no canonical eval partition, so eval here means
  "deterministic held-out shuffle" rather than "held-out documents". For a
  real eval split, point a future ``config.eval_olmo_index_path`` at a
  separate index built over different files; the rest of this function is
  unchanged.
  """
  if not getattr(config, "olmo_index_path", ""):
    raise ValueError("When dataset_type=olmo_grain, please set config.olmo_index_path.")
  loader = _make_loader_for_host(
      config,
      process_indices=process_indices,
      # Distinct seed so eval doesn't overlap train batch order.
      seed=int(getattr(config, "data_shuffle_seed", 0)) ^ 0x1F1F1F1F,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(
      loader,
      global_mesh,
      config.generate_padding_batch_eval,
      expansion_loading_factor_for_grain=config.expansion_factor_real_data,
  )
