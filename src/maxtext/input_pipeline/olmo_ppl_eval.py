# Copyright 2023-2026 Google LLC
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

"""OLMo-core's in-loop perplexity evals (``eval/lm/<set>/CE loss`` and ``PPL``) for the training loop.

OLMo-core's ``LMEvaluator`` runs a ``NumpyPaddedFSLDataset``: every EOS-delimited document (EOS included) becomes one
row, truncated to the sequence length and padded, never packed with other documents. Labels are the inputs shifted
left with the last one ignored, and the per-set metric is the mean of the per-token losses over the positions in
``label_mask``, which includes each document's last token (its loss is 0). So a set's CE is

    sum over documents of the next-token CE of tokens 1 .. L-1   /   sum over documents of L

and PPL = exp(CE). Here each eval batch holds documents of a single set, so the eval step's token-loss sum
(``evaluation/total_loss``, minus the z-loss it includes) is that set's numerator; the denominator is counted on the
host. The router uses the EMO eval pool, as OLMo-core's model in eval mode does.
"""

import os
from typing import Any

import jax
import numpy as np

from maxtext.utils import max_logging


def _read_tokens(path: str) -> np.ndarray:
  """A Dolma-toolkit ``.npy`` token file (raw uint32, no header), from a local path or gs://."""
  if path.startswith("gs://"):
    from etils import epath  # pylint: disable=import-outside-toplevel

    data = epath.Path(path).read_bytes()
    return np.frombuffer(data, dtype=np.uint32)
  return np.fromfile(path, dtype=np.uint32)


def split_documents(tokens: np.ndarray, eos_token_id: int, max_len: int) -> list[np.ndarray]:
  """EOS-delimited documents, EOS included, each truncated to ``max_len`` (OLMo-core's padded eval instances)."""
  ends = np.nonzero(tokens == eos_token_id)[0] + 1
  starts = np.concatenate([[0], ends[:-1]])
  return [tokens[s : min(e, s + max_len)] for s, e in zip(starts, ends) if e > s]


def build_rows(docs: list[np.ndarray], seq_len: int, pad_token_id: int) -> dict[str, np.ndarray]:
  """MaxText batch arrays for one document per row: targets are inputs shifted left, padding carries no weight."""
  n = len(docs)
  inputs = np.full((n, seq_len), pad_token_id, dtype=np.int32)
  lengths = np.array([len(d) for d in docs], dtype=np.int32)
  for i, d in enumerate(docs):
    inputs[i, : len(d)] = d
  pos = np.arange(seq_len, dtype=np.int32)[None, :]
  real = pos < lengths[:, None]
  targets = np.full_like(inputs, pad_token_id)
  targets[:, :-1] = inputs[:, 1:]
  # Position t predicts token t + 1, which exists for t < L - 1.
  has_target = pos < (lengths[:, None] - 1)
  return {
      "inputs": inputs,
      "inputs_position": np.where(real, pos, 0).astype(np.int32),
      "inputs_segmentation": real.astype(np.int32),
      "targets": targets,
      "targets_position": np.where(real, pos, 0).astype(np.int32),
      "targets_segmentation": has_target.astype(np.int32),
  }


class OlmoPplEval:
  """Builds the eval batches once and runs them every ``olmo_ppl_eval_interval`` steps."""

  def __init__(self, config):
    self.interval = int(config.olmo_ppl_eval_interval)
    self.seq_len = int(config.max_target_length)
    # The eval step scores only the first micro_batch_size_to_eval_on rows of a batch (fewer than it loads when
    # eval_per_device_batch_size < 1 or expansion_factor_real_data > 1), so documents go only in those rows; the
    # arrays keep the loaded size, which the input sharding needs, with weightless padding rows after them.
    self.rows_per_batch = int(config.micro_batch_size_to_eval_on)
    self.rows_loaded = int(config.global_batch_size_to_load_eval)
    self.eos = int(config.olmo_ppl_eval_eos_token_id)
    self.pad = int(config.olmo_ppl_eval_pad_token_id)
    # [(label, [(batch arrays, number of documents in it)], denominator tokens)]
    self.sets = []
    for name in config.olmo_ppl_eval_sets:
      path = os.path.join(config.olmo_ppl_eval_dir, name, "val", "part-0-00000.npy")
      docs = split_documents(_read_tokens(path), self.eos, self.seq_len)
      denominator = int(sum(len(d) for d in docs))
      batches = []
      for start in range(0, len(docs), self.rows_per_batch):
        chunk = docs[start : start + self.rows_per_batch]
        rows = build_rows(chunk, self.seq_len, self.pad)
        if len(chunk) < self.rows_loaded:  # weightless padding rows: the last batch's, and any the step doesn't score
          fill = self.rows_loaded - len(chunk)
          rows = {k: np.concatenate([v, np.zeros((fill,) + v.shape[1:], v.dtype)]) for k, v in rows.items()}
        batches.append(rows)
      self.sets.append((f"{name}-validation", batches, denominator))
      max_logging.log(
          f"olmo_ppl_eval: {name}: {len(docs)} documents, {denominator} tokens, {len(batches)} batches of"
          f" {self.rows_per_batch} documents ({self.rows_loaded} rows loaded)"
      )

  def due(self, step: int) -> bool:
    """After the step that completes a multiple of the interval (OLMo-core evaluates after step N, N % interval == 0)."""
    return self.interval > 0 and (step + 1) % self.interval == 0

  def run(self, p_eval_step, state, data_sharding) -> dict[str, Any]:
    """``eval/lm/<label>/CE loss`` and ``.../PPL`` for every set."""
    out = {}
    for label, batches, denominator in self.sets:
      ce_sum = 0.0
      for rows in batches:
        batch = {
            k: jax.make_array_from_callback(v.shape, data_sharding, lambda idx, v=v: v[idx]) for k, v in rows.items()
        }
        scalars = p_eval_step(state, batch)["scalar"]
        total = float(scalars["evaluation/total_loss"])  # token-loss sum, z-loss included
        weights = float(scalars["evaluation/total_weights"])
        z_mean = float(scalars["evaluation/z_loss"])  # per-token z-loss (already weighted by z_loss_multiplier)
        ce_sum += total - z_mean * weights
      ce = ce_sum / max(denominator, 1)
      out[f"eval/lm/{label}/CE loss"] = ce
      out[f"eval/lm/{label}/PPL"] = float(np.exp(ce))
    return out


def ppl_eval_step(eval_step):
  """``eval_step`` with the OLMoE3 router on its EMO eval pool, as OLMo-core evaluates in eval mode."""
  from maxtext.models import olmoe3  # pylint: disable=import-outside-toplevel

  def step(*args, **kwargs):
    with olmoe3.emo_eval_pools():
      return eval_step(*args, **kwargs)

  step.__name__ = "ppl_eval_step"
  return step
