# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from maxtext.kernels.causal_conv1d_gated_delta_rule import config
from maxtext.kernels.causal_conv1d_gated_delta_rule import memory_ref


def compute_batched_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    end_seq: jax.Array,
) -> memory_ref.MetadataRef:
  """Metadata for computing multiple sequences per tile."""

  max_seqs = seq_lens.size
  all_seqs = jnp.arange(max_seqs)

  # NOTE: Only supports use case where query_lens[i] = 1 where i < end_seq.
  # This must be guaranteed by the function caller.
  # TODO: Add error handling when above condition is not met.
  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  is_valid_seqs = jnp.where(all_seqs < end_seq, True, False)
  has_initial_state = (seq_lens - query_lens) > 0
  all_valid_seqs = jnp.where(is_valid_seqs, all_seqs, 0)

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=pl.cdiv(end_seq, cfg.tile_size),
      p_id_to_s_idx=all_valid_seqs,
      p_id_to_r_base=all_valid_seqs,
      p_id_to_r_size=jnp.where(is_valid_seqs, 1, 0),
      p_id_is_first_tile=is_valid_seqs,
      p_id_is_last_tile=is_valid_seqs,
      s_idx_has_initial_state=has_initial_state,
      s_idx_to_state_indices=state_indices,
  )


def compute_per_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
) -> memory_ref.MetadataRef:
  """Metadata for computing single sequence per tile."""

  max_seqs = seq_lens.size
  max_tokens = cfg.batch_size
  all_seqs = jnp.arange(max_seqs)
  all_tokens = jnp.arange(max_tokens)

  # Shift to ensure first element is for start_seq.
  query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
  seq_lens = jnp.roll(seq_lens, shift=-start_seq)
  state_indices = jnp.roll(state_indices, shift=-start_seq)

  query_lens = query_start_loc[1:] - query_start_loc[:-1]
  # NOTE: query_lens is used for calculating num_tiles. Defensive programming
  # that masks out all the other values (seq_lens, state_indices) are not needed
  # since they will not be visited as long as num_tiles is correct.
  num_seqs = end_seq - start_seq
  query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

  # Calculate number of tiles needed for each sequence.
  s_idx_to_num_tiles = pl.cdiv(query_lens, cfg.chunk_size)
  # Calculate starting p_id of each sequence.
  s_idx_to_start_p_id = jnp.cumulative_sum(
      s_idx_to_num_tiles, include_initial=True
  )
  # Map tile index to seq index.
  # Consider following case:
  # all_seqs = [0 1 2 3 4]
  # s_idx_to_num_tiles = [1 2 3 0 1]
  # jnp.repeat will return following results:
  # p_id_to_s_idx = [0 1 1 2 2 2 4]
  # This means p_id_to_s_idx[i] will point to its corresponding seq index.

  # NOTE: To make jnp.repeat jit compilable, we add total_repeat_length. This
  # introduces padding to p_id_to_s_idx[i] where i >= num_tiles. Since the
  # kernel only checks value up-to p_id_to_s_idx[num_tiles-1], padded value
  # will not impact kernel execution.
  p_id_to_s_idx = jnp.repeat(
      all_seqs, s_idx_to_num_tiles, total_repeat_length=max_tokens
  )
  # Map program id (p_id) to tile id of a sequence.
  p_id_to_t_id = all_tokens - s_idx_to_start_p_id[p_id_to_s_idx]
  # Map tile index to starting row of its activation.
  p_id_to_r_base = (
      query_start_loc[p_id_to_s_idx] + p_id_to_t_id * cfg.chunk_size
  )
  # Calculate number of rows to calculate / fetch for each tile.
  p_id_to_r_size = jnp.minimum(
      query_start_loc[p_id_to_s_idx + 1] - p_id_to_r_base,
      cfg.tile_size,
  )

  # Calculate predicate used for state DMA. State is read if program id (p_id)
  # is the first tile of a sequence and the sequence had been computed before
  # (chunked prefill, decode, etc). State is written if the program id is the
  # last tile of a sequence.
  has_initial_state = (seq_lens - query_lens) > 0
  p_id_is_first_tile = p_id_to_t_id == 0
  p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)

  # NOTE: Since query_lens[i] = 0 where i >= num_seqs, s_idx_to_num_tiles[i]
  # where i >= num_seqs will also be 0. Therefore, s_idx_to_num_tiles.sum()
  # will contain number of tiles for valid sequence.
  num_tiles = s_idx_to_num_tiles.sum()

  return memory_ref.MetadataRef.create(
      cfgs=cfg,
      num_tiles=num_tiles,
      p_id_to_s_idx=p_id_to_s_idx,
      p_id_to_r_base=p_id_to_r_base,
      p_id_to_r_size=p_id_to_r_size,
      p_id_is_first_tile=p_id_is_first_tile,
      p_id_is_last_tile=p_id_is_last_tile,
      s_idx_has_initial_state=has_initial_state,
      s_idx_to_state_indices=state_indices,
  )
