# MTP + CP + Packing

## Summary

Fixes 4 correctness issues when combining Multi-Token Prediction (MTP),
Context Parallelism (AG-CP), and Packing.

## Design Highlights

### 1. CP-Aware Left Shift (`_shift_left_one_cp_aware`)

`jnp.roll(x, -1, axis=1)` only sees the local shard. Under CP the sequence
is split across ranks, so the "right neighbor" of rank k's last token lives
on rank k+1 — unreachable by `jnp.roll`. New function uses
`jax.lax.ppermute` in a backward ring to fetch the cross-rank neighbor and
place it at the last position via `jnp.where` (avoiding `.at[...].set()`
scatter). Degrades to `jnp.roll` when CP=1 or no `"context"` axis.

### 2. Segment-Aware Roll (`roll_and_mask_by_segment`)

Under packing, MTP's left-shift can cross document boundaries (Doc A's last
token shifts into Doc B's first). The new function shifts both `x` and
`segment_ids`, then masks positions where `seg_current != seg_next`
(boundary) or `seg_current == 0` (padding). Falls back to `roll_and_mask`
when `segment_ids=None`. All rolling variables now use this function.

### 3. Target Mask Binary Normalization

Under packing, `targets_segmentation` carries document segment IDs
(1, 2, 3, ...) rather than a binary loss mask. `roll_and_mask_by_segment`
preserves these values through boundary-aware rolling, so
`mtp_xent * target_mask` would give segment-2 tokens 2× loss weight.
`MultiTokenPredictionBlock.__call__` now normalizes `target_mask` to 0/1
before the rolling loop: `target_mask = (target_mask != 0).astype(jnp.int32)`.
Also guards `target_ids` / `target_mask` against `None` (when
`mtp_num_layers=0` and `quantize_model` traces the MTP block).

### 4. Synthetic Data with Packed Segment IDs (`_make_packed_segment_ids`)

Upstream synthetic data's `segment_ids` are always all-ones, making segment
boundary logic untestable. The new function uses a fully vectorized
`jnp.argsort`-based random-split approach (no per-row Python loop) to
generate 2..N random segments per row with sequential integer IDs starting
at 1. `train_utils.py` guard against `synthetic + packing + CP` is removed.

### 5. CP Configuration Guards

Two safety checks in `MultiTokenPredictionBlock.__call__`:

- Reject `context_parallel_load_balance=True` with CP (DUAL_CHUNK_SWAP reorder
  breaks `ppermute`-based neighbor fetch).
- Reject `mtp_num_layers > 1` with CP + packing (segment tracking misalignment
  at layers k ≥ 2 is not yet implemented).

## Files Changed

| File                                          | Change                                                                                               |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `layers/multi_token_prediction.py`            | `_shift_left_one_cp_aware`, `roll_and_mask_by_segment`, target_mask normalization, CP guards, wiring |
| `input_pipeline/synthetic_data_processing.py` | `_make_packed_segment_ids` (vectorized)                                                              |
| `utils/train_utils.py`                        | Remove synthetic+packing+CP guard                                                                    |
| `tests/unit/multi_token_prediction_test.py`   | 20 new tests (segment + CP + packed_ids + mask normalization)                                        |

## Backward Compatibility

- `_shift_left_one_cp_aware`: degrades to `jnp.roll` when CP=1 or no `"context"` axis
- `roll_and_mask_by_segment`: degrades to `roll_and_mask` when `segment_ids=None`
- `roll_and_mask(shift=-1)`: equivalent to original path when CP is off
- `target_mask` normalization: no-op for already-binary masks; handles `None` gracefully
- `synthetic_data_processing`: segment IDs remain `jnp.ones` when `packing=False`
