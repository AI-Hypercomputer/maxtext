# MTP + CP + Packing

## Design Highlights

### 1. CP-Aware Left Shift (`_shift_left_one_cp_aware`)

Under context parallelism the sequence is split across ranks. `jnp.roll`
only operates within the local shard, so new function `_shift_left_one_cp_aware`
uses `jax.lax.ppermute` in a backward ring to fetch the cross-rank neighbor.
Falls back to `jnp.roll` when CP=1 or no `"context"` axis.

### 2. Segment-Aware Roll (`roll_and_mask_by_segment`)

With packing, a naive left-shift can cross document boundaries. New function
`roll_and_mask_by_segment` shifts both the tensor and `segment_ids`, then masks
positions where `seg_current != seg_next` (boundary) or `seg_current == 0`
(padding). Falls back to `roll_and_mask` when `segment_ids=None`.

### 3. Target Mask Binary Normalization

Under packing, `targets_segmentation` may carry segment IDs (1, 2, 3, ...)
rather than a 0/1 mask. `MultiTokenPredictionBlock.__call__` normalizes
`target_mask` to binary before the rolling loop with
`target_mask = (target_mask != 0).astype(jnp.int32)`. Also guards against
`None` in the `mtp_num_layers=0` quantization trace path.

### 4. Synthetic Data with Packed Segment IDs (`_make_packed_segment_ids`)

`_make_packed_segment_ids` generates synthetic segment IDs with 2..N random
segments per row via a vectorized `jnp.argsort`-based approach. Used when
`packing=True`.

### 5. CP Configuration Guards

- Reject `context_parallel_load_balance=True` with CP.
- Reject `mtp_num_layers > 1` with CP + packing.

## Files Changed

| File                                          | Change                                                                                               |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `layers/multi_token_prediction.py`            | `_shift_left_one_cp_aware`, `roll_and_mask_by_segment`, target_mask normalization, CP guards, wiring |
| `input_pipeline/synthetic_data_processing.py` | `_make_packed_segment_ids` (vectorized)                                                              |
| `utils/train_utils.py`                        | Remove synthetic+packing+CP guard                                                                    |
| `tests/unit/multi_token_prediction_test.py`   | 20 new tests (segment + CP + packed_ids + mask normalization)                                        |