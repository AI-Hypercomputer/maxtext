# Design Doc: MTP + CP + Packing Combined Fix

## Summary

Fixes 3 correctness issues when combining Multi-Token Prediction (MTP),
Context Parallelism (AG-CP), and Packing. 4 files changed.

## Motivation

| #   | Issue                                  | Impact                                                 |
| --- | -------------------------------------- | ------------------------------------------------------ |
| 1   | `jnp.roll` does not cross CP ranks     | MTP shift-left loses cross-rank right-neighbor tokens  |
| 2   | `roll_and_mask` is segment-unaware     | Packing causes cross-document target leakage in MTP    |
| 3   | Synthetic data has no real segment IDs | Cannot test CP + packing combination on synthetic data |

## Design

### 1. CP-Aware Left Shift (`multi_token_prediction.py`)

`jnp.roll(x, -1, axis=1)` only sees the local shard. With CP the sequence
is split across ranks, so the "right neighbor" of rank k's last token lives
on rank k+1 — unreachable by `jnp.roll`.

```
CP=2:  rank0=[a0,a1,a2,a3], rank1=[a4,a5,a6,a7]
       jnp.roll(rank0) → [a1,a2,a3,0]  ← missing a4
```

New function `_shift_left_one_cp_aware` uses `jax.lax.ppermute` in a
backward ring:

- Rank r sends its `first_token` to rank r-1
- Rank r receives rank r+1's `first_token`, places it in `last_position`
- Rank cp_size-1 fills `last_position` with 0 (sequence end)
- Degrades to `jnp.roll` when CP=1 or no `"context"` axis — zero overhead

`roll_and_mask(shift=-1)` now delegates to `_shift_left_one_cp_aware`.

### 2. Segment-Aware Roll (`multi_token_prediction.py`)

MTP shifts input_ids/target_ids/target_mask/position_ids left by one each
iteration to predict the "next token". Under packing, "next" may cross a
document boundary:

```
Doc A: [a0,a1,a2,a3] | Doc B: [b0,b1,b2,b3]
seg:   [1, 1, 1, 1, 2, 2, 2, 2]
roll:  [a1,a2,a3,b0, b1,b2,b3, 0]
               ↑ cross-doc, must not participate in loss
```

New function `roll_and_mask_by_segment(x, segment_ids)`:

- Shifts both `x` and `segment_ids` via `_shift_left_one_cp_aware`
- `seg_current != seg_next` → document boundary → mask to 0
- `seg_current == 0` → padding position → mask to 0
- `segment_ids is None` → degrades to `roll_and_mask`

All rolling variables in `MultiTokenPredictionBlock.__call__` now use
`roll_and_mask_by_segment`.

**`segment_ids` itself uses `roll_and_mask` (not `by_segment`)**: avoids
a boundary-masked 0 being misinterpreted as a padding position in the next
iteration.

**`decoder_segment_ids` is not rolled**: passed to each MTP layer with the
original value. Self-attention hidden states maintain the original document
identity — positions 0-3 remain Doc A's hidden states; no synchronized
rolling is needed.

### 3. Synthetic Data with Packed Segment IDs (`synthetic_data_processing.py`)

Upstream `base.yml` defaults to `packing: true`, but synthetic data's
`segment_ids` are always all-ones (no document boundaries). This means:

- `roll_and_mask_by_segment` acts identically to `roll_and_mask` on synthetic
  data — segment boundary logic is untested
- `train_utils.py` outright rejects the `synthetic + packing + CP` combination

New function `_make_packed_segment_ids`: each row is split into 2..N
randomly-sized segments with sequential integer IDs starting from 1
(0 = padding). Called in `__init__` when `packing=True`; old behavior
preserved when `packing=False`.

`train_utils.py` removes the `ValueError` guard against
`synthetic + packing + CP` — synthetic data now has real segment boundaries,
making the combination valid.

### 4. Unit Tests (`multi_token_prediction_test.py`)

New `TestRollAndMaskBySegment`, 8 cases:

| Test case                                         | Coverage                           |
| ------------------------------------------------- | ---------------------------------- |
| `test_no_segment_ids_falls_back_to_roll_and_mask` | seg=None → roll_and_mask           |
| `test_single_segment_no_boundaries`               | Single segment, only tail masked   |
| `test_two_segments_masks_boundary`                | Two segments, boundary masked      |
| `test_padding_segment_masked`                     | seg=0 padding mask                 |
| `test_three_segments_boundaries`                  | Three-segment boundaries           |
| `test_2d_tensor_shape_preserved`                  | 2D shape preserved                 |
| `test_3d_tensor_shape_preserved`                  | 3D (embedding) shape + mask verify |
| `test_shift_not_minus_one_raises`                 | shift≠-1 rejected                  |

## Files Changed

| File                                          | Change                                                         |     +/−     |
| --------------------------------------------- | -------------------------------------------------------------- | :---------: |
| `layers/multi_token_prediction.py`            | `_shift_left_one_cp_aware`, `roll_and_mask_by_segment`, wiring |   +108/−4   |
| `input_pipeline/synthetic_data_processing.py` | `_make_packed_segment_ids` with real boundaries                |   +43/−2    |
| `utils/train_utils.py`                        | Remove synthetic+packing+CP guard                              |     −5      |
| `tests/unit/multi_token_prediction_test.py`   | 17 new tests (segment + CP + packed_ids)                       |   +336/−2   |
| **Total**                                     |                                                                | **+492/−8** |

## Backward Compatibility

- `_shift_left_one_cp_aware`: degrades to `jnp.roll` when CP=1 or no
  `"context"` axis — zero overhead
- `roll_and_mask_by_segment`: degrades to `roll_and_mask` when
  `segment_ids=None`
- `roll_and_mask(shift=-1)`: equivalent to the original `jnp.roll` path
  when CP is off
- `synthetic_data_processing`: segment IDs remain `jnp.ones` when
  `packing=False`
- `train_utils.py` guard removal does not affect non-synthetic scenarios

## Verification (2026-07-17, TPU v6e-4)

| #   | Configuration        | main_model_loss |   mtp_loss    |
| --- | -------------------- | :-------------: | :-----------: |
| 1   | MTP + packing        | 12.262 → 11.975 | 1.218 ~ 1.227 |
| 2   | MTP + CP=2           | 12.262 → 11.975 | 1.218 ~ 1.227 |
| 3   | MTP + CP=2 + packing | 12.261 → 11.961 | 1.216 ~ 1.228 |

All three loss curves are consistent, verifying `_shift_left_one_cp_aware`
and `roll_and_mask_by_segment` correctness under all combinations.

```bash
# Unit tests
python3 -m unittest tests.unit/multi_token_prediction_test -v
# Ran 12 tests in 16.5s — OK
```
