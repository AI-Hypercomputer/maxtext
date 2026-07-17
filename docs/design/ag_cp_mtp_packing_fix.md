# Design Doc: MTP + CP + Packing 组合修复

## Summary

修复 Multi-Token Prediction (MTP)、Context Parallelism (AG-CP)、Packing 组合场景下的 4 个正确性问题 + 1 个上游 bug，共 5 个文件改动。

## Motivation

| # | 问题 | 影响 |
|---|------|------|
| 1 | `jnp.roll` 不跨 CP rank 边界 | CP 下 MTP 的 shift-left 丢失跨 rank 右邻居 token |
| 2 | `roll_and_mask` 不感知 segment 边界 | Packing 下 MTP 产生跨 doc 的 target 泄漏 |
| 3 | synthetic data 无真实 segment_ids | 无法在 synthetic 数据上测试 CP + packing 组合 |
| 4 | 无组合测试覆盖 | 未来 regression 无法防御 |
| 5 | MTP loss 始终为 0 | `nnx.pop` 重复执行覆盖正确数据 |

## Design

### 1. CP-Aware Left Shift (`multi_token_prediction.py`)

`jnp.roll(x, -1, axis=1)` 只看本地 shard。CP 把序列分片到多个 rank 后，rank k 最后一个 token 的"右邻居"在 rank k+1，但 `jnp.roll` 拿不到。

```
CP=2:  rank0=[a0,a1,a2,a3], rank1=[a4,a5,a6,a7]
       jnp.roll(rank0) → [a1,a2,a3,0]  ← 缺 a4
```

新增 `_shift_left_one_cp_aware`，用 `jax.lax.ppermute` 做 backward ring：
- rank r 把自己的 `first_token` 发给 rank r-1
- rank r 接收 rank r+1 的 `first_token`，填入自己 `last_position`
- rank cp_size-1 的 last_position 填 0（序列末尾）
- 无 CP 或 cp_size=1 时退化为 `jnp.roll`，零开销

`roll_and_mask(shift=-1)` 内部改为调用 `_shift_left_one_cp_aware`。

### 2. Segment-Aware Roll (`multi_token_prediction.py`)

MTP 每轮对 input_ids/target_ids/target_mask/position_ids 左移一位，预测"下一个 token"。在 packing 下，"下一个"可能跨越到另一个文档：

```
Doc A: [a0,a1,a2,a3] | Doc B: [b0,b1,b2,b3]
seg:   [1, 1, 1, 1, 2, 2, 2, 2]
roll:  [a1,a2,a3,b0, b1,b2,b3, 0]
               ↑ 跨 doc，不应参与 loss 计算
```

新增 `roll_and_mask_by_segment(x, segment_ids)`：
- 对 x 和 segment_ids 同时 `_shift_left_one_cp_aware`
- `seg_current != seg_next` → 跨文档边界 → mask 为零
- `seg_current == 0` → padding 位置 → mask 为零
- `segment_ids is None` 时退化为 `roll_and_mask`

`MultiTokenPredictionBlock.__call__` 中所有 rolling 变量改用 `roll_and_mask_by_segment`。

**segment_ids 自身用 `roll_and_mask`（不用 by_segment）**：避免跨边界被 mask 为 0 后下一轮把后续正常位置也误判为边界。

**`decoder_segment_ids` 不滚**：传给 MTP layer 时用原始值。Self-attention 的 hidden state 维持原始 doc identity——pos 0-3 仍是 doc A 的 hidden state，不需要同步滚动。

### 3. Synthetic Data with Packed Segment IDs (`synthetic_data_processing.py`)

上游 `base.yml` 中 `packing: true` 是默认值，但 synthetic data 的 segment_ids 始终全为 1（无文档边界），导致：

- `roll_and_mask_by_segment` 在 synthetic 数据上等同于 `roll_and_mask`，测不到 segment 边界逻辑
- `train_utils.py` 直接拒绝 `synthetic + packing + CP` 组合

新增 `_make_packed_segment_ids`：每行随机切分为 2~N 个变长段，赋值递增整数 segment ID，从 1 开始（0 为 padding）。packing 模式下 `__init__` 调用之，非 packing 保持原行为。

`train_utils.py` 删除 `synthetic + packing + CP` 的 `ValueError` 拦截——synthetic data 现在有真实 segment 边界，组合合法。

### 4. 单元测试 (`multi_token_prediction_test.py`)

新增 `TestRollAndMaskBySegment`，8 个用例：

| 用例 | 覆盖 |
|------|------|
| `test_no_segment_ids_falls_back_to_roll_and_mask` | seg=None → roll_and_mask |
| `test_single_segment_no_boundaries` | 单段仅尾部 mask |
| `test_two_segments_masks_boundary` | 两段跨边界 mask |
| `test_padding_segment_masked` | seg=0 padding mask |
| `test_three_segments_boundaries` | 三段边界 |
| `test_2d_tensor_shape_preserved` | 2D shape 保持 |
| `test_3d_tensor_shape_preserved` | 3D (embedding) shape + mask 验证 |
| `test_shift_not_minus_one_raises` | shift≠-1 拦截 |

### 5. MTP Loss 为 0 的上游 bug 修复 (`train.py`)

`mtp_losses` / `mtp_acceptance` 是 `nnx.Intermediate` 的子类。`nnx.pop(nnx.Intermediate)` 已经把子类实例全部捕获进 `intermediate_outputs["mtp_block"]`。后续 `nnx.pop(mtp_losses)` 拿到空 dict，覆盖 `intermediate_outputs["mtp_losses"] = {}` → `calculate_mtp_loss` 一直返回 0。

修复：从已捕获的 `intermediate_outputs["mtp_block"]` 中提取 `losses`/`weights` → `mtp_losses`，`mtp_preds`/`mtp_mask` → `mtp_acceptance`。

修复前 `mtp_loss: 0.000`，修复后 `mtp_loss: 1.227`（→ `ln(vocab_size) * scaling_factor`）。

## Files Changed

| 文件 | 改动 | +/− |
|------|------|:---:|
| `layers/multi_token_prediction.py` | `_shift_left_one_cp_aware`, `roll_and_mask_by_segment`, 调用切换 | +104/−4 |
| `trainers/pre_train/train.py` | 修复 MTP loss 始终为 0 | +10/−10 |
| `input_pipeline/synthetic_data_processing.py` | `_make_packed_segment_ids` 生成真实段边界 | +43/−2 |
| `utils/train_utils.py` | 删除 synthetic+packing+CP 拦截 | −5 |
| `tests/unit/multi_token_prediction_test.py` | 8 个 `roll_and_mask_by_segment` 测试 | +87/−2 |
| **合计** | | **+245/−20** |

## Backward Compatibility

- `_shift_left_one_cp_aware`：CP=1 或无 `"context"` axis 时退化 `jnp.roll`，零开销
- `roll_and_mask_by_segment`：`segment_ids=None` 退化 `roll_and_mask`
- `roll_and_mask(shift=-1)`：无 CP 时等价于原 `jnp.roll` 路径
- `synthetic_data_processing`：非 packing 模式 segment 保持 `jnp.ones`
- `train_utils.py` 删除拦截不影响非 synthetic 场景

## 验证 (2026-07-17, TPU v6e-4)

| # | 配置 | main_model_loss | mtp_loss |
|---|------|:---:|:---:|
| 1 | MTP + packing | 12.262 → 11.975 | 1.218 ~ 1.227 |
| 2 | MTP + CP=2 | 12.262 → 11.975 | 1.218 ~ 1.227 |
| 3 | MTP + CP=2 + packing | 12.261 → 11.961 | 1.216 ~ 1.228 |

三组 loss 曲线一致，验证 `_shift_left_one_cp_aware` 和 `roll_and_mask_by_segment` 在所有组合下正确。

```bash
# 单元测试
python3 -m unittest tests.unit.multi_token_prediction_test -v
# Ran 12 tests in 16.5s — OK
```