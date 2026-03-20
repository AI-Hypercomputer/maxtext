# QKV Projection Weight Permutation: TP Sharding Analysis

## Summary

When converting MaxText attention weights to vLLM's fused `qkv_proj.weight`, the row
ordering of the output matrix depends on the **tensor parallelism (TP) degree** used by
vLLM. The current converter produces a fixed GQA-group interleaving that only matches
TP=4 for Qwen3-30B-A3B. Changing to TP=2 (or any other TP) causes an exact row-block
permutation — same values, same shape, same mean, but `allclose` fails.

---

## 1. Model Dimensions (Qwen3-30B-A3B)

| Parameter       | Value |
|-----------------|-------|
| `d_model`       | 2048  |
| `num_q_heads`   | 32    |
| `num_kv_heads`  | 4     |
| `head_dim`      | 128   |
| `heads_per_group` (Q heads per KV head) | 8 |

### vLLM fused `qkv_proj.weight` shape

```
total_rows = (num_q_heads + num_kv_heads + num_kv_heads) * head_dim
           = (32 + 4 + 4) * 128
           = 5120

qkv_proj.weight: (5120, 2048)
```

Each attention head contributes exactly **128 contiguous rows** (= `head_dim`).

---

## 2. MaxText Source Layout

MaxText stores Q, K, V as separate tensors (scanned over layers):

```
query.kernel:  (d_model, num_layers, num_q_heads,  head_dim) = (2048, 48, 32, 128)
key.kernel:    (d_model, num_layers, num_kv_heads,  head_dim) = (2048, 48,  4, 128)
value.kernel:  (d_model, num_layers, num_kv_heads,  head_dim) = (2048, 48,  4, 128)
```

These are **logical** weights with no TP-dependent ordering.

---

## 3. vLLM Expected Layout

vLLM splits `qkv_proj.weight` along **axis 0 (rows)** into `TP` equal shards.
Each TP rank's shard must contain all Q, K, V heads assigned to that rank,
laid out as:

```
[Q_heads_for_rank, K_heads_for_rank, V_heads_for_rank]
```

Therefore the **full (unsharded)** `qkv_proj.weight` must be ordered as:

```
[Q_rank0, K_rank0, V_rank0,  Q_rank1, K_rank1, V_rank1,  ...,  Q_rank(TP-1), K_rank(TP-1), V_rank(TP-1)]
```

This ordering is **TP-dependent**.

---

## 4. Current (Broken) Converter: GQA-Group Interleaving

The current `_to_attn()` uses GQA-group interleaving:

```
[Q_group0, K0, V0,  Q_group1, K1, V1,  ...,  Q_group(num_kv-1), K(num_kv-1), V(num_kv-1)]
```

where each group has `heads_per_group` Q heads + 1 K head + 1 V head.

### Why it works for TP=4

With TP=4 and `num_kv_heads=4`:
- Each TP rank gets exactly **1 KV group** (1 KV head).
- `kv_per_tp = 4 / 4 = 1`, `q_per_tp = 32 / 4 = 8`.
- GQA-group order == TP-rank order (by coincidence).

### Why it breaks for TP=2

With TP=2 and `num_kv_heads=4`:
- Each TP rank gets **2 KV groups** (2 KV heads).
- `kv_per_tp = 4 / 2 = 2`, `q_per_tp = 32 / 2 = 16`.
- vLLM expects: `[Q0..Q15, K0, K1, V0, V1, Q16..Q31, K2, K3, V2, V3]`
- Converter produces: `[Q0..Q7, K0, V0, Q8..Q15, K1, V1, Q16..Q23, K2, V2, Q24..Q31, K3, V3]`
- Same values, different row-block order → **exact permutation of 128-row blocks**.

---

## 5. Empirical Verification

### Test setup

Saved the full `qkv_proj.weight` (layer 0) for both configurations:
- `vllm_qkv_proj_4_1_1.npy`: vLLM with `[TP=4, DP=1, EP=1]`
- `vllm_qkv_proj_2_2_1.npy`: vLLM with `[TP=2, DP=2, EP=1]`

### Observations

| Metric | Result |
|--------|--------|
| Shape | Both `(5120, 2048)` ✅ |
| Mean | `allclose(mean_a, mean_b) = True` ✅ |
| Element-wise | `allclose(a, b) = False` ❌ |
| Max abs diff | `3.123` |

### Block permutation search

Tested block permutations on axis 0 and axis 1 with block sizes 64, 128, 256, 512, 1024:

```
axis=0, block=64,   exact=True,  maxdiff=0       ✅
axis=0, block=128,  exact=True,  maxdiff=0       ✅
axis=0, block=256,  exact=False, maxdiff=2.929   ❌
axis=1, block=*,    exact=False                   ❌ (all sizes)
```

**Conclusion**: The difference is an **exact permutation of 128-row blocks along axis 0**.
This is precisely head-level reordering (`head_dim = 128`).

### Detected permutation (block=64, 80 blocks)

```python
perm_64 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    32, 33, 36, 37,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    34, 35, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    72, 73, 76, 77,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    74, 75, 78, 79,
]
```

### Interpretation of the permutation

Reading the permutation in block-128 terms (pairs of block-64 indices):

| A block (128) | Content in A (TP=4) | B block mapped from | Content in B (TP=2) |
|---|---|---|---|
| 0-7 | Q0..Q7 (group 0 Q) | 0-7 | Q0..Q7 ✅ same |
| 8-9 | K0, V0 | 16-17 | K0, V0 (moved) |
| 10-17 | Q8..Q15 (group 1 Q) | 8-15 | Q8..Q15 ✅ same offset |
| 18-19 | K1, V1 | 18-19 | K1, V1 (moved) |
| ... | ... | ... | ... |

This confirms the shuffle is between **GQA-group interleave** (current code) and
**TP-rank interleave** (what vLLM expects).

---

## 6. Root Cause

The converter concatenates Q/K/V per **GQA group** (1 KV head at a time):

```python
# Current code (GQA-group interleave)
group = jnp.concatenate([q_grouped, k_grouped, v_grouped], axis=3)
# axis=3 iterates over: [heads_per_group Q, 1 K, 1 V] × num_kv_heads
```

But vLLM expects concatenation per **TP rank** (multiple KV heads per rank when TP < num_kv_heads):

```
Per TP rank: [all Q heads for rank, all K heads for rank, all V heads for rank]
```

---

## 7. Fix

Replace GQA-group interleaving with TP-rank interleaving:

```python
@staticmethod
@jax.jit
def _to_attn(attn: PyTree) -> dict[str, jax.Array]:
    q = jnp.transpose(attn['query']['kernel'], (1, 0, 2, 3))
    k = jnp.transpose(attn['key']['kernel'], (1, 0, 2, 3))
    v = jnp.transpose(attn['value']['kernel'], (1, 0, 2, 3))

    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[3]
    l, d_model = q.shape[0], q.shape[1]

    tp = 2  # TODO: parameterize

    q_per_tp = num_q_heads // tp
    kv_per_tp = num_kv_heads // tp

    # Reshape heads dimension into (tp, heads_per_tp)
    q_by_tp = q.reshape(l, d_model, tp, q_per_tp, head_dim)
    k_by_tp = k.reshape(l, d_model, tp, kv_per_tp, head_dim)
    v_by_tp = v.reshape(l, d_model, tp, kv_per_tp, head_dim)

    # Per TP rank: [Q_heads, K_heads, V_heads]
    qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)

    qkv_flat = qkv_by_tp.reshape(l, d_model, -1)
    qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))

    # ... rest unchanged (o_proj, norms) ...
```

### Correctness for different TP values

| TP | q_per_tp | kv_per_tp | Rows per rank | Total rows |
|----|----------|-----------|---------------|------------|
| 1  | 32       | 4         | 5120          | 5120       |
| 2  | 16       | 2         | 2560          | 5120       |
| 4  | 8        | 1         | 1280          | 5120       |
| 8  | 4        | —         | 640 (if num_kv < 8, need kv replication) | 5120 |

> **Note**: For TP > num_kv_heads, vLLM replicates KV heads across ranks.
> The simple reshape above requires `num_kv_heads % tp == 0`.
> KV head replication (TP=8 with 4 KV heads) would need additional handling.

---

## 8. How to Verify

After applying the fix, run with both TP=4 and TP=2 and compare:

```python
import numpy as np

w_tp4 = np.load("vllm_qkv_proj_4_1_1.npy")
w_tp2 = np.load("vllm_qkv_proj_2_2_1.npy")

# These should NOT be allclose (different TP → different valid orderings)
# But each should match vLLM's own weights for that TP config.
```

The real validation is that `llm.generate()` produces correct output after weight
transfer for each TP configuration.