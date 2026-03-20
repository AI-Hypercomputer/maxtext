# KV Head Replication for TP > num_kv_heads (enable_dp_attention)

## Background

Qwen3-30B-A3B has `num_q_heads=32`, `num_kv_heads=4`, `head_dim=128`.

With `enable_dp_attention=True`, vLLM/tpu-inference allows `TP > num_kv_heads`.
For example, `rollout_tp=8` produces a fused `qkv_proj.weight` of shape `(12288, 2048)`:

$$12288 = (32_Q + 32_K + 32_V) \times 128$$

This means KV heads are replicated from 4 → 32 (replication factor = 8).

---

## Full Permutation Pipeline (per tpu-inference `weight_utils.py`)

Starting from HF `k_proj.weight` shape `(512, 2048)` = `(4 × 128, hidden)`:

| Step | Operation | Shape |
|------|-----------|-------|
| HF checkpoint | — | `(512, 2048)` |
| Reshape (`reshape_keys["k_proj"]`) | `(kv_heads, head_dim, hidden)` | `(4, 128, 2048)` |
| Transpose `(2, 0, 1)` | `(hidden, kv_heads, head_dim)` | `(2048, 4, 128)` |
| `jnp.repeat(..., rep=sharding_size//num_kv_heads, axis=1)` | replicate each KV head | `(2048, 32, 128)` |
| Fuse Q+K+V, transpose | final weight | `(12288, 2048)` |

`jnp.repeat` (not `tile`) is used so that head assignment is contiguous:
each TP rank `r` gets heads `[r*4 : (r+1)*4]`, all belonging to the same original KV head `r // (tp // num_kv_heads)`.

---

## Bug: `broadcast_to` gives wrong GQA grouping

The original code used `jnp.broadcast_to` to replicate KV heads:

```python
# WRONG: every TP rank gets all 4 different KV heads
k_by_tp = jnp.broadcast_to(k[:, :, jnp.newaxis, :, :], (l, d_model, tp, kv_per_tp, head_dim))
```

This gives rank 0 heads [K0, K1, K2, K3] — incorrect. Rank 0's Q heads [Q0–Q3] all belong to GQA group 0 (K0), but they'd be paired with K0, K1, K2, K3 respectively → garbage outputs.

---

## Fix: `jnp.repeat` for correct per-rank KV head assignment

```python
if tp > num_kv_heads:
  rep = tp // num_kv_heads          # e.g. 8 // 4 = 2
  kv_per_tp = num_kv_heads          # keep same per-rank KV count
  k_rep = jnp.repeat(k, rep, axis=2).reshape(l, d_model, tp, 1, head_dim)
  v_rep = jnp.repeat(v, rep, axis=2).reshape(l, d_model, tp, 1, head_dim)
  k_by_tp = jnp.repeat(k_rep, kv_per_tp, axis=3)
  v_by_tp = jnp.repeat(v_rep, kv_per_tp, axis=3)
```

### Why this is correct

`jnp.repeat(k, rep=2, axis=2)` on K with shape `(l, d_model, 4, head_dim)` expands to `(l, d_model, 8, head_dim)` with layout:

```
[K0, K0, K1, K1, K2, K2, K3, K3]
  ^rank0 ^rank1 ^rank2 ^rank3 ...
```

After reshape to `(l, d_model, tp=8, 1, head_dim)` and inner `repeat×4`:

| TP rank | Q heads assigned | KV head after repeat | Correct? |
|---------|-----------------|----------------------|----------|
| 0 | Q[0–3] → GQA group 0 | K0 × 4 | ✓ |
| 1 | Q[4–7] → GQA group 0 | K0 × 4 | ✓ |
| 2 | Q[8–11] → GQA group 1 | K1 × 4 | ✓ |
| 3 | Q[12–15] → GQA group 1 | K1 × 4 | ✓ |
| 4 | Q[16–19] → GQA group 2 | K2 × 4 | ✓ |
| 5 | Q[20–23] → GQA group 2 | K2 × 4 | ✓ |
| 6 | Q[24–27] → GQA group 3 | K3 × 4 | ✓ |
| 7 | Q[28–31] → GQA group 3 | K3 × 4 | ✓ |

Each rank has local Q/K/V ratio = 4/4 = 1 (effectively MHA locally), with the correct KV head for its Q group.

---

## Files Modified

- [`bench_weight_sync.py`](bench_weight_sync.py) — `MaxTextToVLLMConverter._to_attn()`, lines ~299–310.
