# Intuitive Explanation: w13_weight — Fusing Gate & Up Projections

## Background: What are gate and up projections?

In a Mixture-of-Experts (MoE) layer, each expert has an MLP with three linear projections:

```
x  →  gate_proj(x) → σ(·)  ┐
x  →  up_proj(x)            ├→  (σ(gate) * up)  →  down_proj  →  output
```

In MaxText's storage format:
- `wi_0` (gate weights) for one layer: shape `(128, 2048, 768)` — `(experts, d_model, d_inner)`
- `wi_1` (up weights) for one layer:   shape `(128, 2048, 768)` — `(experts, d_model, d_inner)`

So for each expert: a `2048→768` linear projection for gate, and a `2048→768` linear for up.

vLLM **fuses** these into a single `w13_weight` of shape `(128, 2048, 1536)` — `(experts, d_model, 2*d_inner)` — to run both projections in one matrix multiply.

---

## The Challenge: Tensor Parallelism (TP)

When `tp=2`, vLLM splits the fused weight across 2 devices along the `d_inner` axis.

- **TP rank 0** will compute with rows `[0:768]` of the fused `(1536,)` output dimension.
- **TP rank 1** will compute with rows `[768:1536]` of the fused output dimension.

Each TP worker needs **both its gate slice AND its up slice** to compute `σ(gate) * up` locally, without any cross-device communication mid-layer.

This means the layout must be:

```
[gate_rows_for_TP0 | up_rows_for_TP0 | gate_rows_for_TP1 | up_rows_for_TP1]
```

Not the naive concatenation `[all_gate | all_up]`.

---

## Step-by-Step: Concrete Example

**Setup:** 1 layer, 128 experts, `d_model=2048`, `d_inner=768`, `tp=2`

```
wi_0 (gate): (128, 2048, 768)   ← expert × d_model × d_inner
wi_1 (up):   (128, 2048, 768)
```

### Step 1 — Transpose to row-oriented view

Transpose from `(e, d_model, d_inner)` to `(e, d_inner, d_model)`:

```
gate: (128, 768, 2048)   ← now 768 "rows", each of length 2048
up:   (128, 768, 2048)
```

Think of each expert as having 768 row vectors of dimension 2048.
These rows are what get split across TP workers.

### Step 2 — Split rows into TP chunks

`chunk_size = 768 // 2 = 384`

```
gate.reshape(128, 2, 384, 2048):
  gate_chunk[tp=0] = gate rows [  0:384]   (128 experts × 384 rows × 2048)
  gate_chunk[tp=1] = gate rows [384:768]

up.reshape(128, 2, 384, 2048):
  up_chunk[tp=0]   = up rows   [  0:384]
  up_chunk[tp=1]   = up rows   [384:768]
```

### Step 3 — Interleave gate and up chunks per TP rank

Stack gate and up along a new axis, then flatten:

```
stack([gate_chunks, up_chunks], axis=2)
  → shape: (128, 2, 2, 384, 2048)
             e   tp g/u chunk d_model

reshape to (128, 1536, 2048):
  Row layout for each expert (768→1536 rows):
  ┌─────────────────────────────────────────────────────┐
  │  gate_chunk0   rows [   0: 384]  (for TP rank 0)   │
  │  up_chunk0     rows [ 384: 768]  (for TP rank 0)   │
  │  gate_chunk1   rows [ 768:1152]  (for TP rank 1)   │
  │  up_chunk1     rows [1152:1536]  (for TP rank 1)   │
  └─────────────────────────────────────────────────────┘
```

### Step 4 — Final transpose to match vLLM's expected layout

vLLM >= 0.17 wants `(e, d_model, 2*d_inner)`, so transpose back:

```
(128, 1536, 2048)  →  transpose(0, 2, 1)  →  (128, 2048, 1536)
```

**Final `w13_weight` per layer: `(128, 2048, 1536)`**

---

## Summary Table

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `wi_0` input | `(128, 2048, 768)` | MaxText gate weight: `(e, d_model, d_inner)` |
| `wi_1` input | `(128, 2048, 768)` | MaxText up weight: `(e, d_model, d_inner)` |
| After transpose | `(128, 768, 2048)` | Row-oriented: `(e, d_inner, d_model)` |
| After chunking | `(128, 2, 384, 2048)` | `(e, tp, chunk, d_model)` |
| After interleave | `(128, 1536, 2048)` | Rows: G₀ U₀ G₁ U₁ |
| Final `w13_weight` | `(128, 2048, 1536)` | vLLM format: `(e, d_model, 2*d_inner)` |

---

## Why not just `concat(gate, up)`?

Naive concat would give layout `[G₀ G₁ | U₀ U₁]`:

```
TP rank 0 gets rows [0:768]  →  all of gate  (no up!)
TP rank 1 gets rows [768:1536] →  all of up  (no gate!)
```

Each worker would have only gate or only up — it couldn't compute `σ(gate) * up` locally. The interleaved layout `[G₀ U₀ G₁ U₁]` ensures every TP rank holds its paired gate+up slice.

---

## Extension: TPU GMM Backend — Padding for Hardware Alignment

The `process_w13_for_gmm` function (used in the `GMM_TP` backend) adds another wrinkle on top of the interleaving above: **each per-TP chunk must be a multiple of 128** for the TPU GMM kernel to work efficiently.

### Why 128?

The TPU GMM kernel tiles computation in 128-element blocks in VMEM. If a chunk dimension isn't a multiple of 128, the hardware wastes a partial tile. So the framework pads each chunk up to the next multiple of 128 before sharding.

### Step-by-Step: `intermediate_size=768`, `tp=4`

```
local_intermediate_size = 768 // 4 = 192
align_to(192, 128) = 256      ← next multiple of 128 above 192
pad_amount         = 256 - 192 = 64
padded_intermediate_size = 256 * 4 = 1024
```

**Inside `_pad_tensor` for w1 (shape `(128, 2048, 768)`):**

```
reshape: (128, 2048, 768) → (128, 2048, 4, 192)   ← expose 4 TP chunks
pad:     (128, 2048, 4, 192) → (128, 2048, 4, 256)  ← 64 zeros appended to each chunk
reshape: → (128, 2048, 1024)
```

Positions `[192:256]` within each chunk are **zeros** (`jnp.pad` defaults to `constant_values=0`). Same happens to w3.

**After pad+concat+reorder:**

```
[w1_c0(256) | w3_c0(256) | w1_c1(256) | w3_c1(256) | w1_c2(256) | w3_c2(256) | w1_c3(256) | w3_c3(256)]
total: 2048 output rows
```

Each TP rank gets a contiguous 512-wide slice: 256 gate rows (192 real + 64 zeros) + 256 up rows (192 real + 64 zeros). Final `w13_weight` shape: `(128, 2048, 2048)`.

### Padding across different TP values (`intermediate_size=768`)

| TP | `local = 768/tp` | `align_to(local, 128)` | `pad_amount` | final `2×padded×tp` |
|----|-----------------|------------------------|--------------|---------------------|
| 2  | 384             | 384 (already aligned)  | 0            | 1536                |
| 4  | 192             | 256                    | 64           | 2048                |
| 6  | 128             | 128 (already aligned)  | 0            | 1536                |
| 8  | 96              | 128                    | 32           | 2048                |

TP=2 and TP=6 need no padding at all (384 and 128 are already multiples of 128). The padding only activates when `intermediate_size / tp` falls between multiples of 128.

### `align_to` is just "next multiple of 128 ≥ x"

| Range of `local` | Aligned to |
|-----------------|-----------|
| 1 – 128         | 128       |
| 129 – 256       | 256       |
| 257 – 384       | **384**   |
| 385 – 512       | 512       |

So yes, 384 is reachable — e.g. `intermediate_size=1152`, `tp=4` → `local=288` → aligned to **384**.
