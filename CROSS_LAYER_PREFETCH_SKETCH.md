# Cross-layer MoE-gather prefetch — design sketch (Tip 1)

**Goal.** Hide the MoE weight all-gather by prefetching **layer i+1's** gather during **layer i's**
attention/QKV matmuls — software pipelining across the layer (scan) boundary, so the gather always
overlaps the *previous* layer's compute instead of being on its own layer's critical path.

**Why this and not the splash window:** the gather hides behind **matmuls** (proven: forward QKV
overlap). Layer i's attention QKV matmuls are a clean, big matmul window that is *independent* of
layer i+1's expert weights → ideal cover. The only reason it doesn't happen today is that layer
i+1's gather isn't *emitted* until layer i+1.

---

## The constraint: `nn.scan`

`decoders.py` scans `DeepSeekMoELayerToLinen` via `nn.scan`; the layer params are stacked
`[num_layers, ...]` and **each iteration sees only layer i's slice**. There is no native handle on
layer i+1's params inside iteration i. So "tie i+1's gather to i's QKV" is **not expressible as a
plain annotation** inside the scan body (the body is traced once; i+1 doesn't exist in scope).

Two tiers:

### Tier A — annotation-only (reviewer's "try first"): **likely a dead end in a scan**
Tagging "layer i+1's gather" with layer i's `_scheduling_group_id` requires i+1's gather op to exist
in i's trace. That only happens if `scan_layers=False` (fully unrolled, all ~61 layers in one
trace) — at which point the annotation + `xla_tpu_enable_layer_scheduler_for_dependent_collectives`
*might* let XLA prefetch across the boundary. **Cost:** unrolling 61 layers = huge HLO, slow compile,
more live memory. Probably impractical, but a 2-layer unroll could be a cheap *existence test* of
whether XLA will prefetch at all. Recommend only as a probe.

### Tier B — software-pipelined scan carry (the real version) — **DECIDED scope below**
**DECISIONS (from review):** prefetch **only w0/w1 (up-projection)**, forward-only. `wo`
(down-projection) is **NOT** prefetched — it's not needed until after the up-GMM + activation, so its
gather is launched *in-layer i* and hides behind layer i's own w0/w1 matmuls (as it does today). This
halves the prefetch HBM (~6 GB → **~4 GB**, w0+w1 only) and avoids the OOM/memory-panic risk.

Restructure the scan to **carry the already-gathered w0/w1** and **gather them one layer ahead**:

```
prologue:  g01_0 = gather(W01_sharded[0])                  # pipeline fill: just w0,w1 of layer 0

scan body (iteration i), carry = g01_i (gathered w0,w1 for THIS layer):
    # 1) consume the PREFETCHED up-proj weights — no exposed w0/w1 gather on this layer's crit path
    out_i = attention(x_i)
    wo_i  = gather(Wo_sharded[i])                          # IN-LAYER; hides behind layer i's up-GMM
    moe(out_i, pregathered=(g01_i.w0, g01_i.w1, wo_i))
    # 2) PREFETCH next layer's w0/w1 — emitted here, overlaps THIS layer's QKV matmuls (the hider)
    g01_next = gather(W01_sharded[i+1])                    # async; hides behind attention(x_i)
    return out_i, carry=g01_next

epilogue:  last iteration's prefetch targets a non-existent layer -> mask/no-op
```
Backward stays on the current in-layer recompute for now (see Q4) — bank the forward win first.

**Mechanics — how iteration i gets `W_sharded[i+1]`:**
- The stacked sharded MoE weights `W_sharded[L, ...]` are passed as an **explicit scan input (`xs`)
  shifted by one**: `W_next = concat(W_sharded[1:], pad_last)` along the layer axis, so iteration i
  receives `W_sharded[i+1]`.
- The scan's normal param axis still gives layer i's own params (attention etc.); the shifted `xs`
  is *only* the sharded expert weights for the prefetch.
- The carry threads the **gathered** weights (g) forward; the gather of g_next is what overlaps.

This is the standard prefetch/pipeline-in-scan pattern; no kernel changes, no batch/seq reshape.

---

## The catch (and the open questions) — HBM

Prefetching means **two layers' gathered expert weights are live at once** during the overlap:
layer i's g_i (being consumed) + layer i+1's g_next (being gathered). Gathered w0+w1+wo ≈ **~6 GB**.
So +~6 GB held across a layer.

We are **HBM-bound and near the ceiling** (kernel run ~89 G; jaxsplash OOM'd at 110 G vs 94.75 G
compile limit). So Tier B risks (a) **OOM**, and (b) the **same scheduler HBM-pressure wall** we hit
all day — XLA may refuse to hold the prefetched weight early for exactly the HBM reason. **The
scheduler-flag sweep running now (`multi_compute_overlap`, shared-mem limit) is probing that very
wall** — its result tells us whether prefetch is even viable before we build it.

Mitigations: prefetch **only w0** (or w0+w1), ~2–4 GB instead of 6; or free HBM elsewhere.

---

## Open questions — RESOLVED (review, 2026-06-20)

1. **HBM headroom.** Bounded by Q2: prefetch holds **only w0/w1 ≈ ~4 GB** (not ~6 GB). Still near the
   ceiling → verify peak at compile. The scheduler-flag sweep (running) reads whether the
   HBM-pressure wall is movable.
2. **Scope → w0/w1 (up-projection) only.** Do NOT cross-layer-prefetch `wo`: down-projection isn't
   needed until after the up-GMM + activation, so its gather is launched **in-layer i** and hides
   behind layer i's own w0/w1 matmuls (as today). Halves prefetch HBM (~6 → ~4 GB), avoids the panic.
3. **Sequencing.** Build **forward-only** (clean scan-carry mod); read the flag-sweep result first
   (cheap) since both touch the same HBM-pressure scheduling.
4. **Forward-only first — without question.** The bwd is shifted async-start/done boundaries inside
   `custom_vjp` *and* much higher peak memory (activation grads + reduce-scatter buffers); a bwd OOM
   would mask whether the forward overlap actually worked. **Bank the forward win, check the trace,
   then tackle the backward.**
