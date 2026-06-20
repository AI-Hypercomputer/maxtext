# Cross-layer w0/w1 prefetch — file-by-file implementation plan (branch: cross-layer-prefetch)

Status: **branch scaffolded; full refactor staged for a validation-checkpointed build** (the param
lift through the nnx↔linen bridge + `nn.scan` is high-risk to land blind — see §Risks). This doc is
the precise plan; flag is added; each step below is meant to be AOT-validated before the next.

Goal (decided scope): forward-only, **w0/w1 only** (~4 GB), `wo` stays in-MoE. Prefetch layer i+1's
w0/w1 gather during layer i's QKV matmuls via the canonical lift-to-parent + shifted-`xs` + carry
software pipeline. We're forced into the **explicit** pipeline because the scheduler-flag sweep was
null (XLA won't relax the HBM-pressure schedule voluntarily — see WAG writeup §7).

## Why the lift is unavoidable
`RoutedMoE.__init__` creates `self.wi_0/self.wi_1` as `nnx.Param` (moe.py:555). Under `nn.scan`
(`variable_axes={"params": param_scan_axis}`, decoders.py:574) these are stacked `[L, ...]` and
**owned by the scanned module** — the parent can't read them to construct a shifted `xs` (the params
don't exist until the scan inits). To pass layer i+1's *sharded* weights into iteration i we must
make them a **parent** param. (Carry can only thread the past; `xs` provides the future slice; a
broadcast of the full stack needs an iteration index AND duplicates HBM. So: lift.)

## File-by-file

### 1. `configs/types.py` + `base.yml` — flag (DONE in scaffold)
`moe_xlayer_prefetch: bool = False` — gate the whole path; default off so the model is unchanged.

### 2. `layers/moe.py` — make `RoutedMoE` accept w0/w1 instead of owning them (when flag on)
- In `__init__`: `if cfg.moe_xlayer_prefetch: skip creating self.wi_0/self.wi_1` (still create `wo`).
- In `gather_routed_weights()` / `sparse_matmul`: when prefetch on, w0/w1 come from an argument
  (the carried, already-gathered weights), NOT `self.wi_0`. `wo` gathers in-layer as today.
- Keep the `pregathered_weights` path (already exists) as the injection point.

### 3. `models/deepseek.py` — thread the prefetched w0/w1 through the layer
- `DeepSeekMoELayer.__call__` gains `prefetched_w01` (the carry, gathered) + `next_w01_sharded`
  (the shifted `xs`, sharded). Body:
  - use `prefetched_w01` as this layer's w0/w1 (already gathered → no exposed gather here);
  - `next_w01_gathered = gather(next_w01_sharded)` — emitted at the attention input → overlaps QKV;
  - return `next_w01_gathered` as the new carry (alongside the normal layer output).
- Forward-only: `fused_bwd` unchanged (recomputes in-layer as today). Backward prefetch deferred.

### 4. `layers/decoders.py` — the parent: stacked param, prologue, shifted `xs`, carry
- Create the stacked parent param `W01_sharded` `[L, E, embed, 2*mlp]` (or two `[L, E, embed, mlp]`).
- **Prologue:** `g0 = gather(W01_sharded[0])` before the scan (pipeline fill).
- **Shifted xs (pad, NOT lax.cond — reviewer #3):**
  `W01_shifted = jnp.concatenate([W01_sharded[1:], W01_sharded[-1:]], axis=0)` → pass via `in_axes`.
- **Carry:** thread `g_i` (gathered w0/w1 for the current layer); init = `g0`; discard final carry.
- `nn.scan` gains the carry + the `W01_shifted` scanned input; `variable_broadcast` the prologue if
  needed.

### 5. Checkpoint shim (reviewer #1) — **NOT needed for our benchmark**
Our runs use `dataset_type=synthetic`, `enable_checkpointing=False` → params are randomly inited in
the new structure; no restore. (For real checkpoints later: a one-time shim that stacks per-layer
`wi_0/wi_1` into `W01_sharded` before load.)

### 6. SC queue (reviewer #2): do nothing
`wo_i` (in-layer) and `g_next` (prefetch) are both SC-eligible during attention; let XLA queue them.
No manual ordering.

## Validation ladder (each is ~3 min local AOT)
1. Flag on, prefetch path inert (params still created) → compiles == baseline.
2. Lift w0/w1 to parent, no prefetch (just use parent param in-layer) → compiles, bit-exact loss.
3. Add prologue + shifted xs + carry, wire the gather → compiles; AOT HLO shows the gather emitted
   before the splash of the *previous* iteration.
4. Cluster chunk=1: step time vs ds-hw-c1 (16.32) + bit-exact loss + peak HBM (watch the ~4 GB bump).

## Risks (why this is checkpoint-validated, not blind)
- **nnx↔linen bridge + scan fragility** — lifting an `nnx.Param` out of a bridged, scanned module is
  exactly the area the memory flags as broken (`nnx.scan` nesting). Step 2 is the make-or-break AOT.
- **Sharding** — `W01_sharded` must keep `wi_kernel_axes` (`exp, embed_moe, mlp_moe`) on the parent
  with the layer axis prepended; the shifted `xs` must not reshard.
- **HBM** — +~4 GB held across a layer; near the ceiling (jaxsplash OOM'd at 110 G). Watch step-4 peak.
- **Init RNG** — the parent param init must reproduce per-layer init (else loss differs); use the
  same `kernel_init` folded by layer index to stay bit-exact vs the scanned init.
