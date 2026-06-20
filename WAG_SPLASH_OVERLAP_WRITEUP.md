# Overlapping the FSDP weight all-gather with the splash attention kernel — what we tried, with code

**Context.** DeepSeek-v3 MoE, generic path (`RoutedMoE.sparse_matmul`, `use_ring_of_experts=True`),
TPU v7x slice 4x8x8 (EP=4, FSDP=128, per-device batch=4, seq=4096). Goal: take step time from
~15.9s toward ~13.5s by hiding exposed collectives. Files: `layers/moe.py`, `layers/attention_op.py`,
`models/deepseek.py` (hand-written bwd), `models/deepseek_batchsplit_fp8.py` (the reference that works).

---

## 1. The target

Per decoder layer the routed expert weights (`wi_0`/`wi_1`/`wo`, ~1.7–2.1 GB gathered each) are
FSDP-all-gathered on the **SparseCore (SC)** offload engine; matmuls and the splash kernel run on the
**TensorCore (TC)**.

- **TC lane = 12.63 s** (binder), **SC lane = 8.81 s** (hidden 5.72 / **exposed 3.09**). Best-overlap
  ceiling = 12.63 s → overlap is **physically possible** (SC not saturated).
- Exposed collective: **forward 14.3 % (~2.3 s), backward 15.4 % (~2.5 s)**.
- Forward already hides ~2/3 of its gathers behind **QKV matmuls** (the structural −0.37 s hoist win).
- Splash windows are big and idle-SC: **splash-fwd ≈ 11 ms, splash-dkv ≈ 19 ms** per layer.

**The question:** can a weight all-gather run on the SC while the splash kernel owns the TC?

Baselines (loss bit-exact across all): autodiff+remat 15.89 (chunk2) / 16.06 (chunk1); hand-written
custom_vjp bwd 16.13 (chunk2) / 16.32 (chunk1).

---

## 2. The gather primitive (shared by every attempt)

`layers/moe.py` — each weight is a distinct `custom_vjp` all-gather; the `_scheduling_group_id`
annotation is applied only in the forward rule:

```python
def _make_cv_gather(in_pspec, out_pspec, gather_axis, sched_group):
  @jax.custom_vjp
  def _g(w):                                   # PRIMAL: plain gather (what remat recomputes)
    return jax.shard_map(lambda x: jax.lax.all_gather(x, "fsdp", axis=gather_axis, tiled=True),
                         mesh=self.mesh, in_specs=(in_pspec,), out_specs=out_pspec, check_vma=False)(w)
  def _g_fwd(w):                               # FORWARD under diff: annotated gather
    def _fn(x):
      ctx = contextlib.nullcontext() if cfg.moe_wag_no_annotation else _scheduling_group(sched_group)
      with ctx:
        return jax.lax.all_gather(x, "fsdp", axis=gather_axis, tiled=True)
    return jax.shard_map(_fn, mesh=self.mesh, in_specs=(in_pspec,), out_specs=out_pspec, check_vma=False)(w), None
  def _g_bwd(_res, ct):                        # transpose of tiled all-gather = tiled psum_scatter
    ...
  _g.defvjp(_g_fwd, _g_bwd); return _g

# group assignment:
_g0 = _WEIGHT_AG_SCHED_GROUP            # = 1
_g1, _g2 = _WEIGHT_AG_SCHED_GROUP + 1, _WEIGHT_AG_SCHED_GROUP + 2   # 2, 3
w0 = _make_cv_gather(wi_in, w0_out, 1, _g0)(w0)
w1 = _make_cv_gather(wi_in, w0_out, 1, _g1)(w1)
wo = _make_cv_gather(wo_in, wo_out, 2, _g2)(wo)
```
where `_scheduling_group(g) = set_xla_metadata(_scheduling_group_id=g)`.

---

## 3. Every attempt, with code

### Attempt 1 — annotation only (groups 1/2/3). **VESTIGIAL.**
The code above, distinct group ids per gather. Result: **16.06 == 16.06 with vs without** the tag
(`moe_wag_no_annotation` ablation). All-gather fusion is already off
(`xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`); the forward win is purely structural
(hoist before attention + distinct ops + no opt-barrier). The annotation does nothing.

### Attempt 2 — tag the whole splash region + w1 closure adjacent, **UNDER AUTO-REMAT.** → CYCLE.
`moe.py`: return w1 as a closure instead of gathering it; `attention_op.py`: tag `wrap_flash_attention`
and run the closure inside the tag, adjacent to the splash:
```python
# moe.py
if cfg.moe_wag_splash_group:
  w1 = lambda _w=w1: _make_cv_gather(wi_in, w0_out, 1, _WEIGHT_AG_SCHED_GROUP)(_w)   # group 1, deferred
# attention_op.py
with xla_metadata.set_xla_metadata(_scheduling_group_id=1):          # tags EVERYTHING inside
  if wag_cell and "gather_w1" in wag_cell:
    wag_cell["w1_gathered"] = wag_cell["gather_w1"]()                # gather emitted adjacent to splash
  ret = wrap_flash_attention(query, key, value, ...)
```
Result: **`FAILED_PRECONDITION: cycle`.** The annotation lives in the *rematted* forward; the MoE
consuming an annotated gather inside the rematted attention closes a scheduling back-edge. → motivated
the hand-written backward (kills auto-remat, kills the cycle).

### Attempt 3 — hand-written bwd + splash-region tag. → GAP (202 ops).
Same `with set_xla_metadata(_scheduling_group_id=1): ret = wrap_flash_attention(...)` but now under the
custom_vjp backward (no auto-remat). Compiles past the cycle, then:
```
UNIMPLEMENTED: Support for annotation groups with gaps doesn't exist yet,
annotation: 1, instr: pallas_call.223 has the same annotation in its operand tree but has gaps...
```
HLO (`before_optimizations`): **`_scheduling_group_id="1" → 202 ops`** — `set_xla_metadata` wraps the
*entire* `wrap_flash_attention` (q/k/v prep + reshapes + splash + output), with ungrouped q/k/v in the
dependency tree.

### Attempt 4 — emit BOTH w0 and w1 adjacent to splash. → IDENTICAL GAP.
`moe.py` returns both w0 and w1 as closures; `attention_op.py` runs both inside the tag. Result: byte-for-byte
the same `pallas_call.223` gap → **the gap is not about gather placement.**

### Attempt 5 — narrow the tag to JUST the splash op. → STILL 196 ops, STILL gaps.
```python
_splash_tag = set_xla_metadata(_scheduling_group_id=1) if _wag_splash else nullcontext()
with _splash_tag:
  attention_output = jax.vmap(splash_kernel, in_axes=(0,0,0,0,None))(query, key, value, seg, sinks)
```
HLO: group 1 = **196 ops**, of which **only 6 are `custom_call_target="tpu_custom_call"`** (the actual
Mosaic kernels); the other ~190 are JAX reshape/transpose/segment wrappers tokamax emits *around* the
kernel, with ungrouped external inputs. The splash cannot be tagged into a contiguous group from our layer.

### Attempt 6 — input anchor (don't tag the splash; tag a clean splash input). → COMPILES, NULL.
```python
_wag_splash = (cfg.moe_wag_splash_group or cfg.moe_handwritten_splash_group) and cfg.moe_weight_ag_scheduling_group
with set_xla_metadata(_scheduling_group_id=1) if _wag_splash else nullcontext():
  value = self._maybe_shard_with_pspec(value, axis_names_kv)
  value = jax.lax.optimization_barrier(value)   # guaranteed group-1 op that FEEDS the splash
# gathers self-tag group 1 via _make_cv_gather -> group 1 = {w0 ag, w1 ag, value-op}: independent, no gap
```
Result: **compiles clean, bit-exact, step time 16.118 == 16.12 baseline.** XLA will not float the gather
over the splash even when grouped with a splash input. **Why it was doomed (in hindsight):** it couples
the gather with a splash *input* (upstream) → "gather done *before* splash starts" → serialize. Wrong
direction.

### Attempt 7 — backward re-gather hoist (structural reorder). → NULL.
`models/deepseek.py` `fused_bwd`, emit `jax.vjp(_gather)` before `jax.vjp(_attn)`:
```python
def fused_bwd(res, cotangents):
  p, x_in, seg, pos, rest_ = res
  with _detached_linen_module_stack():
    weights, vjp_gather = jax.vjp(lambda pp: _gather(pp, rest_), p)                       # HOISTED first
    (hidden, inter), vjp_attn = jax.vjp(lambda pp, xx: _attn(pp, xx, seg, pos, rest_), p, x_in)
    _out, vjp_moe = jax.vjp(lambda pp, hh, ii, ww: _moe(pp, hh, ii, ww, rest_), p, hidden, inter, weights)
    dp_moe, d_hidden, d_inter, d_weights = vjp_moe(cotangents)
    (dp_gather,) = vjp_gather(d_weights)
    dp_attn, dx = vjp_attn((d_hidden, d_inter))
  dp = jax.tree.map(lambda a, b, c: a + b + c, dp_attn, dp_moe, dp_gather)
  return dp, dx
```
Result: **16.307 vs 16.32** (null). Trace-order reorder of the three *separate* `jax.vjp` regions doesn't
change the schedule; the re-gather still `async-done`s 20.3 ms (past the splash that ends 16.5 ms). Trace:
XLA `async-start`s the re-gather ~15.1 ms — i.e. **it starts the gather too late**, near the end of the
splash, so it spills past instead of using the window.

---

## 4. Root causes

1. **`_scheduling_group_id` is a contiguity construct, not "overlap A with B"** — and the splash is a
   ~196-op subgraph with ungrouped inputs, so any tag on it gaps; tagging an input compiles but does nothing.
2. **Splash is not a hider in a single stream** (attempts 2–6). Gathers hide behind **matmuls**, never the splash.
3. **The auto-remat cycle is genuinely cleared** by the hand-written bwd — but that only unlocked the *gap*, not overlap.
4. **The exposed gather is "started too late," not "too big"** (attempt 7 trace) — XLA schedules its
   async-start just-in-time for the MoE GMM consumer (after the splash), so it never uses the splash window.

---

## 5. Why batchsplit succeeds (`deepseek_batchsplit_fp8.py`)

Purely structural — two independent micro-batches staggered with `optimization_barrier`:
```python
def split(x, split_factor=2):                                   # batch -> 2 independent micro-batches
  x = jnp.reshape(x, (-1, split_factor) + x.shape[1:])
  return [x[:, i, ...] for i in range(split_factor)]

def staggered_call(fn, xs):                                     # software pipeline across micro-batches
  for i, x in enumerate(xs):
    if i == len(xs) - 1:
      xs[i] = fn(x)
    else:
      xs[i], xs[i + 1] = jax.lax.optimization_barrier((fn(x), xs[i + 1]))   # <-- the lever
  return xs

# mla_with_norms (attention/splash) and moe BOTH run through staggered_call:
#   return staggered_call(fn, list(zip(inputs, decoder_segment_ids, decoder_positions)))
```
The barrier stops XLA fusing/reordering the two micro-batches, so **mb0's splash (TC) runs while mb1's
gather (SC) runs**. batchsplit **manufactures a second independent stream** to fill the splash window —
a single batch has no independent work to put there, which is precisely why attempts 1–7 cannot.
Note: the lever is `optimization_barrier` used for **pipelining**, not the annotation, and not the
input-anchor misuse of attempt 6.

---

## 6. Proposed next test — `optimization_barrier` as an ordering/deadline lever (single stream)

Attempt 7's trace says the gather is **started too late** (async-start ~15.1 ms, splash 5.7→16.5 ms).
Idea: fence the gathers **across the splash output** so their *deadline = splash-end*, forcing XLA to
start them by ~11.5 ms — i.e. run them *inside* the splash window on the idle SC. This is the
`W1 ag, W2 ag → barrier(splash) → wo ag` shape, and it is the **same structural use of the barrier as
batchsplit** (ordering), not the failed input-coupling of attempt 6 (which fenced *before* the splash).

```python
# forward (and, with care, the bwd recompute). Sketch at the attention call site:
w1g, w2g = w1_gather(), w2_gather()                       # start the two up-proj gathers (async, SC)
hidden, inter = self_attention_with_norm_op(x, ...)       # contains the splash (TC, ~11ms, idle SC)
# fence: tie the gather results to the splash output so XLA must have them done BY splash-end,
# not before splash-start -> deadline pulls async-start earlier, into the splash window:
hidden, w1g, w2g = jax.lax.optimization_barrier((hidden, w1g, w2g))
wo_g = wo_gather()                                        # released after -> hides behind the MoE GMM
mlp = mlp_op(hidden, pregathered_weights=(w1g, w2g, wo_g))
```

**Expected:** forward exposed-collective drops as w1/w2 move into the splash window; wo continues to hide
behind the up-GMM (unchanged).

**Risks / unknowns to flag for review:**
1. A barrier sets a *deadline*, it does not *force concurrency* — XLA may satisfy "done by 16.5 ms" by
   running the gather 11.5→16.5 (the win) **or** push it elsewhere. Plausible, not guaranteed.
2. **Backward:** an `optimization_barrier` on the *gathered weight* is known to fence the backward
   weight-grad reduce-scatter (pins the RS exposed). The bwd version must keep the barrier off the
   weight-grad path; **forward is the clean first test.**
3. Cheap to validate: local AOT confirms compile + barrier placement (~3 min, no cluster); a chunk=1
   cluster run confirms whether the gather's async-start actually moves earlier.

---

## 7. Decision for review

- **Single-stream splash overlap** has failed 7 ways (cycle → gap → null); §6 is the one untried
  *structural* lever left for the single stream.
- If §6 is also null, the conclusion is firm: **the splash window is only reachable via a second
  independent stream** (batchsplit's split+stagger), and the choice is (a) adopt that schedule on the
  generic path, or (b) bank only the matmul-window overlaps and treat ~15.9 s as the single-stream floor.
