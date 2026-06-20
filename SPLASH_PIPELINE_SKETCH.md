# Splash head-pipelining — design sketch (Tip 2)

## RESULT (2026-06-20, image hwbwd-0620c, chunk=1): NULL / slightly negative
`ds-hw-hp` (n_stages=2, threaded carry token) = **16.396s** (steps 3/4), loss bit-exact, vs
`ds-hw-c1` (hand-written, no pipeline) **16.32** → **+0.08s worse**. The barrier-staggered head-split
did not deliver a net win — either the gathers didn't overlap the stages, or the 2× Pallas/wrapper
overhead ate the gain. Compiled clean (after fixing the shard_map-nesting bug via call-site loop).
Implementation lives behind `splash_head_pipeline_stages` (default 1=off), so it's inert by default.

**`sc` diagnostic → DEAD END (fundamental).** fwd weight-AG exposed = **996ms** vs **980ms** no-pipeline
(unchanged), still `covered-by gmm_v2` (the GMM), never the splash. The barrier-stagger forced the
splashes sequential but did NOT overlap the gathers. **Why head-split can't work like batchsplit:**
batchsplit splits the **batch** → each micro-batch has its own attention *and* its own MoE gather, so
mb1's gather overlaps mb0's splash. Head-split only splits the attention **heads** — the MoE/experts
are **shared** (not per-head), so there is still ONE gather whose consumer (the MoE GMM) sits after
*all* head-stages; its deadline is unchanged, XLA still late-starts it, and it spills past the splash
exactly as before. **The thing we need to overlap (the MoE gather) is not head-splittable.** To get
batchsplit's win you must split the batch (independent attention+MoE), not the heads. Tip 2 abandoned.

---


**Goal.** Break the single monolithic splash kernel (one ~11ms `pallas_call` over all 128 heads,
opaque to the scheduler) into a few **head-slice stages**, and stagger them with
`optimization_barrier` so a concurrent SC collective (the MoE weight all-gather, or KV AGs) runs on
the SC while each TC splash stage runs — batchsplit's two-stream pipeline, but on the **head axis**
instead of the batch axis. Local to `attention_op`, transparent to batch/seq, **no extra HBM held**.

## Why this beats the single-stream attempts
- Single splash = one opaque custom-call; the scheduler late-starts the gather to minimize HBM, and
  has no boundary to interleave at → exposed (attempts 6/8 null).
- N head-stages + `optimization_barrier` = genuine pipeline boundaries; the barrier *forces* the
  stagger (the only thing that worked — batchsplit's `staggered_call`), so the gather fills the gaps.
- The heads are independent (attention is per-head), so the slices are a real second stream — and
  the kernel already treats heads as a `parallel` grid dim, so a head-subset is a legal splash call.

## Structure — **n_stages=2 with a THREADED CARRY TOKEN (critical fix)**

**Why a plain per-stage barrier is WRONG:** `o_i, g_i = optimization_barrier((o_i, g_i))` creates no
dependency *between* stages. XLA unrolls the loop, sees N independent barriers, and legally runs all
N splash slices in parallel (recombining into one 11ms TC block) while sinking the gathers →
**collapses back to a single stream.** The barrier must thread a **token** so stage i+1's *inputs*
are blocked by stage i's *outputs* — that is the hard sequence batchsplit relies on.

```python
def pipeline_splash(q, k, v, *, n_stages, weight_gathers=None):   # n_stages = 2
    H = q.shape[1]                                  # local head axis
    step = H // n_stages
    outs, gathered = [], []
    token = jnp.zeros((1,))                          # carry token that forces the stagger
    for i in range(n_stages):
        sl = slice(i*step, (i+1)*step)
        q_i, k_i, v_i = q[:, sl], k[:, sl], v[:, sl]
        # BLOCK this stage's inputs until the PREVIOUS stage's (o,g) hit the barrier:
        q_i, k_i, v_i, token = jax.lax.optimization_barrier((q_i, k_i, v_i, token))
        g_i = weight_gathers[i]() if weight_gathers else None   # SC gather (stage0->w0, stage1->w1)
        o_i = splash_kernel(q_i, k_i, v_i)                       # TC, ~11ms / n_stages
        token = (o_i, g_i)                                       # tie outputs -> next stage's barrier
        if g_i is not None: gathered.append(g_i)
        outs.append(o_i)
    out = jnp.concatenate(outs, axis=1)              # reassemble full heads (~0.05ms kCopy)
    return out, gathered
```

**n_stages=2 is the golden ratio (reviewer):** 11ms splash → two ~5.5ms stages; w0 and w1 are ~5ms
each → stage0∥w0, stage1∥w1, each gather perfectly encapsulated by a splash stage. `wo` hides behind
the MoE up-GMM as today. Minimizes Pallas launch overhead (only 2 launches).

## Integration point
`attention_op.py::wrap_flash_attention` currently does
`attention_output = jax.vmap(splash_kernel, in_axes=(0,0,0,0,None))(q, k, v, seg, sinks)` (vmap over
**batch**, axis 0). The pipeline wraps that call: keep the batch vmap, add the **head-slice loop**
around it, thread the weight-gather closures in (same `wag_cell` plumbing we already built), splice
the gathered weights back into the MoE. Forward first; the bwd `dkv` kernel pipelines the same way.

## Tradeoffs / risks (for review)
1. **Per-stage kernel overhead.** N splash launches instead of 1 (kernel setup, grid, the ~196-op
   wrapper ×N). Keep **n_stages small (2–4)** to amortize; measure the attention-time delta — if a
   2-stage split costs >½ a stage in overhead, the math sours.
2. **Does the barrier actually trigger overlap here?** This is the open empirical question — same as
   batchsplit's premise, but we haven't proven it on the head axis. The local AOT can show whether
   the gather's async-start lands inside a stage; one cluster run confirms step time.
3. **Sharding / head divisibility.** Heads may be sharded (tp); slice the *local* head count and keep
   the slices on-device. Ensure `H_local % n_stages == 0` (DeepSeek: 128 query heads, divides by 2/4).
4. **Concat cost.** Re-concatenating the head slices is cheap (no recompute), but adds a relayout —
   verify it doesn't show up as a new exposed cost.

## Why I'd try this before the cross-layer prefetch
- **No HBM increase** (the killer risk for cross-layer prefetch) — same single-layer gather, just
  overlapped with a staged splash.
- **Local change** — `attention_op` only, no `nn.scan` restructure, no shifted-param carry.
- **Reuses** the `wag_cell` gather-closure plumbing we already built.
- It is the cleanest test of "can a barrier-staggered two-stream actually overlap the gather" — the
  one premise the whole splash-window question rests on, and the thing batchsplit proves at the
  batch level. If it works on heads, it generalizes; if it doesn't, batchsplit's win is suspect too.
