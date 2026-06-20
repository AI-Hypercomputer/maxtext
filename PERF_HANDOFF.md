# DeepSeek-v3 MoE on TPU v7x — perf-opt handoff

Goal: minimize step time for DeepSeek-v3 MoE (MaxText) on **v7x slice 4x8x8, EP=4, FSDP=128, per-device batch=4, seq=4096**.
Generic path = `RoutedMoE.sparse_matmul` with `use_ring_of_experts=True` (NOT `deepseek_batchsplit.py`).
Decoder layer = `models/deepseek.py:DeepSeekMoELayer`, scanned via flax `nn.scan` in `layers/decoders.py`.

## Current state
- **Baseline: 16.24 s/step.**
- **BEST: 15.91 s/step** = `wag-no-barrier + moe_n_chunks=2`. Loss bit-exact.
  - Image: `gcr.io/cloud-tpu-multipod-dev/integrate_v2:wag-attn-0618j`
  - Run: `EXTRA_MAXTEXT_ARGS="moe_weight_ag_scheduling_group=True moe_n_chunks=2"`
  - Pushed to **`ultrons/maxtext` branch `chunked-pipeline`** (HEAD has all wag flags; later commits add flag-gated, off-by-default experiments).
- Target: **13.5 s** (≈74 ms/layer fwd + 148 ms/layer bwd). We're at fwd ≈78 ms/layer (good); the gap is the **backward**.

## What the forward work proved (DONE — don't re-litigate)
1. **The −0.37 s forward win is purely STRUCTURAL**, gated by `moe_weight_ag_scheduling_group`:
   - (a) make the FSDP weight gather (w0/w1/wo) **explicit** and **hoist it into the attention phase** (emitted in `deepseek.py:__call__` *before* `self_attention_with_norm_op`, passed in as `pregathered_weights`) → it overlaps the **QKV/GMM matmuls**;
   - (b) 3 **distinct `custom_vjp` ops** (un-fused) — see `moe.py:RoutedMoE.gather_weights` / `_make_cv_gather`;
   - (c) **NO `optimization_barrier`** — it's self-dual and fences the backward weight-grad RS (removing it took the backward RS 1.9 GB→2.6 MB, and unblocked chunk2 stacking: 16.45→16.06→15.91).
2. **The `_scheduling_group_id` annotation is VESTIGIAL** (16.06 == 16.06 with/without). Fusion is already off via `xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`.
3. **Splash (Pallas) is not a hider** — gathers hide behind matmuls; XLA will not overlap a collective with the splash kernel by placement/annotation.

## THE definitive finding (this is why the backward must be hand-written)
**gather∥splash annotation-overlap CANNOT coexist with auto-remat — it cycles.**
- Tagging the splash kernel (`set_xla_metadata`; no `custom_vjp`) puts the annotation in the *rematted* forward. The MoE consuming an annotated gather that lives inside the rematted attention closes a scheduling back-edge → `FAILED_PRECONDITION: A cycle is detected ... rematted_computation/moe_layers/.../ffn_act` (run `ds-w1splash-0618`, image `0618p`).
- The original wag avoided cycles only because the gather's `custom_vjp` **primal is plain** (no annotation in the remat) and it lives in its own pre-attention region.
- **batchsplit gets the gather∥splash overlap only because it hand-writes its backward** (no auto-remat to cycle against). So to overlap the exposed backward weight re-gathers with the backward splash, **the hand-written backward is REQUIRED, not optional.**

## Backward diagnosis (the ~1.7 s to recover)
- `step_breakdown --normalize`: bwd = 62% of step; **bwd exposed-collective ≈16.3% ≈2.6 s**; bwd compute ≈2× fwd (healthy — not FLOP-bound).
- `xla_shell context` around `tgmm_v2` (the bwd weight-grad GMM): the exposed thing is **3× ~0.94 GB weight re-gathers per layer** (`bf16[16,4096,7168]`, scope `dot_general`, unfused) that over-subscribe per-layer GMM compute. The **token-grad RS (235 MB) is small/hidden** — not the problem. The bwd weight-grad RS already collapsed to 2.6 MB (no-barrier).
- BUT same-layer compute is **abundant**: the **backward splash dkv ≈19 ms + the splash-fwd remat** are big TC kernels with **idle SC**. XLA just front-loads the SC gathers instead of overlapping them with these kernels.
- ⇒ **Cross-iteration prefetch is NOT needed** (and a manual `lax.scan` is out: nesting inside `nnx.scan` is broken, and it's too invasive for MaxText). The fix is **within-layer** scheduling control.

## THE PLAN: layer `custom_vjp` hand-written backward (inside `nn.scan`)
Wrap the DeepSeek layer (gather + attention + MoE) in a `jax.custom_vjp` **inside** the existing `nn.scan` (no manual scan):
- **fwd rule**: run the existing forward — gather **hoisted** (keeps the −0.37 win) → attention → MoE — capturing `jax.vjp` closures for each piece (kernels already ship VJPs: splash, tokamax gmm, `use_custom_sort_vjp`; the rest is autodiff-able). **No gradient math is hand-derived.**
- **bwd rule** (hand-written): replay the captured VJP closures in our chosen order, and **emit the weight re-gathers + the weight-grad reduce-scatter adjacent to the backward splash (dkv/remat) with the scheduling-group annotation — while controlling the remat ourselves, so there's no auto-remat to cycle against.** This is what batchsplit does; adapt the *technique*, keep the generic `RoutedMoE` ring compute.
- Flag-gate it; keep autodiff as the default/reference.

### Skeleton (illustrative)
```python
@jax.custom_vjp
def layer(x, w_sharded, ...): ...
def layer_fwd(x, w_sharded, ...):
    w,  vjp_gather = jax.vjp(gather, w_sharded)      # capture, don't derive
    a,  vjp_attn   = jax.vjp(attention, x)
    y,  vjp_moe    = jax.vjp(moe, a, w)
    return y, (vjp_gather, vjp_attn, vjp_moe, w_sharded)
def layer_bwd(res, dy):
    vjp_gather, vjp_attn, vjp_moe, w_sharded = res
    w  = regather(w_sharded)            # WE place this: adjacent to splash-bwd, annotated, in-region
    da, dw_full = vjp_moe(dy)
    dx          = vjp_attn(da)          # splash-bwd (dkv)
    dw_sharded  = reduce_scatter(dw_full)   # WE place this next to the GMM-bwd matmuls
    return dx, dw_sharded
layer.defvjp(layer_fwd, layer_bwd)
```

### #1 risk + verify ladder
- **Weight-grad sharding spec** is the only real risk — the **1/128 `reduced`/`unreduced` trap**: a wrong `psum_scatter` axis/spec gives a grad off by the shard count. Validate against the known-correct `moe.py:_make_cv_gather._g_bwd` (`psum_scatter` over `"fsdp"`, `scatter_dimension`=gather axis, `tiled=True`).
- **It is NOT a safe blind one-shot** — needs cluster-in-the-loop: edit → cluster run → read loss → fix spec → repeat. Loss must be bit-exact: `9.617 / 9.532 / 9.464 / 9.411 / 9.369 / 9.333` at steps 14–19.
- Verify ladder (per `~/.claude/CLAUDE.md`): AOT Mosaic compile-check (`jax.experimental.topologies`, no HW) → EP=1 exec → EP=N exec → bit-exact loss on the slice.

## Dead ends — DO NOT retry
- Backward weight-grad RS annotation alone (16.62). `context=offload` of `context` (17.49 — host load-back > saved recompute). `use_splash_scheduler` (16.78). `chunk4` (over-chunks). Host-offload of gathered weights / q-k-v (OOM, 184 GB > 128 GB/core). combine=TC (NaN). Tag-splash-only / over-group attention (annotation GAPs). w1-in-splash (cycle — the finding above). Manual `lax.scan` (nesting broken in `nnx.scan`). gate+up GMM fusion won't help the floor (we're TC-bound, not HBM-bound).

## Key file anchors
- `moe.py`: `RoutedMoE.gather_weights` (~2614), `_make_cv_gather` (the gather `custom_vjp`), `_scheduling_group`/`_WEIGHT_AG_SCHED_GROUP` (~57/73), `sparse_matmul` (~1219, `_wag_sched`/`weights_pregathered`), `gather_routed_weights` (2906 → `gather_weights`).
- `deepseek.py`: `DeepSeekMoELayer.__call__` (~589 — hoisted gather + `pregathered_weights`), `self_attention_with_norm_op` (275), `attention_op` (207).
- `attention_op.py`: `tpu_flash_attention` (1160), splash call `ret = wrap_flash_attention(...)` (~1531), `_splash_group_ctx` (~1525), `apply_attention` (903), `AttentionOp.__call__` (2079).
- `attention_mla.py`: `MLA.__call__` (1084).
- `decoders.py`: `scan_decoder_layers` / `nn.scan` (~562), `get_remat_policy` (~333, `remat_policy=custom` → `save_and_offload_only_these_names`).
- Flags (`configs/types.py` + `base.yml`): `moe_weight_ag_scheduling_group`, `moe_n_chunks`, plus off-by-default experiment flags `moe_wag_no_annotation` / `moe_wag_fwd_barrier` / `moe_wag_splash_group` / `moe_wag_attn_group` (the last few cycle/gap — leave off).

## Tooling
- Cluster run: `WORKLOAD_IMAGE=<img> RUN_TAG=<tag> FLAGS_FILE=flags_A.txt EXTRA_MAXTEXT_ARGS="..." PRIORITY=very-high bash repro_variant.sh`. Logs/profiles → `gs://ubench-logs/<tag>/`. Pods GC fast on failure — use a live `kubectl logs -f | tee /tmp/x.log` streamer to catch compile errors.
- Build (local, per `~/.claude/CLAUDE.md`): `sudo docker build ... && sudo docker push ...`. Commit before build.
- Profiles: `cd ~/xla-shell && xla_shell -c "read_xplane <gs:// or local>; analyze_profile; step_breakdown --normalize; list_collectives --overlap"`; `find <op>` → `context <op> --n N` to zoom the trace; xprof compare dropdown = `gs://ubench-logs/_compare/NN-label/`. (See the `/profile` skill.)
- Results log: `sweeps/results.md` (full table of every run). Memory: `deepseek-moe-v7x-perf.md`.
