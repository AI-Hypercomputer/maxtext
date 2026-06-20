# DS-v3 batch-split perf notes (4x8x8, EP=4, FSDP=128, pdbs=4)

## Baseline (reproduced 2026-06-17)
- Workload `ds-repro-0617-1115`, image `integrate_v2:gmmv2_bf16_log27`.
- Steady-state step time: **16.234–16.241s** (matches reported 16.24s). 252.7 TFLOP/s/device.
- Structure (xplane, per step): forward scan **4.82s** (58 layers × 83.1ms), backward scan **10.68s** (58 layers × 184.1ms). Backward dominates.

## Collective attribution (xla_shell list_collectives --overlap)
- Exposed collective stall = **19.2% of step time**. AG dominates (11.25s of trace), then RS (2.83s).
- **Token EP all-gather** (`moe shard_map/all_gather`, ~150–160 MB): all-gather.389=6.29ms, .317=7.22ms, .323=6.63ms/firing — largest exposed bucket. These are `route_impl_fwd` AGs fenced by optimization_barrier.
- **Reduce-scatter**: rs.65=6.45ms (150MB), rs.63=2.96ms (253MB) — token unroute / grad RS.
- **FSDP weight all-gather** (`convert_element_type`, ~2.0–2.2 GB): ag.393=5.17ms, .395=4.62ms, .319=4.50ms exposed (~210–250 GB/s). BUT ag.329 same 2.13GB runs 1503 GB/s / 0.76ms — weight AG CAN be fully offloaded/overlapped. Exposed ones are on critical path.
- all-reduce 2.45GB / 1579ms = DCN data-axis weight-grad AR (grad_ar_dcn).

## Implications
- **Task 2 (flags, no code change)** targets the exposed 2GB weight AGs → push them onto the fast SC-offloaded path (num_sparse_cores_for_gather_offloading=2, single_sc=false, ici_ag_pipelining, ag_backward_pipelining). Validated: headroom is real (.329 proves the fast path exists).
- **Task 3 (chunked pipeline)** targets the exposed token AG (.389/.317/.323) + token RS (.65/.63) — the biggest single bucket. Design = jax-gpt path A: slice local tokens into chunks before EP all_gather, sibling-unrolled, drop per-chunk optimization_barrier so XLA interleaves chunk i+1 AG with chunk i ragged_dot, RS offloaded to SC. Risk (sutra): RS may stay partly on critical path; token AG only ~74% overlapped historically.

## xla-shell workflow (for future A/B)
- Load: `read_xplane <dir>` (pass the *directory* containing .xplane.pb; gs:// listing is flaky → `gcloud storage cp` the .pb to a local dir first).
- One-shot read: `analyze_profile` (engine-lane decomposition + verdict), `roadmap --all` (coupling-ordered work-order with moving floor), `step_breakdown` (phase × resource %), `list_structure` (fwd/bwd scan + per-layer motif).
- Collectives: `list_collectives --overlap` (exposed stall per op), `--verbose` for full source names.
- A/B diff: `compare_profiles <baseline_dir> <new_dir> --depth 1 --top 20` → per-component Δms/Δ%.
- Projection: `what_if --scale collective=0.5` (model hiding half the exposed comm), `what_if --efficiency splash=0.45`.
- Roadmap verdict for THIS profile: scoped comm-overlap work (tasks 2+3) ceiling = 13.11s (gain 3.13s). Kernels→9.18s (out of scope). <9.18s needs comm-volume/sharding cut (out of scope).

## Runs
- A baseline: ds-repro-0617-1115 = 16.24s (DONE)
- B flags additive delta: ds-flagb-0617-1126 (flags_B.txt) = 17.25s REGRESSION +1.01s.
  compare_profiles A->B: SC lane -472ms (comm cheaper) BUT TC lane +422ms (relayout +321ms) + worse overlap. Layer-scheduler + ici-pipelining regimes don't compose; flags force TC relayout. Numerics identical.
- C flags v338 regime: ds-flagc-0617-1126 (flags_C.txt) = COMPILE CRASH (async_collective_start_emitter HandleWhile 26vs6; multiple_steps incompatible w/ MoE while-loop). Deleted.
- B' = B minus num_sparse_cores=2: ds-flagbp-0617-1212 (flags_Bp.txt) = 17.247s, SAME regression as B. => num_sparse_cores=2 NOT the cause; culprit is single_sparse_core=false and/or ici_ag/rs + ag_backward pipelining. lm_loss bit-identical to baseline.

## Task 2 conclusion
Borrowed jax-gpt flags do NOT transfer to this code (generic moe.py ring-of-experts). B/B' regress (~+1s, TC relayout), C crashes. repro.sh flags are near-optimal. Exposed weight AG is structural. Pivoted to task 3 (code).

## Task 3 target (CONFIRMED)
- Branch chunked-pipeline @ captain/pr_branch 95628e62f (= image source). Generic moe.py RoutedMoE.sparse_matmul, use_ring_of_experts=True.
- Real flow (moe.py wrapper :1655): route() -> EP all_gather x/logits/pre_bias (:1456) + permute/sort; gmm_up (:1685, w0+w1 GMM); gmm down (:1688); unpermute (:1704); psum_scatter over EP (:1718, scatter_dim=0).
- Weight AG: NOT fused per-gmm (weight_gather=False since shard_exp_on_fsdp=False); gathered ONCE at shard_map entry via funky mlp_no_fsdp pspec -> reused across chunks (chunking won't re-gather weights. good).
- Exposed comms to hide = EP all_gather (:1456) + psum_scatter (:1718) = profile's biggest bucket.
- Chunk plan: split tokens (seq dim) into moe_n_chunks; per chunk route->gmm->combine as Python-unrolled siblings, no per-chunk opt_barrier -> XLA interleaves AG(c+1) w/ GMM(c), psum_scatter(c) w/ GMM(c+1). lm_loss per-token => bit-identical; moe_lb_loss minor drift (aggregate over chunk). Gate behind moe_n_chunks (default 1 = no-op). Validate via CPU topologies AOT check + n_chunks-equivalence before TPU.

## KEY INSIGHT (compiler team, 2026-06-17): SC custom-calls are scheduling barriers
- The mpmd_map_* SC custom-calls (sc_ragged_gather dispatch + ragged_gather_reduce combine) span BOTH SparseCores. The XLA latency-hiding scheduler models a custom-call's resources conservatively (doesn't know it avoids ICI) -> treats it as a full-resource BARRIER -> FSDP AG/RS get squeezed to iteration ends (exposed). THIS is the "structural" comm exposure.
- Explains ALL our negative results: ici_ag_pipelining (B/B') prefetched the AG but it still landed behind the next mpmd_map (barrier is the bug, not issue-time); async-fusion (C) couldn't thread collectives through the both-SC op; scoped-vmem/scavenging freed VMEM but can't beat a barrier; chunked n=2 won only -0.6% because per-chunk RS/AG still serialize around the barriers.
- FIXES:
  (a) DE-BARRIER (proper, highest leverage): annotate the tokamax custom-call's scheduler resources so XLA knows it's ICI-free -> AG/RS can overlap it. Fixes BOTH SC barriers. XLA/tokamax-side; raise with compiler team.
  (b) use_ragged_sort=False (+use_ring_of_experts=False, validator-coupled): ARG-ONLY, moves dispatch+combine sort/gather to TC (_sort_activations), removing the SC ragged barriers. Dispatch switches to ragged_all_to_all. Staged: raggedoff_run.sh. Metric = step time / FSDP-collective exposure (not gather ms). The compiler-reframe test, cheapest path.
  (c) Move combine->TC einsum (moe.py:932-957 non-ragged unpermute), keep SC gather for dispatch: code change, cuts barriers ~half/layer.

## INFRA BLOCKER (2026-06-17 PM)
- Cluster bodaborg-super-xpk-x8p LOST its 4x8x8 (tpu7x-512) nodepool mid-session. Now only 4x4x4 (48 nodes). Cannot run our EP=4/FSDP=128/256-chip shape. All further cluster experiments (raggedoff, cctrace profiling, gate+up fusion) BLOCKED until 4x8x8 restored or another cluster found.

## FINAL RESULTS (within no-kernel/no-sharding constraint)
- Only win: chunked pipeline n=2 = 16.14s (-0.6%), loss-exact. Everything else (flags B/C, GMM-tile split-K, scoped-vmem/scavenging, splash tile) neutral/regress/crash. Exposed comm is structural (SC custom-call barriers) -> needs (a) de-barrier or (b) raggedoff to unlock.

## tcsort+chunk2 roadmap (de-barrier tradeoff QUANTIFIED, 2026-06-17)
ds-tcsort-c2-0617 (18.24s) vs baseline (16.24s), via xla_shell analyze_profile/roadmap:
- exposed comm: baseline 3.13s -> tcsort+chunk2 2.43s = -0.70s (de-barrier DID hide more FSDP/EP comm; compiler barrier theory confirmed in wall-clock).
- TC lane: 12.72s -> 15.40s = +2.68s (compute +1.3, VPU +1.45). TC _sort_activations gather/combine is VPU-heavy (vpu 3.72->5.17, =24% of step); SC kernels run this on SparseCore keeping TC/VPU free -> why TC route is net loss.
- net +2.0s = +2.68 TC cost - 0.70 comm benefit. Matches 16.24->18.24.
- CONCLUSION (numerically justified): direction-(a) annotate fast SC kernels ICI-free captures the -0.70s comm-hide (+ ~4x-larger chunking overlap) WITHOUT the +2.68s TC/VPU penalty -> path to beat 16.14. tcsort+chunk2 roadmap floor 11.32s (kernels), ceiling 15.40s (TC-bound).

## SC chunk2 roadmap (BEST config, 16.14s) + 3-way (2026-06-17)
ds-chunk2-0617-1235 (16.14, our win): TC-lane-bound 13.01s (compute 8.57+vpu 3.96+relayout 0.48); SC lane 8.91 (6.18 hidden/2.73 exposed). roadmap lever1 schedule-exposed-comm -> 13.37s (gain 2.77); lever2 kernels -> 9.28 (floor, comm-volume-bound past).
3-way exposed comm: baseline 3.13 -> SC chunk2 2.73 -> tcsort+chunk2 2.43. TC lane: 12.72 -> 13.01 -> 15.40 (vpu 3.72->3.96->5.17).
- chunk2's -0.10 net = hid -0.40 comm but +0.29 TC (chunking overhead). Comm-hide throttled by barriers.
- PRIZE: lever-1 on best config = hide 2.77s exposed comm via de-barrier (dir-a annotate SC ICI-free) -> 16.14 -> ~13.37s (-17%), WITHOUT tcsort's +2.68 TC/VPU penalty. <9.28s needs sharding change.

## MECHANISM CORRECTION (user, 2026-06-18): SC resource CONTENTION, not scheduler barrier
- In the best SC chunk2 run, the FSDP WEIGHT GATHER is itself SC-offloaded (sparse_core_collective_offload_all_gather flags) AND the sort/unsort (mpmd_map) runs on SC. So weight-AG + token-AG + RS + sort/unsort ALL contend for the 2 SparseCores. The exposed comm is SC-ENGINE CONTENTION, not (only) a scheduler resource-modeling barrier.
- Reconciles tcsort: moving sort->TC freed the SC for the offloaded collectives (-0.7s exposed comm), not "removed a barrier".
- => direction-(a) ICI-free annotation likely NOT the fix (sort & offloaded-collective both physically need SC). Real lever = REBALANCE SC LOAD.
- Profile op costs (user): gather SC=4.3 / TC=7.8; combine SC=9.x (barriered) / TC unsort+reduce ~10 (frees SC). Decision: gather=SC (cheap), combine=TC (frees SC for RS).
- IMPLEMENTED moe_combine_kernel flag (auto|sc|tc) decoupling combine from gather (use_ragged_sort). Image chunked-cmb-0618. Running ds-scg-tcc-c2-ici-0618 = gather SC + combine TC + chunk2 + ici_ag. WATCH LOSS (SC-gather/TC-combine is new pairing; index/buffer must be compatible).

## PROTOTYPE FINDINGS + SC-idle green light (2026-06-18)
Two on-machine AOT prototypes (/tmp/proto*.py) on virtual tpu7x:4x4x4, plus user's xprof read:
1. shard_map is NOT a scheduling boundary — inlined into main.0_spmd during SPMD partitioning. "Separate shard_maps" vs "one shard_map" compile to identical entry HLO. Cross-region hypothesis FALSE.
2. A plain matmul AND a generic Pallas tpu_custom_call BOTH overlap an independent SC-offloaded all_gather (AG-start before, AG-done after, in flight on SC while compute on TC). Custom-call-barrier theory REFUTED for generic Mosaic kernels.
3. => wagv2's non-overlap was NOT a hard barrier. Root cause = PROGRAM-ORDER PLACEMENT + SC occupancy: wagv2 emitted the weight-AG in the MoE phase (adjacent to the SC-saturating ragged_sort/gather kernels) so it overlapped a BUSY SparseCore -> no benefit. Never hoisted to the attention phase.
4. USER CONFIRMED via xprof: BOTH SparseCore lanes are IDLE during splash attention. => the SC-offloaded weight-AG CAN overlap splash with NO contention. GREEN LIGHT.
5. NEXT EXPERIMENT: emit the weight-AG in the ATTENTION phase (nnx_decoders.__call__ BEFORE self_attention, accessing MoE submodule weights), pass pregathered weights into RoutedMoE. Program-order proximity to splash + idle SC => should hide ~9ms/layer weight-AG. (v2's sparse_matmul placement was the wrong phase.)
6. FOLLOW-UP: dig how deepseek_batchsplit gather_weights/batch_split_schedule positions the next-layer gather in program order to overlap current-layer compute (the working template; user has seen collectives scheduled "over shardmap" = program-order placement, not region).

## wag-attn (attention-phase gather) — XLA "annotation gaps" constraint (2026-06-18)
ds-wagattn-0618 (image wag-attn-0618b, gather_weights() in DeepSeekMoELayer before self_attention, splash tagged _scheduling_group_id=1). Earlier failures: (1) flag not in image types.py/base.yml -> ValueError (fixed: rebuilt 0618b w/ all 4 files); then (2) the real error:
  JaxRuntimeError UNIMPLEMENTED: "Support for annotation groups with gaps doesn't exist yet, annotation:1, instr dot_general.372 has the same annotation in its operand tree but has gaps". 
=> XLA requires a _scheduling_group_id group to be a CONTIGUOUS region in the operand DAG. Tagging two non-adjacent ops (weight-AG before attention + splash) with intervening un-annotated ops (norm, QKV proj) = gaps = unsupported. XLA suggests --xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true (drops gappy groups -> compiles, but may drop OUR group -> inert). Trying that flag (flags_wag.txt). If inert, the explicit-2-op annotation can't express weight-AG||splash; would need contiguous annotation (tag the whole attention block) OR a different mechanism.

## wag-attn FINAL (custom_vjp, 2026-06-18): lever exhausted — overlap won't fire on generic path
ds-wagattn-0618 (0618e custom_vjp) = 16.97s, COMPLETED (cycle gone), loss bit-exact, but +0.72 REGRESSION.
Profile vs baseline: TC 12.72->12.42 (-0.30); SC lane 8.78->10.53 (+1.75); EXPOSED COMM 3.13->4.36 (+1.23 WORSE).
=> custom_vjp solved the cycle (plain-primal remat recompute) + numerics exact, BUT the scheduler does NOT overlap the annotated gather with splash in the real 671B model (toy prototype did; real model doesn't). Explicit fwd-gather + bwd re-gather + psum_scatter ADD exposed SC comm. _scheduling_group_id reaches HLO but scheduler ignores the intent (same as wagv2).
5-iteration arc (config->gaps->cycle->host-OOM->custom_vjp): every COMPILE obstacle solved, final version regresses. SCHEDULING-ANNOTATION APPROACH TO weight-AG||splash DOES NOT WORK on generic path. DEAD.
Overlap is real ONLY in batchsplit (co-designed: manual fwd/bwd + per-layer host-streamed residuals + its schedule). PATHS LEFT: (a) measure batchsplit directly (does it already overlap + beat 16.24?); (b) compiler/tokamax: scheduler must honor overlap intent / forced annotation. Generic-path retrofit exhausted.

## wag-attn unfused: FORWARD WIN, backward drag isolated (2026-06-18)
ds-wagattn-unfused-0618 = 16.45 (+0.21). step_breakdown vs baseline (per-step s):
- FORWARD total 6.27->5.90 = -0.37 WIN; fwd exposed comm 2.73->2.33 = -0.40 (weight-AG||splash overlap FIRED). Lever validated.
- BACKWARD total 9.84->10.41 = +0.57 LOSS; bwd exposed comm 2.53->3.16 = +0.63.
- net +0.20 (=the +0.21 step). If backward matched baseline -> ~15.87 (BEATS baseline by ~0.37).
ROOT of backward drag (list_collectives --overlap): reduce-scatter total exposed 2.83s@366GB/s (baseline) -> 4.36s@235GB/s (wagunf). The new big one = reduce-scatter.55 (weight-grad RS, 1.90GB, 6.83ms/layer, ~1.585s exposed) = the custom_vjp _g_bwd EXPLICIT shard_map psum_scatter. Baseline's weight-grad RS is IMPLICIT GSPMD reduce-scatter (SC-offloaded, fast, overlapped); ours isn't -> slow+exposed. FIX: make _g_bwd use the implicit/offloaded RS (sharding-constraint to sharded layout, let GSPMD insert it) instead of explicit shard_map psum_scatter.

## wag-attn backward fix = JAX-LEVEL DEAD END (2026-06-18, fork-verified)
Attempted fix (with_sharding_constraint for the bwd weight-grad RS) is NUMERICALLY WRONG: it slices (gathered->sharded reshard) but FSDP weight-grad needs reduce-scatter = SUM over fsdp replicas. Verified 4-device: psum_scatter=[4.0,4.4,4.8,5.2] vs wsc=[1.0,1.1,1.2,1.3] (exactly 128x off at fsdp=128). Current psum_scatter is CORRECT (loss bit-exact). Did NOT build/ship it.
FUNDAMENTAL TENSION: forward overlap needs explicit shard_map all-gather (carries _scheduling_group_id); backward offload needs implicit GSPMD reduce-scatter (but that's the wrong slice grad AND loses the fwd tag). Mutually exclusive at JAX level. Manual shard_map psum_scatter won't SC-offload like baseline's implicit RS -> bwd RS slow+exposed (+0.57). Annotating bwd psum_scatter = singleton group, stripped.
=> NO clean JAX-level fix for the backward. wag-attn-unfused 16.45 (+0.21) is the CORRECT best wag-attn (fwd win -0.37 real, bwd un-offloadable). Remaining lever = COMPILER/tokamax (SC-offload a manual shard_map reduce-scatter) = same bucket as "scheduler won't honor overlap intent". Best overall deliverable = SC chunk2 16.14. Pushed code (ultrons/maxtext chunked-pipeline, wag-attn-0618f) is the CORRECT psum_scatter version.
