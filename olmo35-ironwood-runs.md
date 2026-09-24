# OLMo 3.5 on Ironwood: run record, profiles and xla-shell output

Every measured Ironwood number for the OLMo 3.5 partner family, one row per
configuration, with the profile and the xla-shell analysis that goes with it.
The plan and the reasoning live in `olmo35-ironwood-plan.md`; this file is the
evidence.

## Conventions

| item | value |
|---|---|
| MFU | `TF/s/device / 1153.5`. A tpu7x chip is 2307 TFLOP/s bf16 and carries **two** devices |
| step time | median of the **last 10** of 20 steps, so compile and warmup are excluded |
| data | `dataset_type=synthetic`, checkpointing off |
| precision | `dtype=bfloat16 weight_dtype=float32` |
| model | `olmo35-tiny` unless stated: d 1024, 16 layers, 512 experts top-16, latent 512, full attention every 8th layer |
| always on | `sparse_matmul`, `use_tokamax_kda`, `use_gmm_v2`, `shard_exp_on_fsdp`, `num_vocab_tiling=8`, `remat_policy=full`, `TOKAMAX_KDA_DENSE_PAIRS=1`, `TOKAMAX_KDA_BF16_FWD/BWD=1` |
| XLA flags | the Ironwood sparse-core offload set from `run_olmo3_7b_stage1.sh`, via `LIBTPU_INIT_ARGS` |

The always-on set is the winning stack from the 8-device sweep. Anything an arm
passes in its `extra` column overrides it, because MaxText takes the last
occurrence of a repeated key.

## Reference anchors

`tpu-recipes/training/ironwood`, per chip against the 2307 TFLOP/s peak. These
are the yardstick, and they say the expert layout is what decides the class.

| recipe | TF/s/chip | MFU | expert layout |
|---|---|---|---|
| llama3.1-70b | 1207 | 52.3% | dense |
| qwen3-235b-a22b | 630 | 27.3% | MoE, wide experts |
| deepseek3-671b | 608 | 26.3% | MoE, wide experts |
| **gpt-oss-120b** | **330** | **14.3%** | **MoE, many narrow experts** |
| OLMoE3-3p5b, ours | 195 | 8.45% | 512 experts top-16 |

OLMo 3.5 is 512 experts at top-16 with expert hidden = d_model, which is the
gpt-oss class, not the deepseek class.

Measured on the same stack for calibration, 8 devices, pdb=1:

| model | TF/s/dev | MFU |
|---|---|---|
| olmo3-7b dense | 277 | 24.0% |
| olmo3-7b dense, `remat=custom` | 318 | 27.6% |
| olmo3-32b dense | 349 | 30.3% |

## Run log

### 8 devices (2x2x1, single host). The lever stack

Vehicle was `bodaborg-tpu7x-spot-sps` in `cloud-tpu-multipod-dev`, plain pod
create, no Kueue. Each row is one factor changed against the row above.

| step | change | TF/s/dev | MFU | factor |
|---|---|---|---|---|
| 0 | baseline, seq 8192, pdb 1 | 43.1 | 3.7% | |
| 1 | seq 8192 -> 16384 | 56.0 | 4.9% | 1.30x |
| 2 | `use_gmm_v2=True` after the tile-clamp fix, plus the sparse-core XLA flags | 61.0 | 5.3% | 1.09x |
| 3 | `shard_exp_on_fsdp=True` | 63.5 | 5.5% | 1.04x |
| 4 | `TOKAMAX_KDA_DENSE_PAIRS=1` | 80.0 | 6.9% | **1.26x** |
| 5 | `TOKAMAX_KDA_BF16_FWD/BWD=1` | **81.5** | **7.1%** | 1.06x |

Cumulative **1.89x**. Measured neutral and dropped: `use_custom_sort_vjp`,
splash attention, host offload, `remat=custom`, tokamax GMM v1.

**bf16 KDA is a lever-order trap.** It is neutral on its own and worth 1.06x only
after `dense_pairs` removes the backward bottleneck it was hiding behind. A
one-factor-at-a-time sweep that tests it first would have discarded it.

Dispatch comparison at the default tiles that previously aborted:

| path | TF/s/dev |
|---|---|
| megablox | 43.3 |
| tokamax GMM v1 | 44.5 |
| **gmm_v2, after the clamp fix** | **45.9** |

### 128 devices (4x4x4, 16 nodes, 64 chips): run `o35nap240641`

`bodaborg-tpu7x-nap`, namespace `default`, 2026-09-24. First real 4x4x4.

| arm | pdb | seq | TF/s/dev | MFU | step s | outcome |
|---|---|---|---|---|---|---|
| a1_p1s8k | 1 | 8192 | 54.2 | 4.70% | 0.652 | ok |
| a2_p2s8k | 2 | 8192 | **78.2** | **6.78%** | 0.903 | **best** |
| a3_p4s8k | 4 | 8192 | | | | OOM, temporaries **104.02G vs 94.74G** |
| a4_p2s16k | 2 | 16384 | | | | OOM, **101.51G vs 94.74G** |
| b1_rematcustom | 4 | 8192 | 75.0 | 6.50% | 1.885 | fits only via offload |
| b2_ep4 | 4 | 8192 | | | | config error, unsharded params 24.17% |
| b3_noshardexp | 4 | 8192 | | | | OOM, 105.14G |
| b4_megablox | 4 | 8192 | | | | OOM |
| z_profile | 4 | 8192 | | | | OOM, **no capture** |

Raw logs: `gs://agagik-us/olmo35/4x4x4/o35nap240641/hc/`.

#### Why the estimator is a median and not a mean

The per-step trace for `a2_p2s8k` is not smooth, and reading a single step would
give an answer that is wrong by 2x in either direction:

| step | s | TF/s/dev | | step | s | TF/s/dev |
|---|---|---|---|---|---|---|
| 0 | 0.851 | 83.0 | | 10 | 0.906 | 78.0 |
| 1 | **0.379** | **186.3** | | 11 | **1.383** | **51.1** |
| 2 | **3.547** | **19.9** | | 12 | **0.433** | **163.3** |
| 3 | 0.902 | 78.4 | | 13 | 0.898 | 78.7 |
| 4..9 | 0.901-0.910 | 77.7-78.5 | | 14..19 | 0.901-0.906 | 78.0-78.4 |

Steady state is **0.90 s / 78.2 TF/s** and it is extremely stable, 77.7 to 78.7
across twelve steps. The outliers come in pairs, one fast then one slow, which is
the signature of asynchronous dispatch: the fast step measures dispatch while the
following step absorbs the blocked wait. A median over the last 10 rejects both
(it returns 78.22 here, against a mean of 82.0 that the 163.3 outlier drags up).

The same pattern appeared on the flex cluster at the identical configuration
(step 0 at 86.0, step 1 at 215.5), which is useful independent confirmation that
the two clusters agree and that the early fast readings are an artifact, not a
difference in hardware.

#### Where the memory goes

MaxText's own compile-time estimate, same run:

| pdb | estimated temp size |
|---|---|
| 1 | 52.5 GB |
| 2 | 79.8 GB |

That is **27.3 GB per unit of pdb** with a ~25 GB batch-independent floor, so
temporaries are activation-dominated and scale linearly with batch. Extrapolating
puts pdb=4 near 134 GB estimated against the 104.02 GB the runtime actually
demanded, so the compiler over-estimates (it says so in the log) but the slope is
the real constraint. Cutting ~10 GB is enough to reach pdb=4, which is why
`capacity_factor` is the first thing the c-series tries: it is `-1.0` (dropless)
in every run above, so nothing bounds the per-expert row count.

What it establishes:

1. **HBM is the binder, not compute.** pdb=4 is short by 9.3 GiB. Every pdb=4 arm
   died the same way.
2. **128 devices does not beat 8.** 78.2 against 81.5. Scaling this geometry out
   is flat to slightly negative, which retires the "7.1% is slice-limited"
   hypothesis.
3. **perfsim is 4.3x optimistic at 128 devices**, not the 1.7x borrowed from
   OLMoE3. It predicted 29.0% at pdb=2 against 6.78% measured. Direction
   transferred, magnitude did not: pdb 1 -> 2 was +44% measured against +100%
   predicted. Use perfsim for ranking only on this family.
4. **Offload buys memory, not throughput.** b1 is the only pdb=4 arm that ran and
   it still loses to pdb=2.

`b2_ep4` was a harness bug, not a hardware result: `ici_expert_parallelism=4`
with `ici_fsdp_parallelism=-1` leaves 24.17% of parameters unsharded and MaxText
asserts. `fsdp x ep` must equal the device count.

### 128 devices (4x4x4): run `o35fle241730`, in flight

`tpu7x-cluster-flex` pool `tpu7x-full-pod-spot`, no Kueue, spot. The spot
stockout that blocked this route for two days cleared at 18:25 UTC 2026-09-24 and
all 16 pods came up. Running the corrected c-series below. **Spot, so it can be
preempted mid-sweep**; results are pushed to GCS after every arm for that reason.

### Queued: `o35c241729` on nap, same c-series

| arm | pdb | seq | extra | what it answers |
|---|---|---|---|---|
| c1_prof_p2 | 2 | 8192 | `profiler=xplane` | capture at the point that actually runs |
| c2_p3s8k | 3 | 8192 | | does pdb=3 fit the 9.3 GiB gap |
| c3_ep4fix | 4 | 8192 | `ici_fsdp_parallelism=32 ici_expert_parallelism=4` | expert parallelism, sharding corrected |
| c4_p4_cf125 | 4 | 8192 | `capacity_factor=1.25` | does bounding expert capacity buy pdb=4 |
| c5_p4_optoff | 4 | 8192 | `optimizer_memory_host_offload=True` | cheaper offload than activation offload |
| c6_p2_megablox | 2 | 8192 | `megablox=True` | kernel choice at the pdb that runs |
| c7_p2_noshardexp | 2 | 8192 | `shard_exp_on_fsdp=False` | sharding lever at the pdb that runs |
| c8_p4_cf10 | 4 | 8192 | `capacity_factor=1.0` | tighter capacity bound |
| c9_p2_prof_cf125 | 2 | 8192 | `capacity_factor=1.25` | does capacity bounding cost throughput when memory is not binding |

`capacity_factor` is the untested MoE memory knob. It is `-1.0` (dropless) in
every run above, which is exactly what leaves the expert intermediates unbounded
and produces the 104 GiB temporaries.

## Profiles and xla-shell output

Captures are pulled and analysed with `scripts/olmo35_profile_report.sh`, which
locates the `.xplane.pb` under
`<base_output_directory>/<run>-<model>-<arm>/tensorboard/plugins/profile/` and
runs `analyze_profile` plus every `roadmap` view, writing each to a text file.

    scripts/olmo35_profile_report.sh gs://cloud-pathways-staging/agagik/olmo35-out o35fle241730 c1_prof_p2

### 8 devices, best configuration

The only analysis captured so far. Ran against the step-5 configuration above.

| lane | time |
|---|---|
| TensorCore | 906 ms |
| SparseCore | 372 ms |
| host DMA | 50 ms |

Verdict: **TensorCore-bound**, not comm-bound, once the flag set is applied. The
best-overlap moving floor is **372 ms** against a 1.10 s step, so most of the
remaining headroom was software.

`roadmap --kernels`, top entries:

| kernel | time | share of step |
|---|---|---|
| `_fused_dhu_wy_intra_cumsum_pallas_` | **303 ms** | **27%** |
| next three, combined | 51 + 44 + 28 ms | 11% |

The top kernel is the KDA intra-chunk factoring and it is **5.9x the next
kernel**. That single reading is what motivated `TOKAMAX_KDA_DENSE_PAIRS` and the
bf16 KDA flags, which together delivered the 1.34x in steps 4 and 5. It is also
why the latent-MoE fusion prototype and `tokamax_gmm_tile_m` were **not**
pursued: the roadmap puts those kernels at 51+44+28 ms, so even a perfect fix
there is worth far less than the KDA work.

### 128 devices

**Not yet captured.** The first attempt (`z_profile` in `o35nap240641`) was pinned
to pdb=4 and died with the OOM. `c1_prof_p2` re-targets it at pdb=2, the
configuration that actually runs, and is executing now in `o35fle241730`.

The open question the capture has to answer: at 8 devices the binder moved from
comm to TensorCore once tuned, but at 128 devices pdb=1 was 54.2 against pdb=2's
78.2, which is the signature of a large fixed collective cost being amortised.
So the 128-device profile should show whether comm is back on the critical path
at this slice, or whether the KDA cumsum is still the single largest kernel.

## Infrastructure notes that cost real time

| thing | note |
|---|---|
| Kueue TAS on nap | admits only against a **physically existing free topology domain**. Cohort said `nominal=307 used=57 free=250` while refusing a 64-chip ask with "62 more needed" |
| NAP deadlock on nap | NAP builds the pool shape a *Pending* pod asks for, but Kueue suspends the pod first, so NAP only ever saw other people's 2x2x1 requests. No bypass: `manageJobsWithoutQueueName: true` with every framework, and `create provisioningrequests` denied |
| multi-host JobSet on a scaled-to-zero pool | the webhook creates **only the leader pod** and logs `FailedCreate: leader pod not yet scheduled`. One Pending pod for a 16-node request is the normal pre-provision state |
| pod identity, read | shared-capacity pods read `gs://agagik-us`; multipod-dev pods were expected to need `gs://cloud-pathways-staging`, but in fact read `gs://agagik-us` fine. The launcher tries both |
| **pod identity, write** | **multipod-dev pods cannot write ANY bucket.** Both `gs://agagik-us` and `gs://cloud-pathways-staging` return `GcsApiError('')` on upload while reads succeed. This is not a 403 you see in the training log: it **hangs the MaxText summary writer**, so the run sits at step 1 forever with the python process at 349% CPU looking healthy. It also silently breaks the per-arm results upload. Diagnosed by `kubectl exec ... gcloud storage cp` from inside a live pod. Fix: those routes set a **local** `base_output_directory` and the supervisor copies `/tmp/hc` out of the leader pod over the API |
| SPS | does **not** propagate `LIBTPU_INIT_ARGS` (measured 318 vs 318 with the flags on and off), so XLA-flag tuning needs a non-SPS vehicle |
| spot 4x4x4 | `GCE out of resources` in us-central1-c for two days, then cleared without warning. Worth leaving a supervised job queued rather than polling by hand |
