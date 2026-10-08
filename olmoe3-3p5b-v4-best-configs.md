# olmoe3-3p5b on TPU v4: best measured configurations

Model `src/maxtext/configs/models/olmoe3-3p5b.yml` (62.9B total, 3.48B active, 30 layers: 24 KDA + 6 full
attention, 512 experts top-16, latent 768). One v4-128 slice (4x4x4, 64 chips, megacore: 64 devices of
32 GiB) on `v4-128-bodaborg-us-central2-b`, pdb 1, 20 steps of synthetic data, median of the last 10 steps.
Every run is in `olmoe3-3p5b-v4-runs.md`; every artifact needed to rerun one is in GCS (see Reproduce).

## Best

MFU here is MaxText's logged TF/s per device over 275 (the stricter count); the report column uses the reference
report's 91.14 / 184.75 TFLOPs per step per chip.

| config | seq | step s | TF/s/dev (MaxText log) | MFU of 275 | MFU, report accounting | tok/s/chip | run, arm |
|---|---|---|---|---|---|---|---|
| **best 4k** | 4096 | **1.520** | 55.3 | **20.1%** | 21.8% | 2,695 | o3v4o081736 `o0_s4k_m0_ng_sr` |
| **best 8k** | 8192 | **3.095** | 55.1 | **20.0%** | 21.7% | 2,647 | o3v4r081817 `r0_s8k_k3_ng_sr` |
| best 4k at f32 KDA state and conv (strict precision) | 4096 | 1.558 | 53.9 | 19.6% | 21.3% | 2,629 | o3v4w082116 `q0_s4k_ours_f32` |
| best 8k at f32 KDA state and conv | 8192 | 3.167 | 53.9 | 19.6% | 21.2% | 2,587 | o3v4w082116 `q1_s8k_ours_f32` |
| morning best (scanned layers) | 4096 | 1.656 | 50.7 | 18.4% | 20.0% | 2,473 | o3v4x080744 `x0_s4k_tc2a_em` |
| morning best (scanned layers) | 8192 | 3.305 | 51.6 | 18.8% | 20.3% | 2,479 | o3v4y080801 `y0_s8k_tc4a_em` |
| start of the overnight climb | 4096 | 2.078 | 40.5 | 14.7% | 15.9% | 1,971 | o3v4j080423 `j0_s4k_both` |
| start of the overnight climb | 8192 | 4.948 | 34.5 | 12.5% | 13.6% | 1,656 | o3v4j080423 `j3_s8k_comb_dense` |
| first fitting run (2026-10-07) | 4096 | 3.545 | 23.7 | 8.6% | 9.3% | 1,155 | o3v4x071908 |
| reference report, Stage B (FSDP 4 x EP 16, bf16 weights) | 4096 | 1.995 | | | 16.6% | 2,053 | xid 297218573 |
| reference report, 8k live best (hc4) | 8192 | 4.494 | | | 15.0% | 1,823 | xid 297285516 |

Which FLOP count to trust: MaxText's (84.07 / 170.61 TF per step) is within 1.3% of a count from the real
parameter shapes (85.17 / 172.81 TF, `scripts/olmoe3_flops_check.py`); it leaves out only the KDA intra-chunk
matmuls. The report's 91.14 / 184.75 is 7% above that count and its derivation is not given, so it overstates MFU
by about 1.5 points. The logged column is the accurate one (see FLOP accounting below). The f32-state rows (`benchmarks/olmoe3_v4/
v4_arms_ours_f32.txt`) keep the KDA recurrent state and conv in float32. Both bests use MEGACORE_DENSE and f32 weights (bf16 weights do not train, see Cautions). Every lever is
loss-neutral (same model and math, changed only in scheduling, memory placement and kernels) except the bf16 KDA
state and conv (`gdn_state_dtype=bfloat16 kda_conv_in_compute_dtype=True`), a precision change: the f32-state
rows are the strictly neutral bests. In 20-step runs the bf16 state shows no loss difference beyond run-to-run
spread (4k 7.534 vs 7.525, 8k 8.336 vs 8.289 at step 19); a long real-data A/B has not been run.

## FLOP accounting (which MFU to trust)

Per token, forward + backward, from the real parameter shapes (`JAX_PLATFORMS=cpu python scripts/olmoe3_flops_check.py`):

| term | GFLOP/token, seq 4096 | seq 8192 |
|---|---|---|
| weight matmuls, 6 x 3.295 B active matmul params (routed 16/512, shared expert, latent down/up, router, KDA and attention projections, dense layer 0, LM head; input embedding excluded: it is a gather) | 19.769 | 19.769 |
| KDA state recurrence (24 layers) | 0.453 | 0.453 |
| KDA intra-chunk matmuls (chunk 128) | 0.264 | 0.264 |
| full attention core, causal (6 layers) | 0.302 | 0.604 |
| **total (causal attention)** | **20.79** (85.17 TF/step) | **21.10** (172.81 TF/step) |
| total with non-causal attention (PaLM convention) | 21.10 | 21.70 |
| MaxText's formula (logged TF/s) | 20.52 (84.07 TF/step): omits the KDA intra-chunk term | 20.83 (170.61) |
| reference report | 22.25 (91.14 TF/step): +7%, derivation not given | 22.55 (184.75) |

The 3.295 B active matmul parameters plus the 0.180 B input embedding give 3.475 B, the reference's active count.

MFU of the bests on each count (275 TF/s per chip, one megacore device per chip, 4096 / 8192 tokens per chip per step):

| config | step s | MaxText log | parameter-shape count (causal) | non-causal attention | reference report count |
|---|---|---|---|---|---|
| best 4k | 1.520 | 20.1% | **20.4%** | 20.7% | 21.8% |
| best 8k | 3.095 | 20.0% | **20.3%** | 20.9% | 21.7% |
| 4k, f32 KDA state | 1.558 | 19.6% | 19.9% | 20.2% | 21.3% |
| 8k, f32 KDA state | 3.167 | 19.6% | 19.8% | 20.4% | 21.2% |

## Reproduce

Everything for a run lives under `gs://agagik-us/olmo35/v4/runs/<run>/`:

| file | content |
|---|---|
| `src.tgz` | the exact source tree the pods ran (first on `PYTHONPATH`; the image supplies only dependencies) |
| `arms.txt` | the arms: `name\|pdb\|seq\|MaxText flags\|env (+libtpu flag)`, separated by `;` |
| `jobset.yaml` | the rendered JobSet: image, libtpu flags, the full train command |
| `hc/results.tsv`, `hc/olmoe3-3p5b-<arm>.log` | per-arm results (median of the last 10 steps) and full MaxText logs |
| `hc/*.xplane.pb` | profiles of the `*_prof*` arms |
| `pod.log` | the leader pod's stdout |

To rerun the bests:

    gcloud container clusters get-credentials v4-128-bodaborg-us-central2-b --region us-central2 \
        --project cloud-tpu-multipod-dev    # into KUBECONFIG=/tmp/kc-v4.yaml
    # either this tree, or the archived one: gcloud storage cp gs://agagik-us/olmo35/v4/runs/o3v4r081817/src.tgz .
    scripts/olmoe3_v4_launch.sh r benchmarks/olmoe3_v4/best_arms.txt
    scripts/olmoe3_v4_harvest.sh <run printed by the launcher>     # after the arms finish (HOLD_S keeps pods up)

`benchmarks/olmoe3_v4/best_arms.txt` holds the two best arms plus their profiled twins (the morning's scanned
bests are kept in `best_arms_morning.txt`; that file was reproduced to 3-7 ms in run o3v4b080846). The current
bests repeat within 3 ms of themselves: 4k 1.520 / 1.523 (o0, o1 profiled), 8k 3.103 / 3.106 for the same config
without the metric and synthetic-batch knobs (k3, o2 profiled). Reference profiles:
`gs://agagik-us/olmo35/v4/runs/o3v4o081736/hc/olmoe3-3p5b-o1_s4k_m0_prof.xplane.pb` (4k) and
`...o2_s8k_k3_prof.xplane.pb` (8k).

These two docs, the launch/harvest/analysis scripts, `benchmarks/olmoe3_v4/` and the final source tree are
also mirrored in `gs://agagik-us/olmo35/v4/docs/` (`src-final-20261008.tgz`), since the worktree changes are not
committed yet.

To analyze a profile:

    PYTHONPATH=<xla-shell> python -m xla_shell -c "read_xplane X.xplane.pb; analyze_profile; roadmap --all"
    PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_srcsurvey.py X.xplane.pb Xrecs.pkl
    PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_components.py X.xplane.pb Xrecs.pkl ["component|component"]

## Run index

All under `gs://agagik-us/olmo35/v4/runs/<run>/` (src.tgz, arms.txt, jobset.yaml, hc/ logs and profiles); arms
files are also in `benchmarks/olmoe3_v4/`. Series before j are backfilled without `src.tgz` (see the run record).

| run | arms file | what it tested | best arm |
|---|---|---|---|
| o3v4j080423 | v4_arms_j.txt | confirm i-series bests, buffer 1.25, profiles | j0 2.078 s (4k), j3 4.948 s (8k) |
| o3v4k080459 | v4_arms_k.txt | megacore megablox tiles, prefused weights | k0 1.877, k1 4.559 |
| o3v4l080509 | v4_arms_l.txt | expert-major all-to-all | (slower) |
| o3v4m080536 | v4_arms_m.txt | dense routing ops, expert-major with paired VJP | m0 1.833, m1 4.446 |
| o3v4n080602 | v4_arms_n.txt | token-chunked dispatch | n0 3.923 (8k) |
| o3v4o080611 | v4_arms_o.txt | sort-free local permute, `kda_wy` | o0 1.800, o2 3.867 |
| o3v4p080612 | v4_arms_p.txt | async / windowed collective flags | (neutral or worse) |
| o3v4q080629 | v4_arms_q.txt | Pallas KDA state scan, sub-block 8 | (slower) |
| o3v4r080639 | v4_arms_r.txt | paired all-to-all, vocab all-gather once, no 8k offload | r0 1.800 |
| o3v4s080705 | v4_arms_s.txt | MEGACORE_DENSE, async ragged all-to-all | s2 3.516 (8k) |
| o3v4t080708 | v4_arms_t.txt | Pallas scan taking `uw` | (slower) |
| o3v4u080715 | v4_arms_u.txt | vocab tiling, sub-block 8, chunk 256 | u0 1.781 |
| o3v4v080729 | v4_arms_v.txt | async ragged all-to-all + chunks | v0 1.674, v3 3.379 |
| o3v4w080737 | v4_arms_w2.txt | 8k host offload of MoE buffers, vocab tiling 1 | (slower) |
| o3v4x080744 | v4_arms_x2.txt | expert-major with overlap, pipelined copy, Pallas scan | **x0 1.656** |
| o3v4y080801 | v4_arms_y2.txt | 4 / 8 chunks with expert-major, profiles | **y0 3.305** |
| o3v4z080801 | v4_arms_z2.txt | 8k save choice, MEGACORE_DENSE off, vocab 16 | (slower or same) |
| o3v4a080830 | v4_arms_a2.txt | MEGACORE_DENSE at 4k, vocab 4 at 8k, buffer 1.25 | a1 3.295 (vocab 4) |
| o3v4b080846 | best_arms.txt | reproduction of the bests, reference profiles | 1.656 / 3.308 |
| o3v4c080906 | v4_arms_c2.txt | async all-gather, ragged RDMA size | (neutral or worse) |
| o3v4d080916 | v4_arms_d2.txt | fused KDA projection gradient sharding | (neutral) |

## Flags (on top of olmoe3-3p5b.yml and the launcher defaults)

| group | flags |
|---|---|
| kernels | `use_tokamax_kda=False use_tokamax_gmm=False use_gmm_v2=False megablox=True sparse_matmul=True` (tokamax does not run on v4) |
| layout | `ici_expert_parallelism=64 ici_fsdp_parallelism=1 shard_exp_on_fsdp=False` |
| layers | `scan_layers=False` (unrolled: faster, and frees most of the activation memory) |
| MoE routing | `capacity_factor=-1 ragged_buffer_factor=1.125 moe_lean_routing=True moe_topk_pallas=True emo_threshold_by_bisection=True` |
| MoE dispatch | `moe_a2a_expert_major=True`; `moe_a2a_token_chunks=2` at 4k, `=4` at 8k |
| megablox tiles (megacore) | `benchmarks/olmoe3_v4/tiles_megacore.txt`: m 512 everywhere; wi fwd n 896, wi dlhs n 384, wo fwd n 384, wo dlhs n 896, drhs n 896 / 384 |
| KDA | `kda_chunked_impl=subblock gdn_state_dtype=bfloat16 kda_fused_input_proj=True kda_conv_in_compute_dtype=True override_model_config=True gdn_chunk_size=128` |
| remat | `olmoe3_per_layer_remat=True remat_policy=custom moe_routing=device moe_x_sorted=device moe_combine=device`; 4k adds `moe_mlpwi_0=device`; 8k adds `kda_wy=device`; `decoder_layer_input=device` (default) at both |
| LM head | 4k `num_vocab_tiling=1`; 8k `num_vocab_tiling=4 vocab_tiling_ag_once=True` |
| megacore memory | environment variable `TPU_MEGACORE=MEGACORE_DENSE` at both (+1 GiB HBM) |
| metrics / benchmark | `norm_metrics=grad synthetic_data_reuse_batch=True` (training unchanged; together 3-20 ms) |
| precision | `dtype=bfloat16 weight_dtype=float32` |
| libtpu | `--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_scoped_vmem_limit_kib=16384 --xla_tpu_enable_async_ragged_all_to_all=true`; 4k adds `--xla_enable_async_all_gather=false` (with MEGACORE_DENSE XLA otherwise makes the weight all-gathers async, which loses 50 ms at 4k); 8k must keep them async (+404 ms if forced sync) |

Code paths that are always on under `moe_lean_routing` (no flag): dense top-k weights and group sizes, the
sort-free local permute, the paired all-to-all backward, and one all-gather of every chunk's group sizes.

## What each choice is worth (measured)

| choice | 4k | 8k | evidence |
|---|---|---|---|
| per-layer remat | needed to fit (-10 GB) | needed | AOT, v/w-series |
| EP 64 x FSDP 1 | 3.018 s (EP 16) -> 2.850 s; EP 4 / 8 / 32 slower | EP 16 +27 ms | x-series, y4, b6, i5 |
| sub-block KDA | -232 ms | -647 ms | y0, e2 |
| EMo bisection, save routing | -56 ms | | y1, y2 |
| megablox 512-row tiles | -37 ms | | z3, z4 |
| KDA micro-opts (stacked q/k, cumsum as matmul) | -54 ms | | b0 |
| bf16 KDA state + fused KDA projection | -25 ms | | b3 |
| buffer 1.125 (vs 1.25) | -101 ms | -213 ms | j2, j4 |
| KDA chunk 128 | -63 ms | -94 ms | f1, f0 |
| save MoE dispatch + combine (`moe_x_sorted`, `moe_combine`) | -225 ms | combine only: -182 ms | i2, i0 |
| `TPU_MEGACORE=MEGACORE_DENSE` | +48 to +84 ms with async weight all-gathers; neutral with `--xla_enable_async_all_gather=false` | -138 ms (scanned), -521 ms (async chunks) | i4, j1, e0, g2, h0/h2, z1 |
| vocab tiling 8 + all-gather once | | -27 ms | h2 |
| **megacore megablox tiles** (two n-tiles per GMM) | **-201 ms** | **-389 ms** | k0, k1 |
| **dense routing ops** (top-k weights, group sizes) | **-44 ms** | **-113 ms** | m0, m1 |
| **token-chunked dispatch** (2 chunks) | +19 ms (off) | **-523 ms** | n0, n2 |
| **sort-free local permute** | **-33 ms** | **-56 ms** | o0, o2 |
| vocab tiling 2 (from 4) | -19 ms | | u0 |
| **async ragged all-to-all over token chunks** | **-107 ms** (2 chunks) | **-357 ms** (2), -494 ms (4) | v0, s2, v3 |
| **expert-major all-to-all, once overlapped** | **-18 ms** | **-74 ms** (4 chunks), -168 ms (2) | x0, y0, x1 |
| **unrolled layers** (`scan_layers=False`) | **-35 ms** | **-90 ms** | f2, f3 |
| vocab tiling 1 at 4k, 4 at 8k (unrolled) | **-56 ms** (h1 to h0; the same without MEGACORE_DENSE, h2) | -2 ms | h0, h1, h2, i0 |
| **save the dispatch tokens too at 8k** (`moe_x_sorted`, fits once unrolled) | (already) | **-86 ms** | k0 |
| layer inputs on device at 8k (no host offload) | (already) | -21 ms | k1 |
| `kda_wy` saves at 8k | +0 ms | -3 ms | l1, k3 |
| `context` saves at 4k (not with the GMM save) | -20 ms | OOM | l0, k2 |
| **save the routed gate GMM output** (`moe_mlpwi_0`) | **-39 ms** | OOM | m0, n0 |
| `norm_metrics=grad` + synthetic batch reuse | -3 to -20 ms | -8 ms | o0, r0 |

Measured and not adopted (overnight): expert-major all-to-all without overlap (+167 ms at 4k), prefused
gate/up weights (+56 ms), `kda_wy` saves (+17 ms), the Pallas KDA state scan (+41 to +51 ms in the model although
2.6x faster alone: operand relayouts around the custom call), async all-gather (+81 ms), async all-reduce,
windowed einsum, megacore-fusion all-gathers and pipelined local copy (neutral), sub-block 8 (+9 ms), chunk 256
(+134 ms), vocab tiling 8 at 4k and 16 at 8k, `vocab_tiling_ag_once` at 4k (+30 ms), host offload of the MoE
buffers at 8k (+490 to +640 ms), keeping layer inputs on device at 8k (+215 ms), `moe_x_sorted` instead of
`moe_combine` at 8k (+97 ms), MEGACORE_DENSE off at 8k (+521 ms). Earlier: pdb 2, bf16 weights, libtpu ragged
all-to-all knobs alone, megablox 1024-row tiles, scan unroll, saving projections, VMEM 64M.

## Reference report: what carried over

| report finding | here |
|---|---|
| FSDP 4 x EP 16 is the best 2D layout on 4x4x4 | EP 64 x FSDP 1 is faster on this cluster at every stage measured (x-series, b6, i5) |
| `weight_dtype=bfloat16` (-4.2 ms per layer) | does not train (loss 10.32 vs 8.49 at step 19, h4); f32 weights kept |
| 256-aligned, k = full GMM tiles (zero-scratch fast path) | already the case (k 768 / 1792); the larger miss was megacore: one n-tile ran each GMM on one TensorCore |
| fused / hybrid LatentMoE kernel, 1.5-1.7x over XLA `ragged_dot` | megablox with megacore tiles runs the expert block at 192 TF/s fwd+bwd, above the report's fused (141) and hybrid (152) kernels; at ~83% of MXU peak there is little left to fuse on v4 |
| MoE saves `moe_routing` + `moe_x_sorted` (-98 ms) | same idea, split into `moe_x_sorted` (dispatch) and `moe_combine` (combine): -225 ms |
| KDA `context` saves (2-stage checkpointing) | `context=device` does not fit here (34.0 G temps); the narrower `kda_wy` fits but is +17 ms |
| MEGACORE_DENSE (+1 GiB) | used at both: -521 ms at 8k; at 4k neutral only with `--xla_enable_async_all_gather=false` (XLA otherwise spends the GiB on async weight all-gathers, +48 to +84 ms) |
| `--xla_jf_rematerialization_percent_shared_memory_limit=98`, `--xla_tpu_user_reserved_hbm_bytes=0` | flags exist in this libtpu; the 8k both-saves config needs 32.3 G temps, above what they free |
| async collective flags | +81 ms (all-gather) or neutral here |
| chunk 64 / 32 for KDA | chunk 128 is best with the sub-block kernel (f-series) |

## Cautions

| item | why |
|---|---|
| `ragged_buffer_factor=1.125` | drops overflow tokens of experts hotter than 1.125x; probe real packed data with `log_required_ragged_buffer_factor=true` first, else use 1.25 (+61 ms at 4k, +128 ms at 8k on the current bests: s0, s1) |
| `weight_dtype=bfloat16` (reference report) | does not train: loss 10.32 vs 8.49 at step 19 (h4); bf16 master weights and bf16 Adam state round away the updates |
| bf16 KDA state | no measurable cost: 2.09e-2 vs 2.08e-2 error against an exact scan, loss 7.784 vs 7.797 |
| loss at step 19 across arms | moves by up to 0.012 even for exact changes (gradient sums reassociate; synthetic data amplifies it); exactness is pinned by unit tests |
| memory check | XLA's compile-time temporaries check is the real limit; runs with args + temps above the chip size work |
| synthetic data | about 30 ms of every step is the synthetic batch generator, which real data would replace with a host transfer |
