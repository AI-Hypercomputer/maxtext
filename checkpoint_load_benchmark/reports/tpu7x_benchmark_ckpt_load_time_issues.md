# tpu7x Checkpoint Load-Time Benchmark: Issues, Capacity Limits, and Fixes

This document covers the TPU v7x (`bodaborg-tpu7x-nap`, `tpu7x-standard-4t`, topology `2x2x1`) runs of the
checkpoint load-time benchmark. It has three parts:
1. **Part 1, runtime failures and harness fixes**: every failure hit while running Priority 1 (>200B models) and
   Priority 2 (16-model suite) on tpu7x, the configuration that hit it, and how it was resolved.
2. **Part 2, MaxText issues and capacity limits**: problems in `safetensors_dynamic` and `to_maxtext` themselves.
3. **Part 3, commands**: how to reproduce every run.

The v5p findings are in [`benchmark_ckpt_load_time_issues.md`](benchmark_ckpt_load_time_issues.md) and stay
unchanged. Live per-test status: [`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md). Day-to-day log:
[`../handoff.md`](../handoff.md) §7.

**Hardware per slice:** 4 chips = 8 TPU cores, 813.8 GB HBM (8 × 101.7 GB), 1,014.5 GB host RAM (986.7 GB
allocatable to pods), and a ~100 GB boot disk. Large files can therefore only be staged in a RAM-backed `/ramdisk`.
Capacity is spot, admitted through Kueue queue `default`, which has 0 nominal tpu7x quota and borrows from cohort
`tpu-shared-cohort`.

---

## Part 1: Runtime failures and harness fixes

| ID | Affected runs | Where it failed | Symptom | Root cause | Resolution |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **T-1** | Every run on tpu7x | TPU init (any phase) | `Failed to connect to [::]:8353` | The harness stripped `*_SERVICE_*` env vars from worker subprocesses, which dropped `VBAR_CONTROL_SERVICE_URL=unix:///var/run/tpu-plugin/vbar.sock` that tpu7x needs. | Strip only `*_SERVICE_HOST`, `*_SERVICE_PORT*`, `*_TCP*`. |
| **T-2** | 2-slice runs | Remote slice launch | `[Errno 7] Argument list too long` | The harness ships its own source to the slice-1 daemon as one argument, which exceeded `MAX_ARG_STRLEN` (131,072 bytes). | zlib + base64 payload (~28 KB). |
| **T-3** | 2-slice runs | Interpreter teardown | Exit `-6` (SIGABRT) after a complete run | Multi-slice JAX teardown abort. | Don't mark a run failed when its result JSON is already complete. |
| **T-4** | All BEFORE runs before 13:00 UTC | Phase watchdog | Watchdog kills recorded as `OOMKilled / Exit Code 137` | Every `-9` exit was labeled OOM, including `PHASE_TIMEOUT_S` kills. | New records say `TIMEOUT: exceeded PHASE_TIMEOUT_S=<N>s`. Old records were checked against `[TIMEOUT_WATCHDOG]` lines (their peak host RAM was ≤194 GB). |
| **T-5** | First BEFORE (`origin/main`) pass | `safetensors_dynamic` Step 2 | Killed after 360 s | The original manifests set `PHASE_TIMEOUT_S=360`, shorter than the load times WITH the PRs (306–942 s), so these kills prove nothing. | Re-ran BEFORE with a 1800 s watchdog (`before_*` queue tasks, run dir `p1_origin_main/run_tpu7x_200b_t1800`). |
| **T-6** | All pods | Any | Pods disappear mid-run: Kueue `The workload is deactivated` (`c1`, `g1`), `Preempted ... due to reclamation within the cohort`, spot `imminent node shutdown` (`m3`, `s7`), `TaintManagerEviction` (`s6`), unreachable kubelet (node `fd9dfbd0`) | `default` borrows all of its chips, so other queues reclaim them, and spot nodes come and go. Most `The workload is deactivated` events happened ~3 h after the pod started (`c1`, `g1`, pair `i`, `s3`/`s8`, the 397B re-run `s1`). The likely enforcer is the calendar-schedule daemon in `kueue-system`: its code caps individual bookings at 3 h on NAP and can deactivate workloads. The exact rule wasn't confirmed, but in practice one run can't exceed ~3 h. | GCS task queue ([`scripts/tpu7x_task_queue.py`](../scripts/tpu7x_task_queue.py)): atomic locks, 60 s heartbeats, takeover of locks older than 20 min, and resume from the last finished phase (`--skip-completed`). |
| **T-7** | `to_maxtext` S2, scanned 235B (first two attempts) | Conversion | 2 h at leaf 1/15 | Shards were streamed from GCS through a 3-shard LRU cache, but a scanned MoE leaf stacks each expert across all layers, so every leaf reads every shard. | Stage the source locally when it fits, otherwise record why it can't run (see T-8). |
| **T-8** | `to_maxtext` S2: 235B (scanned and unscanned, with full staging) | Conversion | Pod `OOMKilled` at leaf 4/15 (scanned); pod evicted at 944 GB container memory (unscanned) | `to_maxtext` keeps the whole converted state (~checkpoint size) in host RAM until the Orbax save, on top of the 470 GB staged copy in `/ramdisk`. | The harness stages only when `free − checkpoint − 60 GB ≥ checkpoint`; otherwise it streams shards (unscanned) or fails fast with the reason (scanned). It also fails fast when the converted state alone exceeds 93% of host RAM. See U-4. |
| **T-9** | `to_maxtext` S1, 397B (13:26–13:38 UTC) | HF→GCS cache | Two workers re-downloaded 795 GB from HF Hub into the shared GCS cache that other runs were reading | TM Step 1 reused the shared cache only if the scanned 1-slice `safetensors_dynamic` record held a cold Step 1 time; 397B's record was an infra failure (`hf_to_gcs_s=0`). | Reuse the cache when all shards are present. The re-downloads were stopped. Interrupted uploads never replace an object, and the re-uploaded bytes are identical. |
| **T-10** | Running queue workers | Code deployment | Harness fixes (T-8, T-9) don't reach running pods | Workers ran the harness downloaded at pod start, and pods can't be exec'd into (no `pods/exec`). | Workers now sync the harness from GCS before each task. Tasks that needed the fixes before that were limited to newer pods with the `/tmp/maxtext_stagefix` tree. |
| **T-11** | P2 `qwen3-30b-a3b` unscanned | `to_maxtext` S2 | `HFValidationError` | The local staging dir from an older harness version (`/tmp/p2_lance_prs/hf_local/...`) was missing, so the path was treated as a repo id. | Re-stage into `/ramdisk/hf_local/` (re-run in `p2_leftovers`). |
| **T-12** | Docs | — | `SKIP_FAILED_RESULTS` described backwards | In code, `--skip-completed` re-runs FAILED records unless `SKIP_FAILED_RESULTS=1`. | Docs fixed. The 2-slice `to_maxtext` load tasks set `SKIP_FAILED_RESULTS=1`, so they never re-run a conversion that failed for capacity reasons. |
| **T-13** | P2 `llama3.1-70b` scanned · 2 | `to_maxtext` S3 | `ValueError: Requested more bytes than we reserved space for: 37580963840 > 20000000000` | The harness used `checkpoint_storage_concurrent_gb=20` for 2 slices (v5p tuning), but on tpu7x each process still requests the whole 37.58 GB scanned leaf. | 38 GB for all slice counts; re-run as `p2_llama70b_scanned_s2_tm`. |
| **T-14** | `qwen3-235b-a22b` unscanned · 2, first run | `to_maxtext` S3 | Slice 1: `TimeoutError: Timeout waiting at barrier '[op=1] _load_impl'` (`DEADLINE_EXCEEDED`); slice 0 then killed (`-9`) | An Orbax v1 multi-host barrier timed out while the two processes restored 1,131 arrays. | Transient: the retry (`tm_235b_unscanned_s2_retry`) loaded in 617.1 s. If it recurs, raise `ctx.asynchronous.timeout_secs` (MaxText sets 60 min with `enable_continuous_checkpointing`). |
| **T-15** | `to_maxtext` S2 shard streaming (unscanned 235B, 397B) | Conversion | 397B needed ~4.3 h, longer than the ~3 h workload limit (T-6) | The harness fetched each streamed shard (4–8.6 GB) as one sequential GCS read at ~55 MB/s, holding a global lock. | `_parallel_gcs_download`: 16 concurrent byte-range reads per shard (`TM_STREAM_DOWNLOAD_WORKERS`). 235B S2: 5,150 → 2,626 s (569 GB in 1,538 s, ~370 MB/s). Re-runs are in `run_tpu7x_200b_pstream`. |
| **T-16** | The U-1/U-6 fix (unit tests) | `safetensors_dynamic` load | `test_load_safetensors_dynamic_single_key` failed with `hbmfix`'s frees (deleted array) | Deleting a consumed HF tensor also deleted the loaded leaf: `jax.device_put` of an array that already has its target sharding returns an array on the same buffer. | `_delete_inputs` keeps every input that shares a buffer with a result (`unsafe_buffer_pointer`); part of the fix commit. |
| **T-17** | 10-30 runs: multi-host JobSets and queue `p3` workers | Kueue admission, spot nodes | JobSets pending with `topology "tpu-multihost-topology" doesn't allow to fit any of 1 slice(s) … excluded: nodeSelector`; single-host workers killed (`TaintManagerEviction`, exit 137) | The spot `2x2x2`/`2x2x4`/`2x4x4` nodes seen at 04:19 UTC (14 / 8 / 8) were gone by 05:40. The cluster's automation kept creating and deleting spot pools, which waited for GCE spot capacity; the few `2x2x2` nodes that came up went to older workloads. Kueue's spot borrowing limit follows the ready spot chips (56 → 28 → 72). A wave of node restarts (reserved nodes too) killed `t2` five times and `q0` once. | Pending JobSets hold no chips; preempted workers were re-created and took over their stale locks. Reserved capacity (booking system) wasn't used. |

---

## Part 2: MaxText issues and capacity limits

### U-1: `qwen3.5-397b-a17b` scanned runs out of HBM while splitting the `(wi_0, wi_1)` composite (2 slices): fixed on this branch

- **What happens:** each layer's `mlp.experts.gate_up_proj` (512 × 2048 × 4096 BF16, ~8.6 GB) is stacked over the
  45 local layers into one composite `(…, 2)` target (~387 GB, ~22.5 GiB per device on 16 devices). The composite is
  then sliced into `wi_0` and `wi_1`. Creating the first half needs 11.25 GiB while HBM holds the rest of the model
  plus the composite, so only 2.77 GiB is free.
- **Observed:** with Lance's PRs, the load finished in 483.8 s, then `RESOURCE_EXHAUSTED: ... allocate 11.25G, 2.77G
  free`. With [`patches/hbmfix`](../patches/hbmfix.patch) (frees consumed HF inputs and intermediates), the load took
  469.7 s and then failed at the same allocation. The composite is still alive while its first half is created.
  Lance's run on 1 × `2x2x2` (16 devices, ICI) failed the same way after a 488.2 s load.
- **Fix** (`src/maxtext/checkpoint_conversion/utils/load_dynamic.py` and `tensor_handling.py`, commit "Stack composite
  leaves one by one and join reshard slices in pairs"): a composite key whose source is a stacked list of HF tensors
  now splits each HF tensor right after its hook and stacks `wi_0` and `wi_1` as separate leaves (`num_parts`), so the
  stacked composite never exists. Unscanned composites keep the load-then-split path. Consumed HF tensors and the
  stack, cast, moveaxis and slice intermediates are freed as soon as their successor exists (the memory hygiene of
  `hbmfix`, plus the alias check of T-16).
- **Validated** with the fix tree (`/tmp/maxtext_fix`, run dirs `p1_lance_prs/run_tpu7x_200b_fix` and
  `lance_prs/run_tpu7x_multihost_fix`):

  | Configuration | With Lance's PRs | With the fix |
  | :--- | :--- | :--- |
  | 397B scanned · 2 × `2x2x1` | HBM OOM after a 483.8 s load | ✅ **1,608.3 s** (load 513.3 + transform 1,093.4; peak host RAM 93.0 GB) |
  | 397B scanned · 1 × `2x2x2` | HBM OOM after a 488.2 s load | ✅ **1,475.2 s** (load 483.5 + transform 991.6; peak host RAM 63.8 GB) |
  | 397B unscanned · 2 × `2x2x1` (regression check) | 1,311.9 s (515.9 + 795.7; 98.7 GB) | ✅ 1,272.2 s (478.9 + 793.3; 98.9 GB) |
  | 235B scanned · 1 × `2x2x1` (regression check) | 757.5 s (674.8 + 82.7; 391.8 GB) | ✅ 818.7 s (709.5 + 109.2; 390.7 GB) |

- **Workarounds measured before the fix:** unscanned on 2 slices works (1,311.9 s; per-layer composites are ~0.5 GB
  per device). Scanned on 4 slices also works: 1,399.7 s (load 260.0 s + transform 1,139.1 s).
- **Also on 1 slice:** the load finished (942.1 s), then the transform hit `RESOURCE_EXHAUSTED` (795 GB ≈ 98% of
  1-slice HBM). This is a capacity limit, not a bug.

### U-2: `origin/main` has no parameter mapping for `qwen3-480b-a35b`

`KeyError: 'qwen3-480b-a35b'` before any load. PR #5706 adds the mapping (composite keys, legacy key renames,
multi-host hooks).

### U-3: `origin/main` `safetensors_dynamic` loads don't finish within 1800 s

With the 1800 s watchdog, no `origin/main` load finished its load phase, on any model or slice count:

| Model | Config | `origin/main` | WITH the PRs (load + transform) |
| :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | scanned · 1 | timeout at 1800 s, still loading (peak host RAM 528.0 GB) | 757.5 s (674.8 + 82.7; 391.8 GB) |
| `qwen3-235b-a22b` | scanned · 2 | timeout at 1800 s, still loading (300.8 GB) | 792.4 s (708.0 + 83.6; 255.9 GB) |
| `deepseek4-284b` | scanned · 1 | timeout at 1800 s, still loading (501.4 GB) | 453.3 s (366.3 + 86.8; 963.4 GB) |
| `deepseek4-284b` | scanned · 2 | timeout at 1800 s, still loading (576.0 GB) | 440.7 s (352.9 + 87.6; 822.6 GB) |
| `qwen3.5-397b-a17b` | unscanned · 2 | timeout at 1800 s, still loading (301.7 GB) | 1,311.9 s (515.9 + 795.7; 98.7 GB) |
| `qwen3.5-397b-a17b` | scanned · 4 | timeout at 1800 s, still loading (316.6 GB) | 1,399.7 s (260.0 + 1,139.1; 127.7 GB) |
| `deepseek3-671b` | scanned · 4 | timeout at 1800 s, still loading (344.3 GB) | 649.4 s (468.9 + 176.8; 689.9 GB) |
| `qwen3-480b-a35b` | all | `KeyError` (U-2) | 992.2–1,067.9 s on 2 slices |

The harness skips the unscanned run after a scanned timeout for the same slice count. The likely cause is the
strided reads that PR #5828 replaces with leading-axis sharded reads; on v5p the same pattern produced ~245M byte
runs and multi-minute CPU stalls (v5p issue F-9).

> [!NOTE]
> The `origin/main` 397B load overlapped the re-download in T-9. The overlap can't explain the gap: with the PRs,
> the load phase alone takes 515.9 s.

### U-4: `to_maxtext` needs host RAM for the whole converted state

`to_maxtext` materializes every converted leaf (lazy loading only delays it) and keeps them all in host RAM until
the Orbax save. Scanned MoE leaves read every shard once per expert, so scanned conversions also need the whole
source staged locally. On tpu7x the only local storage large enough is RAM, which leads to these results on one
host (1,014.5 GB RAM):

| Model (checkpoint) | Scanned conversion | Unscanned conversion |
| :--- | :--- | :--- |
| `qwen3-235b-a22b` (470 GB) | ❌ staged source + converted state ≈ 940 GB + transients; OOM-killed at leaf 4/15 | ✅ 2,625.5 s with 16 parallel shard reads (5,150.1 s single-stream); peak host RAM 671.5 GB |
| `deepseek4-284b` (582 GB, synthetic sparse files) | ❌ OOM-killed at 989.6 GB | ❌ OOM-killed at 989.6 GB |
| `qwen3.5-397b-a17b` (807 GB on disk) | ❌ fails fast: the source must be staged | ✅ **8,096.5 s** with U-8's Orbax save-buffer cap and 16 parallel shard reads (peak host RAM 884.2 GB); ❌ host OOM-killed at the start of the Orbax save without the cap |
| `qwen3-480b-a35b` (960 GB) | ❌ fails fast: the source must be staged | ❌ fails fast: the converted state exceeds ~943 GB usable RAM |
| `deepseek3-671b` (1.34 TB) | ➖ not attempted | ➖ not attempted |

By contrast, `safetensors_dynamic` loads every one of these models directly from the HF shards, with no conversion
step: 235B and 284B on 1 slice, 480B on 2, and 671B on 4.

> [!NOTE]
> 284B's peak includes the synthetic sparse files: reading the holes of a sparse file on tmpfs through mmap
> allocates zeroed pages. With real files, the 582 GB staged source plus the 582 GB converted state wouldn't fit
> either.

### U-5: `deepseek3-671b` needs 4 slices for `safetensors_dynamic`

| Slices | Result |
| :--- | :--- |
| 1 | TPU HBM OOM (1,342 GB > 813.8 GB) |
| 2 | Host OOM-killed at a sampled 742.1 GB peak |
| 4 (32 cores, 3.26 TB HBM) | ✅ scanned 649.4 s (load 468.9 + transform 176.8); unscanned 531.0 s (409.9 + 120.7); 690 GB peak host RAM per host |

### U-6: `gpt-oss-120b` scanned `safetensors_dynamic` runs out of HBM in a jitted transform: fixed on this branch

With Lance's PRs, the scanned load of `gpt-oss-120b` (233.7 GB) finishes, then the transform fails right after the
router leaves: `RESOURCE_EXHAUSTED: ... HLO temporaries (200.32G) exceeds available HBM (94.74G). HLO module:
jit__lambda` on 1 slice, and 100.21G on 2 slices. `to_maxtext` converts and loads the same model (S2 581.4 s; S3
839.3 s on 1 slice and 894.0 s on 2 slices). Unscanned `safetensors_dynamic` works: 446.7 s on 1 slice and 390.7 s
on 2 (`p2_gptoss120b_unscanned_st`).

- **Cause:** `_reshard` (in `tensor_handling.py`) moves a leaf to its target sharding by slicing it into pieces and
  joining the pieces with one jitted `concatenate`. For the scanned expert weights that is up to 128 pieces. An XLA
  memory analysis of the real shardings on 8 CPU devices (a `(128, 18, 2880, 2880)` leaf, `P(None, None, 'fsdp',
  None)`) gives 17.6 GiB of temporaries per device for a 43-piece concatenate whose result is 4.45 GiB per device,
  and none for 2–8 pieces. On TPU, the failing module (the reshard of the 76 GB `(wi_0, wi_1)` composite) asked for
  200.32 GiB per device, ~22× that leaf's 8.9 GiB share.
- **Fix** (same commit as U-1): `_concatenate_pieces` joins the pieces two at a time, so each jitted concatenate has
  two operands.
- **Validated** with the fix tree (run dir `p2_lance_prs/run_tpu7x_16m_fix`):

  | Configuration | With Lance's PRs | With the fix |
  | :--- | :--- | :--- |
  | scanned · 1 slice | HBM OOM after a 253.6 s load | ✅ **505.1 s** (load 260.4 + transform 244.5; peak host RAM 44.6 GB) |
  | scanned · 2 slices | HBM OOM after a 126.9 s load | ✅ **430.2 s** (load 127.5 + transform 302.5; peak host RAM 71.4 GB) |

  End-to-end with the 350.3 s HF→GCS step, the fixed `safetensor` runs take 855.2 s and 780.3 s, vs 1,771.0 s and
  1,825.7 s for `to_maxtext` (2.07× and 2.34× faster).

### U-7: P2 timing outliers re-run

| Config | First run | Re-run (`run_tpu7x_16m_rerun`) | Conclusion |
| :--- | :--- | :--- | :--- |
| `qwen3-14b` scanned · 1, `safetensors_dynamic` | 342.7 s (transform 298.4 s) | 329.9 s (transform 289.7 s) | Reproducible: the 1-slice scanned transform is ~5× slower than on 2 slices (51 s); worth profiling |
| `qwen3-32b` scanned · 1, `safetensors_dynamic` | 377.7 s (transform 303.9 s) | 100.7 s (transform 18.7 s) | The first run was an outlier; the tables use the re-run |

### U-8: the `to_maxtext` Orbax save ignored `checkpoint_storage_concurrent_gb`: fixed on this branch

- **What happens:** `save_weights_to_checkpoint` (`src/maxtext/checkpoint_conversion/utils/utils.py`) created its
  checkpoint manager without `checkpoint_storage_concurrent_gb`, so the Orbax save used its default of 96 GB of
  concurrent write buffers. The save starts when every converted weight is already in host RAM, so these buffers come
  on top of the converted state: the unscanned 397B conversion (~807 GB converted state) was OOM-killed on a 1 TB
  tpu7x host right after its save started (U-4).
- **Fix** (commit "Cap the to_maxtext Orbax save buffers with checkpoint_storage_concurrent_gb"): the save passes the
  config's `checkpoint_storage_concurrent_gb`. The harness sets 24 GB for unscanned conversions.
- **Validated** (fix tree, run dirs `run_tpu7x_200b_tmfix` and `run_tpu7x_16m_tmfix`):
  - `llama3.1-70b` unscanned conversion (`tmfix_llama70b_unscanned_conv`): peak host RAM dropped from **403.6 GB** to
    **351.9 GB** (−51.7 GB; Step 2 total 640.4 s = transform 296.6 s + save 325.9 s vs 398.1 s = 281.9 s + 101.3 s).
  - `qwen3.5-397b-a17b` unscanned conversion (`tmfix_397b_unscanned_conv`): Step 2 completed in **8,096.5 s**
    (transform+shard 6,642.8 s with 16 parallel shard reads + Orbax save **1,432.0 s** for 738.3 GiB), at **884.2 GB**
    peak host RAM (was OOM-killed at the start of the save without the cap).
  - `qwen3.5-397b-a17b` unscanned Step 3 (Load Orbax):
    - 2 × `2x2x1` (`2 (16)`): **2,025.8 s** (peak host RAM 100.0 GB); Total (No Step 1) **10,122.3 s** vs `safetensor`
      **1,311.9 s** (**7.72×** speedup; E2E **10,936.8 s** vs **2,126.4 s** = **5.14×**).
    - 1 × `2x2x2` (`1 (16)`): **1,838.3 s** (peak host RAM 58.5 GB); Total (No Step 1) **9,934.8 s** vs `safetensor`
      **1,150.9 s** (**8.63×** speedup; E2E **10,749.3 s** vs **1,965.5 s** = **5.47×**).
    - 2 × `2x2x2` (`2 (32)`): on 4 hosts (`397b-tm-2s`), `ocp.load` timed out after **2,306 s** (31.5 s init + 2,274.6 s
      restore; peak host RAM 95.2 GB) with `TimeoutError: DEADLINE_EXCEEDED: Barrier timed out. Id: [op=1] _load_impl::0`
      (`# of tasks that reached the barrier: 3/4`). In Orbax v1, `multihost.coordination_timeout()` returns a hardcoded
      `_DEFAULT_BARRIER_TIMEOUT = 300` s (5 min), so when 4 hosts read thousands of unscanned Zarr3 layer chunks over
      ~33–38 min, a ~15% (>5 min) GCS completion-time skew between the fastest host (Task 3, ~32.9 min) and slowest host
      (Task 2, >37.9 min) triggers the 5-minute barrier timeout. Re-running with `_DEFAULT_BARRIER_TIMEOUT = 1800` s.

---

## Part 3: Commands

The harness is [`benchmark_ckpt_load_time.py`](../benchmark_ckpt_load_time.py). On tpu7x, every run is a task in
[`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json), executed by
[`scripts/tpu7x_task_queue.py`](../scripts/tpu7x_task_queue.py) on pods from
[`generate_tpu7x_manifests.py`](../generate_tpu7x_manifests.py).

```bash
# Upload the harness, bootstrap, queue worker and fix variants; deploy the P1 completion workers
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --priority p1-completion
kubectl apply -f <manifest dir>/tpu7x_p1_completion_scaleout.yaml     # more workers (pairs, singles, 4-slice group)

# One task, as a worker runs it (example: WITH-PRs safetensors_dynamic, 397B unscanned, 2 slices)
python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py \
  --models qwen3.5-397b-a17b --scan-modes unscanned --num-slices-list 2 --skip-to-maxtext \
  --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs --run-id tpu7x_200b \
  --local-work-dir /tmp/p1_lance_prs --devices-per-slice 8 --hardware tpu \
  --reuse-model-hf-cache --skip-completed --allow-step-failure \
  --remote-slice-addrs <slice-1 daemon host:55555>

# BEFORE (origin/main): same command from /tmp/maxtext_origin_main with PHASE_TIMEOUT_S=1800 and
#   --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main --run-id tpu7x_200b_t1800

# CSV from the per-phase result JSONs
python3 checkpoint_load_benchmark/scripts/build_tpu7x_tables.py <p2 logs mirror> <p1 logs mirror> \
  --cold-s1 qwen3.5-397b-a17b=814.53 --csv tpu7x_benchmark_ckpt_load_time_summary.csv --md table.md
# Row-by-row tables (plain text, for the sheet tabs): --suite p1 (>200B, add --multihost-csv) or --suite p2 (up to 120B)
python3 checkpoint_load_benchmark/scripts/build_tpu7x_sheet_table.py --suite p2 <p2 logs mirror> \
  --md-into tpu7x_benchmark_ckpt_load_time_summary.md
```
