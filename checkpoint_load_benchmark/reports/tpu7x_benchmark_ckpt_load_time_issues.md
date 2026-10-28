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
| **T-6** | All pods | Any | Pods disappear mid-run: Kueue `The workload is deactivated` (`c1`, `g1`), `Preempted ... due to reclamation within the cohort`, spot `imminent node shutdown` (`m3`, `s7`), `TaintManagerEviction` (`s6`), unreachable kubelet (node `fd9dfbd0`) | `default` borrows all of its chips, so other queues reclaim them, and spot nodes come and go. Most `The workload is deactivated` events happened ~3 h after the pod started (`c1`, `g1`, pair `i`, `s3`/`s8`, the 397B re-run `s1`). The cluster's calendar-schedule daemon (`kueue-system`) limits individual bookings to 3 h on NAP and deactivates workloads, so in practice one run can't exceed ~3 h. | GCS task queue ([`scripts/tpu7x_task_queue.py`](../scripts/tpu7x_task_queue.py)): atomic locks, 60 s heartbeats, takeover of locks older than 20 min, and resume from the last finished phase (`--skip-completed`). |
| **T-7** | `to_maxtext` S2, scanned 235B (first two attempts) | Conversion | 2 h at leaf 1/15 | Shards were streamed from GCS through a 3-shard LRU cache, but a scanned MoE leaf stacks each expert across all layers, so every leaf reads every shard. | Stage the source locally when it fits, otherwise record why it can't run (see T-8). |
| **T-8** | `to_maxtext` S2: 235B (scanned and unscanned, with full staging) | Conversion | Pod `OOMKilled` at leaf 4/15 (scanned); pod evicted at 944 GB container memory (unscanned) | `to_maxtext` keeps the whole converted state (~checkpoint size) in host RAM until the Orbax save, on top of the 470 GB staged copy in `/ramdisk`. | The harness stages only when `free − checkpoint − 60 GB ≥ checkpoint`; otherwise it streams shards (unscanned) or fails fast with the reason (scanned). It also fails fast when the converted state alone exceeds 93% of host RAM. See U-4. |
| **T-9** | `to_maxtext` S1, 397B (13:26–13:38 UTC) | HF→GCS cache | Two workers re-downloaded 795 GB from HF Hub into the shared GCS cache that other runs were reading | TM Step 1 reused the shared cache only if the scanned 1-slice `safetensors_dynamic` record held a cold Step 1 time; 397B's record was an infra failure (`hf_to_gcs_s=0`). | Reuse the cache when all shards are present. The re-downloads were stopped. Interrupted uploads never replace an object, and the re-uploaded bytes are identical. |
| **T-10** | Running queue workers | Code deployment | Harness fixes (T-8, T-9) don't reach running pods | Workers ran the harness downloaded at pod start, and pods can't be exec'd into (no `pods/exec`). | Workers now sync the harness from GCS before each task. Tasks that needed the fixes before that were limited to newer pods with the `/tmp/maxtext_stagefix` tree. |
| **T-11** | P2 `qwen3-30b-a3b` unscanned | `to_maxtext` S2 | `HFValidationError` | The local staging dir from an older harness version (`/tmp/p2_lance_prs/hf_local/...`) was missing, so the path was treated as a repo id. | Re-stage into `/ramdisk/hf_local/` (re-run in `p2_leftovers`). |
| **T-12** | Docs | — | `SKIP_FAILED_RESULTS` described backwards | In code, `--skip-completed` re-runs FAILED records unless `SKIP_FAILED_RESULTS=1`. | Docs fixed. The 2-slice `to_maxtext` load tasks set `SKIP_FAILED_RESULTS=1`, so they never re-run a conversion that failed for capacity reasons. |
| **T-13** | P2 `llama3.1-70b` scanned · 2 | `to_maxtext` S3 | `ValueError: Requested more bytes than we reserved space for: 37580963840 > 20000000000` | The harness used `checkpoint_storage_concurrent_gb=20` for 2 slices (v5p tuning), but on tpu7x each process still requests the whole 37.58 GB scanned leaf. | 38 GB for all slice counts; re-run as `p2_llama70b_scanned_s2_tm`. |
| **T-14** | `qwen3-235b-a22b` unscanned · 2, first run | `to_maxtext` S3 | Slice 1: `TimeoutError: Timeout waiting at barrier '[op=1] _load_impl'` (`DEADLINE_EXCEEDED`); slice 0 then killed (`-9`) | An Orbax v1 multi-host barrier timed out while the two processes restored 1,131 arrays. | Transient: the retry (`tm_235b_unscanned_s2_retry`) loaded in 617.1 s. If it recurs, raise `ctx.asynchronous.timeout_secs` (MaxText sets 60 min with `enable_continuous_checkpointing`). |

---

## Part 2: MaxText issues and capacity limits

### U-1: `qwen3.5-397b-a17b` scanned runs out of HBM while splitting the `(wi_0, wi_1)` composite (2 slices)

- **What happens:** each layer's `mlp.experts.gate_up_proj` (512 × 2048 × 4096 BF16, ~8.6 GB) is stacked over the
  45 local layers into one composite `(…, 2)` target (~387 GB, ~22.5 GiB per device on 16 devices). The composite is
  then sliced into `wi_0` and `wi_1`. Creating the first half needs 11.25 GiB while HBM holds the rest of the model
  plus the composite, so only 2.77 GiB is free.
- **Observed:** with Lance's PRs, the load finished in 483.8 s, then `RESOURCE_EXHAUSTED: ... allocate 11.25G, 2.77G
  free`. With [`patches/hbmfix`](../patches/hbmfix.patch) (frees consumed HF inputs and intermediates), the load took
  469.7 s and then failed at the same allocation. The composite is still alive while its first half is created.
- **Proposed fix:** split each layer's `gate_up_proj` into gate and up halves before stacking, and stack `wi_0` and
  `wi_1` separately, so that only one target leaf (~12 GB per device) is built at a time. Alternatively, move the
  composite to host memory before splitting it.
- **Workarounds measured:** unscanned on 2 slices works (1,311.9 s; per-layer composites are ~0.5 GB per device).
  Scanned on 4 slices also works: 1,399.7 s (load 260.0 s + transform 1,139.1 s).
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
| `qwen3-235b-a22b` (470 GB) | ❌ staged source + converted state ≈ 940 GB + transients; OOM-killed at leaf 4/15 | ✅ 5,150.1 s, streaming shards (peak host RAM 663.5 GB) |
| `deepseek4-284b` (582 GB, synthetic sparse files) | ❌ OOM-killed at 989.6 GB | ❌ OOM-killed at 989.6 GB |
| `qwen3.5-397b-a17b` (807 GB on disk) | ❌ fails fast: the source must be staged | ❌ not completed: streaming takes ~4.3 h (single-stream GCS reads at ~55 MB/s), longer than the ~3 h workload limit; deactivated at ~65% and ~40% |
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

### U-6: `gpt-oss-120b` scanned `safetensors_dynamic` runs out of HBM in a jitted transform

With Lance's PRs, the scanned load of `gpt-oss-120b` (233.7 GB) finishes, then the transform fails right after the
router leaves: `RESOURCE_EXHAUSTED: ... HLO temporaries (200.32G) exceeds available HBM (94.74G). HLO module:
jit__lambda` on 1 slice, and 100.21G on 2 slices. The temporaries halve with twice the devices, so the jitted expert
transform materializes ~7× the leaf per device. `to_maxtext` converts and loads the same model (S2 581.4 s; S3
839.3 s on 1 slice and 894.0 s on 2 slices), so for now this model is `to_maxtext`-only when scanned. Unscanned
`safetensors_dynamic` (skipped by the harness after the scanned failures) is re-run as `p2_gptoss120b_unscanned_st`.

### U-7: P2 timing outliers re-run

| Config | First run | Re-run (`run_tpu7x_16m_rerun`) | Conclusion |
| :--- | :--- | :--- | :--- |
| `qwen3-14b` scanned · 1, `safetensors_dynamic` | 342.7 s (transform 298.4 s) | 329.9 s (transform 289.7 s) | Reproducible: the 1-slice scanned transform is ~5× slower than on 2 slices (51 s); worth profiling |
| `qwen3-32b` scanned · 1, `safetensors_dynamic` | 377.7 s (transform 303.9 s) | 100.7 s (transform 18.7 s) | The first run was an outlier; the tables use the re-run |

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

# Tables and CSV from the per-phase result JSONs
python3 checkpoint_load_benchmark/scripts/build_tpu7x_tables.py <p2 logs mirror> <p1 logs mirror> \
  --cold-s1 qwen3.5-397b-a17b=814.53 --csv tpu7x_benchmark_ckpt_load_time_summary.csv --md table.md
```
