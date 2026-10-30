# Multi-host slices on TPU v7x (`bodaborg-tpu7x-nap`)

> **Snapshot:** 2026-10-29 ~19:30 UTC · **Owner:** Lance Wang · **Branch:** `docs/checkpoint-load-benchmark`
>
> The P1 runs ([`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md)) use single-host `2x2x1` slices. These
> runs load the same models on multi-host slices. The runs are `MH_RUNS` in
> [`generate_tpu7x_manifests.py`](../generate_tpu7x_manifests.py), deployed with
> `scripts/run_tpu7x_cluster.sh --priority multihost --run <id>` (README §5.1, "Multi-host slices"). The table data is
> in [`tpu7x_multihost_results.csv`](tpu7x_multihost_results.csv).

## Setup

- Every run is `safetensors_dynamic` WITH Lance's 3 PRs (#5706, #5828, #5705), on the same code tree and bootstrap
  patches as the P1 runs. `to_maxtext` and BEFORE (`origin/main`) were not run on multi-host slices.
- Qwen models read the shared P1 HF cache in GCS. DeepSeek models read synthetic sparse BF16 files on each host's local
  disk (`STAGE_SPARSE_DEEPSEEK_OOM=1`), as in P1: they measure load mechanics and capacity, not GCS read bandwidth.
- Each tpu7x host has 4 chips (8 TPU cores, 813.8 GB HBM) and 1,014.5 GB RAM.

| Layout | Hosts | Devices | HBM | Hosts connected over |
| :--- | ---: | ---: | ---: | :--- |
| 1 × `2x2x2` | 2 | 16 | 1,627.6 GB | ICI |
| 1 × `2x2x4` | 4 | 32 | 3,255.2 GB | ICI |
| 2 × `2x2x2` | 4 | 32 | 3,255.2 GB | ICI within a slice, DCN between the slices |
| 4 × `2x2x2` | 8 | 64 | 6,510.4 GB | ICI within a slice, DCN between the slices |
| 2 × `2x2x1`, 4 × `2x2x1` (P1) | 2, 4 | 16, 32 | 1,627.6 GB, 3,255.2 GB | DCN |

- **Load** = `load_sharded_hf_state`, **Transform** = `transform_hf_state_to_mt_state`, **Total** =
  Safetensors+Convert. **Peak host RAM** is the runner host's (JAX process 0), as in the P1 reports.
- Rows marked "P1 (mesakh)" are the P1 multi-slice runs with the same number of devices.

## Results

| Model | Scan mode | Layout | Hosts | Devices | Link | Total (s) | Load (s) | Transform (s) | Peak host RAM (GB) | Run |
| :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| `qwen3-0.6b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | **18.03** | 8.99 | 9.00 | 38.2 | `smoke` |
| `qwen3-0.6b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **19.34** | 8.66 | 10.67 | 78.3 | `smoke-2s` |
| `qwen3-0.6b` | unscanned | 1 × 2x2x2 | 2 | 16 | ICI | **9.61** | 6.32 | 3.28 | 38.0 | `smoke` |
| `qwen3-0.6b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **11.50** | 5.50 | 5.95 | 78.1 | `smoke-2s` |
| `qwen3-235b-a22b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | **908.33** | 826.12 | 82.18 | 220.1 | `235b` |
| `qwen3-235b-a22b` | scanned | 2 × 2x2x1 | 2 | 16 | DCN | **792.38** | 707.97 | 83.58 | 255.8 | P1 (mesakh) |
| `qwen3-235b-a22b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **800.69** | 713.38 | 86.77 | 182.8 | `235b-2s` |
| `qwen3-235b-a22b` | unscanned | 1 × 2x2x2 | 2 | 16 | ICI | **817.59** | 775.01 | 42.52 | 222.0 | `235b` |
| `qwen3-235b-a22b` | unscanned | 2 × 2x2x1 | 2 | 16 | DCN | **749.00** | 704.88 | 44.03 | 254.6 | P1 (mesakh) |
| `qwen3-235b-a22b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **788.66** | 736.01 | 52.54 | 177.7 | `235b-2s` |
| `deepseek4-284b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | **414.93** | 327.55 | 87.14 | 534.7 | `284b` |
| `deepseek4-284b` | scanned | 2 × 2x2x1 | 2 | 16 | DCN | **440.72** | 352.94 | 87.63 | 822.6 | P1 (mesakh) |
| `deepseek4-284b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **302.91** | 209.34 | 93.48 | 318.4 | `284b-2s` |
| `deepseek4-284b` | unscanned | 1 × 2x2x2 | 2 | 16 | ICI | **349.57** | 307.45 | 41.90 | 520.1 | `284b` |
| `deepseek4-284b` | unscanned | 2 × 2x2x1 | 2 | 16 | DCN | **352.96** | 306.30 | 46.47 | 822.6 | P1 (mesakh) |
| `deepseek4-284b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **258.40** | 200.93 | 57.23 | 318.3 | `284b-2s` |
| `qwen3.5-397b-a17b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | ❌ HBM OOM in transform (load 488.18 s; 11.25 GiB needed, 2.77 GiB free) |  |  |  | `397b` |
| `qwen3.5-397b-a17b` | scanned | 2 × 2x2x1 | 2 | 16 | DCN | ❌ HBM OOM in transform (load 483.84 s), also with `hbmfix` |  |  |  | P1 (mesakh) |
| `qwen3.5-397b-a17b` | scanned | 1 × 2x2x4 | 4 | 32 | ICI | **2,618.13** | 261.18 | 2,356.85 | 88.6 | `397b-4h` |
| `qwen3.5-397b-a17b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **1,361.95** | 248.24 | 1,113.67 | 130.8 | `397b-2s` |
| `qwen3.5-397b-a17b` | scanned | 4 × 2x2x1 | 4 | 32 | DCN | **1,399.68** | 260.01 | 1,139.07 | 127.7 | P1 (mesakh) |
| `qwen3.5-397b-a17b` | unscanned | 1 × 2x2x2 | 2 | 16 | ICI | **1,150.92** | 484.68 | 666.14 | 48.9 | `397b-u` |
| `qwen3.5-397b-a17b` | unscanned | 2 × 2x2x1 | 2 | 16 | DCN | **1,311.85** | 515.91 | 795.72 | 98.7 | P1 (mesakh) |
| `qwen3.5-397b-a17b` | unscanned | 1 × 2x2x4 | 4 | 32 | ICI | **954.34** | 268.01 | 686.26 | 88.1 | `397b-4h` |
| `qwen3.5-397b-a17b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **1,018.83** | 266.41 | 751.78 | 130.5 | `397b-2s` |
| `qwen3-480b-a35b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | **1,298.61** | 953.07 | 345.50 | 380.6 | `480b` |
| `qwen3-480b-a35b` | scanned | 2 × 2x2x1 | 2 | 16 | DCN | **1,067.86** | 941.98 | 125.78 | 415.7 | P1 (mesakh) |
| `qwen3-480b-a35b` | unscanned | 1 × 2x2x2 | 2 | 16 | ICI | **1,231.98** | 938.51 | 293.40 | 380.5 | `480b` |
| `qwen3-480b-a35b` | unscanned | 2 × 2x2x1 | 2 | 16 | DCN | **992.17** | 907.65 | 84.30 | 415.5 | P1 (mesakh) |
| `deepseek3-671b` | scanned | 1 × 2x2x4 | 4 | 32 | ICI | **749.86** | 451.28 | 298.30 | 654.9 | `671b` |
| `deepseek3-671b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **657.75** | 490.93 | 166.72 | 673.7 | `671b-2s` |
| `deepseek3-671b` | scanned | 4 × 2x2x1 | 4 | 32 | DCN | **649.35** | 468.92 | 176.84 | 689.9 | P1 (mesakh) |
| `deepseek3-671b` | unscanned | 1 × 2x2x4 | 4 | 32 | ICI | **672.66** | 468.79 | 203.51 | 655.0 | `671b` |
| `deepseek3-671b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **587.07** | 480.07 | 106.76 | 645.0 | `671b-2s` |
| `deepseek3-671b` | unscanned | 4 × 2x2x1 | 4 | 32 | DCN | **530.96** | 409.94 | 120.74 | 689.9 | P1 (mesakh) |
| `deepseek3.2-671b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **672.46** | 493.25 | 178.76 | 653.4 | `ds32-2s` |
| `deepseek3.2-671b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **569.71** | 459.39 | 109.96 | 690.9 | `ds32-2s` |
| `deepseek4-1600b` | scanned | 4 × 2x2x2 | 8 | 64 | ICI + DCN | **1,047.93** | 757.75 | 289.55 | 696.1 | `1600b-4s` |
| `deepseek4-1600b` | unscanned | 4 × 2x2x2 | 8 | 64 | ICI + DCN | **1,008.62** | 771.74 | 236.49 | 676.0 | `1600b-4s` |

## Findings

1. **`qwen3.5-397b-a17b` scanned needs 32 devices.** On 16 devices it runs out of HBM while splitting the stacked
   `(wi_0, wi_1)` composite, on one `2x2x2` slice and on 2 × `2x2x1` alike. On 32 devices it loads on every layout:
   2 × `2x2x2` takes 1,361.95 s, 4 × `2x2x1` 1,399.68 s, and one `2x2x4` slice 2,618.13 s.
2. **On 32 devices, the scanned transform is slowest on one `2x2x4` slice.** The load takes about the same time on
   every layout (410–491 s for `deepseek3-671b`, 248–268 s for `qwen3.5-397b-a17b`), but the scanned transform takes:
   - `deepseek3-671b`: 298.30 s on one `2x2x4` slice, 166.72 s on 2 × `2x2x2`, 176.84 s on 4 × `2x2x1`.
   - `qwen3.5-397b-a17b`: 2,356.85 s on one `2x2x4` slice, 1,113.67 s on 2 × `2x2x2`, 1,139.07 s on 4 × `2x2x1`.

   Unscanned, the gap is smaller for `deepseek3-671b` (203.51 vs 106.76 / 120.74 s), and `qwen3.5-397b-a17b` is
   faster on one `2x2x4` slice (686.26 vs 751.78 s on 2 × `2x2x2`). The cause is not verified.
3. **On 16 devices, one `2x2x2` slice vs 2 × `2x2x1` is mixed:**
   - `qwen3-235b-a22b` is slower (908.33 vs 792.38 s scanned). The gap is in the GCS read; the transform is equal.
   - `qwen3-480b-a35b` is slower (1,298.61 vs 1,067.86 s scanned). The read is equal; the transform takes 345.50 vs
     125.78 s.
   - `deepseek4-284b` and `qwen3.5-397b-a17b` unscanned are faster (414.93 vs 440.72 s, 1,150.92 vs 1,311.85 s).
4. **Peak host RAM per host is lower on one `2x2x2` slice** than on 2 × `2x2x1`, for example 534.7 vs 822.6 GB for
   `deepseek4-284b` scanned.
5. **More devices do not speed up `qwen3-235b-a22b`.** 2 × `2x2x2` (32 devices) takes 800.69 / 788.66 s (scanned /
   unscanned), close to 2 × `2x2x1` (16 devices, 792.38 / 749.00 s): 713–736 s of it is the GCS read.
6. **`deepseek3.2-671b` and `deepseek4-1600b` load on tpu7x.**
   - `deepseek3.2-671b` on 2 × `2x2x2`: 672.46 / 569.71 s, close to `deepseek3-671b` on the same layout
     (657.75 / 587.07 s).
   - `deepseek4-1600b` (3,197.7 GB) on 4 × `2x2x2` (64 devices, 6,510.4 GB HBM): 1,047.93 / 1,008.62 s, with
     676–696 GB peak host RAM per host.

## Not run

- `480b-2s` (`qwen3-480b-a35b` on 2 × `2x2x2`): its JobSet was redeployed at 17:02 UTC, after the admission policy
  in the notes started rewriting new JobSets, so it never started.
- The v5p runs (`--platform v5p`, `V5P_MH_RUNS`): `v5p-smoke-2s` (17:56 UTC) hit the same policy on
  `bodaborg-v5p-nap`.

## Notes

- **JobSet admission policy.** Since 2026-10-29 16:43 UTC, the Kyverno ClusterPolicy `owner-only-exec-stamp-creator`
  on `bodaborg-tpu7x-nap` and `bodaborg-v5p-nap` rewrites every new JobSet that has no ownerReferences: rule
  `stamp-creator-jobset-templates` keeps only the last replicated job and drops its pod spec. The Jobs then fail
  (`spec.template.spec.containers: Required value`), and Kueue creates no Workload. JobSets created before then, and
  xpk JobSets (which have a ConfigMap ownerReference), are not affected. New multi-host runs need the policy fixed.
- **Spot node failures.** On 10-29, several spot `2x2x2` node pools went NotReady during the multi-slice runs. Kueue
  evicted the workloads (`NodeFailures`), the JobSets restarted (`maxRestarts: 2`), and `--skip-completed` resumed
  from the per-phase result JSONs in GCS. When a remote slice dies, the runner's local worker is killed, and the
  harness records it as `OOMKilled / Exit Code 137` although the peak host RAM was far below the limit (handoff §7.4).
  Those records were re-run; the table has only the completed runs.
- **Scan-mode order.** The harness skips unscanned when scanned failed and recorded no load time, which a scanned HBM
  OOM does. So the 397B unscanned point on one `2x2x2` slice is a separate run (`397b-u`), and `397b-2s` and
  `397b-4h` run unscanned first.
- **JAX process order.** libtpu can number the processes differently from the harness: the runner got process index
  1 or 2. The runner (`--slice-id 0`) writes the result JSON, so the results are not affected.
- **Where the data is.** Result CSVs: `gs://mesa-maxtext/wanglance_tpu7x_bench/results/<run>.csv`. Per-phase result
  JSONs: `gs://mesa-maxtext/wanglance_tpu7x_bench/lance_prs/run_tpu7x_multihost/logs/<model>/<scan_mode>[_slices<N>]_hosts<H>/`.

## 2026-10-30 Multi-Host Runs (`mesakh`)

After the Kyverno `owner-only-exec-stamp-creator` admission policy on `bodaborg-tpu7x-nap` was fixed, we ran the
remaining multi-host `safetensors_dynamic` configurations (`MESAKH_MH_RUNS` in
[`generate_tpu7x_manifests.py`](../generate_tpu7x_manifests.py)), the multi-host `safetensor (with fix)` validation for
`qwen3.5-397b-a17b` scanned on 1 × `2x2x2`, and multi-host `to_maxtext` Step 3 (Load Orbax) runs reusing the converted
single-host Orbax checkpoints.

### `safetensors_dynamic` (`source=mesakh` in [`tpu7x_multihost_results.csv`](tpu7x_multihost_results.csv))

| Model | Scan mode | Layout | Hosts | Devices | Link | Total (s) | Load (s) | Transform (s) | Peak host RAM (GB) | Run / Note |
| :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| `qwen3-0.6b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **20.08** | 9.41 | 10.57 | 72.3 | `smoke-tm-2s` |
| `qwen3-0.6b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **16.29** | 10.18 | 6.00 | 72.3 | `smoke-tm-2s` |
| `qwen3.5-397b-a17b` | scanned | 1 × 2x2x2 | 2 | 16 | ICI | **1,475.17** | 483.50 | 991.62 | 63.8 | `397b-fix` (`with fix`: U-1/U-6 HBM-peak fixes; was HBM OOM) |
| `qwen3-480b-a35b` | scanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **757.70** | 644.22 | 111.22 | 244.9 | `480b-2s` |
| `qwen3-480b-a35b` | unscanned | 2 × 2x2x2 | 4 | 32 | ICI + DCN | **718.65** | 642.10 | 76.42 | 244.6 | `480b-2s` |

### `to_maxtext` Step 3 (Load Orbax) on Multi-Host Slices (`source=mesakh`)

Each run loads the Orbax checkpoint written by the single-host (`1 (8)`) conversion (`--skip-to-maxtext-step1-and-2`),
comparing against `safetensors_dynamic` on the same multi-host layout:

| Model | Scan mode | Layout | Hosts | Devices | Step 1 HF→GCS (s) | Step 2 Convert (s) | Step 3 Load Orbax (s) | `to_maxtext` Total (No Step 1 / E2E) (s) | `safetensor` Total (No Step 1 / E2E) (s) | Speedup (E2E / No Step 1) | Run |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `qwen3-0.6b` | scanned | 2 × 2x2x2 | 4 | 32 | 15.98 | 19.40 | **2.94** (peak 59.0 GB) | 22.35 / 38.33 | 19.34 / 35.32 | **1.09× / 1.16×** | `smoke-tm-2s` |
| `qwen3-0.6b` | unscanned | 2 × 2x2x2 | 4 | 32 | 15.98 | 66.78 | **4.03** (peak 59.1 GB) | 70.81 / 86.79 | 11.50 / 27.48 | **3.16× / 6.16×** | `smoke-tm-2s` |
| `qwen3-235b-a22b` | unscanned | 1 × 2x2x2 | 2 | 16 | 583.32 | 2,625.50 | **841.12** (peak 54.9 GB) | 3,466.62 / 4,049.94 | 817.59 / 1,400.91 | **2.89× / 4.24×** | `235b-tm` |
| `qwen3-235b-a22b` | unscanned | 2 × 2x2x2 | 4 | 32 | 583.32 | 2,625.50 | **682.05** (peak 90.4 GB) | 3,307.56 / 3,890.87 | 788.66 / 1,371.98 | **2.84× / 4.19×** | `235b-tm-2s` |
| `qwen3.5-397b-a17b` | unscanned | 1 × 2x2x2 | 2 | 16 | 814.53 | 8,096.53 | **1,838.26** (peak 58.5 GB) | 9,934.79 / 10,749.32 | 1,150.92 / 1,965.45 | **5.47× / 8.63×** | `397b-tm` |

