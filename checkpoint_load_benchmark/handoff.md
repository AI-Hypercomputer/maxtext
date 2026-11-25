# Handoff: Checkpoint Load-Time Benchmark (`v2` 4-Platform Suite: `v7x`, `v6e`, `v5p`, `v5e` + `tpu7x` Legacy Archive)

> **Turnkey Team Reproduction (`REPRODUCE.md`):** For copy-paste commands to reproduce the local CPU test suite (`bash checkpoint_load_benchmark/scripts/repro/run_cpu_repro.sh`), real GCS DeepSeek shard streaming checks, live GKE TPU parity JobSets, and Benchmark `v2` + **PR #6398 / #6353 / #6354** runs, see **[`REPRODUCE.md`](REPRODUCE.md)**.

> **Snapshot:** 2026-11-17 09:08 UTC · **Branch:** `docs/checkpoint-load-benchmark` · **Executive Summary:** [`reports/benchmark_v2_summary.md`](reports/benchmark_v2_summary.md) · **Customer & Buganizer Reports:** [`reports/deepseek_chat_message_short.md`](reports/deepseek_chat_message_short.md), [`reports/deepseek_bug_followup_message.md`](reports/deepseek_bug_followup_message.md) & [`reports/b586746548_deepseek_dequant_issue.md`](reports/b586746548_deepseek_dequant_issue.md) · **Live Progress Log:** [`PROGRESS_LOG.md`](PROGRESS_LOG.md) · **Design Plan:** [`plans/benchmark_v2_plan.md`](plans/benchmark_v2_plan.md) · **Live Google Sheet v2:** [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit) · **Pre-Edit Full Backup Sheet (Never Touch):** `1xZ0ifZW5yD0gPT8CpcsxQ1Q8ndYKnJosJRgVCJ_rh0M`

---

## 0. Benchmark `v2` TL;DR (Current Session State — 2026-11-17 09:08 UTC)

| Workstream | Current Status (2026-11-17 09:08 UTC) |
| :--- | :--- |
| **4-Platform Codebase (`v7x`, `v6e`, `v5p`, `v5e`) & Unit Tests** | ✅ **100% Done, Linted & Tested (`110 passed`, `pylint 10.00/10`).**<br>• Added full TPU `v5p` support across `bench/platforms.py`, `bench/schema_v2.py` (`PLATFORMS = ("v7x", "v6e", "v5p", "v5e")`), `configs/models_v2.json` (**`516` planned `v2` rows** = `164` `v7x` + `140` `v6e` + `164` `v5p` + `48` `v5e`), `bench/manifests_v2.py` (`cpu_kueue_queue="default"` for `bodaborg-v5p-nap`), `scripts/build_sheet_v2.py` (same-region `europe-west4` `step1` / `step1b` lookup), and `benchmark_ckpt_load_time.py` (`9afa3523f`, `d3bb128cd`).<br>• Fixed 2 harness issues discovered during `v5p` Tier C audit (`a2b6d3b4d`, §7.8): (1) wired `_configure_orbax_barrier_timeout()` (`ORBAX_BARRIER_TIMEOUT_S`, default `1800s`) into `run_worker_safetensors_dynamic` as well as `to_maxtext` Step 3; (2) prevented CLI `--skip-to-maxtext` / `--skip-safetensors` `SKIPPED` placeholders from masking subsequent `--skip-completed` runs or overwriting real GCS results.<br>• Rebased & pushed `bench/v5p-with-pr6319` (`8990136d7` = `docs/checkpoint-load-benchmark` + Lance's PR `#6319` `95cc75471`) and uploaded committed bundle + `site_packages_overlay.tar.gz` to **`gs://mesa-maxtext/mesa-bench-v2/code-v5p-r2/`**. |
| **Cold Regional Step 1 (`HF -> GCS`)** | ✅ **`63 / 63` (`100%`) Done & Verified Cold.** `21` models in `us-central1` (`v7x`), `21` in `southamerica-west1` (`v6e`), and **`21 / 21` in `europe-west4`** (`12` measured on `v5e` + **`9` Tier B models measured cold on `v5p`** `n2-standard-64` in `bodaborg-v5p-nap`, all `step1_files_already_in_gcs_count = 0`). |
| **Real-Weight DeepSeek Step 1b (`FP8/FP4 -> BF16`)** | ✅ **`12 / 12` (`100%`) Done & Provenance Audited.** `284B` (`581.9 GB`), `671B` (`1,369.0 GB`), and `3.2-671B` (`1,370.7 GB`) copied from `gs://maxtext-deepseek/` (`copied_from_maxtext_team_bucket`). **`deepseek4-1600b` (`DeepSeek-V4-Pro`, `3,197.7 GB` BF16 / 64 shards)** dequantized by us on CPU via `deepseek_dequantize.py` ([gpaste](https://paste.googleplex.com/6755399437104149), `converted_by_this_benchmark`) in **`10,675.89s`** (`us-central1`) and copied cold to `southamerica-west1` (**`5,803.30s`**) and `europe-west4` (**`5,300.98s`**). |
| **Completed Platforms (`v7x`, `v5p` & `v5e`)** | ✅ **All 3 Platforms 100% Complete, Synced to Sheet & Cleaned Up (`0` active `mesa-*` K8s resources).**<br>• **`v7x` (`us-central1`):** `164/164` `v2` rows (`142 OK`, `22 OOM-HOST`) + **`14 / 14` real-weight GCS DeepSeek `v2real` phases `OK`** (`284B` `855.16–985.29s`, `671B` `1,209.69–1,543.10s`, `3.2-671B` `1,235.74–1,606.99s`, `1600B` `1×2x4x4` `scanned` **`2,493.02s`** / `unscanned` **`2,203.11s`**).<br>• **`v5p` (`europe-west4`, `bodaborg-v5p-nap`):** **`164 / 164` `v2` rows (`136 OK`, `28 OOM-HOST` on single-host `to_maxtext`, `82 / 82` `safetensors_dynamic` `OK`, `0` OOMs, `0` `SKIPPED`, `0` `ERROR`/`TIMEOUT`)**, including `100%` real-weight GCS `BF16` DeepSeek `284B–1600B` loads (`ORBAX_BARRIER_TIMEOUT_S=3600`, PR `#6319`).<br>• **`v5e` (`europe-west4`):** `48/48` `v2` rows + `8/8` baseline `v2real` `OOM-HOST` (`tf.io.gfile` 64 MiB read-ahead) + **`8/8` `v2real-gcsfs` (with PR `#6319`)**: `284B` **`390.59s / 313.77s`**, `671B` **`620.69s / 452.39s`**, `3.2-671B` **`638.18s / 474.81s`**, `1600B` `2×16x16` `unscanned` **`1,245.19s` `OK`** & `scanned` genuine **`OOM-HBM`** (`16.77 GB > 16 GiB/chip`). |
| **Active Run: TPU `v6e` `1×16x16` DeepSeek Re-Run (`bodaborg-v6e-nap`)** | 🔄 **`4 / 8` `v6e` Real-Weight DeepSeek Phases `OK` (`1×8x8` `284B` `2/2 OK` + `1×16x16` `671B` `2/2 OK`, Synced); Last `4` Phases (`3.2-671B`, `1600B`) Actively Running in `mesa-q-v6e-16x16-1s-g1`.** Once `v2realgcsfs/done/` reaches `6/6`, delete `mesa-q-v6e-16x16-1s-g1` and sync the final 4 `v2real|v6e|...` rows. |
| **Buganizer, Customer Updates & Upstream PRs** | ✅ **Tracked, Implemented & Published.**<br>• **Buganizer http://b/586746548 → Draft PR [`#6398`](https://github.com/AI-Hypercomputer/maxtext/pull/6398):** Implemented on-the-fly `FP8` & `MXFP4` → `BF16` dequantization in `safetensors_dynamic` (`safetensors_reader.py`, `dequantize.py`, `hf_streaming_load.py`, `load_dynamic.py`; `223 passed` unit/streaming tests + real GCS `671B`/`284B` bit-exact verification).<br>• **Other Upstream PRs:** Draft PR [`#6353`](https://github.com/AI-Hypercomputer/maxtext/pull/6353) (HBM peak), Draft PR [`#6354`](https://github.com/AI-Hypercomputer/maxtext/pull/6354) (`to_maxtext` Orbax save cap), Lance's PR [`#6319`](https://github.com/AI-Hypercomputer/maxtext/pull/6319) (`gcsfs` reader), and Docs PR [`#6350`](https://github.com/AI-Hypercomputer/maxtext/pull/6350) (`cl/1022390706`, `safetensors_dynamic.md` — `TPU v5p Benchmarks` Markdown section generated & ready!). |
| **Google Sheet `v2` (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`)** | ✅ **`615` `Summary` Rows Live (`516/516` Planned `v2|` Rows = `100%` Done, `0` Formula Errors, `17/17` Lance `wl*` Rows Preserved, `Runs` Expanded to `3,000` Grid Rows & Validated Byte-for-Byte).** |

---

## 1. New Session Pickup Prompt, Immediate Action Plan & Guardrails

### 1.1 Copy-Paste Prompt for Opening a New Jetski Session
> Read `checkpoint_load_benchmark/handoff.md` on branch `docs/checkpoint-load-benchmark` in `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3`. Follow the exact step-by-step action plan in §1.3 and guardrails in §1.2:
> 1. Check live cluster status on `bodaborg-v5p-nap` (last 13 `2x2x1` Tier S/A tasks finishing) and `bodaborg-v6e-nap` (`mesa-q-v6e-16x16-1s-g1`), and delete finished `mesa-*` JobSets.
> 2. Using the already-uploaded fixed bundle `gs://mesa-maxtext/mesa-bench-v2/code-v5p-r2` (`bench/v5p-with-pr6319` @ `8990136d7`), launch the targeted `v5p` fix queue (`ORBAX_BARRIER_TIMEOUT_S=3600`, `SKIP_FAILED_RESULTS=0`) for the 6 DeepSeek `safetensors_dynamic` barrier-timeout phases, the `qwen3-235b-a22b` `2×2x2x1` `safetensors_dynamic` overwrite fix, and the `<2s` `to_maxtext` Host RAM `OOM-HOST` prechecks on `235B` + Tier C.
> 3. Once `v5p` reaches `164/164` clean rows (and `v6e` `16x16` if completed), mirror `logs/` only (never `to_maxtext/` weights!), upload missing `gpaste` logs, sync Google Sheet `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I` (preserving 100% of Lance Wang's `wl*` rows and expanding `Runs` grid >2,000 rows first), and prepare the `TPU v5p Benchmarks` Markdown section for `safetensors_dynamic.md` (PR `#6350` / cl/1022390706).

### 1.2 Mandatory Rules & Guardrails
1. **Never alter or delete Lance Wang's `wl*` rows or cluster jobs:**
   - On Google Sheet `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`, preserve 100% of Lance's `wl*` rows (`11` on `Summary`, `5` on `Step 1 downloads`, `11–12` on `Stages: safetensors_dynamic`, `3` on `Stages: to_maxtext`, `129–140` on `Runs`). On tabs `400`, `500`, `600`, `700`, **append** new `v5p` rows below existing rows rather than clearing `A2:end`!
   - Never touch Lance's `wanglance-*` / `*-wl*` JobSets on `v5p` or `v5e`, and **never touch backup sheet `1xZ0ifZW5yD0gPT8CpcsxQ1Q8ndYKnJosJRgVCJ_rh0M`**.
2. **Never modify `src/maxtext/`** on `docs/checkpoint-load-benchmark` unless explicitly requested, and **never stage unrelated untracked root files** (`benchmark_v2_design_plan.md`, `docs/plans/`, `run_e2e_matrix_xpk.py`, `run_e2e_matrix_xpk.sh`, `vllm_decoder_issue.md`).
3. **Never mirror `runs/*/*/to_maxtext/` Orbax weight directories to local disk (`68+ GiB`):** Only mirror `step1/`, `step1b/`, and `runs/*/run_v2-v5p-*/logs/` (`~200 MB` total; see exact `xargs` command in §9).
4. **Pods run only committed, pushed code bundles:** Always commit & push (`bench/v5p-with-pr6319` in worktree `/usr/local/google/home/mesakh/.gemini/jetski/worktrees/6ae29b9d/bench-gcsfs` when PR `#6319` is included) and build with `scripts/build_code_tarball.py` + copy `site_packages_overlay.tar.gz` into the `code-*` prefix.
5. **Formatting, lint & CLI conventions:**
   - MaxText Python formatting uses **2-space indentation**: `pyink --pyink --pyink-indentation=2 --line-length=120` and `pylint --rcfile=pylintrc` (`/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/`).
   - `gsheets` CLI (`/google/bin/releases/gemini-agents-gsheets/gsheets`) now uses nested subcommands: `gsheets readonly {info,list-sheets,read,...}` and `gsheets mutate {batch,raw-batch,...}`.
   - Buganizer CLI (`/google/bin/releases/issues-cli/issues`): always ask explicit user confirmation before any `issues mutate ...` command.

### 1.3 Exact Step-by-Step Action Plan for the Next Session
1. **Drain & Clean Up Remaining Wave 1/3 `v5p` `2x2x1` Workers (`bodaborg-v5p-nap`):**
   - Run `/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch/v2real/monitor.sh` to verify `2x2x1_1s_sa` (`19/19`) and `2x2x1_2s_sa` (`31/31`) reach `124/124`, and delete all finished `mesa-q-v5p-*` JobSets.
   - Mirror `runs/S` and `runs/A` `logs/` (excluding `to_maxtext/`!) into `/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch/v5p_mirror/` and audit every `v5p` row with `build_sheet_v2.build_v2_dataset`.
2. **Launch Targeted `v5p` Fix Queue Using Already-Uploaded `gs://mesa-maxtext/mesa-bench-v2/code-v5p-r2` (`8990136d7`):**
   - **Why (§7.8):**
     1. **6 DeepSeek `safetensors_dynamic` phases** hit Orbax v1's hardcoded `300s` `[op=1] _load_impl` barrier timeout when per-host GCS read skew across 4–16 hosts exceeded 5 minutes (`Peak RAM` was only `130–175 GB / 473 GB`):
        - `deepseek4-284b` `2×2x2x2` `unscanned` (`run_v2-v5p-deepseek4-284b-2x2x2x2-r1`)
        - `deepseek3-671b` `2×2x2x4` `unscanned` (`run_v2-v5p-deepseek3-671b-2x2x2x4-r1`)
        - `deepseek3.2-671b` `1×2x4x4` `scanned` & `unscanned` (`run_v2-v5p-deepseek3.2-671b-1x2x4x4-r1`)
        - `deepseek4-1600b` `1×4x4x4` `scanned` & `unscanned` (`run_v2-v5p-deepseek4-1600b-1x4x4x4-r1`)
     2. **`qwen3-235b-a22b` `2×2x2x1` (`r1..r3`):** Check if `safetensors_dynamic_result.json` is still `SKIPPED (--skip-safetensors)` after the running `2ssaw*` workers finish; re-run any missing `235B` `2×2x2x1` `safetensors_dynamic` reps.
     3. **`to_maxtext` `OOM-HOST` prechecks (`<2s` each on 1 host):** Re-run `--skip-safetensors` on `qwen3-235b-a22b` (`1×2x2x2` `r1..r3`, `2×2x2x1` `r1..r3`) and the 6 Tier C models (`11` `run_id`s) with `code-v5p-r2` so `_is_completed_result_to_skip` replaces `SKIPPED (--skip-to-maxtext)` with canonical `FAILED / OOM-HOST` (`total_gb + 40 > usable_ram_gb`, matching `v7x`/`v6e`/`v5e`!). Because `to_maxtext` Step 1/2 runs on 1 host before Step 3, all 17 `--skip-safetensors` precheck tasks can run in **<2 minutes total on a single `1×2x2x1` worker** (passing each target `--slice-topology` & `--devices-per-slice` / `--num-hosts-per-slice`)!
   - Set env `ORBAX_BARRIER_TIMEOUT_S=3600` and `SKIP_FAILED_RESULTS=0` on the 6 DeepSeek `safetensors_dynamic` retry tasks, run them on 1 worker each for `2×2x2x2`, `2×2x2x4`, `1×2x4x4`, `1×4x4x4`, and delete each JobSet upon completion.
3. **Check `v6e` `1×16x16` DeepSeek Real-Weight Queue (`bodaborg-v6e-nap`):**
   - Monitor `mesa-q-v6e-16x16-1s-g1` (`gs://mesa-maxtext-southamerica-west1/mesa-bench-v2/queue/v2realgcsfs/`, `6` phases: `671B`, `3.2-671B`, `1600B` `scanned`/`unscanned`). Once `done=6/6`, delete `mesa-q-v6e-16x16-1s-g1` and mirror `runs/C/run_v2realgcsfs*/logs/` to `scratch/v2real_mirror/v6e_gcsfs/`.
4. **Sync Google Sheet `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I` & Validate (`0` Errors):**
   - Adapt `/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch/sync_v2real_sheet.py` into `sync_v5p_sheet.py`:
     - **Expand `Runs` (`sheetId=700`) grid rows first** via `appendDimension` (`+1,000` rows -> `3,000` total rows; currently `92x2000` with `1,532` used rows, and `v5p` adds `~550` phase rows!).
     - Merge `164` `v2|v5p|...` rows into `Summary` (`396` `11-14` base + `164` `v5p` + `11` Lance `wl*` + `30` `v2real|` + `8` [+`6` `v6e`] `v2real-gcsfs|` = `609` [or `615`] rows).
     - Append `v5p` rows below existing rows on `Step 1 downloads` (start after row `60`), `Stages: safetensors_dynamic` (start after row `245`), `Stages: to_maxtext` (start after row `192`), and `Runs` (start after row `1532`) so **zero existing / `wl*` rows are touched**.
     - Add `v5p` row to `Platforms` (`sheetId=900`), `v5p topologies` column/notes to `Models` (`sheetId=800`) if desired, and `v5p` (`164/164`) + `T7` implementation rows to `Progress` (`sheetId=200`) & `README` (`sheetId=100`).
     - Upload all missing `v5p` logs to `gpaste` (`configs/gpaste_log_urls.json`) and run full forensic validation (`0` `#REF!`/`#VALUE!`/`#DIV/0!`, `0` duplicate Row IDs, `100%` `wl*` byte-for-byte parity).
5. **Prepare `TPU v5p Benchmarks` Section for `safetensors_dynamic.md` & Update `reports/benchmark_v2_summary.md`:**
   - Generate the ready-to-paste `TPU v5p Benchmarks` Markdown table/summary for Docs PR `#6350` (cl/1022390706) and refresh [`reports/benchmark_v2_summary.md`](reports/benchmark_v2_summary.md) + [`PROGRESS_LOG.md`](PROGRESS_LOG.md) across all 4 TPU generations (`516` planned `v2` rows).

---

## 2. Benchmark `v2` Architecture, Codebase & Test Suite (`110` Unit Tests)

Built on top of `origin/main` (`5b6773c9b` = `maxtext-v0.2.6`, which includes Lance Wang's merged `safetensors_dynamic` PRs `#5706`, `#5828` / cl/1017682327, `#5705`, `#5610`, `#6113`, `#6116` and is **prior** to the 3 `HF streaming` regression CLs cl/1020481604, cl/1020513918, cl/1023046858) plus branch fixes (`207cabecc` -> Draft PR `#6353`; `96329637f` -> Draft PR `#6354`; and Lance's PR `#6319` `95cc75471` in `bench/v5p-with-pr6319`):

| File | Responsibility |
| :--- | :--- |
| [`bench/schema_v2.py`](bench/schema_v2.py) | Canonical column specs (`SUMMARY_COLUMNS` [37, incl. plain-English verdicts `faster_from_hf_verdict` & `faster_already_orbax_verdict`], `STEP1_COLUMNS` [24], `STAGES_SD_COLUMNS` [55], `STAGES_TM_COLUMNS` [70], `RUNS_COLUMNS` [92]), `PLATFORMS = ("v7x", "v6e", "v5p", "v5e")`, validators, and `load_models_v2()`. |
| [`bench/stages.py`](bench/stages.py) | Per-host `StageRecorder`, `jax.monitoring` listeners, distributed barrier timer, background Host RAM/RSS & TPU HBM (`memory_stats()`) sampler across `v7x`/`v6e`/`v5p`/`v5e`, and `BENCH_STAGES_JSON` aggregator across up to 128 hosts. |
| [`bench/harness_v2.py`](bench/harness_v2.py) | v2 phase result JSON builder (recording true `checkpoint_weights_kind` and `checkpoint_load_source_uri`), pre-load `check_tpu_backend` enforcement, and Tier A parameter checksum verification. |
| [`bench/platforms.py`](bench/platforms.py) | Verified `PlatformProfile` registry for **`v7x`, `v6e`, `v5p`, and `v5e`** (contexts, regional buckets, Kueue/TAS & `cpu_kueue_queue`, reservation affinities, chip/device ratios, Host RAM margins, Step 1 64-vCPU VM types). |
| [`bench/manifests_v2.py`](bench/manifests_v2.py) | Profile-driven K8s Indexed `JobSet` builder (1–128 pods), Step 1/1b CPU `Job` + `PVC` builder (injecting `kueue.x-k8s.io/queue-name` when `cpu_kueue_queue` is set), and `S -> A -> C -> B` queue task generator. |
| [`bench/step1_download_job.py`](bench/step1_download_job.py) | Standalone CPU CLI for cold `HF -> GCS` downloads (`download`), bounded-RAM resumable DeepSeek `FP8/FP4 -> BF16` CPU dequantization (`deepseek-convert`), and resumable cold cross-region GCS copies (`copy-region`). |
| [`benchmark_ckpt_load_time.py`](benchmark_ckpt_load_time.py) & [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py) | Unified multi-host/multi-slice TPU harness (`_configure_orbax_barrier_timeout` on both `to_maxtext` S3 & `safetensors_dynamic`; `_is_completed_result_to_skip` & GCS overwrite guard on `--skip-*`; `a2b6d3b4d`) & committed-tarball pod bootstrap. |
| [`scripts/build_sheet_v2.py`](scripts/build_sheet_v2.py) | Builds all 5 data CSVs (`_PLAT_SORT_ORDER = {"v7x": 0, "v6e": 1, "v5p": 2, "v5e": 3}`, same-region `europe-west4` `step1`/`step1b` lookup) + plain-English verdicts + `merge_external_summary_rows` + `gpaste` uploader (`configs/gpaste_log_urls.json`). |
| [`configs/models_v2.json`](configs/models_v2.json) | 21-model matrix (`0.6B` to `1600B`), HF repos, dtypes, BF16 sizes, Step 1b flags, and per-platform slice/topology lists (**`516` planned `Summary` rows**: `164` `v7x`, `140` `v6e`, `164` `v5p`, `48` `v5e`). |
| [`tests/*_test.py`](tests/) | Unit test suite (**`110` tests passing** via `pytest`, `pylint` `10.00/10`). |

---

## 3. Multi-Cluster Infrastructure & Regional Storage (`bench/platforms.py`)

| Platform | GKE Cluster & Kubectl Context | Region & Regional Bucket | Queue / TAS & Pod Scheduling | Hardware Specs & Topologies |
| :--- | :--- | :--- | :--- | :--- |
| **`v7x`** | `bodaborg-tpu7x-nap`<br>`gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` | `us-central1`<br>`gs://mesa-maxtext-us-central1/mesa-bench-v2` | Kueue `default`, TAS `podset-slice-size`, Spot (`tpu7x-spot` / `tpu7x-flavor`), `reservation-name DoesNotExist`, `gitimg` pull secret, ~3h Kueue cap | `tpu7x-standard-4t`: `4` chips = `8` JAX cores/host (`95 GiB`/core HBM, `960 GiB` Host RAM). Topologies: `2x2x1` (1h), `2x2x2` (2h), `2x2x4` (4h), `2x4x4` (8h). Step 1/1b CPU: `n2d-standard-64` (`cpu-np`). |
| **`v6e`** | `bodaborg-v6e-nap`<br>`gke_cloud-tpu-shared-capacity_southamerica-west1_bodaborg-v6e-nap` | `southamerica-west1`<br>`gs://mesa-maxtext-southamerica-west1/mesa-bench-v2` | Kueue `default`, TAS `False`, `priorityClassName: medium`, NO spot selectors, ~3h Kueue cap | `ct6e-standard-8t` (`2x4`: `8` chips/host, `1,440 GiB` RAM) & `ct6e-standard-4t` (`4` chips/host, `720 GiB` RAM), `32 GiB`/chip HBM. Topologies: `2x4`, `2x2`, `4x4` (4h), `8x8` (16h), `16x16` (64h / 256 chips). Step 1 CPU: `n4-standard-64`. |
| **`v5p`** | `bodaborg-v5p-nap`<br>`gke_cloud-tpu-shared-capacity_europe-west4_bodaborg-v5p-nap` | `europe-west4`<br>`gs://mesa-maxtext-europe-west4/mesa-bench-v2` *(shared regional cache with `v5e`)* | Kueue `default` (**also required on standalone CPU `batch/v1` Jobs** because `manageJobsWithoutQueueName: true`!), TAS `False`, `priorityClassName: medium`, reservation `cloudtpu-20260902214500-1810493672`, 180m Kueue cap | `ct5p-hightpu-4t`: `4` chips = `4` JAX devices/host (`95 GiB`/chip HBM, `448 GiB` / `473.3 GB` measured Host RAM). Topologies: `2x2x1` (1h / 4 devs), `2x2x2` (2h / 8 devs), `2x2x4` (4h / 16 devs), `2x4x4` (8h / 32 devs), `4x4x4` (16h / 64 devs). Step 1 CPU: `n2-standard-64` (`cpu-np`). |
| **`v5e`** | `v5e-256-bodaborg-europe-west4`<br>`gke_tpu-prod-env-multipod_europe-west4_v5e-256-bodaborg-europe-west4` | `europe-west4`<br>`gs://mesa-maxtext-europe-west4/mesa-bench-v2` | **No Kueue** (omit `queue-name`, TAS `gke-v5e-tas-2` broken), `serviceAccountName: mesa-bench`, no 3h cap | `ct5lp-hightpu-4t`: `4` chips = `4` devices/host (`16 GiB`/chip HBM, `192 GiB` Host RAM). Topologies: `1×16x16` (64 hosts / 256 chips) & `2×16x16` (128 hosts / 512 chips). Step 1 CPU: `n2-standard-64`. |

---

## 4. Execution Outcomes, DeepSeek Real-Weight Provenance & `v5p` Highlights

Full `v7x`/`v6e`/`v5e` tables are in [`reports/benchmark_v2_summary.md`](reports/benchmark_v2_summary.md) and on live Sheet [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit):

1. **Two Explicit Scenarios Across Paired `OK` Configurations (`Summary Cols AC & AD`):**
   - **Scenario 1 — Starting from HuggingFace Weights (`no Orbax checkpoint yet`, `Summary Col AC`):** `safetensors_dynamic` is faster in `118 / 121` (`v7x`/`v6e`/`v5e`) paired `OK` configs — median **`4.99x`** (`0.53x – 26.76x`) faster without HF download and median **`1.83x`** faster end-to-end — *because it loads HF safetensors directly into TPU HBM and skips single-host CPU Orbax conversion (`to_maxtext` Step 2)*.
   - **Scenario 2 — Checkpoint Already in Orbax Format (`load-only`: `TM Step 3 ÷ SD Step 2`, `Summary Col AD`):** Pre-converted Orbax Step 3 wins `97%` of `unscanned` pairs (median `0.42x`, ~2.4× faster) and `<30B` `scanned` pairs, whereas **`safetensors_dynamic` is `2x–10.4x` faster on larger `scanned` models (`30B–120B`)** due to Orbax Zarr3/OCDBT stacked-chunk concurrency throttling.
2. **Real-Weight DeepSeek Provenance & Cross-Platform Results (`b/586746548`, `[Weights correction 2026-11-16]`):**
   - **Why Offline Dequantization Was Required (Tracked in http://b/586746548):** Official Hugging Face repos for `DeepSeek-V4-Flash` (`284B`), `DeepSeek-V3` (`671B`), `DeepSeek-V3.2` (`3.2-671B`), and `DeepSeek-V4-Pro` (`1600B`) publish **only quantized `FP8` (`_scale_inv`) / packed `MXFP4` (`.scale`)** weights, while MaxText expects `BF16` and neither `safetensors_dynamic` nor `to_maxtext` dequantizes `FP8`/`MXFP4` on the fly yet.
   - **Exact Provenance of Regional `hf_cache_bf16/` Checkpoints:**
     - `deepseek4-284b` (`581.9 GB`), `deepseek3-671b` (`1,369.0 GB`), `deepseek3.2-671b` (`1,370.7 GB`): copied from MaxText team bucket `gs://maxtext-deepseek/` (`copied_from_maxtext_team_bucket`).
     - `deepseek4-1600b` (`3,197.7 GB`): downloaded raw `FP8+FP4` (`864.7 GB`) from HF and dequantized offline to `BF16` by our benchmark using `deepseek_dequantize.py` ([gpaste](https://paste.googleplex.com/6755399437104149), `converted_by_this_benchmark`).
   - **Real-Weight GCS `safetensors_dynamic` Outcomes:**
     - **`v7x` (`v2real`, `14 / 14 OK`):** `284B` **`855.16–985.29s`**, `671B` **`1,209.69–1,543.10s`**, `3.2-671B` **`1,235.74–1,606.99s`**, `1600B` (`1×2x4x4`) **`2,493.02s`** (`scanned`) / **`2,203.11s`** (`unscanned`).
     - **`v5e` (`v2real` vs `v2real-gcsfs` with PR `#6319`):** Baseline `tf.io.gfile` hit `8/8` `OOM-HOST` (`64 MiB` read-ahead across `33.8k–71.4k` unfused expert keys on `192 GiB` hosts). With PR `#6319` (`gcsfs`), Host RAM dropped to **`68.5–102.3 GB`**: `284B` **`390.59s / 313.77s`**, `671B` **`620.69s / 452.39s`**, `3.2-671B` **`638.18s / 474.81s`**, `1600B` `2×16x16` `unscanned` **`1,245.19s` `OK`** (`scanned` genuine `OOM-HBM` at `16.77 GB > 16 GiB/chip`).
     - **`v6e`:** `1×8x8` `284B` **`1,227.60s / 1,148.10s` `OK`**; `1×16x16` (`6` phases with PR `#6319`) queued in `mesa-q-v6e-16x16-1s-g1`.
     - **`v5p` (with PR `#6319`, `448 GiB` Host RAM, `95 GiB/chip` HBM):** All **`72 / 72` Tier B rows `OK`**; `16 / 22` Tier C `safetensors_dynamic` rows already `OK` (`0` OOMs, Peak Host RAM `35.6–175.0 GB / 473.3 GB`, Peak HBM `28.8–81.6 GB / 95 GiB`); remaining 6 DeepSeek phases re-running with `ORBAX_BARRIER_TIMEOUT_S=3600` (`code-v5p-r2`).

---

## 5. Cold-Cache Guarantees & Google Sheet `v2` Design

1. **Internal `Experiment Tier` (`S`, `A`, `B`, `C`) Hidden (`d524aab2b`):** Kept in `Column C` for `Progress` formulas, with `hiddenByUser: true` enforced on `Summary` (`sheetId=300`) and `Runs` (`sheetId=700`).
2. **Strict Cold / Empty Cache Guarantees Across All Phases:**
   - **Step 1 (`HF -> GCS`) & Step 1b (`copy-region`):** `ensure_empty_gcs_prefix` enforces `step1_files_already_in_gcs_count = 0` across all **`63 / 63`** regional downloads and `12 / 12` Step 1b runs.
   - **JAX XLA Compilation Cache & `/ramdisk` Purge:** Every phase runs in a fresh subprocess with `jax_compilation_cache_state = "empty"`, and `tpu7x_task_queue.py` purges `/ramdisk/bench_*`, `/ramdisk/hf_local`, and `/ramdisk/tmp` across all hosts between tasks.
3. **11-Tab Workbook Layout (`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`):**
   - Sheet IDs: `100` `README`, `200` `Progress`, `300` `Summary` (`445` rows live -> `609` with `v5p`), `400` `Step 1 downloads` (`60` rows live -> `69` with `v5p` Tier B), `500` `Stages: safetensors_dynamic` (`245` rows live), `600` `Stages: to_maxtext` (`192` rows live), `700` `Runs` (`1,532` rows live — **expand `2000 -> 3000` grid rows via `appendDimension` before appending `v5p`!**), `800` `Models`, `900` `Platforms`, `1000` `Legend`, `1100` `Columns`.

---

## 6. Live Google Sheet `v2` Forensic Validation Checklist

Before & after syncing `v5p` (and `v6e` `16x16`) to `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`:
- **Preserve 100% of Lance Wang's `wl*` rows:** Verify all `11` `Summary` `wl*` rows against `/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch/lance_summary_formulas.tsv` (exact cell-for-cell parity) and append-only on tabs `400`, `500`, `600`, `700` so existing `wl*` rows (`5`, `11`, `3`, `140`) are never overwritten.
- **Zero Formula / Token Errors:** Scan all 11 tabs for `#REF!`, `#DIV/0!`, `#VALUE!`, `#NAME?`, `#N/A`, `NaN`, `None`, and verify `0` duplicate `row_id`s.
- **Clickable `gpaste` Links:** Upload all new `v5p` / `v6e` logs via `build_sheet_v2.populate_gpaste_urls` (`configs/gpaste_log_urls.json`).

---

## 7. Engineering Issues Discovered & Resolved During Benchmark `v2`

1. **64-Host `v5e` TPU `SliceBuilder` ALTS gRPC Hang (`c64b44244`):** Fixed remote daemon env forwarding and unbuffered log streaming so `jax.distributed.initialize` across 64–128 hosts completes reliably.
2. **`gpt-oss-20b` / `gpt-oss-120b` on 256-Chip `16x16` Torus (`edad98c96`, `1973efc79`):** Automated `ici_fsdp_parallelism` cap + `allow_split_physical_axes=True` when `total_devices > num_experts`.
3. **Multi-Host Worker Teardown Race (`2884fc50c`):** Added `worker_teardown_results_written` barrier before Indexed Job completion terminates companion pods.
4. **DeepSeek `1600B` (`3.2 TB` BF16) Bounded-RAM Resumable CPU Conversion (`3dffe1209`, `37e0d3cd6`):** Dequantized 64 shards on `n2d-standard-64` + 5 TB PVC with `POSIX_FADV_DONTNEED` and async multipart GCS upload (`10,675.89s`).
5. **Step 1b `copy_bf16_gcs_to_region` `--clean-target` Fix (`a230891c8`):** Enforced cold target cleanup before cross-region `BF16` copies (`5,300.98s` `europe-west4`, `5,803.30s` `southamerica-west1`).
6. **`bodaborg-v5p-nap` Kueue Admission for Standalone CPU Jobs (`9afa3523f`):** Unlike `v7x`/`v6e`/`v5e`, `bodaborg-v5p-nap` sets Kueue `manageJobsWithoutQueueName: true`; added `PlatformProfile.cpu_kueue_queue = "default"` on `v5p` so Step 1 CPU `batch/v1` Jobs admit cleanly onto `cpu-np`.
7. **Unfused MoE `tf.io.gfile` 64 MiB Read-Ahead Amplification (`PR #6319`):** Confirmed on `v5e` (`192 GiB` hosts) and prevented on `v5p`/`v6e` by rebasing `bench/v5p-with-pr6319` (`8990136d7`) onto Lance's `gcsfs` reader (`95cc75471`).
8. **Orbax v1 `300s` `[op=1] _load_impl` Barrier Timeout in `safetensors_dynamic` & `--skip-*` Placeholder Masking (`a2b6d3b4d`):**
   - **Root cause A:** `run_worker_to_maxtext_step3_load_orbax` raised `orbax.checkpoint.experimental.v1._src.synchronization.multihost._DEFAULT_BARRIER_TIMEOUT` from `300s` to `1800s`, but `run_worker_safetensors_dynamic` did not. On `v5p` multi-host DeepSeek reads (up to 16 hosts on `1×4x4x4` `1600B`), GCS read skew > `300s` triggered `TimeoutError: Timeout waiting at barrier '[op=1] _load_impl'` on 6 phases despite `130–175 GB / 473 GB` Host RAM. Fixed via `_configure_orbax_barrier_timeout()` in both workers (`ORBAX_BARRIER_TIMEOUT_S`).
   - **Root cause B:** Splitting `235B` and Tier C tasks into `--skip-to-maxtext` (`sd`) and `--skip-safetensors` (`tm`) under one `run_id` caused `--skip-completed` to treat `--skip-to-maxtext` `SKIPPED` placeholders as completed (skipping the `<2s` Host RAM `OOM-HOST` precheck) and allowed concurrent `--skip-safetensors` runs to overwrite `safetensors_dynamic_result.json` on `235B` `2×2x2x1`. Fixed via `_is_completed_result_to_skip()` + GCS check in `_write_skipped_v2_result()`, covered by 2 new unit tests in `tests/harness_v2_test.py` (`110 passed`).

---

## 8. Patches vs. Upstream `main`, Upstream PRs & Buganizer Tracker

1. **Draft PR [`#6353`](https://github.com/AI-Hypercomputer/maxtext/pull/6353)** (`fix/safetensors-dynamic-composite-hbm-peak` @ `d6b6a0bf2`, from `207cabecc`): Single-leaf composite stacking (`num_parts`) + pairwise reshard concat (`_concatenate_pieces`), fixing `397B` & `gpt-oss-120b` `scanned` HBM OOMs (`97 passed`, `pylint 9.99/10`).
2. **Draft PR [`#6354`](https://github.com/AI-Hypercomputer/maxtext/pull/6354)** (`fix/to-maxtext-orbax-save-concurrent-gb` @ `f917034f3`, from `96329637f`): Wires `config.checkpoint_storage_concurrent_gb` into `to_maxtext` Orbax save, preventing `397B` `unscanned` Host RAM OOM during Step 2 save (`19 passed`, `pylint 9.98/10`).
3. **PR [`#6319`](https://github.com/AI-Hypercomputer/maxtext/pull/6319)** (Lance Wang, `95cc75471`): Reads `safetensors_dynamic` `gs://` checkpoints via `gcsfs` instead of `tf.io.gfile` (`64 MiB` read-ahead), included in `bench/v2real-with-pr6319` (`a8326596c`) and `bench/v5p-with-pr6319` (`8990136d7`).
4. **Docs PR [`#6350`](https://github.com/AI-Hypercomputer/maxtext/pull/6350)** (cl/1022390706, `safetensors_dynamic.md`): Ready to receive the `TPU v5p Benchmarks` section once the 6 `v5p` DeepSeek re-runs finish.
5. **Buganizer http://b/586746548 (`Multipod`, component `1174383`):** Feature request to support in-memory, on-the-fly `FP8` / `MXFP4` → `BF16` dequantization inside `safetensors_dynamic` (`2.0x–3.7x` less GCS I/O, eliminating `582 GB–3.2 TB` offline `BF16` copies; full copy in [`reports/b586746548_deepseek_dequant_issue.md`](reports/b586746548_deepseek_dequant_issue.md)).

---

## 9. Command Cheat Sheet (`v2` 4-Cluster Ops, Logs-Only Mirror & Sheet Sync)

```bash
cd /usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3
PY=/usr/local/google/home/mesakh/Documents/mesakh/py312/bin
SCRATCH_6AE=/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch

# 1. Fast live status across all 4 GKE clusters (v7x, v6e, v5p, v5e)
bash "$SCRATCH_6AE/v2real/monitor.sh"

# 2. Mirror ONLY v5p logs/ & JSONs (NEVER copy runs/*/*/to_maxtext/ Orbax weights!)
MIRROR="$SCRATCH_6AE/v5p_mirror"
gcloud storage rsync -r gs://mesa-maxtext-europe-west4/mesa-bench-v2/step1 "$MIRROR/step1"
gcloud storage rsync -r gs://mesa-maxtext-europe-west4/mesa-bench-v2/step1b "$MIRROR/step1b"
for tier in S A B C; do
  gcloud storage ls "gs://mesa-maxtext-europe-west4/mesa-bench-v2/runs/${tier}/" \
    | grep '/run_v2-v5p-[^/]*/$' \
    | xargs -P 8 -I {} bash -c '
        u="$1"; rname=$(basename "${u%/}")
        dest="'"$MIRROR"'/runs/'"$tier"'/${rname}/logs"
        mkdir -p "$dest" && gcloud storage rsync -q -r "${u}logs" "$dest"
      ' _ {}
done

# 3. Format, lint (10.00/10), and run all 110 unit tests
$PY/pyink --pyink --pyink-indentation=2 --line-length=120 --check checkpoint_load_benchmark/bench/*.py checkpoint_load_benchmark/scripts/build_sheet_v2.py checkpoint_load_benchmark/benchmark_ckpt_load_time.py
PYTHONPATH=checkpoint_load_benchmark $PY/pylint --rcfile=pylintrc checkpoint_load_benchmark/bench/*.py checkpoint_load_benchmark/scripts/build_sheet_v2.py checkpoint_load_benchmark/tests/harness_v2_test.py
$PY/pytest -q checkpoint_load_benchmark/tests/

# 4. Inspect Google Sheet v2 (note new `readonly` / `mutate` subcommands)
/google/bin/releases/gemini-agents-gsheets/gsheets readonly info 1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I
```

---

## 10. Where Everything Lives (`v2` Locations & Local Caches)

| Resource | Exact Location |
| :--- | :--- |
| **Git Repository & Branch** | `/usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3` (`docs/checkpoint-load-benchmark` @ `a2b6d3b4d`) |
| **`v5p` + PR `#6319` Worktree & Branch** | `/usr/local/google/home/mesakh/.gemini/jetski/worktrees/6ae29b9d/bench-gcsfs` (`bench/v5p-with-pr6319` @ `8990136d7`) |
| **GCS Committed Code Bundles** | `gs://mesa-maxtext/mesa-bench-v2/code-v5p-r2/` (`8990136d7`, **use for `v5p` fix pass**), `code-v5p/` (`fea07bef5`), `code-gcsfs/` (`a8326596c`), `code/` (`f2bdfe636`) |
| **GCS Regional Data & Result Buckets** | `gs://mesa-maxtext-{us-central1,southamerica-west1,europe-west4}/mesa-bench-v2/{hf_cache,hf_cache_bf16,step1,step1b,runs,queue}/` |
| **Pre-Mirrored Local `v5p` & `v2real` Artifacts** | `/usr/local/google/home/mesakh/.gemini/jetski/brain/6ae29b9d-fa48-468c-b718-3190159e2336/scratch/{v5p_mirror,v5p,v2real,v2real_mirror,sheet_pre_v5p,sync_v2real_sheet.py,lance_summary_formulas.tsv}` |
| **Canonical `11-14` Sheet Snapshot (`396` Rows)** | `/usr/local/google/home/mesakh/.gemini/jetski/brain/f1a7974d-a23a-4dd5-b86c-79af1a906d19/scratch/sheet_v2_sync/` |
| **Persistent `gpaste` URL Cache** | [`checkpoint_load_benchmark/configs/gpaste_log_urls.json`](configs/gpaste_log_urls.json) (`deepseek_dequantize.py`: `https://paste.googleplex.com/6755399437104149`) |
| **Live Google Sheet `v2` (`11` Tabs, `37`-Col `Summary`)** | [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit) |
| **Pre-Edit Full 11-Tab Backup Sheet (Never Touch)** | `1xZ0ifZW5yD0gPT8CpcsxQ1Q8ndYKnJosJRgVCJ_rh0M` |
| **Customer & Buganizer Reports (`b/586746548`)** | [`reports/deepseek_chat_message_short.md`](reports/deepseek_chat_message_short.md), [`reports/deepseek_bug_followup_message.md`](reports/deepseek_bug_followup_message.md), [`reports/b586746548_deepseek_dequant_issue.md`](reports/b586746548_deepseek_dequant_issue.md), [`reports/benchmark_v2_summary.md`](reports/benchmark_v2_summary.md) |
| **Upstream PR Worktrees (`#6353`, `#6354`)** | `/usr/local/google/home/mesakh/.gemini/jetski/worktrees/6ae29b9d/{pr-hbm,pr-orbax}` |
| **Python 3.12 Toolchain (`pyink`, `pylint`, `pytest`)** | `/usr/local/google/home/mesakh/Documents/mesakh/py312/bin/` |

---

# Legacy Archive: Checkpoint Load-Time Benchmark on TPU v7 (`bodaborg-tpu7x-nap`)

> **Archive Snapshot:** 2026-11-12 ~01:50 UTC · **Legacy Google Sheet:** [`11S2eN8mlpuQDQyMyBoBiBsF9KCRd5ZvNHcDf2T8Bz0Y`](https://docs.google.com/spreadsheets/d/11S2eN8mlpuQDQyMyBoBiBsF9KCRd5ZvNHcDf2T8Bz0Y/edit?usp=sharing) (tabs `tpu-v7x(Large models > 200B)` [`gid=635831153`] and `tpu-v7x (Smaller models)` [`gid=1294150880`], 11 columns including `log`). Legacy P1/P2/P3 clusters below are cleaned up. To refresh legacy status tables, run [`scripts/summarize_tpu7x_results.py`](scripts/summarize_tpu7x_results.py) (§9).

---

## 0. TL;DR

| Workstream | Status |
| :--- | :--- |
| **P1 · Exp 1A WITH Lance's 3 PRs** (`safetensors_dynamic`) | ✅ Done. 235B 4/4, 284B 4/4, 480B 2/2 that fit + `2 × 2x2x2` (757.7 s / 718.7 s), 397B unscanned · 2 (1,311.9 s) and scanned · 4 (1,399.7 s), 671B on 4 slices (649.4 s / 531.0 s), `deepseek3.2-671b` on `1 × 2x2x4` (**776.4 s / 654.2 s**), and `deepseek4-1600b` on `1 × 2x4x4` (**971.5 s / 874.3 s**). With commit `207cabecc` (`safetensor (with fix)`), 397B scanned also completes on `2 × 2x2x1` (**1,608.3 s**) and `1 × 2x2x2` (**1,475.2 s**) |
| **P1 · Exp 1A BEFORE** (`origin/main`) | ✅ Done. Every BEFORE run **timed out at 1800 s while still loading** (235B · 1 and · 2, 284B · 1 and · 2, 397B unscanned · 2, 397B scanned · 4, 671B · 4), except 480B, which hits `KeyError`. With the PRs, the same loads take 353–1,400 s |
| **P1 · Exp 1B** (`to_maxtext` vs `safetensors_dynamic`) | ✅ Done. `to_maxtext` S2 can't run on one tpu7x host for scanned 235B/397B/480B, unscanned 480B, and 284B (§7.16). **Unscanned 235B**: S2 2,625.5 s, S3 545.5 s (`1 × 2x2x1`), 659.1 s (`2 × 2x2x1`), 841.1 s (`1 × 2x2x2`), and 682.1 s (`2 × 2x2x2`). **Unscanned 397B** (with commit `96329637f` save-buffer cap): S2 **8,096.5 s** (transform 6,642.8 s + save 1,432.0 s, peak 884.2 GB RAM), S3 **2,025.8 s** (`2 × 2x2x1`), **1,838.3 s** (`1 × 2x2x2`), **2,233.0 s** (`1 × 2x2x4`), and **2,157.9 s** (`2 × 2x2x2`) |
| **P1 completion & P3 fixes** | ✅ Queue `p1` (`configs/p1_completion_tasks.json`), queue `p3` (`configs/p3_tasks.json`, 10/10 tasks), and all multi-host JobSets (`smoke-tm-2s`, `480b-2s`, `235b-tm`, `235b-tm-2s`, `397b-fix`, `397b-tm`, `397b-tm-4h`, `397b-tm-2s`, `ds32-4h`, `1600b-8h`) are 100% completed and cleaned up |
| **P2 · 16-model suite** | ✅ Done (`p2_leftovers`, re-runs, and `safetensor (with fix)`). With commit `207cabecc`, `gpt-oss-120b` scanned `safetensors_dynamic` now succeeds on 1 slice (**504.9 s**) and 2 slices (**430.0 s**) (U-6 resolved). `llama3.1-70b` unscanned `to_maxtext` S2 peak host RAM drops from 403.6 GB to **351.9 GB** with commit `96329637f` (U-8) |
| **Reports** | ✅ Written and current: `reports/tpu7x_large_model_200b_experiment_report.md`, `tpu7x_benchmark_ckpt_load_time_{summary.md,summary.csv,report.md,issues.md}`, `tpu7x_priority1_progress.md`, `tpu7x_multihost_results.{csv,md}`. Regenerate the CSV with `scripts/build_tpu7x_tables.py` and the row-by-row tables with `scripts/build_tpu7x_sheet_table.py --md-into` (§8) |
| **Branch (11-12)** | ✅ Rebased onto `origin/main` `5b6773c9b` (2026-11-11) and force-pushed with lease. Lance's PRs #5706, #5828, #5705 are merged into `main` now, so the branch only carries this suite, its fixes, and docs (§2) |
| **DeepSeek with real HF weights** | ⏳ Investigated on 11-12, not run. Two options are waiting for a decision (§11) |

---

## 1. Goal & experiment design

- **Priority 1 (>200B models)**: `qwen3-235b-a22b`, `deepseek4-284b`, `qwen3.5-397b-a17b`, `qwen3-480b-a35b`, `deepseek3-671b`.
  - **Exp 1A**: `safetensors_dynamic` **BEFORE** (`origin/main`) vs **WITH** Lance Wang's 3 PRs (#5706, #5828, #5705; all merged into `main` by 11-10).
  - **Exp 1B**: `to_maxtext` (S1 HF→GCS, S2 CPU conversion to an Orbax checkpoint, S3 Orbax load into HBM) vs `safetensors_dynamic` (S1 HF→GCS, S2 sharded safetensors load and in-memory conversion).
- **Priority 2**: rerun the earlier 16-model `to_maxtext` vs `safetensors_dynamic` benchmark on tpu7x (WITH the PRs).
- Every model runs in 4 configs: {`scanned`, `unscanned`} × {1 slice, 2 slices}.
- **Deliverables**: separate `reports/tpu7x_*` files (the existing v5p reports stay untouched), plus scripts and docs in this folder that reproduce every result.

---

## 2. Code & branches

| Ref | Commit | Contents |
| :--- | :--- | :--- |
| `origin/main` | `5b6773c9b` (2026-11-11) | Current base of this branch. Contains Lance's `safetensors_dynamic` PRs: #5706 (`5651d7fe2`; its stacking-rule part landed as #6116 `ce608b918`), #5828 (`9ff34c443`), #5705 (`d92b24c64`), plus #5610 (`67a726321`) and #6113 (`e600234e6`) |
| `f596f5369` | (old `origin/main`, 10-27) | Baseline for the BEFORE runs, and the base that all P1/P2/P3 runs were built on |
| `origin/fix/safetensors-dynamic-lance-prs` | `335d5e880` | Historical: `f596f5369` plus Lance's PR commits as they ran in P1/P2: `df2eacf0d` (#5706: composite keys, multi-host hooks, legacy keys), `1a89e9dfc` (#5705: `safetensors_dynamic` outside the training restore), `1f06b304d` + `335d5e880` (#5828: leading-axis sharded reads, inner MoE stacks kept in the read layout) |
| `origin/docs/checkpoint-load-benchmark` | rebased 11-12 | `origin/main` plus this benchmark suite (`fb7026e81`), the `safetensors_dynamic` HBM-peak fixes (`207cabecc`: single-leaf composite build for U-1 + pairwise non-contiguous reshard slice concat for U-6), the `to_maxtext` Orbax save-buffer cap (`96329637f`, U-8), the harness's 1,800 s Orbax multi-host barrier timeout (`06b04627b`), the 11-column Google Sheet tables with `gpaste` URLs (`fcf49887e`, `4cd977258`), and docs |

### 2.1 History rewrites (read before using an old hash)

- **11-10:** Lance rebased the branch onto `origin/main` `e6ff867c0`. He dropped his 4 PR commits (merged into `main`) and adapted `1dd7ba3a9` to `main`'s newer `tensor_handling.py`: same U-1/U-6 logic, with `delete_input` now passed through `main`'s `_device_put`.
- **11-12:** rebased again onto `5b6773c9b` with no conflicts (all 41 of Lance's commits patch-identical, plus a `handoff.md` path fix) and force-pushed with `--force-with-lease`. Unit tests were not run locally (no JAX environment on the workstation) and there is no PR for this branch, so CI has not run either.
- **Old → current hashes** (older notes and chats cite the old ones):

| Old | Current | Commit |
| :--- | :--- | :--- |
| `ef8488fec` | `fb7026e81` | Add reproducible checkpoint load-time benchmark suite and documentation |
| `1dd7ba3a9` | `207cabecc` | Stack composite leaves one by one and join reshard slices in pairs (U-1, U-6) |
| `27dec98be` | `96329637f` | Cap the to_maxtext Orbax save buffers with checkpoint_storage_concurrent_gb (U-8) |
| `ded32aa7b` | `06b04627b` | Record the 397B 2x2x2x2 Orbax 300s barrier timeout and raise the harness barrier timeout (earlier versions of this handoff wrongly credited this change to `7f3394d94`) |
| `7f3394d94` | `146ce9a5c` | Complete tpu7x multi-host DeepSeek (1x2x2x4, 1x2x4x4) and 397B Orbax load tables |
| `2902000a2` / `5581543f5` | `fcf49887e` / `4cd977258` | Add the `log` column with `gpaste` URLs / fill it for all Not run and Skipped rows |

- **Local-only branches on the workstation:** `backup/checkpoint-load-benchmark-pre-rebase-20261112` (the pre-rewrite branch; the only place the old hashes above still exist, apart from Lance's PR commits on `origin/fix/safetensors-dynamic-lance-prs`), `lance/docs-checkpoint-load-benchmark-20261110` (Lance's 11-10 rebase), and `lance/pr-{5610,5705,5706,5828,6113,6116}` (Lance's final PR heads, all verified to be in `main`).
- **The pods still run pre-rewrite code.** `gs://mesa-maxtext/mesakh_tpu7x_bench/code/maxtext_lance_prs.tar.gz` was built from the old tree (`f596f5369` + Lance's PR commits). `scripts/run_tpu7x_cluster.sh --upload-only` uploads the harness, the bootstrap, and this branch's `src/maxtext/checkpoint_conversion/utils/{load_dynamic,tensor_handling,utils}.py` (as the `fix` tree), but **not** the tarball. The rebased `utils` files may depend on newer `main` code, so rebuild the tarball from the rebased tree (same layout as the current one, see `tar tzf`) before any new cluster run.

**On this branch** (on top of the suite commit `fb7026e81`; the pods run the GCS copies uploaded by `run_tpu7x_cluster.sh`):

| Path | Contents |
| :--- | :--- |
| `benchmark_ckpt_load_time.py` | tpu7x support, zlib-compressed remote payload, env filtering that keeps `VBAR_CONTROL_SERVICE_URL`, SIGABRT teardown tolerance, phase watchdog with `TIMEOUT` labels, `--skip-safetensors`, `--tm-convert-only`, RAM-aware staging and fail-fast RAM checks for `to_maxtext` (§7.15, §7.16), shared-cache reuse when all shards are present (§7.17), parallel shard range-reads (`TM_STREAM_DOWNLOAD_WORKERS`, §7.21), and Orbax 1,800 s multi-host barrier timeout (§7.22) |
| `bootstrap_tpu7x.py`, `generate_tpu7x_manifests.py`, `scripts/run_tpu7x_cluster.sh` | Pod bootstrap (trees: Lance's PRs, `origin/main`, `hbmfix`, `fix`), manifests (P1/P2 pairs, P1-completion pair `g`, daemon controllers, recovery), launcher |
| `scripts/tpu7x_task_queue.py`, `scripts/tpu7x_daemon_controller.py`, `configs/p1_completion_tasks.json`, `configs/p3_tasks.json`, `scripts/run_tpu7x_cluster.sh --priority multihost --run <id> [--platform tpu7x\|v5p]` | P1 completion queue (`queue/p1/`), P3 fixes queue (`queue/p3/`), and multi-host JobSet launcher (`2x2x2`, `2x2x4`, `2x4x4`) |
| `scripts/summarize_tpu7x_results.py` | Status matrix from per-phase result JSONs |
| `scripts/build_tpu7x_tables.py`, `scripts/build_tpu7x_sheet_table.py` | Summary CSV; the 11-column row-by-row tables for the Google Sheet tabs (`--suite p1`: >200B models plus multi-host rows; `--suite p2`: models up to 120B; includes `LOG_URLS` mapping to `gpaste` URLs and not-run explanations) |
| `src/maxtext/checkpoint_conversion/utils/{load_dynamic.py,tensor_handling.py,utils.py}` | Production fixes on this branch: single-leaf composite build (U-1, `207cabecc`), pairwise slice concatenation (U-6, `207cabecc`), and `MAXTEXT_ORBAX_SAVE_CONCURRENT_GB` cap + `hf_state` release before Orbax save (U-8, `96329637f`) |
| `patches/hbmfix.patch`, `patches/hbmfix/src/...` | Earlier intermediate HBM-peak patch (§7.6, superseded by `207cabecc` in `src/`) |
| Not committed: `docs/plans/`, `run_e2e_matrix_xpk.{py,sh}`, `vllm_decoder_issue.md` | Unrelated to this work (never stage) |

Code bundles verified against git (before the rewrite): `code/maxtext_lance_prs.tar.gz` `src/` = this branch's `src/` at `ef8488fec` (now `fb7026e81` on a newer base, §2.1); `code/checkpoint_conversion_origin_main.tar.gz` = `f596f5369:src/maxtext/checkpoint_conversion`.

**What the BEFORE tree contains:** `/tmp/maxtext_origin_main` is the Lance-PRs tree with `src/maxtext/checkpoint_conversion` replaced by the `f596f5369` copy. Lance's PRs also touch 3 files outside that directory (`utils/globals.py`: extra `HF_IDS`; `utils/max_utils.py`: `_uses_checkpoints` for `jax.distributed` init; `utils/model_creation_utils.py`: `from_pretrained` support). None of them is on the measured path: the harness calls `jax.distributed.initialize` itself and calls `checkpointing.load_state_if_possible(source_checkpoint_layout="safetensors_dynamic")` directly (`checkpointing.py` is unchanged by the PRs).

---

## 3. Infrastructure (what the experiments ran on)

| Item | Value |
| :--- | :--- |
| Cluster | `bodaborg-tpu7x-nap` (kube context `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap`), project `cloud-tpu-shared-capacity`, zone `us-central1-c` |
| Accelerator | TPU v7 (Ironwood) `tpu7x`; machine type `tpu7x-standard-4t` |
| One slice | 1 node, topology **`2x2x1`** = **4 chips = 8 JAX TPU devices** (2 TensorCores per chip), `google.com/tpu: 4` |
| TPU HBM | **813.8 GB per slice** (8 × 101.725 GB); **1,627.6 GB on 2 slices** |
| Host | **1,014.5 GB RAM** (944.83 GiB) and 224 vCPUs per node; RAM-backed `emptyDir` mounted at `/ramdisk` |
| Capacity | **Spot** (`cloud.google.com/gke-spot: "true"`, `tpu7x.gke.io/capacity: spot`), admitted through Kueue queue `default`. Pods sit in `SchedulingGated` until Kueue admits them. |
| 1-slice runs | Single process, 8 devices, `ici_fsdp_parallelism=-1` (mesh `fsdp=8`) |
| 2-slice runs | **Multi-slice**: 2 single-host slices over DCN with `jax.distributed.initialize` (2 processes). Slice 0 runs the harness and coordinates; slice 1 runs a daemon on TCP `55555`. Models ≥70B use `dcn_fsdp_parallelism=2` (mesh `fsdp=16`); models ≤47B use `dcn_data_parallelism=2` (mesh `data=2 × fsdp=8`). |
| Container images | Bootstrap pods: `gcr.io/cloud-tpu-shared-capacity/yaoyuchen-runner:ds41-2026-10-22-16-03-48` (JAX 0.11.1). Pair `d` only: `us-central1-docker.pkg.dev/cloud-tpu-shared-capacity/bodaborg-images/maxtext-base-image:latest` (§7.11). |
| HF token | k8s secret `mesakh-hf-token` (key `HF_TOKEN`); locally in `hf_token.txt`. **Never print it.** |
| Permissions | No `pods/exec` for our account, so inspect runs through `kubectl logs` and GCS only |

**Capacity of each P1 model (BF16 weights vs. HBM):**

| Model | BF16 size / shards | 1 slice (813.8 GB) | 2 slices (1,627.6 GB) |
| :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 470.38 GB / 118 | ✅ fits | ✅ fits |
| `deepseek4-284b` | 581.89 GB / 146 | ✅ fits | ✅ fits |
| `qwen3.5-397b-a17b` | 795.48 GB / 94 | ❌ 97.7% of HBM, no headroom left for the transform | ⚠️ weights fit; the scanned transform peak OOMs (§7.6) |
| `qwen3-480b-a35b` | 961.03 GB / 241 | ❌ exceeds HBM | ✅ fits |
| `deepseek3-671b` | 1,342.35 GB (staged sparse copy: 1,365.27 GB / 343) | ❌ exceeds HBM | ❌ host OOM-killed during the load; ✅ loads on 4 slices (690 GB peak host RAM per host) |

---

## 4. How the runs are orchestrated

- **Pod pair `<id>`**: `mesakh-tpu7x-<id>0` (`ROLE=runner`, slice 0, runs `RUNNER_SCRIPT`), `mesakh-tpu7x-<id>1` (`ROLE=daemon`, slice 1, `--remote-slice-daemon --daemon-port 55555`), and the headless Service `mesakh-w1-<id>`.
  [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py) generates them and [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) deploys them.
  - P1 pairs: `a` = 235B, `b` = 397B, `c` = 480B, `f` = 284B + 671B. Each runs 3 stages: WITH PRs → BEFORE (`origin/main`, watchdog `PHASE_TIMEOUT_S`: 360 s when these pairs were deployed, 900 s in the current generator) → full TM vs ST comparison.
  - P2 pairs: `d` = 10 small and medium models, `e` = `qwen3-32b`, `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b`.
- **What [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py) does at pod start** (it is downloaded from `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`):
  1. Installs missing Python packages from `site_packages_overlay.tar.gz`, without shadowing the image's jax, flax, or orbax.
  2. Builds `/tmp/maxtext` from `maxtext_lance_prs.tar.gz` (the Lance-PRs tree), then overwrites `benchmark_ckpt_load_time.py` with the latest copy from GCS.
  3. Builds `/tmp/maxtext_origin_main` by copying that tree and overlaying `checkpoint_conversion_origin_main.tar.gz`. The BEFORE runs use this tree.
  4. **Patches the code by string replacement.** In both trees, `load_dynamic.py` gets single-GET HF→GCS streaming with 16 workers. In `/tmp/maxtext` only, `tensor_handling.py` and `load_dynamic.py` get the older HBM patch: free reshard/stack intermediates and split composite leaves before the reshard.
  5. Builds `/tmp/maxtext_hbmfix`: a copy of the patched `/tmp/maxtext` whose `load_dynamic.py` and `tensor_handling.py` are replaced by the §7.6 fix from `gs://…/code/` (same files as `patches/hbmfix/`), then re-patched for single-GET streaming. Only queue tasks labeled `hbmfix` use this tree.
  > ⚠️ **So the "WITH PRs" numbers are Lance's PRs plus the step-4 string patches**, not the bare PRs. I checked that every patch pattern matches the PR code (so each patch applies). The BEFORE tree gets only the single-GET streaming patch, which affects Step 1 (HF→GCS) and not the timed S2 load.
- **Recovery controller `mesakh-ctrl-c1`** (a `python:3.11-slim` pod on a CPU node). After spot preemptions, it reuses the two surviving daemon pods: `c1` (10.224.42.10) acts as slice 0 and `f1` (10.224.75.10) as slice 1. It first syncs the harness from GCS to both, then runs this queue on `c1`:
  ```bash
  export PYTHONPATH=/tmp/maxtext/src:/tmp/maxtext HF_HOME=/ramdisk/hf_cache TMPDIR=/ramdisk/tmp KEEP_HF_CACHE=1
  COMMON="--devices-per-slice 8 --hardware tpu --remote-slice-addrs 10.224.75.10:55555 --run-id tpu7x_200b --reuse-model-hf-cache --skip-completed --allow-step-failure"
  # 1) qwen3.5-397b WITH PRs, 2 slices                                     -> results/p1_st_lance_b_s2.csv   [DONE: failed, §7.6]
  python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models qwen3.5-397b-a17b \
    --scan-modes scanned unscanned --num-slices-list 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \
    --local-work-dir /tmp/p1_lance_prs --csv-report /tmp/p1_st_lance_b_s2.csv --skip-to-maxtext $COMMON
  # 2) DeepSeek BEFORE (origin/main), 1+2 slices, 360 s watchdog            -> results/p1_origin_main_f.csv   [DONE: inconclusive, §7.5]
  PYTHONPATH=/tmp/maxtext_origin_main/src:/tmp/maxtext_origin_main STAGE_SPARSE_DEEPSEEK_OOM=1 PHASE_TIMEOUT_S=360 \
  SHARED_HF_CACHE_BASE_DIR=gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/safetensors_dynamic \
  python3 -u /tmp/maxtext_origin_main/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models deepseek4-284b deepseek3-671b \
    --scan-modes scanned unscanned --num-slices-list 1 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main \
    --local-work-dir /tmp/p1_origin_main --csv-report /tmp/p1_origin_main_f.csv --skip-to-maxtext $COMMON
  # 3) qwen3-235b full to_maxtext + safetensors_dynamic, 1+2 slices        -> results/p1_full_a.csv          [INTERRUPTED ~09:01 UTC, §7.14]
  python3 -u /tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --models qwen3-235b-a22b \
    --scan-modes scanned unscanned --num-slices-list 1 2 --gcs-base gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs \
    --local-work-dir /tmp/p1_lance_prs --csv-report /tmp/p1_full_a.csv $COMMON
  ```
- **Harness rules that shape the result tables:**
  - If `scanned` fails with OOM or timeout, the harness **skips `unscanned`** for that slice count.
  - `--skip-completed` reuses result JSONs from GCS. FAILED results are **re-run by default**; with `SKIP_FAILED_RESULTS=1` they are skipped instead (§7.7).

---

## 5. Results (snapshot)

### 5.1 / 5.2 P1 · Exp 1A and Exp 1B

The per-test P1 tables (summary by model, the row-by-row `to_maxtext` vs `safetensor` table, and all tests) are kept current in
[`reports/tpu7x_priority1_progress.md`](reports/tpu7x_priority1_progress.md). Highlights are in §0. Each P1 model's cold Step 1 time is 583.3 s (235B), 814.5 s (397B), 1,046.1 s (480B, `to_maxtext` S1 measured separately: 1,135.5 s), or 0 for the synthetic DeepSeek files.

### 5.3 P2 · 16-model suite (WITH PRs + `safetensor (with fix)`)

✅ Complete. All configurations are in [`reports/tpu7x_benchmark_ckpt_load_time_summary.md`](reports/tpu7x_benchmark_ckpt_load_time_summary.md) (row by row, §3.1) and the summary CSV. Points to know:
- `gpt-oss-120b` scanned `safetensors_dynamic` ran out of HBM on the initial Lance-PRs branch (U-6); with commit `207cabecc` (`safetensor (with fix)`, pairwise slice concatenation), it succeeds on 1 slice (**504.92 s**) and 2 slices (**429.98 s**).
- The `llama3.1-70b` scanned 2-slice Orbax load needed `checkpoint_storage_concurrent_gb=38` (T-13); with commit `96329637f` (`MAXTEXT_ORBAX_SAVE_CONCURRENT_GB=40`), `llama3.1-70b` unscanned S2 peak host RAM drops from 403.61 GB to **351.92 GB** (U-8).
- `qwen3-14b` and `qwen3-32b` scanned 1-slice outliers were re-run (U-7, `run_tpu7x_16m_rerun`).

### 5.4 Not run (and why)

- `deepseek3-671b` (`1,342.4 GB`), `deepseek3.2-671b` (`1,367.0 GB`), and `deepseek4-1600b` (`3,197.7 GB`) `to_maxtext`: not run because `to_maxtext` Step 2 is a single-host CPU script and even the converted state alone (`1.34–3.20 TB`) exceeds the **1,014.5 GB** host RAM of one `tpu7x` node. All three are benchmarked with `safetensors_dynamic` across 4 or 8 hosts (`4 × 2x2x1`, `1 × 2x2x4`, `2 × 2x2x2`, `1 × 2x4x4`, `4 × 2x2x2`; [`reports/tpu7x_multihost_results.md`](reports/tpu7x_multihost_results.md)). All DeepSeek runs use synthetic files; for real weights see §11.
- `deepseek3-671b` unscanned on 1–2 slices (`1,342.4 GB` > 813.8 GB HBM on 1 slice and host-OOMs on 2 slices; scanned already failed on both), `qwen3.5-397b-a17b` unscanned `safetensors_dynamic` on 1 slice (scanned failed HBM OOM at 97.7% capacity), `qwen3-480b-a35b` unscanned `safetensors_dynamic` on 1 slice (961.0 GB > 813.8 GB HBM), and `qwen3-0.6b` `to_maxtext` on `1 × 2x2x2` (that smoke test ran `safetensors_dynamic` only; multi-host `to_maxtext` was verified on `2 × 2x2x2`). Every non-executed row has an explicit explanation in Column K (`log`) of the Google Sheet.

---

## 6. Live cluster state (at snapshot)

**Cleaned up on 10-30 after all P1, P2, P3 (`queue/p3/`), and multi-host JobSets finished.** No `mesakh-*` pods, services, or JobSets remain. Every queue task in `queue/p1/` and `queue/p3/` (10/10 tasks) is completed. All results are in GCS, committed to git, and synced to the Google Sheet.

---

## 7. Known issues & findings

1. **tpu7x TPU init needs `VBAR_CONTROL_SERVICE_URL`.** GKE injects `VBAR_CONTROL_SERVICE_URL=unix:///var/run/tpu-plugin/vbar.sock`. Filtering out `*_SERVICE_*` env vars drops it, and TPU init then fails with `Failed to connect to [::]:8353`. *Fixed*: the harness now strips only `*_SERVICE_HOST`, `*_SERVICE_PORT*`, and `*_TCP*`. Two stored records predate the fix: `qwen3.5-397b-a17b` WITH scanned · 1 and an older `deepseek3-671b` `origin/main` scanned · 2. Re-run them.
2. **Linux `MAX_ARG_STRLEN` (131,072 bytes).** The harness ships its own source to the remote daemon as one argument, which grew past the limit (`[Errno 7] Argument list too long`). *Fixed*: zlib then base64, about 28 KB.
3. **SIGABRT (-6) at multi-slice interpreter teardown.** *Fixed*: the run is not marked failed when its result JSON is already complete.
4. **Watchdog kills are recorded as `OOMKilled / Exit Code 137`.** The harness labels every `-9` exit as OOM (`benchmark_ckpt_load_time.py` ~L2276/L2417), including SIGKILLs from `PHASE_TIMEOUT_S`. Every BEFORE `-9` above had a peak host RAM of ≤194 GB out of 1,014.5 GB, so they were timeouts, not OOMs. *Fixed for new runs*: the harness now records `TIMEOUT` when its watchdog fired. Records written before the fix still say OOM; reclassify them from the `[TIMEOUT_WATCHDOG]` lines in the run output when writing the reports.
5. **A watchdog shorter than the WITH-PRs time is inconclusive.** The DeepSeek BEFORE runs were killed at 360 s, but with the PRs they need 353–453 s. Likewise, `qwen3.5-397b-a17b` scanned · 1 took 947.7 s just to load with the PRs. **Re-run BEFORE with `PHASE_TIMEOUT_S=1800`** or more. The original P1 manifests (deployed 06:00 UTC) used `PHASE_TIMEOUT_S=360` for all BEFORE runs, so `qwen3-235b-a22b` (WITH: 697–792 s) is inconclusive too; the generator was later changed to 900 s.
6. **`qwen3.5-397b-a17b` scanned · 2-slice HBM OOM in the composite `(wi_0, wi_1)` path.**
   - **What happens:** each layer's `mlp.experts.gate_up_proj` is 512 × 2048 × 4096 BF16, about 8.6 GB. 45 local layers are stacked into one composite target `(…, 2)` of about 387 GB, about 24.2 GB per device across 16 devices. It is then sliced into `wi_0` and `wi_1` (about 12.1 GB per device each, which is the failed `11.25G` allocation) while the rest of the model already fills HBM.
   - **What changed:** the fix (`patches/hbmfix/`, deployed on the pods as the `/tmp/maxtext_hbmfix` tree and run as `st_397b_scanned_s2_hbmfix`) does three things:
     - Frees each consumed HF input right after its hook, but only tensors popped from `hf_state` on their last use. Tensors shared by several leaves are kept.
     - Frees stack, `astype`, and `moveaxis` intermediates as soon as their successor is ready.
     - Makes `_reshard(..., delete_input=)` opt-in, passed only where the input is a known intermediate.
   - **Caveat:** the composite split itself still briefly holds `loaded` plus both parts. If that is still too much, split at the leaf level instead: split each layer into `wi_0`/`wi_1` and stack the two separately.
   - **Fastest path to a 397B data point:** run unscanned on 2 slices. Per-layer composites are about 0.5 GB per device, and `qwen3-480b-a35b` unscanned · 2 (961 GB) already succeeded. *Done at 13:12 UTC: 1,311.9 s (load 515.9 s + transform 795.7 s), peak host RAM 98.7 GB.*
7. **`SKIP_FAILED_RESULTS` docs were inverted** (now fixed in the README and the harness docstring). In code, with `--skip-completed`, FAILED results are re-run by default, and `SKIP_FAILED_RESULTS=1` skips them. The P1 pair manifests and **pair `d`** set it to `1`.
8. **`qwen3-30b-a3b` unscanned TM S2 failed with `HFValidationError`.** The local staging dir `/tmp/p2_lance_prs/hf_local/qwen3-30b-a3b` was missing (an older harness staging path), so `to_maxtext` treated the path as a repo id. The current harness re-stages into `/ramdisk/hf_local/`, so a plain retry should work.
9. **`llama3.1-70b` unscanned TM S1 was killed (-9) at 21.6 GB RAM during the HF download** (07:45, an external kill). Pair `e` doesn't set `SKIP_FAILED_RESULTS`, so it will retry this automatically.
10. **The DeepSeek P1 numbers use `STAGE_SPARSE_DEEPSEEK_OOM=1`**: synthetic sparse BF16 safetensors on local disk, with no HF→GCS step. They measure load/transform mechanics and capacity, not GCS read bandwidth, so don't compare them directly with the Qwen GCS loads. This is why their `HF -> GCS` cells read `0.00 (local stage)`. Real-weights options: §11.
11. **Pair `d` is inconsistent with the other pairs:**
    - It uses a different image and runs the pushed branch (`git clone`) without the bootstrap patches.
    - It sets `SKIP_FAILED_RESULTS=1`, so it would **not** retry `qwen3-30b-a3b` unscanned.
    - It queues `gpt-oss-120b`, which pair `e` also runs. The two would race on the same GCS result paths.
12. **P2 outliers:** `qwen3-14b` scanned · 1 ST took 342.7 s (298.4 s in the transform, vs. 69.0 s on 2 slices). `qwen3-32b` scanned · 1 is 3–5× slower than unscanned in every phase. Re-run both before reporting.
13. **`run_tpu7x_cluster.sh --cleanup`** deletes the pods labeled `app=mesakh-tpu7x-bench`, every `mesakh-w<k>-<id>` daemon Service (they carry no labels), and `mesakh-ctrl-c1` (created before the controllers got the label). `--status` still only lists the labeled pods.
14. **Kueue can evict running pods.** At ~09:01 UTC, `c1` was stopped after about 3 h with event `EvictedDueToDeactivated: The workload is deactivated`. Our workloads set no `maximumExecutionTimeSeconds`, so the cause isn't visible to us (likely a cluster-level deactivation or quota action). Assume any pod can disappear: keep queues short, rely on the per-phase result JSONs in GCS (`--skip-completed` resumes from them), and don't depend on a 2-pod pairing surviving.
    - Root cause confirmed later: the `default` ClusterQueue has **0 nominal tpu7x quota** and borrows all its chips from the cohort, so `priority-dev` reclaims them (`Preempted ... due to reclamation within the cohort`). Spot nodes also go NotReady. This is why the P1 completion run uses the GCS task queue.
    - **The `The workload is deactivated` events come ~3 h after pod start:** `c1`, `g1` (10:29→13:32), pair `i` (13:13→16:14), `s3`/`s8` (13:33→~16:43), and the 397B re-run `s1` (17:56→20:57). The likely enforcer is the calendar-schedule daemon in `kueue-system` (configmap `calendar-schedule-daemon-code`). Its code caps individual bookings at 3h on NAP (`max_duration_individual`) and deactivates workloads by patching `{"spec": {"active": False}}`. The exact rule for unbooked workloads wasn't confirmed, but the timing is consistent. **Treat ~3 h as the hard limit for one run** on this cluster, unless you have a booking.
15. **Scanned `to_maxtext` conversion thrashed when streaming shards.** For very large models, the harness used to stage only index/config files and stream shards from GCS through a 3-shard LRU cache. A scanned MoE leaf stacks each expert across all layers, so every leaf reads every shard. With a 3-shard cache, the checkpoint was re-downloaded once per expert: the 235B scanned conversion ran 2 h at leaf 1/15.
    - *Fixed*: the harness stages all shards on `/ramdisk` only if `free − checkpoint − 60 GB ≥ checkpoint` (§7.16). For scanned conversions that don't fit (235B: 470 GB, 397B: 795 GB, 480B: 961 GB on a 1,014.5 GB host whose boot disk is only 100 GB), it fails fast with the reason. Unscanned leaves read one layer at a time, so streaming stays for them.
    - Re-staging inside Step 2 is now recorded as `hf_to_local_restage_s`.
16. **`to_maxtext` S2 needs host RAM for the whole converted state.** `to_maxtext` materializes every leaf (lazy loading only delays it) and keeps the full converted state in host RAM until the Orbax save, so it needs about the checkpoint size again on top of any staged shards. A tpu7x node has 1,014.5 GB RAM, 986.7 GB allocatable.
    - **Evidence:** the 235B scanned conversion with all 470 GB staged in `/ramdisk` was OOM-killed at leaf 4/15 (13:23 UTC). The unscanned one was evicted at 944 GB container memory (13:32 UTC). `deepseek4-284b` S2 (both scan modes) was OOM-killed at 989.6 GB.
    - **Fix in the harness:** stage on `/ramdisk` only if `free − checkpoint − 60 GB ≥ checkpoint`, otherwise stream shards (unscanned) or fail fast with the reason (scanned). Also fail fast when the converted state alone exceeds 93% of host RAM (480B unscanned: 960 GB vs ~943 GB).
    - **Result:** TM S2 can't run on one host for scanned 235B/397B/480B, unscanned 397B/480B, and 284B. Only unscanned 235B converts (streaming shards; §7.21).
    - **Deployment:** queue workers run the harness they downloaded at pod start. So the affected tasks name the tree `/tmp/maxtext_stagefix`, a symlink that only pods bootstrapped after the fix have, and older pods never claim them. `tpu7x_task_queue.py` now also syncs the harness from GCS before each task (pods started after 13:31 UTC), so later harness fixes need no gating. The 2-slice TM load tasks set `SKIP_FAILED_RESULTS=1`, so an old pod never re-runs a failed conversion.
17. **Two workers re-downloaded the 397B checkpoint from HF Hub into the shared GCS cache** while other runs read it (13:26–13:38 UTC).
    - **Cause:** TM Step 1 reused the shared cache only if the scanned 1-slice `safetensors_dynamic` record had a cold Step 1 time. For 397B, that record is an infra failure with `hf_to_gcs_s=0`.
    - **What I did:** stopped both workers by replacing their task locks with a far-future heartbeat (worker `hold-coordinator`), which made their workers drop the tasks (LOST LOCK).
    - **Fix:** TM Step 1 now reuses the cache when all shards are present. The cold time recorded with it is the largest Step 1 time recorded for the model, or 0 if only warm checks (under 1 s) exist (397B: use 814.53 s from `results/p1_st_lance_b.csv`).
    - **Impact on other runs:** an interrupted upload never replaces an object, and the re-uploaded bytes are identical. The concurrent `hbmfix` load completed and failed later in HBM, for the reason in §7.19. The BEFORE 397B load (`origin/main`) overlapped the re-download and never left its load phase before the 1800 s timeout. With the PRs, that load takes 515.9 s, so the overlap can't explain the gap; the reports still mention it.
18. **More capacity loss during the scale-out:**
    - Kueue deactivated `g1` at 13:32 UTC (`The workload is deactivated`, same as `c1` in §7.14), so its daemon controller and slice 1 (`h1`) were deleted.
    - Spot node shutdowns killed `m3` and `s7` at 13:42 UTC. `m3`'s loss also failed the BEFORE 671B 4-slice run, which the queue released for retry (attempt 1/3). Its stored `OOMKilled` label comes from the local slice being killed after the remote slice died; it is not a real OOM.
    - Node `gke-tpu-fd9dfbd0-2jfc` stopped answering kubelet calls.
    - **Recovery:** reset the dead worker's lock heartbeat to 0, then recreate the pod from the scale-out manifest. A runner pod resolves its daemons' IPs once at start, so after a daemon is recreated, recreate the runner too (`m0`).
19. **`hbmfix` did not fix `qwen3.5-397b-a17b` scanned · 2.** The load finished (469.7 s), then the split of the stacked `(wi_0, wi_1)` composite failed at the same allocation (11.25 GiB needed, 2.77 GiB free): the composite (~22.5 GiB per device) is still alive while its first half is created. Next fix: split each layer's `gate_up_proj` before stacking, so only one target leaf is built at a time. Alternatively, move the composite to host before splitting. Meanwhile, `st_397b_scanned_s4` measures scanned 397B with 4 slices of HBM headroom.
20. **A queue worker's heartbeat thread can die on a transient auth error.** On the 397B re-run (`s1`, 19:49 UTC), a failed credential refresh raised inside `_beat`, which ended the thread while the harness kept converting. The lock then goes stale, and any new worker that can run the task would take it over and start a duplicate run. *Fixed* in `tpu7x_task_queue.py` (the heartbeat logs and retries), but only for pods started after the fix. For the running re-run: create the 2-slice pair for `tm_397b_unscanned_s2` only after `tm_397b_unscanned_s1` is in `done/`.
21. **Parallel shard reads for `to_maxtext` streaming (10-29).** The single-stream fetch (~55 MB/s) made unscanned 397B take ~4.3 h, longer than the ~3 h workload limit. `_parallel_gcs_download` now fetches each shard as 16 concurrent range reads (`TM_STREAM_DOWNLOAD_WORKERS`). Re-runs went to run dir `p1_lance_prs/run_tpu7x_200b_pstream`.
    - **235B unscanned:** S2 2,625.5 s (569 GB downloaded in 1,538 s), S3 545.5 s / 659.1 s on 1 / 2 slices.
    - **397B unscanned:** the transform finished in 6,860 s, then the pod was OOM-killed when the Orbax save started. The ~807 GB converted state plus save buffers exceed the ~944 GB a pod can use. Its `to_maxtext_step2_result.json` in the pstream run dir was written from the pod log (`recorded_by`).
    - **Capacity:** spot capacity was scarce on 10-29 (3–16 single-host spot nodes, frequent taint evictions). Two 397B attempts and three pair pods were lost before the last attempt ran.
22. **HBM-peak fixes (`207cabecc`), `to_maxtext` Orbax save-buffer cap (`96329637f`), and Orbax 1,800 s barrier timeout (`06b04627b`):**
    - **U-1 (`qwen3.5-397b-a17b` scanned composite `(wi_0, wi_1)` HBM OOM):** `load_dynamic.py` now builds and reshards one composite target leaf at a time instead of stacking the `(wi_0, wi_1)` pair and slicing it in HBM. Validated on `2 × 2x2x1` (**1,608.32 s**) and `1 × 2x2x2` (**1,475.17 s**).
    - **U-6 (`gpt-oss-120b` scanned HLO temporaries HBM OOM):** `tensor_handling.py` now concatenates non-contiguous reshard slices pairwise instead of in a single N-way `jnp.concatenate` that materialized ~7× the leaf in XLA temporaries (200.32 GiB / 100.21 GiB per device). Validated on `1 × 2x2x1` (**504.92 s**) and `2 × 2x2x1` (**429.98 s**).
    - **U-8 (`to_maxtext` Step 2 Orbax save host OOM):** `checkpoint_conversion/utils/utils.py` caps Orbax save concurrency (`MAXTEXT_ORBAX_SAVE_CONCURRENT_GB`, default 40 GB) and drops `hf_state` before saving. Reduced `llama3.1-70b` unscanned S2 peak host RAM from 403.61 GB to **351.92 GB**, and enabled `qwen3.5-397b-a17b` unscanned S2 to complete on one host in **8,096.53 s** (peak **884.23 GB** RAM).
    - **Orbax v1 multi-host barrier timeout:** `orbax.checkpoint.experimental.v1._src.synchronization.multihost._DEFAULT_BARRIER_TIMEOUT` defaults to 300 s, which timed out during `qwen3.5-397b-a17b` unscanned S3 on `2 × 2x2x2` when one host finished reading GCS >5 min after the others. `benchmark_ckpt_load_time.py` raises `_DEFAULT_BARRIER_TIMEOUT` to 1,800 s, allowing all four multi-slice/multi-host 397B Orbax loads (`2 (16)` **2,025.76 s**, `1 (16)` **1,838.26 s**, `1 (32)` **2,233.01 s**, `2 (32)` **2,157.89 s**) to succeed.
23. **Google Sheet 11th column (`log`) with `gpaste` URLs (`fcf49887e`, `4cd977258`):**
    - `scripts/build_tpu7x_sheet_table.py` outputs an 11th column (`log`) containing public `paste.googleplex.com` URLs for all 17 failed or slow-outlier experiments across both tabs (`tpu-v7x(Large models > 200B)` and `tpu-v7x (Smaller models)`) and explicit `Not run` / `Skipped` explanations for all 22 non-executed rows (`LOG_URLS`).

---

## 8. Next steps (in order)

1. **Decide the DeepSeek real-weights follow-up (§11):** option 1 (copy the existing BF16 copies; quick) or option 2 (download from HF and dequantize; the only way to get `deepseek4-1600b`). Nothing has been started.
2. **Before any new cluster run,** rebuild `gs://mesa-maxtext/mesakh_tpu7x_bench/code/maxtext_lance_prs.tar.gz` from the rebased tree, then run `scripts/run_tpu7x_cluster.sh --upload-only` (§2.1).
3. **Lance's local copy** of `docs/checkpoint-load-benchmark` predates the 11-12 force-push; he needs `git fetch origin && git reset --hard origin/docs/checkpoint-load-benchmark` before committing to it.

Everything else (runs, fixes, reports, and the Google Sheet tables with `gpaste` logs in Column K) is finished and pushed; the cluster is cleaned up. To rebuild the tables from local mirrors:
- **Summary CSV:** `scripts/build_tpu7x_tables.py <p2> <p2_rerun> <p1> <p1_pstream> <p1_tmfix> --cold-s1 qwen3.5-397b-a17b=814.53 --csv reports/tpu7x_benchmark_ckpt_load_time_summary.csv --md <scratch.md>`
- **Sheet tables (11 columns with `log`):**
  - `scripts/build_tpu7x_sheet_table.py --suite p1 <p1> <p1_pstream> <p1_tmfix> --cold-s1 qwen3.5-397b-a17b=814.53 --multihost-csv reports/tpu7x_multihost_results.csv --variant "with fix=<p1_fix>" --md-into reports/tpu7x_priority1_progress.md`
  - `scripts/build_tpu7x_sheet_table.py --suite p2 <p2> <p2_rerun> --multihost-csv reports/tpu7x_multihost_results.csv --variant "with fix=<p2_fix>" --md-into reports/tpu7x_benchmark_ckpt_load_time_summary.md`

## 9. Command cheat sheet

```bash
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
kubectl --context=$CTX get pods -o wide | grep mesakh
kubectl --context=$CTX logs --since=15m mesakh-ctrl-g1 | grep -v "Constructed sharding" | tail -n 40
kubectl --context=$CTX logs --tail=40 mesakh-tpu7x-m0

# Queue progress: finished tasks, and running tasks with their worker
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/done/
gcloud storage cat 'gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/locks/*.json'   # {"worker": ..., "heartbeat": <epoch s>}
# A heartbeat older than ~2 min means the worker is gone; another worker takes the task over after 20 min.

# Result CSVs written by the runner scripts
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/results/

# Status matrix from the per-phase result JSONs (works while runs are in progress)
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/logs /tmp/tpu7x_results/p2 --exclude='.*\.log$'
python3 checkpoint_load_benchmark/scripts/summarize_tpu7x_results.py /tmp/tpu7x_results/p2
#   P1: .../p1_lance_prs/run_tpu7x_200b/logs and .../p1_origin_main/run_tpu7x_200b/logs

# Upload harness and bootstrap changes to GCS (pods fetch them at start; ctrl-c1 fetches them when its queue starts)
./checkpoint_load_benchmark/scripts/run_tpu7x_cluster.sh --upload-only
```

## 10. Where things are

| What | Location |
| :--- | :--- |
| Harness | [`benchmark_ckpt_load_time.py`](benchmark_ckpt_load_time.py) (the pods run the copy in `gs://mesa-maxtext/mesakh_tpu7x_bench/code/`) |
| Pod bootstrap / manifests / launcher | [`bootstrap_tpu7x.py`](bootstrap_tpu7x.py), [`generate_tpu7x_manifests.py`](generate_tpu7x_manifests.py), [`scripts/run_tpu7x_cluster.sh`](scripts/run_tpu7x_cluster.sh) (multi-host runs: `--priority multihost --run <id> [--platform tpu7x\|v5p]`) |
| Code tarballs used by the bootstrap | `gs://mesa-maxtext/mesakh_tpu7x_bench/code/{maxtext_lance_prs,checkpoint_conversion_origin_main,site_packages_overlay}.tar.gz` |
| P1 WITH PRs (HF cache, Orbax checkpoints, logs) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/{safetensors_dynamic,to_maxtext,logs}/` |
| P1 parallel-stream & fix runs (`pstream`, `fix`, `tmfix`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_{pstream,fix,tmfix}/logs/` |
| P1 BEFORE (`origin/main`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_origin_main/run_tpu7x_200b/logs/` (360 s watchdog, inconclusive) and `.../p1_origin_main/run_tpu7x_200b_t1800/logs/` (1800 s re-runs) |
| P1 WITH PRs + `hbmfix` (earlier intermediate patch) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b_hbmfix/logs/` |
| Task queues (`p1` and `p3`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/{p1,p3}/{tasks.json,locks/,done/,attempts/}` |
| P2 & P2 reruns / fixes (`rerun`, `fix`, `tmfix`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/{safetensors_dynamic,to_maxtext,logs}/` and `.../run_tpu7x_16m_{rerun,fix,tmfix}/logs/` |
| Multi-host runs (`2x2x2`, `2x2x4`, `2x4x4`) | `gs://mesa-maxtext/mesakh_tpu7x_bench/lance_prs/run_tpu7x_multihost/logs/` and `gs://mesa-maxtext/wanglance_tpu7x_bench/lance_prs/run_tpu7x_multihost/logs/` (summarized in [`reports/tpu7x_multihost_results.csv`](reports/tpu7x_multihost_results.csv)) |
| Per-phase result JSON | `<run dir>/logs/<model>/<scanned\|unscanned>[_slices2\|_slices4]/<phase>_result.json` (+ `<phase>.log`, `<phase>_slice1.log`) |
| Result CSVs | `gs://mesa-maxtext/mesakh_tpu7x_bench/results/{p1_st_lance_*,p1_origin_main_*,p1_full_*,p2_full_*}.csv` |
| Google Sheet (synced from `build_tpu7x_sheet_table.py`) | [`11S2eN8mlpuQDQyMyBoBiBsF9KCRd5ZvNHcDf2T8Bz0Y`](https://docs.google.com/spreadsheets/d/11S2eN8mlpuQDQyMyBoBiBsF9KCRd5ZvNHcDf2T8Bz0Y/edit?usp=sharing) — tab `tpu-v7x(Large models > 200B)` (`sheetId=635831153`, `--suite p1`) and tab `tpu-v7x (Smaller models)` (`sheetId=1294150880`, `--suite p2`); `gpaste` URLs and not-run explanations in `LOG_URLS` inside [`scripts/build_tpu7x_sheet_table.py`](scripts/build_tpu7x_sheet_table.py) |
| v5p reports (keep untouched) | [`reports/benchmark_ckpt_load_time_*`](reports/) |
| tpu7x reports | [`reports/tpu7x_priority1_progress.md`](reports/tpu7x_priority1_progress.md), [`reports/tpu7x_large_model_200b_experiment_report.md`](reports/tpu7x_large_model_200b_experiment_report.md), [`reports/tpu7x_benchmark_ckpt_load_time_{summary.md,summary.csv,report.md,issues.md}`](reports/), [`reports/tpu7x_multihost_results.{csv,md}`](reports/) |
| DeepSeek BF16 copies (MaxText team's bucket; we have read access) | `gs://maxtext-deepseek/deepseek4/hf-bf16/` (`deepseek4-284b`), `gs://maxtext-deepseek/deepseek3-671b/hf/` (`deepseek3-671b`), `gs://maxtext-deepseek/deepseek3.2/hf-bf16/` (`deepseek3.2-671b`); see §11 |
| DeepSeek FP8/FP4 → BF16 tool | [`src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py`](../src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py); MaxText recipes in [`tests/end_to_end/tpu/deepseek/`](../tests/end_to_end/tpu/deepseek/) |

---

## 11. DeepSeek with real HuggingFace weights (investigated 11-12, not run)

All DeepSeek numbers so far come from synthetic sparse BF16 files (§7.10). The HF repos exist and are public and ungated (V4-Flash and V4-Pro since 2026-04-22), but none of them is stored in BF16:

| Model | HF repo (format) | HF download | BF16 size | Existing BF16 copy |
| :--- | :--- | ---: | ---: | :--- |
| `deepseek4-284b` | `deepseek-ai/DeepSeek-V4-Flash` (FP8 + FP4 routed experts) | 159.6 GB | 581.9 GB | `gs://maxtext-deepseek/deepseek4/hf-bf16/` (46 shards) |
| `deepseek3-671b` | `deepseek-ai/DeepSeek-V3` (FP8, 128×128 block scales) | 688.6 GB | 1,369.0 GB | `gs://maxtext-deepseek/deepseek3-671b/hf/` (163 shards) |
| `deepseek3.2-671b` | `deepseek-ai/DeepSeek-V3.2` (FP8) | 689.5 GB | 1,370.7 GB | `gs://maxtext-deepseek/deepseek3.2/hf-bf16/` |
| `deepseek4-1600b` | `deepseek-ai/DeepSeek-V4-Pro` (FP8 + FP4) | 864.7 GB | ~3.2 TB | none |

- HF download sizes are summed from the HF API (`?blobs=true`); BF16 sizes are measured with `gcloud storage du -s` (V4-Pro: `HF_SHAPE` estimate). They match our synthetic files within 0.3%, so real weights hit the same RAM/HBM limits.
- **Neither loader can read the HF files directly** (checked on the rebased branch):
  - `safetensors_dynamic` (`utils/load_dynamic.py`, `utils/tensor_handling.py`) has no FP8/FP4 dequantization.
  - `to_maxtext.py` casts FP8 to BF16 without applying the `weight_scale_inv` block scales (wrong weights). V4's FP4 experts are packed 2 per byte (`I8 [2048, 2048]` for a `[2048, 4096]` weight, with `F8_E8M0` scales), so the shapes don't match either.
- **MaxText's own recipe:** HF download → `deepseek_dequantize.py` (FP8/FP4 → BF16 on CPU) → `to_maxtext` on a large CPU machine (`tests/end_to_end/tpu/deepseek/{v3-671b,v3.2-671b,v4-284b,v4-1600b}/1_test_deepseek.sh`).

**What real weights would allow:**
- **`safetensors_dynamic`:** ✅ the same slice layouts as the synthetic runs, but reading real bytes from GCS, so the results become comparable with the Qwen rows. Needs a small harness change: let the DeepSeek models take a GCS BF16 source instead of `hf://<repo>` or the sparse files (`_should_stage_sparse_bf16`, `STAGE_SPARSE_DEEPSEEK_OOM`). Risk: 284B on 1 slice already peaked at 963–980 GB host RAM out of 1,014.5 GB.
- **`to_maxtext`:**
  - 671B, 3.2-671B, and 1600B: ❌ on a tpu7x host, since the converted state alone (1.37–3.2 TB) exceeds 1,014.5 GB RAM. Only possible on a multi-TB-RAM CPU VM, whose S2 time wouldn't be comparable with the other rows.
  - 284B scanned: ❌ (582 GB staged source + 582 GB converted state).
  - 284B unscanned: ⚠️ worth a retry. The old OOM at 989.6 GB predates the U-8 save-buffer cap (`96329637f`), with which 397B unscanned fit at 884 GB.

**Options (waiting for the owner's decision):**
1. **Quick:** copy the 3 existing BF16 copies (~3.3 TB, GCS → GCS) into our bucket, then re-run `safetensors_dynamic` (plus 284B unscanned `to_maxtext`). The `HF -> GCS` column would then show a GCS copy time, not an HF download.
2. **Full HF path:** download FP8/FP4 from HF → dequantize → upload BF16. Required for 1600B, and gives a true download-plus-dequantize time. Needs a CPU VM with ~4 TB of disk for 1600B (FP8 input + BF16 output), because tpu7x nodes only have a 100 GB boot disk plus the RAM-backed `/ramdisk`.

Either option costs cluster time and adds 3.3–6.5 TB of GCS storage.
