# Handoff: Checkpoint Load-Time Benchmark (`v2` Multi-Platform + `tpu7x` Archive)

> **Snapshot:** 2026-11-12 ~07:31 UTC · **Branch:** `docs/checkpoint-load-benchmark` (rebased onto `origin/main` `5b6773c9b`, code bundle uploaded from `c64b44244`) · **Sheet v2:** [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit)

---

## Quick-Pickup Guide (For Team Members & Jetski)

1. **Plan & task list:** Read [`plans/benchmark_v2_plan.md`](plans/benchmark_v2_plan.md) for the full v2 design, 21-model × 3-platform matrix (`352` planned rows), verified smoke numbers, and remaining Phases `T0.3–T0.6` + `1–5` backlog.
2. **Rules on this branch:**
   - **Never edit `src/maxtext/`** unless explicitly reviewed/approved; keep benchmark & harness tooling self-contained under `checkpoint_load_benchmark/`.
   - **Pods run only committed code:** Push `docs/checkpoint-load-benchmark`, then run `build_code_tarball.py --out /tmp/code_bundle --upload gs://mesa-maxtext/mesa-bench-v2/code` (refuses dirty tracked files or unpushed commits).
   - **Resource prefix `mesa-`:** Name all new Kubernetes JobSets/pods/KSAs/secrets `mesa-*` with label `app: mesa-bench-v2`, and write all v2 GCS outputs under `gs://mesa-maxtext/mesa-bench-v2/`. Leave legacy `mesakh_tpu7x_bench/` untouched.
   - **Never stage unrelated workspace files:** `docs/plans/`, `run_e2e_matrix_xpk.{py,sh}`, `vllm_decoder_issue.md`, and root `benchmark_v2_design_plan.md`.
   - **Teammate setup on another Cloudtop:** Ask the sheet owner (`mesakh`) for Share access to Sheet `1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`, run `gcloud container clusters get-credentials` for the three clusters in `bench/platforms.py` (`bodaborg-tpu7x-nap` / `us-central1`, `bodaborg-v6e-nap` / `southamerica-west1`, `v5e-256-bodaborg-europe-west4` / `europe-west4`), and use a Python environment with `pytest`, `pyyaml`, `etils`, and `google-api-core` (`run_smoke.sh` uses `PYTHON_BIN` if set, falling back cleanly).
3. **Current status (2026-11-12):**
   - **Done (`Phase 0` + `S0–S4` Smoke Tests):**
     - Tarball reproducibility + `CODE_VERSION.json` (`scripts/build_code_tarball.py`), `bootstrap_tpu7x.py` cleanup, `bench/platforms.py` profiles, `scripts/make_smoke_jobset.py`, `scripts/run_smoke.sh`, and mandatory TPU backend/device-count verification (`check_tpu_backend`, fails with `failure_category: "NOT-TPU"` on mismatch).
     - Multi-host TPU `SliceBuilder` gRPC fix (`c64b44244`): `init_worker_runtime` automatically sets `--slicebuilder_use_insecure_grpc=true` (`total_procs > 1`) and `--megascale_use_insecure_grpc=true` (`num_slices > 1`) in `LIBTPU_INIT_ARGS` when supported by `libtpu.so` (override with `KEEP_ALTS_GRPC=1`), sets `PYTHONUNBUFFERED=1`, and attributes remote slice aborts accurately. Local suite: `62` tests pass (`pytest -q -p no:cacheprovider checkpoint_load_benchmark/tests/`).
     - **Smoke verification (`qwen3-0.6b` scanned & unscanned, both pipelines) PASSED across all 3 platforms** (all JobSets deleted, `0 FAILED`, TPU checks OK, 12 Tier S rows recorded in Sheet v2):
       - `v7x` (`2x2x1`, `8` `TPU7x` @ `82cc1c701`): `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v7x/`
       - `v6e` (`2x4`, `8` `TPU v6 lite` @ `82cc1c701`): `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v6e/`
       - `v5e` (`16x16`, `64` hosts / `256` `TPU v5 lite` @ `c64b44244`): `gs://mesa-maxtext/mesa-bench-v2/smoke/run_smoke-v5e-r2/`
   - **Next tasks (Start Here):** Create regional buckets (`T0.3`), create `mesa-hf-token` secrets (`T0.4`), confirm `T0.5–T0.6`, and then implement `Phases 1–4` in [`plans/benchmark_v2_plan.md`](plans/benchmark_v2_plan.md) before launching Tiers `A -> C -> B`.
4. **Commands to test, upload bundle, or launch smoke tests:**
   ```bash
   # Run local unit tests
   python3 -m pytest -q -p no:cacheprovider checkpoint_load_benchmark/tests/

   # Build & upload pushed branch bundle to v2 GCS code dir
   python3 checkpoint_load_benchmark/scripts/build_code_tarball.py \
     --out /tmp/mesa_code_bundle --upload gs://mesa-maxtext/mesa-bench-v2/code

   # Dry-run or apply a smoke JobSet (--platform v7x|v6e|v5e)
   PYTHON_BIN="$(command -v python3)" ./checkpoint_load_benchmark/scripts/run_smoke.sh \
     --platform v6e --topology 2x4 --run-id smoke-v6e [--apply]

   # Monitor active jobs and inspect GCS results
   kubectl --context=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap get jobsets,pods -l app=mesa-bench-v2
   kubectl --context=gke_cloud-tpu-shared-capacity_southamerica-west1_bodaborg-v6e-nap get jobsets,pods -l app=mesa-bench-v2
   kubectl --context=gke_tpu-prod-env-multipod_europe-west4_v5e-256-bodaborg-europe-west4 get jobsets,pods -l app=mesa-bench-v2
   gcloud storage ls -r gs://mesa-maxtext/mesa-bench-v2/
   ```

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
