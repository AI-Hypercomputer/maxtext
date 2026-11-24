# Checkpoint-load benchmark v3: plan

## 1. Goal

Re-run the checkpoint-load benchmark (`safetensors_dynamic` against `to_maxtext`) on TPU with the open fix PRs applied, with every run under the same conditions, and report:

- how long each model takes to load with each pipeline on today's code;
- what PRs #6513 and #6519 change, as a same-condition before/after;
- how the result compares with the v2 sheet of 2026-11-12 to 11-17.

The v2 numbers cannot serve as the "before": they were measured on `main` of 2026-11-11, on four slightly different builds, and their "cold JAX cache" was not cold (section 4.1).

## 2. Code under test

Both builds share one pinned `main` commit (`74ede7cc1`, 2026-11-24), one harness and one container image.

| Build | Branch | Contents |
|---|---|---|
| baseline | `bench/checkpoint-load-v3-baseline` | pinned main + harness and automation + #6353 + #6354 |
| candidate | `bench/checkpoint-load-v3` | baseline + #6513 + #6519 |

The two builds differ only in `tensor_handling.py`, `load_dynamic.py`, `param_mapping.py` and their two unit-test files. #6353 and #6513 both edit `tensor_handling._reshard`; the candidate keeps both changes (`delete_input` and the pairwise join from #6353, the aligned slice length and `_slice_fn` from #6513). #6398 is in neither build.

The harness is `checkpoint_load_benchmark/` as on `bench/v5p-with-pr6319` (`8990136d7`), plus:

1. per-phase cache isolation and cache evidence (section 4.1);
2. exact bit checksums of every loaded leaf (`BENCH_FULL_CHECKSUMS=1`): `bits_*`, and `bitsz_*` with `-0.0` counted as `+0.0`;
3. a `[reshard]` log line for every reshard that takes 1 s or more.

Code bundles are built with `scripts/build_code_tarball.py` from the pushed head of each branch, so every result names the commit it ran (`code_version`).

## 3. Automation

Everything is under `checkpoint_load_benchmark/wl/` and runs from a clean checkout:

| Script | Purpose |
|---|---|
| `plan.py` | The matrix: writes `reports/v3/planned_rows.csv` and the task files of the smoke, `safetensors_dynamic` and `to_maxtext` phases. |
| `wl_bench.py` | One-off operations: code bundles, Step 1 downloads, Step 1b conversion, regional copies, the `.safetensors` header scan, the cache proof. |
| `driver.py` | Runs a task file: one worker JobSet per task, lanes per platform, retries with a recorded reason. |
| `watch.py` | Follows the drivers' event logs. |
| `cache_proof.py` | The three-run proof that the cache evidence works. |

Isolation from the v2 benchmark: Kubernetes objects carry `app=wl-bench-v3`; GCS data is under `gs://mesa-maxtext-<region>/wl-bench-v3/{hf_cache,hf_cache_bf16,queue/<build>,runs-<build>,step1,step1b}` and `gs://mesa-maxtext/wl-bench-v3/code-<build>`. The automation refuses any manifest or path that names `mesa-bench-v2`, and does not submit to a cluster on which v2 benchmark pods are running.

## 4. Same-condition rules

1. One pinned main commit, one container image, one harness for every row. Every result records `code_version`, the image and the versions of jax, orbax, TensorFlow and gcsfs.
2. TensorFlow stays installed in the image, as in MaxText's own Docker images. It decides which GCS reader is used, and #6513 only matters when it is present.
3. No cache may speed up a measured phase (section 4.1).
4. The baseline and the candidate run of a configuration go on the same cluster, back to back. The order alternates from one configuration to the next, so that time-of-day and order effects do not all fall on one build.
5. At most one unit with 64 or more hosts runs per region: two at once hit the GCS egress quota.
6. Both pipelines and both builds read the same weights: the regional copies under `wl-bench-v3`, downloaded once for this benchmark. DeepSeek uses the real BF16 weights, never the synthetic sparse files.
7. Checksums are on for every run. A candidate row is valid only if its checksums equal the baseline's. Scanned gpt-oss `GptOssMlp` leaves can differ in the sign of zeros only, also between two runs of one build, and are compared with `bitsz_*`; every other leaf must match bit for bit.
8. No row is retried silently. When a task's pods die, the results of that attempt are copied to `<run dir>/attempts/a<N>/`, the reason is logged, and the next attempt re-runs the phases that have no OK result.

### 4.1 No cache

| Cache | How it is removed |
|---|---|
| JAX persistent compilation cache | Each phase gets a new, empty, uniquely named directory (`M_JAX_CACHE_DIR`, `JAX_COMPILATION_CACHE_DIR`), and every worker process deletes `~/jax_cache` and `/tmp/jax_cache` on its own host. The cache stays enabled, as it is for a user's first load. |
| In-process jit and tracing caches | One phase (one model, one scan mode, one pipeline, one build) per process. |
| Files left in the pod | One task per pod: each task gets a new JobSet whose queue worker takes one task and exits. The two scan modes of a task share the pod, which the per-phase rules cover. |
| HF hub cache | `HF_HOME` is a new empty directory per phase. |
| Leftover results and checkpoints | Every task has its own run directory, which must be empty before the first attempt. `to_maxtext` Step 3 restores the checkpoint that Step 2 of the same run wrote. |
| gcsfs and `tf.io.gfile` block caches | They live in the process and end with it. |
| Linux page cache | Measured loads read their weights from GCS. Where `to_maxtext` stages files locally as one of its own steps, that is part of the pipeline and is timed. |

Evidence in every phase result (`cache_*` fields): the cache directory and its file count at the start (must be 0) and at the end; JAX's `/jax/compilation_cache/cache_hits` and `cache_misses` counters summed over the hosts (hits must be 0); the pod start times and the number of tasks the pod ran before (must be 0); the number of hosts that reported. These reduce to `cache_valid` and `cache_invalid_reason`. Image pull and pod start are outside the measured time.

Proof that the check works (`wl_bench.py proof`): one small model is loaded three times in one pod. Run 2 follows the rules and must take as long as run 1 with no cache hit. Run 3 keeps run 2's cache directory and must be faster, with cache hits and `cache_valid: false`.

## 5. Matrix

`configs/models_v2.json`: 21 models, each with its placements per platform (v7x, v6e, v5p, v5e), both scan modes; one repetition per row, three for Tier A. `reports/v3/planned_rows.csv` lists every row with its expected outcome.

| Pipeline | Candidate | Baseline |
|---|---|---|
| `safetensors_dynamic` | Full matrix (258 rows) | Subset below (162 rows) |
| `to_maxtext` | Full matrix (258 rows) | Not run: #6513 and #6519 do not touch it |

| Baseline group | Models | Why |
|---|---|---|
| Hidden size not a power of two | `qwen3-14b`, `qwen3-32b`, `gpt-oss-20b`, `gpt-oss-120b` | Slice length fix in #6513; hooks fix in #6519 for gpt-oss |
| Fine-grained MoE on several hosts or slices | `qwen3-30b-a3b`, `qwen3-235b-a22b`, `qwen3-480b-a35b` | GCS read fix in #6513 |
| Scanned stacked MoE on many chips | `deepseek3-671b`, `qwen3.5-397b-a17b`, `deepseek4-1600b` | One-program-per-reshard fix in #6513 |
| Controls | `qwen3-0.6b`, `llama3.1-8b`, `llama3.1-70b` | Expected unchanged |

Step 1 (HF to GCS) and Step 1b (DeepSeek FP8/MXFP4 to BF16) are measured again for this benchmark; their output is the weights that every row reads.

Rows expected to fail, which are not bugs to chase: `to_maxtext` Step 2 on models larger than host RAM (`OOM-HOST`, or `NOT RUN` where the harness refuses before starting); baseline `safetensors_dynamic` of the three Qwen3 MoE models on 64 v5e hosts (`OOM-HOST`, the bug #6513 fixes); `deepseek4-1600b` scanned on two v5e 16x16 slices (`OOM-HBM` while stacking, in both builds, open issue).

## 6. Execution

1. Build: both branches, unit tests, bundles uploaded and compared with their sources.
2. Weights: Step 1 per region, Step 1b in us-central1, regional BF16 copies, then the header scan of every `.safetensors` file.
3. Smoke: `qwen3-0.6b`, both builds, both pipelines, on every platform at the smallest placement and with two slices where the matrix has it. Pass when every phase is OK, checksums are equal between builds and between pipelines, every row is cache-valid, the cache proof passes, and `code_version` matches the bundle.
4. `safetensors_dynamic`: per platform in tier order S, A, C, B.
5. `to_maxtext`, candidate build, same order.
6. Collect and check: status, checksums, cache validity, `code_version`, and that stage times add up to the total.
7. Report.

## 7. Deliverables

The CSV files under `reports/v3/` are the source of truth; the workbook and the report are generated from them.

| File | One row per |
|---|---|
| `planned_rows.csv` | configuration to run, written before anything runs |
| `results.csv` | configuration: status and total seconds per build and pipeline, before/after change, pipeline speedup, peak memory, validity |
| `stages.csv` | configuration, build, pipeline and stage |
| `runs.csv` | attempt of a phase, including failures and retries, with all raw fields |
| `failures.csv` | failed or invalid attempt: category, cause, whether it was expected, what was done |

Workbook tabs, in reading order: Read me, Summary, Before and after, Pipelines, Stages, Failures, Runs. One row per configuration, no merged cells, frozen filterable headers, whole seconds, status as a short word.

A candidate row more than 10% and 10 s slower than its baseline needs a look at the stage times before it is reported. A difference smaller than the spread of the Tier A repetitions on the same platform is reported as "no change".
