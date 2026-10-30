# tpu7x Checkpoint Load Time Benchmark Report: `to_maxtext` vs `safetensor` (`safetensors_dynamic`)

> TPU v7x (`bodaborg-tpu7x-nap`). Companion documents:
> - [`tpu7x_benchmark_ckpt_load_time_summary.md`](tpu7x_benchmark_ckpt_load_time_summary.md): models up to 120B, row by
>   row; the >200B rows are in [`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md).
> - [`tpu7x_benchmark_ckpt_load_time_summary.csv`](tpu7x_benchmark_ckpt_load_time_summary.csv): sub-phase values.
> - [`tpu7x_large_model_200b_experiment_report.md`](tpu7x_large_model_200b_experiment_report.md): >200B models and Lance's PRs.
> - [`tpu7x_benchmark_ckpt_load_time_issues.md`](tpu7x_benchmark_ckpt_load_time_issues.md): failures and fixes.
>
> The v5p report ([`benchmark_ckpt_load_time_report.md`](benchmark_ckpt_load_time_report.md)) is unchanged.

## 1. Executive summary

- **`safetensor` beats `to_maxtext` end-to-end** in 54 of the 56 configurations where both pipelines load the model.
  The median speedup is 1.80×, up to 4.21× (`llama3.1-70b` scanned · 2). The conversion step (`to_maxtext` Step 2) alone
  costs 19–780 s for models up to 120B, and 2,626 s for unscanned 235B (shards streamed with 16 parallel reads).
- **`safetensor` Step 2 on tpu7x is far faster than on v5p** for the same models. For example:
  - `llama3.1-70b` scanned · 1: 199.2 s vs 1,012.5 s.
  - `mixtral-8x7b` scanned · 2: 63.1 s vs 786.3 s.
  - `qwen3-32b` unscanned · 2: 74.7 s vs 661.8 s.

  Lance's PRs (leading-axis sharded reads) and the larger tpu7x hosts both contribute; this benchmark doesn't separate them.
- **`to_maxtext` Step 3 (Orbax load) for scanned MoE checkpoints is slower on tpu7x than on v5p**:
  - `mixtral-8x7b`: 534 / 793 s vs 130 / 123 s.
  - `qwen3-30b-a3b`: 386 / 449 s vs 112 / 186 s.

  Unscanned Orbax loads got faster (`llama3.1-70b` unscanned · 2: 42.5 s vs 140.1 s). This needs profiling; see §5.
- **Above 200B, `to_maxtext` Step 2 can't run on one tpu7x host for most configurations.** It keeps the whole
  converted state in host RAM, and scanned conversions also need the source staged in RAM. `safetensor` loads the
  same checkpoints directly from GCS (issues U-4, U-5).

## 2. Environment and meshes

| Item | Value |
| :--- | :--- |
| Accelerator | TPU v7x (`tpu7x-standard-4t`), topology `2x2x1`: 4 chips = 8 TPU cores per slice, 101.7 GB HBM per core (813.8 GB per slice) |
| Host | 1,014.5 GB RAM (986.7 GB allocatable to pods), 224 vCPUs, ~100 GB boot disk; staging uses a RAM-backed `/ramdisk` |
| 1 slice | 1 process, mesh `fsdp=8` |
| 2 slices | 2 processes over DCN (`jax.distributed`): ≤47B `dcn_data_parallelism=2` (mesh `data=2 × fsdp=8`); ≥70B `dcn_fsdp_parallelism=2` (`fsdp=16`) |
| 4 slices | `dcn_fsdp_parallelism=4` (`fsdp=32`), used for `deepseek3-671b` and scanned `qwen3.5-397b-a17b` |
| Software | Container `yaoyuchen-runner:ds41-2026-10-22-16-03-48` (JAX 0.11.1); MaxText `src/` = `origin/main` + #5706, #5705, #5828, plus the bootstrap patches |
| `to_maxtext` Step 2 | One host's CPU, `--lazy_load_tensors=True` (gemma3: eager), 16 simulated CPU devices for the Orbax save |

## 3. Sub-phase breakdown

### 3.1 `to_maxtext` Step 2 (conversion on one host)

| Model | Scan | Init | Transform + shard | Orbax save to GCS | Total (s) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `qwen3-0.6b` | scanned / unscanned | 0.1 / 1.9 | 4.7 / 47.2 | 12.3 / 15.6 | 19.4 / 66.8 |
| `llama3.1-8b` | scanned / unscanned | 0.1 / 2.5 | 49.2 / 62.5 | 38.0 / 38.2 | 89.6 / 105.3 |
| `qwen3-14b` | scanned / unscanned | 0.1 / 3.6 | 104.3 / 101.0 | 39.1 / 37.9 | 145.8 / 144.7 |
| `qwen3-32b` | scanned / unscanned | 0.1 / 7.8 | 322.2 / 197.9 | 57.1 / – | 381.5 / 208.0 |
| `qwen3-30b-a3b` | scanned / unscanned | 0.1 / 4.1 | 74.2 / 185.7 | 74.8 / 41.7 | 151.6 / 233.6 |
| `mixtral-8x7b` | scanned / unscanned | 0.1 / 2.4 | 82.4 / 147.8 | 49.1 / 56.0 | 134.1 / 208.3 |
| `llama3.1-70b` | scanned / unscanned | 0.1 / 12.3 | 689.9 / 281.9 | 87.5 / 101.3 | 780.3 / 398.1 |
| `gpt-oss-120b` | scanned / unscanned | 0.2 / 4.9 | 441.7 / 417.5 | 136.6 / 192.5 | 581.4 / 617.8 |

The full list is in the CSV (`to_maxtext_2a/2b/2c` columns).
- **Where the time goes:** transform + shard dominates, and the Orbax save is 12–193 s.
- **Scanned vs unscanned for dense models:** scanned costs more for large dense models (`llama3.1-70b`: 690 s vs 282 s), because every layer is stacked into each leaf.
- **Peak host RAM in Step 2** reaches ~2.4× the checkpoint size (`gpt-oss-120b`: 560 GB). That is why >200B conversions don't fit (U-4).

### 3.2 `safetensor` Step 2: load vs transform

`safetensor_2b_load_sharded_hf_s` (GCS reads into HBM) and `safetensor_2c_transform_hf_to_mt_s` (on-device layout
conversion) are in the CSV.
- **Load dominates for unscanned models:** e.g., `llama3.1-70b` unscanned · 1: 159 s load + 9 s transform.
- **Scanned models spend more time in the transform**, because layers and experts are stacked on device:
  - `llama3.1-70b` scanned · 1: 165 + 34 s.
  - `gpt-oss-20b` scanned · 1: 49 + 79 s.
- **Two transforms stand out:**
  - `qwen3-14b` scanned · 1 slice: ~290 s, reproduced in a re-run, vs 51 s on 2 slices.
  - `qwen3.5-397b-a17b` scanned · 4 slices: 1,139 s (composite `(wi_0, wi_1)` split, U-1).

## 4. Scaling from 1 to 2 slices

- **`safetensor`, ≥70B models** (`dcn_fsdp_parallelism=2`): 2 slices are faster.
  - `llama3.1-70b`: 199 → 105 s scanned, 168 → 114 s unscanned.
  - `gpt-oss-120b` unscanned: 447 → 391 s.
- **`safetensor`, ≤47B models** (`dcn_data_parallelism=2`, so each slice loads a full replica): 2 slices are often
  slower. For example:
  - `deepseek2-16b`: 75 → 125 s.
  - `qwen3-30b-a3b` unscanned: 111 → 354 s (load 96 → 333 s).
- **`to_maxtext` Step 3:** roughly flat for unscanned. Scanned MoE got slower on 2 slices (`mixtral-8x7b` 534 → 793 s).

## 5. Open questions

1. **Scanned MoE Orbax restores on tpu7x** (`mixtral-8x7b`, `qwen3-30b-a3b`, `gpt-oss-120b`) are 3–6× slower than on v5p.
   - **Candidate causes:** the `checkpoint_storage_concurrent_gb` settings (45 GB for `mixtral-8x7b`/`gpt-oss-120b`, 20 GB default), and 8 devices per host instead of 4.
   - **Next step:** profile a restore with `jax.profiler` and vary `concurrent_gb`.
2. **The `qwen3-14b` scanned 1-slice `safetensor` transform** (~290 s) is not explained.
3. **`dcn_data_parallelism` for small models on 2 slices** doubles the data read without speeding up the load. For pure load-time comparisons, `dcn_fsdp_parallelism` may be the better default.
