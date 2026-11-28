# FLOP Calculation, Training Benchmarking & Roofline Analysis

After functional correctness is verified, update per-step FLOP accounting, run ahead-of-time TPU memory and static HLO gates, execute a steady-state TPU pre-training benchmark, and compare measured throughput/HBM against an analytical roofline model.

---

## 1. FLOP Calculation (`src/maxtext/utils/maxtext_utils.py`)

Update `calculate_tflops_training_per_device` (and block helpers in `src/maxtext/utils/maxtext_utils.py`) if the new `decoder_block` introduces custom layer math so logged `Total TFLOPs`, `TFLOP/s/device`, and MFU are accurate:

- **Attention FLOPs**: Distinguish local sliding-window layers (`min(seq_len, sliding_window_size)` causal window) from global causal layers (`0.5 * seq_len^2`), GQA/MLA/linear attention, and any extra projections (e.g., attention output-gate matmuls or custom `attention_output_dim` up-projections).
- **FFN / MoE FLOPs**: Count router gate (`2 * B * S * emb_dim * num_experts`), width bottlenecks (`moe_expert_input_dim` pre/post projections and `calculate_ffn_mamtul_tflops_per_device(..., in_dim=config.moe_expert_input_dim)`), active routed experts (`* num_experts_per_tok`), `shared_experts`, and dense vs MoE layer counts (`get_dense_moe_layers`).
- **Unembedding & Aux Blocks**: Include untied/tied LM head projection (`2 * B * S * emb_dim * vocab_size`; token embedding lookup is a gather = `0` FLOPs), MTP (`mtp_num_layers`), PLE, or multimodal encoders, multiplied by `3` (1 fwd + 2 bwd) and `gradient_accumulation_steps`.
- Add unit test coverage verifying the new `decoder_block`'s reported `Total TFLOPs` and `% learnable weight` / `% attention` split.

---

## 2. Ahead-of-Time TPU HBM Preflight & Static HLO Gate (`aot_preflight.py`, `check_hlo_perf.py`)

Before launching multi-host TPU jobs, verify per-device HBM headroom and static HLO collective overlap on CPU:

1. **Multi-Topology Forward HBM Preflight (`scripts/aot_preflight.py`)**:
   - Infers usable per-device HBM automatically from `TOPOLOGY_HBM_GIB` (`v4`, `v5e`/`v5litepod`, `v5p`, `v6e`, `tpu7x`) or accepts `--hbm-limit-gib=<GiB>`, defaulting `compile_topology_num_slices=1` unless explicitly overridden:
   ```bash
   PYTHONPATH=$PWD/src python3 .agents/skills/model-bringup/scripts/aot_preflight.py \
     src/maxtext/configs/base.yml \
     model_name={model_name} \
     compile_topology=v5p-128 \
     --seqs=4096,8192
   ```
2. **Static HLO Overlap, FLOP Inflation & Memory Gate (`scripts/check_hlo_perf.py`)**:
   - Audits `train_compile` HLO text dumps and AOT memory stats for naive/calibrated HBM ceilings, rematerialization FLOP inflation, synchronous collective stalls vs. overlapped `-start`/`-done` pairs, accidental `S(5)` host-memory spills, and fixed-seed loss trajectory parity (`check_loss_trajectory`):
   ```bash
   python3 .agents/skills/model-bringup/scripts/check_hlo_perf.py \
     --summary-json /tmp/{run_name}_aot_summary.json \
     --hlo-path /tmp/{run_name}_after_optimizations.txt \
     --hbm-ceiling-gib 92.0 \
     --forbid-host-offload
   ```

---

## 3. End-to-End Pre-Training Benchmark on TPU (`extract_run_metrics.py`)

Run a synthetic-data steady-state training benchmark on the target TPU slice with XPlane profiling and HLO memory reporting enabled:

```bash
python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name={model_name} \
  run_name={run_name} \
  base_output_directory={gcs_output_dir} \
  dataset_type=synthetic \
  steps=15 \
  max_target_length=4096 \
  per_device_batch_size=1.0 \
  remat_policy=full \
  ici_fsdp_parallelism={FSDP} \
  ici_tensor_parallelism={TP} \
  scan_layers=true \
  opt_type=sgd \
  weight_dtype=bfloat16 \
  grad_dtype=bfloat16 \
  enable_checkpointing=false \
  profiler=xplane \
  skip_first_n_steps_for_profiler=5 \
  profiler_steps=3 \
  log_period=1
```

Extract steady-state step time, `TFLOP/s/device`, MFU %, loss trajectory, and `grad_norm` diagnostics deterministically from the worker log (and verify prose numbers before publishing in `{model_name}_handover.md`):

```bash
python3 .agents/skills/model-bringup/scripts/extract_run_metrics.py \
  /tmp/{run_name}_worker0.log \
  --steady-start-step 10 \
  --steady-end-step 14 \
  --peak-tflops-per-device 459.0
```

Record the following benchmark artifacts and steady-state metrics (e.g., steps 10–14):
- **Configuration**: Hardware topology & TensorCore count, precision, `max_target_length`, `per_device_batch_size`, global batch tokens, and `(FSDP, TP, EP)` sharding (plus any `LIBTPU_INIT_ARGS` / SparseCore offload flags and `shard_map` / per-layer remat optimizations needed to fit HBM).
- **Memory & HLO Artifacts**: Compile-time `Total estimated memory size` (`output`, `temp`, `argument` GB vs device HBM capacity), `after_optimizations-memory-usage-report.txt`, HLO dumps (`before_optimizations`, `after_optimizations`, `buffer-assignment`), and XPlane traces (`.xplane.pb`, `.trace.json.gz`).
- **Throughput**: Step `Total TFLOPs` (and weight/attention % split), steady-state `seconds/step`, `TFLOP/s/device` (and MFU % vs hardware `bf16` peak), `Tokens/s/device`, and cluster `Tokens/s`.

---

## 4. Roofline Analysis & Measured Comparison

Generate a deterministic analytical roofline report (e.g., via `roofline_cli.py` with the `jasmine` backend; do not hand-estimate MFU/HFU figures) matching the exact model architecture (`config.json` / `modeling_{model_name}.py`) and benchmarked `(DP, FSDP, TP, EP, B, S, bf16, remat_policy)` workload (plus any target production topology such as `v7x_8x8x8` alongside the measured slice such as `v7x_4x4x4`).

Include five sections in the roofline report:
1. **Model Summary**: Total & active parameters, layer counts/types (sliding-window vs global attention, dense vs latent/routed MoE, shared experts), norms, and head/soft-cap specs.
2. **Modeling Setup & Limitations**: Simulator backend/components, sharding & batch setup per topology, and **explicit unmodeled ops / caveats** (e.g., zero efficiency derates making step times lower bounds; unmodeled attention output-gate projections, QK/output/post norms, or residual gates; simulator HBM model differences).
3. **Aggregate Roofline Results**: Per-kernel roofline step time (ms), total `Compute / Memory / Comm` time (ms), overall bound (`Compute` vs `Communication` vs `Memory`), primary bottleneck block & % of step, MegaKernel unavoidable-memory bound, and simulator HBM budget.
4. **Per-Block Breakdown Table**: Layers, `Compute (ms)`, `Memory (ms)`, `Comm (ms)`, and `Block step (ms)` for each block (Token Embedding, Sliding-Window Attention, GQA/MLA Attention, Dense/Latent/Routed MoE FFN, RMSNorm, LM Head + Cross-Entropy).
5. **Measured MaxText vs Roofline Comparison**:
   - Side-by-side table on the measured topology comparing `Step time`, `Model TFLOP/s per TensorCore` (MFU %), `Tokens/s per TensorCore`, and **MaxText compile-time HBM estimate vs simulator HBM budget** (rely on MaxText's compiled HBM estimate—not the simulator's—for capacity planning if they disagree).
   - Gap diagnosis identifying why measured step time exceeds the roofline lower bound (e.g., dropless `ragged_dot` across all experts, serialized FSDP weight all-gathers, per-layer MoE rematerialization under `remat_policy=full`), cross-referenced with the captured XPlane profile.
