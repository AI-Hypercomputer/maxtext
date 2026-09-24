# Layer-fused dual-pipe scheduling in normal training

`gradient_accumulation_schedule=dual_pipe` opts the normal
`maxtext.trainers.pre_train.train` entry point into a layer-level fused 1F1B
gradient-accumulation schedule. The default is `serial`, which keeps the existing
training implementation. The launcher does not need a different Python module.

For three microbatches, the decoder schedule is `F0, B0F1, B1F2, B2`.
Each combined scan iteration contains backward for old layer `L-1-k` and forward
for next layer `k`, with independent activation carries. Both paths compile
inside the existing jitted `train_step`. This exposes overlap opportunities; it
does not guarantee GPU concurrency or implement distributed pipeline stages.

This path uses the actual token embedding, final normalization, output head,
and masked causal-LM loss. Output
cotangents come from that loss. Embedding/head and decoder gradients are all
accumulated, normalized by the same valid-token count as normal GA, and returned
in the original parameter tree/scan-axis layout. Existing clipping, optimizer,
checkpointing, evaluation, data loading, and profiling remain in normal training.
Parameters do not update between microbatches.

## Code layout

- `train.py` selects the schedule and keeps the normal optimizer/training loop.
  Its `loss_from_logits` helper shares the existing masked loss calculation;
  loss and z-loss normalization remain with the callers.
- `experimental/dense_training_schedule.py` implements the fused layer scans.
- `experimental/dense_training_nnx.py` adapts scanned Llama or DeepSeek layers and the
  embedding/head to those scans, then restores the full model gradient tree.

There is no separate benchmark entry point.

## Initial supported scope

- NNX dense Llama (`decoder_block=llama2`), including `model_name=llama3-8b`;
  or DeepSeek V3 (`decoder_block=deepseek`) with an initial dense stack and
  a TE-MoE stack, as described below.
- `scan_layers=true`, homogeneous layers within each stack, and `shard_mode=auto`.
- `remat_policy=none` or `full`; `full` uses `nothing_saveable`.
- `dropout_rate=0`, with model parameters stored in FP32/BF16/FP16.
  Llama remains unquantized. DeepSeek allows `quantization=te_no_quant` or
  `te_fp8_currentscaling`, and `te_gmm_quantization=te_no_quant` or `te_mxfp8`.
  These recipes derive scales per invocation; no persistent quantizer state
  is carried or updated by the adapter. Delayed scaling and NVFP4 remain excluded.
- DeepSeek router bias may be enabled with `routed_bias_update_rate=0` and
  `load_balance_loss_weight=0`; it remains read-only, non-parameter layer state.
- `num_vocab_tiling=1`, ordinary causal-LM loss, no mutable internal metrics.
- No non-TE MoE, LoRA, multimodal, MTP/indexer, pipeline parallelism, host offload,
  Tunix accumulation, DiLoCo, or `shard_optimizer_over_data` (ZeRO-1).

Unsupported modes fail explicitly. Full remat recomputes the old microbatch's
forward operations during backward; those are distinct from the next forward
branch. Old and new residuals can coexist in the combined loop, so peak memory
is not guaranteed to be one microbatch's saved activations.

## One-node Llama 3 8B with maxtext-launcher

Set the paths and cluster identifier for your deployment. Keep site-specific
paths, accounts, partitions, and container settings in your private launcher
configuration, not in this repository.

```bash
export LAUNCHER_DIR=/path/to/maxtext-launcher
export MAXTEXT_DIR=/path/to/maxtext
export CLUSTER=your-cluster

python3 "${LAUNCHER_DIR}/launcher.py" llama3-8b \
  --cluster "${CLUSTER}" --nodes 1 \
  --code-dir "${MAXTEXT_DIR}" \
  --maxtext-arg gradient_accumulation_schedule=dual_pipe \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg scan_layers=true \
  --maxtext-arg remat_policy=full \
  --tag llama8b-dualpipe-ga3-full-remat \
  --dry-run
```

Inspect the generated scripts, then rerun without `--dry-run` to submit. Leave
their Python entry point as `maxtext.trainers.pre_train.train`. No generated
script editing is needed. Configure the host checkout and container mount in
your private launcher YAML, for example:

```yaml
code_dir: "/path/to/maxtext"
code_mount: "/workspace/maxtext"
maxtext_path: "/workspace/maxtext"
```

The command leaves batch size, sequence length, PGLE, command buffers, and
profiling at the launcher's configured values. Inspect the resolved preset and
ensure that the number of training steps covers your profiling window. This
adapter requires `remat_policy=full` or `none`, regardless of the preset default.

For comparison, change the schedule to `serial` and use a different tag. Keep
all other options identical. Normal launcher profiling/HLO paths apply to both
runs. Look for `dual_pipe/steady_Bi_Fnext/combined_bf_layers` containing both
`backward` and `forward` operations. Dense FSDP has all-gather/reduce-scatter
communication, not TE-MoE dispatch/combine A2As.

The container must support the mounted MaxText checkout and provide
`jax.fwd_and_bwd`. Focused CPU tests cover true-loss gradients, NNX bookkeeping,
loss masking, and scan structure. HLO interleaving and asynchronous collective
windows are not proof of measured GPU overlap or numerical equivalence. The
training path does not add runtime numerical comparisons or NaN/Inf assertions.

## One-node reduced-layer DeepSeek V3 with TE-MoE

Two matched scripts inherit the regular `deepseek-v3-671b` launcher preset and
use the normal training entry point, without creating another model config.
Use the same private launcher configuration for both runs.

```bash
export LAUNCHER_DIR=/path/to/maxtext-launcher
export CLUSTER=your-cluster
cd /path/to/maxtext
bash scripts/experimental/run_deepseek_small_serial.sh --dry-run
bash scripts/experimental/run_deepseek_small_dualpipe.sh --dry-run
```

Inspect the generated scripts first. Remove `--dry-run` to submit a test.
Use a compatible JAX/Transformer Engine environment. Successful execution alone
does not establish numerical equivalence or a schedule performance improvement.
Other launcher options can be appended unchanged, for example
`--profiler nsys` or `--container IMAGE`. Both scripts pass `--no-pgle` for a
matched comparison; profiling otherwise follows your launcher configuration.
The scripts require `LAUNCHER_DIR` and `CLUSTER`. The MaxText code mount is
derived from the script's own checkout. Allocation details come from your
private launcher configuration.

The commands differ only in the schedule and tag. Their only training overrides
relative to the regular preset are:

- One node, four one-GPU processes, FSDP=2, EP=2; every DCN axis is 1.
- Six layers: `base_num_decoder_layers=6`, with `first_num_dense_layers=1`.
- Sixteen routed experts: `num_experts=16`.
- GA=3 and the selected accumulation schedule.
- Full rematerialization (`remat_policy=full`) instead of the preset's custom policy.
- Explicit quantization selection: `quantization=te_fp8_currentscaling` for
  supported dense TE GEMMs and `te_gmm_quantization=te_mxfp8` for expert GEMMs.
  This does not change parameter or optimizer storage dtypes.
- `override_model_config=true` so the three model-size overrides take effect.
- PGLE disabled (`--no-pgle`) in both scripts.

Everything else comes from your launcher preset and DeepSeek model configuration.
Check the resolved dtype, routing, batch size, sequence length, capacity, and
profiling settings before submitting; they must satisfy the supported scope
above. The preset's per-activation remat/offload settings are not edited, but
selecting `remat_policy=full` replaces its custom remat/offload policy.

To return either script to the unquantized baseline, append
`--maxtext-arg quantization=te_no_quant --maxtext-arg te_gmm_quantization=te_no_quant`.
Use the canonical recipe strings above: `mxfp8` and `te_fp8_current_scaling` are
not valid MaxText values. No TE kernels or scheduling logic are changed by the
recipe selection; full remat also recomputes any needed quantization and amax
reductions inside backward.

The combined decoder body pairs old/new logical layers as follows:

```text
old B(layer 5, MoE)   + next F(layer 0, dense)
old B(layer 4, MoE)   + next F(layer 1, MoE)
old B(layer 3, MoE)   + next F(layer 2, MoE)
old B(layer 2, MoE)   + next F(layer 3, MoE)
old B(layer 1, MoE)   + next F(layer 4, MoE)
old B(layer 0, dense) + next F(layer 5, MoE)
```

The mixed dense/MoE boundaries need distinct scan segments because their
parameter and saved-residual trees differ. Within each segment, backward and
next forward still share the same compiled loop body. Rematerialized old-forward
work inside backward is distinct from the next-microbatch forward branch.

The adapter allows the inherited `routed_bias=true` with its update rate at zero.
The existing per-layer state path carries the bias without parameter gradients
or optimizer updates. Nonzero bias updates, auxiliary load-balancing loss, and
stateful quantization recipes remain unsupported. No routing settings or kernels
were changed to allow frozen bias. When inspecting the optimized HLO, look for
next-forward EP combine windows spanning backward grouped GEMMs, and backward
EP combine windows spanning next-forward dense GEMMs. Confirm actual overlap
with a GPU trace; some dispatch start/done pairs may remain adjacent.

Use a compatible container and four one-GPU processes; normal MaxText setup
performs EP bootstrap. CPU tests cannot validate TE communication,
routing-handle lifetime under scan/remat,
or pinned-host residual handling in the combined scans.
