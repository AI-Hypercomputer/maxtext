# MoE Synthetic Routing Benchmark Harness

Standalone benchmark wrapper that feeds host-generated `forced_routed_experts`
batches (`[batch, seq, num_layers, top_k]` `int32`) into MaxText pre-training
without modifying any core `src/maxtext/` modules or config schemas.

## Supported Models & Decoder Requirements

`forced_routed_experts` (router replay) requires MaxText's pure-NNX decoder
(default in `base.yml`, without pipeline parallelism and with `mtp_num_layers=0`)
and a homogeneous MoE `decoder_block` in `FORCED_ROUTING_SUPPORTED_DECODER_BLOCKS`:
- `qwen3_5` (e.g. `model_name=qwen3.5-397b-a17b`)
- `mixtral` (e.g. `model_name=mixtral-8x7b`)
- `gemma4` (unscanned only: `scan_layers=false`)

## Usage

Pass any `synthetic_routing*` options (either `--key=val` or `key=val`) alongside
standard MaxText configuration arguments. For example, on a single 8-chip TPU
v6e host (`EP=8`, scaled 4-layer `qwen3.5-397b-a17b`, ragged all-to-all):

```bash
python3 -m benchmarks.moe_synthetic_routing.train src/maxtext/configs/base.yml \
  run_name=moe_routing_bench base_output_directory=/tmp/maxtext_bench \
  dataset_type=synthetic steps=10 log_period=1 enable_checkpointing=false \
  model_name=qwen3.5-397b-a17b override_model_config=true use_multimodal=false \
  base_emb_dim=1024 base_num_query_heads=4 base_num_kv_heads=2 head_dim=256 \
  gdn_num_key_heads=4 gdn_num_value_heads=8 \
  base_moe_mlp_dim=256 base_mlp_dim=256 vocab_size=32768 \
  base_num_decoder_layers=4 \
  per_device_batch_size=2 max_target_length=1024 \
  ici_expert_parallelism=8 ici_fsdp_parallelism=1 use_ring_of_experts=false \
  sparse_matmul=true megablox=true ragged_buffer_factor=1.5 \
  --synthetic_routing=lognormal \
  --synthetic_routing_sigmas=0.9 \
  --synthetic_routing_row_taus=1.3 \
  --synthetic_routing_row_len=512
```

For an ideal balanced baseline where every expert receives the exact mean token
count (`required_rbf = 1.0`):

```bash
python3 -m benchmarks.moe_synthetic_routing.train src/maxtext/configs/base.yml \
  ... \
  --synthetic_routing=balanced
```

When benchmarking hot-expert replication (`moe_replica_planner`), pass
`--synthetic_routing_overhead_control=true` to run the overhead-control ablation
arm (moves planned replica weights over ICI each step while keeping token routing
on home experts only).

## Options

| Option | Default | Description |
|---|---|---|
| `synthetic_routing` | `"none"` | `"none"`, `"lognormal"`, or `"balanced"`. |
| `synthetic_routing_seed` | `0` | Base RNG seed for synthetic expert draws. |
| `synthetic_routing_pool_size` | `8` | Number of pre-generated host routing batches to cycle through. |
| `synthetic_routing_sigma` | `1.0` | Default log-normal expert popularity spread (`sigma`). |
| `synthetic_routing_sigmas` | `""` | Comma-separated per-layer `sigma` values (overrides `sigma`). |
| `synthetic_routing_row_tau` | `0.0` | Per-document expert affinity jitter (`0.0` disables). |
| `synthetic_routing_row_taus` | `""` | Comma-separated per-layer `row_tau` values (overrides `row_tau`). |
| `synthetic_routing_row_len` | `0` | Mean exponential document length in tokens (`0` = one document per sequence). |
| `synthetic_routing_profile` | `""` | Optional `.npz` path (`local`, `gs://`, or `/cns/`) with `log_profile` `[P, E]`. |
| `synthetic_routing_overhead_control` | `false` | Benchmark ablation control: moves planned expert replicas but routes no tokens to them. |
