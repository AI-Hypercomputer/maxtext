# Synthetic MoE Routing Benchmark Harness

Standalone benchmark wrapper that feeds host-generated `forced_routed_experts`
batches (`[batch, seq, num_layers, top_k]` `int32`) into MaxText pre-training
without modifying any core `src/maxtext/` modules or config schemas.

## Usage

Pass any `synthetic_routing*` options (either `--key=val` or `key=val`) alongside
standard MaxText configuration arguments:

```bash
python3 -m benchmarks.moe_synthetic_routing.train src/maxtext/configs/base.yml \
  run_name=moe_routing_bench base_output_directory=/tmp/maxtext_bench \
  dataset_type=synthetic steps=20 \
  model_name=qwen3.5-397b-a17b override_model_config=true use_multimodal=false \
  ici_expert_parallelism=8 sparse_matmul=true megablox=true \
  --synthetic_routing=lognormal \
  --synthetic_routing_sigmas=0.9 \
  --synthetic_routing_row_taus=1.3 \
  --synthetic_routing_row_len=768 \
  --synthetic_routing_pool_size=8
```

For an ideal balanced baseline where every expert receives the exact mean token
count:

```bash
python3 -m benchmarks.moe_synthetic_routing.train src/maxtext/configs/base.yml \
  ... \
  --synthetic_routing=balanced
```

## Options

| Option | Default | Description |
|---|---|---|
| `synthetic_routing` | `none` | Routing mode: `none` / `off`, `lognormal`, or `balanced`. |
| `synthetic_routing_sigma` | `0.25` | Per-layer lognormal expert popularity std (used when `sigmas` is empty). |
| `synthetic_routing_sigmas` | `""` | Comma-separated per-layer sigmas (interpolated across layers by depth). |
| `synthetic_routing_pool_size` | `4` | Number of distinct host batches generated once and cycled on device. |
| `synthetic_routing_seed` | `0` | RNG seed for persistent popularity and token draws. |
| `synthetic_routing_row_tau` | `0.0` | Per-document log-popularity std (models sequence-level expert clustering). |
| `synthetic_routing_row_taus` | `""` | Comma-separated per-layer `row_tau` values (overrides `row_tau`). |
| `synthetic_routing_row_len` | `0` | Mean exponential document length in tokens (`0` = one document per sequence). |
| `synthetic_routing_profile` | `""` | Optional `.npz` path (`local`, `gs://`, or `/cns/`) with `log_profile` `[P, E]`. |
