# DeepSeek-V3 671B FP8 training recipes for TPU7x (Ironwood)

These recipes pre-train DeepSeek-V3 671B with FP8 on TPU7x, using FSDP + expert parallelism (EP). They train on the C4 dataset with sequence length 4,096 and a global batch of 16,384 sequences, until the eval loss reaches 3.6. Each recipe is launched with [Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) (`gcluster`) through `maxtext_ctk_runner`.

## Recipes

| Recipe | Chips | Topology | Mesh | Per-device batch |
|---|---|---|---|---|
| `8k` | 8,192 (16,384 TensorCores) | `tpu7x` `8x32x32`, 1 slice | FSDP 1024 × EP 16 | 1 |

2k and 4k recipes are coming.

Main features of the recipes:

- **FP8:** `fp8_full` with qwix, static per-tensor scales, quantized FSDP weight all-gathers and EP token all-gathers, and a fused FP8 MoE combine backward.
- **MoE:** ring-of-experts, ragged sort on the TensorCore with 3D GMM v2 kernels, and a dropless first phase.
- **Attention:** tokamax splash attention with causal diagonal block skipping and `SEQ_MINOR` q/k layouts.
- **GMM v2:** tile sizes tuned for this shape.

## Files

- `config.py`: one `MaxTextModel` per recipe (MaxText overrides + XLA flags), plus `RECIPES`, which maps each recipe name to its TPU7x slice (device type, topology, compute type, number of slices).
- `xla_flags.py`: the XLA / libtpu flags. Flags shared by all scales are grouped by purpose; scale-dependent ones are separate constants.
- `benchmark.py`: launches a recipe with `maxtext_ctk_runner`.

## Prerequisites

- **Tools:** `gcluster` and `kubectl` (≥ 1.31) on `PATH`, and `gcloud` authenticated for the project. See [Run MaxText via Cluster Toolkit](../../docs/run_maxtext/run_maxtext_via_cluster_toolkit.md).
- **Cluster:** a GKE cluster with a TPU7x slice of the recipe's topology.
- **Image:** a MaxText Docker image that includes the features above, passed as `--docker_image`.
- **Dataset:** the preprocessed C4 dataset (`llama-3-1-8b-preprocessed-c4-dataset`) from the [MLCommons LLM MoE pre-training reference](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo), tokenized and converted to TFDS shards that MaxText reads with `dataset_type=c4_mlperf`.
- **Checkpoint:** a DeepSeek-V3 671B checkpoint in MaxText format. See [Run DeepSeek](../../tests/end_to_end/tpu/deepseek/Run_DeepSeek.md) for checkpoint conversion. For fast loading, pre-shard it for the recipe's mesh (flat MoE layout; FSDP 1024 × EP 16 for `8k`).

## Run

```bash
python3 -m benchmarks.deepseek3_671b_tpu7x.benchmark \
  --recipe=8k \
  --project=<project> \
  --location=<zone or region> \
  --cluster_name=<cluster> \
  --docker_image=<maxtext image> \
  --base_output_directory=gs://<bucket>/<dir> \
  --dataset_path=gs://<bucket>/<c4 tfds dataset>/ \
  --load_parameters_path=gs://<bucket>/<checkpoint>/0/items
```

Optional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--run_name` | `<user>-ds3-<recipe>-<MMDDHHMM>` | Workload name (at most 28 characters, a valid DNS label) |
| `--num_steps` | `100` | Maximum training steps. The run stops earlier once it reaches the target eval loss. |
| `--priority` | `high` | Workload priority |
| `--max_restarts` | `1` | JobSet max restarts |
| `--dry_run` | off | Print the `gcluster job submit` command without submitting it |

Use `--dry_run` to check the exact command first: the slice, all MaxText overrides, and the XLA flags in `LIBTPU_INIT_ARGS`.

Evaluation runs on the C4 validation set every step from step 42.

## Adding a recipe

1. In `config.py`, define a new `MaxTextModel` with `_add_to_model_dictionary(tpu7x_deepseek3_model_dict, ...)`. Build its `xla_flags` from the constants in `xla_flags.py`.
2. Add a `RECIPES` entry with its `Tpu7xRecipe(model=..., device_type="tpu7x-<TensorCores>", topology=...)`.
3. Check it with `benchmark.py --recipe=<name> --dry_run`.
