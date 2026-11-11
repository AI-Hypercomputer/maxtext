# OLMo 3.5 Multimodal & 256K Mid-Training on Cloud TPU

End-to-end recipes, checkpoint converters, data pipelines, and parity fixes for running AI2's 3-stage OLMo 3.5 mid-training pipeline (`8K Bridge` -> `8K Joint` -> `256K Unified Text MT + Long-Context + Vision`) in MaxText on Cloud TPU (`v4`, `v5p`, `v7x`).

Supports both:
* **OLMo 3.5 Small (Hero, `olmo35-small-e256k8-sh2x`)**: `72.58B` total / `4.12B` active (`40L` `7:1` KDA:FA, `d_model=1536`, `E=256`, `K=8`) + `25L` SigLIP2-SO400M/14-378 (`414M`) + SwiGLU Connector (`d_ff=9728`, `184M`).
* **OLMo 3.5 Tiny (`olmo35-tiny` / `16L`, `d_model=1024`, `E=512`, `K=16`)**: `12.92B` total / `0.79B` active reference architecture (`bridge_8k`, `joint_8k`, `mt_lc_256k`).

---

## 1. 3-Stage Mid-Training Pipeline Overview

| Stage | Recipe / Config | Seq Len | Global Batch | Trainable Parameters | Learning Rate & Schedule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 8K Bridge** | `olmo35_small_vision_align_8k.yml` + Bridge overrides | `8,192` | `1.05M`–`4.19M` tok (`64` crops/seq) | Connector + `<im_*>` rows `[100278..100283]` (`ViT`, `LM`, `logits_dense`, `expert_bias` frozen) | `2.0e-4` cosine to `0.1x` over `250` steps, held flat steps `250..500` (`warmup_steps_fraction=0.4`) |
| **2. 8K Joint** | `src/maxtext/configs/olmo/olmo35_small_vision_align_8k.yml` | `8,192` | `1.05M`–`4.19M` tok (`64` crops/seq) | `LM` + `25L ViT` + `Connector` + `<im_*>` rows `[100278..100283]` (`logits_dense` frozen; `0.35` text / `0.65` vision loss group weights) | `LM` `1.0e-4` WSD linear to `0.0` (`3,500` steps); `Connector` `2.0e-5` (`w=50`), `ViT` `2.0e-6` (`w=100`), `<im_*>` `2.0e-5` (`w=50`, same schedule group as `Connector`) cosine to `0.1x` |
| **3. 256K Unified MT+LC+MM** | `src/maxtext/configs/olmo/olmo35_small_midtrain_256k.yml` (`use_multimodal=true`) | `262,144` | `16.78M` or `25.17M` tok (`1,024`–`2,048` crops/seq) | All parameters unfrozen (`frozen_parameters_mask=[]`, `olmo_trainable_embedding_row_range=[]`) | `LM` `1.5e-4` (Hero) / `4.4e-4` (Tiny) WSD linear to `0.0` (`2,000`-step warmup); `Connector` `2.0x LM`, `ViT` `0.1x LM` |

---

## 2. Key Parity & System Fixes on `agagik-olmo35-midtraining`

All fixes below are integrated directly into this branch:

1. **Bidirectional Checkpoint Conversion (`src/maxtext/checkpoint_conversion/convert_olmo35_multimodal_checkpoint.py`)**:
   * Converts `OLMo-core` PyTorch Distributed Checkpoints (`DCP`) <-> MaxText `Orbax` (`scan_layers=true` `8`-layer scan cycles or unscanned) with `0.0` max absolute round-trip diff across all `LM`, `25L SigLIP2`, and `SwiGLU Connector` tensors.
   * Supports `--mode bridge_init` from either an `OLMo-core` DCP checkpoint or an existing MaxText Orbax checkpoint (such as Hero `step 162974`), loading `google/siglip2-so400m-patch14-384` (revision `e8e4872`, truncating to `25L` matching `OLMo-core`), initializing the `9,728` SwiGLU Connector, and resetting the `6` image special token rows `[100278..100283]`.
2. **Vision Encoder & Connector Numerics (`src/maxtext/models/olmoe3_vision.py`, `src/maxtext/models/olmoe3.py`)**:
   * Matches `OLMo-core`'s 25-layer SigLIP2-SO400M/14-378 (`eps=1e-6`, `tanh` GELU) and `[-2, -9]` feature concatenation + attention-pooled SwiGLU Connector.
   * Scatter-injects pooled visual embeddings at `<im_patch>` (`token_id=100280`), masks padding tokens (`padding_input_token_id=100257`), and isolates packed documents across both KDA recurrence and Full Attention via `inputs_segmentation`.
   * Shards vision crops across Context Parallel ranks (`olmo_vit_cp_shard_crops=true`) with gradient checkpointing (`olmo_vit_remat_blocks=true`) and configurable crop micro-batching (`olmo_vit_crop_chunk_size=16`).
3. **Loss, Router & Per-Component Optimizer Parity (`src/maxtext/trainers/pre_train/train.py`, `src/maxtext/optimizers/optimizers.py`)**:
   * Normalizes token CE + `z_loss_multiplier=1e-5` by the global `loss_weights` sum (`olmo_loss_denominator_counts_masked_instances=true`, `static_loss_cotangent=true`) and scales MoE auxiliary load-balance loss (`0.01`) by valid non-padding `router_token_mask` tokens.
   * Supports `8K Joint` domain loss-group normalization (`olmo_text_loss_group_weight=0.35`, `olmo_vision_loss_group_weight=0.65`).
   * Implements per-component learning rate scales and schedules (`olmo_vision_lr_scale`, `olmo_connector_lr_scale`, `olmo_embedding_lr_scale`, `olmo_connector_warmup_steps`, `olmo_vision_lr_cosine_alpha_f`, `olmo_hold_final_lr`), per-component global gradient norm clipping (`olmo_per_component_grad_clip=true`), embedding row-range gradient masking (`olmo_trainable_embedding_row_range=[100278, 100284]`), and `load_balance_loss_weight=0.0` during Stage 1 Bridge (keeping router `expert_bias` identical to pretraining, matching `OLMo-core`'s `restore_pretraining_router_lb=True`).
4. **Multimodal & 256K Chunked Text Data Pipelines (`src/maxtext/input_pipeline/olmo_mm_data_processing.py`, `tools/data_generation/export_olmo_mm_batches.py`, `src/maxtext/input_pipeline/olmo_data_grain.py`)**:
   * `dataset_type=olmo_mm` supports both pre-exported `.npz` global batches (`olmo_mm_batches_dir`, for validation and short runs on TPU pods without local media mounts) and live `OLMo-core` `MixtureDataLoader` configs (`olmo_mm_config_path`, including a built-in `merged_mt_lc` composable text-replay shim and `/weka` path remapping).
   * For multi-GiB `.npz` archives (such as `~6.4 GiB` `256K` global batches on GCS), `load_maxtext_batch_npz` parses the uncompressed ZIP local file headers and `.npy` array headers directly over seekable file streams, issuing byte-range reads strictly for each host's local DP row slice (`6.59 s` per `6.37 GiB` batch on GCS vs. downloading the entire archive).
   * `dataset_type=olmo_grain` supports `256K` packed streams over `8K`-chunked Dolmino + `256K` Longmino `.npy` files with `olmo_segment_chunk_length=8192` and `olmo_mask_chunk_boundary_targets_only=true` (masking only synthetic `8K` chunk edges while keeping real intra-chunk `EOS` transitions supervised) plus `olmo_apply_ngram_filter=true`.

> **Operational Guardrails for Production Launches:**
> * **Run `--mode bridge_init` before Stage 1 when starting from a text-only checkpoint** (such as `72.6B` Hero `step 162974`), and start Stage 3 from a Stage 2 checkpoint. Loading a text-only checkpoint directly into `use_multimodal=true` leaves the vision tower and connector at random init, which is not the reference recipe. In one synthetic-data `256K` benchmark arm that did this, the loss went `14.25 -> NaN`. The cause was not isolated, and the full `bridge_init` -> `Bridge` -> `Joint` -> `256K` path has not yet been run at Hero scale.
> * **Keep `freeze_vision_encoder_params=false` explicit when building custom `base.yml` command lines** for Stage 2 Joint or Stage 3 `256K` (`base.yml` defaults `freeze_vision_encoder_params: true`; both `olmo35_small_vision_align_8k.yml` and `olmo35_small_midtrain_256k.yml` already override this to `false`).
> * **Use `olmo_max_crops_per_seq=1024` (with `olmo_max_pooled_patches_per_seq=200704`) on `31.75 GiB` TPU v4 chips at `256K`:** Exported `mt_lc_256k` batches peak at `188` crops/sequence with `16` of `28` vision sources; AI2 reports `700`–`860` for the full mix. On `512` TPU v4 chips (`CP=32, EP=8`, synthetic data), `olmo_max_crops_per_seq=1024` peaks at **`19.03 GiB` XLA HBM (`5.33 GiB` live peak, `14.22 s/step`)**, whereas `olmo_max_crops_per_seq=2048` under `remat_policy=custom` reaches `33.42 GiB` XLA peak (`29.72 GiB` temp + `3.70 GiB` args). `2,048` crops/sequence with `remat_policy=full` has not been measured at `CP=32`.

---

## 3. Step 1: Checkpoint Preparation

Use `src/maxtext/checkpoint_conversion/convert_olmo35_multimodal_checkpoint.py` (`--use_ocdbt=false --use_zarr3=false` is recommended for fast multi-host TPU startup).

### Option A: Initialize Stage 1 (`8K Bridge`) from Pretrained Text Checkpoint + HF SigLIP2

From a MaxText pretrained Orbax checkpoint (e.g. `72.6B` Hero step `162974`):

```bash
export HF_TOKEN=<HF_TOKEN>

python -m maxtext.checkpoint_conversion.convert_olmo35_multimodal_checkpoint \
  --mode bridge_init \
  --maxtext_items_path gs://<BUCKET>/<HERO_PRETRAIN_RUN>/checkpoints/162974/items \
  --output_path gs://<BUCKET>/olmo35/checkpoints/hero_bridge_init \
  --vision_model_id google/siglip2-so400m-patch14-384 \
  --vision_revision e8e4872 \
  --vision_num_layers 25 \
  --connector_mlp_hidden_size 9728 \
  --scan_layers true \
  --use_ocdbt false \
  --use_zarr3 false \
  --validate true
```

Or from an `OLMo-core` DCP text checkpoint (e.g. `810M` Tiny `step870229`):

```bash
python -m maxtext.checkpoint_conversion.convert_olmo35_multimodal_checkpoint \
  --mode bridge_init \
  --checkpoint_path /mnt/weka/.../step870229-unsharded \
  --output_path gs://<BUCKET>/olmo35/checkpoints/tiny_bridge_init \
  --scan_layers true \
  --use_ocdbt false \
  --use_zarr3 false \
  --validate true
```

### Option B: Convert Existing `OLMo-core` Multimodal DCP Checkpoints <-> MaxText Orbax

```bash
# OLMo-core DCP -> MaxText Orbax
python -m maxtext.checkpoint_conversion.convert_olmo35_multimodal_checkpoint \
  --mode olmo_core_to_maxtext \
  --checkpoint_path /mnt/weka/.../step3500 \
  --output_path gs://<BUCKET>/olmo35/checkpoints/step3500_joint \
  --scan_layers true \
  --use_ocdbt false \
  --use_zarr3 false \
  --validate true

# MaxText Orbax -> OLMo-core DCP (for OLMo-core / Molmo evals)
python -m maxtext.checkpoint_conversion.convert_olmo35_multimodal_checkpoint \
  --mode maxtext_to_olmo_core \
  --maxtext_items_path gs://<BUCKET>/olmo35/runs/my_run/checkpoints/499/items \
  --olmo_config_path /path/to/reference_config.json \
  --output_path /mnt/output/step500_olmo_core \
  --validate true
```

---

## 4. Step 2: Data Preparation

### A. Multimodal Streams (`dataset_type=olmo_mm`)

You can either pre-export deterministic global `.npz` batches to GCS (pods then need neither `OLMo-core` nor raw image mounts) or point `olmo_mm_config_path` directly at a mounted `/weka` root. Pre-export suits validation and short runs: a `96`-sequence `256K` batch is roughly `10 GiB`, so the full `14,880`-step Stage 3 needs the live loader. See Section 6 for what has been verified in each mode.

```bash
# Export deterministic global batches from an OLMo-core config.json to GCS
# Optional: pass --global_batch_tokens <N> or --pack_buffer_size <N> (e.g. 64 for 256K fast startup)
python tools/data_generation/export_olmo_mm_batches.py \
  --olmo_config_path /path/to/joint_8k_config.json \
  --output_dir gs://<BUCKET>/olmo35/batches/joint_8k \
  --num_batches 16 \
  --path_remap_from /weka \
  --path_remap_to /mnt/weka \
  --work_dir /tmp/olmo_mm_work \
  --num_workers 4
```

Then pass to `train.py` (each host issues byte-range reads for its own rows of the current batch; nothing is prefetched):

```text
dataset_type=olmo_mm
olmo_mm_batches_dir=gs://<BUCKET>/olmo35/batches/joint_8k
```

> **Note on media coverage:** The batches exported for validation cover `16` of `28` vision sources and both text sources (`dolmino`, `longermino`). The `4` `pixmo_*` sources that read `torch_datasets/pixmo_images` and the `8` `cosyn_*` sources that read `torch_datasets/cosyn_images` need those two image directories under the remapped `/weka` root before a live 28-source run. Live loading also needs `pypdfium2` and `torchvision` on the workers.

### B. 256K Text MT + Long-Context (`dataset_type=olmo_grain`)

For `256K` text-only ablation or `dolmino` (`8K`-chunked) + `longermino` (`256K`) streaming via Grain:

```bash
python tools/data_generation/build_olmo_npy_index.py \
  --mix-file /path/to/mt_lc_mix.txt \
  --gcs-base gs://<BUCKET>/ \
  --tokenizer allenai/dolma3-tokenizer \
  --sequence-length 262144 \
  --output /tmp/mt_lc_seq262144.json \
  --workers 32
```

Enable `8K` chunk-edge masking in `train.py` so packed `8K` replay chunks do not leak cross-document context across chunk boundaries:

```text
dataset_type=olmo_grain
olmo_index_path=/tmp/mt_lc_seq262144.json
olmo_path_remap_from=gs://<BUCKET>/
olmo_path_remap_to=/mnt/olmo-data/
olmo_segment_chunk_length=8192
olmo_mask_cross_segment_targets=false
olmo_mask_chunk_boundary_targets_only=true
olmo_apply_ngram_filter=true
```

---

## 5. Step 3: Running the 3 Mid-Training Stages

Always export these environment variables on TPU pods before invoking `python -m maxtext.trainers.pre_train.train` (disables legacy 2-stage reduce-scatter at `DP >= 8` and enables fast host-loop / SparseCore device transfers):

```bash
export OLMOE3_ENABLE_2STAGE_RS=0
export OLMOE3_FAST_HOST_LOOP=1
export OLMOE3_FAST_SC_DEVICE_PUT=1
```

### Stage 1: `8K` Vision Connector Bridge (`500` steps)

Freezes `25L SigLIP2`, all `40L LM` blocks, `logits_dense`, and router `expert_bias` (`load_balance_loss_weight=0.0`), training only `OLMo35VisionConnector` and the `6` image token rows `[100278..100283]`:

```bash
python -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/olmo/olmo35_small_vision_align_8k.yml \
  run_name=olmo35_hero_bridge_8k \
  base_output_directory=gs://<BUCKET>/olmo35/runs \
  load_parameters_path=gs://<BUCKET>/olmo35/checkpoints/hero_bridge_init/0/items \
  dataset_type=olmo_mm \
  olmo_mm_batches_dir=gs://<BUCKET>/olmo35/batches/bridge_8k \
  freeze_vision_encoder_params=true \
  trainable_parameters_mask="['OLMo35VisionConnector','token_embedder']" \
  olmo_trainable_embedding_row_range="[100278,100284]" \
  frozen_parameters_mask="[]" \
  override_model_config=true \
  load_balance_loss_weight=0.0 \
  olmo_text_loss_group_weight=0.0 \
  olmo_vision_loss_group_weight=0.0 \
  learning_rate=2.0e-4 \
  lr_schedule_type=cosine \
  learning_rate_schedule_steps=250 \
  steps=500 \
  warmup_steps_fraction=0.4 \
  learning_rate_final_fraction=0.1 \
  olmo_hold_final_lr=true \
  olmo_vision_lr_cosine_alpha_f=-1.0 \
  olmo_connector_lr_scale=1.0 \
  olmo_embedding_lr_scale=1.0 \
  checkpoint_period=500 \
  checkpoint_storage_use_ocdbt=false \
  checkpoint_storage_use_zarr3=false
```

### Stage 2: `8K` Vision + Text Replay Joint Alignment (`3,500` steps)

Starts from the Stage 1 Bridge checkpoint (`checkpoints/499/items`), unfreezes `LM` + `25L SigLIP2` + `Connector` + `<im_*>` rows `[100278..100283]`, keeps `logits_dense` frozen, and applies `0.35` text / `0.65` vision loss group weights:

```bash
python -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/olmo/olmo35_small_vision_align_8k.yml \
  run_name=olmo35_hero_joint_8k \
  base_output_directory=gs://<BUCKET>/olmo35/runs \
  load_parameters_path=gs://<BUCKET>/olmo35/runs/olmo35_hero_bridge_8k/checkpoints/499/items \
  dataset_type=olmo_mm \
  olmo_mm_batches_dir=gs://<BUCKET>/olmo35/batches/joint_8k \
  checkpoint_period=500 \
  checkpoint_storage_use_ocdbt=false \
  checkpoint_storage_use_zarr3=false
```

### Stage 3: `256K` Unified Text MT + Long-Context + Multimodal (`mt_lc_256k`)

Starts from the Stage 2 Joint checkpoint (`checkpoints/3499/items`), unfreezes all parameters (`frozen_parameters_mask=[]`, `olmo_trainable_embedding_row_range=[]`), and runs `262,144` sequence length with `2D` Ulysses Context Parallelism (`custom_mesh_and_rule=cp-as-ep`):

```bash
# Default 3,072 TPU v4 chips (8x8x48) at 25.17M tokens/step (DP=12, CP=32, EP=8, zero host offload)
python -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/olmo/olmo35_small_midtrain_256k.yml \
  run_name=olmo35_hero_mt_lc_256k \
  base_output_directory=gs://<BUCKET>/olmo35/runs \
  load_parameters_path=gs://<BUCKET>/olmo35/runs/olmo35_hero_joint_8k/checkpoints/3499/items \
  dataset_type=olmo_mm \
  olmo_mm_batches_dir=gs://<BUCKET>/olmo35/batches/mt_lc_256k \
  use_multimodal=true \
  olmo_use_loss_weights=true \
  olmo_max_crops_per_seq=1024 \
  olmo_max_pooled_patches_per_seq=200704 \
  jax_cache_dir=gs://<BUCKET>/jax_cache \
  checkpoint_period=500 \
  checkpoint_storage_use_ocdbt=false \
  checkpoint_storage_use_zarr3=false
```

> **Note:** For `160B` total tokens (`6,358` steps @ `25.17M` tok/step) instead of `374.5B` tokens (`14,880` steps), set `steps=6358 learning_rate_schedule_steps=6358`. See header comments in `src/maxtext/configs/olmo/olmo35_small_midtrain_256k.yml` for `50.33M` (`DP=12, CP=16, EP=16`), `512`-chip (`8x8x8`), and `128`-chip (`4x4x8`) overrides.

---

## 6. Validation Status

Verified so far:

| Check | Scope | Result |
| :--- | :--- | :--- |
| Converter round trip | Tiny `step870229` (`bridge_init`), `step3500`, `step22320`; `829` tensors | `0.0` max abs diff |
| Hero `162974` compatibility | Key and shape match on `1,113` LM keys; `bridge_init` not yet run on it | Matched |
| Batch export | `8` `bridge_8k`, `8` `joint_8k`, `2` `mt_lc_256k` global batches | `16` of `28` vision sources; Bridge batches use synthetic images |
| Stage 1 Bridge mechanics | Tiny, `64` TPU v5p chips, `500` steps cycling `8` batches | `0.0` change on `ViT`, `LM`, `logits_dense`, `expert_bias`, text embedding rows |
| Stage 2 Joint mechanics | Tiny, `64` TPU v5p chips, `60` steps cycling `8` batches | `30+30` save/resume bit-exact |
| Converted `step3500` eval | `4` `joint_8k` batches | `1.246` mean loss (no reference value to compare yet) |
| Hero `256K` vision step time | `512` TPU v4 chips, synthetic data, `1,024` crops/seq | Measured `14.22 s/step`, `19.03 GiB` XLA peak |

The Stage 1 and Stage 2 runs cycle `8` batches, so their loss curves reflect memorization of those batches. They verify
freeze masks, schedules, and checkpointing, not convergence.

Not yet verified:

1. Step-0 logit and gradient parity against `OLMo-core` on the real `step3500` and `step22320` checkpoints.
2. Stage 3 (`256K`) training on real multimodal batches, and `olmo_grain` chunk-edge masking on hardware.
3. Loss curves against the GPU reference runs.
4. Live `MixtureDataLoader` mode (`olmo_mm_config_path`) on TPU hosts, including its throughput at `256K`.
5. Resume in live mode: the loader replays every earlier batch after a restart (no loader state is saved), which is
   slow late in a long run.
6. The full `bridge_init` -> `Bridge` -> `Joint` -> `256K` path at Hero scale.

---

## 7. Unit Tests

Run the OLMo 3.5 mid-training unit and parity test suites from the repository root:

```bash
pytest -v \
  tests/unit/olmoe3_midtraining_test.py \
  tests/unit/input_pipeline/olmo_data_grain_test.py \
  tests/unit/olmoe3_router_metrics_test.py
```
