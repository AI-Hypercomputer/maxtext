# OLMo-3 7B stage-1 pretraining launchers

End-to-end launchers for reproducing AI2's [OLMo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
stage-1 pretraining in MaxText. Hyperparameters mirror OLMo-core's
[`OLMo-3-1025-7B-pretrain-1.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py):
global batch ≈4M tokens, peak LR 3e-4 cosine to 0.1×, 2k warmup, β=(0.9, 0.95),
ε=1e-8, WD 0.1, grad-clip 1.0, z-loss 1e-5, logits in fp32, skip-step-on-spike.

## Scripts

| Script | Purpose |
|---|---|
| `run_olmo3_7b_stage1.sh` | Direct host launcher. Used by the XPK wrapper inside the runner image; also usable standalone for single-host smoke tests. |
| `xpk_olmo3_7b_stage1.sh` | XPK wrapper for multi-host TPU clusters. Modes: `submit` / `monitor` / `resume_until_done` (deletes + resubmits on Kueue preemption until a target step is reached). |

Both scripts are env-var driven. Header comments in each enumerate required vs. optional env.

## Quick start (multi-host TPU via XPK)

```bash
source ~/.hf_token.sh

export XPK_CLUSTER=<your-cluster>
export XPK_PROJECT=<your-project>
export XPK_ZONE=<your-zone>
export XPK_DEVICE_TYPE=tpu7x-4x8x8       # or tpu7x-4x4x4 for a smoke run
export XPK_TOTAL_DEVICES=512             # set to match device-type * 2 (Ironwood has 2 JAX devices/chip)
export XPK_BASE_OUTPUT_DIR=gs://<your-bucket>/olmo/runs
export XPK_RUN_NAME=olmo3_7b_stage1

export OLMO_INDEX_PATH=/tmp/olmo-data/olmo/indices/olmo_index_seq8192.json
export OLMO_GCS_BASE=gs://<your-bucket>/
export LOAD_PARAMETERS_PATH=gs://<your-bucket>/olmo/checkpoints/stage1-step0/0/items

bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh submit
bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh monitor
# Drive the full pretrain (auto-resubmits on preemption):
STEPS_OVERRIDE=1414078 \
  bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh resume_until_done
```

`STEPS=1414078` (= 5.928T tokens at 512 × 8192/step) matches AI2's stage-1
horizon. The wrapper passes `LIBTPU_INIT_ARGS` (Ironwood XLA flags) and the
full MaxText perf flag set automatically — no manual override needed.

## Quick start (single-host / smoke test)

```bash
source $MAXTEXT_ROOT/maxtext_venv/bin/activate

INDEX_PATH=/tmp/olmo-data/olmo/indices/olmo_index_seq8192.json \
GCS_BASE=gs://<your-bucket>/ \
LOCAL_MOUNT=/tmp/olmo-data \
OUTPUT_DIR=gs://<your-bucket>/olmo/runs \
LOAD_PARAMETERS_PATH=gs://<your-bucket>/olmo/checkpoints/stage1-step0/0/items \
HF_SECRETS=~/.hf_token.sh \
RUN_NAME=olmo3_7b_stage1 \
STEPS=50 WARMUP_STEPS=10 CHECKPOINT_PERIOD=50 \
bash src/maxtext/trainers/pre_train/scripts/olmo/run_olmo3_7b_stage1.sh
```

## Prerequisites

**Tokenized data corpus.** AI2 maintains the OLMo-mix manifests in
[`allenai/OLMo-core/src/olmo_core/data/mixes/`](https://github.com/allenai/OLMo-core/tree/main/src/olmo_core/data/mixes)
(stage-1 uses
[`OLMo-mix-0625-official.txt`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0625-official.txt)
or
[`OLMo-mix-0925-official.txt`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0925-official.txt);
the matching `DataMix` enum values are in
[`mixes/__init__.py`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/__init__.py)).
This MaxText repo also vendors copies of these at root for offline use. The
underlying preprocessed `.npy` tokens live under AI2's `s3://ai2-llm/` bucket
(see the `base_dir` arg on `DataMix.build` in
[`OLMo-core`](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/data/mixes/__init__.py));
mirror them into your own GCS bucket with
[`tools/data_generation/download_olmo_data_to_gcs.py`](../../../../../tools/data_generation/download_olmo_data_to_gcs.py)
(reads a manifest, pulls from AI2's source, uploads to `--gcs-dest`).

**Data index.** Once the corpus is mirrored, build the index with
[`tools/data_generation/build_olmo_npy_index.py`](../../../../../tools/data_generation/build_olmo_npy_index.py)
against the same manifest + sequence length, then upload the resulting JSON
to GCS. Mount your bucket read-only via gcsfuse inside the pod — the XPK
wrapper does this automatically (`MOUNT_GCSFUSE=1`).

See [`docs/guides/data_input_pipeline/olmo_grain.md`](../../../../../../docs/guides/data_input_pipeline/olmo_grain.md)
for the full data-pipeline reference.

**Init weights.** Two supported starting points:

#### Option 1: From AI2's `stage1-step0` (recommended — reproduces AI2's run)

This puts MaxText at exactly the same weights as the PyTorch reference, so
the loss curve can be overlaid against AI2's published WandB curve. AI2
also publishes intermediate checkpoints at every 1000-step boundary on the
[`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B/refs)
HF repo (`stage1-step0`, `stage1-step1000`, …, `stage1-step1413814`); the
same procedure works for any of them — just swap the `--revision` flag.

1. Convert the HF snapshot to Orbax (one-time, ~7 min wall on a 16-core VM,
   peak ~26 GB RAM; needs `huggingface_hub` auth via `HF_TOKEN`):

   ```bash
   export HF_TOKEN=...
   python -m maxtext.checkpoint_conversion.to_maxtext \
     model_name=olmo3-7b-pt scan_layers=True \
     --revision=stage1-step0 \
     --lazy_load_tensors=True --save_dtype=bfloat16
   ```

   The script writes an Orbax checkpoint under
   `<output>/<run_name>/0/items/` (default `<output>` is `/tmp/maxtext`;
   override with `--base_output_directory=...`).

2. Upload the converted checkpoint to GCS so all pods can read it:

   ```bash
   gsutil -m cp -r <output>/0/items gs://<your-bucket>/olmo/checkpoints/stage1-step0/0/items
   ```

3. Point the launcher at it via `LOAD_PARAMETERS_PATH`:

   ```bash
   export LOAD_PARAMETERS_PATH=gs://<your-bucket>/olmo/checkpoints/stage1-step0/0/items
   bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh submit
   ```

   AI2's published checkpoints are parameters only — Adam state is freshly
   initialized on the MaxText side. The cold-start init is ignored once
   `OUTPUT_DIR/RUN_NAME/checkpoints/` has any in-run checkpoints to
   resume from.

#### Option 2: Random MaxText init

Leave `LOAD_PARAMETERS_PATH` unset and the launcher falls through to
MaxText's standard random init from the `olmo3-7b-pt` model config — no
conversion step needed. Useful as a sanity check that MaxText's init
scheme is well-formed independent of the AI2-converted checkpoint (in
practice the init transient resolves by step ~200, after which the curve
tracks the AI2-init curve within data-shuffle noise).

```bash
unset LOAD_PARAMETERS_PATH
bash src/maxtext/trainers/pre_train/scripts/olmo/xpk_olmo3_7b_stage1.sh submit
```

**Runner image** (XPK only). Build + push once:

```bash
sudo bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=stable WORKFLOW=pre-training
sudo bash src/dependencies/scripts/docker_upload_runner.sh \
  CLOUD_IMAGE_NAME=maxtext-olmo3 PROJECT=$XPK_PROJECT
```

Override the resulting image with `XPK_DOCKER_IMAGE` (defaults to
`gcr.io/${XPK_PROJECT}/maxtext-olmo3:latest`).

## Resume

Keep `XPK_RUN_NAME` and `XPK_BASE_OUTPUT_DIR` stable across submissions.
Orbax picks up the latest checkpoint under `${OUTPUT_DIR}/${RUN_NAME}/checkpoints/`;
the OLMo grain sampler resumes its data position via stateless
`initial_step = step × per-host-batch` (no Grain-iterator-state in the
checkpoint).

`resume_until_done` polls the latest checkpoint step, resubmits a fresh
workload on preemption, and exits when checkpoint step ≥ `STEPS_OVERRIDE` or
after `MAX_RETRIES` failed attempts. Tune for preemption-prone clusters:

| Env var | Default | Use |
|---|---|---|
| `MAX_RETRIES` | 10 | Cap on resubmit attempts before giving up. |
| `RETRY_BACKOFF_SECONDS` | 300 | Sleep between resubmits. Bump if Kueue admission is slow. |

## Loss-overlay validation

For a partial-run overlay against AI2's published WandB curve, set
`LR_SCHEDULE_STEPS=1414078` (AI2's stage-1 schedule horizon at 4.19M
tokens/step) independently of `STEPS`. This keeps warmup absolute and the
cosine shape matched even on a short run.
