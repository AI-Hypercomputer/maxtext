# Ming-3.0-audio-tiny thinker: audio SFT (speech + text -> text) handover

Branch: `hengtaoguo-ming-train-thinker` (on top of `hengtaoguo-ming-talker`). TPU VM: `hengtaoguo-dev-v5p2` (v5p, 4 chips).
Work dir on the VM: `/dev/shm/hengtaoguo/ming` (tmpfs: **lost on reboot**, see "Re-provisioning").

## TL;DR

* `train_sft_native.py` + `src/maxtext/configs/post_train/sft-audio-gigaspeech.yml` now fine-tunes the Ming thinker
  on (audio, prompt) -> transcript pairs streamed from parquet on GCS. No new entry point was needed: the HF SFT
  pipeline gained an audio branch (`hf_data_processing.audio_sft_preprocessing_pipeline`) and `loss_fn` forwards the
  mel features as `encoder_audios`.
* Data: GigaSpeech `xs` train (9,389 segments, 0.5–19.9 s, mean 3.8 s) re-sharded to
  `gs://hengtaoguo-maxtext-euw4/data/gigaspeech_xs/train-0000{0..7}-of-00008.parquet` by
  `tools/data_generation/prepare_gigaspeech_parquet.py`.
* Static shapes: every clip is zero-padded (waveform) / truncated to `audio_max_mel_frames` (2000 = 20 s) mel frames;
  the number of `<|audio_pad|>` placeholders follows the clip's **true** length, so the decoder sees exactly what the
  HF processor would produce at inference.
* Smoke run: see "Results" (50 steps, loss curve, checkpoint path).

## What changed (files)

| File | Change |
| --- | --- |
| `src/maxtext/multimodal/processor_ming.py` | `decode_audio` (HF `Audio` dict / bytes / path -> waveform), `preprocess_audio_for_training_ming` (static mel + `audio_lengths`), dummy init shape now uses `audio_max_mel_frames`. |
| `src/maxtext/multimodal/processor.py` | `preprocess_audio_for_training(audio, config)` router (decoder block `ming`). |
| `src/maxtext/input_pipeline/input_pipeline_utils.py` | `reformat_prompt_audio`, `pre_process_audio_sft`, `prepare_text_for_audio_fusion` (example maps); `SFTPromptMaskingVision(media_column=...)`; `PadOrTrimToMaxLength` passes `"audios"` through; `ExtractAudios`. |
| `src/maxtext/input_pipeline/hf_data_processing.py` | `is_audio_sft`, `audio_sft_preprocessing_pipeline`, routing in `make_hf_{train,eval}_iterator`. |
| `src/maxtext/trainers/pre_train/train.py` | `loss_fn` forwards `data["audios"]` as `encoder_audios` when `use_audio`. |
| `src/maxtext/utils/maxtext_utils.py` | `get_shaped_batch` skips the `"images"` entry when the model has no vision tower (the `"audios"` entry already existed). |
| `src/maxtext/configs/types.py`, `base.yml` | `train_audio_column`, `eval_audio_column`, `audio_max_mel_frames` (default 3000 = unchanged inference behaviour). |
| `src/maxtext/configs/post_train/sft-audio-gigaspeech.yml` | Recipe (hf `parquet` from GCS, prompt injected via `default_prompt`). |
| `tools/data_generation/prepare_gigaspeech_parquet.py` | GigaSpeech HF parquet -> normalized, re-sharded parquet on GCS. |
| `tests/unit/audio_sft_pipeline_test.py` | Featurizer shape/length test + end-to-end grain pipeline test (needs `MING_HF_PATH`). |
| `~/projects/.vscode/launch.json` | `train_sft_ming_audio` entry (also usable with `run_detached.py`). |

## Data path

```
HF speechcolab/gigaspeech  parquet-data/xs/train-00000-of-00001.parquet   (audio = 16 kHz WAV bytes, text with <COMMA> tags)
   | tools/data_generation/prepare_gigaspeech_parquet.py  (normalize text, drop > max_duration, 8 shards)
   v
gs://hengtaoguo-maxtext-euw4/data/gigaspeech_xs/train-*.parquet   columns: audio{bytes,path}, text, duration, segment_id
   | datasets.load_dataset("parquet", data_files=gs://..., streaming=True)   (gcsfs; no local copy)
   v
audio_sft_preprocessing_pipeline (HF map stage, python objects):
   add_default_prompt_if_missing -> prompt column = default_prompt
   reformat_prompt_audio        -> "<role>SYSTEM</role>...<role>HUMAN</role><|audio_start|><|audio_pad|><|audio_end|>{prompt}<|role_end|><role>ASSISTANT</role>\n<think></think>"
   reformat_response            -> "{text}<|role_end|>"
   pre_process_audio_sft        -> "audios": MingPreprocessorOutput(audio_values [1, 2000, 128], audio_lengths [n_tokens(true_len)])
   tokenization (batched)       -> prompt / response token ids
   prepare_text_for_audio_fusion-> the single <|audio_pad|> is expanded to audio_lengths[0] copies
   | HFDataSource -> grain
   v
SFTPromptMaskingVision(media_column="audios")  -> inputs = prompt+response, targets = pad*len(prompt)+response
PadOrTrimToMaxLength(max_target_length)       -> *_segmentation / *_position, text padded with pad_id (156892 = <|endoftext|>)
ExtractAudios                                 -> "audios": float32 [2000, 128]
grain.Batch                                   -> [B, 384] text, [B, 2000, 128] audio
ShiftData                                     -> next-token targets, prompt + padding excluded from the loss
```

`loss_fn` -> `model(..., encoder_audios=batch["audios"])` -> `AudioEncoder` (whisper, frozen via `stop_gradient` when
`freeze_audio_encoder_params=true`; projector trainable) -> `merge_mm_embeddings` scatters the i-th projected embedding
into the i-th `<|audio_pad|>` and drops the surplus (padded tail).

Prepare other splits / subsets the same way, e.g. the dev set as an eval set:

```bash
HF_HOME=/dev/shm/hengtaoguo python tools/data_generation/prepare_gigaspeech_parquet.py \
  --subset dev --split validation --num_shards 4 --output_dir gs://hengtaoguo-maxtext-euw4/data/gigaspeech_dev
# then: hf_eval_files=gs://.../gigaspeech_dev/validation-*.parquet hf_eval_split=validation eval_interval=...
```

## Design choices (and why)

1. **Padding, not packing, for v1.** MaxText compiles one static train step; the whisper encoder has no attention
   mask and the Ming decoder path (KDA recurrence + MLA + MoE) is already heavy to compile. Zero-padding the *waveform*
   to a fixed 20 s is exactly whisper's own 30 s convention, so padded frames are ordinary "silence" to the encoder.
   Cost: GigaSpeech xs averages 3.8 s, so ~80% of encoder compute is padding (the encoder is frozen and cheap relative
   to the 24-layer MoE decoder; decoder tokens are not inflated because placeholders follow the true length).
2. **Placeholder count = true length** (`get_num_audio_tokens(true_mel_frames)`), *not* the padded length. This keeps
   train/inference token layouts identical to the HF processor. The surplus encoder outputs are dropped by the
   cumsum-gather in `merge_mm_embeddings`. Flipping one line in `preprocess_audio_for_training_ming`
   (`audio_lengths = [get_num_audio_tokens(max_frames)]`) gives the "all placeholders" variant if ever wanted.
3. **Audio travels as a `PreprocessorOutput` under `"audios"`** through the HF map stage (same pattern as `"images"`
   for vision SFT) and becomes a plain `[T_mel, n_mels]` array only in `ExtractAudios`, right before batching.
   `audio_lengths` is dropped there because the model does not need it; a ragged/packed pipeline can keep it.
4. **Reuse over new code.** `SFTPromptMaskingVision` got a `media_column` argument instead of a copy; prompt/response
   formatting and token expansion route through the existing `mm_processor` functions (decoder block `ming`), so
   adding another audio model means implementing `preprocess_audio_for_training_<model>` + the existing
   `reformat_prompt_*` / `add_extra_tokens_for_*` hooks.
5. **Dataset normalization happens offline** (`prepare_gigaspeech_parquet.py`): GigaSpeech tags
   (`<COMMA>` ...) -> punctuation, garbage-only utterances dropped, lower-cased with sentence capitalization
   ("I" restored). The pipeline itself is dataset-agnostic: any parquet/HF dataset with an audio column
   (HF `Audio` dict, raw bytes or a path) and a text column works; a prompt column is optional.
6. **Checkpoint in GCS, HF weights in tmpfs.** `/dev/shm` had ~40 GB free; the thinker HF weights (18 GB) are kept
   locally and the fp32 MaxText checkpoint (32 GB) was written straight to GCS by `to_maxtext`.

## Caveats

* **Encoder sees padded silence.** Whisper is robust to it, but Ming's encoder was fine-tuned by the publisher
  without padding, so encodings of the real frames differ slightly from inference (no mask in `MingWhisperEncoder`).
  If this matters, add a key-padding mask to the encoder attention (the `audio_mask` is already computed by the
  featurizer) or lower `audio_max_mel_frames` to the dataset's real maximum.
* **Clips longer than `audio_max_mel_frames` are truncated** silently (data prep filters by `--max_duration`).
  `audio_max_mel_frames` must stay <= `ming_audio_chunk_size` (3000) for a single whisper chunk; longer values work
  (multi-chunk) but were not exercised.
* **One audio clip per example.** `pre_process_audio_sft` / `ExtractAudios` assume a single clip.
* **Text truncation.** `max_target_length=384` fits the prompt (~30 tokens) + up to 125 audio tokens + transcript;
  `SFTPromptMaskingVision` truncates the response if it overflows (no error).
* **`pad_id == eos_id` (`<|endoftext|>`, 156892).** Responses end with `<|role_end|>` (156895), so nothing real is
  masked, but do not append `<|endoftext|>` to responses.
* **Only the projector + decoder train** (`freeze_audio_encoder_params=true` uses `stop_gradient`, the encoder
  params still carry optimizer state). Set it to `false` to fine-tune whisper too.
* **MoE router bias** (`routed_bias_update_rate: 0.0`): no loss-free load-balancing updates during SFT; fine for
  short fine-tunes, revisit for long runs.
* **KDA layers run the token-by-token `lax.scan` recurrence** (no fused kernel on this branch); the backward pass
  is another scan, so step time grows linearly with `max_target_length`.
* Eval: `sft-audio-gigaspeech.yml` points `hf_eval_files` at the train shards as a placeholder; prepare the dev split
  (above) before enabling `eval_interval`.
* HF `datasets` streaming + `grain_worker_count=1`: data loading is single-process; mel computation (~ms/clip) was not
  the bottleneck in the smoke run.

## Future directions

* **Length bucketing / dynamic packing**: group clips by duration into a few `audio_max_mel_frames` buckets (one
  compiled step per bucket), or pack several short clips' text into one `max_target_length` row with
  `[B, N_audio, T_mel, n_mels]` audio + `audio_lengths`; `merge_mm_embeddings` already handles concatenated
  per-example embeddings and `ExtractAudios`/`FoldImagesIntoBatch` show the fold pattern. The per-example structure
  (prompt / response / "audios" `PreprocessorOutput` with `audio_lengths`) was kept explicit for this.
* Whisper encoder key-padding mask from `audio_mask` (exact inference parity + enables shorter effective compute).
* Multi-clip examples (`<|audio|>` placeholders in the prompt are already supported by `reformat_prompt_ming`).
* Eval on GigaSpeech dev + a decode-based WER check with `decode_ming_thinker` on the SFT'd checkpoint.
* Scanned layers (`scan_layers=true`, ckpt `fp32_scan_true`) to cut compile time for longer runs.

## Re-provisioning (after a VM reboot: /dev/shm is empty)

```bash
# HF weights (top-level files only; talker/ safetensors collide with thinker keys)
HF_HOME=/dev/shm/hengtaoguo python -c "from huggingface_hub import snapshot_download; snapshot_download('inclusionAI/Ming-3.0-audio-tiny', ignore_patterns=['talker/**'], local_dir='/dev/shm/hengtaoguo/ming/hf')"
mkdir -p /dev/shm/hengtaoguo/ming/hf_thinker && cd /dev/shm/hengtaoguo/ming/hf_thinker && for f in ../hf/*; do [ "$(basename $f)" = talker ] || ln -sfn $f .; done
# MaxText checkpoint already in GCS: gs://hengtaoguo-maxtext-euw4/checkpoints/ming-3.0-audio-tiny/fp32_scan_false/0/items
# (re-create with launch.json `to_mt_ming`, --hf_model_path=/dev/shm/hengtaoguo/ming/hf_thinker, base_output_directory=gs://...)
# Data already in GCS: gs://hengtaoguo-maxtext-euw4/data/gigaspeech_xs/
```

## Commands

```bash
# unit tests (CPU)
cd ~/projects/maxtext && PYTHONPATH=src:. JAX_PLATFORMS=cpu HF_HOME=/dev/shm/hengtaoguo HF_HUB_OFFLINE=1 \
  MING_HF_PATH=/dev/shm/hengtaoguo/ming/hf_thinker venv1/bin/python -m pytest -q tests/unit/audio_sft_pipeline_test.py
# 50-step smoke training (detached; log in /dev/shm/hengtaoguo/ming/logs/gigaspeech_xs_smoke50.log)
python3 ~/projects/run_detached.py train_sft_ming_audio --log_dir /dev/shm/hengtaoguo/ming/logs
```

## Results (2026-11-19, `run_name=gigaspeech_xs_smoke50`)

Setup: v5p 4 chips, FSDP=4, `per_device_batch_size=4` (global 16), `max_target_length=384`, `audio_max_mel_frames=2000`,
bf16 activations / fp32 weights+Adam, `attention=dot_product`, `remat_policy=full`, lr 2e-5 (10% warmup),
`scan_layers=false`, audio encoder frozen. 8.555 B params; 24.4 GB / 95.7 GB HBM per chip after init;
compile ~4.5 min, **~1.15 s/step** (~1340 tokens/s/device; mostly the KDA `lax.scan`, see caveats).

Loss (completion tokens only, ~200–330 supervised tokens per step):

| steps | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0–9   | 0.429 | 0.628 | 0.438 | 0.361 | 0.470 | 0.379 | 0.301 | 0.218 | 0.200 | 0.147 |
| 10–19 | 0.214 | 0.175 | 0.247 | 0.262 | 0.149 | 0.192 | 0.179 | 0.232 | 0.389 | 0.170 |
| 20–29 | 0.145 | 0.168 | 0.236 | 0.146 | 0.219 | 0.184 | 0.190 | 0.137 | 0.157 | 0.195 |
| 30–39 | 0.155 | 0.132 | 0.242 | 0.249 | 0.207 | 0.102 | 0.225 | 0.165 | 0.193 | 0.226 |
| 40–49 | 0.156 | 0.222 | 0.203 | 0.222 | 0.123 | 0.156 | 0.099 | 0.112 | 0.205 | 0.241 |

Mean of steps 0–4: **0.465** -> mean of steps 40–49: **0.174** (the base model is already a capable ASR model, so the
starting loss is low; the drop mostly reflects learning the transcript format: punctuation/casing of the normalized
GigaSpeech text). `moe_lb_loss` is 0 (no load-balance loss configured).

Checkpoint (full train state, fp32 params + Adam, 80 GiB):
`gs://hengtaoguo-maxtext-euw4/checkpoints/ming-3.0-audio-tiny/sft_runs/gigaspeech_xs_smoke50/checkpoints/49/items`
(`checkpoint_period=50` also wrote a step-0 copy of the initial state; it was deleted to save space.)
TensorBoard: `gs://hengtaoguo-maxtext-euw4/checkpoints/ming-3.0-audio-tiny/sft_runs/gigaspeech_xs_smoke50/tensorboard/`.
Logs: `/dev/shm/hengtaoguo/ming/logs/gigaspeech_xs_smoke50.log` (`_attempt1.log` = the `get_shaped_batch` crash fixed in
`maxtext_utils.py`: it emitted an empty-shaped `"images"` entry for a multimodal model without a vision tower).

To decode with the SFT'd weights, point `decode_ming_thinker` at `.../checkpoints/49/items` with
`load_full_state_path`/`load_parameters_path` as appropriate for a train-state checkpoint (params live under `params`).
