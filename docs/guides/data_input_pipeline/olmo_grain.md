# OLMo numpy pipeline (`dataset_type=olmo_grain`)

Grain-based input pipeline for AI2's pre-tokenized OLMo data mixes (e.g.
`OLMo-mix-0925-official.txt`). Reads headerless flat `.npy` token streams
from a gcsfuse mount, shards across hosts, optionally masks repeated-n-gram
instances, and yields the shapes the MaxText pretrain trainer expects.

## Quick start

1. **Download the data** to a GCS bucket. `--mix-file` is a local AI2 manifest listing relative npy paths to fetch from AI2's public bucket (e.g. `OLMo-mix-0925-official.txt` for the 6T pretrain mix or `OLMo-midtraining-mix-0625-100B.txt` for the 100B midtraining mix).

   ```bash
   python tools/data_generation/download_olmo_data_to_gcs.py \
       --mix-file ./OLMo-mix-0925-official.txt \
       --gcs-dest gs://my-bucket/dataset/ \
       --staging-dir /mnt/local-ssd/olmo-staging \
       --workers 16
   ```

2. **Mount it read-only** with gcsfuse (`np.memmap` needs a local path):

   ```bash
   gcsfuse --implicit-dirs --o ro my-bucket /mnt/olmo-readonly
   ```

3. **Build the index**:

   ```bash
   python tools/data_generation/build_olmo_npy_index.py \
       --mix-file /path/to/OLMo-mix-0925-official.txt \
       --gcs-base gs://my-bucket/dataset/ \
       --tokenizer allenai/dolma3-tokenizer \
       --sequence-length 8192 \
       --output /path/to/olmo_index_seq8192.json
   ```

4. **Configure + run** the trainer:

   ```yaml
   dataset_type: olmo_grain
   olmo_index_path: /path/to/olmo_index_seq8192.json
   olmo_path_remap_from: "gs://my-bucket/"
   olmo_path_remap_to:   "/mnt/olmo-readonly/"
   max_target_length: 8192        # must equal index sequence_length
   tokenizer_type: huggingface
   tokenizer_path: allenai/Olmo-3-7B-Instruct
   ```

   See `scripts/run_olmo3_7b_grain_smoke.sh` for a runnable smoke launcher.

## Resume

Stateless sampler: record at step *k* is a pure function of `(seed, shard, k)`. On startup, the trainer adapter reads the latest step from
`config.checkpoint_dir` and shifts the sampler so the data stream picks
up where it left off — no Grain-iterator-state in the checkpoint.

`scripts/run_olmo3_7b_grain_resume_test.sh` validates this end-to-end.

## Notes

- Files are headerless raw uint32 by default (matches AI2's published
  format). The numpy `.npy` extension is misleading.
- Documents may span instance boundaries; this matches OLMo-core.
- `olmo_apply_ngram_filter: True` (default) zeroes loss on instances with
  ≥ 32 repetitions of any 1–13-gram, per OLMo-core.
- For mixing pretraining + midtraining, build a combined index by
  concatenating the two .txt mix files.

## Troubleshooting

| Symptom                                                       | Fix                                                                                                       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `OLMo index sequence_length=N but config.max_target_length=M` | Rebuild the index with `--sequence-length M`.                                                             |
| `q_block_size=512 should divide q_seq_len=…`                  | Set `max_target_length` to a multiple of 512.                                                             |
| OOM during compile on a small TPU                             | Shrink with `override_model_config=True base_num_decoder_layers=N`, use `weight_dtype=bfloat16`.          |
| Resume restarts at step 0                                     | Iterator log should print `resumed_step=N initial_step=…`; if both 0, `checkpoint_dir` is empty or wrong. |
