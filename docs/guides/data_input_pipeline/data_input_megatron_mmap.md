<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Megatron indexed datasets with Grain

MaxText can read pre-tokenized text datasets produced by Megatron-LM as a
Grain file type. The source consists of `<prefix>.bin`, which stores token IDs,
and `<prefix>.idx`, which stores sequence and document metadata. This is not a
separate MaxText dataset pipeline: configure `dataset_type=grain` and select
either `grain_file_type=mmap_npy` or `grain_file_type=mmap`.

Use `mmap_npy` when matching Megatron GPT sample order. It builds or loads
three NumPy indices that define document order, fixed-length sample boundaries,
and sample shuffle. Direct `mmap` is a simpler windowed source and does not
provide the same ordering guarantee. Both modes reuse Grain workers, batching,
multi-host loading, and iterator checkpoints.

## Prepare the data

Generate `.bin` and `.idx` files with Megatron-compatible preprocessing and
append the end-of-document token during preprocessing. MaxText does not insert
EOD tokens at runtime. `mmap_eod_id` must match the written token ID, and
`mmap_split_sentences=true` must be set only when preprocessing used
`--split-sentences`.

The files must be available through a local or mounted filesystem path. Mount
Cloud Storage with Cloud Storage FUSE instead of passing a `gs://` URI.

## Train with mmap_npy

`grain_train_files` has the following single-dataset form:

```text
<npy_index_dir>|<data_prefix>[:<another_data_prefix>...]
```

The index directory stores `document_index`, `sample_index`, and
`shuffle_index` files. A data prefix is the common path without `.bin` and
`.idx`; a directory containing multiple shards is also accepted.

```sh
python3 -m maxtext.trainers.pre_train.train \
  dataset_type=grain \
  grain_file_type=mmap_npy \
  grain_train_files='/cache/wiki_indices|/data/wiki_text_document' \
  mmap_eod_id=2 \
  max_target_length=2048 \
  data_shuffle_seed=1234 \
  reset_attention_mask=true \
  eod_mask_loss=false \
  packing_max_segments_per_sample=0 \
  steps=1000
```

The compatibility guarantee assumes that MaxText and Megatron use the same
input files, sequence length, EOD ID, seed, split or blend recipe, and requested
sample count. For training, the requested sample count is
`steps * global_batch_size_to_load`.

On a cache miss, every host deterministically builds child indices in memory;
host 0 persists them to `npy_index_dir` using atomic writes. No cross-host
barrier is required. The directory must be writable by host 0 unless matching
indices were built in advance.

`reset_attention_mask` and `eod_mask_loss` must match the reference job. A
positive `packing_max_segments_per_sample` deliberately merges short
EOD-derived attention segments; set it to `0` to retain every EOD boundary for
conventional Megatron GPT parity.

## Split training and evaluation documents

Use the same source for training and evaluation with a Megatron-style split:

```sh
grain_train_files='/cache/wiki_indices|/data/wiki_text_document' \
grain_eval_files='/cache/wiki_indices|/data/wiki_text_document' \
mmap_npy_split='99,1'
```

Training uses split 0 and evaluation uses split 1. Without a split,
`grain_eval_files` can point to a separate evaluation dataset.

## Blend mmap_npy datasets

Append a weight to each dataset specification and separate components with
semicolons:

```text
grain_train_files='/cache/wiki_indices|/data/wiki,0.7;/cache/code_indices|/data/code,0.3'
```

Weights must be non-negative and have a positive total; zero-weight components
are ignored. The blend is constructed in global sample order before host
sharding. `blend_cache_dir` optionally persists the generated global dispatch;
a write failure continues with in-memory indices. `blend_index_dir` can instead
contain a prebuilt `dataset_index.npy` and `dataset_sample_index.npy` pair.

## Use direct mmap

Direct mode accepts a `.bin`/`.idx` prefix or a shard directory without the
`npy_index_dir|` portion:

```sh
dataset_type=grain \
grain_file_type=mmap \
grain_train_files='/data/wiki_text_document'
```

A weighted mixture uses `prefix,weight;another_prefix,weight`. Direct mode can
apply the standard Grain shuffle but does not consume `mmap_npy_split`,
`blend_cache_dir`, or `blend_index_dir`, and does not promise Megatron
document/sample or global blend ordering.

## Build indices before training

Runtime generation is sufficient for normal use. To avoid first-job startup
cost, build a single dataset in advance. `--num-samples` must match the training
sample count:

```sh
python3 tools/data_processing/mmap_index_builder.py convert \
  --input /data/wiki_text_document \
  --output-dir /cache/wiki_indices \
  --seq-length 2048 \
  --num-samples 1024000 \
  --seed 1234
```

The `blend` subcommand creates child index directories and the global dispatch:

```sh
python3 tools/data_processing/mmap_index_builder.py blend \
  --datasets '/data/wiki_text_document,0.7;/data/code_text_document,0.3' \
  --output-dir /cache/wiki_code_blend \
  --seq-length 2048 \
  --total-samples 1024000 \
  --seed 1234
```

Consume both kinds of output:

```sh
grain_train_files='/cache/wiki_code_blend/dataset_0|/data/wiki_text_document,0.7;/cache/wiki_code_blend/dataset_1|/data/code_text_document,0.3' \
blend_index_dir='/cache/wiki_code_blend'
```

## Limitations

- Multimodal Megatron indexed-dataset extensions are not supported.
- `grain_use_elastic_iterator=true` is not supported with `mmap` or `mmap_npy`.
- The standard `grain_train_mixture_config_path` JSON mechanism is for
  ArrayRecord; use the inline mixture syntax documented above.
- The mmap preprocessing path accepts exactly one pre-tokenized data column.
- For correct `eod_mask_loss=false` behavior, use `mmap_npy`. Direct `mmap`
  uses EOD as the padding sentinel during shifting and masks EOD-related loss.
