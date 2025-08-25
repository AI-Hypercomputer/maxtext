<!--
 Copyright 2024 Google LLC

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

(data-input-perf)=
# Performance of Data Input Pipeline
* Overview of supported data input pipelines: [](data-input-pipeline)
* Perf data intepretation: for all three data pipelines, there are data prefetch running in parallel with computation. The goal is to hide data loading behind computation. As long as data loading step time < training computation step time, the data pipeline perf is considered sufficient. 

## Methods
* The following results are measured by [standalone_dataloader.py](https://github.com/google/maxtext/blob/main/MaxText/standalone_dataloader.py), which performs data loading without computation.
* c4 data of different formats in GCS bucket are used. For Grain pipeline only, the GCS bucket is mounted to a local path via GCSFUSE ([script](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh))
* The GCS bucket is multi-region (US) and the VMs that read data can be in different regions in the US.

## HuggingFace pipeline
The following data are collected using c4 data in Parquet format.
| Pipeline    | seq_len | VM type    | per_host_batch    | # of host | # of batch | first step (s) | total time (s) |
| ----------- | ------- | ---------- | ----------------- | --------- | ---------- | -------------  | -------------- |
| HuggingFace | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 6              | 72             |
| HuggingFace | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 6              | 72             |

## Grain pipeline
The following data are collected using c4 data in ArrayRecord format.
| Pipeline    | seq_len | VM type    | per_host_batch    | # of host | # of batch | worker | first step (s) | total time (s) |
| ----------- | ------- | ---------- | ----------------- | --------- | ---------- | -----  | -------------- | --------------- |
| Grain       | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 1      | 7              | 1200            |
| Grain       | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 2      | 7              | 355             |
| Grain       | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 4      | 8              | 280             |
| Grain       | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 8      | 15             | 367             |
| Grain       | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 1      | 7              | 691             |
| Grain       | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 2      | 7              | 335             |
| Grain       | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 4      | 8              | 154             |
| Grain       | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 8      | 11             | 120             |

## TFDS pipeline
The following data are collected using c4 data in TFRecord format.
| Pipeline    | seq_len | VM type    | per_host_batch    | # of host | # of batch | first step (s) | total time (s) |
| ----------- | ------- | ---------- | ----------------- | --------- | ---------- | -------------  | -------------- |
| TFDS        | 2048    | TPU v4-8   | 32 (per_device=8) | 1         | 1000       | 2              | 17             |
| TFDS        | 2048    | TPU v4-128 | 32 (per_device=8) | 16        | 1000       | 3              | 18             |
