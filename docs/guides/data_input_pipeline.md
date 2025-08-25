<!--
 Copyright 2023–2025 Google LLC

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

(data-input-pipeline)=
# Data Input Pipeline

Currently MaxText has three data input pipelines:

| Pipeline | Dataset formats | Features | Limitations |
| -------- | --------------- | -------- | ----------- |
| HuggingFace | datasets in HuggingFace Hub<br>local/Cloud Storage datasets in json, parquet, arrow, csv, txt | convenience<br>multiple formats | limit scalability using HuggingFace Hub<br>non-deterministic with preemption<br>(deterministic without preemption) |
| Grain | ArrayRecord (available through Tensorflow Datasets), Parquet | fully deterministic, regardless of preemption; global shuffle with random access datasets; performant |  |
| TFDS | TFRecord, available through Tensorflow Datasets | performant | only supports TFRecords<br>non-deterministic with preemption<br>(deterministic without preemption) |

## Performance
* Perf data for all 3 input pipeline: [](data-input-perf)

## Multihost dataloading best practice
In multihost environment, if use an input pipeline that reads data sequentially (HuggingFace or TFDS), the most performant way is to have each data file only accessed by one host, and each host access a subset of data files (shuffle is within the subset of files). This requires (# of data files) to be multiples of (# of hosts loading data). We recommand users to reshard the dataset or use a subset of hosts to load data by setting expansion_factor_real_data (only available for some topologies, will error out otherwise). In MaxText, since the goal is to demonstrate the most performant experience, the behaviors for different data pipelines are:
### HuggingFace pipeline in multihost
* When (# of data files) >= (# of hosts loading data), assign files to each host as evenly as possible, some host may ended up with 1 file more than the others. When some hosts run out of data, they will produce empty padding batches, so that you are able to utilize the data from the hosts that still have data. But in this stage, training/eval will be less effective, and you will see a decrease in total_weights and slower change in loss. If all hosts run out of data before the step number you set, you will see 0 total_weights and 0 loss. The training/eval will run until the steps/eval_steps set in the config. Note that even each host are assigned the same number of data files, due to the different example count in each data file, and example packing, you will still have different number of batches on each host near the end of the epoch.
* When (# of data files) < (# of hosts loading data), files are read sequentially with multiple hosts accessing each file, perf can degrade quickly as # of host increases.
### TFDS pipeline in multihost
* When (# of data files) >= (# of hosts loading data), assign equal number of files to each host. The remainning files are skipped. Train/eval will hang if steps/eval_steps are not met but some hosts run out of data. Please set steps/eval_steps accordingly.
* When (# of data files) < (# of hosts loading data), files are read sequentially with multiple hosts accessing each file, perf can degrade quickly as # of host increases.
### Grain pipeline in multihost
* Perf not affected by (# of data files) vs (# of hosts loading data). [Data are shuffled globally](#global-shuffle-in-grain). Because grain uses a data format (ArrayRecord) that supports random access by index. Even with multiple hosts accessing the same file, they are accessing different indices and and won't have the issue seen with sequential reading.
* At the end of the dataset, you may still have some hosts runing out of indices and hang, Please set steps/eval_steps accordingly.

## HuggingFace pipeline
The HuggingFace pipeline supports streaming directly from HuggingFace Hub, or from GCS bucket in HuggingFace supported formats (parquet, json, etc.). This is through the HuggingFace [`datasets.load_dataset` API](https://huggingface.co/docs/datasets/en/loading) with `streaming=True`, which take in `hf_*` parameters.
### Example config for streaming from HuggingFace Hub (no download needed):
```
dataset_type: hf
hf_path: 'allenai/c4'  # for using https://huggingface.co/datasets/allenai/c4
hf_data_dir: 'en'
hf_train_files: ''
# set eval_interval > 0 to use the specified eval dataset, otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
hf_eval_split: 'validation'
hf_eval_files: ''
# for HF pipeline, tokenizer_path can be a path in HuggingFace Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
hf_access_token: ''  # provide token if using gated dataset or tokenizer
```

### Example config for streaming from downloaded data in a GCS bucket:
```
dataset_type: hf
hf_path: 'parquet'  # or json, arrow, etc.
hf_data_dir: ''
hf_train_files: 'gs://<bucket>/<folder>/*-train-*.parquet'   # match the train files
# set eval_interval > 0 to use the specified eval dataset. Otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
hf_eval_split: ''
hf_eval_files: 'gs://<bucket>/<folder>/*-validation-*.parquet'  # match the val files
# for HF pipeline, tokenizer_path can be a path in HuggingFace Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
```
### Limitations & Recommendations
1. Streaming data directly from HuggingFace Hub may be impacted by the traffic of the server. During peak hours you may encounter "504 Server Error: Gateway Time-out". It's recommended to download the HuggingFace dataset to a GCS bucket or disk for the most stable experience.
2. Streaming data directly from HuggingFace Hub works in multihost settings with a small number of hosts. We have encountered "read time out" error with host number > 16.
3. Only supports epoch=1 at this moment.

## Grain pipeline - for determinism

### Why do we need determinism for data input pipeline?
Determinism in a data input pipeline means that the same input data always results in the same sequence of batches at each step. This is typically achieved by setting a fixed shuffle seed during pipeline initialization. In an ideal scenario, where training runs uninterrupted, this determinism is straightforward (deterministic without preemption). However, real-world distributed training environments often face preemptions due to maintenance, hardware failures, or resource constraints. 
When a preempted training run resumes, the data input pipeline is re-initialized. If the same shuffle seed is used, the pipeline restarts from the beginning, potentially re-training the model on initial data. Conversely, a new seed produces a different batch sequence, making it difficult to track which data has been seen and how often each example is used for training. This lack of control can impact model performance and reproducibility.

### How does Grain achieve determinism
Grain ensures determinism in data input pipelines by saving the pipeline's state, including dataset metadata and processed data indices, within a small JSON file in checkpoints. When a training run is resumed with the same dataset and shuffle seed, Grain restores the pipeline's exact state from the checkpoint. This enables fully deterministic, reproducible training that is resilient to disruptions.

### Cases where determinism is crucial
* **Model sensitive to repetition.** When models are sensitive to the frequency with which they encounter specific examples, precise control over the order and repetition of data during training is essential.
* **Convergence comparison.** In sensitive convergence experiments like testing quantization techniques, maintaining identical data batches between runs (e.g., quantized vs. unquantized) is essential for comparison. Determinism ensures consistency even when the runs are long and undergo saving/resuming at different steps.
* **Debug training anomalies.** When troubleshooting training spikes or anomalies, the ability to replay the exact data sequence helps distinguish between bad data batches and underlying hardware or software issues.

### Global shuffle in Grain
In HF or TFDS data pipeline, shuffle is performed by file shuffling, interleave from files, and windowshuffle using a fixed size buffer. Similar shuffle strategy applies if using Grain with [Parquet](https://arrow.apache.org/docs/python/parquet.html) format. The global shuffle feature is only available when using Grain with [ArrayRecord](https://github.com/google/array_record) (random access) format, achieved by shuffling indices globally at the beginning of each epoch and then reads the elements according to the random order. We have found this to be generally fast enough, even when using hard drives and distributed file systems.

### Using Grain
1. Grain currently supports two data formats: [ArrayRecord](https://github.com/google/array_record) (random access) and [Parquet](https://arrow.apache.org/docs/python/parquet.html) (partial random-access through row group). Only ArrayRecord format supports global shuffle mentioned above. For converting a dataset into ArrayRecord, see [instructions](https://github.com/google/array_record/tree/main/beam). Additionally, other random accessible data sources can be supported via a custom data source class ([docs](https://github.com/google/grain/blob/main/docs/data_sources.md)).
2. When the dataset is hosted on GCS bucket, grain can read it through [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse). The installation of Cloud Storage FUSE is included in [setup.sh](https://github.com/google/maxtext/blob/main/setup.sh). User then needs to mount the GCS bucket to a local path for each worker, using the script [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh). The script configs some parameters for the mount.
```
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$BUCKET_NAME MOUNT_PATH=$MOUNT_PATH [FILE_PATH=$MOUNT_PATH/my_dataset]
# FILE_PATH is optional, when provided, the script runs "ls -R" for pre-filling the metadata cache
# https://cloud.google.com/storage/docs/cloud-storage-fuse/performance#improve-first-time-reads
```
3. Set `dataset_type=grain` and set `grain_train_files` to match the ArrayRecord files via a local path since the bucket has been mounted.
4. Tune `grain_worker_count` for performance. This parameter controls the number of child process used by Grain (more details in [behind_the_scene](https://github.com/google/grain/blob/main/docs/behind_the_scenes.md), [code](https://github.com/google/grain/blob/main/grain/_src/python/grain_pool.py)). If you use a large number of workers, please check your config for gcsfuse in [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh) to avoid gcsfuse throttling.

5. For multi-source blending, you can specify multiple data sources with their respective weights using semicolon (;) as separator and colon (:) for weights. The weights will be automatically normalized to sum to 1.0. For example:
```
# Blend two data sources with 30% from first source and 70% from second source
grain_train_files=/tmp/gcsfuse/dataset1.array_record*:0.3;/tmp/gcsfuse/dataset2.array_record*:0.7

# Blend three data sources with equal weights (will be normalized to 0.33 each)
grain_train_files=/tmp/gcsfuse/dataset1.array_record*:1;/tmp/gcsfuse/dataset2.array_record*:1;/tmp/gcsfuse/dataset3.array_record*:1
```
Note: When using multiple data sources, only ArrayRecord format is supported.

6. Example command:
```
bash setup_gcsfuse.sh \
DATASET_GCS_BUCKET=maxtext-dataset \
MOUNT_PATH=/tmp/gcsfuse && \
python3 -m MaxText.train MaxText/configs/base.yml \
run_name=<RUN_NAME> base_output_directory=gs://<MY_BUCKET>  \
dataset_type=grain \
grain_file_type=arrayrecord \
grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* \
grain_worker_count=2
```
7. Using validation set for eval
When setting eval_interval > 0, eval will be run with a specified eval dataset. Example config:
```
eval_interval: 10000
grain_eval_files: '/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record*'
```

## TFDS pipeline

1. Download the Allenai c4 dataset in TFRecord format to a GCS bucket (will cost about $100, [details](https://github.com/allenai/allennlp/discussions/5056))
```
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```
2. Use the following config:
```
dataset_type: tfds
dataset_name: 'c4/en:3.0.1'
# set eval_interval > 0 to use the specified eval dataset. Otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
eval_dataset_name: 'c4/en:3.0.1'
eval_split: 'validation'
# TFDS input pipeline only supports tokenizer in spm format
tokenizer_path: "assets/tokenizer.llama2"
```
