<!--
 Copyright 2023 Google LLC

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
## Data Input Pipeline

Currently MaxText has three data input pipelines:

| Pipeline | Dataset formats | Features | Limitations |
| -------- | --------------- | -------- | ----------- |
| HuggingFace | datasets in HuggingFace Hub<br>local/Cloud Storage datasets in json, parquet, arrow, csv, txt | convenience<br>multiple formats | limit scalability using HuggingFace Hub<br>non-deterministic with preemption<br>(deterministic without preemption) |
| Grain | ArrayRecord, available through Tensorflow Datasets | fully deterministic, regardless of preemption | only supports random access datasets |
| TFDS | TFRecord, available through Tensorflow Datasets |  | only supports TFRecords<br>non-deterministic with preemption<br>(deterministic without preemption) |

Performance data for input pipeline: https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Perf.md

### HuggingFace pipeline
The HuggingFace pipeline supports streaming directly from HuggingFace Hub, or from GCS bucket in HuggingFace supported formats (parquet, json, etc.). This is through the HuggingFace [`datasets.load_dataset` API](https://huggingface.co/docs/datasets/en/loading) with `streaming=True`, which take in `hf_*` parameters.
#### Example config for streaming from HuggingFace Hub (no download needed):
```
dataset_type: hf
hf_path: 'allenai/c4'  # for using https://huggingface.co/datasets/allenai/c4
hf_data_dir: 'en'
hf_data_files: ''
# for HF pipeline, tokenizer_path can be a path in HuggingFace Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
hf_access_token: ''  # provide token if using gated dataset or tokenizer
```

#### Example config for streaming from downloaded data in a GCS bucket:
```
dataset_type: hf
hf_path: 'parquet'  # or json, arrow, etc.
hf_data_dir: ''
hf_data_files: 'gs://<bucket>/<folder>/*-train-*.parquet'  # match the train files
# for HF pipeline, tokenizer_path can be a path in HuggingFace Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
```
#### Limitations & Recommendations
1. Streaming data directly from HuggingFace Hub may be impacted by the traffic of the server. During peak hours you may encounter "504 Server Error: Gateway Time-out". It's recommended to download the HuggingFace dataset to a GCS bucket or disk for the most stable experience.
2. Streaming data directly from HuggingFace Hub works in multihost settings with a small number of hosts. We have encountered "read time out" error with host number > 16.
3. Multihost data input is achieved through HuggingFace's `datasets.distributed.split_dataset_by_node` API, which is more performant when (data shards/files) % (number of host reading data) == 0. To give MaxText users a performant experience, we follow this rule and will use data shards up to the number that's divisible by number of host reading data. For instance if your dataset has 100 shards, and you use 8 hosts, at most 96/100 shards of data will be used since 96%8==0. You can reshard or combine shards if you want to utilize the whole dataset.

### Grain pipeline - for determinism

#### Why do we need determinism for data input pipeline?
Determinism in a data input pipeline means that the same input data always results in the same sequence of batches at each step. This is typically achieved by setting a fixed shuffle seed during pipeline initialization. In an ideal scenario, where training runs uninterrupted, this determinism is straightforward (deterministic without preemption). However, real-world distributed training environments often face preemptions due to maintenance, hardware failures, or resource constraints. 
When a preempted training run resumes, the data input pipeline is re-initialized. If the same shuffle seed is used, the pipeline restarts from the beginning, potentially re-training the model on initial data. Conversely, a new seed produces a different batch sequence, making it difficult to track which data has been seen and how often each example is used for training. This lack of control can impact model performance and reproducibility.

#### How does Grain achieve determinism
Grain ensures determinism in data input pipelines by saving the pipeline's state, including dataset metadata and processed data indices, within a small JSON file in checkpoints. When a training run is resumed with the same dataset and shuffle seed, Grain restores the pipeline's exact state from the checkpoint. This enables fully deterministic, reproducible training that is resilient to disruptions.

#### Cases where determinism is crucial
* **Model sensitive to repetition.** When models are sensitive to the frequency with which they encounter specific examples, precise control over the order and repetition of data during training is essential.
* **Convergence comparison.** In sensitive convergence experiments like testing quantization techniques, maintaining identical data batches between runs (e.g., quantized vs. unquantized) is essential for comparison. Determinism ensures consistency even when the runs are long and undergo saving/resuming at different steps.
* **Debug training anomalies.** When troubleshooting training spikes or anomalies, the ability to replay the exact data sequence helps distinguish between bad data batches and underlying hardware or software issues.

#### Using Grain
1. Dataset needs to be in a format that supports random access. The default format is [ArrayRecord](https://github.com/google/array_record). For converting a dataset into ArrayRecord, see [instructions](https://github.com/google/array_record/tree/main/beam). Additionally, other random accessible data sources can be supported via a custom data source class ([docs](https://github.com/google/grain/blob/main/docs/data_sources.md)).
2. ArrayRecord dataset, when hosted on GCS bucket, can only be read through [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse). The installation of Cloud Storage FUSE is included in [setup.sh](https://github.com/google/maxtext/blob/main/setup.sh). User then needs to mount the GCS bucket to a local path for each worker, using the script [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh). The script configs some parameters for the mount.
```
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$BUCKET_NAME MOUNT_PATH=$MOUNT_PATH
```
3. Set `dataset_type=grain` and set `grain_data_files` to match the ArrayRecord files via a local path since the bucket has been mounted.
4. Tune `grain_worker_count` for performance. This parameter controls the number of child process used by Grain (more details in [behind_the_scene](https://github.com/google/grain/blob/main/docs/behind_the_scenes.md), [code](https://github.com/google/grain/blob/main/grain/_src/python/grain_pool.py)). If you use a large number of workers, please check your config for gcsfuse in [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh) to avoid gcsfuse throttling.
5. Example command:
```
bash setup_gcsfuse.sh \
DATASET_GCS_BUCKET=maxtext-dataset \
MOUNT_PATH=/tmp/gcsfuse && \
python3 MaxText/train.py MaxText/configs/base.yml \
run_name=<RUN_NAME> base_output_directory=gs://<MY_BUCKET>  \
dataset_type=grain \
grain_data_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* \
grain_worker_count=2
```

### TFDS pipeline

1. Download the Allenai c4 dataset in TFRecord format to a GCS bucket (will cost about $100, [details](https://github.com/allenai/allennlp/discussions/5056))
```
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```
2. Use the following config:
```
dataset_type: tfds
dataset_name: 'c4/en:3.0.1'
# TFDS input pipeline only supports tokenizer in spm format
tokenizer_path: "assets/tokenizer.llama2"
```
