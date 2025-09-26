# Grain pipeline
## The recommended input pipeline for determinism and resilience!

Grain is a library for reading data for training and evaluating JAX models. It’s designed to be:
* **Powerful**: Users can bring arbitrary Python transformations.
* **Flexible**: Users can readily override Grain components for their needs.
* **Deterministic**: Multiple runs of the same pipeline will produce the same outputs.
* **Resilient to preemptions**: With minimal-sized checkpoints, users can resume the dataloader from the point at which it was preempted and produce the same output as if it was never preempted.
* **Performant**: Achieved with multiprocessing with shared memory. Tested on multiple data modalities.
* **With minimal dependencies**: Does not depend on ML frameworks (Tensorflow).

Reference links: [**Grain repo**](https://github.com/google/grain) | [**Grain-Read the Docs**](https://google-grain.readthedocs.io/en/latest/index.html)

## Why determinism is important for data input pipeline?
Determinism in a data input pipeline means that the same input data always results in the same sequence of batches at each step. This is typically achieved by setting a fixed shuffle seed during pipeline initialization. In an ideal scenario, where training runs uninterrupted, this determinism is straightforward (deterministic without preemption). However, real-world distributed training environments often face preemptions due to maintenance, hardware failures, or resource constraints. 
When a preempted training run resumes, the data input pipeline is re-initialized. If the same shuffle seed is used, the pipeline restarts from the beginning, potentially re-training the model on initial data. Conversely, a new seed produces a different batch sequence, making it difficult to track which data has been seen and how often each example is used for training. This lack of control can impact model performance and reproducibility.

## How Grain achieves determinism
Grain ensures determinism in data input pipelines by saving the pipeline's state, including dataset metadata and processed data indices, within a small JSON file in checkpoints. When a training run is resumed with the same dataset and shuffle seed, Grain restores the pipeline's exact state from the checkpoint. This enables fully deterministic, reproducible training that is resilient to disruptions.

## Cases where determinism is crucial
* **Model sensitive to repetition**: When models are sensitive to the frequency with which they encounter specific examples, precise control over the order and repetition of data during training is essential. All LLMs belong to this category.
* **Convergence comparison**: In sensitive convergence experiments like testing quantization techniques, maintaining identical data batches between runs (e.g., quantized vs. unquantized) is essential for comparison. Determinism ensures consistency even when the runs are long and undergo saving/resuming at different steps.
* **Debug training anomalies**: When troubleshooting training spikes or anomalies, the ability to replay the exact data sequence helps distinguish between bad data batches and underlying hardware or software issues.

## Data shuffling
* **Global shuffle**: This feature is only available when using Grain with [ArrayRecord](https://github.com/google/array_record) (random access) format, achieved by shuffling indices globally at the beginning of each epoch and then reading the elements according to the random order. This shuffle method effectively prevents local overfitting, leading to better training results.
* **Hierarchical shuffle**: For sequential access format [Parquet](https://arrow.apache.org/docs/python/parquet.html), shuffle is performed by these steps: file shuffling, interleave from files, and window shuffle using a fixed size buffer.

## Using Grain
1. Grain currently supports two data formats: [ArrayRecord](https://github.com/google/array_record) (random access) and [Parquet](https://arrow.apache.org/docs/python/parquet.html) (partial random-access through row groups). Only the ArrayRecord format supports the global shuffle mentioned above. For converting a dataset into ArrayRecord, see [Apache Beam Integration for ArrayRecord](https://github.com/google/array_record/tree/main/beam). Additionally, other random access data sources can be supported via a custom [data source](https://google-grain.readthedocs.io/en/latest/data_sources.html) class.
2. When the dataset is hosted on a Cloud Storage bucket, Grain can read it through [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse). The installation of Cloud Storage FUSE is included in [setup.sh](https://github.com/google/maxtext/blob/main/setup.sh). The user then needs to mount the Cloud Storage bucket to a local path for each worker, using the script [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh). The script configures some parameters for the mount.
```
bash setup_gcsfuse.sh \
DATASET_GCS_BUCKET=$BUCKET_NAME \
MOUNT_PATH=$MOUNT_PATH \
[FILE_PATH=$MOUNT_PATH/my_dataset]
# FILE_PATH is optional, when provided, the script runs "ls -R" for pre-filling the metadata cache
# https://cloud.google.com/storage/docs/cloud-storage-fuse/performance#improve-first-time-reads
```
3. Set `dataset_type=grain`, `grain_file_type={arrayrecord|parquet}`, `grain_train_files` to match the file pattern on the mounted local path.
4. Tune `grain_worker_count` for performance. This parameter controls the number of child processes used by Grain (more details in [behind_the_scenes](https://google-grain.readthedocs.io/en/latest/behind_the_scenes.html), [grain_pool.py](https://github.com/google/grain/blob/main/grain/_src/python/grain_pool.py)). If you use a large number of workers, check your config for gcsfuse in [setup_gcsfuse.sh](https://github.com/google/maxtext/blob/main/setup_gcsfuse.sh) to avoid gcsfuse throttling.

5. For multi-source blending, you can specify multiple data sources with their respective weights using semicolon (;) as a separator and colon (:) for weights. The weights will be automatically normalized to sum to 1.0. For example:
```
# Blend two data sources with 30% from first source and 70% from second source
grain_train_files=/tmp/gcsfuse/dataset1.array_record*:0.3;/tmp/gcsfuse/dataset2.array_record*:0.7

# Blend three data sources with equal weights (will be normalized to 0.33 each)
grain_train_files=/tmp/gcsfuse/dataset1.array_record*:1;/tmp/gcsfuse/dataset2.array_record*:1;/tmp/gcsfuse/dataset3.array_record*:1
```
Note: When using multiple data sources, only the ArrayRecord format is supported.

6. Example command:
```
bash setup_gcsfuse.sh \
DATASET_GCS_BUCKET=maxtext-dataset \
MOUNT_PATH=/tmp/gcsfuse && \
python3 -m MaxText.train src/MaxText/configs/base.yml \
run_name=<RUN_NAME> base_output_directory=gs://<MY_BUCKET>  \
dataset_type=grain \
grain_file_type=arrayrecord # or parquet \ 
grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* \
grain_worker_count=2
```
7. Using validation set for evaluation
When setting eval_interval > 0, evaluation will be run with a specified eval dataset. Example config (set in `src/MaxText/configs/base.yml` or through command line):
```
eval_interval: 10000
eval_steps: 50
grain_eval_files: '/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record*'
```
