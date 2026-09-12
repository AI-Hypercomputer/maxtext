# TFDS pipeline (Deprecated)

```{warning}
The TFDS input pipeline (`dataset_type=tfds`) is **deprecated**. We recommend migrating to the [Grain pipeline](data_input_grain.md) with `dataset_type=grain` and `grain_file_type=tfrecord`. You can keep the same TFRecord dataset paths.
```

````{note}
TensorFlow and TensorFlow Datasets (TFDS) are optional dependencies in MaxText. If you need to use the legacy TFDS pipeline, install the optional dependencies by running:

```bash
install_tpu_pre_train_extra_deps --with-tf
# or for GPU:
# install_cuda12_pre_train_extra_deps --with-tf
```
````

1. Download the Allenai C4 dataset in TFRecord format to a Cloud Storage bucket. For information about cost, see [this discussion](https://github.com/allenai/allennlp/discussions/5056)

```shell
bash download_dataset.sh {GCS_PROJECT} {GCS_BUCKET_NAME}
```

2. In [`src/maxtext/configs/base.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml) or through command line, set the following parameters:

```yaml
dataset_type: tfds
dataset_name: 'c4/en:3.0.1'
# set eval_interval > 0 to use the specified eval dataset. Otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
eval_dataset_name: 'c4/en:3.0.1'
eval_split: 'validation'
# TFDS input pipeline only supports tokenizer in spm format
tokenizer_path: 'src/maxtext/assets/tokenizers/tokenizer.llama2'
```
