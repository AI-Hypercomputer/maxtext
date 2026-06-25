# TFDS pipeline

The TensorFlow Datasets (TFDS) pipeline uses datasets in TFRecord format, which is performant and widely supported in the TensorFlow ecosystem.

## Example config for streaming from TFDS dataset in a Cloud Storage bucket

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

### Tokenizer support

TFDS pipeline supports three tokenizer types:

- `sentencepiece`: For SentencePiece tokenizers
- `huggingface`: For HuggingFace tokenizers (requires `hf_access_token` for gated models)
- `tiktoken`: For OpenAI's tiktoken tokenizers
