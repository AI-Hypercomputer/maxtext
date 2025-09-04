# Hugging Face pipeline
The Hugging Face pipeline supports streaming directly from the Hugging Face Hub, or from a Cloud Storage bucket in Hugging Face supported formats (parquet, json, etc.). This is through the Hugging Face [`datasets.load_dataset` API](https://huggingface.co/docs/datasets/en/loading) with `streaming=True`, which takes in `hf_*` parameters.
## Example config for streaming from Hugging Face Hub (no download needed):
In `src/MaxText/configs/base.yml` or through command line, set the following parameters:
```
dataset_type: hf
hf_path: 'allenai/c4'  # for using https://huggingface.co/datasets/allenai/c4
hf_data_dir: 'en'
hf_train_files: ''
# set eval_interval > 0 to use the specified eval dataset, otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
hf_eval_split: 'validation'
hf_eval_files: ''
# for HF pipeline, tokenizer_path can be a path in Hugging Face Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
hf_access_token: ''  # provide token if using gated dataset or tokenizer
```

## Example config for streaming from downloaded data in a Cloud Storage bucket:
In `src/MaxText/configs/base.yml` or through command line, set the following parameters:
```
dataset_type: hf
hf_path: 'parquet'  # or json, arrow, etc.
hf_data_dir: ''
hf_train_files: 'gs://<bucket>/<folder>/*-train-*.parquet'   # match the train files
# set eval_interval > 0 to use the specified eval dataset. Otherwise, only metrics on the train set will be calculated.
eval_interval: 10000
hf_eval_split: ''
hf_eval_files: 'gs://<bucket>/<folder>/*-validation-*.parquet'  # match the val files
# for Hugging Face pipeline, tokenizer_path can be a path in Hugging Face Hub, 
# or a local path containing tokenizer in a format supported by transformers.AutoTokenizer
tokenizer_path: 'google-t5/t5-large'  # for using https://huggingface.co/google-t5/t5-large
```
## Limitations and Recommendations
1. Streaming data directly from Hugging Face Hub may be impacted by the traffic of the server. During peak hours you may encounter "504 Server Error: Gateway Time-out". It's recommended to download the Hugging Face dataset to a Cloud Storage bucket or disk for the most stable experience.
2. Streaming data directly from Hugging Face Hub works in multi-host settings with a small number of hosts. With a host number larger than 16, you might encounter a "read time out" error.
3. Only supports num_epoch=1 at the moment.
