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
# Manage the data input pipeline

Currently MaxText has three data input pipelines:

| Pipeline | Dataset formats | Features | Limitations |
| -------- | --------------- | -------- | ----------- |
| **[Grain](data_input_grain.md)** (recommanded)| [ArrayRecord](https://github.com/google/array_record) (random access, available through [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview), or [conversion](https://github.com/google/array_record/tree/main/beam))<br>[Parquet]((https://arrow.apache.org/docs/python/parquet.html)) (sequential access) | With arrayrecord: fully deterministic, resilient to preemption; global shuffle <br>With parquet: performant; fully deterministic, resilient to preemption; hierarchical shuffle |  |
| **[Hugging Face](data_input_hf.md)** | datasets in [Hugging Face Hub](https://huggingface.co/datasets)<br>local/Cloud Storage datasets in json, parquet, arrow, csv, txt (sequential access) | no download needed, convenience; <br>multiple formats | limit scalability using the Hugging Face Hub (no limit using Cloud Storage); <br>non-deterministic with preemption<br>(deterministic without preemption)<br> |
| **[TFDS](data_input_hf.md)** | TFRecord (sequential access), available through [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview) | performant | only supports TFRecords; <br>non-deterministic with preemption<br>(deterministic without preemption) |


## Multihost dataloading best practice
Training in a multi-host environment presents unique challenges for data input pipelines. An effective data loading strategy must address three key issues:
1. **Concurrent access**: Multiple hosts need to read from the same dataset simultaneously without causing conflicts.
2. **Data uniqueness**: Each host must be fed a unique, non-overlapping subset of the data to ensure the model sees each example correctly.
3. **Uneven completion**: Handling the scenario where some hosts run out of data before others, which can lead to hanging. 
The approaches to solve these challenges depend on whether your dataset supports random access or is limited to sequential access.

### Random access dataset (Recommended)
Random-access formats are highly recommended for multi-host training because they allow any part of the file to be read directly by its index.<br>
In MaxText, this is best supported by the ArrayRecord format using the Grain input pipeline. This approach gracefully handles the key challenges:
* **Concurrent access and uniqueness**: Grain assigns a unique set of indices to each host. ArrayRecord allows different hosts to read from different indices in the same file.
* **Uneven completion**: Data indices are distributed evenly among hosts. Without packing, the data imbalance between hosts will be at most one batch. To handle the final steps where some hosts run out of data, you can enable the `generate_padding_example` flag. This directs hosts to generate empty "padding" batches until the training or evaluation steps are met. **Note**: When sequence packing is enabled, the difference in the number of packed examples per host can be larger. The `generate_padding_example` flag still solves this. However, as more hosts begin generating padding, you will observe a decrease in total_weights and a slower change in the training loss. If all hosts exhaust their data before the target step count is reached, both total_weights and loss will drop to 0.

### Sequential access dataset
* **Concurrent access and uniqueness**: Sequential-access datasets (e.g., Parquet, JSON, TFRecord) cannot be accessed by index, requiring a different strategy -- file-based sharding, where each host is given exclusive access to a specific subset of data files. **Key requirement**: `(Number of data files) % (Number of data-loading hosts) == 0`.  If the file count isn't a multiple of the host count, the files will be distributed unevenly. For example, with 10 files and 8 hosts, some hosts will get two files while others get one, significantly worsening the "uneven completion" problem. If you have fewer files than hosts, performance will be severely degraded as all hosts are concurrently accessing all the files.
* **Uneven completion**: Similar to random-access datasets, you can use the `generate_padding_example` flag to handle hosts that finish their file shards early (currently only supported in Hugging Face pipeline, not available in TFDS pipeline). 



