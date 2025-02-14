<!--
 Copyright 2025 Google LLC

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
## ML Goodput Measurement
MaxText supports automatic measurement and upload of workload metrics such as Goodput, Badput Breakdown and Step Time Deviation using the ML Goodput Measurement library.

The [ML Goodput Measurement](https://github.com/AI-Hypercomputer/ml-goodput-measurement) library currently supports monitoring workloads running on Google Cloud Platform. For more information on details of the library, visit the Github page or the [ml-goodput-measurement](https://pypi.org/project/ml-goodput-measurement/) PyPI package documentation.

### What is Goodput
Goodput is the metric that measures the efficiency of model training jobs, i.e. productive time spent on training progress proportional to the total time spent by the workload. It is an actionable way for users to monitor where they can improve to get the most value from their accelerators. 

### What is Badput
Badput is the metric that measures time that a workload spent on anything that is not productive training proportional to the total time spent by the workload. For example, the time spent in accelerator initialization, training preparation, program startup, data loading, portions of checkpointing, disruptions and wasted progress since the last checkpoint etc. all contribute to Badput. 

The ML Goodput Measurement library exposes Badput Breakdown. Further details of each bucket can be found [here](https://github.com/AI-Hypercomputer/ml-goodput-measurement?tab=readme-ov-file#badput-breakdown-details)

### What is Step Time Deviation

Step Time Deviation is the metric that measures deviation of step time from ideal step time.

The ML Goodput Measurement library exposes step time deviation by computing ideal step time or allowing users to configure ideal step time.


### Prerequisites
The usage of this package requires the setup of a Google Cloud project with
billing enabled to properly use Google Cloud Logging. If you don't have a Google
Cloud project, or if you don't have billing enabled for your Google Cloud
project, then do the following:

1. In the Google Cloud console, on the project selector page,
 [select or create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

2. Make sure that billing is enabled for your Google Cloud project. Instructions can be found [here](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console)

3. Enable the [Cloud Logging API]((https://console.cloud.google.com/flows/enableapi?apiid=logging.googleapis.com&_ga=2.27841276.1571868865.1726250448-123998259.1726107009) ).

4. To run your training on Cloud accelerator, set up the environment by following instructions [here](https://cloud.google.com/tpu/docs/setup-gcp-account).

5. To learn more about Google Cloud Logging, visit this [page](https://cloud.google.com/logging/docs).

### Access Scopes

 You will need both read and write access scopes for cloud logging on both the
 GPU or TPU and CPU node pools. Full cloud logging access is granted by the
 following access scope during node pool creation:

  - `https://www.googleapis.com/auth/cloud-platform`

   XPK adds this access scope to the GPU, TPU and CPU node pools, so XPK is the recommended method to create clusters and node-pools in you intend to run your workloads on GKE.

   Instructions on how to create clusters using XPK can be
   found [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#cluster-create) and how to create workloads using XPK can be found
   [here](https://github.com/AI-Hypercomputer/xpk/blob/main/README.md#workload-create).

   > **_NOTE:_** Access Scopes are immutable and workloads can only be migrated
  to new node pools with required access scopes. Access scopes on already created clusters cannot be updated.

### How to Monitor Goodput and Badput

MaxText enables Goodput recording and monitoring by default with `enable_goodput_recording=True` and `monitor_goodput=True`. You can configure the goodput upload frequency by setting `goodput_upload_interval_seconds`.

```Python
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH run_name=goodput-test-run steps=200 goodput_upload_interval_seconds=30
```

### How to Monitor Step Time Deviation

MaxText enables step time deviation monitoring by default with `monitor_step_time_deviation=True`. You can configure the upload frequency by setting `step_deviation_interval_seconds`.

```Python
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH run_name=goodput-test-run steps=200 step_deviation_interval_seconds=30
```

### How to enable Pathways Goodput

MaxText disables Pathways by default for computation of all Goodput metrics with `enable_pathways_goodput=False`. You can enable Pathways Goodput by setting this flag to true.

> **_NOTE:_** Enabling `enable_pathways_goodput` turns on Goodput measurement for Pathways workloads, and does not update any Pathways features.

```Python
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH run_name=goodput-test-run steps=200 goodput_upload_interval_seconds=30 enable_pathways_goodput=True
```

### Visualize on Tensorboard

1. MaxText installs the required packages on setup: `tensorboard-plugin-profile`, `tensorflow` and `tensorboard`.
2. Follow the Tensorboard URL on MaxText logs to view all metrics in one location.