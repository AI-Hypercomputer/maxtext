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

# Running a workload with Google Cloud ML Diagnostics Enabled
This guide provides an overview on how to enable ML Diagnostics for your MaxText workload.

## Overview
Google Cloud ML Diagnostics is an end-to-end managed platform for ML Engineers to optimize and diagnose their AI/ML workloads on Google Cloud. The product allows ML Engineers to collect and visualize all their workload metrics, configs and profiles with one single platform, all within the same UI. The current product offering focuses on workloads running on XLA-based frameworks (JAX, Pytorch XLA, Tensorflow/Keras) on Google Cloud TPUs and GPUs. Current support is for JAX on Google Cloud TPUs only. 

## Enabling ML Diagnostics on Maxtext Workload
MaxText has integrated the ML Diagnostics [SDK](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics?tab=readme-ov-file) in its code. You can enable ML Diagnostics with the **managed-mldiagnostics** flag. If this is enabled, it will

- Create a managed MachineLearning run with all the MaxText configs.
- Upload profiling traces, if the profiling is enabled by profiler="xplane".
- Upload training metrics, at the defined log_period interval.

### Examples

1.   Enable ML Diagnostics to just capture Maxtext metrics and configs

            python3 -m MaxText.train src/MaxText/configs/base.yml \
               run_name=${USER}-tpu-job \
               base_output_directory="gs://your-output-bucket/" \
               dataset_path="gs://your-dataset-bucket/" \
               steps=100 \
               log_period=10 \
               managed_mldiagnostics=True
    
2.   Enable ML Diagnostics to capture Maxtext metrics, configs and singlehost profiles (on the first TPU device)

            python3 -m MaxText.train src/MaxText/configs/base.yml \
               run_name=${USER}-tpu-job \
               base_output_directory="gs://your-output-bucket/" \
               dataset_path="gs://your-dataset-bucket/" \
               steps=100 \
               log_period=10 \
               profiler=xplane \
               managed_mldiagnostics=True
            
3.   Enable ML Diagnostics to capture Maxtext metrics, configs and multihost profiles (on all TPU devices)

            python3 -m MaxText.train src/MaxText/configs/base.yml \
               run_name=${USER}-tpu-job \
               base_output_directory="gs://your-output-bucket/" \
               dataset_path="gs://your-dataset-bucket/" \
               steps=100 \
               log_period=10 \
               profiler=xplane \
               upload_all_profiler_results=True \
               managed_mldiagnostics=True

Users can deploy the workload across all supported environments, including the standard XPK workload types (**xpk workload create** or **xpk workload create-pathways**) or by running the workload directly on a standalone TPU VM.