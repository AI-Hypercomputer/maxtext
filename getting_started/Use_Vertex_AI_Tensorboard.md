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
## Use Vertex AI Tensorboard
MaxText supports automatic upload of logs collected in a directory to a Tensorboard instance in Vertex AI. For more information on how MaxText supports this feature, visit [cloud-accelerator-diagnostics](https://pypi.org/project/cloud-accelerator-diagnostics) PyPI package documentation.

### What is Vertex AI Tensorboard and Vertex AI Experiment
Vertex AI Tensorboard is a fully managed and enterprise-ready version of open-source Tensorboard. To learn more about Vertex AI Tensorboard, visit [this](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction). Vertex AI Experiment is a tool that helps to track and analyze an experiment run on Vertex AI Tensorboard. To learn more about Vertex AI Experiments, visit [this](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments). 

You can use a single Vertex AI Tensorboard instance to track and compare metrics from multiple Vertex AI Experiments. While you can view metrics from multiple Vertex AI Experiments within a single Tensorboard instance, the underlying log data for each experiment remains separate.

### Prerequisites
* Enable [Vertex AI API](https://cloud.google.com/vertex-ai/docs/start/cloud-environment#enable_vertexai_apis) in your Google Cloud console.
* Assign [Vertex AI User IAM role](https://cloud.google.com/vertex-ai/docs/general/access-control#aiplatform.user) to the service account used by the TPU VMs. This is required to create and access the Vertex AI Tensorboard in Google Cloud console. If you are using XPK for MaxText, the necessary Vertex AI User IAM role will be automatically assigned to your node pools by XPK – no need to assign it manually.

### Upload Logs to Vertex AI Tensorboard
**Scenario 1: Using XPK to run MaxText on GKE**

XPK simplifies MaxText's Vertex AI Tensorboard integration. A Vertex Tensorboard instance and Experiment are automatically created by XPK during workload scheduling. Also, XPK automatically sets the necessary environment variables, eliminating the need to manually configure this in MaxText. Set `use_vertex_tensorboard=False` to avoid setting up Vertex Tensorboard again in MaxText. This is how the configuration will look like for running MaxText via XPK:
```
use_vertex_tensorboard: False
vertex_tensorboard_project: ""
vertex_tensorboard_region: ""
```
The above configuration will upload logs in `config.tensorboard_dir` to Vertex Tensorboard instance set as an environment variable by XPK.

**Scenario 2: Running MaxText on GCE**

Set `use_vertex_tensorboard=True` to upload logs in `config.tensorboard_dir` to a Tensorboard instance in Vertex AI. You can manually create a Tensorboard instance named `<config.vertex_tensorboard_project>-tb-instance` and an Experiment named `config.run_name` in Vertex AI on Google Cloud console. Otherwise, MaxText will create those resources for you when `use_vertex_tensorboard=True`. Note that Vertex AI is available in only [these](https://cloud.google.com/vertex-ai/docs/general/locations#available-regions) regions.

**Scenario 2.1: Configuration to upload logs to Vertex AI Tensorboard**

```
run_name: "test-run"
use_vertex_tensorboard: True
vertex_tensorboard_project: "test-project" # or vertex_tensorboard_project: ""
vertex_tensorboard_location: "us-central1"
```
The above configuration will try to create a Vertex AI Tensorboard instance named `test-project-tb-instance` and a Vertex AI Experiment named `test-run` in the `us-central1` region of `test-project`. If you set `vertex_tensorboard_project=""`, then the default project (`gcloud config get project`) set on the VM will be used to create the Vertex AI resources. It will only create these resources if they do not already exist. Also, the logs in `config.tensorboard_dir` will be uploaded to `test-project-tb-instance` Tensorboard instance and `test-run` Experiment in Vertex AI.

**Scenario 2.2: Configuration to not upload logs to Vertex AI Tensorboard**

The following configuration will not upload any log data collected in `config.tensorboard_dir` to Tensorboard in Vertex AI.
```
use_vertex_tensorboard: False
vertex_tensorboard_project: ""
vertex_tensorboard_location: ""
```
