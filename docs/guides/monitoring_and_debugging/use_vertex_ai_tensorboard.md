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

(vertex-ai-tensorboard)=

# Use Vertex AI Tensorboard

MaxText supports automatic upload of logs collected in a directory to a Tensorboard instance in Vertex AI. For more information on how MaxText supports this feature, visit [cloud-accelerator-diagnostics](https://pypi.org/project/cloud-accelerator-diagnostics) PyPI package documentation.

## What is Vertex AI Tensorboard and Vertex AI Experiment

Vertex AI Tensorboard is a fully managed and enterprise-ready version of open-source Tensorboard. To learn more about Vertex AI Tensorboard, visit [this](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/experiments/tensorboard-introduction). Vertex AI Experiment is a tool that helps to track and analyze an experiment run on Vertex AI Tensorboard. To learn more about Vertex AI Experiments, visit [this](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/experiments/intro-vertex-ai-experiments).

You can use a single Vertex AI Tensorboard instance to track and compare metrics from multiple Vertex AI Experiments. While you can view metrics from multiple Vertex AI Experiments within a single Tensorboard instance, the underlying log data for each experiment remains separate.

## Prerequisites

- Enable [Vertex AI API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/start/cloud-environment#set_up_a_project) in your Google Cloud console.
- Assign [Vertex AI User IAM role](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/access-control#aiplatform.user) to the service account used by the TPU VMs. This is required to create and access the Vertex AI Tensorboard in Google Cloud console. If you are using Cluster Toolkit for MaxText, the necessary Vertex AI User IAM role can be granted via the cluster configuration or cluster policy path; legacy XPK setups handled this automatically.

## Upload logs to Vertex AI Tensorboard

**Scenario 1: Using Cluster Toolkit to run MaxText on GKE**

Unlike legacy XPK (which automatically created a Vertex AI TensorBoard instance and Experiment during workload scheduling and injected `UPLOAD_DATA_TO_TENSORBOARD`), Cluster Toolkit (`gcluster job submit`) does not automatically provision Vertex AI TensorBoard resources or inject `TENSORBOARD_*` environment variables.

To automatically create (or reuse) a Vertex AI TensorBoard instance named `<vertex_tensorboard_project>-tb-instance` and an Experiment named `<run_name>` and upload logs from `config.tensorboard_dir`, set `use_vertex_tensorboard=True` along with your project and region:

```yaml
run_name: "test-run"
use_vertex_tensorboard: True
vertex_tensorboard_project: "test-project" # or vertex_tensorboard_project: ""
vertex_tensorboard_region: "us-central1"
```

Alternatively, if you have already created the Vertex AI TensorBoard instance and Experiment externally (or in a legacy XPK environment that injects these variables automatically), you can set `use_vertex_tensorboard: False` in MaxText so it does not attempt to re-create the resources, and pass the required environment variables via `gcluster job submit`:

```bash
gcluster job submit \
  --env UPLOAD_DATA_TO_TENSORBOARD=1 \
  --env TENSORBOARD_PROJECT="test-project" \
  --env TENSORBOARD_REGION="us-central1" \
  --env TENSORBOARD_NAME="test-project-tb-instance" \
  --env EXPERIMENT_NAME="test-run" \
  ...
```

with the following MaxText configuration:

```yaml
use_vertex_tensorboard: False
vertex_tensorboard_project: ""
vertex_tensorboard_region: ""
```

**Scenario 2: Running MaxText on GCE**

Set `use_vertex_tensorboard=True` to upload logs in `config.tensorboard_dir` to a Tensorboard instance in Vertex AI. You can manually create a Tensorboard instance named `<config.vertex_tensorboard_project>-tb-instance` and an Experiment named `config.run_name` in Vertex AI on Google Cloud console. Otherwise, MaxText will create those resources for you when `use_vertex_tensorboard=True`. Note that Vertex AI is available in only [these](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/locations#available-regions) regions.

**Scenario 2.1: Configuration to upload logs to Vertex AI Tensorboard**

```yaml
run_name: "test-run"
use_vertex_tensorboard: True
vertex_tensorboard_project: "test-project" # or vertex_tensorboard_project: ""
vertex_tensorboard_region: "us-central1"
```

The above configuration will try to create a Vertex AI Tensorboard instance named `test-project-tb-instance` and a Vertex AI Experiment named `test-run` in the `us-central1` region of `test-project`. If you set `vertex_tensorboard_project=""`, then the default project (`gcloud config get project`) set on the VM will be used to create the Vertex AI resources. It will only create these resources if they do not already exist. Also, the logs in `config.tensorboard_dir` will be uploaded to `test-project-tb-instance` Tensorboard instance and `test-run` Experiment in Vertex AI.

**Scenario 2.2: Configuration to not upload logs to Vertex AI Tensorboard**

The following configuration (when `UPLOAD_DATA_TO_TENSORBOARD` is not set in the environment) will not upload any log data collected in `config.tensorboard_dir` to Tensorboard in Vertex AI.

```yaml
use_vertex_tensorboard: False
vertex_tensorboard_project: ""
vertex_tensorboard_region: ""
```
