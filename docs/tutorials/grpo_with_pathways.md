<!--
 Copyright 2024 Google LLC

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

# Try GRPO with Pathways!

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the Llama3.1 70B-IT model on the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can enhance your model's problem-solving skills on mathematical word problems, coding problems, etc.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group's performance to update the policy.

We use Tunix as the library for GRPO. 
And we use vLLM as the library for efficient model inference and generation.

Furthermore, we use Pathways for [orchestration](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro). Using Pathways, you can also run GRPO in a disaggregated mode where the trainer and the samplers are running on separate mesh. Try out the following recipe `v5p-64`. You can submit jobs to a Pathways enabled GKE cluster.
 
## Build and Upload MaxText Docker Image with Tunix, vLLM, tpu-commons dependencies
Run the following bash script to create a docker image with all the dependencies of MaxText, Tunix, vLLM and tpu-commons installed.

In addition to MaxText dependencies, 

1. It installs `pip install keyring keyrings.google-artifactregistry-auth` which enables pip to authenticate with Google Artifact Registry automatically.
2. Next, it installs `vLLM` for Jax and TPUs from the artifact registry `https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/`
3. Then, it installs `tpu-commons` from the same artifact registry.


`tpu_commons` is the TPU backend for vLLM. You will need both libraries to run vLLM on tpus.
We use the scheduler code from vLLM, and the model runner code from `tpu_commons`

```
bash docker_build_dependency_image.sh MODE=post-training
```

You can also use `bash docker_build_dependency_image.sh MODE=post-training-experimental` to try out new features via experimental dependencies such as improved pathwaysutils resharding API



### Upload the dependency docker image along with MaxText code
```
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=path/to/gcr.io
```

### Submit your jobs

Please use a pathways enabled [XPK](https://github.com/AI-Hypercomputer/xpk) cluster, and you can submit the `train_rl.py` script via [XPK](https://github.com/AI-Hypercomputer/xpk).
Note: pass overrides as `key=value` after the YAML path (do not use `--flags`).
```
xpk workload create-pathways --workload $WORKLOAD \
--docker-image path/to/gcr.io:latest --cluster $TPU_CLUSTER \
--tpu-type=$TPU_TYPE --num-slices=1  --zone=$ZONE \
--project=$PROJECT_ID --priority=high \
--command "HF_TOKEN=$HF_TOKEN TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' # Llama3.1-70B-Instruct
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=llama3.1-70b \
  tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=$WORKLOAD \
  base_output_directory=$OUTPUT_PATH \
  hf_access_token=$HF_TOKEN"
```

The overview of the demo script ~/maxtext/src/MaxText/examples/grpo_llama3_1_70b_demo_pw.py` is as follows:

1. We load a policy model and a reference model. Both are copies of `Llama3.1-70b-Instruct`.
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO with potentially different meshes for trainer and rollout depending on the parameters `TRAINER_DEVICES_FRACTION` and `SAMPLER_DEVICES_FRACTION`. If we set both of these to `1.0`, the entire (same) mesh will be used for both trainer and rollout. If we set say `TRAINER_DEVICES_FRACTION=0.5` and `SAMPLER_DEVICES_FRACTION=0.5`, the first half of the devices will be used for trainer and the second half will be used for rollout
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark after the post-training with GRPO.
