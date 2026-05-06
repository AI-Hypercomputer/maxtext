<!--
# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Update MaxText dependencies

## Introduction

This document provides a guide to updating dependencies in MaxText using the
[seed-env](https://github.com/google-ml-infra/actions/tree/main/python_seed_env)
tool. `seed-env` helps generate deterministic and reproducible Python
environments by creating fully-pinned `requirements.txt` files from a base set
of requirements.

Please keep dependencies updated throughout development. This will allow each
commit to work properly from both a feature and dependency perspective. We will
periodically upload commits to PyPI for stable releases. But it is also critical
to keep dependencies in sync for users installing MaxText from source.

## Overview of the process

To update dependencies, you will follow these general steps:

1. **Modify base requirements**: Update the desired dependencies in
   `src/dependencies/requirements/base_requirements/requirements.txt` or the hardware-specific pre-training files
   (`src/dependencies/requirements/base_requirements/tpu-requirements.txt`,
   `src/dependencies/requirements/base_requirements/cuda12-requirements.txt`) or the post-training files (`src/dependencies/requirements/base_requirements/tpu-post-train-requirements.txt`).
2. **Find the JAX build commit hash**: The dependency generation process is
   pinned to a specific nightly build of JAX. You need to find the commit hash
   for the desired JAX build.
3. **Generate the requirement files**: Run `src/dependencies/scripts/generate_requirements.sh`,
   which internally invokes `seed-env` to produce fully-pinned requirements files.
4. **Verify the new dependencies**: Test the new dependencies to ensure the
   project installs and runs correctly.

The following sections provide detailed instructions for each step.

## Step 0: Install `seed-env`

First, you need to install the `seed-env` command-line tool. We recommend
installing `uv` first following
[uv's official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
and then using it to install `seed-env`:

```bash
uv venv --python 3.12 --seed seed_venv
source seed_venv/bin/activate
uv pip install seed-env
```

Alternatively, follow the instructions in the
[seed-env repository](https://github.com/google-ml-infra/actions/tree/main/python_seed_env#install-the-seed-env-tool)
if you want to build `seed-env` from source.

## Step 1: Modify base requirements

Update the desired dependencies in `src/dependencies/requirements/base_requirements/requirements.txt` or the hardware-specific pre-training files (`src/dependencies/requirements/base_requirements/tpu-requirements.txt`, `src/dependencies/requirements/base_requirements/cuda12-requirements.txt`) or the post-training files (`src/dependencies/requirements/base_requirements/tpu-post-train-requirements.txt`).

## Step 2: Find the JAX build commit hash

The dependency generation process is pinned to a specific nightly build of JAX. You need to find the commit hash for the desired JAX build from [JAX `build/` folder](https://github.com/jax-ml/jax/commits/main/build) and copy its full commit hash.

## Step 3: Generate the requirements files

Next, run `generate_requirements.sh` to generate the new requirements files. This
script wraps the `seed-env` CLI and handles exporting the lock, and applying any
overrides. You will need to do this separately for the TPU and GPU environments.

> **Note:** The current `src/dependencies/requirements/generated_requirements/` in the repository were generated using JAX build commit hash: [e0d2967b50abbefd651d563dbcd7afbcb963d08c](https://github.com/jax-ml/jax/commit/e0d2967b50abbefd651d563dbcd7afbcb963d08c).

### TPU Pre-Training

If you have made changes to TPU pre-training dependencies in `src/dependencies/requirements/base_requirements/tpu-requirements.txt`, you need to regenerate the pinned pre-training requirements in `generated_requirements/` directory. Run the following command, replacing `<jax-build-commit-hash>` with the hash you copied in the previous step:

```bash
bash src/dependencies/scripts/generate_requirements.sh \
--base-requirements src/dependencies/requirements/base_requirements/tpu-requirements.txt \
--generated-requirements tpu-requirements.txt \
--override-requirements src/dependencies/extra_deps/tpu_overrides.txt \
--seed-commit <jax-build-commit-hash>

# Copy generated requirements to src/dependencies/requirements/generated_requirements
mv generated_artifacts/python3_12/tpu-requirements.txt \
  src/dependencies/requirements/generated_requirements/tpu-requirements.txt
```

### TPU Post-Training

If you have made changes to the post-training dependencies in `src/dependencies/requirements/base_requirements/tpu-post-train-requirements.txt`, you need to regenerate the pinned post-training requirements in `generated_requirements/` directory. Run the following command, replacing `<jax-build-commit-hash>` with the hash you copied in the previous step:

```bash

bash src/dependencies/scripts/generate_requirements.sh \
--base-requirements src/dependencies/requirements/base_requirements/tpu-post-train-requirements.txt \
--generated-requirements tpu-post-train-requirements.txt \
--override-requirements src/dependencies/extra_deps/tpu_post_train_overrides.txt \
--seed-commit <jax-build-commit-hash>

# Copy generated requirements to src/dependencies/requirements/generated_requirements
mv generated_artifacts/python3_12/tpu-post-train-requirements.txt \
  src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt
```

### GPU Pre-Training

If you have made changes to the GPU pre-training dependencies in `src/dependencies/requirements/base_requirements/cuda12-requirements.txt`, you need to regenerate the pinned pre-training requirements in `generated_requirements/` directory. Run the following command, replacing `<jax-build-commit-hash>` with the hash you copied in the previous step:

```bash
bash src/dependencies/scripts/generate_requirements.sh \
--base-requirements src/dependencies/requirements/base_requirements/cuda12-requirements.txt \
--generated-requirements cuda12-requirements.txt \
--seed-commit <jax-build-commit-hash> \
--hardware cuda12

# Copy generated requirements to src/dependencies/requirements/generated_requirements
mv generated_artifacts/python3_12/cuda12-requirements.txt \
  src/dependencies/requirements/generated_requirements/cuda12-requirements.txt
```

## Step 4: Verify the new dependencies

Finally, test that the new dependencies install correctly and that MaxText runs
as expected.

1. **Install MaxText and dependencies**: For instructions on installing MaxText on your VM, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/install_maxtext.html#from-source).

2. **Run tests:** Run MaxText tests to ensure there are no regressions.
