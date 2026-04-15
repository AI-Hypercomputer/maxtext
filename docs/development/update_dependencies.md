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
   `src/dependencies/requirements/base_requirements/requirements.txt` or the hardware-specific files
   (`src/dependencies/requirements/base_requirements/tpu-base-requirements.txt`,
   `src/dependencies/requirements/base_requirements/gpu-base-requirements.txt`).
2. **Find the JAX build commit hash**: The dependency generation process is
   pinned to a specific nightly build of JAX. You need to find the commit hash
   for the desired JAX build.
3. **Generate the requirement files**: Run the `seed-env` CLI tool to generate
   new, fully-pinned requirements files based on your changes.
4. **Update project files**: Copy the newly generated files into the
   `src/dependencies/requirements/generated_requirements/` directory. If
   necessary, also update any dependencies that are installed directly from
   GitHub from the generated files to `src/dependencies/extra_deps`.
5. **Verify the new dependencies**: Test the new dependencies to ensure the
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

Update the desired dependencies in
`src/dependencies/requirements/base_requirements/requirements.txt` or the
hardware-specific files
(`src/dependencies/requirements/base_requirements/tpu-base-requirements.txt`,
`src/dependencies/requirements/base_requirements/gpu-base-requirements.txt`).

## Step 2: Find the JAX build commit hash

The dependency generation process is pinned to a specific nightly build of JAX.
You need to find the commit hash for the desired JAX build.

You can find the latest commit hashes in the
[JAX `build/` folder](https://github.com/jax-ml/jax/commits/main/build). Choose
a recent, successful build and copy its full commit hash.

## Step 3: Generate the requirements files

Next, run the `seed-env` CLI to generate the new requirements files. You will
need to do this separately for the TPU and GPU environments. The generated files
will be placed in a directory specified by `--output-dir`.

### For TPU

Run the following command, replacing `<jax-build-commit-hash>` with the hash you
copied in the previous step.

```bash
seed-env \
  --local-requirements=src/dependencies/requirements/base_requirements/tpu-base-requirements.txt \
  --host-name=MaxText \
  --seed-commit=<jax-build-commit-hash> \
  --python-version=3.12 \
  --requirements-txt=tpu-requirements.txt \
  --output-dir=generated_tpu_artifacts
```

### For GPU

Similarly, run the command for the GPU requirements.

```bash
seed-env \
  --local-requirements=src/dependencies/requirements/base_requirements/gpu-base-requirements.txt \
  --host-name=MaxText \
  --seed-commit=<jax-build-commit-hash> \
  --python-version=3.12 \
  --requirements-txt=cuda12-requirements.txt \
  --hardware=cuda12 \
  --output-dir=generated_gpu_artifacts
```

## Step 4: Update project files

After generating the new requirements, you need to update the files in the
MaxText repository.

1. **Copy the generated files:**

   - Move `generated_tpu_artifacts/tpu-requirements.txt` to `generated_requirements/tpu-requirements.txt`.
   - Move `generated_gpu_artifacts/cuda12-requirements.txt` to `generated_requirements/cuda12-requirements.txt`.

2. **Update `src/dependencies/extra_deps` (if necessary):**
   Currently, MaxText uses a few dependencies, such as `mlperf-logging` and
   `google-jetstream`, that are installed directly from GitHub source. These are
   defined in `base_requirements/requirements.txt`, and the `seed-env` tool will
   carry them over to the generated requirements files.

## Step 5: Verify the new dependencies

Finally, test that the new dependencies install correctly and that MaxText runs
as expected.

1. **Install MaxText:** Follow the instructions to
   [install MaxText from source](install-from-source).

2. **Run tests:** Run MaxText tests to ensure there are no regressions.
