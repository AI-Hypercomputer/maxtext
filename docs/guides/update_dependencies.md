<!--
 Copyright 2023-2025 Google LLC

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

# How to update MaxText dependencies using seed-env

## Introduction

This document provides a guide to updating dependencies in MaxText using the `seed-env` tool. `seed-env` helps generate deterministic and reproducible Python environments by creating fully-pinned `requirements.txt` files from a base set of requirements.

## Overview of the Process

To update dependencies, you will follow these general steps:

1.  **Modify Base Requirements**: Update the desired dependencies in `base_requirements/requirements.txt` or the hardware-specific files (`base_requirements/tpu-base-requirements.txt`, `base_requirements/gpu-base-requirements.txt`).
2.  **Generate New Files**: Run the `seed-env` CLI tool to generate new, fully-pinned requirements files based on your changes.
3.  **Update Project Files**: Copy the newly generated files into the `generated_requirements/` directory.
4.  **Handle GitHub Dependencies**: Move any dependencies that are installed directly from GitHub from the generated files to `src/install_maxtext_extra_deps/extra_deps_from_github.txt`.
5.  **Verify**: Test the new dependencies to ensure the project installs and runs correctly.

The following sections provide detailed instructions for each step.

## Step 1: Install seed-env

First, you need to install the `seed-env` command-line tool by running `pip install seed-env uv`. Or follow the instructions in the
[seed-env repository](https://github.com/google-ml-infra/actions/tree/main/python_seed_env#install-the-seed-env-tool) if you want to build `seed-env` from source.

## Step 2: Find the JAX Build Commit Hash

The dependency generation process is pinned to a specific nightly build of JAX. You need to find the commit hash for the desired JAX build.

You can find the latest commit hashes in the [JAX `build/` folder](https://github.com/jax-ml/jax/commits/main/build). Choose a recent, successful build and copy its full commit hash.

## Step 3: Generate the Requirements Files

Next, run the `seed-env` CLI to generate the new requirements files. You will need to do this separately for the TPU and GPU environments. The generated files will be placed in a directory specified by `--output-dir`.

### For TPU

Run the following command, replacing `<jax-build-commit-hash>` with the hash you copied in the previous step.

```bash
seed-env \
  --local-requirements=base_requirements/tpu-base-requirements.txt \
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
  --local-requirements=base_requirements/cuda12-base-requirements.txt \
  --host-name=MaxText \
  --seed-commit=<jax-build-commit-hash> \
  --python-version=3.12 \
  --requirements-txt=cuda12-requirements.txt \
  --hardware=cuda12 \
  --output-dir=generated_gpu_artifacts
```

## 4. Update Project Files

After generating the new requirements, you need to update the files in the MaxText repository.

1.  **Copy the generated files:**
    -   Move `generated_tpu_artifacts/tpu-requirements.txt` to `generated_requirements/tpu-requirements.txt`.
    -   Move `generated_gpu_artifacts/cuda12-requirements.txt` to `generated_requirements/cuda12-requirements.txt`.

2.  **Update `extra_deps_from_github.txt` (if necessary):**
    Currently, MaxText uses a few dependencies, such as `mlperf-logging` and `google-jetstream`, that are installed directly from GitHub source. These are defined in `base_requirements/requirements.txt`, and the `seed-env` tool will carry them over to the generated requirements files.

## 5. Verify the New Dependencies

Finally, test that the new dependencies install correctly and that MaxText runs as expected.

1.  **Create a clean environment:** It's best to start with a fresh Python virtual environment.

```bash
uv venv --python 3.12 --seed maxtext_venv
source maxtext_venv/bin/activate
```

2.  **Run the setup script:** Execute `bash setup.sh` to install the new dependencies.

```bash
pip install uv
# install the tpu package
uv pip install -e .[tpu] --resolution=lowest
# or install the gpu package by running the following line:
# uv pip install -e .[cuda12] --resolution=lowest
install_maxtext_github_deps
```

3.  **Run tests:** Run MaxText tests to ensure there are no regressions.