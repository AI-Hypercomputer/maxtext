<!--
# Copyright 2026 Google LLC
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

# CI Docker image pipelines

This page is for MaxText maintainers and contributors with write access to the
`AI-Hypercomputer/maxtext` repository. It explains when the CI builds Docker
images, and how to re-run the image-based tests without paying for a full
rebuild. To build images on your own machine, see
[Build and upload MaxText Docker images](../tutorials/build_maxtext.md).

## When are Docker images built?

Exactly one workflow builds images: `.github/workflows/build_and_push_docker_image.yml`.
It is a reusable workflow (`workflow_call`), so it never starts on its own. Three
workflows call it, and these are the only events that build an image:

| Caller                                                        | Trigger                                              | Images                                                                                                       |
| ------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| TPU Docker Images Pipeline (`tpu_docker_images_pipeline.yml`) | Nightly at 00:00 UTC, or manual **Run workflow**     | `maxtext_jax_stable`, `maxtext_jax_nightly`, `maxtext_post_training_stable`, `maxtext_post_training_nightly` |
| GPU Docker Images Pipeline (`gpu_docker_images_pipeline.yml`) | Nightly at 00:00 UTC, or manual **Run workflow**     | `maxtext_gpu_jax_stable`, `maxtext_gpu_jax_nightly`                                                          |
| Release Pipeline (`release_pipeline.yml`)                     | GitHub release published, or manual **Run workflow** | Release images. Always a full build.                                                                         |

Nothing else builds an image:

- A `git push` or a pull request never builds an image. The PR pipeline
  (`ci_pipeline.yml`) builds a wheel from your commit (about 35 seconds), starts a
  container from the prebuilt `maxtext-unit-test-tpu` / `maxtext-unit-test-cuda12`
  base image, installs the wheel into a fresh virtual environment (about 20 seconds)
  and runs the tests. Pushing a fix re-runs everything in roughly 1.5 minutes plus
  test time, and a push cancels the previous in-progress run for the same PR.
- The `maxtext-unit-test-*` base images contain only system packages, Python, pip
  and uv. They are built by hand from `src/dependencies/dockerfiles/clean_py_env*.Dockerfile`
  and pushed rarely.

Every image pipeline run pushes
`us-docker.pkg.dev/<project>/<repository>/<image>:<run_id>` **before** the tests
run, then runs the CI test suite against that tag and, on success, adds the tag
`verified-ci-<run_id>`. A failed test run does not delete the image, so it can be
reused by the modes below.

## Re-running tests without rebuilding images (fast-rebuild)

When an image pipeline run fails because of a flaky test or a code bug, you often
want to re-run the tests with the **exact same dependencies** but updated code,
without waiting for a full image rebuild.

### Three execution modes

The TPU and GPU Docker Image Pipelines support three modes:

| Mode                                   | What it does                                 | When to use                                                   |
| -------------------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| **build-all** (default)                | Full image build from scratch                | Nightly schedule, first build on a branch, dependency changes |
| **fast-rebuild**                       | Overlay new code on a previous run's image   | Flaky test rerun with a fix, code-only iteration              |
| **Re-run failed jobs** (GitHub native) | Re-run only failed jobs using the same image | Same code, same image, just retry                             |

How to choose:

- Flaky test, code unchanged: open the failed run and click **Re-run failed jobs**.
  The run keeps its `run_id`, so it reuses the `<image>:<run_id>` already in the
  registry. No build at all.
  Needs the run's commit to still be on a branch. After a squash or force-push the
  re-run fails at startup; dispatch a fast-rebuild from that run instead.
- Code fix, dependency files unchanged: **fast-rebuild** with `source_run_id` set to
  the failed run (or to the latest nightly run on `main`).
- Anything under `src/dependencies/`, `pyproject.toml` or `.dockerignore` changed:
  **build-all**.

### Using fast-rebuild

From the GitHub UI (Actions tab):

1. Go to **Actions** → **TPU Docker Images Pipeline** (or GPU).
2. Click **"Run workflow"** and pick your branch.
3. Set **mode** to `fast-rebuild`.
4. Enter the **source_run_id**: the run ID of the previous pipeline run whose image
   you want to reuse as the dependency base. It is the last number in the run's
   URL, for example `https://github.com/.../actions/runs/34793910551` →
   `34793910551`.
5. Click **"Run workflow"**.

From the command line with the [GitHub CLI](https://cli.github.com/):

```bash
# Find the run ID of the source image (the run whose dependencies you want to reuse)
gh run list --workflow=tpu_docker_images_pipeline.yml --limit 5

# Rebuild the images with the code on <branch> on top of that run's dependency layers
gh workflow run tpu_docker_images_pipeline.yml --ref <branch> \
  -f mode=fast-rebuild -f source_run_id=<run_id>
```

Use `gpu_docker_images_pipeline.yml` for the GPU images. The new run appears in the
Actions list as **Fast-rebuild from run \<run_id>**.

Other dispatch inputs:

- `repository_name`: the Artifact Registry repository the images are pushed to
  (`maxtext-images` for the nightly images). A failure issue is only opened for
  `maxtext-images`. In fast-rebuild mode the source run must have pushed to the
  same repository.
- `allow_dependency_drift`: overrides the dependency drift guard. Not recommended,
  see below.
- `run_e2e_tests`: also runs the Airflow end-to-end tests after the build.

The pipeline will:

- Build a new wheel from your current code
- Run the dependency drift guard as a lightweight pre-flight job, so a bad request
  fails before a build runner starts
- Verify the source image exists in the registry
- Build a new image that uses the source image's dependency layers with your new
  code overlaid
- Run the full CI test suite against the new image

### Typical workflow on a feature branch

1. First run on the branch, or after changing dependency files: dispatch with
   `mode=build-all`.
2. Tests fail. Fix the code and push. Dispatch again with `mode=fast-rebuild` and
   `source_run_id` set to the run from step 1. Only the code layers are rebuilt; the
   dependencies are the same bytes as in step 1.
3. A test flaked and the code did not change: **Re-run failed jobs** on that run.
4. Another fix after step 2: dispatch `mode=fast-rebuild` again with `source_run_id`
   set to the step 2 run. A fast-rebuild run can be the source of the next one, so
   one full build carries a branch through many fixes.

If your branch does not touch dependency files, skip step 1 and use the latest
nightly run on `main` as `source_run_id`. The drift guard verifies that the two
commits have identical dependency files before anything is built.

### How it works

Fast-rebuild starts from the source run's image by digest
(`FROM <source_image>@sha256:<digest>`), so the OS packages, Python dependencies and
JAX/XLA layers are reused byte-for-byte. Only the code directories (`src/maxtext/`,
`tests/`, `benchmarks/`, `pytest.ini`) come from your commit. There are two overlay
Dockerfiles under `src/dependencies/dockerfiles/`, and the drift guard picks one:

- **Fast** (`maxtext_code_overlay_fast.Dockerfile`): `COPY --link` only, no `RUN`.
  BuildKit never pulls the source image's layers and only uploads the new code
  layers. Used whenever no code file was deleted or renamed since the source commit
  and, for post-training images, `src/maxtext/integration/vllm/` is unchanged.
- **Clean** (`maxtext_code_overlay.Dockerfile`): removes the old code directories
  first, re-downloads the test assets and re-installs the vLLM adapter for
  post-training images. Exact even when files were removed, but the `RUN` steps
  force BuildKit to pull the 3 GB source image into the builder, so it costs about
  as much as a full build.

The build log prints the choice ("Overlay variant selected by the drift guard: fast")
and the reason appears in the pre-flight job.

Measured on the TPU pipeline (pre-training stable image, September 2026):

| Build                    | Docker build step | Whole build job                           |
| ------------------------ | ----------------- | ----------------------------------------- |
| Full build (`build-all`) | about 200 s       | about 5 min                               |
| Fast overlay             | about 10 s        | about 1 min (2 min if the runner is cold) |
| Clean overlay            | about 200 s       | about 5 min                               |

The remaining time in a fast job is the job's own container start-up, checkout and
wheel install, not the image build. The CI tests that follow take the same time as
in any other run.

This guarantees that when you are debugging a test failure, the dependencies are
**provably identical** to the source run, not just "probably cached".

### Dependency drift guard

If any dependency-related file changed between the source commit and your current
commit, the fast-rebuild fails with an error listing the changed files. This
prevents silently using stale dependencies.

This includes changes you did not make yourself: after a rebase or merge from `main`
that brought in new dependency files, the guard refuses every source run from before
that point. Run `build-all` once on the new base and continue from that run.

The guard runs from the workflow's own checkout and only reads the two commits as
git data; it never executes code from the commit being built.

Files checked:

- `src/dependencies/` (requirements, scripts, dockerfiles, extra_deps)
- `pyproject.toml`
- `.dockerignore`

To override (not recommended): set `allow_dependency_drift` to `true` in the
workflow dispatch form. The resulting image may then have different dependencies
than the source image.

### Limitations

- Fast-rebuild requires the source image to still exist in the registry. Images may
  be garbage-collected after some time. If the lookup fails, the build log lists the
  most recent tags of that image.
- The source run's commit must still be fetchable from GitHub. A commit that was
  rewritten by a squash or force-push stays fetchable until GitHub garbage-collects
  it. If the pre-flight step fails with `Could not fetch commit`, run `build-all`
  once and continue from that run.
- **Re-run failed jobs** needs the run's commit to still be on a branch. After a
  squash or force-push, GitHub fails the re-run at startup ("This run likely failed
  because of a workflow file issue"). A fast-rebuild from that run still works,
  because the pre-flight fetches the source commit by SHA.
- The source run must have pushed to the same `repository_name`.
- Each fast-rebuild adds a code layer of a few tens of MB on top of the source
  image. The nightly `build-all` runs reset this.
- The fast overlay never deletes files. The guard therefore switches to the clean
  overlay when a code file was removed or renamed, which is slow but exact.
- Fast-rebuild is manual by design. Nightly and release builds always use
  `build-all`.
