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
images, how stable builds reuse cached dependencies-only images, and how to
re-run image-based tests without paying for a full dependency rebuild. To build
images on your own machine, see
[Build and upload MaxText Docker images](../tutorials/build_maxtext.md).

## When are Docker images built?

Exactly one workflow builds release/nightly images:
`.github/workflows/build_and_push_docker_image.yml`. It is a reusable workflow
(`workflow_call`), so it never starts on its own. Three workflows call it, and
these are the only events that build an image:

| Caller                                                        | Trigger                                                       | Images                                                                                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| TPU Docker Images Pipeline (`tpu_docker_images_pipeline.yml`) | Daily at 19:00 Pacific, or manual **Run workflow**            | `maxtext_jax_stable`, `maxtext_jax_nightly`, `maxtext_post_training_stable`, `maxtext_post_training_nightly`                           |
| GPU Docker Images Pipeline (`gpu_docker_images_pipeline.yml`) | Nightly at 00:00 UTC, or manual **Run workflow**              | `maxtext_gpu_jax_stable`, `maxtext_gpu_jax_nightly` (each for `amd64` and `arm64`)                                                     |
| Release Candidate Pipeline (`rc_release.yml`)                 | Weekly on Sunday at 19:00 Pacific, or manual **Run workflow** | `tpu_pre_training`, `tpu_post_training`. Always a full build (`release_pipeline.yml` verifies and promotes these certified RC images). |

A `git push` or pull request never triggers the nightly release pipelines. PR CI
(`ci_pipeline.yml`) maintains its own separate cached dependency images in
`maxtext-test-images/maxtext_*_deps_ci` via `build_deps_image.yml`, rebuilt
only when dependencies change.

Every image pipeline run pushes
`us-docker.pkg.dev/<project>/<repository>/<image>:<run_id>` **before** the tests
run, then runs the CI test suite against that tag and, on success, adds the tag
`verified-ci-<run_id>`. A failed test run does not delete the image, so GitHub's
**Re-run failed jobs** reuses the already-built `<image>:<run_id>` without
rebuilding.

## Cached dependencies-only images

For **stable** TPU and GPU builds, `build_and_push_docker_image.yml` separates
dependency installation from MaxText source code so unchanged dependencies are
never rebuilt. The dependencies-only images (`maxtext_tpu_pre_train_deps`,
`maxtext_tpu_post_train_deps`, `maxtext_gpu_pre_train_deps`) are internal build
caches: they live only in the build repository (`maxtext-images-staging` for
scheduled runs) and are never promoted to `maxtext-images`.

1. **Multi-stage dependency Dockerfiles** (`maxtext_tpu_dependencies.Dockerfile`
   and `maxtext_gpu_dependencies.Dockerfile`) define:

   - **Stage 1 (`dependencies`):** installs OS packages, `gcloud`, Python
     packages via `setup.sh`, strips `/deps/src/maxtext`, and sets `PYTHONPATH`.
     It omits `MAXTEXT_*` path env vars so users mounting code at `/app`
     auto-resolve paths from `src/maxtext/utils/globals.py`.
   - **Stage 2 (`maxtext`):** inherits from `dependencies`, sets `ENV MAXTEXT_*`,
     copies MaxText source code, tests, benchmarks, and downloads optional test
     assets.

2. **Content-hash lookup (`src/dependencies/scripts/compute_deps_tag.sh`):**
   computes a deterministic tag `[<platform>-]deps-<16hex>-b<12hex>` from the
   relevant Dockerfile, `setup.sh`, pinned requirement/extra-dep files, and the
   resolved Airlock base image digest, and looks it up in the build repository.

3. **Cache hit vs. cache miss:**

   - **Nightly / RC (`deps_image_name` empty):** skips cache lookup, calls
     `build_deps_image.yml` (`--target dependencies`, `TF=true`, Airlock) to
     push an intermediate staging tag (`:[<platform>-]deps-<run_id>`), and
     overlays code with `maxtext_code_overlay.Dockerfile`.
   - **Cache miss** (first run, dependency pin change, Dockerfile change, or
     Airlock base image security update): calls `build_deps_image.yml`
     (`--target dependencies`, `TF=true`, Airlock) to push
     `maxtext_*_deps:<deps_tag>` to the build repository, then overlays code
     onto that just-built digest via `maxtext_code_overlay.Dockerfile`.
   - **Cache hit:** skips dependency installation completely and runs the same
     `maxtext_code_overlay.Dockerfile` step on top of the cached
     `maxtext_*_deps@sha256:<digest>` image:
     - `overlay-pre-training`: `COPY --link` and `ENV` only (no `RUN`), so
       BuildKit never pulls the cached base layers (finishes in seconds, even
       for `arm64` GPU builds). Test assets are downloaded on the host runner
       beforehand and included via `COPY --link`.
     - `overlay-post-training`: inherits `overlay-pre-training` and runs a
       `--no-deps` reinstall of `/deps/src/maxtext/integration/vllm` so the
       adapter package metadata matches the overlaid code.

   In every case only the full image is SBOM-scanned, promoted through the MOSS
   Exit Gate, and tagged in `maxtext-images`. Its SBOM covers all layers,
   including the dependencies-only base.

### Re-running after a failure

- **Flaky test, code unchanged:** click **Re-run failed jobs** on the failed
  GitHub Actions run. It reuses `<image>:<run_id>` with no build at all.
- **Code fix on a branch:** click **Run workflow** on **TPU Docker Images
  Pipeline** or **GPU Docker Images Pipeline** in GitHub Actions and select your
  branch. Manual runs default to `maxtext-test-images`; once the cached `maxtext_*_deps`
  image exists in that repository, stable builds skip dependency installation
  and overlay only your branch's code.
- **Dependency change:** click **Run workflow** on your branch as usual — the
  content hash changes automatically, triggering a clean rebuild of
  `maxtext_*_deps` before building the full image.
