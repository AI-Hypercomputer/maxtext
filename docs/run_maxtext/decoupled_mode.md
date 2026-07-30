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

# Via Decoupled Mode (No Google Cloud Dependencies)

Set `DECOUPLE_GCLOUD=TRUE` to run MaxText tests and local development without any Google Cloud SDK, `gs://` buckets, JetStream, or Vertex AI integrations.

When enabled:

- Skips external integration tests with markers:
  - `external_serving` (`jetstream`, `serving`, `decode_server`)
  - `external_training` (`goodput`)
- `decoupled` – Applied by `tests/conftest.py` to every collected test except those carrying the external dependency markers above.
- Production / serving entrypoints (`decode.py`, `maxengine_server.py`, `maxengine_config.py`, tokenizer access in `maxengine.py`) **fail fast with a clear RuntimeError** when decoupled. This prevents accidentally running partial serving logic locally when decoupled mode is ON.
- Import-time safety is preserved by lightweight stubs returned from `decouple.py` (so modules import cleanly); only active use of missing functionality raises.
- Conditionally replaces dataset paths in certain tests to point at minimal local datasets.
- Uses a local base output directory (users can override with `LOCAL_BASE_OUTPUT`).
- Many tests use the helper `get_test_config_path()` from `tests/utils/test_helpers.py`. In decoupled mode, this helper selects `src/maxtext/configs/decoupled_base_test.yml` instead of `src/maxtext/configs/base.yml`.

Minimal datasets included (checked into the repo):

- ArrayRecord shards: generated via `python local_datasets/get_minimal_c4_en_dataset.py`,
  located in `local_datasets/c4_en_dataset_minimal/c4/en/3.0.1/c4-{train,validation}.array_record-*`
- Parquet (HF style): generated via `python local_datasets/get_minimal_hf_c4_parquet.py`,
  located in `local_datasets/c4_en_dataset_minimal/hf/c4`

Run a local smoke test fully offline:

```bash
export DECOUPLE_GCLOUD=TRUE
pytest -k train_gpu_smoke_test -q
```

Optional environment variables:

- `LOCAL_GCLOUD_PROJECT` - placeholder project string (default: `local-maxtext-project`).
- `LOCAL_BASE_OUTPUT` - override default local output directory used in tests.

## Installing a decoupled environment

`DECOUPLE_GCLOUD=TRUE` only changes what `gcloud_stub` hands back; it does not hide installed packages, so a module-scope `from google.cloud import storage` still succeeds in an environment that has the Google Cloud SDK. To actually run offline, install `src/dependencies/requirements/generated_requirements/decoupled-requirements.txt`, the GPU pre-training dependency set without the Google Cloud clients and accelerator wheels (see [Update MaxText dependencies](../development/update_dependencies.md)):

```bash
uv venv --seed .venv_decoupled
source .venv_decoupled/bin/activate
uv pip install --resolution=lowest \
  -r src/dependencies/requirements/generated_requirements/decoupled-requirements.txt
# MaxText itself requires the Google Cloud packages, so install it without its dependencies.
uv pip install --no-deps maxtext-*-py3-none-any.whl
```

CI builds this environment on every pull request and runs the tests marked `decoupled_target` in it (the `cpu-unit` job, first worker group), so a newly added unguarded Google Cloud import fails there:

```bash
DECOUPLE_GCLOUD=TRUE pytest -v -m decoupled_target --ignore=tests/post_training
```

Add `decoupled_target` to a test to include it in that gate. The marker only takes effect for tests that the default collection actually reaches, so files listed under `--ignore` in `pytest.ini` cannot be part of it.

### Running decoupled on an accelerator

Decoupling is about the Google Cloud dependencies, not about the hardware, so the file above carries no accelerator wheels and there is no separate decoupled file per accelerator. To run the decoupled suite on an NVIDIA GPU, add the entries that were left out of it, the plugin, the CUDA libraries it loads and Transformer Engine, at the versions the GPU requirements pin:

```bash
python3 src/dependencies/scripts/generate_decoupled_requirements.py --print-accelerator-requirements |
  uv pip install --no-deps --resolution=lowest -r /dev/stdin
```

`--no-deps` is safe because these entries plus the decoupled requirements are the whole GPU lock, and it is also what keeps the resolver from pulling the Google Cloud packages back in.

Other backends follow the same shape: install the vendor's plugin and pjrt wheels on top of the decoupled requirements, for example from a ROCm wheel index. If you want the versions of a different hardware lock rather than the GPU ones, derive the decoupled file from that lock instead, `--source generated_requirements/tpu-requirements.txt`, whose accelerator entries are then `libtpu`.

## Centralized Decoupling API (`gcloud_stub.py`)

MaxText exposes a single module `maxtext.common.gcloud_stub` to avoid scattering environment checks:

```python
from maxtext.common.gcloud_stub import is_decoupled, jetstream

if is_decoupled():
    # Skip optional integrations or use local fallbacks
    pass

# JetStream (serving) components
config_lib, engine_api, token_utils, tokenizer_api, token_params_ns = jetstream()
TokenizerParameters = getattr(token_params_ns, "TokenizerParameters", object)
```

Behavior when `DECOUPLE_GCLOUD=TRUE`:

- `is_decoupled()` returns True.
- Each helper returns lightweight stubs whose attributes are safe to access; calling methods raises a clear `RuntimeError` only when actually invoked.
- Prevents import-time failures for optional dependencies (JetStream).

## Guidelines:

- Prefer calling `jetstream()` once at module import and branching on `is_decoupled()` for functionality that truly requires the dependency.
- Use `is_decoupled()` to avoid direct `os.environ["DECOUPLE_GCLOUD"]` checking.
- Use `get_test_config_path()` instead of hard-coded `base.yml`.
- Prefer conditional local fallbacks for cloud buckets and avoid introducing direct `gs://...` paths.
- Please add the appropriate external dependency marker (`external_serving` or `external_training`) for new tests. Prefer the smallest scope instead of module-wide `pytestmark` when only a part of a file needs an external dependency.
- Tests add a `decoupled` marker if DECOUPLE_GCLOUD && not marked with external dependency markers. Run tests with:

```
pytest -m decoupled -vv tests
```

This centralized approach keeps optional integrations cleanly separated from core MaxText logic, making local development (e.g. on ROCm/NVIDIA GPUs) frictionless.
