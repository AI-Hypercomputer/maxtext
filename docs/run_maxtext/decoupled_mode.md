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


# Decoupled Mode (No Google Cloud Dependencies)

Set `DECOUPLE_GCLOUD=TRUE` to run MaxText tests and local development without any Google Cloud SDK, `gs://` buckets, JetStream, or Vertex AI integrations.

When enabled:
* Skips external integration tests with markers:
  * `external_serving` (`jetstream`, `serving`, `decode_server`)
  * `external_training` (`goodput`)
* `decoupled` – Applied by `tests/conftest.py` to tests that are runnable in decoupled mode (i.e. not skipped for TPU or external markers).
* Production / serving entrypoints (`decode.py`, `maxengine_server.py`, `maxengine_config.py`, tokenizer access in `maxengine.py`) **fail fast with a clear RuntimeError** when decoupled. This prevents accidentally running partial serving logic locally when decoupled mode is ON.
* Import-time safety is preserved by lightweight stubs returned from `decouple.py` (so modules import cleanly); only active use of missing functionality raises.
* Conditionally replaces dataset paths in certain tests to point at minimal local datasets.
* Uses a local base output directory (users can override with `LOCAL_BASE_OUTPUT`).
* All tests that previously hard-coded `configs/base.yml` now use the helper `get_test_config_path()` from `tests/test_utils.py`. This helper ensures usage of `decoupled_base_test.yml`

Minimal datasets included (checked into the repo):
* ArrayRecord shards: generated via `python local_datasets/get_minimal_c4_en_dataset.py`, 
  located in `local_datasets/c4_en_dataset_minimal/c4/en/3.0.1/c4-{train,validation}.array_record-*`
* Parquet (HF style): generated via `python local_datasets/get_minimal_hf_c4_parquet.py`, 
  located in `local_datasets/c4_en_dataset_minimal/hf/c4`


Run a local smoke test fully offline:
```bash
export DECOUPLE_GCLOUD=TRUE
pytest -k train_gpu_smoke_test -q
```

Optional environment variables:
* `LOCAL_GCLOUD_PROJECT` - placeholder project string (default: `local-maxtext-project`).
* `LOCAL_BASE_OUTPUT` - override default local output directory used in tests.

## Centralized Decoupling API (`gcloud_stub.py`)

MaxText exposes a single module `MaxText.gcloud_stub` to avoid scattering environment checks:

```python
from MaxText.gcloud_stub import is_decoupled, cloud_diagnostics, jetstream

if is_decoupled():
  # Skip optional integrations or use local fallbacks
  pass

# Cloud diagnostics (returns diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration)
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = cloud_diagnostics()

# JetStream (serving) components
config_lib, engine_api, token_utils, tokenizer_api, token_params_ns = jetstream()
TokenizerParameters = getattr(token_params_ns, "TokenizerParameters", object)
```

Behavior when `DECOUPLE_GCLOUD=TRUE`:
* `is_decoupled()` returns True.
* Each helper returns lightweight stubs whose attributes are safe to access; calling methods raises a clear `RuntimeError` only when actually invoked.
* Prevents import-time failures for optional dependencies (JetStream).

## Guidelines:
* Prefer calling `jetstream()` / `cloud_diagnostics()` once at module import and branching on `is_decoupled()` for functionality that truly requires the dependency.
* Use `is_decoupled()` to avoid direct `os.environ["DECOUPLE_GCLOUD"]` checking.
* Use `get_test_config_path()` instead of hard-coded `base.yml`.
* Prefer conditional local fallbacks for cloud buckets and avoid introducing direct `gs://...` paths.
* Please add the appropriate external dependency marker (`external_serving` or `external_training`) for new tests. Prefer the smallest scope instead of module-wide `pytestmark` when only a part of a file needs an external dependency.
* Tests add a `decoupled` marker if DECOUPLE_GCLOUD && not marked with external dependency markers. Run tests with:
```
pytest -m decoupled -vv tests
```

This centralized approach keeps optional integrations cleanly separated from core MaxText logic, making local development (e.g. on ROCm/NVIDIA GPUs) frictionless.

