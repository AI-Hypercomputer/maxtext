# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/MaxText/` (core library), `src/install_maxtext_extra_deps/` (extras installer).
- Tests: `tests/` (pytest; files named `*_test.py` or `*_tests.py`).
- Docs and assets: `docs/`, `src/MaxText/assets/`.
- Examples & scripts: `end_to_end/`, `benchmarks/`, `tools/`.
- Config: `pyproject.toml` (build), `pytest.ini` (test), `pylintrc` (lint), `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
- Create env (Python 3.12): `python -m venv .venv && source .venv/bin/activate`.
- Install: `pip install -e .` or with extras `pip install -e .[cuda12]` / `.[tpu]`.
- Lint/format (pre-commit): `pre-commit install && pre-commit run -a`.
- Run tests: `pytest -q`.
  - Select tests: `pytest -k decode`.
  - By marker: `pytest -m "gpu_only"` or skip accelerators: `pytest -m "not gpu_only and not tpu_only"`.
- Build wheel (optional): `hatch build`.

## Coding Style & Naming Conventions
- Python, 2-space indentation (enforced by Pyink). Target line length 122.
- Follow Google Python Style where applicable (see `pylintrc`).
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Keep public APIs in `src/MaxText/` stable; document breaking changes in PR.

## Testing Guidelines
- Framework: `pytest` with markers: `gpu_only`, `tpu_only`, `cpu_only`, `integration_test`, `scheduled_only`.
- Place unit tests in `tests/` mirroring `src/MaxText/` paths; name files `*_test.py`.
- Add tests for new features and bug fixes; prefer small, deterministic tests. Use markers to limit accelerator-specific runs.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., `inference: fix kv-cache index`).
- PRs: clear description, rationale, before/after behavior, linked issues, and any config or doc updates. Include logs for perf changes.
- Quality gates: `pre-commit` clean, tests pass locally (`pytest -q`). Avoid committing large artifacts (models, datasets, secrets).

## Agent-Specific Notes
- Prefer focused patches; update related tests when touching `src/MaxText/`.
- Do not modify tokenizer assets or large data under `src/MaxText/assets/` unless required; discuss in PR.
- Adhere to this file’s guidance for all subdirectories.

