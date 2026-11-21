# MaxText CI Mechanics (`AI-Hypercomputer/maxtext`)

Load this when a MaxText PR's CI is red and the cause is not obvious from
`.agents/skills/maxtext-pr-upload/scripts/ci_status.py`, or when changing
`train_step` / `loss_fn` / `eval_step`.

## Workflow layout

- Marker/flavor matrix: `.github/workflows/run_tests_coordinator.yml`. Actual
  pytest call: `.github/workflows/run_tests_against_package.yml`, step `Run
  Tests`.
- `cpu-unit` marker: `cpu_only and not post_training and not integration_test
  and (not scheduled_only or newly_added)`, sharded with `pytest-split --splits
  N --group G -n auto`, with
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
- `cpu-integration` marker: same, with `integration_test`. 4 shards,
  `least_duration` split from `gh-pages:dev/bench/per_test_baseline.json` (not
  stored in the repo).
- `tests/conftest.py` auto-applies `cpu_only` to tests without a hardware
  marker, so `cpu-unit` runs most of the suite.
- `newly_added` diffs against `origin/<PR base>...HEAD` (the PR's base, not
  necessarily `main`).
- Code Quality runs `pyink`/`mdformat` (mdformat only on `docs/`) over all
  files, and the full pre-commit suite including `pylint` over files changed
  since the merge base.
- `codecov/patch` is the only coverage check (`codecov/project` is not
  configured, so its absence is fine). A helper defined as a closure inside a
  large function is unreachable from CPU unit tests; hoist it to module scope
  so tests can cover it.

## Reading CI without admin rights

- Raw job logs (`/actions/jobs/{id}/logs`) and unauthenticated artifact zips
  return HTTP 403/401 without repo/token access. Readable unauthenticated:
  `/commits/{sha}/check-runs?per_page=100`, `/check-runs/{id}/annotations`,
  `/actions/jobs/{id}` (`steps[]` conclusions),
  `/actions/runs?head_sha=<sha>&per_page=100`,
  `/actions/runs/{id}/attempts/{attempt}/jobs?per_page=25`, and
  `/actions/runs/{id}/artifacts` (byte sizes).
- On re-runs (`run_attempt >= 2`), `/actions/runs/{id}/jobs?per_page=100` can
  return HTTP 502 on large matrix payloads and `/commits/{sha}/check-runs`
  retains stale attempt-1 conclusions while attempt 2 is running. Always query
  `/actions/runs/{id}/attempts/{attempt}/jobs?per_page=25` (handled
  automatically by `.agents/skills/maxtext-pr-upload/scripts/ci_status.py`).
- Unauthenticated rate limit: 60 requests/hour (`GET /rate_limit` is free).
  Exhaustion returns HTTP 403 with `x-ratelimit-remaining: 0`; sleep until
  `x-ratelimit-reset` and space polls `>= 6 min` apart.
- A head SHA carries ~60-90 check runs across ~5-9 workflow runs. The live
  `ci_pipeline` run is the latest `(name, event)` run; a force-push-superseded
  run has 19 placeholder checks and phantom `All Required Tests Passed`
  failures.

## Classifying a red job

Do not classify by generic annotation text (`Executing the custom container
implementation failed` / `Process completed with exit code 1` /
`ScriptExecutorError` appears on every failing job). Inspect `GET
/actions/jobs/{id}` `steps[]`:

| Signal | Meaning |
| :--- | :--- |
| Steps 1-7 green, step 8 `Run Tests` red after normal wall time, `Upload Test Results XML` green | Genuine test failure |
| Exit code 130, both `Run Tests` and `Upload Test Results XML` red | Runner interruption; re-run |
| `Install the maxtext wheel` red with `FetchError ... ETIMEDOUT` | Runner network flake; re-run |
| Only `Track Test Duration` red | Non-blocking `[REGRESSION ALERT]` against rolling `gh-pages` baseline |

Normal wall times: `cpu-unit` shard 1 ~16 min, shards 2-4 ~5-7 min;
`cpu-integration` ~2-3 min. JUnit artifact size hints: a clean
`cpu-integration` shard XML is ~1.0-1.2 KB; ~5.9 KB indicates a recorded
traceback.

## Two CI-only failure classes (missed by plain local `pytest`)

1. **`tests/integration/hlo_diff_test.py`** runs only when `GITHUB_ACTIONS` is
   set. It AOT-compiles `jit_train_step` (`compile_topology=v6e-4`) for
   `deepseek3`, `llama3_8b`, and `qwen3_1.7b`, and compares filtered HLO
   against `tests/utils/reference_hlo_*.txt`. Any `train_step` graph change
   fails 3 `cpu-integration` shards. Gate new `train_step` behavior behind a
   config flag defaulting `false` in `base.yml` (enabled only in the model YAML
   that requires it) so reference HLO graphs stay identical.
2. **`tests/unit/train_nnx_test.py::_Cfg`** is a hand-written dataclass of the
   config fields read by `loss_fn` / `train_step` / `eval_step`. Reading a new
   `config.<attr>` there without adding it to `_Cfg` raises `AttributeError` in
   `cpu-unit`. Short-circuited reads (`config.use_x and config.new_attr`) still
   require the field on `_Cfg`.

## Reproducing in a Local Worktree

- Export `PYTHONPATH="$PWD/src:$PWD"` in any secondary git worktree so editable
  installs from another checkout do not shadow your changes, and set
  `JAX_ENABLE_COMPILATION_CACHE=0` when running parallel workers.
- Known local-only discrepancies: `grain_data_processing_test.py` (requires CI
  dataset fixtures), HF reference tests when local `transformers` differs, and
  `train_compile_test.py` / `train_nnx_test.py` / `check_vma_test` /
  `diloco_test` under `-n` xdist contention (run them serially).
- Exact CI wheel recipe when dependency versions matter: fresh worktree,
  `uv venv --seed`, `python -m build --wheel`,
  `uv pip install "maxtext-*.whl[tpu]" --resolution=lowest` (default resolution
  can pull `jax` ahead of `flax` and break test collection).

## `pull ready` (`.github/workflows/auto_label_pr.yml`)

All conditions must hold on the head SHA:

1. `!isDraft && !isCrossRepository`.
2. `reviewDecision == APPROVED` (`.github/CODEOWNERS` for every touched path)
   and `>= 2` unique approvers.
3. Exactly `1` commit (`commits.totalCount == 1`).
4. Required CI checks on the head SHA completed with `success`/`skipped`
   (excluding `AddPullReady` and `Track Test Duration`). Cancelled runs from a
   double-`synchronize` push block readiness until a new SHA is pushed.
