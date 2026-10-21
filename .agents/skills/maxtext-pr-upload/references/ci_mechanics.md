# MaxText CI Mechanics & Failure Classification

## 1. Required vs. Non-Blocking CI Jobs

`MaxText Package Tests` (`ci_pipeline.yml`) aggregates required jobs in the
final `All Required Tests Passed` gate:
- **Blocking:** `cpu-unit` (4 shards), `cpu-integration` (2 shards), `tpu-unit`,
  `tpu-integration`, `gpu-unit` (when triggered), `RunPrecommitChecks.yml`
  (`pyink`, `pylint`, `codespell`, `yamllint`, copyright header check), and
  `Require Checklist`.
- **Non-blocking:** `Track Test Performance / Track Test Duration` frequently
  reports `[REGRESSION ALERT]` due to runner variance and is excluded from
  `All Required Tests Passed`.

## 2. Classifying Job Failures by Step (`ci_status.py`)

Always inspect the failed step name in `ci_status.py` output before editing code:

| Failed Step Pattern | Root Cause | Action |
| :--- | :--- | :--- |
| `Run Tests` failed AND `Upload Test Results` succeeded | Genuine unit or integration test failure | Reproduce locally with `PYTHONPATH=$PWD/src pytest -q <test_file>` |
| `Run Tests` failed (exit 130 / cancelled) AND `Upload Test Results` skipped/failed | Self-hosted runner preemption or concurrency cancellation | Re-run the failed job or push a fresh SHA if caused by a double-push race |
| `Set up job` / `Pull Docker image` / `Install dependencies` failed | Transient infrastructure or registry failure | Re-run failed jobs in GitHub Actions |

## 3. High-Frequency MaxText CI Traps

1. **`tests/unit/train_nnx_test.py` (`_Cfg` mock):**
   `train_nnx_test.py` uses a lightweight `_Cfg` stub instead of `pyconfig`. Any
   new `config.<field>` accessed in `train_step`, `loss_fn`, or `eval_step` must
   also have a default attribute on `_Cfg`.
2. **`tests/integration/hlo_diff_test.py`:**
   Compares compiled HLO module fingerprints against checked-in golden baselines
   for reference models (`default`, `gpt3-175b`, `llama2-7b`). Gate new decoder
   or loss branches behind a config flag that defaults to `False`/`0.0` on those
   reference configs.
3. **`_Cfg` / `pyconfig` singleton state across tests:**
   If a new unit test mutates a module-level singleton or global config dict in
   place, restore it in `tearDown` / `addCleanup` so subsequent tests in the
   same `cpu-unit` shard are not polluted.

## 4. `AddPullReady` (`pull ready` Label) Conditions

The `AddPullReady` workflow applies `pull ready` only when all four conditions
hold at the moment a review is submitted:
1. `commits == 1` (`git rev-list --count <base>..<head> == 1`).
2. `reviewDecision == APPROVED` (including `.github/CODEOWNERS` for all touched paths).
3. `>= 2` unique approving reviewers.
4. All required checks on the current head SHA have already completed green.
