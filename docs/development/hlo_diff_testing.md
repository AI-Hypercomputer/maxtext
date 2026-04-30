# HLO Graph Diff Verification Testing

This document provides context for the HLO Graph Diff tests, what HLO is, and how to manage reference baselines.

## Related Files

- **Test Logic**: `tests/integration/hlo_diff_test.py`
- **Reference Checkpoints baselines**: `tests/utils/reference_hlo_*.txt`
- **Update Helper script**: `tests/utils/update_hlo_references.py`
- **GitHub Action Trigger Workflow**: `.github/workflows/update_reference_hlo.yml`

## What is HLO?

**HLO (High-Level Optimizer)** is the intermediate representation used by XLA (Accelerated Linear Algebra) to capture the lowering compiler graph structures.

An HLO module records:

- The sequences of low-level math operations (dot products, convolutions, additions).
- Array tensor shapes and numerical precisions.
- Multipod TPU cluster partitioning array sharding mappings.

## Purpose of HloDiffTest

The primary purpose of the `TestHloDiff` validation checks is to ensure that **refactoring PRs are purely refactoring code** and not unintentionally impacting graph compiler lowering or performance.

- **For pure refactors:** The HLO graph layout should remain *strictly identical*. Any detected deviation flags that execution boundaries or operation pipelines might have changed under the hood.
- **For dependency updates:** Changes to framework dependencies (like updating JAX or XLA versions) *are expected* to slightly alter compiled HLO output layouts, which makes baseline updates appropriate in those scenarios.

______________________________________________________________________

## How the Test Works

This test runs automatically as part of the [`tpu-integration`](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml) CI test suite on every Pull Request.

When the test method executes, it performs the following sequence of actions:

1. **Triggers Compilation**: It runs the model training lifecycle compilation-only phase (invoking `train_compile.main()`) without actually allocating hardware compute nodes or running optimization passes.
2. **Dumps HLO modules**: Instructs the XLA compiler back-end to capture optimizer operations lowering structure graphs and dump them to text files.
3. **Strict comparison matches**: Compares the structural lines of the generated representation graph directly against baseline `.txt` copies stored under `tests/utils/`.

______________________________________________________________________

## Updating HLO reference files

When intended architectures transformations alter graph lowering, reference file baselines require updates.

> [!IMPORTANT]\
> While running the update script locally is not the end of the world, **relying on local execution can cause remote CI tests to fail.**
> The PR verification pipelines run the tests in a strictly locked GitHub Actions environment. The smallest discrepancies in local library installations will introduce slight backend lowering graph deviations. If your local execution leads to a remote CI check failure, rely on the GitHub Action trigger described below to generate environment-matching baselines.

### Method 1: Run the manual GitHub Action Workflow (Highly Recommended)

Triggering the CI workflow guarantees execution runs within the correct environment isolation scope.

#### Option A: Using the GitHub UI

1. Go to the Actions tab in the repository browser.
2. Find the manual workflow: `Update HLO References (for hlo_diff_test.py)`.
3. Run it targeting your PR workspace branch. It compiles the graph layout and commits the baseline update files back to the branch automatically.

#### Option B: Using the GitHub CLI (`gh`)

Alternatively, you can trigger the remote workflow via terminal CLI execution:

```bash
gh workflow run update_reference_hlo.yml --ref <branch>
```

> [!NOTE]
> A successful run of the manual update workflow will add a new commit to your Pull Request branch. Once complete, you must:
>
> 1. Pull the new commit from remote.
> 2. Squash the commits in your branch once again to keep your PR history clean.
> 3. Push the squashed commit to remote.
> 4. Retry the `tpu-integration` workflow to verify tests pass on your PR.

### Method 2: Local Execution

If you need to test or update baselines manually during development:

```bash
source .venv/bin/activate
pytest tests/integration/hlo_diff_test.py -v
```

Or to force update the local baselines:

```bash
python3 tests/utils/update_hlo_references.py
```
