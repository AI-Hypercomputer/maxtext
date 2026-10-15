# ci-metrics-code

This branch holds only the code of the CI Pulse collector. It is never merged into `main`.

- The workflow `.github/workflows/ci_metrics_collector.yml` on `main` runs one reviewed commit of
  this branch every 4 hours. That commit is named by `COLLECTOR_SHA` in the workflow file.
- It collects maxtext's CI results for the CI Pulse dashboard:
  https://ai-hypercomputer.github.io/maxtext/dev/bench/#ci
- Raw data goes to the `ci-metrics-store` branch and dashboard files to `gh-pages`. Nothing is
  written to this branch by any robot.

## How to change the code

1. Commit to this branch. Never force-push it: the commit that `main` runs must stay on it, or
   the scheduled runs stop.
2. Open a PR to `main` that changes `COLLECTOR_SHA` to the new commit. Put this link in the PR so
   reviewers can read the change:
   `https://github.com/AI-Hypercomputer/maxtext/compare/<old commit>...<new commit>`
3. After that PR merges, the next run uses the new code.

## Run the tests

    python3 -B -m unittest discover -s tools/ci_metrics/collector/tests -p '*_test.py' -t tools/ci_metrics/collector/tests

The code and its full documentation are in `tools/ci_metrics/collector/` (see its README.md).
