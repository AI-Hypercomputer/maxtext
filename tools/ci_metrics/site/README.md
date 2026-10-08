# CI Pulse - static dashboard for the "MaxText Package Tests" pipeline

The dashboard is one file, hand-rolled SVG, no build step, no
chart library, no database, no AI. Open it in a browser and it renders. It is served from
`dev/bench/ci-pulse/index.html`, one folder below the benchmark page that links to it, so
that its data files sit beside it and every path it fetches is relative. Today it runs
on mock data that is baked into the file as JavaScript constants (`COMMITS`, `JOBS`,
`TH_CATEGORIES`, `RESCUES`, ...). The planned collector
(`../collect.py`, prototype) will replace those constants with view JSON files written to
`views/`, one per chart group per month, rebuilt from an append-only NDJSON store kept
separately. The browser never calls the GitHub API, and the collector never reads or writes
anything outside this project. Write and load rules: data catalog section 8.

What the pages show, the rules behind every number, and the data sources are in
`docs/superpowers/specs/`:

- `2026-08-21-ci-metrics-design.md` - the design and its constraints
- `2026-08-21-ci-metrics-data-catalog.md` - every metric, formula, threshold and
  window rule (section 3.1 has the real test-flavor table)
- `2026-08-31-ci-metrics-developer-guide.md` (+ `.zh-TW.md`) - first-time guide

How the page is published, how it gets real data, and how to refresh that data by hand are
in `../deploy/GOING-LIVE.md`.

## Layout

| Path | What it is |
| --- | --- |
| `../../../dev/bench/ci-pulse/index.html` | the dashboard itself, in the folder it is served from |
| `tests/check_*.js` | jsdom checks: they load the page, click the real controls and assert on the rendered DOM |
| `tests/check_tab.js` | the same, for the CI Pulse tab on `dev/bench/index.html` |
| `tests/run_all.js` | runs every check and exits non-zero on any failure |
| `../collect.py` | collector prototype (GitHub API + JUnit artifacts -> NDJSON) |

## Run the checks

```
cd tools/ci_metrics/site/tests
npm install        # jsdom only
npm test
```

Each check prints one line per assertion and a `N passed, M failed` summary. The
checks are the regression net for the mock: every visible number that was reviewed
(date-range windows, suite names, worker counts, flaky rates, sort orders, guide
panels) has an assertion.

## Facts the mock encodes about the real pipeline

- Suites are the workflow's test flavors, named as GitHub Actions names them:
  tpu-unit, tpu-integration, tpu-post-training-unit, tpu-post-training-integration,
  gpu-unit, gpu-integration, cpu-unit, cpu-integration, cpu-post-training-unit,
  cpu-post-training-integration. Workers per flavor: cpu-unit 4,
  cpu-post-training-unit 4, tpu-unit 2, all others 1 (read per run from the
  `Execute Tests (N)` jobs, never hard-coded).
- `decoupled` is not a job: an extra pytest pass inside cpu-unit worker 1 with its own
  result file; its 50 tests are also counted in cpu-unit, so the two are never added.
- tpu7x flavors run only when the workflow is not started by a pull request; the TPU
  Pathways jobs write no JUnit file. Neither has a suite in pull-request views.
- Runner labels: linux-x86-ct6e-180-4tpu (TPU), linux-x86-a2-48-a100-4gpu (GPU),
  linux-x86-n2-32 (CPU), linux-x86-n2-16-buildkit (package build).

Everything else in the file - run times, queue waits, test counts per suite, flaky
events - is mock data chosen to tell reviewable stories, not measurements.
