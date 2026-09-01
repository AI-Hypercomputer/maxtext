# CI metrics collector

This package reads the "MaxText Package Tests" pipeline (`.github/workflows/ci_pipeline.yml`)
from the GitHub REST API, downloads the JUnit XML artifacts its test jobs upload, and turns
both into the numbers and the row shapes the CI Pulse dashboard is built from.

It is built in three layers:

- **Layer 1, reading.** `github.py` fetches; `junit.py` parses test artifacts into counts.
- **Layer 2, understanding.** `runs.py` finds runs, attempts, jobs and the pull request each
  run belongs to; `derive.py` turns job objects into durations, worker counts and rescues;
  `rows.py` puts those into the shapes that get stored, with their keys.
- **Layer 3, keeping and publishing.** `store.py` is the append-only store on disk;
  `views.py` turns stored rows into the JSON the browser loads; `tick.py` is the command a
  schedule calls, and the only thing here that decides what to fetch.

`tick.py` is what actually runs. `demo.py` stays as the read-only proof of layers 1 and 2:
give it a run id and it prints every number the dashboard would show, and writes nothing.

```bash
# One tick, into a directory of its own. --out is required; there is no default.
GITHUB_TOKEN=$(gh auth token) python3 -m collector.tick --out /path/to/ci-metrics
```

Two rules govern everything here.

- **Read-only, GitHub API only.** Every request is a GET. Nothing writes to GitHub, and
  nothing touches the `gh-pages` branch or the `dev/bench` folder. The only thing this
  package writes is files under the directory `--out` names.
- **No AI anywhere.** Every value is an API field, an XML element, or arithmetic on those.
  Failure text is quoted verbatim; nothing is summarised or scored.

Dependencies: the Python standard library plus `requests`. Nothing else.

## The modules

### `github.py` - the read-only API client

One class, `GitHubClient`, wrapping `requests` with the three things the collector needs:
authentication, pagination that stops correctly, and retries that know which failures are
worth repeating.

Things worth knowing before you use it:

- **The token is attached by host, not by call site.** Only `api.github.com` is ever sent
  an `Authorization` header. Any other host is requested anonymously, whatever URL the
  caller passes in. That matters because an artifact download answers 302 with a redirect to
  a signed storage URL on a Microsoft-owned host.
- **The redirect is followed by hand.** Letting `requests` follow it would strip our header
  but could re-apply a local `.netrc` credential for the new host, so `get_bytes` walks the
  hops itself with the header removed and `.netrc` lookups suppressed.
- **Signed URLs never reach the logs.** Warnings and error messages quote URLs without their
  query string, because that query carries the download signature.
- **Retries:** at most 3 attempts, backoff 2s then 4s. 5xx and transport errors are always
  retried. `429` is always retried and honours `Retry-After`. `403` is retried **only** when
  it carries a rate-limit signal, because a plain permission 403 will keep meaning the same
  thing. `404` is never retried. One call can spend at most one hour asleep across all its
  attempts, so a rate-limited request cannot outlive the collector's own tick.

```python
class GitHubError(RuntimeError)              # .status: int | None, .url: str | None

class GitHubClient:
  def __init__(owner, repo, token=None, session=None)
  def get_json(path, **params) -> dict       # one JSON object; path may be repo-relative
  def paginate(path, key, **params) -> list  # every page, flattened; follows Link headers
  def get_bytes(url) -> bytes                # absolute URL, follows redirects, no token off-host
  def rate_limit() -> dict                   # {"limit", "remaining", "reset"}
  def wait_for_rate_limit(need=50) -> None   # free when the budget is clearly sufficient
  def close() -> None                        # closes only a session it created itself
```

The token comes from the `token=` argument or the `GITHUB_TOKEN` environment variable. With
no token the API allows 60 requests an hour, which is not enough for a single run, so the
client prints one warning per process. A client built without a token also clears any
`Authorization` header it finds on a session it was handed, so it can never send someone
else's credential.

### `junit.py` - artifact listing and JUnit XML parsing

Pure parsers plus three functions that go through the client. `junit.py` does not import
`github.py`: it declares a two-method Protocol (`paginate`, `get_bytes`) that `GitHubClient`
satisfies, so the parsers import and run with no network module present and tests can pass a
stub.

```python
# Constants
ARTIFACT_PREFIX  = "test-results-"
KNOWN_FLAVORS    = (10 flavors; the three tpu7x-* are not in the default ask list)
NESTED_SUITES    = {"decoupled": "cpu-unit"}
REASON_NO_FILE / REASON_UPLOAD_EMPTY / REASON_ARTIFACT_EXPIRED
STATUS_PASSED / STATUS_SKIPPED / STATUS_FAILED / STATUS_ERROR

class JUnitError(Exception)                  # always names the file or artifact at fault
class GitHubClientLike(Protocol)             # paginate(path, key, **params), get_bytes(url)

# Dataclasses
ArtifactRef(name, artifact_id, flavor, worker, expired, size_in_bytes, download_url,
            run_id, created_at, expires_at)
            .from_api(payload) -> ArtifactRef | None
TestRow(name, classname, duration, status, failure_message, worker)
SuiteResult(collected, skipped, executed, junit_seconds, tests, failed, errored,
            reported_tests, suite_seconds, hostname, timestamp, files)
            .count_matches_attribute -> bool
SuiteEntry(suite_id, result, reason, nested_in, per_worker, missing_workers)
            .published_worker_count -> int
            .is_partial -> bool
RunTests(run_id, suites, artifacts)
            .result_for(suite_id) -> SuiteResult | None
            .reason_for(suite_id) -> str | None

# Pure, offline
parse_artifact_name(name) -> tuple[str, int] | None
suite_id_for_file(file_name, flavor) -> tuple[str, str | None]
parse_junit_xml(data, file_name="<junit xml>") -> SuiteResult
merge_suite_results(results) -> SuiteResult | None
parse_artifact_zip(data, flavor, artifact_name="<artifact>") -> dict[str, SuiteResult]

# Through the client
list_test_artifacts(client, run_id) -> list[ArtifactRef]
read_artifact_suites(client, ref) -> dict[str, SuiteResult]
read_run_tests(client, run_id, flavors=None) -> RunTests
```

Four rules this module enforces, all of which exist because breaking them puts a false alarm
on the dashboard:

1. **Missing is None with a reason, never zero.** A zero would draw a "tests vanished" spike
   for a run that simply published nothing. The reason is one of `no_file_published`,
   `upload_empty` or `artifact_expired`.
2. **A partial total says it is partial.** A flavor runs on several parallel workers, and
   their artifacts expire minutes apart. When some workers are readable and others are not,
   the entry keeps the surviving total and lists the rest in `SuiteEntry.missing_workers`
   (`is_partial` is the shorthand). A silent partial total is worse than a zero, because it
   looks plausible.
3. **The test count is counted, not read off an attribute.** T = `<testcase>` elements minus
   the skipped ones. The `<testsuite tests="...">` attribute disagrees with the elements on
   real files (870 against 737 on one of the fixtures) and is kept in `reported_tests` only
   as a cross-check.
4. **`decoupled` is its own suite, nested inside `cpu-unit` worker 1.** Those ~50 tests also
   run in `cpu-unit`'s normal pass, so the two totals must never be added together.
   `nested_in` says so on the entry.

Two more things the next layer needs to keep straight:

- `junit_seconds` is the sum of the per-case `time` attributes. The CPU flavors run pytest
  with `-n auto`, so that sum adds up across parallel processes and is **not** wall-clock
  time. Neither is `<testsuite time>`. Wall-clock duration comes from the job step
  timestamps, which is a different module's input.
- The authoritative worker count W is the number of `Execute Tests (N)` jobs in the run, read
  from the jobs endpoint. `published_worker_count` counts artifacts and is a cross-check
  only, never the source.

### `runs.py` - runs, attempts, jobs and pull requests

Finds what to collect. It knows the allowlist of workflows, how to ask for a window without
hitting GitHub's 1000-result listing cap, which cancelled runs were superseded, and which
pull request a run belongs to.

```python
class RunsError(RuntimeError)                # an unusable payload; request failures stay GitHubError
class GitHubClientLike(Protocol)             # get_json(path, **params), paginate(path, key, **params)

# Constants
WORKFLOW_ALLOWLIST                           # ci_pipeline plus the two nightly image builds
RUNS_API_RESULT_CAP = 1000 / BACKFILL_WINDOW_DAYS = 7
SUPERSEDED_FIELD = "superseded" / CONCLUSION_CANCELLED = "cancelled"
EVENT_PULL_REQUEST / EVENT_SCHEDULE / SUPERSEDING_EVENTS

# Pure - no network, no clock, no state
parse_timestamp / as_utc / run_id_of / run_created_at
sort_runs_newest_first(runs) / dedupe_runs(runs)
filter_runs_to_workflows(runs, workflow_ids=None, paths=None)
filter_runs_to_window(runs, since, until=None)
created_filter(since, until=None) / split_window(since, until, days=7)
mark_superseded(runs) -> list                # shallow copies carrying a `superseded` bool
embedded_pull_request(run) / match_pull_request(pulls, run) / head_owner(run)

# Through the client
resolve_workflow_ids(client, paths=None, refresh=False) / clear_caches()
list_runs(client, since, until=None, workflow_ids=None)
get_run(client, run_id) / list_attempts(client, run) / get_jobs(client, run_id, attempt)
find_pull_requests_for_branch(client, owner, branch, state="all")
link_pull_request(client, run) / get_pull_request(client, number) / resolve_pull_request(client, run)
```

Four rules that are easy to get wrong:

1. **Supersession follows the workflow's own concurrency groups, not the branch.**
   `ci_pipeline.yml` groups a `pull_request` run by the pull request NUMBER, every `schedule`
   run into one group, and everything else into a group of its own run id. So a cancelled
   `push` or `workflow_dispatch` run can never be superseded, and grouping by branch made the
   4-hourly scheduled run look like the successor of every manual run on `main` - which then
   vanished from every statistic. A pull request run whose payload names no number borrows it
   from a sibling run of the same branch, because the array empties once the pull request
   merges.
2. **`run_attempt` is not stable.** One run read 2 from a listing and 3 from the single-run
   endpoint 26 minutes later, so `list_attempts` re-reads the run first. It can still return
   fewer entries than `run_attempt` when GitHub has pruned an early attempt, so read the
   attempt number off each payload rather than from the list position.
3. **The branch lookup only runs for a `pull_request` run.** `main` has been the head branch
   of a pull request once, in 2024, so asking `head=AI-Hypercomputer:main` for a scheduled run
   linked every scheduled run to pull request #771. For any other trigger the answer is a
   wrong link, not a missing one.
4. **A listing at the 1000-run cap is reported, not split silently.** ci_pipeline alone makes
   about 650 runs a week; ask a week at a time and let `split_window` cut a wider window up.

### `derive.py` - job objects to dashboard numbers

Pure arithmetic on job dictionaries. No network, no files, no clock, no global state.

```python
@dataclass PhaseSplit   # queued / setup / tests / tail, plus their boundary moments
@dataclass Rescue       # one job that failed on an attempt and passed on the next

parse_timestamp(value)                       # the one date parser in this module
queue_seconds(job) / run_seconds(job) / setup_seconds(job) / step_span(job, step_name)
is_carried_over(job) / held_a_runner(job)    # the gate in front of every number
suite_duration_seconds(jobs) -> float | None # THE WALL-CLOCK RULE
run_wall_seconds(jobs) / machine_seconds(jobs) / phase_split(jobs)
flavor_of(job) / test_flavors(jobs) / jobs_for_flavor(jobs, flavor)
parse_execute_tests_name(name) / worker_count(jobs, flavor) / device_lane(job)
find_rescues(attempts_jobs) -> list[Rescue]
slowest_tests(rows, per_flavor=25)
```

The rule this module exists to get right: **D is wall clock across a suite's parallel
workers**, the first worker's "Run Tests" start to the last worker's finish. It is not the
sum of the workers' run times and not the sum of the JUnit seconds. On run 33468578834
tpu-unit took 1626 s while its two JUnit files add up to 2519.7 s, 1.55x too much, because
pytest runs with `-n auto` inside each worker as well as across them.

Four job shapes compute to a plausible-looking number that is false, and every helper checks
for them before answering:

1. **Carried-over jobs.** A re-run lists all 42 jobs, but only the ones it re-executed have
   new timestamps; the rest keep attempt-1 timestamps under an attempt-2 `created_at`, so the
   queue wait computes negative (-23,056 s in one measured case). Including them made tpu-unit
   read 23,271 s against a true 1358 s.
2. **Jobs that never held a runner.** A skipped job and a job cancelled while still queued
   both have `created_at == started_at` and no steps. Eight of them in one run each compute to
   13,843 s of machine time on a machine they never had.
3. **Steps that did not execute.** A cancelled job can list "Run Tests" as `skipped` with a
   zero-length span at the moment the cancellation landed.
4. **A job clock that stops before its own steps do.** GitHub stamps a cancelled job's
   `completed_at` when the cancellation is issued, but the steps already running carry on, by
   up to 33 s in the measured case. Taken literally that made a run's tail negative and let a
   worker's setup plus its suite duration exceed its own run time. A job ends at the later of
   the two clocks.

`None` means "this input cannot answer the question". It is never rounded to zero, because a
zero is a value the dashboard would draw as a real drop.

`worker_count` returns 0 for the Pathways flavors, whose jobs are not named
`Execute Tests (N)`. That is a statement about job names, never "the workers disappeared";
their durations still work.

### `rows.py` - the shapes that get stored

Five row types, their keys, and a JSON round trip. No arithmetic: every field is copied from
a payload, so a stored row can be re-read and re-derived. It does not import `junit.py` -
three Protocols describe the shapes structurally.

```python
ROW_VERSION = 1
class RowError(ValueError)
utc_now_iso() -> str

@dataclass RunRow / JobRow / SuiteRow / TestRow / RescueRow
Row = RunRow | JobRow | SuiteRow | TestRow | RescueRow

run_row(run, pr=None, collected_at=None)
job_row(run, job, collected_at=None)
suite_row(run, entry, collected_at=None)                       # entry = junit.SuiteEntry
test_rows(run, flavor, worker, suite_result, ...) -> list
rescue_rows(run, attempts_jobs, collected_at=None)             # rescues only
failed_never_rescued_rows(run, attempts_jobs, collected_at=None)
row_kind(row) / to_json(row) / from_json(payload)
```

| Kind   | Key                                                                    |
| ------ | ---------------------------------------------------------------------- |
| run    | `run\|<run_id>\|<attempt>`                                             |
| job    | `job\|<run_id>\|<attempt>\|<job_id>`                                   |
| suite  | `suite\|<run_id>\|<attempt>\|<suite_id>`                               |
| test   | `test\|<run_id>\|<attempt>\|<suite_id>\|<worker>\|<classname>\|<name>` |
| rescue | `rescue\|<run_id>\|<job_name>\|<failed_attempt>`                       |

Every part is percent-encoded before joining, so a job name full of slashes and brackets, or
a pytest parameter id containing a `|`, cannot run two parts together or fake a separator.

Three decisions worth knowing:

1. **The test key uses `suite_id`, not `flavor`.** They are the same string for every real
   flavor, but the nested `decoupled` pass runs the same 50 tests again inside `cpu-unit`
   worker 1; keying on the flavor would make each decoupled row overwrite the `cpu-unit` row
   of the same test and silently halve that run's history.
2. **The rescue key carries the failed attempt.** A job that goes failure -> success ->
   failure was rescued at attempt 1 and failed for good at attempt 3, and both facts have to
   be stored. Corrections still work, because a correction keeps the same failed attempt: a
   tick that sees only the failure writes `...|1` with `rescued` False, and the next tick
   writes the same key with `rescued` True and a later `collected_at`.
3. **A failure that was never rescued points at the failure the job ended on**, never at an
   earlier one that was rescued - but at the START of that trailing run of failures, so a job
   that failed three times in a row still reports `rerun_after_failure` True. "Re-run and
   failed again" and "never re-run at all" are different cells on the dashboard.

### `demo.py` - the end-to-end proof

A read-only script that takes a run id and prints every number the dashboard would show for
it: the phase split, per-suite workers and duration next to the JUnit counts, per-job queue
and setup, machine time, rescues, and the keys the rows would be stored under. Nothing is
written and every request is a GET.

```bash
# From anywhere; it puts its own package on sys.path:
GITHUB_TOKEN=$(gh auth token) python3 tools/ci_metrics/collector/demo.py 33468578834

# Options:
#   --attempt N   report that attempt instead of the latest
#   --no-tests    skip the JUnit artifact downloads, which are the slow part
#   --repo O/N    point somewhere other than AI-Hypercomputer/maxtext
```

Two things to look for in the output. Per suite, D sits next to the JUnit seconds precisely
so the gap is visible - tpu-unit reads `27m06s` against `2519.7s of JUnit time`. Per job, a
dash is never a zero: it means the number could not be measured, which is what a job that
never held a runner has to report.

Artifacts live about a day, so on a run older than that the suites read
`no test results (artifact_expired)`. That is the honest answer, not a failure.

### `store.py` - the append-only store on disk

Where a row lives, whether it is there already, and how a reader gets the current version
back. No network, no metric.

```
<out>/data/<kind>-YYYY-MM.ndjson   one JSON object a line, appended, never edited
<out>/data/state.json              which attempts are stored, which are in flight
<out>/views/...                    what the browser loads
```

Five rules hold it up, and the module docstring states them in full:

1. **Append-only, with corrections.** A wrong number is fixed by a second line with the same
   key and a later `collected_at`; `read` returns the last row per key. One kind corrects
   itself without being asked - a rescue is keyed by the failure, not the outcome, so the
   tick that later sees the re-run rewrites it (`MUTABLE_KINDS`).
2. **Writes are atomic per file.** Everything lands in a temporary file and is renamed into
   place, so a killed tick leaves either the old file or the new one. `Store.sweep_temp`
   clears the temporary file a killed tick left behind.
3. **`state.json` is an index, not the truth.** It records run attempts, never rows, keeps
   the newest `MAX_INDEXED_RUNS` run ids, and can be deleted: `load_state` rebuilds it by
   scanning the stored rows. The rebuild reads the job files too, because a run row on its
   own only proves a tick started that attempt, not that it finished it.
4. **Missing is None, never zero.** Rows pass through exactly as `rows.py` built them.
5. **A month closes once.** A tick appends only to the month of the run it is reading.
   `compact_month` is run once against a closed month, and running it twice changes nothing.

The month is the RUN's month, not the row's. Run and rescue rows can name it themselves;
job, suite and test rows take `month=store.month_for_run(run)` from the caller, so a re-run
started in the next month cannot scatter one run across two files.

### `views.py` - stored rows to the JSON the browser loads

```
<out>/views/meta.json              what exists, how many rows, when it was built
<out>/views/<group>-YYYY-MM.json   runs, suites, flaky, queue, workflows
<out>/views/pr/<n>.json            one merged pull request in full
```

Four format rules, plus one join rule:

- **Columnar.** A table is `{"columns": [...], "rows": [[...]]}`. `to_columnar` and
  `from_columnar` are exact inverses, None included.
- **Split by month, rebuilt by month.** Only the open months are rewritten; a closed month's
  file is not even read. Its row counts still reach `meta.json`, off the finished file.
- **No timestamp inside a view file.** `generated_at` is in `meta.json` alone, so a month
  whose rows have not changed serialises to identical bytes and git sees no change.
- **Missing is null.** A suite that published no test file carries null counts and a reason
  code; a suite where only some workers reported carries `is_partial`.
- **One row per pull request, in the runs view only.** A pull request can have two completed
  runs - an earlier push that finished before the next push arrived is kept, not superseded.
  The runs view and `pr/<n>.json` describe one of them; the suites and queue tables carry
  both and mark the chosen one with `is_representative`. Join on `run_id`, or filter on that
  flag.

Every duration in every view is `derive.py`'s answer. The only arithmetic this module does
itself is the median, the overlapping-run count and the nearest scheduled probe.

### `tick.py` - the command a schedule calls

```bash
python3 -m collector.tick --out DIR [--repo owner/name] [--since YYYY-MM-DD]
                          [--until YYYY-MM-DD] [--backfill-days 30]
                          [--dry-run] [--max-runs N]
```

| Exit | Meaning |
| --- | --- |
| 0 | The tick finished. Rows written **or** nothing new - both are success. |
| 1 | Data was lost or put at risk: a fetch failed, a run could not be collected, the store could not be written. Always safe to re-run. |
| 2 | The command line was impossible: a bad date, `--until` before `--since`, no `--out`, an `--out` that is a git checkout. Re-running fails the same way. |

What it decides:

- **Which window.** Everything after the stored watermark, plus a re-listing of the last two
  days - a re-run keeps its run id and creation time, so a watermark alone would never see
  it. With no watermark it backfills, a week at a time, checking the rate budget between
  weeks and stopping cleanly when it runs low.
- **Which rows to keep.** Every run keeps per-flavor totals plus every failed, rescued and
  newly-seen test and the slowest 25 per suite. The day's first scheduled main run keeps
  every test row.
- **When to stop.** `--max-runs` stops it cleanly and says so; the next tick continues.
  `--dry-run` fetches and builds everything into a temporary directory and writes nothing
  to `--out`.

Reading the output: the block names what was seen, written and skipped, and the last line is
the one a log greps.

```
ci-metrics tick: ok | 25 run(s) | 27 attempt(s) | 890 job(s) | 6955 test(s) | 6 view file(s) | 313 API request(s)
ci-metrics tick: nothing new | 0 run(s) | 0 attempt(s) | 0 job(s) | 0 test(s) | 1 view file(s) | 19 API request(s)
```

"nothing new" is the normal answer for a tick that found no new run, and it exits 0. Two
lines are worth reading closely. **Test artifacts** splits harvested runs from the ones
whose artifacts had already expired - "too old to have any" is never the same as "no tests".
**Suites with no file** groups the suites that published nothing by reason, so
`no_file_published: 26 (tpu-pathways-integration, ...)` reads as a fact about those
workflows, not as a collection failure. `meta.json` always changes, because it carries
`generated_at`; nothing else does unless its rows did.

The workflow that would run this on a schedule is a template at
`tools/ci_metrics/deploy/collector.yml`. It does not run from there, and installing it is a
separate decision - its header says what that involves.

## Running the tests

The eight test files are plain `unittest`, so they need nothing beyond the standard library
and `requests`. They are named `*_test.py`, which is what the repository's `pytest.ini`
collects, so pytest runs them too.

```bash
# From the repository root, one file at a time (no dependency beyond requests):
python3 tools/ci_metrics/collector/tests/github_test.py
python3 tools/ci_metrics/collector/tests/junit_test.py
python3 tools/ci_metrics/collector/tests/runs_test.py
python3 tools/ci_metrics/collector/tests/derive_test.py
python3 tools/ci_metrics/collector/tests/rows_test.py
python3 tools/ci_metrics/collector/tests/store_test.py
python3 tools/ci_metrics/collector/tests/views_test.py
python3 tools/ci_metrics/collector/tests/tick_test.py

# Or all of them at once - 544 tests, about five seconds:
python3 -m unittest discover -s tools/ci_metrics/collector/tests \
        -p '*_test.py' -t tools/ci_metrics/collector/tests

# Or with pytest, if it is installed:
python3 -m pytest tools/ci_metrics/collector/tests/
```

Every test runs offline. Each file replaces `socket.socket` and `socket.create_connection`
with a raiser, so a test that reached for GitHub would fail instead of hanging.
`github_test.py` drives the client through a fake session that answers from a scripted queue
of real `requests.Response` objects, and patches out the module's `_sleep` and `_warn` seams
so nothing waits or prints. `runs_test.py` uses two seams: a stub client for the pure rules,
and the real client over a fake session wherever paging itself is the question.
`tick_test.py` replaces `github.GitHubClient` with a fake that answers the eight endpoints a
tick reaches from the saved fixtures, builds artifact zips in memory, and pins the clock, so
a whole tick - fetch, store, views - runs with no network at all.

Every expected number in these files is a measurement of a saved fixture. Where no real data
could be captured - a job with a null `started_at`, a run still in flight - the case is
synthesised and the docstring says so.

Style hooks, run with the versions pinned in `.pre-commit-config.yaml`:

```bash
pylint --rcfile=pylintrc --disable=R0401,R0917,W0201,W0613 tools/ci_metrics/collector/
pyink --pyink-indentation=2 --line-length=122 --check tools/ci_metrics/collector/
```

## About the fixtures - read this before changing them

`tests/fixtures/` holds **real files captured from a real pipeline run on 2026-09-01**, not
hand-written samples:

| File                                                         | What it is                                                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `run.json`                                                   | `GET /actions/runs/33468578834`, verbatim                                                  |
| `jobs.json`                                                  | `GET /actions/runs/33468578834/attempts/1/jobs`, 54 jobs with steps, labels, runner groups |
| `artifacts.json`                                             | `GET /actions/runs/33468578834/artifacts`, 28 artifacts                                    |
| `cpu-unit-1.xml`                                             | cpu-unit worker 1; the `<testsuite tests>` attribute lies (870 against 737 elements)       |
| `decoupled-targeted.xml`                                     | the nested pass, from the **same** artifact as cpu-unit worker 1                           |
| `cpu-unit-3.xml`                                             | the everything-skipped case: 737 collected, 737 skipped, 0 executed                        |
| `gpu-integration-1.xml`                                      | 26 collected / 15 skipped / 11 executed                                                    |
| `cpu-post-training-unit-4.xml`                               | second attribute-lies case (86 against 84)                                                 |
| `tpu-post-training-integration-1.xml`                        | smallest TPU flavor, green                                                                 |
| `tpu-post-training-integration-1.failed-run-33467756955.xml` | the only failing test in the capture, from run 33467756955                                 |

**These cannot be regenerated.** Test-result artifacts have `retention-days: 1`, so the
originals were deleted by GitHub about 24 hours after the run and the API now reports them
as `"expired": true`. Run 33468578834 itself stays in the runs API for 90 days, but its
artifact payloads are gone for good. Treat the files as a permanent record:

- Do not edit them to make a test pass. The numbers in the tests are measurements of these
  exact bytes, and changing a byte invalidates the measurement.
- Do not "clean up" the strings inside them. `jobs.json` mentions `gh-pages` and
  `track_performance` because those are the real upstream job step names. They are a
  recording of what GitHub returned, not a reference this code follows.
- If a new case is needed, capture a new fixture from a **fresh** run within its 24-hour
  artifact window and add it alongside these, keeping the `<flavor>-<worker>` naming.

The expected numbers used across `junit_test.py` were measured from these files and are
stated in `FIXTURE_TRUTH` at the top of that file.

Layer 2 added a second set, captured the same day, covering the shapes that only appear when
a run is re-run or cancelled. They are named for what they prove:

| File                                              | What it proves                                                                                             |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `rerun-32772626658-*`                             | 3 rescues out of 8 attempt-1 failures; 28 of 42 jobs carried over into attempt 2 with attempt-1 timestamps |
| `rerun-33037584699-*`                             | 4 rescues across three device lanes, every attempt-1 failure recovered                                     |
| `cancelled-job-32785979907-*`                     | three attempts, zero rescues: a worker goes cancelled -> success, and cancelled is not failure             |
| `queued-then-cancelled-32999133815-*`             | 8 jobs cancelled before they ever held a runner, and 2 whose steps outlive their own `completed_at`        |
| `runs-list-page*` / `runs-list-short-final-page*` | paging: a Link-header hop, and a short final page that stops the loop                                      |
| `superseded-*` / `cancelled-not-superseded-*`     | both sides of the supersession rule on real branch listings                                                |
| `action-required-run-33465601432-*`               | a run that never executed: `{"total_count":0,"jobs":[]}`                                                   |
| `merged-pr-5070-*` / `fork-pr-5042-*`             | `pull_requests` is empty for a merged same-repo run as well as for a fork run                              |
| `tpu-unit-1.xml` / `tpu-unit-2.xml`               | the JUnit sum is 2519.7 s against a real suite duration of 1626 s                                          |
