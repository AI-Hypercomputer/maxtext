# Going live: what the CI Pulse dashboard needs before it shows real numbers

Written 2026-09-02, from the branch `mesa/gh-pages`.

This document answers these questions:

1. What is missing before the dashboard shows real data instead of mock data?
2. How does the page that is already published get its data today?
3. How does the CI Pulse tab work, and how does the page it opens get data?
4. How do I refresh that data by hand, right now, into this branch?
   (Section 4.3 is the runbook.)
5. Do the `tools/` and `tests/` folders belong on this branch, or on `main`?

Everything below was checked against the files on this branch and against
`main`. Where a number appears, the command that produced it is named, so the
next person can re-check it instead of trusting this page.

---

## 0. What is actually on this branch right now

`git ls-tree -r --name-only HEAD` says this branch tracks **two** folders and
nothing else:

| Folder | Tracked files | What it is |
| --- | --- | --- |
| `dev/` | 3 | the live benchmark site: `index.html`, `data.js`, `per_test_baseline.json` |
| `tools/` | 79 | our CI Pulse work: the collector, the dashboard, their tests |

`MaxText/`, `src/`, `tests/`, `meta_checkpoint/` and `.qodo/` are sitting in the
working directory but **are not tracked here**. They are leftovers from a
previous checkout of a source branch. They are not part of this branch and
deleting them from disk would change nothing in git.

One consequence worth knowing before anything is pushed: this branch is the web
root of `https://ai-hypercomputer.github.io/maxtext/`. Anything tracked here is
publicly downloadable. `dev/bench/index.html` is served at
`/maxtext/dev/bench/`, and `tools/ci_metrics/collector/store.py` would be served
at `/maxtext/tools/ci_metrics/collector/store.py`. Nothing here is secret, but
it is worth deciding on purpose rather than by accident.

---

## 1. How the published page gets its data today (question 2)

Short answer: **the page fetches nothing. The data is pushed into the branch by
CI on `main`, and the page just reads a file that is sitting next to it.**

### The page itself

`dev/bench/index.html` is 11,193 bytes. Its whole data story is two lines:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@2.9.2/dist/Chart.min.js"></script>
<script src="data.js"></script>
```

`data.js` is not JSON that gets fetched. It is a JavaScript file that begins:

```js
window.BENCHMARK_DATA = {
  "lastUpdate": 1788323898290,
  "repoUrl": "https://github.com/AI-Hypercomputer/maxtext",
  "entries": { "MaxText Test Execution Times": [ ... ] }
}
```

The browser loads it as a script, the object lands on `window`, and the page
draws charts from it. There is no API call, no server, no database. GitHub Pages
is a plain file host.

Today `data.js` is 685,437 bytes and holds 209 entries (3,474 individual
measurements).

### Who writes it

The writer lives on `main`, not here:

```
main:.github/workflows/ci_pipeline.yml
     on: schedule, cron '0 */4 * * *'      (every 4 hours, on the hour)
     job track_performance
       -> uses main:.github/workflows/track_performance.yml
```

`track_performance.yml` does four things in order:

1. Downloads the JUnit test-result artifacts from the run that called it
   (`test-results-*-<run_id>`).
2. Fetches the previous `per_test_baseline.json` **out of the gh-pages branch**
   with `git show origin/gh-pages:dev/bench/per_test_baseline.json`.
3. Runs `tests/utils/process_test_results.py`, which writes
   `benchmark-results.json` and a new baseline.
4. Publishes, but **only when the run was started by the schedule and is on
   `main`**:
   - the new baseline, by adding a git worktree on gh-pages, copying the file
     in, committing, and `git push origin HEAD:refs/heads/gh-pages`;
   - the benchmark points, through
     `benchmark-action/github-action-benchmark@v1.13.0` with
     `auto-push: true`, `gh-pages-branch: gh-pages`,
     `benchmark-data-dir-path: dev/bench`.

So a pull request never writes here. Only the 4-hourly scheduled run of `main`
does, six times a day.

### Two things about that writer that matter to us

**It is a normal push, not a force push.** Checking six recent bot commits with
`git show --stat` gives the same shape every time:

```
dev/bench/data.js | 114 +++++++++++++++++++++++++-
1 file changed, 113 insertions(+), 1 deletion(-)
```

It appends a block and closes the object again. It has never rewritten history.
Earlier worry about a force push was unfounded.

**It never touches `index.html` once the file exists.** Every one of those six
bot commits changed `data.js` and nothing else. The last human edit to
`dev/bench/index.html` was commit `908aa8913`, and hundreds of bot commits since
then have left it alone. This is the single most important fact for us: **a
customised `index.html` in `dev/bench/` survives the existing pipeline.** We do
not have to fight the benchmark action to publish our own page.

### The picture

```
  main branch                          gh-pages branch (this one)
  -----------                          --------------------------
  ci_pipeline.yml  (cron 0 */4 * * *)
     |
     v
  test jobs  -->  JUnit artifacts
     |
     v
  track_performance.yml
     |  (only if event=schedule and ref=main)
     +---- push baseline ------------> dev/bench/per_test_baseline.json
     +---- github-action-benchmark --> dev/bench/data.js  (append)
                                            ^
                                            |
                       browser <-- index.html reads it as a <script>
```

---

## 2. What is missing before our dashboard shows real data (question 1)

Our dashboard is `tools/ci_metrics/site/index.html`, 2,172 lines, 168,441 bytes.
Running `grep -c 'fetch(' index.html` returns **0**. It makes no network request
at all. Every number on screen comes from JavaScript constants written into the
file by hand.

The collector that produces real numbers is finished and tested. What is missing
is the wire between them, plus three decisions only a person can make.

### 2.1 The collector already produces the right files

`python -m collector.tick --out DIR --repo AI-Hypercomputer/maxtext` writes:

```
DIR/data/*.ndjson              append-only raw rows, one file per month
DIR/data/state.json            which run+attempt is already stored
DIR/views/meta.json            index: which files exist, when they were built
DIR/views/runs-YYYY-MM.json        tables: runs, jobs
DIR/views/suites-YYYY-MM.json      tables: suites
DIR/views/flaky-YYYY-MM.json       tables: rescues, rescue_tests
DIR/views/queue-YYYY-MM.json       tables: queue
DIR/views/workflows-YYYY-MM.json   tables: workflows
DIR/views/pr/<number>.json         tables: attempts, jobs, steps, suites, tests, errors
```

Each table is columnar: a `columns` list and a `rows` list of arrays, so the
browser can read it without a parser of its own.

It has been proven idempotent on live data: first run read 49,974 tests and
stored 25 runs / 890 jobs / 177 suites / 6,955 tests; a second run with the same
arguments stored 0 of everything.

### 2.2 The gap: about 16 constants have to become file reads

These are the constants in `tools/ci_metrics/site/index.html` that hold data, and
where the real value already exists:

| Constant in the page | Real source |
| --- | --- |
| `COMMITS`, `TIMES`, `RUN_IDS`, `TRIGGERS` | `runs-YYYY-MM.json`, table `runs` |
| `JOBS` (per-job times) | `runs-YYYY-MM.json`, table `jobs` |
| `RUNNER_LABELS` | `runs-YYYY-MM.json`, table `jobs`, column `runner_label` |
| `WK_GROUPS`, `TEST_COUNTS` | `suites-YYYY-MM.json`, table `suites` |
| `DEV_PHASES` | `suites` grouped by lane, or `workflows-YYYY-MM.json` |
| `TH_CATEGORIES`, `TH_COMMITS` | `suites-YYYY-MM.json`, table `suites` |
| `PROBE_Q` | `queue-YYYY-MM.json`, table `queue` |
| `RESCUES` | `flaky-YYYY-MM.json`, table `rescues` |
| `FLAKY_TESTS`, `TESTS` | `flaky-YYYY-MM.json`, table `rescue_tests` |
| `MOCK_TODAY` | `meta.json`, field `generated_at` |

These stay in the file, because they are appearance and not data:
`CAT_COLORS`, `LANE_COLORS`, `WK_LANE_HEX`, `JT`, `GUIDE_OPEN`, `GUIDE_ICON`,
`REPO`.

One real gap, and it is small. `STEPS` and `DOCKER_FRAC` drive the "Image pull"
and "Env setup" segments of the first chart. The only place the collector
publishes per-step times is `pr/<number>.json`, which is one file per pull
request. The monthly `runs` view has `setup_seconds` as a single number with no
split. Three ways out, cheapest first:

1. Drop the two segments and show one "Setup" segment. No collector change.
2. Add an `image_pull_seconds` column to `RUN_JOBS_COLUMNS` in `views.py`.
3. Have the page fetch `pr/<n>.json` when a bar is clicked. More requests, more
   code.

Recommendation: option 2. The data is already in the raw rows, the change is one
column, and the chart keeps the detail the reviewers asked for.

### 2.3 What the wiring itself looks like

Replace the constants with one load step that runs before the first draw:

```js
const BASE = 'ci-metrics/views/';        // relative, so it works wherever it is hosted
const meta = await (await fetch(BASE + 'meta.json')).json();
const months = meta.groups.runs.months;  // which months exist
const runs = await Promise.all(months.map(m =>
  fetch(`${BASE}runs-${m}.json`).then(r => r.json())));
```

Then a small helper turns each columnar table into the array of objects the
drawing code already expects, so the charts themselves do not change. That is
the whole of the work: roughly one loader, one columnar-to-object helper, and
one edit per constant.

This can be developed and tested entirely offline with
`python3 -m http.server` pointed at a directory holding real collector output.
It does not need the hosting decision to be made first.

### 2.4 What still needs a human decision

None of these are code problems.

**Where the collector runs.** `tools/ci_metrics/deploy/collector.yml` is a
complete, actionlint-clean workflow, but it is deliberately not in
`.github/workflows/`. GitHub only starts `schedule` and `workflow_dispatch`
workflows from a repository's **default branch**, so a copy on any other branch
never fires. Installing it means a small pull request to `main` that adds one
file. That pull request adds no job to `ci_pipeline.yml` and changes nothing
about the tests; it adds a separate scheduled workflow that only reads.

**Where the output is stored.** The workflow template writes to a branch named
`ci-metrics-store`, which must be created as an orphan branch before the first
run. It is deliberately not gh-pages.

**What serves the page.** If the dashboard is to live at
`/maxtext/dev/bench/`, then this branch is the answer, and the view JSON files
have to arrive here too. That is a second push target for the collector, and it
is the one decision that touches the branch the existing benchmark page lives
on. It is safe on the evidence in section 1 (the benchmark bot only ever writes
`data.js`), but it should still be an explicit decision, not a side effect.

### 2.5 The mock stories that will change when real data arrives

The mock was built to be reviewable, not accurate. When the real files are
plugged in, these will visibly change and that is correct, not a bug:

- the CPU queue story is currently inverted compared with what the API reports;
- the GPU flaky story is told on `gpu-integration`, which runs 0 tests on
  pull-request runs, with invented test names;
- the test counts per suite are the wrong order of magnitude;
- run `#4940` is drawn as FAIL when the real conclusion was `cancelled`.

---

## 3. The tab that opens our dashboard

This already exists. It was added by hand to `dev/bench/index.html` and it
works. Serving the repository root with `python3 -m http.server` and asking for
both pages returns:

```
GET /dev/bench/                                  200   11,193 bytes
GET /tools/ci_metrics/site/index.html            200  168,441 bytes
GET /dev/bench/data.js                           200  685,437 bytes
```

### How it is built

Three pieces, all inside `dev/bench/index.html`:

**The two buttons**, in the header:

```html
<div class="view-toggle" aria-label="View toggle">
  <button class="toggle-button active" data-view="benchmarks" type="button">Benchmarks</button>
  <button class="toggle-button" data-view="ci" type="button">CI Pulse</button>
</div>
```

**The two panes.** The old page is wrapped in `#benchmarks-page`; ours is an
iframe in `#ci-page`:

```html
<div id="ci-page" class="page-view">
  <iframe id="ci-metrics-frame" title="CI Metrics dashboard"
          src="../../tools/ci_metrics/site/index.html"></iframe>
</div>
```

**The switch**, eleven lines of JavaScript that add and remove one class:

```js
function setActiveView(viewName) {
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.toggle('active', view.id === `${viewName}-page`);
  });
  document.querySelectorAll('.toggle-button').forEach(button => {
    button.classList.toggle('active', button.dataset.view === viewName);
  });
}
```

`.page-view { display: none }` and `.page-view.active { display: block }` do the
rest. The iframe is sized with `height: calc(100vh - 88px)`.

### Why the path works

The benchmark page is served at `/maxtext/dev/bench/index.html`. Going up two
levels lands on `/maxtext/`, which is the root of this branch, and `tools/` is
tracked here. So `../../tools/ci_metrics/site/index.html` resolves to
`/maxtext/tools/ci_metrics/site/index.html`, which is exactly where the file is.

It works **because** `tools/` is committed to this branch. That is the coupling
worth being aware of, and section 5 below recommends removing it.

### The one thing that will break when real data arrives

An iframe has its own document URL. Inside the frame the browser thinks it is at
`/maxtext/tools/ci_metrics/site/index.html`, not at `/maxtext/dev/bench/`. So
when the page starts loading real data, a relative `fetch('views/meta.json')`
resolves to:

```
/maxtext/tools/ci_metrics/site/views/meta.json
```

That means the collector's output would have to be published **into a source
folder**, under `tools/`. That is the wrong place for generated data, and it
also means every collector code change and every data refresh land in the same
directory.

The clean fix is to stop serving the page out of `tools/` at all. Move the
served copy next to its data:

```
dev/bench/index.html                 the benchmark page (unchanged)
dev/bench/data.js                    written by the benchmark bot (unchanged)
dev/bench/ci-pulse/index.html        our dashboard, the served copy
dev/bench/ci-pulse/views/meta.json   written by the collector
dev/bench/ci-pulse/views/runs-2026-09.json
...
```

Then the iframe src becomes a sibling path:

```html
<iframe id="ci-metrics-frame" title="CI Metrics dashboard"
        src="ci-pulse/index.html"></iframe>
```

and the loader inside the dashboard is simply `fetch('views/meta.json')`. Both
paths are short, both stay correct if the site ever moves, and `tools/` no
longer has to exist on this branch at all - which is what section 5 recommends
for its own reasons.

### Three smaller things worth fixing while we are here

None of these are broken. They are rough edges that are cheap to remove.

**The hidden iframe is downloaded on every visit.** A `display: none` iframe
still loads its `src` in every major browser. So someone who only ever looks at
the benchmark charts still pays 168 KB, plus whatever the view files come to,
on every page load. Fix: leave the `src` off the tag and set it the first time
the tab is opened.

```html
<iframe id="ci-metrics-frame" title="CI Metrics dashboard"
        data-src="ci-pulse/index.html"></iframe>
```

```js
if (viewName === 'ci') {
  const frame = document.getElementById('ci-metrics-frame');
  if (!frame.src) frame.src = frame.dataset.src;
}
```

**The tab cannot be linked to or reloaded.** Opening CI Pulse and pressing
refresh returns to the Benchmarks tab, and there is no URL to send a colleague.
Fix: read and write `location.hash`.

```js
setActiveView(location.hash === '#ci' ? 'ci' : 'benchmarks');
// and inside the click handler:
history.replaceState(null, '', viewName === 'ci' ? '#ci' : '#benchmarks');
```

**The 88 px is an assumption.** `height: calc(100vh - 88px)` assumes the header
is exactly 88 px tall. The header has `flex-wrap: wrap`, so on a narrow window
it wraps onto two or three rows, the iframe becomes taller than the space left
for it, and the reader gets an outer scrollbar and an inner one at the same
time. Fix: let flexbox do the measuring instead of hard-coding the number.

```css
body { display: flex; flex-direction: column; height: 100vh; }
#ci-page.page-view.active { flex: 1 1 auto; min-height: 0; }
#ci-metrics-frame { flex: 1 1 auto; height: auto; min-height: 0; }
```

**One thing that is fine and needs no change.** Our dashboard uses
`position: fixed` in three places (the tooltip and two dialogs) and
`position: sticky` once (its own header). Inside an iframe those are measured
against the iframe's own box, not the whole window, so the dialogs cover the
dashboard area and stop below the benchmark page's header. That is the
behaviour we want. The page also never reads `window.parent` or `window.top`,
so it does not care whether it is framed.

---

## 4. Getting real data into the page

Section 2 says what is missing. This section says what to actually do, and how to
refresh the data by hand in the meantime.

### 4.1 The two panes get their data in two different ways

They do not share a mechanism, and they should not.

```
  Benchmarks pane                        CI Pulse pane
  ---------------                        -------------
  <script src="data.js">                 iframe -> ci-pulse/index.html
       |                                        |
       |                                   fetch('views/meta.json')
       |                                   fetch('views/runs-2026-09.json')
       |                                        |
  window.BENCHMARK_DATA                    JSON.parse -> charts
       |
  Chart.js draws
```

The benchmark pane loads a **script** that assigns a global. The CI Pulse pane
loads **JSON with `fetch`**. The difference matters for one practical reason: a
script tag works from a `file://` URL, and `fetch` does not. So once the
dashboard reads files, opening it by double-clicking stops working and the
folder has to be served:

```
cd dev/bench
python3 -m http.server 8000
# then open http://localhost:8000/#ci
```

That is the only workflow change for anyone developing the page.

### 4.2 What to change inside the dashboard

Everything below goes in `dev/bench/ci-pulse/index.html`. Nothing in
`dev/bench/index.html` has to change at all - the tab already works, and the
iframe does not care what its contents fetch.

**Step one: a helper that turns a columnar table into rows of objects.** The
collector writes `{"columns": [...], "rows": [[...], ...]}` to keep the files
small. The drawing code wants objects, so convert once at load time:

```js
function rowsOf(table) {
  if (!table) return [];
  const cols = table.columns;
  return table.rows.map(r => {
    const o = {};
    for (let i = 0; i < cols.length; i++) o[cols[i]] = r[i];
    return o;
  });
}
```

**Step two: read `meta.json` first, then only the months it lists.** `meta.json`
is the index. It names which month files exist for each group, so the page never
guesses a filename and never asks for a file that is not there.

```js
const BASE = 'views/';   // relative to ci-pulse/index.html

async function loadGroup(meta, group) {
  const months = meta.groups[group].months;         // e.g. ["2026-08", "2026-09"]
  const files = await Promise.all(months.map(m =>
    fetch(`${BASE}${group}-${m}.json`).then(r => r.json())));
  const out = {};
  for (const file of files) {
    for (const [name, table] of Object.entries(file.tables)) {
      (out[name] = out[name] || []).push(...rowsOf(table));
    }
  }
  return out;                                        // {runs: [...], jobs: [...]}
}

async function loadAll() {
  const meta = await (await fetch(BASE + 'meta.json')).json();
  const [runs, suites, flaky, queue] = await Promise.all([
    loadGroup(meta, 'runs'),
    loadGroup(meta, 'suites'),
    loadGroup(meta, 'flaky'),
    loadGroup(meta, 'queue'),
  ]);
  return {meta, runs, suites, flaky, queue};
}
```

**Step three: replace the constants and draw after the load, not before.**
Today the page draws as soon as the script runs, because the data is already
there. With files it has to wait:

```js
loadAll().then(d => {
  COMMITS     = d.runs.runs;
  JOBS_ROWS   = d.runs.jobs;
  WK_GROUPS   = d.suites.suites;
  RESCUES     = d.flaky.rescues;
  FLAKY_TESTS = d.flaky.rescue_tests;
  PROBE_Q     = d.queue.queue;
  TODAY       = new Date(d.meta.generated_at);
  renderAll();
}).catch(err => {
  document.getElementById('page-main').innerHTML =
    '<div class="card">The data files could not be loaded: ' + err.message + '</div>';
});
```

The `const` declarations become `let`, and every chart function stays exactly as
it is. Section 2.2 has the full constant-to-table mapping.

**Step four: show how old the data is.** `meta.json` carries `generated_at` and
`uncollected_runs`. Both belong on screen, because a dashboard that is quietly
six hours stale is worse than one that says so.

### 4.3 Refreshing the data by hand

This is how to put real numbers into the branch today, without waiting for the
scheduled workflow to be installed. Every command below was run for real while
writing this document, and the output shown is the actual output.

**There is a script that does all of it:**

```
tools/ci_metrics/deploy/refresh-data.sh            # collect, then copy the views in
tools/ci_metrics/deploy/refresh-data.sh --serve    # and open it in a browser
tools/ci_metrics/deploy/refresh-data.sh --help     # every option
```

It checks the token and the branch before it starts, refuses to put the store
inside the repository, refuses to publish a rebuild that is smaller than what is
already published, keeps the previous views so a mistake is recoverable, and
does not commit unless you pass `--commit`. The rest of this section is what it
does, step by step, for when you want to do it by hand or need to understand a
message it printed.

### The steps the script runs

**What you need.** Python 3.12, the `requests` package, and a GitHub token. The
token has to be able to read Actions data on `AI-Hypercomputer/maxtext`: a
classic personal access token with the `repo` scope, or a fine-grained token
with `Actions: read` and `Metadata: read`. Without a token the API allows 60
requests an hour, which is not enough for even one run.

**Step 1 - collect into a store outside the repository.** The store is the raw
history. Keep it out of the working tree so a stray `git add` cannot pick it up.

```
export GITHUB_TOKEN="$(gh auth token)"      # or paste a token
cd tools/ci_metrics
python3 -m collector.tick \
  --out ~/ci-metrics-store \
  --repo AI-Hypercomputer/maxtext \
  --backfill-days 30
```

A real run prints a report, and one grep-able line at the end:

```
CI metrics tick - AI-Hypercomputer/maxtext
------------------------------------------
  Mode                       backfill (2026-08-30T08:53:36Z .. now)
  Runs seen                  3 (0 already stored, 3 collected, 0 failed)
  Jobs written               127
  Rescue events written      11
  Test artifacts             0 run(s) harvested, 3 too old to have any
  Months touched             2026-08
  Views                      6 file(s) written, 0 unchanged
  API requests spent         10 (+0 downloads)

ci-metrics backfill: ok | 3 run(s) | 127 job(s) | 6 view file(s) | 10 API request(s)
```

Three useful things to read out of that:

- It cost **10 API requests for 3 runs**, so roughly 3 to 4 requests per run.
  With 5,000 an hour a 30-day backfill is affordable, but it is not instant.
  `--max-runs N` stops early and the next run continues where it left off.
- **"3 too old to have any"** is normal, not an error. GitHub deletes test
  artifacts after about a day, so backfilled runs have job timings but no test
  counts. Only runs from the last day carry test rows. That is exactly why the
  scheduled collector matters: it is the only way to catch the artifacts before
  they expire.
- Exit code 0 means the tick finished. 1 means it lost data and should be run
  again. 2 means the command line was wrong.

**Step 2 - look at what it produced.**

```
~/ci-metrics-store/
  data/run-2026-08.ndjson        the raw history, append-only
  data/job-2026-08.ndjson
  data/rescue-2026-08.ndjson
  data/state.json                which run+attempt is already stored
  views/meta.json                the index the page reads first
  views/runs-2026-08.json
  views/suites-2026-08.json
  views/flaky-2026-08.json
  views/queue-2026-08.json
  views/workflows-2026-08.json
```

For that 3-run sample, `data/` came to 268 KB and `views/` to 24 KB. Only
`views/` is published. `data/` never goes on the website - it is what the views
are rebuilt from, and it belongs on its own branch.

**Step 3 - copy only the views into the branch.**

```
cd <repo root>
rm -rf dev/bench/ci-pulse/views
cp -R ~/ci-metrics-store/views dev/bench/ci-pulse/views
```

The `rm -rf` first is deliberate. A plain copy would leave behind a month file
the collector has since dropped, `meta.json` would no longer list it, and the
result is a stale file nothing points at - the confusing kind of leftover.

**Step 4 - check it in a browser before committing.**

```
cd dev/bench && python3 -m http.server 8000
# open http://localhost:8000/#ci
```

Read the freshness line first. If it does not match `generated_at` in
`views/meta.json`, the browser is showing a cached copy - hard-reload.

**Step 5 - commit only the views.**

```
git add dev/bench/ci-pulse/views
git status --short          # confirm nothing else is staged
git commit -m "ci-metrics: refresh dashboard data"
```

Check `git status --short` every time. This repository has a long list of files
that must never be committed, and a `git add -A` has staged 98 of them before.

**Repeating it later.** Point `--out` at the same store and leave `--since` off.
The collector reads its own watermark out of `state.json` and asks only for what
is new, so the second run is far cheaper than the first. Running it twice with
the same arguments is safe: a verified repeat stored 25 runs the first time and
0 the second.

### Does it append, or does it replace?

Both, at different layers, and the difference is worth being clear about.

**The store appends.** `data/*.ndjson` is append-only by design: a tick adds
lines and never edits one, and a correction is written as a new line rather than
a change to an old one. `state.json` records which run and attempt is already
there, so re-reading a window costs nothing. Keep pointing `--store` at the same
directory and the history grows without limit.

**The published views are rebuilt, not appended.** `views/*.json` is a rendering
of the store, so the script replaces the whole folder each time. That is not
data loss, because the store it was rendered from still holds everything - the
rebuild always covers what the last one did **and more**.

It stops being true in exactly one case: when the store is a different or
emptier one than last time. A fresh clone, a wiped directory, a shorter
`--since` - any of those produce a rebuild that is smaller than what is already
on the branch, and replacing it whole would delete history that exists nowhere
else.

The script checks for that and stops:

```
  The rebuild is smaller than what is already published:
    - runs: would lose month(s) 2026-08
    - queue: 7 row(s) published, 0 in the rebuild

  This normally means the store is not the one the published views came
  from. Point --store at the store you used last time and run again; it
  appends, so nothing is lost. Use --force only if you mean to shrink it.
```

It also copies the previous views to `<store>/views-previous/` before replacing
them, so even a `--force` is recoverable.

**So: keep the store.** It is the thing that accumulates. Losing it does not
lose the published data, but it does mean the next refresh can only reach back
as far as the API still goes, and test-level rows for older runs are gone for
good - GitHub deletes those artifacts after about a day.

**One thing the script never touches:** `dev/bench/data.js` and
`dev/bench/per_test_baseline.json`. Those belong to the benchmark bot on `main`,
and nothing here reads or writes them.

### 4.4 What the automatic path will do instead

The manual steps above are the scheduled workflow, done by hand. Once
`.github/workflows/ci_metrics_collector.yml` is merged into `main`, the same
thing happens every four hours at twenty minutes past the hour: check out the
store branch, run one tick, commit the store, push. The only piece the template
does not have yet is step 3 - copying `views/` onto this branch - because where
the page is served from was undecided when it was written. That is one more
step in the workflow, and it should be added at the same time as the hosting
decision.

Two writers will then be pushing to this branch: the benchmark bot on the hour,
and the collector at twenty past. They touch different files
(`dev/bench/data.js` against `dev/bench/ci-pulse/views/`), and the collector
already retries with a rebase up to three times, so they will not fight each
other. The twenty-minute offset exists for exactly this reason.

---

## 5. Which folders belong on which branch (question 3)

The honest answer is that they belong in **three different places**, and putting
them all in one is what makes this confusing.

| What | Where it belongs | Why |
| --- | --- | --- |
| the dashboard page, served as `dev/bench/ci-pulse/index.html` | **this branch** (gh-pages) | it is the page the tab opens; it has to be here to be reachable, and next to its data so relative paths work (section 3) |
| `tools/ci_metrics/site/index.html` (the source) | **`main`** | it is edited and reviewed there; the copy under `dev/bench/ci-pulse/` is the build output |
| `tools/ci_metrics/collector/` (9 modules) | **`main`** | GitHub only runs scheduled workflows from the default branch, and the workflow checks the collector out by ref |
| `tools/ci_metrics/collector/tests/` (49 files, 544 tests) | **`main`**, with the collector | they are the collector's regression net and must run in the collector's pull requests |
| `tools/ci_metrics/site/tests/` (14 files, 247 assertions) | **`main`**, with a copy of the page | they are jsdom checks; they need node, not a web host |
| `tools/ci_metrics/deploy/collector.yml` | copied to **`main`** as `.github/workflows/ci_metrics_collector.yml` | see above |
| the view JSON files the collector writes | **wherever the page is served from** | the page reads them with a relative path |

The top-level `tests/` folder on disk is not ours and is not tracked on this
branch; it belongs to MaxText itself and lives on `main`.

### The recommendation

Treat `main` as the home of the source and this branch as the home of the built
site, the same way the existing benchmark page already works.

Concretely:

- **`main` gets a pull request** adding `tools/ci_metrics/` (collector, site
  source, both test suites) and `.github/workflows/ci_metrics_collector.yml`.
  This is where review happens and where CI can run the 544 Python tests and 247
  dashboard assertions.
- **This branch gets only what is served**: `dev/bench/ci-pulse/index.html` and
  `dev/bench/ci-pulse/views/*.json`. No Python, no test fixtures, no
  `__pycache__`, and no `tools/` folder at all - once the iframe points at
  `ci-pulse/index.html` instead of `../../tools/...`, nothing on the web needs
  `tools/` to be here.

Keeping the Python and the 41 test fixtures (2.0 MB) on the branch that backs a
public website means shipping them to the web for no benefit, and it means every
collector change produces a commit on the branch the benchmark bot is also
pushing to. Separating them removes both problems.

If the team would rather have one branch than two for now, that works too, but
then this branch should be the one that is reviewed, and the `main` pull request
still has to happen for the scheduled workflow regardless.

---

## 6. Suggested order of work

1. ~~**Move the served page to `dev/bench/ci-pulse/`**~~ - **done.** The page is
   now at `dev/bench/ci-pulse/index.html`, the iframe points at
   `ci-pulse/index.html`, and the three fixes from section 3 are in: the iframe
   loads only when the tab is opened, the URL remembers the tab, and the height
   is measured from the real header instead of assuming 88 px. The jsdom suite
   covers all of it in `check_tab.js` (267 assertions across 11 files, green).
2. **Wire the page to files** (no decisions blocked). Add the loader, replace
   the ~16 data constants, keep every chart drawing exactly as it does now.
   Prove it with `python3 -m http.server` over real collector output. Update the
   jsdom checks to serve fixtures instead of reading baked-in constants.
3. **Add the `image_pull_seconds` column** to `views.py` so the first chart keeps
   its setup split.
4. **Fix the four mock stories** so nothing on screen contradicts the API.
5. **Open the `main` pull request**: `tools/ci_metrics/` plus the scheduled
   workflow. Nothing runs until this merges.
6. **Create the store branch** as an orphan branch, then run the workflow once by
   hand with `workflow_dispatch` and `backfill_days: 30`.
7. **Decide the hosting question** and, if the answer is this branch, add the
   publish step that copies `views/` into `dev/bench/ci-pulse/views/`.
8. **Delete the mock constants and drop `tools/` from this branch** once real
   data is confirmed on the served page.

Step 1 is done. Steps 2 to 4 are ours and can start now. Steps 5 to 7 need
someone with merge rights on `main` and a decision on where the store lives.

Until step 5 lands, section 4.3 is the way to get real numbers on the page: run
the collector by hand, copy `views/` into `dev/bench/ci-pulse/views/`, commit.
