---
name: maxtext-pr-upload
description: >-
  Prepares, validates, uploads, and shepherds pull requests to the
  AI-Hypercomputer/maxtext GitHub repository through CI and code-owner review.
  Use when committing or force-pushing a MaxText PR branch, opening or
  retargeting a stacked MaxText PR, rebasing dependent PRs, writing MaxText
  commit messages or PR descriptions, running pre-push static/dynamic review
  gates, diagnosing why MaxText CI is red or a PR is not marked pull ready, or
  fetching and resolving reviewer comments on a MaxText PR. Don't use for TPU
  cluster job submission or offline checkpoint conversion.
---

# Uploading and Shepherding MaxText PRs (`AI-Hypercomputer/maxtext`)

MaxText merges require **1 commit per PR**, green required CI, and
`.github/CODEOWNERS` approval (`>= 2` unique approvers for `pull ready`).

- **Pre-pass, Tiers A-C, Pass-0 local review, Gate C, and 2-Incident Bar**:
  [references/review_gates.md](references/review_gates.md).
- **Stacked PR rebase and retargeting**:
  [references/stacked_prs.md](references/stacked_prs.md).
- **Deep CI internals (`hlo_diff_test`, `_Cfg`, job step classification,
  `pull ready`)**: [references/ci_mechanics.md](references/ci_mechanics.md).

| Asset | Purpose |
| :--- | :--- |
| [scripts/audit_repo_and_ci_contracts.py](scripts/audit_repo_and_ci_contracts.py) | Audits torch/TF-free `cpu-unit` imports, `_Cfg`/mock blast radius, `EAGER_VS_JIT_GRAD_COMPARE`, and schema-backed `SILENT_CONFIG_FALLBACK` |
| [scripts/audit_pr_review_hygiene.py](scripts/audit_pr_review_hygiene.py) | Audits Python AST & `.sh` review blind spots (`BENCHMARK_CRIPPLED_XLA_FLAG`, test-laundering, tracer skips, oracle edits, undefined locals) |
| [scripts/sweep_param_mapping_contracts.py](scripts/sweep_param_mapping_contracts.py) | Sweeps `PARAM_MAPPING` / `HOOK_FNS` under BF16, null-subconfig `config.json`, `to_dict()`, FP8 scale safety, and padded tuple splits |
| [scripts/pilot.py](scripts/pilot.py) | Orchestrates Tiers A-C (`abstract_model_diff`, `mini_aot_fingerprint`, `caller_impact_tests`) across base/head worktrees |
| [scripts/abstract_model_diff.py](scripts/abstract_model_diff.py) | Tier A: abstract `nnx.eval_shape` cross-model param/dtype/PartitionSpec/Muon diff + `sharding-sweep` |
| [scripts/mini_aot_fingerprint.py](scripts/mini_aot_fingerprint.py) | Tier B: `tpu7x-8` AOT `train_compile` HLO fingerprint diff + `@dot_product`, `@remat_*`, `@offload_*`, and `@moe_*` variants |
| [scripts/caller_impact_tests.py](scripts/caller_impact_tests.py) | Tier C: static import-graph caller selection and 4-CPU head-vs-base regression `pytest` runner |
| [scripts/build_local_review_prompt.py](scripts/build_local_review_prompt.py) | Builds a full-diff local review prompt from `.gemini/commands/maxtext-review-*.md` + subsystem-filtered KB rules |
| [scripts/review_scope_gate.py](scripts/review_scope_gate.py) | Enforces 1-PR `review-scope` coverage, bans `APPROVED_WITH_FOLLOWUPS`, and gates `--strict-pre-push` on `APPROVED` |
| [scripts/check_doc_train_cmds.py](scripts/check_doc_train_cmds.py) | Validates documented `maxtext.train` commands in Markdown through `pyconfig` and flags `\` comment truncation |
| [scripts/ci_status.py](scripts/ci_status.py) | Queries head SHA, commit count, `per_page=100` runs/checks, `attempts/{attempt}/jobs?per_page=25`, and blocking CI verdict |
| [scripts/pr_comments.py](scripts/pr_comments.py) | Queries `per_page=100` reviews, inline threads, and `[TOP-LEVEL]` `issues/<N>/comments` unanswered status |
| [resources/learned_lessons.jsonl](resources/learned_lessons.jsonl) | Curated MaxText `must_catch` and `do_not_flag` review invariants (validated by [resources/learned_lessons_schema.json](resources/learned_lessons_schema.json)) |

## 1. Ground Rules and Branch Shape

- **Isolated worktree and lease-pinned pushes**: Work in a dedicated git
  worktree (`git worktree add`), back up tips with tags
  (`git tag -f bak/<slug>-<sha> <sha>`), and push only feature branches via
  `--force-with-lease=<branch>:<remote_sha>`. Set `push.default=nothing` and
  never pass `--no-verify`.
- **One commit per PR**: `git rev-list --count <base>..HEAD` must equal `1`
  (`AddPullReady` blocks multi-commit PRs). Amend and force-push review fixes,
  then rebase stacked children bottom-up. Never retarget a child to `main`
  before its parent merges, and never delete a branch that is the base of an
  open PR.
- **`gh` safety**:
  - Space stacked pushes `>= 45s` apart so GitHub Actions does not
    concurrency-cancel child runs.
  - **Never** manually add `"pull ready"`, run `gh pr merge|close|ready`
    without explicit maintainer/author confirmation, or skip required CI suites
    (always remove auto-applied `skip-ci-tests` via
    `gh pr edit <pr> --remove-label skip-ci-tests` when full CI should run).
  - Use `gh >= 2.102` (`gh 2.46`/`2.63` fail `gh pr edit` on deprecated
    classic-project GraphQL fields). After changing a PR's base branch, toggle a
    non-merge label (such as `gemini-review`) once CI settles, because base
    edits alone do not trigger `ci_pipeline.yml`.

## 2. Commit Message and PR Description

- **Commit message**: imperative subject (`<= 72` chars), 1-3 sentences on the
  root cause, and concise bullets for minimal changes and tests. Omit local
  filesystem paths; place issue references (`Fixes #<id>`) in the body rather
  than the PR title.
- **PR body**: include `# Description`, `# Tests` (exact unit/integration test
  commands, negative-control proof, and verbatim log excerpts when hardware
  runs were executed), and all 4 `[X]` boxes under `# Checklist`
  (`Require PR Checklist` fails if any box is unchecked). Keep the PR title and
  description synchronized whenever the diff changes during review.

## 3. Pre-Push Lint, Unit Test, and Mandatory Review Gate (Run Before `git push`)

Before pushing or opening a PR (Section 4), complete all 4 steps in order from
the target worktree (`WT=$PWD`):

1. **Pinned linters and torch-free wheel-mode `pytest`**:

   ```bash
   WT="$PWD"
   MPU="$WT/.agents/skills/maxtext-pr-upload"
   export PYTHONPATH="$WT/src:$WT"
   export XLA_FLAGS=--xla_force_host_platform_device_count=4
   F="<changed .py files>"
   pyink --pyink --pyink-indentation=2 --line-length=122 --check $F
   pylint --disable=R0401,R0917,W0201,W0613 \
     --ignore-patterns='.pytype,.*pyi$' \
     --ignore-paths='src/maxtext/input_pipeline/protos|src/maxtext/eval/simple_evals_template|src/maxtext/experimental/lineage' $F
   pylint -e W0105 $F
   yamllint -c .yamllint <changed .yml files>
   codespell -L ND,nd,sems,TE,te,als,ROUGE,rouge,astroid,ags,dout,VALU,valu <changed files>
   (cd /tmp && JAX_ENABLE_COMPILATION_CACHE=0 PYTHONPATH="$WT/src:$WT" \
     XLA_FLAGS=--xla_force_host_platform_device_count=4 \
     python3 -m pytest -q -p no:cacheprovider "$WT/<affected *_test.py>")
   ```

2. **Deterministic pre-pass**
   ([references/review_gates.md](references/review_gates.md) Section 1):

   ```bash
   python3 "$MPU/scripts/audit_repo_and_ci_contracts.py" \
     --repo-root "$WT" --base origin/main --run-torch-free-import
   python3 "$MPU/scripts/audit_pr_review_hygiene.py" \
     --repo-root "$WT" --base origin/main
   # When checkpoint_conversion / param_mapping / load_dynamic / hf_streaming_load changed:
   PYTHONPATH="$WT/src" python3 "$MPU/scripts/sweep_param_mapping_contracts.py" \
     --maxtext-root "$WT"
   # When Markdown docs containing maxtext.train commands changed:
   PYTHONPATH="$WT/src" python3 "$MPU/scripts/check_doc_train_cmds.py" \
     <doc.md> --start "<start heading>" --end "<end heading>"
   ```

3. **Dynamic `pilot.py` Tiers A-C** (must run under a MaxText virtualenv with
   `jax` and `pytest`, via activated `python3`, `--python`, or
   `$MAXTEXT_PILOT_PYTHON`):

   ```bash
   VENV_PY="${MAXTEXT_PILOT_PYTHON:-$(command -v python3)}"
   "$VENV_PY" "$MPU/scripts/pilot.py" \
     --repo-root "$WT" --python "$VENV_PY" \
     --label "<pr_slug>" --head HEAD --base origin/main
   ```

   Runs behavioral cross-model abstract diff + `sharding-sweep` (Tier A),
   `tpu7x-8` AOT `train_compile` HLO fingerprint diff + policy variants
   (Tier B), and AST caller-impact `pytest` regression checks (Tier C); fails
   closed (`rc=2`) if `VENV_PY` lacks `jax` or `pytest`, or (`rc=1`) if any tier
   detects a regression.

4. **Local Pass-0 review and strict scope gate**:

   ```bash
   python3 "$MPU/scripts/build_local_review_prompt.py" \
     --maxtext-root "$WT" --base origin/main --head "$(git rev-parse HEAD)" \
     --target "PR#<num_or_slug>" --description-file /tmp/pr_body.md \
     --findings-out /tmp/local_review.md --out /tmp/review_prompt.md
   # Run an independent read-only reviewer on /tmp/review_prompt.md -> /tmp/local_review.md, then enforce:
   python3 "$MPU/scripts/review_scope_gate.py" \
     --report /tmp/gate_c_report.md --repo-root "$WT" \
     --base origin/main --head "$(git rev-parse HEAD)" \
     --findings-file /tmp/local_review.md --strict-pre-push
   ```

Key traps:

- **`PYTHONPATH="$WT/src:$WT"` and wheel-mode `cwd=/tmp`**: setting
  `PYTHONPATH` ensures a worktree does not accidentally import an editable
  install from another checkout, while running `pytest` from `/tmp` avoids CWD
  shadowing and verifies package-mode imports.
- **Pinned linters**: match `.pre-commit-config.yaml` (`pyink==24.10.1`,
  `pylint==3.3.8`, `codespell==2.4.1`, `yamllint==1.35.0`). Omit
  `--rcfile=pylintrc` so `pyproject.toml` rules execute, and run
  `pylint -e W0105` explicitly to catch displaced docstrings.
- **Test collection and `codecov/patch`**: name unit tests `*_test.py`
  (`pytest.ini` ignores `test_*.py`). Never put module-level `import torch`,
  module-level `pytest.importorskip("torch")` /
  `pytest.importorskip("jetstream")`, or class-level
  `@unittest.skipUnless(HAS_TORCH)` on mixed pure-JAX test classes in
  `tests/unit/*_test.py`. Prove each new test fails when the fix hunk is
  reverted.
- **`train_step` / `loss_fn` / `eval_step`**: add any new `config.<attr>` read
  to `_Cfg` in `tests/unit/train_nnx_test.py` and gate new behavior behind an
  opt-in `false` default in `base.yml` so `tests/integration/hlo_diff_test.py`
  reference HLO dumps stay unchanged.
- **Doc commands**: validate documented `maxtext.train` invocations with
  `.agents/skills/maxtext-pr-upload/scripts/check_doc_train_cmds.py` (a `#`
  comment line inside a `\` continuation truncates subsequent flags; keys set
  in a model YAML require `override_model_config=true`).

## 4. Pushing, Opening PRs, and Resolving Review Comments

```bash
git push --force-with-lease=<branch>:<remote_sha> origin <branch>
sleep 45  # space stacked pushes >= 45s
gh pr create --repo AI-Hypercomputer/maxtext \
  --head <branch> --base <base> --title "<title>" --body-file /tmp/pr_body.md
gh pr edit <pr> --repo AI-Hypercomputer/maxtext --remove-label skip-ci-tests
```

- Do not add `gemini-review` immediately after `gh pr create`
  (`gemini_dispatch.yml` triggers automatically on `opened` /
  `ready_for_review`, and an immediate label edit concurrency-cancels the
  initial run).
- Poll CI at `>= 6 min` intervals with
  `python3 .agents/skills/maxtext-pr-upload/scripts/ci_status.py <pr>`
  (uses `per_page=100` and `attempts/{attempt}/jobs?per_page=25`; ignores
  non-blocking `Track Test Performance / Track Test Duration`).
- Inspect feedback via
  `python3 .agents/skills/maxtext-pr-upload/scripts/pr_comments.py <pr> --full`,
  which checks **both** inline review threads and substantive top-level PR
  comments (`[TOP-LEVEL]`) — both must have `0` unanswered items before
  requesting final sign-off or merging. Verify every claim against live code,
  fix real defects test-first, and clear all 3 post-upload gates in
  [references/review_gates.md](references/review_gates.md).
