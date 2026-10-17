---
name: maxtext-pr-upload
description: >-
  Prepares, validates, uploads, and shepherds pull requests to the
  AI-Hypercomputer/maxtext GitHub repository through CI and code-owner review.
  Use when formatting or linting MaxText code, writing MaxText commit messages
  or PR descriptions, stacking or rebasing dependent PRs, diagnosing MaxText CI
  failures or missing pull-ready labels, or resolving reviewer comments.
---

# MaxText Pull Request Validation & CI Workflow

Merging into `AI-Hypercomputer/maxtext` requires:
1. **Exactly one commit per PR** (`AddPullReady` blocks multi-commit PRs).
2. **Green required CI** (`RunPrecommitChecks.yml` + `MaxText Package Tests`).
3. **Code-owner approvals** (`.github/CODEOWNERS`, `>= 2` unique approvers for `pull ready`).

| Resource | Purpose |
| :--- | :--- |
| [`scripts/ci_status.py`](scripts/ci_status.py) | Queries PR commit count, active vs. superseded CI runs, and blocking vs. non-blocking job failures |
| [`scripts/pr_comments.py`](scripts/pr_comments.py) | Fetches review states and inline comment threads, flagging unanswered reviewer comments |
| [`scripts/check_doc_train_cmds.py`](scripts/check_doc_train_cmds.py) | Extracts `maxtext.train` commands from Markdown docs and validates them through `pyconfig` |
| [`references/ci_mechanics.md`](references/ci_mechanics.md) | Job step failure classification, `_Cfg` and `hlo_diff_test.py` CI traps, and `pull ready` rules |
| [`references/stacked_prs.md`](references/stacked_prs.md) | Bottom-up merge, retargeting, and rebase sequence for stacked PRs |

---

## 1. Branch Shape & One-Commit Rule

Every PR must have `git rev-list --count <base>..<branch> == 1`:
- **Independent PRs** branch directly from `origin/main`.
- **Dependent PRs** stack on their parent PR branch (`base = <parent-branch>`) with exactly 1 commit on top of the parent. See [`references/stacked_prs.md`](references/stacked_prs.md).
- **Review fixes:** Amend the existing commit (`git commit --amend`) and force-push with `--force-with-lease=<branch>:<expected_sha>`. Never add a second fixup commit.
- **Avoid double-push races:** When force-pushing a stacked chain, wait ~45s between branches so GitHub Actions does not cancel the child PR's `synchronize` run and leave a permanent `cancelled` status check.

---

## 2. Commit Message & PR Description Format

Keep commit messages and PR descriptions concise and free of local machine paths:

```markdown
# Description

<Concise summary of root cause, design change, or feature.>

Fixes #<issue_id>

# Tests

<Unit tests executed (`pytest -q ...`) and end-to-end verification summary.>

# Checklist

Before submitting this PR, please make sure (put X in square brackets):
- [X] I have performed a self-review of my code. For an optional AI review, add the `gemini-review` label.
- [X] I have necessary comments in my code, particularly in hard-to-understand areas.
- [X] I have run end-to-end tests and provided workload links above if applicable.
- [X] I have made or will make corresponding changes to the doc if needed, including adding new documentation pages to the relevant Table of Contents (toctree directive) as explained in [our documentation](https://maxtext.readthedocs.io/en/latest/development.html#adding-new-documentation-files).
```

`Require Checklist` fails if any `- [X]` box in `# Checklist` is unchecked.

---

## 3. Local Pre-Push Gates

Run the exact linters pinned in `.pre-commit-config.yaml` (`pyink==24.10.1`,
`pylint==3.3.8`, `codespell==2.4.1`, `yamllint==1.35.0`) and serial unit tests
before pushing:

```bash
export PYTHONPATH=$PWD/src XLA_FLAGS=--xla_force_host_platform_device_count=4
FILES="<changed .py files>"

pyink --pyink --pyink-indentation=2 --line-length=122 --check $FILES
pylint --rcfile=pylintrc --disable=R0401,R0917,W0201,W0613 \
  --ignore-patterns='.pytype,.*pyi$' \
  --ignore-paths='src/maxtext/input_pipeline/protos|src/maxtext/eval/simple_evals_template' $FILES
yamllint -c .yamllint <changed .yml files>
codespell -L ND,nd,sems,TE,te,als,ROUGE,rouge,astroid,ags,dout <changed files>
python3 -m pytest -q -p no:cacheprovider <affected *_test.py files>
```

### Common MaxText Test & CI Traps
- **Test file discovery:** `pytest.ini` only collects `*_test.py` and `*_tests.py`. Files named `test_*.py` are silently ignored in CI.
- **Run `pytest` serially (no `-n`):** Parallel workers contend on local XLA/TPU lockfiles.
- **Touching `train_step` / `loss_fn` / `eval_step`:**
  1. Any new `config.<attr>` read there must also be added to `_Cfg` in `tests/unit/train_nnx_test.py`.
  2. Any change to the compiled HLO of reference models (`default`, `gpt3-175b`, `llama2-7b`) breaks `tests/integration/hlo_diff_test.py` unless gated behind an opt-in config flag that defaults off.
- **Markdown docs with `maxtext.train` commands:** Validate every documented command block with `check_doc_train_cmds.py`:
  ```bash
  python3 .agents/skills/maxtext-pr-upload/scripts/check_doc_train_cmds.py <doc.md> \
    --start "<section heading>" --end "<next heading>" \
    --subst BASE_OUTPUT_DIRECTORY=/tmp/out --keys model_name,per_device_batch_size
  ```

---

## 4. CI Triage & Reviewer Comments

```bash
# Check commit count, active workflow run, and blocking vs. non-blocking jobs
python3 .agents/skills/maxtext-pr-upload/scripts/ci_status.py <pr_number>

# Fetch reviews and unanswered inline comment threads
python3 .agents/skills/maxtext-pr-upload/scripts/pr_comments.py <pr_number> --full
```

- `Track Test Performance / Track Test Duration` is non-blocking (`All Required Tests Passed` excludes it).
- See [`references/ci_mechanics.md`](references/ci_mechanics.md) for distinguishing real test failures from runner preemptions (exit 130) and diagnosing missing `pull ready` labels.
