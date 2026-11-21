# MaxText PR Review and Verification Gates

Run these gates before uploading a PR, before requesting human code-owner
review on an open PR, and when reviewing changes against `AI-Hypercomputer/maxtext`.

## 1. Mandatory Deterministic Pre-Pass (Before LLM Review)

```bash
WT="$PWD"
MPU="$WT/.agents/skills/maxtext-pr-upload"
BASE_REF="origin/main"
LIVE_HEAD="$(git -C "$WT" rev-parse HEAD)"

python3 "$MPU/scripts/audit_repo_and_ci_contracts.py" \
  --repo-root "$WT" --base "$BASE_REF" --run-torch-free-import
python3 "$MPU/scripts/audit_pr_review_hygiene.py" \
  --repo-root "$WT" --base "$BASE_REF"
# If checkpoint_conversion / param_mapping / load_dynamic / hf_streaming_load touched:
PYTHONPATH="$WT/src" python3 "$MPU/scripts/sweep_param_mapping_contracts.py" \
  --maxtext-root "$WT"
# Behavioral cross-model diff (Tier A), tpu7x-8 AOT HLO diff (Tier B), caller-impact tests (Tier C):
# Run with a MaxText virtualenv Python that has `jax` and `pytest` installed:
VENV_PY="${MAXTEXT_PILOT_PYTHON:-$(command -v python3)}"
"$VENV_PY" "$MPU/scripts/pilot.py" \
  --repo-root "$WT" --python "$VENV_PY" \
  --label "pr_check" --head "$LIVE_HEAD" --base "$BASE_REF"
```

## 2. Pass-0 Local MaxText Reviewer and Scope Gate (Enforced)

Review **one PR per report** on its **full diff against its parent** (never a
range-diff against a prior revision; pure-rebase files still require reading):

1. Build the local MaxText criteria prompt with
   `.agents/skills/maxtext-pr-upload/scripts/build_local_review_prompt.py`:

   ```bash
   python3 "$MPU/scripts/build_local_review_prompt.py" \
     --maxtext-root "$WT" --base "$BASE_REF" --head "$LIVE_HEAD" \
     --target "PR#${PR_NUM}" --description-file /tmp/pr_body.md \
     --findings-out /tmp/local_review.md --out /tmp/review_prompt.md
   ```

   Run a fresh read-only reviewer on the full diff. By default,
   `build_local_review_prompt.py` discovers subsystem-filtered `must_catch` and
   `do_not_flag` lessons from `--kb-ledger`, `$MAXTEXT_REVIEW_KB_LEDGER`,
   `~/.config/maxtext/learned_lessons.jsonl`, and
   `.agents/skills/maxtext-pr-upload/resources/learned_lessons.jsonl` (validated
   against
   `.agents/skills/maxtext-pr-upload/resources/learned_lessons_schema.json`;
   disable with `--no-kb-ledger`). Resolve or dismiss (with an explicit
   evidence-backed `reason="..."`) every `RED` and `ORANGE` finding.
2. In Pass 2, verify every suspected defect against exact `file:line` code or a
   minimal runnable reproduction; drop unverifiable guesses and check repo-wide
   callers when shared configs or signatures change. Specifically verify:
   - **Shared-harness full-function & all-CLI/modality execution**: when a
     shared parity script or runner changes, read the entire enclosing function
     and exercise both `--run_hf_model=true` and `--run_hf_model=false` plus
     every supported modality flag (e.g., `use_audio=True`,
     `use_multimodal=True`), not just the default branch.
   - **Skewed / hot-shard collective capacity probes**: for ragged, MoE, or
     `all_to_all` dispatch collectives, test skewed routing where a single
     shard's token `count > cap` rather than uniform random splits only.
   - **Benchmark script XLA flags & static-vs-runtime HBM**: verify XLA flags
     on benchmark scripts do not disable native kernels without an explicit
     `# intentional-baseline:` comment, and cross-check static AOT HBM
     estimates against runtime memory behavior.
3. End the final report with a 3-backtick `review-scope` block (`target`,
   `base`, `head`, `verdict`, `files:` with single-line `- path: note` bullets
   for every changed file) and enforce with `--strict-pre-push` (requires
   `verdict: APPROVED` only — rejects `APPROVED_WITH_FOLLOWUPS`, unexplained
   dismissed `YELLOW`s, and open defect `YELLOW`s while allowing `[nit]` style
   notes):

   ```bash
   python3 "$MPU/scripts/review_scope_gate.py" \
     --report /tmp/gate_c_report.md --repo-root "$WT" \
     --base "$BASE_REF" --head "$LIVE_HEAD" \
     --findings-file /tmp/local_review.md --strict-pre-push
   ```

   A non-zero exit means the PR was not completely reviewed or still carries
   open blocking/downgraded findings; fix and rerun before pushing or
   approving.

### 2b. Opt-In TPU Hardware Verification & High-Risk Escalation

CPU numerical parity and `tpu7x-8` AOT `train_compile` catch lowering, sharding,
and static HBM regressions, not runtime TPU MXU, Pallas, or SparseCore
numerical edge cases:

- **TPU runtime verification (Pass 2 on kernel/precision/sharding PRs)**: run
  the affected `@pytest.mark.tpu_only` unit and integration suites on TPU
  hardware (for example 4-chip / 8-core TPU v7x, v6e, or v5p runners) and
  include verbatim test output in `# Tests`.
- **Large or high-risk changes**: run the full deterministic pre-pass,
  `pilot.py` Tiers A-C, and `review_scope_gate.py --strict-pre-push` on every
  revision before requesting human review.

## 3. Post-Upload 3-Gate Hold (Before Requesting Human Code-Owner Review)

Wait until all three gates pass on the exact live `headRefOid` before pinging
human code owners:

- **Gate A**: All automated review comments (`github-actions[bot]`,
  `gemini-code-assist[bot]`) evaluated, real defects fixed test-first, and
  threads answered
  (`.agents/skills/maxtext-pr-upload/scripts/pr_comments.py <pr> --full` checks
  both inline review threads and substantive top-level `issues/<N>/comments`
  flagged `[TOP-LEVEL]`; both must have `0` unanswered items).
- **Gate B**: Required GitHub CI checks green
  (`.agents/skills/maxtext-pr-upload/scripts/ci_status.py <pr>`: Pre-commit
  Linters, Documentation Build Check, Development Tests, CPU Pretrain/Posttrain
  Tests, Require PR Checklist; ignores non-blocking
  `Track Test Performance / Track Test Duration`).
- **Gate C**: Independent context-isolated review returns `VERDICT: APPROVED`
  with zero `[P0]`/`[P1][BLOCKING]` issues and passes
  `.agents/skills/maxtext-pr-upload/scripts/review_scope_gate.py` on the live
  `headRefOid`.

## 4. Two-Incident Bar for Adding New Static AST Rules

When triaging post-review feedback or post-mortems, **do not add a new static
AST rule** unless:

1. The same defect class has recurred across **`>= 2` independent PRs**, AND
2. It is invisible to pinned linters, Pass-0 local review (with
   `resources/learned_lessons.jsonl`), and dynamic behavioral Tiers A-C.

Record single-incident misses in `resources/learned_lessons.jsonl` (if general
to a subsystem) or in post-mortem notes so static AST scanners remain lean and
false-positive-free.
