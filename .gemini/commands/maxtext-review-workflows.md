# MaxText Review Focus Area: GitHub Workflows and CI

<!--
Shared focus area for MaxText code review. Used by the GitHub PR review
(.gemini/commands/maxtext-review.toml) and by an internal Code Review Agent
skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `.github/` (workflows, reusable
workflows, `scripts/`, templates, `CODEOWNERS`).

MaxText is developed in the public GitHub repository `AI-Hypercomputer/maxtext`.
Its workflows run on pull requests, use GitHub App tokens and Google Cloud
credentials, run some jobs on self-hosted TPU/GPU runners, and invoke Gemini on
PR content. A mistake here can leak credentials or let an untrusted PR run code
with write access. Keep a finding only if you can name the trigger (who
controls the input, which event fires the job) and the impact (what is leaked,
executed, or broken).

## What to check

1.  **Untrusted input injection.** Values a PR author or commenter controls
    (`github.event.pull_request.title/body/head.ref`, `github.event.comment.body`,
    `github.event.issue.*`, `github.head_ref`, branch names, file contents of
    the PR) interpolated directly as `${{ ... }}` into `run:` scripts,
    `actions/github-script` code, or prompts passed to Gemini. The safe pattern
    is to pass them through `env:` and quote them in the shell.
2.  **Privileged triggers on untrusted code.** `pull_request_target`,
    `workflow_run`, or `issue_comment` jobs that check out or execute PR head
    code (`ref: ${{ github.event.pull_request.head.sha }}`, `refs/pull/*/head`)
    while secrets, a write-scoped `GITHUB_TOKEN`, an App token, or cloud
    credentials are available. Comment-triggered jobs (e.g., `@gemini-cli`)
    must check the commenter's `author_association` before doing anything
    privileged. PR-triggered jobs that use secrets must either exclude forks,
    or (for `pull_request_target`) check out only the base commit, never build
    or execute PR code, read the PR only through the GitHub API, and gate on
    the PR author's `author_association`.
3.  **Permissions.** Missing top-level `permissions:` (defaults may be broad),
    `write-all`, or write scopes (`contents`, `pull-requests`, `issues`,
    `actions`, `id-token`) granted wider than the job needs. Prefer per-job
    least privilege.
4.  **Secrets and credentials.** Secrets echoed, written to files that are
    uploaded as artifacts or caches, passed to untrusted steps, or exposed to
    jobs that run PR code. GitHub App tokens
    (`actions/create-github-app-token`) and cloud credentials should be scoped
    and short-lived.
5.  **Supply chain.** Third-party actions must be pinned to a full commit SHA
    (with a version comment), as the existing workflows do; flag tags or
    branches (`@v4`, `@main`). Flag `curl | bash`, unpinned `pip install` of
    new packages in privileged jobs, and Docker images pulled by mutable tag
    into jobs with credentials.
6.  **Self-hosted runners.** Jobs on self-hosted TPU/GPU runners must not run
    code from forks without an approval gate, and must not leave credentials or
    workspace state behind for later jobs.
7.  **CI contract with the internal mirror.** The names of the required checks
    (e.g., "All Required Tests Passed", "Code Quality Check / Pre-commit
    Linters") are mirrored in the internal Copybara configuration that reports
    GitHub CI results back to internal code review. Flag renaming or removing
    such a job; the internal configuration must be updated in the same change.
8.  **Gemini workflows.** Changes to the Gemini review/invoke/investigate
    workflows or `.gemini/` must keep the tool allowlist minimal (read-only
    shell commands and a short `allowEnv` list without secrets), keep untrusted
    PR content out of instructions, and never give the model write credentials
    beyond posting comments. Automatic runs on fork PRs
    (`pull_request_target`) must stay gated to trusted authors
    (`CONTRIBUTOR`/`MEMBER`/`COLLABORATOR`/`OWNER`) and must keep checking out
    the base commit, so the prompt and workflow come from the base branch.
