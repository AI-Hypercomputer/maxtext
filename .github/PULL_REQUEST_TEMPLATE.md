# Description

Start with a short description of what the PR does and how this is a change from
the past.

The rest of the description includes relevant details and context, examples:
- why is this change being made,
- the problem being solved and any relevant context,
- why this is a good solution,
- some information about the specific implementation,
- shortcomings of the solution and possible future improvements.

If the change fixes a bug or a Github issue, please include a link, e.g.,:
FIXES: b/123456
FIXES: #123456

You can also provide a comma-separated list. If you don't want to close a bug but
simply to reference it, use BUGS, e.g.: 
BUGS: b/123456

*Notice 1:* Once the PR has 2 approvals, the "pull ready" label will automatically be assigned (on PRs from forks, a maintainer adds it). It runs all tests, and the PR is imported for merging once they pass.
This label is used for administrative purposes. Please do not add it manually.

*Notice 2:* For external contributions, our settings currently require an approval from a MaxText maintainer to trigger CI tests.

*Notice 3:* Tests marked `scheduled_only` are skipped on PRs and only run in the periodic scheduled CI.
If you need them (plus the TPU7X suite) on your PR, add the `scheduled-only` label; the full pipeline
re-runs as a scheduled run. Remove and re-add the label to trigger it again.

# Tests

Please describe how you tested this change, and include any instructions and/or
commands to reproduce.

# Checklist

Before submitting this PR, please make sure (put X in square brackets):
- [ ] I have performed a self-review of my code. For an optional AI review, add the `gemini-review` label.
- [ ] I have necessary comments in my code, particularly in hard-to-understand areas.
- [ ] I have run end-to-end tests and provided workload links above if applicable. *(Note: New PRs get `skip-ci-tests` so only CPU/lint/docs checks run during review; a bot removes it after 2 approvals on a 1-commit PR to run TPU/GPU suites, or remove it yourself / ask a maintainer on fork PRs to run them sooner.)*
- [ ] I have made or will make corresponding changes to the doc if needed, including adding new documentation pages to the relevant Table of Contents (toctree directive) as explained in [our documentation](https://maxtext.readthedocs.io/en/latest/development.html#adding-new-documentation-files).
