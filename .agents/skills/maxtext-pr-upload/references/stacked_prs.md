# Stacked MaxText PRs: Open, Merge, Fix

Stack a PR on another only when it depends on that PR's code. Each PR must
still contain **1 commit** relative to its base branch, so GitHub displays only
that PR's own commit.

When a parent PR merges into `main` as a merge commit preserving its head SHA
in `main`'s history, retargeting the child to `main` leaves 1 commit without a
rebase (if the parent was amended or squash-merged as a new SHA, rebase
`--onto origin/main <old_parent_tip> <child>` before retargeting).

## Open

Push bottom-up, one branch at a time, spaced `>= 45s` apart
([SKILL.md](../SKILL.md) Section 4). Open the PRs bottom-up: the root PR with
base `main`, and each child with base `<branch-prefix>/<parent-slug>`. Note
`"Depends on #<parent>; review the top commit only"` in the child PR body
(and remove/update that note once `<parent>` merges). CI runs on every PR
because `ci_pipeline.yml` has no base-branch filter.

## Merge (repeat per level, bottom-up)

1. Merge the bottom PR once CI is green, `pull ready` is applied, and
   code-owner approvals are complete.
2. Verify the parent commit reached `main`:
   `git fetch -q origin && git merge-base --is-ancestor <parent_head_sha> origin/main && echo merged`.
3. Retarget the child's base to `main` (`gh pr edit <child> --base main` or the
   GitHub UI) and confirm the PR shows `commits=1` via
   `.agents/skills/maxtext-pr-upload/scripts/ci_status.py <child>`.
4. Toggle a non-merge label (for example remove/re-add `gemini-review`) or push
   an updated commit timestamp if a fresh `ci_pipeline.yml` run is needed (a
   base change emits `edited`, which does not trigger `ci_pipeline.yml`).
5. Optional cleanup, **only after step 3**: delete the merged parent branch
   (`git push origin --delete <branch-prefix>/<parent-slug>`).
6. Repeat from step 1 with the child as the new root PR.

Never:

- retarget a child to `main` before its parent merges (`commits=2` blocks
  `AddPullReady`);
- close a parent PR in a native GitHub stack without unstacking/retargeting its
  children first (stops `pull_request` workflows on stacked children);
- delete a branch that is still an open PR's base (GitHub auto-closes that PR);
- push a parent and its child in a single `git push` (triggers duplicate
  `synchronize` runs whose cancelled checks block readiness).

## Review fix in a lower PR

Amend that PR's single commit, push it with `--force-with-lease` pinned to the
previous remote SHA, then rebase and push each child bottom-up:

```bash
OLD=<old_parent_head>
NEW=<new_parent_head>
git tag -f "bak/<child>-$(git rev-parse --short <child>)" <child>
git rebase --onto "$NEW" "$OLD" <child>
git push --force-with-lease=<child>:<old_child_remote_sha> origin <child>
sleep 45
```

`git rebase --update-refs --onto "$NEW" "$OLD" <top>` updates intermediate
branches in one pass, which is why backups must be stored as git **tags**
rather than branches.

## Close a redundant middle PR

When `PR_k` (base `PR_{k-1}`) is superseded:

1. `git rebase --onto <PR_{k-1}> <old_PR_k_tip> <PR_{k+1}>`, rebase the rest of
   the stack, and push bottom-up as above.
2. Change `PR_{k+1}`'s base to `<branch-prefix>/<PR_{k-1}-slug>`, then trigger
   CI.
3. Only after `PR_{k+1}` is retargeted, close `PR_k` (or reset `PR_k` to its
   base lease-pinned with explicit confirmation).
