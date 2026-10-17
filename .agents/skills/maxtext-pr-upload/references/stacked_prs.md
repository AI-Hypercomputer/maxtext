# Managing Stacked PRs in MaxText

Because `AddPullReady` requires **1 commit per PR**, stacked PRs (`main <- PR1 <- PR2`)
must be rebased and retargeted in strict bottom-up order.

## 1. Creating a Stacked PR

```bash
# PR1 targets origin/main (1 commit)
git checkout -b <user>/pr1-base origin/main
git commit -m "..."

# PR2 stacks on PR1 (base branch in GitHub = <user>/pr1-base, 1 commit above PR1)
git checkout -b <user>/pr2-child <user>/pr1-base
git commit -m "..."
git rev-list --count <user>/pr1-base..<user>/pr2-child   # must print 1
```

## 2. Amending a Parent PR After Review

When `PR1` is amended, rebase `PR2` onto the new `PR1` tip using `--onto` and
push with a ~45s delay between branches to avoid double-triggering child CI:

```bash
git fetch origin
git tag -f bak/pr1-$(git rev-parse --short <user>/pr1-base) <user>/pr1-base
git tag -f bak/pr2-$(git rev-parse --short <user>/pr2-child) <user>/pr2-child

OLD_PR1=$(git rev-parse <user>/pr1-base)
git checkout <user>/pr1-base
# ... stage review fixes ...
git commit --amend --no-edit

git rebase --onto <user>/pr1-base $OLD_PR1 <user>/pr2-child
git rev-list --count <user>/pr1-base..<user>/pr2-child   # must print 1

git push --force-with-lease=<user>/pr1-base:$OLD_PR1 origin <user>/pr1-base
sleep 45
git push --force-with-lease=<user>/pr2-child:$(git rev-parse bak/pr2-*) origin <user>/pr2-child
```

## 3. Bottom-Up Merge & Retarget Sequence

When `PR1` merges into `main`:
1. Fetch `origin/main` and rebase `PR2` directly onto `origin/main`, dropping the
   pre-squash `PR1` commit:
   ```bash
   git fetch origin
   git rebase --onto origin/main $OLD_PR1 <user>/pr2-child
   git rev-list --count origin/main..<user>/pr2-child    # must print 1
   git push --force-with-lease origin <user>/pr2-child
   ```
2. Change `PR2`'s base branch in GitHub from `<user>/pr1-base` to `main`.
   *(Never change `PR2`'s base to `main` before `PR1` merges, or `PR2` will show
   2 commits and fail `AddPullReady`.)*
