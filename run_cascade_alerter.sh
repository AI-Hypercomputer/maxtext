set -e

# Clear any ongoing rebase or local changes
git rebase --abort 2>/dev/null || true
git reset --hard origin/ckpt-validation-pr6-airflow-integration -q || true

echo "Restacking PR 5..."
git checkout -B ckpt-validation-pr5-agent-sidecar origin/ckpt-validation-pr4-decoding -q
git cherry-pick origin/ckpt-validation-pr5-agent-sidecar

python3 patch_alerter.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 6..."
git checkout -B ckpt-validation-pr6-airflow-integration ckpt-validation-pr5-agent-sidecar -q
git cherry-pick origin/ckpt-validation-pr6-airflow-integration

echo "Checking if we still need to patch PR 6 just in case..."
python3 patch_alerter.py || true
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "All done!"
