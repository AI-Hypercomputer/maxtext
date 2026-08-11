set -e

# Clear any ongoing rebase or local changes
git rebase --abort 2>/dev/null || true
git reset --hard HEAD -q

echo "Restacking PR 1..."
git checkout -B feat/mock-tensor-validation origin/feat/mock-tensor-validation -q
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 2..."
git checkout -B ckpt-validation-pr2-forward-pass feat/mock-tensor-validation -q
git cherry-pick origin/ckpt-validation-pr2-forward-pass
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 3..."
git checkout -B ckpt-validation-pr3-layer-metrics ckpt-validation-pr2-forward-pass -q
git cherry-pick origin/ckpt-validation-pr3-layer-metrics
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 4..."
git checkout -B ckpt-validation-pr4-decoding ckpt-validation-pr3-layer-metrics -q
git cherry-pick origin/ckpt-validation-pr4-decoding
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 5..."
git checkout -B ckpt-validation-pr5-agent-sidecar ckpt-validation-pr4-decoding -q
git cherry-pick origin/ckpt-validation-pr5-agent-sidecar
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Restacking PR 6..."
git checkout -B ckpt-validation-pr6-airflow-integration ckpt-validation-pr5-agent-sidecar -q
git cherry-pick origin/ckpt-validation-pr6-airflow-integration
python3 patch_final_7.py
if ! git diff --quiet; then
  git add src/
  git commit --amend --no-edit -q
fi

echo "Local cascade perfectly completed!"
