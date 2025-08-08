# SPDX-License-Identifier: Apache-2.0

# Clean up Python codes using Pylint & Pyink
# Googlers: please run `sudo apt install pipx; pipx install pylint --force; pipx install pyink==23.10.0` in advance

set -e # Exit immediately if any command fails

FOLDERS_TO_FORMAT=("MaxText" "pedagogical_examples")
LINE_LENGTH=$(grep -E "^max-line-length=" pylintrc | cut -d '=' -f 2)

# Check for --check flag
CHECK_ONLY_PYINK_FLAGS=""
if [[ "$1" == "--check" ]]; then
  CHECK_ONLY_PYINK_FLAGS="--check --diff --color"
fi

for folder in "${FOLDERS_TO_FORMAT[@]}"
do
  pyink "$folder" ${CHECK_ONLY_PYINK_FLAGS} --pyink-indentation=2 --line-length=${LINE_LENGTH}
done

for folder in "${FOLDERS_TO_FORMAT[@]}"
do
  # pylint doesn't change files, only reports errors.
  pylint --disable R0401,R0917,W0201,W0613 "./$folder"
done

echo "Successfully clean up all codes."
