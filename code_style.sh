# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
  pylint --disable C0301,C3001,C0114,C0115,C0116,C0200,C0121,C0201,C0206,C0209,C0412,C0415,C2801,E0102,E0606,E1102,E1111,E1123,E1135,E1136,R0401,R1701,R1703,R1710,R1711,R1735,R0917,R1714,R1716,R1719,R1721,R1728,R1728,W0102,W0107,W0201,W0212,W0221,W0237,W0404,W0611,W0612,W0613,W0621,W0622,W0631,W0707,W0718,W1201,W1203,W1309,W1514,W4901 "./$folder"
done

echo "Successfully clean up all codes."
