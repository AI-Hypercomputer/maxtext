# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates src/dependencies/extra_deps/tf_requirements.txt using uv pip compile.

Resolves the optional top-level TensorFlow, SeqIO, and JetStream packages against
the existing TPU pre-training lock file (tpu-requirements.txt) as a strict
constraint file (`uv pip compile -c ...`), ensuring no version conflicts with
core MaxText dependencies. Outputs only the delta (packages not already present
in tpu-requirements.txt) with exact version pins so they can be installed
reproducibly with `--no-deps`.

Usage:
  python3 src/dependencies/scripts/generate_tf_requirements.py
  python3 src/dependencies/scripts/generate_tf_requirements.py --tensorflow-version 2.20.0
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CONSTRAINTS = (
    REPO_ROOT / "src" / "dependencies" / "requirements" / "generated_requirements" / "tpu-requirements.txt"
)
DEFAULT_OUTPUT = REPO_ROOT / "src" / "dependencies" / "extra_deps" / "tf_requirements.txt"

_PKG_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize_name(line: str) -> str | None:
  stripped = line.strip()
  if not stripped or stripped.startswith("#"):
    return None
  match = _PKG_NAME.match(stripped)
  return match.group(1).lower().replace("_", "-") if match else None


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--tensorflow-version",
      default="2.20.0",
      help="TensorFlow version to lock (default: 2.20.0)",
  )
  parser.add_argument(
      "--tensorflow-text-version",
      default="2.20.1",
      help="TensorFlow Text version to lock (default: 2.20.1)",
  )
  parser.add_argument(
      "--jetstream-commit",
      default="29329e8e73820993f77cfc8efe34eb2a73f5de98",
      help="JetStream GitHub commit hash (default: 29329e8e73820993f77cfc8efe34eb2a73f5de98)",
  )
  parser.add_argument(
      "--constraints",
      type=pathlib.Path,
      default=DEFAULT_CONSTRAINTS,
      help="Base requirements lock file used as constraints (default: generated_requirements/tpu-requirements.txt)",
  )
  parser.add_argument(
      "--output",
      type=pathlib.Path,
      default=DEFAULT_OUTPUT,
      help="Output file to write (default: src/dependencies/extra_deps/tf_requirements.txt)",
  )
  args = parser.parse_args()

  if not args.constraints.is_file():
    sys.exit(f"Constraints file not found: {args.constraints}")

  tpu_lines = [
      l.strip() for l in args.constraints.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")
  ]
  tpu_pkg_names = {_normalize_name(l) for l in tpu_lines if _normalize_name(l)}

  base_tf = (
      f"tensorflow=={args.tensorflow_version}\n"
      f"tensorflow-datasets\n"
      f"tensorflow-text=={args.tensorflow_text_version}\n"
      f"seqio\n"
      f"google-jetstream @ https://github.com/AI-Hypercomputer/JetStream/archive/{args.jetstream_commit}.zip\n"
  )

  with (
      tempfile.NamedTemporaryFile("w", suffix=".in", encoding="utf-8") as tf_in,
      tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as constraints_file,
  ):
    tf_in.write(base_tf)
    tf_in.flush()
    # Convert >= to == in tpu-requirements.txt so uv pip compile enforces exact lock constraints
    constraints_file.write("\n".join(l.replace(">=", "==") for l in tpu_lines) + "\n")
    constraints_file.flush()

    cmd = [
        sys.executable,
        "-m",
        "uv",
        "pip",
        "compile",
        "--no-header",
        "--no-annotate",
        "-c",
        constraints_file.name,
        tf_in.name,
    ]
    print(f"Resolving optional TensorFlow dependencies against {args.constraints.name}...")
    res = subprocess.run(cmd, text=True, capture_output=True, check=True)

  delta_lines = []
  for line in res.stdout.splitlines():
    pkg = _normalize_name(line)
    if pkg and pkg not in tpu_pkg_names:
      delta_lines.append(line.strip())

  args.output.write_text("\n".join(delta_lines) + "\n", encoding="utf-8")
  print(f"Wrote {len(delta_lines)} pinned packages to {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
