# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Names, paths and helpers shared by the benchmark v3 automation.

Benchmark v3 runs the v2 harness of this directory on two code builds. Everything it creates is kept apart from the
v2 benchmark: Kubernetes objects carry the label app=wl-bench-v3, and GCS objects live under
gs://mesa-maxtext-<region>/wl-bench-v3 and gs://mesa-maxtext/wl-bench-v3. `use_v3_profiles` points the platform
profiles of `bench.platforms` at these names, so the manifest builders of `bench.manifests_v2` produce v3 objects;
every script of this package calls it first.
"""

import dataclasses
import json
import os
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH_DIR not in sys.path:
  sys.path.insert(0, BENCH_DIR)

from bench import platforms  # pylint: disable=wrong-import-position

APP = "wl-bench-v3"
V2_APP = "mesa-bench-v2"
EXPERIMENT = "wl-bench-v3"
BUILDS = ("baseline", "candidate")
BUILD_BRANCHES = {"baseline": "bench/checkpoint-load-v3-baseline", "candidate": "bench/checkpoint-load-v3"}
PLATFORMS = ("v7x", "v6e", "v5p", "v5e")
GLOBAL_ROOT = "gs://mesa-maxtext/wl-bench-v3"
REPORT_DIR = os.path.join(BENCH_DIR, "reports", "v3")
# The site-packages overlay that the pod bootstrap reads from every code directory. It is not built from this branch.
OVERLAY_SOURCE = "gs://mesa-maxtext/mesa-bench-v2/code/site_packages_overlay.tar.gz"

_V2_PROFILES = dict(platforms.PROFILES)


def use_v3_profiles() -> None:
  """Points every platform profile at the v3 label, code directory and data roots, for this process."""
  for name, profile in _V2_PROFILES.items():
    platforms.PROFILES[name] = dataclasses.replace(
        profile,
        gcs_code_dir=f"{GLOBAL_ROOT}/code-candidate",
        gcs_data_root=GLOBAL_ROOT,
        regional_gcs_data_root=profile.regional_gcs_data_root.replace(f"/{V2_APP}", f"/{APP}"),
        pod_labels={"app": APP},
    )


def restore_v2_profiles() -> None:
  """Undoes `use_v3_profiles`, for tests that share a process with the v2 harness tests."""
  platforms.PROFILES.update(_V2_PROFILES)


def code_dir(build: str) -> str:
  """Returns the GCS directory that holds the code bundle of `build`."""
  if build not in BUILDS:
    raise ValueError(f"unknown build {build!r}")
  return f"{GLOBAL_ROOT}/code-{build}"


def regional_root(platform: str) -> str:
  """Returns the regional GCS root of `platform`, which is always under wl-bench-v3."""
  root = platforms.get_profile(platform).regional_gcs_data_root.rstrip("/")
  if not root.endswith(f"/{APP}"):
    raise ValueError(f"regional root of {platform} is not a {APP} path: {root}")
  return root


def kube_context(platform: str) -> str:
  return platforms.get_profile(platform).kube_context


def utc_now() -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(cmd: list[str], *, input_text: str | None = None, check: bool = True, timeout: int = 600):
  """Runs `cmd` and returns the completed process; exits with its output if it fails and `check` is set."""
  try:
    res = subprocess.run(cmd, input=input_text, text=True, capture_output=True, check=False, timeout=timeout)
  except subprocess.TimeoutExpired:
    res = subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout} s")
  if check and res.returncode:
    sys.exit(f"FAILED ({res.returncode}): {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
  return res


def kubectl(platform: str, *args: str, input_text: str | None = None, check: bool = True):
  return run(["kubectl", f"--context={kube_context(platform)}", *args], input_text=input_text, check=check)


def gcs_cat(uri: str) -> str | None:
  """Returns the text of the GCS object `uri`, or None if it cannot be read."""
  res = run(["gcloud", "storage", "cat", uri], check=False)
  return res.stdout if res.returncode == 0 else None


def gcs_cat_json(uri: str):
  text = gcs_cat(uri)
  try:
    return json.loads(text) if text else None
  except ValueError:
    return None


def gcs_ls(uri: str) -> list[str]:
  """Returns the object and directory URIs directly under `uri`; an empty list if there are none."""
  res = run(["gcloud", "storage", "ls", uri.rstrip("/") + "/"], check=False)
  return [line for line in res.stdout.split() if line.startswith("gs://")] if res.returncode == 0 else []


def gcs_write_verified(uri: str, text: str) -> None:
  """Writes `text` to `uri` and reads it back: what pods fetch must be what was written."""
  if f"/{APP}/" not in uri:
    raise ValueError(f"refusing to write outside {APP}: {uri}")
  run(["gcloud", "storage", "cp", "-", uri], input_text=text)
  if gcs_cat(uri) != text:
    sys.exit(f"{uri} differs from what was uploaded")


def check_isolated(doc: dict) -> None:
  """Refuses a manifest that is not labelled as ours or that names a v2 path."""
  meta = doc["metadata"]
  if meta.get("labels", {}).get("app") != APP:
    raise ValueError(f"{meta.get('name')} is not labelled app={APP}")
  if f"/{V2_APP}" in json.dumps(doc):
    raise ValueError(f"{meta.get('name')} references a {V2_APP} path")


def apply_manifest(platform: str, doc: dict, dry_run: bool = False) -> bool:
  """Creates the object of `doc` on `platform`. Returns False, without touching it, if the name already exists."""
  check_isolated(doc)
  kind, name = doc["kind"].lower(), doc["metadata"]["name"]
  if dry_run:
    print(json.dumps(doc, indent=1))
    return True
  if kubectl(platform, "get", kind, name, "-o", "name", check=False).returncode == 0:
    print(f"{kind}/{name} already exists on {platform}; not touching it", file=sys.stderr)
    return False
  kubectl(platform, "create", "-f", "-", input_text=json.dumps(doc))
  return True
