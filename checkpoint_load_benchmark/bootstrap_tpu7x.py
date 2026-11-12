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

"""Pod bootstrap and coordination script for TPU benchmark pods.

Responsibilities:
  1. Environment Setup (`setup_environment`):
     - Installs missing Python dependencies from `<BENCH_CODE_DIR>/site_packages_overlay.tar.gz` without shadowing the
       container's native TPU-matched `jax`, `jaxlib`, `flax`, or `orbax` packages.
     - Downloads and extracts `<BENCH_CODE_DIR>/maxtext_lance_prs.tar.gz` into `/tmp/maxtext`. Every pod runs the
       committed code tree inside that tarball verbatim (including `checkpoint_load_benchmark/benchmark_ckpt_load_time.py`
       and `src/`), verified via `/tmp/maxtext/CODE_VERSION.json` (`BENCH_GIT_COMMIT` exported to both runner and daemon).
     - Installs `google-cloud-storage` if missing in the container environment.
  2. Multi-Slice / Multi-Host Role Dispatch (`main`):
     - `ROLE=daemon`: Starts `/tmp/maxtext/checkpoint_load_benchmark/benchmark_ckpt_load_time.py --remote-slice-daemon`.
     - `ROLE=runner`: Waits for all daemon hosts listed in `WAIT_FOR_DAEMON_HOST` on `DAEMON_PORT` (default `55555`),
       exports `REMOTE_SLICE1_ADDR`, runs `RUNNER_SCRIPT`, and sends shutdown signals to the daemons upon completion.
  `BENCH_CODE_DIR` overrides the GCS code directory (default: `gs://mesa-maxtext/mesakh_tpu7x_bench/code`).
"""

import json
import os
import shutil
import site
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any
from etils import epath

GCS_CODE_DIR = os.environ.get("BENCH_CODE_DIR", "gs://mesa-maxtext/mesakh_tpu7x_bench/code").rstrip("/")
CODE_VERSION_FILENAME = "CODE_VERSION.json"
MAXTEXT_TREE_DIR = "/tmp/maxtext"


def download_from_gcs(gcs_uri: str, local_path: str) -> None:
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  data = epath.Path(gcs_uri).read_bytes()
  with open(local_path, "wb") as f:
    f.write(data)
  print(f"[Bootstrap] Downloaded {gcs_uri} -> {local_path} ({len(data)/1e6:.2f} MB)", flush=True)


def load_code_version(tree_dir: str = MAXTEXT_TREE_DIR) -> dict[str, Any]:
  """Reads and validates CODE_VERSION.json from an extracted code tarball tree."""
  version_path = os.path.join(tree_dir, CODE_VERSION_FILENAME)
  try:
    with open(version_path, "r", encoding="utf-8") as f:
      version = json.load(f)
  except (OSError, ValueError) as ex:
    raise RuntimeError(
        f"Missing or invalid {version_path} ({ex}); the code tarball was not built by build_code_tarball.py"
    ) from ex
  if not isinstance(version, dict) or not isinstance(version.get("git_commit"), str) or not version["git_commit"]:
    raise RuntimeError(
        f"Invalid {version_path}: expected a JSON object with a non-empty 'git_commit' "
        f"(tarball was not built by build_code_tarball.py)"
    )
  return version


def extract_and_verify_code_tarball(tarball_path: str, tree_dir: str = MAXTEXT_TREE_DIR) -> dict[str, Any]:
  """Extracts `tarball_path` into `tree_dir`, validates CODE_VERSION.json, and returns its metadata."""
  os.makedirs(tree_dir, exist_ok=True)
  with tarfile.open(tarball_path, "r:gz") as tf:
    tf.extractall(tree_dir, filter="tar")
  return load_code_version(tree_dir)


def setup_environment() -> None:
  """Downloads the site-packages overlay and code tarball from GCS and sets BENCH_GIT_COMMIT."""
  t0 = time.time()
  os.makedirs("/tmp/site_packages_raw", exist_ok=True)
  if os.path.exists(MAXTEXT_TREE_DIR):
    shutil.rmtree(MAXTEXT_TREE_DIR)
  os.makedirs(MAXTEXT_TREE_DIR, exist_ok=True)

  download_from_gcs(f"{GCS_CODE_DIR}/site_packages_overlay.tar.gz", "/tmp/site_packages_overlay.tar.gz")
  with tarfile.open("/tmp/site_packages_overlay.tar.gz", "r:gz") as tf:
    tf.extractall("/tmp/site_packages_raw", filter="tar")

  # Only install overlay packages that do not already exist in the container's native site-packages
  # so we never shadow the container's matched jax/jaxlib/flax/orbax installation.
  site_dirs = site.getsitepackages()
  target_sp = site_dirs[0] if site_dirs else "/usr/local/lib/python3.12/site-packages"
  copied_pkgs = []
  for item in os.listdir("/tmp/site_packages_raw"):
    src = os.path.join("/tmp/site_packages_raw", item)
    dst = os.path.join(target_sp, item)
    if not os.path.exists(dst):
      if os.path.isdir(src):
        shutil.copytree(src, dst)
      else:
        shutil.copy2(src, dst)
      if not item.endswith(".dist-info"):
        copied_pkgs.append(item)
  print(f"[Bootstrap] Added missing packages to {target_sp}: {sorted(copied_pkgs)}", flush=True)

  download_from_gcs(f"{GCS_CODE_DIR}/maxtext_lance_prs.tar.gz", "/tmp/maxtext_lance_prs.tar.gz")
  code_version = extract_and_verify_code_tarball("/tmp/maxtext_lance_prs.tar.gz", MAXTEXT_TREE_DIR)
  os.environ["BENCH_GIT_COMMIT"] = code_version["git_commit"]
  print(
      f"[Bootstrap] Code version: branch={code_version.get('git_branch_name', 'unknown')} "
      f"commit={code_version['git_commit']} pushed={code_version.get('git_commit_pushed')}",
      flush=True,
  )

  shutil.rmtree("/tmp/site_packages_raw", ignore_errors=True)
  for tar_f in ("/tmp/site_packages_overlay.tar.gz", "/tmp/maxtext_lance_prs.tar.gz"):
    try:
      os.remove(tar_f)
    except OSError:
      pass

  try:
    from google.cloud import storage  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
  except ImportError:
    print("[Bootstrap] Installing google-cloud-storage...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "google-cloud-storage"],
        check=True,
    )

  print(f"[Bootstrap] Environment ready in {time.time() - t0:.2f}s", flush=True)


def wait_for_daemon(host: str, port: int = 55555, timeout_s: int = 7200) -> str:
  """Blocks until a remote slice daemon accepts TCP connections on `host:port`."""
  print(f"[Bootstrap] Waiting for remote slice daemon at {host}:{port} ...", flush=True)
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    try:
      ip = socket.gethostbyname(host)
      with socket.create_connection((ip, port), timeout=5):
        print(f"[Bootstrap] Connected to remote slice daemon at {host} ({ip}:{port})!", flush=True)
        return f"{ip}:{port}"
    except Exception:  # pylint: disable=broad-exception-caught
      time.sleep(2)
  raise TimeoutError(f"Timed out waiting for remote slice daemon {host}:{port}")


def shutdown_daemons(addrs: list[str]) -> None:
  """Sends shutdown signals to all remote slice daemons."""
  for addr in addrs:
    try:
      host, port_s = addr.split(":", 1)
      with socket.create_connection((host, int(port_s)), timeout=10) as sock:
        sock.sendall((json.dumps({"shutdown": True}) + "\n").encode("utf-8"))
      print(f"[Bootstrap] Sent shutdown signal to remote slice daemon at {addr}", flush=True)
    except Exception as ex:  # pylint: disable=broad-exception-caught
      print(f"[Bootstrap] Warning: could not send shutdown to {addr}: {ex}", flush=True)


def main() -> None:
  setup_environment()
  os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

  role = os.environ.get("ROLE", "runner")
  if role == "daemon":
    port = int(os.environ.get("DAEMON_PORT", "55555"))
    cmd = [
        sys.executable,
        "-u",
        f"{MAXTEXT_TREE_DIR}/checkpoint_load_benchmark/benchmark_ckpt_load_time.py",
        "--remote-slice-daemon",
        "--daemon-port",
        str(port),
    ]
    os.execvpe(sys.executable, cmd, os.environ)

  addrs: list[str] = []
  daemon_host = os.environ.get("WAIT_FOR_DAEMON_HOST", "").strip()
  if daemon_host:
    daemon_port = int(os.environ.get("DAEMON_PORT", "55555"))
    for dh in [h.strip() for h in daemon_host.split(",") if h.strip()]:
      addrs.append(wait_for_daemon(dh, daemon_port))
    os.environ["REMOTE_SLICE1_ADDR"] = ",".join(addrs)

  runner_script = os.environ.get("RUNNER_SCRIPT", "").strip()
  try:
    if runner_script:
      subprocess.run(["bash", "-c", runner_script], check=True, env=os.environ)
    else:
      print("[Bootstrap] No RUNNER_SCRIPT provided; exiting.", flush=True)
  finally:
    if addrs:
      shutdown_daemons(addrs)


if __name__ == "__main__":
  main()
