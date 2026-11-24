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

"""One-off operations of benchmark v3: code bundles, weight downloads, checks and the cache proof.

  wl_bench.py bundle --build baseline|candidate   # build the bundle of the checked-out branch, upload and verify it
  wl_bench.py step1 --platform v7x [--models m ...] [--dry-run]       # cold HF -> GCS downloads (CPU Jobs)
  wl_bench.py step1b [--models m ...] [--dry-run]                     # DeepSeek FP8/MXFP4 -> BF16 (v7x CPU Jobs)
  wl_bench.py copy-region --platform v6e [--models m ...] [--dry-run] # BF16 copies into another region
  wl_bench.py scan --platform v7x          # compare every .safetensors header with its object size
  wl_bench.py proof --platform v5p [--dry-run]                        # the three-run cache proof (cache_proof.py)
  wl_bench.py status [--platform v5e]

The weights of a region are downloaded once and shared by the platforms in it: use v7x for us-central1, v6e for
southamerica-west1 and v5p for europe-west4 (which v5e shares).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl import common  # pylint: disable=wrong-import-position
from bench import manifests_v2  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

REPO = os.path.dirname(common.BENCH_DIR)
STEP1B_SOURCE_PLATFORM = "v7x"


def _models(platform: str | None, names: list[str] | None, step1b: bool | None = None) -> list[dict]:
  """Returns the models of the matrix placed on a platform of `platform`'s region, filtered by name and Step 1b."""
  region = platforms.get_profile(platform).gcp_region if platform else None
  out = []
  for model in schema_v2.load_models_v2()["models"]:
    placed = any(platforms.get_profile(p).gcp_region == region for p in model["topologies_by_platform"])
    if region and not placed:
      continue
    if names and model["model_name"] not in names:
      continue
    if step1b is not None and bool(model.get("needs_step1b")) != step1b:
      continue
    out.append(model)
  return out


def _sha256_of_gcs(uri: str) -> str:
  with subprocess.Popen(["gcloud", "storage", "cat", uri], stdout=subprocess.PIPE) as proc:
    digest = hashlib.sha256()
    for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
      digest.update(chunk)
  if proc.returncode:
    sys.exit(f"cannot read {uri}")
  return digest.hexdigest()


def _sha256_of_file(path: str) -> str:
  with open(path, "rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def cmd_bundle(args) -> None:
  """Builds the code bundle from HEAD of the build's branch, uploads it and compares every upload with its source."""
  branch = common.run(["git", "-C", REPO, "branch", "--show-current"]).stdout.strip()
  if branch != common.BUILD_BRANCHES[args.build]:
    sys.exit(f"check out {common.BUILD_BRANCHES[args.build]} to build the {args.build} bundle (on {branch})")
  dest = common.code_dir(args.build)
  out_dir = tempfile.mkdtemp(prefix=f"wl3-bundle-{args.build}-")
  builder = os.path.join(common.BENCH_DIR, "scripts", "build_code_tarball.py")
  res = subprocess.run([sys.executable, builder, "--out", out_dir, "--upload", dest], check=False)
  if res.returncode:
    sys.exit("bundle build or upload failed")
  if common.run(["gcloud", "storage", "ls", f"{dest}/site_packages_overlay.tar.gz"], check=False).returncode:
    common.run(["gcloud", "storage", "cp", common.OVERLAY_SOURCE, f"{dest}/site_packages_overlay.tar.gz"], timeout=1800)
  for name in sorted(os.listdir(out_dir)):
    local, remote = _sha256_of_file(os.path.join(out_dir, name)), _sha256_of_gcs(f"{dest}/{name}")
    print(f"{'same' if local == remote else 'DIFFERENT'} {dest}/{name} {remote[:16]}")
    if local != remote:
      sys.exit("an uploaded file differs from its local source")
  overlay = (_sha256_of_gcs(common.OVERLAY_SOURCE), _sha256_of_gcs(f"{dest}/site_packages_overlay.tar.gz"))
  print(f"{'same' if overlay[0] == overlay[1] else 'DIFFERENT'} {dest}/site_packages_overlay.tar.gz {overlay[1][:16]}")
  if overlay[0] != overlay[1]:
    sys.exit("the overlay differs from its source")
  with open(os.path.join(out_dir, "CODE_VERSION.json"), "r", encoding="utf-8") as f:
    print(f"{args.build}: {json.load(f)['git_commit']} -> {dest}")


def cmd_step1(args) -> None:
  """Creates the cold HF -> GCS download Job of every model placed in the platform's region."""
  source_only = args.platform != STEP1B_SOURCE_PLATFORM
  for model in _models(args.platform, args.models):
    if model.get("needs_step1b") and source_only:
      continue  # The quantized files are only needed where Step 1b runs; other regions get the BF16 copy.
    doc = manifests_v2.build_step1_cpu_job(
        platform=args.platform,
        model_name=model["model_name"],
        run_id=f"wl3-s1-{args.platform}",
        clean_target=True,
        code_dir=common.code_dir("candidate"),
    )
    if common.apply_manifest(args.platform, doc, args.dry_run) and not args.dry_run:
      print(f"created job/{doc['metadata']['name']} on {args.platform}")


def cmd_step1b(args) -> None:
  """Creates the scratch PVC and the FP8/MXFP4 -> BF16 conversion Job of every DeepSeek model."""
  platform = STEP1B_SOURCE_PLATFORM
  for model in _models(platform, args.models, step1b=True):
    docs = manifests_v2.build_deepseek_step1b_manifests(
        model_name=model["model_name"], run_id="wl3-s1b", clean_target=True
    )
    # The builder names the scratch PVC after the model only, which is also its name in the v2 benchmark.
    pvc, job = docs
    pvc["metadata"]["name"] = f"mesa-wl3-step1b-{manifests_v2.dns_slug(model['model_name'])}-pvc"
    claims = [v for v in job["spec"]["template"]["spec"]["volumes"] if "persistentVolumeClaim" in v]
    if len(claims) != 1:
      sys.exit("expected one PVC volume in the Step 1b job")
    claims[0]["persistentVolumeClaim"]["claimName"] = pvc["metadata"]["name"]
    for doc in docs:
      if not common.apply_manifest(platform, doc, args.dry_run):
        break
      if not args.dry_run:
        print(f"created {doc['kind'].lower()}/{doc['metadata']['name']} on {platform}")


def cmd_copy_region(args) -> None:
  """Creates the Jobs that copy the converted BF16 weights into the platform's region."""
  if args.platform == STEP1B_SOURCE_PLATFORM:
    sys.exit(f"{STEP1B_SOURCE_PLATFORM} holds the converted files; copy to another platform's region")
  source_root = common.regional_root(STEP1B_SOURCE_PLATFORM)
  for model in _models(args.platform, args.models, step1b=True):
    doc = manifests_v2.build_copy_region_cpu_job(
        dest_platform=args.platform,
        model_name=model["model_name"],
        source_gcs_dir=f"{source_root}/hf_cache_bf16/{model['model_name']}",
        run_id=f"wl3-copy-{args.platform}",
        clean_target=True,
        code_dir=common.code_dir("candidate"),
    )
    if common.apply_manifest(args.platform, doc, args.dry_run) and not args.dry_run:
      print(f"created job/{doc['metadata']['name']} on {args.platform}")


def _read_range(bucket: str, name: str, start: int, end: int, token: str) -> bytes:
  url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media"
  request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}"})
  with urllib.request.urlopen(request, timeout=120) as response:
    return response.read()


def _check_safetensors(uri: str, size: int, token: str) -> str:
  """Returns '' if the tensors that the header of `uri` lists end exactly at the object's size, else the problem."""
  bucket, _, name = uri[len("gs://") :].partition("/")
  try:
    head = _read_range(bucket, name, 0, min(size, 1 << 20) - 1, token)
    (header_len,) = struct.unpack("<Q", head[:8])
    if 8 + header_len > size:
      return f"header of {header_len} bytes is longer than the object ({size})"
    if 8 + header_len > len(head):
      head += _read_range(bucket, name, len(head), 8 + header_len - 1, token)
    header = json.loads(head[8 : 8 + header_len])
    data_end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    expected = 8 + header_len + data_end
    return "" if expected == size else f"header implies {expected} bytes, object has {size}"
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"cannot check: {type(e).__name__}: {e}"


def cmd_scan(args) -> None:
  """Checks every .safetensors file of the platform's region against the size its header implies."""
  root = common.regional_root(args.platform)
  token = common.run(["gcloud", "auth", "print-access-token"]).stdout.strip()
  files = []
  for sub in ("hf_cache", "hf_cache_bf16"):
    listing = common.run(["gcloud", "storage", "ls", "-l", f"{root}/{sub}/**"], check=False, timeout=1800).stdout
    for line in listing.splitlines():
      parts = line.split()
      if len(parts) == 3 and parts[2].endswith(".safetensors"):
        files.append((parts[2], int(parts[0])))
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    problems = list(pool.map(lambda f: _check_safetensors(f[0], f[1], token), files))
  bad = [(uri, problem) for (uri, _), problem in zip(files, problems) if problem]
  for uri, problem in bad:
    print(f"BAD {uri}: {problem}")
  print(f"{root}: {len(files)} .safetensors files, {sum(s for _, s in files) / 1e12:.2f} TB, {len(bad)} bad")
  if bad:
    sys.exit(1)


def cmd_proof(args) -> None:
  """Creates the single-host JobSet that runs cache_proof.py."""
  profile = platforms.get_profile(args.platform)
  topology = next(t for t, hosts in profile.hosts_per_slice_by_topology.items() if hosts == 1)
  root = common.regional_root(args.platform)
  script = (
      "set -euo pipefail\n"
      "export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TMPDIR=/ramdisk/tmp PYTHONPATH=/tmp/maxtext/src:/tmp/maxtext\n"
      "python3 -u /tmp/maxtext/checkpoint_load_benchmark/wl/cache_proof.py \\\n"
      f"  --platform {args.platform} --topology {topology} \\\n"
      f"  --devices-per-slice {profile.devices_per_slice(topology)} \\\n"
      f"  --weights {root}/hf_cache/qwen3-0.6b \\\n"
      f"  --gcs-base {root}/runs-candidate/proof --run-prefix wl3-cache-proof-{args.attempt}\n"
  )
  doc = manifests_v2.build_tpu_jobset(
      platform=args.platform,
      topology=topology,
      num_slices=1,
      name=f"mesa-wl3-cache-proof-{args.platform}-{args.attempt}",
      runner_script=script,
      run_id=f"wl3-cache-proof-{args.attempt}",
      experiment_name=common.EXPERIMENT,
      max_restarts=0,
      code_dir=common.code_dir("candidate"),
  )
  if common.apply_manifest(args.platform, doc, args.dry_run) and not args.dry_run:
    print(f"created jobset/{doc['metadata']['name']} on {args.platform}")


def cmd_status(args) -> None:
  """Prints our JobSets, Jobs and PVCs per platform."""
  for platform in [args.platform] if args.platform else common.PLATFORMS:
    print(f"===== {platform}")
    print(common.kubectl(platform, "get", "jobsets,jobs,pvc", "-l", f"app={common.APP}", check=False).stdout)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)
  p = sub.add_parser("bundle")
  p.add_argument("--build", required=True, choices=common.BUILDS)
  for name in ("step1", "copy-region"):
    p = sub.add_parser(name)
    p.add_argument("--platform", required=True, choices=common.PLATFORMS)
    p.add_argument("--models", nargs="+")
    p.add_argument("--dry-run", action="store_true")
  p = sub.add_parser("step1b")
  p.add_argument("--models", nargs="+")
  p.add_argument("--dry-run", action="store_true")
  p = sub.add_parser("scan")
  p.add_argument("--platform", required=True, choices=common.PLATFORMS)
  p = sub.add_parser("proof")
  p.add_argument("--platform", required=True, choices=common.PLATFORMS)
  p.add_argument("--attempt", default="a1", help="Suffix of the JobSet name and run ids")
  p.add_argument("--dry-run", action="store_true")
  p = sub.add_parser("status")
  p.add_argument("--platform", choices=common.PLATFORMS)
  args = parser.parse_args()
  common.use_v3_profiles()
  commands = {
      "bundle": cmd_bundle,
      "step1": cmd_step1,
      "step1b": cmd_step1b,
      "copy-region": cmd_copy_region,
      "scan": cmd_scan,
      "proof": cmd_proof,
      "status": cmd_status,
  }
  commands[args.cmd](args)


if __name__ == "__main__":
  main()
