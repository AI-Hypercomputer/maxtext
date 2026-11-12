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

"""CLI generator for v2 CPU Step 1 / Step 1b Jobs, TPU JobSets, and GCS task queues."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
import sys
import yaml

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import manifests_v2  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position


def _write_yaml_docs(docs: Sequence[dict], out_path: str | None) -> None:
  rendered = yaml.safe_dump_all(list(docs), sort_keys=False)
  if out_path:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(rendered)
  else:
    sys.stdout.write(rendered)


def _write_json_doc(doc: dict, out_path: str | None, upload_gcs: str = "") -> None:
  """Serializes a JSON queue document locally and/or directly uploads to `upload_gcs`."""
  encoded = json.dumps(doc, indent=2) + "\n"
  if out_path:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(encoded)
  elif not upload_gcs:
    sys.stdout.write(encoded)
  if upload_gcs:
    from etils import epath  # pylint: disable=import-outside-toplevel

    epath.Path(upload_gcs).write_text(encoded)
    print(f"Uploaded tasks.json ({len(doc.get('tasks', []))} tasks) -> {upload_gcs}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
  """Parses CLI args and generates requested v2 Kubernetes Job/JobSet or GCS queue manifests."""
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  s1 = sub.add_parser("step1-cpu", help="Generate CPU Step 1 cold HF -> GCS Job YAML(s)")
  s1.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  s1.add_argument("--models", nargs="+", required=True)
  s1.add_argument("--run-id", required=True)
  s1.add_argument("--clean-target", action="store_true")
  s1.add_argument("--workers", type=int, default=None)
  s1.add_argument("--out", default=None)

  s1b = sub.add_parser("step1b-cpu", help="Generate DeepSeek Step 1b PVC + CPU Job YAML")
  s1b.add_argument("--model", required=True)
  s1b.add_argument("--run-id", required=True)
  s1b.add_argument("--pvc-size-gib", type=int, default=None)
  s1b.add_argument("--clean-target", action="store_true")
  s1b.add_argument("--cache-size", type=int, default=4)
  s1b.add_argument("--out", default=None)

  cp = sub.add_parser("copy-region", help="Generate cross-region BF16 copy CPU Job YAML")
  cp.add_argument("--dest-platform", required=True, choices=schema_v2.PLATFORMS)
  cp.add_argument("--model", required=True)
  cp.add_argument("--source-gcs-dir", required=True)
  cp.add_argument("--run-id", required=True)
  cp.add_argument("--clean-target", action="store_true")
  cp.add_argument("--bf16-source", default="converted_by_this_benchmark", choices=schema_v2.DEEPSEEK_BF16_SOURCES)
  cp.add_argument("--out", default=None)

  qw = sub.add_parser("queue-worker", help="Generate TPU queue-worker JobSet YAML")
  qw.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  qw.add_argument("--topology", required=True)
  qw.add_argument("--slices", type=int, default=1)
  qw.add_argument("--min-slices", type=int, default=1)
  qw.add_argument("--queue", required=True)
  qw.add_argument("--worker", required=True)
  qw.add_argument("--run-id", required=True)
  qw.add_argument("--results", default="")
  qw.add_argument("--experiment-name", default="mesa-bench-v2")
  qw.add_argument("--out", default=None)

  qt = sub.add_parser("queue-tasks", help="Generate tasks.json for a platform and optionally upload to GCS")
  qt.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  qt.add_argument("--experiment-name", default="mesa-bench-v2")
  qt.add_argument("--run-prefix", required=True)
  qt.add_argument("--tiers", nargs="+", default=["S", "A", "C", "B"])
  qt.add_argument("--out", default=None)
  qt.add_argument("--upload", default="", help="Optional gs://.../tasks.json URI to write directly")

  args = parser.parse_args(argv)
  if args.command == "step1-cpu":
    docs = [
        manifests_v2.build_step1_cpu_job(
            platform=args.platform,
            model_name=m,
            run_id=args.run_id,
            clean_target=args.clean_target,
            workers=args.workers,
        )
        for m in args.models
    ]
    _write_yaml_docs(docs, args.out)
  elif args.command == "step1b-cpu":
    docs = manifests_v2.build_deepseek_step1b_manifests(
        model_name=args.model,
        run_id=args.run_id,
        pvc_size_gib=args.pvc_size_gib,
        clean_target=args.clean_target,
        cache_size=args.cache_size,
    )
    _write_yaml_docs(docs, args.out)
  elif args.command == "copy-region":
    doc = manifests_v2.build_copy_region_cpu_job(
        dest_platform=args.dest_platform,
        model_name=args.model,
        source_gcs_dir=args.source_gcs_dir,
        run_id=args.run_id,
        clean_target=args.clean_target,
        bf16_source=args.bf16_source,
    )
    _write_yaml_docs([doc], args.out)
  elif args.command == "queue-worker":
    profile = platforms.get_profile(args.platform)
    root = (profile.regional_gcs_data_root or profile.gcs_data_root).rstrip("/")
    results_uri = args.results or f"{root}/queue_results"
    doc = manifests_v2.build_queue_worker_jobset(
        platform=args.platform,
        topology=args.topology,
        num_slices=args.slices,
        queue_uri=args.queue,
        worker_id=args.worker,
        run_id=args.run_id,
        results_gcs_uri=results_uri,
        experiment_name=args.experiment_name,
        min_slices=args.min_slices,
    )
    _write_yaml_docs([doc], args.out)
  else:
    tasks = manifests_v2.build_planned_queue_tasks(
        platform=args.platform,
        experiment_name=args.experiment_name,
        run_prefix=args.run_prefix,
        tiers=args.tiers,
    )
    _write_json_doc({"platform": args.platform, "tasks": tasks}, args.out, upload_gcs=args.upload)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
