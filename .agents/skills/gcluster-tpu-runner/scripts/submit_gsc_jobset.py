# Copyright 2023–2026 Google LLC
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

"""Generates and submits topology-agnostic Kueue dynamic-slicing TPU JobSets."""

import argparse
import json
import math
import subprocess
import sys
from typing import Any


def nodes_per_slice(topology: str, chips_per_node: int = 4) -> int:
  """Computes the number of GKE nodes in a single slice from a 2D ('AxB') or 3D ('AxBxC') TPU topology."""
  parts = [p.strip() for p in topology.lower().split("x") if p.strip()]
  if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
    raise ValueError(f"Invalid TPU topology '{topology}'. Expected 'AxB' or 'AxBxC' integer dimensions.")
  total_chips = math.prod(int(p) for p in parts)
  if chips_per_node <= 0 or total_chips % chips_per_node != 0:
    raise ValueError(f"Topology '{topology}' ({total_chips} chips) is not divisible by chips_per_node={chips_per_node}.")
  return total_chips // chips_per_node


def build_jobset_spec(
    job_name: str,
    namespace: str,
    queue_name: str,
    priority_class: str,
    topology: str,
    num_slices: int,
    image: str,
    command: str,
    accelerator: str = "tpu-v5p-slice",
    chips_per_node: int = 4,
    max_restarts: int = 0,
    slice_request_annotation: str = "tpu-provisioner.cloud.google.com/slice-selection",
) -> dict[str, Any]:
  """Constructs a Kueue JobSet manifest with dynamic slice selection."""
  nodes = nodes_per_slice(topology, chips_per_node=chips_per_node)
  total_pods = nodes * num_slices
  slice_spec = json.dumps({"tpuTopology": topology, "count": num_slices})

  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {
          "name": job_name,
          "namespace": namespace,
          "labels": {"kueue.x-k8s.io/queue-name": queue_name},
      },
      "spec": {
          "failurePolicy": {"maxRestarts": max_restarts},
          "replicatedJobs": [
              {
                  "name": "main-job",
                  "replicas": 1,
                  "template": {
                      "metadata": {"labels": {"kueue.x-k8s.io/queue-name": queue_name}},
                      "spec": {
                          "parallelism": total_pods,
                          "completions": total_pods,
                          "backoffLimit": 0,
                          "template": {
                              "metadata": {
                                  "annotations": {slice_request_annotation: slice_spec},
                              },
                              "spec": {
                                  "priorityClassName": priority_class,
                                  "restartPolicy": "Never",
                                  "nodeSelector": {
                                      "cloud.google.com/gke-tpu-accelerator": accelerator,
                                  },
                                  "tolerations": [
                                      {
                                          "key": "google.com/tpu",
                                          "operator": "Equal",
                                          "value": "present",
                                          "effect": "NoSchedule",
                                      }
                                  ],
                                  "volumes": [
                                      {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                                  ],
                                  "containers": [
                                      {
                                          "name": "workload-container",
                                          "image": image,
                                          "imagePullPolicy": "Always",
                                          "command": ["bash", "-lc", command],
                                          "resources": {
                                              "requests": {"google.com/tpu": chips_per_node},
                                              "limits": {"google.com/tpu": chips_per_node},
                                          },
                                          "volumeMounts": [
                                              {"name": "dshm", "mountPath": "/dev/shm"},
                                          ],
                                      }
                                  ],
                              },
                          },
                      },
                  },
              }
          ],
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-name", required=True, help="JobSet name")
  parser.add_argument("--context", required=True, help="kubectl context name")
  parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
  parser.add_argument("--queue-name", default="multislice-queue", help="Kueue LocalQueue name")
  parser.add_argument("--priority-class", default="high", help="Pod PriorityClassName")
  parser.add_argument("--topology", default="4x4x4", help="TPU slice topology ('AxB' or 'AxBxC', e.g. 4x4, 4x4x4)")
  parser.add_argument("--num-slices", type=int, default=1, help="Number of slices")
  parser.add_argument("--accelerator", default="tpu-v5p-slice", help="GKE TPU accelerator label")
  parser.add_argument("--chips-per-node", type=int, default=4, help="TPU chips per node (default: 4)")
  parser.add_argument("--image", required=True, help="Container image URI")
  parser.add_argument("--command", required=True, help="Bash command to execute in each worker pod")
  parser.add_argument("--dry-run", action="store_true", help="Print generated JobSet JSON without applying")
  args = parser.parse_args()

  spec = build_jobset_spec(
      job_name=args.job_name,
      namespace=args.namespace,
      queue_name=args.queue_name,
      priority_class=args.priority_class,
      topology=args.topology,
      num_slices=args.num_slices,
      image=args.image,
      command=args.command,
      accelerator=args.accelerator,
      chips_per_node=args.chips_per_node,
  )
  payload = json.dumps(spec, indent=2)
  if args.dry_run:
    print(payload)
    return 0

  cmd = ["kubectl", "--context", args.context, "apply", "-f", "-"]
  res = subprocess.run(cmd, input=payload, text=True, check=False)
  return res.returncode


if __name__ == "__main__":
  sys.exit(main())
