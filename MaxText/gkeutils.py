"""Utilities for GKE."""

import contextlib
import logging
import subprocess


logger = logging.getLogger(__name__)


def get_pod(substring_name: str):
  """Get a pod name."""
  p = subprocess.run(
      "kubectl get pods -o name".split(), capture_output=True, check=True
  )
  for line in p.stdout.decode("utf-8").split("\n"):
    if substring_name in line:
      return line.split("/")[1]
  raise ValueError("Could not find {substring_name} pod")


def port_forward() -> subprocess.Popen[bytes]:
  """Start a port forward process."""
  pod = get_pod("proxy")
  return subprocess.Popen(f"kubectl port-forward {pod} 38676:38676".split())


@contextlib.contextmanager
def port_forward_context():
  """Context manager for port forwarding."""
  p = port_forward()
  try:
    yield p
  finally:
    p.kill()


def try_kill_slice(slice_index: int):
  """Kill a slice."""
  try:
    pod = get_pod(f"worker-{slice_index}-0")
  except ValueError as e:
    logger.info(
        "Could not find worker pod for slice %s. Maybe it was already killed.",
        slice_index,
    )
    return
  subprocess.run([
      "kubectl",
      "exec",
      "-i",
      pod,
      "-c",
      "pathways-worker",
      "--",
      "/bin/bash",
      "-c",
      "kill -s SIGILL 1",
  ], check=True)
