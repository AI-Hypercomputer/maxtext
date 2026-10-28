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

"""Keeps a benchmark daemon pod busy from a small CPU pod (standard library only).

A pod started with `benchmark_ckpt_load_time.py --remote-slice-daemon` runs each shell command it receives
over TCP and streams the output back, ending with `__REMOTE_EXIT_CODE__:<rc>`. When the runner pod of a 2-slice
pair is evicted, the surviving daemon pod can still run single-slice work: this controller sends it the same
command repeatedly (e.g. `tpu7x_task_queue.py --max-tasks 1 --no-wait`), because the daemon's watchdog caps each
command at 4 h. The command's exit code decides what happens next:

  0 -> stop (no work left for this daemon)     2 -> send the command again right away
  3 -> wait --retry-sleep-s, then send again   other -> count an error, wait 60 s, retry (stop after --max-errors)

With --shutdown-at-end, the controller then sends the daemon `{"shutdown": true}`, so its pod completes and
frees the TPU node.

Usage:
  python3 tpu7x_daemon_controller.py --daemon mesakh-w1-f:55555 --command "$(cat worker_cmd.sh)" --shutdown-at-end
"""

import argparse
import json
import socket
import sys
import time


def run_on_daemon(addr: str, command: str) -> int:
  host, port = addr.rsplit(":", 1)
  rc = 1
  with socket.create_connection((host, int(port)), timeout=60) as sock:
    sock.settimeout(None)
    # Older daemons index `env` and `local_work_dir` directly, so always send both.
    payload = {"cmd": ["bash", "-c", command], "env": {}, "local_work_dir": "/tmp/benchmark_remote"}
    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    buf = b""
    while True:
      data = sock.recv(65536)
      if not data:
        break
      buf += data
      *lines, buf = buf.split(b"\n")
      for line in lines:
        text = line.decode("utf-8", errors="replace")
        if text.startswith("__REMOTE_EXIT_CODE__:"):
          rc = int(text.split(":", 1)[1])
        else:
          print(f"[{addr}] {text}", flush=True)
  return rc


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--daemon", required=True, help="host:port of the daemon")
  p.add_argument("--command", required=True, help="Shell command sent on each iteration")
  p.add_argument("--retry-sleep-s", type=int, default=120)
  p.add_argument("--max-errors", type=int, default=5)
  p.add_argument("--shutdown-at-end", action="store_true")
  args = p.parse_args()

  errors = 0
  while True:
    try:
      rc = run_on_daemon(args.daemon, args.command)
    except OSError as ex:
      print(f"[controller] Cannot reach daemon {args.daemon}: {ex}", flush=True)
      return 1
    print(f"[controller] Command finished with exit code {rc}", flush=True)
    if rc == 0:
      break
    if rc == 2:
      continue
    if rc == 3:
      time.sleep(args.retry_sleep_s)
      continue
    errors += 1
    if errors >= args.max_errors:
      print(f"[controller] {errors} failed commands; giving up.", flush=True)
      break
    time.sleep(60)

  if args.shutdown_at_end:
    try:
      host, port = args.daemon.rsplit(":", 1)
      with socket.create_connection((host, int(port)), timeout=30) as sock:
        sock.sendall((json.dumps({"shutdown": True}) + "\n").encode("utf-8"))
      print(f"[controller] Sent shutdown to {args.daemon}", flush=True)
    except OSError as ex:
      print(f"[controller] Shutdown of {args.daemon} failed: {ex}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
