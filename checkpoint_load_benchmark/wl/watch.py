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

"""Follows the event logs of running drivers and returns when something needs attention.

  watch.py --timeout-s 3300 STATE_DIR [STATE_DIR ...]

Prints the events that each DIR/events.log gained since the last call (the position is kept in DIR/watch.offset).
Returns as soon as an ATTN event or a DRIVER_DONE appears, and otherwise after --timeout-s, so a caller that reruns
it gets failures at once and a progress report at that interval.
"""

import argparse
import os
import time


def read_new(state_dir: str) -> list[str]:
  """Returns the complete lines added to the event log of `state_dir` since the last call."""
  log, offset_path = os.path.join(state_dir, "events.log"), os.path.join(state_dir, "watch.offset")
  if not os.path.exists(log):
    return []
  offset = 0
  if os.path.exists(offset_path):
    with open(offset_path, "r", encoding="utf-8") as f:
      offset = int(f.read().strip() or 0)
  with open(log, "rb") as f:
    f.seek(offset)
    data = f.read()
  data = data[: data.rfind(b"\n") + 1]
  with open(offset_path, "w", encoding="utf-8") as f:
    f.write(str(offset + len(data)))
  return data.decode("utf-8", errors="replace").splitlines()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("state_dirs", nargs="+")
  parser.add_argument("--timeout-s", type=int, default=3300)
  parser.add_argument("--poll-s", type=int, default=30)
  args = parser.parse_args()
  deadline = time.time() + args.timeout_s
  done = set()
  while True:
    urgent = False
    for state_dir in args.state_dirs:
      for line in read_new(state_dir):
        print(f"[{os.path.basename(state_dir.rstrip('/'))}] {line}", flush=True)
        parts = line.split(" ", 2)
        urgent |= len(parts) > 1 and parts[1] == "ATTN"
        if len(parts) > 2 and parts[2].startswith("DRIVER_DONE"):
          done.add(state_dir)
    if urgent or len(done) == len(args.state_dirs) or time.time() >= deadline:
      return
    time.sleep(args.poll_s)


if __name__ == "__main__":
  main()
