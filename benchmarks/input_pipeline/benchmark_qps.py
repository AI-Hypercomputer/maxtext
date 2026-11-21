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

"""CLI for the MaxText PyGrain + ArrayRecord GCS QPS and step-time benchmark.

See `qps_runner.py` and `README.md` for the measurement model and flags.

Hermetic smoke run (local GCS emulator, 4 simulated hosts):

  python3 -m benchmarks.input_pipeline.benchmark_qps --backend=emulator --host_counts=4 \
    --grain_worker_count=0,2 --startup_counting=every_host,manifest --steps=5

Live GCS run (on a VM in the bucket's region):

  python3 -m benchmarks.input_pipeline.benchmark_qps --backend=gcs \
    --train_files='gs://BUCKET/c4/en/3.0.1/c4-train.array_record-*' --host_counts=1,4,8 \
    --measure_mode=paired --output_json=/tmp/qps.json --output_markdown=/tmp/qps.md
"""

import sys

from absl import flags

from benchmarks.input_pipeline import qps_runner


def _flags_parser(argv):
  """Marks absl flags as parsed and leaves every benchmark flag to argparse in `qps_runner`."""
  flags.FLAGS(argv[:1])
  return argv


def _main(argv):
  sys.exit(qps_runner.main(argv))


if __name__ == "__main__":
  _main(sys.argv)
