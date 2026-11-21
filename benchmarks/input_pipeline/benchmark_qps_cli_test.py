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

"""Tests the `benchmark_qps` entry point's absl flags parser and its hand-off to argparse."""

import unittest

from absl import flags

from benchmarks.input_pipeline import benchmark_qps
from benchmarks.input_pipeline import qps_runner


class BenchmarkQpsCliTest(unittest.TestCase):

  def test_flags_parser_marks_absl_parsed_and_keeps_argv(self):
    argv = ["benchmark_qps", "--backend=emulator", "--host_counts=1,4", "--startup_counting=every_host,manifest"]
    flags.FLAGS.unparse_flags()
    self.assertFalse(flags.FLAGS.is_parsed())
    # absl.app.run requires FLAGS to be parsed once flags_parser returns.
    self.assertEqual(benchmark_qps._flags_parser(argv), argv)  # pylint: disable=protected-access
    self.assertTrue(flags.FLAGS.is_parsed())
    args = qps_runner.create_parser().parse_args(argv[1:])
    self.assertEqual((args.backend, args.host_counts, args.startup_counting), ("emulator", "1,4", "every_host,manifest"))


if __name__ == "__main__":
  unittest.main()
