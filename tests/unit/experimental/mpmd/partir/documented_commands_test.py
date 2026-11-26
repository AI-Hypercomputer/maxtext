# Copyright 2026 Google LLC
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

"""The command lines the partir package documents parse, and pass the checks the TPUs would otherwise make."""

import contextlib
import io
import os
import re
import shlex
import unittest

from absl import flags

from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import launch
from maxtext.experimental.mpmd.partir import parity
from maxtext.experimental.mpmd.partir import spmd_baseline
from maxtext.experimental.mpmd.partir import train
from maxtext.experimental.mpmd.partir import train_compile

_PACKAGE = os.path.dirname(hp.__file__)
# Where commands are documented: module docstrings, the README and local_pathways.sh's usage.
_DOCUMENTS = (
    "README.md",
    "local_pathways.sh",
    "launch.py",
    "parity.py",
    "spmd_baseline.py",
    "train.py",
    "train_compile.py",
)
_DEFINE_FLAGS = {
    "launch": launch.define_flags,
    "parity": parity.define_flags,
    "spmd_baseline": spmd_baseline.define_flags,
    "train": train.define_flags,
    "train_compile": train_compile.define_flags,
}
# A command's arguments run to the end of its line, or to the backticks closing inline code.
_COMMAND = re.compile(r"-m maxtext\.experimental\.mpmd\.partir\.(\w+)([^\n`]*)")
# A shell operator or redirection ends the arguments.
_SHELL_END = re.compile(r"(;|&&?|\|\|?|\d*>>?(&\d+)?|<)")
# Commands that name no TPU type run on 8 devices: train_compile's fake CPU devices (--num_cpu_devices), or the 4-chip
# TPU7x VM (tpu7x:2x2x1) of local_pathways.sh and the README.
_LOCAL_DEVICES = 8


def documented_commands(text):
  """Returns ``(module, argv)`` for each ``python3 -m maxtext.experimental.mpmd.partir.<module> ...`` in `text`.

  Lines continued with a backslash are joined first, also inside shell comments. The arguments are split as the
  shell splits them (``--flag=value``, ``--flag value`` and quoted values alike) and end at a comment, a shell operator
  or redirection, or the end of the command.
  """
  commands = []
  for match in _COMMAND.finditer(re.sub(r"\\\n[\s#]*", " ", text)):
    argv = []
    for word in shlex.split(match.group(2), comments=True):
      if _SHELL_END.fullmatch(word):
        break
      argv.append(word)
    commands.append((match.group(1), argv))
  return commands


class DocumentedCommandsTest(unittest.TestCase):

  def test_every_documented_command(self):
    found = []
    for document in _DOCUMENTS:
      with open(os.path.join(_PACKAGE, document), encoding="utf-8") as handle:
        found += [(document, module, argv) for module, argv in documented_commands(handle.read())]
    # The README documents train, parity, train_compile, launch and spmd_baseline runs; every module docstring one.
    self.assertGreaterEqual(len(found), 12, found)
    self.assertEqual({module for _, module, _ in found}, set(_DEFINE_FLAGS))
    for document, module, argv in found:
      with self.subTest(document=document, command=" ".join([module, *argv])):
        flag_values = flags.FlagValues()
        _DEFINE_FLAGS[module](flag_values)
        flag_values([module, *argv])
        if module == "launch":
          # The launcher's own checks: required flags, the workload name, slices and stages, the microbatch split.
          with contextlib.redirect_stdout(io.StringIO()):
            launch.main([module], flag_values=flag_values)
          continue
        hparams = hp.from_flags(flag_values)
        if module == "spmd_baseline":
          devices = hp.devices_in_device_type(flag_values.device_type) * flag_values.num_slices
          with contextlib.redirect_stdout(io.StringIO()):
            spmd_baseline.main([module], flag_values=flag_values)  # MaxText's pipeline constraints
        elif module == "train_compile" and flag_values.compile_topology == "cpu":
          devices = flag_values.num_cpu_devices
        else:
          devices = _LOCAL_DEVICES
        # The MPMD run of the same flags must split each microbatch over its stage's batch-sharding devices.
        hp.check_microbatch_split_on(hparams, devices)

  def test_the_extraction(self):
    text = (
        "  python3 -m maxtext.experimental.mpmd.partir.train --pp=4 \\\n"
        "  --num_microbatches=4   # a comment\n"
        "#   python3 -m maxtext.experimental.mpmd.partir.parity --pp=2 \\\n"
        "#       --steps=3\n"
        "python3 -m maxtext.experimental.mpmd.partir.train --num_microbatches 8 --run_name 'a b' \\\n"
        '  --maxtext_overrides="dtype=float32" > /tmp/train.log 2>&1\n'
        "Run `python3 -m maxtext.experimental.mpmd.partir.launch --cluster=<cluster>` first.\n"
    )
    self.assertEqual(
        documented_commands(text),
        [
            ("train", ["--pp=4", "--num_microbatches=4"]),
            ("parity", ["--pp=2", "--steps=3"]),
            ("train", ["--num_microbatches", "8", "--run_name", "a b", "--maxtext_overrides=dtype=float32"]),
            ("launch", ["--cluster=<cluster>"]),
        ],
    )


if __name__ == "__main__":
  unittest.main()
