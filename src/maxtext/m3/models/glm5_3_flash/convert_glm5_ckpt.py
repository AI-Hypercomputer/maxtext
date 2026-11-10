# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line entrypoint for GLM-5.3-Flash HuggingFace-to-Orbax checkpoint conversion."""

from collections.abc import Sequence

from absl import app
from absl import flags
from maxtext.m3.models.glm5_3_flash import glm5_ckpt


def _parse_flags(argv: Sequence[str]) -> list[str]:
  flags.FLAGS.mark_as_parsed()
  return list(argv)


if __name__ == "__main__":
  app.run(lambda argv: glm5_ckpt.run_cli(argv[1:]), flags_parser=_parse_flags)
