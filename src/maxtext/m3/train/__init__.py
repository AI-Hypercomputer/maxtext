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

"""m3's own training stack, built on `nnx.Optimizer`.

Planned modules:
  * `state.py` — optimizer construction.
  * `loss.py` — cross entropy plus auxiliary losses.
  * `step.py` — train/eval step functions.
  * `loop.py` — training loop and infrastructure composition site.

m3 models are drivable by two things: this loop, and the existing
`maxtext.training_engine.maxtext_engine` (which the RL stack targets). Anything
that makes a model only usable by one of the two drivers belongs here or in a
bridge, not in a model file.
"""
