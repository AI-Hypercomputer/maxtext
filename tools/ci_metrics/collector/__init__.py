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

"""Collector package for the CI Pulse dashboard.

Reads the "MaxText Package Tests" pipeline from the GitHub API and its JUnit artifacts,
and turns them into the rows the static dashboard is built from. Nothing here calls an AI
service: every value is an API field, an XML element or arithmetic on those.

Modules:
  github: read-only GitHub REST client (auth, pagination, retries, rate limits).
  junit: artifact listing and JUnit XML parsing.
  runs: run, attempt and job discovery - the allowlist, the window, supersession and the
    pull request each run belongs to.
  derive: job objects to dashboard numbers - suite duration, workers, queue and setup,
    the phase split, machine time and rescues.
  rows: the shapes that get stored, their keys and their JSON round trip.
  store: the append-only store on disk - month files, dedup, corrections, the index in
    state.json, and monthly compaction.
  views: stored rows to the JSON the browser loads - the five month-split view groups,
    one file per merged pull request, and meta.json.
  tick: the command a schedule calls - which window to ask for, which run to read, which
    test rows to keep, and when to stop.
  demo: a read-only command-line proof that prints every number for one run.

The modules are imported directly (`from collector import github`) so that importing the
package never opens a network session.
"""
