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

"""Offline tests for reading one attempt's job list (`runs.get_jobs`).

GitHub can answer a server error (502) to an attempt's job list in pages of 100 while it serves
the same list in smaller pages (run 39983861967 attempt 3 did, on 2026-10-27). `get_jobs` then
asks again with smaller pages. A fake client scripts GitHub's answer per page size, so nothing
here reaches GitHub.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import Any

# The collector package's parent, so the import stays inside tools/ci_metrics.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
  sys.path.insert(0, _PACKAGE_PARENT)

from collector import github
from collector import runs

JOBS = [{"id": number, "name": f"job {number}"} for number in range(1, 69)]
PATH = "actions/runs/39983861967/attempts/3/jobs"


def server_error(status: int = 502) -> github.GitHubError:
  return github.GitHubError(f"GET {PATH} answered {status} after 3 attempt(s)", status=status, url=PATH)


class PageSizeClient:
  """Answers the job list per page size: a list of jobs, or an error to raise."""

  def __init__(self, answers: dict[int | None, Any]) -> None:
    self.answers = answers
    self.calls: list[tuple[str, str, dict[str, Any]]] = []

  def get_json(self, path: str, **params: Any) -> dict[str, Any]:
    raise AssertionError(f"get_jobs must not call get_json ({path})")

  def paginate(self, path: str, key: str, **params: Any) -> list:
    self.calls.append((path, key, dict(params)))
    answer = self.answers[params.get("per_page")]
    if isinstance(answer, Exception):
      raise answer
    return list(answer)

  def page_sizes(self) -> list[int | None]:
    return [params.get("per_page") for _, _, params in self.calls]


def read_jobs(client: PageSizeClient) -> tuple[list[dict[str, Any]], str]:
  warnings = io.StringIO()
  with contextlib.redirect_stderr(warnings):
    jobs = runs.get_jobs(client, 39983861967, 3)
  return jobs, warnings.getvalue()


class GetJobsTest(unittest.TestCase):
  """How `get_jobs` reads a job list, and what it does when GitHub fails."""

  def test_a_normal_list_is_one_call_with_the_default_page_size(self):
    client = PageSizeClient({None: JOBS})
    jobs, warnings = read_jobs(client)
    self.assertEqual(jobs, JOBS)
    self.assertEqual(client.calls, [(PATH, "jobs", {})])
    self.assertEqual(warnings, "")

  def test_a_server_error_is_read_again_in_pages_of_50(self):
    client = PageSizeClient({None: server_error(502), 50: JOBS})
    jobs, warnings = read_jobs(client)
    self.assertEqual(jobs, JOBS)
    self.assertEqual(client.page_sizes(), [None, 50])
    self.assertIn("run 39983861967 attempt 3: GitHub answered 502 for its jobs in pages of 100", warnings)
    self.assertIn("read them in pages of 50 instead", warnings)

  def test_pages_of_30_are_tried_when_pages_of_50_fail_too(self):
    client = PageSizeClient({None: server_error(502), 50: server_error(504), 30: JOBS})
    jobs, warnings = read_jobs(client)
    self.assertEqual(jobs, JOBS)
    self.assertEqual(client.page_sizes(), [None, 50, 30])
    self.assertIn("GitHub answered 504 for its jobs in pages of 100; read them in pages of 30 instead", warnings)

  def test_when_every_page_size_fails_the_last_error_is_raised(self):
    # The tick then records the run as not collected and tries it again next time, as before.
    client = PageSizeClient({None: server_error(502), 50: server_error(502), 30: server_error(503)})
    with self.assertRaises(github.GitHubError) as raised, contextlib.redirect_stderr(io.StringIO()):
      runs.get_jobs(client, 39983861967, 3)
    self.assertEqual(raised.exception.status, 503)
    self.assertEqual(client.page_sizes(), [None, 50, 30])

  def test_a_refusal_is_not_tried_with_smaller_pages(self):
    # 403 means GitHub refused the request (for example the rate limit), not that it failed on the size.
    client = PageSizeClient({None: github.GitHubError("GET answered 403", status=403)})
    with self.assertRaises(github.GitHubError) as raised:
      read_jobs(client)
    self.assertEqual(raised.exception.status, 403)
    self.assertEqual(client.page_sizes(), [None])

  def test_a_refusal_during_the_smaller_pages_is_raised_at_once(self):
    client = PageSizeClient({None: server_error(502), 50: github.GitHubError("GET answered 403", status=403)})
    with self.assertRaises(github.GitHubError) as raised:
      read_jobs(client)
    self.assertEqual(raised.exception.status, 403)
    self.assertEqual(client.page_sizes(), [None, 50])

  def test_an_error_without_a_status_is_not_tried_with_smaller_pages(self):
    # A connection that failed three times has no HTTP status; smaller pages would not help.
    client = PageSizeClient({None: github.GitHubError("GET failed after 3 attempt(s): reset")})
    with self.assertRaises(github.GitHubError):
      read_jobs(client)
    self.assertEqual(client.page_sizes(), [None])

  def test_a_pruned_attempt_still_reads_as_no_jobs(self):
    client = PageSizeClient({None: github.GitHubError("GET answered 404", status=404)})
    jobs, warnings = read_jobs(client)
    self.assertEqual(jobs, [])
    self.assertIn("has no jobs endpoint", warnings)
    self.assertEqual(client.page_sizes(), [None])

  def test_the_smaller_pages_are_smaller_than_the_default(self):
    self.assertTrue(runs.JOBS_FALLBACK_PAGE_SIZES)
    sizes = list(runs.JOBS_FALLBACK_PAGE_SIZES)
    self.assertEqual(sizes, sorted(sizes, reverse=True))
    self.assertLess(max(sizes), github.PER_PAGE)


if __name__ == "__main__":
  unittest.main()
