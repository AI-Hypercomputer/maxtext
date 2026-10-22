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

"""Unit tests for the maxtext-pr-upload skill utilities."""

import pathlib
import sys
import unittest
from unittest import mock
import urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PR_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "maxtext-pr-upload" / "scripts"
sys.path.insert(0, str(PR_SCRIPTS))

import check_doc_train_cmds  # pylint: disable=wrong-import-position
import ci_status  # pylint: disable=wrong-import-position
import pr_comments  # pylint: disable=wrong-import-position


class DevWorkflowSkillsTest(unittest.TestCase):
  """Tests Markdown train command extraction, CI step classification, and PR comment threading."""

  def test_check_doc_train_cmds_parsing_and_validation(self):
    md = (
        "## Training\n"
        "```bash\n"
        "python3 -u -m maxtext.train src/maxtext/configs/base.yml \\\n"
        "  base_output_directory=${BASE_OUTPUT_DIRECTORY} \\\n"
        "  model_name=llama3.1-8b\n"
        "```\n"
        "## Next\n"
    )
    blocks = check_doc_train_cmds.extract_train_blocks(md)
    self.assertEqual(len(blocks), 1)
    subs = check_doc_train_cmds.parse_substitutions(["BASE_OUTPUT_DIRECTORY=/tmp/out"])
    argv, truncated = check_doc_train_cmds.build_pyconfig_argv(blocks[0], subs)
    self.assertFalse(truncated)
    self.assertEqual(
        argv,
        ["train.py", "src/maxtext/configs/base.yml", "base_output_directory=/tmp/out", "model_name=llama3.1-8b"],
    )

    with self.assertRaises(ValueError):
      check_doc_train_cmds.parse_substitutions(["INVALID_SUBST_WITHOUT_EQUALS"])

  def test_ci_status_step_classification_and_retry(self):
    job_test_fail = {
        "conclusion": "failure",
        "steps": [
            {"name": "Run Tests", "conclusion": "failure"},
            {"name": "Upload Test Results", "conclusion": "success"},
        ],
    }
    self.assertEqual(ci_status.classify_job_steps(job_test_fail), "test_failure")

    job_preempted = {
        "conclusion": "failure",
        "steps": [
            {"name": "Run Tests", "conclusion": "cancelled"},
            {"name": "Upload Test Results", "conclusion": "skipped"},
        ],
    }
    self.assertEqual(ci_status.classify_job_steps(job_preempted), "runner_interrupted")

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("transient")):
      with self.assertRaises(urllib.error.URLError):
        ci_status.api_get("/pulls/1", throttle_s=0.0, max_retries=2)

  def test_pr_comments_thread_grouping(self):
    comments = [
        {"id": 10, "user": {"login": "reviewer1"}, "path": "a.py", "line": 12, "body": "Please refactor."},
        {"id": 11, "in_reply_to_id": 10, "user": {"login": "author1"}, "path": "a.py", "line": 12, "body": "Done."},
        {"id": 20, "user": {"login": "reviewer2"}, "path": "b.py", "line": 34, "body": "Missing check."},
    ]
    threads = pr_comments.group_inline_threads(comments, author_login="author1")
    self.assertEqual(len(threads), 2)
    self.assertFalse(threads[0]["needs_reply"])
    self.assertTrue(threads[1]["needs_reply"])


if __name__ == "__main__":
  unittest.main()
