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

"""Unit tests for the gcluster-tpu-runner skill utilities."""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "gcluster-tpu-runner" / "scripts"
sys.path.insert(0, str(RUNNER_SCRIPTS))

import aot_preflight  # pylint: disable=wrong-import-position
import make_stage_log_links  # pylint: disable=wrong-import-position
import preflight_check  # pylint: disable=wrong-import-position
import submit_gsc_jobset  # pylint: disable=wrong-import-position


class TpuInfraSkillsTest(unittest.TestCase):
  """Tests TPU pre-flight, topology-aware HBM/JobSet sizing, and stage log verification."""

  def test_infer_hbm_limit_gib_across_tpu_generations(self):
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("v4-128"), 31.7)
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("v5e-256"), 15.7)
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("v5p-128"), 94.7)
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("v6e-16"), 31.2)
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("tpu7x-128"), 191.5)
    self.assertAlmostEqual(aot_preflight.infer_hbm_limit_gib("v5p-128", override_gib=80.0), 80.0)

  def test_dynamic_nodes_per_slice_and_jobset_spec(self):
    self.assertEqual(submit_gsc_jobset.nodes_per_slice("2x2x1"), 1)
    self.assertEqual(submit_gsc_jobset.nodes_per_slice("4x4"), 4)
    self.assertEqual(submit_gsc_jobset.nodes_per_slice("4x4x4"), 16)
    self.assertEqual(submit_gsc_jobset.nodes_per_slice("4x4x8"), 32)
    self.assertEqual(submit_gsc_jobset.nodes_per_slice("8x16"), 32)
    with self.assertRaises(ValueError):
      submit_gsc_jobset.nodes_per_slice("invalid")

    spec = submit_gsc_jobset.build_jobset_spec(
        job_name="test-job",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="high",
        topology="2x4x4",
        num_slices=2,
        image="gcr.io/test/img:latest",
        command="echo hello",
    )
    job_spec = spec["spec"]["replicatedJobs"][0]["template"]["spec"]
    self.assertEqual(job_spec["parallelism"], 16)
    self.assertEqual(job_spec["completions"], 16)

  def test_dockerignore_and_overlay_command(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      ok_before, missing = preflight_check.validate_dockerignore(tmpdir, fix=False)
      self.assertFalse(ok_before)
      self.assertIn(".git", missing)

      ok_after, missing_after = preflight_check.validate_dockerignore(tmpdir, fix=True)
      self.assertTrue(ok_after)
      self.assertEqual(missing_after, [])

    wrapped = preflight_check.wrap_deps_overlay_command("python3 -m maxtext.train", use_deps_overlay=True)
    self.assertIn("cp -a /src/.", wrapped)
    self.assertEqual(
        preflight_check.wrap_deps_overlay_command("python3 -m maxtext.train", use_deps_overlay=False),
        "python3 -m maxtext.train",
    )

  def test_make_stage_log_links_url_and_cleanup(self):
    url = make_stage_log_links.build_logs_explorer_url(
        project="test-proj",
        cluster="test-cluster",
        pod="job-0-0-abc",
        start_iso="2026-10-12T00:00:00Z",
        end_iso="2026-10-12T00:10:00Z",
    )
    self.assertIn("test-proj", url)
    self.assertIn("job-0-0-abc", url)

    with mock.patch("subprocess.run", side_effect=RuntimeError("gcloud failure")):
      with self.assertRaises(RuntimeError):
        make_stage_log_links.verify_stage_logs(
            "test-proj", "test-cluster", "job-0-0-abc", "2026-10-12T00:00:00Z", "2026-10-12T00:10:00Z", "step"
        )


if __name__ == "__main__":
  unittest.main()
