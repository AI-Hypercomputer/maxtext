# Copyright 2023-2026 Google LLC
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


"""Unit tests for `calculate_distillation_tflops_per_device`.

Verifies the canonical training-FLOPs heuristic for online distillation:
  - student does fwd + bwd ≈ 6 * P_student * tokens
  - teacher does fwd only  ≈ 2 * P_teacher * tokens
  - combined per-step TFLOPs is the sum
  - offline mode zeroes the teacher contribution
"""

import pytest

pytest.importorskip("tunix")

pytestmark = [pytest.mark.cpu_only, pytest.mark.post_training]

import unittest
from unittest import mock

from absl.testing import absltest

from maxtext.trainers.post_train.distillation import distillation_utils


def _make_fake_calculate_tflops(tflops_by_id):
  """Returns a fake `calculate_tflops_training_per_device` keyed by `id(config)`.

  The real function returns `(total_tflops, learnable_weight_tflops, attention_tflops)`
  where `total = 6 * P * tokens / 1e12` for fwd+bwd. The fake mirrors that signature.
  """

  def _fake(config, log=False):
    del log
    total = tflops_by_id[id(config)]
    return total, 0.0, 0.0

  return _fake


class DistillationTflopsTest(unittest.TestCase):
  """Tests for the per-step distillation TFLOPs accounting."""

  def test_online_uses_6P_student_and_2P_teacher(self):
    # P in #params, D in #tokens. Use values where the answer is exact in float.
    p_student, p_teacher, tokens = 1_000_000_000, 4_000_000_000, 8192

    student_tflops_fwd_bwd = 6 * p_student * tokens / 1e12  # 6PD
    teacher_tflops_fwd_bwd = 6 * p_teacher * tokens / 1e12  # underlying util always returns 6PD

    student_config = mock.Mock(name="student_config")
    teacher_config = mock.Mock(name="teacher_config")
    fake = _make_fake_calculate_tflops(
        {
            id(student_config): student_tflops_fwd_bwd,
            id(teacher_config): teacher_tflops_fwd_bwd,
        }
    )

    with mock.patch.object(distillation_utils.maxtext_utils, "calculate_tflops_training_per_device", side_effect=fake):
      combined, student, teacher = distillation_utils.calculate_distillation_tflops_per_device(
          student_config, teacher_config, is_offline=False
      )

    # Student: full fwd + bwd = 6 * P * D.
    self.assertAlmostEqual(student, 6 * p_student * tokens / 1e12, places=9)
    # Teacher: forward only = 2 * P * D (= 6PD / 3).
    self.assertAlmostEqual(teacher, 2 * p_teacher * tokens / 1e12, places=9)
    # Combined: 6 P_s D + 2 P_t D.
    self.assertAlmostEqual(combined, (6 * p_student + 2 * p_teacher) * tokens / 1e12, places=9)

  def test_offline_zeroes_teacher_contribution(self):
    p_student, tokens = 1_000_000_000, 8192
    student_tflops_fwd_bwd = 6 * p_student * tokens / 1e12

    student_config = mock.Mock(name="student_config")
    teacher_config = mock.Mock(name="teacher_config")
    fake = _make_fake_calculate_tflops(
        {
            id(student_config): student_tflops_fwd_bwd,
            # Teacher entry exists but should never be consulted in offline mode;
            # making it large would explode the assertion if it were used.
            id(teacher_config): 1e9,
        }
    )

    with mock.patch.object(distillation_utils.maxtext_utils, "calculate_tflops_training_per_device", side_effect=fake):
      combined, student, teacher = distillation_utils.calculate_distillation_tflops_per_device(
          student_config, teacher_config, is_offline=True
      )

    self.assertEqual(teacher, 0.0)
    self.assertAlmostEqual(student, 6 * p_student * tokens / 1e12, places=9)
    self.assertAlmostEqual(combined, student, places=9)

  def test_none_teacher_config_zeroes_teacher_contribution(self):
    p_student, tokens = 5e8, 4096
    student_tflops_fwd_bwd = 6 * p_student * tokens / 1e12

    student_config = mock.Mock(name="student_config")
    fake = _make_fake_calculate_tflops({id(student_config): student_tflops_fwd_bwd})

    with mock.patch.object(distillation_utils.maxtext_utils, "calculate_tflops_training_per_device", side_effect=fake):
      combined, student, teacher = distillation_utils.calculate_distillation_tflops_per_device(
          student_config, teacher_config=None, is_offline=False
      )

    self.assertEqual(teacher, 0.0)
    self.assertAlmostEqual(combined, student, places=9)


if __name__ == "__main__":
  absltest.main()
