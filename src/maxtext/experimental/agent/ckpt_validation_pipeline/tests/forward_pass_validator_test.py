import unittest
from unittest import mock

import maxtext.experimental.agent.ckpt_validation_pipeline.forward_pass_validator as fpv


class TestForwardPassValidator(unittest.TestCase):

  @mock.patch("maxtext.experimental.agent.ckpt_validation_pipeline.forward_pass_validator.runpy.run_path")
  def test_forward_pass_success(self, mock_run_path):
    # Implement a real test invoking the code under test
    fpv.validate_forward_pass("test_run", "llama", "gs://path", "", [])
    mock_run_path.assert_called_once()


if __name__ == "__main__":
  unittest.main()
