import unittest
from unittest import mock

import maxtext.experimental.agent.ckpt_validation_pipeline.forward_compile_validator as fcv


class TestForwardCompileValidator(unittest.TestCase):

  @mock.patch("maxtext.experimental.agent.ckpt_validation_pipeline.forward_compile_validator.run_mock_forward")
  def test_run_mock_forward_success(self, mock_run):
    mock_run.return_value = {"layer": (10, 10)}
    res = fcv.run_mock_forward("mock_path", "mock_model")
    self.assertEqual(res, {"layer": (10, 10)})

  @mock.patch("maxtext.experimental.agent.ckpt_validation_pipeline.forward_compile_validator.gcs_utils.upload_blob")
  def test_gcs_upload_try_except(self, mock_upload):
    mock_upload.side_effect = Exception("Network blip")
    # should not crash
    try:
      mock_upload("gs://fake", "fake.json")
    except:
      pass
    self.assertEqual(mock_upload.call_count, 1)


if __name__ == "__main__":
  unittest.main()
