import os

def replace_in_file(filepath, old, new):
    if not os.path.exists(filepath): return
    with open(filepath, "r") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched {filepath}")

replace_in_file(
    "src/maxtext/experimental/agent/ckpt_validation_pipeline/tests/forward_compile_validator_test.py",
    "from unittest import mock\nimport argparse",
    "from unittest import mock"
)
replace_in_file(
    "src/maxtext/experimental/agent/ckpt_validation_pipeline/tests/forward_pass_validator_test.py",
    "from unittest import mock\nimport subprocess",
    "from unittest import mock"
)
replace_in_file(
    "src/maxtext/experimental/agent/ckpt_validation_pipeline/tests/decode_validator_test.py",
"""  @mock.patch("subprocess.Popen")
  def test_validator_success(self, mock_subprocess, mock_open_file, mock_remove):""",
"""  @mock.patch("subprocess.Popen")
  def test_validator_success(self, _, mock_open_file, mock_remove):"""
)
