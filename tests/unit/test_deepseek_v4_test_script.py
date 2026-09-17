import unittest
import subprocess
import re

class TestDeepseekV4Script(unittest.TestCase):

  def setUp(self):
    self.script_path = "tests/end_to_end/tpu/deepseek/v4-284b/2_test_deepseek.sh"
    with open(self.script_path, "r") as f:
      self.script_content = f.read()

  def test_max_kl_div(self):
    match = re.search(r"--max_kl_div=\"\$\{MAX_KL_DIV:-([0-9.]+)\}\"", self.script_content)
    self.assertIsNotNone(match, "max_kl_div flag not found in the correct format")
    val = float(match.group(1))
    self.assertEqual(val, 0.35)

  def test_decode_assert(self):
    self.assertIn("autoregressive_decode_assert=\"${DECODE_ASSERT:-", self.script_content)
    match = re.search(r"autoregressive_decode_assert=\"\$\{DECODE_ASSERT:-(.*?)\}\"", self.script_content)
    self.assertIsNotNone(match)
    val = match.group(1).strip()
    self.assertTrue(len(val) > 0, "Decode assert should be non-empty")

  def test_bash_n_syntax(self):
    result = subprocess.run(["bash", "-n", self.script_path], capture_output=True)
    err = result.stderr.decode()
    self.assertEqual(result.returncode, 0, f"bash -n failed: {err}")

if __name__ == "__main__":
  unittest.main()
