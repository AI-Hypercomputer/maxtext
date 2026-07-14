import json
import os
import sys

def convert_entry(key, duration):
  # key: "tests.gather_reduce_sc_test.GatherReduceScTest.test_column0"
  # target: "tests/gather_reduce_sc_test.py::GatherReduceScTest::test_column0"

  parts = key.split('.')
  if not parts or parts[0] != 'tests':
    return None

  file_path = ""
  for i in range(1, len(parts)):
    candidate_path = os.path.join(*parts[:i]) + ".py"
    if os.path.isfile(candidate_path):
      file_path = candidate_path
      remaining_parts = parts[i:]
      break
  else:
    return None

  new_key = f"{file_path}::" + "::".join(remaining_parts)
  return new_key, duration

def test_conversion():
  original_isfile = os.path.isfile
  try:
    os.path.isfile = lambda path: path in [
        os.path.join("tests", "gather_reduce_sc_test.py"),
        os.path.join("tests", "unit", "attention_test.py"),
        os.path.join("tests", "unit", "no_class_test.py"),
    ]

    res1 = convert_entry("tests.gather_reduce_sc_test.GatherReduceScTest.test_column0", 0.003)
    assert res1 == ("tests/gather_reduce_sc_test.py::GatherReduceScTest::test_column0", 0.003), f"Expected tests/gather_reduce_sc_test.py::GatherReduceScTest::test_column0, got {res1}"

    res2 = convert_entry("tests.unit.attention_test.AttentionTest.test_dot_product_reshape_q", 64.84)
    assert res2 == ("tests/unit/attention_test.py::AttentionTest::test_dot_product_reshape_q", 64.84), f"Expected tests/unit/attention_test.py::AttentionTest::test_dot_product_reshape_q, got {res2}"

    res3 = convert_entry("tests.unit.non_existent_test.AttentionTest.test_foo", 1.0)
    assert res3 is None, f"Expected None, got {res3}"

    res4 = convert_entry("tests.unit.no_class_test.test_bar", 1.0)
    assert res4 == ("tests/unit/no_class_test.py::test_bar", 1.0), f"Expected tests/unit/no_class_test.py::test_bar, got {res4}"

    print("All tests passed!")
  finally:
    os.path.isfile = original_isfile

def main():
  if len(sys.argv) > 1 and sys.argv[1] == "--test":
    test_conversion()
    sys.exit(0)

  if len(sys.argv) < 3:
    print("Usage: python convert_durations.py <input_json> <output_json> | --test")
    sys.exit(1)

  input_file = sys.argv[1]
  output_file = sys.argv[2]

  try:
    with open(input_file, 'r') as f:
      data = json.load(f)
  except FileNotFoundError:
    print(f"Input file {input_file} not found.")
    sys.exit(1)

  converted_data = {}
  for key, duration in data.items():
    result = convert_entry(key, duration)
    if result:
      new_key, new_duration = result
      converted_data[new_key] = new_duration

  with open(output_file, 'w') as f:
    json.dump(converted_data, f, indent=2)

if __name__ == "__main__":
  main()
