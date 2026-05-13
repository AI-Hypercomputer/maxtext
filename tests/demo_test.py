"""
python -m tests.demo_test
"""

import sys
import jax
import tests
import maxtext
import os

print("\n== ENV ==")
print("PWD", os.environ["PWD"])
print("PYTHONPATH", os.environ.get("PYTHONPATH",""))

print("\n== SYS PATH ==")
print("sys.path", sys.path)
print("\n== MODULE PATH ==")
print("jax", jax.__file__)
print("tests", tests.__file__)
print("maxtext", maxtext.__file__)


from maxtext import demo_code

def main():
  print("\n== MAIN ==")
  print("DEBUG: tests/demo_test.py")
  demo_code.main()

if __name__ == "__main__":
  main()