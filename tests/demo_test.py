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
print("maxtext", maxtext.__path__)


from maxtext import demo_code
print("demo_code", demo_code.__file__)

from maxtext.checkpoint_conversion import demo_code2
print("demo_code2", demo_code2.__file__)

def main():
  print("\n== MAIN ==")
  print("DEBUG: tests/demo_test.py")
  demo_code.main()
  demo_code2.main()

if __name__ == "__main__":
  main()