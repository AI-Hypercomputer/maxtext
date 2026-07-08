import sys
import os

sys.path.insert(0, os.getcwd() + "/src")
sys.path.insert(0, os.getcwd())

print("Importing maxtext.utils.max_utils...")
sys.stdout.flush()
from maxtext.utils import max_utils
print("Done!")
sys.stdout.flush()
