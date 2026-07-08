import os
import sys

print("--- REMOTE ENVIRONMENT VARIABLES ---")
for k, v in sorted(os.environ.items()):
  print(f"{k}={v}")
print("------------------------------------")
sys.stdout.flush()
