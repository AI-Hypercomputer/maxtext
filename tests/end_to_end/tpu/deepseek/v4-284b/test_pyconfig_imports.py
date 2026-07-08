import sys
import os

# Ensure maxtext is in path
sys.path.insert(0, os.getcwd() + "/src")
sys.path.insert(0, os.getcwd())

print("Importing omegaconf...")
sys.stdout.flush()
import omegaconf
print("Done!")

print("Importing pyconfig_deprecated...")
sys.stdout.flush()
from maxtext.configs import pyconfig_deprecated
print("Done!")

print("Importing utils.globals...")
sys.stdout.flush()
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR, MAXTEXT_ASSETS_ROOT, HF_IDS, MAXTEXT_PKG_DIR
print("Done!")

print("Importing configs.types...")
sys.stdout.flush()
from maxtext.configs import types
print("Done!")

print("Importing utils.max_utils...")
sys.stdout.flush()
from maxtext.utils import max_utils
print("Done!")

print("Importing utils.max_logging...")
sys.stdout.flush()
from maxtext.utils import max_logging
print("Done!")

print("All pyconfig dependency imports succeeded!")
sys.stdout.flush()
