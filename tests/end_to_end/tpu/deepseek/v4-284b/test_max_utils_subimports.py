import sys
import os

print("1. Importing basic libraries...")
sys.stdout.flush()
import collections
from collections.abc import Sequence
import functools
from functools import partial
import socket
import re
import subprocess
import time
from typing import Any
from packaging.version import Version
from pathlib import Path
from contextlib import contextmanager
print("1. Done!")
sys.stdout.flush()

print("2. Importing etils.epath...")
sys.stdout.flush()
from etils import epath
print("2. Done!")
sys.stdout.flush()

print("3. Importing flax...")
sys.stdout.flush()
import flax
print("3. Done!")
sys.stdout.flush()

print("4. Importing jax...")
sys.stdout.flush()
import jax
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
print("4. Done!")
sys.stdout.flush()

print("5. Importing numpy...")
sys.stdout.flush()
import numpy as np
print("5. Done!")
sys.stdout.flush()

print("6. Importing orbax.checkpoint...")
sys.stdout.flush()
import orbax.checkpoint as ocp
print("6. Done!")
sys.stdout.flush()

print("7. Importing orbax initialization...")
sys.stdout.flush()
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization
print("7. Done!")
sys.stdout.flush()

print("8. Importing psutil...")
sys.stdout.flush()
import psutil
print("8. Done!")
sys.stdout.flush()

print("9. Importing elastic_utils...")
sys.stdout.flush()
sys.path.insert(0, os.getcwd() + "/src")
sys.path.insert(0, os.getcwd())
from maxtext.utils import elastic_utils
print("9. Done!")
sys.stdout.flush()

print("All sub-imports of max_utils.py completed successfully!")
sys.stdout.flush()
