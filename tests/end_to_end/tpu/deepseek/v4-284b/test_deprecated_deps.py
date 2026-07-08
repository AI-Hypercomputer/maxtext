import sys
import os

sys.path.insert(0, os.getcwd() + "/src")
sys.path.insert(0, os.getcwd())

print("Importing collections, typing, math, datetime...")
sys.stdout.flush()
from collections import OrderedDict
from typing import Any
from math import prod
import math
import datetime
print("Done!")

print("Importing jax...")
sys.stdout.flush()
import jax
from jax.experimental.compilation_cache import compilation_cache
from jax.tree_util import register_pytree_node_class
print("Done!")

print("Importing omegaconf...")
sys.stdout.flush()
import omegaconf
print("Done!")

print("Importing maxtext.utils.accelerator_to_spec_map...")
sys.stdout.flush()
from maxtext.utils import accelerator_to_spec_map
print("Done!")

print("Importing maxtext.utils.globals...")
sys.stdout.flush()
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_REPO_ROOT, MAXTEXT_PKG_DIR
print("Done!")

print("Importing maxtext.common.common_types...")
sys.stdout.flush()
from maxtext.common.common_types import AttentionType, DecoderBlockType, ReorderStrategy, ShardMode
print("Done!")

print("Importing maxtext.utils.gcs_utils...")
sys.stdout.flush()
from maxtext.utils import gcs_utils
print("Done!")

print("Importing maxtext.utils.max_logging...")
sys.stdout.flush()
from maxtext.utils import max_logging
print("Done!")

print("Importing maxtext.utils.max_utils...")
sys.stdout.flush()
from maxtext.utils import max_utils
print("Done!")

print("All dependencies of pyconfig_deprecated imported successfully!")
sys.stdout.flush()
