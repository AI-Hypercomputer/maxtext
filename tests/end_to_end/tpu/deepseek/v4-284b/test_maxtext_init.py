import sys
import os

os.environ["DECOUPLE_GCLOUD"] = "TRUE"

print("0. Importing torch...")

sys.stdout.flush()
import torch
print("0. Done!")
sys.stdout.flush()

print("0.1. Importing tensorflow...")
sys.stdout.flush()
import tensorflow as tf
print("0.1. Done!")
sys.stdout.flush()

print("0.2. Importing jax...")
sys.stdout.flush()
import jax
print("0.2. Done!")
sys.stdout.flush()

sys.path.insert(0, os.getcwd() + "/src")


sys.path.insert(0, os.getcwd())

print("1. Importing jax.sharding Mesh...")
sys.stdout.flush()
from jax.sharding import Mesh
print("1. Done!")
sys.stdout.flush()

print("2. Importing maxtext.configs.pyconfig...")
sys.stdout.flush()
from maxtext.configs import pyconfig
print("2. Done!")
sys.stdout.flush()

print("3. Importing maxtext.models.models...")
sys.stdout.flush()
from maxtext.models import models
print("3. Done!")
sys.stdout.flush()

print("4. Importing maxtext.utils.maxtext_utils...")
sys.stdout.flush()
from maxtext.utils import maxtext_utils
print("4. Done!")
sys.stdout.flush()

print("5. Importing maxtext.utils.model_creation_utils...")
sys.stdout.flush()
from maxtext.utils import model_creation_utils
print("5. Done!")
sys.stdout.flush()

print("All package-level init imports completed successfully!")
sys.stdout.flush()
