import sys

print("1. Importing jax...")
sys.stdout.flush()
import jax
print("1. Done!")
sys.stdout.flush()

print("2. Importing pathwaysutils.elastic.manager...")
sys.stdout.flush()
from pathwaysutils.elastic import manager
print("2. Done!")
sys.stdout.flush()

print("All imports in JAX -> Pathways manager order succeeded!")

sys.stdout.flush()
