#!/usr/bin/python3

"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

'''This script is used to measure the performance of different sharding schemes on TPU.'''

# https://source.corp.google.com/piper///depot/google3/third_party/py/praxis/layers/pipeline.py

from absl import app
from absl import flags
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.compilation_cache import compilation_cache as cc
from jax._src.pjit import with_sharding_constraint

import argparse
import datetime
import numpy as np
import os
from typing import Sequence

NUM_STAGES = 4
LENGTH = 8

def single_block(input, z):
  return input * z

def main(_argv: Sequence[str]) -> None:
  multi_array = jax.numpy.zeros( (NUM_STAGES, LENGTH) )
  multi_array = multi_array.at[0, :].set(jax.numpy.arange(LENGTH))
  x = 1 + jax.numpy.arange( (NUM_STAGES))
  multiblock = jax.vmap(single_block)
  print(multi_array)
  for i in range(NUM_STAGES):
    multi_array = multiblock(multi_array, x)
    multi_array = jax.numpy.roll(multi_array, shift = 1, axis = 0)
    print(multi_array)

if __name__ == "__main__":
  app.run(main)
