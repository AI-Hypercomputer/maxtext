# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test initialization
"""
import os

# Configure JAX for parallel execution under pytest-xdist (Isolated TPU mode)
if "PYTEST_XDIST_WORKER" in os.environ:
  worker = os.environ["PYTEST_XDIST_WORKER"]
  try:
    worker_id = int(worker.replace("gw", ""))
    # Map worker to one of the 4 TPU chips
    tpu_chip = worker_id % 4
    os.environ["TPU_VISIBLE_DEVICES"] = str(tpu_chip)
    # Bypass libtpu lockfile check to allow concurrency
    os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "true"
    # Ensure workers use TPU, overriding master's CPU setting if inherited
    if "JAX_PLATFORMS" in os.environ:
        del os.environ["JAX_PLATFORMS"]
  except ValueError:
    pass

  import pathwaysutils
  pathwaysutils.initialize()
else:
  # Master process: Force CPU to avoid grabbing TPU resources during collection
  os.environ["JAX_PLATFORMS"] = "cpu"
