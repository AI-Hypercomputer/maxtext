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

__author__ = "Google LLC"
__version__ = "2025.07.24"
__description__ = (
    "MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and "
    "targeting Google Cloud TPUs and GPUs for training and **inference."
)


# maxtext/__init__.py

from collections.abc import Sequence

from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText import dpo_utils
import jax
from jax.sharding import Mesh
from MaxText.train_utils import from_config

Transformer = models.Transformer

