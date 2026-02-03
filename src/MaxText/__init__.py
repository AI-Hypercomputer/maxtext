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
MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and targeting Google Cloud
TPUs and GPUs for training and inference. MaxText achieves high MFUs and scales from single host to very large clusters
while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.
"""

__author__ = "Google LLC"
__version__ = "0.1.1"
__description__ = (
    "MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and "
    "targeting Google Cloud TPUs and GPUs for training and **inference."
)

from collections.abc import Sequence

from jax.sharding import Mesh

from MaxText import pyconfig
from MaxText.layers import models
from maxtext.trainers.post_train.dpo import dpo_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils

Transformer = models.Transformer
transformer_as_linen = models.transformer_as_linen
