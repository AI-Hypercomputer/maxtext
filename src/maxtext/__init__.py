# Copyright 2023–2026 Google LLC
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

# pylint: disable=undefined-all-variable, import-outside-toplevel

from typing import TYPE_CHECKING

from maxtext.version import __author__
from maxtext.version import __description__
from maxtext.version import __version__

# Static analysis tools (such as Pylint and Pytype) statically parse ASTs without
# executing PEP 562 __getattr__. We declare the lazy exports inside `if TYPE_CHECKING:`
# so linters and IDEs resolve exports without eagerly importing heavy dependencies at runtime.
if TYPE_CHECKING:
  from collections.abc import Sequence
  from jax.sharding import Mesh
  from maxtext.configs import pyconfig
  from maxtext.configs.types import MaxTextConfig
  from maxtext.models import models
  from maxtext.models.models import Transformer, transformer_as_linen
  from maxtext.utils import maxtext_utils, model_creation_utils
  from maxtext.utils.model_creation_utils import from_config, from_pretrained

__all__ = [
    "__author__",
    "__description__",
    "__version__",
    "Sequence",
    "Mesh",
    "pyconfig",
    "MaxTextConfig",
    "models",
    "Transformer",
    "transformer_as_linen",
    "maxtext_utils",
    "model_creation_utils",
    "from_config",
    "from_pretrained",
]


def __dir__():
  return __all__


import os
# In order to have any effect on the C++ logging this has to be set before we import anything from jax.
# When jax is imported, its `__init__.py` calls `cloud_tpu_init()`, which also initializes the C++ logger.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")
del os


def __getattr__(name: str):
  # Lazy-load exports to avoid eagerly pulling in heavy transitive dependencies
  # (such as jax or omegaconf) when importing lightweight submodules or running
  # in minimal launcher environments (e.g. XManager CLI scripts).
  module_dict = globals()
  match name:
    case "Sequence":
      from collections.abc import Sequence  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["Sequence"] = Sequence
      return Sequence
    case "Mesh":
      from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["Mesh"] = Mesh
      return Mesh
    case "pyconfig":
      from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["pyconfig"] = pyconfig
      return pyconfig
    case "MaxTextConfig":
      from maxtext.configs.types import MaxTextConfig  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["MaxTextConfig"] = MaxTextConfig
      return MaxTextConfig
    case "models" | "Transformer" | "transformer_as_linen":
      from maxtext.models import models  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["models"] = models
      module_dict["Transformer"] = models.Transformer
      module_dict["transformer_as_linen"] = models.transformer_as_linen
      return module_dict[name]
    case "maxtext_utils":
      from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["maxtext_utils"] = maxtext_utils
      return maxtext_utils
    case "from_config" | "from_pretrained" | "model_creation_utils":
      from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel, g-import-not-at-top

      module_dict["model_creation_utils"] = model_creation_utils
      module_dict["from_config"] = model_creation_utils.from_config
      module_dict["from_pretrained"] = model_creation_utils.from_pretrained
      return module_dict[name]
    case _:
      raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
