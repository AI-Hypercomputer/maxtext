
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.machinery
import importlib.metadata
import importlib.util
import os
from itertools import chain
from types import ModuleType
from typing import Any, Optional

from . import (
    BACKENDS_MAPPING,
    IMPORT_STRUCTURE_T,
    Backend,
    DummyObject,
    requires_backends,
)


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: Optional[importlib.machinery.ModuleSpec] = None,
        extra_objects: Optional[dict[str, object]] = None,
        explicit_import_shortcut: Optional[dict[str, list[str]]] = None,
    ):
        super().__init__(name)

        self._object_missing_backend = {}
        self._explicit_import_shortcut = explicit_import_shortcut if explicit_import_shortcut else {}

        if any(isinstance(key, frozenset) for key in import_structure):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []

            _import_structure = {}

            for backends, module in import_structure.items():
                missing_backends = []

                # This ensures that if a module is importable, then all other keys of the module are importable.
                # As an example, in module.keys() we might have the following:
                #
                # dict_keys(['models.nllb_moe.configuration_nllb_moe', 'models.sew_d.configuration_sew_d'])
                #
                # with this, we don't only want to be able to import these explicitly, we want to be able to import
                # every intermediate module as well. Therefore, this is what is returned:
                #
                # {
                #     'models.nllb_moe.configuration_nllb_moe',
                #     'models.sew_d.configuration_sew_d',
                #     'models',
                #     'models.sew_d', 'models.nllb_moe'
                # }

                module_keys = set(
                    chain(*[[k.rsplit(".", i)[0] for i in range(k.count(".") + 1)] for k in list(module.keys())])
                )

                for backend in backends:
                    if backend in BACKENDS_MAPPING:
                        callable, _ = BACKENDS_MAPPING[backend]
                    else:
                        if any(key in backend for key in ["=", "<", ">"]):
                            backend = Backend(backend)
                            callable = backend.is_satisfied
                        else:
                            raise ValueError(
                                f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}"
                            )

                    try:
                        if not callable():
                            missing_backends.append(backend)
                    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError, RuntimeError):
                        missing_backends.append(backend)

                self._modules = self._modules.union(module_keys)

                for key, values in module.items():
                    if missing_backends:
                        self._object_missing_backend[key] = missing_backends

                    for value in values:
                        self._class_to_module[value] = key
                        if missing_backends:
                            self._object_missing_backend[value] = missing_backends
                    _import_structure.setdefault(key, []).extend(values)

                # Needed for autocompletion in an IDE
                self.__all__.extend(module_keys | set(chain(*module.values())))

            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = _import_structure

        # This can be removed once every exportable object has a `require()` require.
        else:
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend:
            missing_backends = self._object_missing_backend[name]

            class Placeholder(metaclass=DummyObject):
                _backends = missing_backends

                def __init__(self, *args, **kwargs):
                    requires_backends(self, missing_backends)

                def call(self, *args, **kwargs):
                    pass

            Placeholder.__name__ = name

            if name not in self._class_to_module:
                module_name = f"transformers.{name}"
            else:
                module_name = self._class_to_module[name]
                if not module_name.startswith("transformers."):
                    module_name = f"transformers.{module_name}"

            Placeholder.__module__ = module_name

            value = Placeholder
        elif name in self._class_to_module:
            try:
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            except (ModuleNotFoundError, RuntimeError) as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                ) from e

        elif name in self._modules:
            try:
                value = self._get_module(name)
            except (ModuleNotFoundError, RuntimeError) as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                ) from e
        else:
            value = None
            for key, values in self._explicit_import_shortcut.items():
                if name in values:
                    value = self._get_module(key)

            if value is None:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))

import importlib
import importlib.metadata
import importlib.util
import logging
from typing import Union

# In MaxText, the logger is often configured globally.
# We assume `logger` is available in the scope, similar to the source file.
# e.g., from absl import logging as logger
# For direct compatibility, we can use:
logger = logging.getLogger(__name__)


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[tuple[bool, str], bool]:
  """Checks if a package is available and optionally returns its version.

  This function avoids importing the package directly to check for its existence,
  first using `importlib.util.find_spec` and then `importlib.metadata.version`.
  It includes fallback mechanisms for specific packages with non-standard
  installation names or versioning schemes.

  Args:
    pkg_name: The name of the package to check.
    return_version: If True, returns a tuple of (bool, version_str). Otherwise,
      returns a bool.

  Returns:
    A boolean indicating if the package is available, or a tuple of
    (availability, version string) if return_version is True.
  """
  # Check if the package spec exists and grab its version to avoid importing a
  # local directory
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      # TODO: Once python 3.9 support is dropped,
      # `importlib.metadata.packages_distributions()`
      # should be used here to map from package name to distribution names
      # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark;
      # onnxruntime -> onnxruntime-gpu.
      # `importlib.metadata.packages_distributions()` is not available in
      # Python 3.9.

      # Primary method to get the package version
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      # Fallback method: Only for "torch" and versions containing "dev"
      if pkg_name == "torch":
        try:
          package = importlib.import_module(pkg_name)
          temp_version = getattr(package, "__version__", "N/A")
          # Check if the version contains "dev"
          if "dev" in temp_version:
            package_version = temp_version
            package_exists = True
          else:
            package_exists = False
        except ImportError:
          # If the package can't be imported, it's not available
          package_exists = False
      elif pkg_name == "quark":
        # TODO: remove once `importlib.metadata.packages_distributions()` is
        # supported.
        try:
          package_version = importlib.metadata.version("amd-quark")
        except Exception:
          package_exists = False
      elif pkg_name == "triton":
        try:
          # import triton works for both linux and windows
          package = importlib.import_module(pkg_name)
          package_version = getattr(package, "__version__", "N/A")
        except Exception:
          try:
            package_version = importlib.metadata.version(
                "pytorch-triton"
            )  # pytorch-triton
          except Exception:
            package_exists = False
      else:
        # For packages other than "torch", don't attempt the fallback and set
        # as not available
        package_exists = False
    logger.debug(f"Detected {pkg_name} version: {package_version}")
  if return_version:
    return package_exists, package_version
  else:
    return package_exists

from typing import Any

# The variable _auto_gptq_available is defined in the same file,
# similar to the PyTorch implementation.
_auto_gptq_available: Any


def is_auto_gptq_available() -> bool:
  """Returns True if auto-gptq is available."""
  return _auto_gptq_available

from packaging import version

# The following constants and variables are assumed to be defined in the same file.
# from maxtext.config import ACCELERATE_MIN_VERSION
# _accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION):
  """Checks if accelerate is available and its version is >= min_version."""
  return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_av_available() -> bool:
  """Check if PyAv is available."""
  return _av_available
def is_bs4_available() -> bool:
  """Returns True if the `bs4` (Beautiful Soup) library is available."""
  return _bs4_available
import importlib.util

# `importlib.metadata.util` doesn't work with `opencv-python-headless`.
_cv2_available = importlib.util.find_spec("cv2") is not None


def is_cv2_available() -> bool:
  """Returns True if the OpenCV library is available."""
  return _cv2_available

import importlib.util


def is_cython_available() -> bool:
  return importlib.util.find_spec("pyximport") is not None
def is_decord_available() -> bool:
  """Returns True if decord is available."""
  return _decord_availabledef is_essentia_available() -> bool:
  """Returns True if essentia is available."""
  return _essentia_availabledef is_faiss_available() -> bool:
  """Returns True if faiss is available."""
  return _faiss_availabledef is_fastapi_available():
  return _fastapi_available
# from src.MaxText.import_utils import _jinja_available

def is_jinja_available() -> bool:
  """Returns True if Jinja2 is available."""
  return _jinja_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

def is_levenshtein_available() -> bool:
  """Returns whether Levenshtein is available."""
  return _levenshtein_available

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_librosa_available() -> bool:
  """Returns True if the librosa library is available."""
  return _librosa_available

import importlib.metadata
import importlib.util

_mistral_common_available: bool = False
if importlib.util.find_spec("mistral_common") is not None:
  try:
    importlib.metadata.version("mistral_common")
    _mistral_common_available = True
  except importlib.metadata.PackageNotFoundError:
    # This can happen if the package is found as a local directory.
    _mistral_common_available = False


def is_mistral_common_available() -> bool:
  """Check if mistral_common is available."""
  return _mistral_common_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

from MaxText.pyconfig import _nltk_available


def is_nltk_available() -> bool:
  """Check if NLTK is available."""
  return _nltk_available
def is_openai_available() -> bool:
  """Returns True if OpenAI is available."""
  return _openai_availabledef is_pandas_available() -> bool:
  """Check if pandas is available."""
  return _pandas_available
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The `_peft_available` flag is assumed to be defined at the module level,
# similar to the original PyTorch implementation.
# e.g.,
# import importlib.util
# _peft_available = importlib.util.find_spec("peft") is not None


def is_peft_available() -> bool:
  """Check if peft is available."""
  return _peft_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

from . import _phonemizer_available


def is_phonemizer_available() -> bool:
  """Checks if phonemizer is available."""
  return _phonemizer_available

# The _pretty_midi_available flag is defined in the same file,
# similar to the PyTorch version.
# It checks for the availability of the 'pretty_midi' package.

def is_pretty_midi_available() -> bool:
  return _pretty_midi_available

import importlib.util


def is_protobuf_available():
  """Check if protobuf is available."""
  if importlib.util.find_spec("google") is None:
    return False
  return importlib.util.find_spec("google.protobuf") is not None

def is_pyctcdecode_available() -> bool:
  """Check if pyctcdecode is available."""
  return _pyctcdecode_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# This function relies on the module-level variable `_pydantic_available`,
# which is set during module initialization by checking for the `pydantic` package.


def is_pydantic_available() -> bool:
  """Checks if the pydantic library is installed and available."""
  return _pydantic_available


# From transformers.utils.import_utils
# This is a framework-agnostic utility function.

def is_pytesseract_available() -> bool:
  """Returns True if pytesseract is available."""
  return _pytesseract_available

from typing import Union

# The following code is an adaptation of HuggingFace's `_is_package_available` utility.
# It is included here to support the dependency checks in the translated functions.
# For the full implementation and further context, refer to the original HuggingFace Transformers library.
def _is_package_available(pkg_name: str) -> bool:
  """
  Checks if a package is available in the environment.
  """
  import importlib.util

  return importlib.util.find_spec(pkg_name) is not None


_pytorch_quantization_available = _is_package_available("pytorch_quantization")


def is_pytorch_quantization_available() -> bool:
  """Checks if pytorch_quantization is available."""
  return _pytorch_quantization_available
def is_rich_available() -> bool:
  """Check if the 'rich' library is available."""
  return _rich_available
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
from maxtext.utils.import_utils import _rjieba_available


def is_rjieba_available() -> bool:
  return _rjieba_available

import importlib.metadata
import importlib.util

_sacremoses_available: bool
if importlib.util.find_spec("sacremoses") is not None:
  try:
    importlib.metadata.version("sacremoses")
    _sacremoses_available = True
  except importlib.metadata.PackageNotFoundError:
    _sacremoses_available = False
else:
  _sacremoses_available = False


def is_sacremoses_available() -> bool:
  """Returns True if sacremoses is available."""
  return _sacremoses_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# This file is an almost-direct-copy of https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
# The following modifications were made:
# 1. The file is formatted with the internal formatter.
# 2. Unused imports are removed.
# 3. All PyTorch-specific availability checks are removed.
# 4. The logic of `_is_package_available` is simplified to just use `importlib.util.find_spec`.
# 5. Type annotations are added.


def is_scipy_available() -> bool:
  """Returns True if the scipy library is available."""
  return _scipy_available

import importlib.util
import importlib.metadata

# The following code is a JAX-compatible equivalent of the functionality
# used to initialize the `_sentencepiece_available` variable in the original file.
_sentencepiece_available: bool = False
if importlib.util.find_spec("sentencepiece") is not None:
  try:
    importlib.metadata.version("sentencepiece")
    _sentencepiece_available = True
  except importlib.metadata.PackageNotFoundError:
    # A spec may exist for a local directory, but the package is not installed.
    _sentencepiece_available = False


def is_sentencepiece_available() -> bool:
  """Returns True if SentencePiece is available."""
  return _sentencepiece_available

def is_sklearn_available():
  """Returns True if scikit-learn is available."""
  return _sklearn_available

# The variable `_timm_available` is assumed to be defined at the module level.

def is_timm_available() -> bool:
  """Check if timm is available."""
  return _timm_available
def is_tokenizers_available() -> bool:
  """Checks if the 'tokenizers' library is available."""
  return _tokenizers_availabledef is_torch_available() -> bool:
  """Returns True if PyTorch is available."""
  return _torch_available
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

from maxtext.utils.import_utils import _torchaudio_available


def is_torchaudio_available() -> bool:
  """Returns True if torchaudio is available."""
  return _torchaudio_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# This function is framework-agnostic and used as-is.
# It depends on `_torchcodec_available` being defined in the same file,
# which would be translated as:
# _torchcodec_available = importlib.util.find_spec("torchcodec") is not None
def is_torchcodec_available() -> bool:
  """Check if torchcodec is available."""
  return _torchcodec_available

import importlib.util
import importlib.metadata

# The following variable is defined at the module level in the source file.
# It checks for the availability of the torchvision package.
try:
  _torchvision_available = importlib.util.find_spec("torchvision") is not None
  if _torchvision_available:
    # Check if we can actually get a version, to avoid local directories
    # named 'torchvision'
    importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
  _torchvision_available = False


def is_torchvision_available() -> bool:
  """Returns True if torchvision is available."""
  return _torchvision_available
def is_uroman_available() -> bool:
  """Returns True if the uroman package is available."""
  # The _uroman_available flag is set at module-level.
  return _uroman_availabledef is_uvicorn_available() -> bool:
  """Checks if uvicorn is available."""
  return _uvicorn_available
import importlib.metadata
import importlib.util
from functools import lru_cache

from absl import logging


@lru_cache
def is_vision_available() -> bool:
  """Checks if the PIL library is available."""
  _pil_available = importlib.util.find_spec("PIL") is not None
  if _pil_available:
    try:
      package_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
      try:
        package_version = importlib.metadata.version("Pillow-SIMD")
      except importlib.metadata.PackageNotFoundError:
        return False
    logging.debug(f"Detected PIL version {package_version}")
  return _pil_available
import importlib.util

_yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None


def is_yt_dlp_available() -> bool:
  """Check if yt-dlp is available."""
  return _yt_dlp_available
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_speech_available() -> bool:
  """Checks if a speech processing library is available."""
  # For now this depends on torchaudio but the exact dependency might evolve in the future.
  return _torchaudio_available

from collections import OrderedDict

# The is_*_available functions and *_IMPORT_ERROR constants are assumed to be
# defined in the same file, as in the original source.
from . import (
    ACCELERATE_IMPORT_ERROR,
    AV_IMPORT_ERROR,
    BS4_IMPORT_ERROR,
    CCL_IMPORT_ERROR,
    CV2_IMPORT_ERROR,
    CYTHON_IMPORT_ERROR,
    DATASETS_IMPORT_ERROR,
    DECORD_IMPORT_ERROR,
    DETECTRON2_IMPORT_ERROR,
    ESSENTIA_IMPORT_ERROR,
    FAISS_IMPORT_ERROR,
    FASTAPI_IMPORT_ERROR,
    FLAX_IMPORT_ERROR,
    FTFY_IMPORT_ERROR,
    G2P_EN_IMPORT_ERROR,
    JIEBA_IMPORT_ERROR,
    JINJA_IMPORT_ERROR,
    KERAS_NLP_IMPORT_ERROR,
    LEVENSHTEIN_IMPORT_ERROR,
    LIBROSA_IMPORT_ERROR,
    MISTRAL_COMMON_IMPORT_ERROR,
    NATTEN_IMPORT_ERROR,
    NLTK_IMPORT_ERROR,
    OPENAI_IMPORT_ERROR,
    PANDAS_IMPORT_ERROR,
    PEFT_IMPORT_ERROR,
    PHONEMIZER_IMPORT_ERROR,
    PRETTY_MIDI_IMPORT_ERROR,
    PROTOBUF_IMPORT_ERROR,
    PYCTCDECODE_IMPORT_ERROR,
    PYDANTIC_IMPORT_ERROR,
    PYTESSERACT_IMPORT_ERROR,
    PYTORCH_IMPORT_ERROR,
    PYTORCH_QUANTIZATION_IMPORT_ERROR,
    RICH_IMPORT_ERROR,
    SACREMOSES_IMPORT_ERROR,
    SCIPY_IMPORT_ERROR,
    SENTENCEPIECE_IMPORT_ERROR,
    SKLEARN_IMPORT_ERROR,
    SPEECH_IMPORT_ERROR,
    TENSORFLOW_IMPORT_ERROR,
    TENSORFLOW_PROBABILITY_IMPORT_ERROR,
    TENSORFLOW_TEXT_IMPORT_ERROR,
    TIMM_IMPORT_ERROR,
    TOKENIZERS_IMPORT_ERROR,
    TORCHAUDIO_IMPORT_ERROR,
    TORCHCODEC_IMPORT_ERROR,
    TORCHVISION_IMPORT_ERROR,
    UROMAN_IMPORT_ERROR,
    UVICORN_IMPORT_ERROR,
    VISION_IMPORT_ERROR,
    YT_DLP_IMPORT_ERROR,
    is_accelerate_available,
    is_av_available,
    is_bs4_available,
    is_ccl_available,
    is_cv2_available,
    is_cython_available,
    is_datasets_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_faiss_available,
    is_fastapi_available,
    is_flax_available,
    is_ftfy_available,
    is_g2p_en_available,
    is_jieba_available,
    is_jinja_available,
    is_keras_nlp_available,
    is_levenshtein_available,
    is_librosa_available,
    is_mistral_common_available,
    is_natten_available,
    is_nltk_available,
    is_openai_available,
    is_pandas_available,
    is_peft_available,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_protobuf_available,
    is_pyctcdecode_available,
    is_pydantic_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_rich_available,
    is_sacremoses_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_speech_available,
    is_tf_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_timm_available,
    is_tokenizers_available,
    is_torchaudio_available,
    is_torch_available,
    is_torchcodec_available,
    is_torchvision_available,
    is_uroman_available,
    is_uvicorn_available,
    is_vision_available,
    is_yt_dlp_available,
)


BACKENDS_MAPPING = OrderedDict(
    [
        ("av", (is_av_available, AV_IMPORT_ERROR)),
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("uroman", (is_uroman_available, UROMAN_IMPORT_ERROR)),
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        (
            "pytorch_quantization",
            (
                is_pytorch_quantization_available,
                PYTORCH_QUANTIZATION_IMPORT_ERROR,
            ),
        ),
        (
            "sentencepiece",
            (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR),
        ),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        (
            "tensorflow_probability",
            (
                is_tensorflow_probability_available,
                TENSORFLOW_PROBABILITY_IMPORT_ERROR,
            ),
        ),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        (
            "tensorflow_text",
            (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR),
        ),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("torchcodec", (is_torchcodec_available, TORCHCODEC_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
        ("yt_dlp", (is_yt_dlp_available, YT_DLP_IMPORT_ERROR)),
        ("rich", (is_rich_available, RICH_IMPORT_ERROR)),
        ("keras_nlp", (is_keras_nlp_available, KERAS_NLP_IMPORT_ERROR)),
        ("pydantic", (is_pydantic_available, PYDANTIC_IMPORT_ERROR)),
        ("fastapi", (is_fastapi_available, FASTAPI_IMPORT_ERROR)),
        ("uvicorn", (is_uvicorn_available, UVICORN_IMPORT_ERROR)),
        ("openai", (is_openai_available, OPENAI_IMPORT_ERROR)),
        (
            "mistral-common",
            (is_mistral_common_available, MISTRAL_COMMON_IMPORT_ERROR),
        ),
    ]
)

# Copyright 2024 The MaxText Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.metadata
import operator
import re
from enum import Enum
from functools import lru_cache
from packaging import version

# The following BACKENDS_MAPPING is assumed to be defined elsewhere in the project
# and available in the scope where this Backend class is used.
# from . import BACKENDS_MAPPING


class VersionComparison(Enum):
    """Enum for version comparison operators."""

    EQUAL = operator.eq
    NOT_EQUAL = operator.ne
    GREATER_THAN = operator.gt
    LESS_THAN = operator.lt
    GREATER_THAN_OR_EQUAL = operator.ge
    LESS_THAN_OR_EQUAL = operator.le

    @staticmethod
    def from_string(version_string: str) -> "VersionComparison":
        """Converts a string to a VersionComparison operator."""
        string_to_operator = {
            "=": VersionComparison.EQUAL.value,
            "==": VersionComparison.EQUAL.value,
            "!=": VersionComparison.NOT_EQUAL.value,
            ">": VersionComparison.GREATER_THAN.value,
            "<": VersionComparison.LESS_THAN.value,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL.value,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL.value,
        }

        return string_to_operator[version_string]


@lru_cache
def split_package_version(package_version_str: str) -> tuple[str, str, str]:
    """Splits a package version string into package name, comparison, and version."""
    pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
    match = re.match(pattern, package_version_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        raise ValueError(f"Invalid package version string: {package_version_str}")


class Backend:
    """Represents a backend requirement with package name and version constraints."""

    def __init__(self, backend_requirement: str):
        self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

        # This check assumes BACKENDS_MAPPING is defined in the scope of use.
        # from . import BACKENDS_MAPPING
        # if self.package_name not in BACKENDS_MAPPING:
        #     raise ValueError(
        #         f"Backends should be defined in the BACKENDS_MAPPING. Offending backend: {self.package_name}"
        #     )

    def is_satisfied(self) -> bool:
        """Checks if the backend requirement is satisfied in the current environment."""
        return VersionComparison.from_string(self.version_comparison)(
            version.parse(importlib.metadata.version(self.package_name)), version.parse(self.version)
        )

    def __repr__(self) -> str:
        """Returns a string representation of the Backend object."""
        return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

    @property
    def error_message(self):
        """Generates an error message for when the backend requirement is not met."""
        return (
            f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
            f" library was not found with this version in your environment."
        )

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
from typing import Callable, Tuple

from ._import_structure import BACKENDS_MAPPING, Backend


def requires(*, backends: Tuple[str, ...] = ()) -> Callable:
  """
  This decorator enables two things:
  - Attaching a `__backends` tuple to an object to see what are the necessary backends for it
    to execute correctly without instantiating it
  - The '@requires' string is used to dynamically import objects
  """

  if not isinstance(backends, tuple):
    raise TypeError("Backends should be a tuple.")

  applied_backends = []
  for backend in backends:
    if backend in BACKENDS_MAPPING:
      applied_backends.append(backend)
    else:
      if any(key in backend for key in ["=", "<", ">"]):
        applied_backends.append(Backend(backend))
      else:
        raise ValueError(f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}")

  def inner_fn(fun: Callable) -> Callable:
    fun.__backends = applied_backends
    return fun

  return inner_fn
