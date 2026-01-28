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
# pylint: skip-file

# Shim for Implementation of Sparse Flash Attention, a.k.a. "Splash" attention.

import sys
import importlib
from MaxText import max_logging

OLD_MODULE_PATH = "MaxText.kernels.splash_attention_kernel"
NEW_MODULE_PATH = "maxtext.kernels.splash_attention_kernel"

try:
  _new_module = importlib.import_module(NEW_MODULE_PATH)
  max_logging.warning(f"'{OLD_MODULE_PATH}' is deprecated; use '{NEW_MODULE_PATH}' instead.\n")
  sys.modules[OLD_MODULE_PATH] = _new_module

except ImportError as e:
  max_logging.error(f"Shim could not find target module: '{NEW_MODULE_PATH}'\n")
  raise e
