# Copyright 2026 Google LLC
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

# pylint: disable=unused-import,unused-wildcard-import,wildcard-import
"""Canonical Gated Delta Net (GDN) backward pass facade re-exporting from .gdn_bwd."""

try:
  from maxtext.models.kernels.gdn import gdn_bwd
  from maxtext.models.kernels.gdn.gdn_bwd import *  # pylint: disable=wildcard-import
  from maxtext.models.kernels.gdn.gdn_bwd import __all__ as _gdn_bwd_all
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import gdn_bwd
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd import *  # pylint: disable=wildcard-import
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd import __all__ as _gdn_bwd_all
  except (ImportError, ModuleNotFoundError):
    from . import gdn_bwd
    from .gdn_bwd import *  # pylint: disable=wildcard-import
    from .gdn_bwd import __all__ as _gdn_bwd_all

__all__ = list(_gdn_bwd_all)
