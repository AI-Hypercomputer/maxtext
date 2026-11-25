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

"""Deprecated backward-compatible re-export shim for benchmarks.ctk_configs.

XPK is deprecated. MaxText benchmarks now run on Cluster Toolkit (gcluster);
import from `benchmarks.ctk_configs` instead. This module only re-exports
those symbols for existing callers and may be removed in a future release.
"""

import warnings

# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from benchmarks.ctk_configs import *  # noqa: F401,F403
from benchmarks.ctk_configs import ClusterConfig, XpkClusterConfig

warnings.warn(
    "benchmarks.xpk_configs is deprecated because XPK is deprecated; "
    "use benchmarks.ctk_configs (Cluster Toolkit) instead.",
    DeprecationWarning,
    stacklevel=2,
)
