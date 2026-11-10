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

"""Deprecated backward-compatible re-export shim for benchmarks.maxtext_ctk_runner.

XPK is deprecated. MaxText benchmarks now run on Cluster Toolkit (gcluster);
import from `benchmarks.maxtext_ctk_runner` instead. This module only re-exports
those symbols for existing callers and may be removed in a future release.
"""

import warnings

# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,protected-access
from benchmarks.maxtext_ctk_runner import *  # noqa: F401,F403
from benchmarks.maxtext_ctk_runner import (
    COMPLETION_TIMEOUT_SECONDS,
    ClusterConfig,
    LibTpuType,
    PathwaysConfig,
    WorkloadConfig,
    XpkClusterConfig,
    _DEFAULT_MAXTEXT_BASE_DOCKER_IMAGE_NAME,
    _V5E_V6E_TOPOLOGIES,
    _V5P_TOPOLOGIES,
    _build_args_from_config,
    _combine_flag_strings,
    _get_config_tuning_params,
    _get_pathways_proxy_flags,
    _get_pathways_server_flags,
    _get_pathways_specific_flags,
    _get_pathways_worker_flags,
    build_user_command,
    ctk_benchmark_runner,
    generate_workload_cmd,
    generate_xpk_workload_cmd,
    hardware_id_to_num_chips_per_node,
    main,
    on_device_benchmark_runner,
    run_workload,
    run_xpk_workload,
    wait_for_workload_completion,
    wait_for_workloads_completion_async,
    wait_for_xpk_workload_completion,
    wait_for_xpk_workloads_completion_async,
    xpk_benchmark_runner,
)

warnings.warn(
    "benchmarks.maxtext_xpk_runner is deprecated because XPK is deprecated; "
    "use benchmarks.maxtext_ctk_runner (Cluster Toolkit) instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
  main()
