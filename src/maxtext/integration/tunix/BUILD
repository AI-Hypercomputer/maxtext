# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//devtools/python/blaze:pytype.bzl",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = ["//third_party/py/maxtext:license"],
    default_visibility = ["//third_party/py/maxtext:__subpackages__"],
)

pytype_strict_library(
    name = "weight_mapping",
    srcs = [
        "weight_mapping/__init__.py",
        "weight_mapping/deepseek3.py",
        "weight_mapping/gpt_oss.py",
        "weight_mapping/llama3.py",
        "weight_mapping/qwen2.py",
        "weight_mapping/qwen3.py",
    ],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":weight_mapping",
        "//third_party/py/maxtext:checkpoint_conversion_utils_param_mapping",
    ],
)

pytype_strict_library(
    name = "tunix_adapter",
    srcs = ["tunix_adapter.py"],
    deps = [
        ":utils",
        "//third_party/py/flax/nnx",
        "//third_party/py/jax",
        "//third_party/py/maxtext:checkpoint_conversion_utils_hf_model_configs",
        "//third_party/py/maxtext:layers",
    ],
)
