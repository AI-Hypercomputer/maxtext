# Copyright 2026 Google LLC
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

"""Throwaway: verifies the TPU7X gate runs a PR's tpu_only test and excludes skip_on_tpu7x ones."""

import jax
import pytest


@pytest.mark.tpu_only
def test_runnable_on_tpu7x():
  assert jax.devices()[0].platform == "tpu"


@pytest.mark.tpu_only
@pytest.mark.skip_on_tpu7x
def test_excluded_on_tpu7x():
  assert True
