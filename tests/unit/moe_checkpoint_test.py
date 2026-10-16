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

"""Tests for optional TransformerEngine MoE checkpoint arguments."""

# These tests directly exercise the private checkpoint compatibility helper.
# pylint: disable=protected-access

from maxtext.layers import moe


def test_te_moe_checkpoint_kwargs_for_supported_signature():
  def te_moe(
      *,
      wi_0_checkpoint_name=None,
      wi_1_checkpoint_name=None,
      wo_checkpoint_name=None,
      dispatch_checkpoint_name=None,
      combine_checkpoint_name=None,
  ):
    del wi_0_checkpoint_name, wi_1_checkpoint_name, wo_checkpoint_name
    del dispatch_checkpoint_name, combine_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
      "wo_checkpoint_name": "moe_mlpwo",
      "dispatch_checkpoint_name": "te_moe_dispatch_remat_location",
  }


def test_te_moe_checkpoint_kwargs_for_previous_signature():
  def te_moe(*, wi_0_checkpoint_name=None, wi_1_checkpoint_name=None, wo_checkpoint_name=None):
    del wi_0_checkpoint_name, wi_1_checkpoint_name, wo_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
      "wo_checkpoint_name": "moe_mlpwo",
  }


def test_te_moe_checkpoint_kwargs_for_partial_signature():
  def older_te_moe(*, wi_0_checkpoint_name=None, wi_1_checkpoint_name=None):
    del wi_0_checkpoint_name, wi_1_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(older_te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
  }


def test_te_moe_checkpoint_kwargs_for_ep_only_signature():
  def te_moe(*, dispatch_checkpoint_name=None):
    del dispatch_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "dispatch_checkpoint_name": "te_moe_dispatch_remat_location",
  }


def test_te_moe_checkpoint_kwargs_for_legacy_signature():
  def te_moe():
    pass

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {}
