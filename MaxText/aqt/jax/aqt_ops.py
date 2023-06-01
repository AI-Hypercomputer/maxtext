# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Accurate Quantized Training ops.

Refer to //knowledge/cerebra/catalyst/tensorflow/aqt/aqt_ops.py for a core of
the quantization algorithm in AQTp.
"""
# TODO(jihwanlee): Create a shared doc, referred to by both TF and Jax.


from aqt.jax import aqt_conv_general
from aqt.jax import aqt_dot_general
from aqt.jax import aqt_matmul


aqt_conv_general_dilated = aqt_conv_general.conv_general_dilated
aqt_dot = aqt_dot_general.dot
aqt_dot_general = aqt_dot_general.dot_general
aqt_matmul = aqt_matmul.matmul
