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

A core of the quantization algorithm is replacing float matmul:

r = matmul(a, b)

with a following equation:

sa = calibrate(a)
sb = calibrate(b)
sa1 = 1 / sa
sb1 = 1 / sb

def q(x):
  return round(clip(x, -clip_bound, clip_bound))

qa = q(sa * a)  # each row can be scaled separately
qb = q(b * sb)  # each column can be scaled separately
qr = int_matmul(qa, qb)
r = sa1 * qr * sb1

We can observe that if 'q' function was identity, both computations
would be equivalent. A similar approach applies to most linear operations.

Calibration is a major difficulty, i.e. how to find best scales sa, sb.
This is solved (or rather tackled) by Stats.bound() and StatsConfig.

Another difficulty is feedback loops between Stats.bound() calculation and
the changing distribution of inputs.
This can be tackled by setting freeze_scale_at_begin=True

Another difficulty is 'sparse' distributions.
E.g. quantization of the output of Relu and other similar functions.
This can be tackled with filter_zeros=True setting.

Typical quantization uses mean-of-batch-max calibration.
It is problematic as that statistics is dependant on batch size and can't
take example weights into account.
lp_dev with large lp_order is a suitable replacement.

Typical matmuls have weights that can have per-output-channel scales.
In the above equation 'b' would be weights and 'sb' diagonal matrix.
While activations usually share distribution across examples in batch and
may have 'sa' as a scalar. This can be configured using 'share_stats_axes'.

Note that share_stats_axes should always include contraction axis.
For instance in case of MatMul, share_stats_axes in lhs should include '1', and
share_stats_axes in rhs should include '0'.

There are several reasons to control schedule of quantization:

  - Avoid quantization of initial turbulent period of training to
    accelerate convergence.
  - Separately quantize weights and activations in various layers.
  - One also may wish to incrementally increase quantization.
  - Disable quantization just before the training is finished.
  - Try multiple configurations in one long 'anytime' training.

All and any of the above can be achieved by providing multiple
lhs_configs and rhs_configs with begin_at_event and end_at_event set properly.
The intervals [begin_at_event, end_at_event) are expected to be disjoint.
"""

from aqt.tensorflow import aqt_conv2d
from aqt.tensorflow import aqt_einsum
from aqt.tensorflow import aqt_matmul

# TODO(b/193796715): Consider using Union[tf.Tensor, tf.Variable].
# TODO(vladf): aqt_ops.aqt_matmul is redundant, we should consider
# renaming to just the short operation name, e.g., aqt_ops.matmul.

aqt_depthwise_conv2d = aqt_conv2d.depthwise_conv2d
aqt_conv2d = aqt_conv2d.conv2d
aqt_einsum = aqt_einsum.einsum
aqt_matmul = aqt_matmul.matmul
