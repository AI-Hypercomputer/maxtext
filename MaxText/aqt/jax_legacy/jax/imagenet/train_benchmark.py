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

"""Benchmark for the ImageNet example."""
import tempfile
import time

from absl import flags
from absl.testing import absltest
from absl.testing.flagsaver import flagsaver
from aqt.jax_legacy.jax.imagenet import train
from aqt.jax_legacy.jax.imagenet.configs.paper import resnet50_bfloat16
from flax.testing import Benchmark
import jax
import numpy as np


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


FLAGS = flags.FLAGS


class ImagenetBenchmark(Benchmark):
  """Benchmarks for the ImageNet Flax example."""

  @flagsaver
  def test_8x_v100_half_precision(self):
    """Run ImageNet on 8x V100 GPUs in half precision for 2 epochs."""
    model_dir = tempfile.mkdtemp()

    config = resnet50_bfloat16.get_config()
    FLAGS.hparams_config_dict = config
    FLAGS.batch_size = 2048
    FLAGS.half_precision = True
    FLAGS.num_epochs = 2
    FLAGS.model_dir = model_dir

    start_time = time.time()
    train.main([])
    benchmark_time = time.time() - start_time
    summaries = self.read_summaries(model_dir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_accuracy = zip(*summaries['eval_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_accuracy = eval_accuracy[-1]

    # Assertions are deferred until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertBetween(sec_per_epoch, 210, 240)
    self.assertBetween(end_accuracy, 0.06, 0.09)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({'sec_per_epoch': sec_per_epoch,
                         'accuracy': end_accuracy})
    self.report_extras({
        'description':
            'Toy 8 x V100 test for ImageNet ResNet50.',
        'model_name':
            'resnet50'
    })


if __name__ == '__main__':
  absltest.main()
