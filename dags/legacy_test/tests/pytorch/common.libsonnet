// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local common = import '../common.libsonnet';
local metrics = import 'templates/metrics.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  local PyTorchBaseTest = common.CloudAcceleratorTest {
    configMaps+: ['pytorch-nfs-ip'],

    image: 'us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla',

    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          exclude_tags: [
            'LearningRate',
          ],
          merge_runs: true,
          aggregateAssertionsMap+:: {
            ExecuteTime__Percentile_99_sec: {
              FINAL: {
                std_devs_from_mean: {
                  comparison: 'LESS',
                  std_devs: 5.0,
                },
                inclusive_bounds: false,
                wait_for_n_data_points: 10,
              },
            },
            aten_ops_sum: {
              FINAL: {
                inclusive_bounds: true,
                std_devs_from_mean: {
                  comparison: 'LESS',
                  std_devs: 0.0,
                },
                wait_for_n_data_points: 0,
              },
            },
          },
        },
      },
    },
  },
  PyTorchTest:: PyTorchBaseTest {
    local config = self,

    entrypoint: [
      'bash',
      '-cxue',
      |||
        if [[ ! -z "$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)" ]]; then
          # Trim grpc:// prefix
          export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"
        fi

        # Run whatever is in `command` here
        docker-entrypoint.sh "${@:0}"
      |||,
    ],

    volumeMap+: {
      dshm: volumes.MemoryVolumeSpec {
        name: 'dshm',
        mountPath: '/dev/shm',
      },
    },

    cpu: '4.5',
    memory: '8Gi',

    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              XLA_USE_BF16: '0',
            },
          },
        },
      },
    },
  },
}
