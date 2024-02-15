// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  PyTorchTpuVmMixin:: experimental.BaseTpuVmMixin {
    local config = self,

    // Don't need to mount datasets within Kubernetes for TPU VM.
    volumeMap+: { datasets: null },

    tpuSettings+: {
      tpuVmPytorchSetup: |||
        echo No PyTorch setup required.
      |||,
      tpuVmExtraSetup: |||
        echo No extra setup required.
      |||,
      // XRT_TPU_CONFIG set up by xla_dist on pods
      tpuVmExports:
        if config.accelerator.replicas == 1 then
          |||
            export XRT_TPU_CONFIG='localservice;0;localhost:51011'
            export TPU_NUM_DEVICES=%d
          ||| % config.accelerator.numCores
        else
          '',
      tpuVmCreateSleepSeconds:
        if config.accelerator.replicas == 1 then
          super.tpuVmCreateSleepSeconds
        else
          180,
      tpuVmXlaDistPrefix:
        if config.accelerator.replicas == 1 then
          null
        else
          [
            'python3',
            '-m',
            'torch_xla.distributed.xla_dist',
            '--tpu=tpu-$(POD_UID)',
            '--',
          ],
      tpuVmMainCommandWorkers: '0',
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              // Distribute command with xla_dist on pods
              testCommand: if config.tpuSettings.tpuVmXlaDistPrefix == null then
                utils.toCommandString(config.command)
              else
                utils.toCommandString(
                  config.tpuSettings.tpuVmXlaDistPrefix + config.command
                ),
              commandWorkers: config.tpuSettings.tpuVmMainCommandWorkers,
              pytorchSetup: config.tpuSettings.tpuVmPytorchSetup,
              extraSetup: config.tpuSettings.tpuVmExtraSetup,
              exports: config.tpuSettings.tpuVmExports,
            },
            args: null,
            // PyTorch tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u

                cat > workersetup.sh << TEST_SCRIPT_EOF
                sudo apt-get -y update
                // Ensure lock is released after udpate
                sudo kill -9 $(lsof /var/lib/dpkg/lock-frontend | awk '{print $2}')
                sudo dpkg --configure -a
                sudo apt-get -y install nfs-common
                sudo mkdir /datasets && sudo mount.nfs $(PYTORCH_DATA_LOCATION) /datasets

                yes '' | gcloud compute config-ssh

                cd
                %(pytorchSetup)s

                cd
                %(extraSetup)s
                TEST_SCRIPT_EOF
                gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) --zone=$(cat /scripts/zone) --ssh-key-file=/scripts/id_rsa --strict-host-key-checking=no --internal-ip --worker=all --command "$(cat workersetup.sh)"

                cat > testscript.sh << 'TEST_SCRIPT_EOF'
                %(exports)s
                %(testCommand)s
                TEST_SCRIPT_EOF
                gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) --zone=$(cat /scripts/zone) --ssh-key-file=/scripts/id_rsa --strict-host-key-checking=no --internal-ip --worker=%(commandWorkers)s --command "$(cat testscript.sh)"

                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
  PjRt:: {
    tpuSettings+: {
      tpuVmExports: |||
        export PJRT_DEVICE=TPU
      |||,
      tpuVmXlaDistPrefix: null,
      tpuVmMainCommandWorkers: 'all',
    },
  },
}
