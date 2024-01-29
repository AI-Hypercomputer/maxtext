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

local timeouts = import 'templates/timeouts.libsonnet';
local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

{
  BaseTpuVmTest:: {
    local config = self,
    local cleanupHook = {
      preStop: {
        exec: {
          command: [
            'bash',
            '/scripts/cleanup.sh',
          ],
        },
      },
    },

    volumeMap+: {
      scripts: volumes.MemoryVolumeSpec {
        name: 'scripts',
        mountPath: '/scripts',
      },
    },

    testName+: '-1vm',

    tpuSettings+: {
      local tpuSettings = self,

      softwareVersion: 'v2-nightly',

      // Startup script in TPU VM metadata.
      tpuVmStartupScript: 'echo Running startup script',

      // Amount of time to sleep after TPU is READY.
      tpuVmCreateSleepSeconds:
        if config.accelerator.version <= 3 then
          60
        else
          90,

      // Additional arguments for test Docker container.
      tpuVmDockerArgs: '',
    },
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            lifecycle: cleanupHook,
            resources+: {
              // HACK: remove standard Cloud TPU resource.
              local originalLimits = super.limits,
              limits: {
                [field]: originalLimits[field]
                for field in std.objectFields(originalLimits)
                if !std.startsWith(field, 'cloud-tpus.google.com')
              },
            },
          },
        },
        initContainerMap+:: {
          'create-tpu': {
            image: 'google/cloud-sdk',
            local tpuCreateSettings = {
              acceleratorName: std.escapeStringBash(config.accelerator.name),
              softwareVersion: std.escapeStringBash(config.tpuSettings.softwareVersion),
              startupScript: std.escapeStringBash(config.tpuSettings.tpuVmStartupScript),
              sleepTime: config.tpuSettings.tpuVmCreateSleepSeconds,
              testName: std.strReplace(config.testName, '.', '-'),
            },
            command: utils.scriptCommand(|||
              project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              tpu_name=tpu-${POD_UID}
              ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              echo "
              gcloud alpha compute tpus tpu-vm delete -q --async ${tpu_name} --zone=${zone}
              sleep 60
              " > /scripts/cleanup.sh

              echo "xl-ml-test:$(cat /scripts/id_rsa.pub)" > ssh-keys.txt
              echo %(startupScript)s > startup-script.txt

              # Retry every 30 seconds for up to 10 minutes
              start_time="$(date -u +%%s)"
              for i in {1..40}; do
                set +e
                gcloud alpha compute tpus tpu-vm create ${tpu_name} \
                  --accelerator-type=%(acceleratorName)s \
                  --version=%(softwareVersion)s  \
                  --metadata-from-file='ssh-keys=ssh-keys.txt,startup-script=startup-script.txt' \
                  --labels='test-name=%(testName)s' \
                  --zone=${zone}

                exit_code=$?
                set -e

                current_time="$(date -u +%%s)"
                elapsed_seconds=$(($current_time-$start_time))
                # Break if command passed or 10-minute limit reached
                test $exit_code = 0 && break
                test $elapsed_seconds -gt 600 && break
                sleep 30
              done

              if [ $exit_code -ne 0 ]; then
                exit $exit_code
              fi

              echo ${zone} > /scripts/zone
              echo ${tpu_name} > /scripts/tpu_name
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/tpu_ip
              gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" > /scripts/all_tpu_ips

              sleep %(sleepTime)d
            ||| % tpuCreateSettings),
            env: [
              {
                name: 'POD_UID',
                valueFrom: {
                  fieldRef: {
                    fieldPath: 'metadata.uid',
                  },
                },
              },
            ],
            volumeMounts: [
              {
                mountPath: '/scripts',
                name: 'scripts',
              },
            ],
          },
        },
      },
    },
  },
  // `BaseTpuVmMixin` is used to convert a 2VM target to 1VM.
  BaseTpuVmMixin:: self.BaseTpuVmTest {
    local config = self,

    // Disable retries
    jobTemplate+:: {
      spec+: {
        activeDeadlineSeconds: std.max(2 * config.timeout, 24 * timeouts.one_hour),
        backoffLimit: 0,
      },
    },

    // TPU VM tests don't run the models directly
    cpu: 1,
    memory: '2Gi',

    // Pass TPU VM name to test container
    podTemplate+:: {
      spec+: {
        activeDeadlineSeconds: config.timeout,
        containerMap+:: {
          train+: {
            image: 'google/cloud-sdk',
            envMap+:: {
              LOCAL_OUTPUT_DIR: '/tmp/model_dir',
              KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS: if config.accelerator.replicas == 1 then
                'local'
              else
                'tpu-$(POD_UID)',
            },
          },
        },
      },
    },
  },
}
