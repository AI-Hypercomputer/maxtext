// Copyright 2021 Google LLC
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
local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  JaxTest:: common.CloudAcceleratorTest + experimental.BaseTpuVmMixin {
    local config = self,

    frameworkPrefix: 'jax',
    image: 'google/cloud-sdk',
    accelerator: tpus.v2_8,

    metricConfig+: {
      sourceMap+:: {
        tensorboard+: {
          exclude_tags: ['_hparams_/session_start_info'],
          merge_runs: true,
        },
        // Remove default duration assertion.
        literals+: {
          assertions+: {
            duration: null,
          },
        },
      },
    },

    jaxlibVersion:: error 'Add jaxlib version mixin',
    scriptConfig:: {
      maybeBuildJaxlib: error 'Must define `maybeBuildJaxlib`',
      installLocalJax: error 'Must define `installLocalJax`',
      installLatestJax: error 'Must define `installLatestJax`',
      testEnvWorkarounds: error 'Must define `testEnvWorkarounds`',
      printDiagnostics: |||
        python3 -c 'import jax; print("jax version:", jax.__version__)'
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
        python3 -c 'import jax; print("libtpu version:",
          jax.lib.xla_bridge.get_backend().platform_version)'
      |||,
    },

    tpuSettings+: {
      tpuVmCreateSleepSeconds: 60,
    },

    setup: error 'Must define `setup`',
    runTest: error 'Must define `runTest`',

    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      %(setup)s
      %(runTest)s
    ||| % self,
    command: [
      'bash',
      '-c',
      |||
        set -x
        set -u

        cat > testsetup.sh << 'TEST_SCRIPT_EOF'
        %s
        TEST_SCRIPT_EOF

        gcloud alpha compute tpus tpu-vm ssh xl-ml-test@$(cat /scripts/tpu_name) \
        --zone=$(cat /scripts/zone) \
        --ssh-key-file=/scripts/id_rsa \
        --strict-host-key-checking=no \
        --internal-ip \
        --worker=all \
        --command "$(cat testsetup.sh)"

        exit_code=$?
        bash /scripts/cleanup.sh
        exit $exit_code
      ||| % config.testScript,
    ],
  },

  jaxlibHead:: {
    jaxlibVersion:: 'head',
    scriptConfig+: {
      // Install jax without jaxlib or libtpu deps
      installLocalJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git
        cd jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .
      |||,
      installLatestJax: 'pip install jax',
      maybeBuildJaxlib: |||
        echo "Installing latest jaxlib-nightly..."
        pip install --pre jaxlib \
          -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
        pip list | grep jaxlib
        python3 -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'

        echo "Installing latest libtpu-nightly..."
        pip install libtpu-nightly --no-index --pre \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
    },
  },

  jaxlibLatest:: {
    jaxlibVersion:: 'latest',
    scriptConfig+: {
      installLocalJax: |||
        echo "Checking out and installing JAX..."
        git clone https://github.com/google/jax.git
        cd jax
        echo "jax git hash: $(git rev-parse HEAD)"
        pip install -r build/test-requirements.txt

        pip install .[tpu] \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      installLatestJax: |||
        pip install jax[tpu] \
          -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      maybeBuildJaxlib: '',
    },
  },

  tpuVmBaseImage:: {
    local config = self,

    tpuSettings+: {
      softwareVersion: 'tpu-ubuntu2204-base',
    },
    scriptConfig+: {
      testEnvWorkarounds: |||
        # b/192016388: fix host_callback_to_tf_test.py
        pip install tensorflow
      ||| + self.maybeInstallLibtpuV4,
      maybeInstallLibtpuV4: if config.accelerator.type == 'tpu' && config.accelerator.version == 4 then |||
        gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev20211028-py3-none-any.whl .
        pip install libtpu_tpuv4-0.1.dev20211028-py3-none-any.whl
      ||| else '',
    },
  },

  tpuVmV4Base:: {
    local config = self,
    accelerator: tpus.v4_8,

    tpuSettings+: {
      softwareVersion: 'tpu-ubuntu2204-base',
    },
    scriptConfig+: {
      testEnvWorkarounds: |||
        pip install tensorflow
      |||,
    },
  },

  huggingFaceTransformer:: {
    scriptConfig+: {
      installPackages: |||
        pip install --upgrade pip
        git clone https://github.com/huggingface/transformers.git
        cd transformers && pip install .
        pip install -r examples/flax/_tests_requirements.txt
        pip install --upgrade huggingface-hub urllib3 zipp

        pip install tensorflow
        pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      verifySetup: |||
        python3 -c 'import flax; print("flax version:", flax.__version__)'
        num_devices=`python3 -c "import jax; print(jax.device_count())"`
        if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
        fi
      |||,
    },
  },
  huggingFaceDiffuser:: {
    scriptConfig+: {
      installPackages: |||
        pip install --upgrade pip
        git clone https://github.com/huggingface/diffusers.git
        cd diffusers && pip install .
        pip install -U -r examples/text_to_image/requirements_flax.txt
        export PATH=$PATH:$HOME/.local/bin

        pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      |||,
      verifySetup: |||
        python3 -c 'import flax; print("flax version:", flax.__version__)'
        num_devices=`python3 -c "import jax; print(jax.device_count())"`
        if [ "$num_devices" = "1" ]; then
          echo "No TPU devices detected"
          exit 1
        fi
      |||,
    },
  },
}
