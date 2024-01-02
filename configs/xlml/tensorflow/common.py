# Copyright 2023 Google LLC
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

"""Utilities to construct common configs."""

from typing import Tuple


def set_up_pjrt_nightly() -> Tuple[str]:
  """Common set up for PJRT nightly tests."""
  return (
      "pip install tensorflow-text-nightly",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/tensorflow/tf-nightly/latest/*.whl /tmp/ && pip install /tmp/tf*.whl --force",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/libtpu/latest/libtpu.so /lib/",
  )


# TODO(ranran): migrate repo `tf2-api-tests` from xl-ml-test to cloud-ml-auto-solutions project
def set_up_tensorflow_keras() -> Tuple[str]:
  """Common set up for tensorflow Keras tests."""
  return (
      "pip install --upgrade --force-reinstall tf-keras-nightly",
      "export PATH=$PATH:/root/google-cloud-sdk/bin && cd /tmp && sudo gcloud source repos clone tf2-api-tests --project=xl-ml-test",
      "cd /tmp/tf2-api-tests && pip install behave matplotlib",
  )


def set_up_google_tensorflow_models() -> Tuple[str]:
  """Common set up for tensorflow models."""
  return (
      "sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu && git clone https://github.com/tensorflow/models.git",
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      "pip install tensorflow-recommenders --no-deps",
      "pip install --upgrade --force-reinstall tf-keras-nightly",
  )
