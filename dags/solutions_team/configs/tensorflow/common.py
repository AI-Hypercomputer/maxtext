# Copyright 2024 Google LLC
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


# Keras API
AAA_CONNECTION = "aaa_connection"
CUSTOM_LAYERS_MODEL = "custom_layers_model"
CUSTOM_TRAINING_LOOP = "custom_training_loop"
FEATURE_COLUMN = "feature_column"
RNN = "rnn"
UPSAMPLE = "upsample"
SAVE_AND_LOAD_LOCAL_DRIVER = "save_and_load_io_device_local_drive"
SAVE_AND_LOAD_FEATURE = "save_and_load.feature"
TRAIN_AND_EVALUATE = "train_and_evaluate"
TRANSFER_LEARNING = "transfer_learning"
FEATURE_NAME = {
    AAA_CONNECTION: "connection",
    CUSTOM_LAYERS_MODEL: "custom_layers",
    CUSTOM_TRAINING_LOOP: "ctl",
    FEATURE_COLUMN: "feature_column",
    RNN: "rnn",
    UPSAMPLE: "upsample",
    SAVE_AND_LOAD_LOCAL_DRIVER: "save_load_localhost",
    SAVE_AND_LOAD_FEATURE: "save_and_load",
    TRAIN_AND_EVALUATE: "train_and_evaluate",
    TRANSFER_LEARNING: "transfer_learning",
}
FEATURE_TIMEOUT = {
    AAA_CONNECTION: 60,
    CUSTOM_LAYERS_MODEL: 60,
    CUSTOM_TRAINING_LOOP: 60,
    FEATURE_COLUMN: 120,
    RNN: 60,
    UPSAMPLE: 60,
    SAVE_AND_LOAD_LOCAL_DRIVER: 120,
    SAVE_AND_LOAD_FEATURE: 120,
    TRAIN_AND_EVALUATE: 180,
    TRANSFER_LEARNING: 60,
}


CMD_PRINT_TF_VERSION = "python3 -c \"import tensorflow; print('Running using TensorFlow Version: ' + tensorflow.__version__)\""
CMD_INSTALL_KERAS_NIGHTLY = (
    "pip install --upgrade --no-deps --force-reinstall tf-keras-nightly"
)


def set_up_se_nightly() -> Tuple[str]:
  """Adjust grpc_tpu_worker for SE tests"""
  return (
      "sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images-dev\/grpc_tpu_worker:nightly\"/' /etc/systemd/system/tpu-runtime.service",
      "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime",
      "cat /etc/systemd/system/tpu-runtime.service",
  )


def install_tf_nightly() -> Tuple[str]:
  """Install tf nightly + libtpu."""
  return (
      "pip install tensorflow-text-nightly",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/tensorflow/tf-nightly/latest/*.whl /tmp/ && pip install /tmp/tf*.whl --force",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/libtpu/latest/libtpu.so /lib/",
      CMD_PRINT_TF_VERSION,
  )


def install_tf_2_16() -> Tuple[str]:
  """Install tf 2.16 + libtpu."""
  return (
      "pip install tensorflow-text-nightly",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/tensorflow/2.16/2024-02-20/t*.whl /tmp/ && pip install /tmp/t*.whl --force",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/libtpu/1.10.0/rc0/libtpu.so /lib/",
      CMD_PRINT_TF_VERSION,
  )


def set_up_tensorflow_keras() -> Tuple[str]:
  """Common set up for tensorflow Keras tests."""
  return (
      CMD_INSTALL_KERAS_NIGHTLY,
      "export PATH=$PATH:/root/google-cloud-sdk/bin && cd /tmp && sudo gcloud source repos clone tf2-api-tests --project=cloud-ml-auto-solutions",
      "cd /tmp/tf2-api-tests && pip install behave matplotlib",
  )


def set_up_tensorflow_2_16_keras() -> Tuple[str]:
  """Common set up for tensorflow Keras 2.16 tests."""
  return (
      "pip install --upgrade --force-reinstall tf-keras==2.16.0rc0",
      "export PATH=$PATH:/root/google-cloud-sdk/bin && cd /tmp && sudo gcloud source repos clone tf2-api-tests --project=cloud-ml-auto-solutions",
      "cd /tmp/tf2-api-tests && pip install behave matplotlib",
  )


def set_up_google_tensorflow_models() -> Tuple[str]:
  """Common set up for tensorflow models."""
  return (
      'if [ ! -d "/usr/share/tpu/models" ]; then sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu && git clone https://github.com/tensorflow/models.git; fi',
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      "pip install tensorflow-recommenders --no-deps",
      CMD_INSTALL_KERAS_NIGHTLY,
  )


def set_up_google_tensorflow_2_16_models() -> Tuple[str]:
  """Common set up for tensorflow models."""
  return (
      "sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu",
      "sudo git clone -b r2.16.0 https://github.com/tensorflow/models.git",
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      "pip install tensorflow-recommenders --no-deps",
      "pip install --upgrade --force-reinstall tf-keras==2.16.0rc0",
  )
