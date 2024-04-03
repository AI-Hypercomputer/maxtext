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

from __future__ import annotations


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


def set_up_se_nightly() -> tuple[str]:
  """Adjust grpc_tpu_worker for SE tests"""
  return (
      "sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images-dev\/grpc_tpu_worker:nightly\"/' /etc/systemd/system/tpu-runtime.service",
      "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime",
      "cat /etc/systemd/system/tpu-runtime.service",
  )


def install_tf(
    major: Optional[int] = None,
    minor: Optional[int] = None,
    patch: Optional[int] = None,
    libtpu_version: Optional[str] = None,
) -> tuple[str]:
  """Install tf + libtpu.

  If the version numbers are set, installs that version. Otherwise just installs using nightly.
  Either all of the version numbers need to be set or none of them should be set.

  Args:
      major (Optional[int]): The major version number
      minor (Optional[int]): The minor version number
      patch (Optional[int]): The minor version number
      libtpu_version (Optional[str]): The libtpu version to install
  """
  gs_version_str = "tf-nightly"
  if any(x is not None for x in {major, minor, patch}):
    msg = "All parts of a version should be specified if any of them are"
    assert all(x is not None for x in {major, minor, patch}), msg
    gs_version_str = f"tf-{major}-{minor}-{patch}"

  libtpu_version_str = "latest"
  if libtpu_version is not None:
    libtpu_version_str = f"{libtpu_version}/latest"

  return (
      "pip install tensorflow-text-nightly",
      f"sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/tensorflow/{gs_version_str}/latest/t*.whl /tmp/ && pip install /tmp/t*.whl --force",
      f"sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/libtpu/{libtpu_version_str}/libtpu.so /lib/",
      CMD_PRINT_TF_VERSION,
  )


def set_up_keras(version: Optional[str] = None) -> tuple[str]:
  """Common set up for tensorflow Keras tests.

  If a version is not set, defaults to nightly.

  Args:
    version(Optional[str]): The keras version to install
  """
  cmd_install_keras = CMD_INSTALL_KERAS_NIGHTLY
  if version is not None:
    cmd_install_keras = (
        f"pip install --upgrade --force-reinstall --no-deps tf-keras=={version}"
    )

  return (
      cmd_install_keras,
      "export PATH=$PATH:/root/google-cloud-sdk/bin && cd /tmp",
      "gcloud source repos clone tf2-api-tests --project=cloud-ml-auto-solutions || (cd tf2-api-tests && git pull)",
      "cd /tmp/tf2-api-tests && pip install behave matplotlib",
  )


def set_up_tensorflow_models(
    models_branch: Optional[str] = None,
    keras_version: Optional[str] = None,
) -> tuple[str]:
  """Common set up for tensorflow models for the release.

  If any versions are not set, defaults to nightly.

  Args:
      models_branch (Optional[str]): The models branch to use
  """
  if models_branch is None:
    models_branch = "master"

  cmd_install_keras = CMD_INSTALL_KERAS_NIGHTLY
  if keras_version is not None:
    cmd_install_keras = f"pip install --upgrade --force-reinstall --no-deps tf-keras=={keras_version}"

  return (
      "sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu",
      f'if [ ! -d "/usr/share/tpu/models" ]; then sudo git clone -b {models_branch} https://github.com/tensorflow/models.git; fi',
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      "pip install tensorflow-recommenders --no-deps",
      cmd_install_keras,
  )


def set_up_dlrm_v5p(models_branch: Optional[str] = None) -> tuple[str]:
  """Setup DLRM on v5p TPUs

  If any versions are not set, defaults to nightly.
  """
  if models_branch is None:
    models_branch = "master"

  return (
      "sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu",
      f'if [ ! -d "/usr/share/tpu/models" ]; then sudo git clone -b {models_branch} http    s://github.com/tensorflow/models.git; fi',
      'if [ ! -d "/usr/share/tpu/recommenders" ]; then sudo git clone https://github.com/tensorflow/recommenders.git; fi',
      "pip install gin-config tensorflow-datasets tbp-nightly tf_keras_nightly tbp-nightly",
      "sudo pip install --no-deps /usr/share/tpu/recommenders",
      "source /usr/share/tpu/four_tasks_per_host.sh && setup_four_tasks_per_host",
  )


def export_env_variables(
    tpu_name: str,
    is_pod: bool,
    is_pjrt: bool,
    is_v5p_sc: Optional[bool] = False,
) -> str:
  """Export environment variables for training."""
  stmts = [
      "export WRAPT_DISABLE_EXTENSIONS=true",
      "export TF_USE_LEGACY_KERAS=1",
  ]

  if is_v5p_sc:
    stmts.append("source /usr/share/tpu/four_tasks_per_host.sh")
    stmts.append("export `get_tpu_name`")
    stmts.append(
        "export PYTHONPATH='/usr/share/tpu/recommenders:/usr/share/tpu/models'"
    )
    stmts.append("export TPU_LOAD_LIBRARY=0")
    stmts.append(
        "export TF_XLA_FLAGS='--tf_mlir_enable_mlir_bridge=true --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true'"
    )
  else:
    stmts.append(f"export TPU_NAME={tpu_name}")
    stmts.append("export PYTHONPATH='.'")
    if is_pod:
      stmts.append("export TPU_LOAD_LIBRARY=0")
    elif is_pjrt:
      stmts.append("export NEXT_PLUGGABLE_DEVICE_USE_C_API=true")
      stmts.append("export TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so")

  return " && ".join(stmts)
