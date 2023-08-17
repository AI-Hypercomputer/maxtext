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

from typing import Iterable

UPGRADE_PIP = "pip install --upgrade pip"
INSTALL_LATEST_JAX = (
    "pip install jax[tpu] -f"
    " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
)


def set_up_google_flax() -> Iterable[str]:
  """Common set up for flax repo."""
  return (
      UPGRADE_PIP,
      INSTALL_LATEST_JAX,
      "pip install --upgrade clu tensorflow tensorflow-datasets",
      "git clone https://github.com/google/flax.git /tmp/flax",
      "pip install --user flax",
  )


def set_up_hugging_face_transformer() -> Iterable[str]:
  """Common set up for hugging face transformer repo."""
  return (
      UPGRADE_PIP,
      INSTALL_LATEST_JAX,
      (
          "git clone https://github.com/huggingface/transformers.git"
          " /tmp/transformers"
      ),
      "cd /tmp/transformers && pip install .",
      "pip install -r /tmp/transformers/examples/flax/_tests_requirements.txt",
      "pip install --upgrade huggingface-hub urllib3 zipp",
      "pip install tensorflow",
  )


def set_up_hugging_face_diffuser() -> Iterable[str]:
  """Common set up for hugging face diffuser repo."""
  return (
      UPGRADE_PIP,
      INSTALL_LATEST_JAX,
      "git clone https://github.com/huggingface/diffusers.git /tmp/diffusers",
      "cd /tmp/diffusers && pip install .",
  )
