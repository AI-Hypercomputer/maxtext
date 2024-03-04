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

"""Utility for ssh."""

import dataclasses

from airflow.decorators import task
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


@dataclasses.dataclass
class SshKeys:
  """Represents a pair of SSH keys."""

  private: str
  public: str


@task
def generate_ssh_keys() -> SshKeys:
  """Generate an RSA key pair in the OpenSSH format."""
  key = rsa.generate_private_key(
      public_exponent=65537,
      key_size=2048,
  )

  private_key = key.private_bytes(
      serialization.Encoding.PEM,
      serialization.PrivateFormat.OpenSSH,
      serialization.NoEncryption(),
  )
  public_key = key.public_key().public_bytes(
      serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
  )

  return SshKeys(private=private_key.decode(), public=public_key.decode())
