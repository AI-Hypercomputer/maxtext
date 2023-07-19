import dataclasses

from airflow.decorators import task

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

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
      serialization.NoEncryption()
  )
  public_key = key.public_key().public_bytes(
      serialization.Encoding.OpenSSH,
      serialization.PublicFormat.OpenSSH
  )

  return SshKeys(private=private_key.decode(), public=public_key.decode())
