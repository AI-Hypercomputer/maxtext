
import importlib.metadata
from packaging import version

# The following dependencies are referenced from the original file and are assumed to be in scope:
# - split_package_version
# - BACKENDS_MAPPING
# - VersionComparison

class Backend:
  """A class to represent a backend requirement."""

  def __init__(self, backend_requirement: str):
    self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

    if self.package_name not in BACKENDS_MAPPING:
      raise ValueError(f"Backends should be defined in the BACKENDS_MAPPING. Offending backend: {self.package_name}")

  def is_satisfied(self) -> bool:
    """Checks if the backend requirement is satisfied in the current environment."""
    return VersionComparison.from_string(self.version_comparison)(
        version.parse(importlib.metadata.version(self.package_name)), version.parse(self.version)
    )

  def __repr__(self) -> str:
    return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

  @property
  def error_message(self):
    """Generates an error message for when the backend is not satisfied."""
    return (
        f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
        f" library was not found with this version in your environment."
    )
