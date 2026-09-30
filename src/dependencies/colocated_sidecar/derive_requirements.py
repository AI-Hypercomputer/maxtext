"""Derive deterministic sidecar requirements from a trainer pip freeze."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
import sys

_PKG_NAME_RE = re.compile(r"^([A-Za-z0-9_.\-]+)")


@dataclass(frozen=True)
class SidecarRequirementsSpec:
  drop: frozenset[str] = frozenset({
      "jax",
      "jaxlib",
      "libtpu",
      "libtpu-nightly",
      "pytest",
      "pylint",
      "pyink",
      "pre-commit",
      "pytype",
  })
  must_include: frozenset[str] = frozenset({
      "orbax-checkpoint",
      "tensorstore",
      "gcsfs",
      "google-cloud-storage",
  })


def _canonical_name(name: str) -> str:
  return name.lower().replace("_", "-")


def derive_sidecar_requirements(
    trainer_freeze: Sequence[str],
    spec: SidecarRequirementsSpec = SidecarRequirementsSpec(),
) -> list[str]:
  """Filter and sort trainer pip-freeze lines for the colocated Python sidecar."""
  drop_set = {_canonical_name(p) for p in spec.drop}
  must_include_set = {_canonical_name(p) for p in spec.must_include}

  surviving: dict[str, str] = {}
  for raw_line in trainer_freeze:
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if match := _PKG_NAME_RE.match(line):
      pkg = _canonical_name(match.group(1))
      if (
          pkg in drop_set
          or ("jax" in drop_set and pkg.startswith("jax"))
          or ("libtpu" in drop_set and pkg.startswith("libtpu"))
      ):
        continue
      surviving[pkg] = line

  missing = sorted(must_include_set - surviving.keys())
  if missing:
    raise ValueError(
        f"Missing mandatory sidecar package(s) in trainer freeze: {', '.join(missing)}"
    )

  return sorted(surviving.values())


if __name__ == "__main__":
  for req_line in derive_sidecar_requirements(sys.stdin.read().splitlines()):
    print(req_line)
