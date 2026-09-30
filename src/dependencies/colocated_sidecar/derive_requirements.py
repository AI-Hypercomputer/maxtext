"""Derive deterministic sidecar requirements from a trainer pip freeze."""

from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
      "tokamax",
      "qwix",
  })
  must_include: frozenset[str] = frozenset({
      "orbax-checkpoint",
      "tensorstore",
      "gcsfs",
      "google-cloud-storage",
  })
  unpin: frozenset[str] = frozenset({"protobuf"})
  allowlist: frozenset[str] | None = None


def _canonical_name(name: str) -> str:
  return name.lower().replace("_", "-")


def parse_package_names(lines: Sequence[str]) -> frozenset[str]:
  """Extract normalized package names from a requirements file."""
  names: set[str] = set()
  for raw in lines:
    line = raw.strip()
    if not line or line.startswith(("#", "-")):
      continue
    if match := _PKG_NAME_RE.match(line):
      names.add(_canonical_name(match.group(1)))
  return frozenset(names)


def derive_sidecar_requirements(
    trainer_freeze: Sequence[str],
    spec: SidecarRequirementsSpec = SidecarRequirementsSpec(),
) -> list[str]:
  """Filter and sort trainer pip-freeze lines for the colocated Python sidecar."""
  drop_set = {_canonical_name(p) for p in spec.drop}
  must_include_set = {_canonical_name(p) for p in spec.must_include}
  unpin_set = {_canonical_name(p) for p in spec.unpin}
  allow_set = (
      {_canonical_name(p) for p in spec.allowlist}
      | must_include_set
      | {"cloudpickle", "etils"}
      if spec.allowlist is not None
      else None
  )

  surviving: dict[str, str] = {}
  for raw_line in trainer_freeze:
    line = raw_line.strip()
    if not line or line.startswith(("#", "-")) or " @ file://" in line:
      continue
    if match := _PKG_NAME_RE.match(line):
      pkg = _canonical_name(match.group(1))
      if (
          pkg in drop_set
          or ("jax" in drop_set and pkg.startswith("jax"))
          or ("libtpu" in drop_set and pkg.startswith("libtpu"))
      ):
        continue
      if allow_set is not None and pkg not in allow_set:
        continue
      surviving[pkg] = pkg if pkg in unpin_set else line

  missing = sorted(must_include_set - surviving.keys())
  if missing:
    raise ValueError(
        f"Missing mandatory sidecar package(s) in trainer freeze: {', '.join(missing)}"
    )

  return sorted(surviving.values())


if __name__ == "__main__":
  parser = ArgumentParser(description=__doc__)
  parser.add_argument(
      "--base-requirements",
      type=Path,
      default=None,
      help="Optional base requirements.txt to bound candidate packages.",
  )
  args = parser.parse_args()
  allow = (
      parse_package_names(
          args.base_requirements.read_text(encoding="utf-8").splitlines()
      )
      if args.base_requirements
      else None
  )
  for req_line in derive_sidecar_requirements(
      sys.stdin.read().splitlines(), SidecarRequirementsSpec(allowlist=allow)
  ):
    print(req_line)
