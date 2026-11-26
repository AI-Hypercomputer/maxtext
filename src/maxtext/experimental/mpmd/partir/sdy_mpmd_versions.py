# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which Shardy MPMD frontend matches which jaxlib, and how to install it.

Shardy MPMD's Python frontend (``shardy.integrations.python.jax.mpmd``) is neither on PyPI nor in the jax or jaxlib
wheels, and it must come from the Shardy commit the installed jaxlib was built from, so that it matches the C++
bindings in ``jaxlib._sdy_mpmd``. :data:`SHARDY_COMMITS` lists those commits; install the frontend with pip::

  pip install --no-deps \
    "shardy-mpmd @ git+https://github.com/openxla/shardy@<commit>#subdirectory=shardy/integrations/python/jax/mpmd"

(:func:`install_command` prints the line for a jaxlib version). A Shardy checkout on ``PYTHONPATH`` works too.

This module imports neither jax nor Shardy, so the version table and its checks are usable (and tested) without
shardy-mpmd installed; ``sdy_mpmd_compat`` imports the frontend itself.
"""

import importlib.metadata
import json
import re
import types

SHARDY_MPMD_DISTRIBUTION = "shardy-mpmd"
SHARDY_MPMD_SUBDIRECTORY = "shardy/integrations/python/jax/mpmd"

# The Shardy commit each jaxlib release was built from (XLA's third_party/shardy pin at that jaxlib). jaxlib records
# only its jax commit, so the matching Shardy commit cannot be discovered at run time.
SHARDY_COMMITS = types.MappingProxyType(
    {
        "0.11.1": "2832731619ffb4bcc718faa4aa8054214a68c969",
        "0.12.0.1": "38ab99f4d252c59d491899a694fe8cbadfeccec3",
        "0.12.1": "57fffc23dfdf139b80013ad681701cee59c22f96",
    }
)

_COMMIT_RE = re.compile(r"\b[0-9a-f]{40}\b")


def install_command(jaxlib_version: str) -> str:
  """Returns the pip command that installs the Shardy MPMD frontend matching `jaxlib_version`."""
  commit = SHARDY_COMMITS.get(jaxlib_version)
  if commit is None:
    known = ", ".join(f"jaxlib {version}" for version in SHARDY_COMMITS)
    return (
        f"no Shardy commit is known for jaxlib {jaxlib_version} (known: {known}); install the frontend from the Shardy"
        " commit that XLA's third_party/shardy pins at the commit this jaxlib was built from, or install one of the"
        " known jaxlib versions"
    )
  return (
      f'pip install --no-deps "{SHARDY_MPMD_DISTRIBUTION} @ git+https://github.com/openxla/shardy@{commit}'
      f'#subdirectory={SHARDY_MPMD_SUBDIRECTORY}"'
  )


def commit_from_direct_url(direct_url: str) -> str | None:
  """Returns the Shardy commit recorded in a distribution's PEP 610 ``direct_url.json``, or None.

  pip records the resolved commit of a ``git+https://...@<commit>`` install under ``vcs_info``; an archive install
  (``https://github.com/openxla/shardy/archive/<commit>.zip``) carries the commit in its URL. A local directory install
  records neither.
  """
  try:
    record = json.loads(direct_url)
  except ValueError:
    return None
  commit = record.get("vcs_info", {}).get("commit_id")
  if commit:
    return commit
  match = _COMMIT_RE.search(record.get("url", ""))
  return match.group(0) if match else None


def installed_commit() -> str | None:
  """Returns the Shardy commit the installed ``shardy-mpmd`` distribution came from, or None if it is unknown."""
  try:
    direct_url = importlib.metadata.distribution(SHARDY_MPMD_DISTRIBUTION).read_text("direct_url.json")
  except importlib.metadata.PackageNotFoundError:  # a Shardy checkout on PYTHONPATH
    return None
  return commit_from_direct_url(direct_url) if direct_url else None


def version_mismatch(jaxlib_version: str, commit: str | None) -> str | None:
  """Returns why `commit` may not match `jaxlib_version`'s C++, or None when it matches or cannot be checked."""
  expected = SHARDY_COMMITS.get(jaxlib_version)
  if expected is None:
    return f"Shardy MPMD's Python must match the C++ in jaxlib: {install_command(jaxlib_version)}."
  if commit is None or commit == expected:
    return None
  return (
      f"shardy-mpmd was installed from Shardy {commit[:12]}, but jaxlib {jaxlib_version} was built with Shardy"
      f" {expected[:12]}, and its Python must match the C++ in jaxlib. Reinstall it: {install_command(jaxlib_version)}"
      " --force-reinstall"
  )
