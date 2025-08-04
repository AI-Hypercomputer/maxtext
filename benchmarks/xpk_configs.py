# SPDX-License-Identifier: Apache-2.0

import dataclasses


# This is needed to prevent circular imports.
@dataclasses.dataclass
class XpkClusterConfig:
  """Holds details related to a XPK cluster to run workloads on."""

  cluster_name: str
  project: str
  zone: str
  device_type: str
