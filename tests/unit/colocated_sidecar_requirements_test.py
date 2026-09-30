"""Unit tests for colocated sidecar requirements derivation and Dockerfile."""

import os
import re
import unittest

from dependencies.colocated_sidecar.derive_requirements import (
    SidecarRequirementsSpec,
    derive_sidecar_requirements,
)

SAMPLE_TRAINER_FREEZE = (
    "# Trainer pip freeze snapshot",
    "jax==0.11.0",
    "jaxlib==0.11.0",
    "libtpu-nightly==0.1.dev20260316",
    "libtpu==0.1.4",
    "pytest==8.0.0",
    "pylint==3.1.0",
    "pyink==24.3.0",
    "pre-commit==3.6.0",
    "pytype==2024.02.27",
    "",
    "# Mandatory storage & checkpointing",
    "tensorstore==0.1.85",
    "orbax-checkpoint==0.12.4",
    "cloudpickle==3.0.0",
    "gcsfs==2024.9.0",
    "google-cloud-storage==2.18.2",
    "# Other trainer packages",
    "numpy==1.26.4",
    "etils==1.9.4",
    "absl-py==2.1.0",
)


class ColocatedSidecarRequirementsTest(unittest.TestCase):

  def test_derive_sidecar_requirements_filtering_and_sorted_pins(self):
    result = derive_sidecar_requirements(SAMPLE_TRAINER_FREEZE)

    # Load-bearing counter: 8 surviving lines out of 17 package lines
    self.assertEqual(len(result), 8)
    self.assertEqual(result, sorted(result))

    for line in result:
      self.assertIsNone(re.match(r"^(jax|libtpu)\b", line, re.IGNORECASE))
      pkg = line.split("==")[0].lower()
      self.assertNotIn(pkg, SidecarRequirementsSpec().drop)

    self.assertEqual(
        result,
        [
            "absl-py==2.1.0",
            "cloudpickle==3.0.0",
            "etils==1.9.4",
            "gcsfs==2024.9.0",
            "google-cloud-storage==2.18.2",
            "numpy==1.26.4",
            "orbax-checkpoint==0.12.4",
            "tensorstore==0.1.85",
        ],
    )

  def test_missing_must_include_raises_value_error(self):
    without_gcsfs = [
        line for line in SAMPLE_TRAINER_FREEZE if not line.startswith("gcsfs")
    ]
    with self.assertRaisesRegex(
        ValueError, r"Missing mandatory sidecar package\(s\).*gcsfs"
    ):
      derive_sidecar_requirements(without_gcsfs)

  def test_canonical_name_normalization(self):
    dirty_freeze = [
        "  Google_Cloud_Storage==2.18.2  ",
        "Orbax_Checkpoint==0.12.4",
        "TensorStore==0.1.85",
        "GCSFS==2024.9.0",
        "Pre_Commit==3.6.0",
        "JAX==0.11.0",
    ]
    result = derive_sidecar_requirements(dirty_freeze)
    self.assertEqual(len(result), 4)
    self.assertNotIn("JAX==0.11.0", result)
    self.assertNotIn("Pre_Commit==3.6.0", result)

  def test_dockerfile_contract(self):
    dockerfile_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../src/dependencies/colocated_sidecar/Dockerfile",
        )
    )
    with open(dockerfile_path, "r", encoding="utf-8") as f:
      content = f.read()
    for required_fragment in (
        "FROM ${BASE_IMAGE}",
        "ARG REQUIREMENTS_FILE",
        "COPY ${REQUIREMENTS_FILE} .",
        "COPY maxtext /app/maxtext",
        "ENV PYTHONPATH=/app/maxtext/src:${PYTHONPATH}",
        "uv pip install -r ${REQUIREMENTS_FILE} -c /opt/venv/server_constraints.txt",
        "from jax._src.lib import _jax; _jax.colocated_python_cpu_client",
        "import orbax.checkpoint, tensorstore, gcsfs",
    ):
      self.assertIn(required_fragment, content)


if __name__ == "__main__":
  unittest.main()
