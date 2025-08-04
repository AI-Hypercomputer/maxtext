# SPDX-License-Identifier: Apache-2.0

import os.path

PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # MaxText directory path
EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

__all__ = ["DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE", "EPS", "PKG_DIR"]
