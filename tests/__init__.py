# Copyright 2023–2025 Google LLC
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

"""
Test initialization
"""

try:
  import pathwaysutils

  pathwaysutils.initialize()
except ImportError:
  import sys
  from unittest.mock import MagicMock

  mock_pathwaysutils = MagicMock()
  mock_pathwaysutils.__path__ = []
  mock_pathwaysutils.is_pathways_backend_used.return_value = False
  sys.modules["pathwaysutils"] = mock_pathwaysutils

  mock_elastic = MagicMock()
  mock_elastic.__path__ = []
  sys.modules["pathwaysutils.elastic"] = mock_elastic

  mock_manager = MagicMock()
  sys.modules["pathwaysutils.elastic.manager"] = mock_manager

try:
  import tokamax
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel
except ImportError:
  import sys
  from unittest.mock import MagicMock

  mock_tokamax = MagicMock()
  mock_tokamax.__path__ = []
  sys.modules["tokamax"] = mock_tokamax
  sys.modules["tokamax._src"] = MagicMock()
  sys.modules["tokamax._src.ops"] = MagicMock()
  sys.modules["tokamax._src.ops.experimental"] = MagicMock()
  sys.modules["tokamax._src.ops.experimental.tpu"] = MagicMock()
  sys.modules["tokamax._src.ops.experimental.tpu.splash_attention"] = MagicMock()

try:
  import tensorflow
except ImportError:
  import sys
  from unittest.mock import MagicMock

  mock_tf = MagicMock()
  mock_tf.__path__ = []
  sys.modules["tensorflow"] = mock_tf
  sys.modules["tensorflow.io"] = MagicMock()
  sys.modules["tensorflow.data"] = MagicMock()
  sys.modules["tensorflow.compat"] = MagicMock()
  sys.modules["tensorflow.compat.v1"] = MagicMock()
  sys.modules["tensorflow.compat.v1.io"] = MagicMock()
  sys.modules["tensorflow.compat.v1.io.gfile"] = MagicMock()
