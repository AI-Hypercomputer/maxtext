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

"""Convenience entrypoint re-exporting ``standalone_scripts.convert_olmo35_multimodal_ckpt``."""

from maxtext.checkpoint_conversion.standalone_scripts.convert_olmo35_multimodal_ckpt import (
    DEFAULT_IMAGE_TOKEN_IDS,
    DEFAULT_VISION_MODEL_ID,
    DEFAULT_VISION_REVISION,
    OLMO_CORE_CONFIG_FILE,
    build_bridge_vision_and_connector_state,
    convert_maxtext_to_olmo_core,
    convert_olmo_core_to_maxtext,
    init_bridge_maxtext_checkpoint,
    is_multimodal_olmo_config,
    load_maxtext_params_any,
    load_olmo_core_checkpoint_weights,
    main,
    reset_image_embedding_rows,
    vision_and_connector_shapes,
)

__all__ = [
    "DEFAULT_IMAGE_TOKEN_IDS",
    "DEFAULT_VISION_MODEL_ID",
    "DEFAULT_VISION_REVISION",
    "OLMO_CORE_CONFIG_FILE",
    "build_bridge_vision_and_connector_state",
    "convert_maxtext_to_olmo_core",
    "convert_olmo_core_to_maxtext",
    "init_bridge_maxtext_checkpoint",
    "is_multimodal_olmo_config",
    "load_maxtext_params_any",
    "load_olmo_core_checkpoint_weights",
    "main",
    "reset_image_embedding_rows",
    "vision_and_connector_shapes",
]

if __name__ == "__main__":
  main()
