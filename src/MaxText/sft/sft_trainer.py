#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Shim for SFT Trainer in `src/MaxText/trainers/post_train/sft`."""

import sys
import importlib

from MaxText import max_logging

OLD_MODULE_PATH = "MaxText.sft.sft_trainer"
NEW_MODULE_PATH = "maxtext.trainers.post_train.sft.train_sft"

if __name__ == "__main__":
  try:
    _new_module = importlib.import_module(NEW_MODULE_PATH)
    if hasattr(_new_module, "main"):
      max_logging.warning(f"'{OLD_MODULE_PATH}' is deprecated; use '{NEW_MODULE_PATH}' instead.\n")
      _new_module.main(sys.argv)
  except ImportError as e:
    max_logging.error(f"Shim could not find target module: '{NEW_MODULE_PATH}'\n")
    raise e
