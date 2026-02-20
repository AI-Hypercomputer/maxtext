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

"""Shim for RL Utils in `src/maxtext/trainers/post_train/rl`."""

import importlib

from maxtext.utils import max_logging

OLD_MODULE_PATH = "MaxText.rl.utils_rl"
NEW_MODULE_PATH = "maxtext.trainers.post_train.rl.utils_rl"

max_logging.warning(f"'{OLD_MODULE_PATH}' is deprecated; use '{NEW_MODULE_PATH}' instead.\n")
_new_module = importlib.import_module(NEW_MODULE_PATH)

# Re-export all public names for backward compatibility
SUBSTITUTIONS = _new_module.SUBSTITUTIONS
REMOVED_EXPRESSIONS = _new_module.REMOVED_EXPRESSIONS
get_match_format_regex = _new_module.get_match_format_regex
match_format_exactly = _new_module.match_format_exactly
match_format_approximately = _new_module.match_format_approximately
normalize_final_answer = _new_module.normalize_final_answer
check_answer = _new_module.check_answer
get_match_numbers_regex = _new_module.get_match_numbers_regex
check_numbers = _new_module.check_numbers
extract_hash_answer = _new_module.extract_hash_answer
get_optimizer = _new_module.get_optimizer
process_data = _new_module.process_data
