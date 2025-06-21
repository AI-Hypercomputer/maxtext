"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utility functions for MaxText.configs"""

from typing import TypeVar

from pydantic import BaseModel

from deepmerge import always_merger

T = TypeVar("T", bound=BaseModel)


# https://github.com/pydantic/pydantic/discussions/3416#discussioncomment-12267413
def merge_pydantic_models(base: T, nxt: T) -> T:
  """Merge two Pydantic model instances.

  The attributes of 'base' and 'nxt' that weren't explicitly set are dumped into dicts
  using '.model_dump(exclude_unset=True)', which are then merged using 'deepmerge',
  and the merged result is turned into a model instance using '.model_validate'.

  For attributes set on both 'base' and 'nxt', the value from 'nxt' will be used in
  the output result.
  """
  base_dict = base.model_dump(exclude_unset=True)
  nxt_dict = nxt.model_dump(exclude_unset=True)
  merged_dict = always_merger.merge(base_dict, nxt_dict)
  return base.model_validate(merged_dict)


__all__ = ["merge_pydantic_models"]
