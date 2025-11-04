
# Copyright 2024 The MaxText Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hub utilities: utilities related to download and cache models
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict

from huggingface_hub import (
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    create_branch,
    create_commit,
    create_repo,
    HfHubHTTPError,
    EntryNotFoundError,
)

# Assuming a logging utility is available in the MaxText project structure.
# from .. import logging
# For demonstration, using standard logging.
import logging

# Assuming working_or_temp_dir is available from a generic utility module.
# from ..generic import working_or_temp_dir
# For self-containment, its implementation is included below.
from contextlib import contextmanager
import tempfile

# Assuming ENV_VARS_TRUE_VALUES is available from an import utility module.
# from ..import_utils import ENV_VARS_TRUE_VALUES
# For self-containment, its definition is included below.
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


@contextmanager
def working_or_temp_dir(working_dir: Union[str, os.PathLike], use_temp_dir: bool = False):
  """Context manager for a working directory that can be temporary."""
  if use_temp_dir:
    with tempfile.TemporaryDirectory() as tmp_dir:
      yield tmp_dir
  else:
    yield working_dir


logger = logging.getLogger(__name__)

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", _default_endpoint)


def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
  """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`bool`, *optional*, defaults to `False`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """
  try:
    # Check if the model card is present on the remote repo
    model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
  except EntryNotFoundError:
    # Otherwise create a simple model card from template
    model_description = (
        "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been"
        " automatically generated."
    )
    card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
    model_card = ModelCard.from_template(card_data, model_description=model_description)

  if tags is not None:
    # Ensure model_card.data.tags is a list and not None
    if model_card.data.tags is None:
      model_card.data.tags = []
    for model_tag in tags:
      if model_tag not in model_card.data.tags:
        model_card.data.tags.append(model_tag)

  return model_card


class PushToHubMixin:
  """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

  def _create_repo(
      self,
      repo_id: str,
      private: Optional[bool] = None,
      token: Optional[Union[bool, str]] = None,
      repo_url: Optional[str] = None,
      organization: Optional[str] = None,
  ) -> str:
    """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
    if repo_url is not None:
      warnings.warn(
          "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
          "instead.",
          FutureWarning,
      )
      if repo_id is not None:
        raise ValueError("`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`.")
      repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
    if organization is not None:
      warnings.warn(
          "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
          "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`).",
          FutureWarning,
      )
      if not repo_id.startswith(organization):
        if "/" in repo_id:
          repo_id = repo_id.split("/")[-1]
        repo_id = f"{organization}/{repo_id}"

    url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
    return url.repo_id

  def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]) -> Dict[str, float]:
    """
        Returns the list of files with their last modification timestamp.
        """
    return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

  def _upload_modified_files(
      self,
      working_dir: Union[str, os.PathLike],
      repo_id: str,
      files_timestamps: Dict[str, float],
      commit_message: Optional[str] = None,
      token: Optional[Union[bool, str]] = None,
      create_pr: bool = False,
      revision: Optional[str] = None,
      commit_description: Optional[str] = None,
  ):
    """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
    if commit_message is None:
      if "Model" in self.__class__.__name__:
        commit_message = "Upload model"
      elif "Config" in self.__class__.__name__:
        commit_message = "Upload config"
      elif "Tokenizer" in self.__class__.__name__:
        commit_message = "Upload tokenizer"
      elif "FeatureExtractor" in self.__class__.__name__:
        commit_message = "Upload feature extractor"
      elif "Processor" in self.__class__.__name__:
        commit_message = "Upload processor"
      else:
        commit_message = f"Upload {self.__class__.__name__}"
    modified_files = [
        f
        for f in os.listdir(working_dir)
        if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
    ]

    # filter for actual files + folders at the root level
    modified_files = [
        f
        for f in modified_files
        if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
    ]

    operations = []
    # upload standalone files
    for file in modified_files:
      if os.path.isdir(os.path.join(working_dir, file)):
        # go over individual files of folder
        for f in os.listdir(os.path.join(working_dir, file)):
          operations.append(
              CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f))
          )
      else:
        operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))

    if revision is not None and not revision.startswith("refs/pr"):
      try:
        create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
      except HfHubHTTPError as e:
        if e.response.status_code == 403 and create_pr:
          # If we are creating a PR on a repo we don't have access to, we can't create the branch.
          # so let's assume the branch already exists. If it's not the case, an error will be raised when
          # calling `create_commit` below.
          pass
        else:
          raise

    logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
    return create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=commit_message,
        commit_description=commit_description,
        token=token,
        create_pr=create_pr,
        revision=revision,
    )

  def push_to_hub(
      self,
      repo_id: str,
      use_temp_dir: Optional[bool] = None,
      commit_message: Optional[str] = None,
      private: Optional[bool] = None,
      token: Optional[Union[bool, str]] = None,
      max_shard_size: Optional[Union[int, str]] = "5GB",
      create_pr: bool = False,
      safe_serialization: bool = True,
      revision: Optional[str] = None,
      commit_description: Optional[str] = None,
      tags: Optional[List[str]] = None,
      **deprecated_kwargs,
  ) -> str:
    """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hugging Face auto-conversion utilities."""

from typing import Optional, Tuple
from huggingface_hub import HfApi
from MaxText.utils import hf_utils


def auto_conversion(
    pretrained_model_name_or_path: str,
    ignore_errors_during_conversion: bool = False,
    **cached_file_kwargs,
) -> Optional[Tuple[str, str, bool]]:
  """
  Handles automatic conversion of models to safetensors on the Hugging Face Hub.

  This function attempts to find a pre-existing safetensors conversion PR for a
  given model. If found, it downloads the converted files.

  Args:
    pretrained_model_name_or_path: The name or path of the model on the Hub.
    ignore_errors_during_conversion: If True, suppresses exceptions during the
      conversion and download process.
    **cached_file_kwargs: Additional keyword arguments passed to Hugging Face Hub
      API calls.

  Returns:
    A tuple containing (resolved_archive_file, sha, sharded) on success,
    or (None, None, None) on failure if errors are ignored.
  """
  try:
    api = HfApi(
        token=cached_file_kwargs.get("token"),
        headers={"user-agent": hf_utils.http_user_agent()},
    )
    sha = hf_utils.get_conversion_pr_reference(
        api, pretrained_model_name_or_path, **cached_file_kwargs
    )

    if sha is None:
      return None, None, None
    cached_file_kwargs["revision"] = sha
    if "_commit_hash" in cached_file_kwargs:
      del cached_file_kwargs["_commit_hash"]

    # This is an additional HEAD call that could be removed if we could infer
    # sharded/non-sharded from the PR description.
    sharded = api.file_exists(
        pretrained_model_name_or_path,
        "model.safetensors.index.json",
        revision=sha,
        token=cached_file_kwargs.get("token"),
    )
    filename = "model.safetensors.index.json" if sharded else "model.safetensors"

    resolved_archive_file = hf_utils.cached_file(
        pretrained_model_name_or_path, filename, **cached_file_kwargs
    )
    return resolved_archive_file, sha, sharded
  except Exception as e:
    if not ignore_errors_during_conversion:
      raise e
    return None, None, None

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utils for downloading and caching model weights.
"""

import json
import os
import warnings
from typing import Any, Optional, Union, Dict, List, Tuple

# from .. import hub_utils
# from .hub import cached_files (if it were in a different file)


def get_checkpoint_shard_files(
    pretrained_model_name_or_path: str,
    index_filename: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: Optional[bool] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    revision: Optional[str] = None,
    subfolder: str = "",
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r", encoding="utf-8") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(list(set(index["weight_map"].values())))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub. Try to get everything from cache,
    # or download the files
    # pylint: disable-next=line-too-long
    # from: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L1153
    cached_filenames = cached_files(
        pretrained_model_name_or_path,
        shard_filenames,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
        token=token,
        user_agent=user_agent,
        revision=revision,
        subfolder=subfolder,
        _commit_hash=_commit_hash,
    )

    return cached_filenames, sharded_metadata
