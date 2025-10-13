# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing for translation dataset."""

import datasets

from MaxText import max_logging

LANGUAGE_CODE_TO_MAP = {
    "en": "English",
    "fr": "France",
    #"de": "German",
    #"it": "Italian",
    #"hi": "Hindi",
}


def map_to_conversation(example, data_column):
    """Maps an example to a prompt-completion format."""
    keys = list(example[data_column].keys())
    try:
        source_language = LANGUAGE_CODE_TO_MAP[keys[0]]
        destination_language = LANGUAGE_CODE_TO_MAP[keys[1]]
    except KeyError:
        max_logging.log(f"Unsupported language codes. Expected language codes to match one of {LANGUAGE_CODE_TO_MAP.keys()}, but got {keys}")
    prompt = {
        "role": "user",
        "content": f"Translate {source_language} to {destination_language}: {example[data_column][keys[0]]}"
    }
    completion = {
        "role": "assistant",
        "content": example[data_column][keys[1]]
    }
    example["messages"] = [prompt, completion]
    return example


def convert_to_conversational_format(
        dataset,
        data_column,
):
    """Converts translation dataset to conversational format."""
    dataset_features = datasets.Features(
        {"messages": [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
    )
    data_column_names = ["messages"]
    dataset = dataset.map(
        map_to_conversation,
        fn_kwargs={"data_column": data_column},
        remove_columns=[data_column],
        features=dataset_features,
    )
    return dataset, data_column_names
